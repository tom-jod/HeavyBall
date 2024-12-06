import functools
import random
from typing import Optional, Union

import torch

from . import utils

balance_probability: float = 0.01


def _key_in_state(state, key):
    if isinstance(key, str):
        return key in state
    for k in key:
        if isinstance(k, (tuple, list)):
            continue
        if k not in state:
            return False
    return True


def _inplace_guard_(state, key, template_fn):
    key_not_in_state = not _key_in_state(state, key)
    if key_not_in_state:
        template_fn()
    return key_not_in_state


def _guard_in_state(state, key, template_fn):
    if not _key_in_state(state, key):
        state[key] = template_fn()
    return state[key]


def _zero_guard(state, key, ref, dtype):
    return _guard_in_state(state, key,
                           lambda: torch.zeros_like(ref, dtype=torch.float32, memory_format=torch.preserve_format))


def _storage_dtype(group):
    dtype = group.get('storage_dtype', "float32")
    return getattr(torch, dtype)


def zero_guard(*names):
    def _outer(fn):
        def _fn(state, group, update, grad, param, *args, **kwargs):
            vars = [[_zero_guard(state(p), name, p, _storage_dtype(group)) for p in param] for name in names]
            return fn(state, group, update, grad, param, *args, *vars, **kwargs)

        return _fn

    return _outer


def general_guard(*names, init_fn):
    def _outer(fn):
        def _fn(state, group, update, grad, param, *args, **kwargs):
            vars = []
            skip_update = False
            for p, g, u in zip(param, grad, update):
                st = state(p)
                skip_update |= _inplace_guard_(st, names, lambda: init_fn(st, group, u, g, p, **kwargs))
                vars.append([st[name] if isinstance(name, str) else st.get(name[0], name[1]) for name in names])
            if skip_update:
                raise SkipUpdate
            return fn(state, group, update, grad, param, *args, *zip(*vars), **kwargs)

        return _fn

    return _outer


def no_state(fn):
    def _fn(state, *args, **kwargs):
        return fn(*args, **kwargs)

    return _fn


def use_state_no_foreach(fn):
    def _fn(state, *args, **kwargs):
        return fn(state(param), *args, **kwargs)

    return _fn


def no_state_no_foreach(fn):
    def _fn(state, group, *args, **kwargs):
        for a in zip(*args):
            return fn(group, *a, **kwargs)

    return _fn


class SkipUpdate(ValueError):
    pass


@zero_guard("exp_avg")
@no_state
def exp_avg(group, update, grad, param, exp_avg):
    utils.stochastic_lerp_(exp_avg, grad, utils.beta_debias(utils.get_beta1(group), group["step"]))
    return exp_avg


@zero_guard("exp_avg_sq")
@no_state
def scale_by_exp_avg_sq(group, update, grad, param, exp_avg_sq):
    return utils.scale_by_exp_avg_sq_(exp_avg_sq, grad, utils.beta_debias(utils.get_beta2(group), group["step"]),
                                      group['eps'])[0]


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_adam(group, update, grad, param, exp_avg, exp_avg_sq):
    return utils.adam_(exp_avg, exp_avg_sq, grad, utils.get_beta1(group), utils.get_beta2(group), group['step'],  #
                       group['eps'])[0]


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_adam(group, update, grad, param, exp_avg, exp_avg_sq):
    utils.fused_adam_(param, exp_avg, exp_avg_sq, grad, utils.get_beta1(group), utils.get_beta2(group), group['step'],
                      group['lr'], group['eps'], group['weight_decay'], group['caution'])
    raise SkipUpdate


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_laprop(group, update, grad, param, exp_avg, exp_avg_sq):
    return utils.laprop_(exp_avg, exp_avg_sq, grad, utils.get_beta1(group), utils.get_beta2(group), group['step'],
                         group['eps'])[0]


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_laprop(group, update, grad, param, exp_avg, exp_avg_sq):
    utils.fused_laprop_(param, exp_avg, exp_avg_sq, grad, utils.get_beta1(group), utils.get_beta2(group), group['step'],
                        group['lr'], group['weight_decay'], group['caution'])
    raise SkipUpdate


@zero_guard("z")
@no_state
def update_by_schedule_free(group, update, grad, param, z):
    group['weight_sum'] = utils.schedule_free_(group['lr'], group['weight_lr_power'], group.get('weight_sum', 0), utils.get_beta1(group), param, z, update,
                              group['r'], group['step'])
    raise SkipUpdate


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    if group['step'] == 1:
        utils.exp_avg_sq_(exp_avg_sq, update, 0, 1)
        raise SkipUpdate

    if group['step'] == 2:
        update = utils.promote(update)
        easq = utils.promote(exp_avg_sq)
        [utils.set_(ea, u / easq.sqrt().clamp_(min=group['eps'])) for ea, u, easq in zip(exp_avg, update, exp_avg_sq)]
        utils.exp_avg_sq_(exp_avg_sq, update, utils.beta_debias(utils.get_beta2(group), group['step']), 1)
        raise SkipUpdate

    utils.fused_adopt_(param, update, exp_avg_sq, exp_avg, utils.get_beta1(group), utils.get_beta2(group),
                       group['step'] - 2, group['lr'], group['eps'], group['weight_decay'], group['caution'])
    raise SkipUpdate


def _init_soap(state, group, update, grad, param):
    utils.init_preconditioner(grad, state, utils.get_beta2(group), group['max_precond_dim'], group['precondition_1d'])


def _init_psgd(state, group, update, grad, param, cached: bool = False, prob: Optional[callable] = None):
    Q, state["exprs"] = utils.init_Q_exprs(grad, group['precond_init_scale'], group['max_size_triangular'],
                                           group['min_ndim_triangular'], group['memory_save_mode'],
                                           dtype=getattr(torch, group['q_dtype']))
    state["Q"] = utils.triu_to_line(Q) if group['store_triu_as_line'] else Q

    if not cached:
        return

    state['Q_cache'] = [torch.empty_like(q) for q in Q]


def precond_schedule(group, prob: Union[callable, float, None] = None, name: str = 'cumulative_prob'):
    step = group['step']
    if 'precondition_frequency' in group:
        return step > 0 and step % group['precondition_frequency'] == 0
    rng = random.Random(0x172381 ^ step)
    if 'precond_scheduler' in group:
        return utils.precond_schedule(step, group['precond_scheduler'], rng)
    if prob is not None:
        return utils.psgd_should_update(group, prob, rng, name=name)
    raise ValueError("No preconditioner update schedule specified.")


@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_soap(group, update, grad, param, exp_avg, exp_avg_sq, Q, GG):
    update = utils.promote(update)
    grad = utils.promote(grad)

    grad_projected = [utils.project(u, q, False) for u, q in zip(update, Q)]
    precond = utils.adam_(exp_avg, exp_avg_sq, grad_projected, utils.get_beta1(group), utils.get_beta2(group),
                          utils.scalar_guard(group['step'], exp_avg[0]))
    precond = [utils.project(p, q, False) for p, q in zip(precond, Q)]

    for u, q, gg, eas in zip(update, Q, GG, exp_avg_sq):
        utils.update_preconditioner(u, q, gg, eas, group['max_precond_dim'], group['precondition_1d'],
                                    utils.beta_debias(utils.get_beta2(group), group['step']), precond_schedule(group))
    return precond


def _update_psgd_precond(group, param, grad, Q_mat, Q, exprs, prob: Optional[callable] = None):
    if prob is None:
        prob = utils.precond_update_prob_schedule()
    if not precond_schedule(group, prob, name=f"cumulative_prob_{id(Q)}"):
        return

    Q = [utils.promote(q_) for q_ in Q]
    utils.psgd_update_precond(Q_mat, exprs, grad, group['precond_lr'], Q, group['store_triu_as_line'])

    if grad.dim() > 1 and precond_schedule(group, balance_probability, "balance_prob"):
        if group['store_triu_as_line']:
            utils.psgd_balance_Q([q_ for _, q_ in Q])
        else:
            utils.psgd_balance_Q(Q)


def _update_psgd_cache(cached, Q_cache, q):
    if not cached:
        return q

    for c_, q_ in zip(Q_cache, q):
        if q_.ndim == 2:
            torch.matmul(q_.T, q_, out=c_)
        else:
            torch.mul(q_, q_, out=c_)
    return Q_cache


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd)
@no_state_no_foreach
def scale_by_psgd(group, update, grad, param, Q, exprs, Q_cache, cache_expr: str, cached: bool = False,
                  prob: Optional[callable] = None):
    Q_mat = utils.line_to_triu(Q) if group['store_triu_as_line'] else Q
    _update_psgd_precond(group, param, update, Q_mat, Q, exprs, prob)
    return utils.psgd_precond_grad(False, exprs[-1], update, *_update_psgd_cache(cached, Q_cache, Q_mat))


@general_guard("Q", "exprs", ("Q_cache", None), ("cache_expr", None), init_fn=_init_psgd)
@no_state_no_foreach
def scale_by_delayed_psgd(group, update, grad, param, Q, exprs, Q_cache, cache_expr: str, cached: bool = False,
                          prob: Optional[callable] = None):
    Q_mat = utils.line_to_triu(Q) if group['store_triu_as_line'] else Q
    precond = utils.psgd_precond_grad(False, exprs[-1], update, *_update_psgd_cache(cached, Q_cache, Q_mat))
    # TODO: Use actual cache equations
    _update_psgd_precond(group, param, update, Q_mat, Q, exprs, prob)
    return precond


def palm_beta2(state, group, update, grad, param):
    beta2 = 1 - group['step'] ** -group['beta2_scale']
    group['betas'] = (utils.get_beta1(group), beta2)
    return update


def apply_to_idx(fn, idx):
    def _fn(state, group, update, grad, param):
        args = [state, group, update, grad, param]
        return fn(args[idx])

    return _fn


def chain(state: Union[callable, dict], group, grad, param, *fns):
    update = grad
    skip_update = False
    for fn in fns:
        try:
            update = fn(state, group, update, grad, param)
        except SkipUpdate:
            skip_update = True
            continue
        if update is None:
            break
    if not skip_update and update is not None:
        utils.update_param_(param, update, group['lr'], group['weight_decay'], caution=group['caution'], grad=grad)


class ChainOpt(utils.StatefulOptimizer):
    def __init__(self, params, defaults, foreach: bool, *fns):
        super().__init__(params, defaults, foreach)

        self.fns = tuple(fns)

    def _step(self, group):
        if 'base_lr' not in group:
            group['base_lr'] = group['lr']
        step = group['step'] = group.get('step', 0) + 1
        if group['warmup_steps'] and step < group['warmup_steps']:
            group['lr'] = -group['base_lr'] * step / group['warmup_steps']
        else:
            group['lr'] = -group['base_lr']

        p, g = zip(*list(self.split_p_and_g_in_group(group, should_promote=False, beta1=utils.get_beta1(group))))

        if not group['foreach']:
            for param, grad in zip(p, g):
                state = self.state_(p)
                chain(state, group, g, p, *self.fns)
            return

        chain(self.state_, group, g, p, *self.fns)


str_or_fn = Union[str, callable, None]


def _get_clip_fn(name: str_or_fn, default: str_or_fn):
    if name is None:
        name = default
    if callable(name):
        return name
    elif name not in ('l2_clip_', 'rmsnorm_clip_', 'trust_region_clip_', 'a_law_compress', 'mu_law_compress'):
        raise ValueError(f"Clipping function {name} not found")
    return getattr(utils, name)


def default(a, b):
    return b if a is None else a


class BaseOpt(ChainOpt):
    gradient_clipping: str_or_fn = None
    update_clipping: str_or_fn = None
    palm: bool = False

    def __init__(self, params, defaults, foreach: bool, gradient_clipping: str_or_fn, update_clipping: str_or_fn,
                 palm: bool = None, *fns):
        if update_clipping is not None and any(
                fn in (update_by_adopt, update_by_adam, update_by_laprop, update_by_schedule_free) for fn in fns):
            raise ValueError("`update_by` functions do not support update clipping. Use `scale_by` functions instead.")
        fns = tuple(fns)

        if default(palm, self.palm):
            fns = (palm_beta2,) + fns
        if default(gradient_clipping, self.gradient_clipping) is not None:
            fns = (apply_to_idx(gradient_clipping, 2),) + fns
        if default(update_clipping, self.update_clipping) is not None:
            fns = fns + (apply_to_idx(update_clipping, 2),)

        super().__init__(params, defaults, foreach, *fns)


class ScheduleFree(BaseOpt):
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta'] if 'beta' in group else group['betas'][0]
            if beta1 > 0 and train_mode:
                for p in group['params']:
                    state = self.state_(p)
                    if 'z' in state:
                        # Set p.data to x
                        z = utils.promote(state['z'])
                        p32 = utils.promote(p.data)
                        p32.lerp_(end=z, weight=1 - 1 / beta1)
                        utils.copy_stochastic_(p.data, p32)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta'] if 'beta' in group else group['betas'][0]
            if beta1 > 0 and not train_mode:
                for p in group['params']:
                    state = self.state_(p)
                    if 'z' in state:
                        z = utils.promote(state['z'])
                        p32 = utils.promote(p.data)
                        p32.lerp_(end=z, weight=1 - beta1)
                        utils.copy_stochastic_(p.data, p32)
                group['train_mode'] = True
