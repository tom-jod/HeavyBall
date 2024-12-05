import random
from typing import Optional, Union

import torch

from . import utils

balance_probability: float = 0.01


def _inplace_guard_(state, key, template_fn):
    key_not_in_state = key not in state
    if key_not_in_state:
        template_fn()
    return key_not_in_state


def _guard_in_state(state, key, template_fn):
    if key not in state:
        state[key] = template_fn()
    return state[key]


def _zero_guard(state, key, ref, dtype):
    return _guard_in_state(state, key,
                           lambda: torch.zeros_like(ref, dtype=torch.float32, memory_format=torch.preserve_format))


def _storage_dtype(group):
    dtype = group.get('storage_dtype', "float32")
    return getattr(torch, dtype)

class SkipUpdate(ValueError):
    pass

def exp_avg(state, group, update, grad, param):
    exp_avg = _zero_guard(state, "exp_avg", grad, _storage_dtype(group))
    utils.stochastic_lerp_(exp_avg, grad, utils.beta_debias(utils.get_beta1(group), group["step"]))
    return exp_avg


def scale_by_exp_avg_sq(state, group, update, grad, param):
    exp_avg_sq = _zero_guard(state, "exp_avg_sq", grad, _storage_dtype(group))
    return utils.scale_by_exp_avg_sq_(exp_avg_sq, grad, utils.beta_debias(utils.get_beta2(group), group["step"]),
                                      group['eps'])[0]


def scale_by_adam(state, group, update, grad, param):
    exp_avg = _zero_guard(state, "exp_avg", grad, _storage_dtype(group))
    exp_avg_sq = _zero_guard(state, "exp_avg_sq", grad, _storage_dtype(group))
    return utils.adam_(exp_avg, exp_avg_sq, grad, utils.get_beta1(group), utils.get_beta2(group), group['step'],  #
                       group['eps'])[0]


def update_by_adam(state, group, update, grad, param):
    exp_avg = _zero_guard(state, "exp_avg", grad, _storage_dtype(group))
    exp_avg_sq = _zero_guard(state, "exp_avg_sq", grad, _storage_dtype(group))
    utils.fused_adam_(param, exp_avg, exp_avg_sq, grad, utils.get_beta1(group), utils.get_beta2(group), group['step'],
                      group['lr'], group['eps'], group['weight_decay'], group['caution'])
    raise SkipUpdate


def scale_by_laprop(state, group, update, grad, param):
    exp_avg = _zero_guard(state, "exp_avg", grad, _storage_dtype(group))
    exp_avg_sq = _zero_guard(state, "exp_avg_sq", grad, _storage_dtype(group))
    return utils.laprop_(exp_avg, exp_avg_sq, grad, utils.get_beta1(group), utils.get_beta2(group), group['step'],
                         group['eps'])[0]


def update_by_laprop(state, group, update, grad, param):
    exp_avg = _zero_guard(state, "exp_avg", grad, _storage_dtype(group))
    exp_avg_sq = _zero_guard(state, "exp_avg_sq", grad, _storage_dtype(group))
    utils.fused_laprop_(param, exp_avg, exp_avg_sq, grad, utils.get_beta1(group), utils.get_beta2(group), group['step'],
                        group['lr'], group['weight_decay'], group['caution'])
    raise SkipUpdate


def update_by_schedule_free(state, group, update, grad, param):
    z = _zero_guard(state, "z", grad, _storage_dtype(group))
    state['weight_sum'] = utils.schedule_free_(group['lr'], group['weight_lr_power'], state.get('weight_sum', 0),
                                               utils.get_beta1(group), param, z, update, group['r'], group['step'])
    raise SkipUpdate


def update_by_adopt(state, group, update, grad, param):
    if 'exp_avg_sq' not in state:
        state['exp_avg_sq'] = utils.promote(update).square().to(_storage_dtype(group))
        raise SkipUpdate

    if 'exp_avg' not in state:
        update = utils.promote(update)
        easq = utils.promote(state['exp_avg_sq'])
        state['exp_avg'] = (update / easq.sqrt().clamp_(min=group['eps'])).to(_storage_dtype(group))
        utils.set_(state['exp_avg_sq'],
                   easq.lerp_(update.square(), utils.beta_debias(utils.get_beta2(group), group['step'])))
        raise SkipUpdate

    utils.fused_adopt_(param, update, state['exp_avg_sq'], state['exp_avg'], utils.get_beta1(group),
                       utils.get_beta2(group), group['step'] - 2, group['lr'], group['eps'], group['weight_decay'],
                       group['caution'])
    raise SkipUpdate


def _init_soap(state, group, grad):
    utils.init_preconditioner(grad, state, group['max_precond_dim'], group['precondition_1d'])
    utils.update_preconditioner(grad, state, group['max_precond_dim'], group['precondition_1d'], 0, True)


def _init_psgd(state, group, grad, cached_q):
    Q, state["exprs"] = utils.init_Q_exprs(grad, group['precond_init_scale'], group['max_size_triangular'],
                                           group['min_ndim_triangular'], group['memory_save_mode'],
                                           dtype=getattr(torch, group['q_dtype']))
    state["Q"] = utils.triu_to_line(Q) if group['store_triu_as_line'] else Q

    if not cached_q:
        return

    state['Q_cache'] = [torch.empty_like(q) for q in Q]

    expr = [f'{c.upper()}{c}' if q_.ndim == 2 else c for c, q_ in zip(utils.einsum_base, Q)]
    expr = ','.join(expr)
    grad_expr = ''.join(c for c, _ in zip(utils.einsum_base, grad.shape))
    out_expr = ''.join(c.upper() if c.upper() in expr else c for c in grad_expr)
    expr = f'{expr},{grad_expr}->{out_expr}'
    state['cache_expr'] = expr


def precond_schedule(state, group, prob: Union[callable, float, None] = None, name: str = 'cumulative_prob'):
    step = group['step']
    if 'precondition_frequency' in group:
        return step > 0 and step % group['precondition_frequency'] == 0
    rng = random.Random(0x172381 ^ step)
    if 'precond_scheduler' in group:
        return utils.precond_schedule(step, group['precond_scheduler'], rng)
    if prob is not None:
        return utils.psgd_should_update(state, group, prob, rng, name=name)
    raise ValueError("No preconditioner update schedule specified.")


def scale_by_soap(state, group, update, grad, param):
    update = utils.promote(update)
    grad = utils.promote(grad)

    exp_avg = _zero_guard(state, "exp_avg", grad, _storage_dtype(group))
    exp_avg_sq = _zero_guard(state, "exp_avg_sq", grad, _storage_dtype(group))
    if _inplace_guard_(state, "Q", lambda: _init_soap(state, group, grad)):
        raise SkipUpdate

    grad_projected = utils.project(update, state['Q'], False)
    precond = utils.adam_(exp_avg, exp_avg_sq, grad_projected, utils.get_beta1(group), utils.get_beta2(group),
                          utils.scalar_guard(group['step'], exp_avg))[0]
    precond = utils.project(precond, state['Q'], True)

    utils.update_preconditioner(update, state, group['max_precond_dim'], group['precondition_1d'],
                                utils.beta_debias(utils.get_beta2(group), group['step']),
                                precond_schedule(state, group))
    return precond


def _update_psgd_precond(state, group, param, grad, Q, prob: Optional[callable] = None):
    if prob is None:
        prob = utils.precond_update_prob_schedule()
    if not precond_schedule(state, group, prob):
        return

    Q = [utils.promote(q_) for q_ in Q]
    utils.psgd_update_precond(Q, state['exprs'], grad, group['precond_lr'], state['Q'], group['store_triu_as_line'])

    if grad.dim() > 1 and precond_schedule(state, group, balance_probability, "balance_prob"):
        if group['store_triu_as_line']:
            utils.psgd_balance_Q([q_ for _, q_ in state['Q']])
        else:
            utils.psgd_balance_Q(state['Q'])


def _update_psgd_cache(state, cached, q):
    if not cached:
        return q

    for c_, q_ in zip(state['Q_cache'], q):
        if q_.ndim == 2:
            torch.matmul(q_.T, q_, out=c_)
        else:
            torch.mul(q_, q_, out=c_)
    return state['Q_cache']


def scale_by_psgd(state, group, update, grad, param, cached: bool = False, prob: Optional[callable] = None):
    if _inplace_guard_(state, "Q", lambda: _init_psgd(state, group, grad, cached)):
        raise SkipUpdate

    q = utils.line_to_triu(state['Q']) if group['store_triu_as_line'] else state['Q']
    _update_psgd_precond(state, group, param, update, q, prob)
    return utils.psgd_precond_grad(False, state["exprs"][-1], update, *_update_psgd_cache(state, cached, q))


def scale_by_delayed_psgd(state, group, update, grad, param, cached: bool = False, prob: Optional[callable] = None):
    if _inplace_guard_(state, "Q", lambda: _init_psgd(state, group, grad, cached)):
        raise SkipUpdate

    q = utils.line_to_triu(state['Q']) if group['store_triu_as_line'] else state['Q']
    precond = utils.psgd_precond_grad(False, state["exprs"][-1], update, *_update_psgd_cache(state, cached, q))
    _update_psgd_precond(state, group, param, update, q, prob)
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


def chain(state, group, grad, param, *fns):
    update = grad
    skip_update = False
    for fn in fns:
        try:
            update = fn(state, group, update, grad, param)
        except SkipUpdate:
            skip_update = True
            continue
    if not skip_update and update is not None:
        utils.update_param_(param, update, group['lr'], group['weight_decay'], caution=group['caution'], grad=grad)


class ChainOpt(utils.StatefulOptimizer):
    def __init__(self, params, defaults, foreach: bool, *fns):
        super().__init__(params, defaults, foreach)
        self.fns = fns

    def _step(self, group):
        if 'base_lr' not in group:
            group['base_lr'] = group['lr']
        step = group['step'] = group.get('step', 0) + 1
        if group['warmup_steps'] and step < group['warmup_steps']:
            group['lr'] = -group['base_lr'] * step / group['warmup_steps']
        else:
            group['lr'] = -group['base_lr']
        for p, g in self.split_p_and_g_in_group(group, should_promote=False, beta1=utils.get_beta1(group)):
            state = self.state_(p)
            chain(state, group, g, p, *self.fns)
