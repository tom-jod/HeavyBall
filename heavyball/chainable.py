import copy
import time
import functools
import math
import random
from typing import Iterable, List, Literal, Optional, Union
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ExponentialLR,
    StepLR
)
from torch.optim import SGD
from . import utils
import sys
import os
from optimizers.nadamw import NAdamW

use_default = utils.use_default

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


class FunctionTransform:
    def __init__(self, fn, names: list[str] | None = None):
        if names is None:
            names = []
        self.fn = fn
        self.fn_name = self.get_fn().__name__
        self.transform_idx = None
        self.is_initialized = False
        self.names = names

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        raise NotImplementedError

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        states = [state(p) for p in param]
        skip_update = False
        for st, a in zip(states, zip(update, grad, param, *args)):
            if self.transform_idx not in st.get("is_initialized", set()):
                try:
                    self._init(st, group, *a, **kwargs)
                except SkipUpdate:
                    skip_update = True
                except:
                    raise
                finally:
                    if "is_initialized" not in st:
                        st["is_initialized"] = set()
                    st["is_initialized"].add(self.transform_idx)
        if skip_update:
            raise SkipUpdate from None
        vars = [[st.get(self.val_name(name), None) for st in states] for name in self.names]
        return self._call(state, group, update, grad, param, vars, *args, **kwargs)

    def get_fn(self):
        if utils.hasattr_none(self.fn, "get_fn"):
            return self.fn.get_fn()
        return self.fn

    def val_name(self, name):
        assert self.transform_idx is not None
        return f"{self.fn_name}_{name}_{self.transform_idx}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.fn}, transform_idx={self.transform_idx})"


def _zero_guard(state, key, ref, dtype):
    return _guard_in_state(state, key, lambda: torch.zeros_like(ref, dtype=dtype, memory_format=torch.preserve_format))


def _storage_dtype(group):
    dtype = group.get("storage_dtype", "float32")
    return getattr(torch, dtype)


class ZeroGuard(FunctionTransform):
    def __init__(self, fn, names):
        super().__init__(fn, names)

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        for name in self.names:
            _zero_guard(state, self.val_name(name), param, _storage_dtype(group))

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)


class PrecondGradAccumGuard(FunctionTransform):
    def __init__(self, fn):
        super().__init__(fn, ["precond_grad_accum"])
        self.steps_taken = 0
        self.pass_through = None

    def _accum(self, state, new):
        self.steps_taken += 1
        utils.stochastic_add_(state, new)

    def _reset(self, state):
        if self.steps_taken == 0:
            self.steps_taken = 0
            utils.zero_(state)

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        if self.pass_through is None:
            self.pass_through = not group.get("precond_grad_accum", False)
        if self.pass_through is False:
            for name in self.names:
                _zero_guard(state, self.val_name(name), param, _storage_dtype(group))

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        base_grad = update if group.get("momentum_into_precond_update", True) else grad
        if self.pass_through:
            return self.fn(state, group, update, grad, param, *args, base_grad, **kwargs)

        (vars,) = vars
        if group["is_preconditioning"]:
            if self.steps_taken:
                self._accum(vars, base_grad)
                utils.stochastic_multiply_(vars, 1 / self.steps_taken)
            else:
                vars = base_grad
        else:
            self._accum(vars, base_grad)
            vars = base_grad
        try:
            out = self.fn(state, group, update, grad, param, *args, vars, **kwargs)
        finally:
            if group["is_preconditioning"]:
                self._reset(vars)

        return out


class CopyGuard(FunctionTransform):
    def __init__(self, fn, index, names):
        super().__init__(fn, names)
        self.index = index

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        val = [update, grad, param, *args][self.index]
        for name in self.names:
            state[self.val_name(name)] = torch.clone(val)

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)


class GeneralGuard(FunctionTransform):
    def __init__(self, fn, names, init_fn, skip_first: bool = True):
        super().__init__(fn, names)
        self.init_fn = init_fn
        self.skip_first = skip_first
        self.named_to_anonymous = None
        self.anonymous_to_named = None

    def _map(self, state_fn, param, mapping):
        for p in param:
            state = state_fn(p)
            for name, mapped in mapping.items():
                if mapped in state:
                    raise ValueError(f"Name {name} already mapped to {mapped}")
                if name in state:
                    state[mapped] = state.pop(name)

    def _init(self, state: dict, group: dict, update: Tensor, grad: Tensor, param: Tensor, *args, **kwargs):
        self.init_fn(state, group, update, grad, param, **kwargs)
        for name in self.names:
            state[self.val_name(name)] = state.pop(name, None)
        if self.skip_first:
            raise SkipUpdate from None

    def _call(self, state, group, update, grad, param, vars, *args, **kwargs):
        return self.fn(state, group, update, grad, param, *args, *vars, **kwargs)


class NoState(FunctionTransform):
    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        return self.fn(group, update, grad, param, *args, **kwargs)


class NoStateNoForeach(FunctionTransform):
    def __call__(self, state, group, update, grad, param, *args, **kwargs):
        updates = []
        skip_update = False
        for a in zip(update, grad, param, *args):
            try:
                updates.append(self.fn(group, *a, **kwargs))
            except SkipUpdate:
                skip_update = True
                pass
        if skip_update:
            raise SkipUpdate from None
        return updates


def zero_guard(*names):
    return functools.partial(ZeroGuard, names=names)


def copy_guard(index, *names):
    return functools.partial(CopyGuard, index=index, names=names)


def general_guard(*names, init_fn, skip_first: bool = True):
    return functools.partial(GeneralGuard, names=names, init_fn=init_fn, skip_first=skip_first)


def no_state(fn):
    return NoState(fn)


def no_state_no_foreach(fn):
    return NoStateNoForeach(fn)


class SkipUpdate(ValueError):
    pass


@zero_guard("exp_avg")
@no_state
def exp_avg(group, update, grad, param, exp_avg):
    return utils.scale_by_exp_avg_(exp_avg, update, utils.beta_debias(utils.get_beta1(group), group["step"]))


@copy_guard(2, "init")
@no_state
def weight_decay_to_init(group, update, grad, param, init):
    utils.stochastic_lerp_(param, init, group["weight_decay_to_ema"] * group["lr"])
    return update


@zero_guard("exp_avg")
@no_state
def weight_decay_to_ema(group, update, grad, param, exp_avg):
    utils.weight_decay_to_ema_(
        param,
        exp_avg,
        utils.beta_debias(group["ema_beta"], group["step"]),
        group["weight_decay_to_ema"] * group["lr"],
    )
    return update


@zero_guard("exp_avg")
@no_state
def l1_weight_decay_to_ema(group, update, grad, param, exp_avg):
    utils.l1_weight_decay_to_ema_(
        param,
        exp_avg,
        utils.beta_debias(group["ema_beta"], group["step"]),
        group["weight_decay_to_ema"] * group["lr"],
    )
    return update


@zero_guard("exp_avg_sq")
@no_state
def scale_by_exp_avg_sq(group, update, grad, param, exp_avg_sq):
    return utils.scale_by_exp_avg_sq_(
        exp_avg_sq, update, utils.beta_debias(utils.get_beta2(group), group["step"]), group["eps"]
    )


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_adam(group, update, grad, param, exp_avg, exp_avg_sq):
    return utils.adam_(
        exp_avg,
        exp_avg_sq,
        update,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],  #
        group["eps"],
    )

@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_nadam(group, update, grad, param, exp_avg, exp_avg_sq):
    return utils.nadam_(
        exp_avg,
        exp_avg_sq,
        update,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],  #
        group["eps"],
    )

@zero_guard("exp_avg")
@no_state
def update_by_SGD(group, update, grad, param, exp_avg):
    utils.fused_SGD_(
        param,
        exp_avg,
        update,
        grad,
        utils.get_beta1(group),
        group["step"],
        group["lr"],
        group["eps"],
        group["weight_decay"],
        group["caution"],
    )
    raise SkipUpdate from None


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_adam(group, update, grad, param, exp_avg, exp_avg_sq):
    utils.fused_adam_(
        param,
        exp_avg,
        exp_avg_sq,
        update,
        grad,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],
        group["lr"],
        group["eps"],
        group["weight_decay"],
        group["caution"],
    )
    raise SkipUpdate from None

@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_nadam(group, update, grad, param, exp_avg, exp_avg_sq):
    utils.fused_nadam_(
        param,
        exp_avg,
        exp_avg_sq,
        update,
        grad,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],
        group["lr"],
        group["eps"],
        group["weight_decay"],
        group["caution"],
    )
    raise SkipUpdate from None

@zero_guard("exp_avg", "exp_avg_sq", "sum_of_norm_grad_sq",  "sum_of_norm_d_sq", "eta")
@no_state
def update_by_STORM_(group, update, grad, param, exp_avg_d, exp_avg_g, sum_of_norm_grad_sq, sum_of_norm_d_sq, eta):
    prev_grads = group.get("prev_grads", [])
    utils.fused_STORM_plus_(
        param,
        exp_avg_d,
        exp_avg_g,
        sum_of_norm_grad_sq,
        sum_of_norm_d_sq,
        update,
        grad,
        prev_grads,
        group["step"],
        group["lr"],
        group["eps"],
        group["weight_decay"],
        group["caution"],
        eta,
    )
    raise SkipUpdate from None


@zero_guard("exp_avg", "exp_avg_sq", "sum_of_norm_grad_sq",  "sum_of_norm_d_sq")
@no_state
def update_by_STORM_plus(group, update, grad, param, exp_avg_d, exp_avg_g, sum_of_norm_grad_sq, sum_of_norm_d_sq):
    prev_grads = group.get("prev_grads", [])
    utils.fused_STORM_plus_(
        param,
        exp_avg_d,
        exp_avg_g,
        sum_of_norm_grad_sq,
        sum_of_norm_d_sq,
        update,
        grad,
        prev_grads,
        group["step"],
        group["lr"],
        group["eps"],
        group["weight_decay"],
        group["caution"],
        utils.get_beta1(group),
    )
    raise SkipUpdate from None


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_MARSAdamW(group, update, grad, param, exp_avg, exp_avg_sq):
    prev_grads = group.get("prev_grads", [])
    mars_gamma = group.get("mars_gamma", 0.0)
    #print(f"above{prev_grads[0]}")
    utils.fused_MARSAdamW_(
        param,
        exp_avg,
        exp_avg_sq,
        update,
        grad,
        prev_grads,
        mars_gamma,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],
        group["lr"],
        group["eps"],
        group["weight_decay"],
        group["caution"],
    )
    raise SkipUpdate from None


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_laprop(group, update, grad, param, exp_avg, exp_avg_sq):
    return utils.laprop_(exp_avg, exp_avg_sq, update, utils.get_beta1(group), utils.get_beta2(group), group["step"])


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_laprop(group, update, grad, param, exp_avg, exp_avg_sq):
    utils.fused_laprop_(
        param,
        exp_avg,
        exp_avg_sq,
        update,
        grad,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"],
        group["lr"],
        group["weight_decay"],
        group["caution"],
    )
    raise SkipUpdate from None


@no_state
def orthogonalize_grad_to_param(group, update, grad, param):
    return utils.orthogonalize_grad_to_param(param, update, group["eps"])


@copy_guard(2, "z")
@no_state
def update_by_schedule_free(group, update, grad, param, z):
    group["weight_sum"] = utils.schedule_free_(
        group["lr"],
        group["weight_lr_power"],
        group.get("weight_sum", 0),
        utils.get_beta1(group),
        param,
        z,
        update,
        grad,
        group["caution"],
        group["r"],
        group["step"],
        group["weight_decay"],
    )
    raise SkipUpdate from None


@copy_guard(2, "z")
@zero_guard("exp_avg")
@no_state
def update_by_msam(group, update, grad, param, z, exp_avg):
    utils.msam_(
        group["lr"],
        utils.beta_debias(utils.get_beta1(group), group["step"]),
        param,
        z,
        update,
        grad,
        exp_avg,
        group["caution"],
        group["weight_decay"],
        group["sam_step_size"],
    )
    raise SkipUpdate from None


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def update_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    if group["step"] == 1:
        utils.scale_by_exp_avg_sq_(exp_avg_sq, update, 0, group["eps"])
        raise SkipUpdate from None

    if group["step"] == 2:
        update = utils.promote(update)
        easq = utils.promote(exp_avg_sq)
        [utils.set_(ea, u / easq_.sqrt().clamp_(min=group["eps"])) for ea, u, easq_ in zip(exp_avg, update, easq)]
        utils.scale_by_exp_avg_sq_(
            exp_avg_sq,
            update,
            utils.beta_debias(utils.get_beta2(group), group["step"]),
            group["eps"],
        )
        raise SkipUpdate from None

    utils.fused_adopt_(
        param,
        update,
        grad,
        exp_avg_sq,
        exp_avg,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 2,
        group["lr"],
        group["eps"],
        group["weight_decay"],
        group["caution"],
    )
    raise SkipUpdate from None


@zero_guard("exp_avg", "exp_avg_sq")
@no_state
def scale_by_adopt(group, update, grad, param, exp_avg, exp_avg_sq):
    if group["step"] == 1:
        utils.scale_by_exp_avg_sq_(exp_avg_sq, update, 0, group["eps"])
        raise SkipUpdate from None

    if group["step"] == 2:
        update = utils.promote(update)
        easq = utils.promote(exp_avg_sq)
        [utils.set_(ea, u / easq_.sqrt().clamp_(min=group["eps"])) for ea, u, easq_ in zip(exp_avg, update, easq)]
        utils.scale_by_exp_avg_sq_(
            exp_avg_sq,
            update,
            utils.beta_debias(utils.get_beta2(group), group["step"]),
            group["eps"],
        )
        raise SkipUpdate from None

    return utils.adopt(
        update,
        exp_avg_sq,
        exp_avg,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 2,
    )


def _init_soap(state, group, update, grad, param, inner: str = ""):
    utils.init_preconditioner(grad, state, group["max_precond_dim"], group["precondition_1d"])

def _init_cosmos(state, group, update, grad, param, inner: str = ""):
    utils.init_preconditioner_cosmos(grad, state) 

def _init_psgd_kron(state, group, update, grad, param, cached: bool = False, prob: Optional[callable] = None):
    Q = utils.init_Q_exprs(
        grad,
        group["precond_init_scale"],
        group["precond_init_scale_scale"],
        group["precond_init_scale_power"],
        group["max_size_triangular"],
        group["min_ndim_triangular"],
        group["memory_save_mode"],
        getattr(param, "hessian_vector", None),
        getattr(param, "vector", None),
        dtype=getattr(torch, group["q_dtype"]),
    )
    state["Q"] = utils.triu_to_line(Q) if group["store_triu_as_line"] else Q
    state["running_lower_bound"] = [torch.zeros((), device=q.device, dtype=q.dtype) for q in Q]
    if group["adaptive"]:
        state["velocity"] = [torch.zeros((), device=q.device, dtype=q.dtype) for q in Q]
    if not cached:
        return

    state["Q_cache"] = [torch.empty_like(q) for q in Q]


def _init_psgd_lra(state, group, update, grad, param, cached: bool = False, prob: Optional[callable] = None):
    state["U"], state["V"], state["d"] = utils.init_lra(
        grad,
        group["param_count"],
        group["precond_init_scale"],
        group["precond_init_scale_scale"],
        group["precond_init_scale_power"],
        group["rank"],
        getattr(param, "hessian_vector", None),
        getattr(param, "vector", None),
        dtype=getattr(torch, group["q_dtype"]),
    )


def precond_schedule(group, prob: Union[callable, float, None] = None, name: str = "cumulative_prob"):
    step = group["step"]
    if "precondition_frequency" in group:
        return step > 0 and step % group["precondition_frequency"] == 0
    if isinstance(step, torch.Tensor):
        utils.warn_once("Preconditioner schedule is not supported with torch.Tensor step.")
        rng = random.Random(0x172381)
    else:
        rng = random.Random(0x172381 ^ step)
    if "precond_scheduler" in group:
        return utils.precond_schedule(step, group["precond_scheduler"], rng)
    if prob is not None:
        return utils.psgd_should_update(group, prob, rng, name=name)
    raise ValueError("No preconditioner update schedule specified.")


@no_state_no_foreach
def orthogonalize_update(group, update, grad, param, scale_mode: str = "scale"):  # explore scale_mode="graft"
    if update.dim() < 2:
        return update
    original_shape = update.shape
    # doing it this way, as tmp and update are not guaranteed to share memory address or layout
    tmp = update.flatten(1, -1)
    utils.inplace_orthogonal_(tmp, out=tmp, scale_mode=scale_mode)
    return tmp.reshape(original_shape)


@zero_guard("momentum")
@no_state
def nesterov_momentum(group, updates, grads, params, momentum):
    return utils.nesterov_momentum(momentum, updates, utils.get_beta1(group))


@zero_guard("momentum")
@no_state
def nesterov_ema(group, updates, grads, params, momentum):  # equivalent to Grokfast
    return utils.nesterov_ema(momentum, updates, utils.get_beta1(group))


def _store_std(state, group, update, grad, param):
    state["init_std"] = torch.std(param)


@general_guard("init_std", init_fn=_store_std, skip_first=False)
@no_state
def mup_approx(group, updates, grads, params, init_std):
    _updates = [(u, i) for u, i in zip(updates, init_std) if u.ndim > 1]
    _updates, _init_std = zip(*_updates)
    utils.stochastic_multiply_(_updates, _init_std)
    return updates


def _init_delta(state, group, update, grad, param, log_space: bool):
    val = group["initial_d"]
    state["delta"] = torch.full((), math.log(val) if log_space else val, dtype=param.dtype, device=param.device)


def _init_full_delta(state, group, update, grad, param, log_space: bool):
    val = group["initial_d"]
    state["delta"] = torch.full_like(param, math.log(val) if log_space else val)


@zero_guard("state")
@general_guard("delta", init_fn=functools.partial(_init_delta, log_space=False), skip_first=False)
@no_state
def scale_by_d_adaptation(group, update, grad, param, state, delta):
    utils.d_adaptation(grad, update, state, delta)
    return update


@zero_guard("state")
@general_guard("delta", init_fn=functools.partial(_init_delta, log_space=True), skip_first=False)
@no_state
def scale_by_lr_adaptation(group, update, grad, param, state, delta):
    utils.lr_adaptation(grad, update, state, delta, group["lr_lr"])
    return update


@zero_guard("state")
@general_guard("delta", init_fn=functools.partial(_init_full_delta, log_space=True), skip_first=False)
@no_state
def scale_by_pointwise_lr_adaptation(group, update, grad, param, state, delta):
    utils.pointwise_lr_adaptation(grad, update, state, delta, group["lr_lr"])
    return update


@zero_guard("momentum")
@no_state
def heavyball_momentum(group, updates, grads, params, momentum):
    return utils.heavyball_momentum(momentum, updates, utils.get_beta1(group))


def _init_tensor_shampoo(state, group, update, grad, param):
    """Initialize Shampoo preconditioner states for arbitrary tensor dimensions"""
    G_modes = []
    Q_modes = []
    merged_shapes = []
    
    for p in param:
        if _should_skip_preconditioning(p, group):
            # Skip preconditioning - store empty lists
            G_modes.append([])
            Q_modes.append([])
            merged_shapes.append(param.shape)
        else:
            # Determine tensor shape (with optional dimension merging)
            if group["use_merge_dims"]:
                merged_shape = utils.merge_small_dims(
                    p.shape, group["merge_small_dims_threshold"]
                )
            else:
                merged_shape = p.shape
            
            merged_shapes.append(merged_shape)
            
            # Initialize preconditioner matrices for each mode
            param_G_modes = []
            param_Q_modes = []
            
            modes_to_precondition = _get_modes_to_precondition(merged_shape, group)
            
            for mode_idx, should_precondition in enumerate(modes_to_precondition):
                if should_precondition:
                    dim_size = merged_shape[mode_idx]
                    
                    # Skip if dimension is too large
                    if (group["skip_preconditioning_dim_size_gt"] > 0 and 
                        dim_size > group["skip_preconditioning_dim_size_gt"]):
                        param_G_modes.append(None)
                        param_Q_modes.append(None)
                        continue
                    
                    # Limit dimension size for memory efficiency
                    actual_dim = min(dim_size, group["max_preconditioner_dim"])
                    
                    # Initialize G matrix (statistics accumulator)
                    device, dtype = p.device, getattr(torch, group["storage_dtype"])
                    G_matrix = torch.eye(actual_dim, device=device, dtype=dtype) * group["eps"]
                    param_G_modes.append(G_matrix)
                    
                    # Initialize Q matrix (orthogonal basis)
                    Q_matrix = torch.eye(actual_dim, device=device, dtype=dtype)
                    param_Q_modes.append(Q_matrix)
                else:
                    param_G_modes.append(None)
                    param_Q_modes.append(None)
            
            G_modes.append(param_G_modes)
            Q_modes.append(param_Q_modes)
    
    return {
        "G_modes": G_modes,
        "Q_modes": Q_modes, 
        "merged_shapes": merged_shapes
    }

@zero_guard("exp_avg", "exp_avg_sq", "momentum")  # For grafting + momentum
@general_guard("G_modes", "Q_modes", "merged_shapes", init_fn=_init_tensor_shampoo)
@no_state
def update_by_tensor_shampoo(group, update, grad, param, exp_avg, exp_avg_sq, momentum,
                            G_modes, Q_modes, merged_shapes):
    """Unified tensor Shampoo update supporting arbitrary tensor dimensions"""
    
    # 1. Apply momentum/gradient filtering if enabled
    if group["betas"][0] > 0:
        filtered_grad = utils.update_momentum_list_(
            momentum, update, group["betas"][0], group["step"], group.get("use_bias_correction", True)
        )
    else:
        filtered_grad = update
    
    # 2. Determine which tensors to precondition
    use_shampoo = group["step"] >= group["start_preconditioning_step"]
    
    if use_shampoo:
        # Apply tensor Shampoo preconditioning
        preconditioned_grad = _apply_tensor_shampoo_preconditioning(
            filtered_grad, param, G_modes, Q_modes, merged_shapes, group
        )
    else:
        # During warmup, skip Shampoo preconditioning
        preconditioned_grad = filtered_grad
    
    # 3. Apply grafting for layer-wise scaling
    if group.get("grafting_type") and group["grafting_type"] != "none":
        final_update = _apply_tensor_grafting(
            preconditioned_grad, filtered_grad, exp_avg, exp_avg_sq, group, use_shampoo
        )
    else:
        final_update = preconditioned_grad
    
    # 4. Apply final parameter update with weight decay
    utils.fused_parameter_update_list_(
        param, final_update, -group["lr"], group["weight_decay"], group["caution"]
    )
    
    raise SkipUpdate from None


def _should_skip_preconditioning(param, group):
    """Determine if we should skip preconditioning for this parameter"""
    # Skip based on rank
    if param.dim() < group["skip_preconditioning_rank_lt"]:
        return True
    
    # Skip based on dimension size
    if (group["skip_preconditioning_dim_size_gt"] > 0 and 
        any(s > group["skip_preconditioning_dim_size_gt"] for s in param.shape)):
        return True
    
    return False

def _get_modes_to_precondition(shape, group):
    """Determine which tensor modes to precondition based on preconditioner_type"""
    rank = len(shape)
    
    if group["preconditioner_type"] == "all" or rank <= 1:
        return [True] * rank  # Precondition all dimensions
    elif group["preconditioner_type"] == "input":
        return [True] * (rank - 1) + [False]  # All but last dimension
    elif group["preconditioner_type"] == "output":  
        return [False] * (rank - 1) + [True]  # Only last dimension
    else:
        raise ValueError(f"Unknown preconditioner_type: {group['preconditioner_type']}")

def _apply_tensor_shampoo_preconditioning(grad_list, param_list, G_modes, Q_modes, merged_shapes, group):
    """Apply Shampoo preconditioning to gradients of arbitrary tensor dimensions"""
    preconditioned_grads = []
    should_update_preconditioners = (group["step"] % group["precondition_frequency"] == 0)
    
    for grad, param, G_list, Q_list, merged_shape in zip(
        grad_list, param_list, G_modes, Q_modes, merged_shapes
    ):
        if not G_list:  # Skip preconditioning for this parameter
            preconditioned_grads.append(grad)
            continue
        
        # Reshape gradient to merged shape if needed
        if merged_shape != param.shape:
            reshaped_grad = grad.view(merged_shape)
        else:
            reshaped_grad = grad
        
        # Apply multi-mode preconditioning
        preconditioned_grad = reshaped_grad
        
        for mode_idx, (G_matrix, Q_matrix) in enumerate(zip(G_list, Q_list)):
            if G_matrix is not None and Q_matrix is not None:
                # Update statistics (G matrix) 
                if should_update_preconditioners:
                    utils.update_preconditioner_mode_(
                        G_matrix, reshaped_grad, mode_idx, 
                        group["betas"][1], group["eps"]
                    )
                    
                    # Update orthogonal basis (Q matrix) using QR decomposition
                    utils.update_orthogonal_basis_(G_matrix, Q_matrix)
                
                # Apply preconditioning for this mode
                preconditioned_grad = utils.apply_mode_preconditioning(
                    preconditioned_grad, Q_matrix, mode_idx
                )
        
        # Reshape back to original parameter shape
        if merged_shape != param.shape:
            preconditioned_grad = preconditioned_grad.view(param.shape)
        
        preconditioned_grads.append(preconditioned_grad)
    
    return preconditioned_grads

def _apply_tensor_grafting(shampoo_updates, original_grads, exp_avg, exp_avg_sq, group, use_shampoo):
    """Apply grafting to scale Shampoo updates using first-order method norms"""
    if not use_shampoo:
        # During warmup, just use the grafting method directly
        return _compute_grafting_updates(original_grads, exp_avg, exp_avg_sq, group)
    
    # Compute grafting updates for comparison
    grafting_updates = _compute_grafting_updates(original_grads, exp_avg, exp_avg_sq, group)
    
    # Apply grafting: scale Shampoo updates by grafting norms
    grafted_updates = []
    for shampoo_update, grafting_update in zip(shampoo_updates, grafting_updates):
        # Compute norms
        shampoo_norm = torch.norm(shampoo_update) + 1e-16  # Avoid division by zero
        grafting_norm = torch.norm(grafting_update)
        
        # Scale Shampoo update by grafting norm ratio
        scale_factor = grafting_norm / shampoo_norm
        grafted_update = shampoo_update * scale_factor
        
        grafted_updates.append(grafted_update)
    
    return grafted_updates

def _compute_grafting_updates(grad_list, exp_avg, exp_avg_sq, group):
    """Compute updates using the grafting method (Adam, RMSprop, etc.)"""
    grafting_type = group["grafting_type"]
    
    if grafting_type == "adam":
        return utils.compute_adam_updates_(
            exp_avg, exp_avg_sq, grad_list, 
            group["betas"][1], group["step"], group["eps"],
            group.get("use_bias_correction", True)
        )
    elif grafting_type == "rmsprop":
        return utils.compute_rmsprop_updates_(
            exp_avg_sq, grad_list,
            group["betas"][1], group["eps"]
        )
    elif grafting_type == "sgd":
        return grad_list  # No preconditioning
    else:
        raise ValueError(f"Unknown grafting_type: {grafting_type}")


_optim_fns = {"adam": utils.adam_, "laprop": utils.laprop_, "lion": utils.LION_}


@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("Q", "GG", init_fn=_init_soap)
@no_state
def scale_by_soap(group, update, grad, param, exp_avg, exp_avg_sq, Q, GG, inner: str = "adam"):
    grad_projected = [utils.project(utils.promote(u), q, False) for u, q in zip(update, Q)]
    fn = _optim_fns[inner]
    precond = fn(
        exp_avg,
        exp_avg_sq,
        grad_projected,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["step"] - 1,
        group["eps"],
    )
    precond = [utils.project(p, q, True) for p, q in zip(precond, Q)]

    for u, q, gg, ea in zip(update, Q, GG, exp_avg):
        utils.update_preconditioner(
            utils.promote(u),
            q,
            gg,
            ea,
            group["max_precond_dim"],
            group["precondition_1d"],
            utils.beta_debias(group["shampoo_beta"], group["step"]),
            group["is_preconditioning"],
        )
    return precond

_optim_fns_cosmos = {"cosmos": utils.cosmos_, "laprop": utils.laprop_, "lion": utils.LION_}

@zero_guard("exp_avg", "exp_avg_sq")
@general_guard("U", "S", "adam_exp_avg_sq", init_fn=_init_cosmos)
@no_state
def scale_by_cosmos(group, update, grad, param, exp_avg, exp_avg_sq, adam_exp_avg_sq, U, S, inner: str = "cosmos"):
   
    fn = _optim_fns_cosmos[inner]
   
    precond = fn(
        exp_avg,
        exp_avg_sq,
        adam_exp_avg_sq,
        grad,
        U, 
        S,
        utils.get_beta1(group),
        utils.get_beta2(group),
        group["lr"],
        utils.get_beta3(group), # let beta3 be gamma for cosmos
        group["step"] - 1,
        group["eps"],
    )
    return precond

def _update_psgd_precond(
    cached, Q_cache, group, param, grad, Q, velocity, running_lower_bound, prob: Optional[callable] = None
) -> Optional[Tensor]:
    if prob is None:
        prob = utils.precond_update_prob_schedule()

    if not group["is_preconditioning"]:
        return

    if utils.hasattr_none(param, "vector"):
        vector, hessian_vector = param.vector, param.hessian_vector
        del param.vector
        del param.hessian_vector
    elif group["inverse_free"]:
        vector, hessian_vector = None, grad
    else:
        vector, hessian_vector = utils.dampen_grad(grad, group["dampening"])

    precond = (utils.inverse_free_psgd_update_precond if vector is None else utils.psgd_update_precond)(
        hessian_vector,
        group["precond_lr"],
        Q,
        group["store_triu_as_line"],
        velocity,
        utils.beta_debias(utils.get_beta2(group), group["step"]),
        group["ortho_method"],
        vector,
        running_lower_bound,
        group["lower_bound_beta"],
        group["precond_update_power_iterations"],
    )
    del vector, hessian_vector

    if isinstance(prob, float):
        float_prob = prob
    else:
        float_prob = prob(group.get(f"cumulative_prob_{id(Q)}_prob_step", 1))
    group["is_cached"] = should_use_cache = cached and float_prob < 0.5

    if precond is not None:
        return precond
    if not should_use_cache or not cached:
        return None  # caching adds extra ops and is not worth the overhead when we precondition at every step

    for c_, q_ in zip(Q_cache, utils.line_to_triu(Q, group["inverse_free"]) if group["store_triu_as_line"] else Q):
        if q_.ndim == 2:
            torch.matmul(q_.T, q_, out=c_)
        else:
            torch.mul(q_, q_, out=c_)
    return None


def _cached_psgd_precond_grad(group, update, Q, Q_cache, grad):
    kwargs = {"ea": update, "caution": group["caution"], "grad": grad}
    if group.get("is_cached", False):
        out = utils.precond_grad_cached_(cached_q=Q_cache, **kwargs)
    else:
        out = utils.psgd_precond_grad(
            preconds=Q, store_triu_as_line=group["store_triu_as_line"], symmetric_output=group["inverse_free"], **kwargs
        )
    group["caution"] = False  # we already cautioned here - shouldn't do it again
    return out


def _fused_cached_psgd_precond_grad(group, grad, param, update, Q, Q_cache):
    kwargs = {
        "ea": update,
        "caution": group["caution"],
        "grad": grad,
        "param": param,
        "lr": group["lr"],
        "decay": group["weight_decay"],
    }
    if group.get("is_cached", False):
        utils.fused_precond_grad_cached_(cached_q=Q_cache, **kwargs)
    else:
        utils.fused_psgd_precond_grad(
            preconds=Q, store_triu_as_line=group["store_triu_as_line"], symmetric_output=group["inverse_free"], **kwargs
        )


def _update_lra(
    group, U: List[Tensor], V: List[Tensor], d: List[Tensor], params: List[Tensor], grads: List[Tensor], delayed: bool
):
    if not group["is_preconditioning"]:
        return utils.multi_flatten((U, 1), (V, 1), (d, 0))

    if utils.hasattr_none(params[0], "hessian_vector"):
        vector = utils.flatten([p.vector for p in params])
        hessian_vector = utils.flatten([p.hessian_vector for p in params])
        for p in params:
            del p.vector
            del p.hessian_vector
    else:
        vector, hessian_vector = utils.dampen_multiple(grads)
    precond_step = group["precond_step"] = group.get("precond_step", -1) + 1
    return utils.update_lra_precond_(
        U, V, d, vector, hessian_vector, group["eps"], group["precond_lr"], delayed, bool(precond_step % 2)
    )


@PrecondGradAccumGuard
@general_guard("U", "V", "d", init_fn=_init_psgd_lra, skip_first=False)
@no_state
def scale_by_psgd_lra(group, update, grad, param, update_to_precond, U, V, d):
    u, v, d = _update_lra(group, U, V, d, param, update_to_precond, False)
    return utils.extract_from_flat_update(param, utils.lra_precond(u, v, d, utils.flatten(update)))


@PrecondGradAccumGuard
@general_guard("U", "V", "d", init_fn=_init_psgd_lra, skip_first=False)
@no_state
def update_by_psgd_lra(group, update, grad, param, update_to_precond, U, V, d):
    u, v, d = _update_lra(group, U, V, d, param, update_to_precond, False)
    utils.apply_lra_update(param, update, u, v, d, group["lr"], group["weight_decay"], group["caution"], grad)
    raise SkipUpdate from None


@PrecondGradAccumGuard
@general_guard("U", "V", "d", init_fn=_init_psgd_lra, skip_first=False)
@no_state
def scale_by_delayed_psgd_lra(group, update, grad, param, update_to_precond, U, V, d):
    u, v, d = _update_lra(group, U, V, d, param, update_to_precond, True)
    return utils.extract_from_flat_update(param, utils.lra_precond(u, v, d, utils.flatten(update)))


@PrecondGradAccumGuard
@general_guard("U", "V", "d", init_fn=_init_psgd_lra, skip_first=False)
@no_state
def update_by_delayed_psgd_lra(group, update, grad, param, update_to_precond, U, V, d):
    u, v, d = _update_lra(group, U, V, d, param, update_to_precond, True)
    utils.apply_lra_update(param, update, u, v, d, group["lr"], group["weight_decay"], group["caution"], grad)
    raise SkipUpdate from None


@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "velocity", "running_lower_bound", init_fn=_init_psgd_kron, skip_first=False)
@no_state_no_foreach
def scale_by_psgd(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    velocity: Optional[List[Tensor]],
    running_lower_bound: List[Tensor],
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, velocity, running_lower_bound, prob)
    return _cached_psgd_precond_grad(group, update, Q, Q_cache, grad)


@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "velocity", "running_lower_bound", init_fn=_init_psgd_kron, skip_first=False)
@no_state_no_foreach
def scale_by_delayed_psgd(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    velocity: Optional[List[Tensor]],
    running_lower_bound: List[Tensor],
    cached: bool = False,
    prob: Optional[callable] = None,
):
    if group.get("inverse_free", False):
        precond = None
    else:
        precond = _cached_psgd_precond_grad(group, update, Q, Q_cache, grad)
    new = _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, velocity, running_lower_bound, prob)
    return new if precond is None else precond


@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "velocity", "running_lower_bound", init_fn=_init_psgd_kron, skip_first=False)
@no_state_no_foreach
def update_by_psgd(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    velocity: Optional[List[Tensor]],
    running_lower_bound: List[Tensor],
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, velocity, running_lower_bound, prob)
    _fused_cached_psgd_precond_grad(group, update, param, update, Q, Q_cache)
    raise SkipUpdate from None


@no_state
def sign(group, update, grad, param, graft: bool = True):
    return utils.sign_(update, graft)


@no_state
def global_clip(group, update, grad, param, clip_fn: Optional[callable] = None):
    assert clip_fn is not None
    return clip_fn(update)


@PrecondGradAccumGuard
@general_guard("Q", "Q_cache", "velocity", "running_lower_bound", init_fn=_init_psgd_kron, skip_first=False)
@no_state_no_foreach
def update_by_delayed_psgd(
    group,
    update,
    grad,
    param,
    update_to_precond,
    Q,
    Q_cache,
    velocity: Optional[List[Tensor]],
    running_lower_bound: List[Tensor],
    cached: bool = False,
    prob: Optional[callable] = None,
):
    _fused_cached_psgd_precond_grad(group, update, param, update, Q, Q_cache)
    _update_psgd_precond(cached, Q_cache, group, param, update_to_precond, Q, velocity, running_lower_bound, prob)
    raise SkipUpdate from None


def palm_beta2(state, group, update, grad, param):
    beta2 = 1 - group["step"] ** -group["beta2_scale"]
    group["betas"] = (utils.get_beta1(group), beta2)
    return update


def apply_to_idx(fn, idx):
    def _fn(state, group, update, grad, param):
        args = [state, group, update, grad, param]
        return fn(args[idx])

    return _fn


def _inner_chain(state, group, update, grad, param, *fns):
    skip_update = False
    for fn in fns:
        try:
            update = fn(state, group, update, grad, param)
        except SkipUpdate:
            skip_update = True
            continue
        if update is None:
            break
    return update, skip_update


def chain(state: Union[callable, dict], group, grad, param, *fns):
    
    update = [torch.clone(g, memory_format=torch.preserve_format) for g in grad]
    update, skip_update = _inner_chain(state, group, update, grad, param, *fns)
    
    if not skip_update and update is not None:
        
        utils.update_param_(param, update, group["lr"], group["weight_decay"], caution=group["caution"], grad=grad)


def create_branch(branches: List[List[callable]], merge_fn: callable):
    def _branch(state, group, update, grad, param):
        outputs = []
        for branch in branches:
            branch_update = [torch.clone(u, memory_format=torch.preserve_format) for u in update]
            branch_update, skip_update = _inner_chain(state, group, branch_update, grad, param, *branch)
            if skip_update:
                raise ValueError("Branches should not skip updates")
            outputs.append(branch_update)
        return merge_fn(outputs)

    return _branch


def set_indices(fns: Iterable[callable], retain: bool = True, offset: int = 0):
    if retain:
        if offset:
            raise ValueError("offset cannot be retained")

        offset = -1
        for fn in fns:
            while isinstance(fn, (FunctionTransform, functools.partial)):
                if isinstance(fn, functools.partial):
                    fn = fn.func
                    continue
                if fn.transform_idx is not None:
                    offset = max(offset, fn.transform_idx)
                fn = fn.fn
        offset += 1  # if we found nothing, this will be 0. if we found something, we START at N+1

    fns = [copy.deepcopy(fn) for fn in fns]
    for fn in fns:
        while isinstance(fn, (FunctionTransform, functools.partial)):
            if isinstance(fn, functools.partial):
                fn = fn.func
                continue
            if not retain or fn.transform_idx is None:
                fn.transform_idx = offset
                offset += 1
            fn = fn.fn
    return fns


def create_step_schedule_fn(total_steps, warmup_ratio=0.05, decay_points=[0.5, 0.75], gamma=0.1):
    """Creates a function that returns LR multiplier for step decay schedule"""
    milestones = [int(p * total_steps) for p in decay_points]
    warmup_steps = int(warmup_ratio * total_steps)
    def get_lr_multiplier(step):
        if step <= warmup_steps:
            # Linear warmup: scale from 0 to 1
            return step / warmup_steps if warmup_steps > 0 else 1.0
        multiplier = 1.0
        for milestone in milestones:
            if step >= milestone:
                multiplier *= gamma
        return multiplier
    
    return get_lr_multiplier

def create_warmup_cosine_schedule_fn(total_steps, warmup_ratio=0.05, min_lr_ratio=0.0):
    """Creates a function that returns LR multiplier for linear warmup + cosine annealing schedule"""
    warmup_steps = int(warmup_ratio * total_steps)
 
    def get_lr_multiplier(step):
        if step <= warmup_steps:
            # Linear warmup: scale from 0 to 1
            return step / warmup_steps if warmup_steps > 0 else 1.0
        else:
            # Cosine annealing after warmup
            cosine_steps = total_steps - warmup_steps
            current_cosine_step = step - warmup_steps
            
            # Cosine annealing formula
            cosine_factor = 0.5 * (1 + math.cos(math.pi * current_cosine_step / cosine_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor
    
    return get_lr_multiplier

class ChainOpt(utils.StatefulOptimizer):
    promote: bool = False
    
    def __init__(self, params, defaults, foreach: bool, *fns):
        defaults = {k: v for k, v in defaults.items() if v is not use_default}
        super().__init__(params, defaults, foreach)
        self.fns = fns
        
        # Initialize step decay function with defaults
        # Get total_steps from defaults, use a reasonable default if not provided
        self.total_steps = defaults.get("total_steps", 100000)
        decay_points = defaults.get("decay_points", [0.5, 0.75])
        decay_gamma = defaults.get("decay_gamma", 0.1)
        self.warmup_ratio = defaults.get("warmup_ratio", 0.05)  # 5% warmup by default
        self.min_lr_ratio = defaults.get("min_lr_ratio", 0.0)   # Minimum LR as ratio of base LR
        
        # Create the warmup + cosine schedule function
        self._lr_schedule_fn = create_warmup_cosine_schedule_fn(
            self.total_steps, self.warmup_ratio, self.min_lr_ratio
        )
        # Step decay
        #self._lr_schedule_fn = create_step_schedule_fn(self.total_steps, self.warmup_ratio, decay_points, decay_gamma)
        # Keep existing scheduler initialization for backward compatibility
        self.lr_scheduler = None
        self._scheduler_initialized = False
        self._global_external_optimizers = {}
        # Global optimizers for DistributedShampoo (created once)
        self._global_shampoo_optimizers = {}
        self._param_to_optimizer_map = {}  # Track which optimizer handles which param

    def _step(self, group, closure=None):
        
        if "base_lr" not in group:
            group["base_lr"] = group["lr"]
            
        if "prev_lr" in group and group["prev_lr"] != group["lr"]:
            utils.warn_once(
                f"Learning rate changed between steps. This is an experimental feature and "
                f"only supported with foreach=True (currently foreach={group['foreach']})."
            )
            group["base_lr"] = group["lr"]
            
        caution = group["caution"]
        use_ema = group.get("use_ema", False)
        prev_grads = group.get("prev_grads", [])
        prev_loss = group.get("prev_loss", None)
        group["prev_grads"] = prev_grads
        
        vals = list(self.split_p_and_g_in_group(
            group, should_promote=self.promote,
            beta1=utils.get_beta1(group), use_ema=use_ema
        ))
        
        if not vals:
            return
            
        p, g = zip(*vals)
        
        # Get step count from first parameter (they should all be the same)
        step = 0
        for param in p:
            state = self.state_(param)
            if "step" in state:
                step = state["step"]
                break
            elif self.compile_step:
                step = utils.scalar_guard(0, param)
                break

        # Increment step for ALL parameters
        step = step + 1
        for param in p:
            state = self.state_(param)
            state["step"] = step

        group["step"] = step
                
        # Handle learning rate scheduling
        current_lr = group["base_lr"]
        external_optimizer_type = group.get("external_optimizer", None)
        
        # Check if we should use the new warmup + cosine schedule
        use_warmup_cosine = group.get("use_warmup_cosine", True)  # Default to True
        
        if use_warmup_cosine:
            
            # Use warmup + cosine annealing schedule
            lr_multiplier = self._lr_schedule_fn(step)
            current_lr = group["base_lr"] * lr_multiplier
        else:
            # Fall back to existing logic for backward compatibility
            # Apply warmup if specified
            warmup_steps = self.warmup_ratio * self.total_steps
            
            if warmup_steps > 0 and step <= warmup_steps:
                warmup_factor = step / warmup_steps
                current_lr = group["base_lr"] * warmup_factor
            
            # Apply other LR schedules if explicitly enabled
            if group.get("lr_schedule", False):
                
                self._initialize_lr_scheduler()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    current_lr = self.lr_scheduler.get_last_lr()[0]
        
       # if step%1000==0:
        #    print(current_lr)    
        group["lr"] = current_lr
        group["prev_lr"] = current_lr
        # Apply optimization step
        
        
        
        if external_optimizer_type is not None:
            if external_optimizer_type.lower() == "distributedshampoo":
                # Just assign gradients
                for param, grad in zip([param for param, _ in vals], g):
                    param.grad = grad
            else:
                # Handle other external optimizers as before
                param_tensors = [param for param, _ in vals]
                self._use_external_optimizer(group, external_optimizer_type, param_tensors, g, closure)
            
        else:
            # Use your custom chain-based optimizers
            if not group["foreach"] or len(p) == 1:
                for param, grad in zip(p, g):
                    chain(self.state_, group, [grad], [param], *self.fns)
            else:
                chain(self.state_, group, g, p, *self.fns)
            
        group["caution"] = caution
        group["step"] = None
        
    def _create_global_shampoo_optimizers(self):
        """Create DistributedShampoo optimizers following AlgoPerf's approach"""
        if self._global_shampoo_optimizers:
            return  # Already created
        
        print("Creating global Shampoo optimizers...")
        
        # Collect all parameters that will use DistributedShampoo
        all_shampoo_params = []
        shampoo_groups = []
        
        for group in self.param_groups:
            
            all_shampoo_params.extend(group['params'])
            shampoo_groups.append(group)
        
        if not all_shampoo_params:
            print("No Shampoo parameters found!")
            return
        
        print(f"Total Shampoo parameters: {len(all_shampoo_params)}")
        
        # Split parameters like AlgoPerf does
        small_params = []
        large_params = []
        
        for param in all_shampoo_params:
            # Use AlgoPerf's criterion: any dimension >= 1M
            if torch.any(torch.tensor(param.shape) >= 1_000_000):
                large_params.append(param)
                print(f"Large param: {param.shape}, id={id(param)}")
            else:
                small_params.append(param)
                print(f"Small param: {param.shape}, id={id(param)}")
        
        print(f"Split: {len(small_params)} small, {len(large_params)} large")
        
        # Get configuration from first shampoo group
        if not shampoo_groups:
            print("No shampoo groups found!")
            return
            
        first_group = shampoo_groups[0]
        
        # Create optimizers following AlgoPerf's exact approach
        if small_params:
            print(f"Creating DistributedShampoo for {len(small_params)} small parameters")
            self._global_shampoo_optimizers['small'] = self._create_algoperf_shampoo(
                small_params, first_group, use_full_shampoo=True
            )
            
            # Map parameters to optimizer
            for param in small_params:
                param_id = id(param)
                self._param_to_optimizer_map[param_id] = 'small'
                print(f"Mapped small param {param.shape} (id={param_id}) to 'small'")
        
        if large_params:
            print(f"Creating DistributedShampoo for {len(large_params)} large parameters")
            self._global_shampoo_optimizers['large'] = self._create_algoperf_shampoo(
                large_params, first_group, use_full_shampoo=False
            )
            
            # Map parameters to optimizer
            for param in large_params:
                param_id = id(param)
                self._param_to_optimizer_map[param_id] = 'large'
                print(f"Mapped large param {param.shape} (id={param_id}) to 'large'")
        
        print(f"Created {len(self._global_shampoo_optimizers)} optimizers")
        print(f"Parameter map has {len(self._param_to_optimizer_map)} entries")

    def _create_algoperf_shampoo(self, params, group, use_full_shampoo=True):
        """Create DistributedShampoo exactly like AlgoPerf"""
        print(group)
        lr = group["lr"]
        try:
            weight_decay = group.get("weight_decay", 0.0)
            eps = group.get("eps", 1e-8)
       
            from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
            from optimizers.distributed_shampoo.shampoo_types import (
                AdamGraftingConfig, DDPShampooConfig, CommunicationDType
            )
            import torch.distributed as dist
            # Initialize distributed if needed
            port_rng = random.Random()
            port_rng.seed(time.time_ns())  # Seed with high-resolution timestamp
            socket = port_rng.randint(20000, 65530)
            
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='gloo',
                    init_method=f'tcp://localhost:{socket}',
                    rank=0,
                    world_size=1
                )
            
            # Create grafting config (AlgoPerf always uses Adam for grafting)
            grafting_config = AdamGraftingConfig(
                beta2=group.get("beta2", 0.999),
                epsilon=group.get("grafting_epsilon", 1e-8),
            )
            
            # Different configs for small vs large params (like AlgoPerf)
            if use_full_shampoo:
                # Small parameters - full Shampoo
                max_preconditioner_dim = group.get("max_preconditioner_dim", 1024)
                start_preconditioning_step = group.get("start_preconditioning_step", 1)
            else:
                # Large parameters - RWS AdaGrad mode
                max_preconditioner_dim = 524_288  # AlgoPerf's value for large params
                start_preconditioning_step = torch.inf  # Disable Shampoo, use only grafting
                
            optimizer = DistributedShampoo(
                params,
                lr=lr,
                betas=(group.get("beta1", 0.9), group.get("beta2", 0.999)),
                epsilon=eps,
                momentum=group.get("momentum", 0.0),
                weight_decay=weight_decay,
                max_preconditioner_dim=max_preconditioner_dim,
                precondition_frequency=int(group.get("precondition_frequency", 100)),
                start_preconditioning_step=start_preconditioning_step,
                inv_root_override=group.get("inv_root_override", 0),
                exponent_multiplier=group.get("exponent_multiplier", 1.0),
                use_nadam=group.get("use_nadam", True),
                use_nesterov=True,
                use_bias_correction=True,
                use_decoupled_weight_decay=True,
                grafting_config=grafting_config,
                use_normalized_grafting=group.get("use_normalized_grafting", False),
                use_merge_dims=True,
                use_pytorch_compile=False,
            #    distributed_config=DDPShampooConfig(
            #        communication_dtype=CommunicationDType.FP32,
            #        num_trainers_per_group=1,
            #        communicate_params=False,
            #    ),
                preconditioner_dtype=torch.float32,
                use_protected_eigh=True,
                track_root_inv_residuals=False,
            )
            
            return optimizer
            
        except Exception as e:
            print(f"Failed to create AlgoPerf-style DistributedShampoo: {e}")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, eps=eps)
        
    def step_lbfgs(self):
        """Step method specifically for L-BFGS"""
        if not hasattr(self, '_lbfgs_closure'):
            raise ValueError("L-BFGS closure not set. Call from _inner method.")
    
        return self.step(self._lbfgs_closure)
    
    def _use_external_optimizer(self, group, optimizer_type, param_tensors, grads, closure=None):
        """Handle external PyTorch optimizers using AlgoPerf approach"""
        
        if optimizer_type.lower() == "distributedshampoo":
            # Ensure global optimizers are created
            self._create_global_shampoo_optimizers()
            # Assign gradients to parameters
            for param, grad in zip(param_tensors, grads):
                param.grad = grad
            return
                
        elif optimizer_type.lower() == "lbfgs":
            optimizer = self._get_or_create_global_external_optimizer(optimizer_type)
            
            for param, grad in zip(param_tensors, grads):
                param.grad = grad
            
            self._update_optimizer_params(optimizer, group)
            
            if closure is None:
                raise ValueError("L-BFGS requires a closure function")
            
            # L-BFGS calls closure multiple times - don't interfere with gradients
            def lbfgs_closure():
                # Don't zero gradients - L-BFGS manages this
                loss = closure()
                # Verify gradients exist
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        if param.requires_grad and param.grad is None:
                            print("Warning: Parameter missing gradient in L-BFGS closure")
                            param.grad = torch.zeros_like(param)
                
                return loss
            
            try:
                optimizer.step(lbfgs_closure)
            except Exception as e:
                print(f"Error in L-BFGS optimizer: {e}")
                
                # Complete state reset on any error
                optimizer.state.clear()
                
                # Try once more with clean state
                try:
                    optimizer.step(lbfgs_closure)
                except Exception as e2:
                    print(f"L-BFGS failed after reset: {e2}")
                    # Create entirely new optimizer
                    if optimizer_type.lower() in self._global_external_optimizers:
                        del self._global_external_optimizers[optimizer_type.lower()]
                    
                    # Get fresh optimizer and try once more
                    optimizer = self._get_or_create_global_external_optimizer(optimizer_type)
                    self._update_optimizer_params(optimizer, group)
                    
                    try:
                        optimizer.step(lbfgs_closure)
                    except Exception as e3:
                        print(f"L-BFGS completely failed: {e3}")

                for param in param_tensors:
                    param.grad = None
            
        else:
            # Get the global optimizer (created once with all parameters)
            optimizer = self._get_or_create_global_external_optimizer(optimizer_type)
            # Assign gradients to the specific parameters in this group
            for param, grad in zip(param_tensors, grads):
                param.grad = grad
            # Update hyperparameters for this step
            self._update_optimizer_params(optimizer, group)
            # Step the optimizer (it will only update parameters that have gradients)
            if hasattr(optimizer, '_initialize_state'):
                optimizer._initialize_state(param_tensors)
            try:
                optimizer.step()
            except KeyError as e:
                print(f"KeyError in external optimizer {optimizer_type}: {e}")
                # Try to initialize missing state and retry
                self._fix_missing_state(optimizer, param_tensors, str(e))
                #optimizer.step()
            # Clear gradients for parameters in this group
            for param in param_tensors:
                param.grad = None

    def _initialize_lr_scheduler(self):
        """Initialize LR scheduler based on the first parameter group that needs it"""
        if self._scheduler_initialized:
            return
            
        # Find the first group that has lr_schedule enabled
        schedule_group = None
        for group in self.param_groups:
            if group.get("lr_schedule", False):
                schedule_group = group
                break
                
        if schedule_group is None:
            self._scheduler_initialized = True
            return

        # Get scheduler configuration from the group
        scheduler_type = schedule_group.get("scheduler_type", "cosine")
        total_steps = schedule_group.get("total_steps", 100000)
        
        # Handle step_decay scheduler
        if scheduler_type == "step_decay":
            decay_points = schedule_group.get("decay_points", [0.5, 0.75])
            gamma = schedule_group.get("gamma", 0.1)
            self._step_decay_fn = create_step_schedule_fn(total_steps, decay_points, gamma)
            self._scheduler_initialized = True
            return
            
        # Create a dummy optimizer for PyTorch schedulers
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        dummy_optimizer = SGD([dummy_param], lr=schedule_group["lr"])
        
        # Create the appropriate scheduler
        if scheduler_type == "cosine":
            self.lr_scheduler = CosineAnnealingLR(
                dummy_optimizer,
                T_max=total_steps,
                eta_min=schedule_group.get("min_lr", 0.0)
            )
        elif scheduler_type == "onecycle":
            self.lr_scheduler = OneCycleLR(
                dummy_optimizer,
                max_lr=schedule_group.get("max_lr", schedule_group["lr"] * 10),
                total_steps=total_steps,
                anneal_strategy=schedule_group.get("anneal_strategy", "cos"),
                div_factor=schedule_group.get("div_factor", 25),
                final_div_factor=schedule_group.get("final_div_factor", 1e4)
            )
        elif scheduler_type == "exponential":
            self.lr_scheduler = ExponentialLR(
                dummy_optimizer,
                gamma=schedule_group.get("gamma", 0.95)
            )
        elif scheduler_type == "step":
            self.lr_scheduler = StepLR(
                dummy_optimizer,
                step_size=schedule_group.get("step_size", 30000),
                gamma=schedule_group.get("gamma", 0.1)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
            
        self._scheduler_initialized = True
    
    @property
    def fns(self):
        return self._fns

    @fns.setter
    def fns(self, value):
        self._fns = value
        self._set_indices(retain=True)

    def _set_indices(self, retain=True):
        self._fns = set_indices(self.fns, retain)

         
    def _get_or_create_global_external_optimizer(self, optimizer_type):
        """Create external optimizer once with ALL parameters"""
        if optimizer_type not in self._global_external_optimizers:
            # Collect ALL parameters from ALL groups
            all_params = []
            for group in self.param_groups:
                all_params.extend(group['params'])
            
            # Use the first group's settings as defaults
            first_group = self.param_groups[0]
            
            self._global_external_optimizers[optimizer_type] = self._create_external_optimizer(
                optimizer_type, all_params, first_group
            )
            if optimizer_type == 'distributedshampoo':
                for group in self._global_external_optimizers[optimizer_type].param_groups:
                    group["optimizer_type"] = "distributedshampoo"

        return self._global_external_optimizers[optimizer_type]

    def _fix_missing_state(self, optimizer, param_tensors, error_msg):
        """Fix missing state in external optimizers"""
        if "MOMENTUM_LIST" in error_msg:
            # Initialize momentum state for all parameters
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        state = optimizer.state[p]
                        if 'momentum_buffer' not in state:
                            state['momentum_buffer'] = torch.zeros_like(p.data)
        
        if "MASKED_MOMENTUM_LIST" in error_msg or "masked_momentum_list" in error_msg:
            
            # Initialize momentum state for all parameters
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        
                        state = optimizer.state[p]
                        if 'masked_momentum_list' not in state:
                            print("YES")
                            state['masked_momentum_list'] = torch.zeros_like(p.data)
                            


    def _create_external_optimizer(self, optimizer_type, params, group):
        """Factory method to create different external optimizers"""

        # Ensure params is a list
        if not isinstance(params, list):
            params = [params]

        # Common hyperparameters
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        eps = group.get("eps", 1e-8)

        if optimizer_type.lower() == "adamw":
            
            return torch.optim.AdamW(
                params,
                lr=lr,
                betas=(utils.get_beta1(group), utils.get_beta2(group)),
                eps=eps,
                weight_decay=weight_decay
            )
        
        elif optimizer_type.lower() == "nadamw":
            return NAdamW(
            params,
            lr=lr,
            betas=(utils.get_beta1(group), utils.get_beta2(group)),
            eps=eps,
            weight_decay=weight_decay
        )
        elif optimizer_type.lower() == "distributedshampoo":
            # Import Distributed Shampoo
            from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
            from optimizers.distributed_shampoo.shampoo_types import (
                AdamGraftingConfig,
                AdaGradGraftingConfig,
                DDPShampooConfig,
                GraftingConfig,
                RMSpropGraftingConfig,
                SGDGraftingConfig,
            )
            import torch.distributed as dist
            torch.backends.cuda.preferred_linalg_library()
            # Initialize distributed processing for single process
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='gloo',  # Use 'nccl' if you have CUDA
                    init_method='tcp://localhost:23456',
                    rank=0,
                    world_size=1
            )
            # Create grafting configuration
            grafting_type = group.get("grafting_type", "ADAM")

            if grafting_type == "ADAM":
                grafting_config = AdamGraftingConfig(
                    beta2=0.999,
                    epsilon=1e-8,
                )
            elif grafting_type == "ADAGRAD":
                grafting_config = AdaGradGraftingConfig(
                    epsilon=1e-8,
                )
            elif grafting_type == "SGD":
                grafting_config = SGDGraftingConfig()
            elif grafting_type == "RMSPROP":
                grafting_config = RMSpropGraftingConfig(
                    beta2=0.999,
                    epsilon=1e-8,
                )
            else:  
                grafting_config = None
            max_preconditioner_dim = group.get("max_preconditioner_dim", 1024)
            precondition_frequency = group.get("precondition_frequency", 1)
            print(f'precondition_freq: {precondition_frequency}')
            start_preconditioning_step = group.get("start_preconditioning_step", 1)
           
            # Create DistributedShampoo with minimal required parameters
            return DistributedShampoo(
                params,
                lr=lr,
                betas=(utils.get_beta1(group), utils.get_beta2(group)),
                epsilon=eps,
                momentum=0.0,
                weight_decay=weight_decay,
                max_preconditioner_dim=max_preconditioner_dim,
                precondition_frequency=precondition_frequency,
                start_preconditioning_step=start_preconditioning_step,
                inv_root_override=0,
                exponent_multiplier=1.0,
                use_nadam=False,
                use_nesterov=False,
                use_bias_correction=True,
                use_decoupled_weight_decay=True,
                grafting_config=grafting_config,
                use_normalized_grafting=False,
                use_merge_dims=True,
                use_pytorch_compile=False,
                distributed_config=None,  
                preconditioner_dtype=torch.float32,
                use_protected_eigh=True,
                track_root_inv_residuals=False,
            )

        elif optimizer_type.lower() == "adam":
            return torch.optim.Adam(
                params,
                lr=lr,
                betas=(utils.get_beta1(group), utils.get_beta2(group)),
                eps=eps,
                weight_decay=weight_decay
            )

        elif optimizer_type.lower() == "sgd":
            momentum = utils.get_beta1(group)
            return torch.optim.SGD(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )

        elif optimizer_type.lower() == "rmsprop":
            alpha = group.get("alpha", 0.99)
            momentum = utils.get_beta1(group)
            return torch.optim.RMSprop(
                params,
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum
            )

        elif optimizer_type.lower() == "adagrad":
            return torch.optim.Adagrad(
                params,
                lr=lr,
                eps=eps,
                weight_decay=weight_decay
            )

        elif optimizer_type.lower() == "adadelta":
            rho = group.get("rho", 0.9)
            return torch.optim.Adadelta(
                params,
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay
            )

        elif optimizer_type.lower() == "adamax":
            return torch.optim.Adamax(
                params,
                lr=lr,
                betas=(utils.get_beta1(group), utils.get_beta2(group)),
                eps=eps,
                weight_decay=weight_decay
            )
        
        elif optimizer_type.lower() == "lbfgs":
            # L-BFGS specific parameters with proper defaults
            max_iter = group.get("max_iter", 20)
            max_eval = group.get("max_eval", max_iter * 2)  
            tolerance_grad = group.get("tolerance_grad", 1e-7)
            tolerance_change = group.get("tolerance_change", 1e-9)
            history_size = group.get("history_size", 100)
            line_search_fn = group.get("line_search_fn", "strong_wolfe")
            
            # Ensure no None values that could cause comparison errors
            if max_eval is None:
                max_eval = max_iter * 2
            
            return torch.optim.LBFGS(
                params,
                lr=lr,
                max_iter=max_iter,
                max_eval=max_eval,  
                tolerance_grad=tolerance_grad,
                tolerance_change=tolerance_change,
                history_size=history_size,
                line_search_fn=line_search_fn
            )
        else:
            raise ValueError(f"Unsupported external optimizer: {optimizer_type}")

    def _update_optimizer_params(self, optimizer, group):
        """Update optimizer hyperparameters to match current group settings"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = group["lr"]
            param_group['weight_decay'] = group["weight_decay"]
            
            # Update optimizer-specific parameters
            if 'betas' in param_group:
                param_group['betas'] = (utils.get_beta1(group), utils.get_beta2(group))
            if 'eps' in param_group:
                param_group['eps'] = group.get("eps", 1e-8)
            if 'momentum' in param_group:
                param_group['momentum'] = utils.get_beta1(group)
            if 'alpha' in param_group:
                param_group['alpha'] = group.get("alpha", 0.99)
            if 'rho' in param_group:
                param_group['rho'] = group.get("rho", 0.9)
                
            # L-BFGS specific parameters - ensure no None values
            if 'max_iter' in param_group:
                max_iter = group.get("max_iter", 20)
                param_group['max_iter'] = max_iter
            if 'max_eval' in param_group:
                max_eval = group.get("max_eval", None)
                if max_eval is None:
                    max_eval = param_group.get('max_iter', 20) * 2
                param_group['max_eval'] = max_eval
            if 'tolerance_grad' in param_group:
                param_group['tolerance_grad'] = group.get("tolerance_grad", 1e-7)
            if 'tolerance_change' in param_group:
                param_group['tolerance_change'] = group.get("tolerance_change", 1e-9)
            if 'history_size' in param_group:
                param_group['history_size'] = group.get("history_size", 100)
            if 'line_search_fn' in param_group:
                param_group['line_search_fn'] = group.get("line_search_fn", None)

    def _step_global_shampoo_optimizers(self):
        """Step all global Shampoo optimizers once per overall step"""
        if not self._global_shampoo_optimizers:
            return
            
        # Inside _step_global_shampoo_optimizers():
        for k, opt in self._global_shampoo_optimizers.items():
            for group in opt.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        print(f"[{k}] grad norm before step: {p.grad.norm().item()}")
            opt.step()


    def _clear_all_gradients(self):
        """Clear gradients for all parameters"""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad = None

    def _step_global_shampoo_optimizers(self):
        """Step all global Shampoo optimizers once per overall step"""
        if not self._global_shampoo_optimizers:
            return
            
        for optimizer_key, optimizer in self._global_shampoo_optimizers.items():
            try:
                optimizer.step()
            except Exception as e:
                print(f"Error stepping {optimizer_key} DistributedShampoo: {e}")

    def _clear_shampoo_gradients(self):
        """Clear gradients for all Shampoo parameters"""
        for group in self.param_groups:
            if group.get("optimizer_type", "").lower() == "distributedshampoo":
                for param in group['params']:
                    if param.grad is not None:
                        param.grad = None


class CosineSandwichSchedule:
    def __init__(self, total_steps, gamma, warmup_ratio=0.125, hold_ratio=0.5):
        self.total_steps = total_steps
        self.gamma = gamma

        self.T1 = int(total_steps * warmup_ratio)
        self.T2 = int(total_steps * hold_ratio)
        self.T3 = total_steps - self.T1 - self.T2

    def get_gamma(self, step):
        if step < self.T1:
            # Linear ramp-up: 0 → gamma
            return (step / self.T1) * self.gamma
        elif step < self.T1 + self.T2:
            # Hold at gamma
            return self.gamma
        elif step < self.total_steps:
            # Linear decay: gamma → 0
            t = step - (self.T1 + self.T2)
            return self.gamma * (1 - t / self.T3)
        else:
            # After schedule ends, stay at 0
            return 0.0

str_or_fn = Union[str, callable, None, Literal[use_default]]


def _get_clip_fn(name: str_or_fn, default_val: str_or_fn):
    name = default(name, default_val)
    if callable(name):
        return name
    elif name not in (
        "l2_clip_",
        "rmsnorm_clip_",
        "trust_region_clip_",
        "a_law_compress",
        "mu_law_compress",
    ):
        raise ValueError(f"Clipping function {name} not found")
    return getattr(utils, name)


def default(a, b):
    return b if a is use_default else a


# not supported: update_by_schedule_free, scale_by_soap, scale_by_exp_avg_sq
_scale_to_update_map = {
    scale_by_delayed_psgd.get_fn(): update_by_delayed_psgd,  #
    scale_by_psgd.get_fn(): update_by_psgd,  #
    scale_by_psgd_lra.get_fn(): update_by_psgd_lra,  #
    scale_by_delayed_psgd_lra.get_fn(): update_by_delayed_psgd_lra,  #
    scale_by_adam.get_fn(): update_by_adam,  #
    scale_by_laprop.get_fn(): update_by_laprop,  #
    scale_by_adopt.get_fn(): update_by_adopt,  #
}
_scale_to_update_map_inv = {
    update_by_delayed_psgd.get_fn(): scale_by_delayed_psgd,  #
    update_by_psgd.get_fn(): scale_by_psgd,  #
    update_by_psgd_lra.get_fn(): scale_by_psgd_lra,  #
    update_by_delayed_psgd_lra.get_fn(): scale_by_delayed_psgd_lra,  #
    update_by_adam.get_fn(): scale_by_adam,  #
    update_by_laprop.get_fn(): scale_by_laprop,  #
    update_by_adopt.get_fn(): scale_by_adopt,  #
}


class BaseOpt(ChainOpt):
    """
    Base Optimizer

    compile_step: bool = False
    Whether to change some internals to try to make the optimizer compilable
    This does not compile the step by itself and breaks some optimizers loudly (e.g. SOAP)

    promote: bool = False
    Whether to promote the gradients to fp32 before applying the optimizer
    Improves update quality for low-precision parameters, but increases costs
    Compiling the optimizer step would reduce memory and compute. Alternatively, `foreach=False` decreases memory at the cost of runtime

    gradient_clipping: str_or_fn = None
    The function to use for clipping the incoming gradients, before any other transformations.
    This is syntactic sugar, equivalent to manually passing the function as the first element of the optimizer chain.

    update_clipping: str_or_fn = None
    The function to use for clipping the outgoing updates before applying them, after all other transformations.
    This will turn off
    This is syntactic sugar, equivalent to manually passing the function as the last element of the optimizer chain.

    requires_prev_minibatch: bool = False
    Whether this optimizer requires access to the previous minibatch for computation.
    When True, the training loop should provide both current and previous batch data.
    """
    
    gradient_clipping: str_or_fn = None
    update_clipping: str_or_fn = None
    palm: bool = False
    auto_fuse: bool = True
    requires_prev_minibatch: bool = False  
    requires_prev_model: bool = False
    def __init__(
        self,
        params,
        defaults,
        foreach: bool,
        gradient_clipping: str_or_fn,
        update_clipping: str_or_fn,
        palm: bool = use_default, 
        *fns,
        compile_step: bool = use_default,
        promote: bool = use_default,
        requires_prev_minibatch: bool = use_default, 
        requires_prev_model: bool = use_default, 
    ):
        
        self.compile_step = default(compile_step, self.compile_step)
        self.promote = default(promote, self.promote)

        self.requires_prev_minibatch = default(requires_prev_minibatch, self.requires_prev_minibatch)  
        defaults['requires_prev_minibatch'] = self.requires_prev_minibatch 

        self.requires_prev_model = default(requires_prev_model, self.requires_prev_model)  
        defaults['requires_prev_model'] = self.requires_prev_model

        if not fns:
            raise ValueError("No functions provided. If that's on purpose (SGD-like), use `identity`")

        args, kwargs = None, None
        fn = fns[-1]
        if isinstance(fn, functools.partial):
            fn, args, kwargs = fn.func, fn.args, fn.keywords
        if isinstance(fn, FunctionTransform):
            fn = fn.get_fn()

        if default(update_clipping, self.update_clipping) is None:
            if self.auto_fuse:
                if fn in _scale_to_update_map:
                    fn = _scale_to_update_map[fn]
                    if args is not None:
                        fn = functools.partial(fn, *args, **kwargs)
                    fns = tuple(fns)[:-1] + (fn,)
        elif fn in _scale_to_update_map_inv:
            if not self.auto_fuse:
                raise ValueError(
                    "update_clipping is currently not compatible with update_by_* functions. "
                    "Manually select scale_by_* functions or set auto_fuse=True."
                )
            fn = _scale_to_update_map_inv[fn]
            if args is not None:
                fn = functools.partial(fn, *args, **kwargs)
            fns = tuple(fns)[:-1] + (fn,)

        self.compile_step = default(compile_step, self.compile_step)
        self.promote = default(promote, self.promote)
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
            group["train_mode"] = train_mode = not group.get("train_mode")
            beta1 = utils.get_beta1(group)
            if beta1 > 0 and not train_mode:
                for p in group["params"]:
                    state = self.state_(p)
                    if "z" in state:
                        # Set p.data to x
                        z = utils.promote(state["z"])
                        p32 = utils.promote(p.data)
                        p32.lerp_(end=z, weight=1 - 1 / beta1)
                        utils.copy_stochastic_(p.data, p32)

    def train(self):
        for group in self.param_groups:
            group["train_mode"] = train_mode = not group.get("train_mode")
            beta1 = utils.get_beta1(group)
            if beta1 > 0 and train_mode:
                for p in group["params"]:
                    state = self.state_(p)
                    if "z" in state:
                        z = utils.promote(state["z"])
                        p32 = utils.promote(p.data)
                        p32.lerp_(end=z, weight=1 - beta1)
                        utils.copy_stochastic_(p.data, p32)


class MSAM(BaseOpt):
    def eval(self):
        for group in self.param_groups:
            group["train_mode"] = train_mode = not group.get("train_mode")
            if not train_mode:
                for p in group["params"]:
                    state = self.state_(p)
                    if "z" in state:
                        p_copy = p.data.clone()
                        utils.copy_stochastic_(p.data, state["z"])
                        utils.copy_stochastic_(state["z"], p_copy)

    def train(self):
        for group in self.param_groups:
            group["train_mode"] = train_mode = not group.get("train_mode")
            if train_mode:
                for p in group["params"]:
                    state = self.state_(p)
                    if "z" in state:
                        p_copy = p.data.clone()
                        utils.copy_stochastic_(p.data, state["z"])
                        utils.copy_stochastic_(state["z"], p_copy)
