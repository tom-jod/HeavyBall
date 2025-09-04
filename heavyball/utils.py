import collections
import contextlib
import functools
import gc
import inspect
import math
import pickle
import random
import re
import string
import warnings
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch._dynamo.exc import TorchDynamoException
from torch.backends import cudnn, opt_einsum
from torch.utils._pytree import tree_map

compile_mode = "max-autotune-no-cudagraphs"
dynamic = False
compile_mode_recommended_to_none = None
zeroth_power_mode = "newtonschulz"
precise_zeroth_power_mode = "qr"  # or svd
tiny_bf16 = torch.finfo(torch.bfloat16).tiny
_cudnn_double_backward_pattern = re.compile(
    r"the derivative for .* is not implemented\. Double backwards .* To run double backwards"
)
_torch_compile_double_backward_pattern = re.compile(r"compile.*does not currently support double backward")
_fd_error = (
    "You can accelerate startup by globally enabling finite_differences first "  #
    "(via opt.finite_differences=True or by subclassing it)\n"
    "Original Error: "
)


def decorator(func):
    compiled = None

    @functools.wraps(func)
    def _fn(*args, **kwargs):
        if is_compiling() or compile_mode_recommended_to_none is None:
            return func(*args, **kwargs)
        nonlocal compiled
        if compiled is None:
            compiled = torch.compile(fullgraph=True, dynamic=dynamic, mode=compile_mode_recommended_to_none)(func)
        return compiled(*args, **kwargs)

    return _fn


def decorator_knowngood(func: Callable, fullgraph: bool = True):
    compiled = None

    @functools.wraps(func)
    def _fn(*args, **kwargs):
        if is_compiling() or compile_mode is None:
            return func(*args, **kwargs)
        nonlocal compiled
        if compiled is None:
            compiled = torch.compile(fullgraph=fullgraph, dynamic=dynamic, mode=compile_mode)(func)
        return compiled(*args, **kwargs)

    return _fn


einsum_base = string.ascii_lowercase


@decorator_knowngood
def compiled_einsum(expr, *args):
    """
    this is necessary to avoid the slowdown introduced by uncompiled einsum
    uncompiled einsum is twice as slow if we add three 1-sized dimensions
    for more, see https://gist.github.com/ClashLuke/a9530f1b9ba4e525369e2dba48528957
    """
    return torch.einsum(expr, *args)


@decorator_knowngood
def _compilable_schedule_free_(
    p: List[Tensor],
    z: List[Tensor],
    ckp1: Tensor,
    update: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    decay: float,
    grad: List[Tensor],
    caution,
):
    for op, oz, u_, g_ in zip(p, z, update, grad):
        u_ = u_.view_as(op)
        p_, z_, u_ = map(promote, (op, oz, u_))
        if decay != 0:
            u_ = u_ + p_ * decay
        if caution:
            u_ = _compilable_cautioning(u_, g_)
        p_ = p_.lerp(z_, ckp1)
        p_ = p_ + u_ * (lr * (beta1 * (1 - ckp1)) - lr)
        z_ = z_ + u_ * -lr
        copy_stochastic_(op, p_)
        copy_stochastic_(oz, z_)


def schedule_free_(
    lr: float,
    weight_lr_power: float,
    weight_sum: float,
    beta1: float,
    parameters: List[Tensor],
    z: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    caution: bool = False,
    r: float = 0.0,
    step: int = 0,
    decay: float = 0.0,
):
    weight = abs(lr) ** weight_lr_power * max(step, 1) ** r
    weight_sum = weight_sum + weight

    try:
        ckp1 = weight / weight_sum
    except ZeroDivisionError:
        ckp1 = 0

    update, parameters, z, grad = list_guard(update, parameters, z, grad)
    lr, ckp1, beta1 = scalar_guard(lr, ckp1, beta1, grad[0])
    _compilable_schedule_free_(parameters, z, ckp1, update, lr, beta1, decay, grad, caution)
    return weight_sum


@decorator_knowngood
def _compilable_msam(
    lr: Tensor,
    beta1: Tensor,
    param: List[Tensor],
    z: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    exp_avg: List[Tensor],
    caution: bool,
    decay: Tensor,
    sam_step_size: Tensor,
):
    exp_avg32 = _lerp(exp_avg, update, beta1)
    for u_, g_, z_, p_ in zip(exp_avg32, grad, z, param):
        u_ = u_.view_as(z_)
        z32_ = promote(z_)
        if caution:
            u_ = _compilable_cautioning(promote(g_), u_)
        z32_ = z32_ * (1 - decay * lr) + u_ * -lr
        copy_stochastic_(z_, z32_)
        copy_stochastic_(p_, z32_ + u_ / u_.norm().clamp(min=1e-8) * -sam_step_size)


def msam_(
    lr: float,
    beta1: float,
    param: List[Tensor],
    z: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    exp_avg: List[Tensor],
    caution: bool,
    weight_decay: float,
    sam_step_size: float,
):
    param, z, update, grad, exp_avg = list_guard(param, z, update, grad, exp_avg)
    lr, beta1, weight_decay, sam_step_size = scalar_guard(lr, beta1, weight_decay, sam_step_size, exp_avg[0])
    _compilable_msam(lr, beta1, param, z, update, grad, exp_avg, caution, weight_decay, sam_step_size)


def append_or_extend(base, new):
    if isinstance(new, list):
        base.extend(new)
    else:
        base.append(new)


def dim_merger(grad, max_precond_dim, split: bool = False):
    """
    Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.

    we don't want to merge fan-in into fan-out,
    but we want to merge conv kernels into fan-in or at least merge the kernel
    so, [128, 64, 3, 3] should result in [128, 576] or [128, 64, 9] instead of [73728] or [8192, 3, 3] the baseline
    would've done

    By @francois-rozet (commit: 68cde41eaf7e73b4c46eacb6a944865dcc081f1d), re-commited due to faulty merge
    """
    new_shape = []
    cum_size = 1

    for s in grad.shape[1:][::-1]:
        temp_size = cum_size * s
        if temp_size > max_precond_dim:
            if cum_size > 1:
                new_shape.append(cum_size)
                cum_size = s
            else:
                new_shape.append(s)
                cum_size = 1
        else:
            cum_size = temp_size

    if cum_size > 1:
        new_shape.append(cum_size)

    new_shape = [grad.shape[0], *new_shape[::-1]]
    new_grad = grad.reshape(new_shape)
    if not split:
        return new_grad.to(memory_format=torch.contiguous_format).contiguous()

    grads = [new_grad]
    for i, sh in reversed(list(enumerate(new_shape[:]))):
        if sh == 1:
            grads = [g.squeeze(dim=i) for g in grads]
            continue
        if sh <= max_precond_dim:
            continue
        grads = [a for g in grads for a in g.split(max_precond_dim, dim=i)]
    if len(grads) == 1:
        return new_grad.to(memory_format=torch.contiguous_format).contiguous()
    new_grads = []
    for g in grads:
        append_or_extend(new_grads, dim_merger(g, max_precond_dim, split))
    return new_grads


def beta_debias(beta, step):
    return 1 - (1 - beta) / (1 - beta**step)


def eps_sqrt(item, eps):
    return item.sqrt().clamp(min=eps)


@decorator_knowngood
def _compilable_exp_avg_sq_(
    state: List[Tensor], grad: List[Tensor], beta2: Tensor, eps: Tensor, out: List[Optional[Tensor]]
):
    g32 = promote(grad)
    s32 = _lerp(state, torch._foreach_mul(g32, g32), beta2)

    denom = [eps_sqrt(d, eps) for d in s32]

    if out[0] is None:
        return denom

    copy_stochastic_list_(out, denom)
    return out

@decorator_knowngood
def _compilable_exp_avg_sq_(
    state: List[Tensor], grad: List[Tensor], beta2: Tensor, eps: Tensor, out: List[Optional[Tensor]]
):
    # Filter out None gradients before processing
    valid_indices = []
    valid_grad = []
    valid_state = []
    
    for i, (g, s) in enumerate(zip(grad, state)):
        if g is not None:
            valid_indices.append(i)
            valid_grad.append(g)
            valid_state.append(s)
    
    if not valid_grad:
        # No valid gradients, return original state
        return state
    
    # Process only valid gradients
    g32 = [promote(g) for g in valid_grad]
    s32 = _lerp(valid_state, torch._foreach_mul(g32, g32), beta2)
    
    denom = [eps_sqrt(d, eps) for d in s32]
    
    # Reconstruct the full result list with None values in correct positions
    result = [None] * len(state)
    for i, idx in enumerate(valid_indices):
        result[idx] = denom[i]
    
    return result


def exp_avg_sq_(state, grad, beta2, eps, out=None):
    state, grad, out = list_guard(state, grad, out)
    beta2, eps = scalar_guard(beta2, eps, state[0])
    return _compilable_exp_avg_sq_(state, grad, beta2, eps, out)


@decorator_knowngood
def _compilable_scale_by_exp_avg_sq_(state: List[Tensor], grad: List[Tensor], beta2: Tensor, eps: Tensor):
    g32 = promote(grad)
    denom = _compilable_exp_avg_sq_(state, g32, beta2, eps, [None])
    out = torch._foreach_div(g32, denom)
    copy_stochastic_list_(grad, out)


def scale_by_exp_avg_sq_(exp_avg_sq, grad, beta2, eps):
    grad, exp_avg_sq = list_guard(grad, exp_avg_sq)
    beta2, eps = scalar_guard(beta2, eps, grad[0])
    _compilable_scale_by_exp_avg_sq_(exp_avg_sq, grad, beta2, eps)
    return grad


@decorator_knowngood
def _compilable_exp_avg_(state, grad, beta):
    lerped = _lerp(state, grad, beta)
    copy_stochastic_list_(grad, lerped)


def scale_by_exp_avg_(state, grad, beta):
    state, grad = list_guard(state, grad)
    beta = scalar_guard(beta, state[0])
    _compilable_exp_avg_(state, grad, beta)
    return grad


@decorator_knowngood
def _compilable_agc_(parameters: List[Tensor], gradients: List[Tensor], clip_val: float, minimum: float, eps: float):
    p32, g32 = [list(map(promote, x)) for x in (parameters, gradients)]
    p_norm = torch._foreach_norm(p32)
    g_norm = torch._foreach_norm(g32)
    p_norm = torch._foreach_maximum(p_norm, minimum)
    g_norm = torch._foreach_maximum(g_norm, eps)
    p_norm = torch._foreach_div(p_norm, g_norm)
    p_norm = torch._foreach_mul(p_norm, clip_val)
    p_norm = torch._foreach_minimum(p_norm, 1)
    g32 = torch._foreach_mul(g32, p_norm)
    copy_stochastic_list_(gradients, g32)


def adaptive_gradient_clipping_(
    parameters: List[Tensor], gradients: List[Tensor], clip_val: float, minimum: float = 1e-3, eps: float = 1e-8
):
    if clip_val <= 0:
        return gradients
    parameters, gradients = list_guard(parameters, gradients)
    clip_val = scalar_guard(clip_val, parameters[0])
    _compilable_agc_(parameters, gradients, clip_val, minimum, eps)
    return gradients


def is_compiling():
    try:
        return torch.compiler.is_compiling()
    except TorchDynamoException:
        return True


def set_(dst: Tensor, src: Tensor):
    dst.copy_(src)


def clean():
    torch.cuda.empty_cache()
    gc.collect()


def _ignore_warning(msg):
    warnings.filterwarnings("ignore", f".*{re.escape(msg)}.*")


def set_torch(benchmark_limit: int = 32, einsum_strategy: str = "auto-hq"):
    import opt_einsum as _opt_einsum

    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.benchmark_limit = benchmark_limit
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")  # highest: FP32, high: TF32, medium: bf16
    opt_einsum.set_flags(True)
    if einsum_strategy == "heavyball":
        opt_einsum.strategy = "auto-hq"
        choices = _opt_einsum.paths._AUTO_HQ_CHOICES
        for max_val, fn in ((20, _opt_einsum.paths.dynamic_programming), (64, 512), (128, 256)):
            if isinstance(fn, int):
                fn = functools.partial(_opt_einsum.path_random.random_greedy, max_repeats=fn)
            for i in range(max(choices.keys()), max_val):
                if i not in choices:
                    choices[i] = fn
    else:
        opt_einsum.strategy = einsum_strategy

    # Torch calls these for 2nd-order optimization in HeavyBall, but they are explicitly handled.
    _ignore_warning(
        "Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient which can cause a memory leak"
    )
    _ignore_warning(
        "We recommend using autograd.grad when creating the graph to avoid this. If you have to use this function, make sure to reset the .grad fields of your parameters to None after use to break the cycle and avoid the leak"
    )
    _ignore_warning(
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead."
    )


@decorator
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16 if G.dtype != torch.float64 else G.dtype)  # Preserve float64 if present
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


@decorator_knowngood
def _compilable_heavyball_momentum_(state, grad, beta):
    s32, g32 = [list(map(promote, x)) for x in (state, grad)]
    s32 = torch._foreach_mul(s32, beta)
    s32 = torch._foreach_add(s32, g32)
    copy_stochastic_list_(state, s32)
    copy_stochastic_list_(grad, s32)


@decorator_knowngood
def _compilable_nesterov_momentum_(state, grad, beta):
    s32, g32 = [list(map(promote, x)) for x in (state, grad)]
    s32 = torch._foreach_mul(s32, beta)
    s32 = torch._foreach_add(s32, g32)
    g32 = [g + s * beta for g, s in zip(g32, s32)]
    copy_stochastic_list_(state, s32)
    copy_stochastic_list_(grad, g32)


def heavyball_momentum(state, grad, beta):
    state, grad = list_guard(state, grad)
    beta = scalar_guard(beta, state[0])
    _compilable_heavyball_momentum_(state, grad, beta)
    return grad


def nesterov_momentum(state, grad, beta):
    state, grad = list_guard(state, grad)
    beta = scalar_guard(beta, state[0])
    _compilable_nesterov_momentum_(state, grad, beta)
    return grad


@decorator_knowngood
def _compilable_nesterov_ema_(state, grad, beta):
    ema32 = _lerp(state, grad, beta)
    stochastic_add_(grad, ema32, 1)


def nesterov_ema(state, grad, beta):
    state, grad = list_guard(state, grad)
    beta = scalar_guard(beta, state[0])
    _compilable_nesterov_ema_(state, grad, beta)
    return grad


@decorator_knowngood
def _compilable_grafting(magnitude, direction):
    return direction * (magnitude.norm() / direction.norm().clamp(min=1e-6))


@decorator_knowngood
def _compilable_orthogonal_(x: Tensor, mode: str, out: Tensor | None, scale_mode: str):
    if mode == "newtonschulz" or x.shape[0] != x.shape[1]:
        y = zeropower_via_newtonschulz5(x, 5)
    elif mode == "qr":
        y = torch.linalg.qr(promote(x)).Q
    elif mode == "svd":
        u, _s, v = torch.linalg.svd(promote(x))
        y = u @ v.T
    else:
        raise NotImplementedError(f"Unknown zeroth_power_mode: {mode}")
    if scale_mode == "none":
        pass
    elif scale_mode == "scale":
        y *= max(1, x.size(0) / x.size(1)) ** 0.5
    elif scale_mode == "graft":
        y = _compilable_grafting(x, y)
    else:
        raise NotImplementedError(f"Unknown scale_mode: {scale_mode}")
    if out is None:
        return y

    set_(out, y)


def inplace_orthogonal_(x: Tensor, mode: str | None = None, out: Tensor | None = None, scale_mode: str = "none"):
    return _compilable_orthogonal_(x, mode or zeroth_power_mode, out, scale_mode)


@decorator_knowngood
def _compilable_scatter_set(target, source, index):
    target[:] = source.contiguous()[index].reshape_as(target)


# @decorator_knowngood
def get_orthogonal_matrix_QR(GG: List[Tensor], Q: List[Tensor], exp_avg: Optional[Tensor] = None):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration
    followed by torch.linalg.qr decomposition, and updates exp_avg in-place from old to new eigenspace.

    :param GG: List of accumulated gradient outer products.
    :param Q: List of current eigenbases (updated in-place to Q_new).
    :param exp_avg: Exponential moving average in the old eigenspace (updated in-place if provided).
    """
       # Handle None Q
    if Q is None:
        #print("Warning: Q is None in get_orthogonal_matrix_QR, skipping update")
        return
        
    # Handle empty GG
    if isinstance(GG, list) and not GG:
        print("Warning: GG is empty in get_orthogonal_matrix_QR, skipping update")
        return
        
    # Handle empty Q
    if isinstance(Q, list) and not Q:
        print("Warning: Q is empty in get_orthogonal_matrix_QR, skipping update")
        return
    
    # Validate dimensions
    if exp_avg is not None and exp_avg.dim() != len(Q):
        print(f"Warning: exp_avg dim {exp_avg.dim()} does not match Q length {len(Q)}, skipping update")
        return
    
    if exp_avg is not None and exp_avg.dim() == 0:
        Q.clear()
        return

    if isinstance(Q, list) and not Q:
        return

    if exp_avg is not None and exp_avg.dim() != len(Q):
        raise ValueError(f"exp_avg dim {exp_avg.dim()} does not match Q length {len(Q)}")

    new_qs = []

    for m, q in zip(GG, Q):
        if m is None:
            new_qs.append(None)
            continue

        m = promote(m.data)
        q_old = promote(q.data)

        tmp = m @ q_old
        est_eig = compiled_einsum("ij,ij->j", q_old, tmp)
        sort_idx = torch.argsort(est_eig, descending=True)

        tmp[:, sort_idx] = inplace_orthogonal_(tmp[:, sort_idx], precise_zeroth_power_mode)
        new_qs.append(tmp)

    if exp_avg is None:
        for q, q_new in zip(Q, new_qs):
            copy_stochastic_(q, q_new)
        return

    assert exp_avg.ndim < 13, "exp_avg.ndim must be less than 13"
    in_str = einsum_base[: exp_avg.dim()]
    out_str = einsum_base[exp_avg.dim() : 2 * exp_avg.dim()]

    from_shampoo = ",".join([o + i for m, i, o in zip(Q, in_str, in_str.upper()) if m is not None])
    if not from_shampoo:
        return

    to_shampoo = ",".join([i + o for m, i, o in zip(new_qs, in_str.upper(), out_str) if m is not None])
    out_str = "".join([o if o in to_shampoo else i for i, o in zip(in_str, out_str)])

    subscripts = f"{in_str},{from_shampoo},{to_shampoo}->{out_str}"
    exp_avg_new = compiled_einsum(
        subscripts, exp_avg, *[q for q in Q if q is not None], *[q for q in new_qs if q is not None]
    )
    copy_stochastic_(exp_avg, exp_avg_new)

    for q, q_new in zip(Q, new_qs):
        if q is not None:
            copy_stochastic_(q, q_new)

def get_orthogonal_matrix(mat):
    """
    Robust orthogonal matrix computation with comprehensive error handling
    """
    print(f"get_orthogonal_matrix called with mat: {mat}")
    
    # Handle empty or None matrices
    if not mat or all(m is None for m in mat):
        print("All matrices are None or empty")
        return []
    
    # Filter out None matrices
    valid_matrices = [m for m in mat if m is not None]
    if not valid_matrices:
        print("No valid matrices found")
        return []
    
    print(f"Processing {len(valid_matrices)} valid matrices")
    
    final = []
    
    for i, matrix_to_decompose in enumerate(valid_matrices):
        print(f"Processing matrix {i} with shape {matrix_to_decompose.shape}")
        
        try:
            # Validate input matrix
            if matrix_to_decompose is None:
                print(f"Matrix {i} is None")
                final.append(None)
                continue
                
            # Check for invalid values before any operations
            if not torch.isfinite(matrix_to_decompose).all():
                print(f"Matrix {i} contains non-finite values")
                device = matrix_to_decompose.device
                dtype = matrix_to_decompose.dtype
                final.append(torch.eye(matrix_to_decompose.shape[0], device=device, dtype=dtype))
                continue
            
            # Store original properties
            original_device = matrix_to_decompose.device
            original_dtype = matrix_to_decompose.dtype
            
            print(f"Matrix {i} stats: min={matrix_to_decompose.min():.6f}, max={matrix_to_decompose.max():.6f}")
            print(f"Matrix {i} condition number estimate: {torch.linalg.cond(matrix_to_decompose):.6f}")
            
            # Try eigendecomposition
            m = matrix_to_decompose.clone()
            success = False
            eigvec = None
            
            # Method 1: Try on original device with original dtype
            try:
                print(f"Attempting eigendecomposition for matrix {i}")
                eps = 1e-30
                eye = torch.eye(m.shape[0], device=m.device, dtype=m.dtype)
                eigval, eigvec = torch.linalg.eigh(m + eps * eye)
                print(f"Eigendecomposition successful for matrix {i}")
                print(f"Eigenvalues range: {eigval.min():.6f} to {eigval.max():.6f}")
                success = True
            except (RuntimeError, torch._C._LinAlgError) as e:
                print(f"Eigendecomposition failed for matrix {i}: {e}")
                if "illegal memory access" in str(e):
                    print(f"Memory access error, trying CPU fallback")
                    # Try on CPU
                    try:
                        m_cpu = m.cpu()
                        eps = 1e-30
                        eye_cpu = torch.eye(m_cpu.shape[0], dtype=m_cpu.dtype)
                        eigval, eigvec = torch.linalg.eigh(m_cpu + eps * eye_cpu)
                        eigvec = eigvec.to(original_device)
                        print(f"CPU fallback successful for matrix {i}")
                        success = True
                    except Exception as cpu_error:
                        print(f"CPU fallback also failed: {cpu_error}")
                        success = False
                else:
                    success = False
            
            # If eigendecomposition failed, use identity matrix
            if not success or eigvec is None:
                print(f"Using identity matrix for matrix {i}")
                try:
                    identity = torch.eye(m.shape[0], dtype=original_dtype, device=original_device)
                    eigvec = identity
                    print(f"Identity matrix created successfully for matrix {i}")
                except Exception as identity_error:
                    print(f"Failed to create identity matrix: {identity_error}")
                    eigvec = torch.eye(m.shape[0], dtype=torch.float32)
            
            final.append(eigvec)
            print(f"Matrix {i} processed successfully")
            
        except Exception as outer_error:
            print(f"Catastrophic error processing matrix {i}: {outer_error}")
            import traceback
            traceback.print_exc()
            # Emergency fallback
            try:
                shape = matrix_to_decompose.shape[0] if matrix_to_decompose is not None else 1
                final.append(torch.eye(shape, dtype=torch.float32))
            except:
                final.append(torch.eye(1, dtype=torch.float32))
    
    print(f"get_orthogonal_matrix returning: {len(final)} matrices")
    for i, f in enumerate(final):
        if f is not None:
            print(f"  Result[{i}]: shape {f.shape}")
        else:
            print(f"  Result[{i}]: None")
    
    return final

@decorator_knowngood
def _compilable_stochastic_lerp_(x: List[Tensor], y: List[Tensor], a: Union[float, int, Tensor]):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        if x32.dtype != y32.dtype:
            y32 = y32.to(x32.dtype)
        copy_stochastic_(x_, x32 * (1 - a) + y32 * a)

@decorator_knowngood
def _compilable_stochastic_sub_(x: List[Tensor], y: List[Tensor], a: Union[float, int, Tensor]):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        if x32.dtype != y32.dtype:
            y32 = y32.to(x32.dtype)
        copy_stochastic_(x_, (x32 - y32 * a) / (1 - a + 1e-10))

def get_beta1(group):
    beta = None
    if "beta" in group:
        beta = group["beta"]
    if beta is None and "betas" in group:
        beta = group["betas"][0]
    if beta is None:
        raise ValueError("Beta not found in group.")
    return beta


def get_beta2(group):
    if "palm" in group and group["palm"] is True and "beta2_scale" in group:
        step = max(group.get("step", 1), 1)
        return 1 - step ** -group["beta2_scale"]
    if "betas" in group:
        return group["betas"][1]
    raise ValueError("Beta2 not found in group.")


def get_beta3(group):
    if "betas" in group:
        return group["betas"][2]
    raise ValueError("Beta3 not found in group.")


def stochastic_lerp_(x: List[Tensor], y: List[Tensor], a: Union[float, int, Tensor]):
    x, y = list_guard(x, y)
    a = scalar_guard(a, x[0])
    _compilable_stochastic_lerp_(x, y, a)

def stochastic_sub_(x: List[Tensor], y: List[Tensor], a: Union[float, int, Tensor]):
    x, y = list_guard(x, y)
    a = scalar_guard(a, x[0])
    _compilable_stochastic_sub_(x, y, a)

def list_guard(*xs):
    out = []
    for x in xs:
        if isinstance(x, (list, tuple)):
            out.append(x)
        else:
            out.append([x])
    if len(xs) == 1:
        return out[0]
    return out


def scalar_guard(*args):
    *xs, ref = args
    out = []
    for x in xs:
        if isinstance(x, float):
            out.append(torch.empty((), dtype=promote(ref.dtype), device=ref.device).fill_(x))
        elif isinstance(x, int):
            out.append(torch.empty((), dtype=torch.int64, device=ref.device).fill_(x))
        else:
            out.append(x)
    if len(xs) == 1:
        return out[0]
    return out


def broadcastable_list_guard(*xs):
    xs = list_guard(*xs)
    for x in xs:
        if isinstance(x[0], Tensor):
            ref = x[0]
            break
    else:
        raise ValueError("No tensor-valued input given")
    xs = [x if isinstance(x[0], Tensor) else list_guard(scalar_guard(*x, ref)) for x in xs]
    max_len = max(len(x) for x in xs)
    return [x if len(x) > 1 else x * max_len for x in xs]


@decorator_knowngood
def _compilable_stochastic_add_(x: List[Tensor], y: List[Tensor], alpha: Union[float, int, Tensor]):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        copy_stochastic_(x_, x32 + y32 * alpha)


def stochastic_add_(x: List[Tensor], y: List[Tensor], alpha: Union[float, int, Tensor] = 1):
    x, y = broadcastable_list_guard(x, y)
    alpha = scalar_guard(alpha, x[0])
    _compilable_stochastic_add_(x, y, alpha)


@decorator_knowngood
def _compilable_stochastic_add_divide_(x: List[Tensor], y: List[Tensor], alpha: Tensor, divisor: Tensor):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        copy_stochastic_(x_, (x32 + y32 * alpha) / divisor)


def stochastic_add_divide_(x: List[Tensor], y: List[Tensor], alpha: Union[float, int, Tensor] = 1, divisor: float = 1):
    x, y = broadcastable_list_guard(x, y)
    alpha, divisor = scalar_guard(alpha, divisor, x[0])
    _compilable_stochastic_add_divide_(x, y, alpha, divisor)


@decorator_knowngood
def _compilable_stochastic_multiply_(x: List[Tensor], y: List[Tensor]):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        copy_stochastic_(x_, x32 * y32)


def stochastic_multiply_(x: List[Tensor], y: List[Tensor]):
    x, y = broadcastable_list_guard(x, y)
    _compilable_stochastic_multiply_(x, y)


@decorator
def update_ggt(grad, GG, max_precond_dim, precondition_1d, beta):
    """
    Simplified by @francois-rozet in commit 704ccc4bab52429f945df421647ec82c54cdd65f
    Re-commited due to faulty merge
    """
    if grad.dim() == 1 and (not precondition_1d or grad.shape[0] > max_precond_dim):
        return
    
    for idx, m in enumerate(GG):
        # Get the dimension for this index
        if idx >= grad.dim():
            break
            
        dim = grad.shape[idx]
        
        # Skip if dimension is too large
        if dim > max_precond_dim:
            continue
            
        # Initialize matrix if it doesn't exist
        if not isinstance(m, Tensor):
            # Create new matrix with small regularization
            device = grad.device
            dtype = grad.dtype
            # Initialize with small identity matrix instead of zeros
            regularization = 1e-6
            GG[idx] = torch.eye(dim, device=device, dtype=dtype) * regularization
            print(f"Initialized GG[{idx}] with shape {dim}x{dim} and regularization {regularization}")
        
        # Now update the matrix
        b = einsum_base[idx]
        g0 = einsum_base[: grad.dim()]
        g1 = g0.replace(b, b.upper())
        outer_product = compiled_einsum(f"{g0},{g1}->{b + b.upper()}", grad, grad)
        
        # Add regularization to prevent singular matrices
        regularization = 1e-8
        outer_product = outer_product + torch.eye(dim, device=grad.device, dtype=grad.dtype) * regularization
        
        stochastic_lerp_(GG[idx], outer_product, 1 - beta)


@decorator
def update_gtg_no_momentum(grad, max_precond_dim, precondition_1d, beta):
    """
    Compute G^T G for each dimension consistently
    """
    if grad.dim() == 1 and (not precondition_1d or grad.shape[0] > max_precond_dim):
        return
    
    gtg_list = []
    
    for idx in range(grad.dim()):
        b = einsum_base[idx]
        g0 = einsum_base[: grad.dim()]
        
        # Create g1 by making ALL other dimensions uppercase except the target dimension
        g1 = ""
        for i, char in enumerate(g0):
            if i == idx:
                g1 += char  # Keep target dimension lowercase
            else:
                g1 += char.upper()  # Make other dimensions uppercase
        
        # This contracts over all dimensions except idx, giving G^T G
        outer_product = compiled_einsum(f"{g1},{g0}->{b + b.upper()}", grad, grad)
        gtg_list.append(outer_product)
    
    return gtg_list


def tree_apply(fn):
    def _fn(*args):
        return tree_map(fn, *args)

    return _fn


@tree_apply
def promote(x):
    if isinstance(x, torch.dtype) and x in (torch.bfloat16, torch.float16):
        return torch.float32
    if isinstance(x, Tensor) and x.dtype in (torch.bfloat16, torch.float16):
        return x.float()
    return x


def promote_detach(x, should_promote):
    if x is None:
        return x
    if should_promote:
        x = promote(x)
    return x.detach()


def detach(x):
    if isinstance(x, Tensor):
        return x.detach()
    return x


def min_dtype(xs: List[Tensor]):
    dtypes = [x.dtype for x in xs]
    for d in (torch.float32, torch.bfloat16, torch.float16):
        if all(x in (d, torch.float32, torch.float64) for x in dtypes):
            return d
    return torch.float32


def update_preconditioner(grad, Q, GG, exp_avg, max_precond_dim, precondition_1d, beta, update_precond):
    """
    Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
    """
    update_ggt(grad, GG, max_precond_dim, precondition_1d, beta)
    if update_precond:
        get_orthogonal_matrix_QR(GG, Q, exp_avg)

def update_preconditioner_cosmos(grad, U, S, exp_avg, max_precond_dim, precondition_1d, beta2, beta3, update_precond):
    """
    Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
    """
    #GG = update_gtg_no_momentum(grad, max_precond_dim, precondition_1d, beta2)
    
    # Only update basis if we have valid GG and this tensor uses COSMOS
    if update_precond and U is not None and S is not None:
        cosmos_update_basis(grad, U, S, beta2, beta3)

def update_preconditioner_CASPR(grad, Q, GG, exp_avg, max_precond_dim, precondition_1d, beta, update_precond):
    """
    Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
    """
    update_ggt(grad, GG, max_precond_dim, precondition_1d, beta)
    if update_precond:
        get_orthogonal_matrix_QR(GG, Q, exp_avg)


def init_preconditioner(grad, state, max_precond_dim, precondition_1d):
    """
    Initialize preconditioner with comprehensive error handling
    """
    try:
        # Validate input gradient
        if grad is None:
            print("Warning: grad is None, skipping preconditioner initialization")
            return
            
        # Check if gradient is accessible
        try:
            grad_shape = grad.shape
            grad_device = grad.device
            grad_dtype = grad.dtype
        except RuntimeError as e:
            if "illegal memory access" in str(e):
                print("Warning: Cannot access gradient due to memory corruption, skipping preconditioner")
                return
            raise
        
        # Check for invalid values
        if not torch.isfinite(grad).all():
            print("Warning: Gradient contains non-finite values, skipping preconditioner")
            return
            
        # Clear CUDA cache before operations
        if grad_device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Initialize GG if not exists
        if "GG" not in state:
            state["GG"] = []
        
        # Ensure GG has correct length
        sh = grad.shape
        print(sh)
        if len(sh) == 1:
            if not precondition_1d:
                print("1D gradient with precondition_1d=False, skipping")
                return
            if len(state["GG"]) == 0:
                state["GG"].append(None)
        else:
            while len(state["GG"]) < len(sh):
                state["GG"].append(None)
        
        # Update GG matrices with error handling
        try:
            update_ggt(grad, state["GG"], max_precond_dim, precondition_1d, 0)
            
        except Exception as e:
            print(f"Error in update_ggt: {e}, initializing with identity matrices")
            # Initialize with identity matrices
            state["GG"] = []
            for dim in sh:
                if dim <= max_precond_dim:
                    state["GG"].append(torch.eye(dim, device=grad_device, dtype=grad_dtype))
                else:
                    state["GG"].append(None)
        
        # Get orthogonal matrices with error handling
        try:
            state["Q"] = get_orthogonal_matrix(state["GG"])
            print(state["Q"])
        except Exception as e:
            print(f"Error in get_orthogonal_matrix: {e}, using identity matrices")
            state["Q"] = []
            for gg in state["GG"]:
                if gg is not None:
                    state["Q"].append(torch.eye(gg.shape[0], device=gg.device, dtype=gg.dtype))
                else:
                    state["Q"].append(None)
                    
    except Exception as e:
        print(f"Critical error in init_preconditioner: {e}")
        # Emergency fallback - disable preconditioning for this parameter
        state["Q"] = []
        if "GG" in state:
            for _ in state["GG"]:
                state["Q"].append(None)


def cosmos_update_basis(g, U, S, beta2, beta3):
    """
    COSMOS basis update following Algorithm 1 from the paper
    """
    
    # Handle the case where inputs might not be lists
    if not isinstance(g, (list, tuple)):
        g = [g]
    if not isinstance(U, (list, tuple)):
        U = [U]
    if not isinstance(S, (list, tuple)):
        S = [S]
    
    for i, (g_tensor, u_tensor, s_tensor) in enumerate(zip(g, U, S)):
        if u_tensor is None or s_tensor is None:
            continue
            
        # Promote to higher precision
        g_promoted = promote(g_tensor)     # Gt: gradient
       
        u_promoted = promote(u_tensor)     # Ut-1: basis matrix (n, r)
        s_promoted = promote(s_tensor)     # St-1: S matrix (r, r)
       
        # For 2D gradients, we need to compute Gt^T @ Gt
        if g_promoted.dim() == 2:
            # Line 5: Update U using QR decomposition
            # Ut ← QR(β₃Ut-1St-1 + (1 - β₃)Gt^T GtUt-1)
            
            # First term: β₃ * Ut-1 @ St-1
            term1 = beta3 * torch.matmul(u_promoted, s_promoted)
            
            # Second term: (1 - β₃) * Gt^T @ Gt @ Ut-1
            
            gt_gt = torch.matmul(g_promoted.T,g_promoted)
 
            term2 = (1 - beta3) * torch.matmul(gt_gt, u_promoted)
            
            # Combined matrix for QR decomposition
            M_combined = term1 + term2
            
            # QR decomposition to get new orthogonal basis
            u_new, _ = torch.linalg.qr(M_combined, mode='reduced')
            
            # Line 6: Update S matrix
            # St ← Ut^T(β₃Ut-1St-1Ut-1^T + (1 - β₃)Gt^T Gt)Ut
            
            # First term: β₃ * Ut-1 @ St-1 @ Ut-1^T
            old_s_full = beta3 * torch.matmul(torch.matmul(u_promoted, s_promoted), u_promoted.T)
            
            # Second term: (1 - β₃) * Gt^T @ Gt
            new_contrib = (1 - beta3) * gt_gt
            
            # Combined full matrix
            full_matrix = old_s_full + new_contrib
            
            # Project to new basis: St = Ut^T @ full_matrix @ Ut
            s_new = torch.matmul(torch.matmul(u_new.T, full_matrix), u_new)
            
        else:
            # For 1D gradients, fall back to simpler update
            continue
        
        # Update U and S in-place
        copy_stochastic_(u_tensor, u_new)
        copy_stochastic_(s_tensor, s_new)
    
    return U, S

def project(grad, Q, back=False):
    """
    Project gradient using orthogonal matrices Q.
    
    :param grad: gradient tensor
    :param Q: list of orthogonal matrices (can be None)
    :param back: whether to project back
    :return: projected gradient
    """
    if Q is None:
        # No preconditioning, return original gradient
        return grad
        
    param = einsum_base[: grad.dim()]
    
    # Filter out None matrices and create valid preconditioners
    valid_preconditioners = []
    valid_matrices = []
    
    for i, (m, g) in enumerate(zip(Q, param)):
        if m is not None:
            valid_preconditioners.append((g + g.upper())[:: -1 if back else 1])
            valid_matrices.append(m)
    
    if not valid_preconditioners:
        # No valid preconditioners, return original gradient
        return grad
        
    preconditioners = ",".join(valid_preconditioners)
    
    if preconditioners:
        out = "".join([c.upper() if c.upper() in preconditioners else c for c in param])
        out = compiled_einsum(f"{param},{preconditioners}->{out}", promote(grad), *valid_matrices)
        return out
    else:
        return grad


@contextlib.contextmanager
def patch_backward():
    @contextlib.contextmanager
    def patch_module(module):
        original = module.backward
        try:
            signature = inspect.signature(original)

            @functools.wraps(original)
            def patched_backward(*args, **kwargs):
                new_kwargs = signature.bind(*args)
                new_kwargs.apply_defaults()
                new_kwargs = new_kwargs.arguments
                new_kwargs.update(kwargs)
                new_kwargs["create_graph"] = True
                return original(**new_kwargs)

            module.backward = patched_backward
            yield
        finally:
            module.backward = original

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch_module(torch.Tensor))
        stack.enter_context(patch_module(torch.autograd))
        yield


def hasattr_none(obj, name):
    return getattr(obj, name, None) is not None


class ExactHVPFailed(ValueError):
    pass


use_default = object()


class StatefulOptimizer(torch.optim.Optimizer):
    """
    finite_differences saves memory, but needs more compute. (Alternative is true HVP)
    Both `True` and `False` have some edge cases they don't support, so experiment with it.
    The previous (heavyball<=1.5.3) default was `True`, which is incompatible with some benchmarks but works better with RevNet
    Further notice that both methods have different numerics outputs
    """

    ema_decay: float = 0.001
    compile_step: bool = False
    hessian_approx: bool = False
    precond_schedule: Union[Callable, float, None] = None
    stochastic_schedule: bool | Literal[use_default] = use_default
    finite_differences: bool = False
    fallback_to_finite_differences: bool = True
    _fallback_enabled: bool = False
    hvp_interval: int = 1  # grad is faster initially, hvp later

    def __init__(self, params, defaults, foreach: bool = True, use_ema: bool = False):
        super().__init__(params, {**defaults, "foreach": foreach})
        self.use_ema = use_ema
        self.mapping = {}
        self.mapping_inverse = {}

        if self.stochastic_schedule is use_default:
            stochastic_schedule = None
            for group in self.param_groups:
                new = group.get("stochastic_schedule", stochastic_schedule)
                if stochastic_schedule is not None and new != stochastic_schedule:
                    raise ValueError("All parameter groups must have the same stochastic_schedule.")
                stochastic_schedule = new
            self.stochastic_schedule = stochastic_schedule

        self.inner_group = {"stochastic_schedule": self.stochastic_schedule}
        self.precond_rng = random.Random(0x12312)
        self._is_preconditioning = None

        if self.hessian_approx and self.compile_step:
            raise ValueError("Hessian approximation can't be used with compile_step.")

        self.register_state_dict_post_hook(StatefulOptimizer._store_stats)
        self.register_load_state_dict_pre_hook(StatefulOptimizer._load_stats)
        self._init_mapping()

    def _store_stats(self, state_dict: dict[str, any]):
        state_dict["heavyball"] = {
            "inner_group": self.inner_group,
            "precond_rng": pickle.dumps(self.precond_rng),
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "compile_step": self.compile_step,
            "hessian_approx": self.hessian_approx,
            "precond_schedule": pickle.dumps(self.precond_schedule),
            "stochastic_schedule": self.stochastic_schedule,
            "fallback_to_finite_differences": self.fallback_to_finite_differences,
            "_fallback_enabled": self._fallback_enabled,
            "hvp_interval": self.hvp_interval,
        }

    def _load_stats(self, state_dict):
        sd = state_dict.pop("heavyball", {})
        for k, v in sd.items():
            if k in ("precond_rng", "precond_schedule"):
                v = pickle.loads(v)
            setattr(self, k, v)

    def get_groups(self, group):
        return [group]

    @functools.lru_cache(maxsize=None)
    def state_(self, arg: Tensor, fail: bool = True):
        if not fail and arg not in self.mapping:
            return {}
        state_param, index = self.mapping_inverse[arg]
        if state_param not in self.state:
            self.state[state_param] = collections.defaultdict(dict)
        return self.state[state_param][index]

    def mars_correct_list(self, group, p_list, g_list, mars_gamma, beta):
        for p, g in zip(p_list, g_list):
            state = self.state_(p)
            if "mars_old_grad" not in state:
                state["mars_old_grad"] = torch.zeros_like(g)
        old_gs = [self.state_(p)["mars_old_grad"] for p in p_list]
        mars_correction(g_list, old_gs, mars_gamma, beta)

    def mars_correct_list_ema_c(self, group, p_list, g_list, mars_gamma, beta, ema_update):
        mars_correction(g_list, ema_update, mars_gamma, beta)

    def mars_correct_list_ema_m(self, group, p_list, g_list, mars_gamma, beta):
        for p, g in zip(p_list, g_list):
            state = self.state_(p)
            if "mars_old_grad" not in state:
                state["mars_old_grad"] = torch.zeros_like(g)
        old_gs = [self.state_(p)["mars_old_grad"] for p in p_list]
        mars_correction_ema(g_list, old_gs, mars_gamma, beta)

    def _init_mapping(self, group: dict | None = None):
        if group is None:
            for group in self.param_groups:
                self._init_mapping(group)
            return

        for p in group["params"]:
            if p not in self.mapping:
                self.mapping[p] = p_views = merge_group(group, p)
                for i, pv in enumerate(p_views):
                    self.mapping_inverse[pv] = (p, i)

    def split_p_and_g_in_group(
        self,
        group: dict,
        skip_none: bool = True,
        should_promote: bool = True,
        beta1: float = -1.0,
        raw: bool = False,
        use_ema: bool = False,
    ):
        
        for p in group["params"]:
           
            grad = getattr(p, "grad", None)
            
            if grad is None and skip_none:
                continue

            p.grad = None

            if raw:
                yield p, grad
                continue

            if p in self.mapping:
                p_views = self.mapping[p]
            else:
                self.mapping[p] = p_views = merge_group(group, p)
                for i, pv in enumerate(p_views):
                    self.mapping_inverse[pv] = (p, i)

            vector = getattr(p, "vector", None)
            hessian_vector = getattr(p, "hessian_vector", None)
            p.vector = None
            p.hessian_vector = None

            grad, vs, hvs = [
                [None] * len(p_views) if x is None else merge_group(group, x)  #
                for x in (grad, vector, hessian_vector)
            ]

            for pv, g, v, hv in zip(p_views, grad, vs, hvs):
                
                g = promote_detach(g, should_promote)
                if beta1 >= 0 and group.get("mars", False):
                    
                    if use_ema and "update_by_adam_exp_avg_0" in self.state_(p):
                        ema_update = self.state_(p)["update_by_adam_exp_avg_0"]  
                        #self.mars_correct_list_ema_c(group, [pv], [g], group["mars_gamma"], beta1, [ema_update])
                        self.mars_correct_list_ema_m(group, [pv], [g], group["mars_gamma"], beta1)
            
                    else:
                        self.mars_correct_list(group, [pv], [g], group["mars_gamma"], beta1)
                pv.vector = promote_detach(v, should_promote)
                pv.hessian_vector = promote_detach(hv, should_promote)
                yield pv, g

    def state_size(self) -> int:
        total_bytes = 0

        def _add(x):
            nonlocal total_bytes
            if isinstance(x, Tensor):
                total_bytes += x.numel() * x.element_size()

        for group in self.param_groups:
            for p, _ in self.split_p_and_g_in_group(group, skip_none=False):
                tree_map(_add, self.state_(p))
        return total_bytes

    def _step(self, group):
        raise NotImplementedError

    def ema_update(self):
        with torch.no_grad():
            for group in self.param_groups:
                active_p = [p for p in group["params"]]

                if not active_p:
                    return

                k = group["ema_step"] = group.get("ema_step", -1) + 1

                for p in active_p:
                    if "param_ema" not in self.state_(p):
                        self.state_(p)["param_ema"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                y, param_ema = zip(*[(p.data, self.state_(p)["param_ema"]) for p in active_p])
                torch._foreach_lerp_(param_ema, y, weight=beta_debias(1 - self.ema_decay, k + 1))

    def copy_emas_to_params(self):
        with torch.no_grad():
            for group in self.param_groups:
                active_p = [p for p in group["params"]]

                if not active_p:
                    return

                for p in active_p:
                    if "param_ema" in self.state_(p):
                        p_clone = p.data.clone()
                        set_(p.data, self.state_(p)["param_ema"])
                        set_(self.state_(p)["param_ema"], p_clone)

    def copy_params_to_emas(self):
        with torch.no_grad():
            for group in self.param_groups:
                active_p = [p for p in group["params"]]

                if not active_p:
                    return

                for p in active_p:
                    if "param_ema" in self.state_(p):
                        ema_clone = self.state_(p)["param_ema"].data.clone()
                        set_(self.state_(p)["param_ema"], p.data)
                        set_(p.data, ema_clone)

    def _finite_differences_hvp(self, closure):
        with torch.enable_grad():
            loss = closure()  # closure without retain_graph=True

        grads = []
        for group in self.param_groups:
            for p, g in self.split_p_and_g_in_group(group, skip_none=True, raw=True):
                grads.append(g)
                p.vector = torch.randn_like(p)
                p.orig = p.data.clone()
                # scale taken from https://github.com/lixilinx/psgd_torch/blob/1943e66596111e78157ca1b72b31c1dfdf0653ef/preconditioned_stochastic_gradient_descent.py#L2161
                stochastic_add_(p.data, p.vector, torch.finfo(p.dtype).eps ** 0.5)

        with torch.enable_grad():
            closure()

        # we don't subtract the vector here again to avoid accumulating error from (x + eps - eps + eps - eps)
        # this costs more memory, but the imprecision seems too severe to use the other method
        for group in self.param_groups:
            for p, g in self.split_p_and_g_in_group(group, skip_none=True, raw=True):
                p.grad = grads.pop(0)
                stochastic_add_(g, p.grad, -1)  # technically, we have to divide by the scale here
                p.hessian_vector = g
                p.data.copy_(p.orig)
                del p.orig
        return loss

    def _double_backward_hvp(self, closure):
        with torch.enable_grad(), patch_backward():
            loss = closure()

        params, grads = [], []
        for group in self.param_groups:
            for p, g in self.split_p_and_g_in_group(group, skip_none=True, raw=True):
                params.append(p)
                grads.append(g)

        if not params:
            raise ValueError("No parameter has gradients")

        vs = [torch.randn_like(p) for p in params]
        with torch.enable_grad():
            try:
                hvs = torch.autograd.grad(grads, params, vs, create_graph=False, retain_graph=False, allow_unused=True)
            except RuntimeError as e:
                raise ExactHVPFailed(str(e.args))

        unused = []
        for p, g, v, hv in zip(params, grads, vs, hvs):
            p.hessian_vector = detach(hv)
            p.grad = detach(g)
            p.vector = detach(v)
            if hv is None:
                unused.append(list(p.shape))

        if unused:
            raise ExactHVPFailed(f"Parameters with the following shapes have no 2nd order derivative: {unused}")

        return loss

    def _handle_closure(self, closure):
        hessian_approx = self.hessian_approx and self._is_preconditioning

        if closure is None:
            if hessian_approx:
                raise ValueError("Hessian approximation requires a closure.")
            return None

        step = self.inner_group["total_hvp_steps"] = self.inner_group.get("total_hvp_steps", 0) + 1
        if not hessian_approx or (step - 1) % self.hvp_interval == 0:  # hvp in 0th step for better precond init
            with torch.enable_grad():
                loss = closure()
            return loss

        if self.finite_differences or self._fallback_enabled:
            return self._finite_differences_hvp(closure)

        try:
            return self._double_backward_hvp(closure)
        except NotImplementedError as e:
            if not self.fallback_to_finite_differences:
                raise
            if not any(isinstance(arg, str) and _cudnn_double_backward_pattern.match(arg) for arg in e.args):
                raise
            warn_once(
                "CUDNN doesn't support double-backward for some models (including RNNs). "  #
                f"Falling back to finite_differences.\n{_fd_error}{e}"
            )
        except RuntimeError as e:
            if not self.fallback_to_finite_differences:
                raise
            if not any(isinstance(arg, str) and _torch_compile_double_backward_pattern.match(arg) for arg in e.args):
                raise
            warn_once(
                f"torch.compile does not support double-backward. Disabling it may be beneficial, depending on "
                f"the model.\n{_fd_error}{e}"
            )
        except ExactHVPFailed as e:
            if not self.fallback_to_finite_differences:
                raise
            warn_once(f"Exact HVP calculation failed.\n{_fd_error}{e}")
        self._fallback_enabled = True
        return self._handle_closure(closure)
    

    """ def _handle_prev_closure(self, prev_closure):
        """#Handle previous model closure and store gradients separately
    """
        if prev_closure is None:
            return None
        
        # Store current gradients temporarily
        current_grads = {}
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    current_grads[id(p)] = p.grad.clone()
        
        # Clear gradients for previous model computation
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
        
        # Compute previous model loss and gradients
        with torch.enable_grad():
            prev_loss = prev_closure()  # This calls loss.backward() and fills p.grad
        
        # Store previous gradients in param_groups
        for group in self.param_groups:
            if 'prev_grads' not in group:
                group['prev_grads'] = []
            if 'prev_loss' not in group:
                group['prev_loss'] = prev_loss.item() if prev_loss is not None else None
            
            group['prev_grads'].clear()
            for p in group['params']:
                if p.grad is not None:
                    group['prev_grads'].append(p.grad.clone())  # Store prev gradient
                    
                else:
                    group['prev_grads'].append(torch.zeros_like(p.data))
        
        # Restore current gradients
        for group in self.param_groups:
            for p in group['params']:
                if id(p) in current_grads:
                    p.grad = current_grads[id(p)]  # Restore current gradient
                else:
                    p.grad = None
        
        return prev_loss
    """
    def _handle_prev_closure(self, prev_closure):
        """Handle previous model closure and store gradients separately"""
        # Clear gradients for previous model computation
        self.zero_grad()  
        
        # Compute previous model loss and gradients
        with torch.enable_grad():
            prev_loss = prev_closure()  # This calls loss.backward() and fills p.grad
        
        # Store previous gradients in param_groups
        for group in self.param_groups:
            if 'prev_grads' not in group:
                group['prev_grads'] = []
            if 'prev_loss' not in group:
                group['prev_loss'] = prev_loss.item() if prev_loss is not None else None
            
            group['prev_grads'].clear()
            for p in group['params']:
                if p.grad is not None:
                    group['prev_grads'].append(p.grad.clone())  # Store prev gradient
                else:
                    group['prev_grads'].append(torch.zeros_like(p.data))
        return prev_loss
        
    def step(self, closure: Optional[Callable] = None):
        if self.precond_schedule is None:
            self._is_preconditioning = False
        else:
            self._is_preconditioning = psgd_should_update(self.inner_group, self.precond_schedule, self.precond_rng)
        
        loss = self._handle_closure(closure)
        
        # Track if we have any DistributedShampoo groups
        has_shampoo_groups = False
        
        with torch.no_grad(), torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                if "param_count" not in group:
                    group["param_count"] = sum(p.numel() for p in group["params"])
                group["is_preconditioning"] = self._is_preconditioning
                external_optimizer_type = group.get("external_optimizer", None)
                # Check if this group uses an external optimizer
                if external_optimizer_type is not None:
                    
                    if external_optimizer_type.lower() == "distributedshampoo":
                        
                        has_shampoo_groups = True
                        # For Shampoo: just assign gradients, don't step yet
                        param_tensors = [p for p in group['params'] if p.grad is not None]
                        grads = [p.grad for p in param_tensors]
                        
                        if param_tensors:
                            # Ensure global optimizers are created
                            self._create_global_shampoo_optimizers()
                            # Just assign gradients
                            for param, grad in zip(param_tensors, grads):
                                param.grad = grad
                    else:
                        # Handle other external optimizers normally
                        param_tensors = [p for p in group['params'] if p.grad is not None]
                        grads = [p.grad for p in param_tensors]
                        
                        if param_tensors:
                            torch.nn.utils.clip_grad_norm_(param_tensors, max_norm=self.clip_norm)
                            self._use_external_optimizer(group, external_optimizer_type, param_tensors, grads, closure)
                else:
                    # Use your existing chain-based approach
                    self._step(group, closure)
            
            # Step global Shampoo optimizers once after all groups are processed
            if has_shampoo_groups:
                self._step_global_shampoo_optimizers()
                self._clear_shampoo_gradients()
        
        if self.use_ema:
            self.ema_update()
        
        for real, views in self.mapping.items():
            for tensor in (real, *views):
                for key in ("grad", "vector", "hessian_vector", "orig"):
                    if hasattr(tensor, key):
                        setattr(tensor, key, None)
        
        return loss
    
    def step_with_prev(self, closure: Optional[Callable] = None, prev_closure: Optional[Callable] = None):
        if self.precond_schedule is None:
            self._is_preconditioning = False
        else:
            self._is_preconditioning = psgd_should_update(self.inner_group, self.precond_schedule, self.precond_rng)
       
        if prev_closure is not None:
            prev_loss = self._handle_prev_closure(prev_closure)
        loss = self._handle_closure(closure)
        # we assume that parameters are constant and that there are no excessive recompiles
        with torch.no_grad(), torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                if "param_count" not in group:
                    group["param_count"] = sum(p.numel() for p in group["params"])
                group["is_preconditioning"] = self._is_preconditioning
                self._step(group)
                if self.use_ema:
                    self.ema_update()
                for real, views in self.mapping.items():
                    for tensor in (real, *views):
                        for key in ("grad", "vector", "hessian_vector", "orig"):
                            if hasattr(tensor, key):
                                setattr(tensor, key, None)
        return loss



def copy_stochastic_list_(target: List[Tensor], source: List[Tensor]):
    for t, s in zip(target, source):
        copy_stochastic_(t, s)


@decorator_knowngood
def _lerp(state: List[Tensor], grad: List[Tensor], beta):
    ea32 = list(map(promote, state))
    grad = list(map(promote, grad))
    beta = promote(beta)
    stochastic_lerp_(ea32, grad, 1 - beta)
    copy_stochastic_list_(state, ea32)
    return ea32

@decorator_knowngood
def _undo_lerp(state: List[Tensor], grad: List[Tensor], beta):
    ea32 = list(map(promote, state))
    grad = list(map(promote, grad))
    beta = promote(beta)
    stochastic_sub_(ea32, grad, 1 - beta)
    copy_stochastic_list_(state, ea32)
    return ea32

@decorator_knowngood
def _compilable_SGD_(
    grad: List[Tensor],
    step: Tensor,
    eps: Tensor,
):
    g32 = list(map(promote, grad))
    copy_stochastic_list_(grad, g32)


def SGD_(
    grad: List[Tensor],
    step: int,
    eps: float = 1e-8,
):
    grad = map(list_guard, ( grad))
    step, eps = scalar_guard( step, eps, grad[0])
    _compilable_SGD_( grad, step, eps)
    return grad


@decorator_knowngood
def _fused_compilable_SGD_(
    y: List[Tensor],
    exp_avg: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    beta1: Tensor,
    step: Tensor,
    decay: Tensor,
    lr: Tensor,
    eps: Tensor,
    caution: bool,
):
    u32, g32 = [list(map(promote, x)) for x in [update, grad]]
    exp_avg32 = _lerp(exp_avg, g32, beta1)
    _compilable_update_(y, exp_avg32, decay, lr, caution, g32)


def fused_SGD_(
    y: List[Tensor],
    exp_avg: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    beta1: float,
    step: int,
    lr: float,
    eps: float,
    decay: float,
    caution: bool,
):
    y, grad, exp_avg = list_guard(y, grad, exp_avg)
    beta1, step, lr = scalar_guard(beta1, step, lr, y[0])
    _fused_compilable_SGD_(y, exp_avg, update, grad, beta1, step, decay, lr, eps, caution)


@decorator_knowngood
def _compilable_adam_(
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    grad: List[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    eps: Tensor,
):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)

    g32 = list(map(promote, grad))
    exp_avg32 = _lerp(exp_avg, g32, beta1)
    denom = _compilable_exp_avg_sq_(exp_avg_sq, g32, beta2, eps, [None])
    u32 = torch._foreach_div(exp_avg32, denom)
    copy_stochastic_list_(grad, u32)


def adam_(
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    grad: List[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    eps: float = 1e-8,
):
    
    exp_avg, exp_avg_sq, grad = map(list_guard, (exp_avg, exp_avg_sq, grad))
    beta1, beta2, step, eps = scalar_guard(beta1, beta2, step, eps, exp_avg[0])
    _compilable_adam_(exp_avg, exp_avg_sq, grad, beta1, beta2, step, eps)
    return grad

@decorator_knowngood
def _compilable_nadam_(
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    grad: List[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    eps: Tensor,
):
    beta1_original = beta1
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)

    g32 = list(map(promote, grad))
    exp_avg32 = _lerp(exp_avg, g32, beta1)
    denom = _compilable_exp_avg_sq_(exp_avg_sq, g32, beta2, eps, [None])
    exp_avg32 = _lerp(exp_avg, g32, beta1_original)
    u32 = torch._foreach_div(exp_avg32, denom)

    copy_stochastic_list_(grad, u32)
    exp_avg32 = _undo_lerp(exp_avg, g32, beta1_original)


def nadam_(
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    grad: List[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    eps: float = 1e-8,
):
    
    exp_avg, exp_avg_sq, grad = map(list_guard, (exp_avg, exp_avg_sq, grad))
    beta1, beta2, step, eps = scalar_guard(beta1, beta2, step, eps, exp_avg[0])
    _compilable_nadam_(exp_avg, exp_avg_sq, grad, beta1, beta2, step, eps)
    return grad


def matmul_tensor_lists(tensor_list_a, tensor_list_b):
    """
    Calculate matrix multiplication between two lists of tensors.
    Handles nested lists by flattening them first.
    """
    # Flatten any nested lists
    list_a = flatten_tensor_lists(tensor_list_a)
    list_b = flatten_tensor_lists(tensor_list_b)
    
    if len(list_a) != len(list_b):
        raise ValueError(f"Lists must have same length: {len(list_a)} != {len(list_b)}")
    
    result_list = []
    for i, (A, B) in enumerate(zip(list_a, list_b)):
        if A is None or B is None:
            result_list.append(None)
            continue
        
        # Use torch.matmul for standard matrix multiplication
        result = torch.matmul(A, B)
        result_list.append(result)
    
    return result_list


def flatten_tensor_lists(tensor_list):
    """Flatten nested lists of tensors"""
    result = []
    for item in tensor_list:
        if isinstance(item, list):
            # If it's a list, take the first tensor (or handle as needed)
            if len(item) > 0 and item[0] is not None and isinstance(item[0], torch.Tensor):
                result.append(item[0])
            else:
                result.append(None)
        elif isinstance(item, torch.Tensor):
            result.append(item)
        else:
            result.append(None)
    return result

def matmul_tensor_lists_batched(tensor_list_a, tensor_list_b, grad_shape, einsum_base="abcdefghijklmnopqrstuvwxyz", compiled_einsum=None):
    """
    Calculate matrix multiplication between two lists of tensors using einsum,
    following the pattern from update_ggt_no_momentum for handling gradients.
    
    Args:
        tensor_list_a: List of tensors (left matrices)
        tensor_list_b: List of tensors (right matrices)  
        grad_shape: Shape of the gradient tensor (used for dimension indexing)
        einsum_base: String of characters to use for einsum indices
        compiled_einsum: Compiled einsum function (optional, falls back to torch.einsum)
    
    Returns:
        List of matrix products A_i @ B_i
    """
    if compiled_einsum is None:
        compiled_einsum = torch.einsum
    
    if len(tensor_list_a) != len(tensor_list_b):
        raise ValueError(f"Lists must have same length: {len(tensor_list_a)} != {len(tensor_list_b)}")
    
    result_list = []
    
    for idx in range(len(grad_shape)):
        A = tensor_list_a[idx] if idx < len(tensor_list_a) else None
        B = tensor_list_b[idx] if idx < len(tensor_list_b) else None
        
        if A is None or B is None:
            result_list.append(None)
            continue
        
        # Use the same indexing pattern as update_ggt_no_momentum
        i_char = einsum_base[idx]
        j_char = einsum_base[idx].upper()  # Use uppercase for the contracted dimension
        k_char = einsum_base[(idx + len(grad_shape)) % len(einsum_base)]
        
        # For A @ B: A[i,j] * B[j,k] -> C[i,k]
        einsum_expr = f"{i_char}{j_char},{j_char}{k_char}->{i_char}{k_char}"
        result = compiled_einsum(einsum_expr, A, B)
        result_list.append(result)
    
    return result_list


# Specialized functions for common COSMOS operations
def matmul_U_S(U_list, S_list, compiled_einsum=None):
    """Calculate U @ S for COSMOS algorithm (line 6: U_t^T * ... * U_t)"""
    return matmul_tensor_lists(U_list, S_list, compiled_einsum=compiled_einsum)


def matmul_A_UT(A_list, U_list):
    """Calculate A @ U^T for COSMOS algorithm"""
    result_list = []
    for A, U in zip(A_list, U_list):
        if A is None or U is None:
            result_list.append(None)
            continue
        
        # A @ U^T: multiply A by transpose of U
        result = torch.matmul(A, U.T)
        result_list.append(result)
    
    return result_list


def matmul_UT_A(U_list, A_list):
    """Calculate U^T @ A for COSMOS algorithm"""
    result_list = []
    for U, A in zip(U_list, A_list):
        if U is None or A is None:
            result_list.append(None)
            continue
        
        # U^T @ A: transpose U then multiply
        result = torch.matmul(U.T, A)
        result_list.append(result)
    
    return result_list

def cosmos_project_gradients(grad_list, U_list):
    """Project gradients using COSMOS U matrices (following official implementation)."""
    result_list = []
    
    for grad, U in zip(grad_list, U_list):
        if grad is None or U is None:
            result_list.append(None)
            continue
            
        # Official COSMOS does: U.T @ grad (not grad @ U)
        # This projects the gradient into the low-rank subspace
        if isinstance(U, list) and len(U) > 0 and U[0] is not None:
            result = torch.matmul(U[0].T, grad)  # Shape: (rank, grad.size(1))
        else:
            result = grad
            
        result_list.append(result)
    
    return result_list


def cosmos_(
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    adam_exp_avg_sq: List[Tensor],
    grad: List[Tensor],
    U: List[Tensor],
    S: List[Tensor],
    beta1: float,
    beta2: float,
    lr: float,
    gamma: float,
    step: int,
    eps: float = 1e-8,
):
    exp_avg, exp_avg_sq, adam_exp_avg_sq, grad, U, S = map(list_guard, (exp_avg, exp_avg_sq, adam_exp_avg_sq, grad, U, S))
    beta1, beta2, gamma, step, eps, lr = scalar_guard(beta1, beta2, gamma, step, eps, lr, exp_avg[0])
    _compilable_cosmos_(exp_avg, exp_avg_sq, adam_exp_avg_sq, grad, U, S, beta1, beta2, lr, gamma, step, eps)
    return grad

@decorator_knowngood
def _compilable_cosmos_(
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    adam_exp_avg_sq: List[Tensor],
    grad: List[Tensor],
    U: List[Tensor],
    S: List[Tensor],
    beta1: float,
    beta2: float,
    lr: float,
    gamma: float,
    step: int,
    eps: float = 1e-8,
):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)
    
    g32 = list(map(promote, grad))
    exp_avg32 = list(map(promote, exp_avg))
    U32 = [promote(u) if u is not None else None for u in U]
    S32 = [promote(s) if s is not None else None for s in S]
    
    # Update first moment (same as Adam)
    exp_avg32 = _lerp(exp_avg32, g32, beta1)
    
    # Find which parameters use COSMOS vs Adam
    cosmos_mask = []
    for i, (grad_tensor, U_tensor, S_tensor) in enumerate(zip(g32, U, S)):
        use_cosmos = (grad_tensor is not None and 
                     grad_tensor.dim() == 2 and 
                     grad_tensor.size(0) <= 10000 and 
                     U_tensor is not None and 
                     S_tensor is not None)
        cosmos_mask.append(use_cosmos)
    
    # Process Adam parameters with vectorized operations
    adam_grads = [g for g, use_cosmos in zip(g32, cosmos_mask) if not use_cosmos and g is not None]
    adam_exp_avg_sq = [sq for sq, use_cosmos in zip(exp_avg_sq, cosmos_mask) if not use_cosmos]
    adam_exp_avg32 = [ea for ea, use_cosmos in zip(exp_avg32, cosmos_mask) if not use_cosmos]
    
    if adam_grads:
        # Use optimized Adam processing
        denom = _compilable_exp_avg_sq_(adam_exp_avg_sq, adam_grads, beta2, eps, [None])
        adam_updates = torch._foreach_div(adam_exp_avg32, denom)
    else:
        adam_updates = []
    
    # Process COSMOS parameters individually (harder to vectorize)
    cosmos_updates = []
    
    for i, use_cosmos in enumerate(cosmos_mask):
        if use_cosmos:
           
            update = cosmos_low_rank_update(
                g32[i], exp_avg32[i], exp_avg_sq[i], U[i], S[i],
                beta1, beta2, lr, gamma, step, eps
            )
            
            cosmos_updates.append(update)
       
    # Merge results
    updates = []
    adam_idx = 0
    cosmos_idx = 0
    for use_cosmos in cosmos_mask:
        if use_cosmos:
            updates.append(cosmos_updates[cosmos_idx])
            cosmos_idx += 1
        else:
            updates.append(adam_updates[adam_idx] if adam_idx < len(adam_updates) else None)
            adam_idx += 1
    
    copy_stochastic_list_(grad, updates)

def cosmos_low_rank_update(grad, exp_avg, exp_avg_sq, U, S, beta1, beta2, lr, gamma, step, eps):
    """
    Returns the parameter update (not modifying exp_avg directly)
    """
   
    if step == 1:
        # Initialize U using SVD
        W = torch.matmul(grad.T, grad)
        U_svd, _, _ = torch.linalg.svd(W, full_matrices=False)
        U.data = U_svd[:, :U.size(1)]
        S.data = torch.matmul(torch.matmul(U.T, grad.T), torch.matmul(grad, U)) * (1 - beta2)
        exp_avg_sq.data = torch.zeros(grad.size(0), U.size(1), device=grad.device, dtype=grad.dtype)
        # For first step, return a small update or zero
        return torch.zeros_like(grad) * lr
    
    # Update basis
    cosmos_update_llama_style(grad, U, S, beta2)
    
    # Project to low-rank space
    low_rank_grad = torch.matmul(grad, U)
    
    #exp_avg_sq.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad, value=1 - beta2)
    exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * (low_rank_grad * low_rank_grad)
    

    V_add_eps = exp_avg_sq + eps
    # Compute update in low-rank space
    t = torch.matmul(exp_avg, U) 
    corrected = t / beta1
    denom = (V_add_eps / beta2)**0.5
    t1 = corrected / denom
    A = torch.matmul(t1, U.T)
    
    # Orthogonal complement update
    t_orth = exp_avg - torch.matmul(t, U.T)
    t_orth = t_orth / (t_orth.norm() + eps)
    t_orth = zeropower_via_newtonschulz5(t_orth, steps=5)
    sqrt_n = (grad.size(1) ** 0.5)
    sqrt_m = (grad.size(0) ** 0.5)
    B = t_orth * sqrt_n / (t_orth.norm() + eps)
    
    G = A + gamma * sqrt_m * B
    final_update = G * sqrt_n / (G.norm() + eps)
    
    return final_update

def adam_fallback_update(grad, exp_avg, exp_avg_sq, beta1, beta2, lr, step, eps):
    """
    Returns the Adam parameter update
    """
    denom = _compilable_exp_avg_sq_(exp_avg_sq, grad, beta2, eps, [None])
    u32 = torch._foreach_div(exp_avg, denom)
    print(f"Adam returning: type={type(u32)}, shape={u32.shape}")
    return u32

def cosmos_update_llama_style(grad, U, S, beta2):
    """Update U and S matrices for LLaMA-style COSMOS."""
    U_old = U.detach().clone()
     
    U.data = beta2 * torch.matmul(U, S) + (1 - beta2) * torch.matmul(grad.T, torch.matmul(grad, U))
    U.data, _ = torch.linalg.qr(U, mode='reduced')
    S.data = beta2 * torch.matmul(U_old, torch.matmul(S, U_old.T)) + \
             (1 - beta2) * torch.matmul(grad.T,grad)
    S.data = torch.matmul(U.T, torch.matmul(S,U))

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration for matrix inverse square root.
    From the official COSMOS implementation.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)  # ensure top singular value <= 1
    
    if G.size(0) > G.size(1):
        X = X.T
        
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
        
    if G.size(0) > G.size(1):
        X = X.T
        
    return X

def init_preconditioner_cosmos(grad, state, rank=64):
    """
    Initialize COSMOS preconditioner for a single gradient tensor.
    """

    device = grad.device
    dtype = grad.dtype
    
    # LLaMA-style: U is (cols, rank), S is (rank, rank) 
    if len(grad.shape) > 1:
        actual_rank = min(rank, grad.size(1))
        state["U"] = torch.eye(grad.size(1), actual_rank, dtype=dtype, device=device)
    else:
        actual_rank = rank
        state["U"] = torch.eye(grad.size(0), actual_rank, dtype=dtype, device=device)

    
    state["S"] = torch.eye(actual_rank, dtype=dtype, device=device)
    state["adam_exp_avg_sq"] = torch.zeros_like(grad)
    # exp_avg_sq should be in low-rank space: (rows, rank)
    state["exp_avg_sq"] = torch.zeros(grad.size(0), actual_rank, dtype=dtype, device=device)


@decorator_knowngood
def _compilable_LION_(
    exp_avg_c: List[Tensor],  # c_t (momentum buffer)
    exp_avg_m: List[Tensor],  # m_t (second moment buffer) 
    grad: List[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)
    
    g32 = list(map(promote, grad))
    
    # Update c_t (first moment with beta1)
    exp_avg_c32 = _lerp(exp_avg_c, g32, beta1)
    
    # Update m_t (second moment with beta2) 
    exp_avg_m32 = _lerp(exp_avg_m, g32, beta2)
    
    # Compute update direction using interpolation between c_t and m_t
    # LION uses: sign(beta1 * m_{t-1} + (1 - beta1) * g_t)
    # which is equivalent to: sign(lerp(m_{t-1}, g_t, 1-beta1))
    interpolated = _lerp(exp_avg_m32, g32, 1 - beta1)
    if isinstance(interpolated, (list, tuple)) and len(interpolated) > 0:
        # If interpolated is already a list/tuple of tensors
        update_direction = torch._foreach_sign(interpolated)
    else:
        # If interpolated is a single tensor, wrap it in a list
        update_direction = torch._foreach_sign([interpolated])
        # If it was a single tensor, unwrap the result
        if not isinstance(grad, (list, tuple)):
            update_direction = update_direction[0]
    
    # Copy results back
    copy_stochastic_list_(exp_avg_c, exp_avg_c32)
    copy_stochastic_list_(exp_avg_m, exp_avg_m32) 
    copy_stochastic_list_(grad, update_direction)


def LION_(
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    grad: List[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    eps: float = 1e-8,
):
    
    exp_avg, exp_avg_sq, grad = map(list_guard, (exp_avg, exp_avg_sq, grad))
    beta1, beta2, step = scalar_guard(beta1, beta2, step , exp_avg[0])
    _compilable_LION_(exp_avg, exp_avg_sq, grad, beta1, beta2, step)
    return grad


@decorator_knowngood
def _fused_compilable_adam_(
    y: List[Tensor],
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    decay: Tensor,
    lr: Tensor,
    eps: Tensor,
    caution: bool,
):
    # Use PyTorch's standard bias correction instead of beta_debias
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    
    # Promote tensors to float32
    u32, g32 = [list(map(promote, x)) for x in [update, grad]]
    
    exp_avg32 = _lerp(exp_avg, u32, beta1)  
    exp_avg_sq_updated = _compilable_exp_avg_sq_(exp_avg_sq, u32, beta2, eps, [None])
    
    # Apply PyTorch-style bias correction
    # Divide by bias correction factors instead of debiased betas
    exp_avg_corrected = torch._foreach_div(exp_avg32, bias_correction1)
    #exp_avg_sq_corrected = torch._foreach_div(exp_avg_sq_updated, bias_correction2)
    
    # Compute denominator: sqrt(corrected_second_moment) + eps
    denom = torch._foreach_sqrt(exp_avg_sq_updated)
    denom = torch._foreach_add(denom, eps)
    
    # Compute update: corrected_first_moment / denom
    u32 = torch._foreach_div(exp_avg_corrected, denom)
    
    # Apply parameter update
    _compilable_update_(y, u32, decay, lr, caution, g32)


def fused_adam_(
    y: List[Tensor],
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    lr: float,
    eps: float,
    decay: float,
    caution: bool,
):
    
    y, exp_avg, exp_avg_sq, grad = list_guard(y, exp_avg, exp_avg_sq, grad)
    beta1, beta2, step, lr = scalar_guard(beta1, beta2, step, lr, y[0])
    _fused_compilable_adam_(y, exp_avg, exp_avg_sq, update, grad, beta1, beta2, step, decay, lr, eps, caution)


@decorator_knowngood
def _fused_compilable_adan_(
    y: List[Tensor],
    exp_avg: List[Tensor],           # m_k (first moment)
    exp_avg_sq: List[Tensor],        # n_k (second moment for Adan)
    exp_avg_diff: List[Tensor],      # v_k (gradient difference)
    prev_grad: List[Tensor],         # g_{k-1}
    update: List[Tensor],
    grad: List[Tensor],              # g_k
    beta1: Tensor,                   # β₁
    beta2: Tensor,                   # β₂  
    beta3: Tensor,                   # β₃
    step: Tensor,
    decay: Tensor,
    lr: Tensor,
    eps: Tensor,
    caution: bool,
):
    # Promote tensors to float32
    u32, g32 = [list(map(promote, x)) for x in [update, grad]]
    prev_grad32 = list(map(promote, prev_grad))
    
    # Compute gradient difference: g_k - g_{k-1}
    grad_diff = torch._foreach_sub(g32, prev_grad32)
    
    # Update first moment: m_k = (1 - β₁) * m_{k-1} + β₁ * g_k
    exp_avg32 = _lerp(exp_avg, g32, beta1)
    
    # Update gradient difference moment: v_k = (1 - β₂) * v_{k-1} + β₂ * (g_k - g_{k-1})
    exp_avg_diff32 = _lerp(exp_avg_diff, grad_diff, beta2)
    
    # Compute weighted gradient: g_k + (1 - β₂) * (g_k - g_{k-1})
    one_minus_beta2 = 1.0 - beta2
    weighted_grad_diff = torch._foreach_add(g32, grad_diff, alpha=one_minus_beta2)
    
    # Square the weighted gradient difference
    weighted_grad_diff_sq = torch._foreach_mul(weighted_grad_diff, weighted_grad_diff)
    
    # Update second moment: n_k = (1 - β₃) * n_{k-1} + β₃ * [g_k + (1 - β₂) * (g_k - g_{k-1})]²
    exp_avg_sq32 = _lerp(exp_avg_sq, weighted_grad_diff_sq, beta3)
    
    # Bias correction (optional - can be disabled for standard Adan)
    bias_correction1 = 1 - beta1 ** step
    bias_correction3 = 1 - beta3 ** step
    
    # Apply bias correction to first moment
    exp_avg_corrected = torch._foreach_div(exp_avg32, bias_correction1)
    exp_avg_sq_corrected = torch._foreach_div(exp_avg_sq32, bias_correction3)
    
    # Compute denominator: sqrt(n_k) + eps
    denom = torch._foreach_sqrt(exp_avg_sq_corrected)
    denom = torch._foreach_add(denom, eps)
    
    # Compute step size for each parameter: η / (√n_k + ε)
    step_size = torch._foreach_div([lr] * len(denom), denom)
    
    # Compute update direction: m_k + (1 - β₂) * v_k
    update_direction = torch._foreach_add(exp_avg_corrected, exp_avg_diff32, alpha=one_minus_beta2)
    
    # Scale by step size: η_k ∘ (m_k + (1 - β₂) * v_k)
    u32 = torch._foreach_mul(update_direction, step_size)
    
    # Update previous gradient for next iteration
    torch._foreach_copy_(prev_grad, g32)
    
    # Apply parameter update with weight decay
    _compilable_update_(y, u32, decay, lr, caution, g32)

def fused_adan_(
    y: List[Tensor],
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    lr: float,
    eps: float,
    decay: float,
    caution: bool,
):
    
    y, exp_avg, exp_avg_sq, grad = list_guard(y, exp_avg, exp_avg_sq, grad)
    beta1, beta2, step, lr = scalar_guard(beta1, beta2, step, lr, y[0])
    _fused_compilable_adan_(y, exp_avg, exp_avg_sq, update, grad, beta1, beta2, step, decay, lr, eps, caution)


@decorator_knowngood
def _fused_compilable_nadam_(
    y: List[Tensor],
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    decay: Tensor,
    lr: Tensor,
    eps: Tensor,
    caution: bool,
):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)
    
    u32, g32 = [list(map(promote, x)) for x in [update, grad]]
    exp_avg32 = _lerp(exp_avg, u32, beta1)
    denom = _compilable_exp_avg_sq_(exp_avg_sq, u32, beta2, eps, [None])
    u32 = torch._foreach_div(exp_avg32, denom)
    
    beta1_original = beta1
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)

    u32, g32 = [list(map(promote, x)) for x in [update, grad]]
    exp_avg32 = _lerp(exp_avg, u32, beta1)
    denom = _compilable_exp_avg_sq_(exp_avg_sq, u32, beta2, eps, [None])
    exp_avg32 = _lerp(exp_avg, u32, beta1_original)
    u32 = torch._foreach_div(exp_avg32, denom)
    _compilable_update_(y, u32, decay, lr, caution, g32)
    exp_avg32 = _undo_lerp(exp_avg, g32, beta1_original)

def fused_nadam_(
    y: List[Tensor],
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    lr: float,
    eps: float,
    decay: float,
    caution: bool,
):
    
    y, exp_avg, exp_avg_sq, grad = list_guard(y, exp_avg, exp_avg_sq, grad)
    beta1, beta2, step, lr = scalar_guard(beta1, beta2, step, lr, y[0])
    _fused_compilable_nadam_(y, exp_avg, exp_avg_sq, update, grad, beta1, beta2, step, decay, lr, eps, caution)


@decorator_knowngood
def _fused_compilable_STORM_plus_(
    y: List[Tensor],
    exp_avg_d: List[Tensor],
    exp_avg_g: List[Tensor],
    sum_of_norm_grad_sq: List[Tensor],
    sum_of_norm_d_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    prev_grads: List[Tensor],
    step: Tensor,
    decay: Tensor,
    lr: Tensor,
    eps: Tensor,
    caution: bool,
    a: float,
):
    u32, g32, prev_g32 = [list(map(promote, x)) for x in [update, grad, prev_grads]]
    
    if prev_g32 and len(prev_g32) == len(exp_avg_d):
        exp_avg_corrected = torch._foreach_sub(exp_avg_d, prev_g32)
    else:
        exp_avg_corrected = exp_avg_d

    exp_avg_d = torch._foreach_mul(exp_avg_corrected, 1 - a)
    exp_avg_d = torch._foreach_add(u32, exp_avg_corrected)
    
    norm_grad = torch._foreach_norm(g32)
    norm_grad_sq = torch._foreach_mul(norm_grad, norm_grad)
    # use old sum of the norm of grads squared
    one_plus = torch._foreach_add(sum_of_norm_grad_sq, 1.0)
    a = torch._foreach_pow(one_plus, -2.0/3.0)
    
    norm_d = torch._foreach_norm(exp_avg_d)
    norm_d_sq = torch._foreach_mul(norm_d, norm_d)
    sum_of_norm_d_sq = torch._foreach_add(sum_of_norm_d_sq, norm_d_sq)

    non_zero_a = torch._foreach_add(a, eps)
    eta_t_denom = torch._foreach_div(sum_of_norm_d_sq, non_zero_a)
    eta_t = torch._foreach_pow(eta_t_denom, -1.0/3.0) 
    # Apply eta_t to all parameters
    u32 = torch._foreach_mul(exp_avg_d, eta_t)
    sum_of_norm_grad_sq = torch._foreach_add(sum_of_norm_grad_sq, norm_grad_sq)
    _compilable_update_(y, u32, decay, torch.tensor(1.0), caution, g32)


def fused_STORM_plus_(
    y: List[Tensor],
    exp_avg_d: List[Tensor],
    exp_avg_g: List[Tensor],
    sum_of_norm_grad_sq: List[Tensor],
    sum_of_norm_d_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    prev_grads: List[Tensor],
    step: Tensor,
    decay: Tensor,
    lr: Tensor,
    eps: Tensor,
    caution: bool,
    a: Tensor,
):
    
    y, exp_avg_d, exp_avg_g, grad, prev_grads = list_guard(y, exp_avg_d, exp_avg_g, grad, prev_grads)
    step, lr, a = scalar_guard(step, lr, a, y[0])
    _fused_compilable_STORM_plus_(y, exp_avg_d, exp_avg_g, sum_of_norm_grad_sq, sum_of_norm_d_sq, update, grad, prev_grads, step, decay, lr, eps, caution, a)


@decorator_knowngood
def _fused_compilable_STORM_(
    y: List[Tensor],
    exp_avg_d: List[Tensor],
    exp_avg_g: List[Tensor],
    sum_of_norm_grad_sq: List[Tensor],
    sum_of_norm_d_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    prev_grads: List[Tensor],
    step: Tensor,
    decay: Tensor,
    lr: Tensor,
    eps: Tensor,
    caution: bool,
    eta: Tensor, 
):
    """ Setting k=lr, w=eps and c=1??"""
    u32, g32, prev_g32 = [list(map(promote, x)) for x in [update, grad, prev_grads]]
    
    if prev_g32 and len(prev_g32) == len(exp_avg_d):
        exp_avg_corrected = torch._foreach_sub(exp_avg_d, prev_g32)
    else:
        exp_avg_corrected = exp_avg_d
    a = torch._foreach_pow(eta, 2.)
    exp_avg_d = _lerp(exp_avg_corrected, u32, a)
   
    exp_avg_g = torch._foreach_add(exp_avg_g, g32)
    
    norm_grad_sq = torch._foreach_norm(g32)
    sum_of_norm_grad_sq = torch._foreach_add(sum_of_norm_grad_sq, norm_grad_sq)
    # add epsilon to protect against dividing by zero
    sum_of_norm_grad_sq = torch._foreach_add(sum_of_norm_grad_sq, eps)
    
    eta_denom = torch._foreach_pow(sum_of_norm_grad_sq, 1.0/3.0)
    eta = torch._foreach_div(lr, eta_denom)
    
    u32 = torch._foreach_mul(exp_avg_d, eta)
    
    _compilable_update_(y, u32, decay, torch.tensor(1.0), caution, g32)


def fused_STORM_(
    y: List[Tensor],
    exp_avg_d: List[Tensor],
    exp_avg_g: List[Tensor],
    sum_of_norm_grad_sq: List[Tensor],
    sum_of_norm_d_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    prev_grads: List[Tensor],
    step: Tensor,
    decay: Tensor,
    lr: Tensor,
    eps: Tensor,
    caution: bool,
    eta: Tensor,
):
    
    y, exp_avg_d, exp_avg_g, grad, prev_grads = list_guard(y, exp_avg_d, exp_avg_g, grad, prev_grads)
    step, lr, eta = scalar_guard(step, lr, eta, y[0])
    _fused_compilable_STORM_(y, exp_avg_d, exp_avg_g, sum_of_norm_grad_sq, sum_of_norm_d_sq, update, grad, prev_grads, step, decay, lr, eps, caution, eta)


@decorator_knowngood
def _fused_compilable_MARSAdamW_(
    y: List[Tensor],
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    prev_grads: List[Tensor],
    scaled_gamma: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    decay: Tensor,
    lr: Tensor,
    eps: Tensor,
    caution: bool,
):
   
    u32, g32, prev_g32 = [list(map(promote, x)) for x in [update, grad, prev_grads]]
 
    if prev_g32 and len(prev_g32) == len(u32):
        u32_corrected = torch._foreach_sub(u32, prev_g32)
    else:
        print("No Mars correction applied as tensor sizes didn't match")
        u32_corrected = u32
   
    gradient_correction = torch._foreach_mul(u32_corrected, scaled_gamma)
    u32_gamma_corrected = torch._foreach_add(u32, gradient_correction)
    
    norms = torch._foreach_norm(u32_gamma_corrected)
    
    ones_tensor = torch.tensor(1, device=u32_gamma_corrected[0].device, dtype=u32_gamma_corrected[0].dtype)
    # update clipping
    clip_coef = torch._foreach_minimum(torch._foreach_reciprocal(norms), ones_tensor)
    
    u32_gamma_corrected = torch._foreach_mul(u32_gamma_corrected, clip_coef)
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)
    exp_avg32 = _lerp(exp_avg, u32_gamma_corrected, beta1)
    denom = _compilable_exp_avg_sq_(exp_avg_sq, u32_gamma_corrected, beta2, eps, [None])
    u32 = torch._foreach_div(exp_avg32, denom)
    _compilable_update_(y, u32, decay, lr, caution, g32)


def fused_MARSAdamW_(
    y: List[Tensor],
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    prev_grads: List[Tensor],
    mars_gamma: float,
    beta1: float,
    beta2: float,
    step: int,
    lr: float,
    eps: float,
    decay: float,
    caution: bool,
):
    
    scaled_gamma = (mars_gamma * beta1)/(1 - beta1)
    y, exp_avg, exp_avg_sq, grad, prev_grads = list_guard(y, exp_avg, exp_avg_sq, grad, prev_grads)
    scaled_gamma, beta1, beta2, step, lr = scalar_guard(scaled_gamma, beta1, beta2, step, lr, y[0])
    _fused_compilable_MARSAdamW_(y, exp_avg, exp_avg_sq, update, grad, prev_grads, scaled_gamma, beta1, beta2, step, decay, lr, eps, caution)


@decorator_knowngood
def _compilable_laprop_(
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    grad: List[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    eps: Tensor,
):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)

    gp32 = list(map(promote, grad))
    denom = _compilable_exp_avg_sq_(exp_avg_sq, gp32, beta2, eps, [None])
    gp32 = torch._foreach_div(gp32, denom)
    gp32 = _lerp(exp_avg, gp32, beta1)
    copy_stochastic_list_(grad, gp32)


def laprop_(
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    grad: List[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    eps: float = 1e-8,
):
    exp_avg, exp_avg_sq, grad = list_guard(exp_avg, exp_avg_sq, grad)
    beta1, beta2, step, eps = scalar_guard(beta1, beta2, step, eps, exp_avg[0])
    _compilable_laprop_(exp_avg, exp_avg_sq, grad, beta1, beta2, step, eps)
    return grad


@decorator_knowngood
def _fused_compilable_laprop_(
    y: List[Tensor],
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    beta1: Tensor,
    beta2: Tensor,
    step: Tensor,
    lr: Tensor,
    decay: Tensor,
    caution: bool,
    eps: Tensor,
):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)

    u32, gp32 = [list(map(promote, x)) for x in [update, grad]]
    denom = _compilable_exp_avg_sq_(exp_avg_sq, u32, beta2, eps, [None])
    u32 = torch._foreach_div(u32, denom)
    u32 = _lerp(exp_avg, u32, beta1)
    _compilable_update_(y, u32, decay, lr, caution, gp32)


def fused_laprop_(
    y: List[Tensor],
    exp_avg: List[Tensor],
    exp_avg_sq: List[Tensor],
    update: List[Tensor],
    grad: List[Tensor],
    beta1: float,
    beta2: float,
    step: int,
    lr: float,
    decay: float,
    caution: bool,
    eps: float = 1e-8,
):
    exp_avg, exp_avg_sq, grad, y = list_guard(exp_avg, exp_avg_sq, grad, y)
    beta1, beta2, step, lr, eps = scalar_guard(beta1, beta2, step, lr, eps, exp_avg[0])
    _fused_compilable_laprop_(y, exp_avg, exp_avg_sq, update, grad, beta1, beta2, step, lr, decay, caution, eps)


@decorator_knowngood
def _fused_compilable_adopt_(y, update, grad, exp_avg_sq, exp_avg, beta1, beta2, step, lr, eps, decay, caution):
    u32, g32, exp_avg_sq32 = [list(map(promote, x)) for x in [update, grad, exp_avg_sq]]
    _compilable_update_(y, u32, decay, lr, caution, g32)

    beta1 = beta_debias(beta1, step)
    denom = [eps_sqrt(d, eps) for d in exp_avg_sq32]
    stochastic_lerp_(exp_avg, torch._foreach_div(g32, denom), 1 - beta1)

    beta2 = beta_debias(beta2, step + 1)
    stochastic_lerp_(exp_avg_sq, torch._foreach_mul(g32, g32), 1 - beta2)


def fused_adopt_(y, update, grad, exp_avg_sq, exp_avg, beta1, beta2, step, lr, eps, decay, caution):
    exp_avg, exp_avg_sq, grad, y = list_guard(exp_avg, exp_avg_sq, grad, y)
    beta1, beta2, step, lr = scalar_guard(beta1, beta2, step, lr, exp_avg[0])
    _fused_compilable_adopt_(y, update, grad, exp_avg_sq, exp_avg, beta1, beta2, step, lr, eps, decay, caution)


@decorator_knowngood
def _compilable_adopt_(grad, exp_avg_sq, exp_avg, beta1, beta2, step, eps):
    g32, exp_avg_sq32 = [list(map(promote, x)) for x in [grad, exp_avg_sq]]
    update = [e.clone() for e in exp_avg]

    beta1 = beta_debias(beta1, step)
    denom = [eps_sqrt(d, eps) for d in exp_avg_sq32]
    stochastic_lerp_(exp_avg, torch._foreach_div(g32, denom), 1 - beta1)

    stochastic_lerp_(exp_avg_sq, torch._foreach_mul(g32, g32), 1 - beta2)

    copy_stochastic_list_(grad, update)


def adopt(grad, exp_avg_sq, exp_avg, beta1, beta2, step, eps: float = 1e-8):
    exp_avg, exp_avg_sq, grad = list_guard(exp_avg, exp_avg_sq, grad)
    beta1, beta2, step, eps = scalar_guard(beta1, beta2, step, eps, exp_avg[0])
    _compilable_adopt_(grad, exp_avg_sq, exp_avg, beta1, beta2, step, eps)
    return grad


def stochastic_round_list_(ref: List[Tensor], source: List[Tensor]):
    return [stochastic_round_(r, s) for r, s in zip(ref, source)]


@decorator_knowngood
def stochastic_round_(ref: Tensor, source: Tensor):
    if source.dtype == torch.bfloat16 or ref.dtype == source.dtype:
        return source
    if ref.dtype != torch.bfloat16:
        return source.to(ref.dtype)
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32
    return result.view(dtype=torch.float32).bfloat16()


@decorator_knowngood
def _compilable_copy_stochastic_(target: Tensor, source: Tensor):
    target.copy_(stochastic_round_(target, source))


def copy_stochastic_(target: Tensor, source: Tensor):
    if target.dtype == torch.bfloat16 and source.dtype in (torch.float16, torch.float32, torch.float64):
        _compilable_copy_stochastic_(target, source.float())
    set_(target, source)


@decorator_knowngood
def _compilable_update_(
    p: List[Tensor], u: List[Tensor], decay: Tensor, lr: Tensor, caution: bool, g: List[Optional[Tensor]]
):
    for u_, g_, p_ in zip(u, g, p):  # lr is data-dependent -> can't compile a foreach
        u_ = promote(u_.view_as(p_))
        p32_ = promote(p_)
        if caution:
            u_ = _compilable_cautioning(promote(g_), u_)
        p32_ = p32_ * (1 - decay * lr) + u_ * -lr
        copy_stochastic_(p_, p32_)


def update_param_(
    param: List[Tensor], update: List[Tensor], lr: float, decay: float, caution: bool = False, grad: List[Tensor] = None
):
    param, update, grad = list_guard(param, update, grad)
    lr = scalar_guard(lr, param[0])
    if not caution:
        grad = [None] * len(param)
    _compilable_update_(param, update, decay, lr, caution, grad)


def precond_schedule(step, precond_scheduler):
    precond_prob = max(step, 1) ** precond_scheduler[0]
    precond_prob = math.log10(precond_prob)
    precond_prob = precond_prob ** precond_scheduler[1] + 1
    return 1 / precond_prob


def get_soap_precond_schedule(precond_scheduler):
    return functools.partial(precond_schedule, precond_scheduler=precond_scheduler)


def _max_idx(x: List[int]):
    return len(x) - 1 - np.argmax(x[::-1])  # we want to start counting from the back, as torch is fan-out/fan-in


@decorator_knowngood
def stable_exp(x: Tensor):
    # fp16:
    #   exp(x) is stable in [-17, 11]
    #   `stable_exp` extends to [-17, 17]
    #   average error (in [-10, 10]) increased from 2.288e-3 to 2.299e-3
    # fp32:
    #   exp(x) is stable in [-103, 88]
    #   `stable_exp` extends to [-103, 103]
    #   average error (in [-87, 87]) reduced from 3.309-06 to 3.224-06
    return torch.where(x > 0, 1 / (-x).exp(), x.exp())


def _lse_mean(x: Tensor, pow: float, eps: float) -> Tensor:
    # ln(mean(x ** pow) ** (1 / pow / 2))
    normalization = math.log(x.numel())
    x = x.double()
    x = x.abs()
    x = x.clamp(min=eps)
    x = x.log()
    x = x * pow
    x = x.flatten()
    x = x.logsumexp(dim=0)  # log(sum(exp( log(x) * P ) - more stable than sum(x ** P)
    x = x - normalization  # sum -> mean (divide by x.numel() in log space)
    return x / pow / 2


@decorator_knowngood
def mean_root(x: torch.Tensor, pow: float, eps=1e-12):
    # 1 / (mean(x ** pow) ** (1 / pow / 2))
    return stable_exp(-_lse_mean(x, pow, eps))


@decorator_knowngood
def divided_root(x: torch.Tensor, y: torch.Tensor, pow0: float, pow1: float, eps=1e-12):
    # mean(x ** pow0) ** (1 / pow0 / 2) / mean(y ** pow1) ** (1 / pow1 / 2)
    return stable_exp(_lse_mean(x, pow0, eps) - _lse_mean(y, pow1, eps))


class PrecondInitError(ValueError):
    pass


def precond_init_scale(scale, scale_scale, scale_power, grad, hessian_vector, vector, scale_max: float = 100):
    automatic_scale = True
    manual_hint = " Set it manually using `precond_init_scale=0.1`"
    scale_scale = 1 if scale_scale is None else scale_scale

    if scale is not None:
        automatic_scale = False
        warn_once(
            "It's recommended to use precond_init_scale=None (default since 1.7.x), which uses advanced heuristics."
        )
        if scale_scale != 1:
            warn_once(
                "precond_init_scale_scale multiplies the precond_init_scale by a constant factor. With a fixed precond_init_scale, you should explicitly fuse it."
            )
        if scale_power is not None:
            warn_once(
                "precond_init_scale_power is used to compute precond_init_scale ** precond_init_scale_power. With a fixed precond_init_scale, you should explicitly fuse it."
            )
    elif hessian_vector is None:
        scale = mean_root(grad, 4) * scale_scale
    else:
        scale = divided_root(vector, hessian_vector, 2, 4) * scale_scale

    if automatic_scale:
        scale_power = 0.5 if scale_power is None else scale_power
        scale = scale**scale_power

    if isinstance(scale, torch.Tensor):
        scale = scale.item()  # slow, but necessary

    if np.isfinite(scale):
        if scale > scale_max:  # fallthrough to later checks
            warn_once(f"The computed precond_init_scale {scale} is outside of the expected range.{manual_hint}")
        else:
            return scale

    if not automatic_scale:
        raise PrecondInitError("The manually set precond_init_scale is not finite")

    for x in (grad, hessian_vector, vector):
        if x is None:
            continue
        if torch.allclose(x, torch.zeros_like(x)):
            raise PrecondInitError(
                f"Grad or HVP is all 0s, causing NaNs in precond_init_scale computation.{manual_hint}"
            )
        if not torch.isfinite(x).all().item():
            raise PrecondInitError("Grad or HVP is not finite")

    if np.isfinite(scale):
        return scale

    raise PrecondInitError(f"Computed precond_init_scale is not finite.{manual_hint}")


def init_lra(
    grad, param_count, scale, scale_scale, scale_power, rank, hessian_vector, vector, dtype=None, eps: float = 10
):
    # "+10 to 1) avoid /0; 2) make sure that norm(U*V') << 1 even when rank_of_approximation=1" from @lixilinx at
    # https://github.com/lixilinx/psgd_torch/blob/590cd3f125552998ed20028be096652540e2a200/preconditioned_stochastic_gradient_descent.py#L829C11-L829C14
    scale = precond_init_scale(scale, scale_scale, scale_power, grad, hessian_vector, vector)
    uv_scale = (param_count * (rank + eps)) ** -0.5
    U = torch.randn((*grad.shape, rank), dtype=dtype, device=grad.device) * uv_scale
    V = torch.randn((*grad.shape, rank), dtype=dtype, device=grad.device) * uv_scale
    d = torch.full_like(grad, scale, dtype=dtype, device=grad.device)
    return U, V, d


def init_Q_exprs(
    grad,
    scale,
    scale_scale,
    scale_power,
    max_size,
    min_ndim_triangular,
    memory_save_mode,
    hessian_vector,
    vector,
    dtype=None,
):
    """
    For a scalar or tensor `grad`, we initialize its preconditioner Q and
    reusable einsum expressions for updating Q and preconditioning gradient.

    precond init scale computation from
    https://github.com/lixilinx/psgd_torch/blob/1943e66596111e78157ca1b72b31c1dfdf0653ef/preconditioned_stochastic_gradient_descent.py#L2208-L2227
    """
    scale = precond_init_scale(scale, scale_scale, scale_power, grad, hessian_vector, vector)
    dtype = dtype if dtype is not None else grad.dtype
    shape = grad.shape

    if len(shape) == 0:  # scalar
        Q = [scale * torch.ones_like(grad, dtype=dtype)]
        return Q

    scale = scale ** (1 / len(shape))

    dim_diag = [False for _ in shape]
    if memory_save_mode is None:
        pass
    elif memory_save_mode == "one_diag":
        dim_diag[_max_idx(shape)] = True
    elif memory_save_mode == "smart_one_diag":
        sorted_shape = sorted(shape)
        if len(shape) >= 2 and sorted_shape[-1] > sorted_shape[-2]:
            dim_diag[_max_idx(shape)] = True
    elif memory_save_mode == "one_triu":
        shape_ranks = np.argsort(np.argsort(shape))  # ranks
        dim_diag = (shape_ranks != 0).tolist()  # only triu the smallest
    elif memory_save_mode == "all_diag":
        dim_diag = [True for _ in shape]
    else:
        raise ValueError(
            f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
            "[None, 'one_diag', 'all_diag', 'smart_one_diag']"
        )

    Q = []
    for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
        if size == 1 or size > max_size or len(shape) < min_ndim_triangular or dim_d:
            # use diagonal matrix as preconditioner for this dim
            Q.append(scale * torch.ones(size, dtype=promote(dtype), device=grad.device))
        else:
            # use triangular matrix as preconditioner for this dim
            Q.append(scale * torch.eye(size, dtype=dtype, device=grad.device))
    return Q


@decorator_knowngood
def psgd_balance_Q(Q):
    norms = [promote(q.norm(float("inf"))).log() for q in Q]
    geometric_mean = sum([n for n in norms]) / len(Q)
    for q, n in zip(Q, norms):
        q *= (geometric_mean - n).exp()


@decorator_knowngood
def _lra_flatten_and_balance(U: List[Tensor], V: List[Tensor], d: List[Tensor]):
    u_norm = sum(u.square().sum().double() for u in U)
    v_norm = sum(v.square().sum().double() for v in V)
    scale = (u_norm / v_norm) ** 0.25  # sqrt of L2 norms; sqrt, as it's 2 factors
    scale = torch.where(torch.logical_and(torch.isfinite(scale), scale > 1e-6), scale, 1)
    stochastic_multiply_(U, [1 / scale] * len(U))
    stochastic_multiply_(V, [scale] * len(V))
    return multi_flatten((U, 1), (V, 1), (d, 0))


@decorator
def low_rank_mm(U: Tensor, V: Tensor, x: Tensor) -> Tensor:
    dtype = min_dtype([U, V, x])
    return x + compiled_einsum("br,gr,g->b", U.to(dtype), V.to(dtype), x.to(dtype)).to(x.dtype)


@decorator_knowngood
def _compilable_d_step(
    d: Tensor,
    d_orig: List[Tensor],
    invQtv: Tensor,
    vector: Tensor,
    inverse_precond_vector: Tensor,
    hessian_vector: Tensor,
    precond_hessian_vector: Tensor,
    eps: Tensor,
    step: Tensor,
    delayed: bool,
):
    precond_hessian_vector = promote(precond_hessian_vector)
    hessian_vector = promote(hessian_vector)
    vector = promote(vector)
    inverse_precond_vector = promote(inverse_precond_vector)
    invQtv = promote(invQtv)
    inverse_precond_vector = invQtv - inverse_precond_vector

    nablaD = promote(d).square() * precond_hessian_vector * hessian_vector - vector * inverse_precond_vector

    """
    1) Sketching
        1.1) multiply, square, etc. in high precision (to avoid numerical errors + doesn't increase cost)
        1.2) reduced-precision selection of largest element (halves memory traffic)
    2) Computation
        2.1) select relevant indices
        2.2) redo 1.1 in double precision for scalar values
        2.3) return high-precision normalized step-size
    overall, this should REDUCE the cost of the operation compared to baseline (-> less memory traffic) while
    improving precision
    """
    a0 = promote(d) * precond_hessian_vector
    a1 = vector
    b0 = inverse_precond_vector / promote(d)
    b1 = hessian_vector

    divisor = (a0.square() + a1.square()) * (b0.square() + b1.square())
    idx = divisor.bfloat16().flatten().argmax()
    a = a0.index_select(0, idx).double().square() + a1.index_select(0, idx).double().square()
    b = b0.index_select(0, idx).double().square() + b1.index_select(0, idx).double().square()
    divisor = (a * b).sqrt().clamp(min=eps)
    step = -step / divisor

    # fused update(s)
    apply_flat_add(d_orig, nablaD, step)
    if not delayed:
        copy_stochastic_(d, promote(d) - nablaD * step)


def update_lra_precond_(
    U: List[Tensor],
    V: List[Tensor],
    d: List[Tensor],
    vector: Tensor,
    hessian_vector: Tensor,
    eps: float,
    step: float,
    delayed: bool,
    precond_u: bool,
):
    """
    Adapted from https://github.com/lixilinx/psgd_torch/blob/6dbea94915679d08a289928e6431b6ce07931aaf/preconditioned_stochastic_gradient_descent.py#L657
    """
    U_orig, V_orig, d_orig = U, V, d

    U, V, d = _lra_flatten_and_balance(U, V, d)

    dtype = min_dtype([U, V, vector, hessian_vector])
    U, V, vector, hessian_vector = U.to(dtype), V.to(dtype), vector.to(dtype), hessian_vector.to(dtype)

    eps = scalar_guard(eps, vector)

    Qh = low_rank_mm(U, V, d * hessian_vector)
    Ph = low_rank_mm(V, U, Qh)
    rank = U.size(1)

    VtU = compiled_einsum("br,bn->rn", V, U)  # (rank, rank)
    I = torch.eye(rank, dtype=VtU.dtype, device=VtU.device)
    IpVtU = I + VtU
    invQtv = vector / d

    # LU factorization to reuse computation
    try:
        LU, pivots = torch.linalg.lu_factor(IpVtU)
    except RuntimeError:
        # Error:
        # U[2,2] is zero and using it on lu_solve would result in a division by zero.
        # If you still want to perform the factorization, consider calling
        # linalg.lu(A, pivot) or linalg.lu_factor_ex(A, pivot)
        # ---
        # So, we skip this step and reattempt on the next one
        return U.to(U_orig[0].dtype), V.to(V_orig[0].dtype), d.to(d_orig[0].dtype)

    invQtv = invQtv - V @ torch.linalg.lu_solve(LU, pivots, (U.T @ invQtv).view(-1, 1), adjoint=True).flatten()
    invPv = U @ torch.linalg.lu_solve(LU, pivots, (V.T @ invQtv).view(-1, 1)).flatten()

    eps, step = scalar_guard(eps, step, vector)
    _compilable_d_step(d, d_orig, invQtv, vector, invPv, hessian_vector, Ph, eps, step, delayed)

    a, b = Qh, invQtv

    precond = V if precond_u else U
    atV = compiled_einsum("b,br->r", a, precond)  # o == one
    btV = compiled_einsum("b,br->r", b, precond)
    atVVt = compiled_einsum("r,br->b", atV, precond)
    btVVt = compiled_einsum("r,br->b", btV, precond)
    precond_step = step / (a.norm() * atVVt.norm() + b.norm() * btVVt.norm()).clamp(min=eps)
    if precond_u:
        a = compiled_einsum("b,r,rg->bg", a, atV, IpVtU)
        b = compiled_einsum("b,r,rg->bg", b, btV, IpVtU)
    else:
        a = a + compiled_einsum("br,r->b", V, atV)
        b = b + compiled_einsum("br,r->b", V, btV)
        a = compiled_einsum("b,r->br", a, atV)
        b = compiled_einsum("b,r->br", b, btV)
    apply_flat_add(U_orig if precond_u else V_orig, b - a, precond_step)
    if not delayed:
        stochastic_add_([U if precond_u else V], [b - a], precond_step)
    return U.to(U_orig[0].dtype), V.to(V_orig[0].dtype), d.to(d_orig[0].dtype)


def lra_precond(U: Tensor, V: Tensor, d: Tensor, g: Tensor):
    """
    As-is from https://github.com/lixilinx/psgd_torch/blob/6dbea94915679d08a289928e6431b6ce07931aaf/preconditioned_stochastic_gradient_descent.py#L744
    """
    new_g = low_rank_mm(U, V, d * g)
    return d * low_rank_mm(V, U, new_g)


@decorator_knowngood
def dampen_grad(g: Tensor, damp: float = 2**-13):
    # https://github.com/lixilinx/psgd_torch/blob/1943e66596111e78157ca1b72b31c1dfdf0653ef/preconditioned_stochastic_gradient_descent.py#L50
    v = torch.randn_like(g)
    return v, g + damp * g.abs().mean() * v


@decorator_knowngood
def _compilable_lra_update_(
    params: List[Tensor],
    update: List[Tensor],
    U: Tensor,
    V: Tensor,
    d: Tensor,
    lr: Tensor,
    decay: Tensor,
    caution: bool,
    grads: List[Tensor],
):
    update = lra_precond(U, V, d, flatten(update))
    start = 0
    update = update.flatten()
    for p, g in zip(params, grads):
        size = p.numel()
        update_param_(p, update[start : start + size].view_as(p), lr, decay, caution, g)
        start += size


def apply_lra_update(
    params: List[Tensor],
    update: Tensor,
    U: Tensor,
    V: Tensor,
    d: Tensor,
    lr: float,
    decay: float,
    caution: bool,
    grads: List[Tensor],
):
    params, grads = list_guard(params, grads)
    lr, decay = scalar_guard(lr, decay, params[0])
    _compilable_lra_update_(params, update, U, V, d, lr, decay, caution, grads)


@decorator_knowngood
def apply_flat_update(params: List[Tensor], update: Tensor):
    start = 0
    update = update.flatten()
    for p in params:
        size = p.numel()
        copy_stochastic_(p, update[start : start + size].view_as(p))
        start += size


@decorator_knowngood
def zero_(x: List[Tensor]):
    for i in x:
        i.zero_()


@decorator_knowngood
def apply_flat_add(params: List[Tensor], update: Tensor, alpha: Tensor):
    start = 0
    update = update.flatten()
    for p in params:
        size = p.numel()
        stochastic_add_([p], [update[start : start + size].view_as(p)], alpha)
        start += size


@decorator_knowngood
def extract_from_flat_update(params: List[Tensor], update: Tensor):
    start = 0
    outputs = []
    update = update.flatten()
    for p in params:
        size = p.numel()
        outputs.append(update[start : start + size].view_as(p))
        start += size
    return outputs


@decorator_knowngood
def flatten(x: List[Tensor], remaining: int = 0) -> Tensor:
    last_dim = x[0].shape[-remaining:] if remaining else []
    return torch.cat([i.reshape(-1, *last_dim) for i in x if i.numel()], 0)


@decorator_knowngood
def multi_flatten(*xs: Tuple[List[Tensor], int]):
    return [flatten(x, i) for x, i in xs]


@decorator_knowngood
def dampen_multiple(g: List[Tensor], damp: float = 2**-13):
    vs = []
    gs = []
    for g_ in g:
        v, g = dampen_grad(g_, damp)
        vs.append(v)
        gs.append(g)
    return flatten(vs), flatten(gs)


def casted_einsum(expr: str, *args: Tensor) -> Tensor:
    md = min_dtype(args)
    return compiled_einsum(expr, *[a.to(md) for a in args]).to(args[-1].dtype)


@decorator_knowngood
def _psgd_calc_scalars_(Qs: List[Tensor], conjB: Tensor):
    triangular_qs = []
    conjB = promote(conjB)
    for i, q in enumerate(Qs):
        q = promote(q)
        if q.dim() <= 1:
            if conjB.ndim == 0:
                conjB = conjB / q
            else:
                shape = [1] * conjB.ndim
                shape[i] = -1
                conjB = conjB / q.view(shape)
        else:
            triangular_qs.append((i, q))
    return triangular_qs, conjB


@decorator_knowngood
def _reshape_conjB(solved: Tensor, transposed_shape: List[int], original_shape: List[int], last_dim: int, new_dim: int):
    solved = solved.reshape(transposed_shape)
    solved = solved.transpose(-1, last_dim)
    solved = solved.reshape(original_shape)
    solved = solved.transpose(-1, new_dim)
    return solved.contiguous(), solved.shape


def ndim_tuple(Q: list[Tensor]) -> tuple:
    return tuple(q.ndim for q in Q)


def psgd_calc_A_and_conjB(G, Q, conjB):  # conjB ("V", "vector") == randn during hvp/whitening
    exprA = cached_precond_grad_expr(ndim_tuple(Q), G.ndim)  # calcA expr and cached precond expr are the same
    A = casted_einsum(exprA, *Q, G)
    solve = torch.compiler.disable(torch.linalg.solve_triangular)
    transposed_shape = original_shape = conjB.shape
    prev_i = -1
    qs, conjB = _psgd_calc_scalars_(Q, conjB)
    for i, tri_q in qs:
        conjB, transposed_shape = _reshape_conjB(conjB, transposed_shape, original_shape, prev_i, i)
        prev_i = i
        conjB = solve(tri_q, conjB, upper=True, left=False)
    conjB, _ = _reshape_conjB(conjB, transposed_shape, original_shape, prev_i, -1)
    return A, conjB


@decorator_knowngood
def _random_projection(x: Tensor, scale: Optional[Tensor]):
    if scale is None:
        scale = x.norm(float("inf")).clamp(min=1e-8)
    k = 2 ** math.ceil(math.log2(math.log2(min(x.shape))))  # next-largest-power-of-2 of log2-of-size
    norm = x.square().sum(0)
    indices = torch.topk(norm, k, largest=True).indices
    return x.index_select(1, indices).contiguous() / scale, scale


def max_singular_value_exact(A, use_lobpcg: bool = False):
    try:
        if use_lobpcg:
            A = A @ A.T
            eigval, _ = torch.compiler.disable(torch.lobpcg)(A, k=1, largest=True)
            return eigval[0].sqrt()
        else:
            return torch.linalg.svd(A, driver="gesvdj")[1].max()  # == linalg.matrix_norm(A, ord=2)
    except torch.linalg.LinAlgError:
        return torch.zeros((), device=A.device, dtype=A.dtype)


@decorator_knowngood
def max_singular_value_power_iter(A: Tensor, max_abs: Optional[Tensor] = None, iterations: int = 5):
    """
    Rayleigh quotient of row with the largest norm + optional power iterations
    """
    x_norm, max_idx = A.norm(dim=1).max(dim=0)
    x = A.index_select(0, max_idx).flatten().contiguous()
    A = A / x_norm
    x = x / x_norm
    for _ in range(iterations):
        x = A.T.mv(A.mv(x))  # A @ A.T @ x, but explicitly telling torch.compile not to compute the full matrix
        x = x / x.norm()
    return (x @ A.T.mv(A.mv(x))).sqrt() * x_norm


@decorator_knowngood
def max_singular_value_cholesky(A: Tensor, max_abs: Optional[Tensor] = None):
    """
    Adapted from @evanatyourservice
    """
    Y, max_abs = _random_projection(A, max_abs)
    Q = inplace_orthogonal_(Y, precise_zeroth_power_mode)
    Q = Q / max_abs
    Z = A.T @ Q
    W = inplace_orthogonal_(Z, precise_zeroth_power_mode)
    sketch_norm = max_singular_value_exact(Z.T @ W)
    return sketch_norm * max_abs


@decorator_knowngood
def max_singular_value(
    A: Tensor, max_abs: Optional[Tensor], max_svd: int = 32, use_cholesky: bool = False, power_iter: int = 0
) -> Tensor:
    if min(A.shape) <= max_svd:
        return max_singular_value_exact(A)  # SVD needs ~25% more runtime for size=32, but 0% error instead of 5%
    if use_cholesky or power_iter < 0:
        return max_singular_value_cholesky(A, max_abs)
    return max_singular_value_power_iter(A, None, iterations=power_iter)


@decorator_knowngood
def _psgd_default_preconditioner_grad(
    terms: List[Tuple[Tensor, Tensor]],
    Q: List[Tensor],
) -> List[Tensor]:
    out = []
    for q, (x, y) in zip(Q, terms):
        x = promote(x)
        y = promote(y)
        update = x - y
        if q.ndim < 2:
            update = q * update
        else:
            update = (q @ update).triu()
        out.append(update)
    return out


@decorator_knowngood
def _balance_to_triu(Q: "TriuOrLine", symmetric_output: bool = False):
    if isinstance(Q[0], tuple):
        psgd_balance_Q([o[1] for o in Q])
        return line_to_triu(Q, symmetric_output)
    psgd_balance_Q(Q)
    return Q


@functools.lru_cache(maxsize=None)
def calcG_expr(q_dim, g_dim):
    exprs = []
    base = einsum_base[:g_dim]
    for i, q in enumerate(q_dim):
        new = list(base)
        if q == 2:
            new[i] = "Z"
            out = f"{base[i]}Z"
        else:
            out = base[i]
        exprs.append(f"{base},{''.join(new)}->{out}")
    return exprs


@decorator
def psgd_update_precond(
    G: Tensor,
    precond_lr: float,
    oq: "TriuOrLine",
    store_triu_as_line: bool,
    velocity: Optional[List[Tensor]],
    beta2: float,
    ortho_method: Optional[str],
    V: Tensor,
    running_lower_bound: List[Tensor],
    lower_bount_beta: float,
    power_iter: int,
) -> None:
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    Q = _balance_to_triu(oq)
    exprGs = calcG_expr(ndim_tuple(Q), G.ndim)
    precond_lr, beta2, lower_bount_beta = scalar_guard(precond_lr, beta2, lower_bount_beta, G)

    A, conjB = psgd_calc_A_and_conjB(G, Q, V)
    terms = [(compiled_einsum(exprG, A, A), compiled_einsum(exprG, conjB, conjB)) for exprG in exprGs]
    del A, conjB, V
    updates = _psgd_default_preconditioner_grad(terms, Q)
    _psgd_precond_update_(
        updates, oq, running_lower_bound, lower_bount_beta, precond_lr, store_triu_as_line, power_iter
    )
    return None


@decorator_knowngood
def _psgd_precond_update_(
    matmuled: List[Optional[Tensor]],
    Q: "TriuOrLine",
    running_lower_bound: List[Tensor],
    lower_bount_beta: Tensor,
    precond_lr: Tensor,
    store_triu_as_line: bool,
    power_iter: int,
):
    for update, oq, lb_state in zip(matmuled, Q, running_lower_bound):
        if isinstance(oq, tuple):
            oq = oq[1]

        q = promote(oq)
        if update.ndim < 2:
            lb = update.norm(float("inf"))
        else:
            lb = max_singular_value(update, None, power_iter=power_iter)
            update = promote(update)
            if store_triu_as_line:
                update = triu_to_line([update])[0][1]

        lb = promote(lb)
        lb = lb.maximum(promote(lb_state) + (lb - promote(lb_state)) * (1 - lower_bount_beta))
        copy_stochastic_(lb_state, lb)
        copy_stochastic_(oq, q - update / lb * precond_lr)


@decorator_knowngood
def _psgd_quad_preconditioner_grad(GG: List[Tensor], Q: List[Tensor], numel: int):
    """
    I: Identity
    U: Update / gg / target
    Q: q, preconditioner
    scale: scalar scale
    ---
    U = T * scale - I
    F = I - U  # = 2I - U * scale
    O = F @ Q @ F - Q
    """
    out = []
    for gg, q in zip(GG, Q):
        if gg.ndim < 2:
            scale = max(1, gg.numel()) / numel
            target = promote(gg)
            update = target * scale - 1
            out.append(q - (1 - update) * q * (1 - update))
        else:
            scale = gg.size(0) / numel
            gg = 2 * torch.eye(gg.size(0), device=gg.device, dtype=gg.dtype) - gg * scale
            update = q - gg @ q @ gg
            out.append(update + update.T)  # make matrix symmetric
    return out


@decorator
def inverse_free_psgd_update_precond(
    G: Tensor,
    precond_lr: float,
    oq: List[Tensor],
    store_triu_as_line: bool,
    velocity: Optional[List[Tensor]],
    beta2: float,
    ortho_method: Optional[str],
    V: None,
    running_lower_bound: List[Tensor],
    lower_bount_beta: float,
    power_iter: int,
) -> Tensor:
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    assert V is None
    assert ortho_method is None
    assert velocity is None
    del V, ortho_method, velocity

    Q = _balance_to_triu(oq, True)
    precond_lr, beta2, lower_bount_beta = scalar_guard(precond_lr, beta2, lower_bount_beta, G)
    exprGs = calcG_expr(ndim_tuple(Q), G.ndim)

    G = psgd_precond_grad(G, Q)
    terms = [compiled_einsum(exprG, G, G) for exprG in exprGs]
    matmuled = _psgd_quad_preconditioner_grad(terms, Q, G.numel())
    _psgd_precond_update_(
        matmuled, oq, running_lower_bound, lower_bount_beta, precond_lr, store_triu_as_line, power_iter
    )
    return G


@decorator_knowngood
def _compilable_l2_clip_(x, clip_at):
    ref = x
    x = list(map(promote, x))
    norm = torch._foreach_norm(x)
    torch._foreach_maximum_(norm, clip_at)
    out = torch._foreach_div(x, norm)
    return stochastic_round_list_(ref, out)


def l2_normalization_(x, clip_at: float = 1e-8):
    x = list_guard(x)
    return _compilable_l2_clip_(x, clip_at)


def l2_clip_(x, clip_at: float = 1.0):
    x = list_guard(x)
    return _compilable_l2_clip_(x, clip_at)


@decorator_knowngood
def _compilable_rmsnorm_clip_(x, clip_at):
    x = list(map(promote, x))
    norm = torch._foreach_norm(x)
    norm = [n.div_(x_.numel() ** 0.5) for n, x_ in zip(norm, x)]
    torch._foreach_maximum_(norm, clip_at)
    return torch._foreach_div(x, norm)


def rmsnorm_clip_(x, clip_at: float = 1.0):
    x = list_guard(x)
    return _compilable_rmsnorm_clip_(x, clip_at)


@decorator_knowngood
def _compilable_global_rmsnorm_clip_(x, clip_at):
    norm = 0
    items = 0
    for i in x:
        norm += promote(i.square().sum())
        items += i.numel()
    norm = (norm / items) ** 0.5
    norm = norm.clamp(min=clip_at)
    stochastic_multiply_(x, 1 / norm)
    return x


def global_rmsnorm_clip(x, clip_at: float = 1.0):
    x = list_guard(x)
    clip_at = scalar_guard(clip_at, x[0])
    return _compilable_global_rmsnorm_clip_(x, clip_at)


@decorator_knowngood
def _compilable_global_l2norm_clip_(x, clip_at):
    norm = 0
    for i in x:
        norm += promote(i.square().sum())
    norm = norm**0.5
    norm = norm.clamp(min=clip_at)
    stochastic_multiply_(x, 1 / norm)
    return x


def global_l2norm_clip(x, clip_at: float = 1.0):
    x = list_guard(x)
    clip_at = scalar_guard(clip_at, x[0])
    return _compilable_global_rmsnorm_clip_(x, clip_at)


def rmsnorm_normalize_(x, clip_at: float = 1e-6):
    x = list_guard(x)
    return _compilable_rmsnorm_clip_(x, clip_at)


@decorator_knowngood
def _compilable_mu_law_compress_(x, mu):
    """
    original at https://github.com/opooladz/modded-nanogpt-psgd/blob/dc7c78082ac15fbf326f1bacd9e0ead0a2b45908/kron_mu.py
    """

    for x_ in x:
        xa = promote(x_.abs()) * mu
        xa = xa.log1p()
        xa = xa / math.log1p(mu)
        xa = xa.copysign(x_)
        copy_stochastic_(x_, xa)


def mu_law_compress(x, mu=127.0):
    """
    μ-law compression
    Args:
        x: Input tensor
        mu: Compression parameter (default 127.0 for behavior similar to trust_region=1.5)
    """
    x = list_guard(x)
    mu = scalar_guard(mu, x[0])
    _compilable_mu_law_compress_(x, mu)
    return x


@decorator_knowngood
def _compilable_a_law_compress_(x, A):
    """
    original at https://github.com/opooladz/modded-nanogpt-psgd/blob/dc7c78082ac15fbf326f1bacd9e0ead0a2b45908/kron_mu.py
    """
    for x_ in x:
        xa = promote(x_.abs()) * A
        xa = torch.where(xa < 1, xa, 1 + xa.log())
        xa = xa.copysign(x_)
        xa = xa * (1 / (1 + math.log(A)))
        copy_stochastic_(x_, xa)


def a_law_compress(x, A=87.6):
    """
    A-law compression
    Args:
        x: Input tensor
        A: Compression parameter (default 87.6 - European PCM standard)
    :param x:
    :param A:
    :return:
    """
    x = list_guard(x)
    A = scalar_guard(A, x[0])
    _compilable_a_law_compress_(x, A)
    return x


def identity(x):
    return x


@decorator_knowngood
def _compilable_weight_decay_to_ema_(p, ema, ema_decay, weight_decay):
    ema32 = _lerp(ema, p, ema_decay)
    _lerp(p, ema32, 1 - weight_decay)


def weight_decay_to_ema_(p, ema, ema_decay, weight_decay):
    p, ema = list_guard(p, ema)
    ema_decay, weight_decay = scalar_guard(ema_decay, weight_decay, p[0])
    _compilable_weight_decay_to_ema_(p, ema, ema_decay, weight_decay)


@decorator_knowngood
def _compilable_l1_weight_decay_to_ema_(p, ema, ema_decay, weight_decay):
    ema32 = _lerp(ema, p, ema_decay)
    for p_, e_ in zip(p, ema32):
        p32 = promote(p_)
        p32 = p32 + (p32 - e_).sign() * weight_decay
        copy_stochastic_(p_, p32)


def l1_weight_decay_to_ema_(p, ema, ema_decay, weight_decay):
    p, ema = list_guard(p, ema)
    ema_decay, weight_decay = scalar_guard(ema_decay, weight_decay, p[0])
    _compilable_l1_weight_decay_to_ema_(p, ema, ema_decay, weight_decay)


@decorator_knowngood
def _compilable_sign_(grad: List[Tensor], graft: bool):
    for g_ in grad:
        gs = g_.sign()
        if graft:
            gs = _compilable_grafting(g_, gs)
        copy_stochastic_(g_, gs)


def sign_(grad: List[Tensor], graft: bool = True):
    grad = list_guard(grad)
    _compilable_sign_(grad, graft)
    return grad


@decorator_knowngood
def _compilable_trust_region_clip_(grad, lerp, scale):
    # (sgn(x) * log(1 + |x|) * 0.1 + tanh(x) * 0.9).clamp_(min=-2, max=2)
    for x_ in grad:
        x = promote(x_)
        x = x / scale
        tanh = x.tanh()
        x = x.abs().log1p()
        x = x.copysign(tanh) * (1 - lerp) + tanh * lerp
        x = x * scale
        x = x.clamp(min=-2, max=2)
        copy_stochastic_(x_, x)


def trust_region_clip_(grad, lerp=0.9, scale=1.5):
    grad = list_guard(grad)
    lerp, scale = scalar_guard(lerp, scale, grad[0])
    _compilable_trust_region_clip_(grad, lerp, scale)
    return grad


@decorator
def triu_to_line(Q_list: List[Tensor]):
    out = []
    for q in Q_list:
        if q.dim() < 2:
            out.append((None, q))
        else:
            out.append((tuple(q.shape), q[tuple(torch.triu_indices(*q.shape))]))
    return out


@decorator_knowngood
def line_to_triu(Q_list: List[Tuple[Optional[List[int]], Tensor]], symmetric_output: bool = False):
    new = []
    for shape, q in Q_list:
        if shape is not None:
            x, y = torch.triu_indices(*shape, device=q.device)
            q_mat = torch.zeros(shape, device=q.device, dtype=q.dtype)
            q_mat[x, y] = q
            if symmetric_output:
                q_mat[y, x] = q
            q = q_mat
        new.append(q)
    return new


_warned = set()


def warn_once(msg):
    if msg not in _warned:
        warnings.warn(msg)
        _warned.add(msg)


def psgd_should_update(
    group, prob: Union[float, callable], rng: Optional[random.Random] = None, name: str = "cumulative_prob"
):
    group[f"{name}_prob_step"] = group.get(f"{name}_prob_step", 0) + 1
    if not isinstance(prob, float):
        prob = prob(group[f"{name}_prob_step"])
    if group["stochastic_schedule"]:
        return rng.random() < prob
    cumulative_prob = group.get(name, 0)
    group[name] = cumulative_prob + prob
    return int(group[name]) > int(cumulative_prob)


@functools.lru_cache(maxsize=None)
def cached_precond_grad_expr(Q_dim, grad_dim):
    expr = [f"{c.upper()}{c}" if q_ == 2 else c for c, q_ in zip(einsum_base, Q_dim)]
    expr = ",".join(expr)
    grad_expr = "".join(c for c, _ in zip(einsum_base, range(grad_dim)))
    out_expr = "".join(c.upper() if c.upper() in expr else c for c in grad_expr)
    return f"{expr},{grad_expr}->{out_expr}"


@decorator_knowngood
def precond_grad_cached_(
    ea: Tensor,
    cached_q: List[Tensor],
    caution: bool = False,
    grad: Optional[Tensor] = None,
    cast: bool = True,
):
    if caution:
        ea = _compilable_cautioning(grad, ea)
    md = min_dtype(list(cached_q) + [ea])
    args = [q.to(md) for q in cached_q]
    args = args + [ea.to(md)]
    expr = cached_precond_grad_expr(ndim_tuple(cached_q), grad.ndim)
    new = compiled_einsum(expr, *args)
    if cast:
        return new.to(ea.dtype)
    return new


TriuOrLine = Union[List[Tensor], List[Tuple[Optional[List[int]], Tensor]]]


@decorator_knowngood
def _compilable_fused_precond_grad_cached_(ea: Tensor, param, lr, grad, decay, caution, cached_q: List[Tensor]):
    precond = precond_grad_cached_(ea, cached_q, caution=caution, grad=grad, cast=False)
    update_param_(param, precond, lr, decay, caution=False)


def fused_precond_grad_cached_(ea: Tensor, param, lr, grad, decay, caution, cached_q: List[Tensor]):
    lr = scalar_guard(lr, param[0])
    _compilable_fused_precond_grad_cached_(ea, param, lr, grad, decay, caution, cached_q)


@functools.lru_cache(maxsize=None)
def precond_grad_expr(Q_dim, grad_dim):
    expr = [
        f"{c2}{c.upper()},{c2}{c}" if q_ == 2 else f"{c},{c}" for c, c2, q_ in zip(einsum_base, einsum_base[13:], Q_dim)
    ]
    expr = ",".join(expr)
    grad_expr = "".join(c for c, _ in zip(einsum_base, range(grad_dim)))
    out_expr = "".join(c.upper() if c.upper() in expr else c for c in grad_expr)
    return f"{expr},{grad_expr}->{out_expr}"


@decorator_knowngood
def psgd_precond_grad(
    ea: Tensor,
    preconds: TriuOrLine,
    caution: bool = False,
    grad: Optional[Tensor] = None,
    store_triu_as_line: bool = False,
    symmetric_output: bool = False,
):
    if caution:
        ea = _compilable_cautioning(grad, ea)
    if store_triu_as_line:
        preconds = line_to_triu(preconds, symmetric_output)
    md = min_dtype(list(preconds) + [ea])
    args = [q.to(md) for q in preconds]
    expr = precond_grad_expr(ndim_tuple(args), ea.ndim)
    new = compiled_einsum(expr, *[a for a in args for _ in (0, 1)], ea.to(md))
    return new.to(ea.dtype)


@decorator_knowngood
def _compilable_fused_psgd_precond_grad(
    ea: Tensor,
    param,
    lr,
    grad,
    decay,
    caution,
    preconds: TriuOrLine,
    store_triu_as_line: bool = False,
    symmetric_output: bool = False,
):
    precond = psgd_precond_grad(
        ea,
        preconds,
        caution=caution,
        grad=grad,
        store_triu_as_line=store_triu_as_line,
        symmetric_output=symmetric_output,
    )
    update_param_(param, precond, lr, decay, caution=False, grad=grad)


def fused_psgd_precond_grad(
    ea: Tensor,
    param,
    lr,
    grad,
    decay,
    caution,
    preconds: TriuOrLine,
    store_triu_as_line: bool = False,
    symmetric_output: bool = False,
):
    lr = scalar_guard(lr, param[0])
    _compilable_fused_psgd_precond_grad(
        ea, param, lr, grad, decay, caution, preconds, store_triu_as_line, symmetric_output
    )


@decorator_knowngood
def _compilable_mars_correction_(g: Tensor, old_g: Tensor, a: Tensor):
    g_copy = [g_.clone() for g_ in g]
    _compilable_stochastic_lerp_(g, old_g, a)
    copy_stochastic_list_(old_g, g_copy)


def mars_correction(g, old_g, beta1, gamma):
    a = -gamma * beta1 / (1 - beta1)
    g, old_g = list_guard(g), list_guard(old_g)
    a = scalar_guard(a, g[0])
    _compilable_mars_correction_(g, old_g, a)


@decorator_knowngood
def _compilable_mars_correction_ema_(g: Tensor, old_g: Tensor, a: Tensor, beta1: Tensor):
    g_copy = [g_.clone() for g_ in g]
    _compilable_stochastic_lerp_(g, old_g, a)
    old_g = _compilable_stochastic_lerp_(g, old_g, beta1)
    copy_stochastic_list_(old_g, old_g)


def mars_correction_ema(g, old_g, beta1, gamma):
    a = -gamma * beta1 / (1 - beta1)
    g, old_g = list_guard(g), list_guard(old_g)
    a, beta1 = scalar_guard(a, beta1, g[0])
    _compilable_mars_correction_ema_(g, old_g, a, beta1)


@decorator_knowngood
def _compilable_mars_correction_ema_(g: Tensor, old_g: Tensor, a: Tensor):
    g_copy = [g_.clone() for g_ in g]
    _compilable_stochastic_lerp_(g, old_g, a)
    copy_stochastic_list_(old_g, g_copy)


def mars_correction_ema(g, old_g, beta1, gamma):
    a = -gamma * beta1 / (1 - beta1)
    g, old_g = list_guard(g), list_guard(old_g)
    a = scalar_guard(a, g[0])
    _compilable_mars_correction_ema_(g, old_g, a)


@decorator_knowngood
def _compilable_orthogonalization(weight: List[Tensor], grad: List[Tensor], eps: Tensor, graft: bool = True):
    """
    Implements OrthoGrad from "Grokking at the Edge of Numerical Stability" (https://arxiv.org/abs/2501.04697)
    """

    for w, g in zip(weight, grad):
        proj = promote((w * g).sum()) / promote((w * w).sum()).add(eps)
        out = promote(g) - proj * promote(w)  # promote in this funky way to keep traffic minimal

        if graft:
            out = _compilable_grafting(g, out)
        copy_stochastic_(g, out)


def orthogonalize_grad_to_param(weight, grad, eps, graft=True):
    weight, grad = list_guard(weight, grad)
    eps = scalar_guard(eps, weight[0])
    _compilable_orthogonalization(weight, grad, eps, graft)
    return grad


@decorator_knowngood
def _compilable_cautioning(g: Tensor, update: Tensor):
    mask = g.signbit() ^ update.signbit()  # "Mask if they point in different directions"
    update = update.masked_fill(mask, 0)
    scale = mask.numel() / (mask.numel() - mask.sum()).clamp(min=1)
    update.mul_(scale)
    return update


def caution(g, update):
    return _compilable_cautioning(g, update)


def _inner_precond_update_prob_schedule(
    n: int, max_prob: float = 1.0, min_prob: float = 0.03, decay: float = 0.999, flat_start: float = 1000
):
    return max(min_prob, max_prob * decay ** max(n - flat_start, 0))


def precond_update_prob_schedule(
    max_prob: float = 1.0, min_prob: float = 0.03, decay: float = 0.999, flat_start: float = 1000
):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at `max_prob` for 1000 steps then exponentially anneal down to
    `min_prob` by ~4000 steps. Default settings work very well for most models and
    training regimes.
    """
    return functools.partial(
        _inner_precond_update_prob_schedule, max_prob=max_prob, min_prob=min_prob, decay=decay, flat_start=flat_start
    )


def merge_group(group, *tensors):
    if not group.get("merge_dims", False):
        return tensors
    if isinstance(tensors[0], list):
        return [merge_group(group, *t) for t in tensors]

    out = []
    for t in tensors:
        append_or_extend(
            out,
            dim_merger(
                t,
                group["max_size_triangular"] if "max_size_triangular" in group else group["max_precond_dim"],
                group.get("split", False),
            ),
        )
    return out


@decorator_knowngood
def _compilable_d_adapt_(grads: List[Tensor], update: List[Tensor], state: List[Tensor], delta: List[Tensor]):
    for g_, u_, s_, d_ in zip(grads, update, state, delta):
        g, u, s, d = promote(g_), promote(u_), promote(s_), promote(d_)
        next_d = d * (g * s).sum()
        s = s + u * d
        next_d = next_d / s.abs().sum()
        next_d = torch.maximum(next_d, d)
        copy_stochastic_(u_, u * d)
        copy_stochastic_(d_, next_d)
        copy_stochastic_(s_, s)


def d_adaptation(grads: List[Tensor], update: List[Tensor], state: List[Tensor], delta: List[Tensor]):
    grads, update, state, delta = list_guard(grads, update, state, delta)
    _compilable_d_adapt_(grads, update, state, delta)


@decorator_knowngood
def _compilable_lr_adapt_(
    grads: List[Tensor], update: List[Tensor], state: List[Tensor], delta: List[Tensor], lr_lr: Tensor
):
    for g_, u_, s_, d_ in zip(grads, update, state, delta):
        g, u, s, d = promote(g_), promote(u_), promote(s_), promote(d_)
        lr_grad = d.sigmoid()
        lr_grad = lr_grad * (1 - lr_grad)
        lr_grad = lr_grad * (s * g).mean()
        d = d - lr_grad * lr_lr
        copy_stochastic_(d_, d)
        copy_stochastic_(u_, u * d.sigmoid())
        copy_stochastic_(s_, u)


def lr_adaptation(grads: List[Tensor], update: List[Tensor], state: List[Tensor], delta: List[Tensor], lr_lr: float):
    grads, update, state, delta = list_guard(grads, update, state, delta)
    lr_lr = scalar_guard(lr_lr, grads[0])
    _compilable_lr_adapt_(grads, update, state, delta, lr_lr)


@decorator_knowngood
def _compilable_pointwise_lr_adapt_(
    grads: List[Tensor], update: List[Tensor], state: List[Tensor], delta: List[Tensor], lr_lr: Tensor
):
    for g_, u_, s_, d_ in zip(grads, update, state, delta):
        g, u, s, d = promote(g_), promote(u_), promote(s_), promote(d_)
        lr_grad = d.sigmoid()
        lr_grad = lr_grad * (1 - lr_grad)
        lr_grad = lr_grad * s * g
        d = d - lr_grad * lr_lr
        copy_stochastic_(d_, d)
        copy_stochastic_(u_, u * d.sigmoid())
        copy_stochastic_(s_, u)


def pointwise_lr_adaptation(
    grads: List[Tensor], update: List[Tensor], state: List[Tensor], delta: List[Tensor], lr_lr: float
):
    grads, update, state, delta = list_guard(grads, update, state, delta)
    lr_lr = scalar_guard(lr_lr, grads[0])
    _compilable_lr_adapt_(grads, update, state, delta, lr_lr)


def hook_optimizer_into_model(model, optimizer, *args, **kwargs):
    optimizers = {}

    def _step(p: Tensor):
        o = optimizers[p]
        o.step()
        o.zero_grad()

    for p in model.parameters():
        optimizers[p] = optimizer([p], *args, **kwargs)
        p.register_post_accumulate_grad_hook(_step)

    return optimizers


def fused_hook(parameters, optimizer, *args, **kwargs):
    parameters = list(parameters)
    param_count = len(parameters)
    seen_params = set()

    o = optimizer(parameters, *args, **kwargs)
    step_fn = o.step
    o.step = functools.partial(
        warn_once, msg="You're trying to call `step` on a fused optimizer. This will not do anything."
    )

    def _step(p: Tensor):
        seen_params.add(p)

        if len(seen_params) < param_count:
            step_fn()
            o.zero_grad()
            seen_params.clear()

    for p in parameters:
        p.register_post_accumulate_grad_hook(_step)

    return o


@decorator_knowngood
def _compilable_caution_no_scale(g: Tensor, update: Tensor):
    mask = g.signbit() ^ update.signbit()  # "Mask if they point in different directions"
    update = update.masked_fill(mask, 0)
    return update


def disable_caution_scaling():
    global _compilable_cautioning
    _compilable_cautioning = _compilable_caution_no_scale


@decorator_knowngood
def sam_step(parameters, ball_size, adaptive: bool = True):
    old_params = []
    for p in parameters:
        old_params.append(p.detach().clone())
        grad = promote(p.grad)
        if adaptive:
            grad = grad * promote(p).square()
        stochastic_add_(p.data, grad, ball_size)
        p.grad.zero_()
    return old_params

def merge_small_dims(shape, max_dim):
    """Merge small dimensions to reduce memory usage.
    
    Example: [1, 2, 512, 1, 2048, 1, 3, 4] -> [1024, 2048, 12] if max_dim = 1024
    """
    if not shape or len(shape) <= 1:
        return shape
    
    # Convert to list for easier manipulation
    shape_list = list(shape)
    
    # If all dimensions are small enough, return as-is
    if all(s <= max_dim for s in shape_list):
        return tuple(shape_list)
    
    # Merge small consecutive dimensions
    merged = []
    current_product = 1
    
    for dim in shape_list:
        if current_product * dim <= max_dim:
            current_product *= dim
        else:
            if current_product > 1:
                merged.append(current_product)
            current_product = dim
    
    # Don't forget the last dimension
    if current_product > 1:
        merged.append(current_product)
    
    return tuple(merged) if merged else tuple(shape_list)

def update_preconditioner_mode_(G_matrix, grad, mode_idx, beta2, eps):
    """Update preconditioner matrix for a specific tensor mode"""
    # Unfold the gradient tensor along the specified mode
    grad_unfolded = unfold_tensor_mode(grad, mode_idx)
    
    # Compute outer product for this mode: G += grad_mode @ grad_mode.T
    outer_product = torch.mm(grad_unfolded, grad_unfolded.t())
    
    # Add regularization
    regularization = eps * torch.eye(outer_product.size(0), 
                                   device=outer_product.device, 
                                   dtype=outer_product.dtype)
    outer_product = outer_product + regularization
    
    # Update with exponential moving average
    G_matrix.mul_(beta2).add_(outer_product, alpha=1 - beta2)

def unfold_tensor_mode(tensor, mode):
    """Unfold tensor along specified mode (mode-n matricization)"""
    # Move the mode dimension to the front
    dims = list(range(tensor.dim()))
    dims[0], dims[mode] = dims[mode], dims[0]
    
    # Permute and reshape
    unfolded = tensor.permute(dims)
    mode_size = unfolded.shape[0]
    unfolded = unfolded.reshape(mode_size, -1)
    
    return unfolded

def update_orthogonal_basis_(G_matrix, Q_matrix):
    """Update orthogonal basis using QR decomposition"""
    try:
        # Compute one step of power iteration followed by QR
        temp = torch.mm(G_matrix, Q_matrix)
        
        # QR decomposition
        Q_new, _ = torch.linalg.qr(temp)
        
        # Update Q_matrix in place
        Q_matrix.copy_(Q_new)
        
    except torch.linalg.LinAlgError:
        # If QR fails, just keep the old Q_matrix
        pass

def apply_mode_preconditioning(grad, Q_matrix, mode_idx):
    """Apply preconditioning for a specific tensor mode"""
    # This applies the preconditioner Q^T to the specified mode
    
    # Get original shape
    original_shape = grad.shape
    
    # Move mode to the front
    dims = list(range(grad.dim()))
    dims[0], dims[mode_idx] = dims[mode_idx], dims[0]
    
    # Permute tensor
    grad_permuted = grad.permute(dims)
    
    # Reshape for matrix multiplication
    mode_size = grad_permuted.shape[0]
    grad_flat = grad_permuted.reshape(mode_size, -1)
    
    # Apply preconditioner: Q^T @ grad_flat
    preconditioned_flat = torch.mm(Q_matrix.t(), grad_flat)
    
    # Reshape back
    new_shape = list(grad_permuted.shape)
    new_shape[0] = Q_matrix.shape[0]  # Might be different due to dimensionality reduction
    
    if preconditioned_flat.numel() == torch.prod(torch.tensor(new_shape)):
        preconditioned = preconditioned_flat.reshape(new_shape)
        
        # Permute back to original dimension order
        inv_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            inv_dims[d] = i
        
        result = preconditioned.permute(inv_dims)
        
        # If shape changed due to preconditioning, we need to handle it
        if result.shape != original_shape:
            # For now, just return the original gradient
            # In a full implementation, you'd need to handle dimension changes properly
            return grad
        
        return result
    else:
        # Shape mismatch, return original gradient
        return grad

def compute_adam_updates_(exp_avg, exp_avg_sq, grad_list, beta2, step, eps, use_bias_correction=True):
    """Compute Adam updates for grafting"""
    updates = []
    
    for i, grad in enumerate(grad_list):
        # Update exponential moving averages
        exp_avg[i].mul_(0.9).add_(grad, alpha=0.1)  # beta1 = 0.9
        exp_avg_sq[i].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Bias correction
        if use_bias_correction:
            bias_correction1 = 1 - 0.9 ** step
            bias_correction2 = 1 - beta2 ** step
            corrected_exp_avg = exp_avg[i] / bias_correction1
            corrected_exp_avg_sq = exp_avg_sq[i] / bias_correction2
        else:
            corrected_exp_avg = exp_avg[i]
            corrected_exp_avg_sq = exp_avg_sq[i]
        
        # Compute update
        denom = corrected_exp_avg_sq.sqrt().add_(eps)
        update = corrected_exp_avg / denom
        updates.append(update)
    
    return updates

def compute_rmsprop_updates_(exp_avg_sq, grad_list, beta2, eps):
    """Compute RMSprop updates for grafting"""
    updates = []
    
    for i, grad in enumerate(grad_list):
        # Update second moment
        exp_avg_sq[i].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Compute update
        denom = exp_avg_sq[i].sqrt().add_(eps)
        update = grad / denom
        updates.append(update)
    
    return updates

def update_momentum_list_(momentum_list, grad_list, beta1, step, use_bias_correction=True):
    """Update momentum for a list of gradients and return filtered gradients"""
    filtered_grads = []
    
    for momentum, grad in zip(momentum_list, grad_list):
        # Update momentum: m = beta1 * m + (1 - beta1) * grad
        momentum.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        # Apply bias correction if enabled
        if use_bias_correction:
            bias_correction = 1 - beta1 ** step
            filtered_grad = momentum / bias_correction
        else:
            filtered_grad = momentum
        
        filtered_grads.append(filtered_grad)
    
    return filtered_grads

def fused_parameter_update_list_(param_list, update_list, lr, weight_decay, caution):
    """Apply parameter updates with weight decay"""
    for param, update in zip(param_list, update_list):
        # Apply weight decay if specified
        if weight_decay != 0:
            param.data.add_(param.data, alpha=-weight_decay * lr)
        
        # Apply the update
        if caution:
            # Cautious update - clip large updates
            update_norm = torch.norm(update)
            param_norm = torch.norm(param.data)
            if update_norm > param_norm:
                update = update * (param_norm / update_norm)
        
        param.data.add_(update, alpha=lr)