import gc
import math
import string
from typing import List

import torch

_mode = None

if _mode is None:
    def decorator(func):
        return func
else:
    decorator = torch.compile(fullgraph=False, dynamic=True, mode=_mode)

_einsum_base = string.ascii_lowercase + string.ascii_uppercase


def warmup(lr: float, step: int, warmup_steps: int):
    return lr * min(step / warmup_steps, 1)


def schedule_free_(lr: float, weight_lr_power: float, weight_sum: float, beta1: float, parameters: List[torch.Tensor],
                   z: List[torch.Tensor], grad: list[torch.Tensor]):
    weight = lr ** weight_lr_power
    weight_sum = weight_sum + weight

    try:
        ckp1 = weight / weight_sum
    except ZeroDivisionError:
        ckp1 = 0

    # These operations update y in-place,
    # without computing x explicitly.
    p32 = [p.float() for p in parameters]
    z32 = [z_.float() for z_ in z]
    torch._foreach_lerp_(p32, z32, weight=ckp1)
    torch._foreach_add_(p32, grad, alpha=lr * (beta1 * (1 - ckp1) - 1))
    copy_stochastic_list_(parameters, p32)

    # z step
    torch._foreach_sub_(z, grad, alpha=lr)
    copy_stochastic_list_(z, z32)
    return weight_sum


def _merge_dims(grad, max_precond_dim):
    """
    Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.

    we don't want to merge fan-in into fan-out,
    but we want to merge conv kernels into fan-in or at least merge the kernel
    so, [128, 64, 3, 3] should result in [128, 576] or [128, 64, 9] instead of [73728] or [8192, 3, 3] the baseline
    would've done
    """
    shape = grad.shape
    new_shape = []

    curr_shape = 1

    for sh in shape[1:][::-1]:
        temp_shape = curr_shape * sh
        if temp_shape >= max_precond_dim:
            if curr_shape > 1:
                new_shape.append(curr_shape)
                curr_shape = sh
            else:
                new_shape.append(sh)
                curr_shape = 1
        else:
            curr_shape = temp_shape
    new_shape = [*shape[:1], *new_shape[::-1]]

    if curr_shape > 1 or len(new_shape) == 0:
        new_shape.append(curr_shape)

    new_grad = grad.reshape(new_shape)
    return new_grad


def beta_debias(beta, step):
    return 1 - (1 - beta) / (1 - beta ** step)


def exp_avg_sq_(state, grad, beta2, eps):
    torch._foreach_mul_(state, beta2)
    torch._foreach_addcmul_(state, grad, grad, value=1 - beta2)
    denom = torch._foreach_sqrt(state)
    torch._foreach_maximum_(denom, eps)
    return denom


def adaptive_gradient_clipping_(parameters: List[torch.Tensor], gradients: List[torch.Tensor], clip_val: float,
                                minimum: float = 1e-3, eps: float = 1e-8):
    if clip_val <= 0:
        return
    p_norm = torch._foreach_norm(parameters)
    g_norm = torch._foreach_norm(gradients)
    torch._foreach_maximum_(p_norm, minimum)
    torch._foreach_maximum_(g_norm, eps)
    torch._foreach_div_(p_norm, g_norm)
    torch._foreach_mul_(p_norm, clip_val)
    torch._foreach_minimum_(p_norm, 1)
    torch._foreach_mul_(gradients, p_norm)


def set_(dst: torch.Tensor, src: torch.Tensor):
    if src.is_contiguous():
        dst.set_(src)
    else:
        dst.copy_(src)


def clean():
    torch.cuda.empty_cache()
    gc.collect()


@decorator
def _get_orthogonal_matrix_QR(GG, Q, exp_avg_sq, max_precond_dim=10000, merge_dims=False):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration
    followed by torch.linalg.qr decomposition.
    """
    matrix = []
    orth_matrix = []
    for m, o in zip(GG, Q):
        if len(m) == 0:
            matrix.append([])
            orth_matrix.append([])
            continue
        if m.data.dtype != torch.float:
            matrix.append(m.data.float())
            orth_matrix.append(o.data.float())
        else:
            matrix.append(m.data.float())
            orth_matrix.append(o.data.float())

    orig_shape = exp_avg_sq.shape
    if merge_dims:
        exp_avg_sq_new = _merge_dims(exp_avg_sq, max_precond_dim)
    else:
        exp_avg_sq_new = exp_avg_sq

    indices = []

    for ind, (m, o, q) in enumerate(zip(matrix, orth_matrix, Q)):
        if len(m) == 0:
            indices.append(None)
            continue

        tmp = m @ o
        del m
        est_eig = torch.einsum('ij,ij->i', o, tmp)
        del o
        sort_idx = torch.argsort(est_eig, descending=True)
        del est_eig
        indices.append(sort_idx)
        power_iter = tmp[:, sort_idx]
        set_(q, torch.linalg.qr(power_iter)[0])
        del power_iter

    indices = tuple(slice(None) if ind is None else ind.view(*(1,) * i, -1, *(1,) * (exp_avg_sq_new.dim() - i - 1))  #
                    for i, ind in enumerate(indices))
    exp_avg_sq_new = exp_avg_sq_new[indices]

    if merge_dims:
        exp_avg_sq_new = exp_avg_sq_new.reshape(orig_shape)
    set_(exp_avg_sq, exp_avg_sq_new)


def _get_orthogonal_matrix(mat):
    """
    Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
    """
    matrix = []
    for m in mat:
        if len(m) == 0:
            matrix.append([])
            continue
        if m.data.dtype != torch.float:
            float_data = False
            original_type = m.data.dtype
            original_device = m.data.device
            matrix.append(m.data.float())
        else:
            float_data = True
            matrix.append(m.data)

    final = []
    for m in matrix:
        if len(m) == 0:
            final.append([])
            continue

        device, dtype = m.device, m.dtype
        for modifier in (None, torch.double, 'cpu'):
            clean()
            if modifier is not None:
                m = m.to(modifier)
                clean()
            try:
                Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))[1].to(device=device,
                                                                                                dtype=dtype)
                break
            except:
                continue
        else:
            raise RuntimeError("Failed to compute eigenvalues.")

        clean()
        Q = torch.flip(Q, [1])

        if not float_data:
            Q = Q.to(original_device).type(original_type)
        final.append(Q)

    clean()
    return final


@decorator
def _compute_ggt(grad, GG, max_precond_dim, merge_dims, precondition_1d, beta):
    if grad.dim() == 1:
        if precondition_1d and grad.shape[0] <= max_precond_dim:
            GG[0].lerp_((grad.unsqueeze(1) * grad.unsqueeze(0)).float(), 1 - beta)
        return

    if merge_dims:
        new_grad = _merge_dims(grad, max_precond_dim)
        for idx, sh in enumerate(new_grad.shape):
            if sh <= max_precond_dim:
                outer_product = torch.tensordot(new_grad, new_grad,
                                                dims=[[*range(idx), *range(idx + 1, len(new_grad.shape))]] * 2)
                GG[idx].lerp_(outer_product.float(), 1 - beta)
        return

    for idx, sh in enumerate(grad.shape):
        if sh <= max_precond_dim:
            outer_product = torch.tensordot(grad, grad, dims=[[*range(idx), *range(idx + 1, len(grad.shape))]] * 2)
            GG[idx].lerp_(outer_product.float(), 1 - beta)


def _update_preconditioner(grad, state, max_precond_dim, merge_dims, precondition_1d, beta, update_precond):
    """
    Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
    """
    _compute_ggt(grad, state['GG'], max_precond_dim, merge_dims, precondition_1d, beta)
    if state['Q'] is None:
        state['Q'] = _get_orthogonal_matrix(state['GG'])
    if update_precond:
        _get_orthogonal_matrix_QR(state['GG'], state['Q'], state['exp_avg_sq'], max_precond_dim, merge_dims)


def _init_preconditioner(grad, state, max_precond_dim=10000, precondition_1d=False, merge_dims=False):
    """
    Initializes the preconditioner matrices (L and R in the paper).
    """
    state['GG'] = []  # Will hold all the preconditioner matrices (L and R in the paper).
    if grad.dim() == 1:
        if not precondition_1d or grad.shape[0] > max_precond_dim:
            state['GG'].append([])
        else:
            state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device))
    else:
        if merge_dims:
            grad = _merge_dims(grad, max_precond_dim)

        for sh in grad.shape:
            if sh > max_precond_dim:
                state['GG'].append([])
            else:
                state['GG'].append(torch.zeros(sh, sh, device=grad.device))

    state['Q'] = None  # Will hold all the eigenbases of the preconditioner.


@decorator
def _project(grad, Q, merge_dims, max_precond_dim, back: bool):
    """

    :param grad:
    :param Q:
    :param merge_dims:
    :param max_precond_dim:
    :param back: whether to project to Shampoo eigenbases or back to original space
    :return:
    """
    original_shape = grad.shape
    if merge_dims:
        grad = _merge_dims(grad, max_precond_dim)

    param = _einsum_base[:grad.dim()]
    preconditioners = ",".join(
        (g + g.upper())[::-1 if back else 1] for m, g in zip(Q, param) if isinstance(m, torch.Tensor))
    if preconditioners:
        out = ''.join(c.upper() if c.upper() in preconditioners else c for c in param)
        grad = torch.einsum(f'{param},{preconditioners}->{out}', grad, *[q for q in Q if isinstance(q, torch.Tensor)])
    if merge_dims:
        grad = grad.reshape(original_shape)
    return grad


class ScheduleFree(torch.optim.Optimizer):
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta'] if 'beta' in group else group['betas'][0]
            if beta1 > 0 and train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to x
                        z = state['z'].float()
                        p32 = p.data.float()
                        p32.lerp_(end=z, weight=1 - 1 / beta1)
                        copy_stochastic_(p.data, p32)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta'] if 'beta' in group else group['betas'][0]
            if beta1 > 0 and not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        z = state['z'].float()
                        p32 = p.data.float()
                        p32.lerp_(end=z, weight=1 - beta1)
                        copy_stochastic_(p.data, p32)
                group['train_mode'] = True

    def _step(self):
        raise NotImplementedError


def copy_stochastic_list_(target: List[torch.Tensor], source: List[torch.Tensor]):
    for t, s in zip(target, source):
        if t.dtype == torch.bfloat16:
            copy_stochastic_(t, s)
        elif t.data_ptr() != s.data_ptr():
            t.copy_(s)


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    if target.data_ptr() == source.data_ptr():
        return

    """Taken as-is from https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905"""
    # create a random 16 bit integer
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))


def update_param_(param: List[torch.Tensor], update: List[torch.Tensor], lr: float, decay: float,
                  add_fn: callable = None):
    param32 = [p.float() for p in param]
    update32 = [u.float() for u in update]
    if decay > 0:
        torch._foreach_mul_(param32, 1 - decay * lr)
    if add_fn is None:
        torch._foreach_add_(param32, update32, alpha=lr)
    else:
        add_fn(param32, update32, lr)
    copy_stochastic_list_(param, param32)


def precond_schedule(step, precond_scheduler, rng):
    precond_prob = max(step, 1) ** precond_scheduler[0]
    precond_prob = math.log10(precond_prob)
    precond_prob = precond_prob ** precond_scheduler[1] + 1
    precond_prob = 1 / precond_prob
    update_precond = rng.random() < precond_prob
    return update_precond
