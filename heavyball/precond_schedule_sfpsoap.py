import gc
import math
import random
import string
from itertools import chain

import torch
import torch.optim as optim

_mode = None

if _mode is None:
    def decorator(func):
        return func
else:
    decorator = torch.compile(fullgraph=False, dynamic=True, mode=_mode)

_einsum_base = string.ascii_lowercase + string.ascii_uppercase


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


def set_(dst: torch.Tensor, src: torch.Tensor):
    if dst.is_contiguous():
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

        est_eig = torch.einsum('ij,kj,ik->j', o, o, m)
        sort_idx = torch.argsort(est_eig, descending=True)
        del est_eig
        indices.append(sort_idx)
        o = o[:, sort_idx]
        power_iter = m @ o
        del m, o
        set_(q, torch.linalg.qr(power_iter)[0].contiguous())
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
                Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))[1].to(device=device, dtype=dtype)
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
            GG[0].lerp_(grad.unsqueeze(1) * grad.unsqueeze(0), 1 - beta)
        return

    if merge_dims:
        new_grad = _merge_dims(grad, max_precond_dim)
        for idx, sh in enumerate(new_grad.shape):
            if sh <= max_precond_dim:
                outer_product = torch.tensordot(new_grad, new_grad,
                                                dims=[[*chain(range(idx), range(idx + 1, len(new_grad.shape)))]] * 2)
                GG[idx].lerp_(outer_product, 1 - beta)
        return

    for idx, sh in enumerate(grad.shape):
        if sh <= max_precond_dim:
            outer_product = torch.tensordot(grad, grad,  # Contracts across all dimensions except for k.
                                            dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2)
            GG[idx].lerp_(outer_product, 1 - beta)


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


# Default precond scheduler has the following values
# | Input       | f⁻¹(Input) | Total | constant, every 2 | constant, every 16 |
# |-------------|-----------|-------|-------------------|--------------------|
# | 10          | 1.00005   |    10 |                 5 |                  0 |
# | 100         | 1.026     |    99 |                50 |                  6 |
# | 1,000       | 2.0       |   738 |               500 |                 62 |
# | 10,000      | 14.3      | 2,168 |             5,000 |                625 |
# | 100,000     | 100.2     | 4,049 |            50,000 |              6,250 |
# | 1,000,000   | 513       | 7,245 |           500,000 |             62,500 |
class PrecondScheduleSFPaLMSOAP(optim.Optimizer):
    """
    SFPaLMForeachSOAP with preconditioner schedules

    Sources:
        Preconditioner Schedules:
            Preconditioned Stochastic Gradient Descent
            Xi-Lin Li, Omead Pooladzandi, Evan Walters
            https://arxiv.org/abs/1512.04202
            https://github.com/evanatyourservice/kron_torch
            https://github.com/lixilinx/psgd_torch

        Baseline SOAP:
            SOAP: Improving and Stabilizing Shampoo using Adam
            Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
            https://arxiv.org/abs/2409.11321
            https://github.com/nikhilvyas/SOAP

        ScheduleFree
            The Road Less Scheduled
            Aaron Defazio, Xingyu Alice Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky
            https://arxiv.org/abs/2405.15682
            https://github.com/facebookresearch/schedule_free

        Beta2 Schedule:
            PaLM: Scaling Language Modeling with Pathways
            Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, Noah Fiedel
            https://arxiv.org/abs/2204.02311
    """
    def __init__(self, params, lr: float = 3e-3, beta=0.9, beta2_scale: float = 0.8, eps: float = 1e-8,
                 weight_decay: float = 0.01, precondition_frequency: int = 2, max_precond_dim: int = 2048,  #
                 merge_dims: bool = True, precondition_1d: bool = False, normalize_grads: bool = False,
                 data_format: str = "channels_first", correct_bias: bool = True, warmup_steps: int = 1, r=0.0,
                 weight_lr_power=2.0, gradient_clip_val: float = 0.01, precond_scheduler=(1 / 3, 9)):
        defaults = {"lr": lr, "beta": beta, "beta2_scale": beta2_scale, "eps": eps, "weight_decay": weight_decay,
                    "precondition_frequency": precondition_frequency, "max_precond_dim": max_precond_dim,
                    "merge_dims": merge_dims, "precondition_1d": precondition_1d, "normalize_grads": normalize_grads,
                    "correct_bias": correct_bias, 'warmup_steps': warmup_steps, 'r': r,
                    'weight_lr_power': weight_lr_power, 'train_mode': True, 'step': -1,
                    'gradient_clip_val': gradient_clip_val, 'precond_scheduler': precond_scheduler}
        super().__init__(params, defaults)
        self._data_format = data_format
        self.rng = random.Random(0x120983109)

    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta']
            if beta1 > 0 and train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to x
                        p.data.lerp_(end=state['z'], weight=1 - 1 / beta1)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta']
            if beta1 > 0 and not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to y
                        p.data.lerp_(end=state['z'], weight=1 - beta1)
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            vals = []
            merge_dims = group['merge_dims']
            max_precond_dim = group['max_precond_dim']
            precondition_1d = group['precondition_1d']

            step = group['step'] = group.get("step", -1) + 1

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                p.grad = None
                vals.append((p, grad))

            p_list, grad = zip(*vals)

            # adaptive gradient clipping
            if group["gradient_clip_val"] > 0:
                p_norm = torch._foreach_norm(p_list)
                g_norm = torch._foreach_norm(grad)
                torch._foreach_maximum_(p_norm, 1e-3)
                torch._foreach_maximum_(g_norm, group["eps"])
                torch._foreach_div_(p_norm, g_norm)
                torch._foreach_mul_(p_norm, group["gradient_clip_val"])
                torch._foreach_minimum_(p_norm, 1)
                torch._foreach_mul_(grad, p_norm)

            vals = []

            for p, grad in zip(p_list, grad):
                state = self.state[p]

                if "z" not in state:
                    state["z"] = torch.clone(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    _init_preconditioner(grad, state, max_precond_dim, precondition_1d, merge_dims)
                    _update_preconditioner(grad, state, max_precond_dim, merge_dims, precondition_1d, 0, True)
                    continue  # first step is skipped so that we never use the current gradients in the projection.

                # Projecting gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                grad_projected = _project(grad, state['Q'], merge_dims, max_precond_dim, False)
                z, exp_avg_sq = state["z"], state["exp_avg_sq"]
                vals.append((p, grad, grad_projected, z, exp_avg_sq))

            if not vals:
                continue

            p_list, grad, grad_projected, z, exp_avg_sq = zip(*vals)
            beta1 = group["beta"]

            beta2 = 1 - max(step, 1) ** -group['beta2_scale']
            bias_correction2 = 1.0 - beta2 ** step
            new_debiased2 = (1 - beta2) / bias_correction2

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            torch._foreach_mul_(exp_avg_sq, 1 - new_debiased2)
            torch._foreach_addcmul_(exp_avg_sq, grad_projected, grad_projected, value=new_debiased2)
            denom = torch._foreach_sqrt(exp_avg_sq)
            torch._foreach_maximum_(denom, group["eps"])
            torch._foreach_div_(grad_projected, denom)

            precond_prob = max(step, 1) ** group['precond_scheduler'][0]
            precond_prob = math.log10(precond_prob)
            precond_prob = precond_prob ** group['precond_scheduler'][1] + 1
            precond_prob = 1 / precond_prob
            update_precond = self.rng.random() < precond_prob

            for p, g, gp in zip(p_list, grad, grad_projected):
                state = self.state[p]
                # Projecting back the preconditioned (by Adam) exponential moving average of gradients
                # to the original space
                set_(gp, _project(gp, state['Q'], merge_dims, max_precond_dim, back=True))

                _update_preconditioner(g, state, max_precond_dim, merge_dims, precondition_1d, 1 - new_debiased2,
                                       update_precond)

            # Weight decay calculated at y
            if group["weight_decay"] > 0:
                torch._foreach_add_(grad, p_list, alpha=group["weight_decay"])

            lr = group["lr"] * min(step / group['warmup_steps'], 1)
            weight = lr ** group['weight_lr_power']
            weight_sum = group['weight_sum'] = group.get('weight_sum', 0) + weight

            try:
                ckp1 = weight / weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            # These operations update y in-place,
            # without computing x explicitly.
            torch._foreach_lerp_(p_list, z, weight=ckp1)
            torch._foreach_add_(p_list, grad_projected, alpha=lr * (beta1 * (1 - ckp1) - 1))

            # z step
            torch._foreach_sub_(z, grad_projected, alpha=lr)
        return loss
