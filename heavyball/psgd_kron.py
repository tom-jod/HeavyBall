"""
Originally from Evan Walters and Omead Pooladzandi, 2024
Modified under Creative Commons Attribution 4.0 International
Source available at https://github.com/evanatyourservice/kron_torch/blob/97a2b5ee8a1a4c29e4780bbf6c521e545189eff9/kron_torch/kron.py
"""

import random
import string

import numpy as np
import torch

from .utils import promote, update_param_, warmup


def precond_update_prob_schedule(max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=250):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 200 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work very well for most models and
    training regimes.
    """

    def _schedule(n):
        """Exponential anneal with flat start."""
        n = torch.tensor(n, dtype=torch.float32)
        prob = max_prob * torch.exp(-decay * (n - flat_start))
        prob.clamp_(min=min_prob, max=max_prob)
        return prob

    return _schedule


class ForeachPSGDKron(torch.optim.Optimizer):
    """Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate.
        b1 (float): Momentum parameter.
        weight_decay (float): Weight decay (L2 penalty).
        preconditioner_update_probability (callable or float, optional): Probability of
            updating the preconditioner. If None, defaults to a schedule that anneals
            from 1.0 to 0.03 by 4000 steps.
        max_size_triangular (int): Max size for dim's preconditioner to be triangular.
        min_ndim_triangular (int): Minimum number of dimensions a layer needs
            to have triangular preconditioners.
        memory_save_mode: (string, optional), None, 'one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular, 'one_diag' sets the largest
            or last dim to be diagonal per layer, and 'all_diag' sets all preconditioners
            to be diagonal.
        momentum_into_precond_update: (bool), whether to send momentum into preconditioner
            update instead of raw gradients.
    """

    def __init__(self, params, lr=0.001, beta=0.9, weight_decay=0.0, preconditioner_update_probability=None,
                 max_size_triangular=2048, min_ndim_triangular=2, memory_save_mode=None,
                 momentum_into_precond_update=True, warmup_steps: int = 1):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay,
                        preconditioner_update_probability=preconditioner_update_probability,
                        max_size_triangular=max_size_triangular, min_ndim_triangular=min_ndim_triangular,
                        memory_save_mode=memory_save_mode, momentum_into_precond_update=momentum_into_precond_update,
                        precond_lr=0.1,  # precond lr hardcoded to 0.1
                        precond_init_scale=1.0,  # precond init scale hardcoded to 1.0
                        step=0, warmup_steps=warmup_steps)
        super().__init__(params, defaults)

        self._tiny = torch.finfo(torch.bfloat16).tiny
        self._prob_step = 0
        self.rng = random.Random(5318008)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # update preconditioners all together
        update_prob = self.param_groups[0]["preconditioner_update_probability"]
        if callable(update_prob):
            update_prob = update_prob(self._prob_step)
        do_update = self.rng.random() < update_prob
        self._prob_step += 1

        balance = self.rng.random() < 0.01 and do_update

        for group in self.param_groups:
            momentum_into_precond_update = group.get("momentum_into_precond_update", True)
            precond_init_scale = group['precond_init_scale']
            max_size_triangular = group['max_size_triangular']
            min_ndim_triangular = group['min_ndim_triangular']
            memory_save_mode = group['memory_save_mode']
            precond_lr = group['precond_lr']
            weight_decay = group['weight_decay']
            lr = group['lr']
            beta = group['beta']

            vals = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = promote(p.grad)
                p.grad = None
                state = self.state[p]

                if 'step' not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["Q"], state["exprs"] = init_Q_exprs(p, precond_init_scale, max_size_triangular,
                                                              min_ndim_triangular, memory_save_mode, dtype=grad.dtype)
                    state['step'] = 0

                vals.append((p, grad, state["exp_avg"], state["Q"]))

            if not vals:
                continue

            p_list, grad_list, exp_avg_list, Q_list = zip(*vals)

            group["step"] += 1

            torch._foreach_lerp_(exp_avg_list, grad_list, (1 - beta) / (1 - beta ** group["step"]))

            if balance:
                filtered_q = []
                norms = []
                for g, q in zip(grad_list, Q_list):
                    if g.dim() > 1:
                        norms.extend(_balance_Q(q))
                        filtered_q.extend(q)
                torch._foreach_mul_(filtered_q, norms)
                del filtered_q, norms

            if do_update:
                for p, grad, exp_avg, Q in vals:
                    _update_precond(Q, self.state[p]["exprs"], torch.randn_like(exp_avg),
                                    exp_avg if momentum_into_precond_update else grad, precond_lr, self._tiny)

            pre_grads = [_precond_grad(Q, self.state[p]["exprs"], exp_avg) for p, Q, exp_avg, state in
                         zip(p_list, Q_list, exp_avg_list, vals)]

            torch._foreach_mul_(pre_grads, 1 / 1.5)
            tanh = torch._foreach_tanh(pre_grads)
            sign = torch._foreach_sign(pre_grads)
            torch._foreach_abs_(pre_grads)
            torch._foreach_log1p_(pre_grads)
            torch._foreach_mul_(pre_grads, sign)
            torch._foreach_lerp_(pre_grads, tanh, 0.9)  # sgn(x) * log(1 + |x|) * 0.1 + tanh(x) * 0.9
            torch._foreach_mul_(pre_grads, 1.5)

            torch._foreach_maximum_(pre_grads, -2)
            torch._foreach_minimum_(pre_grads, 2)

            lr = -warmup(lr, group['step'], group['warmup_steps'])
            update_param_(p_list, pre_grads, lr, weight_decay)

        return loss


def init_Q_exprs(t, scale, max_size, min_ndim_triangular, memory_save_mode, dtype=None):
    """For a scalar or tensor t, we initialize its preconditioner Q and
    reusable einsum expressions for updating Q and preconditioning gradient.
    """
    letters = string.ascii_lowercase + string.ascii_uppercase

    dtype = dtype if dtype is not None else t.dtype
    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = [scale * torch.ones_like(t, dtype=dtype)]
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!")

        scale = scale ** (1 / len(shape))

        if memory_save_mode is None:
            dim_diag = [False for _ in shape]
        elif memory_save_mode == "one_diag":
            rev_sorted_dims = np.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            dim_diag[rev_sorted_dims[0]] = True
        elif memory_save_mode == "all_diag":
            dim_diag = [True for _ in shape]
        else:
            raise ValueError(f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
                             "[None, 'one_diag', 'all_diag']")

        Q = []
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            if (size == 1 or size > max_size or len(shape) < min_ndim_triangular or dim_d):
                # use diagonal matrix as preconditioner for this dim
                Q.append(scale * torch.ones(size, dtype=dtype, device=t.device))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
                subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
                exprGs.append(subscripts)

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                Q.append(scale * torch.eye(size, dtype=dtype, device=t.device))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
                piece2 = "".join([(letters[i + 26] if j == i else letters[j]) for j in range(len(shape))])
                subscripts = (piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26])
                exprGs.append(subscripts)

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = (",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P)

    exprGs = tuple(exprGs)
    return [Q, (exprA, exprGs, exprP)]


@torch.compile(fullgraph=True, dynamic=False)
def _balance_Q(Q_in):
    norms = torch.stack([q.norm(float("inf")) for q in Q_in])
    geometric_mean = norms.log().mean().exp()
    norms = geometric_mean / norms
    return list(norms)


def _calc_A_and_conjB(exprA, G, Q, V):
    A = torch.einsum(exprA, *Q, G)
    order = G.dim()
    p = list(range(order))
    conjB = torch.permute(V.conj(), p[1:] + p[:1])
    for i, q in enumerate(Q):
        if q.dim() <= 1:
            conjB /= q
        else:
            unsqueeze = conjB.dim() <= 1
            if unsqueeze:
                conjB = conjB.unsqueeze(0)
            conjB = torch.linalg.solve_triangular(q, conjB, upper=True, left=False, out=conjB)
            if unsqueeze:
                conjB = conjB.squeeze(0)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)
    return A, conjB


def _lb(A, max_abs):
    A /= max_abs
    aa = torch.real(A * A.conj())
    value0, i = torch.max(torch.sum(aa, dim=0), 0)
    value1, j = torch.max(torch.sum(aa, dim=1), 0)

    ah = A.H
    comp = value0 > value1
    x = torch.where(comp, A[:, i], A[j])
    x = x.conj()
    if x.dim() > 1:
        x = torch.where(comp, x, x.T)
    torch.matmul(x, torch.where(comp, A, A.T), out=x.view(1, -1))
    x /= torch.linalg.vector_norm(x)
    torch.matmul(x, torch.where(comp, ah, ah.T), out=x.view(1, -1))
    x = torch.linalg.vector_norm(x)
    x *= max_abs
    return x


def _update_precond(Q, exprs, V, G, step, tiny):
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    exprA, exprGs, _ = exprs

    A, conjB = _calc_A_and_conjB(exprA, G, Q, V)

    for q, exprG in zip(Q, exprGs):
        term1 = torch.einsum(exprG, A, A.conj())
        term2 = torch.einsum(exprG, conjB.conj(), conjB)

        term2 += term1  # a + b
        term1 *= 2  # 2a
        term1 -= term2  # 2a - (a + b) == a - b

        term1 *= step
        norm = term2.norm(float('inf'))
        if q.dim() < 2:
            term1 *= q
            q.addcdiv_(term1, norm.clamp_(min=tiny), value=-1)
        else:
            torch.triu(term1, out=term1)
            term1 /= torch.where(norm > 0, _lb(term2, norm), norm).clamp_(tiny)
            term1 @= q
            q.sub_(term1)


def _precond_grad(Q, exprs, G):
    """Precondition gradient G with preconditioner Q."""
    return torch.einsum(exprs[-1], *[q.conj() for q in Q], *Q, G)
