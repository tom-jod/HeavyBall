"""
Originally from Evan Walters and Omead Pooladzandi, 2024
Modified under Creative Commons Attribution 4.0 International
Source available at https://github.com/evanatyourservice/kron_torch/blob/97a2b5ee8a1a4c29e4780bbf6c521e545189eff9/kron_torch/kron.py
"""

import random

import torch

from .utils import promote, update_param_, warmup, psgd_precond_grad, init_Q_exprs, \
    PSGDBase, precond_update_prob_schedule


class ForeachPurePSGD(PSGDBase):
    """
    Kronecker Factorized PSGD WITHOUT Momentum

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate.
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

    def __init__(self, params, lr=0.001, weight_decay=0.0, preconditioner_update_probability=None,
                 max_size_triangular=2048, min_ndim_triangular=2, memory_save_mode=None,
                 momentum_into_precond_update=True, warmup_steps: int = 1):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()
        self.preconditioner_update_probability = preconditioner_update_probability

        defaults = dict(lr=lr, weight_decay=weight_decay,
                        max_size_triangular=max_size_triangular, min_ndim_triangular=min_ndim_triangular,
                        memory_save_mode=memory_save_mode, momentum_into_precond_update=momentum_into_precond_update,
                        precond_lr=0.1,  # precond lr hardcoded to 0.1
                        precond_init_scale=1.0,  # precond init scale hardcoded to 1.0
                        step=0, warmup_steps=warmup_steps)
        super().__init__(params, defaults)

        self._prob_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # update preconditioners all together
        update_prob = self.preconditioner_update_probability
        if callable(update_prob):
            update_prob = update_prob(self._prob_step)
        do_update = self.rng.random() < update_prob
        self._prob_step += 1

        for group in self.param_groups:
            precond_init_scale = group['precond_init_scale']
            max_size_triangular = group['max_size_triangular']
            min_ndim_triangular = group['min_ndim_triangular']
            memory_save_mode = group['memory_save_mode']
            precond_lr = group['precond_lr']
            weight_decay = group['weight_decay']
            lr = group['lr']

            vals = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = promote(p.grad)
                p.grad = None
                state = self.state[p]

                if 'step' not in state:
                    state["Q"], state["exprs"] = init_Q_exprs(p, precond_init_scale, max_size_triangular,
                                                              min_ndim_triangular, memory_save_mode, dtype=grad.dtype)
                    state['step'] = 0

                vals.append((p, grad, state["Q"]))

            if not vals:
                continue

            p_list, grad_list, Q_list = zip(*vals)
            del vals

            group["step"] += 1

            self.balance(do_update, grad_list, Q_list)
            if do_update:
                self.do_update(p_list, grad_list, Q_list, precond_lr)

            for p, Q, g in zip(p_list, Q_list, grad_list):
                psgd_precond_grad(Q, self.state[p]["exprs"], g, inplace=True)

            lr = -warmup(lr, group['step'], group['warmup_steps'])
            update_param_(p_list, grad_list, lr, weight_decay)

        return loss
