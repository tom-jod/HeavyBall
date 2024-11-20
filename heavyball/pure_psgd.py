"""
Originally from Evan Walters and Omead Pooladzandi, 2024
Modified under Creative Commons Attribution 4.0 International
Source available at https://github.com/evanatyourservice/kron_torch/blob/97a2b5ee8a1a4c29e4780bbf6c521e545189eff9/kron_torch/kron.py
"""

import torch

from heavyball.utils import identity
from .utils import update_param_, warmup, psgd_precond_grad, init_Q_exprs, PSGDBase, split_p_and_g_in_group, \
    line_to_triu, triu_to_line, promote


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
                 momentum_into_precond_update=True, warmup_steps: int = 1, merge_dims: bool = False,
                 split: bool = False, clip_fn: callable = None, store_triu_as_line: bool = True, foreach: bool = True,
                 q_dtype='float32', stochastic_schedule: bool = True,  #
                 # expert parameters
                 precond_init_scale=1.0, precond_lr=0.1):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if clip_fn is None:
            clip_fn = identity

        defaults = dict(lr=lr, weight_decay=weight_decay, max_size_triangular=max_size_triangular,
                        min_ndim_triangular=min_ndim_triangular, memory_save_mode=memory_save_mode,
                        momentum_into_precond_update=momentum_into_precond_update, precond_lr=precond_lr,
                        precond_init_scale=precond_init_scale, step=0, warmup_steps=warmup_steps, merge_dims=merge_dims,
                        split=split, store_triu_as_line=store_triu_as_line, q_dtype=q_dtype)
        super().__init__(params, defaults, foreach, stochastic_schedule, clip_fn, preconditioner_update_probability)

    def _step(self, group):
        # update preconditioners all together
        precond_init_scale = group['precond_init_scale']
        max_size_triangular = group['max_size_triangular']
        min_ndim_triangular = group['min_ndim_triangular']
        memory_save_mode = group['memory_save_mode']
        precond_lr = group['precond_lr']
        weight_decay = group['weight_decay']
        lr = group['lr']
        store_triu_as_line = group['store_triu_as_line']
        q_dtype = getattr(torch, group['q_dtype'])

        vals = []

        for p, g in split_p_and_g_in_group(group):
            state = self.state_(p)

            if 'Q' not in state:
                Q, state["exprs"] = init_Q_exprs(p, precond_init_scale, max_size_triangular, min_ndim_triangular,
                                                 memory_save_mode, dtype=q_dtype)
                state['Q'] = triu_to_line(Q) if store_triu_as_line else Q

            vals.append((p, g, state["Q"]))

        if not vals:
            return

        p_list, grad_list, Q_list = zip(*vals)
        del vals

        group["step"] += 1

        Q_list = list(Q_list)
        for i, (p, g) in enumerate(zip(p_list, grad_list)):
            q_orig = Q_list.pop(0)
            q = line_to_triu(q_orig) if store_triu_as_line else q_orig

            if self.should_update(group):
                q32 = [promote(q_) for q_ in q]
                self.do_update(group, [p], [g], [q32], precond_lr, [q_orig], store_triu_as_line)
            psgd_precond_grad(q, self.state_(p)["exprs"], g, inplace=True)

        grad_list = self.clip_fn(grad_list)

        lr = -warmup(lr, group['step'], group['warmup_steps'])
        update_param_(p_list, grad_list, lr, weight_decay)
