"""
Originally from Evan Walters and Omead Pooladzandi, 2024
Modified under Creative Commons Attribution 4.0 International
Source available at https://github.com/evanatyourservice/kron_torch/blob/97a2b5ee8a1a4c29e4780bbf6c521e545189eff9/kron_torch/kron.py
"""

import torch

from heavyball.utils import triu_to_line, line_to_triu, identity
from .utils import update_param_, warmup, psgd_precond_grad, init_Q_exprs, PSGDBase, exp_avg_sq_, beta_debias, \
    split_p_and_g_in_group, promote


class ForeachPaLMPAdam(PSGDBase):
    """
    Kronecker Factorized Adam with PSGD preconditioner

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
                 momentum_into_precond_update=True, warmup_steps: int = 1, betas=(None, None), beta: float = 0.9,
                 beta2_scale: float = 0.8, merge_dims: bool = False, split: bool = False, clip_fn: callable = None,
                 store_triu_as_line: bool = True, foreach: bool = True, q_dtype='float32',
                 stochastic_schedule: bool = True,  #
                 # expert parameters
                 precond_init_scale=1.0, precond_lr=0.1):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if betas[0] is not None:
            beta = betas[0]

        if clip_fn is None:
            clip_fn = identity

        defaults = dict(lr=lr, weight_decay=weight_decay, max_size_triangular=max_size_triangular,
                        min_ndim_triangular=min_ndim_triangular, memory_save_mode=memory_save_mode,
                        momentum_into_precond_update=momentum_into_precond_update, precond_lr=precond_lr,
                        precond_init_scale=precond_init_scale, step=0, warmup_steps=warmup_steps, beta=beta,
                        beta2_scale=beta2_scale, merge_dims=merge_dims, split=split,
                        store_triu_as_line=store_triu_as_line, q_dtype=q_dtype)
        super().__init__(params, defaults, foreach, stochastic_schedule, clip_fn, preconditioner_update_probability)

    def _step(self, group):
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
                state['exp_avg'] = torch.zeros_like(g)
                state['exp_avg_sq'] = torch.zeros_like(g)
                Q, state["exprs"] = init_Q_exprs(p, precond_init_scale, max_size_triangular, min_ndim_triangular,
                                                 memory_save_mode, dtype=q_dtype)
                state['Q'] = triu_to_line(Q) if store_triu_as_line else Q

            vals.append((p, g, state["Q"], state['exp_avg'], state['exp_avg_sq']))

        if not vals:
            return

        p_list, grad_list, Q_list, exp_avg, exp_avg_sq = zip(*vals)
        del vals

        group["step"] += 1

        Q_triu = [line_to_triu(q) if store_triu_as_line else q for q in Q_list]
        if self.should_update(group):
            for g, p, q_, q_orig in zip(grad_list, p_list, Q_triu, Q_list):
                q32 = [promote(qq_) for qq_ in q_]
                self.do_update(group, [p], [g], [q32], precond_lr, [q_orig], store_triu_as_line)
        torch._foreach_lerp_(exp_avg, grad_list, 1 - beta_debias(group['beta'], group['step']))

        beta2 = 1 - group['step'] ** -group['beta2_scale']

        for p, Q, g, ea, eas in zip(p_list, Q_triu, grad_list, exp_avg, exp_avg_sq):
            psgd_precond_grad(Q, self.state_(p)["exprs"], g, inplace=True)
            ea = psgd_precond_grad(Q, self.state_(p)["exprs"], ea)
            exp_avg_sq_(eas, g, beta_debias(beta2, group['step']), 1e-8, out=g)
            torch.div(ea, g, out=g)
            """
            divide by g here, because g == denom (from exp_avg_sq_(out=g)), avoids denom allocation
            divide into g so we can deallocate ea, avoids one allocation (-> less memory than equivalent foreach)
            """

        grad_list = self.clip_fn(grad_list)

        lr = -warmup(lr, group['step'], group['warmup_steps'])
        update_param_(p_list, grad_list, lr, weight_decay)
