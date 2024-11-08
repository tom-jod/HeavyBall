import torch
import torch.optim

from .utils import schedule_free_, warmup, ScheduleFree, exp_avg_sq_, beta_debias


class PaLMForeachSFAdamW(ScheduleFree):
    r"""
    Schedule-Free AdamW
    As the name suggests, no scheduler is needed with this optimizer.
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.

    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Should be significantly faster, but will have higher peak memory
            usage (default True if supported in your PyTorch version).
    """

    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0, r=0.0,
                 weight_lr_power=2.0, foreach=hasattr(torch, "_foreach_mul_")):

        defaults = dict(lr=lr, betas=betas, eps=eps, r=r, k=0, warmup_steps=warmup_steps, train_mode=True,
                        weight_sum=0.0, lr_max=-1.0, weight_lr_power=weight_lr_power, weight_decay=weight_decay,
                        foreach=foreach)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            decay = group['weight_decay']
            k = group['k']

            if not group['train_mode']:
                raise Exception("Not in train mode!")

            active_p = [p for p in group['params'] if p.grad is not None]

            for p in active_p:
                if 'z' not in self.state[p]:
                    self.state[p]['z'] = torch.clone(p.data)
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)

            y, grad, exp_avg_sq, z = zip(
                *[(p.data, p.grad.float(), self.state[p]['exp_avg_sq'], self.state[p]['z']) for p in active_p])

            # Decay the first moment running average coefficient
            beta2 = 1 - (k + 1) ** -0.8
            old_debiased = beta_debias(beta2, k + 1)

            # Decay the first and second moment running average coefficient
            denom = exp_avg_sq_(exp_avg_sq, grad, old_debiased, eps)

            # Normalize grad in-place for memory efficiency
            torch._foreach_div_(grad, denom)

            # Weight decay calculated at y
            if decay != 0:
                torch._foreach_add_(grad, y, alpha=decay)

            lr = warmup(group['lr'], k + 1, group['warmup_steps'])
            group['weight_sum'] = schedule_free_(lr, group['weight_lr_power'], group['weight_sum'], group['betas'][0],
                                                 y, z, grad)

            group['k'] = k + 1
        return loss
