import datetime
import gc
import inspect

import numpy as np
import torch

import heavyball.utils

base_args = {'betas': (0.9, 0.999), 'precondition_frequency': 1, 'merge_dims': False, 'warmup_steps': 100,
             'max_precond_dim': 2 ** 16, 'beta': 0.9, 'preconditioner_update_probability': 1}


def get_optim(optim, params, **kwargs):
    args = {**base_args, **kwargs}
    signature = inspect.signature(optim)
    o = optim(params, **{k: v for k, v in args.items() if k in signature.parameters})
    return o


def trial(model, data, loss_fn, win_condition, steps, opt, dtype, size, batch, weight_decay, method, length, depth,
          trials=10, failure_threshold=3):
    opt = getattr(heavyball, opt)
    if "soap" not in opt.__name__.lower() and method != 'qr':
        return

    print(f'{opt.__name__} | {dtype} {size=} {batch=} {weight_decay=} {method=} {length=} {depth=}')

    heavyball.utils.zeroth_power_mode = method

    lr = 1e-3
    losses = []
    lrs = []
    loss0 = None
    for _ in range(trials):
        torch.cuda.empty_cache()
        gc.collect()
        torch.manual_seed(0x1239121)

        m = torch.compile(model.to(dtype).cuda(), mode='max-autotune', fullgraph=False, dynamic=False)
        o = get_optim(opt, m.parameters(), lr=lr, weight_decay=weight_decay)
        torch.cuda.empty_cache()
        gc.collect()

        start = datetime.datetime.now()
        loss_hist = []

        for i in range(steps):
            inp, tgt = data()
            out = m(inp)
            loss = loss_fn(out, tgt)
            loss.backward()
            o.step()
            o.zero_grad()
            loss_hist.append(loss.item())
            if loss0 is None:
                loss0 = loss.item()
            if win_condition(loss_hist[-1]):
                dist = datetime.datetime.now() - start
                print(f'Took {dist} | {opt.__name__}, {dtype=}, {size=}, {length=}, {batch=}, {lr=}, {weight_decay=}, '
                      f'{method=} | Iteration: {i}')
                return
            if loss_hist[-1] > failure_threshold * loss0 or not np.isfinite(loss_hist[-1]):
                print(f'{opt.__name__} diverged at {i=}, loss={loss_hist[-1]}, {loss0}')
                break

        print(f'{lr=} did not converge')
        lrs.append(lr)
        lrs = sorted(lrs)
        losses.insert(lrs.index(lr), loss.item())

        argmin = np.argmin(losses)
        if argmin == 0:
            lr /= 2
        elif argmin == len(losses) - 1:
            lr *= 3
        else:
            lr = (lrs[argmin - 1] + lrs[argmin + 1]) / 2

    raise ValueError(f"Failed to converge")
