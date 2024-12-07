import copy
import datetime
import gc
import inspect
import random

import heavyball.utils
import numpy as np
import torch

base_args = {'betas': (0.9, 0.999), 'precondition_frequency': 1, 'merge_dims': True, 'warmup_steps': 100,
             'max_precond_dim': 2 ** 16, 'beta': 0.9, 'max_size_triangular': 2 ** 16, 'split': False}


def get_optim(optim, params, **kwargs):
    args = {**base_args, **kwargs}
    signature = inspect.signature(optim)
    o = optim(params, **{k: v for k, v in args.items() if k in signature.parameters})
    return o


def trial(model, data, loss_fn, win_condition, steps, opt, dtype, size, batch, weight_decay, method, length, depth,
          trials=10, failure_threshold=3, group=100, base_lr: float = 1e-3):
    opt = getattr(heavyball, opt)
    if "soap" not in opt.__name__.lower() and method != 'qr':
        return

    heavyball.utils.zeroth_power_mode = method

    lr = base_lr
    losses = []
    lrs = []
    loss0 = None
    for attempt in range(trials):
        torch.cuda.empty_cache()
        gc.collect()
        torch.manual_seed(0x1239121)

        m = torch.compile(copy.deepcopy(model).to(dtype).cuda(), mode='max-autotune', fullgraph=False, dynamic=False)
        o = get_optim(opt, m.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99),
                      store_triu_as_line=False)
        torch.cuda.empty_cache()
        gc.collect()

        start = datetime.datetime.now()
        loss_hist = []
        rng = random.Random(0x9128391)

        for i in range(steps // group):
            for _ in range(group):
                inp, tgt = data()
                out = m(inp)
                loss = loss_fn(out, tgt)
                loss.backward()
                o.step()
                o.zero_grad()
                loss_hist.append(loss.detach())
                if loss0 is None:
                    loss0 = loss_hist[-1].item()
            if hasattr(o, 'eval'):
                o.eval()
            for j, loss in enumerate(loss_hist[-group:], group * i):
                loss = loss.item()
                if win_condition(m, loss):
                    dist = datetime.datetime.now() - start
                    print(
                        f'Took {dist} | {opt.__name__}, {dtype=}, {size=}, {length=}, {batch=}, {lr=}, {weight_decay=}, '
                        f'{method=} | Iteration: {j} | Attempt: {attempt + 1} | Loss: {loss}')
                    return
                if loss > failure_threshold * loss0 or not np.isfinite(loss):
                    print(f'{opt.__name__} diverged at {j=}, loss={loss}, {loss0}')
                    loss = 1e6 + lr
                    break
            if hasattr(o, 'train'):
                o.train()
            if loss > failure_threshold * loss0 or not np.isfinite(loss):
                break
            #print(datetime.datetime.now() - start, (i + 1) * group, sum(loss_hist[-group:]).div(group).item())

        lrs.append(lr)
        lrs = sorted(lrs)
        losses.insert(lrs.index(lr), loss)
        print(m.param.add(1).norm().item(), loss)
        print(
            f'{opt.__name__} failed at attempt {attempt + 1} | {datetime.datetime.now() - start} | {dtype=}, {size=}, {length=}, {batch=}, {lr=}, {weight_decay=}, '
            f'{method=}')

        if len(losses) > 1 and all(ls == losses[0] for ls in losses):
            if rng.random() > 0.5:
                lr /= 7
            else:
                lr *= 4
            continue

        argmin = np.argmin(losses)
        if argmin == 0:
            lr /= 3
        elif argmin == len(losses) - 1:
            lr *= 5
        else:
            lr = (lrs[argmin - 1] + lrs[argmin + 1]) / 2

    raise ValueError(f"Failed to converge")
