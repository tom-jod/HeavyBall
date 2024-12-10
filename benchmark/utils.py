import copy
import gc
import inspect
import random
import warnings

import hyperopt
import numpy as np
import torch

import heavyball.utils

np.warnings = warnings

base_args = {'betas': (0.9, 0.999), 'precondition_frequency': 1, 'merge_dims': True, 'warmup_steps': 100,
             'max_precond_dim': 2 ** 16, 'beta': 0.9, 'max_size_triangular': 2 ** 16, 'split': False}


def get_optim(optim, params, **kwargs):
    args = {**base_args, **kwargs}
    signature = inspect.signature(optim)
    o = optim(params, **{k: v for k, v in args.items() if k in signature.parameters})
    return o


def ema(hist, beta=0.9):
    hist = [h.item() if hasattr(h, 'item') else h for h in hist]
    fac = beta ** np.flip(np.arange(len(hist)))
    fac /= fac.sum()
    return np.array(hist) @ fac


def _get_objective(failure_threshold, model, opt, steps, group, data, loss_fn, win_condition, weight_decay, **kwargs):
    m = None
    loss0 = None
    attempt = 0
    best_loss = None
    avg = None

    def _inner(params):
        nonlocal m, loss0, attempt
        params = {'lr': params[0], 'betas': (1 - params[1], 1 - params[2]), 'shampoo_beta': 1 - params[3]}
        m = copy.deepcopy(model)
        o = get_optim(opt, m.parameters(), **params, weight_decay=weight_decay, **kwargs)
        loss_hist = []

        for i in range(steps // group):
            for _ in range(group):
                inp, tgt = data()
                loss = m() if inp is None else m(inp)
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
                if win_condition(m, loss)[0]:
                    return ema(loss_hist), m
                if loss > failure_threshold * loss0 or not np.isfinite(loss):
                    return ema(loss_hist), m
            if hasattr(o, 'train'):
                o.train()
            if loss > failure_threshold * loss0 or not np.isfinite(loss):
                return ema(loss_hist), m
        return ema(loss_hist), m

    def _objective(params):
        nonlocal m, loss0, attempt, avg, best_loss
        attempt += 1
        loss = _inner(params)[0]
        if best_loss is None or loss < best_loss:
            best_loss = loss
            avg = np.log(np.array(params))
        return loss

    def objective(params):
        nonlocal best_loss
        return _objective(params)

    def _get_m():
        return m

    def _get_best():
        print(np.exp(avg))
        return _inner(np.exp(avg))[1]

    return objective, _get_m, _get_best


def trial(model, data, loss_fn, win_condition, steps, opt, dtype, size, batch, weight_decay, method, length, depth,
          trials=10, failure_threshold=3, group=100, base_lr: float = 1e-3):
    opt = getattr(heavyball, opt)
    if "soap" not in opt.__name__.lower() and method != 'qr':
        return

    heavyball.utils.zeroth_power_mode = method

    for attempt in range(trials):
        torch.cuda.empty_cache()
        gc.collect()

        torch.manual_seed(0x1239121)
        np.random.seed(0x1239122)
        random.seed(0x1239123)

        obj, get_m, get_best = _get_objective(failure_threshold, model, opt, steps, group, data, loss_fn, win_condition,
                                              weight_decay)
        out = hyperopt.fmin(obj, (hyperopt.hp.loguniform('lr', np.log(1e-6), np.log(0.1)),  #
                                  hyperopt.hp.loguniform('1mbeta1', np.log(1e-3), np.log(1)),  #
                                  hyperopt.hp.loguniform('1mbeta2', np.log(1e-5), np.log(1)),  #
                                  hyperopt.hp.loguniform('1mshampoo_beta', np.log(1e-3), np.log(1))),  #
                            max_evals=trials, algo=hyperopt.atpe.suggest,
                            early_stop_fn=lambda x: win_condition(get_m(), x), return_argmin=True)
        print(
            f"{opt.__name__}(lr={out['lr']:.5f}, betas=({1 - out['1mbeta1']:.3f}, {1 - out['1mbeta2']:.4f}), shampoo_beta={1 - out['1mshampoo_beta']:.3f})")
        return get_best()
