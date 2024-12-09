import copy
import datetime
import gc
import inspect

import hyperopt
import numpy as np
import torch

import heavyball.utils

base_args = {'betas': (0.9, 0.999), 'precondition_frequency': 1, 'merge_dims': True, 'warmup_steps': 100,
             'max_precond_dim': 2 ** 16, 'beta': 0.9, 'max_size_triangular': 2 ** 16, 'split': False}


def get_optim(optim, params, **kwargs):
    args = {**base_args, **kwargs}
    signature = inspect.signature(optim)
    o = optim(params, **{k: v for k, v in args.items() if k in signature.parameters})
    return o


def _get_objective(failure_threshold, model, opt, steps, group, data, loss_fn, win_condition, weight_decay, **kwargs):
    m = None
    loss0 = None
    start = datetime.datetime.now()
    attempt = 0
    best_loss = None
    best_m = None

    def _objective(params):
        nonlocal m, loss0, attempt
        attempt += 1
        lr = params['lr']
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
                    dist = datetime.datetime.now() - start
                    return loss
                if loss > failure_threshold * loss0 or not np.isfinite(loss):
                    return 1e6 + lr
            if hasattr(o, 'train'):
                o.train()
            if loss > failure_threshold * loss0 or not np.isfinite(loss):
                return loss
        return loss

    def objective(params):
        nonlocal best_loss, best_m
        loss = _objective(params)
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_m = copy.deepcopy(m)
        return loss

    def _get_m():
        return m

    def _get_best():
        return best_m

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

        obj, get_m, get_best = _get_objective(failure_threshold, model, opt, steps, group, data, loss_fn, win_condition,
                                              weight_decay, betas=(0.9, 0.99))
        hyperopt.fmin(obj, {"lr": hyperopt.hp.loguniform('lr', np.log(1e-6), np.log(0.1))}, max_evals=trials,
                      algo=hyperopt.tpe.suggest, early_stop_fn=lambda x: win_condition(get_m(), x))
        return get_best()
