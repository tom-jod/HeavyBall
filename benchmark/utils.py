import copy
import gc
import inspect
import random
import sys
import time
import warnings
from datetime import datetime
from typing import Union

import heavyball.utils
import hyperopt
import numpy as np
import torch

np.warnings = warnings

base_args = {'betas': (0.9, 0.999), 'precondition_frequency': 1, 'merge_dims': True, 'warmup_steps': 100,
             'max_precond_dim': 2 ** 16, 'beta': 0.9, 'max_size_triangular': 2 ** 16, 'split': False, 'eps': 1e-8,
             'weight_decay': 1e-4}


def get_optim(optim, params, **kwargs):
    args = {**base_args, **kwargs}
    signature = inspect.signature(optim)
    o = optim(params, **{k: v for k, v in args.items() if k in signature.parameters})
    return o


def ema(hist, beta=0.9):
    fac = beta ** np.arange(len(hist), 0, -1)
    fac /= fac.sum()
    return hist @ fac


class Objective:
    def __init__(self, failure_threshold, model, opt, steps, group, data, loss_fn, win_condition, weight_decay,
                 max_consecutive_failures=1, minimal_improvement=1e-2, **kwargs):
        self.failure_threshold = failure_threshold
        self.model = torch.compile(model.cuda(), mode='max-autotune-no-cudagraphs')
        self.opt = opt
        self.steps = steps
        self.group = group
        self.data = data
        self.loss_fn = loss_fn
        self.win_condition = win_condition
        self.weight_decay = weight_decay
        self.kwargs = kwargs
        self.max_consecutive_failures = max_consecutive_failures    
        self.minimal_improvement = minimal_improvement

        self.m = None
        self.loss0 = None
        self.attempt = 0
        self.best_loss = None
        self.avg = None

    def _inner(self, params):
        params = {'lr': params[0], 'betas': (1 - params[1], 1 - params[2]), 'shampoo_beta': 1 - params[3], 'eps': 1e-8,
                  'precond_lr': params[3]  # we never have both precond_lr and shampoo_beta
                  }
        self.m = copy.deepcopy(self.model)
        o = get_optim(self.opt, self.m.parameters(), **params, weight_decay=self.weight_decay, **self.kwargs)
        torch_hist = []
        loss_hist = np.empty(self.steps, dtype=np.float64)
        ema_states = np.empty(8, dtype=np.float64)
        update_factor = 3.0 ** (-np.arange(1, 9))
        consecutive_failures = 0


        for i in range(self.steps // self.group):
            if hasattr(o, 'train'):
                o.train()

            for _ in range(self.group):
                inp, tgt = self.data()

                def _closure():
                    loss = self.m() if inp is None else self.m(inp)
                    if self.loss_fn is not None:
                        loss = self.loss_fn(loss, tgt)
                    loss.backward()
                    return loss

                loss = o.step(_closure)
                o.zero_grad()
                torch_hist.append(loss.detach())
                if self.loss0 is None:
                    self.loss0 = torch_hist[-1].item()
            if hasattr(o, 'eval'):
                o.eval()
            for j, h in enumerate(torch_hist):
                loss_hist[i * self.group + j] = h.item()
            torch_hist.clear()
            lh = loss_hist[:(i + 1) * self.group]
            if i == 0:
                ema_states[:] = np.stack([ema(lh, 1 - u) for u in update_factor])            
            else:
                for loss in lh[-self.group:]:
                    ema_states += update_factor * (loss - ema_states) 
                    failed = np.any(ema_states[1:] * (1 - self.minimal_improvement) < ema_states[:-1])
                    if failed:
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 0
                    if consecutive_failures >= self.max_consecutive_failures:
                        return ema_states[-1], self.m, loss
            for j, loss in enumerate(lh[-self.group:]):
                if self.win_condition(self.m, loss)[0] or loss > self.failure_threshold * self.loss0 or not np.isfinite(loss):
                    return ema_states[-1], self.m, loss                    
        return ema_states[-1], self.m, loss

    def objective(self, params):
        self.attempt += 1
        target, _, loss = self._inner(params)
        if self.best_loss is None or loss < self.best_loss or not np.isfinite(self.best_loss):
            self.best_loss = loss
            self.avg = np.log(np.array(params))
        return target

    def get_best(self):
        self.group = 1
        return self._inner(np.exp(self.avg))[1]


def loss_win_condition(target):
    def win(_model, loss: Union[float, hyperopt.Trials]):
        if not isinstance(loss, float):
            loss = loss.results[-1]['loss']
        return loss <= target, {}

    return win


def param_norm_win_condition(target, offset):
    def win(model, loss):
        with torch.no_grad():
            return model.param.add(offset).norm().item() < target, {}

    return win


def trial(model, data, loss_fn, win_condition, steps, opt, dtype, size, batch, weight_decay, method, length, depth,
          trials=10, failure_threshold=3, group=256, base_lr: float = 1e-3, return_best: bool = False):
    kwargs = {'caution': False, 'mars': False}
    if opt.startswith('cautious-'):
        opt = opt[len('cautious-'):]
        kwargs['caution'] = True
    if opt.startswith('unscaled_cautious-'):
        opt = opt[len('unscaled_cautious-'):]
        heavyball.utils.disable_caution_scaling()
        kwargs['caution'] = True
    if opt.startswith('mars-'):
        opt = opt[len('mars-'):]
        kwargs['mars'] = True
    opt = getattr(heavyball, opt)
    if "soap" not in opt.__name__.lower() and method != 'qr':
        return

    heavyball.utils.zeroth_power_mode = method

    torch.cuda.empty_cache()
    gc.collect()

    torch.manual_seed(0x1239121)
    np.random.seed(0x1239122)
    random.seed(0x1239123)

    did_win = False

    def _win_condition(*args):
        nonlocal did_win
        win_state, out = win_condition(*args)
        did_win |= win_state
        return did_win, out

    obj = Objective(failure_threshold, model, opt, steps, group, data, loss_fn, _win_condition, weight_decay, **kwargs)
    start_time = time.time()
    sys.stdout = sys.stderr
    out = hyperopt.fmin(obj.objective, (hyperopt.hp.loguniform('lr', np.log(1e-6), np.log(0.1)),  #
                                        hyperopt.hp.loguniform('1mbeta1', np.log(1e-3), np.log(1)),  #
                                        hyperopt.hp.loguniform('1mbeta2', np.log(1e-5), np.log(1)),  #
                                        hyperopt.hp.loguniform('1mshampoo_beta', np.log(1e-4), np.log(1))),  #
                        max_evals=trials, algo=hyperopt.atpe.suggest, early_stop_fn=lambda x: _win_condition(obj.m, x),
                        return_argmin=True, show_progressbar=True)
    torch.cuda.synchronize()
    sys.stdout = sys.__stdout__
    end_time = time.time()
    if did_win:
        print("Successfully found the minimum.")
    print(f"Took: {end_time - start_time} | Attempt: {obj.attempt} | "  #
          f"{opt.__name__}(lr={out['lr']:.5f}, betas=({1 - out['1mbeta1']:.3f}, {1 - out['1mbeta2']:.4f}), "  #
          f"shampoo_beta={1 - out['1mshampoo_beta']:.3f}) | Best Loss: {obj.best_loss}")
    if return_best:
        return obj.get_best()
