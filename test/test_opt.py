import datetime
import gc
import inspect

import numpy as np
import pytest
import torch
import torch.backends.opt_einsum
import torch.nn as nn
from torch.backends import cudnn

import heavyball.utils
from heavyball import ForeachSOAP, ForeachPSGDKron, ForeachSFAdamW, ForeachADOPT

steps = 10_000
cudnn.benchmark = True
cudnn.deterministic = False
torch.use_deterministic_algorithms(False)
torch.set_float32_matmul_precision("high")  # highest: FP32, high: TF32, medium: bf16
torch.backends.opt_einsum.enabled = True
torch.backends.opt_einsum.strategy = "optimal"

args = {'betas': (0.9, 0.95), 'precondition_frequency': 4, 'merge_dims': False, 'warmup_steps': 100,
        'max_precond_dim': 2 ** 16, 'beta': 0.9, 'preconditioner_update_probability': 1 / 4}


def data(size, batch, dtype):
    inp = torch.randn((batch, size, 1), device='cuda', dtype=torch.float)
    inp = inp > 0
    i0, i1 = inp.chunk(2, 1)
    i0 = torch.logical_xor(i0, i1)
    return inp.to(dtype), i0.to(dtype)


@pytest.mark.parametrize('method', ['qr'])
@pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.parametrize('length,size,batch', [(16, 32, 32)])  # 21, 34 for (16, 32) without compile
@pytest.mark.parametrize('opt', [ForeachSFAdamW, ForeachADOPT, ForeachSOAP, ForeachPSGDKron])
@pytest.mark.parametrize('weight_decay', [0])
def test_f(opt, dtype, size, batch, weight_decay, method, length):
    if "soap" not in opt.__name__.lower() and method != 'qr':
        return
    heavyball.utils.zeroth_power_mode = method

    lr = 1e-3
    losses = []
    lrs = []
    for _ in range(10):
        torch.cuda.empty_cache()
        gc.collect()
        torch.manual_seed(0x1239121)

        a = nn.LSTM(1, size, proj_size=1, batch_first=True).cuda().train()
        signature = inspect.signature(opt)
        o = opt(a.parameters(), lr, weight_decay=weight_decay,
                **{k: v for k, v in args.items() if k in signature.parameters})
        torch.cuda.empty_cache()
        gc.collect()

        start = datetime.datetime.now()

        for i in range(steps):
            inp, tgt = data(length, batch, dtype)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(a(inp)[0].chunk(2, 1)[1], tgt)
            loss.backward()
            o.step()
            o.zero_grad()
            if loss.item() < 0.1:
                dist = datetime.datetime.now() - start
                print(
                    f'Took {dist} | {opt.__name__}, {dtype=}, {size=}, {length=}, {batch=}, {lr=}, {weight_decay=}, {method=} | Iteration: {i}')
                return

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
