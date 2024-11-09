import datetime
import gc
import inspect

import pytest
import torch
import torch.nn as nn
from torch.backends import cudnn

import heavyball.utils
from heavyball import PaLMForeachSOAP, SFPaLMForeachSOAP, PaLMForeachSFAdamW, PrecondScheduleSFPaLMSOAP, ForeachADOPT, \
    ForeachSOAP, ForeachSFAdamW, ForeachLaProp, PrecondScheduleForeachSOAP, PrecondSchedulePaLMForeachSOAP

steps = 1000
cudnn.benchmark = True
cudnn.deterministic = False
torch.use_deterministic_algorithms(False)
torch.set_float32_matmul_precision("high")  # highest: FP32, high: TF32, medium: bf16

args = {'betas': (0.9, 0.95), 'precondition_frequency': 1, 'merge_dims': False, 'warmup_steps': 100,
        'max_precond_dim': 2 ** 16, 'beta': 0.9}


def data(size, batch, dtype):
    inp = torch.randn((batch, size), device='cuda', dtype=dtype)
    tgt = (inp.cumsum(1) - torch.flip(inp, [1]).cumsum(1)) / size ** 0.5
    return inp, tgt


@pytest.mark.parametrize('method',
                         ['newtonschulz2', 'newtonschulz5', 'newtonschulz10', 'newtonschulz20', 'newtonschulz50'])
@pytest.mark.parametrize('opt', [PaLMForeachSOAP, SFPaLMForeachSOAP, PaLMForeachSFAdamW, PrecondScheduleSFPaLMSOAP,
                                 ForeachADOPT, ForeachSOAP, ForeachSFAdamW, ForeachLaProp, PrecondScheduleForeachSOAP,
                                 PrecondSchedulePaLMForeachSOAP, torch.optim.AdamW, torch.optim.Adam])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.double])
@pytest.mark.parametrize('size,batch', [(128, 32)])
@pytest.mark.parametrize('lr', [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
@pytest.mark.parametrize('weight_decay', [0])
def test_f(opt, dtype, size, batch, lr, weight_decay, method):
    heavyball.utils.zeroth_power_mode = method
    torch.cuda.empty_cache()
    gc.collect()
    torch.manual_seed(0x1239121)

    a = nn.Sequential(nn.Linear(size, size, bias=False), nn.ReLU(), nn.LayerNorm(size),
                      nn.Linear(size, size)).cuda().to(dtype)
    signature = inspect.signature(opt)
    o = opt(a.parameters(), lr, weight_decay=weight_decay,
            **{k: v for k, v in args.items() if k in signature.parameters})
    torch.cuda.empty_cache()
    gc.collect()

    loss_mean = 0
    start = datetime.datetime.now()

    for i in range(steps):
        inp, tgt = data(size, batch, dtype)
        loss = (a(inp) - tgt).square().mean()
        loss.backward()
        o.step()
        o.zero_grad()
        with torch.no_grad():
            loss_mean += loss.detach() / steps

    if hasattr(o, 'eval'):
        o.eval()

    inp, tgt = data(size, batch, dtype)
    eval_loss = (a(inp) - tgt).square().mean()
    dist = datetime.datetime.now() - start
    print(f'Took {dist} | {opt.__name__}, {dtype=}, {size=}, {batch=}, {lr=}, {weight_decay=}, {method=} | '
          f'Loss: {loss.item()} - Eval: {eval_loss.item()} - Mean: {loss_mean.item()}')
