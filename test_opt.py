import datetime
import gc
import inspect
import time

import pytest
import torch
import torch.nn as nn
from torch.backends import cudnn

from heavyball import PaLMForeachSOAP, SFPaLMForeachSOAP, PaLMForeachSFAdamW, PrecondScheduleSFPaLMSOAP, ForeachADOPT, \
    ForeachSOAP, ForeachSFAdamW, ForeachLaProp, PrecondScheduleForeachSOAP, PrecondSchedulePaLMForeachSOAP

steps = 10000

cudnn.benchmark = True
cudnn.deterministic = False
torch.use_deterministic_algorithms(False)
torch.set_float32_matmul_precision("high")  # highest: FP32, high: TF32, medium: bf16

args = {'betas': (0.9, 0.95), 'precondition_frequency': 2, 'merge_dims': False, 'warmup_steps': 100,
        'max_precond_dim': 2 ** 16, 'beta': 0.9}


@pytest.mark.parametrize('opt', [PrecondSchedulePaLMForeachSOAP, SFPaLMForeachSOAP, PaLMForeachSFAdamW, PrecondScheduleSFPaLMSOAP,
                                 ForeachADOPT, ForeachSOAP, ForeachSFAdamW, ForeachLaProp, PrecondScheduleForeachSOAP,
                                 PrecondSchedulePaLMForeachSOAP, torch.optim.AdamW, torch.optim.Adam])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('size,batch', [(128, 128)])
@pytest.mark.parametrize('lr', [0.1, 1e-2, 1e-3, 1e-4])
@pytest.mark.parametrize('weight_decay', [1e-2])
def test_f(opt, dtype, size, batch, lr, weight_decay):
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)
    torch.manual_seed(0x1239121)

    a = nn.Linear(size, size, bias=False).cuda().to(dtype)
    signature = inspect.signature(opt)
    o = opt(a.parameters(), lr, weight_decay=weight_decay,
            **{k: v for k, v in args.items() if k in signature.parameters})
    torch.cuda.empty_cache()
    gc.collect()

    loss_mean = 0
    start = datetime.datetime.now()

    for i in range(steps):
        inp = torch.randn((batch, size), device='cuda', dtype=dtype)
        loss = (a(inp) - inp).square().mean()
        loss.backward()
        o.step()
        o.zero_grad()
        with torch.no_grad():
            loss_mean += loss.detach() / steps

    if hasattr(o, 'eval'):
        o.eval()
    eval_loss = (a(inp) - inp).square().mean()
    dist = datetime.datetime.now() - start
    print(f'Took {dist} | {opt.__name__} | Loss: {loss.item()} - Eval: {eval_loss.item()} - Mean: {loss_mean.item()}')
