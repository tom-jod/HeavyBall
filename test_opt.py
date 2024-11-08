import datetime
import gc
import inspect
import time

import pytest
import torch
import torch.nn as nn
from torch.backends import cudnn

from heavyball import PaLMForeachSOAP, SFPaLMForeachSOAP, PaLMForeachSFAdamW, PrecondScheduleSFPaLMSOAP

dtype = torch.bfloat16
steps = 100_000
size = 2 ** 7
batch = 128

cudnn.benchmark = True
cudnn.deterministic = False
torch.use_deterministic_algorithms(False)
torch.set_float32_matmul_precision("high")  # highest: FP32, high: TF32, medium: bf16

args = {'lr': 0.001, 'betas': (0.9, 0.95), 'precondition_frequency': 2, 'merge_dims': False, 'warmup_steps': 100,
        'max_precond_dim': 2 ** 16, 'beta': 0.9}

@pytest.mark.parametrize('opt', [PaLMForeachSOAP, SFPaLMForeachSOAP, PaLMForeachSFAdamW, PrecondScheduleSFPaLMSOAP])
def test_f(opt):
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)
    torch.manual_seed(0x1239121)

    a = nn.Linear(size, size, bias=False).cuda().to(dtype)
    signature = inspect.signature(opt)
    o = opt(a.parameters(), **{k: v for k, v in args.items() if k in signature.parameters})
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
    try:
        o.eval()
    except:
        pass
    eval_loss = (a(inp) - inp).square().mean()
    dist = datetime.datetime.now() - start
    print(f'Took {dist} | {opt.__name__} | Loss: {loss.item()} - Eval: {eval_loss.item()} - Mean: {loss_mean.item()}')
