import pytest
import torch
from torch import nn

import heavyball
import heavyball.utils
from benchmark.utils import get_optim
from heavyball.utils import clean, set_torch


def get_memory():
    clean()
    torch.cuda.synchronize()
    clean()
    torch.cuda.synchronize()
    out = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    torch.cuda.reset_accumulated_memory_stats()
    return out


@pytest.mark.parametrize("opt", ["NewtonHybrid2PSGDKron", "ForeachPSGDKron", "ForeachADOPT", "SOAP"])
@pytest.mark.parametrize("size,depth", [(8192, 2), (2048, 16)])
def test_memory(opt, size, depth: int, iterations: int = 500, warmup: int = 50):
    set_torch()

    opt = getattr(heavyball, opt)
    model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
    print(model)

    o = get_optim(opt, model.parameters(), lr=1e-3)
    peak = 0
    for i in range(iterations):
        data = torch.randn((1, size), device="cuda").requires_grad_(True)

        def _closure():
            nonlocal model
            loss = (model(data) - data).square().mean()
            loss.backward()
            return loss

        o.step(_closure)

        if i <= warmup:
            peak = max(peak, get_memory())
        if i > warmup:
            assert peak >= get_memory() * 0.95  # fudge factor
