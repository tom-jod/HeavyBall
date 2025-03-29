import pytest
import torch
from torch import nn

import heavyball
import heavyball.utils
from benchmark.utils import get_optim
from heavyball.utils import clean, set_torch


def get_memory():
    clean()
    clean()
    torch.cuda.synchronize()
    out = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    torch.cuda.reset_accumulated_memory_stats()
    return out


@pytest.mark.parametrize("opt", ["NewtonHybrid2PSGDKron", "ForeachPSGDKron", "ForeachADOPT", "SOAP"])
@pytest.mark.parametrize("size,depth", [(256, 1), (128, 4)])
@pytest.mark.parametrize("mars", [True, False])
@pytest.mark.parametrize("merge_dims", [True, False])
@pytest.mark.parametrize("split", [True, False])
@pytest.mark.parametrize("finite_differences", [True, False])
def test_memory(
    opt,
    size,
    depth: int,
    mars: bool,
    merge_dims: bool,
    split: bool,
    finite_differences: bool,
    iterations: int = 500,
    warmup: int = 100,
    max_growth: float = 1.05,
):
    set_torch()

    opt = getattr(heavyball, opt)
    model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
    print(model)

    o = get_optim(opt, model.parameters(), lr=1e-3, mars=mars, merge_dims=merge_dims, split=split)
    if finite_differences:
        if not o.hessian_approx:
            pytest.skip("Finite Differences is an HVP calculation - can't do it on non-hvp optimizers")
        o.finite_differences = True

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
            assert peak * max_growth >= get_memory()  # fudge factor
