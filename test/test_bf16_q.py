import heavyball
import heavyball.utils
import pytest
import torch
from benchmark.utils import get_optim
from heavyball.utils import clean, set_torch, PSGDBase
from torch import nn


def get_memory():
    clean()
    torch.cuda.synchronize()
    clean()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(256, 2)])
def test_foreach(opt, size, depth: int, iterations: int = 128, outer_iterations: int = 3):
    set_torch()

    opt = getattr(heavyball, opt)
    if not issubclass(opt, PSGDBase):
        raise pytest.skip('Only PSGD is supported')

    peaks = []
    losses = []

    for q_dtype in ['float32', 'bfloat16']:
        peaks.append([])
        losses.append([])

        for i in range(outer_iterations):
            torch.manual_seed(0x2131290)
            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
            o = get_optim(opt, model.parameters(), lr=1e-3, q_dtype=q_dtype)

            for _ in range(iterations):
                loss = model(torch.randn((1024, size)).cuda()).square().mean()
                loss.backward()
                o.step()
                o.zero_grad()
                losses[-1].append(loss.detach())

            del model, o
            clean()


    for i, (l0, l1) in enumerate(zip(*losses)):
        print(i, l0.item(), l1.item())
        assert torch.allclose(l0, l1, rtol=0.1)
