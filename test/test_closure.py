from typing import List

import heavyball
import heavyball.utils
import pytest
import torch
from benchmark.utils import get_optim
from heavyball.utils import set_torch, clean
from torch import nn


class Param(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size))

    def forward(self, inp):
        return self.weight.mean() * inp


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size", [(4, 4, 4, 4), ])
def test_closure(opt, size: List[int], depth: int = 2, iterations: int = 5, outer_iterations: int = 3):
    clean()
    set_torch()

    opt = getattr(heavyball, opt)

    for _ in range(outer_iterations):
        clean()
        model = nn.Sequential(*[Param(size) for _ in range(depth)]).cuda()
        o = get_optim(opt, model.parameters(), lr=1e-3)

        def _closure():
            loss = model(torch.randn((1, size[0]), device='cuda')).sum()
            loss.backward()
            return loss

        for i in range(iterations):
            o.step(_closure)
            o.zero_grad()
            print(o.state_size())

        del model, o
