import os

import heavyball
import heavyball.utils
import pytest
import torch
from benchmark.utils import get_optim
from heavyball.utils import clean, set_torch
from torch import nn
from torch._dynamo import config
import torch._inductor.config as ind_cfg

os.environ['TORCH_LOGS'] = '+recompiles'

config.cache_size_limit = 128

@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(256, 1)])
def test_foreach(opt, size, depth: int, iterations: int = 16, outer_iterations: int = 3):
    set_torch()
    opt = getattr(heavyball, opt)

    peaks = []
    losses = []

    for dtype in [torch.float32, torch.bfloat16]:
        torch.manual_seed(0x2131290)
        peaks.append([])
        losses.append([])

        for i in range(outer_iterations):
            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda().to(dtype)
            o = get_optim(opt, model.parameters(), lr=1e-3, weight_decay=1e-4, warmup_steps=16,
                          max_size_triangular=2048, merge_dims=True, split=False, memory_save_mode='one_diag',
                          store_triu_as_line=False, stochastic_schedule=False, storage_dtype='float32',
                          q_dtype='float32')

            for _ in range(iterations):
                loss = model(torch.randn((1024, size), device='cuda', dtype=dtype)).square().mean()
                loss.backward()
                o.step()
                o.zero_grad()
                losses[-1].append(loss.detach())

            del model, o
            clean()

    for i, (l0, l1) in enumerate(zip(*losses)):
        print(i, l0.item(), l1.item())
        assert torch.allclose(l0.float(), l1.float(), rtol=0.1)
