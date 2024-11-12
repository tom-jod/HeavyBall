import pytest
import torch
from torch import nn

import heavyball
import heavyball.utils
from benchmark.utils import get_optim, set_torch
from heavyball.utils import clean


def get_memory():
    clean()
    torch.cuda.synchronize()
    clean()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


expected_memory = {'adamw': {'after': 4, 'peak': 5.1}, 'soap': {'after': 7, 'peak': 14}, 'psgd': {'after': 2, 'peak': 6}}


@pytest.mark.parametrize("opt", ['ForeachAdamW', 'ForeachSOAP', 'ForeachPSGDKron'])
@pytest.mark.parametrize("method", ['qr', 'newtonschulz2', 'svd', 'eigh'])
@pytest.mark.parametrize("size", [8192])
def test_memory(opt, method, size, iterations: int = 5):
    if 'soap' not in opt.lower() and method != 'qr':
        return
    set_torch()

    for k, v in expected_memory.items():
        if k in opt.lower():
            break
    else:
        raise ValueError(f'Unknown optimizer {opt}')

    opt = getattr(heavyball, opt)
    heavyball.utils.zeroth_power_mode = method

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    torch.cuda.reset_accumulated_memory_stats()

    for i in range(iterations):
        model = nn.Linear(size, size).cuda()

        model_allocated = get_memory()
        o = get_optim(opt, model.parameters(), lr=1e-3)
        model(torch.randn((1, size)).cuda()).sum().backward()
        o.step()

        opt_allocated = get_memory()
        o.zero_grad()

        del model, o
        peak = torch.cuda.memory_stats()['allocated_bytes.all.peak']

        assert peak / model_allocated < v['peak']

    assert opt_allocated / model_allocated < v['after']
