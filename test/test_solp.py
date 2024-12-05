import heavyball
import heavyball.utils
import pytest
import torch
from benchmark.utils import get_optim
from heavyball.utils import clean, set_torch, ScheduleFree
from torch import nn
from torch._dynamo import config

config.cache_size_limit = 128

@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(128, 2)])
def test_solp(opt, size, depth: int, iterations: int = 65536, outer_iterations: int = 2):
    set_torch()
    if 'SOAP' not in opt:
        raise pytest.skip('This test is only for SOAP')

    opt_name = opt
    peaks = []
    losses = []

    for use_solp in [True, False]:
        try:
            opt = getattr(heavyball, opt_name.replace("SOAP", "SOLP") if use_solp else opt_name)
        except AttributeError:
            raise pytest.skip(f'{opt_name} does not have a SOLP variant')
        print(opt, opt_name.replace("SOAP", "SOLP"))

        torch.manual_seed(0x2131290)
        peaks.append([])
        losses.append([])

        for i in range(outer_iterations):
            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
            o = get_optim(opt, model.parameters(), lr=1e-5)

            for _ in range(iterations):
                loss = model(torch.randn((1024, size), device='cuda')).square().mean()
                loss.backward()
                o.step()
                o.zero_grad()
                losses[-1].append(loss.detach())

            del model, o
            clean()

    for i, (l0, l1) in enumerate(zip(*losses)):
        print(i, l0.item(), l1.item())
        assert l0.item() <= l1.item() * 1.1
