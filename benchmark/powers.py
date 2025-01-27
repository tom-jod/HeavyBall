import itertools
from typing import List

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.nn import functional as F

import heavyball
from heavyball.utils import set_torch
from benchmark.utils import trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class Model(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.register_buffer('scale', torch.arange(size).float().add(1))

    def forward(self, inp):
        return self.param.pow(self.scale)


def win(_model, loss):
    if not isinstance(loss, float):
        loss = loss.results[-1]['loss']
    return loss < 1e-10, {}


@app.command()
def main(
    method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
    dtype: List[str] = typer.Option(['float32'], help='Data type to use'),
    size: int = 128,
    steps: int = 10,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(['ForeachSOAP'], help='Optimizers to use'),
):
    dtype = [getattr(torch, d) for d in dtype]
    model = Model(size).cuda()

    def data():
        inp = torch.zeros((1,), device='cuda', dtype=dtype[0])
        return inp, torch.zeros_like(model.param, device='cuda', dtype=dtype[0])

    trial(model, data, F.mse_loss, win, steps, opt[0], dtype[0], 1, 1, weight_decay, method[0], 1, 1,
          failure_threshold=3, group=100, base_lr=1e-3, trials=20)


if __name__ == '__main__':
    app()
