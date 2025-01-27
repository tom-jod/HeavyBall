import itertools
from typing import List

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.nn import functional as F

import heavyball
from heavyball.utils import set_torch
from benchmark.utils import trial, loss_win_condition

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn((2,)))

    def forward(self, inp):
        x, y = self.param
        # Return a tensor with batch dimension
        return ((1 - x) ** 2 + 1 * (y - x ** 2) ** 2).unsqueeze(0)


@app.command()
def main(
    method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
    dtype: List[str] = typer.Option(['float32'], help='Data type to use'),
    steps: int = 10,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(['ForeachSOAP'], help='Optimizers to use'),
    trials: int = 10,
    win_condition_multiplier: float = 1.0,
):
    dtype = [getattr(torch, d) for d in dtype]
    model = Model().cuda()

    def data():
        inp = torch.zeros((1,), device='cuda', dtype=dtype[0])
        target = torch.zeros((1,), device='cuda', dtype=dtype[0])
        return inp, target

    trial(model, data, F.mse_loss, loss_win_condition(1e-5 * win_condition_multiplier), steps, opt[0], dtype[0], 1, 1, weight_decay, method[0], 1, 1,
          failure_threshold=3, group=100, base_lr=1e-3, trials=trials)


if __name__ == '__main__':
    app()
