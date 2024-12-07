import itertools
from typing import List

import heavyball
import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from heavyball.utils import set_torch
from utils import trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn((2,)))

    def forward(self, inp):
        x, y = self.param
        return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2


@app.command()
def main(method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
         dtype: List[str] = typer.Option(["float32"], help='Data type to use'), steps: int = 30_000,
         weight_decay: float = 0,
         opt: List[str] = typer.Option(['SFAdamW'], help='Optimizers to use')):
    dtype = [getattr(torch, d) for d in dtype]
    for args in itertools.product(method, dtype, opt, [weight_decay]):
        m, d, o, wd = args

        model = Model()

        def data():
            inp = torch.zeros((), device='cuda', dtype=d)
            return inp, torch.zeros((), device='cuda', dtype=d)

        def win(_model, loss):
            return loss < 1e-5

        trial(model, data, torch.nn.functional.mse_loss, win, steps, o, d, 1, 1, wd, m, 1, 1, group=5_000,
              base_lr=1e-4, trials=30)


if __name__ == '__main__':
    app()
