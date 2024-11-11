import itertools
from typing import List

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer

from utils import set_torch, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class Model(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.param = nn.Parameter(torch.randn((size,)))

    def forward(self, inp):
        y = None
        y0 = self.param.view(1, -1).expand(inp.size(0), -1) + 1  # offset, so weight decay doesnt help
        for i in inp.unbind(1):
            y = torch.einsum('bi,bik->bk', y0, i)
            y0 = torch.nn.functional.leaky_relu(y)
        return y



@app.command()
def main(method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
         dtype: List[str] = typer.Option(["float32"], help='Data type to use'), length: int = 60, size: int = 32,
         depth: int = 1, batch: int = 32, steps: int = 100_000, weight_decay: float = 0,
         opt: List[str] = typer.Option(['ForeachLaProp', 'ForeachSOAP', 'ForeachPSGDKron'], help='Optimizers to use')):
    dtype = [getattr(torch, d) for d in dtype]

    for args in itertools.product(method, dtype, [(length, size, depth, batch)], opt, [weight_decay]):
        m, d, (l, s, dp, b), o, wd = args

        model = Model(s)

        def data():
            inp = torch.randn((b, dp, s, s), device='cuda', dtype=d)
            return inp, torch.zeros((b, s), device='cuda', dtype=d)

        def win(_loss):
            with torch.no_grad():
                return model.param.add(1).norm().item() < 1e-3

        trial(model, data, torch.nn.functional.mse_loss, win, steps, o, d, s, b, wd, m, l, dp, failure_threshold=s ** 2)


if __name__ == '__main__':
    app()
