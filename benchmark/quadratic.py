import itertools
from typing import List
import time
import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.nn import functional as F

import heavyball
from heavyball.utils import set_torch
from utils import trial
from image_descent import FunctionDescent2D
app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class Model(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.register_buffer('scale', F.normalize(torch.arange(size).float().add(1).square(), dim=0))

    def forward(self, inp):
        return torch.einsum('a,a,a->', self.param, self.param, self.scale)


@app.command()
def main(method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
         dtype: List[str] = typer.Option(["float32"], help='Data type to use'), steps: int = 100,
         weight_decay: float = 0,
         opt: List[str] = typer.Option(['ForeachSOAP', 'PaLMForeachSOAP', 'PrecondScheduleForeachSOAP'], help='Optimizers to use')):
    dtype = [getattr(torch, d) for d in dtype]
    for args in itertools.product(method, dtype, opt, [weight_decay]):
        m, d, o, wd = args

        model = Model().cuda()

        def data():
            inp = torch.zeros((), device='cuda', dtype=d)
            return inp, torch.zeros((), device='cuda', dtype=d)

        def win(_model, loss):
            if isinstance(loss, float):
                return loss < 1e-5, {}
            return False, {}

        start_time = time.time()
        trial(model, data, torch.nn.functional.mse_loss, win, steps, o, d, 1, 1, wd, m, 1, 1)
        end_time = time.time()
        print(f"{o} took {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    app()
