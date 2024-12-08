import itertools
import pathlib
import random
from typing import List, Union

import heavyball
import hyperopt
import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from heavyball.utils import set_torch
from image_descent import FunctionDescent2D
from utils import trial
from hyperopt import early_stop

early_stop.no_progress_loss()
app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


def beale(x, y):
    return torch.log((1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2)


@app.command()
def main(method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
         dtype: List[str] = typer.Option(["float32"], help='Data type to use'), steps: int = 1_000,
         weight_decay: float = 0, opt: List[str] = typer.Option(heavyball.__all__, help='Optimizers to use')):
    dtype = [getattr(torch, d) for d in dtype]
    coords = (-3.5, -3.5)

    for path in pathlib.Path('.').glob('beale_*.png'):
        path.unlink()

    for args in itertools.product(method, dtype, opt, [weight_decay]):
        m, d, o, wd = args

        model = FunctionDescent2D(beale, coords=coords, xlim=(-4, 4), ylim=(-4, 4), normalize=False, after_step=torch.exp)
        model.double()

        def data():
            return None, torch.zeros((), dtype=torch.float)

        def win(_model, loss: Union[float, hyperopt.Trials]):
            if not isinstance(loss, float):
                loss = loss.results[-1]['loss']
            return loss < 1e-8, {}

        model = trial(model, data, torch.nn.functional.l1_loss, win, steps, o, d, 1, 1, wd, m, 1, 1, group=100,
                      base_lr=0.1, trials=50)
        fig, _ = model.plot_path(return_fig=True)
        fig.savefig(f'beale_{m}_{o}.png')


if __name__ == '__main__':
    app()
