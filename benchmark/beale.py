import copy
import pathlib
import random
import time
from typing import List

import matplotlib.colors
import matplotlib.pyplot as plt
import torch
import torch.backends.opt_einsum
import typer
from hyperopt import early_stop
from image_descent import FunctionDescent2D
from torch import nn

from benchmark.utils import trial, loss_win_condition
from heavyball.utils import set_torch

early_stop.no_progress_loss()
app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


def beale(x, y):
    x = x + 3
    y = y + 0.5
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2


class Model(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(x).float())

    def forward(self):
        return beale(*self.param)


@app.command()
def main(method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
         dtype: List[str] = typer.Option(['float32'], help='Data type to use'), steps: int = 100,
         weight_decay: float = 0, opt: List[str] = typer.Option(['ForeachSOAP'], help='Optimizers to use'),
         show_image: bool = False, trials: int = 100, win_condition_multiplier: float = 1.0, ):
    dtype = [getattr(torch, d) for d in dtype]
    coords = (-7, -4)

    # Clean up old plots
    for path in pathlib.Path('.').glob('beale_*.png'):
        path.unlink()

    img = None
    colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    stride = max(1, steps // 20)
    rng = random.Random(0x1239121)
    rng.shuffle(colors)

    if show_image:
        model = FunctionDescent2D(lambda *x: beale(*x).log(), coords=coords, xlim=(-8, 2), ylim=(-8, 2), normalize=8,
                                  after_step=torch.exp)
    else:
        model = Model(coords)
    model.double()

    def data():
        return None, None

    start_time = time.time()
    model = trial(model, data, None, loss_win_condition(win_condition_multiplier * 1e-6 * (not show_image)), steps,
                  opt[0], dtype[0], 1, 1, weight_decay, method[0], 1, 1, group=100, base_lr=1e-4, trials=trials,
                  return_best=show_image)
    end_time = time.time()
    print(f"{opt[0]} took {end_time - start_time:.2f} seconds")

    if not show_image:
        return

    if img is None:
        fig, ax = model.plot_image(cmap="gray", levels=20, return_fig=True, xlim=(-8, 2), ylim=(-8, 2))
        ax.set_frame_on(False)
        img = fig, ax

    fig, ax = img
    c = colors[0]
    ax.plot(*list(zip(*model.coords_history)), linewidth=1, color=c, zorder=2, label=f'{method[0]} {opt[0]}')
    ax.scatter(*list(zip(*model.coords_history[::stride])), s=8, zorder=1, alpha=0.75, marker='x', color=c)
    ax.scatter(*model.coords_history[-1], s=64, zorder=3, marker='x', color=c)

    f = copy.deepcopy(fig)
    f.legend()
    f.savefig('beale.png', dpi=1000)
    plt.close(fig)


if __name__ == '__main__':
    app()
