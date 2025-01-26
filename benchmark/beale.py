import copy
import itertools
import pathlib
import random
import time
from typing import List, Union

import hyperopt
import matplotlib.colors
import matplotlib.pyplot as plt
import torch
import torch.backends.opt_einsum
import typer
from heavyball.utils import set_torch
from hyperopt import early_stop
from image_descent import FunctionDescent2D
from utils import trial

early_stop.no_progress_loss()
app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


def beale(x, y):
    x = x + 3
    y = y + 0.5
    return torch.log((1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2)


@app.command()
def main(method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
         dtype: List[str] = typer.Option(["float32"], help='Data type to use'), steps: int = 100,
         weight_decay: float = 0,
         opt: List[str] = typer.Option(['ForeachSOAP', 'PaLMForeachSOAP', 'PrecondScheduleForeachSOAP'], help='Optimizers to use')):
    dtype = [getattr(torch, d) for d in dtype]
    coords = (-7, -4)

    for path in pathlib.Path('.').glob('beale_*.png'):
        path.unlink()

    img = None
    colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    stride = max(1, steps // 20)
    rng = random.Random(0x1239121)
    rng.shuffle(colors)

    for args in itertools.product(method, dtype, opt, [weight_decay]):
        m, d, o, wd = args

        model = FunctionDescent2D(beale, coords=coords, xlim=(-8, 2), ylim=(-8, 2), normalize=8, after_step=torch.exp)
        model.double()

        def data():
            inp = torch.zeros((), device='cuda', dtype=d)
            return None, None

        def win(_model, loss: Union[float, hyperopt.Trials]):
            if not isinstance(loss, float):
                loss = loss.results[-1]['loss']
            return loss < 0, {}

        start_time = time.time()
        model = trial(model, data, None, win, steps, o, d, 1, 1, wd, m, 1, 1, group=100, base_lr=1e-4, trials=100)
        end_time = time.time()
        print(f"{o} took {end_time - start_time:.2f} seconds")

        if img is None:
            fig, ax = model.plot_image(cmap="gray", levels=20, return_fig=True, xlim=(-8, 2), ylim=(-8, 2))
            ax.set_frame_on(False)
            img = fig, ax

        fig, ax = img
        c = colors.pop(0)
        ax.plot(*list(zip(*model.coords_history)), linewidth=1, color=c, zorder=2, label=f'{m} {o}')
        ax.scatter(*list(zip(*model.coords_history[::stride])), s=8, zorder=1, alpha=0.75, marker='x', color=c)
        ax.scatter(*model.coords_history[-1], s=64, zorder=3, marker='x', color=c)

        f = copy.deepcopy(fig)
        f.legend()
        f.savefig(f'beale.png', dpi=1000)
    plt.close(fig)


if __name__ == '__main__':
    app()
