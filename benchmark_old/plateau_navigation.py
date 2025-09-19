import math
import pathlib
import random
from typing import List, Optional

import matplotlib.colors
import torch
import torch.backends.opt_einsum
import typer
from torch import nn
from utils import Plotter

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

configs = {
    "trivial": {"scale": 1},
    "easy": {"scale": 4},
    "medium": {"scale": 8},
    "hard": {"scale": 12},
    "extreme": {"scale": 16},
    "nightmare": {"scale": 20},
}


def objective(x, y, scale: float):
    """Tests optimizer's ability to handle regions with very small gradients and sharp plateaus."""
    output = 1 / (1 + torch.exp((x**2 + y**2 - 1) * -scale))
    minimum = 1 / (1 + math.exp(scale))
    return output - minimum  # ensure the minimum is at 0


class Model(nn.Module):
    def __init__(self, x, scale):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(x).float())
        self.scale = scale

    def forward(self):
        return objective(*self.param, scale=self.scale)


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    show_image: bool = False,
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    config: Optional[str] = None,
):
    scale = configs.get(config, {}).get("scale", 4)

    dtype = [getattr(torch, d) for d in dtype]
    coords = (1.5, 1.5)  # Start outside the plateau

    # Clean up old plots
    for path in pathlib.Path(".").glob("plateau_navigation.png"):
        path.unlink()

    colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    rng = random.Random(0x1239121)
    rng.shuffle(colors)

    if show_image:
        model = Plotter(lambda *x: objective(*x, scale=scale).log())
    else:
        model = Model(coords, scale=scale)
    model.double()

    def data():
        return None, None

    trial(
        model,
        data,
        None,
        loss_win_condition(win_condition_multiplier * 1e-4),
        steps,
        opt[0],
        dtype[0],
        1,
        1,
        weight_decay,
        method[0],
        1,
        1,
        failure_threshold=3,
        base_lr=1e-3,
        trials=trials,
    )


if __name__ == "__main__":
    app()
