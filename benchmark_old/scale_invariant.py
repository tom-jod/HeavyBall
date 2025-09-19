from typing import List, Optional

import torch
import torch.backends.opt_einsum
import typer
from torch import nn

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
configs = {
    "trivial": {"range": 1},
    "easy": {"range": 2},
    "medium": {"range": 3},
    "hard": {"range": 4},
    "extreme": {"range": 5},
    "nightmare": {"range": 6},
}


def objective(x):
    """Tests optimizer's ability to handle different parameter scales."""
    return torch.log1p(x.square()).mean()


class Model(nn.Module):
    def __init__(self, size, value_range):
        super().__init__()
        # Initialize with different scales
        scales = torch.logspace(-value_range, value_range, size)
        self.param = nn.Parameter(scales * torch.randn(size))

    def forward(self):
        return objective(self.param)


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    size: int = 512,
    config: Optional[str] = None,
):
    value_range = configs.get(config, {}).get("range", 3)

    dtype = [getattr(torch, d) for d in dtype]
    model = Model(size, value_range).cuda().double()

    def data():
        return None, None

    trial(
        model,
        data,
        None,
        loss_win_condition(win_condition_multiplier * 1e-3),
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
