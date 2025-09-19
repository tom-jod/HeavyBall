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


class Model(nn.Module):
    def __init__(self, size, value_range):
        super().__init__()
        self.scale = nn.Buffer(torch.logspace(-value_range, value_range, size))
        self.param = nn.Parameter(torch.randn(size))

    def forward(self):
        return (self.param**2 * self.scale).mean()


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

    model = Model(size, value_range).cuda().double()

    trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-3),
        steps,
        opt[0],
        weight_decay,
        failure_threshold=3,
        trials=trials,
    )


if __name__ == "__main__":
    app()
