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
    "trivial": {"scale": 2},
    "easy": {"scale": 1e1},
    "medium": {"scale": 1e3},
    "hard": {"size": 1e5},
    "extreme": {"scale": 1e7},
    "nightmare": {"scale": 1e9},
}


class Model(nn.Module):
    def __init__(self, scale: float, size=1024):
        super().__init__()
        # Simulate different layer scales in deep networks
        self.layer1 = nn.Parameter(torch.randn(size))  # Small gradients
        self.layer2 = nn.Parameter(torch.randn(size))  # Medium gradients
        self.layer3 = nn.Parameter(torch.randn(size))  # Large gradients
        self.scale = scale

    def forward(self):
        """Test optimizer's ability to handle different gradient scales across layers."""
        # Each layer contributes equally to the loss but has very different scales
        return (
            self.layer1.square().mean() * self.scale
            + self.layer2.square().mean()
            + self.layer3.square().mean() / self.scale
        ) / 3


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    config: Optional[str] = None,
):
    scale = configs.get(config, {}).get("scale", 1e3)
    dtype = [getattr(torch, d) for d in dtype]
    model = Model(scale).cuda().double()

    def data():
        return None, None

    # More lenient win condition due to vastly different scales
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
        failure_threshold=5,
        base_lr=1e-4,
        trials=trials,
    )  # Lower learning rate and more attempts


if __name__ == "__main__":
    app()
