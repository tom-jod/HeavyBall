from typing import List, Optional

import torch
import torch.backends.opt_einsum
import typer
from torch import nn

from benchmark.utils import param_norm_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

configs = {
    "trivial": {"offset": 32},
    "easy": {"offset": 16},
    "medium": {"offset": 8},
    "hard": {"offset": 4},
    "extreme": {"offset": 2},
    "nightmare": {"offset": 1},
}


class Model(nn.Module):
    def __init__(self, offset, size=4096):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.register_buffer("step", torch.zeros(1))
        self.offset = offset

    def forward(self):
        """Test optimizer's ability to handle changing noise levels during training."""
        self.step += 1
        # Noise that decreases over time
        noise_scale = 1.0 / (self.offset + self.step)
        noise = torch.randn_like(self.param) * noise_scale
        return (self.param + noise).square().mean()


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
    offset = configs.get(config, {}).get("offset", 4)
    dtype = [getattr(torch, d) for d in dtype]
    model = Model(offset).cuda().double()

    def data():
        return None, None

    # Lenient initial condition due to high initial noise
    trial(
        model,
        data,
        None,
        param_norm_win_condition(win_condition_multiplier * 1e-3, 0),
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
        base_lr=1e-3,
        trials=trials,
    )


if __name__ == "__main__":
    app()
