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
    "trivial": {"size": 4},
    "easy": {"size": 32},
    "medium": {"size": 512},
    "hard": {"size": 2048},
    "extreme": {"size": 8192},
    "nightmare": {"size": 2**14},
}


class Model(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.param = nn.Parameter(torch.randn((size, size)))
        self.target = nn.Buffer(torch.triu(torch.ones_like(self.param)))
        self.size = size

    def forward(self):
        inp = torch.randn((16, self.size), device="cuda", dtype=self.param.dtype)
        target = inp.cumsum(1)
        return (inp @ self.param - target).square().mean()


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    size: int = 1024,
    depth: int = 4,
    batch: int = 16,
    steps: int = 10,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    config: Optional[str] = None,
):
    size = configs.get(config, {}).get("size", size)
    dtype = getattr(torch, dtype[0])
    model = Model(size).cuda()

    def data():
        return None, None

    trial(
        model,
        None,
        None,
        param_norm_win_condition(1e-3 * win_condition_multiplier, model.target),
        steps,
        opt[0],
        weight_decay=weight_decay,
        failure_threshold=depth * 2,
        trials=trials,
    )


if __name__ == "__main__":
    app()
