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
    "trivial": {"classes": 64, "items": 2},  # 4096 unique samples, easy to overfit
    "easy": {"classes": 64, "items": 4},  # 16e6 samples, will see entire dataset in 1M-step run (batch=16)
    "medium": {"classes": 64, "items": 6},  # 68e9, 0.025% will be seen
    "hard": {"classes": 64, "items": 8},  # 281e12,
    "extreme": {"classes": 64, "items": 10},  # 1e18
    "nightmare": {"classes": 64, "items": 12},  # 4e21, 1e-13%
}


class Model(nn.Module):
    def __init__(self, classes: int, items: int, features: int = 4, hidden: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Embedding(classes, features),
            nn.Flatten(),
            nn.Linear(items * features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, classes + 1),
        )
        self.classes = classes
        self.items = items
        self.weight = nn.Buffer(torch.randn((items,)))

    def forward(self):
        data = torch.randint(0, self.classes, (16, self.items), device=self.model[0].weight.device)
        y = self.model(data)
        target = (data.float() @ self.weight.float()).round().long() % self.classes
        return torch.nn.functional.cross_entropy(y, target)


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
    config = configs[config or "trivial"]
    model = Model(**config)

    trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-8 * (not show_image)),
        steps,
        opt[0],
        weight_decay,
        trials=trials,
        return_best=False,
    )


if __name__ == "__main__":
    app()
