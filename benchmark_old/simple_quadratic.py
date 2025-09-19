import pathlib
import random
from typing import List, Optional

import matplotlib.colors
import torch
import torch.backends.opt_einsum
import typer
from torch import nn

from benchmark.utils import Plotter, SkipConfig, loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
app = typer.Typer()

torch._dynamo.config.disable = True

def simple_quadratic(x, y):
    """Simple quadratic: f(x,y) = x^2 + y^2"""
    return x**2 + y**2


class SimpleModel(nn.Module):
    def __init__(self, x, y):
        super().__init__()
        self.x = nn.Parameter(torch.tensor(x).float())
        self.y = nn.Parameter(torch.tensor(y).float())
    
    def forward(self):
        return simple_quadratic(self.x, self.y)


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    steps: int = 5,  # Keep small for manual verification
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["MARSAdamW"], help="Optimizers to use"),
    show_image: bool = False,
    trials: int = 1,  # Single trial for debugging
    win_condition_multiplier: float = 1.0,
    config: Optional[str] = None,
):
    if config is not None and config != "trivial":
        raise SkipConfig("'config' must be 'trivial'.")
    
    dtype = [getattr(torch, d) for d in dtype]
    
    # Simple starting point
    start_x, start_y = 2.0, 1.0
    
    # Clean up old plots
    for path in pathlib.Path(".").glob("simple_test.png"):
        path.unlink()
    
    if show_image:
        model = Plotter(SimpleModel(start_x, start_y), (start_x, start_y), 
                       x_limits=(-1, 3), y_limits=(-1, 2), should_normalize=True)
    else:
        model = SimpleModel(start_x, start_y)
        model.double()
    
    def data():
        return None, None
    
    print(f"Starting parameters: x={start_x}, y={start_y}")
    print(f"Initial loss: {simple_quadratic(start_x, start_y)}")
    print(f"Initial gradients: dx=2*{start_x}={2*start_x}, dy=2*{start_y}={2*start_y}")
    
    model = trial(
        model,
        data,
        None,
        loss_win_condition(win_condition_multiplier * 1e-9 * (not show_image)),
        steps,
        opt[0],
        dtype[0],
        1,
        1,
        weight_decay,
        method[0],
        1,
        1,
        base_lr=0.1,  # Simple learning rate
        trials=trials,
        return_best=show_image,
        warmup_trial_pct = 0.0,
    )
    
    if show_image:
        model.plot(save_path="simple_test.png")


if __name__ == "__main__":
    app()