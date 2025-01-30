import math
from typing import List

import torch
import torch.backends.opt_einsum
import typer
from torch import nn

from benchmark.utils import trial, loss_win_condition
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class Model(nn.Module):
    def __init__(self, size=1024):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.register_buffer('batch_sizes', torch.tensor([1, 4, 16, 64, 256]))
        self.current_batch = 0

    def forward(self):
        """Test optimizer's ability to handle different batch sizes and noise scales."""
        batch_size = self.batch_sizes[self.current_batch].item()
        self.current_batch = (self.current_batch + 1) % len(self.batch_sizes)
        
        # Add noise scaled by batch size
        noise = torch.randn_like(self.param) / math.sqrt(batch_size)
        return (self.param + noise).square().mean()


@app.command()
def main(method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
         dtype: List[str] = typer.Option(['float32'], help='Data type to use'), steps: int = 100,
         weight_decay: float = 0, opt: List[str] = typer.Option(['ForeachSOAP'], help='Optimizers to use'),
         trials: int = 100, win_condition_multiplier: float = 1.0, ):
    dtype = [getattr(torch, d) for d in dtype]
    model = Model().cuda().double()

    def data():
        return None, None

    # Use a more lenient win condition since we have inherent noise
    trial(model, data, None, loss_win_condition(win_condition_multiplier * 1e-4), steps, opt[0], dtype[0], 1, 1,
          weight_decay, method[0], 1, 1, failure_threshold=3, base_lr=1e-3, trials=trials)


if __name__ == '__main__':
    app()
