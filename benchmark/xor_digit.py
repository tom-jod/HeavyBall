import os
import itertools
from typing import List

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.nn import functional as F

import heavyball
from heavyball.utils import set_torch
from benchmark.utils import trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class Model(nn.Module):
    def __init__(self, size, depth):
        super().__init__()
        self.embed = nn.Embedding(2, size)
        self.enc = nn.LSTM(size, size, depth, batch_first=True)
        self.proj = nn.Linear(size, 1, bias=False)

    def forward(self, inp):
        inp = self.embed(inp.squeeze(-1).long())
        out, _ = self.enc(inp)
        return self.proj(out[:, -1])


def win(_model, loss):
    if isinstance(loss, float):
        return loss < 0.01, {}
    return False, {}


@app.command()
def main(
    method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
    dtype: List[str] = typer.Option(['float32'], help='Data type to use'),
    length: int = 4,
    size: int = 2048,
    depth: int = 1,
    batch: int = 16,
    steps: int = 10,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(['ForeachSOAP'], help='Optimizers to use'),
):
    dtype = [getattr(torch, d) for d in dtype]
    torch.manual_seed(0x1239121)
    model = Model(size, depth).cuda()

    def data():
        inp = torch.randn((batch, length, 1), device='cuda', dtype=dtype[0])
        inp = inp > 0
        return inp.to(dtype[0]), (inp.sum(1) % 2).to(dtype[0])

    trial(model, data, F.binary_cross_entropy_with_logits, win, steps, opt[0], dtype[0], size, batch, weight_decay, method[0], length, depth,
          failure_threshold=10, base_lr=1e-6, trials=20)


if __name__ == '__main__':
    app()
