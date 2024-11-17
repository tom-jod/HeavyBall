import itertools
from typing import List

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer

from heavyball.utils import set_torch
from utils import trial

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


def win(loss):
    return loss < 0.1


@app.command()
def main(method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
         dtype: List[str] = typer.Option(["float32"], help='Data type to use'), length: int = 128, size: int = 128,
         depth: int = 1, batch: int = 256, steps: int = 100_000, weight_decay: float = 0,
         opt: List[str] = typer.Option(['ForeachAdamW', 'ForeachADOPT', 'ForeachLaProp', 'ForeachSOAP'], help='Optimizers to use')):
    dtype = [getattr(torch, d) for d in dtype]

    for args in itertools.product(method, dtype, [(length, size, depth, batch)], opt, [weight_decay]):
        m, d, (l, s, dp, b), o, wd = args

        model = Model(s, dp)

        def data():
            inp = torch.randn((b, l, 1), device='cuda', dtype=d)
            inp = inp > 0
            return inp.to(d), (inp.sum(1) % 2).to(d)

        trial(model, data, torch.nn.functional.binary_cross_entropy_with_logits, win, steps, o, d, s, b, wd, m, l, dp,
              failure_threshold=10)


if __name__ == '__main__':
    app()
