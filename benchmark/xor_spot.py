"""
Inspired by https://github.com/lixilinx/psgd_torch/blob/master/rnn_xor_problem_general_purpose_preconditioner.py
This version is strongly simplified but follows the same basic idea:
1) Generate random sequence
2) Mark two spots
3) Train a model to predict the xor of the two spots
This does NOT elicit memory in the RNN, but it does force it to learn a pointwise forget mechanism.
"""
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
        self.embed = nn.Embedding(4, size)
        self.enc = nn.LSTM(size, size, depth, batch_first=True)
        self.proj = nn.Linear(size, 1, bias=False)

    def forward(self, inp):
        inp = self.embed(inp.squeeze(-1).long())
        inp = inp[0] + inp[1]
        out, _ = self.enc(inp)
        return self.proj(out[:, -1])


def win(loss):
    return loss < 0.1


@app.command()
def main(method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
         dtype: List[str] = typer.Option(["float32"], help='Data type to use'), length: int = 8, size: int = 32,
         depth: int = 1, batch: int = 32, steps: int = 100_000, weight_decay: float = 0,
         opt: List[str] = typer.Option(['ForeachPaLMPAdam', 'ForeachPSGDKron'], help='Optimizers to use')):
    dtype = [getattr(torch, d) for d in dtype]

    for args in itertools.product(method, dtype, [(length, size, depth, batch)], opt, [weight_decay]):
        m, d, (l, s, dp, b), o, wd = args

        model = Model(s, dp)

        def data():
            inp = torch.randn((b, l, 1), device='cuda', dtype=d)
            inp = inp > 0
            zeros = torch.zeros_like(inp)
            zeros[:, torch.randint(0, l, (b,), device='cuda')] = 1
            zeros[:, torch.randint(0, l, (b,), device='cuda')] = 1
            target = (inp * zeros).sum(1) % 2
            return torch.stack((inp, zeros + 2), 0).to(d), target.to(d)

        trial(model, data, torch.nn.functional.binary_cross_entropy_with_logits, win, steps, o, d, s, b, wd, m, l, dp,
              failure_threshold=10)


if __name__ == '__main__':
    app()
