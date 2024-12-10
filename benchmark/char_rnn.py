import datetime

import heavyball
import torch
from heavyball import ForeachPSGDKron
from heavyball.utils import set_torch
from torch import nn
from torch.nn import functional as F

set_torch()


class Take0(nn.Module):
    def forward(self, x):
        return x[0]


def main(features: int = 512, sequence: int = 256, batch: int = 16, printervall: int = 1000):
    model = nn.Sequential(nn.Embedding(256, 512), nn.LSTM(features, features, 1, dropout=0.5, batch_first=True), Take0(),
                          nn.Linear(features, 256)).cuda()
    opt = heavyball.ForeachPSGDKron(model.parameters(), lr=1e-3)

    with open('shakespeare.txt', 'rb') as f:
        text = f.read()
    chars = torch.frombuffer(text, dtype=torch.uint8).cuda().long()

    holdout = chars[:(sequence + 1) * batch].view(batch, sequence + 1)
    chars = chars[(sequence + 1) * batch:]

    offsets = torch.arange(0, sequence + 1, device='cuda').repeat(batch, 1)

    step = 0
    start = datetime.datetime.now()
    losses = 0
    accuracy = 0

    for step in range(1, 10 ** 9):
        batch_offsets = torch.randint(0, len(chars) - sequence - 1, (batch,), device='cuda')
        batch_offsets = batch_offsets[:, None] + offsets
        batch_chars = chars[batch_offsets]
        batch_chars = batch_chars.view(batch, sequence + 1)
        src = batch_chars[:, :-1]
        tgt = batch_chars[:, 1:]
        out = model(src)
        loss = F.cross_entropy(out.view(-1, 256), tgt.flatten())
        loss.backward()
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            accuracy = accuracy + (out.argmax(-1) == tgt).sum().detach()
            losses = losses + loss.detach()
            if step % printervall == 0:
                accuracy = accuracy.item() / (printervall * batch * sequence)
                loss = losses / printervall
                eval_loss = F.cross_entropy(model(holdout[:, :-1]).view(-1, 256), holdout[:, 1:].flatten())
                eval_accuracy = (model(holdout[:, :-1]).argmax(-1) == holdout[:, 1:]).sum().item() / (batch * sequence)
                print(f'{datetime.datetime.now() - start} | {step:5d} | Train Loss: {loss.item():8.5f} - '
                      f'Accuracy: {accuracy * 100:.2f}% | Eval Loss: {eval_loss.item():8.5f} - Accuracy: {eval_accuracy * 100:.2f}%')
                losses = 0
                accuracy = 0


main()
