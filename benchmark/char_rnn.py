import datetime

import heavyball
import torch
from heavyball import ForeachPSGDKron
from heavyball.utils import set_torch
from numba.np.npdatetime import datetime_minus_timedelta
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


"""
LaProp, no dropout, lr=1e-3
0:00:19.349051 |  1000 | Train Loss:  1.73716 - Accuracy: 49.06% | Eval Loss:  1.49760 - Accuracy: 54.54%
0:00:38.476921 |  2000 | Train Loss:  1.42494 - Accuracy: 56.51% | Eval Loss:  1.41697 - Accuracy: 56.20%
0:00:57.589616 |  3000 | Train Loss:  1.34504 - Accuracy: 58.52% | Eval Loss:  1.37730 - Accuracy: 57.84%
0:01:16.776616 |  4000 | Train Loss:  1.29590 - Accuracy: 59.75% | Eval Loss:  1.35823 - Accuracy: 57.84%
0:01:36.029076 |  5000 | Train Loss:  1.26080 - Accuracy: 60.64% | Eval Loss:  1.33327 - Accuracy: 58.45%
0:01:55.932413 |  6000 | Train Loss:  1.23130 - Accuracy: 61.41% | Eval Loss:  1.32194 - Accuracy: 59.25%
0:02:14.910573 |  7000 | Train Loss:  1.20622 - Accuracy: 62.13% | Eval Loss:  1.33725 - Accuracy: 58.76%
0:02:34.706537 |  8000 | Train Loss:  1.18389 - Accuracy: 62.73% | Eval Loss:  1.32489 - Accuracy: 59.38%

LaProp, dropout=0.5, decay=0.1, lr=1e-3
0:00:18.707218 |  1000 | Train Loss:  1.72649 - Accuracy: 49.20% | Eval Loss:  1.48849 - Accuracy: 55.91%
0:00:37.325233 |  2000 | Train Loss:  1.41182 - Accuracy: 56.77% | Eval Loss:  1.40718 - Accuracy: 56.54%
0:00:56.403623 |  3000 | Train Loss:  1.33651 - Accuracy: 58.63% | Eval Loss:  1.36069 - Accuracy: 58.45%
0:01:15.169717 |  4000 | Train Loss:  1.29757 - Accuracy: 59.59% | Eval Loss:  1.36479 - Accuracy: 57.71%
0:01:33.906788 |  5000 | Train Loss:  1.26813 - Accuracy: 60.34% | Eval Loss:  1.34825 - Accuracy: 59.28%
0:01:53.695672 |  6000 | Train Loss:  1.24695 - Accuracy: 60.89% | Eval Loss:  1.34470 - Accuracy: 59.30%
0:02:13.276058 |  7000 | Train Loss:  1.23555 - Accuracy: 61.17% | Eval Loss:  1.35736 - Accuracy: 58.50%

"""