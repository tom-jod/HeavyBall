import os
from datetime import datetime

import torch
import torch.nn as nn
import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torchvision.utils import make_grid

import heavyball

heavyball.utils.compile_mode = None
heavyball.utils.set_torch()


class Residual(nn.Sequential):
    def forward(self, input):
        out = super().forward(input)
        return out + F.interpolate(input, out.shape[2:])


class Block(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        intermediate: int,
        out_features: int,
        kernel: int,
        stride: int,
        up: bool,
        depth: int,
    ):
        padding = kernel // 2
        layers = [nn.Conv2d(in_features, intermediate, kernel_size=kernel, padding=padding)]

        for _ in range(depth):
            layers.append(
                Residual(
                    nn.Upsample(scale_factor=stride) if up else nn.MaxPool2d(stride),
                    nn.BatchNorm2d(intermediate),
                    nn.ReLU(),
                    nn.Conv2d(intermediate, intermediate, kernel_size=kernel, padding=padding),
                )
            )

        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(intermediate, out_features, kernel_size=kernel, padding=padding))

        super().__init__(*layers)


class Autoencoder(nn.Module):
    def __init__(self, kernel: int = 3, stride: int = 2, hidden: int = 2, intermediate: int = 128):
        super(Autoencoder, self).__init__()
        self.enc = Block(1, intermediate, hidden, kernel, stride, False, 5)
        self.dec = Block(hidden, intermediate, 1, kernel, stride, True, 5)

    def forward(self, x):
        x = self.enc(x).sigmoid()
        # label = x > torch.rand_like(x)
        # x = label.detach().float() + x - x.detach()
        out = self.dec(x)
        return out


class RandomPad(nn.Module):
    def __init__(self, amount: int):
        super().__init__()
        self.amount = amount

    def forward(self, inp):
        x = torch.randint(0, self.amount, (inp.size(0),))
        y = torch.randint(0, self.amount, (inp.size(0),))
        new = torch.zeros(
            [inp.shape[0], inp.shape[1] + self.amount, inp.shape[2] + self.amount],
            device=inp.device,
            dtype=inp.dtype,
        )
        new[:, x : x + inp.size(1), y : y + inp.size(2)] = inp
        return new


class FastMNIST(torch.utils.data.Dataset):
    def __init__(self, train: bool):
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])
        data = list(MNIST(root="./data", train=train, download=True, transform=transform))
        images, _labels = zip(*data)
        self._data = torch.stack(images)
        self._data.share_memory_()
        self.transform = RandomPad(4)

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, index):
        return self.transform(self._data[index]) / 255.0


def main(epochs: int, batch: int, log_interval: int = 16):
    # Setup tensorboard logging
    log_dir = os.path.join("runs", f"soap_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)

    model = torch.compile(Autoencoder().cuda(), mode="default")
    optimizer = heavyball.PSGDKron(
        model.parameters(),
        lr=1e-4,
        mars=True,
        lower_bound_beta=0.999,
        update_clipping=heavyball.utils.global_l2norm_clip,
    )
    # optimizer = heavyball.AdamW(model.parameters(), lr=1e-3, mars=True)

    trainset = FastMNIST(True)
    testset = FastMNIST(False)

    dataloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    testloader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    eval_batch = next(iter(testloader))
    eval_batch_cuda = eval_batch.cuda()
    del testloader
    step = 0
    total_loss = 0

    for epoch in range(epochs):
        for img in tqdm.tqdm(dataloader):
            step += 1
            img = img.cuda()

            def _closure():
                output = model(img)
                loss = F.mse_loss(output, img)
                loss.backward()
                return loss

            loss = optimizer.step(_closure)
            optimizer.zero_grad()
            with torch.no_grad():
                total_loss = total_loss + loss.detach()

            if step % log_interval == 0:
                avg_loss = (total_loss / log_interval).item()
                writer.add_scalar("Loss/train", avg_loss, step)
                total_loss = 0
            if step % (log_interval * 10) == 0:
                writer.flush()

        with torch.no_grad():
            model.eval()
            samples = model(eval_batch_cuda)
            comparison = torch.cat([eval_batch, samples.cpu()], dim=0)
            grid = make_grid(comparison, nrow=8, normalize=True, padding=2)
            writer.add_image("reconstructions", grid, epoch)
            model.train()
        writer.flush()


if __name__ == "__main__":
    main(epochs=10, batch=16)
