import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm
from preconditioned_stochastic_gradient_descent import LRA
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
    def __init__(self, kernel: int = 5, stride: int = 2, hidden: int = 8, intermediate: int = 128):
        super(Autoencoder, self).__init__()
        self.enc = Block(1, intermediate, hidden, kernel, stride, False, 1)
        self.balancer = nn.BatchNorm2d(hidden, affine=False)
        self.dec = Block(hidden, intermediate, 1, kernel, stride, True, 1)

    def forward(self, x):
        x = self.enc(x)
        x = self.balancer(x).sigmoid()
        out = self.dec(x)
        return out


def plot_samples(model, data, epoch, save_dir="samples"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        samples = model(data.cuda())
        # Create a grid of original and reconstructed images
        comparison = torch.cat([data, samples.cpu() * 255.0], dim=0)
        grid = make_grid(comparison, nrow=8, normalize=True, padding=2)
        plt.figure(figsize=(10, 5))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"))
        plt.close()
    model.train()


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


def mean(updates):
    return [sum(us) / len(us) for us in zip(*updates)]


def main(epochs: int, batch: int):
    # Setup tensorboard logging
    log_dir = os.path.join("runs", f"soap_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)

    torch.manual_seed(0x12783)
    model = Autoencoder().cuda()
    # optimizer = heavyball.ForeachPSGDLRA(
    #     model.parameters(), lr=1e-3, mars=True,precond_init_scale=1, precond_lr=0.1
    # )
    optimizer = LRA(
        model.parameters(),
        rank_of_approximation=20,
        preconditioner_init_scale=1,
        lr_params=1e-4,
        lr_preconditioner=0.1,
        exact_hessian_vector_product=False,
        preconditioner_type="whitening",
        momentum=0.9,
    )
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32), RandomPad(4)])
    trainset = list(MNIST(root="./data", train=True, download=True, transform=transform)) * epochs
    dataloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    step = 0
    losses = []

    for data in tqdm.tqdm(dataloader):
        img, _ = data
        img = img.to(device="cuda", non_blocking=True) / 255.0

        def _closure():
            output = model(img)
            loss = F.mse_loss(output, img)
            # loss.backward()
            return loss

        loss = optimizer.step(_closure)
        losses.append(loss.detach())
        if len(losses) >= 64:
            for loss in losses:
                writer.add_scalar("Loss/train", loss.item(), step)
                step += 1
            losses.clear()
            writer.flush()


if __name__ == "__main__":
    main(epochs=2000, batch=64)
