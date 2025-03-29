import math
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
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
from heavyball import chainable as C

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


class BranchedPSGD(C.BaseOpt):
    delayed: bool = False
    cached: bool = True
    exp_avg_input: bool = True
    hvp_interval = 2
    hessian_approx = True

    def __init__(
        self,
        params,
        lr=0.001,
        beta=0.9,
        weight_decay=0.0,  #
        # Kron:
        preconditioner_update_probability=None,
        max_size_triangular=2048,
        min_ndim_triangular=2,
        memory_save_mode=None,  #
        # LRA:
        rank: Optional[int] = None,
        eps: float = 1e-8,  #
        momentum_into_precond_update=True,
        warmup_steps: int = 0,
        merge_dims: bool = False,
        split: bool = False,
        store_triu_as_line: bool = True,
        foreach: bool = True,
        q_dtype="float32",
        stochastic_schedule: bool = False,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        delayed: Optional[bool] = C.use_default,
        cached: Optional[bool] = C.use_default,
        exp_avg_input: Optional[bool] = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,  #
        initial_d: float = 1e-6,
        lr_lr: float = 0.1,  # expert parameters
        precond_init_scale=None,
        precond_init_scale_scale=1,
        precond_lr=0.1,
    ):
        defaults = locals()
        defaults.pop("self")
        self.precond_schedule = (
            defaults.pop("preconditioner_update_probability") or heavyball.utils.precond_update_prob_schedule()
        )
        params = defaults.pop("params")

        if rank is None:
            heavyball.utils.warn_once(
                f"{rank=}. It will be set to log2(param_count). This requires `params` to be of type list. Currently, {type(params)=}"
            )
            params = list(params)
            defaults["rank"] = round(math.log2(sum(p.numel() for p in params)))
            heavyball.utils.warn_once(f"rank was set to {defaults['rank']}")

        delayed = C.default(delayed, self.delayed)
        exp_avg_input = C.default(exp_avg_input, self.cached)
        update_clipping = C.default(update_clipping, heavyball.utils.trust_region_clip_)

        branches = C.create_branch([[C.scale_by_psgd], [C.scale_by_psgd_lra]], mean)

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            False,  #
            *(C.exp_avg,) * exp_avg_input,
            branches,
        )


def main(epochs: int, batch: int):
    # Setup tensorboard logging
    log_dir = os.path.join("runs", f"soap_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)

    torch.manual_seed(0x12783)
    model = Autoencoder().cuda()
    optimizer = BranchedPSGD(model.parameters(), lr=1e-3, mars=True)

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
            loss.backward()
            return loss

        loss = optimizer.step(_closure)
        optimizer.zero_grad()
        losses.append(loss.detach())
        if len(losses) >= 64:
            for loss in losses:
                writer.add_scalar("Loss/train", loss.item(), step)
                step += 1
            losses.clear()
            writer.flush()


if __name__ == "__main__":
    main(epochs=10, batch=128)
