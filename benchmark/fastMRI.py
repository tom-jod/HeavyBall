from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from heavyball.utils import set_torch
from benchmark.utils import loss_win_condition, trial

# Set up environment
set_torch()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Import AlgoPerf fastMRI workload utilities
from algorithmic_efficiency.workloads.fastmri.input_pipeline import (
    load_dataset,
    build_input_queue
)
from algorithmic_efficiency.workloads.fastmri.model import UNet  # assuming such exists

app = typer.Typer(pretty_exceptions_enable=False)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()  # AlgoPerf U-Net architecture

    def forward(self, x, y, mask):
        # returns output image
        return self.unet(x, mask)

@app.command()
def main(
    batch: int = 32,
    steps: int = 1000,
    weight_decay: float = 0.0,
    opt: List[str] = typer.Option(["Adam"], help="Optimizer(s)"),
    dtype: List[str] = typer.Option(["float32"], help="dtype"),
    win_mult: float = 1.0,
    trials: int = 3,
):
    # Setup
    dtype = [getattr(torch, d) for d in dtype]
    model = Model().cuda().to(dtype[0])
    
    # Load fastMRI dataset
    data_dir = Path(__file__).parent / "data" / "fastmri"
    ds = load_dataset(data_dir, split="train")
    loader = build_input_queue(ds, batch_size=batch, shuffle=True)
    data_iter = iter(loader)

    def data():
        nonlocal data_iter
        try:
            x, y, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y, mask = next(data_iter)
        return x.cuda(), y.cuda(), mask.cuda()

    def loss_fn(output, target):
        # AlgoPerf uses L1 loss and SSIM targets, but L1 is used for training
        return F.l1_loss(output, target)

    trial(
        model,
        data,
        loss_fn,
        loss_win_condition(win_mult * 0.723653),  # fastMRI validation target SSIM
        steps,
        opt[0],
        dtype[0],
        0,       # hidden features unused
        batch,
        weight_decay,
        "none",  # no eigenvector method
        0,       # sequence length N/A
        0,       # extra param
        failure_threshold=10,
        base_lr=1e-3,
        trials=trials,
        estimate_condition_number=True,
    )

if __name__ == "__main__":
    app()
