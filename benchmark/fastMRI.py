from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
import jax
import numpy as np
from heavyball.utils import set_torch
from benchmark.utils import loss_win_condition, trial

# Set up environment
set_torch()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Import AlgoPerf fastMRI workload utilities
from benchmark.algoperf.input_pipeline import load_fastmri_split  # Updated import
# from benchmark.algoperf.model import UNet  # assuming such exists

app = typer.Typer(pretty_exceptions_enable=False)

# Simple U-Net implementation (you'll need to replace this with the actual AlgoPerf UNet)
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified U-Net - replace with actual AlgoPerf implementation
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)
        
    def forward(self, x, mask=None):
        # Simplified forward pass - replace with actual implementation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()
    
    def forward(self, x, y=None, mask=None):
        # x should be the input image, returns output image
        return self.unet(x, mask)

def tf_to_torch_batch(tf_batch):
    """Convert TensorFlow batch to PyTorch tensors."""
    # tf_batch contains: {'inputs': ..., 'targets': ..., 'mean': ..., 'std': ..., 'volume_max': ...}
    inputs = torch.from_numpy(tf_batch['inputs'].numpy()).float()
    targets = torch.from_numpy(tf_batch['targets'].numpy()).float()
    
    # Add channel dimension if needed (fastMRI outputs are single channel)
    if len(inputs.shape) == 3:  # [batch, height, width]
        inputs = inputs.unsqueeze(1)   # [batch, 1, height, width]
    if len(targets.shape) == 3:
        targets = targets.unsqueeze(1)
    
    return inputs, targets

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
    
    # Load fastMRI dataset using the actual function
    data_dir = Path(__file__).parent / "data" / "fastmri"
    
    # Create JAX RNG for shuffling
    shuffle_rng = jax.random.PRNGKey(42)
    
    # Load the dataset using the actual function from the FastMRI pipeline
    dataset_iterator = load_fastmri_split(
        global_batch_size=batch,
        split="train",
        data_dir=str(data_dir),
        shuffle_rng=shuffle_rng,
        num_batches=None,  # None for infinite training
        repeat_final_eval_dataset=False
    )
    
    print(f"Created FastMRI dataset iterator with batch size {batch}")
    
    def data():
        """Get next batch from the dataset."""
        try:
            tf_batch = next(dataset_iterator)
            inputs, targets = tf_to_torch_batch(tf_batch)
            return inputs.cuda(), targets.cuda(), None  # No mask needed for this simple case
        except StopIteration:
            # This shouldn't happen with infinite training dataset, but just in case
            print("Dataset exhausted - this shouldn't happen with training split")
            raise
    
    def loss_fn(output, target):
        # AlgoPerf uses L1 loss for training
        return F.l1_loss(output, target)
    
    trial(
        model,
        data,
        loss_fn,
        loss_win_condition(win_mult * 0.723653),  # fastMRI validation target SSIM
        steps,
        opt[0],
        dtype[0],
        0,  # hidden features unused
        batch,
        weight_decay,
        "none",  # no eigenvector method
        0,  # sequence length N/A
        0,  # extra param
        failure_threshold=10,
        base_lr=1e-3,
        trials=trials,
        estimate_condition_number=False,
        test_loader=None,
        track_variance=False
    )

if __name__ == "__main__":
    app()