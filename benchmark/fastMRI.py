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
from benchmark.algoperf.model import UNet  # assuming such exists

app = typer.Typer(pretty_exceptions_enable=False)
"""
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
"""
def tf_to_torch_batch(tf_batch):
    """Convert TensorFlow or NumPy batch to PyTorch tensors."""
    def convert(tensor):
        # Convert TensorFlow tensor or NumPy array to torch.Tensor
        if hasattr(tensor, 'numpy') and callable(getattr(tensor, 'numpy')):  # TF tensor
            array = tensor.numpy()
        elif isinstance(tensor, np.ndarray):  # Already NumPy array
            array = tensor
        else:
            # Handle other cases (maybe already a torch tensor?)
            try:
                array = np.array(tensor)
            except:
                raise ValueError(f"Cannot convert tensor of type {type(tensor)}")
        
        return torch.from_numpy(array).float()
    
    inputs = convert(tf_batch['inputs'])
    targets = convert(tf_batch['targets'])
    print(inputs.size())
    # Add channel dimension if needed (fastMRI outputs are single channel)
    if inputs.ndim == 3:  # [batch, height, width]
        inputs = inputs.unsqueeze(1)   # [batch, 1, height, width]
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    print(f'after: {inputs.size()}')
    return inputs, targets

@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    hidden_size: int = 32,
    batch: int = 1,
    steps: int = 0,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    estimate_condition_number: bool = True,
    test_loader: bool = None,
    track_variance: bool = True,
    runtime_limit: int = 3600 * 24,
    step_hint: int = 317000
):
    dtype = [getattr(torch, d) for d in dtype]
    model = UNet().cuda().to(dtype[0])
    
    # Load fastMRI dataset using the actual function
    #data_dir = Path(__file__).parent / "data" / "fastmri"
    data_dir = '/mnt/storage01/home/tomjodrell/'
    # Create JAX RNG for shuffling
    shuffle_rng = jax.random.PRNGKey(42)
    
    # Load the dataset using the actual function from the FastMRI pipeline
    dataset_iterator = load_fastmri_split(
        global_batch_size=batch,
        split="train",
        data_dir=str(data_dir),
        shuffle_rng=shuffle_rng,
        num_batches=batch,  # None for infinite training
        repeat_final_eval_dataset=False
    )
    test_iterator = load_fastmri_split(
        global_batch_size=batch,
        split="test",
        data_dir=str(data_dir),
        shuffle_rng=shuffle_rng,
        num_batches=batch,  # None for infinite training
        repeat_final_eval_dataset=False
    )
    print(f"Created FastMRI dataset iterator with batch size {batch}")
    
    def data():
        batch = next(dataset_iterator)
        
        # Extract tensors from TensorFlow dataset structure
        if isinstance(batch, dict):
            inp = batch['inputs']
            tgt = batch.get('targets', None)
            
            # Convert to numpy if they're TF tensors
            if hasattr(inp, 'numpy'):
                inp = inp.numpy()
            if tgt is not None and hasattr(tgt, 'numpy'):
                tgt = tgt.numpy()
                
            # Convert to PyTorch tensors
            inp = torch.from_numpy(inp).float() if isinstance(inp, np.ndarray) else inp
            tgt = torch.from_numpy(tgt).float() if isinstance(tgt, np.ndarray) and tgt is not None else tgt
            
            if inp.ndim == 3:  # [batch, height, width]
                inp = inp.unsqueeze(1)   # [batch, 1, height, width]
            if tgt is not None and tgt.ndim == 3:
                tgt = tgt.unsqueeze(1)
                
            
        else:
            # Handle case where batch is not a dict
            inp = batch
            tgt = None
            if hasattr(inp, 'numpy'):
                inp = inp.numpy()
            inp = torch.from_numpy(inp).float() if isinstance(inp, np.ndarray) else inp
            
            # Add channel dimension here too
            if inp.ndim == 3:
                inp = inp.unsqueeze(1)
        
        return inp, tgt
    
    def loss_fn(output, target):
        # AlgoPerf uses L1 loss for training
        return F.l1_loss(output, target)
    
    trial(
        model,
        data,
        loss_fn,
        loss_win_condition(win_condition_multiplier * 0.0),
        steps,
        opt[0],
        dtype[0],
        hidden_size, 
        batch,
        weight_decay,
        method[0],
        128,  # sequence parameter (not really applicable for MNIST, but required)
        1,    # some other parameter
        failure_threshold=10,
        base_lr=1e-3,
        trials=trials,
        estimate_condition_number=estimate_condition_number,
        test_loader=test_iterator,
        track_variance=track_variance,
        runtime_limit=runtime_limit,
        step_hint=step_hint
    )

if __name__ == "__main__":
    app()
