import os
import glob
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List
import torch.nn.functional as F
import typer
from heavyball.utils import set_torch
from benchmark.utils import loss_win_condition, trial

# Set up environment
set_torch()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
from benchmark.algoperf.model import UNet
app = typer.Typer(pretty_exceptions_enable=False)

# Constants from original
_TRAIN_DIR = 'singlecoil_train'
_VAL_DIR = 'singlecoil_val'
_EVAL_SEED = 42

class FastMRIDataset(Dataset):
    """PyTorch Dataset for FastMRI data that mimics the original preprocessing."""
    
    def __init__(self, h5_paths, split, shuffle_seed=None):
        self.h5_paths = h5_paths
        self.split = split
        self.shuffle_seed = shuffle_seed
        self.is_train = split == 'train'
        self.shuffle = self.is_train or split == 'eval_train'
        
        # Build index of all slices across all files
        self.slice_indices = []
        for file_idx, h5_path in enumerate(h5_paths):
            with h5py.File(h5_path, 'r') as f:
                num_slices = f['kspace'].shape[0]
                for slice_idx in range(num_slices):
                    self.slice_indices.append((file_idx, slice_idx))
    
    def __len__(self):
        return len(self.slice_indices)
    
    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_indices[idx]
        h5_path = self.h5_paths[file_idx]
        
        with h5py.File(h5_path, 'r') as f:
            kspace = f['kspace'][slice_idx]
            
            # Convert to torch tensor
            kspace = torch.from_numpy(kspace).float()
            
            # Apply the same preprocessing as original _process_example
            example_index = idx
            if self.shuffle and self.shuffle_seed is not None:
                rng = torch.Generator()
                rng.manual_seed(self.shuffle_seed + example_index)
            else:
                rng = torch.Generator()
                rng.manual_seed(_EVAL_SEED)
            
            # Apply the same processing logic as original
            processed_input, processed_target = self._process_example(kspace, rng)
            
        # Return tuple instead of dictionary
        return processed_input, processed_target
    
    def _process_example(self, kspace, rng):
        """
        Apply the same preprocessing as the original _process_example function.
        This should match the logic from the original TF/JAX implementation.
        """
        # Convert k-space to image domain
        image = torch.fft.ifft2(kspace, dim=(-2, -1))
        image = torch.abs(image)
        
        # Apply center crop (typical FastMRI preprocessing)
        # You'll need to adjust these dimensions based on your specific preprocessing
        target_height, target_width = 320, 320  # Adjust as needed
        
        # Center crop
        h, w = image.shape[-2:]
        top = (h - target_height) // 2
        left = (w - target_width) // 2
        
        if top >= 0 and left >= 0:
            image = image[..., top:top+target_height, left:left+target_width]
        else:
            # Pad if image is smaller than target
            pad_h = max(0, target_height - h)
            pad_w = max(0, target_width - w)
            image = F.pad(image, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2))
        
        # Normalize (you may need to adjust this based on your specific normalization)
        image = image / torch.max(image)
        
        # For supervised learning, target is the fully sampled image
        target = image.clone()
        
        # Apply undersampling mask to create input (you'll need to implement your specific mask)
        # This is a simplified example - replace with your actual undersampling logic
        mask = self._create_mask(image.shape, rng)
        
        # Apply mask in k-space
        kspace_masked = torch.fft.fft2(image, dim=(-2, -1)) * mask
        input_image = torch.abs(torch.fft.ifft2(kspace_masked, dim=(-2, -1)))
        
        return input_image, target
    
    def _create_mask(self, shape, rng):
        """
        Create undersampling mask. Replace this with your specific mask logic.
        This is a simplified example.
        """
        # Simple random mask - replace with your actual mask generation
        mask = torch.ones(shape[-2:])
        # Randomly zero out some k-space lines
        num_lines_to_keep = int(0.3 * shape[-2])  # Keep 30% of lines
        indices = torch.randperm(shape[-2], generator=rng)[:num_lines_to_keep]
        mask_1d = torch.zeros(shape[-2])
        mask_1d[indices] = 1
        mask = mask_1d.unsqueeze(-1).expand(shape[-2:])
        return mask

def load_fastmri_split_pytorch(global_batch_size, split, data_dir, shuffle_seed=None, num_batches=None):
    """
    PyTorch version of load_fastmri_split that returns a DataLoader.
    """
    if split not in ['train', 'eval_train', 'validation', 'test']:
        raise ValueError('Unrecognized split {}'.format(split))
    
    # Check if data directories exist
    if not os.path.exists(os.path.join(data_dir, _TRAIN_DIR)):
        raise NotADirectoryError('Directory not found: {}'.format(
            os.path.join(data_dir, _TRAIN_DIR)))
    if not os.path.exists(os.path.join(data_dir, _VAL_DIR)):
        raise NotADirectoryError('Directory not found: {}'.format(
            os.path.join(data_dir, _VAL_DIR)))
    
    # Get file paths
    if split in ['train', 'eval_train']:
        file_pattern = os.path.join(data_dir, _TRAIN_DIR, '*.h5')
        h5_paths = glob.glob(file_pattern)
    elif split == 'validation':
        file_pattern = os.path.join(data_dir, _VAL_DIR, '*.h5')
        h5_paths = sorted(glob.glob(file_pattern))[:100]
    elif split == 'test':
        file_pattern = os.path.join(data_dir, _VAL_DIR, '*.h5')
        h5_paths = sorted(glob.glob(file_pattern))[100:]
    
    # Create dataset
    dataset = FastMRIDataset(h5_paths, split, shuffle_seed)
    
    # Create DataLoader
    is_train = split == 'train'
    shuffle = is_train or split == 'eval_train'
    
    dataloader = DataLoader(
        dataset,
        batch_size=global_batch_size,
        shuffle=shuffle,
        num_workers=4,  # Adjust as needed
        pin_memory=True,
        drop_last=is_train
    )
    
    return dataloader

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
    
    # Data directory
    data_dir = '/mnt/storage01/home/tomjodrell/'
    print("START")
    # Load datasets using PyTorch DataLoader
    trainloader = load_fastmri_split_pytorch(
        global_batch_size=batch,
        split="train",
        data_dir=data_dir,
        shuffle_seed=42,
        num_batches=None
    )
    
    test_dataloader = load_fastmri_split_pytorch(
        global_batch_size=batch,
        split="test", 
        data_dir=data_dir,
        shuffle_seed=42,
        num_batches=batch
    )
    l=[]
    for i in range(1000):
        l.append(i)
        print(i)
    print(f"Created FastMRI dataset with batch size {batch}")
    
    # Create data iterator that matches the expected format
    train_iter = iter(trainloader)
    
    def data():
        nonlocal train_iter
        try:
            inputs, targets = next(train_iter)  # Clean tuple unpacking!
        except StopIteration:
            train_iter = iter(trainloader)
            inputs, targets = next(train_iter)
        
        # Add channel dimension if needed
        if inputs.ndim == 3:  # [batch, height, width]
            inputs = inputs.unsqueeze(1)   # [batch, 1, height, width]
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
            
        return inputs, targets
    
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
        128,  # sequence parameter
        1,    # some other parameter
        failure_threshold=10,
        base_lr=1e-3,
        trials=trials,
        estimate_condition_number=estimate_condition_number,
        test_loader=test_dataloader,
        track_variance=track_variance,
        runtime_limit=runtime_limit,
        step_hint=step_hint,
        group=10
    )

if __name__ == "__main__":
    app()