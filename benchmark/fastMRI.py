from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
import jax
import numpy as np
import glob
import os
import h5py
import datetime
import tensorflow as tf
from heavyball.utils import set_torch
from benchmark.utils import loss_win_condition, trial

# Set up environment
set_torch()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Import AlgoPerf fastMRI workload utilities
from benchmark.algoperf.model import UNet

app = typer.Typer(pretty_exceptions_enable=False)

def _process_example(kspace, kspace_shape, target, target_shape, volume_max, seed):
    """Generate a single example (slice from mri image)."""
    
    # sample_mask
    num_cols = kspace_shape[1]
    num_cols_float = tf.cast(num_cols, dtype=tf.float32)
    
    # choose_acceleration
    center_fraction = tf.convert_to_tensor(0.08, dtype=tf.float32)
    acceleration = tf.convert_to_tensor(4.0, dtype=tf.float32)
    
    num_low_frequencies = tf.cast(
        num_cols_float * center_fraction, dtype=tf.int32)
    
    # calculate_center_mask
    mask = tf.zeros(num_cols, dtype=tf.float32)
    pad = (num_cols - num_low_frequencies + 1) // 2
    mask = tf.tensor_scatter_nd_update(
        mask,
        tf.reshape(tf.range(pad, pad + num_low_frequencies), (-1, 1)),
        tf.ones(num_low_frequencies))
    
    # reshape_mask
    center_mask = tf.reshape(mask, (1, num_cols))
    
    # calculate_acceleration_mask
    num_low_frequencies_float = tf.cast(num_low_frequencies, dtype=tf.float32)
    prob = (num_cols_float / acceleration - num_low_frequencies_float) / (
        num_cols_float - num_low_frequencies_float)
    
    mask = tf.cast(
        tf.random.stateless_uniform((num_cols,), seed) < prob, dtype=tf.float32)
    acceleration_mask = tf.reshape(mask, (1, num_cols))
    
    mask = tf.math.maximum(center_mask, acceleration_mask)
    mask = tf.cast(mask, dtype=tf.complex64)
    
    # apply_mask
    masked_kspace = kspace * mask + 0.0
    
    # ifft2c
    shifted_kspace = tf.signal.ifftshift(masked_kspace, axes=(0, 1))
    shifted_image = tf.signal.ifft2d(shifted_kspace)
    image = tf.signal.fftshift(shifted_image, axes=(0, 1))
    scaling_norm = tf.cast(
        tf.math.sqrt(
            tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), 'float32')),
        kspace.dtype)
    image = image * scaling_norm
    image = tf.stack((tf.math.real(image), tf.math.imag(image)), axis=-1)
    
    # complex_center_crop
    w_from = (kspace_shape[0] - target_shape[0]) // 2
    h_from = (kspace_shape[1] - target_shape[1]) // 2
    w_to = w_from + target_shape[0]
    h_to = h_from + target_shape[1]
    
    image = image[..., w_from:w_to, h_from:h_to, :]
    
    # complex_abs
    abs_image = tf.math.sqrt(tf.math.reduce_sum(image**2, axis=-1))
    
    # normalize_instance
    mean = tf.math.reduce_mean(abs_image)
    std = tf.math.reduce_std(abs_image)
    norm_image = (abs_image - mean) / std
    
    # clip_image
    image = tf.clip_by_value(norm_image, -6, 6)
    
    # process target
    norm_target = (target - mean) / std
    target = tf.clip_by_value(norm_target, -6, 6)
    
    return {
        'inputs': image,
        'targets': target,
        'mean': mean,
        'std': std,
        'volume_max': volume_max,
    }

def _h5_to_examples(path, log=False):
    """Yield MRI slices from an hdf5 file containing a single MRI volume."""
    if log:
        tf.print('fastmri_dataset._h5_to_examples call:',
                 path,
                 datetime.datetime.now().strftime('%H:%M:%S:%f'))
    with open(path, 'rb') as gf:
        with h5py.File(gf, 'r') as hf:
            # NOTE(dsuo): logic taken from reference code
            volume_max = hf.attrs.get('max', 0.0)
            
            for i in range(hf['kspace'].shape[0]):
                yield hf['kspace'][i], hf['kspace'][i].shape, hf['reconstruction_esc'][
                    i], hf['reconstruction_esc'][i].shape, volume_max

def _create_generator(filename):
    signature = (
        tf.TensorSpec(shape=(640, None), dtype=tf.complex64),
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
        tf.TensorSpec(shape=(320, 320), dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(
        _h5_to_examples, args=(filename,), output_signature=signature)


def load_fastmri_split_minimal_memory(global_batch_size, split, data_dir, shuffle_rng):
    """Minimal memory version - load one file at a time"""
    _TRAIN_DIR = 'singlecoil_train'
    _VAL_DIR = 'singlecoil_val'
    
    # FIX: Actually use the split parameter!
    if split in ['train', 'eval_train']:
        file_pattern = os.path.join(data_dir, _TRAIN_DIR, '*.h5')
        h5_paths = glob.glob(file_pattern)
    elif split == 'validation':
        file_pattern = os.path.join(data_dir, _VAL_DIR, '*.h5')
        h5_paths = sorted(glob.glob(file_pattern))[:100]
    elif split == 'test':
        file_pattern = os.path.join(data_dir, _VAL_DIR, '*.h5')
        h5_paths = sorted(glob.glob(file_pattern))[100:]
    else:
        raise ValueError(f"Unknown split: {split}")
    
    print(f"Loading {split} split with {len(h5_paths)} files")  # Debug info
    
    def simple_generator():
        # FIX: Don't loop infinitely for test/validation
        loop_count = 0 if split in ['test', 'validation'] else float('inf')
        
        while True:
            for h5_path in h5_paths:
                for example in _h5_to_examples(h5_path):
                    kspace, kspace_shape, target, target_shape, volume_max = example
                    
                    # FIX: Actually process the data properly
                    # This is a simplified version - you should implement proper masking
                    processed_example = _process_example(
                        kspace, kspace_shape, target, target_shape, 
                        volume_max, shuffle_rng
                    )
                    yield processed_example
            
            if loop_count != float('inf'):
                break
            loop_count -= 1
    
    return simple_generator()


def tf_to_torch_batch_efficient(tf_batch):
    """Memory-efficient conversion from TensorFlow to PyTorch tensors."""
    def convert(tensor):
        if hasattr(tensor, 'numpy') and callable(getattr(tensor, 'numpy')):
            array = tensor.numpy()
            del tensor
        elif isinstance(tensor, np.ndarray):
            array = tensor
        else:
            try:
                array = np.array(tensor)
            except:
                raise ValueError(f"Cannot convert tensor of type {type(tensor)}")
        
        torch_tensor = torch.from_numpy(array).float()
        return torch_tensor
    
    inputs = convert(tf_batch['inputs'])
    targets = convert(tf_batch['targets'])
    
    # Ensure proper dimensions: [batch, channel, height, width]
    # If inputs is 2D [H, W], make it [1, 1, H, W]
    if inputs.ndim == 2:
        inputs = inputs.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif inputs.ndim == 3:
        inputs = inputs.unsqueeze(1)  # Add channel dim, assuming first dim is batch
    
    # Same for targets
    if targets.ndim == 2:
        targets = targets.unsqueeze(0).unsqueeze(0)
    elif targets.ndim == 3:
        targets = targets.unsqueeze(1)
    
    return inputs, targets

class FastMRITestDataset:
    """Wrapper to make the test iterator compatible with the evaluation function - DEBUG VERSION"""
    def __init__(self, test_iterator):
        self.test_iterator = test_iterator
        self.count = 0
        print(f"DEBUG: FastMRITestDataset initialized with iterator type: {type(test_iterator)}")
    
    def __iter__(self):
        print("DEBUG: FastMRITestDataset.__iter__ called")
        self.count = 0
        return self
    
    def __next__(self):
        self.count += 1
        print(f"DEBUG: FastMRITestDataset.__next__ called (batch #{self.count})")
        
        try:
            batch_data = next(self.test_iterator)
            print(f"DEBUG: Got batch_data type: {type(batch_data)}")
            
            if isinstance(batch_data, dict):
                print(f"DEBUG: batch_data keys: {batch_data.keys()}")
            
        except StopIteration:
            print("DEBUG: StopIteration in FastMRITestDataset")
            raise StopIteration
        except Exception as e:
            print(f"DEBUG: Error getting batch_data: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Convert efficiently
        try:
            if isinstance(batch_data, dict):
                inputs, targets = tf_to_torch_batch_efficient(batch_data)
                print(f"DEBUG: Converted dict - inputs: {inputs.shape}, targets: {targets.shape}")
            else:
                # Handle non-dict case
                inputs = batch_data
                if hasattr(inputs, 'numpy'):
                    inputs = torch.from_numpy(inputs.numpy()).float()
                
                # Fix dimension handling here too
                if inputs.ndim == 2:  # [H, W]
                    inputs = inputs.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                elif inputs.ndim == 3:  # [B, H, W] or [H, W, C]
                    if inputs.shape[-1] == 1:  # [H, W, 1] -> [1, 1, H, W]
                        inputs = inputs.squeeze(-1).unsqueeze(0).unsqueeze(0)
                    else:  # [B, H, W] -> [B, 1, H, W]
                        inputs = inputs.unsqueeze(1)
                
                targets = inputs.clone()  # For unsupervised case
                print(f"DEBUG: Converted non-dict - inputs: {inputs.shape}")
            
            print(f"DEBUG: Returning batch - inputs: {inputs.shape}, targets: {targets.shape}")
            return inputs, targets
            
        except Exception as e:
            print(f"DEBUG: Error in data conversion: {e}")
            import traceback
            traceback.print_exc()
            raise
    
@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    hidden_size: int = 32,
    batch: int = 1,  # Keep batch size small for memory
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
    
    data_dir = '/mnt/storage01/home/tomjodrell/'
    shuffle_rng = jax.random.PRNGKey(42)
    
    # Use memory-efficient dataset loader
    dataset_iterator = load_fastmri_split_minimal_memory(
        global_batch_size=batch,
        split="train",
        data_dir=str(data_dir),
        shuffle_rng=shuffle_rng,
       
    )
    
    test_iterator = load_fastmri_split_minimal_memory(
        global_batch_size=batch,
        split="test",
        data_dir=str(data_dir),
        shuffle_rng=shuffle_rng,
       
    )
    test_loader = FastMRITestDataset(test_iterator)
    print(f"Created FastMRI dataset iterator with batch size {batch}")
    
    def data():
        nonlocal dataset_iterator
        try:
            batch_data = next(dataset_iterator)
        except StopIteration:
            # Reset iterator if needed
            dataset_iterator = load_fastmri_split_minimal_memory(
                global_batch_size=batch,
                split="train",
                data_dir=str(data_dir),
                shuffle_rng=shuffle_rng,
            )
            batch_data = next(dataset_iterator)
        
        # Convert efficiently
        if isinstance(batch_data, dict):
            inputs, targets = tf_to_torch_batch_efficient(batch_data)
        else:
            # Handle non-dict case
            inputs = batch_data
            if hasattr(inputs, 'numpy'):
                inputs = torch.from_numpy(inputs.numpy()).float()
            
            # Fix dimension handling
            if inputs.ndim == 2:  # [H, W]
                inputs = inputs.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif inputs.ndim == 3:  # [B, H, W]
                inputs = inputs.unsqueeze(1)  # [B, 1, H, W]
            
            targets = inputs.clone()
        
        # Move to GPU
        inputs = inputs.cuda()
        targets = targets.cuda() if targets is not None else None
        
        return inputs, targets
    
    def loss_fn(output, target):
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
        128,
        1,
        failure_threshold=10,
        base_lr=1e-3,
        trials=trials,
        estimate_condition_number=estimate_condition_number,
        test_loader=test_loader,
        track_variance=track_variance,
        runtime_limit=runtime_limit,
        step_hint=step_hint,
    )
    
if __name__ == "__main__":
    app()