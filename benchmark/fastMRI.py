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
    """Preprocess one MRI slice (adapted from AlgoPerf)."""
    with tf.device('/CPU:0'):
        num_cols = kspace_shape[1]

        # --------------------------
        # 1. Create sampling mask
        # --------------------------
        center_fraction = 0.08
        acceleration = 4

        num_low_frequencies = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_frequencies) / (
            num_cols - num_low_frequencies
        )

        # Center mask
        mask = tf.zeros(num_cols, dtype=tf.float32)
        pad = (num_cols - num_low_frequencies + 1) // 2
        mask = tf.tensor_scatter_nd_update(
            mask,
            tf.reshape(tf.range(pad, pad + num_low_frequencies), (-1, 1)),
            tf.ones(num_low_frequencies),
        )
        center_mask = tf.reshape(mask, (1, num_cols))

        # Acceleration mask (stateless RNG per-seed)
        mask = tf.cast(
            tf.random.stateless_uniform((num_cols,), seed) < prob,
            dtype=tf.float32,
        )
        accel_mask = tf.reshape(mask, (1, num_cols))

        mask = tf.math.maximum(center_mask, accel_mask)
        mask = tf.cast(mask, dtype=tf.complex64)

        # --------------------------
        # 2. Apply mask in k-space
        # --------------------------
        masked_kspace = kspace * mask

        shifted = tf.signal.ifftshift(masked_kspace, axes=(0, 1))
        image = tf.signal.ifft2d(shifted)
        image = tf.signal.fftshift(image, axes=(0, 1))

        scaling_norm = tf.sqrt(
            tf.cast(tf.reduce_prod(tf.shape(kspace)[-2:]), tf.float32)
        )
        scaling_norm = tf.cast(scaling_norm, image.dtype)
        image = image * scaling_norm

        # Convert complex -> 2 channels
        image = tf.stack((tf.math.real(image), tf.math.imag(image)), axis=-1)

        # --------------------------
        # 3. Center crop to target shape
        # --------------------------
        w_from = (kspace_shape[0] - target_shape[0]) // 2
        h_from = (kspace_shape[1] - target_shape[1]) // 2
        w_to = w_from + target_shape[0]
        h_to = h_from + target_shape[1]

        image = image[w_from:w_to, h_from:h_to, :]

        # --------------------------
        # 4. Convert to magnitude image
        # --------------------------
        input_image = tf.sqrt(tf.reduce_sum(tf.square(image), axis=-1))

        # --------------------------
        # 5. Normalize input and target separately
        # --------------------------
        # Input normalization
        mean = tf.reduce_mean(input_image)
        std = tf.math.reduce_std(input_image)
        norm_input = (input_image - mean) / (std + 1e-8)
        norm_input = tf.clip_by_value(norm_input, -6, 6)

        # Target normalization (independent)
        target_mean = tf.reduce_mean(target)
        target_std = tf.math.reduce_std(target)
        norm_target = (target - target_mean) / (target_std + 1e-8)

        # --------------------------
        # Final dict
        # --------------------------
        result = {
            "inputs": norm_input,
            "targets": norm_target,
            "mean": mean,
            "std": std,
            "volume_max": volume_max,
        }

        return result

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
    del tf_batch
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
    """Iterator wrapper for FastMRI test set that preserves normalization stats."""
    def __init__(self, test_iterator):
        self.test_iterator = test_iterator
        self.count = 0
    
    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        self.count += 1
        try:
            batch_data = next(self.test_iterator)
        except StopIteration:
            raise StopIteration
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
        if isinstance(batch_data, dict):
            # Convert inputs and targets to torch tensors
            inputs, targets = tf_to_torch_batch_efficient(batch_data)

            # Keep normalization info
            mean = batch_data["mean"]
            std = batch_data["std"]
            volume_max = batch_data["volume_max"]

            # Convert scalars to torch tensors
            if not torch.is_tensor(mean):
                mean = torch.tensor(mean, dtype=torch.float32)
            if not torch.is_tensor(std):
                std = torch.tensor(std, dtype=torch.float32)
            if not torch.is_tensor(volume_max):
                volume_max = torch.tensor(volume_max, dtype=torch.float32)

            return inputs, targets, mean, std, volume_max

        else:
            # Fallback: treat inputs as numpy/tensor, assume no stats
            inputs = batch_data
            if hasattr(inputs, "numpy"):
                inputs = torch.from_numpy(inputs.numpy()).float()
            if inputs.ndim == 2:  # [H, W]
                inputs = inputs.unsqueeze(0).unsqueeze(0)
            elif inputs.ndim == 3:  # [B, H, W]
                inputs = inputs.unsqueeze(1)

            targets = inputs.clone()
            mean = torch.tensor(0.0)
            std = torch.tensor(1.0)
            volume_max = torch.tensor(1.0)

            return inputs, targets, mean, std, volume_max

    
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
    estimate_condition_number: bool = False,
    test_loader: bool = None,
    track_variance: bool = False,
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
        
        del batch_data
    
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