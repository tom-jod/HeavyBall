from pathlib import Path
from typing import List
import h5py
import numpy as np
import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

# Import Transolver components
from benchmark.transolver_layers.transolver import TransolverReg

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
import torch._dynamo
torch._dynamo.config.suppress_errors = True

torch._dynamo.config.disable = True

class CarCFDDataset(Dataset):
    """Dataset for car CFD data from h5 files."""
    
    def __init__(self, dataset_path: str, normalize: bool = True):
        self.dataset_path = Path(dataset_path)
        self.normalize = normalize
        self.data = []  # Will store concatenated verts + normals
        self.targets = []  # Will store pressure values
        
        # Load all h5 files
        self._load_data()
        
        if len(self.data) == 0:
            raise ValueError("No data was loaded! Check your h5 files and data structure.")
        
        # Convert to numpy arrays first for normalization
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        
        # Normalize if requested
        if self.normalize:
            self._normalize_data()
        
        # Convert to tensors
        self.data = torch.FloatTensor(self.data)
        self.targets = torch.FloatTensor(self.targets)
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Data shape: {self.data.shape}")  # Should be [N, num_vertices, 6]
        print(f"Target shape: {self.targets.shape}")  # Should be [N, num_vertices]

    def _load_data(self):
        """Load surface data from h5 files."""
        h5_files = list(self.dataset_path.glob("*.h5"))
        if not h5_files:
            raise ValueError(f"No h5 files found in {self.dataset_path}")
        
        print(f"Found {len(h5_files)} h5 files")
        
        for i, h5_file in enumerate(h5_files):
            try:
                with h5py.File(h5_file, 'r') as f:
                    # Print structure for first file
                    if i == 0:
                        print(f"H5 file structure: {list(f.keys())}")
                        if 'surface.verts' in f.keys():
                            print(f"surface.verts shape: {f['surface.verts'].shape}")
                        if 'surface.verts_normals' in f.keys():
                            print(f"surface.verts_normals shape: {f['surface.verts_normals'].shape}")
                        if 'surface.pressure' in f.keys():
                            print(f"surface.pressure shape: {f['surface.pressure'].shape}")
                    
                    # Extract surface data
                    verts = None
                    normals = None
                    pressure = None
                    
                    # Get vertices (3D coordinates)
                    if 'surface.verts' in f.keys():
                        verts = f['surface.verts'][:]  # Shape: [num_vertices, 3]
                        print(f"File {i}: Loaded verts with shape {verts.shape}")
                    else:
                        print(f"Warning: 'surface.verts' not found in {h5_file}")
                        continue
                    
                    # Get vertex normals (3D normal vectors)
                    if 'surface.verts_normals' in f.keys():
                        normals = f['surface.verts_normals'][:]  # Shape: [num_vertices, 3]
                        print(f"File {i}: Loaded normals with shape {normals.shape}")
                    else:
                        print(f"Warning: 'surface.verts_normals' not found in {h5_file}")
                        continue
                    
                    # Get pressure values (scalar per vertex)
                    if 'surface.pressure' in f.keys():
                        pressure = f['surface.pressure'][:]  # Shape: [num_vertices,] or [num_vertices, 1]
                        print(f"File {i}: Loaded pressure with shape {pressure.shape}")
                    else:
                        print(f"Warning: 'surface.pressure' not found in {h5_file}")
                        continue
                    
                    # Verify we have all required data
                    if verts is None or normals is None or pressure is None:
                        print(f"Skipping {h5_file} - missing required data")
                        continue
                    
                    # Ensure consistent shapes
                    if len(pressure.shape) == 1:
                        pressure = pressure.reshape(-1, 1)  # Make it [num_vertices, 1]
                    
                    # Verify shapes match
                    if verts.shape[0] != normals.shape[0] or verts.shape[0] != pressure.shape[0]:
                        print(f"Warning: Shape mismatch in {h5_file}:")
                        print(f" verts: {verts.shape}")
                        print(f" normals: {normals.shape}")
                        print(f" pressure: {pressure.shape}")
                        continue
                    
                    # Concatenate vertices and normals
                    features = np.concatenate([verts, normals], axis=1)  # Shape: [num_vertices, 6]
                    self.data.append(features)
                    self.targets.append(pressure.squeeze(-1))  # Remove last dimension for pressure
                    
                    if i == 0:  # Print info for first sample
                        print(f"Sample shapes - Features: {features.shape}, Pressure: {pressure.squeeze(-1).shape}")
                        print(f"Verts range: [{verts.min():.3f}, {verts.max():.3f}]")
                        print(f"Normals range: [{normals.min():.3f}, {normals.max():.3f}]")
                        print(f"Pressure range: [{pressure.min():.3f}, {pressure.max():.3f}]")
                    
                    print(f"Successfully loaded sample {len(self.data)} from {h5_file.name}")
                    
            except Exception as e:
                print(f"Error loading {h5_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Total samples loaded: {len(self.data)}")

    def _normalize_data(self):
        """Normalize the features and targets."""
        print("Normalizing data...")
        if len(self.data) == 0:
            raise ValueError("No data to normalize!")
        
        # Check if all samples have the same number of vertices
        vertex_counts = [sample.shape[0] for sample in self.data]
        if len(set(vertex_counts)) > 1:
            print(f"Warning: Different number of vertices per sample: {set(vertex_counts)}")
            print("Using per-sample normalization instead of global normalization")
            # Normalize each sample individually
            normalized_data = []
            normalized_targets = []
            
            for features, pressure in zip(self.data, self.targets):
                # Normalize features
                feature_mean = features.mean(axis=0, keepdims=True)
                feature_std = features.std(axis=0, keepdims=True) + 1e-8
                features_norm = (features - feature_mean) / feature_std
                
                # Normalize pressure
                pressure_mean = pressure.mean()
                pressure_std = pressure.std() + 1e-8
                pressure_norm = (pressure - pressure_mean) / pressure_std
                
                normalized_data.append(features_norm)
                normalized_targets.append(pressure_norm)
            
            self.data = normalized_data
            self.targets = normalized_targets
        else:
            # All samples have same number of vertices - use global normalization
            n_samples = len(self.data)
            n_vertices = self.data[0].shape[0]
            
            # Reshape to [total_vertices, features]
            data_flat = np.concatenate(self.data, axis=0)  # [n_samples * n_vertices, 6]
            targets_flat = np.concatenate(self.targets, axis=0)  # [n_samples * n_vertices]
            
            # Normalize features (vertices + normals)
            self.feature_scaler = StandardScaler()
            data_flat_norm = self.feature_scaler.fit_transform(data_flat)
            
            # Normalize targets (pressure)
            self.target_scaler = StandardScaler()
            targets_flat_norm = self.target_scaler.fit_transform(targets_flat.reshape(-1, 1)).squeeze(-1)
            
            # Reshape back to original structure
            self.data = data_flat_norm.reshape(n_samples, n_vertices, 6)
            self.targets = targets_flat_norm.reshape(n_samples, n_vertices)
        
        print("Normalization complete")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]  # [num_vertices, 6]
        pressure = self.targets[idx]  # [num_vertices]
        return features, pressure
class DebugTransolverReg(nn.Module):
    """Debug version of TransolverReg to understand input processing."""
    
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        
        # Replicate the original forward logic with debug prints
        fx = None
        x = x[None, :, :]  # Add batch dimension
        print(f"After adding batch dim: {x.shape}")
        
        if len(x.shape) == 4:
            x = x.squeeze(1)
            print(f"After squeeze: {x.shape}")
        
        # The issue is likely here - what shape does the preprocess expect?
        print(f"Shape going into preprocess: {x.shape}")
        print(f"Preprocess expects input_dim: {self.original_model.preprocess.linear_pre[0].in_features}")
        
        # Let's see what happens if we reshape x properly
        # The MLP likely expects [N, features] format
        if len(x.shape) == 3:  # [batch, features, N]
            x = x.transpose(1, 2)  # [batch, N, features]
            print(f"After transpose for MLP: {x.shape}")
        
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.original_model.preprocess(fx)
        else:
            fx = self.original_model.preprocess(x)
            
        print(f"After preprocess: {fx.shape}")
        
        fx = fx + self.original_model.placeholder[None, None, :]
        print(f"After adding placeholder: {fx.shape}")
        
        for i, block in enumerate(self.original_model.blocks):
            fx = block(fx)
            print(f"After block {i}: {fx.shape}")
        
        result = fx[0]
        print(f"Final output: {result.shape}")
        return result

class TransolverCFDWrapper(nn.Module):
    """Wrapper for Transolver to handle CFD surface data with proper input formatting."""
    
    def __init__(self,
                 feature_dim: int = 6,  # 3 for verts + 3 for normals
                 n_layers: int = 5,
                 n_hidden: int = 256,
                 dropout: float = 0.1,
                 n_head: int = 8,
                 act: str = "gelu",
                 mlp_ratio: float = 4,
                 function_dim: int = 0,  # Set to 0 since we don't have function features
                 out_dim: int = 1,  # Pressure is scalar
                 slice_num: int = 32,
                 ref: int = 8):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_hidden = n_hidden
        
        # Create the actual TransolverReg model
        self.transolver = TransolverReg(
            feature_dim=feature_dim,  # 6
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            function_dim=function_dim,  # 0 - no auxiliary features
            out_dim=out_dim,
            slice_num=slice_num,
            ref=ref
        )
        
    def forward(self, x):
        """
        Forward pass for CFD data.
        Args:
            x: Input tensor of shape [batch_size, num_vertices, 6]
        Returns:
            Output tensor of shape [batch_size, num_vertices] for pressure prediction
        """
        batch_size, num_vertices, feature_dim = x.shape
        
        # Process each sample in the batch
        outputs = []
        for i in range(batch_size):
            # Get single sample: [num_vertices, 6]
            sample = x[i]  # [3586, 6]
            
            # The TransolverReg has a bug in its forward method
            # It expects the input to be processed correctly by the MLP
            # Let's manually implement the correct forward pass
            
            # Step 1: Add batch dimension as the original code does
            sample_batched = sample[None, :, :]  # [1, 3586, 6]
            
            # Step 2: Process through the preprocessing MLP
            # The MLP expects [batch, N, features] format
            fx = self.transolver.preprocess(sample_batched)  # [1, 3586, n_hidden]
            
            # Step 3: Add placeholder
            fx = fx + self.transolver.placeholder[None, None, :]  # [1, 3586, n_hidden]
            
            # Step 4: Process through transformer blocks
            for block in self.transolver.blocks:
                fx = block(fx)  # [1, 3586, n_hidden] or [1, 3586, out_dim] for last block
            
            # Step 5: Remove batch dimension
            output = fx[0]  # [3586, out_dim] or [3586, n_hidden]
            
            # Handle output shape - squeeze if needed
            if output.dim() > 1 and output.shape[-1] == 1:
                output = output.squeeze(-1)  # [3586]
            
            outputs.append(output)
        
        # Stack outputs back to batch format
        return torch.stack(outputs, dim=0)  # [batch_size, num_vertices]
    
@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    feature_dim: int = 6,  # 3 verts + 3 normals
    n_hidden: int = 256,
    n_layers: int = 5,
    n_head: int = 8,
    mlp_ratio: float = 4,
    slice_num: int = 32,
    batch: int = 4,  # Very small batch for debugging
    steps: int = 100,
    weight_decay: float = 1e-4,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    train_split: float = 0.8,
    normalize: bool = True,
    dropout: float = 0.1,
    act: str = "gelu",
):
    """
    Car CFD benchmark using Transolver architecture.
    Compatible with Heavyball repository format.
    """
    dtype = [getattr(torch, d) for d in dtype]
    
    # Dataset path
    dataset_path = "/udl/dinozaur/car_cfd_lethe/02_processed_data/"
    
    # Load dataset
    print("Loading car CFD surface dataset...")
    full_dataset = CarCFDDataset(dataset_path, normalize=normalize)
    
    # Split into train and test
    train_size = int(train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=True
    )
    
    # Test data loading
    print("Testing data loading...")
    batch_data, batch_targets = next(iter(train_loader))
    print(f"Batch data shape: {batch_data.shape}")
    print(f"Batch targets shape: {batch_targets.shape}")
    
    model = TransolverCFDWrapper(
    feature_dim=feature_dim,
    n_layers=n_layers,
    n_hidden=n_hidden,
    dropout=dropout,
    n_head=n_head,
    act=act,
    mlp_ratio=mlp_ratio,
    out_dim=1,
    slice_num=slice_num,
    ref=8
).cuda()
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        test_output = model(batch_data.cuda())
        print(f"Test output shape: {test_output.shape}")
    
    # Create data iterator that matches heavyball format
    data_iter = iter(train_loader)
    
    def data():
        nonlocal data_iter
        try:
            batch_data, batch_targets = next(data_iter)
        except StopIteration:
            # Reset iterator when exhausted
            data_iter = iter(train_loader)
            batch_data, batch_targets = next(data_iter)
        return batch_data.cuda(), batch_targets.cuda()
    
    # Loss function for pressure regression
    def loss_fn(output, target):
        return F.mse_loss(output, target)
    
    # Run trial
    trial(
        model,
        data,
        loss_fn,
        loss_win_condition(win_condition_multiplier * 0.01),  # Adjust threshold for regression
        steps,
        opt[0],
        dtype[0],
        n_hidden,  # features parameter
        batch,
        weight_decay,
        method[0],
        feature_dim,  # sequence parameter
        n_layers,  # some other parameter (using n_layers instead of hardcoded 1)
        failure_threshold=10,
        base_lr=1e-4,  # Lower learning rate for regression
        trials=trials,
        estimate_condition_number=True
    )


if __name__ == "__main__":
    app()