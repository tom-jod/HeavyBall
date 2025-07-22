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
from benchmark.transolver_layers.transolver import TransolverReg
app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
app = typer.Typer()

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
                    
                    # Extract surface data - note the keys use dots, not groups
                    verts = None
                    normals = None
                    pressure = None
                    
                    # Get vertices (3D coordinates)
                    if 'surface.verts' in f.keys():
                        verts = f['surface.verts'][:]  # Shape: [num_vertices, 3]
                        
                    else:
                        print(f"Warning: 'surface.verts' not found in {h5_file}")
                        print(f"Available keys: {[k for k in f.keys() if 'verts' in k]}")
                        continue
                    
                    # Get vertex normals (3D normal vectors)
                    if 'surface.verts_normals' in f.keys():
                        normals = f['surface.verts_normals'][:]  # Shape: [num_vertices, 3]
                        
                    else:
                        print(f"Warning: 'surface.verts_normals' not found in {h5_file}")
                        print(f"Available keys: {[k for k in f.keys() if 'normal' in k]}")
                        continue
                    
                    # Get pressure values (scalar per vertex)
                    if 'surface.pressure' in f.keys():
                        pressure = f['surface.pressure'][:]  # Shape: [num_vertices,] or [num_vertices, 1]
                        
                    else:
                        print(f"Warning: 'surface.pressure' not found in {h5_file}")
                        print(f"Available keys: {[k for k in f.keys() if 'pressure' in k]}")
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
                        print(f"  verts: {verts.shape}")
                        print(f"  normals: {normals.shape}")
                        print(f"  pressure: {pressure.shape}")
                        continue
                    
                    # Concatenate vertices and normals
                    features = np.concatenate([verts, normals], axis=1)  # Shape: [num_vertices, 6]
                    
                    self.data.append(features)
                    self.targets.append(pressure.squeeze(-1))  # Remove last dimension for pressure
                        
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
class TransolverWrapper(nn.Module):
    """Wrapper for Transolver to handle CFD surface data."""
    def __init__(self,
                 feature_dim: int = 6,  # 3 for verts + 3 for normals
                 n_layers: int = 8,
                 n_hidden: int = 256,
                 dropout: float = 0.0,
                 n_head: int = 8,
                 act: str = "gelu",
                 mlp_ratio: float = 2,
                 function_dim: int = 0,  # Set to 0 since we don't have function features
                 out_dim: int = 1,  # Pressure is scalar
                 slice_num: int = 32,
                 ref: int = 8):
        super().__init__()
        # Use the actual TransolverReg
        self.transolver = TransolverReg(
            feature_dim=feature_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            function_dim=function_dim,
            out_dim=out_dim,
            slice_num=slice_num,
            ref=ref
        )

    def forward(self, x):
        """
        Forward pass for CFD surface data.
        Args:
            x: Input tensor of shape [batch_size, num_vertices, feature_dim]
        Returns:
            output: Pressure predictions of shape [batch_size, num_vertices]
        """
        batch_size, num_vertices, feature_dim = x.shape
    
        outputs = []
        for i in range(batch_size):
            # Process each sample individually since TransolverReg expects single samples
            single_input = x[i:i+1]  # [1, feature_dim, num_vertices]
            single_output = self.transolver(single_input)  # Should return [num_vertices, out_dim] or [num_vertices]
            
            if single_output.dim() == 2 and single_output.shape[-1] == 1:
                single_output = single_output.squeeze(-1)  # [num_vertices]
            
            outputs.append(single_output)
        
        # Stack outputs back into batch
        output = torch.stack(outputs, dim=0)  # [batch_size, num_vertices]
        
        return output

@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    feature_dim: int = 6,  # 3 verts + 3 normals
    n_hidden: int = 256,
    n_layers: int = 5,
    n_head: int = 8,  
    slice_num: int = 32, 
    batch: int = 1,  
    steps: int = 10000,
    weight_decay: float = 1e-4,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    train_split: float = 0.8,
    normalize: bool = True,
):
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
        num_workers=2,  # Set to 0 for debugging
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch,  # Can use larger batch for testing
        shuffle=False,  # No need to shuffle test data
        num_workers=0,
        pin_memory=True
    )
    
    batch_data, batch_targets = next(iter(train_loader))
    
    # Create model with TransolverReg
    model = TransolverWrapper().cuda()
    model.train()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        test_output = model(batch_data.cuda())
        print(f"Test output shape: {test_output.shape}")
        print(f"Test output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
    
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
        loss_win_condition(win_condition_multiplier * 0.00),  
        steps,
        opt[0],
        dtype[0],
        n_hidden,  
        batch,
        weight_decay,
        method[0],
        feature_dim,  
        1,  
        failure_threshold=10,
        base_lr=1e-4,
        trials=trials,
        estimate_condition_number=False,
        test_loader=None,
        track_variance=True
    )

if __name__ == "__main__":
    app()