from pathlib import Path
from typing import List
import math

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.nn import functional as F

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
app = typer.Typer()

torch._dynamo.config.disable = True
class SimpleNeRFModel(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        
        # Calculate input dimensions after positional encoding
        pos_encoded_dim = 3 + 2 * 3 * 10  # 3 + 60 = 63
        dir_encoded_dim = 3 + 2 * 3 * 4   # 3 + 24 = 27
        total_input_dim = pos_encoded_dim + dir_encoded_dim  # 63 + 27 = 90
        
        # Input: 90D (after positional encoding)
        # Output: 4D (3D color + 1D density)
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_size),  # Changed from 6 to 90
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)  # 3D color + 1D density
        )
        
        # Add positional encoding as a preprocessing step
        self.register_buffer('pos_freqs', torch.pow(2.0, torch.linspace(0, 9, 10)))
        self.register_buffer('dir_freqs', torch.pow(2.0, torch.linspace(0, 3, 4)))
        
    def positional_encoding(self, x, freqs):
        """Apply positional encoding"""
        encoded = [x]
        for freq in freqs:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)
        
    def forward(self, x):
        # Split input into position and direction
        positions = x[:, :3]
        directions = x[:, 3:]
        
        # Apply positional encoding
        pos_encoded = self.positional_encoding(positions, self.pos_freqs)
        dir_encoded = self.positional_encoding(directions, self.dir_freqs)
        
        # Concatenate encoded features
        encoded_input = torch.cat([pos_encoded, dir_encoded], dim=1)
        
        # Forward pass
        output = self.net(encoded_input)
        
        # Apply activations
        colors = torch.sigmoid(output[:, :3])  # Colors in [0, 1]
        density = F.relu(output[:, 3:4])       # Density >= 0
        
        return torch.cat([colors, density], dim=1)
    
def generate_synthetic_scene(batch_size, device):
    """Generate synthetic NeRF training data"""
    # Generate random rays
    # Ray origins (camera positions on a sphere)
    theta = torch.rand(batch_size, 1, device=device) * 2 * math.pi
    phi = torch.rand(batch_size, 1, device=device) * math.pi
    radius = 2.0
    
    ray_origins = torch.cat([
        radius * torch.sin(phi) * torch.cos(theta),
        radius * torch.sin(phi) * torch.sin(theta),
        radius * torch.cos(phi)
    ], dim=1)
    
    # Ray directions (pointing towards origin with some noise)
    ray_directions = F.normalize(-ray_origins + 0.1 * torch.randn_like(ray_origins), dim=1)
    
    # Sample points along rays
    t_vals = torch.linspace(0.5, 3.5, 64, device=device).expand(batch_size, -1)
    t_vals = t_vals + torch.randn_like(t_vals) * 0.01  # Add noise
    
    # 3D positions along rays
    positions = ray_origins.unsqueeze(1) + t_vals.unsqueeze(2) * ray_directions.unsqueeze(1)
    directions = ray_directions.unsqueeze(1).expand(-1, 64, -1)
    
    # Ground truth: simple sphere at origin
    distances = torch.norm(positions, dim=2)
    sphere_radius = 0.5
    
    # Create ground truth colors and densities
    gt_density = torch.exp(-10 * F.relu(distances - sphere_radius))
    gt_colors = torch.stack([
        0.8 * torch.ones_like(gt_density),  # Red sphere
        0.2 * torch.ones_like(gt_density),
        0.2 * torch.ones_like(gt_density)
    ], dim=2)
    
    return positions, directions, gt_colors, gt_density

@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    hidden_size: int = 256,
    batch: int = 512,
    steps: int = 1000,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
):
    dtype = [getattr(torch, d) for d in dtype]
    model = SimpleNeRFModel(hidden_size).cuda()
    
    def data():
        positions, directions, gt_colors, gt_density = generate_synthetic_scene(batch, 'cuda')
        # Flatten everything for easier processing - use reshape instead of view
        batch_size, n_samples = positions.shape[:2]
        positions_flat = positions.reshape(-1, 3)  # Use reshape instead of view
        directions_flat = directions.reshape(-1, 3)  # Use reshape instead of view
        gt_colors_flat = gt_colors.reshape(-1, 3)   # Use reshape instead of view
        gt_density_flat = gt_density.reshape(-1)    # Use reshape instead of view
        
        # Combine inputs and targets
        inputs = torch.cat([positions_flat, directions_flat], dim=1)  # 6D input
        targets = torch.cat([gt_colors_flat, gt_density_flat.unsqueeze(1)], dim=1)  # 4D target
        
        return inputs, targets
    
    def loss_fn(output, target):
        # Split output into colors and density
        pred_colors = output[:, :3]
        pred_density = output[:, 3]
        
        # Split target into colors and density
        gt_colors = target[:, :3]
        gt_density = target[:, 3]
        
        # Compute loss
        color_loss = F.mse_loss(pred_colors, gt_colors)
        density_loss = F.mse_loss(pred_density, gt_density)
        
        return color_loss + 0.1 * density_loss
    
    trial(
        model,
        data,
        loss_fn,
        loss_win_condition(win_condition_multiplier * 0.01),  # Target loss
        steps,
        opt[0],
        dtype[0],
        hidden_size,
        batch * 64,  # Total number of samples (batch_size * samples_per_ray)
        weight_decay,
        method[0],
        128,
        1,
        failure_threshold=10,
        base_lr=5e-4,  # Good learning rate for NeRF
        trials=trials,
        estimate_condition_number=False,
        test_loader=None,
        track_variance=True
    )

if __name__ == "__main__":
    app()