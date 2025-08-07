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

class PINNModel(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        # Network to approximate u(x, y, t)
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size),  # Input: (x, y, t)
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)  # Output: u(x, y, t)
        )
        
    def forward(self, coords):
        return self.net(coords)

def generate_training_data(batch_size, device):
    """Generate training data for 2D heat equation"""
    # Simplified data generation that's more compatible with condition number estimation
    
    # Generate random coordinates in [0,1]^3
    coords = torch.rand(batch_size, 3, device=device, requires_grad=True)
    
    # Simple target function: u(x,y,t) = sin(π*x)*sin(π*y)*exp(-t)
    x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
    targets = torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.exp(-t)
    
    return coords, targets

def simplified_physics_loss(model, coords, targets):
    """Simplified physics loss that's more stable for condition number estimation"""
    # Forward pass
    pred = model(coords)
    
    # Data fitting loss
    data_loss = F.mse_loss(pred, targets)
    
    # Simple physics regularization (without second derivatives)
    # Compute first derivatives
    grad_outputs = torch.ones_like(pred)
    grads = torch.autograd.grad(
        outputs=pred,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    if grads is not None:
        # Simple physics constraint: ∂u/∂t + ∂u/∂x + ∂u/∂y ≈ 0
        u_t = grads[:, 2:3]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        
        physics_residual = u_t + 0.1 * (u_x + u_y)
        physics_loss = torch.mean(physics_residual**2)
    else:
        physics_loss = torch.tensor(0.0, device=coords.device)
    
    # Combine losses
    total_loss = data_loss + 0.1 * physics_loss
    
    return total_loss

@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    hidden_size: int = 128,
    batch: int = 1024,
    steps: int = 1000,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
):
    dtype = [getattr(torch, d) for d in dtype]
    model = PINNModel(hidden_size).cuda()
    
    def data():
        """Return data in the format expected by the framework (x, y)"""
        coords, targets = generate_training_data(batch, 'cuda')
        return coords, targets
    
    def loss_fn(model_output, target):
        """Simplified loss function"""
        return F.mse_loss(model_output, target)
    
    trial(
        model,
        data,
        loss_fn,
        loss_win_condition(win_condition_multiplier * 0.000),  # Lower target for convergence
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
        base_lr=1e-4,  # Lower learning rate for stability
        trials=trials,
        estimate_condition_number=True,
        test_loader=None,
        track_variance=True
    )

if __name__ == "__main__":
    app()