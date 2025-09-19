from pathlib import Path
from typing import List
  
import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.nn import functional as F
from torchvision import datasets, transforms
  
from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch
  
app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
app = typer.Typer()
  
torch._dynamo.config.disable = True
class Model(nn.Module):
    def __init__(self, hidden_size: int = 16, num_layers: int = 12):  # Less extreme
        super().__init__()
        self.flatten = nn.Flatten()
        self.num_layers = num_layers
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(28 * 28, hidden_size, bias=True))  # Keep some bias
        
        # Hidden layers
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, 10, bias=True))
        
        self.layers = nn.ModuleList(layers)
        
        # Better initialization - not too small
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)  # Smaller gain but not tiny
    
    def forward(self, x):
        x = self.flatten(x)
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = torch.tanh(x)  # Use tanh instead of ReLU to avoid dead neurons
        
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)

@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    hidden_size: int = 8,  # Much smaller hidden size
    num_layers: int = 25,  # Very deep network
    batch: int = 64,
    steps: int = 100,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
):
    dtype = [getattr(torch, d) for d in dtype]
    model = Model(hidden_size, num_layers).cuda()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} parameters across {num_layers} layers")
    print(f"Hidden size: {hidden_size}")
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download data to a data directory relative to the script
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    
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
    
    # Custom loss function that matches the expected signature
    def loss_fn(output, target):
        return F.nll_loss(output, target)
    
    trial(
        model,
        data,
        loss_fn,
        loss_win_condition(win_condition_multiplier * 0.0),
        steps,
        opt[0],
        dtype[0],
        hidden_size, # features parameter
        batch,
        weight_decay,
        method[0],
        128, # sequence parameter (not really applicable for MNIST, but required)
        1, # some other parameter
        failure_threshold=10,
        base_lr=1e-3,
        trials=trials,
        estimate_condition_number = True
    )

if __name__ == "__main__":
    app()