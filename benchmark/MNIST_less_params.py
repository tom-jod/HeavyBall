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

class Model3Layer(nn.Module):
    """Baseline 3-layer MLP."""
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class ModelLogReg(nn.Module):
    """Much lower condition number: single linear classifier."""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
class ModelWide(nn.Module):
    """ lower condition number: single linear classifier."""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


class ModelDeepSigmoid(nn.Module):
    """Much higher condition number: deep narrow sigmoid MLP."""
    def __init__(self, hidden_size: int = 16, depth: int = 6):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = [nn.Linear(28 * 28, hidden_size), nn.Sigmoid()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Sigmoid()]
        layers += [nn.Linear(hidden_size, 10)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        return F.log_softmax(x, dim=1)

class ModelDeeperSigmoid(nn.Module):
    """Much higher condition number: deep narrow sigmoid MLP."""
    def __init__(self, hidden_size: int = 16, depth: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = [nn.Linear(28 * 28, hidden_size), nn.Sigmoid()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Sigmoid()]
        layers += [nn.Linear(hidden_size, 10)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        return F.log_softmax(x, dim=1)

def set_deterministic_weights(model, seed=42):
    """Initialize model with deterministic weights using a fixed seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Re-initialize all parameters
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Use Xavier/Glorot uniform initialization with fixed seed
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    return model

@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    hidden_size: int = 16,
    batch: int = 128,
    steps: int = 0,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    estimate_condition_number: bool = True,
    test_loader: bool = None,
    track_variance: bool = False,
    runtime_limit: int = 3600 * 24,
    step_hint: int = 317000,
    model_type: str = "deepsigmoid" # "Choose: mlp | logreg | deepsigmoid | deepersigmoid | wide")
):
    dtype = [getattr(torch, d) for d in dtype]

    # Pick model
    if model_type == "mlp":
        model = Model3Layer(hidden_size).cuda()
    elif model_type == "logreg":
        model = ModelLogReg().cuda()
    elif model_type == "deepsigmoid":
        model = ModelDeepSigmoid(hidden_size=hidden_size).cuda()
    elif model_type == "deepersigmoid":
        model = ModelDeeperSigmoid(hidden_size=8).cuda()
    elif model_type == "wide":
        model = ModelWide().cuda()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
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
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_dataset = datasets.MNIST(
    data_dir, train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True
)
    
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
    
    win_target = 1 - 0.9851

    trial(
        model,
        data,
        loss_fn,
        loss_win_condition(win_condition_multiplier * 0),
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
        test_loader=test_loader,
        train_loader=train_loader,
        track_variance=track_variance,
        runtime_limit=runtime_limit,
        step_hint=step_hint
    )

if __name__ == "__main__":
    app()

# steps per epoch: int(60000/64)