import heavyball
import itertools
from typing import List
from collections import defaultdict
import copy
import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from heavyball.utils import set_torch
from benchmark.utils import trial, get_optim

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


class ModularMLP(nn.Module):
    def __init__(self, p=23, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(p, hidden_dim),
            nn.Flatten(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, p)
        )
        
    def forward(self, x):
        return self.net(x)

class ModuloDataset(torch.utils.data.Dataset):
    def __init__(self, p, min_idx, length):
        self.p = p  
        self.n_samples = length
        self.min_idx = min_idx
        self.max_idx = min_idx + length

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        generator = torch.Generator()
        generator.manual_seed(min(idx + self.min_idx, self.max_idx))
        x1 = torch.randint(0, self.p, (), generator=generator)
        x2 = torch.randint(0, self.p, (), generator=generator)
        x = torch.stack([x1, x2])
        y = ((x1 + x2) % self.p).long()
        return x, y


def evaluate(model, loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().detach()
            total += y.size(0)
    return correct / total


def plot_results(train_losses, test_accs, steps_to_grok=None, save_path=None):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot training loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_yscale('log')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True)
    
    # Plot test accuracy
    eval_steps = np.arange(0, len(train_losses), len(train_losses) // len(test_accs))
    ax2.plot(eval_steps, test_accs, label='Test Accuracy', color='orange')
    ax2.axhline(y=0.9, color='r', linestyle='--', label='Grokking Threshold')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Steps')
    ax2.set_title('Test Accuracy Over Time')
    ax2.grid(True)
    
    if steps_to_grok is not None:
        ax2.axvline(x=steps_to_grok, color='g', linestyle='--', 
                   label=f'Grokking Step ({steps_to_grok})')
    
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


@app.command()
def main(method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
         dtype: List[str] = typer.Option(["float32"], help='Data type to use'),
         opt: List[str] = typer.Option(['AdamW', 'OrthoAdamW', 'AdamWOrtho'], help='Optimizers to use'),
         steps: int = 2000,
         batch_size: int = 32,
         hidden_dim: int = 32,
         p: int = 257,
         weight_decay: float = 0,
         lr: float = 1e-3):
    
    dtype = [getattr(torch, d) for d in dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Clean up old plots
    plot_dir = Path('.')
    for path in plot_dir.glob('grokking_*.png'):
        path.unlink()
    
    # Pre-generate datasets
    train_data = ModuloDataset(p, 0, 10000)
    test_data = ModuloDataset(p, train_data.max_idx, 1000)

    
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size * 2,
        pin_memory=True,
        num_workers=4 
    )
    
    train_iter = iter(train_loader)
    history = defaultdict(list)
    
    def data():
        """Get next batch from the dataloader"""
        nonlocal train_iter
        try:
            x, y = next(train_iter)
        except (StopIteration, NameError):
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        return x.to(device), y.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    def win_condition(model, loss_hist):
        """Check if model has achieved grokking"""
        if not isinstance(loss_hist, float):
            loss = loss_hist
        else:
            loss = loss_hist
            
        history['loss'].append(loss)
            
        if loss > 0.1:  # Not converged yet
            return False, {}
            
        # If loss is low, check test accuracy
        acc = evaluate(model, test_loader, device)
        history['test_acc'].append(acc)
        return acc > 0.9, {'test_acc': acc}
    
    global_model = ModularMLP(p=p, hidden_dim=hidden_dim).to(device)
    for d, o in itertools.product(dtype, opt):
        print(f"\nRunning {o} with {d}")
        model = copy.deepcopy(global_model)
        model.to(dtype=d)
        
        history.clear()
        
        # Get optimizer class
        optimizer_class = getattr(heavyball, o)
        optimizer = get_optim(optimizer_class, model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training loop
        for step in range(steps):
            model.train()
            x, y = data()
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                history['loss'].append(loss.detach())
                
                if step % 100 == 0:
                    acc = evaluate(model, test_loader, device).item()
                    history['test_acc'].append(acc)
                    print(f"Step {step}: Loss = {loss.item():.4f}, Test Acc = {acc:.4f}")
        
        # Plot results
        plot_name = plot_dir / f"grokking_{o}_{d}_lr{lr}_h{hidden_dim}_p{p}.png"
        plot_results(
            [i.item() for i in history['loss']],
            history['test_acc'],
            next((i for i, acc in enumerate(history['test_acc']) if acc > 0.9), None),
            plot_name
        )
        print(f"Training curves saved to {plot_name}")


if __name__ == '__main__':
    app()
