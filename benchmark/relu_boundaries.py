import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import imageio
import heavyball
import typer
from typing import List, Optional, Literal
from pathlib import Path
import subprocess
from benchmark.utils import get_optim
from torch.utils.data import TensorDataset, DataLoader
from heavyball.utils import set_torch
from abc import ABC, abstractmethod

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

class BaseDataset(torch.utils.data.Dataset, ABC):
    classes = 2
    def __init__(self, n_samples: int, batch_size: int, seed: int = 42):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.seed = seed
        self.X, self.y = self._generate_data()
        self.indices = torch.arange(len(self.X))
    
    @abstractmethod
    def _generate_data(self):
        """Generate dataset-specific data"""
        pass
    
    def __len__(self):
        return self.n_samples // self.batch_size
    
    def __getitem__(self, idx):
        batch_idx = torch.randperm(len(self.indices))[:self.batch_size]
        return self.X[batch_idx], self.y[batch_idx]
    
    def get_full_data(self):
        """Return all data for plotting"""
        return self.X, self.y

class CircleDataset(BaseDataset):
    def _generate_data(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        # Generate points in polar coordinates
        r1 = torch.normal(mean=2.0, std=0.2, size=(self.n_samples//2,), generator=generator)
        theta1 = torch.rand(self.n_samples//2, generator=generator) * 2 * np.pi
        
        r2 = torch.normal(mean=4.0, std=0.2, size=(self.n_samples//2,), generator=generator)
        theta2 = torch.rand(self.n_samples//2, generator=generator) * 2 * np.pi
        
        # Convert to Cartesian coordinates
        x1 = torch.stack([r1 * torch.cos(theta1), r1 * torch.sin(theta1)], dim=1)
        x2 = torch.stack([r2 * torch.cos(theta2), r2 * torch.sin(theta2)], dim=1)
        
        # Combine data
        X = torch.cat([x1, x2], dim=0).float()
        y = torch.cat([torch.zeros(self.n_samples//2), 
                      torch.ones(self.n_samples//2)]).reshape(-1).long()
     
        return X, y

class ModularAdditionDataset(BaseDataset):
    def __init__(self, n_samples: int, batch_size: int, modulo: int = 11, seed: int = 42):
        self.modulo = modulo
        super().__init__(n_samples, batch_size, seed)
        self.classes = modulo
    
    def _generate_data(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        # Generate random pairs of numbers
        x1 = torch.randint(0, self.modulo, (self.n_samples,), generator=generator).float()
        x2 = torch.randint(0, self.modulo, (self.n_samples,), generator=generator).float()
        
        # Normalize to [0, 1]
        X = torch.stack([x1, x2], dim=1) / (self.modulo - 1)
        
        # Compute modular addition
        y = ((x1 + x2) % self.modulo).long().reshape(-1)
        
        return X, y

class XORDataset(BaseDataset):
    def _generate_data(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        # Generate random binary inputs
        x1 = torch.randint(0, 2, (self.n_samples,), generator=generator).float()
        x2 = torch.randint(0, 2, (self.n_samples,), generator=generator).float()
        
        # Add some noise for visualization
        noise = torch.normal(0, 0.1, (self.n_samples, 2), generator=generator)
        X = torch.stack([x1, x2], dim=1) + noise
        
        # Compute XOR
        y = (x1 != x2).reshape(-1).long()
        
        return X, y

class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=32, classes=2):
        super().__init__()
        # Keep layers separate to access pre-activation values
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, classes)
    
    def forward(self, x):
        x1 = self.fc1(x)
        r1 = torch.relu(x1)
        x2 = self.fc2(r1)
        r2 = torch.relu(x2)
        out = self.fc3(r2)
        return out
    
    def get_boundaries(self, x):
        """Get pre-activation values for ReLU boundaries"""
        x1 = self.fc1(x)  # First layer pre-activations
        r1 = torch.relu(x1)
        x2 = self.fc2(r1)  # Second layer pre-activations
        return x1, x2

def plot_decision_boundary(model, loader, ax, resolution, device='cuda'):
    """Plot ReLU decision boundaries"""
    model.eval()
    
    # Get full dataset for plotting
    X, y = loader.dataset.get_full_data()
    
    # Determine bounds with margin
    margin = 1.0
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    
    # Create grid points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                        np.linspace(y_min, y_max, resolution))
    
    # Prepare grid points as input
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    
    # Get ReLU boundaries in batches
    batch_size = 10000
    boundaries1 = []
    boundaries2 = []
    
    with torch.no_grad():
        for i in range(0, len(grid), batch_size):
            batch = grid[i:i+batch_size]
            b1, b2 = model.get_boundaries(batch)
            boundaries1.append(b1.cpu())
            boundaries2.append(b2.cpu())
    
    boundaries1 = torch.cat(boundaries1, dim=0).numpy()
    boundaries2 = torch.cat(boundaries2, dim=0).numpy()
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.squeeze(), 
                        cmap=plt.cm.RdYlBu, alpha=0.6, label='Training Data')
    
    # Plot ReLU boundaries
    # First layer (blue)
    for i in range(boundaries1.shape[1]):
        values = boundaries1[:, i].reshape(xx.shape)
        ax.contour(xx, yy, values, levels=[0], colors=['#0066CC'], 
                  alpha=0.4, linewidths=1.0,
                  label='Layer 1 ReLU' if i == 0 else None)
    
    # Second layer (red)
    for i in range(boundaries2.shape[1]):
        values = boundaries2[:, i].reshape(xx.shape)
        ax.contour(xx, yy, values, levels=[0], colors=['#CC0000'], 
                  alpha=0.4, linewidths=1.0,
                  label='Layer 2 ReLU' if i == 0 else None)
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.legend(loc='upper right')
    
    # Add explanatory text
    ax.text(0.02, 0.98, 'Blue: Layer 1 ReLU boundaries\nRed: Layer 2 ReLU boundaries', 
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

@app.command()
def main(
    hidden_size: int = typer.Option(32, help='Number of neurons in hidden layers'),
    n_samples: int = typer.Option(2 ** 14, help='Number of training samples'),
    learning_rate: float = typer.Option(0.001, help='Learning rate for the optimizer'),
    n_epochs: int = typer.Option(10, help='Number of training epochs'),
    batch_size: int = typer.Option(128, help='Batch size for training'),
    n_frames: int = typer.Option(10, help='Number of frames to generate'),
    output_file: str = typer.Option('relu_boundaries.mp4', help='Output filename'),
    output_format: str = typer.Option('mp4', help='Output format: mp4 or gif'),
    fps: int = typer.Option(10, help='Frames per second in output video'),
    seed: int = typer.Option(42, help='Random seed for reproducibility'),
    optimizer: str = typer.Option('PrecondScheduleForeachSOAP', help=f'Optimizer to use'),
    weight_decay: float = typer.Option(0.0, help='Weight decay for the optimizer'),
    beta1: float = typer.Option(0.9, help='Beta1 parameter for Adam-like optimizers'),
    beta2: float = typer.Option(0.999, help='Beta2 parameter for Adam-like optimizers'),
    resolution: int = typer.Option(32, help='Resolution of the decision boundary plot'),
    dataset: str = typer.Option('circle', help='Dataset to use: circle, modular, xor'),
    modulo: int = typer.Option(11, help='Modulo for modular addition dataset')
):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset based on choice
    dataset_classes = {
        'circle': lambda: CircleDataset(n_samples, batch_size, seed),
        'modular': lambda: ModularAdditionDataset(n_samples, batch_size, modulo, seed),
        'xor': lambda: XORDataset(n_samples, batch_size, seed)
    }
    
    if dataset not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(dataset_classes.keys())}")
    
    train_data = dataset_classes[dataset]()
    train_loader = DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Create model and optimizer
    model = SimpleMLP(hidden_size=hidden_size, classes=train_data.classes).to(device)
    model = torch.compile(model, mode='max-autotune-no-cudagraphs')
    
    optimizer_class = getattr(heavyball, optimizer)
    optimizer = get_optim(
        optimizer_class,
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2)
    )
    criterion = nn.CrossEntropyLoss()
    
    # Setup for animation
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 10))
    frames = []
    frame_files = []
    frame_count = 0
    
    # Calculate logarithmically spaced epochs for frame capture
    log_space = np.logspace(0, np.log10(n_epochs + 1), n_frames) - 1
    frame_epochs = np.unique(log_space.astype(int))
    print(f"Will capture frames at epochs: {frame_epochs.tolist()}")
    
    # Ensure output directory exists and is empty
    output_dir = Path('frames')
    output_dir.mkdir(exist_ok=True)
    for f in output_dir.glob('*.png'):
        f.unlink()
    
    # Training loop
    train_iter = iter(train_loader)
    for epoch in range(n_epochs):
        model.train()
        
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.squeeze(0).to(device), y.squeeze(0).to(device)
        
        # Training step
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch in frame_epochs:
            ax.clear()
            plot_decision_boundary(model, train_loader, ax, resolution, device=device)
            ax.set_title(f'Epoch {epoch}\nLoss: {loss.item():.4f}')
            plt.tight_layout()
            
            frame_file = output_dir / f'frame_{frame_count:05d}.png'
            frame_files.append(frame_file)
            plt.savefig(frame_file, dpi=100, bbox_inches='tight')
            frame_count += 1
            
            if output_format == 'gif':
                frames.append(imageio.imread(frame_file))
            
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            
        if epoch % (n_epochs // 10) == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    print(f"Generated {frame_count} frames")
    
    # Save animation
    output_path = Path(output_file)
    if output_format != output_path.suffix[1:]:
        output_path = output_path.with_suffix(f'.{output_format}')
    
    if output_format == 'mp4':
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(output_dir / 'frame_%05d.png'),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            str(output_path)
        ]
        print(f"Running FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("FFmpeg error:")
            print(result.stderr)
            raise RuntimeError("FFmpeg failed to create video")
    else:
        imageio.mimsave(output_path, frames, fps=fps)
    
    # Clean up
    plt.close('all')
    for frame_file in frame_files:
        try:
            frame_file.unlink()
        except FileNotFoundError:
            pass
    try:
        output_dir.rmdir()
    except OSError:
        pass
    
    print(f"Animation saved as {output_path}")

if __name__ == "__main__":
    app()
