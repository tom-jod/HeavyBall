from pathlib import Path
from typing import List
import requests
import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
app = typer.Typer()

torch._dynamo.config.disable = True

class TolstoiDataset(Dataset):
    def __init__(self, text, seq_length=50, train=True, train_split=0.8):
        self.seq_length = seq_length
        
        # Create character vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Convert text to indices
        self.data = [self.char_to_idx[ch] for ch in text]
        
        # Split data
        split_idx = int(len(self.data) * train_split)
        if train:
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Input sequence
        x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        # Target sequence (shifted by 1)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y

class CharRNN(nn.Module):
    def __init__(self, vocab_size=83, hidden_size=128, num_layers=2, seq_length=50):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers, 
            dropout=0.2 if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layers
        self.dropout = nn.Dropout(0.2)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Initialize hidden state
        self.hidden = None
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state if None or batch size changed
        if self.hidden is None or self.hidden[0].size(1) != batch_size:
            self.hidden = self.init_hidden(batch_size)
        
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM forward pass
        lstm_out, self.hidden = self.lstm(embedded, self.hidden)
        
        # Detach hidden state to prevent backprop through entire sequence
        self.hidden = tuple([h.detach() for h in self.hidden])
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Reshape for linear layer
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        
        # Output layer
        output = self.fc(lstm_out)
        
        # Reshape back to (batch_size, seq_length, vocab_size)
        output = output.view(batch_size, self.seq_length, self.vocab_size)
        
        return output
    
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
    
    def reset_hidden(self):
        self.hidden = None

def download_war_and_peace():
    """Download War and Peace text from Project Gutenberg"""
    url = "https://www.gutenberg.org/files/2600/2600-0.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except:
        # Fallback to a simple text if download fails
        return "This is a simple fallback text for character-level language modeling. " * 1000

@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    hidden_size: int = 128,
    batch: int = 50,
    steps: int = 1000,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    seq_length: int = 50,
):
    dtype = [getattr(torch, d) for d in dtype]
    
    # Download and prepare data
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    text_file = data_dir / "war_and_peace.txt"
    if not text_file.exists():
        print("Downloading War and Peace...")
        text = download_war_and_peace()
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    
    # Create datasets
    train_dataset = TolstoiDataset(text, seq_length=seq_length, train=True)
    test_dataset = TolstoiDataset(text, seq_length=seq_length, train=False)
    
    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    model = CharRNN(
        vocab_size=vocab_size, 
        hidden_size=hidden_size, 
        num_layers=2, 
        seq_length=seq_length
    ).cuda()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create data iterator
    data_iter = iter(train_loader)
    
    def data():
        nonlocal data_iter
        try:
            batch_data, batch_targets = next(data_iter)
        except StopIteration:
            # Reset iterator and hidden state when exhausted
            data_iter = iter(train_loader)
            model.reset_hidden()
            batch_data, batch_targets = next(data_iter)
        return batch_data.cuda(), batch_targets.cuda()
    
    # Custom loss function for sequence modeling
    def loss_fn(output, target):
        # output: (batch_size, seq_length, vocab_size)
        # target: (batch_size, seq_length)
        output = output.view(-1, vocab_size)  # (batch_size * seq_length, vocab_size)
        target = target.view(-1)  # (batch_size * seq_length,)
        return F.cross_entropy(output, target)
    
    # Calculate steps per epoch for group parameter
    steps_per_epoch = len(train_loader)
    
    trial(
        model,
        data,
        loss_fn,
        loss_win_condition(win_condition_multiplier * 2.0),  # Reasonable loss target for char-level modeling
        steps,
        opt[0],
        dtype[0],
        hidden_size,  # features parameter
        batch,
        weight_decay,
        method[0],
        seq_length,  # sequence parameter
        vocab_size,    # vocab size parameter
        failure_threshold=10,
        group=steps_per_epoch,  # set to epoch size
        base_lr=1e-3,
        trials=trials,
        estimate_condition_number=False,
        test_loader=test_loader,
    )

if __name__ == "__main__":
    app()