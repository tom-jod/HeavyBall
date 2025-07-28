from pathlib import Path
from typing import List
import math

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.nn import functional as F
from torchvision import datasets, transforms

from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Copy the NAdamW optimizer and scheduler classes from your template
class NAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        defaults = {
            'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('NAdamW does not support sparse gradients')
                grads.append(p.grad)
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                state_steps.append(state['step'])
            
            nadamw(params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps,
                   beta1=beta1, beta2=beta2, lr=group['lr'], 
                   weight_decay=group['weight_decay'], eps=group['eps'])
        return loss

def nadamw(params, grads, exp_avgs, exp_avg_sqs, state_steps, 
           beta1, beta2, lr, weight_decay, eps):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        
        step_t += 1
        param.mul_(1 - lr * weight_decay)
        
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # NAdam modification
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        step = step_t.item()
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        param.addcdiv_(exp_avg, denom, value=-step_size)
        exp_avg.sub_(grad, alpha=1 - beta1).div_(beta1)

class WarmCosine(object):
    def __init__(self, optimizer, lr_min, lr_max, warmup_steps, T):
        self.optimizer = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.T = T
        self.t = 0

    def schedule(self, t):
        if t <= self.warmup_steps:
            return self.lr_min + (self.lr_max - self.lr_min) / self.warmup_steps * t
        elif t <= self.T:
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + math.cos((t - self.warmup_steps) * math.pi / (self.T - self.warmup_steps))
            )
        return self.lr_min

    def step(self):
        self.t += 1
        lr = self.schedule(self.t)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

class WarmCosineCycles(object):
    def __init__(self, optimizer, lr_min, lr_max, warmup_steps, T, cycles):
        self.optimizer = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.T = T
        self.cycles = cycles
        self.alpha = 1 / cycles
        self.t = 0

    def f(self, t, phi):
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + math.cos((t - self.warmup_steps - phi * self.alpha * self.T) * math.pi / 
                        (self.alpha * self.T - self.warmup_steps))
        )

    def warmup(self, t, phi):
        return self.lr_min + (self.lr_max - self.lr_min) / self.warmup_steps * (t - phi * self.alpha * self.T)

    def schedule(self, t):
        for phi in range(0, self.cycles):
            if t <= phi * self.alpha * self.T + self.warmup_steps:
                return self.warmup(t, phi)
            elif t <= (phi + 1) * self.alpha * self.T:
                return self.f(t, phi)
        return self.lr_min

    def step(self):
        self.t += 1
        lr = self.schedule(self.t)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

class Model(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        #x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


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
    hidden_size: int = 128,
    batch: int = 64,
    steps: int = 100,
    weight_decay: float = 0.0,
    lr: float = 0.001,
    warmup_factor: float = 0.05,
    cycles: int = 1,
    eval_every: int = 10,
    optimizer_type: str = "adamw",  # adamw, nadamw, sgd, shampoo
    # Shampoo-specific parameters
    max_preconditioner_dim: int = 1024,
    precondition_frequency: int = 1,
    start_preconditioning_step: int = 1,
    grafting_type: str = "ADAM",  # ADAM, ADAGRAD, SGD, RMSPROP, or NONE
):
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Usage in your script:
    model = Model(hidden_size).cuda()
    model = set_deterministic_weights(model, seed=42)
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    # Initialize optimizer based on type
    if optimizer_type.lower() == "shampoo":
        # Import Distributed Shampoo
        from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
        from optimizers.distributed_shampoo.shampoo_types import (
            AdamGraftingConfig,
            AdaGradGraftingConfig,
            DDPShampooConfig,
            GraftingConfig,
            RMSpropGraftingConfig,
            SGDGraftingConfig,
        )
        import torch.distributed as dist
        torch.backends.cuda.preferred_linalg_library()
        # Initialize distributed processing for single process
        if not dist.is_initialized():
            dist.init_process_group(
                backend='gloo',  # Use 'nccl' if you have CUDA
                init_method='tcp://localhost:23456',
                rank=0,
                world_size=1
        )
        # Create grafting configuration
        if grafting_type == "ADAM":
            grafting_config = AdamGraftingConfig(
                beta2=0.999,
                epsilon=1e-8,
            )
        elif grafting_type == "ADAGRAD":
            grafting_config = AdaGradGraftingConfig(
                epsilon=1e-8,
            )
        elif grafting_type == "SGD":
            grafting_config = SGDGraftingConfig()
        elif grafting_type == "RMSPROP":
            grafting_config = RMSpropGraftingConfig(
                beta2=0.999,
                epsilon=1e-8,
            )
        else:  # NONE
            grafting_config = None
        
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            epsilon=1e-8,
            momentum=0.0,
            weight_decay=weight_decay,
            max_preconditioner_dim=max_preconditioner_dim,
            precondition_frequency=precondition_frequency,
            start_preconditioning_step=start_preconditioning_step,
            inv_root_override=0,
            exponent_multiplier=1.0,
            use_nadam=False,
            use_nesterov=True,
            use_bias_correction=True,
            use_decoupled_weight_decay=True,
            grafting_config=grafting_config,
            use_normalized_grafting=False,
            use_merge_dims=True,
            use_pytorch_compile=False,
            distributed_config=None,  # No distributed training for single GPU
            preconditioner_dtype=torch.float32,
            use_protected_eigh=True,
            track_root_inv_residuals=False,
        )
        print(f"Using Distributed Shampoo with {grafting_type} grafting")
        
    elif optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
        print("Using AdamW")
        
    elif optimizer_type.lower() == "nadamw":
        optimizer = NAdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
        print("Using NAdamW")
        
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.0,
            weight_decay=weight_decay
        )
        print("Using SGD")
        
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Choose from: adamw, nadamw, sgd, shampoo")
    

    # Initialize scheduler
    warmup_steps = int(warmup_factor * steps)
    if cycles == 1:
        scheduler = WarmCosine(
            optimizer, lr_min=0, lr_max=lr,
            warmup_steps=warmup_steps, T=steps
        )
    else:
        scheduler = WarmCosineCycles(
            optimizer, lr_min=0, lr_max=lr,
            warmup_steps=warmup_steps, T=steps, cycles=cycles
        )
    
    # Training loop
    data_iter = iter(train_loader)
    def debug_model_weights(model, script_name):
        print(f"{script_name} - Model weight checksums:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.sum().item():.6f} (shape: {param.shape})")

    debug_model_weights(model, "SImple")  # or "Simple"
    model.train()
    losses = []
    print(f"Starting training for {steps} steps...")
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Steps per epoch: {len(train_dataset) // batch}")
    torch.manual_seed(42)
    data_new = torch.randn(64, 1, 28, 28).cuda()
    targets = torch.randint(0, 10, (64,)).cuda()

    def data_func():
        return data_new, targets
    for step in range(steps):
        # Get next batch
        try:
            data, target = next(data_iter)
            
        except StopIteration:
            data_iter = iter(train_loader)
            data, target = next(data_iter)
            
        data, target = data_func()
       # data, target = data.cuda(), target.cuda()
        


        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        
                
        loss.backward()
        #scheduler.step()  
        optimizer.step()
        
        # Logging and evaluation
        if step % eval_every == 0 or step == steps - 1:
            losses.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            test_loss, accuracy = evaluate_model(model, test_loader)
            print(f"Step {step:4d} | Train Loss: {loss.item():.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {accuracy:.2f}% | "
                  f"LR: {current_lr:.2e}")
            model.train()  # Switch back to training mode
    
    print("Training completed!")
    print(losses)
if __name__ == "__main__":
    app()