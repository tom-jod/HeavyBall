import copy
import functools
import gc
import inspect
import os
import random
import sys
import time
import warnings
import json
import numpy as np
import optuna
import torch
from torch import nn
from torch._dynamo import config
import functools
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pad as pad_fn
import heavyball.utils
from heavyball import chainable as C
from heavyball.helpers import AutoSampler
from heavyball.utils import PrecondInitError

config.cache_size_limit = 2**16

np.warnings = warnings

base_args = {
    "betas": (0.9, 0.995, 0.995),
    "precondition_frequency": 16,
    "merge_dims": True,
    "warmup_steps": 100,
    "max_precond_dim": 2**16,
    "beta": 0.9,
    "max_size_triangular": 2**16,
    "split": False,
    "precond_grad_accum": True,
    "momentum_into_precond_update": True,
    "eps": 1e-8,
    "weight_decay": 0,
    "precond_update_power_iterations": 8,
    "dampening": 2**-18,             
    "shampoo_beta": 0.9,         
    "warmup_factor": 0.05,            # warmup_factor
    "dropout_rate": 0,             # dropout_rate
    "label_smoothing": 0,         # label_smoothing
    "momentum": 0,               # momentum (one_minus_momentum)
    "one_minus_momentum": 1,   # one_minus_momentum
    "use_momentum": True,          # use_momentum
    "max_preconditioner_dim": 1024,    # max_preconditioner_dim
    "precondition_frequency":  110,    # precondition_frequency
    "start_preconditioning_step": -1,  # start_preconditioning_step
    "inv_root_override":  0,         # inv_root_override
    "exponent_multiplier": 1,        # exponent_multiplier
    "grafting_type": "ADAM",         # grafting_type
    "grafting_epsilon":  1e-8,        # grafting_epsilon
    "use_normalized_grafting":  False,    # use_normalized_grafting
    "communication_dtype":  "FP32",       # communication_dtype
    "communicate_params": True,         # communicate_params
    "use_cosine_decay": True,           # use_cosine_decay
    "use_nadam":  True,               # use_nadam
    "step_hint_factor": 1,          # step_hint_factor

}


def get_optim(optim, params, **kwargs) -> C.BaseOpt:
    args = {**base_args, **kwargs}
    signature = inspect.signature(optim)
    
    # Always include total_steps if it exists, even if not in signature
    filtered_args = {k: v for k, v in args.items() if k in signature.parameters}
    
    # Force total_steps to be included if it exists in kwargs
    if 'total_steps' in kwargs:
        filtered_args['total_steps'] = kwargs['total_steps']
    if 'warmup_ratio' in kwargs:
        filtered_args['warmup_ratio'] = kwargs['warmup_ratio']

    # Use all args not filtered args
    if optim.__name__ == 'ExternalDistributedShampoo':
        o = optim(params, **args)
    else:
        o = optim(params, **filtered_args)

    return o


class FailureCounter:
    def __init__(self, mapping, broadcast: int = 1):
        self.mapping = mapping
        self.broadcast = broadcast
        max_consecutive_failures, minimal_improvement = zip(*mapping.items())
        self.max_consecutive_failures = torch.tensor(max_consecutive_failures, dtype=torch.float64, device="cuda")
        self.minimal_improvement = torch.tensor(minimal_improvement, dtype=torch.float64, device="cuda")
        self.consecutive_failures = torch.zeros(len(minimal_improvement), dtype=torch.int64, device="cuda").repeat(
            broadcast
        )

    def compare(self, inp, other):
        old_state = inp.reshape(1, -1, 1)  # vertical
        new_state = other.reshape(1, 1, -1)  # horizontal

        return new_state * (1 - self.minimal_improvement.reshape(-1, 1, 1)) < old_state

    def new(self):
        return FailureCounter(self.mapping, self.broadcast)

    def __call__(self, comparison, failure_scale: float = 1):
        failed = torch.any(comparison, axis=tuple(range(1, comparison.ndim)))
        self.consecutive_failures.copy_(torch.where(failed, self.consecutive_failures + 1, 0))
        mask = self.consecutive_failures >= (self.max_consecutive_failures.view(-1, 1) * failure_scale).flatten()
        return torch.any(mask)


class Validator:
    ema_index: int = 4
    warmup: int = 128
    ema_patience: float = 2
    ema_start: int = 0

    def __init__(self, ema_mapping, global_mapping, steps, emas: int = 20):
        self.step = 0
        self.emas = emas

        self.ema_states = torch.zeros((self.emas,), dtype=torch.float64, device="cuda")
        es = self.ema_start + 1
        self.update_factor = 2.0 ** (-torch.arange(es, 20 + es, dtype=torch.float64, device="cuda"))
        self.ema_failures = FailureCounter(ema_mapping)
        self.triu_indices = torch.triu_indices(self.emas, self.emas, offset=1)

        self.global_min_loss = torch.tensor((float("inf"),) * steps, dtype=torch.float64, device="cuda")
        self.global_min_failures = FailureCounter({1: 0}, steps)

        self.global_avg_loss = torch.zeros_like(self.global_min_loss)
        self.global_avg_step = torch.zeros_like(self.global_avg_loss)

        self.weighting = torch.arange(1, 1 + self.global_min_loss.size(0), device="cuda")
        self.weighting = self.weighting.clamp(min=self.warmup).view(1, -1)
        self.weighting = functools.reduce(torch.minimum, [self.weighting**p * f for f, p in global_mapping.items()])

        self.seen_until = np.zeros((), dtype=np.int64)  # seen_until has to be shared
        self.global_avg_failures = FailureCounter({1: 0}, steps)

    def new(self):
        new = copy.copy(self)
        new.ema_failures = new.ema_failures.new()
        new.global_min_failures = new.global_min_failures.new()
        new.global_avg_failures = new.global_avg_failures.new()
        new.ema_states = torch.zeros_like(new.ema_states)
        new.step = 0
        return new

    def _update_ema(self, loss):
        self.step += 1
        np.copyto(self.seen_until, np.maximum(self.seen_until, self.step - 1))

        uf = 1 - heavyball.utils.beta_debias(1 - self.update_factor, self.step)
        self.ema_states += uf * (loss - self.ema_states)

    def _global_min(self):
        loss = self.ema_states[self.ema_index]
        comparison = self.global_min_failures.compare(loss, self.global_min_loss)
        global_failed = self.global_min_failures(comparison.view(-1, 1), self.weighting)
        loss_slice = self.global_min_loss[self.step - 1 :]
        loss_slice.copy_(torch.where(torch.logical_and(loss < loss_slice, torch.isfinite(loss)), loss, loss_slice))
        return global_failed

    def _global_avg(self):
        loss = self.ema_states[self.ema_index]

        self.global_avg_step[self.step - 1] += 1
        self.global_avg_loss[self.step - 1].lerp_(loss, 1 / self.global_avg_step[self.step - 1])

        comparison = self.global_avg_failures.compare(loss, self.global_avg_loss).view(-1, 1)
        comparison[self.seen_until - 1 :].fill_(False)
        return self.global_avg_failures(comparison, self.weighting)

    def _local_convergence(self):
        comparison = self.ema_failures.compare(self.ema_states, self.ema_states)
        comparison = comparison[tuple([slice(None), *self.triu_indices])]
        return self.ema_failures(comparison, self.ema_patience)

    def __call__(self, loss):
        self._update_ema(loss)

        outputs = [self._global_min(), self._global_avg(), self._local_convergence()]
        if self.step < self.warmup:
            return torch.zeros_like(outputs[0])
        return functools.reduce(torch.logical_or, outputs)


class Stop(Exception):
    pass


class SkipConfig(ValueError):
    pass


class WinConditionMet(ValueError):
    pass


class Plotter(nn.Module):
    def __init__(
        self,
        objective_fn,
        x_limits=(-5, 5),
        y_limits=(-5, 5),
        resolution=300,
        transform=None,
        inverse_transform=None,
        should_normalize: bool = True,
    ):
        super().__init__()
        self.should_normalize = should_normalize
        self.objective = objective_fn
        self.initial = objective_fn.param.data.clone()
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.resolution = resolution
        self.transform = transform if transform else lambda x: x
        self.inverse_transform = inverse_transform if inverse_transform else lambda x: x
        
        self.param = objective_fn.param
        
        with torch.no_grad():
            x = torch.linspace(x_limits[0], x_limits[1], resolution)
            y = torch.linspace(y_limits[0], y_limits[1], resolution)
            self.X, self.Y = torch.meshgrid(x, y, indexing="ij")
            Z = torch.zeros_like(self.X)
            for i in range(resolution):
                for j in range(resolution):
                    objective_fn.param.data[:] = torch.tensor([self.X[i, j].item(), self.Y[i, j].item()], device="cuda")
                    Z[i, j] = self.transform(objective_fn())
            objective_fn.param.data[:] = self.initial
            self.Z = Z
        
        self.trajectory = [self.initial.detach().cpu().numpy()]
        # Track function values along trajectory
        self.loss_trajectory = [self.transform(objective_fn()).item()]
    
    def forward(self, *args):
        value = self.objective(*args)
        with torch.no_grad():
            self.trajectory.append(self.param.cpu().detach().numpy())
            self.loss_trajectory.append(self.transform(value).item())
        return self.transform(value)
    
    def plot_loss_curve(self, title="Loss During Optimization", save_path=None):
        """Create a line graph of the objective function values along the trajectory.
        
        Args:
            title: Title for the plot
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        iterations = range(len(self.loss_trajectory))
        
        plt.plot(iterations, self.loss_trajectory, 'b-', linewidth=2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot(self, title=None, save_path=None):
        """Create contour plot with optimization trajectory.
        
        Args:
            title: Optional title for the plot
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        z = self.Z
        if self.should_normalize:
            z = z - z.min()
            z = z / z.max()
            z = z + 1e-8
        plt.contourf(self.X.numpy(), self.Y.numpy(), z.log().numpy(), levels=1000)
        
        # Plot trajectory
        trajectory = np.array(self.trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], "r.-", label="Optimization path")
        plt.plot(trajectory[0, 0], trajectory[0, 1], "go", label="Start")
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", label="End")
        
        plt.colorbar(label=f"Log({'Normalized' * self.should_normalize}ObjectiveValue)")
        plt.xlabel("x")
        plt.ylabel("y")
        if title:
            plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

class Objective:
    def __init__(
        self,
        failure_threshold,
        model,
        opt: str,
        steps, # Use steps = 0 as a flag to run for a fixed amount of time rather than fixed number of steps
        group,
        data,
        loss_fn,
        win_condition,
        weight_decay,
        warmup_trials,
        estimate_condition_number,
        test_loader,
        track_variance,
        runtime_limit,
        step_hint,
        ema_index: int = 0,
        **kwargs,
    ):
        self.failure_threshold = failure_threshold
        self.model = model.cuda()
        for mod in self.model.modules():
            if isinstance(mod, torch.nn.RNNBase):
                mod.flatten_parameters()
        self.opt = opt
        self.steps = steps
        if steps == 0:
            self.steps = int(1e9) # Very large number, so we are not limited by number of steps
        self.group = group
        self.data = data
        self.loss_fn = loss_fn
        self._win_condition = win_condition
        self.weight_decay = weight_decay
        self.warmup_trials = int(warmup_trials)
        self.kwargs = kwargs
        self.ema_index = ema_index

        # up to 32768 consecutive times can the new loss be (1 - 1e-7)x larger than the preceding loss
        self.validator = Validator(
            {  # We can't check for improvement, as it's not guaranteed - "1% in 1k steps" may not happen
                256: 0,  # 0 improvement in 256 steps
                128: -1e-4,  # 1.01x over 128 steps
                64: -1e-3,  # 1.06x over 64 steps
                32: -0.01,  # 1.4x over 32 steps
                16: -0.1,  # 5.4x over 16 steps
                8: -0.33,  # 24x over 8 steps
                4: -0.5,  # 16x over 4 steps
                2: -0.75,  # 16x over 2 steps
                1: -0.99,  # 100x over 1 step
            },
            {1: 2, 4: 1.5, 16: 1},  # step_count^2 * 1 | step_count ^ 1.5 * 4 | step_count ^ 1 * 16
            steps,
        )
        self.m = None
        self.attempt = 0
        self.best_loss = None
        self.best_at = 0
        self.avg = None
        self.use_cudnn = True
        self.set_precond_init_scale = False
        #self.end_time = int(os.environ.get("HEAVYBALL_BENCHMARK_TIMEOUT", 3600 * 24)) + time.time() # was 3600 * 4
        self.start_time = time.time()
        self.runtime_limit = runtime_limit
        self._last_loss = float('inf')
        self.best_losses = []
        self.current_losses = []
        self.estimate_condition_number = estimate_condition_number
        self.test_loader = test_loader
        self.track_variance = track_variance
        # Set total_steps for lr schedules
        self.kwargs["total_steps"] = step_hint
        self.step_counter = 0
        self.step_counter = 0

    def win_condition(self, loss=None):
        if loss is not None:
            self._last_loss = loss
        return self._win_condition(self.m, self._last_loss)[0]
    
    def requires_prev_minibatch(self, opt):
        """Safely check if optimizer requires previous minibatch"""
        
        return getattr(opt, 'requires_prev_minibatch', False)
    
    def requires_prev_model(self, opt):
        """Safely check if optimizer requires previous model"""
        
        return getattr(opt, 'requires_prev_model', False)
    
    def _inner(self, params):
        
        input_kwargs = locals()
        input_kwargs.pop("self")
       
        params = {
            "lr": params[0],                           # learning_rate
            "betas": (1 - params[1], 1 - params[2], 1 - params[3]),  # (beta1, beta2, beta3)
            "beta": 1 - params[1],                     # beta1 (one_minus_beta1)
            "beta3": 1 - params[3],                    # beta3 (one_minus_beta3)
            "shampoo_beta": 1 - params[4],             # shampoo_beta (one_minus_shampoo_beta)
            "sam_step_size": params[4],                # one_minus_shampoo_beta
            "eps": params[6],                          # epsilon
            "precond_lr": params[4],                   # one_minus_shampoo_beta
            "weight_decay": params[5],                 # weight_decay
            "warmup_factor": params[7],                # warmup_factor
            "dropout_rate": params[8],                 # dropout_rate
            "label_smoothing": params[9],              # label_smoothing
            "momentum": 1 - params[10],                # momentum (one_minus_momentum)
            "one_minus_momentum": params[10],          # one_minus_momentum
            "use_momentum": params[11],                # use_momentum
            "max_preconditioner_dim": params[12],      # max_preconditioner_dim
            "precondition_frequency": params[13],      # precondition_frequency
            "start_preconditioning_step": params[14],  # start_preconditioning_step
            "inv_root_override": params[15],           # inv_root_override
            "exponent_multiplier": params[16],         # exponent_multiplier
            "grafting_type": params[17],               # grafting_type
            "grafting_epsilon": params[18],            # grafting_epsilon
            "use_normalized_grafting": params[19],     # use_normalized_grafting
            "communication_dtype": params[20],         # communication_dtype
            "communicate_params": params[21],          # communicate_params
            "use_cosine_decay": params[22],            # use_cosine_decay
            "use_nadam": params[23],                   # use_nadam
            "step_hint_factor": params[24],            # step_hint_factor
        }
        if params["use_momentum"] == False:
            params["momentum"] = 0.0

        self.kwargs["warmup_ratio"] = params["warmup_factor"]
        
        torch.cuda.reset_peak_memory_stats()
        if self.set_precond_init_scale:
            params["precond_init_scale"] = 0.1
        self.m = copy.deepcopy(self.model)
        o = get_optim(self.opt, self.m.parameters(), **params, **self.kwargs)
        is_lbfgs = hasattr(o, 'external_optimizers') and any(
            isinstance(opt, torch.optim.LBFGS) 
            for opt in o.external_optimizers.values()
            )
        is_lbfgs = hasattr(o, 'external_optimizers') and any(
            isinstance(opt, torch.optim.LBFGS) 
            for opt in o.external_optimizers.values()
            )
        torch_hist = torch.empty(self.group, dtype=torch.float64, device="cuda")
        validator = self.validator.new()
        self.test_accuracies = []
        self.grad_variances = []
        self.condition_numbers = []
        self.current_losses = []
        prev_model_params = None
        self.step_counter = 0
        timeout_reached = False
        self.start_time = time.time()
        self.end_time = self.runtime_limit + time.time() # was 3600 * 4
        self._last_loss = float('inf')
        # iterate through each epoch
        for i in range(self.steps // self.group):
           
            if not hasattr(self, 'test_accuracies'):
                self.test_accuracies = []
            if self.test_loader != None:
                test_accuracy = evaluate_test_accuracy(self.m, self.test_loader)
                self.test_accuracies.append(test_accuracy)
                
                if self.win_condition(1 - test_accuracy):
                    runtime = time.time() - self.start_time
                    print({"WIN"})
                    return validator.ema_states.min().item(), self.m, torch_hist[-1].item(), self.current_losses, self.test_accuracies, self.grad_variances, self.condition_numbers, self.step_counter, runtime
                
            if self.estimate_condition_number: 
                with torch.backends.cudnn.flags(enabled=False):
                    #if i % int(self.steps // (self.group*5)) == 0:
                    #self.condition_numbers.append(estimate_condition_number_hvp(self.m, self.data, n_probes=20, n_samples=50, loss_fn=self.loss_fn)[0])
                    self.condition_numbers.append(estimate_condition_number_grid_search(self.m, self.data, loss_fn=self.loss_fn, timout=30))
    
            if self.track_variance:
                # track variance at every 1000 steps through training
                #grad_variance = estimate_minibatch_variance_detailed(self.m, self.data, n_samples=10, loss_fn=self.loss_fn)[0]
                grad_variance = estimate_minibatch_variance_grid_search(self.m, self.data, loss_fn=self.loss_fn, timout=30)
                self.grad_variances.append(grad_variance["gradient_variance"])
            
            if hasattr(o, "train"):
                o.train()
            
            for j in range(self.group):
               
                self.step_counter += 1
        
                if self.requires_prev_minibatch(o):
                    # Get current batch
                    curr_inp, curr_tgt = self.data()
                    
                    # Initialize previous batch storage if needed
                    if not hasattr(self, '_prev_batch'):
                        self._prev_batch = None
                    
                    if self._prev_batch is None:
                        # First iteration: duplicate current batch
                        prev_inp, prev_tgt = curr_inp.clone(), curr_tgt.clone()
                    else:
                        # Use actual previous batch
                        prev_inp, prev_tgt = self._prev_batch
                    
                    # Update previous batch for next iteration
                    # Use .detach().clone() to avoid keeping gradients
                    self._prev_batch = (curr_inp.detach().clone(), curr_tgt.detach().clone())
                    
                    # Return both batches (you can modify this based on how you want to use them)
                    inp, tgt = {
                        'current': (curr_inp, curr_tgt),
                        'previous': (prev_inp, prev_tgt)
                    }, None  # tgt set to None since it's in the dict
                   
                else:
                    inp, tgt = self.data()

                
                def _closure():
                    # Move BOTH input and target to GPU
                    if inp is not None:
                        gpu_inp = inp.cuda() if not inp.is_cuda else inp
                    else:
                        gpu_inp = None
                    
                    if tgt is not None:
                        gpu_tgt = tgt.cuda() if not tgt.is_cuda else tgt  
                    else:
                        gpu_tgt = None
                    
                    # Now both are on GPU
                    loss = self.m() if gpu_inp is None else self.m(gpu_inp)
                    
                    if self.loss_fn is not None:
                        loss = self.loss_fn(loss, gpu_tgt)  
                        loss = self.loss_fn(loss, gpu_tgt)  
                    
                    loss.backward()
                    
                    # Clean up GPU tensors
                    del gpu_inp
                    if gpu_tgt is not None:
                        del gpu_tgt
                    
                    return loss
                
                # Later, when you need previous model:
                def _prev_closure():
                    nonlocal prev_model_params
                    
                    if prev_model_params is None:
                        # First step: no previous model, use current closure
                        prev_model_params = {name: param.data.clone().detach() 
                                        for name, param in self.m.named_parameters()}
                        return _closure()
                    
                    # Save current state
                    current_state = {name: param.clone() for name, param in self.m.named_parameters()}
                    
                    try:
                        # Restore previous model state
                        for name, param in self.m.named_parameters():
                            if name in prev_model_params:
                                param.data.copy_(prev_model_params[name])
                        
                        # Compute loss with previous model
                        loss = self.m() if inp is None else self.m(inp)
                        if self.loss_fn is not None:
                            loss = self.loss_fn(loss, tgt)
                        loss.backward()
                        return loss
                    finally:
                        # Restore current state
                        for name, param in self.m.named_parameters():
                            if name in current_state:
                                param.data.copy_(current_state[name])
                        # Update prev_model_params
                        prev_model_params = {name: param.data.clone().detach() 
                                        for name, param in self.m.named_parameters()}
                
                try:
                    if is_lbfgs:
                        # Store closure on the chainable optimizer for L-BFGS
                        o._lbfgs_closure = _closure
                        
                    # Always use the same step methods
                    if self.requires_prev_model(o):
                        loss = o.step_with_prev(_closure, _prev_closure)
                    else:
                        loss = o.step(_closure)
               
                
                try:
                    if is_lbfgs:
                        # Store closure on the chainable optimizer for L-BFGS
                        o._lbfgs_closure = _closure
                    loss = o.step(_closure)
                      
                except PrecondInitError:
                    self.set_precond_init_scale = True
                    return self._inner(**input_kwargs)
                o.zero_grad()
                with torch.no_grad():
                    torch_hist[j] = loss.detach()
                if 'inp' in locals() and inp is not None:
                    if isinstance(inp, dict):
                        # Handle the prev_minibatch case
                        for key, (data, target) in inp.items():
                            if hasattr(data, 'cpu'):
                                del data, target
                else:
                    # Handle normal case
                    if hasattr(inp, 'cpu'):
                        del inp
                    if 'tgt' in locals() and tgt is not None and hasattr(tgt, 'cpu'):
                        del tgt
                # Add loss to step_losses and current_losses (once per epoch: outer loop)
                if j == 0:
                    self.current_losses.append(float(torch_hist[j]))
                if time.time() > self.end_time:  # timeout
                    timeout_reached = True

            if timeout_reached:
                break
            
        # Get current memory usage before returning
        self.current_memory_usage = get_gpu_memory_usage()
        runtime = time.time() - self.start_time
        return validator.ema_states.min().item(), self.m, torch_hist[-1].item(), self.current_losses, self.test_accuracies, self.grad_variances, self.condition_numbers, self.step_counter, runtime
    
    def objective(self, params):
        self.attempt += 1
        target, model, loss, step_losses, test_accuracies, grad_variances, condition_numbers, step_counter, runtime = self._inner(params)
        self.loss = loss
        #if self.best_loss is None or loss < self.best_loss or not np.isfinite(self.best_loss):
        self.best_loss = loss
        self.best_at = self.attempt
        self.avg = None
        self.best_losses = step_losses.copy() 
        self.test_accuracies = test_accuracies.copy()
        self.grad_variances = grad_variances.copy()
        self.condition_numbers = condition_numbers.copy()
        self.memory_usage = getattr(self, 'current_memory_usage', 0)/ (1024**2)
        self.step_counter = step_counter
        self.runtime = runtime

        #if self.best_at * 100 < self.attempt and self.attempt - self.best_at > self.warmup_trials:  # no improvements
        #    raise Stop
        
        return target
    
    def reset(self):
        self._last_loss = None
        self.test_accuracies = []
        self.grad_variances = []
        self.condition_numbers = []
        self.current_losses = []
        self.m = None

    def get_best(self):
        self.group = 1
        # Modified to return losses with the model
        result = self._inner(np.exp(self.avg))
        # Return the model and its loss trajectory
        return result[1], result[3]


def loss_win_condition(target):
    def win(_model, loss: float):
        return loss <= target, {}

    return win


def param_norm_win_condition(target, offset):
    target = torch.full((), target, device="cuda")

    def win(model, loss):
        with torch.no_grad():
            norm = model.param.add(offset).square().mean().sqrt()
            return (norm < target).item(), {}

    return win


def param0_win_condition(target):
    target = torch.full((), target, device="cuda")

    def win(model, loss):
        with torch.no_grad():
            return (model.param[0] < target).item(), {}

    return win


def set_seed(seed: int = 0x1239121):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def cleanup():
    gc.enable()
    gc.collect()
    gc.disable()
    with torch.cuda.device("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def trial(
    model,
    data,
    loss_fn,
    win_condition,
    steps,
    opt,
    dtype,
    size,
    batch,
    weight_decay,
    method,
    length,
    depth,
    trials=10,
    failure_threshold=3,
    group=1000,
    base_lr: float = 1e-3,
    return_best: bool = False,
    warmup_trial_pct: int = 0.2,
    random_trials: int = 10,
    estimate_condition_number: bool = False,
    test_loader=None,
    track_variance=False,
    test_optimizer_implementation=False,
    runtime_limit: int = 3600 * 24, # Default runtime limit is a day
    step_hint: int = 0,
):
    use_fixed_hyperparams = False
    #if opt in ["SFAdamW", "ExternalDistributedShampoo"]:
    if opt in ["SFAdamW", "ExternalDistributedShampoo"]:
        opt_name = opt
        print("using_list")
        use_fixed_hyperparams = True

    if test_optimizer_implementation:
        group = 10
    # If no step hint given assume we do not need one as we are limiting the trial by number of steps rather than runtime
    if steps != 0:
        step_hint = steps
    
    if isinstance(opt, list):
        opt = opt[0]
    
    kwargs = {"caution": False, "mars": False, "decay": False}
    if opt.startswith("cautious-"):
        opt = opt[len("cautious-") :]
        kwargs["caution"] = True
    #kwargs["total_steps"] = steps 
    if opt.startswith("unscaled_cautious-"):
        opt = opt[len("unscaled_cautious-") :]
        heavyball.utils.disable_caution_scaling()
        kwargs["caution"] = True
    if opt.startswith("mars-"):
        opt = opt[len("mars-") :]
        kwargs["mars"] = True
    if opt.startswith("unscaled-"):
        opt = opt[len("unscaled-") :]
        kwargs["unscaled"] = True
    if opt.startswith("adaptive-"):
        opt = opt[len("adaptive-") :]
        kwargs["adaptive"] = True
    if opt.startswith("ortho-"):
        opt = opt[len("ortho-") :]
        kwargs["ortho_method"] = "newtonschulz-graft"
    opt = getattr(heavyball, opt)
    if "soap" not in opt.__name__.lower() and method != "qr":
        return
    
    heavyball.utils._ignore_warning("logei_candidates_func is experimental")
    heavyball.utils._ignore_warning("BoTorchSampler is experimental")
    heavyball.utils._ignore_warning("It will be set to log2(param_count). This requires `params` to be of type list.")
    heavyball.utils._ignore_warning("rank was set to")
    heavyball.utils._ignore_warning(
        "The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior."
    )
    heavyball.utils.zeroth_power_mode = method
    
    cleanup()
    set_seed()
    
    did_win = False
    
    def _win_condition(*args):
        nonlocal did_win
        win_state, out = win_condition(*args)
        did_win |= win_state
        #return did_win, out
        return win_state, out
    
    obj = Objective(
        failure_threshold,
        model,
        opt,
        steps,
        group,
        data,
        loss_fn,
        _win_condition,
        weight_decay,
        max(trials * warmup_trial_pct, 1 + random_trials),
        estimate_condition_number,
        test_loader,
        track_variance,
        runtime_limit,
        step_hint,
        **kwargs,
    )
    
    torch.cuda.synchronize()
    start_time = time.time()
    #stdout, sys.stdout = sys.stdout, sys.stderr
    
    set_seed()
    
    # Use a list to store win condition status (mutable object that can be modified from inner function)
    win_status = [False]
    global current_loss_trajectory
    current_loss_trajectory = []

    def create_params_dict(tuned_values, tuned_indices):
        """Create the required params dictionary format"""
        return {f"param_{i}_{param_names[i]}": val for i, val in zip(tuned_indices, tuned_values)}
    
    # Define all default hyperparameters
    default_hyperparams = {
        0: 0.004,      # learning_rate
        1: 0.1,        # one_minus_beta1
        2: 0.0045,     # one_minus_beta2
        3: 0.0045,     # one_minus_beta3
        4: 0.001,      # one_minus_shampoo_beta
        5: 0.08,       # weight_decay
        6: 1e-8,       # epsilon
        7: 0.02,       # warmup_factor
        8: 0.1,        # dropout_rate
        9: 0.4,        # label_smoothing
        10: 0.0,       # one_minus_momentum
        11: False,     # use_momentum
        12: 1024,      # max_preconditioner_dim
        13: 100,       # precondition_frequency
        14: -1,        # start_preconditioning_step
        15: 0,         # inv_root_override
        16: 1.0,       # exponent_multiplier
        17: "ADAM",    # grafting_type
        18: 1e-8,      # grafting_epsilon
        19: False,     # use_normalized_grafting
        20: "FP32",    # communication_dtype
        21: True,      # communicate_params
        22: True,      # use_cosine_decay
        23: True,      # use_nadam
        24: 1.0,       # step_hint_factor
        25: 1.0,       # beta2
    }
    
    param_names = {
        0: "learning_rate", 1: "one_minus_beta1", 2: "one_minus_beta2", 
        3: "one_minus_beta3", 4: "one_minus_shampoo_beta", 5: "weight_decay",
        6: "epsilon", 7: "warmup_factor", 8: "dropout_rate", 9: "label_smoothing",
        10: "one_minus_momentum", 11: "use_momentum", 12: "max_preconditioner_dim",
        13: "precondition_frequency", 14: "start_preconditioning_step", 15: "inv_root_override",
        16: "exponent_multiplier", 17: "grafting_type", 18: "grafting_epsilon",
        19: "use_normalized_grafting", 20: "communication_dtype", 21: "communicate_params",
        22: "use_cosine_decay", 23: "use_nadam", 24: "step_hint_factor", 25: "beta2"
    }
    
    def create_full_params_tuple(tuned_values, tuned_indices):
        """Create a full parameter tuple with specific indices tuned and others using defaults"""
        full_params = []
        tuned_dict = dict(zip(tuned_indices, tuned_values))
        
        for i in range(26):  # Total number of parameters
            if i in tuned_dict:
                full_params.append(tuned_dict[i])
            else:
                full_params.append(default_hyperparams[i])

        # update 1-beta2 if beta2 has been specified (this is the case in SFAdamW)
        if full_params[25] != 1.0:
            full_params[2] = 1 - full_params[25]
        
        # ignore beta2 param (number 25)
        return tuple(full_params[:25])
    
    if test_optimizer_implementation:
        print("Testing Optimizer Implementation")
        # Specify exactly which parameter indices to tune
        tuned_indices = [0, 1, 2, 3, 4, 5]  # You can change this to [0, 3, 4] or any other combination
        
        sampler = AutoSampler(
            seed=0x123125,
            search_space={
                f"param_{i}": optuna.distributions.FloatDistribution(1e-3, 1e-3) if i in [1,2,3,4] 
                            else optuna.distributions.FloatDistribution(0, 0) if i == 5
                            else optuna.distributions.FloatDistribution(1e-3, 1e-3)
                for i in tuned_indices
            },
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        winning_params = {}
        all_trial_results = []
        
        def _optuna_objective(trial):
            did_win = False
            set_seed(0x12312)
            
            # Get tuned parameter values
            tuned_values = []
            for i in tuned_indices:
                if i == 0:  # learning_rate
                    value = trial.suggest_float(f"param_{i}", 1e-3, 1e-3)
                elif i in [1, 2, 3, 4]:  # beta parameters
                    value = trial.suggest_float(f"param_{i}", 1e-3, 1e-3)
                elif i == 5:  # weight_decay
                    value = trial.suggest_float(f"param_{i}", 0, 0)
                else:
                    value = trial.suggest_float(f"param_{i}", 1e-3, 1e-3)  # default range
                tuned_values.append(value)
            
            # Create full params tuple
            full_params = create_full_params_tuple(tuned_values, tuned_indices)
            params_dict = create_params_dict(tuned_values, tuned_indices)
            loss = float("inf")
            
            out = obj.objective(full_params)
            
            # Create params dict for logging
            params_dict = {f"param_{i}_{param_names[i]}": val for i, val in zip(tuned_indices, tuned_values)}
            
            trial_result = {
            'tuned_indices': tuned_indices,
            'params': params_dict,
            'full_params': full_params,
            'losses': obj.current_losses.copy() if hasattr(obj, 'current_losses') else [],
            'test_accuracies': obj.test_accuracies.copy() if hasattr(obj, 'test_accuracies') else [],
            'grad_variances': obj.grad_variances.copy() if hasattr(obj, 'grad_variances') else [],
            'condition_numbers': obj.condition_numbers.copy() if hasattr(obj, 'condition_numbers') else [],
            'runtime': obj.runtime if hasattr(obj, 'runtime') else [],
            'steps': obj.step_counter if hasattr(obj, 'step_counter') else steps
            'steps': obj.step_counter if hasattr(obj, 'step_counter') else steps
            }
            all_trial_results.append(trial_result)
                
           
            return obj.runtime if hasattr(obj, 'runtime') else float('inf')
        
        set_seed()
        study.optimize(_optuna_objective, n_trials=1)
       
    elif use_fixed_hyperparams:
        hyperparam_counter = 0
        print("using fixed hypers")
        with open(f'benchmark/hyperparams/{opt_name}_hyperparams.json', 'r') as f:
            fixed_hyperparams_list = json.load(f)

        tuned_indices = []
        sampler = AutoSampler(
            seed=0x123125,
            search_space={
                f"param_{i}": optuna.distributions.FloatDistribution(1e-3, 1e-3) if i in [1,2,3,4] 
                            else optuna.distributions.FloatDistribution(0, 0) if i == 5
                            else optuna.distributions.FloatDistribution(1e-3, 1e-3)
                for i in tuned_indices
            },
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        winning_params = {}
        all_trial_results = []
        
        def _optuna_objective(trial):
            did_win = False
            nonlocal hyperparam_counter
            set_seed(0x12312)
            if hyperparam_counter < len(fixed_hyperparams_list):
            if hyperparam_counter < len(fixed_hyperparams_list):
                hyperparams = fixed_hyperparams_list[hyperparam_counter]
                hyperparam_counter += 1
            
                
                # Convert named hyperparams to indexed format
                indexed_hyperparams = {}
                name_to_index = {v: k for k, v in param_names.items()}
                
                for param_name, value in hyperparams.items():
                    if param_name in name_to_index:
                        indexed_hyperparams[name_to_index[param_name]] = value
                
                # Get tuned indices and values
                tuned_indices = list(indexed_hyperparams.keys())
                tuned_values = [indexed_hyperparams[i] for i in tuned_indices]
                
                # Create full params tuple
                full_params = create_full_params_tuple(tuned_values, tuned_indices)
                params_dict = create_params_dict(tuned_values, tuned_indices)
                
                loss = float("inf")
                obj = Objective(
                    failure_threshold,
                    model,
                    opt,
                    steps,
                    group,
                    data,
                    loss_fn,
                    _win_condition,
                    weight_decay,
                    max(trials * warmup_trial_pct, 1 + random_trials),
                    estimate_condition_number,
                    test_loader,
                    track_variance,
                    runtime_limit,
                    step_hint,
                    **kwargs,
                )
                out = obj.objective(full_params)
                
                trial_result = {
                    'trial_number': hyperparam_counter,
                    'tuned_indices': tuned_indices,
                    'original_json_params': hyperparams.copy(),
                    'params': params_dict,
                    'losses': obj.current_losses.copy() if hasattr(obj, 'current_losses') else [],
                    'test_accuracies': obj.test_accuracies.copy() if hasattr(obj, 'test_accuracies') else [],
                    'grad_variances': obj.grad_variances.copy() if hasattr(obj, 'grad_variances') else [],
                    'condition_numbers': obj.condition_numbers.copy() if hasattr(obj, 'condition_numbers') else [],
                    'runtime': obj.runtime if hasattr(obj, 'runtime') else 0,
                    'steps': obj.step_counter if hasattr(obj, 'step_counter') else steps,
                    'steps': obj.step_counter if hasattr(obj, 'step_counter') else steps,
                }
                print(trial_result)
                all_trial_results.append(trial_result)   
                print(trial_result)
                all_trial_results.append(trial_result)   
                return obj.runtime if hasattr(obj, 'runtime') else float('inf')
            
        set_seed()
        n_trials = 5
        # If trials is less than 5, only do trials many runs
        if trials < n_trials:
            n_trials = trials

        study.optimize(_optuna_objective, n_trials)
           
        n_trials = 5
        # If trials is less than 5, only do trials many runs
        if trials < n_trials:
            n_trials = trials

        study.optimize(_optuna_objective, n_trials)
           
    else:
        print("Using Quasi-random Sampling")
        # Specify exactly which parameter indices to tune - you can change this list
        tuned_indices = [0, 1, 2, 5]  # Example: tune learning_rate, one_minus_beta3, one_minus_shampoo_beta
        
        # Define search ranges for each parameter index
        search_ranges = {
            0: (1e-4, 1e-2, True),   # learning_rate (log scale)
            1: (1e-3, 0.5, True),    # one_minus_beta1 (log scale)
            2: (1e-3, 0.5, True),    # one_minus_beta2 (log scale)
            3: (1e-3, 0.5, True),    # one_minus_beta3 (log scale)
            4: (1e-3, 0.5, True),      # one_minus_shampoo_beta (log scale)
            5: (1e-5, 1e-0, True),   # weight_decay (log scale)
        }
        
        sampler = AutoSampler(
            seed=0x123125,
            search_space={
                f"param_{i}": optuna.distributions.FloatDistribution(
                    search_ranges[i][0], search_ranges[i][1], log=search_ranges[i][2]
                ) for i in tuned_indices if i in search_ranges
            },
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        winning_params = {}
        all_trial_results = []
        
        def _optuna_objective(trial):
            did_win = False
            set_seed(0x12312)
            obj = Objective(
                failure_threshold,
                model,
                opt,
                steps,
                group,
                data,
                loss_fn,
                _win_condition,
                weight_decay,
                max(trials * warmup_trial_pct, 1 + random_trials),
                estimate_condition_number,
                test_loader,
                track_variance,
                runtime_limit,
                step_hint,
                **kwargs,
            )
            # Get tuned parameter values
            tuned_values = []
            for i in tuned_indices:
                if i in search_ranges:
                    low, high, log_scale = search_ranges[i]
                    value = trial.suggest_float(f"param_{i}", low, high, log=log_scale)
                else:
                    value = default_hyperparams[i]  # fallback to default
                tuned_values.append(value)
            
            # Create full params tuple
            full_params = create_full_params_tuple(tuned_values, tuned_indices)
            
            loss = float("inf")
            try:
                out = obj.objective(full_params)
                
                # Create params dict for logging
                params_dict = {f"param_{i}_{param_names[i]}": val for i, val in zip(tuned_indices, tuned_values)}
                
                trial_result = {
                    'tuned_indices': tuned_indices,
                    'params': params_dict,
                    'full_params': full_params,
                    'losses': obj.current_losses.copy() if hasattr(obj, 'current_losses') else [],
                    'test_accuracies': obj.test_accuracies.copy() if hasattr(obj, 'test_accuracies') else [],
                    'grad_variances': obj.grad_variances.copy() if hasattr(obj, 'grad_variances') else [],
                    'condition_numbers': obj.condition_numbers.copy() if hasattr(obj, 'condition_numbers') else [],
                    'runtime': obj.runtime if hasattr(obj, 'runtime') else [],
                    'steps': obj.step_counter if hasattr(obj, 'step_counter') else steps,
                    'steps': obj.step_counter if hasattr(obj, 'step_counter') else steps,
                }
                print(trial_result)
                all_trial_results.append(trial_result)   
            except Stop:
                pass
            return obj.runtime if hasattr(obj, 'runtime') else float('inf')
        set_seed()
        study.optimize(_optuna_objective, n_trials=trials)
    
    # After all trials complete, find the best one
    if all_trial_results:
        #best_trial = min(all_trial_results, key=lambda x: x['losses'][-1])
        #best_trial = min(all_trial_results, key=lambda x: x['runtime'])
        best_trial = max(all_trial_results, key=lambda x: max(x['test_accuracies']) if x['test_accuracies'] else 0)
        winning_params = best_trial['params']
        
        # Use the best trial's metrics
        final_losses = best_trial['losses']
        final_test_accuracies = best_trial['test_accuracies']
        final_grad_variances = best_trial['grad_variances']
        final_condition_numbers = best_trial['condition_numbers']
        final_runtime = best_trial['runtime']
        final_steps = best_trial['steps']
        final_steps = best_trial['steps']
        
        print("Successfully found the minimum runtime")

    if all_trial_results:
        #best_trial = min(all_trial_results, key=lambda x: x['losses'][-1])
        most_accurate_trial = max(all_trial_results, key=lambda x: max(x['test_accuracies']) if x['test_accuracies'] else 0)
        
        best_test_accuracies = most_accurate_trial['test_accuracies']
        best_test_accuracy = max(best_test_accuracies)
        
        print("Successfully found the maximum test accuracy")

    
    else:
        # Fallback if no trials succeeded
        winning_params = {"lr": base_lr, "1mbeta1": 0.9, "1mbeta2": 0.999, "1mshampoo_beta": 0.999, "1mbeta3": 0.999, "weight_decay": 0.999}
        final_losses = []
        final_test_accuracies = []
        final_grad_variances = []
        final_condition_numbers = []
    
    # Calculate condition number stats
    mean_cond = np.mean(final_condition_numbers) if final_condition_numbers else 0
    std_err_cond = np.std(final_condition_numbers) / np.sqrt(len(final_condition_numbers)) if final_condition_numbers else 0
    torch.cuda.synchronize() 
    end_time = time.time()
    print(best_test_accuracy)
    print(best_test_accuracy)
    print(
        f"Took: {end_time - start_time} | Highest_accuracy: {best_test_accuracy} | Winning_runtime: {final_runtime} | Trials: {obj.attempt + 1} | "
        f"Params: {opt.__name__}_{winning_params} | "
        f"Took: {end_time - start_time} | Highest_accuracy: {best_test_accuracy} | Winning_runtime: {final_runtime} | Trials: {obj.attempt + 1} | "
        f"Params: {opt.__name__}_{winning_params} | "
        f"loss_trajectory: {final_losses} | test_accuracies: {final_test_accuracies} | "
        f"grad_variances: {final_grad_variances} | mean_cond: {mean_cond} | std_err_cond: {std_err_cond} | steps: {final_steps} "
        f"grad_variances: {final_grad_variances} | mean_cond: {mean_cond} | std_err_cond: {std_err_cond} | steps: {final_steps} "
    )
    
    loss_trajectory = obj.best_losses
    if return_best:
    # Get the best model
        best_model, _ = obj.get_best()
        # Add the loss trajectory to the model as an attribute
        best_model.loss_trajectory = loss_trajectory
        return best_model

    
def evaluate_test_accuracy(model, test_loader):
    # Save the current training state
    was_training = model.training
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            
            # Handle different output shapes
            if output.dim() > 2:  # Sequence modeling: [batch, seq_len, vocab_size]
                pred = output.argmax(dim=-1)  # [batch, seq_len]
                pred_flat = pred.view(-1)
                target_flat = target.view(-1)
                correct += pred_flat.eq(target_flat).sum().item()
                total += target_flat.numel()
            else:  # Regular classification: [batch, num_classes]
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.numel()
    
    # Restore the original training state
    model.train(was_training)
    
    return correct / total

def to_device(obj, device):
    """Recursively move tensors to device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {key: to_device(value, device) for key, value in obj.items()}
    else:
        return obj
    
def evaluate_model(model, test_loader, loss_fn=torch.nn.functional.cross_entropy):
    # Save the current training state
    was_training = model.training
    model.eval()
    with torch.no_grad():
        # Try to get an iterator from the test_loader
        if hasattr(test_loader, '__iter__') and not hasattr(test_loader, '__next__'):
            # This is likely a PyTorch DataLoader or similar
            test_iterator = iter(test_loader)
            batch = next(test_iterator)
        else:
            # This is already an iterator or TF dataset
            batch = next(test_loader)
        #FastMRI handling
        if isinstance(batch, dict):
            
            
            # Extract tensors from TensorFlow dataset structure
            if isinstance(batch, dict):
                inp = batch['inputs']
                tgt = batch.get('targets', None)
                
                # Convert to numpy if they're TF tensors
                if hasattr(inp, 'numpy'):
                    inp = inp.numpy()
                if tgt is not None and hasattr(tgt, 'numpy'):
                    tgt = tgt.numpy()
                    
                # Convert to PyTorch tensors
                inp = torch.from_numpy(inp).float() if isinstance(inp, np.ndarray) else inp
                tgt = torch.from_numpy(tgt).float() if isinstance(tgt, np.ndarray) and tgt is not None else tgt
               
                if inp.ndim == 3:  # [batch, height, width]
                    inp = inp.unsqueeze(1)   # [batch, 1, height, width]
                if tgt is not None and tgt.ndim == 3:
                    tgt = tgt.unsqueeze(1)
                    
                print(f"Input shape after processing: {inp.shape}")
                
            else:
                # Handle case where batch is not a dict
                inp = batch
                tgt = None
                if hasattr(inp, 'numpy'):
                    inp = inp.numpy()
                inp = torch.from_numpy(inp).float() if isinstance(inp, np.ndarray) else inp
                
                # Add channel dimension here too
                if inp.ndim == 3:
                    inp = inp.unsqueeze(1)
            target = tgt
            model.zero_grad()
            gpu_target = target.cuda() if hasattr(target, 'cuda') else torch.from_numpy(target).cuda()
            gpu_inp = inp.cuda() if hasattr(inp, 'cuda') else torch.from_numpy(inp).cuda()
            loss = model(gpu_inp)
            
            if loss_fn is not None:
                loss = loss_fn(loss, gpu_target).item()

        else:        
            for data, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), to_device(target,'cuda')
                
                output = model(data)
                loss = loss_fn(output, target).item()
        
    model.train(was_training)
    return loss

def get_gpu_memory_usage():
    """Returns the current GPU memory usage in bytes"""
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()

def estimate_minibatch_variance_grid_search(model, data_fn, loss_fn=torch.nn.functional.cross_entropy, task_type='auto', timout=3600):
    n_samples = [5,10,20,50,100,1000]
    results = [0]

    for n in n_samples:
        result, reached_timout = estimate_minibatch_variance_detailed(model, data_fn, n_samples=n, loss_fn=loss_fn, task_type='auto', timout=timout)
        if reached_timout:
            return results[-1]
        else:
            results.append(result)

    return results[-1]

def estimate_minibatch_variance_detailed(model, data_fn, n_samples=100, loss_fn=torch.nn.functional.cross_entropy, task_type='auto', timout=3600):
    """
    Enhanced minibatch variance estimation that works for both classification and regression
    """
    model.train()
    gradient_samples = []
    loss_samples = []
    reached_timeout = False
    start_time = time.time()
    for sample_idx in range(n_samples):
       
        inp, target = data_fn()
        model.zero_grad()
        gpu_target = target.cuda() if hasattr(target, 'cuda') else torch.from_numpy(target).cuda()
        gpu_inp = inp.cuda() if hasattr(inp, 'cuda') else torch.from_numpy(inp).cuda()
        loss = model(gpu_inp)
        
        if loss_fn is not None:
            loss = loss_fn(loss, gpu_target)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        flat_grad = torch.cat([g.flatten() for g in grads])
        gradient_samples.append(flat_grad.detach().cpu().numpy())
        loss_samples.append(loss.item())

        if time.time() - start_time > timout:
            reached_timeout = True
            break
    
    if not gradient_samples:
        return _get_default_metrics(task_type), reached_timeout
    
    # Convert to numpy
    gradients = np.array(gradient_samples)
    
    # Standard gradient variance metrics (same for both tasks)
    grad_var = np.var(gradients, axis=0)
    total_grad_variance = np.mean(grad_var)
    grad_norm_variance = np.var([np.linalg.norm(g) for g in gradients])
    loss_variance = np.var(loss_samples)
    
    # Gradient direction variance
    cosine_similarities = []
    for i in range(len(gradients) - 1):
        g1, g2 = gradients[i], gradients[i + 1]
        cos_sim = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-8)
        cosine_similarities.append(cos_sim)
    
    gradient_direction_variance = np.var(cosine_similarities) if cosine_similarities else 0
    
    # Base metrics
    result = {
        "gradient_variance": float(total_grad_variance),
        "gradient_norm_variance": grad_norm_variance,
        "loss_variance": loss_variance,
        "gradient_direction_variance": gradient_direction_variance,
        "mean_loss": np.mean(loss_samples),
        "task_type": task_type
    }
  
    return result, reached_timeout

def _get_default_metrics(task_type):
    """Return default metrics when no samples are available"""
    base_metrics = {
        "gradient_variance": float('inf'),
        "gradient_norm_variance": float('inf'),
        "loss_variance": float('inf'),
        "gradient_direction_variance": float('inf'),
        "mean_loss": float('inf'),
        "task_type": task_type
    }
    
    return base_metrics


def estimate_condition_number_grid_search(model, data_fn, loss_fn=torch.nn.functional.cross_entropy, timout=3600):
    n_samples = [5,10,20,50,100,300,1000]
    results = [0]

    for n in n_samples:
        result, reached_timout = estimate_condition_number_hvp(model, data_fn, n_probes=int(n/2), n_samples=n, loss_fn=loss_fn, timout=timout)
        if reached_timout:
            return results[-1]
        else:
            results.append(result)

    return float(results[-1])


def estimate_condition_number_hvp(model, data_fn, n_probes=20, n_samples=50, loss_fn=None, timout=3600):
    """Estimate condition number using Hessian-vector products - works for any loss function"""
    start_time = time.time()
    reached_timout = False
    # Auto-detect loss function if not provided
    inp, target = data_fn()
    model.zero_grad()
    gpu_target = target.cuda() if hasattr(target, 'cuda') else torch.from_numpy(target).cuda()
    gpu_inp = inp.cuda() if hasattr(inp, 'cuda') else torch.from_numpy(inp).cuda()
    loss = model(gpu_inp)

    output, y = gpu_inp, gpu_target
    
    if loss_fn is None:
        try:
            if output.dim() > 2 or (y.dim() == output.dim() and y.shape == output.shape):
                loss_fn = torch.nn.functional.mse_loss
            else:
                loss_fn = torch.nn.functional.cross_entropy
        except:
            loss_fn = torch.nn.functional.cross_entropy  # Default
    
    def compute_hvp(loss, params, v):
        """Compute Hessian-vector product"""
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_v = sum(torch.sum(g * v_i) for g, v_i in zip(grads, v))
        hvp = torch.autograd.grad(grad_v, params, retain_graph=True)
        return hvp
    
    eigenvals = []
    for _ in range(n_samples):
        try:
            inp, target = data_fn()
            model.zero_grad()
            gpu_target = target.cuda() if hasattr(target, 'cuda') else torch.from_numpy(target).cuda()
            gpu_inp = inp.cuda() if hasattr(inp, 'cuda') else torch.from_numpy(inp).cuda()
            loss = model(gpu_inp)
            
            if loss_fn is not None:
                loss = loss_fn(loss, gpu_target)
            params = list(model.parameters())
            
            # Probe with random vectors
            for i in range(n_probes):
                # Random probe vector
                v = [torch.randn_like(p) for p in params]
                v_norm = sum(torch.sum(v_i**2) for v_i in v).sqrt()
                v = [v_i / v_norm for v_i in v]  # Normalize
                
                # Compute Hv
                hvp = compute_hvp(loss, params, v)
                
                # Rayleigh quotient: v^T H v / v^T v = v^T H v (since v normalized)
                eigenval = sum(torch.sum(hv * v_i) for hv, v_i in zip(hvp, v))
                eigenvals.append(eigenval.item())

            if time.time() - start_time > timout:
                reached_timout = True
                break

                
        except Exception as e:
            print(f"Error in condition number estimation: {e}")
            continue
    
    eigenvals = np.array(eigenvals)
    eigenvals = eigenvals[np.isfinite(eigenvals)]  # Remove NaN/inf
    
    if len(eigenvals) == 0:
        return float('inf'), False
    
    # Condition number approximation
    abs_eigenvals = np.abs(eigenvals)
    max_eig = np.max(abs_eigenvals)
    min_eig = np.min(abs_eigenvals)
    
    if min_eig <= 0:
        return float('inf'), False  # Indefinite Hessian
        
    return max_eig / (min_eig + 1e-8), reached_timout


def evaluate_test_accuracy(model, test_loader):
    """Evaluate model performance with automatic task detection"""
    # Save the current training state
    was_training = model.training
    model.eval()
    
    # Try to get a sample batch to determine task type
    try:
        sample_data, sample_target = next(iter(test_loader))
        if torch.cuda.is_available():
            sample_data, sample_target = sample_data.cuda(), sample_target.cuda()
        
        with torch.no_grad():
            sample_output = model(sample_data)
        
        # Detect task type based on output characteristics
        task_type = detect_task_type(sample_output, sample_target, model)
        
        if task_type == 'fastmri':
            result = evaluate_fastmri_ssim_internal(model, test_loader)
        else:
            result = evaluate_classification_accuracy(model, test_loader)
            
    except Exception as e:
        print(f"Error in evaluation: {e}")
        result = 0.0
    
    # Restore the original training state
    model.train(was_training)
    return result

def detect_task_type(output, target, model):
    """Detect whether this is a FastMRI reconstruction task or classification"""
    
    # Check 1: If model is a UNet (common for FastMRI)
    model_name = model.__class__.__name__.lower()
    if 'unet' in model_name:
        return 'fastmri'
    
    # Check 2: If output and target are both 2D images (after removing batch/channel dims)
    output_spatial_dims = len([d for d in output.shape[2:] if d > 1])  # Skip batch and channel
    target_spatial_dims = len([d for d in target.shape[2:] if d > 1])
    
    if output_spatial_dims == 2 and target_spatial_dims == 2:
        # Both are 2D images
        if output.shape[-2:] == target.shape[-2:]:  # Same spatial dimensions
            return 'fastmri'
    
    # Check 3: If output is continuous values (not logits) and target is also continuous
    if output.dtype == torch.float32 and target.dtype == torch.float32:
        # Check if output values are in a reasonable range for images (not logits)
        output_range = output.max() - output.min()
        target_range = target.max() - target.min()
        
        # If both have reasonable image-like ranges and similar scales
        if (0.1 < output_range < 100) and (0.1 < target_range < 100):
            # Check if target has image-like statistics (not one-hot encoded)
            if target.numel() > 1000 and len(target.unique()) > 10:
                return 'fastmri'
    
    # Check 4: Spatial dimensions suggest image reconstruction
    if len(output.shape) == 4 and len(target.shape) == 4:  # [B, C, H, W]
        if output.shape[2] > 64 and output.shape[3] > 64:  # Large spatial dimensions
            if output.shape == target.shape:  # Same shape (reconstruction task)
                return 'fastmri'
    
    # Default to classification
    return 'classification'

def evaluate_classification_accuracy(model, test_loader):
    """Original classification evaluation logic"""
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            
            # Handle different output shapes
            if output.dim() > 2:
                # Sequence modeling: [batch, seq_len, vocab_size]
                pred = output.argmax(dim=-1)  # [batch, seq_len]
                pred_flat = pred.view(-1)
                target_flat = target.view(-1)
                correct += pred_flat.eq(target_flat).sum().item()
                total += target_flat.numel()
            else:
                # Regular classification: [batch, num_classes]
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.numel()
    
    return correct / total

def evaluate_fastmri_ssim_internal(model, test_loader):
    """SSIM evaluation for FastMRI tasks"""
    total_ssim = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= 10:  # Limit evaluation to prevent memory issues
                break
            
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            
            # Ensure output and target have the same shape
            if output.shape != target.shape:
                if len(output.shape) == 4 and len(target.shape) == 4:
                    output = F.interpolate(output, size=target.shape[-2:], 
                                         mode='bilinear', align_corners=False)
            
            try:
                # Compute SSIM for each example in the batch
                batch_ssim = 0.0
                batch_size = output.shape[0]
                
                for j in range(batch_size):
                    # Remove batch and channel dimensions for SSIM computation
                    pred_img = output[j].squeeze()
                    target_img = target[j].squeeze()
                    
                    # Simple SSIM computation (you can use your more complex version)
                    ssim_val = structural_similarity(pred_img, target_img, data_range=1.0)
                    batch_ssim += ssim_val.item()
                
                total_ssim += batch_ssim
                total_samples += batch_size
                
            except Exception as e:
                print(f"SSIM computation error: {e}")
                # Fallback to MSE-based metric
                mse = F.mse_loss(output, target)
                # Convert MSE to a "higher is better" metric
                total_ssim += max(0, 1.0 - mse.item())
                total_samples += output.shape[0]
    
    if total_samples == 0:
        return 0.0
    
    avg_ssim = total_ssim / total_samples
    print(f"FastMRI SSIM evaluation: {avg_ssim:.4f}")
    return avg_ssim

# Add the SSIM functions from before
def structural_similarity(im1, im2, data_range=1.0, win_size=7, k1=0.01, k2=0.03):
    """Compute the mean structural similarity index between two images."""
    # Add a simple fallback for very small images or errors
    try:
        if im1.numel() < win_size * win_size or im2.numel() < win_size * win_size:
            # Fallback to normalized correlation for very small images
            im1_flat = im1.flatten()
            im2_flat = im2.flatten()
            correlation = F.cosine_similarity(im1_flat.unsqueeze(0), im2_flat.unsqueeze(0))
            return correlation.squeeze()
        
        filter_func = functools.partial(_uniform_filter, size=win_size)
        num_points = win_size ** len(im1.shape)
        cov_norm = num_points / (num_points - 1)
        
        # compute (weighted) means
        ux = filter_func(im1)
        uy = filter_func(im2)
        
        # compute (weighted) variances and covariances
        uxx = filter_func(im1 * im1)
        uyy = filter_func(im2 * im2)
        uxy = filter_func(im1 * im2)
        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)
        
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2
        a1 = 2 * ux * uy + c1
        a2 = 2 * vxy + c2
        b1 = ux**2 + uy**2 + c1
        b2 = vx + vy + c2
        d = b1 * b2
        s = (a1 * a2) / d
        
        # to avoid edge effects will ignore filter radius strip around edges
        pad = (win_size - 1) // 2
        if s.shape[0] > 2*pad and s.shape[1] > 2*pad:
            return torch.mean(s[pad:-pad, pad:-pad])
        else:
            return torch.mean(s)
            
    except Exception as e:
        print(f"SSIM computation failed: {e}, using fallback")
        # Fallback to simple correlation
        return F.cosine_similarity(im1.flatten().unsqueeze(0), im2.flatten().unsqueeze(0)).squeeze()

def _uniform_filter(im, size=7):
    """Uniform filter for SSIM computation"""
    try:
        pad_size = size // 2
        def conv(im):
            padded_im = pad_fn(im.unsqueeze(0), pad_size, padding_mode='symmetric')
            padded_im = padded_im[0, pad_size:-pad_size]
            filters = torch.ones(1, 1, size, dtype=padded_im.dtype, device=im.device)
            return F.conv1d(padded_im.unsqueeze(1), filters).squeeze(1) / size
        
        im = conv(im)
        im = conv(im.T)
        return im.T
    except Exception:
        # Fallback to simple averaging
        return im