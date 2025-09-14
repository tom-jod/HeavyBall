import copy
import csv
import functools
import gc
import inspect
import os
import random
import sys
import time
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from torch import nn
from torch._dynamo import config
import functools
from datetime import datetime
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pad as pad_fn
from torch.nn.utils import parameters_to_vector
from torch.func import functional_call
import heavyball.utils
from heavyball import chainable as C
from heavyball.helpers import AutoSampler
from heavyball.utils import PrecondInitError
from torch.func import functional_call
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
    "precondition_frequency":  100,    # precondition_frequency
    "start_preconditioning_step": -1,  # start_preconditioning_step
    "inv_root_override":  0,         # inv_root_override
    "exponent_multiplier": 1,        # exponent_multiplier
    "grafting_type": "None",         # grafting_type
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
        train_loader,
        track_variance,
        runtime_limit,
        step_hint,
        best_test_accuracies = [],
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
        self.best_test_accuracies = best_test_accuracies
 
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
        self.train_loader = train_loader
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
        
        weight_decay = params[5]
        
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
        self.condition_number_variances = []
        self.current_losses = []
        prev_model_params = None
        self.step_counter = 0
        timeout_reached = False
        self.start_time = time.time()
        self.end_time = self.runtime_limit + time.time() # was 3600 * 4
        self._last_loss = float('inf')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
                    return validator.ema_states.min().item(), self.m, torch_hist[-1].item(), self.current_losses, self.test_accuracies, self.grad_variances, self.condition_numbers, self.condition_number_variances, self.step_counter, runtime
                
                # Early stopping for trials that are out of the 5 percent test performance of current best trial
                early_stopping_enabled = True
                if self.best_test_accuracies != [] and len(self.best_test_accuracies) == self.steps//self.group and early_stopping_enabled:
                    if test_accuracy < 0.95 * self.best_test_accuracies[i]:
                        runtime = time.time() - self.start_time
                        print({"STOPPING EARLY"})
                        return validator.ema_states.min().item(), self.m, torch_hist[-1].item(), self.current_losses, self.test_accuracies, self.grad_variances, self.condition_numbers, self.condition_number_variances, self.step_counter, runtime
                
            if self.estimate_condition_number: 
                with torch.backends.cudnn.flags(enabled=False):
                    x, y = self.data()

                    estimate = True
                    visualisation = True
            
                    if estimate:
                        condition_number = estimate_effective_condition_number_reorth(self.m, self.data, loss_fn=self.loss_fn, weight_decay=weight_decay, lanczos_steps=200, n_samples=5)
                        self.condition_numbers.append(condition_number['condition_number'])
                        self.condition_number_variances.append(condition_number['condition_number_variance'])
                        # Extract eigenvalues
                        true_eigs_l = np.array(condition_number["eigenvals_lanczos"])
                        true_eigs_r = np.array(condition_number["eigenvals_rayleigh"])
                        true_eigs_l = true_eigs_l[np.isfinite(true_eigs_l)]  
                        true_eigs_r = true_eigs_r[np.isfinite(true_eigs_r)]  
                        eigs_abs_l = np.abs(true_eigs_l)
                        eigs_abs_r = np.abs(true_eigs_r)
                       
                        # Split into pos/neg subspaces
                        pos_eigs_r = true_eigs_r[true_eigs_r > 0]
                        neg_eigs_r = true_eigs_r[true_eigs_r < 0]
                        pos_eigs_l = true_eigs_l[true_eigs_l > 0]
                        neg_eigs_l = true_eigs_l[true_eigs_l < 0]

                        pos_lambda_min = np.min(pos_eigs_l) if len(pos_eigs_l) > 0 else np.nan
                        pos_lambda_max = np.max(pos_eigs_l) if len(pos_eigs_l) > 0 else np.nan
                        neg_lambda_min = np.min(neg_eigs_l) if len(neg_eigs_l) > 0 else np.nan
                        neg_lambda_max = np.max(neg_eigs_l) if len(neg_eigs_l) > 0 else np.nan
                        
                        pos_neg_ratio = len(pos_eigs_l) / (len(neg_eigs_l) + 1e-8)
                        
                        # --- Save directory ---
                        save_dir = "Condition_number_estimates/timestamp"
                        os.makedirs(save_dir, exist_ok=True)

                        # --- Plot both histograms side by side ---
                        plt.rcParams['font.size'] = 16
                        plt.rcParams['font.family'] = 'serif'
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                        # Left: abs eigenvalues (log scale)
                        if len(eigs_abs_l) > 0:
                            log_bins = np.logspace(np.log10(np.min(eigs_abs_l)), np.log10(np.max(eigs_abs_l)), 100)
                            axes[0].hist(eigs_abs_l, bins=log_bins, color="navy", label="Lanczos")
                            log_bins = np.logspace(np.log10(np.min(eigs_abs_r)), np.log10(np.max(eigs_abs_r)), 100)
                            axes[0].hist(eigs_abs_r, bins=log_bins, color="red", label="Rayleigh")
                            axes[0].set_xscale("log")
                            axes[0].set_xlabel("Absolute value of estimated eigenvalues")
                            axes[0].set_ylabel("Count")
                            axes[0].legend()
                            

                        # Right: raw eigenvalues
                        axes[1].hist(true_eigs_l, bins=200, color="navy")
                        axes[1].set_xlabel("Lanczos Ritz values")
                        axes[1].set_ylabel("Count")
                        

                        plt.tight_layout()
                        plt.savefig(f"{save_dir}/step_{i}.png", dpi=500)
                        plt.close(fig)

                        # --- CSV file inside save_dir ---
                        csv_file = os.path.join(save_dir, "eigenspectrum_stats.csv")

                        if not os.path.exists(csv_file):
                            with open(csv_file, mode="w", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    "iteration",
                                    "condition_number", "condition_number_abs", "condition_number_variance",
                                    "pos_lambda_min", "pos_lambda_max",
                                    "neg_lambda_min", "neg_lambda_max", "pos_neg_ratio_Lanczos", "condition_numbers_all_samples",
                                    "Lanczos_condition_number", "Lanczos_condition_number_variance"
                                ])

                        # Append row
                        with open(csv_file, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                i,
                                condition_number['condition_number'], condition_number['condition_number_abs'], condition_number["condition_number_variance"],
                                pos_lambda_min, pos_lambda_max,
                                neg_lambda_min, neg_lambda_max, pos_neg_ratio, condition_number['condition_numbers_all_samples'],
                                condition_number['lanczos_condition_number'], condition_number['lanczos_condition_number_variance']
                            ])
                    elif visualisation:
                        # Full caculation of the eigenvalues of the Hessian
                        condition_number = estimate_condition_number_full(self.m, self.data, num_batches=1, loss_fn=self.loss_fn, weight_decay=weight_decay)
                        condition_number_approx = estimate_effective_condition_number_reorth(self.m, self.data, loss_fn=self.loss_fn, weight_decay=weight_decay, lanczos_steps=200, n_samples=1)
                        self.condition_numbers.append(condition_number['condition_number'])
                        self.condition_number_variances.append(condition_number['condition_number_variance'])
                        true_eigs = np.array(condition_number["eigenvalues"])
                        sorted_true_eigs = np.sort(true_eigs)
                        ninety_fifth = sorted_true_eigs[int(0.95*len(sorted_true_eigs))]
                        ninety_nineth = sorted_true_eigs[int(0.99*len(sorted_true_eigs))]
                        true_eigs = true_eigs[np.isfinite(true_eigs)]  
                        eigs_abs = np.abs(true_eigs[true_eigs>10**(-10)])
                        eigs_abs = eigs_abs[np.isfinite(eigs_abs)]
                        
                        true_eigs_l = np.array(condition_number_approx["eigenvals_lanczos"])
                        true_eigs_r = np.array(condition_number_approx["eigenvals_rayleigh"])
                        true_eigs_l = true_eigs_l[np.isfinite(true_eigs_l)]  # clean
                        true_eigs_r = true_eigs_r[np.isfinite(true_eigs_r)]  # clean
                        eigs_abs_l = np.abs(true_eigs_l)
                        eigs_abs_r = np.abs(true_eigs_r)


                        # Split into pos/neg subspaces
                        pos_eigs = true_eigs[true_eigs > 0]
                        neg_eigs = true_eigs[true_eigs < 0]

                        pos_lambda_min = np.min(pos_eigs) if len(pos_eigs) > 0 else np.nan
                        pos_lambda_max = np.max(pos_eigs) if len(pos_eigs) > 0 else np.nan
                        neg_lambda_min = np.min(neg_eigs) if len(neg_eigs) > 0 else np.nan
                        neg_lambda_max = np.max(neg_eigs) if len(neg_eigs) > 0 else np.nan
                        pos_neg_ratio = len(pos_eigs) / len(neg_eigs)

                        # --- Save directory ---
                        save_dir = "eigenspectrum_visualisation/timestamp"
                        os.makedirs(save_dir, exist_ok=True)

                        # --- Plot both histograms side by side ---
                        plt.rcParams['font.size'] = 16
                        plt.rcParams['font.family'] = 'serif'
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                        # Left: abs eigenvalues (log scale)
                        if len(eigs_abs) > 0:
                            log_bins = np.logspace(np.log10(np.min(eigs_abs)), np.log10(np.max(eigs_abs)), 200)
                            axes[0].hist(eigs_abs, bins=log_bins, color="navy")
                            
                            axes[0].hist(eigs_abs_l, bins=log_bins, color="purple")
                           
                            axes[0].hist(eigs_abs_r, bins=log_bins, color="red")
                            axes[0].axvline(ninety_fifth, color="red", linestyle="--", linewidth=2, label="95th percentile", alpha=0.8)
                            axes[0].axvline(ninety_nineth, color="red", linestyle="--", linewidth=2, label="99th percentile", alpha=0.6)
                            axes[0].set_xscale("log")
                            axes[0].set_yscale("log")
                            axes[0].set_xlabel("Absolute value of eigenvalues")
                            axes[0].set_ylabel("Count")
                            axes[0].legend()
                            

                        # Right: raw eigenvalues
                        bins = np.linspace(np.min(true_eigs), np.max(true_eigs), 200)
                        axes[1].hist(true_eigs, bins=bins, color="navy", label="True eigenspectrum")
                        axes[1].hist(true_eigs_l, bins=bins, color="purple", label="Lanczos approximation")
                        axes[1].hist(true_eigs_r, bins=bins, color="red", label="Rayleigh approximation")
                        
                        axes[1].set_xlabel("Eigenvalues")
                        axes[1].set_ylabel("Count")
                        axes[1].set_yscale("log")
                        axes[1].legend()
                        

                        plt.tight_layout()
                        plt.savefig(f"{save_dir}/step_{i}.png", dpi=500)
                        plt.close(fig)

                        # --- CSV file inside save_dir ---
                        csv_file = os.path.join(save_dir, "eigenspectrum_stats.csv")

                        # If file doesn't exist, write header
                         # If file doesn't exist, write header
                        if not os.path.exists(csv_file):
                            with open(csv_file, mode="w", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    "iteration",
                                    "condition_number", "condition_number_abs", "condition_number_variance", "condition_number_threshold", "condition_number_abs_threshold",
                                    "pos_lambda_min", "pos_lambda_max",
                                    "neg_lambda_min", "neg_lambda_max", "pos_neg_ratio_Lanczos", "condition_numbers_all_samples"
                                ])

                        # Append row
                        with open(csv_file, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                i,
                                condition_number['condition_number'], condition_number['condition_number_abs'], condition_number["condition_number_variance"], condition_number['condition_number_threshold'], condition_number['condition_number_abs_threshold'],
                                pos_lambda_min, pos_lambda_max,
                                neg_lambda_min, neg_lambda_max, pos_neg_ratio, condition_number['condition_numbers_all_samples']
                            ])


                    else:
                        # Full caculation of the eigenvalues of the Hessian
                        condition_number = estimate_condition_number_full(self.m, self.data, num_batches=5, loss_fn=self.loss_fn, weight_decay=weight_decay)
                        
                        self.condition_numbers.append(condition_number['condition_number'])
                        self.condition_number_variances.append(condition_number['condition_number_variance'])
                        true_eigs = np.array(condition_number["eigenvalues"])
                        true_eigs = true_eigs[np.isfinite(true_eigs)]  
                        eigs_abs = np.abs(true_eigs[true_eigs>10**(-10)])
                        eigs_abs = eigs_abs[np.isfinite(eigs_abs)]
                        print(f"length of eigs: {len(true_eigs)}")
                        # Split into pos/neg subspaces
                        pos_eigs = true_eigs[true_eigs > 0]
                        neg_eigs = true_eigs[true_eigs < 0]

                        pos_lambda_min = np.min(pos_eigs) if len(pos_eigs) > 0 else np.nan
                        pos_lambda_max = np.max(pos_eigs) if len(pos_eigs) > 0 else np.nan
                        neg_lambda_min = np.min(neg_eigs) if len(neg_eigs) > 0 else np.nan
                        neg_lambda_max = np.max(neg_eigs) if len(neg_eigs) > 0 else np.nan
                        pos_neg_ratio = len(pos_eigs) / len(neg_eigs)

                        # --- Save directory ---
                        save_dir = "true_eigenspectrum/timestamp"
                        os.makedirs(save_dir, exist_ok=True)

                        # --- Plot both histograms side by side ---
                        plt.rcParams['font.size'] = 16
                        plt.rcParams['font.family'] = 'serif'
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                        # Left: abs eigenvalues (log scale)
                        if len(eigs_abs) > 0:
                            log_bins = np.logspace(np.log10(np.min(eigs_abs)), np.log10(np.max(eigs_abs)), 100)
                            axes[0].hist(eigs_abs, bins=log_bins, color="navy")
                            axes[0].set_xscale("log")
                            axes[0].set_xlabel("Absolute value of eigenvalues")
                            axes[0].set_ylabel("Count")
                            
                        # Right: raw eigenvalues
                        axes[1].hist(true_eigs, bins=200, color="navy")
                        axes[1].set_xlabel("Eigenvalues")
                        axes[1].set_ylabel("Count")
                        axes[1].set_yscale("log")
                        

                        plt.tight_layout()
                        plt.savefig(f"{save_dir}/step_{i}.png", dpi=500)
                        plt.close(fig)

                        # --- CSV file inside save_dir ---
                        csv_file = os.path.join(save_dir, "eigenspectrum_stats.csv")

                        # If file doesn't exist, write header
                         # If file doesn't exist, write header
                        if not os.path.exists(csv_file):
                            with open(csv_file, mode="w", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    "iteration",
                                    "condition_number", "condition_number_abs", "condition_number_variance", "condition_number_threshold", "condition_number_abs_threshold",
                                    "pos_lambda_min", "pos_lambda_max",
                                    "neg_lambda_min", "neg_lambda_max", "pos_neg_ratio_Lanczos", "condition_numbers_all_samples"
                                ])

                        # Append row
                        with open(csv_file, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                i,
                                condition_number['condition_number'], condition_number['condition_number_abs'], condition_number["condition_number_variance"], condition_number['condition_number_threshold'], condition_number['condition_number_abs_threshold'],
                                pos_lambda_min, pos_lambda_max,
                                neg_lambda_min, neg_lambda_max, pos_neg_ratio, condition_number['condition_numbers_all_samples']
                            ])
                        
    
            if self.track_variance:
                # track variance at every 1000 steps through training
                grad_variance = compute_true_minibatch_variance(self.m, self.train_loader, loss_fn=self.loss_fn)
                self.grad_variances.append(grad_variance["gradient_noise_variance"])
                
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
                    
                    if not torch.isfinite(loss):
                    # Raising TrialPruned tells Optuna to stop this trial and record it as pruned.
                    # This prevents the entire script from crashing.
                        print("WARNING: Loss is NaN or infinite. Pruning trial.")
                        raise optuna.exceptions.TrialPruned("Loss became non-finite.")
                    
                    loss.backward()
                    # Gradient check for shampoo
                    for p in self.m.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            print("WARNING: Found NaN or infinite gradients. Pruning trial.")
                            raise optuna.exceptions.TrialPruned("Gradients became non-finite.")
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
                    # Always use the same step methods
                    if self.requires_prev_model(o):
                        loss = o.step_with_prev(_closure, _prev_closure)
                    else:
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
        return validator.ema_states.min().item(), self.m, torch_hist[-1].item(), self.current_losses, self.test_accuracies, self.grad_variances, self.condition_numbers, self.condition_number_variances, self.step_counter, runtime
    
    def objective(self, params):
        self.attempt += 1
        target, model, loss, step_losses, test_accuracies, grad_variances, condition_numbers, condition_number_variances, step_counter, runtime = self._inner(params)
        self.loss = loss
        #if self.best_loss is None or loss < self.best_loss or not np.isfinite(self.best_loss):
        self.best_loss = loss
        self.best_at = self.attempt
        self.avg = None
        self.best_losses = step_losses.copy() 
        self.test_accuracies = test_accuracies.copy()
        self.grad_variances = grad_variances.copy()
        self.condition_numbers = condition_numbers.copy()
        self.condition_number_variances = condition_number_variances.copy()
        self.memory_usage = getattr(self, 'current_memory_usage', 0)/ (1024**2)
        self.step_counter = step_counter
        self.runtime = runtime

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
    estimate_condition_number: bool = True,
    test_loader=None,
    train_loader=None,
    track_variance=False,
    test_optimizer_implementation=False,
    runtime_limit: int = 3600 * 24, # Default runtime limit is a day
    step_hint: int = 0,
    use_fixed_hyperparams: bool = False
):
    opt_name = opt
    

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
        train_loader,
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
        7: 0.05,       # warmup_factor
        8: 0.1,        # dropout_rate
        9: 0.4,        # label_smoothing
        10: 0.0,       # one_minus_momentum
        11: False,     # use_momentum
        12: 1024,      # max_preconditioner_dim
        13: 100,       # precondition_frequency
        14: 100,       # start_preconditioning_step (was -1)
        15: 0,         # inv_root_override
        16: 1.0,       # exponent_multiplier
        17: "ADAM",    # grafting_type
        18: 1e-8,      # grafting_epsilon
        19: False,     # use_normalized_grafting
        20: "FP64",    # communication_dtype
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
            'condition_number_variances': obj.condition_number_variances.copy() if hasattr(obj, 'condition_number_variances') else [],
            'runtime': obj.runtime if hasattr(obj, 'runtime') else [],
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
                    train_loader,
                    track_variance,
                    runtime_limit,
                    step_hint,
                    **kwargs,
                )
                out = obj.objective(full_params)
                
                best_test_accuracy = max(obj.test_accuracies.copy())
                
                trial_result = {
                    'trial_number': hyperparam_counter,
                    'tuned_indices': tuned_indices,
                    'original_json_params': hyperparams.copy(),
                    'params': params_dict,
                    'losses': obj.current_losses.copy() if hasattr(obj, 'current_losses') else [],
                    'test_accuracies': obj.test_accuracies.copy() if hasattr(obj, 'test_accuracies') else [],
                    'grad_variances': obj.grad_variances.copy() if hasattr(obj, 'grad_variances') else [],
                    'condition_numbers': obj.condition_numbers.copy() if hasattr(obj, 'condition_numbers') else [],
                    'condition_number_variances': obj.condition_number_variances.copy() if hasattr(obj, 'condition_number_variances') else [],
                    'runtime': obj.runtime if hasattr(obj, 'runtime') else 0,
                    'steps': obj.step_counter if hasattr(obj, 'step_counter') else steps,
                }
                print(trial_result) 
                all_trial_results.append(trial_result)   
                return 1-best_test_accuracy
           
        n_trials = 5
        # If trials is less than 5, only do trials many runs
        if trials < n_trials:
            n_trials = trials

        study.optimize(_optuna_objective, n_trials)
           
    else:
        tuning_freq = False
        print("Using Quasi-random Sampling")
        best_test_accuracies = []
        # Specify exactly which parameter indices to tune - you can change this list
        # Example: tune learning_rate, one_minus_beta3, one_minus_shampoo_beta
        if tuning_freq and opt_name == "ExternalDistributedShampoo":
            tuned_indices = [0, 1, 2, 13, 5] 
            print("Tuning precond. freq. instead of beta2")
            search_ranges = {
                0: (1e-4, 1e-2, True),   # learning_rate (log scale)
                1: (1e-3, 0.5, True),    # one_minus_beta1 (log scale)
                2: (1e-3, 0.5, True),    # one_minus_beta2 (log scale)
                13: (10, 200, True),    # Precond. freq. (log scale)
                5: (1e-5, 1e-2, True),   # weight_decay (log scale)
            }

        elif opt_name == "SGD":
            tuned_indices = [0, 1, 5] 
            # Define search ranges for each parameter index
            search_ranges = {
                0: (1e-2, 10, True),   # learning_rate (log scale)
                1: (1e-3, 1, True),    # one_minus_beta1 (log scale)
                2: (1e-3, 0.5, True),    # one_minus_beta2 (log scale)
                3: (1e-3, 0.5, True),    # one_minus_beta3 (log scale)
                4: (1e-3, 0.5, True),      # one_minus_shampoo_beta (log scale)
                5: (1e-7, 1e-2, True),   # weight_decay (log scale)
            }

        elif opt_name == "ExternalDistributedShampoo":
            print("Using lr default ranges")
            # Define search ranges for each parameter index
            tuned_indices = [0, 1, 2, 5]  
            search_ranges = {
                0: (1e-4, 1e-2, True),   # learning_rate (log scale)
                1: (1e-3, 0.5, True),    # one_minus_beta1 (log scale)
                2: (1e-3, 0.5, True),    # one_minus_beta2 (log scale)
                3: (1e-3, 0.5, True),    # one_minus_beta3 (log scale)
                4: (1e-3, 0.5, True),      # one_minus_shampoo_beta (log scale)
                5: (1e-5, 1e-0, True),   # weight_decay (log scale)
            }
        
        else:
            print("Using higher lr default ranges")
            # Define search ranges for each parameter index
            tuned_indices = [0, 1, 2, 5]  
            search_ranges = {
                0: (1e-3, 1e-1, True),   # learning_rate (log scale)
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
            nonlocal best_test_accuracies
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
                train_loader,
                track_variance,
                runtime_limit,
                step_hint,
                best_test_accuracies,
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
            
            
            try:
                out = obj.objective(full_params)
                
                # Create params dict for logging
                params_dict = {f"param_{i}_{param_names[i]}": val for i, val in zip(tuned_indices, tuned_values)}
                best_test_accuracy = max(obj.test_accuracies.copy())
                trial_result = {
                    'tuned_indices': tuned_indices,
                    'params': params_dict,
                    'full_params': full_params,
                    'losses': obj.current_losses.copy() if hasattr(obj, 'current_losses') else [],
                    'test_accuracies': obj.test_accuracies.copy() if hasattr(obj, 'test_accuracies') else [],
                    'grad_variances': obj.grad_variances.copy() if hasattr(obj, 'grad_variances') else [],
                    'condition_numbers': obj.condition_numbers.copy() if hasattr(obj, 'condition_numbers') else [],
                    'condition_number_variances': obj.condition_number_variances.copy() if hasattr(obj, 'condition_number_variances') else [],
                    'runtime': obj.runtime if hasattr(obj, 'runtime') else [],
                    'steps': obj.step_counter if hasattr(obj, 'step_counter') else steps,
                }
                print(trial_result)
                all_trial_results.append(trial_result)   
                if (
                    best_test_accuracies == [] 
                    or (best_test_accuracy > max(best_test_accuracies) 
                        and len(trial_result["test_accuracies"]) == steps // group)
                ):
                    best_test_accuracies = trial_result["test_accuracies"]
            except Stop:
                pass
            return 1-best_test_accuracy
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
        final_condition_number_variances = best_trial['condition_number_variances']
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

    print(
        f"Took: {end_time - start_time} | Highest_accuracy: {best_test_accuracy} | Winning_runtime: {final_runtime} | Trials: {obj.attempt + 1} | "
        f"Params: {opt.__name__}_{winning_params} | "
        f"loss_trajectory: {final_losses} | test_accuracies: {final_test_accuracies} | "
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


def estimate_condition_number_grid_search(model, data_fn, loss_fn=torch.nn.functional.cross_entropy, timout=3600, weight_decay=1e-4):
    n_samples = [5,10,20,50,100,200,500,1000]
    results = [0]

    for n in n_samples:
        result = estimate_condition_number_hvp(model, data_fn, n_probes=int(n*2), n_samples=n, loss_fn=loss_fn, timout=timout, weight_decay=weight_decay)
        condition_number = result["condition_number"]
        reached_timout = result["reached_timout"]
        sorted_eigs = sorted(result['eigenvals'])
        print(f"Bottom 20 for n={n}: {sorted_eigs[0:20]}")
        print(f"Top 20 for n={n}: {sorted_eigs[-21:-1]}")
        print(f"Median for n={n}: {sorted_eigs[int(len(sorted_eigs)/2)]}")
        if reached_timout:
            return results[-1]
        else:
            results.append(condition_number)
    
    return float(results[-1])


def estimate_condition_number_hvp(model, data_fn, n_probes=20, n_samples=50, loss_fn=None, timout=3600, weight_decay=1e-4):
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
                
                # Rayleigh quotient:
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
        result = {
        "condition_number" : 0,
        "eigenvals" : eigenvals,
        "reached_timout" : False
        } 
        return result
    
    # Condition number approximation
    abs_eigenvals = np.abs(eigenvals)
    max_eig = np.max(abs_eigenvals)
    min_eig = np.min(abs_eigenvals)

    # Convert to python float for better printing visibility 
    eigenvals = [float(ev) for ev in eigenvals if np.isfinite(ev)]

    result = {
        "condition_number" : (max_eig ) / (min_eig ),
        "lambda_max": max_eig,
        "eigenvals" : eigenvals,
        "reached_timout" : reached_timout
    } 
    return result


def evaluate_test_accuracy(model, test_loader):
    """Evaluate model performance with automatic task detection (robust)."""
    was_training = model.training
    model.eval()

    try:
        # Grab one batch safely
        sample_batch = next(iter(test_loader))
        # Detect format: FastMRI returns 5 items, classification returns 2
        if isinstance(sample_batch, (list, tuple)):
            if len(sample_batch) >= 5:
                # FastMRI dataset: data, target, mean, std, volume_max
                sample_data, sample_target = sample_batch[0], sample_batch[1]
            else:
                # Classification dataset: data, target
                sample_data, sample_target = sample_batch
        else:
            raise ValueError(f"Unexpected batch format: {type(sample_batch)}")

        # Move to GPU if available
        if torch.cuda.is_available():
            sample_data, sample_target = sample_data.cuda(), sample_target.cuda()

        with torch.no_grad():
            sample_output = model(sample_data)
     
        # Detect task type
        task_type = detect_task_type(sample_output, sample_target, model)

        if task_type == 'fastmri':
            result = evaluate_fastmri_ssim_internal(model, test_loader)
        else:
            result = evaluate_classification_accuracy(model, test_loader)

    except Exception as e:
        print(f"Error in evaluation: {e}")
        result = 0.0

    # Restore original training state
    model.train(was_training)
    return result



def detect_task_type(output, target, model):
    """Detect whether this is a FastMRI reconstruction task or classification"""
    
    model_name = model.__class__.__name__.lower()
    if 'unet' in model_name:
        return 'fastmri'
    
    # Ensure output and target have at least 2 spatial dims
    output_spatial_dims = len([d for d in output.shape[2:] if d > 1])
    target_spatial_dims = len([d for d in target.shape[2:] if d > 1])
    
    if output_spatial_dims == 2 and target_spatial_dims == 2:
        if output.shape[-2:] == target.shape[-2:]:
            return 'fastmri'
    
    if output.dtype == torch.float32 and target.dtype == torch.float32:
        output_range = output.max() - output.min()
        target_range = target.max() - target.min()
        
        # Safe unique count
        target_unique = target.unique()
        num_unique = target_unique.numel() if target_unique.numel() > 0 else 0
        
        if (0.1 < output_range < 100) and (0.1 < target_range < 100) and num_unique > 10:
            return 'fastmri'
    
    if len(output.shape) == 4 and len(target.shape) == 4:
        if output.shape[2] > 64 and output.shape[3] > 64:
            if output.shape == target.shape:
                return 'fastmri'
    
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
            
            # Ensure output & target are at least 2D
            if output.ndim < 2:
                output = output.view(1, -1)
            if target.ndim < 2:
                target = target.view(1, -1)
            
            if output.ndim == 3:  # (batch, seq_len, vocab_size)
                pred_flat = output.argmax(dim=2).reshape(-1)   # take vocab dimension
                target_flat = target.reshape(-1)
            else:
                pred_flat = output.argmax(dim=1).view(-1)
                target_flat = target.view(-1)
            correct += pred_flat.eq(target_flat).sum().item()
            total += target_flat.numel()

    
    return correct / total

from skimage.metrics import structural_similarity as skimage_ssim

def evaluate_fastmri_ssim_internal(model, test_loader):
    total_ssim = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Unpack with fallback
            if isinstance(batch, tuple) and len(batch) == 5:
                data, target, mean, std, volume_max = batch
            else:
                data, target = batch, batch  # fallback
                mean = torch.tensor(0.0)
                std = torch.tensor(1.0)
                volume_max = torch.tensor(1.0)
            
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            
            # Make sure output is at least 4D [B,C,H,W]
            while output.ndim < 4:
                output = output.unsqueeze(0)
            while target.ndim < 4:
                target = target.unsqueeze(0)
            
            # Denormalize
            output = output * std + mean
            target = target * std + mean
            output = torch.clamp(output, 0, volume_max)
            target = torch.clamp(target, 0, volume_max)
            
            batch_size = output.shape[0]
            for j in range(batch_size):
                pred_img = output[j,0].cpu().numpy()
                target_img = target[j,0].cpu().numpy()
                try:
                    ssim_val = skimage_ssim(pred_img, target_img, data_range=volume_max.item())
                except Exception as e:
                    print(f"Error in skimage_ssim: {e}")
                    ssim_val = 0.0
                total_ssim += ssim_val
                total_samples += 1

    
    return total_ssim / max(1, total_samples)


def structural_similarity(im1, im2, data_range=1.0, win_size=7, k1=0.01, k2=0.03):
    try:
        if im1.numel() < win_size * win_size or im2.numel() < win_size * win_size:
            # fallback
            im1_flat = im1.flatten()
            im2_flat = im2.flatten()
            correlation = F.cosine_similarity(im1_flat.unsqueeze(0), im2_flat.unsqueeze(0))
            return float(correlation.item())
        
        filter_func = functools.partial(_uniform_filter, size=win_size)
        try:
            num_points = win_size ** len(im1.shape)
        except:
            print(f"im1 shape: {im1.shape}")
            num_points = win_size ** 2

        cov_norm = num_points / (num_points - 1)

        ux = filter_func(im1)
        uy = filter_func(im2)

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

        pad = (win_size - 1) // 2
        if s.shape[0] > 2*pad and s.shape[1] > 2*pad:
            return float(torch.mean(s[pad:-pad, pad:-pad]).item())
        else:
            return float(torch.mean(s).item())

    except Exception as e:
        print(f"SSIM computation failed: {e}, using fallback")
        return float(F.cosine_similarity(im1.flatten().unsqueeze(0),
                                        im2.flatten().unsqueeze(0)).item())


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
    
import time
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

@torch.no_grad()
def _clone_params(params):
    return [p.detach().clone() for p in params]

def _auto_loss_fn(output, y):
    # Match your heuristic
    try:
        if output.dim() > 2 or (y.dim() == output.dim() and y.shape == output.shape):
            return torch.nn.functional.mse_loss
        else:
            return torch.nn.functional.cross_entropy
    except:
        return torch.nn.functional.cross_entropy

def _prepare_batch(data_fn, device):
    inp, target = data_fn()

    return inp.to(device), target.to(device)

def _compute_loss(model, data_fn, loss_fn, device, dtype=torch.float64):
    model.zero_grad(set_to_none=True)
    if data_fn()[0] is not None:
        x, y = _prepare_batch(data_fn, device)
    else: 
        x, y = data_fn()
    # Forward
      # Compute loss with previous model
    loss = model() if x is None else model(x)
    
    # Auto-detect loss if needed
    if loss_fn is not None:
        loss = loss_fn(loss, y)
    return loss, loss_fn

from torch.func import functional_call
from torch.nn.utils import parameters_to_vector
import numpy as np
import time
import torch

def estimate_condition_number_full_old(
    model,
    data_fn,
    loss_fn=None,
    device=None,
    timeout=1800,
    max_params=150_000,
    use_double=False,
    return_eigs=True,
):
    t0 = time.time()
    reached_timeout = False

    if device is None:
        device = next(model.parameters(), torch.tensor([])).device
        if device.type not in ("cuda", "mps"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- collect trainable params metadata (ORDER + SHAPES) ---
    named = [(n, p) for (n, p) in model.named_parameters() if p.requires_grad]
    if len(named) == 0:
        return {
            'condition_number': float('inf'),
            'lambda_max': float('nan'),
            'lambda_min_pos': None,
            'num_params': 0,
            'num_pos': 0,
            'num_neg': 0,
            'num_zero': 0,
            'reached_timeout': False,
            'dtype_used': next(model.parameters(), torch.empty(())).dtype if any(True for _ in model.parameters()) else torch.float32,
        }

    names = [n for (n, _) in named]
    params = [p for (_, p) in named]
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    n = sum(numels)

    if n > max_params:
        raise RuntimeError(
            f"Refusing to build a dense {n}x{n} Hessian (>{max_params} params). "
            "Use a smaller model/batch or increase max_params explicitly."
        )

    # dtype/device handling
    dtype_used = torch.float64 if use_double else params[0].dtype
    model = model.to(device=device)
    if use_double:
        model = model.to(dtype=torch.float64)

    # --- FIXED BATCH (avoid calling data_fn multiple times) ---
    x0, y0 = _prepare_batch(data_fn, device)

    # choose loss
    if loss_fn is None:
        loss_scalar = model(x0)
        loss_fn = _auto_loss_fn(loss_scalar, y0)

    # --- build flat initial vector (not used for values, just shape) ---
    with torch.no_grad():
        flat0 = parameters_to_vector(params).to(dtype_used)

    # unflatten helper that **creates differentiable views from theta**
    def unflatten(theta):
        splits = list(torch.split(theta, numels))
        tensors = [t.view(s) for t, s in zip(splits, shapes)]
        return {n: t for n, t in zip(names, tensors)}

    # closure f(theta): NO copy_, NO vector_to_parameters
    def f(theta):
        # theta is the ONLY source of parameters here
        pmap = unflatten(theta)
        out = functional_call(model, pmap, (x0,))
        loss = loss_fn(out, y0)
        return loss

    # build theta with grad
    theta0 = flat0.detach().clone().requires_grad_(True)

    # eigens via dense Hessian of f wrt theta
    from torch.autograd.functional import hessian as autograd_hessian

    class _TimeoutFlag(Exception): pass
    def _check_timeout():
        nonlocal reached_timeout
        if time.time() - t0 > timeout:
            reached_timeout = True
            raise _TimeoutFlag()

    was_training = model.training
    model.eval()  # deterministic forward for Hessian

    try:
        _check_timeout()
        H = autograd_hessian(f, theta0)  # shape [n, n]
        _check_timeout()
    except _TimeoutFlag:
        model.train(was_training)
        return {
            'condition_number': float('inf'),
            'lambda_max': float('nan'),
            'lambda_min_pos': None,
            'num_params': n,
            'num_pos': 0,
            'num_neg': 0,
            'num_zero': 0,
            'reached_timeout': True,
            'dtype_used': dtype_used,
        }
    finally:
        model.train(was_training)

    # symmetrize, move to CPU, eig
    H = (H + H.transpose(0, 1)) * 0.5
    evals = torch.linalg.eigvalsh(H.cpu())
    evals_np = evals.numpy()

    lam_abs_max = float(np.max(np.abs(evals_np))) if evals_np.size else 0.0
    tol = max(1e-12, 1e-10 * (1.0 + lam_abs_max))

    num_pos = int(np.sum(evals_np >  tol))
    num_neg = int(np.sum(evals_np < -tol))
    num_zero = int(evals_np.size - num_pos - num_neg)

    pos_eigs = np.abs(evals_np)
    lambda_max = float(np.max(pos_eigs)) if pos_eigs.size else float('nan')
    lambda_min_pos = float(np.min(pos_eigs)) if pos_eigs.size else None
    kappa = lambda_max / max(lambda_min_pos, 1e-300)

    result = {
        'condition_number': float(kappa),
        'lambda_max': lambda_max,
        'lambda_min_pos': lambda_min_pos,
        'num_params': n,
        'num_pos': num_pos,
        'num_neg': num_neg,
        'num_zero': num_zero,
        'reached_timeout': reached_timeout,
        'dtype_used': dtype_used,
    }
    if return_eigs:
        result['eigenvalues'] = evals_np
    return result

def compute_true_minibatch_variance(model, trainloader, loss_fn=torch.nn.functional.cross_entropy, device=None):
    """
    Compute variance between minibatch gradients and the full-batch gradient.

    Returns a dict with relative variance (normalized by the full gradient norm).
    """
    if device is None:
        device = next(model.parameters()).device
    model.train()

    # ---- Step 1: compute full-batch gradient ----
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    n_samples = 0
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_fn(output, y) * x.size(0)  # weight by batch size
        loss.backward()
        total_loss += loss.item()
        n_samples += x.size(0)

    full_grad = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]) / n_samples
    full_grad_np = full_grad.cpu().numpy()
    full_grad_norm_sq = np.sum(full_grad_np ** 2)

    # ---- Step 2: compute minibatch gradient deviations ----
    deviations = []
    for x, y in trainloader:
        model.zero_grad(set_to_none=True)
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_fn(output, y)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        flat_grad = torch.cat([g.detach().flatten() for g in grads])
        diff = flat_grad.cpu().numpy() - full_grad_np
        deviations.append(diff)

    deviations = np.stack(deviations)  # shape [num_batches, num_params]

    # ---- Step 3: compute relative variance ----
    grad_noise_variance_total = float(np.mean(np.sum(deviations ** 2, axis=1)))
    relative_variance = grad_noise_variance_total / (full_grad_norm_sq + 1e-12)

    result = {
        "gradient_noise_variance": float(relative_variance),  # now normalized
        "num_batches": len(trainloader),
        "num_params": full_grad_np.shape[0],
    }
    return result

def estimate_condition_number_full(
    model,
    data_fn=None,
    trainloader=None,   
    num_batches=1,     
    loss_fn=None,
    device=None,
    timeout=1800,
    max_params=150_000,
    use_double=False,
    return_eigs=True,
    weight_decay=1e-4,
    cutoff=100
):
    """
    Compute condition number of the Hessian averaged over multiple batches.
    Either supply data_fn() (single batch sampler) or a trainloader.
    """
    results = []
    results_abs = []
    results_threshold = []
    results_abs_threshold = []
    max_eigs = []
    eigs_all = [] if return_eigs else None

    # pick data source
    if trainloader is not None:
        batch_iter = iter(trainloader)
        get_batch = lambda: next(batch_iter)
    elif data_fn is not None:
        get_batch = data_fn
    else:
        raise ValueError("Must supply either data_fn or trainloader")

    for b in range(num_batches):
        try:
            x0, y0 = get_batch()
        except StopIteration:
            break  # exhausted loader

        # --- per-batch estimate ---
        
        res = _estimate_condition_number_single_batch(
            model=model,
            x0=x0,
            y0=y0,
            loss_fn=loss_fn,
            device=device,
            timeout=timeout,
            max_params=max_params,
            use_double=use_double,
            return_eigs=return_eigs,
            weight_decay=weight_decay,
            cutoff=cutoff
        )

        results.append(res["condition_number"])
        results_abs.append(res["condition_number_abs"])
        results_threshold.append(res["condition_number_threshold"])
        results_abs_threshold.append(res["condition_number_abs_threshold"])
        max_eigs.append(res["lambda_max"])
        if return_eigs:
            eigs_all.extend(res.get("eigenvalues", []))

    if len(results) == 0:
        return {"condition_number": float("nan")}

    # aggregate
    avg_kappa = float(np.mean(results))
    avg_kappa_abs = float(np.mean(results_abs))
    avg_kappa_threshold = float(np.mean(results_threshold))
    avg_kappa_abs_threshold = float(np.mean(results_abs_threshold))
    avg_max = float(np.mean(max_eigs))
    out = {
        "condition_number": avg_kappa,
        "condition_number_abs": avg_kappa_abs,
        "condition_number_threshold": avg_kappa_threshold,
        "condition_number_abs_threshold": avg_kappa_abs_threshold,
        "condition_numbers_all_samples": results,
        "condition_number_variance": float(np.var(results)),
        "condition_number_variance_threshold": float(np.var(results_threshold)),
        "lambda_max": avg_max,
        "num_batches": len(results),
        "per_batch": results,
    }
    if return_eigs:
        out["eigenvalues"] = np.array(eigs_all)
    return out


def _estimate_condition_number_single_batch(
    model,
    x0,
    y0,
    loss_fn=None,
    device=None,
    timeout=1800,
    max_params=150_000,
    use_double=True,
    return_eigs=True,
    weight_decay=1e-4,
    cutoff=100
):
    """
    Estimate Hessian condition number for a *single fixed batch* (x0, y0).
    """

    t0 = time.time()
    reached_timeout = False

    if device is None:
        device = next(model.parameters(), torch.tensor([])).device
        if device.type not in ("cuda", "mps"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- collect trainable params metadata (ORDER + SHAPES) ---
    named = [(n, p) for (n, p) in model.named_parameters() if p.requires_grad]
    if len(named) == 0:
        return {
            "condition_number": float("inf"),
            "lambda_max": float("nan"),
            "lambda_min_pos": None,
            "num_params": 0,
            "num_pos": 0,
            "num_neg": 0,
            "num_zero": 0,
            "reached_timeout": False,
            "dtype_used": next(model.parameters(), torch.empty(())).dtype
            if any(True for _ in model.parameters())
            else torch.float32,
        }

    names = [n for (n, _) in named]
    params = [p for (_, p) in named]
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    n = sum(numels)

    if n > max_params:
        raise RuntimeError(
            f"Refusing to build a dense {n}x{n} Hessian (>{max_params} params). "
            "Use a smaller model/batch or increase max_params explicitly."
        )

    # dtype/device handling
    dtype_used = torch.float64 if use_double else params[0].dtype
    model = model.to(device=device)
    if use_double:
        model = model.to(dtype=torch.float64)

    # --- move batch to device ---
    x0, y0 = x0.to(device), y0.to(device)

    # choose loss
    if loss_fn is None:
        loss_scalar = model(x0)
        loss_fn = _auto_loss_fn(loss_scalar, y0)

    # --- build flat initial vector (not used for values, just shape) ---
    with torch.no_grad():
        flat0 = parameters_to_vector(params).to(dtype_used)

    # unflatten helper that **creates differentiable views from theta**
    def unflatten(theta):
        splits = list(torch.split(theta, numels))
        tensors = [t.view(s) for t, s in zip(splits, shapes)]
        return {n: t for n, t in zip(names, tensors)}

    # closure f(theta): NO copy_, NO vector_to_parameters
    def f(theta):
        pmap = unflatten(theta)
        out = functional_call(model, pmap, (x0,))
        loss = loss_fn(out, y0)
        return loss

    # build theta with grad
    theta0 = flat0.detach().clone().requires_grad_(True)

    # eigens via dense Hessian of f wrt theta
    from torch.autograd.functional import hessian as autograd_hessian

    class _TimeoutFlag(Exception):
        pass

    def _check_timeout():
        nonlocal reached_timeout
        if time.time() - t0 > timeout:
            reached_timeout = True
            raise _TimeoutFlag()

    was_training = model.training
    model.eval()  # deterministic forward for Hessian

    try:
        _check_timeout()
        H = autograd_hessian(f, theta0)  # shape [n, n]
        _check_timeout()
    except _TimeoutFlag:
        model.train(was_training)
        return {
            "condition_number": float("inf"),
            "lambda_max": float("nan"),
            "lambda_min_pos": None,
            "num_params": n,
            "num_pos": 0,
            "num_neg": 0,
            "num_zero": 0,
            "reached_timeout": True,
            "dtype_used": dtype_used,
        }
    finally:
        model.train(was_training)

    # symmetrize, move to CPU, eig
    H = (H + H.transpose(0, 1)) * 0.5
    evals = torch.linalg.eigvalsh(H.cpu())
    evals_np = evals.numpy()

    lam_abs_max = float(np.max(np.abs(evals_np))) if evals_np.size else 0.0
    tol = max(1e-12, 1e-10 * (1.0 + lam_abs_max))

    num_pos = int(np.sum(evals_np > tol))
    num_neg = int(np.sum(evals_np < -tol))
    num_zero = int(evals_np.size - num_pos - num_neg)


    # robust top-end estimate and requested lambda_min definition
    evals_pos = evals[evals > 0]
    evals_sorted = np.sort(evals_pos)
    evals_over_threshold = evals_sorted[evals_sorted>(10**(-6))]
    ninetyth_percentile = int(0.9 * len(evals_sorted))
    lambda_max = float(np.max(evals_sorted)) 
    # Take the top 10 percent of eigenvalues to approximate the right hand tail of the distribution
    lambda_min = float(np.median(evals_sorted[ninetyth_percentile : len(evals_sorted)]))  
    lambda_min_threshold = float(np.median(evals_over_threshold)) 

    kappa_pos_subspace = lambda_max / lambda_min
    kappa_pos_threshold = lambda_max / lambda_min_threshold

    
    abs_evals_sorted = np.sort(np.abs(evals))
    abs_evals_threshold = np.sort(np.abs(evals[evals>(10**(-6))]))
    lambda_max = float(np.max(abs_evals_sorted))
    lambda_min = float(np.median(abs_evals_sorted))  
    lambda_min_threshold = float(np.median(abs_evals_threshold))  
    
    kappa_abs = lambda_max / lambda_min
    kappa_abs_threshold = lam_abs_max / lambda_min_threshold

    result = {
        "condition_number": kappa_pos_subspace,
        "condition_number_abs": kappa_abs,
        "condition_number_threshold": kappa_pos_threshold,
        "condition_number_abs_threshold": kappa_abs_threshold,
        "lambda_max": lambda_max,
        "lambda_min_pos": lambda_min,
        "num_params": n,
        "num_pos": num_pos,
        "num_neg": num_neg,
        "num_zero": num_zero,
        "reached_timeout": reached_timeout,
        "dtype_used": dtype_used,
    }
    if return_eigs:
        result["eigenvalues"] = evals_np

    return result


def _estimate_condition_number_single_batch_slq(
    model,
    x0,
    y0,
    loss_fn=None,
    device=None,
    timeout=1800,
    use_double=False,
    weight_decay=1e-4,
    lanczos_steps=50,
    cutoff=100,
):
    """
    Estimate 'effective' Hessian condition number using stochastic Lanczos quadrature (SLQ).
    - Uses HVPs instead of building the dense Hessian.
    - Returns kappa_eff = lambda_max / (lambda_percentile + wd).
    """

    t0 = time.time()
    reached_timeout = False

    if device is None:
        device = next(model.parameters(), torch.tensor([])).device
        if device.type not in ("cuda", "mps"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # collect params
    named = [(n, p) for (n, p) in model.named_parameters() if p.requires_grad]
    names = [n for (n, _) in named]
    params = [p for (_, p) in named]
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    n = sum(numels)

    dtype_used = torch.float64 if use_double else params[0].dtype
    model = model.to(device=device)
    if use_double:
        model = model.to(dtype=torch.float64)

    x0, y0 = x0.to(device), y0.to(device)

    # choose loss
    if loss_fn is None:
        loss_scalar = model(x0)
        loss_fn = _auto_loss_fn(loss_scalar, y0)

    with torch.no_grad():
        flat0 = parameters_to_vector(params).to(dtype_used)

    # unflatten helper
    def unflatten(theta):
        splits = list(torch.split(theta, numels))
        tensors = [t.view(s) for t, s in zip(splits, shapes)]
        return {n: t for n, t in zip(names, tensors)}

    def closure(theta):
        pmap = unflatten(theta)
        out = functional_call(model, pmap, (x0,))
        return loss_fn(out, y0)

    # build theta with grad
    theta0 = flat0.detach().clone().requires_grad_(True)

    # HVP function with weight decay
    def hvp_fn(v):
        loss = closure(theta0)
        grads = torch.autograd.grad(loss, theta0, create_graph=True)[0]
        grad_v = torch.dot(grads, v)
        Hv = torch.autograd.grad(grad_v, theta0, retain_graph=False)[0]
        return Hv

    # ---- Lanczos algorithm ----
    def lanczos(hvp_fn, dim, m):
        # start with random unit vector
        q = torch.randn(dim, device=device, dtype=dtype_used)
        q = q / q.norm()
        Q = []
        alpha, beta = [], []
        prev_q = torch.zeros_like(q)

        for j in range(m):
            if time.time() - t0 > timeout:
                break
            z = hvp_fn(q)
            a = torch.dot(q, z).item()
            z = z - a * q - (beta[-1] * prev_q if j > 0 else 0)
            b = z.norm().item()
            alpha.append(a)
            beta.append(b)
            Q.append(q)
            if b < 1e-10:
                break
            prev_q, q = q, z / b

        T = np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1)
        return np.linalg.eigvalsh(T)

    ritz_vals = lanczos(hvp_fn, n, lanczos_steps)

    if len(ritz_vals) == 0:
        return {
            "condition_number": float("nan"),
            "lambda_max": None,
            "lambda_percentile": None,
            "reached_timeout": reached_timeout,
        }

    lambda_max = float(np.max(ritz_vals))
    # choose effective lower eigenvalue as percentile of positive spectrum
    pos_vals = np.sort(np.abs(ritz_vals))

   # cutoff_index = int(len(pos_vals) * percentile / 100.0)
   # cutoff_index = min(max(cutoff_index, 0), len(pos_vals) - 1)
    cutoff_index = -cutoff
    lambda_p = float(pos_vals[cutoff_index])

    kappa_eff = (lambda_max) / lambda_p 

    return {
        "condition_number": kappa_eff,
        "lambda_max": lambda_max,
        "lambda_percentile": lambda_p,
        "num_params": n,
        "reached_timeout": reached_timeout,
        "dtype_used": dtype_used,
        "eigenvals": ritz_vals,
    }

def estimate_effective_condition_number(
    model,
    data_fn,
    loss_fn=None,
    device=None,
    weight_decay=1e-4,
    lanczos_steps=200,
    lanczos_topk=10,
    n_probes=20,
    n_samples=1,
    timeout=1800,
    use_double=False,
):
    """
    Combine Lanczos and Rayleigh methods to estimate an 'effective' Hessian condition number.
    
    - Numerator: mean of top-K Lanczos Ritz eigenvalues
    - Denominator: median Rayleigh quotient eigenvalue (across random probes)
    
    Returns dict with all intermediate results.
    """

    start_time = time.time()
    if device is None:
        device = next(model.parameters(), torch.tensor([])).device
        if device.type not in ("cuda", "mps"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get batch
    x0, y0 = data_fn()
    x0, y0 = x0.to(device), y0.to(device)

    # dtype
    params = [p for p in model.parameters() if p.requires_grad]
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    n = sum(numels)
    dtype_used = torch.float64 if use_double else params[0].dtype
    if use_double:
        model = model.to(dtype=torch.float64)
    model = model.to(device)

    # pick loss if not given
    with torch.no_grad():
        flat0 = parameters_to_vector(params).to(dtype_used)
    def _auto_loss_fn(out, y):
        if out.ndim > 1 and out.shape[-1] > 1:
            return torch.nn.functional.cross_entropy
        else:
            return torch.nn.functional.mse_loss
    if loss_fn is None:
        loss_scalar = model(x0)
        loss_fn = _auto_loss_fn(loss_scalar, y0)

    # unflatten helper
    def unflatten(theta):
        splits = list(torch.split(theta, numels))
        tensors = [t.view(s) for t, s in zip(splits, shapes)]
        return {n: t for n, t in zip([n for n, _ in model.named_parameters() if _.requires_grad], tensors)}

    def closure(theta):
        pmap = unflatten(theta)
        out = functional_call(model, pmap, (x0,))
        return loss_fn(out, y0)

    theta0 = flat0.detach().clone().requires_grad_(True)

    def hvp_fn(v):
        loss = closure(theta0)
        grads = torch.autograd.grad(loss, theta0, create_graph=True)[0]
        grad_v = torch.dot(grads, v)
        Hv = torch.autograd.grad(grad_v, theta0, retain_graph=False)[0]
        return Hv

    # ---- Lanczos ----
    def lanczos(hvp_fn, dim, m):
        q = torch.randn(dim, device=device, dtype=dtype_used)
        q = q / q.norm()
        Q, alpha, beta = [], [], []
        prev_q = torch.zeros_like(q)
        for j in range(m):
            if time.time() - start_time > timeout:
                break
            z = hvp_fn(q)
            a = torch.dot(q, z).item()
            z = z - a * q - (beta[-1] * prev_q if j > 0 else 0)
            b = z.norm().item()
            alpha.append(a)
            beta.append(b)
            Q.append(q)
            if b < 1e-10:
                break
            prev_q, q = q, z / b
        T = np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1)
        return np.linalg.eigvalsh(T)

    ritz_vals = lanczos(hvp_fn, n, lanczos_steps)
    if len(ritz_vals) == 0:
        return {"condition_number": float("nan"), "eigenvals_lanczos": [], "eigenvals_rayleigh": []}

    topk_vals = np.sort(np.abs(ritz_vals))[-lanczos_topk:]
    lanczos_mean_topk = float(np.mean(topk_vals))
    pos_lanczos_mean_topk = float(np.mean(np.sort(ritz_vals)[-lanczos_topk:]))
    # ---- Rayleigh estimates ----
    eigenvals_rayleigh = []
    for _ in range(n_samples):
        if time.time() - start_time > timeout:
            break
        inp, target = data_fn()
        inp, target = inp.to(device), target.to(device)
        loss = model(inp)
        loss = loss_fn(loss, target)
        params = list(model.parameters())

        # probe
        for _ in range(n_probes):
            v = [torch.randn_like(p) for p in params]
            v_norm = sum(torch.sum(v_i**2) for v_i in v).sqrt()
            v = [v_i / v_norm for v_i in v]
            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_v = sum(torch.sum(g * v_i) for g, v_i in zip(grads, v))
            hvp = torch.autograd.grad(grad_v, params, retain_graph=True)
            eig_est = sum(torch.sum(hv * v_i) for hv, v_i in zip(hvp, v))
            eigenvals_rayleigh.append(eig_est.item())

    eigenvals_rayleigh = np.array(eigenvals_rayleigh)
    eigenvals_rayleigh = eigenvals_rayleigh[np.isfinite(eigenvals_rayleigh)]
    if len(eigenvals_rayleigh) == 0:
        return {"condition_number": float("nan"), "eigenvals_lanczos": ritz_vals, "eigenvals_rayleigh": []}

    rayleigh_median = float(np.median(np.abs(eigenvals_rayleigh)))
    pos_rayleigh_median = float(np.median(eigenvals_rayleigh[eigenvals_rayleigh > 0]))

    # effective condition number
    kappa_eff = lanczos_mean_topk / rayleigh_median
    pos_kappa_eff = pos_lanczos_mean_topk / pos_rayleigh_median 
    return {
        "condition_number": pos_kappa_eff,
        "condition_number_abs": kappa_eff,
        "lanczos_mean_topk": lanczos_mean_topk,
        "rayleigh_median": rayleigh_median,
        "eigenvals_lanczos": ritz_vals,
        "eigenvals_rayleigh": eigenvals_rayleigh,
        "dtype_used": dtype_used,
        "num_params": n,
    }

@torch.no_grad()
def _flatten_params(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return parameters_to_vector(params)

def estimate_condition_lanczos_reorth(
    model,
    data_fn,
    loss_fn=None,
    device=None,
    weight_decay=1e-4,
    lanczos_steps=200,     # you mentioned using ~200 vectors
    n_probes=1,            # multiple random starts improve the bulk estimate
    topk_for_lambda_max=10,
    tol_breakdown=1e-10,
    use_double=False,
    timeout=1800,
    seed=None,
):
    """
    Lanczos with FULL reorthogonalization + multiple probes (no Rayleigh).

    - For each probe, run m-step Lanczos with full reorthogonalization.
    - Collect all Ritz eigenvalues from the tridiagonal(s).
    - lambda_max := mean of top-K pooled Ritz eigenvalues (configurable).
    - lambda_min := median of pooled Ritz eigenvalues (bulk).
    - kappa := lambda_max / lambda_min

    Returns:
      dict with kappa, lambda_max, lambda_min, pooled Ritz values, per-probe Ritz, etc.
    """
    start = time.time()
    if device is None:
        device = next(model.parameters(), torch.tensor([])).device
        if device.type not in ("cuda", "mps"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # one batch (you can change to multiple if desired)
    x0, y0 = data_fn()
    x0, y0 = x0.to(device), y0.to(device)

    # dtype
    params = [p for p in model.parameters() if p.requires_grad]
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    n = sum(numels)
    dtype_used = torch.float64 if use_double else params[0].dtype
    if use_double:
        model = model.to(dtype=torch.float64)
    model = model.to(device)

    # auto-select loss if not provided
    def _auto_loss_fn(out, y):
        if out.ndim > 1 and out.shape[-1] > 1:
            return torch.nn.functional.cross_entropy
        else:
            return torch.nn.functional.mse_loss
    if loss_fn is None:
        with torch.no_grad():
            trial_out = model(x0)
        loss_fn = _auto_loss_fn(trial_out, y0)

    # helpers to (un)flatten
    with torch.no_grad():
        theta0 = parameters_to_vector(params).to(device=device, dtype=dtype_used)
    theta0 = theta0.detach().clone().requires_grad_(True)

    names = [n for n, p in model.named_parameters() if p.requires_grad]
    def unflatten(theta):
        splits = list(torch.split(theta, numels))
        tensors = [t.view(s) for t, s in zip(splits, shapes)]
        return {name: t for name, t in zip(names, tensors)}

    # Hessian-vector product (adds L2 weight decay on the fly)
    def hvp_fn(v):
        loss = loss_fn(functional_call(model, unflatten(theta0), (x0,)), y0)
        grads = torch.autograd.grad(loss, theta0, create_graph=True)[0]
        grad_v = torch.dot(grads, v)
        Hv = torch.autograd.grad(grad_v, theta0, retain_graph=False)[0]
        return Hv 

    # ---- FULL reorthogonalized Lanczos (single probe) ----
    def lanczos_full_reorth(hvp, dim, m, q0=None):
        """
        Returns: eigvals(T_m), and (alpha, beta) actually used (length k, k-1)
        """
        if q0 is None:
            q = torch.randn(dim, device=device, dtype=dtype_used)
        else:
            q = q0.to(device=device, dtype=dtype_used)
        q = q / (q.norm() + 1e-32)

        Q = []                     # list of orthonormal basis vectors (len k)
        alpha, beta = [], []

        prev_time = time.time()
        for j in range(m):
            if time.time() - start > timeout:
                break

            # z = H q_j
            z = hvp(q)

            # alpha_j = q_j^T z
            a = torch.dot(q, z).item()
            alpha.append(a)

            # z <- z - alpha_j q_j - sum_{i=0}^{j-1} (q_i^T z) q_i
            #   (full reorthogonalization against ALL previous Q, including current q)
            # First remove the current component
            z = z - a * q
            # Now remove components on all previous basis vectors
            if Q:
                # single-pass full reorthogonalization
                # (optionally, you can do a second pass for "double reorth")
                coeffs = torch.stack([torch.dot(z, qi) for qi in Q])
                # z <- z - sum_i coeff_i * q_i
                for ci, qi in zip(coeffs, Q):
                    z = z - ci * qi

            b = z.norm().item()
            beta.append(b)

            # save current q
            Q.append(q)

            # breakdown / happy convergence
            if b < tol_breakdown or j == m - 1:
                break

            # next q
            q = (z / (b + 1e-32))

        k = len(alpha)
        if k == 0:
            return np.array([]), alpha, beta

        # Build the used tridiagonal (size k x k)
        T = np.diag(alpha)
        if k >= 2:
            off = np.array(beta[:k-1], dtype=np.float64)
            T = T + np.diag(off, 1) + np.diag(off, -1)

        # NOTE: T is symmetric by construction
        ritz = np.linalg.eigvalsh(T)
        return ritz, alpha, beta

    # ---- multiple probes ----
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    pooled_ritz = []
    ritz_by_probe = []

    for p in range(n_probes):
        if time.time() - start > timeout:
            break
        q0 = torch.randn(n, device=device, dtype=dtype_used)
        q0 = q0 / (q0.norm() + 1e-32)
        ritz, _, _ = lanczos_full_reorth(hvp_fn, n, lanczos_steps, q0=q0)
        if ritz.size > 0:
            pooled_ritz.append(ritz)
            ritz_by_probe.append(ritz)

    if len(pooled_ritz) == 0:
        return {
            "kappa": float("nan"),
            "lambda_max": float("nan"),
            "lambda_min": float("nan"),
            "ritz_all": np.array([]),
            "ritz_by_probe": [],
            "dtype_used": dtype_used,
            "num_params": n,
        }

    ritz_all = np.concatenate(pooled_ritz, axis=0)

    # robust top-end estimate and requested lambda_min definition
    ritz_pos = ritz_all[ritz_all > 0]
    ritz_sorted = np.sort(ritz_pos)
    k_top = min(topk_for_lambda_max, len(ritz_sorted))
    lambda_max = float(np.mean(ritz_sorted[-k_top:])) if k_top > 0 else float("nan")
    lambda_min = float(np.median(ritz_sorted))  # as requested
    
    kappa_pos_subspace = lambda_max / lambda_min

    ritz_sorted = np.sort(np.abs(ritz_all))
    k_top = min(topk_for_lambda_max, len(ritz_sorted))
    lambda_max = float(np.mean(ritz_sorted[-k_top:])) if k_top > 0 else float("nan")
    lambda_min = float(np.median(ritz_sorted))  # as requested
    
    kappa_abs = lambda_max / lambda_min

    return {
        "condition_number": kappa_pos_subspace,
        "condition_number_abs": kappa_abs,
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "eigenvals": ritz_all,           # pooled Ritz eigenvalues over all probes
        "ritz_by_probe": ritz_by_probe, # list of arrays per probe
        "dtype_used": dtype_used,
        "num_params": n,
        "lanczos_steps": lanczos_steps,
        "n_probes": n_probes,
        "topk_for_lambda_max": topk_for_lambda_max,
        "weight_decay": weight_decay,
       
    }
def estimate_effective_condition_number_reorth(
    model,
    data_fn,
    loss_fn=None,
    device=None,
    weight_decay=1e-4,
    lanczos_steps=200,
    lanczos_topk=10,
    n_probes=1,
    n_samples=1,
    n_rayleigh_probes=200,
    timeout=1800,
    use_double=False,
    seed=None,
):
    """
    Hybrid estimator (batch-averaged):
      - For each of n_samples batches:
          * Run Lanczos (with n_probes) to get top eigenvalue estimates
          * Run Rayleigh quotient probes (n_rayleigh_probes) to estimate small eigenvalues
          * Compute per-batch condition number
      - Return the mean and variance of condition numbers across batches.

    Also records all Lanczos Ritz values and Rayleigh estimates.
    """
    start_time = time.time()
    if device is None:
        device = next(model.parameters(), torch.tensor([])).device
        if device.type not in ("cuda", "mps"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        return {
            "condition_number": float("nan"),
            "condition_number_abs": float("nan"),
            "condition_number_variance": float("nan"),
            "eigenvals_lanczos": np.array([]),
            "eigenvals_rayleigh": np.array([]),
            "num_params": 0,
        }
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    n = int(sum(numels))

    dtype_used = torch.float64 if use_double else params[0].dtype
    if use_double:
        model = model.to(torch.float64)
    model = model.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # helpers
    names = [n for n, p in model.named_parameters() if p.requires_grad]

    def unflatten(theta):
        splits = list(torch.split(theta, numels))
        tensors = [t.view(s) for t, s in zip(splits, shapes)]
        return {name: t for name, t in zip(names, tensors)}

    def lanczos_full_reorth(hvp, dim, m, q0=None, tol_breakdown=1e-12):
        q = torch.randn(dim, device=device, dtype=dtype_used) if q0 is None else q0.to(device, dtype=dtype_used)
        q = q / (q.norm() + 1e-32)
        Q, alpha, beta = [], [], []
        for j in range(m):
            if time.time() - start_time > timeout:
                break
            z = hvp(q)
            a = float(torch.dot(q, z))
            alpha.append(a)
            z = z - a * q
            if Q:
                for qi in Q:
                    ci = float(torch.dot(z, qi))
                    z = z - ci * qi
            b = float(z.norm())
            beta.append(b)
            Q.append(q)
            if b < tol_breakdown:
                break
            q = z / (b + 1e-32)
        k = len(alpha)
        if k == 0:
            return np.array([])
        T = np.diag(alpha)
        if k >= 2:
            off = np.array(beta[:k-1], dtype=np.float64)
            T += np.diag(off, 1) + np.diag(off, -1)
        return np.linalg.eigvalsh(T)

    # containers
    cond_pos_list, cond_abs_list, cond_lanczos_list = [], [], []
    per_batch_stats = []
    all_lanczos_eigs, all_rayleigh_eigs = [], []

    for s in range(n_samples):
        if time.time() - start_time > timeout:
            break
        x, y = data_fn()
        x, y = x.to(device), y.to(device)

        # flatten parameters
        theta0 = parameters_to_vector(params).to(device=device, dtype=dtype_used)
        theta0 = theta0.detach().clone().requires_grad_(True)

        def closure_theta(theta):
            pmap = unflatten(theta)
            out = functional_call(model, pmap, (x,))
            return loss_fn(out, y) if loss_fn is not None else torch.nn.functional.cross_entropy(out, y)

        def hvp_fn(v):
            loss = closure_theta(theta0)
            grads = torch.autograd.grad(loss, theta0, create_graph=True)[0]
            grad_v = torch.dot(grads, v)
            Hv = torch.autograd.grad(grad_v, theta0, retain_graph=False)[0]
            return Hv 

        # Lanczos pooled Ritz values
        pooled_ritz = []
        for p in range(n_probes):
            q0 = torch.randn(n, device=device, dtype=dtype_used)
            q0 = q0 / (q0.norm() + 1e-32)
            ritz = lanczos_full_reorth(hvp_fn, n, lanczos_steps, q0=q0)
            if ritz.size > 0:
                pooled_ritz.append(ritz)
        if not pooled_ritz:
            continue
        ritz_all = np.concatenate(pooled_ritz, axis=0)
        all_lanczos_eigs.append(ritz_all)

        ritz_sorted_abs = np.sort(np.abs(ritz_all))
        k_top = min(lanczos_topk, len(ritz_sorted_abs))
        lanczos_mean_topk = float(np.mean(ritz_sorted_abs[-k_top:])) if k_top > 0 else float("nan")
        ritz_pos = ritz_all[ritz_all > 0]
        ritz_pos_sorted = np.sort(ritz_pos) if ritz_pos.size > 0 else np.array([])
        k_top_pos = min(lanczos_topk, len(ritz_pos_sorted))
        pos_lanczos_mean_topk = float(np.mean(ritz_pos_sorted[-k_top_pos:])) if k_top_pos > 0 else float("nan")
        lanczos_median = float(np.median(ritz_pos))
        # Rayleigh probes
        eigenvals_rayleigh = []
        out = model(x)
        loss_batch = loss_fn(out, y) if loss_fn is not None else torch.nn.functional.cross_entropy(out, y)
        grads = torch.autograd.grad(loss_batch, params, create_graph=True)
        for _ in range(n_rayleigh_probes):
            v_list = [torch.randn_like(p) for p in params]
            v_norm = torch.sqrt(sum(torch.sum(v_i * v_i) for v_i in v_list))
            v_list = [v_i / (v_norm + 1e-32) for v_i in v_list]
            grad_v = sum(torch.sum(g * v_i) for g, v_i in zip(grads, v_list))
            hvp = torch.autograd.grad(grad_v, params, retain_graph=True)
            eig_est = sum(torch.sum(hv * v_i) for hv, v_i in zip(hvp, v_list))
            eigenvals_rayleigh.append(float(eig_est))
        eigenvals_rayleigh = np.array(eigenvals_rayleigh)
        eigenvals_rayleigh = eigenvals_rayleigh[np.isfinite(eigenvals_rayleigh)]
        all_rayleigh_eigs.append(eigenvals_rayleigh)

        rayleigh_median = float(np.median(np.abs(eigenvals_rayleigh))) if eigenvals_rayleigh.size > 0 else float("nan")
        pos_vals = eigenvals_rayleigh[eigenvals_rayleigh > 0]
        pos_rayleigh_median = float(np.median(pos_vals)) if pos_vals.size > 0 else float("nan")

        kappa_abs = lanczos_mean_topk / rayleigh_median
        kappa_pos = pos_lanczos_mean_topk / pos_rayleigh_median
        
        kappa_lanczos = lanczos_mean_topk / lanczos_median
        cond_abs_list.append(kappa_abs)
        cond_pos_list.append(kappa_pos)
        cond_lanczos_list.append(kappa_lanczos)

        per_batch_stats.append({
            "lanczos_mean_topk": lanczos_mean_topk,
            "pos_lanczos_mean_topk": pos_lanczos_mean_topk,
            "rayleigh_median": rayleigh_median,
            "pos_rayleigh_median": pos_rayleigh_median,
            "condition_number_abs": kappa_abs,
            "condition_number": kappa_pos,
        })

    cond_abs_arr, cond_pos_arr, cond_lanczos_arr = np.array(cond_abs_list), np.array(cond_pos_list), np.array(cond_lanczos_list)
    return {
        "condition_number": float(np.mean(cond_pos_arr)) if cond_pos_arr.size else float("nan"),
        "condition_number_abs": float(np.mean(cond_abs_arr)) if cond_abs_arr.size else float("nan"),
        "condition_numbers_all_samples": cond_pos_arr,
        "condition_number_variance": float(np.var(cond_pos_arr, ddof=1)) if cond_pos_arr.size > 1 else float("nan"),
        "lanczos_condition_number": float(np.mean(cond_lanczos_arr)) if cond_lanczos_arr.size else float("nan"),
        "lanczos_condition_number_variance": float(np.var(cond_lanczos_arr, ddof=1)) if cond_lanczos_arr.size > 1 else float("nan"),
        "per_batch": per_batch_stats,
        "dtype_used": dtype_used,
        "num_params": n,
        "lanczos_steps": lanczos_steps,
        "n_probes": n_probes,
        "n_samples": n_samples,
        "n_rayleigh_probes": n_rayleigh_probes,
        "weight_decay": weight_decay,
        "eigenvals_lanczos": np.concatenate(all_lanczos_eigs, axis=0) if all_lanczos_eigs else np.array([]),
        "eigenvals_rayleigh": np.concatenate(all_rayleigh_eigs, axis=0) if all_rayleigh_eigs else np.array([]),
        "timeout_reached": (time.time() - start_time) > timeout,
    }
