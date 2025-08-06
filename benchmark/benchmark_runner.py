import subprocess
import json
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
import typer
import datetime

app = typer.Typer()

def ensure_str(data):
    if data is None:
        return ""
    return data.decode() if isinstance(data, bytes) else data

def run_single_benchmark(script_path, optimizer, steps, trials, seed, output_dir, runtime_limit, step_hint):
    """Run a single benchmark instance and capture output."""
    print(f"running {optimizer}")
    # Create a unique output file for this run
    timestamp = int(time.time() * 1000)
    log_file = f"{output_dir}/run_{optimizer}_{seed}_{timestamp}.log"
    script_name = Path(script_path).stem
    module_path = f"benchmark.{script_name}"
    # Build the command
    cmd = [
        "python", "-m", module_path,
        "--opt", optimizer,
        "--steps", str(steps),
        "--trials", str(trials),
        "--runtime-limit", str(runtime_limit),
        "--step-hint", str(step_hint)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=72*3600,  # 24 hour timeout
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "3"}  # Ensure single GPU
        )
        
        # Save the full output
        with open(log_file, 'w') as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
        
        # Parse the output
        parsed_result = parse_benchmark_output(result.stdout, result.stderr)
        parsed_result.update({
            "optimizer": optimizer,
            "seed": seed,
            "steps": steps,
            "trials": trials,
            "log_file": log_file,
            "success": result.returncode == 0
        })
        
        return parsed_result
    except subprocess.TimeoutExpired as e:
        with open(log_file, 'w') as f:
            f.write("STDOUT:\n")
            f.write(ensure_str(e.stdout))
            f.write("\nSTDERR:\n")
            f.write(ensure_str(e.stderr))

        parsed_result = parse_benchmark_output(ensure_str(e.stdout), ensure_str(e.stderr))
        parsed_result.update({
            "optimizer": optimizer,
            "seed": seed,
            "steps": steps,
            "trials": trials,
            "log_file": log_file,
            "success": False
        })
        return parsed_result
  
    except Exception as e:
        with open(log_file, 'w') as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
        return {
            "optimizer": optimizer,
            "seed": seed,
            "steps": steps,
            "trials": trials,
            "success": False,
            "error": str(e),
            "log_file": log_file
        }

def parse_benchmark_output(stdout, stderr):
    """Parse the output from your benchmark script."""
    result = {
        "runtime": None,
        "attempts": None,
        "best_loss": None,
        "loss_trajectory": [],
        "test_accuracies": [],
        "grad_variances": [],
        "config_str": "",
        "mean_cond": None,
        "std_err_cond": None,
        "memory_usage": None
    }
    
    # Parse the final output line that contains all the metrics
    # Looking for pattern like: "Took: X | Attempt: Y | OptimizerName(...) | Best Loss: Z | ..."
    
    lines = stdout.split('\n')
    for line in lines:
        # Look for the summary line
        if "Took:" in line and "Best Loss:" in line:
            # Extract runtime
            runtime_match = re.search(r"Took: ([0-9.]+)", line)
            if runtime_match:
                result["runtime"] = float(runtime_match.group(1))
            
            # Extract attempts
            attempts_match = re.search(r"Attempt: ([0-9]+)", line)
            if attempts_match:
                result["attempts"] = int(attempts_match.group(1))
            
            # Extract best loss
            loss_match = re.search(r"Best Loss: ([0-9.e\-+]+)", line)
            if loss_match:
                result["best_loss"] = float(loss_match.group(1))
            
            # Extract loss trajectory
            traj_match = re.search(r"loss_trajectory: (\[.*?\])", line)
            if traj_match:
                try:
                    # Safely evaluate the list string
                    import ast
                    result["loss_trajectory"] = ast.literal_eval(traj_match.group(1))
                except:
                    pass
            
            # Extract test accuracies
            acc_match = re.search(r"test_accuracies: (\[.*?\])", line)
            if acc_match:
                try:
                    import ast
                    result["test_accuracies"] = ast.literal_eval(acc_match.group(1))
                except:
                    pass

            # Extract gradient variances
            acc_match = re.search(r"grad_variances: (\[.*?\])", line)
            if acc_match:
                try:
                    import ast
                    result["grad_variances"] = ast.literal_eval(acc_match.group(1))
                except:
                    pass
            
            # Extract condition numbers
            cond_match = re.search(r"mean_cond: ([0-9.e\-+]+)", line)
            if cond_match:
                result["mean_cond"] = float(cond_match.group(1))
            
            # Extract memory usage
            mem_match = re.search(r"memory_usage: ([0-9.e\-+]+)", line)
            if mem_match:
                result["memory_usage"] = float(mem_match.group(1))
    
    return result

def aggregate_and_plot(results, benchmark_name, output_dir):
    """Aggregate results and create plots."""
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Filter successful runs
    successful_df = df[df['success'] == True].copy()
    
    if successful_df.empty:
        print("No successful runs to aggregate!")
        return
    
    # Group by optimizer
    optimizer_groups = successful_df.groupby('optimizer')
    
    # Create summary statistics
    summary_stats = []
    
    for optimizer, group in optimizer_groups:
        stats = {
            'optimizer': optimizer,
            'n_runs': len(group),
            'success_rate': len(group) / len(df[df['optimizer'] == optimizer]),
            'mean_runtime': group['runtime'].mean(),
            'std_runtime': group['runtime'].std(),
            'mean_loss': group['best_loss'].mean(),
            'std_loss': group['best_loss'].std(),
            'mean_attempts': group['attempts'].mean(),
            'std_attempts': group['attempts'].std(),
        }
        summary_stats.append(stats)
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_stats)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_df.to_csv(f"{output_dir}/{benchmark_name}_summary_{timestamp}.csv", index=False)
    
    # Plot loss trajectories
    plt.figure(figsize=(12, 8))
    
    for optimizer, group in optimizer_groups:
        # Get all loss trajectories for this optimizer
        trajectories = [traj for traj in group['loss_trajectory'] if traj]
        
        if not trajectories:
            continue
        
        # Find minimum length
        min_length = min(len(traj) for traj in trajectories)
        
        # Truncate all trajectories
        truncated = [traj[:min_length] for traj in trajectories]
        mean_runtime = group["runtime"].mean()
        if truncated:
            # Calculate statistics
            mean_traj = np.mean(truncated, axis=0)
            std_traj = np.std(truncated, axis=0)
            se_traj = std_traj / np.sqrt(len(truncated))
            
            # Plot
            x = np.arange(min_length)
            plt.plot(x, mean_traj, label=f"{optimizer} (n={mean_runtime})", linewidth=1)
            plt.fill_between(x, mean_traj - se_traj, mean_traj + se_traj, alpha=0.2)
    
    plt.xlabel('Iteration (x1000)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Often helpful for loss curves
    plt.savefig(f"{output_dir}/{benchmark_name}_loss_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot gradient variances
    plt.figure(figsize=(12, 8))
    
    for optimizer, group in optimizer_groups:
        # Get all loss trajectories for this optimizer
        trajectories = [traj for traj in group['grad_variances'] if traj]
        
        if not trajectories:
            continue
        
        # Find minimum length
        min_length = min(len(traj) for traj in trajectories)
        
        # Truncate all trajectories
        truncated = [traj[:min_length] for traj in trajectories]
        
        if truncated:
            # Calculate statistics
            mean_traj = np.mean(truncated, axis=0)
            std_traj = np.std(truncated, axis=0)
            se_traj = std_traj / np.sqrt(len(truncated))
            
            # Plot
            x = np.arange(min_length)
            plt.plot(x, mean_traj, label=f"{optimizer} (n={len(truncated)})", linewidth=2)
            plt.fill_between(x, mean_traj - se_traj, mean_traj + se_traj, alpha=0.3)
    
    plt.xlabel('Iteration (x1000)')
    plt.ylabel('Gradient Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Often helpful for loss curves
    plt.savefig(f"{output_dir}/{benchmark_name}_grad_variances_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()


    # Plot test accuracies if available
    plt.figure(figsize=(12, 8))
    
    for optimizer, group in optimizer_groups:
        # Get all test accuracy trajectories for this optimizer
        acc_trajectories = [acc for acc in group['test_accuracies'] if acc]
        
        if not acc_trajectories:
            continue
        
        # Find minimum length
        min_length = min(len(acc) for acc in acc_trajectories)
        
        # Truncate all trajectories
        truncated = [acc[:min_length] for acc in acc_trajectories]
        
        if truncated:
            # Calculate statistics
            mean_acc = np.mean(truncated, axis=0)
            std_acc = np.std(truncated, axis=0)
            se_acc = std_acc / np.sqrt(len(truncated))
            
            # Plot
            x = np.arange(min_length)
            plt.plot(x, mean_acc, label=f"{optimizer} (n={len(truncated)})", linewidth=2)
            plt.fill_between(x, mean_acc - se_acc, mean_acc + se_acc, alpha=0.3)
    
    plt.xlabel('`Steps (x1000)')
    plt.ylabel('Test loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(f"{output_dir}/{benchmark_name}_test_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"{output_dir}/{benchmark_name}_detailed_results_{timestamp}.csv", index=False)
    
@app.command()
def main(
    benchmark: str = typer.Argument(..., help="Path to benchmark script"),
    optimizers: str = typer.Argument(..., help="Comma-separated optimizers"),
    runs_per_optimizer: int = typer.Option(5, help="Number of runs per optimizer"),
    steps: int = typer.Option(0, help="Steps per run"),
    trials: int = typer.Option(5, help="Optuna trials per run"),
    output_dir: str = typer.Option("comparison_results", help="Output directory"),
    runtime_limit: int = typer.Option(3600 * 24, help="Timeout in seconds"),
    step_hint: int = typer.Option(67000, help="Step hint per run"),
):
    """
    Run optimizer comparison using subprocess calls.
    
    Example:
        python comparison_runner.py MNIST.py "SGD,AdamW" --runs-per-optimizer=5
    """
    
    # Setup
    benchmark_path = Path(benchmark)
    benchmark_name = benchmark_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    optimizer_list = [opt.strip() for opt in optimizers.split(',')]
    
    print(f"Running comparison for {benchmark_name}")
    print(f"Optimizers: {', '.join(optimizer_list)}")
    print(f"Runs per optimizer: {runs_per_optimizer}")
    print(f"Steps per run: {steps}")
    print(f"Optuna trials per run: {trials}")
    
    # Run all experiments
    all_results = []
    
    for optimizer in optimizer_list:
        print(f"\n=== Running {optimizer} ===")
        
        for run_idx in range(runs_per_optimizer):
            print(f"Run {run_idx + 1}/{runs_per_optimizer}")
            
            # Use different seeds for each run
            seed = 42 + run_idx
            
            # Run the benchmark
            result = run_single_benchmark(
                benchmark_path, optimizer, steps, trials, seed, output_dir, runtime_limit, step_hint
            )
            
            all_results.append(result)
            print(result)
            # Print immediate feedback
            if result['success']:
                print(f"  ✓ Success - Loss: {result.get('best_loss', 'N/A')}")
            else:
                print(f"  ✗ Failed - {result.get('error', 'Unknown error')}")
    
    # Aggregate and visualize results
    print(f"\n=== Aggregating Results ===")
    aggregate_and_plot(all_results, benchmark_name, output_dir)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Plots: {output_dir}/{benchmark_name}_*.png")

if __name__ == "__main__":
    app()