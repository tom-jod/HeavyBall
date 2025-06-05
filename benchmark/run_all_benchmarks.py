import itertools
import multiprocessing
import os
import random
import re
import time
import traceback
from datetime import datetime
from queue import Empty
import matplotlib.pyplot as plt
import numpy as np
import typer
import logging
import contextlib
import ast
import torch._dynamo
torch._dynamo.config.suppress_errors = True
app = typer.Typer()

torch._dynamo.config.disable = True

benchmarks = [
    "beale.py",
    "rosenbrock.py",
    "rastrigin.py",
    "quadratic_varying_scale.py",
    "quadratic_varying_target.py",
    "noisy_matmul.py",
    "xor_sequence.py",
    "xor_digit.py",
    "xor_spot.py",
    "xor_sequence_rnn.py",
    "xor_digit_rnn.py",
    "xor_spot_rnn.py",
    "saddle_point.py",
    "discontinuous_gradient.py",
    "wide_linear.py",
    "minimax.py",
    "plateau_navigation.py",
    "scale_invariant.py",
    "momentum_utilization.py",
    "batch_size_scaling.py",
    "sparse_gradient.py",
    "layer_wise_scale.py",
    "parameter_scale.py",
    "gradient_delay.py",
    "gradient_noise_scale.py",
    "adversarial_gradient.py",
    "dynamic_landscape.py",
    "constrained_optimization.py",
]


def last_match(pattern, text):
    matches = re.findall(pattern, text)
    if not matches:
        return None
    last = matches[-1]
    return float(last)


def parse_config(text):
    match = re.search(r"Attempt: \d+ \| (.*?) \| Best Loss:", text)
    if match:
        return match.group(1).strip()
    return ""


_module_cache = {}


def run_benchmark(script, opt, steps, dtype, trials, seed, difficulty):
    import io
    import pathlib
    import sys
    import time

    import torch

    from benchmark.utils import SkipConfig

    sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    start_time = time.time()
    loss_trajectory = []

    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        module_name = script.replace(".py", "")

        if module_name in _module_cache:
            module = _module_cache[module_name]
        else:
            module = __import__(module_name)
            _module_cache[module_name] = module

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Build arguments
        arguments = {
            "method": ["qr"],
            "dtype": [dtype],
            "steps": steps,
            "weight_decay": 0,
            "opt": [opt],
            "trials": trials,
            "win_condition_multiplier": 1.0,
            "config": difficulty,
        }
        # Run the main function and capture its return value
         # Run the main function
        module.main(**arguments)

    except SkipConfig:
        raise
    except Exception:
        output = sys.stdout.getvalue()
        error = traceback.format_exc()
    else:
        output = sys.stdout.getvalue()
        error = ""
    finally:
        sys.stdout = stdout
    # Parse output
    
    success = "Successfully found the minimum." in output
    runtime = last_match(r"Took: ([0-9.]+)", output)
    loss = last_match(r"Best Loss: ([0-9.e\-+]+)", output)
    attempts = int(last_match(r"Attempt: ([0-9]+)", output) or trials)
    # Try to find a list called loss_trajectory in the output
    if not loss_trajectory:  # Only search if we haven't found it yet
        # Look for patterns like "[1.2, 3.4, 5.6, ...]" in the output
        # This regex looks for a list pattern that might be printed as part of "loss_trajectory"
        list_pattern = r'\[([\d\s,.e\-+]+)\]'
        list_matches = re.findall(list_pattern, output)
        
        if list_matches:
            for list_str in list_matches:
                try:
                    # Try to parse the list string
                    # First, make it a proper list by adding brackets
                    list_str = "[" + list_str + "]"
                    # Use ast.literal_eval to safely evaluate the string representation of the list
                    parsed_list = ast.literal_eval(list_str)
                    
                    # If it's a list of numbers and reasonably long, it might be our loss trajectory
                    if (isinstance(parsed_list, list) and 
                        all(isinstance(x, (int, float)) for x in parsed_list) and 
                        len(parsed_list) > 5):  # Assuming loss trajectory has more than 5 points
                        
                        loss_trajectory = parsed_list
                        print(f"Extracted loss_trajectory from output with length {len(loss_trajectory)}")
                        break
                except (SyntaxError, ValueError):
                    # If parsing fails, try the next match
                    continue
    total_runtime = time.time() - start_time

    # Parse winning config string
    config_str = parse_config(output)
    
    return {
        "name": f"{script.replace('.py', '')}-{difficulty}",
        "opt": opt,
        "success": success,
        "runtime": float(runtime or total_runtime),
        "loss": float(loss) if loss else float("inf"),
        "attempts": attempts,
        "error": error if error else "",
        "seed": seed,
        "config_str": config_str,
        "loss_trajectory": loss_trajectory,
    }


def opt_to_config(opt):
    caution = "No"
    mars = "No"
    if opt.startswith("cautious-"):
        opt = opt[len("cautious-") :]
        caution = "Yes"
    if opt.startswith("unscaled_cautious-"):
        opt = opt[len("unscaled_cautious-") :]
        caution = "Unscaled"
    if opt.startswith("mars-"):
        opt = opt[len("mars-") :]
        mars = "Yes"
    return opt, caution, mars


def write_progress(results, opt, output):
    with open(output, "w") as f:
        f.write(f"# Benchmark Results\nGenerated: {datetime.now()}\nLast updated: {datetime.now()}\n\n")
        f.write("## Summary (In Progress)\n\n")
        f.write("| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |\n")
        f.write("|-----------|---|---|---------|----------|------|\n")

        for o in opt:
            opt_results = [r for r in results if r["opt"] == o]
            if not opt_results:
                continue
            success = sum(r["success"] for r in opt_results)
            runtime = np.mean([r["runtime"] for r in opt_results if r["success"]]) if success else 0
            attempts = np.mean([r["attempts"] for r in opt_results if r["success"]]) if success else 0

            o, caution, mars = opt_to_config(o)
            f.write(f"| {o} | {caution} | {mars} | {success}/{len(opt_results)} | {runtime:.2f}s | {attempts:.1f} |\n")

        f.write("\n## Details\n\n")
        # Add Winning Config column
        f.write(
            "| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |\n"
        )
        f.write("|-----------|-----------|---------|---|---|----------|------|---|---|----------------|\n")

        for r in sorted(results, key=lambda x: (x["name"], x["opt"])):
            mark = "✓" if r["success"] else "✗"
            runtime = f"{r['runtime']:.2f}s"
            loss = f"{r['loss']:.2e}"
            attempts = f"{r['attempts']:d}"
            seed = f"{r['seed']}"
            config_str = f"`{r['config_str']}`" if r.get("config_str", False) else "N/A"

            opt, caution, mars = opt_to_config(r["opt"])
            f.write(
                f"| {r['name']} | {opt} | {caution} | {mars} | {mark} | {runtime} | {loss} | {attempts} | {seed} | {config_str} |\n"
            )

        if any(not r["success"] for r in results):
            f.write("\n## Errors\n\n")
            for r in sorted(results, key=lambda x: (x["name"], x["opt"])):
                if not r["success"] and r["error"] and r["error"] != "None" and r["error"] != "":
                    f.write(f"\n### {r['name']} - {r['opt']}\n```\n{r['error']}\n```\n")


_difficulty_order = ["trivial", "easy", "medium", "hard", "extreme", "nightmare"]


def plot_loss_curves(results, opt, output):

    time_string = time.strftime("%H:%M:%S", time.localtime())
    benchmarks = set()
    for r in results:
        benchmarks.add(r["name"])

    for benchmark in list(benchmarks):
        benchmark_results = [r for r in results if r["name"] == benchmark]
        if not benchmark_results:
            continue
        plt.rcParams.update({'font.size': 16})
        # create figure 
        plt.figure(figsize=(12, 8))

        for o in opt:
            opt_results = [r for r in benchmark_results if r["opt"] == o]
            if not opt_results:
                continue
            
            # add line to graph
            if opt_results[0]['loss_trajectory'] != []:
                #target = opt_results[0]['target']
                loss_trajectory = opt_results[0]['loss_trajectory']
                iterations = range(len(loss_trajectory))
                plt.plot(iterations, loss_trajectory, linewidth=1, label=o)
                #plt.axhline(y=target, color='r', linestyle='--', label='Target')
        # save plot to a common folder
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.legend()
        folder_path = "loss_curves"
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{benchmark}_optimiser_loss_comparison.png")
        plt.savefig(file_path)
        plt.close()

            
def worker(task_queue, result_queue, worker_index, difficulties: list, timeout: int):
    import torch

    from benchmark.utils import SkipConfig

    torch.cuda.set_device(worker_index % torch.cuda.device_count())
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["HEAVYBALL_BENCHMARK_TIMEOUT"] = str(round(timeout))

    # Create the worker_logs directory if it doesn't exist
    os.makedirs("worker_logs", exist_ok=True)

    logging.basicConfig(
        filename=f"worker_logs/worker_{worker_index}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Worker {worker_index} starting")
    # Initialize CUDA context
    dummy = torch.zeros(1, device="cuda")
    del dummy
    torch.cuda.empty_cache()

    difficulties = [d for d in _difficulty_order if d in difficulties]

    while True:
        try:
            try:
                script, o, steps, dtype, trials, seed = task_queue.get(timeout=0.1)
            except Empty:
                result_queue.put(None)
                return

            inner_difficulties = difficulties.copy()
            for _ in range(len(difficulties)):
                d = inner_difficulties.pop(0)
                try:
                    logging.info(f"running benchmark")
                    result = run_benchmark(script, o, steps, dtype, trials, seed, d)
                    exc = ""
                    logging.info(f"result:{result}")
                except Exception as e:
                    
                    result = isinstance(e, SkipConfig)
                    exc = str(e)
                    logging.info(exc)
                if result is True:  # SkipConfig
                    break
                if result is not False and result is not None:
                    result_queue.put(result)
                    if result["success"]:
                        continue
                else:
                    inner_difficulties.insert(0, d)

                # model failed this task - no need to try harder ones
                for d_ in inner_difficulties:
                    result = {
                        "name": f"{script.replace('.py', '')}-{d_}",
                        "opt": o,
                        "success": False,
                        "runtime": timeout if "timed out" in exc else 0,
                        "attempts": 0,
                        "loss": float("inf"),
                        "error": str(exc),
                        "seed": seed,
                        "config_str": "N/A",
                    }
                    result_queue.put(result)
                break

        except Exception:
            traceback.print_exc()
            break


@app.command()
def main(
    opt: list[str] = typer.Option([], help="Optimizers"),
    steps: int = 100_000,
    timeout: int = 3600,
    output: str = "benchmark_results.md",
    trials: int = 1000,
    dtype: str = "float32",
    parallelism: int = typer.Option(8, help="Number of parallel worker processes"),
    caution: bool = False,
    mars: bool = False,
    unscaled_caution: bool = typer.Option(False, "--unscaled-caution", help="Use unscaled caution"),
    seeds: int = 4,
    difficulties: str = typer.Option([], help=f"{_difficulty_order} or any combination of these"),
    plot_loss_trajectories: bool = True,
    exclude: str = typer.Option([], help="List of benchmarks to exclude"),
):  
    # Parse list based argments, to allow the user to pass in --flag "option1,option2..." instead of typing --flag each time
    #opt = [item.strip() for item in opt.split(',')] if opt else []
    exclude = [item.strip() for item in exclude.split(',')] if exclude else []
    difficulties = [item.strip() for item in difficulties.split(',')] if difficulties else []
    if plot_loss_trajectories and len(difficulties) > 1:
        raise AssertionError("Please choose only one difficulty if plotting loss trajectories")

    multiprocessing.set_start_method("spawn", force=True)  # spawn appears to be safer with CUDA MPS

    # Filter out benchmarks that should be removed
    filtered_benchmarks = []
    if exclude != []:
        for benchmark in benchmarks:
            # Remove .py extension for comparison if it exists
            benchmark_name = benchmark.replace(".py", "")
            if not any(benchmark_name == remove.replace(".py", "") for remove in exclude):
                filtered_benchmarks.append(benchmark)
    else:
        filtered_benchmarks = benchmarks
    if not filtered_benchmarks:
        raise ValueError("No benchmarks left after filtering")
    
    print(f"Running {len(filtered_benchmarks)} benchmarks: {', '.join(filtered_benchmarks)}")

    if mars:
        opt = ["mars-" + o for o in opt]
    if caution:
        opt = ["cautious-" + o for o in opt]
    if unscaled_caution:
        opt = ["unscaled_cautious-" + o for o in opt]

    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    total_tasks = 0
    for script, o, i in itertools.product(filtered_benchmarks, opt, range(seeds)):
        task_queue.put((script, o, steps, dtype, trials, i))
        total_tasks += len(difficulties)

    if not total_tasks:
        raise ValueError("No benchmarks found")

    processes = []
    workers = min(parallelism, total_tasks)
    for idx in range(workers):
        p = multiprocessing.Process(
            target=worker, args=(task_queue, result_queue, idx, difficulties, timeout), daemon=False
        )
        p.start()
        processes.append(p)
        time.sleep(3)  # we can't start too many processes very quickly - otherwise there's errors with the cuda context

    # Collect results
    results = []
    completed = 0

    try:
        while workers > 0:
            prev_completed = completed
            while True:  # clear entire backlog and continue
                try:
                    result = result_queue.get(timeout=1)
                except Empty:
                    break
                if result is None:
                    workers -= 1
                    continue
                results.append(result)
                completed += 1
                print(
                    f"Progress: [{completed}/{total_tasks}] {result['name']}.py - {result['opt']}: "  #
                    f"{'✓' if result['success'] else '✗'}"
                )
            if prev_completed != completed:  # >= 1 task finished
                write_progress(results, opt, output)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current progress...")
    finally:
        # Clean up
        for p in processes:
            p.terminate()

        # Save final results
        if results:
            write_progress(results, opt, output)
            if plot_loss_trajectories:
                plot_loss_curves(results, opt, output)


if __name__ == "__main__":
    app()
