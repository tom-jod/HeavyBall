import concurrent.futures
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from tabulate import tabulate
from torch import Tensor

from heavyball.utils import decorator_knowngood, set_torch
from heavyball.utils import psgd_lb as _lb_random_sketch

set_torch()


@decorator_knowngood
def _max_select(to_index: Tensor):
    to_argmax = to_index.square().sum(0)
    idx = to_argmax.argmax()
    return to_index.index_select(1, idx).flatten().contiguous()


def psgd_lb(A: Tensor, max_abs: Tensor):
    """
    ~1.5 power iter steps & vector norm.
    Adapted from Evan Walters (https://github.com/evanatyourservice)
    """
    x = _max_select(A)  # / max_abs
    x = torch.einsum("i,ij->j", x, A)
    x /= x.norm()
    x = torch.einsum("i,ji,jk->k", x, A, A)
    return x.norm().sqrt() * max_abs


def psgd_lb_single(A: Tensor, max_abs: Tensor):
    """
    ~1.5 power iter steps & vector norm.
    Adapted from Evan Walters (https://github.com/evanatyourservice)
    """
    x = _max_select(A)  # / max_abs
    x = torch.einsum("i,ij->j", x, A)
    x /= x.norm()
    x = torch.einsum("i,ji->j", x, A)
    return x.norm() * max_abs


def calculate_ground_truth(A: torch.Tensor, max_abs: torch.Tensor) -> torch.Tensor:
    """Calculate the true spectral norm using SVD."""
    return torch.linalg.norm(A, ord=2)


def generate_test_matrix(rows: int, cols: int, dtype: torch.dtype, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a test matrix with known spectral structure and return it with its max_abs value."""
    # Create a test matrix with known spectral structure
    min_dim = min(rows, cols)

    # Use batched operations for QR decomposition
    U = torch.randn(rows, min_dim, device=device, dtype=dtype)
    U, _ = torch.linalg.qr(U)

    V = torch.randn(cols, min_dim, device=device, dtype=dtype)
    V, _ = torch.linalg.qr(V)

    # Create singular values with exponential decay
    s = torch.logspace(0, -3, min_dim, device=device, dtype=dtype)

    # Construct matrix A = U @ diag(s) @ V.T
    A = U @ (s.unsqueeze(1) * V.T)

    # Calculate max_abs
    max_abs = torch.max(torch.abs(A))

    return A, max_abs


def time_function(
    func,
    A: torch.Tensor,
    max_abs: torch.Tensor,
    num_repeats: int = 10,
    num_warmup: int = 5,
    k: Optional[int] = None,
    device: str = "cuda",
) -> float:
    """Time a function call and return average execution time in milliseconds."""

    # Handle optional k parameter for _lb_random_sketch
    def test_func():
        return func(A, max_abs)

    # Warmup
    for _ in range(num_warmup):
        _ = test_func()
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_repeats):
        _ = test_func()
    torch.cuda.synchronize()
    end_time = time.time()

    return (end_time - start_time) / num_repeats * 1000  # Convert to ms


def benchmark_method(
    method_func,
    A: torch.Tensor,
    max_abs: torch.Tensor,
    ground_truth_norm: torch.Tensor,
    device: str,
    num_repeats: int = 10,
    num_warmup: int = 5,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """Benchmark a single method against ground truth."""
    # Time the method
    method_time = time_function(method_func, A, max_abs, num_repeats, num_warmup, k, device)

    # Calculate accuracy
    if k is not None:
        method_norm = method_func(A, max_abs, k)
    else:
        method_norm = method_func(A, max_abs)

    rel_error = ((method_norm - ground_truth_norm) / ground_truth_norm).item()

    # Time the ground truth method for comparison
    ground_truth_time = time_function(calculate_ground_truth, A, max_abs, num_repeats, num_warmup, device=device)

    # Calculate speedup
    speedup = ground_truth_time / method_time

    return {"time_ms": method_time, "rel_error": rel_error, "speedup": speedup}


def benchmark_fixed_methods(
    sizes: List[Tuple[int, int]],
    dtype: torch.dtype = torch.float32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_repeats: int = 10,
    num_warmup: int = 5,
) -> Dict:
    """
    Benchmark psgd_lb and psgd_lb_single for different matrix sizes.

    Args:
        sizes: List of (rows, cols) tuples representing matrix sizes to benchmark
        dtype: Torch data type to use
        device: Device to run benchmarks on ("cuda" or "cpu")
        num_repeats: Number of repetitions for timing measurements
        num_warmup: Number of warmup iterations before timing

    Returns:
        Dictionary with benchmark results
    """
    results = {"method": [], "sizes": [], "time_ms": [], "rel_error": [], "speedup": []}

    print(f"Running fixed method benchmarks on {device} with {dtype}")

    # Use ThreadPoolExecutor for parallel processing when on CPU
    use_parallel = device == "cpu" and len(sizes) > 1

    if use_parallel:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(sizes)))
        futures = []

    for rows, cols in sizes:
        print(f"Benchmarking matrix size ({rows}, {cols}) for fixed methods")

        # Generate test matrix
        A, max_abs = generate_test_matrix(rows, cols, dtype, device)

        # Calculate ground truth once
        ground_truth_norm = calculate_ground_truth(A, max_abs)

        if use_parallel:
            # Submit benchmark tasks to thread pool
            for method_name, method_func in [("psgd_lb", psgd_lb), ("psgd_lb_single", psgd_lb_single)]:
                future = executor.submit(
                    benchmark_method, method_func, A, max_abs, ground_truth_norm, device, num_repeats, num_warmup
                )
                futures.append((method_name, f"{rows}x{cols}", future))
        else:
            # Run benchmarks sequentially
            for method_name, method_func in [("psgd_lb", psgd_lb), ("psgd_lb_single", psgd_lb_single)]:
                benchmark_result = benchmark_method(
                    method_func, A, max_abs, ground_truth_norm, device, num_repeats, num_warmup
                )

                # Store results
                results["method"].append(method_name)
                results["sizes"].append(f"{rows}x{cols}")
                results["time_ms"].append(benchmark_result["time_ms"])
                results["rel_error"].append(benchmark_result["rel_error"])
                results["speedup"].append(benchmark_result["speedup"])

    if use_parallel:
        # Collect results from futures
        for method_name, size, future in futures:
            benchmark_result = future.result()

            # Store results
            results["method"].append(method_name)
            results["sizes"].append(size)
            results["time_ms"].append(benchmark_result["time_ms"])
            results["rel_error"].append(benchmark_result["rel_error"])
            results["speedup"].append(benchmark_result["speedup"])

        executor.shutdown()

    return results


def benchmark_random_sketch(
    sizes: List[Tuple[int, int]],
    k_values: List[int],
    dtype: torch.dtype = torch.float32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_repeats: int = 10,
    num_warmup: int = 5,
) -> Dict:
    """
    Benchmark the _lb_random_sketch function for different matrix sizes and k values.

    Args:
        sizes: List of (rows, cols) tuples representing matrix sizes to benchmark
        k_values: List of k values to use in the random sketch
        dtype: Torch data type to use
        device: Device to run benchmarks on ("cuda" or "cpu")
        num_repeats: Number of repetitions for timing measurements
        num_warmup: Number of warmup iterations before timing

    Returns:
        Dictionary with benchmark results
    """
    results = {"method": [], "sizes": [], "k": [], "time_ms": [], "rel_error": [], "speedup": []}

    print(f"Running random sketch benchmarks on {device} with {dtype}")

    # Cache test matrices for each size to avoid regenerating for each k value
    matrix_cache = {}

    for rows, cols in sizes:
        # Generate and cache test matrix
        if (rows, cols) not in matrix_cache:
            A, max_abs = generate_test_matrix(rows, cols, dtype, device)
            ground_truth_norm = calculate_ground_truth(A, max_abs)
            matrix_cache[(rows, cols)] = (A, max_abs, ground_truth_norm)
        else:
            A, max_abs, ground_truth_norm = matrix_cache[(rows, cols)]

        print(f"Benchmarking matrix size ({rows}, {cols}) with k={1}")

        # Benchmark random sketch
        benchmark_result = benchmark_method(
            _lb_random_sketch, A, max_abs, ground_truth_norm, device, num_repeats, num_warmup
        )

        # Store results
        results["method"].append(f"lb_random_sketch k={1}")
        results["sizes"].append(f"{rows}x{cols}")
        results["k"].append(1)
        results["time_ms"].append(benchmark_result["time_ms"])
        results["rel_error"].append(benchmark_result["rel_error"])
        results["speedup"].append(benchmark_result["speedup"])

    return results


def print_results(results: Dict, method_key: bool = True):
    """Print benchmark results in a nicely formatted table."""
    if method_key:
        headers = ["Method", "Matrix Size", "Time (ms)", "Rel. Error", "Speedup vs SVD"]
        rows = []

        for i in range(len(results["sizes"])):
            rows.append([
                results["method"][i],
                results["sizes"][i],
                f"{results['time_ms'][i]:.2f}",
                f"{results['rel_error'][i]:.2e}",
                f"{results['speedup'][i]:.2f}x",
            ])
    else:
        headers = ["Matrix Size", "k", "Time (ms)", "Rel. Error", "Speedup vs SVD"]
        rows = []

        for i in range(len(results["sizes"])):
            rows.append([
                results["sizes"][i],
                results["k"][i],
                f"{results['time_ms'][i]:.2f}",
                f"{results['rel_error'][i]:.2e}",
                f"{results['speedup'][i]:.2f}x",
            ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))


def plot_results(results: Dict, output_file: str = None, method_key: bool = False):
    """Plot benchmark results."""
    if method_key:
        # Create figure with subplots
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Get unique matrix sizes and methods
        unique_methods = list(dict.fromkeys(results["method"]))

        # Group by method
        for method in unique_methods:
            indices = [i for i, val in enumerate(results["method"]) if val == method]
            sizes = [results["sizes"][i] for i in indices]
            times = [results["time_ms"][i] for i in indices]
            errors = [results["rel_error"][i] for i in indices]

            ax1.plot(sizes, times, "o-", label=method)
            ax2.plot(sizes, errors, "o-", label=method)

        ax1.set_xlabel("Matrix Size")
        ax1.set_ylabel("Time (ms)")
        ax1.set_title("Performance")
        ax1.grid(True)
        ax1.legend()

        ax2.set_xlabel("Matrix Size")
        ax2.set_ylabel("Relative Error")
        ax2.set_title("Accuracy")
        ax2.set_yscale("log")
        ax2.grid(True)
        ax2.legend()
    else:
        # Create figure with subplots
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Get unique matrix sizes and k values
        unique_k = list(dict.fromkeys(results["k"]))

        # Group by k value
        for k in unique_k:
            indices = [i for i, val in enumerate(results["k"]) if val == k]
            sizes = [results["sizes"][i] for i in indices]
            times = [results["time_ms"][i] for i in indices]
            errors = [results["rel_error"][i] for i in indices]

            ax1.plot(sizes, times, "o-", label=f"k={k}")
            ax2.plot(sizes, errors, "o-", label=f"k={k}")

        ax1.set_xlabel("Matrix Size")
        ax1.set_ylabel("Time (ms)")
        ax1.set_title("Performance")
        ax1.grid(True)
        ax1.legend()

        ax2.set_xlabel("Matrix Size")
        ax2.set_ylabel("Relative Error")
        ax2.set_title("Accuracy")
        ax2.set_yscale("log")
        ax2.grid(True)
        ax2.legend()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


def main():
    # Configure benchmark parameters
    sizes = [
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]

    k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    dtype = torch.float32
    device = "cuda"

    # Set number of repeats based on device (fewer for CPU)
    num_repeats = 64
    num_warmup = 4

    # Run benchmarks for fixed methods (psgd_lb and psgd_lb_single)
    print("\n=== Benchmarking fixed methods (psgd_lb and psgd_lb_single) ===")
    fixed_results = benchmark_fixed_methods(
        sizes=sizes, dtype=dtype, device=device, num_repeats=num_repeats, num_warmup=num_warmup
    )

    # Print results for fixed methods
    print_results(fixed_results, method_key=True)

    # Plot results for fixed methods
    plot_results(fixed_results, "fixed_methods_benchmark.png", method_key=True)

    # Run benchmarks for random sketch
    print("\n=== Benchmarking random sketch method ===")
    random_sketch_results = benchmark_random_sketch(
        sizes=sizes, k_values=k_values, dtype=dtype, device=device, num_repeats=num_repeats, num_warmup=num_warmup
    )

    # Print results for random sketch
    print_results(random_sketch_results, method_key=False)

    # Plot results for random sketch
    plot_results(random_sketch_results, "random_sketch_benchmark.png", method_key=False)

    # Combine all results for comparison
    combined_results = {
        "method": fixed_results["method"] + random_sketch_results["method"],
        "sizes": fixed_results["sizes"] + random_sketch_results["sizes"],
        "time_ms": fixed_results["time_ms"] + random_sketch_results["time_ms"],
        "rel_error": fixed_results["rel_error"] + random_sketch_results["rel_error"],
        "speedup": fixed_results["speedup"] + random_sketch_results["speedup"],
    }

    # Plot combined results
    plot_results(combined_results, "combined_benchmark.png", method_key=True)


if __name__ == "__main__":
    main()
