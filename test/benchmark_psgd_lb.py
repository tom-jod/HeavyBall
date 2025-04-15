import functools
import math
import time

import torch

from heavyball.utils import max_singular_value_cholesky, max_singular_value_power_iter, set_torch


def display(exact, name, approx, duration):
    exact = exact.double().cpu()
    approx = approx.double().cpu()
    error = torch.abs(approx - exact)
    rel_error = error / exact.clamp(min=1e-8)
    print(
        f"{name} | Took: {duration:.6f}s | Approx={approx.mean():.4e}, Exact={exact.mean():.4e}, "  #
        f"Abs Error={error.mean():.4e}, Rel Error={rel_error.mean():.5f}"
    )


def measure(xs, fn):
    out = []
    for x in xs:  # warmup
        fn(x)
    torch.cuda.synchronize()
    start = time.time()
    for x in xs:
        out.append(fn(x))
    torch.cuda.synchronize()
    end = time.time()
    return end - start, torch.tensor(out)


def baseline(x):
    return torch.linalg.matrix_norm(x, ord=2)


@torch.inference_mode()
def test_singular_value_approx(min_val: int = 8, max_val: int = 512, attempts: int = 128):
    torch.manual_seed(0x12378)
    test_cases = [
        (lambda x: torch.randn((x, x)), "Normal"),  #
        (lambda x: torch.rand((x, x)), "Uniform"),  #
        (lambda x: torch.randn((x, x)) * torch.arange(x).view(1, -1), "Normal * Arange"),  #
        (lambda x: torch.randn((x, x)).exp(), "exp(Normal)"),  #
        (lambda x: torch.randn((x, x)) ** 16, "Normal ** 16"),  #
    ]
    max_name_len = max(len(name) for _, name in test_cases)
    test_cases = [(fn, f"{name:{max_name_len}}") for fn, name in test_cases]
    for size in range(int(math.log2(min_val)), int(math.log2(max_val)) + 1, 3):
        size = 2**size
        for matrix_fn, name in test_cases:
            matrices = [matrix_fn(size) for _ in range(attempts)]
            # _, exact = measure([m.double() for m in matrices], baseline)
            matrices = [m.cuda().float() for m in matrices]
            baseline_time, cuda_exact = measure(matrices, baseline)
            exact = cuda_exact  # they're close enough
            cholesky_time, ch_approx = measure(matrices, max_singular_value_cholesky)
            power_iter0_time, pow0_approx = measure(matrices, max_singular_value_power_iter)
            power_iter1_time, pow1_approx = measure(
                matrices, functools.partial(max_singular_value_power_iter, iterations=1)
            )
            power_iter2_time, pow2_approx = measure(
                matrices, functools.partial(max_singular_value_power_iter, iterations=2)
            )

            size_str = f"{size:{len(str(max_val))}d}"
            display(exact, f"{name} ({size_str}) |            exact (GPU)", cuda_exact, baseline_time)
            display(exact, f"{name} ({size_str}) |         cholesky (GPU)", ch_approx, cholesky_time)
            display(exact, f"{name} ({size_str}) | power iter (0it) (GPU)", pow0_approx, power_iter0_time)
            display(exact, f"{name} ({size_str}) | power iter (1it) (GPU)", pow1_approx, power_iter1_time)
            display(exact, f"{name} ({size_str}) | power iter (2it) (GPU)", pow2_approx, power_iter2_time)


if __name__ == "__main__":
    set_torch()
    with torch._dynamo.utils.disable_cache_limit():
        test_singular_value_approx()
