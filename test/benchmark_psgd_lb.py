import functools
import math
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from heavyball.utils import max_singular_value_cholesky, max_singular_value_power_iter, set_torch


def display_stats(exact, name, approx, duration):
    exact = exact.double().cpu()
    approx = approx.double().cpu()
    error = torch.abs(approx - exact)
    rel_error = error / exact.clamp(min=1e-8)
    print(
        f"{name} | Took: {duration:.6f}s | Approx={approx.mean():.4e}, Exact={exact.mean():.4e}, "
        f"Abs Error={error.mean():.4e}, Rel Error={rel_error.mean():.5f}"
    )
    return {
        "approx": approx.mean().item(),
        "exact": exact.mean().item(),
        "abs_error": error.mean().item(),
        "rel_error": rel_error.mean().item(),
        "duration": duration,
    }


def measure_time(xs, fn):
    for x in xs:  # Warmup
        fn(x)
    torch.cuda.synchronize()
    start = time.time()
    results = [fn(x) for x in xs]
    torch.cuda.synchronize()
    return time.time() - start, torch.tensor(results)


def baseline_norm(x):
    return torch.linalg.matrix_norm(x, ord=2)


@torch.inference_mode()
def test_singular_value_approx(min_val=2, max_val=512, attempts=128):
    torch.manual_seed(0x12378)
    test_cases = [
        (lambda x: torch.randn((x, x)), "Normal"),
        (lambda x: torch.rand((x, x)), "Uniform"),
        (lambda x: torch.randn((x, x)) * torch.arange(x).view(1, -1), "Normal * Arange"),
        (lambda x: torch.randn((x, x)).exp(), "exp(Normal)"),
        (lambda x: torch.randn((x, x)) ** 16, "Normal ** 16"),
    ]
    max_name_len = max(len(name) for _, name in test_cases)
    test_cases = [(fn, f"{name:{max_name_len}}") for fn, name in test_cases]
    methods = (
        ("exact", baseline_norm),
        ("cholesky", max_singular_value_cholesky),
        ("power_iter_0", max_singular_value_power_iter),
        ("power_iter_1", functools.partial(max_singular_value_power_iter, iterations=1)),
        ("power_iter_2", functools.partial(max_singular_value_power_iter, iterations=2)),
    )
    results = []

    sizes = [2**s for s in range(int(math.log2(min_val)), int(math.log2(max_val)) + 1)]
    for size in sizes:
        size_str = f"{size:{len(str(max_val))}d}"
        for matrix_fn, name in test_cases:
            matrices = [matrix_fn(size).cuda().float() for _ in range(attempts)]
            exact_vals = None
            for method_name, method_fn in methods:
                duration, approx_vals = measure_time(matrices, method_fn)
                if method_name == "exact":
                    exact_vals = approx_vals
                stats = display_stats(exact_vals, f"{name} ({size_str}) | {method_name}", approx_vals, duration)
                results.append({"size": size, "matrix_type": name.strip(), "method": method_name, **stats})
    return pd.DataFrame(results)


def plot_results(df):
    # Set a more modern aesthetic with clear colors
    sns.set_theme(style="whitegrid", font_scale=1.3)
    palette = sns.color_palette("viridis", n_colors=4)

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={"hspace": 0.35, "wspace": 0.25})

    # Add a title to the overall figure
    fig.suptitle("Singular Value Approximation Method Comparison", fontsize=20, y=0.98)

    # 1. Duration vs Size (Top Left)
    method_order = ["cholesky", "power_iter_0", "power_iter_1", "power_iter_2"]
    method_names = {
        "cholesky": "Cholesky",
        "power_iter_0": "Power Iteration (Default)",
        "power_iter_1": "Power Iteration (1 iter)",
        "power_iter_2": "Power Iteration (2 iter)",
    }

    # Filter out the exact method for time comparison and ensure consistent order
    plot_df = df[df["method"] != "exact"].copy()
    plot_df["method_name"] = plot_df["method"].map(method_names)

    # Add speedup compared to exact method
    exact_times = df[df["method"] == "exact"].set_index(["size", "matrix_type"])["duration"]
    plot_df["speedup"] = plot_df.apply(
        lambda row: exact_times.loc[(row["size"], row["matrix_type"])] / row["duration"], axis=1
    )

    # Plot duration vs size
    sns.lineplot(
        data=plot_df,
        x="size",
        y="duration",
        hue="method_name",
        style="method_name",
        markers=True,
        markersize=10,
        linewidth=3,
        ax=axes[0, 0],
        palette=palette,
        hue_order=[method_names[m] for m in method_order],
    )

    # Formatting for first plot
    axes[0, 0].set(
        xscale="log",
        yscale="log",
        title="Computation Time vs Matrix Size",
        xlabel="Matrix Size (n×n)",
        ylabel="Time (seconds)",
    )
    axes[0, 0].grid(True, which="both", ls="-", alpha=0.2)
    axes[0, 0].legend(title="Method", frameon=True, title_fontsize=14, fontsize=12)

    # Add exact method time as a reference line
    exact_avg = df[df["method"] == "exact"].groupby("size")["duration"].mean()
    axes[0, 0].plot(exact_avg.index, exact_avg.values, "r--", linewidth=2, alpha=0.7, label="Exact (SVD)")
    axes[0, 0].legend(title="Method", frameon=True, title_fontsize=14, fontsize=12)

    # 2. Relative Error vs Size (Top Right)
    sns.lineplot(
        data=plot_df,
        x="size",
        y="rel_error",
        hue="method_name",
        style="method_name",
        markers=True,
        markersize=10,
        linewidth=3,
        ax=axes[0, 1],
        palette=palette,
        hue_order=[method_names[m] for m in method_order],
    )

    # Formatting for second plot
    axes[0, 1].set(
        xscale="log", title="Relative Error vs Matrix Size", xlabel="Matrix Size (n×n)", ylabel="Relative Error"
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, which="both", ls="-", alpha=0.2)
    axes[0, 1].legend(title="Method", frameon=True, title_fontsize=14, fontsize=12)

    # 3. Speedup Factor vs Size (Bottom Left)
    sns.lineplot(
        data=plot_df,
        x="size",
        y="speedup",
        hue="method_name",
        style="method_name",
        markers=True,
        markersize=10,
        linewidth=3,
        ax=axes[1, 0],
        palette=palette,
        hue_order=[method_names[m] for m in method_order],
    )

    # Formatting for third plot
    axes[1, 0].set(
        xscale="log",
        yscale="log",
        title="Speedup vs Matrix Size (compared to exact SVD)",
        xlabel="Matrix Size (n×n)",
        ylabel="Speedup Factor (×)",
    )
    axes[1, 0].grid(True, which="both", ls="-", alpha=0.2)
    axes[1, 0].legend(title="Method", frameon=True, title_fontsize=14, fontsize=12)

    # 4. Error vs Matrix Type (Bottom Right) - Boxplot
    sns.boxplot(data=plot_df, x="method_name", y="rel_error", hue="matrix_type", ax=axes[1, 1], palette="Set2")

    # Formatting for fourth plot
    axes[1, 1].set(
        yscale="log",
        title="Relative Error by Method and Matrix Type",
        xlabel="Method",
        ylabel="Relative Error (log scale)",
    )
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].legend(title="Matrix Type", frameon=True, title_fontsize=14, fontsize=12)

    # Apply consistent formatting to all subplots
    for ax in axes.flatten():
        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(labelsize=12)

        # For logarithmic scales, add minor gridlines
        if ax.get_xscale() == "log":
            ax.xaxis.grid(True, which="minor", linestyle="--", alpha=0.2)
        if ax.get_yscale() == "log":
            ax.yaxis.grid(True, which="minor", linestyle="--", alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the figure title

    return fig


def main():
    set_torch()
    with torch._dynamo.utils.disable_cache_limit():
        results = test_singular_value_approx()
        fig = plot_results(results)
        fig.savefig("singular_value_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
