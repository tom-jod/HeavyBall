import numpy as np

# Mean values extracted from your LaTeX tables
results = {
    "MNIST": {
        "AdamW": 1.43e-3,
        "Distributed Shampoo": 1.62e-4,
        "Schedule-Free AdamW": 7.92e-6,
        "NadamW": 5.27e-4,
        "Mars-AdamW": 3.31e-3,
        "SGD": 7.32e-3,
    },
    "SVHN": {
        "AdamW": 5.35e-2,
        "Distributed Shampoo": 5.39e-3,
        "Schedule-Free AdamW": 2.94e-2,
        "NadamW": 3.73e-2,
        "Mars-AdamW": 4.36e-2,
        "SGD": 2.15,
    },
    "Tolstoi": {
        "AdamW": 1.51,
        "Distributed Shampoo": 1.39,
        "Schedule-Free AdamW": 1.38,
        "NadamW": 1.34,
        "Mars-AdamW": 1.35,
        "SGD": 1.47,
    },
    "C10-wide": {
        "AdamW": 9.03e-3,
        "Distributed Shampoo": 3.83e-3,
        "Schedule-Free AdamW": 1.70e-3,
        "NadamW": 5.82e-4,
        "Mars-AdamW": 4.58e-3,
        "SGD": 9.95e-4,
    },
    "C100": {
        "AdamW": 3.33e-2,
        "Distributed Shampoo": 1.19e-2,
        "Schedule-Free AdamW": 7.46e-3,
        "NadamW": 4.42e-3,
        "Mars-AdamW": 7.21e-3,
        "SGD": 2.35e-2,
    },
}


def compute_relative_scores(results):
    # Collect optimizers
    optimizers = set()
    for bench in results.values():
        optimizers.update(bench.keys())

    scores = {opt: [] for opt in optimizers}

    # Per benchmark: find best optimizer, compute ratios
    for bench, bench_data in results.items():
        best = min(bench_data.values())
        for opt, mean_val in bench_data.items():
            scores[opt].append(best / mean_val)

    # Average over benchmarks
    avg_scores = {opt: np.mean(vals) for opt, vals in scores.items()}
    return avg_scores


def to_latex(value, digits=3):
    """Format a float into LaTeX scientific notation."""
    s = f"{value:.{digits}e}"
    base, exp = s.split("e")
    exp = int(exp)
    return f"${base}\\times 10^{{{exp}}}$"


if __name__ == "__main__":
    avg_scores = compute_relative_scores(results)
    for opt, score in avg_scores.items():
        print(f"{opt}: {to_latex(score)}")
