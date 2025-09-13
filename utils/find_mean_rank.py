import numpy as np
import pandas as pd

# -----------------------------
# Data (means, stds) without last aggregated column
# -----------------------------
data = {
    "AdamW": [
        (98.64, 0.01), (94.1, 0.4), (57.0, 3.0),
        (95.9, 0.2), (76.7, 0.8)
    ],
    "Distributed Shampoo": [
        (98.36, 0.02), (94.29, 0.01), (59.27, 0.05),
        (94.4, 0.2), (71.4, 0.2)
    ],
    "Schedule-Free AdamW": [
        (98.25, 0.09), (94.4, 0.5), (59.93, 0.09),
        (93.9, 0.2), (70.3, 0.2)
    ],
    "NadamW": [
        (98.49, 0.07), (95.4, 0.8), (60.46, 0.06),
        (95.2, 0.2), (73.6, 0.3)
    ],
    "Mars-AdamW": [
        (98.58, 0.05), (94.4, 0.7), (59.7, 0.1),
        (95.5, 0.4), (75.8, 0.6)
    ],
    "SGD": [
        (98.517, 0.005), (19.6, 0.0), (57.6, 0.3),
        (95.4, 0.2), (75.0, 2.0)
    ],
    "Mars-NadamW": [
        (98.52, 0.02), (95.91, 0.06), (60.49, 0.09),
        (95.23, 0.02), (73.5, 0.4)
    ],

}

optimizers = list(data.keys())
benchmarks = range(len(next(iter(data.values()))))

# -----------------------------
# Helper to compute mean ranks with adjustments
# -----------------------------
def compute_rank_for_optimizer(target_opt, case="mean"):
    """
    Compute average rank across benchmarks for one optimizer
    case = "mean", "best", "worst"
    """
    ranks_per_bench = []
    for b in benchmarks:
        # baseline: all means
        scores = {opt: data[opt][b][0] for opt in optimizers}
        mean, std = data[target_opt][b]

        # adjust only target optimizer
        if case == "best":
            scores[target_opt] = mean + std
        elif case == "worst":
            scores[target_opt] = mean - std

        # rank (higher is better)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rank_dict = {opt: i+1 for i, (opt, _) in enumerate(sorted_scores)}

        ranks_per_bench.append(rank_dict[target_opt])

    return np.mean(ranks_per_bench)

# -----------------------------
# Compute all ranks
# -----------------------------
results = {}
for opt in optimizers:
    mean_rank = compute_rank_for_optimizer(opt, "mean")
    best_rank = compute_rank_for_optimizer(opt, "best")
    worst_rank = compute_rank_for_optimizer(opt, "worst")
    results[opt] = (mean_rank, best_rank, worst_rank)

# -----------------------------
# Format nicely
# -----------------------------
df = pd.DataFrame(results, index=["Mean Rank", "Best Case", "Worst Case"]).T
df["Range"] = df.apply(lambda row: f"{row['Mean Rank']:.2f} [{row['Best Case']:.2f}, {row['Worst Case']:.2f}]", axis=1)
df = df.sort_values("Mean Rank")

print("\nAverage Ranking of Optimizers (range vs others' means):\n")
print(df[["Range"]])
