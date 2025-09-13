import numpy as np
import pandas as pd

# -----------------------------
# Data: (mean, std) for each benchmark
# First 3 benchmarks = first table, next 3 = second table
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

# -----------------------------
# Identify the best mean per benchmark
# -----------------------------
all_means = list(zip(*[ [m for m, _ in runs] for runs in data.values() ]))
best_means = [max(col) for col in all_means]

# -----------------------------
# Benchmarks to ignore per optimizer (e.g., SGD catastrophic failure)
# (indices: 0=first benchmark, ..., 5=sixth benchmark)
# -----------------------------
ignore = {
    "SGD": [1],  # ignore second benchmark (value=19.6)
}

# -----------------------------
# Compute normalized performance
# -----------------------------
results = {}
for opt, vals in data.items():
    norm_means, norm_highs, norm_lows = [], [], []
    for i, ((mean, std), best) in enumerate(zip(vals, best_means)):
        if opt in ignore and i in ignore[opt]:
            continue
        norm = mean / best
        norm_hi = (mean + std) / best
        norm_lo = max(0, (mean - std) / best)
        norm_means.append(norm)
        norm_highs.append(norm_hi)
        norm_lows.append(norm_lo)
    overall_mean = np.mean(norm_means)
    overall_low = np.mean(norm_lows)
    overall_high = np.mean(norm_highs)
    err_low = overall_mean - overall_low
    err_high = overall_high - overall_mean
    results[opt] = (overall_mean, err_low, err_high)

# -----------------------------
# Make a DataFrame
# -----------------------------
df = pd.DataFrame(results, index=["Mean", "ErrLow", "ErrHigh"]).T
df_percent = df.copy()
df_percent["Mean"] = df_percent["Mean"] * 100
df_percent["Err"] = ((df_percent["ErrLow"] + df_percent["ErrHigh"]) / 2) * 100

print("\nNormalized Overall Results (%):\n")
print(df_percent[["Mean", "Err"]].round(2).sort_values("Mean", ascending=False))
