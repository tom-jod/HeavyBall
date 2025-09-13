import os
import glob
import ast
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Config ---
log_folder = "logs/report_results/plotting"   # <-- adjust
plot_mode = "all"   # "range" = mean + shaded min/max
                      # "all"   = plot every run individually
smooth = 1           # smoothing window size (1 = no smoothing)
pad_mode = "edge"     # "edge" or "reflect" (how to pad at boundaries)
verbose = True        # print run-length diagnostics

results = defaultdict(list)

# --- Helper: smoothing with padding (keeps same length) ---
def smooth_curve(x, window=1, mode="edge"):
    if window <= 1:
        return x.copy()
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return 
    kernel = np.ones(window, dtype=float) / window
    left = window // 2
    right = window - 1 - left
    # pad with edge or reflect to avoid zero-bias at edges
    xp = np.pad(x, (left, right), mode=mode)
    sm = np.convolve(xp, kernel, mode="valid")
    # sm should have same length as x
    assert sm.shape[0] == x.shape[0], f"smoothed length {sm.shape[0]} != original {x.shape[0]}"
    return sm

# --- Read logs ---
for filepath in glob.glob(os.path.join(log_folder, "*.out")):
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    record = ast.literal_eval(line)
                except Exception:
                    continue
                
                if "optimizer" in record and "test_accuracies" in record:
                    optimizer = record["optimizer"]
                    accs = np.array(record["test_accuracies"], dtype=float)
                    results[optimizer].append(accs)

# quick diagnostics
if verbose:
    print("Found optimizers and run lengths:")
    for opt, runs in results.items():
        lengths = [len(r) for r in runs]
        print(f"  {opt}: {len(runs)} run(s), lengths = {lengths}, min_len = {min(lengths)}")

# --- Prepare consistent colors ---
optimizers = list(results.keys())
cmap = plt.get_cmap("tab10")
color_map = {opt: cmap(i % 10) for i, opt in enumerate(optimizers)}

# --- Plotting ---
plt.figure(figsize=(9, 6))

for optimizer, runs in results.items():
    # Align to the shortest run
    min_len = min(len(r) for r in runs)
    runs = [r[:min_len] for r in runs]
    stacked = np.stack(runs)  # shape (num_runs, min_len)
    steps = np.arange(1, min_len + 1)
    color = color_map[optimizer]

    if plot_mode == "range":
        mean = stacked.mean(axis=0)
        min_vals = stacked.min(axis=0)
        max_vals = stacked.max(axis=0)

        # smooth (keeps same length)
        mean_s = smooth_curve(mean, smooth, mode=pad_mode)
        min_s = smooth_curve(min_vals, smooth, mode=pad_mode)
        max_s = smooth_curve(max_vals, smooth, mode=pad_mode)

        plt.plot(steps, mean_s, label=optimizer, color=color)
        plt.fill_between(steps, min_s, max_s, alpha=0.2, color=color)

    elif plot_mode == "all":
        for r in runs:
            r_s = smooth_curve(r, smooth, mode=pad_mode)
            plt.plot(steps, r_s, color=color, alpha=0.5, linewidth=1)
        # one legend entry per optimizer
        plt.plot([], [], color=color, label=optimizer)

# --- Style ---
plt.ylim((0.95, 1))          # keep your limits if desired
plt.xlabel("Evaluation step")
plt.ylabel("Test accuracy")
#plt.title("Optimizer Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("MNIST_Accuracies.png")

