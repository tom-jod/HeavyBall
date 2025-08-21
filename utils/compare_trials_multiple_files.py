import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os
import importlib.util

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'serif'

MARKERS = ['o', '*', 'D', '^', 'v', '<', '>', 'P', '+', 's', 'h', 'X']


def load_experiments_from_file(file_path):
    """Load experiments_data_dict from a Python file."""
    try:
        spec = importlib.util.spec_from_file_location("experiments_module", file_path)
        experiments_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiments_module)
        if hasattr(experiments_module, 'experiments_data_dict'):
            return experiments_module.experiments_data_dict
        else:
            print(f"Error: 'experiments_data_dict' not found in {file_path}")
            return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def plot_aggregated_parameter_comparison(folder_path, output_prefix="Aggregated"):
    """Plot 2×2 grid comparing hyperparameter effects across multiple parsed_results files."""
    parsed_files = [f for f in os.listdir(folder_path) if f.startswith("parsed_experiments") and f.endswith(".py")]
    if not parsed_files:
        print(f"No parsed_results*.py files found in {folder_path}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Prepare parameter containers across all files
    parameters = [
        {"name": "Learning Rate", "key": "learning_rate", "use_log_norm": True, "cmap": plt.cm.seismic, "ax_idx": 0},
        {"name": r"$1-\beta_1$", "key": "beta1", "use_log_norm": True, "cmap": plt.cm.seismic, "ax_idx": 1},
        {"name": r"$1-\beta_2$", "key": "beta2", "use_log_norm": True, "cmap": plt.cm.seismic, "ax_idx": 2},
        {"name": "Weight Decay", "key": "weight_decay", "use_log_norm": True, "cmap": plt.cm.seismic, "ax_idx": 3},
    ]

    # For each subplot, collect values across all files
    param_values_all = {p["key"]: [] for p in parameters}
    for file_name in parsed_files:
        file_path = os.path.join(folder_path, file_name)
        experiments_data_dict = load_experiments_from_file(file_path)
        if experiments_data_dict is None:
            continue
        for i in range(len(experiments_data_dict)):
            trial = experiments_data_dict[f"Exp_{i+1}"]
            param_values_all["learning_rate"].append(trial["learning_rate"])
            param_values_all["beta1"].append(1 - trial["beta1"])
            param_values_all["beta2"].append(1 - trial["beta2"])
            param_values_all["weight_decay"].append(trial["weight_decay"])

    # Plot each subplot
    for param in parameters:
        ax = axes[param["ax_idx"]]
        values = param_values_all[param["key"]]

        if param["use_log_norm"]:
            min_val = min([v for v in values if v > 0])
            max_val = max(values)
            norm = mcolors.LogNorm(vmin=min_val, vmax=max_val)
        else:
            norm = mcolors.Normalize(vmin=min(values), vmax=max(values))

        cmap = param["cmap"]

        # Iterate files with different markers
        for idx, file_name in enumerate(sorted(parsed_files)):
            file_path = os.path.join(folder_path, file_name)
            experiments_data_dict = load_experiments_from_file(file_path)
            if experiments_data_dict is None:
                continue

            marker = MARKERS[idx % len(MARKERS)]
            label = (
                    file_name.replace("parsed_experiments", "")  # remove prefix
                            .replace(".py", "")                 # drop extension
                            .strip("_")                         # remove leading/trailing underscores
                            .replace("_", " ")                  # make underscores into spaces
                    )

            for i in range(len(experiments_data_dict)):
                trial = experiments_data_dict[f"Exp_{i+1}"]
                if param["key"] == "learning_rate":
                    val = trial["learning_rate"]
                elif param["key"] == "beta1":
                    val = 1 - trial["beta1"]
                elif param["key"] == "beta2":
                    val = 1 - trial["beta2"]
                else:
                    val = trial["weight_decay"]

                color = cmap(norm(val))
                test_accuracies = trial["test_accuracies"]
                x_vals = np.arange(len(test_accuracies))

                # Line + scatter with small markers
                ax.plot(x_vals, test_accuracies, color=color, alpha=0.8, linewidth=1.2)
                ax.scatter(x_vals, test_accuracies, color=color, marker=marker,
                           s=30, alpha=0.8, label=label if i == 0 else "")

        ax.set_ylim([0.4, 0.8])
        ax.set_xlabel("Steps (x1000)")
        ax.set_ylabel("Test Accuracy")
        #ax.set_title(f"Test Accuracy vs {param['name']}", fontsize=15)
        ax.grid(True, alpha=0.3)

          # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(param["name"])
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0e}"))

        # === ADD MARKERS TO COLORBAR ===
        for idx, file_name in enumerate(sorted(parsed_files)):
            file_path = os.path.join(folder_path, file_name)
            experiments_data_dict = load_experiments_from_file(file_path)
            if experiments_data_dict is None:
                continue

            marker = MARKERS[idx % len(MARKERS)]
            # x-position: left side for file 0, right side for file 1, etc.
            # 0 = left (0), 1 = right (1.1), 2 = slightly further right, etc.
            x_pos = -0.1 if idx == 0 else 1.1  

            for i in range(len(experiments_data_dict)):
                trial = experiments_data_dict[f"Exp_{i+1}"]
                if param["key"] == "learning_rate":
                    val = trial["learning_rate"]
                elif param["key"] == "beta1":
                    val = 1 - trial["beta1"]
                elif param["key"] == "beta2":
                    val = 1 - trial["beta2"]
                else:
                    val = trial["weight_decay"]

                y = norm(val)  # map value to [0,1]
                cbar.ax.plot([x_pos], [y], marker=marker, color="black",
                             transform=cbar.ax.transAxes, clip_on=False)


    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{output_prefix}_all_params.png", dpi=500, bbox_inches="tight")
    plt.show()


def main(folder_path, output_prefix=None):
    """Main function that loads all parsed results from a folder and plots aggregated scatter+line plots."""
    print(f"Loading experiments from folder: {folder_path}")

    if output_prefix is None:
        base_name = os.path.basename(os.path.normpath(folder_path))
        output_prefix = f"plots_{base_name}"

    plot_aggregated_parameter_comparison(folder_path, output_prefix)
    print(f"\nPlots saved with prefix: {output_prefix}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_experiments.py <folder_of_parsed_results> [output_prefix]")
        sys.exit(1)

    folder = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else None
    main(folder, prefix)
