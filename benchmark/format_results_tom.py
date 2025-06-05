import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib
plt.rcParams.update({'font.size': 14})



pd.set_option('future.no_silent_downcasting', True)

def parse_loss(loss_str):
    if loss_str.strip() == "inf":
        return float("inf")
    try:
        if "e" in loss_str:
            base, exp = loss_str.split("e")
            return float(base) * 10 ** float(exp)
        return float(loss_str)
    except (ValueError, IndexError):
        return float("nan")


def read_benchmark_results(path):
    """
    Read benchmark results from a file or all files in a directory.
    
    Args:
        path: Path to a file or directory
    
    Returns:
        DataFrame with benchmark results, averaged if there are duplicates
    """
    path = Path(path)
    
    # If path is a file, process it directly
    if path.is_file():
        return _process_single_file(path)
    
    # If path is a directory, process all files and combine results
    elif path.is_dir():
        all_data = []
        for file_path in path.glob('*'):
            if file_path.is_file():
                try:
                    df = _process_single_file(file_path)
                    all_data.append(df)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
        
        if not all_data:
            raise ValueError(f"No valid files found in directory {path}")
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Group by benchmark and optimizer, and average the numeric columns
        numeric_cols = ['runtime', 'loss', 'attempts']
        grouped_df = combined_df.groupby(['benchmark', 'optimizer', 'cautious', 'mars']).agg({
            'success': 'mean',  # This will give the success rate
            'runtime': 'mean',
            'loss': 'mean',
            'attempts': 'mean'
        }).reset_index()
        
        return grouped_df
    
    else:
        raise ValueError(f"Path {path} is neither a file nor a directory")

def process_str(x, truthy):
    if x == "No":
        return ""
    if x == "Yes":
        return f"{truthy}-"
    return f"{x}-"


def _process_single_file(file_path):
    """Process a single benchmark results file."""
    print(f"Processing file: {file_path}")
    with open(file_path, "r") as f:
        content = f.read()
    
    details_section = re.search(r"## Details\n\n(.*?)(?=\n\n|$)", content, re.DOTALL)
    if not details_section:
        raise ValueError(f"Could not find Details section in {file_path}")
    
    lines = details_section.group(1).strip().split("\n")[2:]
    data = []
    for line in lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 8:
            continue
        caution = process_str(parts[2], "cautious")
        mars = process_str(parts[3], "mars")
        optimizer = f"{caution}{mars}{parts[1]}"
        optimizer = optimizer.replace("Foreach", "").replace("Cached", "").strip()
        data.append({
            "benchmark": parts[0],
            "optimizer": optimizer,
            "cautious": parts[2] == "Yes",
            "mars": parts[3] == "Yes",
            "success": parts[4] == "✓",
            "runtime": float(parts[5].replace("s", "")),
            "loss": parse_loss(parts[6]),
            "attempts": int(parts[7]),
        })
    
    return pd.DataFrame(data)


def create_result_matrix(df):
    benchmarks = sorted(df["benchmark"].unique())
    optimizers = sorted(df["optimizer"].unique())

    success_matrix = pd.DataFrame(index=benchmarks, columns=optimizers)
    attempts_matrix = pd.DataFrame(index=benchmarks, columns=optimizers)
    runtime_matrix = pd.DataFrame(index=benchmarks, columns=optimizers)
    loss_matrix = pd.DataFrame(index=benchmarks, columns=optimizers)

    for _, row in df.iterrows():
        success_matrix.loc[row["benchmark"], row["optimizer"]] = row["success"]
        attempts_matrix.loc[row["benchmark"], row["optimizer"]] = row["attempts"]
        runtime_matrix.loc[row["benchmark"], row["optimizer"]] = row["runtime"]
        loss_matrix.loc[row["benchmark"], row["optimizer"]] = row["loss"]

    return success_matrix, attempts_matrix, runtime_matrix, loss_matrix


def normalize_row_attempts(row_attempts, row_success):
    """Normalize attempts within a row, considering only successful runs"""
    # Convert to boolean and handle NaN
    success_mask = to_bool(row_success)
    successful_attempts = row_attempts[success_mask]

    if len(successful_attempts) == 0:
        return pd.Series(np.nan, index=row_attempts.index)

    min_attempts = successful_attempts.min()
    max_attempts = successful_attempts.max()

    if max_attempts == min_attempts:
        # all successful attempts are the same, so they should be 0
        normalized = pd.Series(0, index=row_attempts.index)
    else:
        normalized = (row_attempts - min_attempts) / (max_attempts - min_attempts)

    # Set unsuccessful attempts to NaN
    normalized[~success_mask] = np.nan
    return normalized


attempts_cmap = matplotlib.colormaps.get_cmap('RdYlGn')  # Or try 'RdYlGn_r' for green to red reversed

def get_color_for_cell(normalized_value, success, best_in_row=False):
    if pd.isna(normalized_value) or not success:
        return "#000000"  # Red for failure

    # Use colormap to get RGB color (normalized_value is between 0 and 1)
    color = attempts_cmap(1 - normalized_value)  # Invert so lower attempts → greener

    # Add golden tint if best
    if best_in_row:
        color = np.array(color[:3]) * 0.9 + np.array([0.8, 0.6, 0.0]) * 0.1
        return tuple(color)

    return color[:3]  # Remove alpha


def to_bool(x):
    return x.fillna(False).infer_objects(copy=False).astype(bool)


def create_visual_matrix(success_matrix, attempts_matrix, runtime_matrix, loss_matrix):
    plt.style.use("default")
    fig = plt.figure(figsize=(20, 15), facecolor="white")

    # Create grid for multiple panels with adjusted width ratios
    gs = plt.GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[4, 0.1, 1.5, 1.5],
        height_ratios=[1, 1],
        wspace=0.3,
        hspace=0.3,
    )

    # Main heatmap
    main_ax = fig.add_subplot(gs[:, 0])

    # Normalize attempts per row
    normalized_attempts = pd.DataFrame(index=success_matrix.index, columns=success_matrix.columns)

    for idx in success_matrix.index:
        normalized_attempts.loc[idx] = normalize_row_attempts(attempts_matrix.loc[idx], success_matrix.loc[idx])

    # Calculate best performers per benchmark
    best_performers = pd.DataFrame(index=success_matrix.index, columns=success_matrix.columns)

    for idx in success_matrix.index:
        row_success = success_matrix.loc[idx]
        row_attempts = attempts_matrix.loc[idx]
        row_runtime = runtime_matrix.loc[idx]

        # Find best performer (successful with minimum attempts, then minimum runtime)
        successful_mask = to_bool(row_success)
        if successful_mask.any():
            min_attempts = row_attempts[successful_mask].min()
            min_attempts_mask = (row_attempts == min_attempts) & successful_mask
            
            if min_attempts_mask.sum() > 1 and min_attempts_mask.sum()!=len(success_matrix.columns):
                # If multiple with same attempts, use runtime as tiebreaker
                print(min_attempts_mask)
                best_idx = row_runtime[min_attempts_mask].idxmin()
              
            else:
                best_idx = min_attempts_mask[min_attempts_mask].index[0]

            best_performers.loc[idx, best_idx] = True
        
    # Convert to boolean after loop
    best_performers = to_bool(best_performers)

    # === Add summary row with star counts ===
    star_counts = best_performers.sum(axis=0)
    best_performers_with_header = pd.concat(
        [pd.DataFrame([star_counts], index=["★ Count"]), best_performers],
        axis=0
)
    
    # Plot main heatmap
    for i in range(success_matrix.shape[0] + 1):
        for j in range(success_matrix.shape[1]):
            success = success_matrix.iloc[i-1, j]
            attempts = attempts_matrix.iloc[i-1, j]
            runtime = runtime_matrix.iloc[i-1, j]
            normalized = normalized_attempts.iloc[i-1, j]
            is_best = best_performers_with_header.iloc[i, j]
            
            if i == 0:
                # Top summary row with star counts
                count = int(best_performers_with_header.iloc[0, j])
                main_ax.text(
                    j,
                    i,
                    f"{count}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=14,
                    fontweight="bold",
                )
                continue
            # Get cell color
            color = get_color_for_cell(normalized, success, is_best)

            # Create cell rectangle
            rect = Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                facecolor=color,
                alpha=1.0,
                edgecolor="white",
                linewidth=1,
            )
            main_ax.add_patch(rect)

            if pd.notna(attempts):
                # Add attempt count and runtime
                attempts_text = f"{int(attempts)}"
                runtime_text = f"{runtime:.1f}s"

                # Determine text color based on background brightness
                if isinstance(color, tuple):
                    brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                    text_color = "white" if brightness < 0.65 else "black"
                else:
                    text_color = "white" if color == "#000000" else "black"

                # Add text with better formatting
                main_ax.text(
                    j,
                    i ,
                    attempts_text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=14,
                    fontweight="bold",
                )
                """
                main_ax.text(
                    j,
                    i + 0.15,
                    runtime_text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
                """
                # Add star for best performer
                if is_best:
                    main_ax.text(
                        j - 0.4,
                        i - 0.4,
                        "★",
                        ha="center",
                        va="center",
                        color="#FFD700",
                        fontsize=14,
                        fontweight="bold",
                    )
                

    # Add grid lines
    for i in range(success_matrix.shape[0] + 1):
        main_ax.axhline(y=i - 0.5, color="#DDD", linewidth=0.5, alpha=0.5)
    for j in range(success_matrix.shape[1] + 1):
        main_ax.axvline(x=j - 0.5, color="#DDD", linewidth=0.5, alpha=0.5)

    # Format axis labels
    main_ax.set_xticks(range(len(success_matrix.columns)))
    main_ax.set_yticks(range(len(success_matrix.index) + 1))
    main_ax.set_xticklabels(success_matrix.columns, rotation=45, ha="right", fontsize=14, fontweight="bold")
    main_ax.set_yticklabels(["Benchmarks won (★)"] + list(success_matrix.index), fontsize=14, fontweight="bold")
    

    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])

    return fig


def main(file: str = "benchmark_results.md"):
    df = read_benchmark_results(file)
    success_matrix, attempts_matrix, runtime_matrix, loss_matrix = create_result_matrix(df)

    # Create the enhanced visual matrix
    _fig = create_visual_matrix(success_matrix, attempts_matrix, runtime_matrix, loss_matrix)

    # Save with high quality
    plt.savefig("benchmark_matrix.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.5)
    plt.close()

    # Print text summary
    print("\nSummary Statistics:")
    success_rates = success_matrix.mean().sort_values(ascending=False)
    avg_attempts = attempts_matrix.mean()
    median_attempts = attempts_matrix.median()
    avg_runtime = runtime_matrix.mean()
    median_runtime = runtime_matrix.median()
    avg_loss = loss_matrix.mean()
    median_loss = loss_matrix.median()

    print("\nSuccess Rates by Optimizer:")
    for optimizer in success_rates.index:
        rate = success_rates[optimizer] * 100
        print(f"{optimizer}: {rate:.1f}%")

    print("\nAverage Attempts by Optimizer (Mean, Median):")
    for optimizer in avg_attempts.index:
        mean_val = avg_attempts[optimizer]
        median_val = median_attempts[optimizer]
        print(f"{optimizer}: Mean={mean_val:.1f}, Median={median_val:.1f}")

    print("\nAverage Runtime by Optimizer (Mean, Median):")
    for optimizer in avg_runtime.index:
        mean_val = avg_runtime[optimizer]
        median_val = median_runtime[optimizer]
        print(f"{optimizer}: Mean={mean_val:.1f}s, Median={median_val:.1f}s")

    print("\nAverage Loss by Optimizer (Mean, Median):")
    for optimizer in avg_loss.index:
        mean_val = avg_loss[optimizer]
        median_val = median_loss[optimizer]
        print(f"{optimizer}: Mean={mean_val:.2e}, Median={median_val:.2e}")


if __name__ == "__main__":
    typer.run(main)