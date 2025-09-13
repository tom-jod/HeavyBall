import os
import ast
import numpy as np

def extract_loss_trajectories_from_file(filepath):
    """Extracts all loss_trajectory lists from a log file."""
    trajectories = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip().startswith("{'runtime':"):
                try:
                    data = ast.literal_eval(line.strip())
                    if "loss_trajectory" in data:
                        trajectories.append(data["loss_trajectory"])
                except Exception as e:
                    print(f"Failed to parse line in {filepath}: {e}")
    return trajectories

def format_scientific_latex(value, error):
    """Formats mean ± std in LaTeX scientific notation."""
    # Use scientific notation with 2 significant figures for value and error
    mean_str = f"{value:.2e}"
    std_str = f"{error:.0e}"

    # Convert Python-style 'e' to LaTeX \times10^
    def sci_to_latex(s):
        base, exp = s.split("e")
        exp = int(exp)
        return f"{base}\\times 10^{{{exp}}}"

    return f"${sci_to_latex(mean_str)} \\pm {sci_to_latex(std_str)}$"

def analyze_folder(folder):
    """Scans all files, extracts minima, prints LaTeX mean ± std."""
    all_minima = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            continue
        runs = extract_loss_trajectories_from_file(filepath)
        for traj in runs:
            if traj:
                all_minima.append(min(traj))

    if not all_minima:
        print("No loss_trajectory data found.")
        return None

    mean_min = np.mean(all_minima)
    std_min = np.std(all_minima)

    latex_str = format_scientific_latex(mean_min, std_min)
    print(latex_str)
    return latex_str


if __name__ == "__main__":
    folder = "logs/report_results/loss_analysis"  # <-- replace with your folder
    analyze_folder(folder)
