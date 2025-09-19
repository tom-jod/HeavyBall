import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import typer
import ast
from pathlib import Path
import datetime
import glob

app = typer.Typer()

def safe_eval_list(list_str):
    """Safely evaluate a string representation of a list."""
    if pd.isna(list_str) or list_str == '' or list_str == '[]':
        return []
    try:
        if isinstance(list_str, str):
            return ast.literal_eval(list_str)
        return list_str
    except:
        return []

def get_run_settings(row):
    """Extract run settings to identify matching configurations."""
    settings = {
        'optimizer': row.get('optimizer', ''),
        'steps': row.get('steps', 0),
        'trials': row.get('trials', 0),
    }
    return tuple(sorted(settings.items()))

def load_and_combine_csvs(folder_path):
    """Load all CSV files from folder and combine them."""
    folder = Path(folder_path)
    
    # Find all CSV files that look like detailed results
    csv_files = []
    for pattern in ['*detailed_results*.csv', '*_results*.csv', '*.csv']:
        csv_files.extend(glob.glob(str(folder / pattern)))
    
    if not csv_files:
        print(f"No CSV files found in {folder}")
        return None
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {Path(f).name}")
    
    # Load and combine all CSV files
    all_dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df['source_file'] = Path(csv_file).name
            all_dfs.append(df)
            print(f"Loaded {len(df)} rows from {Path(csv_file).name}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_dfs:
        print("No valid CSV files could be loaded")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined total: {len(combined_df)} rows")
    
    return combined_df

def plot_from_folder(folder_path, output_dir=None, benchmark_name=None):
    """Create plots from all CSV files in a folder."""
    
    # Load and combine all CSV files
    df = load_and_combine_csvs(folder_path)
    if df is None:
        return
    
    # Set default output directory and benchmark name if not provided
    if output_dir is None:
        output_dir = Path(folder_path)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    if benchmark_name is None:
        benchmark_name = Path(folder_path).name
    
    # Filter successful runs
    successful_df = df[df['success'] == True].copy()
    
    if successful_df.empty:
        print("No successful runs found in the CSV files!")
        return
    
    print(f"Found {len(successful_df)} successful runs")
    
    # Parse list columns
    list_columns = ['loss_trajectory', 'test_accuracies', 'grad_variances']
    for col in list_columns:
        if col in successful_df.columns:
            successful_df[col] = successful_df[col].apply(safe_eval_list)
    
    # Group by optimizer and settings (to aggregate runs with same configuration)
    successful_df['run_config'] = successful_df.apply(get_run_settings, axis=1)
    
    # Group by optimizer for plotting
    optimizer_groups = successful_df.groupby('optimizer')
    
    # Print aggregation info
    print("\nAggregation summary:")
    for optimizer, group in optimizer_groups:
        n_runs = len(group)
        n_files = group['source_file'].nunique()
        print(f"  {optimizer}: {n_runs} runs from {n_files} files")
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Loss Trajectories
    plt.figure(figsize=(12, 8))
    has_loss_data = False
    
    for optimizer, group in optimizer_groups:
        # Get all loss trajectories for this optimizer
        trajectories = [traj for traj in group['loss_trajectory'] if len(traj) > 0]
        
        if not trajectories:
            continue
            
        has_loss_data = True
        
        # Find minimum length
        min_length = min(len(traj) for traj in trajectories)
        
        # Truncate all trajectories
        truncated = [traj[:min_length] for traj in trajectories]
        
        if truncated:
            # Calculate statistics
            mean_traj = np.mean(truncated, axis=0)
            std_traj = np.std(truncated, axis=0)
            se_traj = std_traj / np.sqrt(len(truncated))
            
            # Plot
            x = np.arange(min_length)
            plt.plot(x, mean_traj, label=f"{optimizer} (n={len(truncated)})", linewidth=2)
            plt.fill_between(x, mean_traj - se_traj, mean_traj + se_traj, alpha=0.3)
    
    if has_loss_data:
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'{benchmark_name} - Loss Trajectories')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig(f"{output_dir}/{benchmark_name}_loss_curves_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved loss curves to: {output_dir}/{benchmark_name}_loss_curves_{timestamp}.png")
    else:
        plt.close()
        print("No loss trajectory data found")
    
    # Plot 2: Gradient Variances
    if 'grad_variances' in successful_df.columns:
        plt.figure(figsize=(12, 8))
        has_grad_data = False
        
        for optimizer, group in optimizer_groups:
            # Get all gradient variance trajectories for this optimizer
            trajectories = [traj for traj in group['grad_variances'] if len(traj) > 0]
            
            if not trajectories:
                continue
                
            has_grad_data = True
            
            # Find minimum length
            min_length = min(len(traj) for traj in trajectories)
            
            # Truncate all trajectories
            truncated = [traj[:min_length] for traj in trajectories]
            
            if truncated:
                # Calculate statistics
                mean_traj = np.mean(truncated, axis=0)
                std_traj = np.std(truncated, axis=0)
                se_traj = std_traj / np.sqrt(len(truncated))
                
                # Plot
                x = np.arange(min_length)
                plt.plot(x, mean_traj, label=f"{optimizer} (n={len(truncated)})", linewidth=2)
                plt.fill_between(x, mean_traj - se_traj, mean_traj + se_traj, alpha=0.3)
        
        if has_grad_data:
            plt.xlabel('Iteration')
            plt.ylabel('Gradient Variance')
            plt.title(f'{benchmark_name} - Gradient Variances')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.savefig(f"{output_dir}/{benchmark_name}_grad_variances_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved gradient variances to: {output_dir}/{benchmark_name}_grad_variances_{timestamp}.png")
        else:
            plt.close()
            print("No gradient variance data found")
    
    # Plot 3: Test Accuracies
    if 'test_accuracies' in successful_df.columns:
        plt.figure(figsize=(12, 8))
        has_acc_data = False
        
        for optimizer, group in optimizer_groups:
            # Get all test accuracy trajectories for this optimizer
            acc_trajectories = [acc for acc in group['test_accuracies'] if len(acc) > 0]
            
            if not acc_trajectories:
                continue
                
            has_acc_data = True
            
            # Find minimum length
            min_length = min(len(acc) for acc in acc_trajectories)
            
            # Truncate all trajectories
            truncated = [acc[:min_length] for acc in acc_trajectories]
            
            if truncated:
                # Calculate statistics
                mean_acc = np.mean(truncated, axis=0)
                std_acc = np.std(truncated, axis=0)
                se_acc = std_acc / np.sqrt(len(truncated))
                
                # Plot
                x = np.arange(min_length)
                plt.plot(x, mean_acc, label=f"{optimizer} (n={len(truncated)})", linewidth=2)
                plt.fill_between(x, mean_acc - se_acc, mean_acc + se_acc, alpha=0.3)
        
        if has_acc_data:
            plt.xlabel('Epoch')
            plt.ylabel('Test Accuracy (%)')
            plt.title(f'{benchmark_name} - Test Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{output_dir}/{benchmark_name}_test_accuracy_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved test accuracy to: {output_dir}/{benchmark_name}_test_accuracy_{timestamp}.png")
        else:
            plt.close()
            print("No test accuracy data found")

@app.command()
def main(
    folder_path: str = typer.Argument(..., help="Path to folder containing CSV files with detailed results"),
    output_dir: str = typer.Option(None, help="Output directory for plots (default: same as input folder)"),
    benchmark_name: str = typer.Option(None, help="Benchmark name for plot titles (default: folder name)"),
):
    """
    Create plots from all CSV files in a folder, aggregating results with matching configurations.
    
    Example:
        python plot_from_folder.py results/
        python plot_from_folder.py results/ --output-dir=plots --benchmark-name=MyBenchmark
    """
    
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return
    
    if not folder_path.is_dir():
        print(f"Error: Path is not a directory: {folder_path}")
        return
    
    print(f"Reading results from folder: {folder_path}")
    plot_from_folder(folder_path, output_dir, benchmark_name)
    print("Plotting completed!")

if __name__ == "__main__":
    app()