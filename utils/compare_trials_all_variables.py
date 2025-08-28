import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import importlib.util

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'serif'
def load_experiments_from_file(file_path):
    """
    Load experiments_data_dict from a Python file
    """
    try:
        # Load the module from file path
        spec = importlib.util.spec_from_file_location("experiments_module", file_path)
        experiments_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiments_module)
        # Get the experiments_data_dict
        if hasattr(experiments_module, 'experiments_data_dict'):
            return experiments_module.experiments_data_dict
        else:
            print(f"Error: 'experiments_data_dict' not found in {file_path}")
            return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None
  
def plot_parameter_comparison(experiments_data_dict, output_prefix="Compare_target_trials"):
    """
    Create 4 subplots showing test accuracy curves colored by different parameters
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()  # Make it easier to index
    n_trials = len(experiments_data_dict)
    
    # Extract all parameter values
    learning_rates = []
    beta1_values = []  # Will store 1-beta1
    beta2_values = []  # Will store 1-beta2
    weight_decays = []
    
    for i in range(n_trials):
        trial = experiments_data_dict[f"Exp_{i+1}"]
        learning_rates.append(trial['learning_rate'])
        beta1_values.append(1 - trial['beta1'])  # Convert to 1-beta1
        beta2_values.append(1 - trial['beta2'])  # Convert to 1-beta2
        weight_decays.append(trial['weight_decay'])
    
    # Define parameters and their properties
    parameters = [
        {
            'name': 'Learning Rate',
            'values': learning_rates,
            'use_log_norm': True,
            'cmap': plt.cm.seismic,
            'ax_idx': 0
        },
        {
            'name': r'$1-\beta_1$',  # Updated label
            'values': beta1_values,
            'use_log_norm': True,    # Changed to True for log scale
            'cmap': plt.cm.seismic,
            'ax_idx': 1
        },
        {
            'name': r'$1-\beta_2$',  # Updated label
            'values': beta2_values,
            'use_log_norm': True,    # Changed to True for log scale
            'cmap': plt.cm.seismic,
            'ax_idx': 2
        },
        {
            'name': 'Weight Decay',
            'values': weight_decays,
            'use_log_norm': True,
            'cmap': plt.cm.seismic,
            'ax_idx': 3
        }
    ]
    
    # Create each subplot
    for param in parameters:
        ax = axes[param['ax_idx']]
        
        # Create normalization
        if param['use_log_norm']:
            # For parameters that span many orders of magnitude
            # Handle potential zero values for log scale
            min_val = min([v for v in param['values'] if v > 0])
            max_val = max(param['values'])
            norm = mcolors.LogNorm(vmin=min_val, vmax=max_val)
        else:
            # For parameters with smaller ranges
            norm = mcolors.Normalize(vmin=min(param['values']), vmax=max(param['values']))
        
        # Map parameter values to colors
        colors = param['cmap'](norm(param['values']))
        
        # Plot each trial
        for i in range(n_trials):
            trial = experiments_data_dict[f"Exp_{i+1}"]
            test_accuracies = trial['test_accuracies']
            ax.plot(range(len(test_accuracies)), test_accuracies,
                   color=colors[i], alpha=0.8, linewidth=1.5)
        
        # Customize subplot
        ax.set_ylim([0.0, 1])
        ax.set_xlabel('Steps (x1000)')
        ax.set_ylabel('Test Accuracy')
        #ax.set_title(f'Test Accuracy vs {param["name"]}', fontsize=15)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=param['cmap'], norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(param['name'])
        
        # Format colorbar labels for better readability
        if param['name'] == 'Learning Rate':
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0e}'))
        elif param['name'] == 'Weight Decay':
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0e}'))
        elif param['name'] in [r'$1-\beta_1$', r'$1-\beta_2$']:
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0e}'))

        ### ADD STARS TO COLORBAR ###
        for val in param['values']:
            y = norm(val)  # normalized position (0–1)
            cbar.ax.plot([1.05], [y], marker="*", color="black",
                         transform=cbar.ax.transAxes, clip_on=False)

    
    plt.tight_layout()
    print(output_prefix)
    plt.savefig(f'{output_prefix}_all_params', dpi=500, bbox_inches='tight')
    plt.show()
  
def plot_parameter_statistics(experiments_data_dict, output_prefix="Parameter_Analysis"):
    """
    Additional function to show parameter distributions and final accuracies
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    n_trials = len(experiments_data_dict)
    
    # Extract data
    learning_rates = []
    beta1_values = []  # Will store 1-beta1
    beta2_values = []  # Will store 1-beta2
    weight_decays = []
    final_accuracies = []
    max_accuracies = []
    
    for i in range(n_trials):
        trial = experiments_data_dict[f"Exp_{i+1}"]
        learning_rates.append(trial['learning_rate'])
        beta1_values.append(1 - trial['beta1'])  # Convert to 1-beta1
        beta2_values.append(1 - trial['beta2'])  # Convert to 1-beta2
        weight_decays.append(trial['weight_decay'])
        final_accuracies.append(trial['test_accuracies'][-1])
        max_accuracies.append(max(trial['test_accuracies']))
    
    # Parameter vs Final Accuracy scatter plots
    parameters = [
        {'name': r'Learning Rate', 'values': learning_rates, 'log_scale': True},
        {'name': r'$1-\beta_1$', 'values': beta1_values, 'log_scale': True},  # Updated
        {'name': r'$1-\beta_2$', 'values': beta2_values, 'log_scale': True},  # Updated
        {'name': r'Weight Decay', 'values': weight_decays, 'log_scale': True}
    ]
    
    for i, param in enumerate(parameters):
        ax = axes[i]
        
        # Scatter plot with final accuracy
        scatter = ax.scatter(param['values'], final_accuracies,
                           c=max_accuracies, cmap='RdYlGn',
                           alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(param['name'])
        ax.set_ylabel('Final Test Accuracy')
        ax.set_title(f'Final Accuracy vs {param["name"]}')
        
        if param['log_scale']:
            ax.set_xscale('log')
        
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Max Accuracy')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_vs_Accuracy', dpi=500, bbox_inches='tight')
    plt.show()
  
def print_parameter_summary(experiments_data_dict):
    """
    Print summary statistics of parameters
    """
    n_trials = len(experiments_data_dict)
    learning_rates = []
    beta1_values = []  # Will store 1-beta1
    beta2_values = []  # Will store 1-beta2
    weight_decays = []
    final_accuracies = []
    
    for i in range(n_trials):
        trial = experiments_data_dict[f"Exp_{i+1}"]
        learning_rates.append(trial['learning_rate'])
        beta1_values.append(1 - trial['beta1'])  # Convert to 1-beta1
        beta2_values.append(1 - trial['beta2'])  # Convert to 1-beta2
        weight_decays.append(trial['weight_decay'])
        final_accuracies.append(trial['test_accuracies'][-1])
    
    print("=== Parameter Summary ===")
    print(f"Number of experiments: {n_trials}")
    print(f"Learning Rate: min={min(learning_rates):.2e}, max={max(learning_rates):.2e}")
    print(f"1-Beta1: min={min(beta1_values):.6f}, max={max(beta1_values):.6f}")  # Updated
    print(f"1-Beta2: min={min(beta2_values):.6f}, max={max(beta2_values):.6f}")  # Updated
    print(f"Weight Decay: min={min(weight_decays):.2e}, max={max(weight_decays):.2e}")
    print(f"Final Accuracy: min={min(final_accuracies):.4f}, max={max(final_accuracies):.4f}")
    
    # Find best performing experiment
    best_idx = np.argmax(final_accuracies)
    best_trial = experiments_data_dict[f"Exp_{best_idx+1}"]
    print(f"\nBest performing experiment: Exp_{best_idx+1}")
    print(f"  Learning Rate: {best_trial['learning_rate']:.2e}")
    print(f"  1-Beta1: {1 - best_trial['beta1']:.6f}")  # Updated
    print(f"  1-Beta2: {1 - best_trial['beta2']:.6f}")  # Updated
    print(f"  Weight Decay: {best_trial['weight_decay']:.2e}")
    print(f"  Final Accuracy: {final_accuracies[best_idx]:.4f}")
  
def main(experiments_file_path, output_prefix=None):
    """
    Main function that loads experiments and creates plots
    Args:
        experiments_file_path (str): Path to the Python file containing experiments_data_dict
        output_prefix (str, optional): Prefix for output files
    """
    # Load experiments data
    print(f"Loading experiments from: {experiments_file_path}")
    experiments_data_dict = load_experiments_from_file(experiments_file_path)
    
    if experiments_data_dict is None:
        print("Failed to load experiments data!")
        return
    
    # Set output prefix based on input file if not provided
    if output_prefix is None:
        import os
        base_name = os.path.splitext(os.path.basename(experiments_file_path))[0]
        output_prefix = f"plots_{base_name}"
    
    # Print parameter summary
    print_parameter_summary(experiments_data_dict)
    
    # Create the main 4-subplot comparison
    print("\nCreating parameter comparison plots...")
    plot_parameter_comparison(experiments_data_dict, output_prefix)
    
    # Create additional analysis plots
    print("Creating parameter vs accuracy analysis...")
    plot_parameter_statistics(experiments_data_dict, output_prefix)
    
    print(f"\nPlots saved with prefix: {output_prefix}")
  
if __name__ == "__main__":
    # Command line usage
    if len(sys.argv) < 2:
        print("Usage: python plot_experiments.py <experiments_file.py> [output_prefix]")
        print("Example: python plot_experiments.py parsed_experiments.py")
        print("Example: python plot_experiments.py parsed_experiments.py my_analysis")
        sys.exit(1)
    
    experiments_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(experiments_file, output_prefix)