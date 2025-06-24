import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Updated data with error bars - trivial benchmarks
benchmark_data = {
    'rastrigin-trivial': {'condition_number': 0.9999999998387041, 'condition_std_err': 3.845925372767128e-17, 'n_params': 2},
    'gradient_delay-trivial': {'condition_number': None, 'condition_std_err': None, 'n_params': 256},
    'rosenbrock-trivial': {'condition_number': 861.6806699956081, 'condition_std_err': 0.008240347581799829, 'n_params': 2},
    'saddle_point-trivial': {'condition_number': 91888.68086097848, 'condition_std_err': 53567.78989839244, 'n_params': 2},
    'beale-trivial': {'condition_number': 210732.5860281166, 'condition_std_err': 95209.28364513395, 'n_params': 2},
    'constrained_optimization-trivial': {'condition_number': 0.9999999200000074, 'condition_std_err': 6.2803698347351e-17, 'n_params': 16},
    'quadratic_varying_scale-trivial': {'condition_number': 3.9702629456170464, 'condition_std_err': 0.01086372361603261, 'n_params': 4},
    'plateau_navigation-trivial': {'condition_number': 12681.722238982658, 'condition_std_err': 9155.354527262856, 'n_params': 2},
    'dynamic_landscape-trivial': {'condition_number': 0.9999187065226719, 'condition_std_err': 3.198402122563329e-08, 'n_params': 16384},
    'quadratic_varying_target-trivial': {'condition_number': 1.0000004806791334, 'condition_std_err': 1.3058732973790058e-08, 'n_params': 4},
    'noisy_matmul-trivial': {'condition_number': 1.1722476228840366, 'condition_std_err': 0.002458656341032482, 'n_params': 64},
    'discontinuous_gradient-trivial': {'condition_number': 1.4145736016953239, 'condition_std_err': 0.010273191112878279, 'n_params': 1024},
    'adversarial_gradient-trivial': {'condition_number': 0.9999948800262152, 'condition_std_err': 0.0, 'n_params': 1024},
    'sparse_gradient-trivial': {'condition_number': 1.1331852724683795, 'condition_std_err': 0.09125763424492048, 'n_params': 65536},
    'wide_linear-trivial': {'condition_number': 8.456737694574993, 'condition_std_err': 0.4959897201919112, 'n_params': 16},
    'scale_invariant-trivial': {'condition_number': 1.6681626636766373, 'condition_std_err': 0.023063879228518092, 'n_params': 512},
    'parameter_scale-trivial': {'condition_number': 0.9999846402359267, 'condition_std_err': 5.4389598220420725e-17, 'n_params': 3072},
    'batch_size_scaling-trivial': {'condition_number': 0.9999948800262152, 'condition_std_err': 0.0, 'n_params': 1024},
    'momentum_utilization-trivial': {'condition_number': 1.0437173596740965, 'condition_std_err': 0.0012081070607304091, 'n_params': 1024},
    'layer_wise_scale-trivial': {'condition_number': 1.1111882264867392, 'condition_std_err': 0.0031824561313248644, 'n_params': 3072},
    'gradient_noise_scale-trivial': {'condition_number': 0.9999795204194228, 'condition_std_err': 7.691850745534256e-17, 'n_params': 4096},
    'xor_digit_rnn-trivial': {'condition_number': 417934.51732124516, 'condition_std_err': 148954.88131527413, 'n_params': 8641},
    'xor_spot_rnn-trivial': {'condition_number': 135088.171192115, 'condition_std_err': 53305.38363290111, 'n_params': 8769},
    'xor_sequence_rnn-trivial': {'condition_number': 171255.15299611475, 'condition_std_err': 74936.96709224746, 'n_params': 1201},
    'xor_digit-trivial': {'condition_number': 261057.59900440424, 'condition_std_err': 129801.91259445195, 'n_params': 8641},
    'xor_sequence-trivial': {'condition_number': 886809.5831448702, 'condition_std_err': 484391.34644175763, 'n_params': 1201},
    'xor_spot-trivial': {'condition_number': 294032.2174859221, 'condition_std_err': 164934.5337048027, 'n_params': 8769},
}

nightmare_benchmark_data = {
    'saddle_point-nightmare': {'condition_number': 91888.68086097848, 'condition_std_err': 53567.78989839244, 'n_params': 2},
    'constrained_optimization-nightmare': {'condition_number': 0.9999999200000074, 'condition_std_err': 6.2803698347351e-17, 'n_params': 16},
    'quadratic_varying_target-nightmare': {'condition_number': 0.9993455455843698, 'condition_std_err': 0.0, 'n_params': 131072},
    'plateau_navigation-nightmare': {'condition_number': 1.5901764117372556e-21, 'condition_std_err': 1.8665673278228616e-27, 'n_params': 2},
    'dynamic_landscape-nightmare': {'condition_number': 0.9999187065226719, 'condition_std_err': 3.198402122563329e-08, 'n_params': 16384},
    'scale_invariant-nightmare': {'condition_number': 1.6371300015299801, 'condition_std_err': 0.009076823399866653, 'n_params': 512},
    'momentum_utilization-nightmare': {'condition_number': 277591.31185500085, 'condition_std_err': 113845.87252228722, 'n_params': 1024},
    'batch_size_scaling-nightmare': {'condition_number': 0.9999948800262152, 'condition_std_err': 0.0, 'n_params': 1024},
    'gradient_noise_scale-nightmare': {'condition_number': 0.9999795204194228, 'condition_std_err': 7.691850745534256e-17, 'n_params': 4096},
    'sparse_gradient-nightmare': {'condition_number': 1.7330184595058562, 'condition_std_err': 0.04429894454408897, 'n_params': 65536},
    'adversarial_gradient-nightmare': {'condition_number': 0.9999948800262152, 'condition_std_err': 0.0, 'n_params': 1024},
    'parameter_scale-nightmare': {'condition_number': 0.9999846402359267, 'condition_std_err': 5.4389598220420725e-17, 'n_params': 3072},
    'layer_wise_scale-nightmare': {'condition_number': 1.3179572626139584, 'condition_std_err': 0.010496221065780716, 'n_params': 3072},
    'noisy_matmul-nightmare': {'condition_number': 1.4463247067635787, 'condition_std_err': 0.12077412891157314, 'n_params': 64},
    'wide_linear-nightmare': {'condition_number': 1.1364118082990058, 'condition_std_err': 0.0019245728367553523, 'n_params': 268435456},
    'xor_sequence_rnn-nightmare': {'condition_number': 207880.62798076178, 'condition_std_err': 26755.248237981687, 'n_params': 1201},
    'xor_sequence-nightmare': {'condition_number': 423664.97668437194, 'condition_std_err': 123294.03901190669, 'n_params': 1201},
    'xor_digit_rnn-nightmare': {'condition_number': 166977.7050313757, 'condition_std_err': 55281.997201904036, 'n_params': 8641},
    'xor_spot_rnn-nightmare': {'condition_number': 81838.02576099725, 'condition_std_err': 20289.38285308829, 'n_params': 8769},
    'xor_digit-nightmare': {'condition_number': 291830.48785368446, 'condition_std_err': 187499.23098520897, 'n_params': 8641},
    'xor_spot-nightmare': {'condition_number': 144902.0033495639, 'condition_std_err': 48892.63173132324, 'n_params': 8769},
    # Missing benchmarks from original nightmare data (no error data provided)
    'quadratic_varying_scale-nightmare': {'condition_number': 1.0159356926438396, 'condition_std_err': None, 'n_params': 131072},
    'minimax-nightmare': {'condition_number': None, 'condition_std_err': None, 'n_params': 262144},
    'gradient_delay-nightmare': {'condition_number': None, 'condition_std_err': None, 'n_params': 256},
}

def create_combined_dataframe(trivial_data, nightmare_data):
    """Convert both benchmark datasets to combined DataFrame with error bars"""
    df_data = []
    
    # Process trivial data
    for name, values in trivial_data.items():
        if values['condition_number'] is not None and values['condition_number'] > 1e-10:
            df_data.append({
                'benchmark': name.replace('-trivial', ''),
                'condition_number': values['condition_number'],
                'condition_std_err': values['condition_std_err'] if values['condition_std_err'] is not None else 0.0,
                'n_params': values['n_params'],
                'log_condition': np.log10(values['condition_number']),
                'log_params': np.log10(values['n_params']),
                'difficulty': 'Trivial'
            })
    
    # Process nightmare data
    for name, values in nightmare_data.items():
        if values['condition_number'] is not None and values['condition_number'] > 1e-10:
            df_data.append({
                'benchmark': name.replace('-nightmare', ''),
                'condition_number': values['condition_number'],
                'condition_std_err': values['condition_std_err'] if values['condition_std_err'] is not None else 0.0,
                'n_params': values['n_params'],
                'log_condition': np.log10(values['condition_number']),
                'log_params': np.log10(values['n_params']),
                'difficulty': 'Nightmare'
            })
    
    return pd.DataFrame(df_data)


def categorize_benchmarks(df):
    """Add categories based on benchmark characteristics"""
    def get_category(name):
        if any(keyword in name for keyword in ['xor_digit_rnn', 'xor_sequence_rnn', 'xor_spot_rnn']):
            return 'RNN'
        elif any(keyword in name for keyword in ['xor_digit', 'xor_sequence', 'xor_spot']):
            return 'LSTM'
        elif any(keyword in name for keyword in ['saddle_point', 'beale', 'rosenbrock', 'rastrigin', 'plateau_navigation']):
            return 'Classic Optimization'
        elif any(keyword in name for keyword in ['quadratic', 'dynamic_landscape']):
            return 'Quadratic/Convex'
        elif any(keyword in name for keyword in ['gradient', 'noise', 'adversarial', 'sparse', 'minimax', 'wide_linear']):
            return 'Gradient Challenges'
        else:
            return 'Other'
    
    df['category'] = df['benchmark'].apply(get_category)
    return df

def plot_condition_vs_params(df, save_path=None):
    """Create log-log scatter plot of condition number vs number of parameters"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Color map for categories
    categories = df['category'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, colors))
    
    # Marker styles for difficulty
    marker_map = {'Trivial': 'o', 'Nightmare': 's'}  # circle vs square
    
    # Plot log-log scale
    for category in categories:
        for difficulty in ['Trivial', 'Nightmare']:
            cat_diff_data = df[(df['category'] == category) & (df['difficulty'] == difficulty)]
            if len(cat_diff_data) > 0:
                # Calculate error bars in log space
                log_errors = []
                for _, row in cat_diff_data.iterrows():
                    if row['condition_std_err'] > 0:
                        # Convert standard error to log space
                        # For log(y), error = std_err / (y * ln(10))
                        log_error = row['condition_std_err'] / (row['condition_number'] * np.log(10))
                    else:
                        log_error = 0
                    log_errors.append(log_error)
                
                # Plot with error bars
                ax.errorbar(cat_diff_data['log_params'], cat_diff_data['log_condition'],
                           yerr=log_errors,
                           fmt=marker_map[difficulty], 
                           color=color_map[category],
                           label=f'{category}' if difficulty == 'Trivial' else '',
                           alpha=0.7, markersize=8, 
                           markeredgecolor='black', markeredgewidth=0.5,
                           capsize=3, capthick=1, elinewidth=1)
    
    ax.set_xlabel('Log₁₀(Number of Parameters)', fontsize=12)
    ax.set_ylabel('Log₁₀(Condition Number)', fontsize=12)
    ax.set_title('Condition Number vs Number of Parameters (Log-Log Scale)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add benchmark names as annotations for classic optimization problems - styled like "beale.py"
    classic_problems = ['saddle_point', 'beale', 'rosenbrock', 'rastrigin']
    
    for _, row in df.iterrows():
        if any(classic in row['benchmark'] for classic in classic_problems):
            # Format name like "beale.py" - lowercase with .py extension
            formatted_name = f"{row['benchmark'].replace('_', '')}.py"
            ax.annotate(formatted_name, 
                       (row['log_params'], row['log_condition']),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, alpha=0.8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        # Also annotate extreme cases - styled like "beale.py"
        elif row['condition_number'] > 1000000 or row['n_params'] > 1000000:
            formatted_name = f"{row['benchmark'].replace('_', '')}.py"
            ax.annotate(formatted_name, 
                       (row['log_params'], row['log_condition']),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, alpha=0.8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add custom legend for marker shapes
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                             markersize=8, label='Trivial', markeredgecolor='black'),
                      Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                             markersize=8, label='Nightmare', markeredgecolor='black')]
    
    # Add second legend for shapes
    shape_legend = ax.legend(handles=legend_elements, title="Difficulty", 
                            loc='upper left', bbox_to_anchor=(1.05, 0.3), fontsize=11)
    ax.add_artist(shape_legend)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.savefig("analyse_benchmarks_combined.png", dpi=500, bbox_inches='tight')
    plt.show()

def analyze_patterns(df):
    """Print analysis of patterns in the data"""
    print("=== Benchmark Analysis ===\n")
    
    # Summary statistics
    print("Summary Statistics:")
    print(f"Total benchmarks: {len(df)}")
    print(f"Condition number range: {df['condition_number'].min():.2e} to {df['condition_number'].max():.2e}")
    print(f"Parameter count range: {df['n_params'].min()} to {df['n_params'].max():,}")
    print()
    
    # Category analysis
    print("By Category:")
    category_stats = df.groupby('category').agg({
        'condition_number': ['count', 'mean', 'median', 'max'],
        'n_params': ['mean', 'median']
    }).round(2)
    print(category_stats)
    print()
    
    # Classic optimization problems detail
    print("Classic Optimization Problems:")
    classic_data = df[df['category'] == 'Classic Optimization'].sort_values('condition_number', ascending=False)
    for _, row in classic_data.iterrows():
        print(f"  {row['benchmark'].replace('_', ' ').title()}: {row['condition_number']:.2e} (params: {row['n_params']:,}) [{row['difficulty']}]")
    print()
    
    # Correlation analysis
    correlation = df['log_condition'].corr(df['log_params'])
    print(f"Correlation between log(condition number) and log(parameters): {correlation:.3f}")

# Main execution
if __name__ == "__main__":
    # Create combined DataFrame
    df_combined = create_combined_dataframe(benchmark_data, nightmare_benchmark_data)
    df_combined = categorize_benchmarks(df_combined)
    
    # Create plot
    plot_condition_vs_params(df_combined, save_path='condition_vs_params_combined.png')
    
    # Print analysis
    analyze_patterns(df_combined)
    
    # Save data to CSV for further analysis
    df_combined.to_csv('benchmark_conditioning_analysis_combined.csv', index=False)
    print(f"\nData saved to 'benchmark_conditioning_analysis_combined.csv'")