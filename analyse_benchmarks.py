import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from your benchmarks
benchmark_data = {
    'rosenbrock-trivial': {'condition_number': 861.70, 'n_params': 2},
    'rastrigin-trivial': {'condition_number': 1.00, 'n_params': 2},
    'gradient_delay-trivial': {'condition_number': None, 'n_params': 256},
    'saddle_point-trivial': {'condition_number': 322832.83, 'n_params': 2},
    'beale-trivial': {'condition_number': 571809.33, 'n_params': 2},
    'constrained_optimization-trivial': {'condition_number': 1.00, 'n_params': 16},
    'plateau_navigation-trivial': {'condition_number': 53579.12, 'n_params': 2},
    'quadratic_varying_scale-trivial': {'condition_number': 3.996, 'n_params': 4},
    'dynamic_landscape-trivial': {'condition_number': 1.00, 'n_params': 16384},
    'quadratic_varying_target-trivial': {'condition_number': 1.00, 'n_params': 4},
    'noisy_matmul-trivial': {'condition_number': 1.18, 'n_params': 64},
    'discontinuous_gradient-trivial': {'condition_number': 1.45, 'n_params': 1024},
    'gradient_noise_scale-trivial': {'condition_number': 1.00, 'n_params': 4096},
    'adversarial_gradient-trivial': {'condition_number': 1.00, 'n_params': 1024},
    'layer_wise_scale-trivial': {'condition_number': 1.13, 'n_params': 3072},
    'scale_invariant-trivial': {'condition_number': 1.76, 'n_params': 512},
    'batch_size_scaling-trivial': {'condition_number': 1.00, 'n_params': 1024},
    'momentum_utilization-trivial': {'condition_number': 1.05, 'n_params': 1024},
    'sparse_gradient-trivial': {'condition_number': 1.55, 'n_params': 65536},
    'parameter_scale-trivial': {'condition_number': 1.00, 'n_params': 3072},
    'xor_digit_rnn-trivial': {'condition_number': 783206.26, 'n_params': 8641},
    'xor_sequence_rnn-trivial': {'condition_number': 561801.19, 'n_params': 1201},
    'xor_spot_rnn-trivial': {'condition_number': 225475.07, 'n_params': 8769},
    'xor_sequence-trivial': {'condition_number': 3410584.65, 'n_params': 1201},
    'xor_digit-trivial': {'condition_number': 292965.66, 'n_params': 8641},
    'xor_spot-trivial': {'condition_number': 1063636.66, 'n_params': 8769},
}

def create_benchmark_dataframe(data):
    """Convert benchmark data to pandas DataFrame"""
    df_data = []
    
    for name, values in data.items():
        if values['condition_number'] is not None:  # Skip None values
            df_data.append({
                'benchmark': name.replace('-trivial', ''),
                'condition_number': values['condition_number'],
                'n_params': values['n_params'],
                'log_condition': np.log10(values['condition_number']),
                'log_params': np.log10(values['n_params'])
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
        elif any(keyword in name for keyword in ['gradient', 'noise', 'adversarial', 'sparse']):
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
    
    # Plot log-log scale
    for category in categories:
        cat_data = df[df['category'] == category]
        ax.scatter(cat_data['log_params'], cat_data['log_condition'], 
                  c=[color_map[category]], label=category, alpha=0.7, s=80)
    
    ax.set_xlabel('Log₁₀(Number of Parameters)', fontsize=12)
    ax.set_ylabel('Log₁₀(Condition Number)', fontsize=12)
    ax.set_title('Condition Number vs Number of Parameters (Log-Log Scale)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add benchmark names as annotations for all classic optimization problems
    classic_problems = ['saddle_point', 'beale', 'rosenbrock', 'rastrigin', 'plateau_navigation']
    
    for _, row in df.iterrows():
        if any(classic in row['benchmark'] for classic in classic_problems):
            ax.annotate(row['benchmark'].replace('_', ' ').title(), 
                       (row['log_params'], row['log_condition']),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, alpha=0.8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        # Also annotate extreme cases
        elif row['condition_number'] > 1000000 or row['n_params'] > 50000:
            ax.annotate(row['benchmark'].replace('_', ' '), 
                       (row['log_params'], row['log_condition']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.savefig("analyse_benchmarks.png", dpi=500, bbox_inches='tight')

def analyze_patterns(df):
    """Print analysis of patterns in the data"""
    
    print("=== Benchmark Analysis ===\n")
    
    # Summary statistics
    print("Summary Statistics:")
    print(f"Total benchmarks: {len(df)}")
    print(f"Condition number range: {df['condition_number'].min():.2f} to {df['condition_number'].max():.2e}")
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
        print(f"  {row['benchmark'].replace('_', ' ').title()}: {row['condition_number']:.2e} (params: {row['n_params']:,})")
    print()
    
    # Correlation analysis
    correlation = df['log_condition'].corr(df['log_params'])
    print(f"Correlation between log(condition number) and log(parameters): {correlation:.3f}")

# Main execution
if __name__ == "__main__":
    # Create DataFrame
    df = create_benchmark_dataframe(benchmark_data)
    df = categorize_benchmarks(df)
    
    # Create plot
    plot_condition_vs_params(df, save_path='condition_vs_params_log.png')
    
    # Print analysis
    analyze_patterns(df)
    
    # Save data to CSV for further analysis
    df.to_csv('benchmark_conditioning_analysis.csv', index=False)
    print(f"\nData saved to 'benchmark_conditioning_analysis.csv'")