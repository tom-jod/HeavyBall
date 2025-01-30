#!/usr/bin/env python3
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from datetime import datetime
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import colorsys

def parse_loss(loss_str):
    if loss_str.strip() == 'inf':
        return float('inf')
    try:
        if 'e' in loss_str:
            base, exp = loss_str.split('e')
            return float(base) * 10**float(exp)
        return float(loss_str)
    except (ValueError, IndexError):
        return float('nan')

def read_benchmark_results(file_path):
    # Previous implementation remains the same
    with open(file_path, 'r') as f:
        content = f.read()
    
    details_section = re.search(r'## Details\n\n(.*?)(?=\n\n|$)', content, re.DOTALL)
    if not details_section:
        raise ValueError("Could not find Details section")
    
    lines = details_section.group(1).strip().split('\n')[2:]
    data = []
    for line in lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split('|')[1:-1]]
        if len(parts) < 8:
            continue
        data.append({
            'benchmark': parts[0],
            'optimizer': parts[1],
            'cautious': parts[2] == 'Yes',
            'mars': parts[3] == 'Yes',
            'success': parts[4] == '✓',
            'runtime': float(parts[5].replace('s', '')),
            'loss': parse_loss(parts[6]),
            'attempts': int(parts[7])
        })
    
    return pd.DataFrame(data)

def create_result_matrix(df):
    # Previous implementation remains the same
    benchmarks = sorted(df['benchmark'].unique())
    optimizers = sorted(df['optimizer'].unique())
    
    success_matrix = pd.DataFrame(index=benchmarks, columns=optimizers)
    attempts_matrix = pd.DataFrame(index=benchmarks, columns=optimizers)
    runtime_matrix = pd.DataFrame(index=benchmarks, columns=optimizers)
    
    for _, row in df.iterrows():
        success_matrix.loc[row['benchmark'], row['optimizer']] = row['success']
        attempts_matrix.loc[row['benchmark'], row['optimizer']] = row['attempts']
        runtime_matrix.loc[row['benchmark'], row['optimizer']] = row['runtime']
    
    return success_matrix, attempts_matrix, runtime_matrix

def create_colorblind_friendly_palette():
    # Create a colorblind-friendly diverging palette
    success_color = '#2ECC71'  # Bright green
    failure_color = '#E74C3C'  # Soft red
    neutral_color = '#F7DC6F'  # Soft yellow
    
    return success_color, neutral_color, failure_color

def create_attempt_colormap(attempts_matrix):
    # Create a custom colormap based on attempt counts
    min_attempts = attempts_matrix.min().min()
    max_attempts = attempts_matrix.max().max()
    
    # Create more sophisticated color gradients
    colors = ['#D5F5E3', '#82E0AA', '#2ECC71', '#229954', '#145A32']
    positions = [0, 0.25, 0.5, 0.75, 1]
    
    return LinearSegmentedColormap.from_list('custom_attempts', 
                                           list(zip(positions, colors)))

def add_fancy_background(ax, color='#F8F9F9', border_color='#BDC3C7'):
    # Fixed version - use the figure's add_artist method instead of add_patch
    bbox = ax.get_position()
    fancy_box = FancyBboxPatch((bbox.x0, bbox.y0),
                              bbox.width, bbox.height,
                              boxstyle="round,pad=0.02",
                              fc=color,
                              ec=border_color,
                              transform=ax.figure.transFigure,
                              zorder=-1)
    ax.figure.add_artist(fancy_box)

def create_visual_matrix(success_matrix, attempts_matrix, runtime_matrix):
    # Set style
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(20, 14), facecolor='white')
    
    # Create main axis for heatmap with more space for labels
    main_ax = plt.subplot2grid((1, 5), (0, 0), colspan=4)
    
    # Define softer colors
    success_color = '#7CB9E8'  # Soft blue
    failure_color = '#F08080'  # Soft red
    
    # Create success/failure mask
    success_mask = success_matrix.astype(bool)
    
    # Create base color matrix
    colors = np.where(success_mask, success_color, failure_color)
    
    # Plot main heatmap
    #im = main_ax.imshow(np.zeros_like(success_matrix.values), 
    #                   aspect='auto', 
    #                   cmap='RdYlBu')
    
    # Add cell content with enhanced clarity
    for i in range(success_matrix.shape[0]):
        for j in range(success_matrix.shape[1]):
            if not pd.isna(attempts_matrix.iloc[i, j]):
                success = success_mask.iloc[i, j]
                attempts = int(attempts_matrix.iloc[i, j])
                runtime = runtime_matrix.iloc[i, j]
                
                # Set cell color
                color = success_color if success else failure_color
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                               facecolor=color, 
                               alpha=0.7)
                main_ax.add_patch(rect)
                
                # Format text
                attempts_text = f'{attempts}'
                runtime_text = f'{runtime:.1f}s'
                
                # Add text with better spacing
                main_ax.text(j, i - 0.15, attempts_text,
                           ha='center', va='center',
                           color='black',
                           fontsize=8,
                           fontweight='bold')
                main_ax.text(j, i + 0.15, runtime_text,
                           ha='center', va='center',
                           color='black',
                           fontsize=7)
    
    # Enhance grid
    for i in range(success_matrix.shape[0] + 1):
        main_ax.axhline(y=i-0.5, color='white', linewidth=1)
    for j in range(success_matrix.shape[1] + 1):
        main_ax.axvline(x=j-0.5, color='white', linewidth=1)
    
    # Improve axis labels
    main_ax.set_xticks(range(len(success_matrix.columns)))
    main_ax.set_yticks(range(len(success_matrix.index)))
    
    # Rotate and align the tick labels so they look better
    main_ax.set_xticklabels(success_matrix.columns,
                           rotation=45,
                           ha='right',
                           fontsize=10)
    main_ax.set_yticklabels(success_matrix.index,
                           fontsize=10)
    
    # Add success rate bars on the right
    success_ax = plt.subplot2grid((1, 5), (0, 4))
    success_rates = success_matrix.mean(axis=0) * 100
    
    # Create horizontal bars
    bars = success_ax.barh(range(len(success_rates)),
                          success_rates,
                          color=success_color,
                          alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(success_rates):
        success_ax.text(v + 1, i,
                       f'{v:.1f}%',
                       va='center',
                       fontsize=8)
    
    # Configure success rate axis
    success_ax.set_yticks([])
    success_ax.set_xlabel('Success Rate (%)')
    success_ax.grid(True, axis='x', alpha=0.3)
    
    # Add title and subtitle
    plt.suptitle('Optimizer Benchmark Performance Matrix',
                fontsize=16,
                y=0.95)
    
    # Add legend with clearer labels
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=success_color, alpha=0.7, label='Success'),
        Rectangle((0, 0), 1, 1, fc=failure_color, alpha=0.7, label='Failure')
    ]
    main_ax.legend(handles=legend_elements,
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.15),
                  ncol=2,
                  frameon=True,
                  fontsize=10)
    
    # Add explanatory text
    fig.text(0.02, 0.02,
             'Cell Format:\nAttempts (top)\nRuntime in seconds (bottom)',
             fontsize=8,
             ha='left',
             va='bottom')
    
	    # Add summary statistics
    best_optimizer = success_rates.idxmax()
    best_rate = success_rates.max()
    avg_attempts = attempts_matrix.mean().mean()
    median_runtime = runtime_matrix.median().median()
    
    stats_text = (
        f'Best Performer: {best_optimizer} ({best_rate:.1f}% success)\n'
        f'Average Attempts: {avg_attempts:.1f}\n'
        f'Median Runtime: {median_runtime:.2f}s'
    )
    
    fig.text(0.98, 0.02,
             stats_text,
             fontsize=8,
             ha='right',
             va='bottom',
             bbox=dict(facecolor='white', 
                      edgecolor='gray', 
                      alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.15)
    
    # Save with high quality
    plt.savefig('benchmark_matrix.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                pad_inches=0.5)
    plt.close()

def main():
    # Read and process results
    df = read_benchmark_results('benchmark_results.md')
    success_matrix, attempts_matrix, runtime_matrix = create_result_matrix(df)
    
    # Create the enhanced visual matrix
    create_visual_matrix(success_matrix, attempts_matrix, runtime_matrix)
    
    # Print text summary
    print("\nSummary Statistics:")
    success_rates = success_matrix.mean().sort_values(ascending=False)
    avg_attempts = attempts_matrix.mean()
    
    print("\nSuccess Rates by Optimizer:")
    for optimizer in success_rates.index:
        rate = success_rates[optimizer] * 100
        attempts = avg_attempts[optimizer]
        print(f"{optimizer}: {rate:.1f}% (avg {attempts:.1f} attempts)")

if __name__ == "__main__":
    main()
