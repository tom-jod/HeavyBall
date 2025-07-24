import matplotlib.pyplot as plt
import numpy as np

def plot_optimizer_comparison():
     # HeavyBall SGD data (new results)
    # These are sampled every 100 steps based on the loss_trajectory length
    hb_steps = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    hb_losses = np.array([3.0330703258514404, 2.7274816036224365, 2.4513914585113525, 
                         2.002882242202759, 2.0375466346740723, 2.0237557888031006, 
                         1.7503620386123657, 1.9077682495117188, 1.7975537776947021, 
                         1.8356080055236816])
    hb_test_acc = np.array([7.93, 14.78, 28.33, 43.51, 52.63, 57.81, 60.53, 61.98, 62.85, 63.04])
    
    # Original SGD data (from your provided output)
    orig_steps = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999])
    orig_train_losses = np.array([3.0358, 2.3943, 2.2489, 2.0426, 1.9285, 1.9409, 
                                 1.7334, 1.8928, 1.7464, 1.7829, 1.8967])
    orig_test_losses = np.array([2.7199, 2.3049, 2.0536, 1.8949, 1.7781, 1.6966, 
                                1.6381, 1.6016, 1.5820, 1.5745, 1.5733])
    orig_test_acc = np.array([7.94, 14.68, 28.47, 43.94, 52.79, 57.58, 60.25, 62.09, 62.83, 63.11, 63.14])
    orig_lr = np.array([2.00e-05, 9.93e-04, 9.39e-04, 8.37e-04, 6.99e-04, 5.40e-04, 
                       3.76e-04, 2.25e-04, 1.04e-04, 2.66e-05, 0.00e+00])
    """
    # HeavyBall AdamW data (new results)
    # These are sampled every 100 steps based on the loss_trajectory length
    hb_steps = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    hb_losses = np.array([3.0330703258514404, 0.6630929112434387, 0.5271769165992737, 
                         0.4640238881111145, 0.27277663350105286, 0.27711135149002075, 
                         0.14471471309661865, 0.4565425515174866, 0.22511038184165955, 
                         0.44733762741088867])
    hb_test_acc = np.array([7.93, 86.96, 89.1, 92.21, 93.02, 93.78, 94.14, 94.45, 94.69, 94.67])
    
    # Original AdamW data (from your provided output)
    orig_steps = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999])
    orig_train_losses = np.array([3.0358, 0.8401, 0.5289, 0.2862, 0.3456, 0.3708, 
                                 0.4061, 0.4164, 0.3848, 0.4692, 0.3586])
    orig_test_losses = np.array([2.7142, 0.4605, 0.3754, 0.2501, 0.2353, 0.2108, 
                                0.1949, 0.1889, 0.1827, 0.1792, 0.1788])
    orig_test_acc = np.array([7.96, 86.34, 89.02, 92.49, 93.10, 93.74, 94.17, 94.27, 94.50, 94.66, 94.60])
    orig_lr = np.array([2.00e-05, 9.93e-04, 9.39e-04, 8.37e-04, 6.99e-04, 5.40e-04, 
                       3.76e-04, 2.25e-04, 1.04e-04, 2.66e-05, 0.00e+00])
    """
    # Create subplots
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot 1: Training Loss Comparison
    ax1.plot(hb_steps, hb_losses, 'b-o', label='HeavyBall', linewidth=2, markersize=6)
    ax1.plot(orig_steps, orig_train_losses, 'r-s', label='Original (NAdamW)', linewidth=2, markersize=6)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Test Accuracy Comparison
    ax2.plot(hb_steps, hb_test_acc, 'b-o', label='HeavyBall', linewidth=2, markersize=6)
    ax2.plot(orig_steps, orig_test_acc, 'r-s', label='Original (NAdamW)', linewidth=2, markersize=6)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=500, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("=== COMPARISON SUMMARY ===")
    print(f"Final Training Loss:")
    print(f"  HeavyBall: {hb_losses[-1]:.4f}")
    print(f"  Original:  {orig_train_losses[-1]:.4f}")
    print(f"  Winner: {'HeavyBall' if hb_losses[-1] < orig_train_losses[-1] else 'Original'}")
    
    print(f"\nFinal Test Accuracy:")
    print(f"  HeavyBall: {hb_test_acc[-1]:.2f}%")
    print(f"  Original:  {orig_test_acc[-1]:.2f}%")
    print(f"  Winner: {'HeavyBall' if hb_test_acc[-1] > orig_test_acc[-1] else 'Original'}")
    
    print(f"\nConvergence Speed (steps to reach 90% accuracy):")
    hb_90_step = hb_steps[np.where(hb_test_acc >= 90)[0][0]] if np.any(hb_test_acc >= 90) else "Not reached"
    orig_90_step = orig_steps[np.where(orig_test_acc >= 90)[0][0]] if np.any(orig_test_acc >= 90) else "Not reached"
    print(f"  HeavyBall: {hb_90_step}")
    print(f"  Original:  {orig_90_step}")
    
    print(f"\nStability (std deviation of last 5 measurements):")
    print(f"  HeavyBall Loss Std: {np.std(hb_losses[-5:]):.4f}")
    print(f"  Original Loss Std:  {np.std(orig_train_losses[-5:]):.4f}")

def plot_detailed_comparison_with_interpolation():
    """More detailed version with interpolation for smoother curves"""
    
    # Same data as above
    hb_steps = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    hb_losses = np.array([0.6816403865814209, 0.6770703196525574, 0.33210989832878113, 
                         0.3376680612564087, 0.31155869364738464, 0.5053976774215698, 
                         0.2650371193885803, 0.23870964348316193, 0.19927029311656952, 
                         0.215846985578537])
    hb_test_acc = np.array([9.05, 87.17, 90.76, 91.16, 92.58, 93.67, 93.82, 94.4, 94.58, 94.67])
    
    orig_steps = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999])
    orig_train_losses = np.array([2.3187, 0.5932, 0.2983, 0.4928, 0.3536, 0.2746, 
                                 0.2875, 0.3403, 0.2426, 0.1612, 0.1167])
    orig_test_acc = np.array([14.75, 87.23, 91.33, 92.40, 93.43, 94.21, 94.54, 94.51, 94.86, 95.01, 95.01])
    
    # Create interpolated curves for smoother visualization
    from scipy.interpolate import interp1d
    
    # Common step range
    common_steps = np.linspace(0, 900, 100)
    
    # Interpolate HeavyBall data
    hb_loss_interp = interp1d(hb_steps, hb_losses, kind='cubic')(common_steps)
    hb_acc_interp = interp1d(hb_steps, hb_test_acc, kind='cubic')(common_steps)
    
    # Interpolate Original data (up to step 900)
    orig_steps_900 = orig_steps[orig_steps <= 900]
    orig_loss_900 = orig_train_losses[orig_steps <= 900]
    orig_acc_900 = orig_test_acc[orig_steps <= 900]
    
    orig_loss_interp = interp1d(orig_steps_900, orig_loss_900, kind='cubic')(common_steps)
    orig_acc_interp = interp1d(orig_steps_900, orig_acc_900, kind='cubic')(common_steps)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Smooth curves
    ax1.plot(common_steps, hb_loss_interp, 'b-', label='HeavyBall', linewidth=3, alpha=0.7)
    ax1.plot(common_steps, orig_loss_interp, 'r-', label='Original', linewidth=3, alpha=0.7)
    # Original data points
    ax1.scatter(hb_steps, hb_losses, color='blue', s=50, zorder=5)
    ax1.scatter(orig_steps_900, orig_loss_900, color='red', s=50, zorder=5)
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Test accuracy
    ax2.plot(common_steps, hb_acc_interp, 'b-', label='HeavyBall', linewidth=3, alpha=0.7)
    ax2.plot(common_steps, orig_acc_interp, 'r-', label='Original', linewidth=3, alpha=0.7)
    ax2.scatter(hb_steps, hb_test_acc, color='blue', s=50, zorder=5)
    ax2.scatter(orig_steps_900, orig_acc_900, color='red', s=50, zorder=5)
    
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    print("here")
    plt.tight_layout()
    plt.savefig('detailed_optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run basic comparison
    plot_optimizer_comparison()
    
    # Run detailed comparison (requires scipy)
    try:
        plot_detailed_comparison_with_interpolation()
    except ImportError:
        print("Install scipy for detailed interpolated plots: pip install scipy")