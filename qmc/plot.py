import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec
from scipy import stats

def analyze_vmc_results(results, save_plots=True, plot_dir="./plots", dpi=300, kmax=None):
    """
    Focused analysis of VMC simulation results with only essential plots.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing VMC simulation results
    save_plots : bool, optional
        Whether to save plots to disk (default: True)
    plot_dir : str, optional
        Directory to save plots (default: './plots')
    dpi : int, optional
        Resolution of saved plots (default: 300)
    kmax : float, optional
        Theoretical kmax value for comparison with electron ages
    
    Returns:
    --------
    None (plots are displayed and optionally saved)
    """
    if save_plots and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    sns.set_style("whitegrid")
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['figure.titlesize'] = 16
    
    # Use consistent colors across plots
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    electron_colors = plt.cm.plasma(np.linspace(0, 0.8, 2))  # Just for electron #1 and #2
    
    plot_block_energy_with_errorbars(results, colors, save_plots, plot_dir, dpi)
    
    # 1. Cumulative Monte Carlo Energy, Variance, Error
    plot_cumulative_energy(results, colors, save_plots, plot_dir, dpi)
    
    # 2. Cumulative Acceptance Ratio
    plot_acceptance_ratio(results, colors, save_plots, plot_dir, dpi)
    
    # 3. MSD of Electrons #1 and #2
    plot_electron_msd(results, electron_colors, save_plots, plot_dir, dpi)
    
    # 3b. MSD with threshold analysis
    plot_msd_threshold_analysis(results, electron_colors, threshold=0.7, save_plots=save_plots, plot_dir=plot_dir, dpi=dpi)
    
    # 4. Max Age of Each Electron (with kmax if provided)
    plot_max_and_mean_ages(results, colors, save_plots, plot_dir, dpi, kmax=kmax)

    # 5. Autocorrelation
    plot_autocorrelation(results, electron_colors, save_plots, plot_dir, dpi)
    
    # Print summary including kmax comparison if provided
    print_vmc_summary(results, kmax=kmax)
    
    print("Analysis complete!")
    
def plot_block_energy_with_errorbars(results, colors, save_plots=True, plot_dir="./plots", dpi=300, block_offset=3):
    """
    Plot block energies with error bars showing the standard error within each block.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing VMC simulation results
    colors : array-like
        Colors for plots
    save_plots : bool
        Whether to save plots to disk
    plot_dir : str
        Directory to save plots
    dpi : int
        Resolution of saved plots
    block_offset : int
        Starting block number (default: 3, means starting from block 4)
    """
    plt.figure(figsize=(12, 8))
    
    # Extract data
    block_energies = np.array(results["total"])
    blocks = np.arange(block_offset + 1, block_offset + 1 + len(block_energies))
    
    # Extract or calculate within-block errors
    if "block_errors" in results:
        # If block errors are directly available in results
        block_errors = np.array(results["block_errors"])
    else:
        # If not directly available, we'll use block variances
        # In a real VMC code, this would normally be calculated from the individual samples
        # Here we'll approximate it from the cumulative variance
        cumul_variance = np.array(results["cumulative_variance"])
        samples_per_block = len(results.get("msd", {}).get("0", [])) / len(blocks)
        block_errors = np.sqrt(cumul_variance / samples_per_block)
    
    # Plot block energies with error bars
    plt.errorbar(blocks, block_energies, yerr=np.array(results["cumulative_error"]), fmt='o-', 
                 color=colors[0], ecolor=colors[0], capsize=5, 
                 lw=2, markersize=8, label='Block Energy with Error')
    
    # Add mean energy line
    mean_energy = results["mean_energy"]
    std_error = results["std_error"]
    plt.axhline(mean_energy, color='red', linestyle='--', lw=2, 
                label=f'Mean Energy: {mean_energy:.6f}')
    
    # Add shaded region for mean energy error
    plt.fill_between(blocks, 
                    [mean_energy - std_error] * len(blocks), 
                    [mean_energy + std_error] * len(blocks), 
                    color='red', alpha=0.2, 
                    label=f'Mean Energy Error: ±{std_error:.6f}')
    
    # Cumulative energy for comparison
    cumul_energy = np.array(results["cumulative_energy"])
    plt.plot(blocks, cumul_energy, '-', color=colors[1], lw=2, label='Cumulative Energy')
    
    # Set axis labels and title
    plt.xlabel('Block', fontsize=14)
    plt.ylabel('Energy (a.u.)', fontsize=14)
    plt.title('Block Energies with Error Bars', fontsize=16)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)
    
    # Adjust y-axis limits to better show the error bars
    y_range = np.max(block_energies) - np.min(block_energies)
    plt.ylim(np.min(block_energies) - y_range*0.2, 
             np.max(block_energies) + y_range*0.2)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'block_energy_errorbars.png'), dpi=dpi, bbox_inches='tight')
    plt.show()
    
    # Additional plot: Standard errors vs. block number
    plt.figure(figsize=(10, 6))
    
    plt.semilogy(blocks, block_errors, 'o-', color=colors[0], lw=2, 
                 label='Within-Block Std Error')
    plt.semilogy(blocks, np.array(results["cumulative_error"]), 'o-', color=colors[1], lw=2, 
                 label='Cumulative Std Error')
    
    # Add horizontal line for final cumulative error
    plt.axhline(results["std_error"], color='red', linestyle='--', lw=2, 
                label=f'Final Std Error: {std_error:.6f}')
    
    # Set axis labels and title
    plt.xlabel('Block', fontsize=14)
    plt.ylabel('Standard Error (a.u.)', fontsize=14)
    plt.title('Standard Errors vs. Block Number', fontsize=16)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'standard_errors.png'), dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_cumulative_energy(results, colors, save_plots, plot_dir, dpi):
    """Plot cumulative energy, variance, and error."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Extract data
    block_energies = np.array(results["total"])
    blocks = np.arange(1, len(block_energies) + 1)
    cumul_energy = np.array(results["cumulative_energy"])
    cumul_error = np.array(results["cumulative_error"])
    mean_energy = results["mean_energy"]
    std_error = results["std_error"]
    
    # Block and cumulative energy
    ax1 = axes[0]
    ax1.plot(blocks, block_energies, 'o-', color=colors[0], lw=2, label='Block Energy')
    ax1.plot(blocks, cumul_energy, '-', color=colors[1], lw=2, label='Cumulative Energy')
    ax1.fill_between(blocks, cumul_energy - cumul_error, cumul_energy + cumul_error, 
                    color=colors[1], alpha=0.3)
    
    # Horizontal line for mean energy
    if mpl.rcParams['text.usetex']:
        label_text = f'Mean: {mean_energy:.6f} $\\pm$ {std_error:.6f}'
    else:
        label_text = f'Mean: {mean_energy:.6f} ± {std_error:.6f}'
    ax1.axhline(mean_energy, color='red', linestyle='--', lw=1.5, label=label_text)
    
    ax1.set_ylabel('Energy (a.u.)')
    ax1.set_title('Energy Convergence')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax2 = axes[1]
    cumul_variance = np.array(results["cumulative_variance"])
    ax2.plot(blocks, cumul_variance, 'o-', color=colors[2], lw=2, label='Cumulative Variance')
    ax2.plot(blocks, cumul_error, '-', color=colors[3], lw=2, label='Cumulative Error')
    ax2.set_xlabel('Block')
    ax2.set_ylabel('Value')
    ax2.set_title('Cumulative Variance and Error')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale to better visualize error decay
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'cumulative_energy.png'), dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_acceptance_ratio(results, colors, save_plots, plot_dir, dpi):
    """Plot cumulative acceptance ratio."""
    plt.figure(figsize=(10, 6))
    
    # Extract data
    blocks = np.arange(1, len(results["acceptance"]) + 1)
    acceptance = np.array(results["acceptance"])
    cumul_acceptance = np.array(results["cumulative_acceptance"])
    
    plt.plot(blocks, acceptance, 'o-', color=colors[0], lw=2, label='Block Acceptance Ratio')
    plt.plot(blocks, cumul_acceptance, '-', color=colors[1], lw=2, label='Cumulative Acceptance Ratio')
    plt.axhline(0.5, color='r', linestyle='--', lw=1.5, label='Ideal Ratio (50%)')
    plt.xlabel('Block')
    plt.ylabel('Acceptance Ratio')
    plt.title('Monte Carlo Acceptance Ratio')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'acceptance_ratio.png'), dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_electron_msd(results, electron_colors, save_plots, plot_dir, dpi):
    """Plot MSD for electrons #1 and #2."""
    plt.figure(figsize=(10, 6))
    
    # Extract MSD data for electrons #1 and #2 (index 0 and 1)
    steps = np.arange(len(results["msd"]["0"]))
    
    # Plot MSD for electrons #1 and #2
    msd_0 = np.array(results["msd"]["0"])
    msd_1 = np.array(results["msd"]["1"])
    
    plt.plot(steps, msd_0, '-', color=electron_colors[0], lw=2, label='Electron #1')
    plt.plot(steps, msd_1, '-', color=electron_colors[1], lw=2, label='Electron #2')
    
    # Add linear fit to estimate diffusion coefficient
    fit_start = len(steps) // 4  # Skip initial transient
    
    # Fit for electron #1
    coeffs_0 = np.polyfit(steps[fit_start:], msd_0[fit_start:], 1)
    plt.plot(steps[fit_start:], np.polyval(coeffs_0, steps[fit_start:]), 
             '--', color=electron_colors[0], lw=1.5, 
             label=f'Fit #1: {coeffs_0[0]:.4e} (D≈{coeffs_0[0]/6:.4e})')
    
    # Fit for electron #2
    coeffs_1 = np.polyfit(steps[fit_start:], msd_1[fit_start:], 1)
    plt.plot(steps[fit_start:], np.polyval(coeffs_1, steps[fit_start:]), 
             '--', color=electron_colors[1], lw=1.5, 
             label=f'Fit #2: {coeffs_1[0]:.4e} (D≈{coeffs_1[0]/6:.4e})')
    
    plt.xlabel('Monte Carlo Step')
    plt.ylabel('Mean Square Displacement')
    plt.title('Electron Mean Square Displacement')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'electron_msd.png'), dpi=dpi, bbox_inches='tight')
    plt.show()
    
def plot_msd_threshold_analysis(results, electron_colors, threshold=0.7, save_plots=True, plot_dir="./plots", dpi=300):
    """
    Plot MSD data with threshold analysis and a separate plot for the linear region.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing simulation results with MSD data
    electron_colors : array-like
        Colors for electron plots
    threshold : float, optional
        MSD threshold value (default: 0.7)
    save_plots : bool, optional
        Whether to save plots to disk (default: True)
    plot_dir : str, optional
        Directory to save plots (default: './plots')
    dpi : int, optional
        Resolution of saved plots (default: 300)
        
    Returns:
    --------
    diffusion_coeffs : dict
        Dictionary containing diffusion coefficients for each electron
    """
    # Create directories for plots if needed
    if save_plots and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Dictionary to store diffusion coefficients
    diffusion_coeffs = {}
    
    # Extract MSD data
    steps = np.arange(len(results["msd"]["0"]))
    
    # Figure 1: Full MSD plot with threshold points
    plt.figure(figsize=(10, 6))
    
    # Plot MSD for electrons
    for e in range(min(2, len(results["msd"]))):
        e_str = str(e)
        if e_str in results["msd"]:
            msd_e = np.array(results["msd"][e_str])
            plt.plot(steps, msd_e, '-', color=electron_colors[e], lw=2, label=f'Electron #{e+1}')
            
            # Find the first index where MSD reaches or exceeds the threshold
            threshold_idx = np.where(msd_e >= threshold)[0]
            if len(threshold_idx) > 0:
                threshold_idx = threshold_idx[0]
            else:
                # If threshold is never reached, use the entire data
                threshold_idx = len(msd_e) - 1
            
            # Get the linear region data (from step 0 to threshold_idx)
            x_linear = steps[:threshold_idx+1]
            y_linear = msd_e[:threshold_idx+1]
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_linear, y_linear)
            
            # Calculate diffusion coefficient (D = slope/6 for 3D)
            diffusion = slope / 6.0
            r2 = r_value ** 2
            
            # Store diffusion coefficient data
            diffusion_coeffs[e_str] = {
                'threshold_step': int(threshold_idx),
                'threshold_value': float(msd_e[threshold_idx]),
                'slope': float(slope),
                'diffusion_coef': float(diffusion),
                'r2': float(r2)
            }
            
            # Plot the linear fit
            fit_y = slope * x_linear + intercept
            plt.plot(x_linear, fit_y, '--', color=electron_colors[e], lw=1.5,
                    label=f'Linear fit #{e+1}: D≈{diffusion:.4e}, R²={r2:.4f}')
            
            # Mark the threshold point
            plt.scatter(threshold_idx, msd_e[threshold_idx], color=electron_colors[e], 
                        marker='o', s=100, edgecolor='black', zorder=10)
            plt.axvline(threshold_idx, color=electron_colors[e], linestyle=':', lw=1, alpha=0.5)
            plt.text(threshold_idx+2, msd_e[threshold_idx], f'MSD={msd_e[threshold_idx]:.2f}', 
                     color=electron_colors[e], fontweight='bold')
    
    # Add a horizontal line at threshold value
    plt.axhline(threshold, color='gray', linestyle='--', lw=1.5, label=f'Threshold = {threshold}')
    
    plt.xlabel('Monte Carlo Step')
    plt.ylabel('Mean Square Displacement')
    plt.title(f'Electron MSD with Linear Fit to First MSD>{threshold} Point')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(plot_dir, f'electron_msd_threshold_{threshold}.png'), dpi=dpi, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Zoomed-in linear region only
    plt.figure(figsize=(10, 6))
    
    for e in range(min(2, len(results["msd"]))):
        e_str = str(e)
        if e_str in results["msd"] and e_str in diffusion_coeffs:
            msd_e = np.array(results["msd"][e_str])
            threshold_idx = diffusion_coeffs[e_str]['threshold_step']
            
            # Get the linear region data
            x_linear = steps[:threshold_idx+1]
            y_linear = msd_e[:threshold_idx+1]
            
            # Plot the actual MSD data in this region
            plt.plot(x_linear, y_linear, 'o-', color=electron_colors[e], lw=2, 
                     markersize=5, alpha=0.7, label=f'Electron #{e+1}')
            
            # Plot the linear fit
            slope = diffusion_coeffs[e_str]['slope']
            intercept = float(y_linear[0] - slope * x_linear[0])  # Recalculate intercept to be sure
            fit_y = slope * x_linear + intercept
            
            plt.plot(x_linear, fit_y, '--', color=electron_colors[e], lw=2,
                    label=f'Fit #{e+1}: y = {slope:.4e}x + {intercept:.4e}')
            
            # Add text with diffusion coefficient
            diffusion = diffusion_coeffs[e_str]['diffusion_coef']
            r2 = diffusion_coeffs[e_str]['r2']
            plt.text(x_linear[-1]*0.1, y_linear[-1]*0.9, 
                     f'D{e+1} = {diffusion:.4e}\nR² = {r2:.4f}', 
                     color=electron_colors[e], fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    plt.xlabel('Monte Carlo Step')
    plt.ylabel('Mean Square Displacement')
    plt.title(f'Linear Region of MSD (up to threshold {threshold})')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(plot_dir, f'msd_linear_region_{threshold}.png'), dpi=dpi, bbox_inches='tight')
    plt.show()
    
    # Display diffusion coefficients
    print("\nDiffusion Coefficients (MSD Threshold Analysis):")
    print("-" * 80)
    print(f"{'Electron':<10} {'Threshold Step':<15} {'Threshold MSD':<15} {'Slope':<15} {'Diffusion Coef':<15} {'R²':<8}")
    print("-" * 80)
    
    for e_id, data in diffusion_coeffs.items():
        print(f"#{int(e_id)+1:<9} {data['threshold_step']:<15d} {data['threshold_value']:<15.6f} "
              f"{data['slope']:<15.6e} {data['diffusion_coef']:<15.6e} {data['r2']:<8.4f}")
    
    print("-" * 80)
    
    return diffusion_coeffs

def plot_max_and_mean_ages(results, colors, save_plots=True, plot_dir="./plots", dpi=300, kmax=None):
    """
    Plot maximum and mean ages of walkers across blocks.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing VMC simulation results
    colors : array-like
        Colors for plots
    save_plots : bool
        Whether to save plots to disk
    plot_dir : str
        Directory to save plots
    dpi : int
        Resolution of saved plots
    kmax : float, optional
        Theoretical kmax value for comparison
    """
    if "max_ages" not in results and "mean_ages" not in results:
        print("No age data found in results.")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    blocks = np.arange(1, len(results["max_ages"]) + 1)
    
    # Plot maximum ages
    if "max_ages" in results:
        max_ages = np.array(results["max_ages"])
        ax1.plot(blocks, max_ages, 'o-', lw=2, color=colors[0], label='Maximum Age')
        
        # Add theoretical kmax if provided
        if kmax is not None:
            ax1.axhline(kmax, color='red', linestyle='--', lw=2, 
                       label=f'Theoretical kmax = {kmax:.1f}')
            
            # Calculate ratio of observed max age to theoretical kmax
            observed_max = np.max(max_ages)
            ratio = observed_max / kmax
            ax1.text(blocks[-1] * 0.5, kmax * 0.95, 
                    f'Max Age / kmax = {ratio:.2f}', 
                    color='red', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        ax1.set_ylabel('Maximum Age (MC Steps)')
        ax1.set_title('Maximum Walker Ages')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
    
    # Plot mean ages
    if "mean_ages" in results:
        mean_ages = np.array(results["mean_ages"])
        ax2.plot(blocks, mean_ages, 'o-', lw=2, color=colors[1], label='Mean Age')
        
        ax2.set_xlabel('Block')
        ax2.set_ylabel('Mean Age (MC Steps)')
        ax2.set_title('Mean Walker Ages')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'walker_ages.png'), dpi=dpi, bbox_inches='tight')
    plt.show()
    
    # Additional plot: Final age histogram if available
    if "final_age_histogram" in results:
        plt.figure(figsize=(10, 6))
        
        age_histogram = np.array(results["final_age_histogram"])
        age_bins = np.arange(len(age_histogram))
        
        # Plot histogram as bars
        plt.bar(age_bins, age_histogram, color=colors[2], alpha=0.7)
        
        # Add vertical line for average age
        if "final_mean_age" in results:
            mean_age = results["final_mean_age"]
            plt.axvline(mean_age, color=colors[1], linestyle='--', lw=2, 
                        label=f'Mean Age = {mean_age:.1f}')
        
        # Add vertical line for maximum age
        if "final_max_age" in results:
            max_age = results["final_max_age"]
            plt.axvline(max_age, color=colors[0], linestyle='--', lw=2, 
                        label=f'Max Age = {max_age:.0f}')
            
            # Add kmax reference if provided
            if kmax is not None:
                plt.axvline(kmax, color='red', linestyle='--', lw=2, 
                           label=f'Theoretical kmax = {kmax:.1f}')
        
        plt.xlabel('Age (MC Steps)')
        plt.ylabel('Number of Walkers')
        plt.title('Final Age Distribution of Walkers')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(plot_dir, 'age_histogram.png'), dpi=dpi, bbox_inches='tight')
        plt.show()
        
    # Add plot for stuck walkers if available
    if "stuck_walker_count" in results:
        plt.figure(figsize=(10, 6))
        
        stuck_counts = np.array(results["stuck_walker_count"])
        
        plt.plot(blocks, stuck_counts, 'o-', lw=2, color='red', 
                label=f'Stuck Walkers (age > {results.get("max_age_threshold", 20)})')
        
        plt.xlabel('Block')
        plt.ylabel('Number of Stuck Walkers')
        plt.title('Stuck Walker Count per Block')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(plot_dir, 'stuck_walkers.png'), dpi=dpi, bbox_inches='tight')
        plt.show()

def plot_autocorrelation(results, electron_colors, save_plots, plot_dir, dpi):
    """Plot autocorrelation function for electrons."""
    
    if "autocorr" in results and results["autocorr"]:
        plt.figure(figsize=(10, 6))
        
        # Get the autocorrelation data
        max_lag = len(results["autocorr"]["0"])
        lags = np.arange(max_lag)
        
        # Plot autocorrelation for electrons #1 and #2
        for e in range(2):  # Just the first two electrons
            if str(e) in results["autocorr"]:
                autocorr_e = np.array(results["autocorr"][str(e)])
                plt.plot(lags, autocorr_e, '-', color=electron_colors[e], lw=2, label=f'Electron #{e+1}')
        
        # Add horizontal line at zero
        plt.axhline(0, color='gray', linestyle='--', lw=1)
        
        # Add horizontal line at 1/e
        plt.axhline(1/np.e, color='black', linestyle=':', lw=1, alpha=0.7, label='1/e threshold')
        
        # Add exponential decay reference using the first electron's data
        if "0" in results["autocorr"]:
            autocorr_0 = np.array(results["autocorr"]["0"])
            # Find the autocorrelation time (where value drops to 1/e)
            e_value = 1/np.e
            tau_idx = np.argmin(np.abs(autocorr_0 - e_value))
            if tau_idx > 0:  # Make sure we found a valid tau
                tau_0 = tau_idx
                x_exp = np.arange(max_lag)
                y_exp = np.exp(-x_exp/tau_0)
                plt.plot(x_exp, y_exp, ':', color=electron_colors[0], lw=1.5, 
                         label=f'exp(-t/{tau_0:.1f})')
                
                # Add vertical line at tau
                plt.axvline(tau_0, color=electron_colors[0], linestyle=':', lw=1)
                plt.text(tau_0+0.5, 0.4, f'τ₁ ≈ {tau_0:.1f}', color=electron_colors[0])
        
        # Do the same for electron #2
        if "1" in results["autocorr"]:
            autocorr_1 = np.array(results["autocorr"]["1"])
            tau_idx = np.argmin(np.abs(autocorr_1 - e_value))
            if tau_idx > 0:
                tau_1 = tau_idx
                plt.axvline(tau_1, color=electron_colors[1], linestyle=':', lw=1)
                plt.text(tau_1+0.5, 0.3, f'τ₂ ≈ {tau_1:.1f}', color=electron_colors[1])
        
        plt.xlabel('Time Lag (MC Steps)')
        plt.ylabel('Autocorrelation')
        plt.title('Position Autocorrelation Function')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.2, 1.05)  # Give some space above 1 and below 0
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(plot_dir, 'autocorrelation.png'), dpi=dpi, bbox_inches='tight')
        plt.show()
    else:
        print("No autocorrelation data found in results.")

def print_vmc_summary(results, kmax=None):
    """
    Print a concise summary of VMC results focusing on the requested metrics.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing VMC simulation results
    kmax : float, optional
        Theoretical kmax value for comparison with electron ages
    """
    print("\n" + "="*60)
    print("VMC Simulation Summary".center(60))
    print("="*60)
    
    # Energy results
    print("\nEnergy Results:")
    print(f"  Mean Energy:      {results['mean_energy']:.8f} ± {results['std_error']:.8f} a.u.")
    
    # Acceptance ratio
    if "mean_acceptance" in results:
        print(f"  Mean Acceptance:  {results['mean_acceptance']:.4f}")
    
    # Autocorrelation times
    if "autocorr" in results and results["autocorr"]:
        print("\nAutocorrelation Times:")
        for e in range(2):  # Just for electrons #1 and #2
            if str(e) in results["autocorr"]:
                autocorr_e = np.array(results["autocorr"][str(e)])
                tau_idx = np.argmin(np.abs(autocorr_e - 1/np.e))
                if tau_idx > 0:
                    print(f"  Electron #{e+1}:     {tau_idx:.2f} MC steps")
    
    # MSD and diffusion coefficients
    if "msd" in results and "0" in results["msd"] and "1" in results["msd"]:
        print("\nMSD Analysis:")
        steps = np.arange(len(results["msd"]["0"]))
        fit_start = len(steps) // 4  # Skip initial transient
        
        for e in range(2):  # Just for electrons #1 and #2
            msd_e = np.array(results["msd"][str(e)])
            coeffs = np.polyfit(steps[fit_start:], msd_e[fit_start:], 1)
            diffusion = coeffs[0] / 6.0  # D = slope/6 for 3D
            print(f"  Electron #{e+1} Diffusion:  {diffusion:.4e} a.u.²/step")
    
    print("\n" + "="*60)

def create_powerpoint_plots(results, case_name="", save_plots=True, plot_dir="./ppt_plots", dpi=300, kmax=None):
    """
    Create PowerPoint-friendly subplots for a single VMC case.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing VMC simulation results
    case_name : str, optional
        Name or description of the case
    save_plots : bool, optional
        Whether to save plots to disk (default: True)
    plot_dir : str, optional
        Directory to save plots (default: './ppt_plots')
    dpi : int, optional
        Resolution of saved plots (default: 300)
    kmax : float, optional
        Theoretical kmax value for comparison
        
    Returns:
    --------
    summary : dict
        Dictionary with summary statistics
    """
    if save_plots and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Set style for presentation-quality plots
    sns.set_style("whitegrid")
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['figure.titlesize'] = 14
    
    # Define colors
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    electron_colors = plt.cm.plasma(np.linspace(0, 0.8, 2))
    
    # -------------------------------------------------------------------------
    # Figure 1: Acceptance Rate, Cumulative Energy, and Electron Ages combined
    # -------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(10, 7.5))  # 4:3 aspect ratio for PowerPoint
    gs = GridSpec(2, 2, figure=fig1)
    
    # 1. Acceptance Rate - Top Left
    ax1 = fig1.add_subplot(gs[0, 0])
    blocks = np.arange(1, len(results["acceptance"]) + 1)
    acceptance = np.array(results["acceptance"])
    cumul_acceptance = np.array(results["cumulative_acceptance"])
    
    ax1.plot(blocks, acceptance, 'o-', color=colors[0], lw=2, label='Block')
    ax1.plot(blocks, cumul_acceptance, '-', color=colors[1], lw=2, label='Cumulative')
    ax1.axhline(0.5, color='r', linestyle='--', lw=1.5, label='Ideal (50%)')
    ax1.set_xlabel('Block')
    ax1.set_ylabel('Acceptance Ratio')
    ax1.set_title('Acceptance Ratio')
    ax1.legend(loc='best', frameon=True, framealpha=0.7)
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative Energy - Top Right
    ax2 = fig1.add_subplot(gs[0, 1])
    block_energies = np.array(results["total"])
    cumul_energy = np.array(results["cumulative_energy"])
    cumul_error = np.array(results["cumulative_error"])
    mean_energy = results["mean_energy"]
    std_error = results["std_error"]
    
    ax2.plot(blocks, block_energies, 'o-', color=colors[0], lw=2, label='Block')
    ax2.plot(blocks, cumul_energy, '-', color=colors[1], lw=2, label='Cumulative')
    ax2.fill_between(blocks, cumul_energy - cumul_error, cumul_energy + cumul_error, 
                    color=colors[1], alpha=0.3)
    
    # Mean energy line
    label_text = f'Mean: {mean_energy:.6f} ± {std_error:.6f}'
    ax2.axhline(mean_energy, color='red', linestyle='--', lw=1.5, label='Mean')
    
    ax2.set_xlabel('Block')
    ax2.set_ylabel('Energy (a.u.)')
    ax2.set_title('Energy Convergence')
    ax2.legend(loc='best', frameon=True, framealpha=0.7)
    ax2.grid(True, alpha=0.3)
    
    # 3. Electron Ages - Bottom (span both columns)
    if "max_ages" in results:
        ax3 = fig1.add_subplot(gs[1, :])
        max_ages = np.array(results["max_ages"])
        
        for e in range(max_ages.shape[1]):
            if e < 2:  # Use special colors for electrons #1 and #2
                ax3.plot(blocks, max_ages[:, e], 'o-', lw=2, 
                         color=electron_colors[e], label=f'Electron #{e+1}')
            else:
                ax3.plot(blocks, max_ages[:, e], 'o-', lw=1.5, alpha=0.7,
                         label=f'Electron #{e+1}')
        
        # Add theoretical kmax if provided
        if kmax is not None:
            ax3.axhline(kmax, color='red', linestyle='--', lw=2, 
                       label=f'Theoretical kmax = {kmax:.1f}')
        
        ax3.set_xlabel('Block')
        ax3.set_ylabel('Maximum Age (MC Steps)')
        ax3.set_title('Maximum Electron Ages')
        ax3.legend(loc='best', frameon=True, framealpha=0.7)
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        fig1.savefig(os.path.join(plot_dir, f'{case_name}_metrics.png'), dpi=dpi, bbox_inches='tight')
    
    # -------------------------------------------------------------------------
    # Figure 2: Autocorrelation and Electron MSD combined
    # -------------------------------------------------------------------------
    fig2 = plt.figure(figsize=(10, 7.5))  # 4:3 aspect ratio for PowerPoint
    gs = GridSpec(2, 1, figure=fig2)
    
    # 1. Autocorrelation - Top
    tau_values = {}
    if "autocorr" in results and results["autocorr"]:
        ax1 = fig2.add_subplot(gs[0, 0])
        
        max_lag = len(results["autocorr"]["0"])
        lags = np.arange(max_lag)
        
        # Plot autocorrelation for electrons #1 and #2
        for e in range(2):  # Just the first two electrons
            if str(e) in results["autocorr"]:
                autocorr_e = np.array(results["autocorr"][str(e)])
                ax1.plot(lags, autocorr_e, '-', color=electron_colors[e], lw=2, label=f'Electron #{e+1}')
        
        # Add horizontal line at zero and 1/e
        ax1.axhline(0, color='gray', linestyle='--', lw=1)
        ax1.axhline(1/np.e, color='black', linestyle=':', lw=1, alpha=0.7, label='1/e threshold')
        
        # Calculate and show tau values
        e_value = 1/np.e
        for e in range(2):
            if str(e) in results["autocorr"]:
                autocorr_e = np.array(results["autocorr"][str(e)])
                tau_idx = np.argmin(np.abs(autocorr_e - e_value))
                if tau_idx > 0:
                    tau = tau_idx
                    tau_values[str(e)] = tau
                    ax1.axvline(tau, color=electron_colors[e], linestyle=':', lw=1)
                    ax1.text(tau+0.5, 0.4 - e*0.1, f'τ{e+1} ≈ {tau:.1f}', color=electron_colors[e])
        
        ax1.set_xlabel('Time Lag (MC Steps)')
        ax1.set_ylabel('Autocorrelation')
        ax1.set_title('Position Autocorrelation Function')
        ax1.legend(loc='best', frameon=True, framealpha=0.7)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.2, 1.05)
    
    # 2. MSD - Bottom
    ax2 = fig2.add_subplot(gs[1, 0])
    
    # Extract MSD data
    steps = np.arange(len(results["msd"]["0"]))
    msd_0 = np.array(results["msd"]["0"])
    msd_1 = np.array(results["msd"]["1"])
    
    ax2.plot(steps, msd_0, '-', color=electron_colors[0], lw=2, label='Electron #1')
    ax2.plot(steps, msd_1, '-', color=electron_colors[1], lw=2, label='Electron #2')
    
    # Add linear fit to estimate diffusion coefficient
    fit_start = len(steps) // 4  # Skip initial transient
    
    # Fit for electron #1
    coeffs_0 = np.polyfit(steps[fit_start:], msd_0[fit_start:], 1)
    ax2.plot(steps[fit_start:], np.polyval(coeffs_0, steps[fit_start:]), 
             '--', color=electron_colors[0], lw=1.5, 
             label=f'D₁≈{coeffs_0[0]/6:.4e}')
    
    # Fit for electron #2
    coeffs_1 = np.polyfit(steps[fit_start:], msd_1[fit_start:], 1)
    ax2.plot(steps[fit_start:], np.polyval(coeffs_1, steps[fit_start:]), 
             '--', color=electron_colors[1], lw=1.5, 
             label=f'D₂≈{coeffs_1[0]/6:.4e}')
    
    ax2.set_xlabel('Monte Carlo Step')
    ax2.set_ylabel('Mean Square Displacement')
    ax2.set_title('Electron Mean Square Displacement')
    ax2.legend(loc='best', frameon=True, framealpha=0.7)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        fig2.savefig(os.path.join(plot_dir, f'{case_name}_correlation.png'), dpi=dpi, bbox_inches='tight')
    
    # Show plots if interactive session
    plt.show()
    
    # Return summary statistics as a bonus
    summary = {
        "Energy": f"{mean_energy:.6f} ± {std_error:.6f}",
        "Acceptance": f"{results['mean_acceptance']:.4f}",
        "Max_Age_E1": f"{results['max_ages'][-1][0]:.0f}",
        "Max_Age_E2": f"{results['max_ages'][-1][1]:.0f}",
        "Autocorr_Time_E1": tau_values.get('0', 'N/A'),
        "Diffusion_E1": f"{coeffs_0[0]/6:.4e}",
        "Diffusion_E2": f"{coeffs_1[0]/6:.4e}"
    }
    
    # Add kmax comparison if provided
    if kmax is not None:
        summary["kmax"] = kmax
        summary["Max_Age_E1_Ratio"] = f"{results['max_ages'][-1][0]/kmax:.2f}"
        summary["Max_Age_E2_Ratio"] = f"{results['max_ages'][-1][1]/kmax:.2f}"
    
    print(f"Analysis complete for {case_name}!")
    return summary