import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec

def analyze_vmc_results(results, save_plots=True, plot_dir="./plots", dpi=300):
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
    
    # 1. Cumulative Monte Carlo Energy, Variance, Error
    plot_cumulative_energy(results, colors, save_plots, plot_dir, dpi)
    
    # 2. Cumulative Acceptance Ratio
    plot_acceptance_ratio(results, colors, save_plots, plot_dir, dpi)
    
    # 3. MSD of Electrons #1 and #2
    plot_electron_msd(results, electron_colors, save_plots, plot_dir, dpi)
    
    # 4. Max Age of Each Electron
    plot_electron_ages(results, electron_colors, save_plots, plot_dir, dpi)
    
    # 5. Autocorrelation
    plot_autocorrelation(results, electron_colors, save_plots, plot_dir, dpi)
    
    print("Analysis complete!")

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

def plot_electron_ages(results, electron_colors, save_plots, plot_dir, dpi):
    """Plot maximum age of each electron."""
    
    if "max_ages" in results:
        # Max age trajectory across blocks
        plt.figure(figsize=(10, 6))
        
        max_ages = np.array(results["max_ages"])
        blocks = np.arange(1, len(max_ages) + 1)
        
        for e in range(max_ages.shape[1]):
            if e < 2:  # Use special colors for electrons #1 and #2
                plt.plot(blocks, max_ages[:, e], 'o-', lw=2, 
                         color=electron_colors[e], label=f'Electron #{e+1}')
            else:
                plt.plot(blocks, max_ages[:, e], 'o-', lw=1.5, alpha=0.7,
                         label=f'Electron #{e+1}')
        
        plt.xlabel('Block')
        plt.ylabel('Maximum Age (MC Steps)')
        plt.title('Maximum Electron Ages')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(plot_dir, 'electron_ages.png'), dpi=dpi, bbox_inches='tight')
        plt.show()
        
        # Final max ages as bar chart
        plt.figure(figsize=(12, 6))
        
        # Use final_max_ages if available, otherwise use the last entry in max_ages
        if "final_max_ages" in results:
            final_ages = results["final_max_ages"]
        else:
            final_ages = max_ages[-1]
        
        # Create bar colors with special highlighting for electrons #1 and #2
        bar_colors = ['gray'] * len(final_ages)
        bar_colors[0] = electron_colors[0]
        if len(final_ages) > 1:
            bar_colors[1] = electron_colors[1]
        
        plt.bar(np.arange(1, len(final_ages) + 1), final_ages, color=bar_colors)
        plt.xlabel('Electron')
        plt.ylabel('Maximum Age (MC Steps)')
        plt.title('Final Maximum Electron Ages')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(np.arange(1, len(final_ages) + 1))
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(plot_dir, 'final_electron_ages.png'), dpi=dpi, bbox_inches='tight')
        plt.show()
    else:
        print("No electron age data found in results.")

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

def print_vmc_summary(results):
    """Print a concise summary of VMC results focusing on the requested metrics."""
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
    
    # Maximum ages
    if "final_max_ages" in results:
        print("\nMaximum Ages:")
        for e in range(2):  # Just for electrons #1 and #2
            if e < len(results["final_max_ages"]):
                print(f"  Electron #{e+1}:     {results['final_max_ages'][e]:.0f} MC steps")
    
    print("\n" + "="*60)
    
    

def create_powerpoint_plots(results, case_name="", save_plots=True, plot_dir="./ppt_plots", dpi=300):
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
        
    Returns:
    --------
    None (plots are saved to disk)
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
        "Autocorr_Time_E1": tau_idx if "autocorr" in results else "N/A",
        "Diffusion_E1": f"{coeffs_0[0]/6:.4e}",
        "Diffusion_E2": f"{coeffs_1[0]/6:.4e}"
    }
    
    print(f"Analysis complete for {case_name}!")
    return summary