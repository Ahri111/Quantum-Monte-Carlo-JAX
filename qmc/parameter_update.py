import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, List
import numpy as np

# Import required functions from mc.py
from qmc.mc import vmc_simple_improved, orthogonalize_orbitals_s_based
from qmc.determinants import parameter_gradient

# ===========================
# Minimal JIT Functions (Pure Mathematical Operations Only)
# ===========================

@jax.jit
def apply_gradient_clipping(gradient: jnp.ndarray, max_norm: float = 0.1) -> jnp.ndarray:
    """Apply gradient clipping - only vector normalization uses JIT"""
    grad_norm = jnp.linalg.norm(gradient.flatten())
    clip_factor = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
    return gradient * clip_factor

@jax.jit
def compute_natural_gradient(fim_inverse: jnp.ndarray, energy_gradient: jnp.ndarray) -> jnp.ndarray:
    """Compute natural gradient - only matrix multiplication uses JIT"""
    return jnp.dot(fim_inverse, energy_gradient.flatten()).reshape(energy_gradient.shape)

# ===========================
# Fisher Information Matrix Calculation
# ===========================

def compute_fisher_information_matrix(
    psi_gradients: Dict[str, jnp.ndarray], 
    regularization: float = 1e-4
) -> Dict[str, jnp.ndarray]:
    """
    Compute Fisher Information Matrix (equation-based)
    
    Mathematical definition:
    F_ij = ⟨∂ln|Ψ|/∂θ_i ∂ln|Ψ|/∂θ_j⟩ - ⟨∂ln|Ψ|/∂θ_i⟩⟨∂ln|Ψ|/∂θ_j⟩
    
    Args:
        psi_gradients: Dictionary of wavefunction gradients
        regularization: Tikhonov regularization parameter
        
    Returns:
        Dictionary of FIM inverse matrices for each parameter
    """
    fim_inverses = {}
    
    for param_name, grad in psi_gradients.items():
        nconf = grad.shape[0]
        grad_flat = grad.reshape(nconf, -1)  # [nconf, n_params]
        
        # Gradient mean: ⟨∂ln|Ψ|/∂θ⟩
        mean_grad = jnp.mean(grad_flat, axis=0)
        
        # Centered gradient: ∂ln|Ψ|/∂θ - ⟨∂ln|Ψ|/∂θ⟩
        centered_grad = grad_flat - mean_grad
        
        # Fisher Information Matrix: F = (1/N) Σ(g - ⟨g⟩)(g - ⟨g⟩)^T
        fim = jnp.dot(centered_grad.T, centered_grad) / nconf
        
        # Tikhonov regularization: F_reg = F + λI
        fim_regularized = fim + regularization * jnp.eye(fim.shape[0])
        
        # Safe matrix inversion
        try:
            # Cholesky decomposition (most stable)
            L = jnp.linalg.cholesky(fim_regularized)
            fim_inv = jnp.linalg.inv(L.T) @ jnp.linalg.inv(L)
        except:
            try:
                # Regular inverse
                fim_inv = jnp.linalg.inv(fim_regularized)
            except:
                # Pseudo-inverse (last resort)
                fim_inv = jnp.linalg.pinv(fim_regularized)
                print(f"Warning: Using pseudo-inverse for {param_name}")
        
        fim_inverses[param_name] = fim_inv
        
        # Monitor numerical stability
        condition_number = jnp.linalg.cond(fim_regularized)
        if condition_number > 1e12:
            print(f"Warning: Poor conditioning for {param_name}: {condition_number:.2e}")
            
    return fim_inverses

def compute_energy_gradients(
    local_energies: jnp.ndarray,
    psi_gradients: Dict[str, jnp.ndarray],
    outlier_threshold: float = 3.0
) -> Dict[str, jnp.ndarray]:
    """
    Compute energy gradients (variational principle)
    
    Mathematical definition:
    ∂⟨E⟩/∂θ = 2 Re[⟨(E_L - ⟨E_L⟩) ∂ln|Ψ|/∂θ⟩]
    
    Args:
        local_energies: Array of local energies
        psi_gradients: Wavefunction gradients
        outlier_threshold: Outlier removal threshold (MAD-based)
        
    Returns:
        Dictionary of energy gradients
    """
    # Outlier removal (Median Absolute Deviation method)
    energy_median = jnp.median(local_energies)
    energy_mad = jnp.median(jnp.abs(local_energies - energy_median))
    
    # Create outlier mask based on MAD
    outlier_mask = jnp.abs(local_energies - energy_median) < outlier_threshold * energy_mad
    
    # Use only clean data
    clean_energies = local_energies[outlier_mask]
    energy_mean = jnp.mean(clean_energies)
    
    energy_gradients = {}
    
    for param_name, psi_grad in psi_gradients.items():
        # Apply outlier mask
        clean_psi_grad = psi_grad[outlier_mask]
        
        # Variational principle: ∂⟨E⟩/∂θ = 2⟨(E_L - ⟨E_L⟩) ∂ln|Ψ|/∂θ⟩
        energy_centered = clean_energies - energy_mean
        
        # Compute covariance: ⟨(E_L - ⟨E_L⟩) ∂ln|Ψ|/∂θ⟩
        covariance = jnp.mean(
            energy_centered[:, jnp.newaxis, jnp.newaxis] * clean_psi_grad, 
            axis=0
        )
        
        # Energy gradient (factor of 2 from variational principle)
        energy_gradients[param_name] = 2.0 * jnp.real(covariance)
        
    return energy_gradients

# ===========================
# Parameter Update Methods
# ===========================

def fisher_information_update(
    mo_coeff: Tuple[jnp.ndarray, jnp.ndarray],
    energy_gradients: Dict[str, jnp.ndarray],
    psi_gradients: Dict[str, jnp.ndarray],
    learning_rate: float = 0.005,
    regularization: float = 1e-4,
    max_gradient_norm: float = 0.1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fisher Information Matrix-based update (Stochastic Reconfiguration)
    
    Mathematical definition:
    θ_new = θ_old - α F^(-1) ∇E
    
    Where:
    - F is the Fisher Information Matrix
    - ∇E is the energy gradient
    - α is the learning rate
    
    Args:
        mo_coeff: Current MO coefficients (alpha, beta)
        energy_gradients: Energy gradients
        psi_gradients: Wavefunction gradients for FIM computation
        learning_rate: Learning rate
        regularization: FIM regularization parameter
        max_gradient_norm: Gradient clipping threshold
        
    Returns:
        Updated MO coefficients
    """
    # Compute Fisher Information Matrix inverses
    fim_inverses = compute_fisher_information_matrix(psi_gradients, regularization)
    
    new_mo_coeff = list(mo_coeff)
    
    # Update alpha orbitals
    if "mo_coeff_alpha" in energy_gradients:
        # Natural gradient: F^(-1) ∇E
        natural_grad = compute_natural_gradient(
            fim_inverses["mo_coeff_alpha"],
            energy_gradients["mo_coeff_alpha"]
        )
        
        # Gradient clipping
        clipped_grad = apply_gradient_clipping(natural_grad, max_gradient_norm)
        
        # Parameter update: θ_new = θ_old - α * (clipped natural gradient)
        new_mo_coeff[0] = mo_coeff[0] - learning_rate * clipped_grad
    
    # Update beta orbitals
    if "mo_coeff_beta" in energy_gradients:
        natural_grad = compute_natural_gradient(
            fim_inverses["mo_coeff_beta"],
            energy_gradients["mo_coeff_beta"]
        )
        clipped_grad = apply_gradient_clipping(natural_grad, max_gradient_norm)
        new_mo_coeff[1] = mo_coeff[1] - learning_rate * clipped_grad
    
    return tuple(new_mo_coeff)

def gradient_descent_update(
    mo_coeff: Tuple[jnp.ndarray, jnp.ndarray],
    energy_gradients: Dict[str, jnp.ndarray],
    learning_rate: float = 0.002,
    momentum: float = 0.9,
    velocity: Optional[Dict[str, jnp.ndarray]] = None,
    max_gradient_norm: float = 0.1
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    Gradient descent with momentum
    
    Mathematical definition:
    v_t = β v_{t-1} - α ∇E
    θ_new = θ_old + v_t
    
    Args:
        mo_coeff: Current MO coefficients
        energy_gradients: Energy gradients
        learning_rate: Learning rate
        momentum: Momentum coefficient
        velocity: Previous velocity (for momentum)
        max_gradient_norm: Gradient clipping threshold
        
    Returns:
        (Updated MO coefficients, new velocity)
    """
    new_mo_coeff = list(mo_coeff)
    
    if velocity is None:
        velocity = {}
    
    # Update alpha orbitals
    if "mo_coeff_alpha" in energy_gradients:
        grad = energy_gradients["mo_coeff_alpha"]
        clipped_grad = apply_gradient_clipping(grad, max_gradient_norm)
        
        # Initialize velocity
        if "mo_coeff_alpha" not in velocity:
            velocity["mo_coeff_alpha"] = jnp.zeros_like(grad)
        
        # Momentum update: v = β v_prev - α ∇E
        velocity["mo_coeff_alpha"] = (
            momentum * velocity["mo_coeff_alpha"] - learning_rate * clipped_grad
        )
        
        # Parameter update: θ_new = θ_old + v
        new_mo_coeff[0] = mo_coeff[0] + velocity["mo_coeff_alpha"]
    
    # Update beta orbitals
    if "mo_coeff_beta" in energy_gradients:
        grad = energy_gradients["mo_coeff_beta"]
        clipped_grad = apply_gradient_clipping(grad, max_gradient_norm)
        
        if "mo_coeff_beta" not in velocity:
            velocity["mo_coeff_beta"] = jnp.zeros_like(grad)
        
        velocity["mo_coeff_beta"] = (
            momentum * velocity["mo_coeff_beta"] - learning_rate * clipped_grad
        )
        new_mo_coeff[1] = mo_coeff[1] + velocity["mo_coeff_beta"]
    
    return tuple(new_mo_coeff), velocity

# ===========================
# Adaptive Learning Rate and Safety Mechanisms
# ===========================

def adapt_learning_rate(
    current_lr: float,
    energy_history: List[float],
    patience: int = 5,
    factor: float = 0.8,
    min_lr: float = 1e-6,
    max_lr: float = 0.1
) -> float:
    """
    Adaptive learning rate adjustment based on energy improvement
    
    Args:
        current_lr: Current learning rate
        energy_history: Recent energy history
        patience: Number of steps to wait without improvement
        factor: Reduction factor
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        
    Returns:
        Adjusted learning rate
    """
    if len(energy_history) < patience + 1:
        return current_lr
    
    recent_energies = energy_history[-patience:]
    
    # Decrease learning rate if energy doesn't improve
    if all(recent_energies[i] >= recent_energies[i-1] - 1e-8 for i in range(1, len(recent_energies))):
        new_lr = max(current_lr * factor, min_lr)
        return new_lr
    
    # Slightly increase learning rate if energy consistently improves
    elif all(recent_energies[i] < recent_energies[i-1] for i in range(1, len(recent_energies))):
        new_lr = min(current_lr * 1.05, max_lr)
        return new_lr
    
    return current_lr

def check_convergence(
    energy_history: List[float],
    gradient_norms: List[float],
    energy_tolerance: float = 1e-6,
    gradient_tolerance: float = 1e-4,
    window_size: int = 10
) -> bool:
    """
    Check convergence criteria
    
    Args:
        energy_history: Energy history
        gradient_norms: Gradient norm history
        energy_tolerance: Energy change tolerance
        gradient_tolerance: Gradient norm tolerance
        window_size: Convergence assessment window size
        
    Returns:
        Whether converged
    """
    if len(energy_history) < window_size:
        return False
    
    # Check recent energy changes
    recent_energies = energy_history[-window_size:]
    energy_std = jnp.std(jnp.array(recent_energies))
    
    # Check recent gradient norms
    if len(gradient_norms) >= window_size:
        recent_gradients = gradient_norms[-window_size:]
        avg_gradient_norm = jnp.mean(jnp.array(recent_gradients))
        
        return energy_std < energy_tolerance and avg_gradient_norm < gradient_tolerance
    
    return energy_std < energy_tolerance

# ===========================
# Safe Parameter Settings by System Size
# ===========================

def get_safe_optimization_params(system_size: str = "medium") -> Dict:
    """
    Safe optimization parameters by system size
    
    Args:
        system_size: "small" (< 10 electrons), "medium" (10-50 electrons), "large" (> 50 electrons)
        
    Returns:
        Optimization parameter dictionary
    """
    params = {
        "small": {      # H2, LiH, etc.
            "n_iterations": 100,
            "fisher_lr": 0.03,
            "gradient_lr": 0.01,
            "vmc_blocks": 10,
            "vmc_steps_per_block": 100,
            "equilibration": 1000,
            "regularization": 1e-5,
            "gradient_clip": 0.1,
            "patience": 3
        },
        "medium": {     # H2O, NH3, CH4, etc.
            "n_iterations": 50,
            "fisher_lr": 0.005,
            "gradient_lr": 0.002,
            "vmc_blocks": 8,
            "vmc_steps_per_block": 40,
            "equilibration": 1000,
            "regularization": 1e-4,
            "gradient_clip": 0.08,
            "patience": 5
        },
        "large": {      # Benzene, naphthalene, etc.
            "n_iterations": 80,
            "fisher_lr": 0.008,
            "gradient_lr": 0.003,
            "vmc_blocks": 10,
            "vmc_steps_per_block": 50,
            "equilibration": 1500,
            "regularization": 1e-3,
            "gradient_clip": 0.05,
            "patience": 8
        }
    }
    
    return params.get(system_size, params["medium"])

# ===========================
# Main Optimization Function
# ===========================

def optimize_mo_coeff_improved(
    mol, configs, mo_coeff, overlap_matrix, det_coeff, det_map, _nelec, occup_hash,
    atom_charges, atom_coords, key, get_phase,
    n_iterations: int = None,
    method: str = "fisher",
    time_step: int = None,#
    initial_learning_rate: float = None,
    system_size: str = "medium",
    vmc_blocks: int = None,
    vmc_steps_per_block: int = None,
    verbose: bool = True,
    save_history: bool = True
) -> Dict:
    """
    Improved MO coefficient optimization (Fisher Information vs Gradient Descent)
    
    Args:
        mol: PySCF molecule object
        configs: Initial electron configurations
        mo_coeff: Initial MO coefficients
        overlap_matrix: AO overlap matrix
        det_coeff: Determinant coefficients
        det_map: Determinant mapping
        _nelec: Electron count information
        occup_hash: Occupation hash
        atom_charges: Atomic charges
        atom_coords: Atomic coordinates
        key: JAX random key
        get_phase: Phase function
        n_iterations: Number of optimization iterations
        method: "fisher" (FIM) or "gradient" (GD)
        initial_learning_rate: Initial learning rate
        system_size: System size ("small", "medium", "large")
        vmc_blocks: Number of VMC blocks
        vmc_steps_per_block: Steps per block
        verbose: Whether to print progress
        save_history: Whether to save history
        
    Returns:
        Optimization results dictionary
    """
    # Load safe parameters
    safe_params = get_safe_optimization_params(system_size)
    
    # Set parameters (use defaults if None)
    n_iterations = n_iterations or safe_params["n_iterations"]
    vmc_blocks = vmc_blocks or safe_params["vmc_blocks"]
    vmc_steps_per_block = vmc_steps_per_block or safe_params["vmc_steps_per_block"]
    
    if initial_learning_rate is None:
        learning_rate = safe_params["fisher_lr"] if method == "fisher" else safe_params["gradient_lr"]
    else:
        learning_rate = initial_learning_rate
    
    # Initialize
    current_mo_coeff = mo_coeff
    current_configs = configs
    current_lr = learning_rate
    velocity = None  # For GD momentum
    
    # Tracking variables
    energy_history = []
    gradient_norms = []
    acceptance_history = []
    learning_rates = []
    
    # Track best parameters
    best_energy = float('inf')
    best_mo_coeff = current_mo_coeff
    best_configs = current_configs
    best_iteration = 0
    
    if verbose:
        print(f"\n=== MO Coefficient Optimization ===")
        print(f"Method: {method.upper()}")
        print(f"System size: {system_size}")
        print(f"Iterations: {n_iterations}")
        print(f"VMC sampling: {vmc_blocks} blocks × {vmc_steps_per_block} steps")
        print(f"Initial learning rate: {learning_rate:.6f}")
        print(f"Regularization: {safe_params['regularization']}")
        print("-" * 50)
    
    for iteration in range(n_iterations):
        # 1. VMC sampling (using vmc_simple_improved from mc.py)
        vmc_results = vmc_simple_improved(
            current_configs, mol, atom_charges, atom_coords,
            current_mo_coeff, det_coeff, det_map, _nelec, occup_hash,
            get_phase, key,
            equilibration_step=safe_params["equilibration"],
            tstep=time_step,  # Safe timestep
            n_blocks=vmc_blocks,
            nsteps_per_block=vmc_steps_per_block,
            mode="langevin"
        )
        
        # Extract results
        mean_energy = vmc_results["mean_energy"]
        local_energies = vmc_results["local_energies"]
        acceptance = vmc_results["mean_acceptance"]
        
        energy_history.append(float(mean_energy))
        acceptance_history.append(float(acceptance))
        learning_rates.append(current_lr)
        
        # Update best parameters
        if mean_energy < best_energy:
            best_energy = mean_energy
            best_mo_coeff = current_mo_coeff
            best_configs = vmc_results["final_configs"]
            best_iteration = iteration + 1
        
        # 2. Compute wavefunction gradients (using parameter_gradient from determinants.py)
        psi_gradients = parameter_gradient(
            vmc_results["final_configs"],
            vmc_results["aovals"],
            vmc_results["dets"],
            det_coeff,
            current_mo_coeff,
            det_map,
            occup_hash,
            _nelec,
            vmc_results["inverse"]
        )
        
        # 3. Compute energy gradients
        energy_gradients = compute_energy_gradients(
            local_energies, psi_gradients, outlier_threshold=3.0
        )
        
        # Track gradient norms
        total_grad_norm = 0.0
        for grad in energy_gradients.values():
            total_grad_norm += jnp.linalg.norm(grad.flatten())**2
        total_grad_norm = jnp.sqrt(total_grad_norm)
        gradient_norms.append(float(total_grad_norm))
        
        # 4. Parameter update
        if method == "fisher":
            new_mo_coeff = fisher_information_update(
                current_mo_coeff, energy_gradients, psi_gradients,
                learning_rate=current_lr,
                regularization=safe_params["regularization"],
                max_gradient_norm=safe_params["gradient_clip"]
            )
        else:  # gradient descent
            new_mo_coeff, velocity = gradient_descent_update(
                current_mo_coeff, energy_gradients,
                learning_rate=current_lr,
                momentum=0.9,
                velocity=velocity,
                max_gradient_norm=safe_params["gradient_clip"]
            )
        
        # 5. Enforce orthogonality constraints (using orthogonalize_orbitals_s_based from mc.py)
        new_mo_coeff = list(new_mo_coeff)
        new_mo_coeff[0] = orthogonalize_orbitals_s_based(new_mo_coeff[0], overlap_matrix)
        new_mo_coeff[1] = orthogonalize_orbitals_s_based(new_mo_coeff[1], overlap_matrix)
        current_mo_coeff = tuple(new_mo_coeff)
        
        # 6. Adaptive learning rate adjustment
        current_lr = adapt_learning_rate(
            current_lr, energy_history,
            patience=safe_params["patience"], 
            factor=0.8, 
            min_lr=1e-6,
            max_lr=learning_rate * 2
        )
        
        # 7. Print progress
        if verbose:
            print(f"Iter {iteration + 1:3d}: "
                  f"E = {mean_energy:.8f} (Best: {best_energy:.8f}), "
                  f"|∇| = {total_grad_norm:.6f}, "
                  f"LR = {current_lr:.6f}, "
                  f"Acc = {acceptance:.3f}")
        
        # 8. Check convergence
        if check_convergence(energy_history, gradient_norms):
            if verbose:
                print(f"\nConverged after {iteration + 1} iterations!")
            break
        
        # 9. Update state
        current_configs = vmc_results["final_configs"]
        # Use updated key from vmc_results if available
    
    # Compile final results
    results = {
        "optimized_mo_coeff": current_mo_coeff,
        "best_mo_coeff": best_mo_coeff,
        "best_energy": best_energy,
        "best_iteration": best_iteration,
        "final_configs": current_configs,
        "best_configs": best_configs,
        "method_used": method,
        "final_learning_rate": current_lr,
        "converged": iteration < n_iterations - 1,
        "total_iterations": iteration + 1
    }
    
    if save_history:
        results.update({
            "energy_history": energy_history,
            "gradient_norms": gradient_norms,
            "acceptance_history": acceptance_history,
            "learning_rates": learning_rates
        })
    
    if verbose:
        print(f"\n=== Optimization Complete ===")
        print(f"Best energy: {best_energy:.8f} (iteration {best_iteration})")
        print(f"Final energy: {energy_history[-1]:.8f}")
        print(f"Total iterations: {results['total_iterations']}")
        print(f"Converged: {results['converged']}")
        print(f"Method: {method.upper()}")
    
    return results

# ===========================
# Convenience Functions
# ===========================

def compare_optimization_methods(
    mol, configs, mo_coeff, overlap_matrix, det_coeff, det_map, _nelec, occup_hash,
    atom_charges, atom_coords, key, get_phase,
    system_size: str = "medium",
    n_iterations: int = 30,
    verbose: bool = True
) -> Dict:
    """
    Compare Fisher Information and Gradient Descent methods
    
    Args:
        Same input parameters
        system_size: System size
        n_iterations: Number of iterations for comparison
        verbose: Whether to print output
        
    Returns:
        Dictionary comparing results of both methods
    """
    if verbose:
        print("=" * 60)
        print("OPTIMIZATION METHODS COMPARISON")
        print("=" * 60)
    
    # Fisher Information method
    if verbose:
        print("\n1. Fisher Information Method (Stochastic Reconfiguration)")
    
    results_fim = optimize_mo_coeff_improved(
        mol, configs, mo_coeff, overlap_matrix, det_coeff, det_map, _nelec, occup_hash,
        atom_charges, atom_coords, key, get_phase,
        n_iterations=n_iterations,
        method="fisher",
        system_size=system_size,
        verbose=verbose
    )
    
    # Gradient Descent method
    if verbose:
        print("\n2. Gradient Descent Method")
    
    results_gd = optimize_mo_coeff_improved(
        mol, configs, mo_coeff, overlap_matrix, det_coeff, det_map, _nelec, occup_hash,
        atom_charges, atom_coords, key, get_phase,
        n_iterations=n_iterations,
        method="gradient",
        system_size=system_size,
        verbose=verbose
    )
    
    # Compare results
    comparison = {
        "fisher_results": results_fim,
        "gradient_results": results_gd,
        "energy_improvement": {
            "fisher": results_fim["energy_history"][0] - results_fim["best_energy"],
            "gradient": results_gd["energy_history"][0] - results_gd["best_energy"]
        },
        "convergence": {
            "fisher": results_fim["converged"],
            "gradient": results_gd["converged"]
        },
        "final_energies": {
            "fisher": results_fim["best_energy"],
            "gradient": results_gd["best_energy"]
        }
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Fisher Information:")
        print(f"  Best Energy: {results_fim['best_energy']:.8f}")
        print(f"  Energy Improvement: {comparison['energy_improvement']['fisher']:.8f}")
        print(f"  Converged: {results_fim['converged']}")
        print(f"  Iterations: {results_fim['total_iterations']}")
        
        print(f"\nGradient Descent:")
        print(f"  Best Energy: {results_gd['best_energy']:.8f}")
        print(f"  Energy Improvement: {comparison['energy_improvement']['gradient']:.8f}")
        print(f"  Converged: {results_gd['converged']}")
        print(f"  Iterations: {results_gd['total_iterations']}")
        
        # Determine winner
        if results_fim['best_energy'] < results_gd['best_energy']:
            winner = "Fisher Information"
            energy_diff = results_gd['best_energy'] - results_fim['best_energy']
        else:
            winner = "Gradient Descent"
            energy_diff = results_fim['best_energy'] - results_gd['best_energy']
        
        print(f"\nWinner: {winner}")
        print(f"Energy difference: {energy_diff:.8f}")
    
    return comparison

def quick_fisher_optimization(
    mol, configs, mo_coeff, overlap_matrix, det_coeff, det_map, _nelec, occup_hash,
    atom_charges, atom_coords, key, get_phase,
    system_size: str = "medium"
) -> Dict:
    """
    Quick Fisher Information optimization (recommended method)
    
    Uses Fisher Information method with safe settings by default
    """
    return optimize_mo_coeff_improved(
        mol, configs, mo_coeff, overlap_matrix, det_coeff, det_map, _nelec, occup_hash,
        atom_charges, atom_coords, key, get_phase,
        method="fisher",
        system_size=system_size,
        verbose=True
    )

def conservative_optimization(
    mol, configs, mo_coeff, overlap_matrix, det_coeff, det_map, _nelec, occup_hash,
    atom_charges, atom_coords, key, get_phase
) -> Dict:
    """
    Very conservative optimization (for unstable systems)
    
    Uses small learning rates and extensive sampling for stability
    """
    return optimize_mo_coeff_improved(
        mol, configs, mo_coeff, overlap_matrix, det_coeff, det_map, _nelec, occup_hash,
        atom_charges, atom_coords, key, get_phase,
        method="fisher",
        initial_learning_rate=0.001,  # Very small learning rate
        vmc_blocks=15,                # More blocks
        vmc_steps_per_block=60,       # More steps
        n_iterations=100,             # Sufficient iterations
        system_size="large",          # Use conservative settings
        verbose=True
    )

# ===========================
# Diagnostic and Analysis Functions
# ===========================

def analyze_optimization_stability(results: Dict, plot: bool = True) -> Dict:
    """
    Analyze optimization stability
    
    Args:
        results: Results from optimize_mo_coeff_improved
        plot: Whether to generate plots
        
    Returns:
        Analysis results dictionary
    """
    analysis = {}
    
    if "energy_history" not in results:
        print("Warning: No energy history available for analysis")
        return analysis
    
    energy_history = np.array(results["energy_history"])
    gradient_norms = np.array(results.get("gradient_norms", []))
    
    # Energy stability analysis
    analysis["energy_stats"] = {
        "initial_energy": float(energy_history[0]),
        "final_energy": float(energy_history[-1]),
        "best_energy": float(results["best_energy"]),
        "total_improvement": float(energy_history[0] - results["best_energy"]),
        "final_improvement": float(energy_history[0] - energy_history[-1])
    }
    
    # Convergence analysis
    if len(energy_history) > 10:
        last_10_std = np.std(energy_history[-10:])
        analysis["convergence_stats"] = {
            "converged": results["converged"],
            "last_10_std": float(last_10_std),
            "is_stable": last_10_std < 1e-6
        }
    
    # Gradient analysis
    if len(gradient_norms) > 0:
        analysis["gradient_stats"] = {
            "initial_gradient_norm": float(gradient_norms[0]),
            "final_gradient_norm": float(gradient_norms[-1]),
            "max_gradient_norm": float(np.max(gradient_norms)),
            "gradient_reduction": float(gradient_norms[0] / (gradient_norms[-1] + 1e-10))
        }
    
    # Learning rate analysis
    if "learning_rates" in results:
        lr_history = np.array(results["learning_rates"])
        analysis["learning_rate_stats"] = {
            "initial_lr": float(lr_history[0]),
            "final_lr": float(lr_history[-1]),
            "min_lr": float(np.min(lr_history)),
            "max_lr": float(np.max(lr_history)),
            "lr_adaptations": int(np.sum(np.diff(lr_history) != 0))
        }
    
    # Acceptance rate analysis
    if "acceptance_history" in results:
        acc_history = np.array(results["acceptance_history"])
        analysis["acceptance_stats"] = {
            "mean_acceptance": float(np.mean(acc_history)),
            "min_acceptance": float(np.min(acc_history)),
            "max_acceptance": float(np.max(acc_history)),
            "acceptance_stable": float(np.std(acc_history) < 0.1)
        }
    
    # Generate recommendations
    recommendations = []
    
    if analysis.get("convergence_stats", {}).get("is_stable", True) == False:
        recommendations.append("Increase number of iterations for better convergence")
    
    if analysis.get("gradient_stats", {}).get("gradient_reduction", 10) < 2:
        recommendations.append("Gradient reduction is poor - consider different method or parameters")
    
    if analysis.get("acceptance_stats", {}).get("mean_acceptance", 0.5) < 0.3:
        recommendations.append("Acceptance rate too low - decrease timestep")
    elif analysis.get("acceptance_stats", {}).get("mean_acceptance", 0.5) > 0.7:
        recommendations.append("Acceptance rate too high - increase timestep")
    
    analysis["recommendations"] = recommendations
    
    # Generate plots
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Energy convergence
            axes[0, 0].plot(energy_history, 'o-', markersize=4)
            axes[0, 0].axhline(y=results["best_energy"], color='r', linestyle='--', label='Best')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Energy')
            axes[0, 0].set_title('Energy Convergence')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Gradient norm
            if len(gradient_norms) > 0:
                axes[0, 1].semilogy(gradient_norms, 'o-', markersize=4)
                axes[0, 1].set_xlabel('Iteration')
                axes[0, 1].set_ylabel('Gradient Norm (log scale)')
                axes[0, 1].set_title('Gradient Norm Evolution')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Learning rate
            if "learning_rates" in results:
                axes[1, 0].plot(results["learning_rates"], 'o-', markersize=4)
                axes[1, 0].set_xlabel('Iteration')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_title('Learning Rate Adaptation')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Acceptance rate
            if "acceptance_history" in results:
                axes[1, 1].plot(results["acceptance_history"], 'o-', markersize=4)
                axes[1, 1].axhline(y=0.5, color='r', linestyle='--', label='Target')
                axes[1, 1].set_xlabel('Iteration')
                axes[1, 1].set_ylabel('Acceptance Rate')
                axes[1, 1].set_title('Acceptance Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('optimization_analysis.png', dpi=300, bbox_inches='tight')
            print("Analysis plot saved as 'optimization_analysis.png'")
            
        except ImportError:
            print("Matplotlib not available - skipping plots")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    return analysis

def print_optimization_summary(results: Dict):
    """
    Print optimization results summary
    """
    print("\n" + "=" * 50)
    print("OPTIMIZATION SUMMARY")
    print("=" * 50)
    
    print(f"Method: {results['method_used'].upper()}")
    print(f"Total iterations: {results['total_iterations']}")
    print(f"Converged: {'Yes' if results['converged'] else 'No'}")
    
    if "energy_history" in results:
        initial_energy = results['energy_history'][0]
        final_energy = results['energy_history'][-1]
        print(f"\nEnergy Results:")
        print(f"  Initial energy: {initial_energy:.8f}")
        print(f"  Final energy: {final_energy:.8f}")
        print(f"  Best energy: {results['best_energy']:.8f} (iteration {results['best_iteration']})")
        print(f"  Total improvement: {initial_energy - results['best_energy']:.8f}")
    
    if "gradient_norms" in results:
        print(f"\nGradient Norms:")
        print(f"  Initial: {results['gradient_norms'][0]:.6f}")
        print(f"  Final: {results['gradient_norms'][-1]:.6f}")
        print(f"  Reduction factor: {results['gradient_norms'][0] / (results['gradient_norms'][-1] + 1e-10):.1f}")
    
    if "acceptance_history" in results:
        mean_acc = np.mean(results['acceptance_history'])
        print(f"\nAcceptance Rate: {mean_acc:.3f}")
    
    print(f"\nFinal learning rate: {results['final_learning_rate']:.6f}")
    
def test_parameter_update():
    """
    Simple test of parameter update functions
    """
    print("Testing parameter update functions...")
    
    # Generate dummy data
    nconf, nao, nmo = 100, 10, 5
    
    # Dummy MO coefficients
    mo_coeff = (
        jnp.array(np.random.randn(nao, nmo)),
        jnp.array(np.random.randn(nao, nmo))
    )
    
    # Dummy gradients
    psi_gradients = {
        "mo_coeff_alpha": jnp.array(np.random.randn(nconf, nao, nmo)),
        "mo_coeff_beta": jnp.array(np.random.randn(nconf, nao, nmo))
    }
    
    energy_gradients = {
        "mo_coeff_alpha": jnp.array(np.random.randn(nao, nmo)) * 0.01,
        "mo_coeff_beta": jnp.array(np.random.randn(nao, nmo)) * 0.01
    }
    
    print("✓ Test data generated")
    
    # Test Fisher Information
    try:
        fim_inv = compute_fisher_information_matrix(psi_gradients)
        print("✓ Fisher Information Matrix computed")
        
        new_mo_coeff = fisher_information_update(
            mo_coeff, energy_gradients, psi_gradients
        )
        print("✓ Fisher Information update successful")
    except Exception as e:
        print(f"✗ Fisher Information test failed: {e}")
    
    # Test Gradient Descent
    try:
        new_mo_coeff, velocity = gradient_descent_update(
            mo_coeff, energy_gradients
        )
        print("✓ Gradient Descent update successful")
    except Exception as e:
        print(f"✗ Gradient Descent test failed: {e}")
    
    print("Parameter update tests completed!")