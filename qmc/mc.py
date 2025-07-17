import jax
import jax.numpy as jnp
from jax import random

from typing import Dict, Any, Tuple, List, Optional
import time
import pickle
import os

from typing import Tuple, Optional

from pyscf import gto, scf, mcscf
import pyqmc.api as pyq

from qmc.pyscftools import  orbital_evaluator_from_pyscf
from qmc.orbitals import *
from qmc.determinants import *
from qmc.extract import jax_ee_energy, jax_ei_energy, jax_ii_energy, compute_potential_energy, jax_kinetic_energy

@jax.jit
def limdrift(g, cutoff=1):
    """
    Limit a vector to have a maximum magnitude of cutoff while maintaining direction

    :parameter g: a [nconf,ndim] vector
    :parameter cutoff: the maximum magnitude
    :returns: The vector with the cutoff applied.
    """
    tot = jnp.linalg.norm(g, axis=1)
    mask = tot > cutoff
    scaling = jnp.where(mask, cutoff / tot, 1.0)
    return g * scaling[:, jnp.newaxis]

@jax.jit
def compute_transition_probability(gauss, grad, new_grad, tstep):
    """
    Compute the transition probability ratio for the Metropolis-Hastings algorithm.
    
    Parameters:
    gauss: Gaussian random displacement
    grad: Current position gradient
    new_grad: Proposed position gradient
    tstep: Time step size
    
    Returns:
    Transition probability ratio
    """
    forward = jnp.sum(gauss**2, axis=1)
    backward = jnp.sum((gauss + tstep * (grad + new_grad))**2, axis=1)
    t_prob = jnp.exp(1 / (2 * tstep) * (forward - backward))
    return t_prob

@jax.jit
def update_coordinates(coords, e, accept, newcoorde):
    """
    Update the electron coordinates based on acceptance using JAX-friendly operations.
    """
    # Create a condition array that's True only for the accepted moves of electron e
    # We'll use broadcasting to make a 3D mask matching coords dimensions
    mask = jnp.zeros_like(coords, dtype=bool)
    mask = mask.at[:, e, :].set(accept[:, None])  # Broadcast accept across coordinates
    
    # Create the updated array with new coordinates for electron e
    new_coords = coords.at[:, e, :].set(newcoorde)
    
    # Use where to conditionally select between original and new coordinates
    return jnp.where(mask, new_coords, coords)

def mc_step_langevin(coords, mol, mo_coeff, det_coeff, det_map, _nelec, occup_hash,
                     get_phase, key, aovals, dets, inverse, tstep=0.5):
    """
    Run a quantum Monte Carlo simulation using regular Python loops with JIT-compiled subfunctions.
    
    Parameters:
    coords: Initial electron coordinates [nconf, nelec, 3]
    mol: PySCF molecule object (cannot be used in JIT-compiled functions)
    mo_coeff: Molecular orbital coefficients
    det_coeff: Determinant coefficients
    det_map: Determinant mapping
    _nelec: Electron count information
    occup_hash: Occupation hash
    get_phase: Function to calculate the phase
    equilibration_step: Number of equilibration steps
    tstep: Time step size
    seed: Random seed
    
    Returns:
    Final coordinates, acceptance ratio, and elapsed time
    """
    nconf, nelec, _ = coords.shape
    step_acc = 0.0
    
    for e in range(nelec):
        key, gauss_key = random.split(key)
        
        # Calculate gradient at current position
        g, _, _ = gradient_value(mol, e, coords[:, e, :], dets, inverse, mo_coeff,
                               det_coeff, det_map, _nelec, occup_hash)
        
        # Apply drift limitation
        grad = limdrift(jnp.real(g.T))
        
        # Generate random displacement
        gauss = random.normal(gauss_key, shape=(nconf, 3)) * jnp.sqrt(tstep)
        
        # Propose new position
        newcoorde = coords[:, e, :] + gauss + grad * tstep
        
        #calculate gradient at proposed position
        g, new_val, saved = gradient_value(mol, e, newcoorde, dets, inverse, mo_coeff,
                                         det_coeff, det_map, _nelec, occup_hash)
        
        # Apply drift limitation to new gradient
        new_grad = limdrift(jnp.real(g.T))
        
        # Compute transition probability
        t_prob = compute_transition_probability(gauss, grad, new_grad, tstep)

        # Calculate acceptance ratio
        ratio = jnp.abs(new_val) ** 2 * t_prob
        
        # Metropolis acceptance/rejection
        key, uniform_key = random.split(key)
        uniform_rand = random.uniform(uniform_key, shape=(nconf,))
        accept = ratio > uniform_rand
        
        coords = update_coordinates(coords, e, accept, newcoorde)
        
        aovals, dets, inverse = sherman_morrison(e, newcoorde, coords, accept, aovals, 
                                                 saved, get_phase, dets, inverse, 
                                                 mo_coeff, occup_hash, _nelec)

        step_acc += jnp.mean(accept) / nelec
        
    return coords, aovals, dets, inverse, step_acc, key

def mc_step_symmetric(coords, mol, mo_coeff, det_coeff, det_map, _nelec, occup_hash,
                      get_phase, key, aovals, dets, inverse, tstep=0.5):
    """
    Run a quantum Monte Carlo simulation using regular Python loops with JIT-compiled subfunctions.
    
    Parameters:
    coords: Initial electron coordinates [nconf, nelec, 3]
    mol: PySCF molecule object (cannot be used in JIT-compiled functions)
    mo_coeff: Molecular orbital coefficients
    det_coeff: Determinant coefficients
    det_map: Determinant mapping
    _nelec: Electron count information
    occup_hash: Occupation hash
    get_phase: Function to calculate the phase
    equilibration_step: Number of equilibration steps
    tstep: Time step size
    seed: Random seed
    
    Returns:
    Final coordinates, acceptance ratio, and elapsed  time
    """
    nconf, nelec, _ = coords.shape
    step_acc = 0.0
    
    for e in range(nelec):
        key, gauss_key = random.split(key)
        
        # Current wavefunction values
        _, old_val, _ = gradient_value(mol, e, coords[:, e, :], dets, inverse, mo_coeff,
                                       det_coeff, det_map, _nelec, occup_hash)
        
        # Generate random displacement (symmetric proposal)
        gauss = random.normal(gauss_key, shape=(nconf, 3)) * jnp.sqrt(tstep)
        
        # Propose new position
        newcoorde = coords[:, e, :] + gauss
        
        # New wavefunction value at propose position
        _, new_val, saved = gradient_value(mol, e, newcoorde, dets, inverse, mo_coeff,
                                     det_coeff, det_map, _nelec, occup_hash)
        
        # Calculate acceptance ratio
        ratio = jnp.abs(new_val / old_val) ** 2
        
        # Metropolis acceptance/rejection
        key, uniform_key = random.split(key)
        uniform_rand = random.uniform(uniform_key, shape=(nconf,))
        accept = ratio > uniform_rand
        
        # Update coordinates for accepted moves
        coords = update_coordinates(coords, e, accept, newcoorde)
        
        # Update wavefunction values
        aovals, dets, inverse = sherman_morrison(e, newcoorde, coords, accept, aovals, 
                                                 saved, get_phase, dets, inverse, 
                                                 mo_coeff, occup_hash, _nelec)
        
        step_acc += jnp.mean(accept) / nelec
        
    return coords, aovals, dets, inverse, step_acc, key

def vmc(
   coords, 
   mol,
   atom_charges,
   atom_coords,
   mo_coeff, 
   det_coeff, 
   det_map, 
   _nelec, 
   occup_hash,
   get_phase, 
   key, 
   equilibration_step=500, 
   tstep=0.5,
   n_blocks=10,
   nsteps_per_block=10,
   blockoffset=0,
   mode="langevin",
   compute_autocorr=True,
   autocorr_length=20,
   max_age_threshold=20,
):
    
    """
    Run Variational Monte Carlo simulation with enhanced analysis.
    
    Parameters:
    coords: Initial electron coordinates [nconf, nelec, 3]
    mol: PySCF molecule object
    atom_charges: Atomic charges array
    atom_coords: Atomic coordinates array
    mo_coeff: Molecular orbital coefficients
    det_coeff: Determinant coefficients
    det_map: Determinant mapping
    _nelec: Electron count information
    occup_hash: Occupation hash
    get_phase: Function to calculate the phase
    key: JAX random key
    equilibration_step: Number of equilibration steps
    tstep: Time step size
    n_blocks: Number of blocks for statistics
    nsteps_per_block: Number of steps per block
    blockoffset: Block offset for resuming simulations
    mode: "langevin" or "symmetric" for the type of MC step
    compute_autocorr: Whether to compute autocorrelation
    autocorr_length: Maximum lag for autocorrelation
    
    Returns:
    Dictionary containing energy statistics and simulation data
    """
    
    nconf, nelec, _ = coords.shape
    total_acc = 0
    
    # Initialize electron ages (tracks how many steps each electron has gone without moving)
    # Start with zero for all electrons in all configurations
    walker_ages = jnp.zeros(nconf, dtype = jnp.int32)
    
    age_histogram = jnp.zeros((nconf, nelec), dtype = jnp.int32)
        
    # Track maximum age reached by each electron
    max_ages_reached = 0    
    #  Initialize wavefunction
    aovals, dets, inverse = recompute(mol, coords, mo_coeff, _nelec, occup_hash)

    # Save initial positions for MSD calculation
    initial_coords = coords.copy()
    
    # Initialize list for analysis
    all_energies = []
    all_corrds = []
    all_acc = []
    
    # print(f"Starting equilibriation : {equilibration_step} steps")
    
    for step in range(equilibration_step):
        
        if mode == "langevin":
            coords, aovals, dets, inverse, step_acc, key = mc_step_langevin(coords, mol, mo_coeff, det_coeff, det_map, _nelec, 
                                                               occup_hash, get_phase, key, aovals, dets, inverse, tstep)
            
        else:
            coords, aovals, dets, inverse, step_acc, key = mc_step_symmetric(coords, mol, mo_coeff, det_coeff, det_map, _nelec, 
                                                                occup_hash, get_phase, key, aovals, dets, inverse, tstep)
            
        total_acc += step_acc
        
        # Update electron ages - Here we should update based on which electrons moved
        # But we don't have acceptance information per electron at this level
        # This will be properly tracked in the production phase
        
    # if (step + 1) % 100 == 0:
    #     print(f"Step {step + 1} Acceptance: {total_acc / (step + 1)}")
        
    block_energies = {
        "kinetic" : [],
        "ee" : [],
        "ei" : [],
        "total" : [],
        "acceptance" : [],
        "cumulative_energy" : [],
        "cumulative_variance" : [],
        "cumulative_error" : [],
        "cumulative_acceptance" : [],
        "msd" : {str(i) : [] for i in range(nelec)},
        "mean_msd" : [],
        "autocorr" : {str(i) : [] for i in range(nelec) if compute_autocorr},
        "max_ages" : [],  # Track maximum ages of electrons,
        "mean_ages" : [],
        "age_histogram" : [],
        "stuck_walker_count" : [],  # Track indices of stuck configurations
        "stuck_walker_indices" : [],
        "std_error_medium" : [],
    }        
    
    block_coords = []
    
    # print(f"Starting production : {n_blocks} blocks of {nsteps_per_block} steps")
    
    # nuclear - nuclear repulsion energy
    ii_energy = jax_ii_energy(mol)
    
    # All production phase energy values for cumulative statistics
    all_total_energies = []
    
    # All acceptance ratio for cumulative statistics
    all_acceptance = []
    
    # for MSD and autocorrelation
    step_coords = []
    step_coords.append(coords)
    
    for block in range(n_blocks):
        
        block_acc = 0.0        
        block_kinetic = []
        block_ee = []
        block_ei = []
        block_total = []
        
        aovals, dets, inverse = recompute(mol, coords, mo_coeff, _nelec, occup_hash)
        
        for step in range(nsteps_per_block):
            # Reset electron_ages for electrons that are going to be moved
            # Store original coordinates to detect movement later
            prev_coords = coords.copy()
            
            if mode == "langevin":
                coords, aovals, dets, inverse, step_acc, key = mc_step_langevin(coords, mol, mo_coeff, det_coeff, det_map, _nelec, 
                                                                   occup_hash, get_phase, key, aovals, dets, inverse, tstep)
            else:
                coords, aovals, dets, inverse, step_acc, key = mc_step_symmetric(coords, mol, mo_coeff, det_coeff, det_map, _nelec, 
                                                                    occup_hash, get_phase, key, aovals, dets, inverse, tstep)
            
            # Update electron ages based on movement
            # For each electron and configuration, check if the coordinates changed
            
            moved = jnp.zeros(nconf, dtype = bool)
            
            for e in range(nelec):
                # Calculate if electron moved (any change in x, y, or z coordinates)
                # True if moved is ture or jnp.any is true
                moved = moved | jnp.any(jnp.abs(coords[:, e, :] - prev_coords[:, e, :]) > 1e-10, axis=1)
     
            walker_ages = jnp.where(moved, 0, walker_ages + 1)
            max_ages_reached = jnp.max(walker_ages)
            
            age_histogram = jnp.zeros(100, dtype = jnp.int32)
            for age in range(min(100, jnp.max(walker_ages) + 1)):
                age_histogram = age_histogram.at[age].set(jnp.sum(walker_ages == age))
            
            block_acc += step_acc
            
            step_coords.append(coords)
            # Compute energy components
            ee = jax_ee_energy(coords)
            ei = jax_ei_energy(coords, atom_charges, atom_coords)
            ke = jax_kinetic_energy(coords, mol, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash)[0]
            total_energy = ee + ei + ke + ii_energy
            
            block_kinetic.append(ke)
            block_ee.append(ee)
            block_ei.append(ei)
            block_total.append(total_energy)
            all_total_energies.append(total_energy)
            
            block_coords.append(coords)
        
        
        block_avg_acc = block_acc / nsteps_per_block
        
        mean_age = jnp.mean(walker_ages)
        stuck_walkers = jnp.where(walker_ages > max_age_threshold)[0]
        stuck_count = len(stuck_walkers)
        
        block_energies["max_ages"].append(max_ages_reached)
        block_energies["mean_ages"].append(mean_age)
        block_energies["age_histogram"].append(age_histogram)
        block_energies["stuck_walker_count"].append(stuck_count)
        block_energies["stuck_walker_indices"].append(stuck_walkers)
        
        # Calculate energies
        block_energies["kinetic"].append(jnp.mean(jnp.array(block_kinetic)))
        block_energies["ee"].append(jnp.mean(jnp.array(block_ee)))
        block_energies["ei"].append(jnp.mean(jnp.array(block_ei)))
        block_energies["total"].append(jnp.mean(jnp.array(block_total)))
        block_energies["acceptance"].append(block_avg_acc)
        
        cumulative_energy = jnp.array(all_total_energies)
        all_acceptance.append(block_avg_acc)
        block_energies["cumulative_energy"].append(jnp.mean(cumulative_energy))
        block_energies["cumulative_variance"].append(jnp.var(cumulative_energy))
        total_samples_so_far = nconf * (block + 1)
        block_energies["cumulative_error"].append(jnp.std(cumulative_energy) / jnp.sqrt(total_samples_so_far))
        block_energies["cumulative_acceptance"].append(jnp.mean(jnp.array(all_acceptance)))
        
        std_error = jnp.std(jnp.array(block_total)) / jnp.sqrt(nconf * nsteps_per_block)
        block_energies["std_error_medium"].append(std_error)       
        block_coords.append(coords)
        
        print(f"Block {block}/{n_blocks}, Energy: {block_energies['total'][-1]:.6f} ± "
              f"{std_error:.6f},"
              f"Acceptance: {block_energies['acceptance'][-1]:.4f}, "
              f"Max Age: {max_ages_reached}, "
              f"Stuck: {stuck_count}, ")
    
    mean_energy = jnp.mean(jnp.array(block_energies["total"]))
    total_samples = nconf * n_blocks
    std_error = jnp.std(jnp.array(block_energies["total"])) / jnp.sqrt(total_samples)

    step_coords_array = jnp.array(step_coords)
    
    for e in range(nelec):
        
        # Extract coordinate for electron e across all steps
        electron_coords = step_coords_array[:, :, e, :] # [nsteps, nconf, 3]
        
        initial_pos = step_coords_array[0, :, e, :] # [nconf, 3]
        
        disp_sq = jnp.sum((electron_coords - initial_pos[jnp.newaxis, :, :])**2, axis=2) # [steps, nconf]
        
        msd = jnp.mean(disp_sq, axis = 1) # [steps]
        
        block_energies["msd"][str(e)] = msd
        
        
    all_msd = jnp.array([block_energies["msd"][str(e)] for e in range(nelec)])  # [nelec, steps]
    block_energies["mean_msd"] = jnp.mean(all_msd, axis=0)  # [steps]

    # Calculate autocorrelation if requested
    if compute_autocorr:
        for e in range(nelec):
            # Extract coordinates for electron e across all steps and first configuration
            electron_coords = step_coords_array[:, 0, e, :]  # [steps, 3]
            
            # Calculate autocorrelation up to specified lag
            max_lag = min(autocorr_length, len(electron_coords) - 1)
            autocorr = []
            
            for lag in range(max_lag + 1):
                # Calculate correlation between positions at different time lags
                if lag == 0:
                    # At lag 0, correlation is 1 by definition
                    autocorr.append(1.0)
                else:
                    # Calculate correlation for this lag
                    x_t = electron_coords[:-lag]
                    x_t_plus_lag = electron_coords[lag:]
                    
                    # Calculate dot product between vectors at different times
                    dot_products = jnp.sum(x_t * x_t_plus_lag, axis=1)
                    
                    # Calculate norms
                    norm_t = jnp.sqrt(jnp.sum(x_t ** 2, axis=1))
                    norm_t_plus_lag = jnp.sqrt(jnp.sum(x_t_plus_lag ** 2, axis=1))
                    
                    # Calculate correlation
                    correlation = jnp.mean(dot_products / (norm_t * norm_t_plus_lag))
                    autocorr.append(float(correlation))
            
            block_energies["autocorr"][str(e)] = jnp.array(autocorr)
    
    # stuck config
    
    
    # Add summary statistics
    block_energies["mean_energy"] = mean_energy
    block_energies["std_error"] = std_error
    block_energies["mean_acceptance"] = jnp.mean(jnp.array(block_energies["acceptance"]))
    block_energies["final_coords"] = coords
    block_energies["block_coords"] = block_coords
    block_energies["final_max_age"] = max_ages_reached
    block_energies["final_mean_age"] = jnp.mean(walker_ages)
    block_energies["final_age_histogram"] = age_histogram
    block_energies["final_stuck_count"] = jnp.sum(walker_ages > max_age_threshold)
    block_energies["final_stuck_indices"] = jnp.where(walker_ages > max_age_threshold)[0]
        
    # print(f"\nFinal energy: {mean_energy:.6f} ± {std_error:.6f}")
    # print(f"Average acceptance ratio: {block_energies['mean_acceptance']:.4f}")
    # print(f"Final maximum electron ages: {max_ages}")
    
    return block_energies, all_total_energies

def vmc_simple_improved(
    coords, mol, atom_charges, atom_coords, mo_coeff, det_coeff, det_map, 
    _nelec, occup_hash, get_phase, key, 
    equilibration_step=200, tstep=0.3, n_blocks=5, nsteps_per_block=50, mode="langevin"
):
    """
    Improved VMC simulation with minimal JIT usage
    
    Returns properly structured local energies for parameter optimization
    """
    nconf, nelec, _ = coords.shape
    
    # Initialize wavefunction (no JIT - contains mol object)
    aovals, dets, inverse = recompute(mol, coords, mo_coeff, _nelec, occup_hash)
    
    # Equilibration (no JIT - uses mol object)
    total_equil_acc = 0.0
    for step in range(equilibration_step):
        if mode == "langevin":
            coords, aovals, dets, inverse, step_acc, key = mc_step_langevin(
                coords, mol, mo_coeff, det_coeff, det_map, _nelec, 
                occup_hash, get_phase, key, aovals, dets, inverse, tstep
            )
        else:
            coords, aovals, dets, inverse, step_acc, key = mc_step_symmetric(
                coords, mol, mo_coeff, det_coeff, det_map, _nelec, 
                occup_hash, get_phase, key, aovals, dets, inverse, tstep
            )
        total_equil_acc += step_acc
    
    equil_acceptance = total_equil_acc / equilibration_step
    
    # Nuclear-nuclear energy (can use JIT)
    ii_energy = jax_ii_energy(mol)
    
    # Block-wise sampling
    block_energies = []
    block_acceptances = []
    all_step_energies = []  # Store all individual step energies
    final_local_energies = None  # Store final step local energies
    
    for block in range(n_blocks):
        block_acc = 0.0
        block_energy_list = []
        
        # Recompute wavefunction (no JIT)
        aovals, dets, inverse = recompute(mol, coords, mo_coeff, _nelec, occup_hash)
        
        for step in range(nsteps_per_block):
            # MC step (no JIT - uses mol object)
            if mode == "langevin":
                coords, aovals, dets, inverse, step_acc, key = mc_step_langevin(
                    coords, mol, mo_coeff, det_coeff, det_map, _nelec, 
                    occup_hash, get_phase, key, aovals, dets, inverse, tstep
                )
            else:
                coords, aovals, dets, inverse, step_acc, key = mc_step_symmetric(
                    coords, mol, mo_coeff, det_coeff, det_map, _nelec, 
                    occup_hash, get_phase, key, aovals, dets, inverse, tstep
                )
            
            block_acc += step_acc
            
            # Energy calculation (partial JIT usage)
            ee = jax_ee_energy(coords)  # Already JIT compiled, returns [nconf] array
            ei = jax_ei_energy(coords, atom_charges, atom_coords)  # Already JIT compiled, returns [nconf] array
            ke, _ = jax_kinetic_energy(  # No JIT - uses mol object, returns [nconf] array
                coords, mol, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash
            )
            
            # Total energy per configuration
            total_energy_per_config = ee + ei + ke + ii_energy  # Shape: [nconf]
            
            # Store individual configuration energies for this step
            all_step_energies.extend(total_energy_per_config.tolist())
            
            # Average energy for this step (for block statistics)
            step_mean_energy = jnp.mean(total_energy_per_config)
            block_energy_list.append(step_mean_energy)
            
            # Keep final step local energies for parameter gradients
            if block == n_blocks - 1 and step == nsteps_per_block - 1:
                final_local_energies = total_energy_per_config
        
        # Block statistics (Python - simple operations)
        block_mean = jnp.mean(jnp.array(block_energy_list))
        block_energies.append(block_mean)
        block_acceptances.append(block_acc / nsteps_per_block)
    
    # Overall statistics (Python - simple operations)
    all_step_energies = jnp.array(all_step_energies)
    mean_energy = jnp.mean(all_step_energies)
    std_energy = jnp.std(all_step_energies)
    
    block_energies = jnp.array(block_energies)
    std_error_blocks = jnp.std(block_energies) / jnp.sqrt(n_blocks)
    std_error_total = std_energy / jnp.sqrt(len(all_step_energies))
    
    # Ensure we have final local energies
    if final_local_energies is None:
        # Compute final local energies if not captured
        ee = jax_ee_energy(coords)
        ei = jax_ei_energy(coords, atom_charges, atom_coords)
        ke, _ = jax_kinetic_energy(
            coords, mol, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash
        )
        final_local_energies = ee + ei + ke + ii_energy
    
    return {
        "local_energies": final_local_energies,  # ✅ Final step local energies [nconf] for parameter gradients
        "all_energies": all_step_energies,       # ✅ All individual configuration energies from all steps
        "block_energies": block_energies,        # ✅ Average energy per block
        "mean_energy": mean_energy,              # ✅ Overall mean energy
        "std_error": std_error_blocks,           # ✅ Block-based standard error
        "std_error_total": std_error_total,      # ✅ Total standard error
        "block_acceptances": jnp.array(block_acceptances),
        "mean_acceptance": jnp.mean(jnp.array(block_acceptances)),
        "equil_acceptance": equil_acceptance,
        "final_configs": coords,                 # ✅ Final configurations for parameter gradients
        "aovals": aovals,                        # ✅ Final AO values for parameter gradients
        "dets": dets,                           # ✅ Final determinants for parameter gradients
        "inverse": inverse,               
        "n_blocks": n_blocks,
        "nsteps_per_block": nsteps_per_block,
        "total_samples": len(all_step_energies)
    }
    
@jax.jit
def orthogonalize_orbitals_s_based(mo_matrix, S):
    """
    PySCF의 중첩 행렬 S를 고려하여 Löwdin 직교화를 수행합니다.
    
    Args:
        mo_matrix: 분자 궤도 계수 행렬 [n_ao, n_mo]
        S: AO 중첩 행렬 [n_ao, n_ao]
    
    Returns:
        직교화된 분자 궤도 계수 행렬
    """
    # C^T S C 계산
    CtSC = jnp.dot(mo_matrix.T, jnp.dot(S, mo_matrix))
    
    # (C^T S C)^{-1/2} 계산
    eigvals, eigvecs = jnp.linalg.eigh(CtSC)
    eigvals = jnp.maximum(eigvals, 1e-10)  # 수치적 안정성
    sqrt_inv_eigvals = 1.0 / jnp.sqrt(eigvals)
    CtSC_inv_sqrt = jnp.dot(eigvecs, jnp.dot(jnp.diag(sqrt_inv_eigvals), eigvecs.T))
    
    # C' = C (C^T S C)^{-1/2}
    mo_matrix_ortho = jnp.dot(mo_matrix, CtSC_inv_sqrt)
    
    return mo_matrix_ortho

def optimize_mo_coeff(mol, configs, mo_coeff, overlap_matrix, det_coeff, det_map, _nelec, occup_hash,
                     atom_charges, atom_coords, key, get_phase, n_iterations=100, learning_rate=0.01):
    """
    vmc_simple을 사용하여 분자 궤도 계수를 최적화하며, 가장 낮은 에너지를 저장합니다.
    
    Args:
        mol: PySCF 분자 객체
        configs: 초기 전자 좌표 [nconf, nelec, 3]
        mo_coeff: 초기 분자 궤도 계수
        overlap_matrix: AO 중첩 행렬 [n_ao, n_ao]
        det_coeff: 행렬식 계수
        det_map: 행렬식 맵핑
        _nelec: 전자 수 정보
        occup_hash: 점유 해시
        atom_charges: 원자 전하 배열
        atom_coords: 원자 좌표 배열
        key: JAX 랜덤 키
        get_phase: 위상 계산 함수
        n_iterations: 최적화 반복 횟수
        learning_rate: 학습률
    
    Returns:
        dict: 최적화된 MO 계수, 최종 전자 구성, 에너지 리스트, 최소 에너지 정보
    """
    # 해시 가능한 형식으로 occup_hash 변환
    if not isinstance(occup_hash[0], tuple):
        occup_hash = (tuple(occup_hash[0]), tuple(occup_hash[1]))    
    
    current_mo_coeff = mo_coeff
    energies = []
    
    # 최소 에너지 추적 변수
    min_energy = float('inf')
    min_energy_mo_coeff = current_mo_coeff
    min_energy_configs = configs
    min_energy_iteration = 0
    
    # 전체 최적화 과정
    for iteration in range(n_iterations):
        # 1. 간소화된 VMC 시뮬레이션 실행
        vmc_results = vmc_simple(
            configs, 
            mol,
            atom_charges,
            atom_coords,
            current_mo_coeff, 
            det_coeff, 
            det_map, 
            _nelec, 
            occup_hash,
            get_phase,
            key,
            equilibration_step=500,
            tstep=2.116,
            nsteps=100,
            mode="langevin"
        )
        
        # 샘플링된 구성 및 필요한 값들 추출
        sampled_configs = vmc_results["final_configs"]
        local_energies = vmc_results["local_energies"]
        aovals = vmc_results["aovals"]
        dets = vmc_results["dets"]
        inverse = vmc_results["inverse"]
        
        # 2. 파동함수 그래디언트 계산
        psi_gradients = parameter_gradient(
            sampled_configs, aovals, dets, det_coeff, 
            current_mo_coeff, det_map, occup_hash, _nelec, inverse
        )
        
        # 3. 에너지 그래디언트 계산 (변분 원리 공식)
        energy_mean = jnp.mean(local_energies)
        energy_gradients = {}
        
        energies.append(float(energy_mean))
        
        # 최소 에너지 갱신
        if energy_mean < min_energy:
            min_energy = energy_mean
            min_energy_mo_coeff = current_mo_coeff
            min_energy_configs = sampled_configs
            min_energy_iteration = iteration + 1
        
        for param_name, psi_grad in psi_gradients.items():
            # 로컬 에너지와 파동함수 그래디언트의 곱에 대한 기대값
            energy_psi_grad_mean = jnp.mean(local_energies[:, jnp.newaxis, jnp.newaxis] * psi_grad, axis=0)
            
            # 파동함수 그래디언트의 기대값
            psi_grad_mean = jnp.mean(psi_grad, axis=0)
            
            # 에너지 그래디언트 계산
            energy_gradients[param_name] = 2.0 * (energy_psi_grad_mean - energy_mean * psi_grad_mean)
        
        # 4. 파라미터 업데이트
        new_mo_coeff = list(current_mo_coeff)
        
        if "mo_coeff_alpha" in energy_gradients:
            new_mo_coeff[0] = new_mo_coeff[0] - learning_rate * energy_gradients["mo_coeff_alpha"]
            new_mo_coeff[0] = orthogonalize_orbitals_s_based(new_mo_coeff[0], overlap_matrix)
        
        if "mo_coeff_beta" in energy_gradients:
            new_mo_coeff[1] = new_mo_coeff[1] - learning_rate * energy_gradients["mo_coeff_beta"]
            new_mo_coeff[1] = orthogonalize_orbitals_s_based(new_mo_coeff[1], overlap_matrix)
        
        current_mo_coeff = tuple(new_mo_coeff)
        
        print(f"Iteration {iteration + 1}, Energy: {energy_mean:.6f}, Min Energy: {min_energy:.6f} (Iteration {min_energy_iteration})")
                
        # 구성 업데이트
        configs = sampled_configs
    
    # 최종 결과 반환
    return {
        "optimized_mo_coeff": current_mo_coeff,
        "final_configs": configs,
        "energies": energies,
        "min_energy": float(min_energy),
        "min_energy_mo_coeff": min_energy_mo_coeff,
        "min_energy_configs": min_energy_configs,
        "min_energy_iteration": min_energy_iteration
    }
    
@jax.jit
def calculate_autocorrelation(data, max_lag):
    """Calculate autocorrelation function for a time series."""
    autocorr = []
    
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr.append(1.0)
        else:
            x_t = data[:-lag]
            x_t_plus_lag = data[lag:]
            
            # Calculate normalized correlation
            x_mean = jnp.mean(data)
            numerator = jnp.mean((x_t - x_mean) * (x_t_plus_lag - x_mean))
            denominator = jnp.var(data)
            
            if denominator > 0:
                correlation = numerator / denominator
            else:
                correlation = 0.0
                
            autocorr.append(float(correlation))
    
    return jnp.array(autocorr)

def calculate_energy_autocorrelation(energies, max_lag=None):
    """Calculate autocorrelation function for energy time series."""
    if max_lag is None:
        max_lag = len(energies) // 2
    
    autocorr = []
    mean_energy = jnp.mean(energies)
    var_energy = jnp.var(energies)
    
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr.append(1.0)
        else:
            corr = jnp.mean((energies[:-lag] - mean_energy) * (energies[lag:] - mean_energy)) / var_energy
            autocorr.append(float(corr))
    
    return jnp.array(autocorr)

def integrated_autocorrelation_time(autocorr):
    """Calculate integrated autocorrelation time from autocorrelation function."""
    # Find where autocorrelation first crosses zero or becomes negative
    for i, ac in enumerate(autocorr):
        if ac <= 0 and i > 0:
            cutoff = i
            break
    else:
        cutoff = len(autocorr) // 3  # Default cutoff if no crossing
    
    # Integrate up to the cutoff
    integrated_time = 1.0 + 2.0 * jnp.sum(autocorr[1:cutoff])
    return max(1.0, integrated_time)  # Ensure it's at least 1.0

def estimate_correlation_time(data):
    """Estimate correlation time of a time series."""
    autocorr = calculate_energy_autocorrelation(jnp.array(data))
    return integrated_autocorrelation_time(autocorr)

def analyze_vmc_results(block_energies, plot=True):
    """
    Analyze VMC simulation results to diagnose statistical issues.
    
    Args:
        block_energies: Result dictionary from vmc_improved
        plot: Whether to generate diagnostic plots
    
    Returns:
        Dictionary with analysis results
    """
    import numpy as np
    analysis = {}
    
    # Calculate block-by-block variance
    energies = np.array(block_energies["total"])
    block_variances = []
    
    for i in range(len(block_energies["total"])):
        if i < 3:  # Need a few blocks to calculate variance trend
            continue
            
        # Calculate variance of energy up to this block
        block_var = np.var(energies[:i+1])
        block_variances.append(block_var)
    
    analysis["block_variances"] = block_variances
    
    # Check for energy drift
    analysis["energy_drift"] = linear_trend_test(energies)
    
    # Analyze autocorrelation
    if "energy_autocorrelation" in block_energies:
        autocorr = block_energies["energy_autocorrelation"]
        
        # Find autocorrelation decay time (time to reach 1/e)
        for i, ac in enumerate(autocorr):
            if ac < 1/np.e:
                analysis["autocorr_decay_time"] = i
                break
        else:
            analysis["autocorr_decay_time"] = len(autocorr)
        
        analysis["integrated_autocorr_time"] = block_energies["integrated_autocorr_time"]
        
        # Check if blocks are independent
        if analysis["integrated_autocorr_time"] > 1:
            analysis["blocks_independent"] = False
            analysis["recommended_block_length"] = int(5 * analysis["integrated_autocorr_time"])
        else:
            analysis["blocks_independent"] = True
            analysis["recommended_block_length"] = "current length is sufficient"
    
    # Analyze stuck walkers
    if "final_stuck_count" in block_energies:
        analysis["stuck_walker_percentage"] = 100 * block_energies["final_stuck_count"] / len(block_energies["final_coords"])
        
        if analysis["stuck_walker_percentage"] > 10:
            analysis["stuck_walker_issue"] = True
            analysis["recommended_timestep"] = "decrease from current value"
        else:
            analysis["stuck_walker_issue"] = False
    
    # Check acceptance ratio
    if "mean_acceptance" in block_energies:
        mean_acc = block_energies["mean_acceptance"]
        analysis["mean_acceptance"] = mean_acc
        
        if mean_acc < 0.3:
            analysis["acceptance_issue"] = "too low"
            analysis["recommended_timestep_acc"] = "decrease"
        elif mean_acc > 0.7:
            analysis["acceptance_issue"] = "too high"
            analysis["recommended_timestep_acc"] = "increase"
        else:
            analysis["acceptance_issue"] = "acceptable"
            analysis["recommended_timestep_acc"] = "maintain current"
    
    # Check for equilibration issues
    if len(energies) > 10:
        first_quarter = energies[:len(energies)//4]
        rest = energies[len(energies)//4:]
        
        f_mean, r_mean = np.mean(first_quarter), np.mean(rest)
        f_std, r_std = np.std(first_quarter), np.std(rest)
        
        if abs(f_mean - r_mean) > 2 * (f_std / np.sqrt(len(first_quarter)) + r_std / np.sqrt(len(rest))):
            analysis["equilibration_issue"] = True
            analysis["recommended_equil_steps"] = "increase significantly"
        else:
            analysis["equilibration_issue"] = False
    
    # Effective sample size analysis
    if "effective_sample_size" in block_energies:
        analysis["effective_sample_size"] = block_energies["effective_sample_size"]
        
        if analysis["effective_sample_size"] < 10:
            analysis["sample_size_issue"] = True
            analysis["recommended_steps"] = "increase total steps significantly"
        else:
            analysis["sample_size_issue"] = False
    
    # Generate summary of issues
    issues = []
    recommendations = []
    
    if analysis.get("equilibration_issue", False):
        issues.append("Insufficient equilibration")
        recommendations.append(f"Increase equilibration steps (current recommendation: {analysis['recommended_equil_steps']})")
    
    if not analysis.get("blocks_independent", True):
        issues.append("High autocorrelation between samples")
        recommendations.append(f"Increase block length to {analysis['recommended_block_length']} steps")
    
    if analysis.get("stuck_walker_issue", False):
        issues.append(f"Stuck walkers ({analysis['stuck_walker_percentage']:.1f}% of configurations)")
        recommendations.append("Decrease timestep and/or improve trial wavefunction")
    
    if analysis.get("acceptance_issue", "acceptable") != "acceptable":
        issues.append(f"Acceptance ratio is {analysis['acceptance_issue']} ({analysis['mean_acceptance']:.2f})")
        recommendations.append(f"Try {analysis['recommended_timestep_acc']}ing the timestep")
    
    if analysis.get("sample_size_issue", False):
        issues.append(f"Effective sample size too small ({analysis['effective_sample_size']:.1f})")
        recommendations.append("Run more blocks or increase steps per block")
    
    if analysis.get("energy_drift", False):
        issues.append("Energy shows a drift over time")
        recommendations.append("Increase equilibration and check for instability in trial wavefunction")
    
    analysis["issues"] = issues
    analysis["recommendations"] = recommendations
    
    # Generate plots if requested
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            # Energy convergence
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(energies, 'o-')
            plt.xlabel('Block')
            plt.ylabel('Energy')
            plt.title('Energy per Block')
            
            plt.subplot(2, 2, 2)
            plt.plot(block_energies["cumulative_energy"], 'o-')
            plt.xlabel('Block')
            plt.ylabel('Cumulative Energy')
            plt.title('Cumulative Energy')            
            plt.subplot(2, 2, 3)
            if "energy_autocorrelation" in block_energies:
                plt.plot(block_energies["energy_autocorrelation"][:50], 'o-')  # Show first 50 lags
                plt.axhline(y=0, color='r', linestyle='-')
                plt.xlabel('Lag')
                plt.ylabel('Autocorrelation')
                plt.title('Energy Autocorrelation')
            
            plt.subplot(2, 2, 4)
            plt.plot(block_energies["cumulative_error"], 'o-', label='Standard Error')
            if "cumulative_error_reblocked" in block_energies:
                plt.plot(block_energies["cumulative_error_reblocked"], 'o-', label='Reblocked Error')
            plt.xlabel('Block')
            plt.ylabel('Error Estimate')
            plt.legend()
            plt.title('Error Convergence')
            
            plt.tight_layout()
            plt.savefig('vmc_energy_analysis.png')
            
            # Acceptance and MSD
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(block_energies["acceptance"], 'o-')
            plt.axhline(y=0.5, color='r', linestyle='--')
            plt.xlabel('Block')
            plt.ylabel('Acceptance Ratio')
            plt.title('Acceptance Ratio per Block')
            
            plt.subplot(2, 2, 2)
            if "mean_msd" in block_energies and len(block_energies["mean_msd"]) > 0:
                plt.plot(block_energies["mean_msd"], 'o-')
                plt.xlabel('Step')
                plt.ylabel('Mean Squared Displacement')
                plt.title('Mean Squared Displacement')
            
            plt.subplot(2, 2, 3)
            if "max_ages" in block_energies:
                plt.plot(block_energies["max_ages"], 'o-')
                plt.xlabel('Block')
                plt.ylabel('Maximum Walker Age')
                plt.title('Maximum Walker Age per Block')
            
            plt.subplot(2, 2, 4)
            if "final_age_histogram" in block_energies:
                ages = np.arange(len(block_energies["final_age_histogram"]))
                plt.bar(ages, block_energies["final_age_histogram"])
                plt.xlabel('Walker Age')
                plt.ylabel('Count')
                plt.title('Walker Age Distribution')
                plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig('vmc_performance_analysis.png')
            
            analysis["plots_generated"] = True
            
        except Exception as e:
            print(f"Error generating plots: {e}")
            analysis["plots_generated"] = False
    
    return analysis

def linear_trend_test(data, threshold=0.05):
    """Test if there's a significant linear trend in the data."""
    import numpy as np
    from scipy import stats
    
    x = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
    
    return p_value < threshold