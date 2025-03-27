import jax
import jax.numpy as jnp
from jax import random

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
    Final coordinates, acceptance ratio, and elapsed time
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
   autocorr_length=20
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
    electron_ages = jnp.zeros((nconf, nelec), dtype=jnp.int32)
    
    # Track maximum age reached by each electron
    max_ages = jnp.zeros(nelec, dtype=jnp.int32)
    
    #  Initialize wavefunction
    aovals, dets, inverse = recompute(mol, coords, mo_coeff, _nelec, occup_hash)

    # Save initial positions for MSD calculation
    initial_coords = coords.copy()
    
    # Initialize list for analysis
    all_energies = []
    all_corrds = []
    all_acc = []
    
    print(f"Starting equilibriation : {equilibration_step} steps")
    
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
        
    if (step + 1) % 100 == 0:
        print(f"Step {step + 1} Acceptance: {total_acc / (step + 1)}")
        
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
        "max_ages" : []  # Track maximum ages of electrons
    }        
    
    block_coords = []
    
    print(f"Starting production : {n_blocks} blocks of {nsteps_per_block} steps")
    
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
            for e in range(nelec):
                # Calculate if electron moved (any change in x, y, or z coordinates)
                moved = jnp.any(coords[:, e, :] != prev_coords[:, e, :], axis=1)
                
                # Where moved is True, reset age to 0; where False, increment age by 1
                electron_ages = electron_ages.at[:, e].set(
                    jnp.where(moved, 0, electron_ages[:, e] + 1)
                )
                
                # Update maximum age for this electron
                max_ages = max_ages.at[e].set(jnp.maximum(max_ages[e], jnp.max(electron_ages[:, e])))
            
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
        
        # Store maximum ages for this block
        
        block_avg_acc = block_acc / nsteps_per_block
        block_energies["max_ages"].append(max_ages.copy())
        block_energies["kinetic"].append(jnp.mean(jnp.array(block_kinetic)))
        block_energies["ee"].append(jnp.mean(jnp.array(block_ee)))
        block_energies["ei"].append(jnp.mean(jnp.array(block_ei)))
        block_energies["total"].append(jnp.mean(jnp.array(block_total)))
        block_energies["acceptance"].append(block_avg_acc)
        
        cumulative_energy = jnp.array(all_total_energies)
        all_acceptance.append(block_avg_acc)
        block_energies["cumulative_energy"].append(jnp.mean(cumulative_energy))
        block_energies["cumulative_variance"].append(jnp.var(cumulative_energy))
        block_energies["cumulative_error"].append(jnp.std(cumulative_energy) / jnp.sqrt(len(cumulative_energy)))
        block_energies["cumulative_acceptance"].append(jnp.mean(jnp.array(all_acceptance)))
        
        block_coords.append(coords)
        
        # Print block summary with max ages
        print(f"Block {block + blockoffset + 1}/{n_blocks + blockoffset}, "
              f"Energy: {block_energies['total'][-1]:.6f} ± "
              f"{jnp.std(jnp.array(block_total)) / jnp.sqrt(len(block_total)):.6f}, "
              f"Acceptance: {block_energies['acceptance'][-1]:.4f}")
        print(f"Maximum ages: {max_ages}")
    
    mean_energy = jnp.mean(jnp.array(block_energies["total"]))
    std_error = jnp.std(jnp.array(block_energies["total"])) / jnp.sqrt(n_blocks)

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
    
    # Add summary statistics
    block_energies["mean_energy"] = mean_energy
    block_energies["std_error"] = std_error
    block_energies["mean_acceptance"] = jnp.mean(jnp.array(block_energies["acceptance"]))
    block_energies["final_coords"] = coords
    block_energies["block_coords"] = block_coords
    block_energies["final_max_ages"] = max_ages
    
    print(f"\nFinal energy: {mean_energy:.6f} ± {std_error:.6f}")
    print(f"Average acceptance ratio: {block_energies['mean_acceptance']:.4f}")
    print(f"Final maximum electron ages: {max_ages}")
    
    return block_energies