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

# # Main simulation function that uses the JIT-compiled subfunctions
def mc_simulation(coords, mol, mo_coeff, det_coeff, det_map, _nelec, occup_hash,
                     get_phase, key, equilibration_step=500, tstep=0.5):
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
    
    # Initialize wavefunction values
    aovals, dets, inverse = recompute(mol, coords, mo_coeff, _nelec, occup_hash)
    
    # Set random seed for reproducibility    
    total_acc = 0.0
    
    # Main simulation loop
    for i in range(equilibration_step):
        step_acc = 0.0
        
        # Update each electron
        for e in range(nelec):
            
            key, gauss_key = random.split(key)
            
            # Calculate gradient at current position (uses mol, can't be JIT-compiled)
            g, _, _ = gradient_value(mol, e, coords[:, e, :], dets, inverse, mo_coeff,
                                    det_coeff, det_map, _nelec, occup_hash)
            
            # Apply drift limitation (JIT-compiled)
            grad = limdrift(jnp.real(g.T))
            
            # Generate random displacement
            # gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
            # gauss = jnp.array(gauss)
            gauss = random.normal(gauss_key, shape = (nconf, 3)) * jnp.sqrt(tstep)
            
            # Propose new position
            newcoorde = coords[:, e, :] + gauss + grad * tstep
            
            # Calculate gradient at proposed position (uses mol, can't be JIT-compiled)
            g, new_val, saved = gradient_value(mol, e, newcoorde, dets, inverse, mo_coeff,
                                             det_coeff, det_map, _nelec, occup_hash)
            
            # Apply drift limitation to new gradient (JIT-compiled)
            new_grad = limdrift(jnp.real(g.T))
            
            # Compute transition probability (JIT-compiled)
            t_prob = compute_transition_probability(gauss, grad, new_grad, tstep)
            
            # Calculate acceptance ratio
            ratio = jnp.abs(new_val) ** 2 * t_prob
            
            # Metropolis acceptance/rejection
            key, uniform_key = random.split(key)
            uniform_rand = random.uniform(uniform_key, shape=(nconf,))
            accept = ratio > uniform_rand
            # accept = ratio > np.random.rand(nconf)
            
            # Update coordinates for accepted moves (JIT-compiled)
            coords = update_coordinates(coords, e, accept, newcoorde)
            
            # Update wavefunction values (uses mol, can't be JIT-compiled)
            aovals, dets, inverse = sherman_morrison(e, newcoorde, coords, accept, aovals, 
                                                     saved, get_phase, dets, inverse, 
                                                     mo_coeff, occup_hash, _nelec)
            
            # Update acceptance counter
            step_acc += jnp.mean(accept) / nelec
        
        # Track average acceptance
        total_acc += step_acc / equilibration_step
    
    return coords, total_acc


def jax_energy_mc_simulation(coords, mol, mo_coeff, det_coeff, det_map, _nelec, occup_hash,
                             get_phase, key, equilibration_step=500, tstep=0.5):
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
    atom_coords = jnp.array(mol.atom_coords())
    atom_charges = jnp.array(mol.atom_charges())    
    # Initialize wavefunction values
    aovals, dets, inverse = recompute(mol, coords, mo_coeff, _nelec, occup_hash)
    
    total_acc = 0.0
    ee_total = 0.0
    ei_total = 0.0
    ke_total = 0.0
    ii_total = 0.0
    te_total = 0.0
    # Main simulation loop
    for i in range(equilibration_step):
        step_acc = 0.0
        
        # Update each electron
        for e in range(nelec):
            
            key, gauss_key = random.split(key)

            # Calculate gradient at current position (uses mol, can't be JIT-compiled)
            g, _, _ = gradient_value(mol, e, coords[:, e, :], dets, inverse, mo_coeff,
                                    det_coeff, det_map, _nelec, occup_hash)
            
            # Apply drift limitation (JIT-compiled)
            grad = limdrift(jnp.real(g.T))
            
            # Generate random displacement
            # gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
            # gauss = jnp.array(gauss)
            gauss = random.normal(gauss_key, shape = (nconf, 3)) * jnp.sqrt(tstep)
    
            # Propose new position
            newcoorde = coords[:, e, :] + gauss + grad * tstep
            
            # Calculate gradient at proposed position (uses mol, can't be JIT-compiled)
            g, new_val, saved = gradient_value(mol, e, newcoorde, dets, inverse, mo_coeff,
                                             det_coeff, det_map, _nelec, occup_hash)
            
            # Apply drift limitation to new gradient (JIT-compiled)
            new_grad = limdrift(jnp.real(g.T))
            
            # Compute transition probability (JIT-compiled)
            t_prob = compute_transition_probability(gauss, grad, new_grad, tstep)
            
            # Calculate acceptance ratio
            ratio = jnp.abs(new_val) ** 2 * t_prob
            
            # Metropolis acceptance/rejection
            key, uniform_key = random.split(key)
            uniform_rand = random.uniform(uniform_key, shape=(nconf,))
            accept = ratio > uniform_rand
            # accept = ratio > np.random.rand(nconf)
            
            # Update coordinates for accepted moves (JIT-compiled)
            coords = update_coordinates(coords, e, accept, newcoorde)
            
            # Update wavefunction values (uses mol, can't be JIT-compiled)
            aovals, dets, inverse = sherman_morrison(e, newcoorde, coords, accept, aovals, 
                                                   saved, get_phase, dets, inverse, 
                                                   mo_coeff, occup_hash, _nelec)
            
            # Update acceptance counter
            step_acc += jnp.mean(accept) / nelec
        
        # Track average acceptance
        total_acc += step_acc / equilibration_step
        
    #     # ie = jnp.mean(ei_energy)
    #     ee_current =  jnp.mean(jax_ee_energy(coords), axis = 0)
    #     ei_current = jnp.mean(jax_ei_energy(coords, atom_charges, atom_coords), axis = 0)
    #     ii_current = jax_ii_energy(mol)
    #     ke_current = jnp.mean(jax_kinetic_energy(coords, mol, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash)[0], axis = 0)
        
    #     ee_total += ee_current / equilibration_step
    #     ei_total += ei_current / equilibration_step
    #     ke_total += ke_current / equilibration_step
    #     ii_total += ii_current / equilibration_step
        
    # te_total = ee_total + ei_total + ke_total + ii_total
    
    return coords, te_total, total_acc

def jax_energy_mc_wo_LG_simulation(coords, mol, mo_coeff, det_coeff, det_map, _nelec, occup_hash,
                                   get_phase, key, equilibration_step=500, tstep=0.5):
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
    atom_coords = jnp.array(mol.atom_coords())
    atom_charges = jnp.array(mol.atom_charges())    
    # Initialize wavefunction values
    aovals, dets, inverse = recompute(mol, coords, mo_coeff, _nelec, occup_hash)
    
    total_acc = 0.0
    ee_total = 0.0
    ei_total = 0.0
    ke_total = 0.0
    ii_total = 0.0
    
    # Main simulation loop
    for i in range(equilibration_step):
        step_acc = 0.0
        
        # Update each electron
        for e in range(nelec):
            
            # Generate random displacement
            key, gauss_key = random.split(key)
            
            gauss = random.normal(gauss_key, shape=(nconf, 3)) * jnp.sqrt(tstep)
                        
            # Calculate gradient at proposed position (uses mol, can't be JIT-compiled)
            _, old_val, _ = gradient_value(mol, e, newcoorde, dets, inverse, mo_coeff,
                                             det_coeff, det_map, _nelec, occup_hash)
            
            newcoorde = coords[:, e, :] + gauss

            _, new_val, saved = gradient_value(mol, e, newcoorde, dets, inverse, mo_coeff,
                                             det_coeff, det_map, _nelec, occup_hash)            
            # Calculate acceptance ratio
            ratio = jnp.abs(new_val) ** 2
            
            # Metropolis acceptance/rejection
            key, uniform_key = random.split(key)
            uniform_rand = random.uniform(uniform_key, shape=(nconf,))
            accept = ratio > uniform_rand

            # Update coordinates for accepted moves (JIT-compiled)
            coords = update_coordinates(coords, e, accept, newcoorde)
            
            # Update wavefunction values (uses mol, can't be JIT-compiled)
            aovals, dets, inverse = sherman_morrison(e, newcoorde, coords, accept, aovals, 
                                                   saved, get_phase, dets, inverse, 
                                                   mo_coeff, occup_hash, _nelec)
            
            # Update acceptance counter
            step_acc += jnp.mean(accept) / nelec
        
        # Track average acceptance
        total_acc += step_acc / equilibration_step
        ee_current = jnp.mean(jax_ee_energy(coords), axis=0)
        ei_current = jnp.mean(jax_ei_energy(coords, atom_charges, atom_coords), axis=0)
        ii_current = jax_ii_energy(mol)
        ke_current = jnp.mean(jax_kinetic_energy(coords, mol, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash)[0], axis=0)
        
        # Accumulate energy components for averaging
        ee_total += ee_current / equilibration_step
        ei_total += ei_current / equilibration_step
        ke_total += ke_current / equilibration_step
        ii_total += ii_current / equilibration_step
     
    te_total = ee_total + ei_total + ke_total + ii_total

    return coords, te_total

def vmc(
   coords, 
   mol, 
   mo_coeff, 
   det_coeff, 
   det_map, 
   _nelec, 
   occup_hash,
   get_phase, 
   key, 
   equilibration_step=500, 
   tstep=0.5,
   n_blocks = 10,
   nsteps_per_block = 10,
   blockoffset = 0,
   mode = ""
):
    
    block_energies = {
        
    }
    

