import jax
import jax.numpy as jnp
import jax.random as jrand

from typing import Tuple, Optional

from pyscf import gto, scf, mcscf
import pyqmc.api as pyq

from qmc.pyscftools import  orbital_evaluator_from_pyscf
from qmc.orbitals import *
from qmc.determinants import *

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

def metropolis_step(e, carry, mol, mo_coeff, det_coeff, det_map, _nelec, occup_hash, tstep, get_phase):
    """
    Performs a Metropolis step for a single electron.
    
    This function stochastically updates the position of electron e in a quantum Monte Carlo 
    simulation. It uses the Metropolis-Hastings algorithm to sample electron positions according 
    to the probability distribution of the wavefunction squared.
    
    :parameter e: Index of the electron being processed
    :parameter carry: Tuple of state variables (coords, aovals, dets, inverse, key, acc)
        coords: Positions of all electrons [nconf, nelec, 3]
        aovals: Atomic orbital values
        dets: Slater determinant values
        inverse: Inverse of the Slater matrices
        key: JAX random number generator key
        acc: Accumulated acceptance ratio
    :parameter mol: Molecule information (PySCF mol object)
    :parameter mo_coeff: Molecular orbital coefficients
    :parameter det_coeff: Determinant coefficients
    :parameter det_map: Determinant mapping
    :parameter _nelec: Electron count information (total electrons, alpha electrons, beta electrons)
    :parameter occup_hash: Occupation hash
    :parameter tstep: Time step parameter controlling the magnitude of moves
    :parameter get_phase: Function to calculate the phase of Slater determinants
    
    :returns: Updated tuple of state variables (coords, aovals, dets, inverse, key, acc)
    """
    coords, aovals, dets, inverse, key, acc = carry
    nconf = coords.shape[0]  # Number of configurations
    
    # The gradient_value function likely uses mol, so we can't JIT across this boundary
    g, _, _ = gradient_value(mol, e, coords[:, e, :], dets, inverse, mo_coeff, 
                           det_coeff, det_map, _nelec, occup_hash)
    
    # This part doesn't use mol directly, so we could JIT it separately
    grad, key, newcoorde, gauss = _metropolis_compute(
        g, coords, e, key, tstep, nconf)
    
    # The gradient_value call again uses mol
    g, new_val, saved = gradient_value(mol, e, newcoorde, dets, inverse, mo_coeff, 
                                     det_coeff, det_map, _nelec, occup_hash)
    
    # This part could be another JIT-compiled function
    new_grad, t_prob, ratio, key, accept, acc, indices, coords = _metropolis_decision(
        g, newcoorde, new_val, coords, gauss, grad, tstep, e, key, nconf, _nelec[0])
    
    # Sherman-Morrison update likely involves mol, so keep outside JIT
    aovals, dets, inverse = sherman_morrison(e, newcoorde, coords, mask=accept, 
                                           aovals=aovals, saved_value=saved, 
                                           get_phase=get_phase, dets=dets, 
                                           inverse=inverse, mo_coeff=mo_coeff, 
                                           occup_hash=occup_hash, _nelec=_nelec)
    
    return coords, aovals, dets, inverse, key, acc


# JIT-compiled helper functions for the parts that don't use mol
@jax.jit
def _metropolis_compute(g, coords, e, key, tstep, nconf):
    """Computes the gradient drift and proposed move (JIT-compatible part)"""
    grad = limdrift(jnp.real(g.T))
    
    key, subkey = jrand.split(key)
    gauss = jrand.normal(subkey, shape=(nconf, 3)) * jnp.sqrt(tstep)
    
    newcoorde = coords[:, e, :] + gauss + grad * tstep
    
    return grad, key, newcoorde, gauss


@jax.jit
def _metropolis_decision(g, newcoorde, new_val, coords, gauss, grad, tstep, e, key, nconf, nelec_total):
    """Computes the acceptance ratio and makes the decision (JIT-compatible part)"""
    new_grad = limdrift(jnp.real(g.T))
    
    forward = jnp.sum(gauss**2, axis=1)
    backward = jnp.sum((gauss + tstep * (grad + new_grad))**2, axis=1)
    t_prob = jnp.exp(1 / (2 * tstep) * (forward - backward))
    
    ratio = jnp.abs(new_val) ** 2 * t_prob
    
    key, subkey = jrand.split(key)
    accept = ratio > jrand.uniform(subkey, shape=(nconf,))
    
    acc = jnp.mean(accept) / nelec_total
    
    indices = jnp.where(accept)[0]
    coords = coords.at[indices, e, :].set(newcoorde[indices, :])
    
    return new_grad, t_prob, ratio, key, accept, acc, indices, coords




@jax.jit
def run_equilibration_step(i, carry, mol, mo_coeff, det_coeff, det_map, _nelec, occup_hash, nelec, tstep, get_phase):
    """
    Performs a single equilibration step by updating all electrons once.
    
    This function applies the Metropolis algorithm to each electron in sequence,
    completing one full sweep through all electrons in the system.
    
    :parameter i: Current equilibration step index
    :parameter carry: Tuple of state variables (coords, aovals, dets, inverse, key, total_acc)
        coords: Positions of all electrons [nconf, nelec, 3]
        aovals: Atomic orbital values
        dets: Slater determinant values
        inverse: Inverse of the Slater matrices
        key: JAX random number generator key
        total_acc: Total accumulated acceptance ratio
    :parameter mol: Molecule information
    :parameter mo_coeff: Molecular orbital coefficients
    :parameter det_coeff: Determinant coefficients
    :parameter det_map: Determinant mapping
    :parameter _nelec: Electron count information (total electrons, alpha electrons, beta electrons)
    :parameter occup_hash: Occupation hash
    :parameter nelec: Number of electrons
    :parameter tstep: Time step parameter controlling the magnitude of moves
    :parameter get_phase: Function to calculate the phase of Slater determinants
    
    :returns: Updated tuple of state variables (coords, aovals, dets, inverse, key, total_acc)
    """
    coords, aovals, dets, inverse, key, total_acc = carry
    
    # Initialize acceptance ratio for this step
    acc = 0.0
    
    # Define a function for metropolis_step with fixed parameters except electron index
    def step_fn(e, carry):
        return metropolis_step(e, carry, mol, mo_coeff, det_coeff, det_map, _nelec, 
                              occup_hash, tstep, get_phase)
    
    # Apply metropolis step to all electrons using JAX's fori_loop
    coords, aovals, dets, inverse, key, acc = jax.lax.fori_loop(
        0, nelec, step_fn, (coords, aovals, dets, inverse, key, acc))
    
    # Update total acceptance ratio
    total_acc = total_acc + acc
    
    return coords, aovals, dets, inverse, key, total_acc

@jax.jit
def run_simulation(coords, aovals, dets, inverse, key, mol, mo_coeff, det_coeff, det_map, _nelec, occup_hash, nelec, equilibration_step, tstep, get_phase):
    """
    Runs a complete quantum Monte Carlo simulation with equilibration.
    
    This function performs multiple equilibration steps to sample electron configurations
    according to the wavefunction probability distribution. Each equilibration step
    updates all electrons once.
    
    :parameter coords: Initial electron positions [nconf, nelec, 3]
    :parameter aovals: Initial atomic orbital values
    :parameter dets: Initial Slater determinant values
    :parameter inverse: Initial inverse of the Slater matrices
    :parameter key: JAX random number generator key
    :parameter mol: Molecule information
    :parameter mo_coeff: Molecular orbital coefficients
    :parameter det_coeff: Determinant coefficients
    :parameter det_map: Determinant mapping
    :parameter _nelec: Electron count information (total electrons, alpha electrons, beta electrons)
    :parameter occup_hash: Occupation hash
    :parameter nelec: Number of electrons
    :parameter equilibration_step: Number of equilibration steps to perform
    :parameter tstep: Time step parameter controlling the magnitude of moves
    :parameter get_phase: Function to calculate the phase of Slater determinants
    
    :returns: Tuple containing updated electron configurations, wavefunction values, and average acceptance ratio
        coords: Final electron positions
        aovals: Final atomic orbital values
        dets: Final Slater determinant values
        inverse: Final inverse of the Slater matrices
        key: Updated JAX random number generator key
        avg_acc: Average acceptance ratio across all equilibration steps
    """
    # Initialize acceptance ratio
    total_acc = 0.0
    
    # Define a function for run_equilibration_step with fixed parameters except step index
    def equil_fn(i, carry):
        return run_equilibration_step(i, carry, mol, mo_coeff, det_coeff, det_map, _nelec, 
                                     occup_hash, nelec, tstep, get_phase)
    
    # Execute all equilibration steps using JAX's fori_loop
    coords, aovals, dets, inverse, key, total_acc = jax.lax.fori_loop(
        0, equilibration_step, equil_fn, (coords, aovals, dets, inverse, key, total_acc))
    
    # Calculate average acceptance ratio
    avg_acc = total_acc / equilibration_step
    
    return coords, aovals, dets, inverse, key, avg_acc