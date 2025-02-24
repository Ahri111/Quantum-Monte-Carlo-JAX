import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf, mcscf
import pyqmc.api as pyq
from qmc.pyscftools import orbital_evaluator_from_pyscf
from qmc.orbitals import check_parameters_complex
from qmc.determinants import get_complex_phase

def initialize_calculation(mol: gto.Mole, nconfig: int, seed):
    """
    Initial Calculation
    Args:
        mol: pyscf object
        nconfig: configuration
        
    Returns:
        coords, max_orb, det_coeff, det_map, mo_coeff, occup_hash, nelec
    """
    np.random.seed(seed)
    mf = scf.RHF(mol)
    mf.kernel()
    
    configs = pyq.initial_guess(mol, nconfig)
    coords = configs.configs
    
    max_orb, det_coeff, det_map, mo_coeff, occup_hash, _nelec = \
        orbital_evaluator_from_pyscf(mol, mf)
    
    nelec = np.sum(mol.nelec)
    return coords, max_orb, det_coeff, det_map, mo_coeff, occup_hash, _nelec, nelec

def determine_complex_settings(mo_coeff: jnp.ndarray, 
                             det_coeff: jnp.ndarray):
    """
    
    Args:
        mo_coeff: MO's coefficient
        det_coeff: determinant's coefficent
        
    Returns:
        iscomplex, mo_dtype, get_phase
    """
    ao_dtype = float
    
    iscomplex = check_parameters_complex(mo_coeff)
    mo_dtype = complex if iscomplex else float
    
    
    iscomplex = mo_dtype == complex or check_parameters_complex(det_coeff, mo_coeff)
    get_phase = get_complex_phase if iscomplex else jnp.sign
    
    return iscomplex, mo_dtype, get_phase