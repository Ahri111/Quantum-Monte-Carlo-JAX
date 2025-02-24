import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

import numpy as np
import functools
from functools import partial

from typing import Dict, List, Optional, Tuple, Union
from qmc.determinant_tools import jax_single_occupation_list
from qmc.determinant_tools import jax_organize_determinant_data

def mol_eval_gto(mol, evalstr: str, walker: np.ndarray) -> np.ndarray:
    """
    Calculate the value of molecule orbitals
    
    Parameters:
    mol (object) : The pyscf molecular objects
    evalstr (str) : The evaluation string for the molecular orbitals
    primcoords (np.ndarray): The coordinates of random walker for each electron
    
    Return
    np.ndarray --> to jnx.numpy.ndarray since mol cannot be utilized in jax environement
    """
    
    aos = mol.eval_gto(evalstr, walker)
    if "deriv2" in evalstr:
        aos[4] += aos[7] + aos[9]
        aos = aos[:5]
    
    return aos

def aos(mol, eval_str, configs, mask = None) -> jnp.ndarray:
    '''
    Evaluate atomic orbitals at given configurations.
    
    1) Parameters:
    configs -> mycoords (np.ndarray) : Configuration object containing electron positions
    mask (np.ndarray) : Optional mask for selecting specific configurations
    eval_str : Type of pyscf evaluation string
    φₖ(r) = Nᵢ × Rₙₗ(r) × Yₗₘ(θ,φ)
    
    2) eval_str
    (1) 'GTOval' : Evaluate the value of GTOs
    (2) 'GTOval_ip' : Evaluate the value of GTOs and their first derivatives
        ∂φ/∂x, ∂φ/∂y, ∂φ/∂z 반환
    (3) 'GTOval_ip_ip' : Evaluate the value of GTOs and their first and second derivatives
        ∂²φ/∂x², ∂²φ/∂y², ∂²φ/∂z², ∂²φ/∂x∂y, ∂²φ/∂x∂z, ∂²φ/∂y∂z 반환

    Returns:
    np.ndarray : Atomic orbitals
    [1, nconf, nao] or [1, 3, nconf, nao] (gradients)
    nao: Number of atomic orbitals
    nmo : Nuber of molecular orbitals
    '''
    mycoords = configs if mask is None else configs[mask] # [nconf, nelec, 3]
    mycoords = mycoords.reshape((-1, mycoords.shape[-1])) #[nconf*nelec, 3]
    eval_gto = functools.partial(mol_eval_gto, mol)
    aos = jnp.asarray(eval_gto(eval_str, mycoords))[jnp.newaxis] # [1, nconf*nelec, 3]
    
    if len(aos.shape) == 4:  # derivatives included
        return aos.reshape((1, aos.shape[1], *mycoords.shape[:-1], aos.shape[-1]))

    return aos.reshape((1, *mycoords.shape[:-1], aos.shape[-1]))

@jit
def mos(ao: jnp.ndarray, parameters) -> jnp.ndarray:
    '''
    Convert atomic orbitals to molecular orbitals for given spin.
    
    Parameters:
    ao : Atomic orbital values [1, nconf, nao]
    
    spin : Spin index (0 for alpha, 1 for beta)
    
    Φᵢ(r) = Σₖ cᵢₖ φₖ(r)
    
    Returns:
    Molecular orbital values
    '''
    return jnp.dot(ao[0], parameters)

def check_parameters_complex(*params):
    """
    Check whether detcoeff and mo_coeff are complex
        
    Args:
        *params: param1: mo_coeff, param2 : det_coeff
        
    Returns 1 if complex, 0 if not
    """
    # 각 파라미터를 평탄화하고 연결
    flattened = [param.ravel() for param in params if param is not None]
    if not flattened:  # 빈 입력 체크
        return False
        
    combined = jnp.concatenate(flattened)
    return bool(jnp.iscomplex(combined).any())