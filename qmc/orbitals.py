import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

import functools
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

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

class Orbitals:

    '''
    Mo orbital calculator
    
    Wave function:
    Ψ = det[Φᵢ(rⱼ)]

    Each Mo can be written as:
    Φᵢ(r) = Σₖ cᵢₖ φₖ(r)

    cᵢₖ : Mo coefficient
    φₖ(r) : Atomic orbital
    '''
    
    def __init__(self, mol, mo_coeff):
        '''
        Parameters:
        mol (object) : The pyscf molecular objectsmol (object) : The pyscf molecular objects
        mo_coeff (np.ndarray) : The molecular orbital coefficients
        spin = 0 : Alpha spin
        spin = 1 : Beta spin
        '''
        
        self.parameters = {
            "mo_coeff_alpha": jnp.asarray(mo_coeff[0]),
            "mo_coeff_beta": jnp.asarray(mo_coeff[1]),
        }
        
        self.parm_names = ["mo_coeff_alpha", "mo_coeff_beta"]
        
        self.mo_dtype = complex if any(jnp.iscomplexobj(v) for v in self.parameters.values()) else float
        self.eval_gto = functools.partial(mol_eval_gto, mol)
        self._mol = mol
        
    @jit
    def nmo(self) -> Tuple[int, int]:
        '''
        Return the number of molecular orbitals for alpha and beta spins
        '''
        return (self.parameters["mo_coeff_alpha"].shape[-1], 
                self.parameters["mo_coeff_beta"].shape[-1])
        
    def aos(self, eval_str, configs, mask = None) -> jnp.ndarray:
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
        aos = jnp.asarray(self.eval_gto(eval_str, mycoords))[jnp.newaxis] # [1, nconf*nelec, 3]
        
        if len(aos.shape) == 4:  # derivatives included
            return aos.reshape((1, aos.shape[1], *mycoords.shape[:-1], aos.shape[-1]))
    
        return aos.reshape((1, *mycoords.shape[:-1], aos.shape[-1]))
    
    @jit
    def mos(self, ao: jnp.ndarray, parameters) -> jnp.ndarray:
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
    
    @jit
    def pgradient(self, ao: jnp.ndarray, spin: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        nelec = [self.parameters[self.parm_names[spin]].shape[1]]

        return (jnp.array(nelec), ao)