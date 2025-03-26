import jax
import jax.numpy as jnp
import numpy as np
from qmc.determinants import gradient_laplacian

@jax.jit
def jax_ee_energy(configs):
    """
    Calculate electron-electron Coulomb interaction energy (1/r).
    Mathematical form: V_ee = Σ_i<j (1/|r_i - r_j|)
    
    Pure JAX operations, so JIT-compatible, but use of if conditions 
    and jnp.where may affect performance depending on the situation.
    """
    
    nconf, nelec, _ = configs.shape
    # if nelec == 1:
    #     return jnp.zeros(nconf)
    r_ee = configs[:, :, None, :] - configs[:, None, :, :]
    r_ee_dist = jnp.sqrt(jnp.sum(r_ee**2, axis=-1))
    mask = ~jnp.eye(nelec, dtype=bool)
    ee = jnp.where(mask, 1.0 / r_ee_dist, 0.0)
    return jnp.sum(ee, axis=(1,2)) * 0.5

@jax.jit
def jax_ei_energy(configs, atom_charges, atom_coords):
    """
    Calculate electron-ion Coulomb interaction energy using JAX.
    Mathematical form: V_ei = -Σ_i Σ_A (Z_A/|r_i - R_A|)
    where Z_A is nuclear charge and R_A is nuclear position.
    
    Args:
        configs: Configuration of electrons
        atom_charges: Array of atomic charges
        atom_coords: Array of atomic coordinates
    
    Returns:
        Electron-ion interaction energy
    """
    ei = 0.0
    
    for c, coord in zip(atom_charges, atom_coords):
        delta = configs - coord[jnp.newaxis, jnp.newaxis, :]
        deltar = jnp.sqrt(jnp.sum(delta**2, axis=2))
        ei += -c * jnp.sum(1.0 / deltar, axis=1)
    
    return ei

def dist_matrix(configs):
    """
    Calculate distance vectors between all pairs of coordinates.
    Mathematical form: r_ij = r_i - r_j for all i < j
    
    Uses Python loop → Not JIT-compatible.
    """
    nconf, n = configs.shape[:2]
    npairs = int(n * (n - 1) / 2)
    if npairs == 0:
        return jnp.zeros((nconf, 0, 3)), []
    vs = []
    ij = []
    for i in range(n):
        dist_vectors = configs[:, i+1:, :] - configs[:, i:i+1, :]
        vs.append(dist_vectors)
        ij.extend([(i, j) for j in range(i+1, n)])
    vs = jnp.concatenate(vs, axis=1)
    return vs, ij

def jax_ii_energy(mol):
    """
    Calculate ion-ion Coulomb interaction energy.
    Mathematical form: V_ii = Σ_A<B (Z_A Z_B/|R_A - R_B|)
    where Z_A, Z_B are nuclear charges and R_A, R_B are nuclear positions.
    
    Simple calculation but includes Python loop → Not JIT-compatible.
    """
    coords = jnp.array(mol.atom_coords())[jnp.newaxis, :, :]
    charges = jnp.array(mol.atom_charges())
    rij, ij = dist_matrix(coords)
    if len(ij) == 0:
        return jnp.array(0.0)
    rij = jnp.linalg.norm(rij, axis=2)[0, :]
    energy = sum(charges[i] * charges[j] / r for (i, j), r in zip(ij, rij))
    return energy

def compute_potential_energy(mol, configs):
    """
    Calculate total potential energy = V_ee + V_ei + V_ii
    Mathematical form: V_total = V_ee + V_ei + V_ii
    
    JIT application possible if component functions are pure operations.
    """
    ee = jax_ee_energy(configs)
    ei = jax_ei_energy(mol, configs)
    ii = jax_ii_energy(mol)
    
    PE = ee + ei + ii
    
    return PE


def jax_kinetic_energy(configs, mol, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash):
    """
    운동에너지 계산
    
    Parameters
    ----------
    configs : jnp.ndarray
    configurations
    
    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        (kinetic_energy, gradient_squared)
    """
    nconf = configs.shape[0]
    ke = jnp.zeros(nconf)
    grad2 = jnp.zeros(nconf)
    for e in range(configs.shape[1]):
        grad, lap = gradient_laplacian(mol, e, configs[:, e, :], dets, inverse, 
                                       mo_coeff, det_coeff, det_map, _nelec, occup_hash)

        # -1/2 ∇²Ψ/Ψ
        ke += -0.5 * jnp.real(lap)
        
        # gradient culmulation
        grad2 += jnp.sum(jnp.abs(grad)**2, axis=0)
    
    return ke, grad2