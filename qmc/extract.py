from pyscf import gto, scf
import numpy as np
import jax
import jax.numpy as jnp

def pyscf_parameter(mol, mf):
    
    mo_coeff = mf.mo_coeff
    
    ao_labels = mol.ao_labels()
    
    n_electrons = mol.nelectron
    n_orbitals = mol.nao
    
    ao_centers = []
    ao_types = []
    
    for label in ao_labels:
        atom_id = int(label.split()[0])
        ao_type = label.split()[2]
        ao_centers.append(mol.atom_coord(atom_id))  # atom position
        ao_types.append(ao_type)                # atom type

    ao_centers = np.array(ao_centers)
    orbital_energies = mf.mo_energy
    
    parameters = {
        'mo_coeff': mo_coeff,
        "n_electrons": n_electrons,
        "n_orbitals": n_orbitals,
        "ao_centers": ao_centers,
        "ao_types": ao_types,
        "orbital_energies": orbital_energies,
        "total_energy": mf.e_tot
    }
    
    return parameters