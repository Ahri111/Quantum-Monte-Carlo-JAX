import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import h5py
import logging

def np_initial_guess(mol, nconfig, r = 1.0):
    
    epos = np.zeros((nconfig, np.sum(mol.nelec), 3))
    wts = mol.atom_charges()
    wts = wts / np.sum(wts) # investigate charge contribution to config the elctron
    
    for s in [0, 1]:
        neach = np.array(
            np.floor(mol.nelec[s] * wts), dtype=int
        )
        
        nassigned = np.sum(neach)
        totleft = int(mol.nelec[s] - nassigned)
        ind0 = s * mol.nelec[0] # arrange spin 0 first then arrange spin 1
        
        epos[:, ind0:ind0 + nassigned, :] = np.repeat(
            mol.atom_coords(), neach, axis = 0
        ) # neach is array -> we can repeat the atom position based on their integral configuration.        
        
        if totleft > 0:
            inds = np.argpartition(
                np.random.random((nconfig, len(wts))), totleft, axis=1
            )[:, :totleft]
            epos[:, ind0 + nassigned:ind0 + mol.nelec[s], :] = mol.atom_coords()[
                inds
            ]
        epos += r * np.random.randn(*epos.shape)
        
        return epos