import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Optional

from pyscf import gto, scf, mcscf
import pyqmc.api as pyq

from qmc.pyscftools import  orbital_evaluator_from_pyscf
from qmc.orbitals import *
from qmc.determinants import *

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



