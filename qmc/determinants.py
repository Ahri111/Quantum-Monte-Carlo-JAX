import jax
import jax.numpy as jnp
from functools import partial
from qmc.orbitals import aos
from qmc.orbitals import mos

def convert_to_hashable(occup):
    
    up_orbs = tuple(occup[0])
    dn_orbs = tuple(occup[1])
    
    return (up_orbs, dn_orbs)


@partial(jax.jit, static_argnums= (2,))
def compute_determinants(mo_values, det_occup, s):
    mo_vals = jnp.swapaxes(mo_values[:, :, det_occup[s]], 1, 2)
    compute_det = jnp.asarray(jnp.linalg.slogdet(mo_vals))
    inverse = jax.vmap(jnp.linalg.inv)(mo_vals)
    return compute_det, inverse

def recompute(configs, 
              atomic_orbitals, 
              mo_coeff, 
              _nelec, 
              occup_hash, 
              s):
        
    nconf, nelec_tot, ndim = configs.shape
    aovals = atomic_orbitals.reshape(-1, nconf, nelec_tot, atomic_orbitals.shape[-1])
    
    if s == 0:
        ao_slice = aovals[:, :, :_nelec[0], ]
        param = mo_coeff[0]
        
    else:
        ao_slice = aovals[:, :, _nelec[0]:_nelec[0] + _nelec[1], :]
        param = mo_coeff[1]
        
    mo = mos(ao_slice, param)
    
    return compute_determinants(mo, occup_hash, s)

def compute_wf_value(configs, 
                     atomic_orbitals, 
                     mo_coeff, 
                     det_coeff, 
                     _nelec, 
                     occup_hash,
                     det_map):
    
    if det_coeff is None:
        det_coeff = jnp.array([1.0])
        
    updets, up_inverse = recompute(
        configs = configs, 
        atomic_orbitals = atomic_orbitals, 
        mo_coeff = mo_coeff, 
        _nelec = _nelec,
        occup_hash = occup_hash, 
        s=0
    )
    
    updets = updets[:, :, det_map[0]]
    
    dndets, down_inverse = recompute(
        configs = configs,
        atomic_orbitals = atomic_orbitals,
        mo_coeff = mo_coeff,
        _nelec = _nelec,
        occup_hash = occup_hash,
        s=1
    )
        
    if det_coeff is None:
        det_coeff = jnp.array([1.0])
        
    upref = jnp.amax(updets[1]).real
    dnref = jnp.amax(dndets[1]).real
    
    phases = updets[0] * dndets[0]
    logvals = updets[1] - upref + dndets[1] - dnref
    
    wf_val = jnp.einsum("d,id->i", det_coeff, phases * jnp.exp(logvals))
    wf_sign = jnp.where(wf_val == 0, 0.0, wf_val / jnp.abs(wf_val))
    wf_logval = jnp.where(wf_val == 0, -jnp.inf,
                            jnp.log(jnp.abs(wf_val)) + upref + dnref)
    
    return wf_sign, wf_logval

def gradient_value(mol, 
                   e, 
                   epos, 
                   inverse,
                   updets,
                   dndets, 
                   mo_coeff, 
                   det_coeff,
                   det_map, 
                   _nelec, 
                   occup_hash):
    """
    Compute the gradient value of the wave function
    """
    
    s = int(e >= _nelec[0])
    
    # (φ ∂φ/∂x, ∂φ/∂y, ∂φ/∂z) -> (1, 4, config, number of coefficients)
    aograd = aos(mol, "GTOval_sph_deriv1", epos)
    
    
    # ∂Φᵢ/∂r = Σₖ cᵢₖ ∂φₖ/∂r -> (4, config, number_of_electrons)
    mograd = mos(aograd, mo_coeff[s])

    # (4, config, 1, number_of_electrons)
    mograd_vals = mograd[:, :, occup_hash[s]]
    
    ratio = _testrow_deriv(e = e, 
                           vec = mograd_vals, 
                           inverse = inverse, 
                           s = s,
                           updets = updets,
                           dndets = dndets,
                           det_coeff = det_coeff,
                           det_map = det_map,
                           _nelec = _nelec
                           )
    
    derivatives = ratio[1:] / ratio[0]
    derivatives = derivatives.at[~jnp.isfinite(derivatives)].set(0.0)
    
    values = ratio[0]
    values = values.at[~jnp.isfinite(values)].set(1.0)
    
    return derivatives, values, (aograd[:, 0], mograd[0])

def gradient_laplacian(mol, 
                       e, 
                       epos, 
                       inverse,
                       updets,
                       dndets,
                       mo_coeff,
                       det_coeff,
                       det_map,
                       _nelec,
                       occup_hash
                       ):
    
    s = int(e >= _nelec[0])
    
    ao = aos(mol, "GTOval_sph_deriv2", epos)
    mo = mos(ao, mo_coeff[s])
    mo_vals = mo[..., occup_hash[s]]
    
    ratio = _testrow_deriv(e = e, 
                           vec = mo_vals,
                           inverse = inverse,
                           s = s,
                           updets = updets,
                           dndets = dndets,
                           det_coeff = det_coeff,
                           det_map = det_map,
                           _nelec = _nelec)
    
    ratio = ratio/ratio[:1]
    
    return ratio[1:-1], ratio[-1]

def _testrow_deriv(e, 
                   vec, 
                   inverse, 
                   s,
                   updets,
                   dndets, 
                   det_coeff,
                   det_map,
                   _nelec
                   ):
    
    ratios = jnp.einsum("ei...dj, idj... ->ei...d",
                        vec,
                        inverse[s][..., e - s*_nelec[0]])
    
    upref = jnp.amax(updets[1]).real
    dnref = jnp.amax(dndets[1]).real
    
    det_array = (updets[0, :, det_map[0]] *
                 dndets[0, :, det_map[1]] *
                 jnp.exp(
                     updets[1][:, det_map[0]] +
                     dndets[1][:, det_map[1]] +
                     upref - dnref
                 )
    )
    
    numer = jnp.einsum("ei...d,d,di->ei...",
                        ratios[..., det_map[s]],
                        det_coeff,               
                        det_array                 
    )
    
    denom = jnp.einsum("d,di->i...",
                      det_coeff,
                      det_array
    )
    
    if len(numer.shape) == 3:
        denom = denom[jnp.newaxis, :, jnp.newaxis]

    return numer / denom