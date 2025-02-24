import jax
import jax.numpy as jnp
from functools import partial
from qmc.orbitals import aos
from qmc.orbitals import mos

def convert_to_hashable(occup):
    
    up_orbs = tuple(occup[0])
    dn_orbs = tuple(occup[1])
    
    return (up_orbs, dn_orbs)

def check_complex_components(
    mo_dtype,
    parameters,
):
    is_mo_complex = mo_dtype == jnp.complex64 or mo_dtype == jnp.complex128
    param_values = jnp.stack([p.ravel()[0] for p in parameters.values()])
    is_params_complex = jnp.any(jnp.iscomplex(param_values))
    return bool(is_mo_complex or is_params_complex)

def get_complex_phase(x):
    return x / jnp.abs(x)

@partial(jax.jit, static_argnums= (2,))
def compute_determinants(mo_values, occup_hash, s):
    mo_vals = jnp.swapaxes(mo_values[:, :, occup_hash[s]], 1, 2)
    compute_det = jnp.asarray(jnp.linalg.slogdet(mo_vals))
    inverse = jax.vmap(jnp.linalg.inv)(mo_vals)
    return compute_det, inverse

def recompute(mol, configs, mo_coeff, _nelec, occup_hash):
    nconf, nelec_tot, ndim = configs.shape
    atomic_orbitals = aos(mol,"GTOval_sph", configs)
    aovals = atomic_orbitals.reshape(-1, nconf, nelec_tot, atomic_orbitals.shape[-1])

    updets, upinverse = compute_determinants(mos(aovals[:, :, :_nelec[0]], mo_coeff[0]), occup_hash, 0)
    dndets, dninverse = compute_determinants(mos(aovals[:, :, _nelec[0]:], mo_coeff[1]), occup_hash, 1)
    
    return (updets, dndets), (upinverse, dninverse), aovals

def compute_wf_value(configs, dets, det_coeff, det_map):
    
    det_coeff = jnp.array([1.0]) if det_coeff is None else det_coeff
    updets, dndets = dets
    updets, dndets = updets[:, :, det_map[0]], dndets[:, :, det_map[1]]
    
    upref, dnref = jnp.amax(updets[1]).real, jnp.amax(dndets[1]).real
    phases = updets[0] * dndets[0]
    logvals = updets[1] - upref + dndets[1] - dnref
    
    wf_val = jnp.einsum("d,id->i", det_coeff, phases * jnp.exp(logvals))
    wf_sign = jnp.where(wf_val == 0, 0.0, wf_val / jnp.abs(wf_val))
    wf_logval = jnp.where(wf_val == 0, -jnp.inf,
                            jnp.log(jnp.abs(wf_val)) + upref + dnref)
    
    return wf_sign, wf_logval

def gradient_value(mol, e, epos, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash):
    
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
    
    ratio = _testrow_deriv(e, mograd_vals, inverse, s, dets, det_coeff, det_map, _nelec)
    
    derivatives = ratio[1:] / ratio[0]
    derivatives = derivatives.at[~jnp.isfinite(derivatives)].set(0.0)
    
    values = ratio[0]
    values = values.at[~jnp.isfinite(values)].set(1.0)
    
    return derivatives, values, (aograd[:, 0], mograd[0])

def gradient_laplacian(mol, e,  epos, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash):
    
    s = int(e >= _nelec[0])
    
    ao = aos(mol, "GTOval_sph_deriv2", epos)
    mo = mos(ao, mo_coeff[s])
    mo_vals = mo[..., occup_hash[s]]
    
    ratio = _testrow_deriv(e, mo_vals, inverse, s, dets, det_coeff, det_map, _nelec)
    
    ratio = ratio/ratio[:1]
    
    return ratio[1:-1], ratio[-1]

def _testrow_deriv(e, vec, inverse, s, dets, det_coeff, det_map, _nelec):
    
    ratios = jnp.einsum("ei...dj, idj... ->ei...d",
                        vec,
                        inverse[s][..., e - s*_nelec[0]])
    
    upref = jnp.amax(dets[0][1]).real
    dnref = jnp.amax(dets[1][1]).real
    
    det_array = (dets[0][0, :, det_map[0]] *
                 dets[1][0, :, det_map[1]] *
                 jnp.exp(
                     dets[0][1][:, det_map[0]] 
                     + dets[1][1][:, det_map[1]] 
                     - upref 
                     - dnref
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

def sherman_morrison_ms(e, inv, vec):
    
    # v'tA-1
    tmp = jnp.einsum("edk, edkj -> edj", vec, inv)
    
    # uv'tA-1
    ratio = tmp[:, :, e]
    
    # (A-1u)/(uv'tA-1)    
    inv_ratio = inv[:, :, :, e] / ratio[:, :, jnp.newaxis]
    
    # A-1 - (A-1uv'tA-1)/(uv'tA-1)
    invnew = inv - jnp.einsum("kdi, kdj -> kdij", inv_ratio, tmp)
    
    invnew = invnew.at[:, :, :, e].set(inv_ratio)
    
    return ratio, invnew  

def sherman_morrison(e, epos, configs, mask, gtoval, aovals, saved_value, get_phase, dets, inverse, mo_coeff, occup_hash, _nelec):
    
    s = int(e >= _nelec[0])
    
    if mask is None:
        mask = jnp.ones(epos.shape[0], dtype=bool)
    
    
    eeff = e - s * _nelec[0]
    
    if saved_value is None:
        ao = aos(gtoval, epos, mask)
        aovals[:, mask, e, :] = ao
        mo = mos(ao, mo_coeff[s])
    
    else:
        ao, mo = saved_value
        aovals = aovals.at[:, mask, e, :].set(ao[:, mask])
        mo = mo[mask]
        
    mo_vals = mo[:, occup_hash[s]]
        
    det_ratio, new_inverse_value = sherman_morrison_ms(eeff, inverse[s][mask], mo_vals)
    
    inverse_list = list(inverse)
    inverse_list[s] = inverse[s].at[mask, :, :, :].set(new_inverse_value)
    inverse = tuple(inverse_list)
        
    dets_list = list(dets)
    phase_val = dets_list[s][0].at[mask].set(dets_list[s][0][mask] * get_phase(det_ratio))
    log_val = dets_list[s][1].at[mask].set(dets_list[s][1][mask] + jnp.log(jnp.abs(det_ratio)))
    dets_list[s] = jnp.array([phase_val, log_val])  # JAX 배열로 통합
    dets = tuple(dets_list)
    
    return aovals, dets, inverse