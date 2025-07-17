import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from qmc.orbitals import aos
from qmc.orbitals import mos

def convert_to_hashable(occup):
    
    up_orbs = tuple(occup[0])
    dn_orbs = tuple(occup[1])
    
    return (up_orbs, dn_orbs)

def convert_to_numpy(occup):
    
    up_orbs = np.array(occup[0])
    dn_orbs = np.array(occup[1])
    
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

@partial(jax.jit, static_argnums=(2,))
def recompute_jit(aovals, mo_coeff, _nelec, occup_hash):
    
    updets, upinverse = compute_determinants(mos(aovals[:, :, :_nelec[0]], mo_coeff[0]), occup_hash, 0)
    dndets, dninverse = compute_determinants(mos(aovals[:, :, _nelec[0]:], mo_coeff[1]), occup_hash, 1)
    
    return aovals, (updets, dndets), (upinverse, dninverse)

def recompute(mol, configs, mo_coeff, _nelec, occup_hash):
    
    nconf, nelec_tot, ndim = configs.shape
    atomic_orbitals = aos(mol,"GTOval_sph", configs)
    aovals = atomic_orbitals.reshape(-1, nconf, nelec_tot, atomic_orbitals.shape[-1])

    return recompute_jit(aovals, mo_coeff, _nelec, occup_hash)

@jax.jit
def compute_wf_value(configs, dets, det_coeff, det_map):
    
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

@partial(jax.jit, static_argnums=(1,))
def gradient_value_jit(e, s, aograd, epos, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash):
    # ∂Φᵢ/∂r = Σₖ cᵢₖ ∂φₖ/∂r -> (4, config, number_of_electrons)
    mograd = mos(aograd, mo_coeff[s])
    
    # (4, config, 1, number_of_electrons)
    mograd_vals = mograd[:, :, occup_hash[s]]
    
    ratio = _testrow_deriv(e, mograd_vals, inverse, s, dets, det_coeff, det_map, _nelec)
    
    derivatives = ratio[1:] / ratio[0]
    derivatives = jnp.where(jnp.isfinite(derivatives), derivatives, 0.0)
    
    values = ratio[0]
    values = jnp.where(jnp.isfinite(values), values, 1.0)
    
    return derivatives, values, (aograd[:, 0], mograd[0])

def gradient_value(mol, e, epos, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash):
    s = int(e >= _nelec[0])
    
    aograd = aos(mol, "GTOval_sph_deriv1", epos)
    
    return gradient_value_jit(e, s, aograd, epos, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash)

@partial(jax.jit, static_argnums=(1,))
def gradient_laplacian_jit(e, s, ao, epos, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash):
    mo = mos(ao, mo_coeff[s])
    mo_vals = mo[..., occup_hash[s]]

    ratio = _testrow_deriv(e, mo_vals, inverse, s, dets, det_coeff, det_map, _nelec)

    ratio = ratio / ratio[:1]

    return ratio[1:-1], ratio[-1]

def gradient_laplacian(mol, e,  epos, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash):
    
    s = int(e >= _nelec[0])
    
    ao = aos(mol, "GTOval_sph_deriv2", epos)
    
    return gradient_laplacian_jit(e, s, ao, epos, dets, inverse, mo_coeff, det_coeff, det_map, _nelec, occup_hash)

@partial(jax.jit, static_argnums=(5,))
def _compute_det_ratio(e, s, vec, dets, inverse_s, det_map, det_coeff, _nelec):

    e_eff = e - s * _nelec[0]
    
    ratios = jnp.einsum("ei...dj, idj... ->ei...d",
                    vec,
                    inverse_s[..., e_eff])
    
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
    
    return numer, denom

def _testrow_deriv(e, vec, inverse, s, dets, det_coeff, det_map, _nelec):
    
    numer, denom = _compute_det_ratio(
        e, s, vec, dets, inverse[s], det_map, det_coeff, _nelec
    )
    
    if len(numer.shape) == 3:
        denom = denom[jnp.newaxis, :, jnp.newaxis]

    return numer / denom

@jax.jit
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

def safe_mask_indexing(mask):
    return jnp.where(mask)[0]

def sherman_morrison(e, epos, configs, mask, aovals, saved_value, get_phase, dets, inverse, mo_coeff, occup_hash, _nelec):
    s = jnp.where(e >= _nelec[0], 1, 0)
    eeff = e - s * _nelec[0]

    ao, mo = saved_value
    indices = safe_mask_indexing(mask)
    mo = mo[indices]
    mo_vals = mo[:, occup_hash[s]]

    det_ratio, new_inverse_value = sherman_morrison_ms(eeff, inverse[s][indices], mo_vals)

    inverse_list = [inverse[0], inverse[1]]
    dets_list = [dets[0], dets[1]]

    aovals = aovals.at[:, indices, e, :].set(ao[:, indices])
    inverse_list[s] = inverse[s].at[indices, :, :, :].set(new_inverse_value)

    phase_val = dets_list[s][0].at[indices].set(dets_list[s][0][indices] * get_phase(det_ratio))
    log_val = dets_list[s][1].at[indices].set(dets_list[s][1][indices] + jnp.log(jnp.abs(det_ratio)))
    dets_list[s] = jnp.array([phase_val, log_val])

    inverse = tuple(inverse_list)
    dets = tuple(dets_list)

    return aovals, dets, inverse

import jax
import jax.numpy as jnp

# JAX를 64비트 부동소수점으로 설정
jax.config.update("jax_enable_x64", True)

# einsum 연산만을 위한 JIT 컴파일된 함수
@jax.jit
def compute_col_derivative(ao_vals, inverse_slice):
    """
    원자 궤도함수 값과 역행렬 요소를 사용하여 열 파생값을 계산합니다.
    
    Args:
        ao_vals: 원자 궤도함수 값 (shape: [nconf, nao, ...])
        inverse_slice: 역행렬 요소 (shape: [nconf, nao, ...])
        
    Returns:
        열 파생값 (shape: [nconf, ...])
    """
    return jnp.einsum(
        "ij...,ij->i...", 
        ao_vals, 
        inverse_slice,
        optimize="greedy"
    )

# 이제 메인 함수 내에서 JIT 컴파일된 함수 사용
def parameter_gradient(configs, aovals, dets, det_coeff, mo_coeff, det_map, occup_hash, _nelec, inverse):
    curr_val = compute_wf_value(configs, dets, det_coeff, det_map)
    nonzero = curr_val[0] != 0.0
    
    # det_map에 따라 행렬식 매핑
    dets_mapped = (
        dets[0][:, :, det_map[0]],
        dets[1][:, :, det_map[1]],
    )
    
    # 행렬식 계수에 대한 미분 계산
    d = {}
    d["det_coeff"] = jnp.zeros(dets_mapped[0].shape[1:], dtype=dets_mapped[0].dtype)
    d["det_coeff"] = d["det_coeff"].at[nonzero, :].set(
        dets_mapped[0][0, nonzero, :]
        * dets_mapped[1][0, nonzero, :]
        * jnp.exp(
            dets_mapped[0][1, nonzero, :]
            + dets_mapped[1][1, nonzero, :]
            - jnp.array(curr_val[1][nonzero, jnp.newaxis])
        )
        / jnp.array(curr_val[0][nonzero, jnp.newaxis])
    )
    
    # 분자 궤도 계수에 대한 미분 계산
    for s, parm in zip([0, 1], ["mo_coeff_alpha", "mo_coeff_beta"]):
        ao = aovals[:, :, s * _nelec[0] : _nelec[s] + s * _nelec[0], :]
        split = [mo_coeff[s].shape[1]]
        aos = ao
        mos = jnp.split(jnp.arange(split[-1]), jnp.asarray(split).astype(int))
        
        nao = aos[0].shape[-1]
        nconf = aos[0].shape[0]
        nmo = int(split[-1])
        
        deriv = jnp.zeros((len(occup_hash[s]), nconf, nao, nmo), dtype=curr_val[0].dtype)
        
        for det_idx in range(len(occup_hash[s])):
            occ = occup_hash[s][det_idx]  # JAX 배열
            
            for ao_idx, (ao_vals, mo_indices) in enumerate(zip(aos, mos)):
                for i in mo_indices:
                    # JAX 배열에서 i가 있는지 확인
                    is_in_occ = jnp.any(occ == i)
                    
                    if is_in_occ:
                        # JAX에서 인덱스 찾기
                        col = jnp.argmax(occ == i).item()  # 스칼라로 변환
                        
                        # JIT 컴파일된 einsum 함수 사용
                        col_deriv = compute_col_derivative(
                            ao_vals, 
                            inverse[s][:, det_idx, col, :]
                        )
                        
                        deriv = deriv.at[det_idx, :, :, i].set(col_deriv)
        
        # 행렬식에 대한 축소
        d[parm] = jnp.zeros(deriv.shape[1:], dtype=curr_val[0].dtype)
            
        for di, coeff in enumerate(det_coeff):
            whichdet = det_map[s][di]
            d[parm] = d[parm] + (
                deriv[whichdet]
                * coeff
                * d["det_coeff"][:, di, jnp.newaxis, jnp.newaxis]
            )
    
    
    for k in list(d.keys()):
        if jnp.prod(jnp.array(d[k].shape)) == 0:
            del d[k]
            
    return d