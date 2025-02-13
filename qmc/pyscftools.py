import pyscf
import pyscf.pbc
import pyscf.mcscf
import pyscf.fci
import h5py
import json
import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple

from qmc.determinant_tools import flatten_determinants
from qmc.determinant_tools import create_packed_objects
from qmc.determinant_tools import binary_to_occ
from qmc.determinant_tools import reformat_binary_dets
from qmc.determinant_tools import jax_organize_determinant_data
from qmc.determinants import convert_to_hashable

import pyqmc.supercell as supercell
import pyqmc.twists as twists


def recover_pyscf(chkfile, ci_checkfile=None, cancel_outputs=True):
    """
    pyscf checkfile로부터 Molecule, SCF (및 필요한 경우 MC) 객체를 복구합니다.
    
    Typical usage:
        mol, mf = recover_pyscf("dft.hdf5")
        
    :param chkfile: 체크파일 이름
    :param ci_checkfile: CI 또는 MC 관련 체크파일 (선택사항)
    :param cancel_outputs: True이면 객체 내부의 출력 스트림을 제거
    :return: (mol, mf) 또는 (mol, mf, mc)
    """
    with h5py.File(chkfile, "r") as f:
        periodic = "a" in json.loads(f["mol"][()]).keys()

    if not periodic:
        mol = pyscf.lib.chkfile.load_mol(chkfile)
        with h5py.File(chkfile, "r") as f:
            mo_occ_shape = f["scf/mo_occ"].shape
        if cancel_outputs:
            mol.output = None
            mol.stdout = None
        if len(mo_occ_shape) == 2:
            mf = pyscf.scf.UHF(mol)
        elif len(mo_occ_shape) == 1:
            mf = pyscf.scf.ROHF(mol) if mol.spin != 0 else pyscf.scf.RHF(mol)
        else:
            raise Exception("Couldn't determine type from chkfile")
    else:
        mol = pyscf.pbc.lib.chkfile.load_cell(chkfile)
        with h5py.File(chkfile, "r") as f:
            has_kpts = "mo_occ__from_list__" in f["/scf"].keys()
            if has_kpts:
                rhf = "000000" in f["/scf/mo_occ__from_list__/"].keys()
            else:
                rhf = len(f["/scf/mo_occ"].shape) == 1
        if cancel_outputs:
            mol.output = None
            mol.stdout = None
        if not rhf and has_kpts:
            mf = pyscf.pbc.scf.KUHF(mol)
        elif has_kpts:
            mf = pyscf.pbc.scf.KROHF(mol) if mol.spin != 0 else pyscf.pbc.scf.KRHF(mol)
        elif rhf:
            mf = pyscf.pbc.scf.ROHF(mol) if mol.spin != 0 else pyscf.pbc.scf.RHF(mol)
        else:
            mf = pyscf.pbc.scf.UHF(mol)
    mf.__dict__.update(pyscf.scf.chkfile.load(chkfile, "scf"))

    if ci_checkfile is not None:
        casdict = pyscf.lib.chkfile.load(ci_checkfile, "ci")
        if casdict is None:
            casdict = pyscf.lib.chkfile.load(ci_checkfile, "mcscf")
        with h5py.File(ci_checkfile, "r") as f:
            hci = "ci/_strs" in f.keys()
        if hci:
            mc = pyscf.hci.SCI(mol)
        else:
            if len(casdict["mo_coeff"].shape) == 3:
                mc = pyscf.mcscf.UCASCI(mol, casdict["ncas"], casdict["nelecas"])
            else:
                mc = pyscf.mcscf.CASCI(mol, casdict["ncas"], casdict["nelecas"])
        mc.__dict__.update(casdict)
        return mol, mf, mc
    return mol, mf


class OrbitalEvaluatorResult(NamedTuple):
    max_orb: jnp.ndarray
    detcoeff: jnp.ndarray
    det_map: jnp.ndarray
    mo_coeff: jnp.ndarray
    occup: tuple
    elec_n : tuple
           
def orbital_evaluator_from_pyscf(
    mol, mf, mc=None, twist=0, determinants=None, tol=None, eval_gto_precision=None,
    evaluate_orbitals_with="pyscf",
):
    """
    mol: Molecule 객체
    mf: pyscf mean-field 객체
    mc: pyscf multiconfigurational 객체 (HCI 또는 CAS 지원)
    twist: 계산의 twist (필요한 경우)
    determinants: 결정자 리스트 (없으면 자동 생성)
    tol: 가장 작은 결정자 weight (이보다 작은 것은 버림)
    eval_gto_precision: (선택사항)
    
    :returns: OrbitalEvaluatorResult(NamedTuple)
        - detcoeff: 각 결정자의 weight 배열
        - occup: 각 결정자에 해당하는 오비탈 점유 정보
        - det_map: 결정자와 occup 간의 매핑 정보
    """
    periodic = hasattr(mol, "a")
    # f_max_orb: 입력 배열 a에 대해, 요소가 있으면 최대값+1, 없으면 0
    f_max_orb = lambda a: int(jnp.max(jnp.array(a), initial=0)) + 1 if len(a) > 0 else 0
    
    tol = -1 if tol is None else tol
    
    if periodic:
        mf = pyscf.pbc.scf.addons.convert_to_khf(mf)

    try:
        mf = mf.to_uhf()
    except TypeError:
        mf = mf.to_uhf(mf)
     
    if determinants is None:
        determinants = determinants_from_pyscf(mol, mf, mc=mc, tol=tol)


    # MO 계수 결정: mc가 있으면 mc.mo_coeff, 없으면 mf.mo_coeff 사용
    if hasattr(mc, "mo_coeff"):
        _mo_coeff = mc.mo_coeff
        if len(_mo_coeff.shape) == 2:  # restricted spin: up, down 동일 처리
            _mo_coeff = [_mo_coeff, _mo_coeff]
        if periodic:
            _mo_coeff = [m[np.newaxis] for m in _mo_coeff]  # k-point 차원 추가
    else:
        _mo_coeff = mf.mo_coeff

    if periodic:
        if not hasattr(mol, "original_cell"):
            mol = supercell.get_supercell(mol, np.eye(3))
        kinds = twists.create_supercell_twists(mol, mf)["primitive_ks"][twist]
        if len(kinds) != mol.scale:
            raise ValueError(f"Found {len(kinds)} k-points but should have found {mol.scale}.")
        kpts = mf.kpts[kinds]
        max_orb = [[[f_max_orb(k) for k in s] for s in det] for wt, det in determinants]
        max_orb = np.amax(max_orb, axis=0)
        mo_coeff = [
            [_mo_coeff[s][k][:, 0 : max_orb[s][k]] for k in kinds] for s in [0, 1]
        ]

        determinants = flatten_determinants(
            determinants, max_orb, kinds
        )
    else:
        max_orb = [[f_max_orb(s) for s in det] for wt, det in determinants]
        max_orb = np.amax(max_orb, axis=0)
        mo_coeff = [_mo_coeff[spin][:, 0 : max_orb[spin]] for spin in [0, 1]]

    
    detcoeff, occup, det_map = jax_organize_determinant_data(determinants, tol)
 
    max_orb = jnp.array(max_orb)
    # detcoeff = jnp.array(detcoeff)
    # det_map = jnp.array(det_map)
    mo_coeff = jnp.array(mo_coeff)

    elec_n = mol.nelec
    occup = convert_to_hashable(occup)
    
    return OrbitalEvaluatorResult(max_orb, detcoeff, det_map,
                                  mo_coeff, occup, elec_n)

def determinants_from_pyscf(mol, mf, mc=None, tol=-1):
    periodic = hasattr(mol, "a")
    if mc is None:
        determinants = single_determinant_from_mf(mf)
    elif periodic:
        determinants = pbc_determinants_from_casci(mc, cutoff=tol)
    if mc is not None and not periodic:
        determinants = interpret_ci(mc, tol)
    return determinants

def single_determinant_from_mf(mf, weight=1.0):
    """
    SCF 객체로부터 단일 결정자 리스트를 생성합니다.
    """
    try:
        mf = mf.to_uhf()
    except TypeError:
        mf = mf.to_uhf(mf)
    if hasattr(mf, "kpts"):
        occupation = [[list(np.nonzero(k > 0.5)[0]) for k in s] for s in mf.mo_occ]
    else:
        occupation = [list(np.nonzero(s > 0.5)[0]) for s in mf.mo_occ]
    return [(weight, occupation)]

def pbc_determinants_from_casci(mc, orbitals=None, cutoff=0.05):
    if hasattr(mc.ncore, "__len__"):
        nocc = [c + e for c, e in zip(mc.ncore, mc.nelecas)]
    else:
        nocc = [mc.ncore + e for e in mc.nelecas]
    if orbitals is None:
        orbitals = np.arange(mc.ncore, mc.ncore + mc.ncas)
    if not hasattr(orbitals[0], "__len__"):
        orbitals = [orbitals, orbitals]
    deters = pyscf.fci.addons.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=-1)
    determinants = []
    for x in deters:
        if abs(x[0]) > cutoff:
            allorbs = [
                [translate_occ(x[1], orbitals[0], nocc[0])],
                [translate_occ(x[2], orbitals[1], nocc[1])],
            ]
            determinants.append((x[0], allorbs))
    return determinants

def translate_occ(x, orbitals, nocc):
    a = binary_to_occ(x, 0)[0]
    orbitals_without_active = list(range(nocc))
    for o in orbitals:
        if o in orbitals_without_active:
            orbitals_without_active.remove(o)
    return orbitals_without_active + [orbitals[i] for i in a]

def interpret_ci(mc, tol):
    """
    MC 객체로부터 결정자 계수와 MO 점유 정보를 추출합니다.
    
    :returns: 재구성된 결정자 데이터 (detwt, occup, map_dets)
    """
    ncore = mc.ncore if hasattr(mc, "ncore") else 0
    if hasattr(mc, "_strs"):
        deters = deters_from_hci(mc, tol)
    else:
        deters = pyscf.fci.addons.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=-1)
    return reformat_binary_dets(deters, ncore=ncore, tol=tol)

def deters_from_hci(mc, tol):
    bigcis = np.abs(mc.ci) > tol
    nstrs = int(mc._strs.shape[1] / 2)
    deters = []
    for c, s in zip(mc.ci[bigcis], mc._strs[bigcis, :]):
        s1 = "".join(str(bin(p)).replace("0b", "") for p in s[0:nstrs])
        s2 = "".join(str(bin(p)).replace("0b", "") for p in s[nstrs:])
        deters.append((c, s1, s2))
    return deters