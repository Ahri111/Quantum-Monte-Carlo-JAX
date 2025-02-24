import pyscf
from pyscf import gto, scf, mcscf
from pyqmc.api import Slater
import pyqmc.api as pyq
import numpy as np
from pyqmc.api import vmc
from pyqmc.energy import kinetic

def limdrift(g, cutoff=1):
    """
    Limit a vector to have a maximum magnitude of cutoff while maintaining direction

    :parameter g: a [nconf,ndim] vector
    :parameter cutoff: the maximum magnitude
    :returns: The vector with the cutoff applied.
    """
    tot = np.linalg.norm(g, axis=1)
    mask = tot > cutoff
    g[mask, :] = cutoff * g[mask, :] / tot[mask, np.newaxis]
    return g

# 물 분자 정의
np.random.seed(42)

mol = gto.Mole()
mol.atom = '''
O 0.000000 0.000000 0.117790
H 0.000000 0.755453 -0.471161
H 0.000000 -0.755453 -0.471161
'''
mol.basis = 'sto-3g'
mol.build()

mf = scf.RHF(mol)
mf.kernel()

nconfig = 10
configs = pyq.initial_guess(mol, nconfig)
# # CASSCF 계산
# # 6개 궤도함수에 6개 전자를 넣고 계산
# mc = mcscf.CASSCF(mf, ncas=6, nelecas=6)
# mc.kernel()
# coords = configs.configs
wf = Slater(mol, mf)
nconf, nelec, _ = configs.configs.shape
block_avg = {}
wf.recompute(configs)
nsteps = 1
tstep = 0.5
equilibration_step = 800

np.random.seed(42)

for _ in range(equilibration_step):
    acc = 0.0
    for e in range(nelec):
        # Propose move
        g, _, _ = wf.gradient_value(e, configs.electron(e))
        grad = limdrift(np.real(g.T))
        gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
        newcoorde = configs.configs[:, e, :] + gauss + grad * tstep
        newcoorde = configs.make_irreducible(e, newcoorde)

        # Compute reverse move
        g, new_val, saved = wf.gradient_value(e, newcoorde)
        new_grad = limdrift(np.real(g.T))
        forward = np.sum(gauss**2, axis=1)
        backward = np.sum((gauss + tstep * (grad + new_grad)) ** 2, axis=1)

        # Acceptance
        t_prob = np.exp(1 / (2 * tstep) * (forward - backward))
        ratio = np.abs(new_val) ** 2 * t_prob
        accept = ratio > np.random.rand(nconf)
        # Update wave function
        configs.move(e, newcoorde, accept)
        wf.updateinternals(e, newcoorde, configs, mask=accept, saved_values=saved)
        acc += np.mean(accept) / nelec
print(acc)

# wf.recompute(configs)
# # print(wf._inverse[0])
# # print(wf.recompute(configs))
# tstep = 0.5
# nconf, nelec, _ = configs.configs.shape
# e = 0

# acc = 0
# g, _, _ = wf.gradient_value(e, configs.electron(e))
# grad = limdrift(np.real(g.T))
# np.random.seed(42)
# gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
# newcoorde = configs.configs[:, e, :] + gauss + grad * tstep
# newcoorde = configs.make_irreducible(e, newcoorde)
# print(newcoorde.configs)

# # print("--------------------")
# g, new_val, saved = wf.gradient_value(e, newcoorde)
# # print(g)

# new_grad = limdrift(np.real(g.T))
# forward = np.sum(gauss**2, axis=1)
# backward = np.sum((gauss + tstep * (grad + new_grad)) ** 2, axis=1)

# t_prob = np.exp(1 / (2 * tstep) * (forward - backward))
# ratio = np.abs(new_val) ** 2 * t_prob
# accept = ratio > np.random.rand(nconf)
# # print(accept)

# configs.move(e, newcoorde, accept)
# wf.updateinternals(e, newcoorde, configs, mask=accept, saved_values=saved)
# acc += np.mean(accept) / nelec
# # print(wf._dets)
# print(wf._inverse[0])