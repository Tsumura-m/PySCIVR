from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import CubicSpline

from .units import Hartree


def _as_batch(positions):
    """Ensure positions have a batch axis."""
    x = np.asarray(positions, dtype=float)
    if x.ndim == 2:
        return x[None], True
    if x.ndim == 3:
        return x, False
    raise ValueError(f"positions must be (N, 3) or (B, N, 3), got {x.shape}")


##################################################################################################
# Base class for potential energy calculators
##################################################################################################

class Calculator(ABC):
    """Base class for potential energy calculators."""

    @abstractmethod
    def _energy_forces_batch(self, x):
        """"x : (B, N, 3) -> energy : (B,), forces : (B, N, 3)"""
        pass

    def _hessian_batch(self, x):
        """x : (B, N, 3) -> hessian : (B, 3N, 3N)"""
        raise NotImplementedError(f"{type(self).__name__} does not implement an analytic Hessian."
                                   "Use finite differences in hessian.py.")

    def energy_forces(self, positions):
        """positions [Angs] -> (energy [eV], forces [eV/Angs])"""
        x, is_single = _as_batch(positions)
        ener, force = self._energy_forces_batch(x)
        if is_single:
            return ener[0], force[0]
        return ener, force

    def energy(self, positions):
        return self.energy_forces(positions)[0]

    def forces(self, positions):
        return self.energy_forces(positions)[1]

    def hessian(self, positions):
        """positions [Angs] -> hessian [eV/Angs^2]"""
        x, is_single = _as_batch(positions)
        hess = self._hessian_batch(x)
        if is_single:
            return hess[0]
        return hess


# V(q) = 0
class NullPotential(Calculator):
    """Zero potential energy calculator. Forces and Hessian are zero.
    """

    def _energy_forces_batch(self, x):
        n_batch, n_atoms, _ = x.shape
        return np.zeros(n_batch), np.zeros_like(x)

    def _hessian_batch(self, x):
        n_batch, n_atoms, _ = x.shape
        return np.zeros((n_batch, n_atoms * 3, n_atoms * 3))



###################################################################################################
# hydrogen : CCSD(T)/aug-cc-pV5Z scan + cubic spline
###################################################################################################

_H2_ENENER_HARTREE = np.array([
    -0.98163485470E+00,-0.10146982768E+01,-0.10427841611E+01,-0.10666345859E+01,-0.10868635139E+01,-0.11039825385E+01,
    -0.11184206959E+01,-0.11305398917E+01,-0.11406470591E+01,-0.11490038686E+01,-0.11558343459E+01,-0.11613312863E+01,
    -0.11656611580E+01,-0.11689682309E+01,-0.11713779485E+01,-0.11729997224E+01,-0.11739292665E+01,-0.11742505492E+01,
    -0.11740374330E+01,-0.11733550521E+01,-0.11722609079E+01,-0.11708061217E+01,-0.11690358828E+01,-0.11669904242E+01,
    -0.11647055778E+01,-0.11622133277E+01,-0.11595421926E+01,-0.11567177408E+01,-0.11537628634E+01,-0.11506980943E+01,
    -0.11475418749E+01,-0.11443107844E+01,-0.11410197413E+01,-0.11376821813E+01,-0.11343102132E+01,-0.11309147566E+01,
    -0.11275056642E+01,-0.11240918298E+01,-0.11206812837E+01,-0.11172812777E+01,-0.11138983603E+01,-0.11105384316E+01,
    -0.11072068458E+01,-0.11039084015E+01,-0.11006474289E+01,-0.10974278217E+01,-0.10942530746E+01,-0.10911263171E+01,
    -0.10880503432E+01,-0.10850276381E+01,-0.10820604028E+01,-0.10791505749E+01,-0.10762998496E+01,-0.10735096919E+01,
    -0.10707813591E+01,-0.10681159100E+01])

# 0.4 ~ 1.5 Angstroms, 0.02 Angstrom spacing
_H2_DIST_ANGS = 0.4 + np.arange(len(_H2_ENENER_HARTREE)) * 0.02


class H2SplinePotential(Calculator):
    """H2 potential energy calculator using cubic spline interpolation.

    Forces and Hessian are exact derivatives of the spline:
        H_BB = V''(r) u u^T + V'(r) / r (I - u u^T),  u = (x_j - x_i) / r
    """

    def __init__(self, atom_indices=(0, 1)):
        self._spline = CubicSpline(_H2_DIST_ANGS, _H2_ENENER_HARTREE * Hartree)
        self.atom_indices = tuple(atom_indices)

    def _bond(self, x):
        i, j = self.atom_indices
        d = x[:, j] - x[:, i]
        r = np.linalg.norm(d, axis=1)
        u = d / r[:, None]
        return r, u

    def _energy_forces_batch(self, x):
        i, j = self.atom_indices
        r, u = self._bond(x)
        ener = self._spline(r)
        g_B = self._spline(r, 1)[:, None] * u   # dV/dx_j

        force = np.zeros_like(x)
        force[:, i] = g_B
        force[:, j] = -g_B
        return ener, force

    def _hessian_batch(self, x):
        i, j = self.atom_indices
        r, u = self._bond(x)
        dv = self._spline(r, 1)  # V'(r)
        ddv = self._spline(r, 2)  # V''(r)

        P = u[:, :, None] * u[:, None, :]  # u u^T
        H_BB = ddv[:, None, None] * P + (dv / r)[:, None, None] * (np.eye(3) - P)

        n_batch, n_atoms, _ = x.shape
        hess = np.zeros((n_batch, n_atoms, 3, n_atoms, 3))

        hess[:, i, :, i, :] = H_BB
        hess[:, j, :, j, :] = H_BB
        hess[:, i, :, j, :] = -H_BB
        hess[:, j, :, i, :] = -H_BB
        return hess.reshape(n_batch, n_atoms * 3, n_atoms * 3)


##################################################################################################
# ASE calculators (MACE etc.)
##################################################################################################

class ASEPotential(Calculator):
    """Wrapper for ASE calculators. Geometries in a batch are evaluated sequentially for now."""

    def __init__(self, symbols, calc):
        from ase import Atoms

        self.atoms = Atoms(symbols=symbols, positions=np.zeros((len(symbols), 3)))
        self.atoms.calc = calc

    def _energy_forces_batch(self, x):
        ener = np.empty(len(x))
        force = np.empty_like(x)
        for i, pos in enumerate(x):
            self.atoms.set_positions(pos)
            ener[i] = self.atoms.get_potential_energy()
            force[i] = self.atoms.get_forces()
        return ener, force


def _patch_torch_jit_load_cpu():
    import torch

    if hasattr(torch.jit, "__is_patched__"):
        return
    _real = torch.jit.load

    def _jit_load_cpu(f, map_location=None, *args, **kwargs):
        return _real(f, map_location=torch.device('cpu'), *args, **kwargs)
    _jit_load_cpu.__is_patched__ = True
    torch.jit.load = _jit_load_cpu

def mace_anicc_potential(symbols, device=None, num_threads=1):
    """MACE potential energy calculator using a pretrained model."""
    import torch
    from mace.calculators import mace_anicc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        _patch_torch_jit_load_cpu()
        torch.set_num_threads(num_threads)

    return ASEPotential(symbols, mace_anicc(device=device))
