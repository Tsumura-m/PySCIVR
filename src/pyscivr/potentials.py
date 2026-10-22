"""Potential energy calculators.

All calculators take positions in Angstrom and return energies in eV,
forces in eV/Angs and Hessians in eV/Angs^2.

Notation:
    - ``N``: number of atoms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline

from .units import Bohr, Hartree

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator as ASECalculator


def _as_batch(positions: ArrayLike) -> tuple[NDArray[np.float64], bool]:
    """Ensure positions have a batch axis.

    Args:
        positions: ``(N, 3)`` or ``(n_batch, N, 3)`` positions.

    Returns:
        ``(x, is_single)``, where ``x`` is ``(n_batch, N, 3)`` positions and
        ``is_single`` is ``True`` if a batch axis was added.

    Raises:
        ValueError: If ``positions`` is neither 2D nor 3D.
    """
    x = np.asarray(positions, dtype=float)
    if x.ndim == 2:
        return x[None], True
    if x.ndim == 3:
        return x, False
    raise ValueError(f"positions must be (N, 3) or (n_batch, N, 3), got {x.shape}")


##################################################################################################
# Base class for potential energy calculators
##################################################################################################


class Calculator(ABC):
    """Base class for potential energy calculators.

    Subclasses implement :meth:`_energy_forces_batch` and optionally :meth:`_hessian_batch`
    for a batch of geometries. The public methods accept either a single geometry ``(N, 3)``
    or a batch of geometries ``(n_batch, N, 3)``, and return the results without the batch axis
    for a single geometry.
    """

    @abstractmethod
    def _energy_forces_batch(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute the energies and forces for a batch of geometries.

        Args:
            x: ``(n_batch, N, 3)`` positions [Angs].

        Returns:
            ``(energy, forces)``, with shape ``(n_batch,)`` [eV] and ``(n_batch, N, 3)`` [eV/Angs].
        """
        pass

    def _hessian_batch(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the Hessians for a batch of geometries.

        Args:
            x: ``(n_batch, N, 3)`` positions [Angs].

        Returns:
            ``(n_batch, 3N, 3N)`` Hessians [eV/Angs^2].

        Raises:
            NotImplementedError: If the subclass does not implement an analytic Hessian.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement an analytic Hessian. Use finite differences in hessian.py."
        )

    def energy_forces(self, positions: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute the energy and forces for a single geometry or a batch of geometries.

        Args:
            positions: ``(N, 3)`` or ``(n_batch, N, 3)`` positions [Angs].

        Returns:
            ``(energy, forces)`` [eV, eV/Angs], with shapes ``()`` and ``(N, 3)``
            for a single geometry, or ``(n_batch,)`` and ``(n_batch, N, 3)`` for a batch.
        """
        x, is_single = _as_batch(positions)
        ener, force = self._energy_forces_batch(x)
        if is_single:
            return ener[0], force[0]
        return ener, force

    def energy(self, positions: ArrayLike) -> NDArray[np.float64]:
        """Compute the energy [eV] (see :meth:`energy_forces`)."""
        return self.energy_forces(positions)[0]

    def forces(self, positions: ArrayLike) -> NDArray[np.float64]:
        """Compute the forces [eV/Angs] (see :meth:`energy_forces`)."""
        return self.energy_forces(positions)[1]

    def hessian(self, positions: ArrayLike) -> NDArray[np.float64]:
        """Compute the analytical Hessian.

        Args:
            positions: ``(N, 3)`` or ``(n_batch, N, 3)`` positions [Angs].

        Returns:
            ``(3N, 3N)`` Hessian [eV/Angs^2] for a single geometry,
            or ``(n_batch, 3N, 3N)`` for a batch.

        Raises:
            NotImplementedError: If the subclass does not implement an analytic Hessian.
                Use :func:`pyscivr.hessian.eval_hess_active` instead.
        """
        x, is_single = _as_batch(positions)
        hess = self._hessian_batch(x)
        if is_single:
            return hess[0]
        return hess


# V(q) = 0
class ZeroPotential(Calculator):
    """Zero potential. The energies, forces and Hessians are all zero."""

    def _energy_forces_batch(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n_batch, n_atoms, _ = x.shape
        return np.zeros(n_batch), np.zeros_like(x)

    def _hessian_batch(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        n_batch, n_atoms, _ = x.shape
        return np.zeros((n_batch, n_atoms * 3, n_atoms * 3))


###################################################################################################
# hydrogen: CCSD(T)/aug-cc-pV5Z Scan + Cubic spline
# Computed with Gaussian 16, Revision A.03, M. J. Frisch et al., Gaussian, Inc., Wallingford CT, 2016.
###################################################################################################

_H2_ENERGY_HARTREE = np.array(
    [
        -0.98163485470e00,
        -0.10146982768e01,
        -0.10427841611e01,
        -0.10666345859e01,
        -0.10868635139e01,
        -0.11039825385e01,
        -0.11184206959e01,
        -0.11305398917e01,
        -0.11406470591e01,
        -0.11490038686e01,
        -0.11558343459e01,
        -0.11613312863e01,
        -0.11656611580e01,
        -0.11689682309e01,
        -0.11713779485e01,
        -0.11729997224e01,
        -0.11739292665e01,
        -0.11742505492e01,
        -0.11740374330e01,
        -0.11733550521e01,
        -0.11722609079e01,
        -0.11708061217e01,
        -0.11690358828e01,
        -0.11669904242e01,
        -0.11647055778e01,
        -0.11622133277e01,
        -0.11595421926e01,
        -0.11567177408e01,
        -0.11537628634e01,
        -0.11506980943e01,
        -0.11475418749e01,
        -0.11443107844e01,
        -0.11410197413e01,
        -0.11376821813e01,
        -0.11343102132e01,
        -0.11309147566e01,
        -0.11275056642e01,
        -0.11240918298e01,
        -0.11206812837e01,
        -0.11172812777e01,
        -0.11138983603e01,
        -0.11105384316e01,
        -0.11072068458e01,
        -0.11039084015e01,
        -0.11006474289e01,
        -0.10974278217e01,
        -0.10942530746e01,
        -0.10911263171e01,
        -0.10880503432e01,
        -0.10850276381e01,
        -0.10820604028e01,
        -0.10791505749e01,
        -0.10762998496e01,
        -0.10735096919e01,
        -0.10707813591e01,
        -0.10681159100e01,
    ]
)

# 0.4 ~ 1.5 Angstroms, 0.02 Angstrom spacing
_H2_DIST_ANGS = 0.4 + np.arange(len(_H2_ENERGY_HARTREE)) * 0.02


class H2SplinePotential(Calculator):
    r"""H2 potential by cubic spline interpolation of CCSD(T)/aug-cc-pV5Z scan computed with Gaussian 16.

    The potential :math:`V(r)` depends only on the bond length :math:`r` between the two atoms.
    The forces and Hessians are the exact derivatives of the spline,
    with the Hessian blocks ``H_ii = H_jj = H_BB`` and ``H_ij = H_ji = -H_BB``, where

    .. math::

        H_{BB} = V''(r) \hat{u}\hat{u}^T + \frac{V'(r)}{r} (I - \hat{u}\hat{u}^T), \quad
        \hat{u} = \frac{q_j - q_i}{r}

    Warning:
        The scan covers only 0.4 to 1.5 Angs (0.02 Angs spacing).
        The spline is not reliable outside this range.
    """

    def __init__(self) -> None:
        self._spline = CubicSpline(_H2_DIST_ANGS, _H2_ENERGY_HARTREE * Hartree)
        self.atom_indices = tuple((0, 1))

    def _bond(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        i, j = self.atom_indices
        d = x[:, j] - x[:, i]
        r = np.linalg.norm(d, axis=1)
        u = d / r[:, None]
        return r, u

    def _energy_forces_batch(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        i, j = self.atom_indices
        r, u = self._bond(x)
        ener = self._spline(r)
        g_B = self._spline(r, 1)[:, None] * u  # dV/dx_j

        force = np.zeros_like(x)
        force[:, i] = g_B
        force[:, j] = -g_B
        return ener, force

    def _hessian_batch(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
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
    """Wrapper for ASE calculators.

    Geometries in a batch are evaluated sequantially.
    The analytical Hessian is not available.

    Args:
        symbols: ``(N,)`` atomic symbols.
        calc: ASE calculator.
    """

    def __init__(self, symbols: Sequence[str], calc: ASECalculator) -> None:
        from ase import Atoms

        self.atoms = Atoms(symbols=symbols, positions=np.zeros((len(symbols), 3)))
        self.atoms.calc = calc

    def _energy_forces_batch(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ener = np.empty(len(x))
        force = np.empty_like(x)
        for i, pos in enumerate(x):
            self.atoms.set_positions(pos)
            ener[i] = self.atoms.get_potential_energy()
            force[i] = self.atoms.get_forces()
        return ener, force


def _patch_torch_jit_load_cpu() -> None:
    """Patch torch.jit.load to always load on CPU."""
    import torch

    if hasattr(torch.jit.load, "__is_patched__"):
        return
    _real = torch.jit.load

    def _jit_load_cpu(f, map_location=None, *args, **kwargs):
        return _real(f, map_location=torch.device("cpu"), *args, **kwargs)

    _jit_load_cpu.__is_patched__ = True
    torch.jit.load = _jit_load_cpu


def mace_anicc_potential(symbols: Sequence[str], *, device: str | None = None, num_threads: int = 1) -> ASEPotential:
    """Make a calculator with the pre-trained MACE-ANICC model.

    Args:
        symbols: ``(N,)`` atomic symbols.
        device: Device to run the model on. If ``None``, use GPU if available, otherwise CPU.
        num_threads: Number of threads to use for CPU inference.

    Returns:
        Calculator wrapping the MACE-ANICC model.

    Note:
        Requires the ``torch``, ``mace`` and ``ase`` packages.
        On the CPU, ``torch.jit.load`` is patched to load the model onto the CPU.
    """
    import torch
    from mace.calculators import mace_anicc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        _patch_torch_jit_load_cpu()
        torch.set_num_threads(num_threads)

    return ASEPotential(symbols, mace_anicc(device=device))


##################################################################################################
# PySCF DFT calculator
##################################################################################################


class DFTPotential(Calculator):
    """Potential by DFT calculations with PySCF.

    Geometries in a batch are evaluated sequentially.
    The analytical Hessian is not available.

    Args:
        symbols: ``(N,)`` atomic symbols.
        basis: Basis set.
        xc: Exchange-correlation functional.
        charge: Total charge.
        spin: Number of unpaired electrons (``2S``). Restricted Kohn-Sham is used
            if ``spin == 0``, otherwise unrestricted Kohn-Sham.
        conv_tol: Convergence threshold of the SCF energy [Hartree].
    """

    def __init__(
        self,
        symbols: Sequence[str],
        *,
        basis: str = "def2-svp",
        xc: str = "b3lyp",
        charge: int = 0,
        spin: int = 0,
        conv_tol: float = 1e-10,
    ) -> None:
        self.symbols = list(symbols)
        self.basis = basis
        self.xc = xc
        self.charge = charge
        self.spin = spin
        self.conv_tol = conv_tol

    def _energy_forces_batch(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        from pyscf import dft, gto

        ener = np.empty(len(x))
        force = np.empty_like(x)
        for i, pos in enumerate(x):
            mol = gto.M(
                atom=[(s, tuple(p)) for s, p in zip(self.symbols, pos)],
                basis=self.basis,
                charge=self.charge,
                spin=self.spin,
                unit="Angstrom",
                verbose=0,
            )
            mf = dft.RKS(mol, xc=self.xc) if self.spin == 0 else dft.UKS(mol, xc=self.xc)
            mf.conv_tol = self.conv_tol
            ener[i] = mf.kernel() * Hartree  # Hartree -> eV
            force[i] = -mf.nuc_grad_method().kernel() * Hartree / Bohr  # Hartree/Bohr -> eV/Angs
        return ener, force

class ArgonPotential(Calculator):
  """Wrapper for Argon (Unpublished) calculators.

  Geometries in a batch are evaluated sequantially.
  The analytical Hessian is not available.

  Args:
    symbols: ``(N,)`` atomic symbols.
    calc: Argon calculator.
  """

  def __init__(self, calc, root=None, fname_qc_base=None):
    self.calc = calc
    self.root = root
    self.fname_qc_base = fname_qc_base

  def _energy_forces_batch(self, x):
    ener = np.empty(len(x))
    force = np.empty_like(x)
    for i, pos, in enumerate(x):
      ener[i] = self.calc.get_energy(self.root, self.fname_qc_base, pos)
      force[i] = -self.calc.get_gradient(self.root, self.fname_qc_base, pos)
    return ener, force

