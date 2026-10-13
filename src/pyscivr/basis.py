"""Coherent state overlaps for 1D coordinates with various topologies ``(R, R+, S1, I-, I+)``.

See section II E 2 or appendix H of the paper for details.

Notation:
    - ``T``: number of time steps of the trajectory.
    - ``m``: number of active coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .units import HBar

TOPOLOGIES = ("R", "R+", "S1", "I-", "I+")

def _overlap_r1(
        posi_ref: ArrayLike, mome_ref: ArrayLike, posi: ArrayLike, mome: ArrayLike,
        gamma: float) -> NDArray[np.complex128]:
    dq = posi_ref - posi
    dp = mome_ref - mome
    return np.exp(-0.25 * gamma * dq**2 - 0.25 / (gamma * HBar**2) * dp**2
                  + 0.5j / HBar * (mome_ref + mome) * dq)

def overlap_1d(
        topology: str,
        posi_ref: float,
        mome_ref: float,
        posi: ArrayLike,
        mome: ArrayLike,
        gamma: float,
        *,
        n_image: int = 3) -> NDArray[np.complex128]:
    r"""Compute the overlap :math:`\langle q_{\text{ref}}, p_{\text{ref}} | q, p \rangle` for a 1D topology.

    Args:
        topology: Topology of the 1D coordinate.
        posi_ref: Reference position.
        mome_ref: Reference momentum.
        posi: ``(T,)`` positions to evaluate the overlap at.
        mome: ``(T,)`` momenta to evaluate the overlap at.
        gamma: Width parameter of the coherent state basis.
        n_image: Number of images to consider for periodic boundary conditions.

    Returns:
        Overlap values for each position and momentum in ``posi`` and ``mome``.

    Raises:
        ValueError: If the provided topology is not recognized.

    Note:
        For ``R`` the overlap is

        .. math::

            \langle q_{\text{ref}}, p_{\text{ref}} | q, p \rangle_{\mathbb{R}^1} =
            \exp\left( -\frac{\gamma}{4} (q_{\text{ref}} - q)^2
            - \frac{1}{4 \gamma \hbar^2} (p_{\text{ref}} - p)^2
            + \frac{i}{2 \hbar} (p_{\text{ref}} + p)(q_{\text{ref}} - q) \right)

        The other topologies are built from it by image summations.
    """
    posi = np.asarray(posi, dtype=float)
    if topology == "R":
        return _overlap_r1(posi_ref, mome_ref, posi, mome, gamma)
    if topology == "R+":
        return _overlap_r1(posi_ref, mome_ref, posi, mome, gamma) \
               - _overlap_r1(-posi_ref, -mome_ref, posi, mome, gamma)

    shifts = 2.0 * np.pi * np.arange(-n_image, n_image + 1).reshape((-1,) + (1,) * posi.ndim)
    direct = np.sum(_overlap_r1(posi_ref + shifts, mome_ref, posi, mome, gamma), axis=0)

    if topology == "S1":
        return direct

    mirror = np.sum(_overlap_r1(-posi_ref + shifts, -mome_ref, posi, mome, gamma), axis=0)
    if topology == "I-":
        return direct - mirror
    if topology == "I+":
        return direct + mirror

    raise ValueError(f"Unknown topology: {topology}")

def get_overlap(
        topologies: Sequence[str],
        posi_ref: ArrayLike,
        mome_ref: ArrayLike,
        posi: ArrayLike,
        mome: ArrayLike,
        gammas: ArrayLike,
        *,
        n_image: int = 3) -> NDArray[np.complex128]:
    r"""Compute the overlap :math:`\langle q_{\text{ref}}, p_{\text{ref}} | q, p \rangle` for multiple 1D topologies.

    The total overlap is the product of the 1D overlaps for each coordinate.

    Args:
        topologies: ``(m,)`` topologies of the 1D coordinates.
        posi_ref: ``(m,)`` reference positions.
        mome_ref: ``(m,)`` reference momenta.
        posi: ``(T, m)`` positions.
        mome: ``(T, m)`` momenta.
        gammas: ``(m,)`` width parameters of the coherent state basis.
        n_image: Number of images to consider for periodic boundary conditions.

    Returns:
        ``(T,)`` overlap values for each position and momentum in ``posi`` and ``mome``.
    """
    posi = np.asarray(posi, dtype=float)
    mome = np.asarray(mome, dtype=float)
    res = np.ones(posi.shape[0], dtype=complex)

    for i, topology in enumerate(topologies):
        res *= overlap_1d(topology, posi_ref[i], mome_ref[i], posi[:, i], mome[:, i], gammas[i], n_image=n_image)

    return res
