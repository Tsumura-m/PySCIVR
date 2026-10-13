"""Hessian evaluation in the active internal coordinates along a trajectory.

See section II C of the paper for details.

Notation:
    - ``N``: number of atoms.
    - ``T``: number of time steps of the trajectory.
    - ``S``: number of steps at which the Hessian is evaluated.
    - ``m``: number of active coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from .coordinates import InternalCoordinates
    from .potentials import Calculator


def eval_hess_active(
        calculator: Calculator,
        coord_all: InternalCoordinates,
        active_idx: ArrayLike,
        posi: ArrayLike,
        steps: ArrayLike,
        *,
        eps: float = 1e-4,
        chunk_size: int | None = 100,
        rcond: float = 1e-10
        )-> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Evaluate directional derivative of the gradient at selected steps of a trajectory.

    Computes :math:`K A_A \equiv \nabla^2 V A_A` (Eq. (52) in the paper), where

    .. math::

        A_A = \frac{dq}{dQ_A}, \quad B_A = \frac{dQ_A}{dq}


    Args:
        calculator: Calculator whose ``energy_forces`` accepts ``(n_batch, N, 3)`` geometries
            and returns ``(energy, forces)``.
        coord_all: InternalCoordinates for the full system. It must define ``3N``
            coordinates for coordinate transformation.
        active_idx: ``(m,)`` row indices of the active coordinates in ``coord_all``, in the row order
            after ``build()`` (e.g. ``InternalCoordinates.find_coordinate`` or
            ``coord_all.inv_argsort[active_idx_original]``).
        posi: ``(T, N, 3)`` trajectory.
        steps: ``(S,)`` indices of the steps to evaluate.
        eps: Displacement [Angs].
        chunk_size: Number of geometries passed to the calculator at once
            If ``None``, all geometries are passed at once.
        rcond: Relative cutoff for the pseudo-inverse.

    Returns:
        ``(KA, AA, BA)``, where

        - ``KA``: ``(S, 3N, m)``   :math:`K A_A`
        - ``AA``: ``(S, 3N, m)``   :math:`A_A`
        - ``BA``: ``(S, m, 3N)``   :math:`B_A`

    Note:
        :math:`A_A` is decomposed as :math:`A_A = Q R` via QR decomposition, the gradient is differentiated
        along the orthonormal columns of :math:`Q` by central differences with
        step ``eps``, and :math:`K A_A` is obtained as :math:`(K Q) R`. This requires
        :math:`2 S m` gradient evaluations.

    Warning:
        The obtained :math:`K A_A` can be numerically unstable near singularities
        (i.e., where the active coordinates are not independent).
        It is recommended to check the trajectory in advance and choose ``steps`` that avoid such regions.
    """
    steps = np.asarray(steps, dtype=int)
    posi = np.asarray(posi, dtype=float)
    n_atom = posi.shape[1]
    x = np.asarray(posi[steps], dtype=float).reshape(len(steps), -1)  # (S, 3N)
    const_grad_func_batch = coord_all.get_const_grad_batch # (B, 3N) -> (B, 3N, 3N)


    B = np.asarray(const_grad_func_batch(x)) # (S, 3N, 3N)
    assert B.shape[1] == B.shape[2] == n_atom * 3, \
        f"B must have 3N = {n_atom * 3} rows, but has {B.shape[1]} rows."

    A = np.linalg.pinv(B, rcond=rcond)  # (S, 3N, 3N)
    idx_A = np.asarray(active_idx, dtype=int)
    AA = A[:, :, idx_A]  # (S, 3N, m)
    BA = B[:, idx_A, :]  # (S, m, 3N)

    Q, R = np.linalg.qr(AA)  # (S, 3N, m), (S, m, m)
    disp = eps * Q.transpose(0, 2, 1)  # (S, m, 3N)
    x_disp = np.stack([x[:, None, :] + disp, x[:, None, :] - disp]) # (2, S, m, 3N)

    geoms = x_disp.reshape(-1, n_atom, 3)  # (2*S*m, N, 3)
    if chunk_size is None:
        chunk_size = len(geoms)

    force = np.concatenate([calculator.energy_forces(geoms[i:i + chunk_size])[1]
                            for i in range(0, len(geoms), chunk_size)])  # (2*S*m, N, 3)
    grad = -force.reshape(x_disp.shape)  # (2, S, m, 3N)

    KQ = ((grad[0] - grad[1]) / (2 * eps)).transpose(0, 2, 1)  # (S, 3N, m)
    KA = KQ @ R
    return KA, AA, BA

def projected_hessian(
        KA: NDArray[np.float64],
        AA: NDArray[np.float64],
        BA: NDArray[np.float64],
        *,
        symmetrize: bool = True) -> NDArray[np.float64]:
    r"""Compute the Hessian projected onto the active coordinates.

    Eq. (50) in the paper:

    .. math::
        \mathfrak{P}^T K P = \mathfrak{P}^T (K A_A) B_A, \quad \mathfrak{P} = A_A B_A

    Args:
        KA: ``(S, 3N, m)`` ``K A_A`` from :func:`eval_hess_active`.
        AA: ``(S, 3N, m)`` ``A_A`` from :func:`eval_hess_active`.
        BA: ``(S, m, 3N)`` ``B_A`` from :func:`eval_hess_active`.
        symmetrize: If ``True``, the Hessian is symmetrized.

    Returns:
        ``(S, 3N, 3N)`` projected Hessians ``\mathfrak{P}^T K \mathfrak{P}``.
    """
    P = AA @ BA
    K = P.transpose(0, 2, 1) @ KA @ BA

    if symmetrize:
        K = (K + K.transpose(0, 2, 1)) * 0.5

    return K

def interpolate_hessian(
        steps: ArrayLike,
        hess: NDArray[np.float64],
        n_step: int) -> NDArray[np.float64]:
    """Interpolate the Hessian along the trajectory using cubic splines.

    Args:
        steps: ``(S,)`` strictly increasing indices of the steps where the Hessians are evaluated.
        hess: ``(S, 3N, 3N)`` Hessians at the given steps.
        n_step: Total number of steps in the trajectory.

    Returns:
        ``(n_step, 3N, 3N)`` interpolated Hessians at every step ``0, 1, ..., n_step-1``.
    """
    return CubicSpline(steps, hess, axis=0)(np.arange(n_step))
