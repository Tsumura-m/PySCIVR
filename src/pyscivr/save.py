"""Saving and loading of trajectories and Hessians in the ``.npz`` format.

See sections II B 2 and II C of the paper for details.

Notation:
    - ``N``: number of atoms.
    - ``T``: number of time steps of the trajectory.
    - ``K``: number of constraints.
    - ``S``: number of steps at which the Hessian is evaluated.
    - ``m``: number of active coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .hessian import interpolate_hessian, projected_hessian

if TYPE_CHECKING:
    from .coordinates import InternalCoordinates

TRAJ_KEYS = ["posi", "mome", "force", "potential_energy", "total_energy"]

##################################################################################################
# trajectory save/load
##################################################################################################


def save_trajectory(fpath: str, traj: dict[str, Any], **meta: Any) -> None:
    """Save a trajectory.

    Only the keys ``(posi, mome, force, potential_energy, total_energy, dt, mass)``
    are saved.
    The Lagrange multipliers ``lam`` and ``mu`` are not saved, since they can be
    restored by :func:`load_trajectory`.

    Args:
        fpath: Path to save the trajectory.
        traj: Trajectory returned by :meth:`pyscivr.integrators.Integrator.run`.
        **meta: Additional metadata to save with the trajectory.
    """
    data = {key: traj[key] for key in TRAJ_KEYS if key in traj}
    data["dt"] = np.asarray(traj["dt"])
    data["mass"] = np.asarray(traj["mass"])
    np.savez(fpath, **data, **meta)


def restore_multipliers(
    coord_const: InternalCoordinates,
    posi: ArrayLike,
    mome: ArrayLike,
    force: ArrayLike,
    mass: ArrayLike,
    dt: float,
    *,
    atol: float = 1e-10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Restore the Lagrange multipliers of SHAKE/RATTLE from a trajectory.

    The multipliers are obtained by solving (Eqs. (25)-(27) in the paper)

    .. math::

        q_{n+1} - \tilde{q}_{n+1} &= \Delta t M^{-1} \mathcal{G}(q_{n})^T \lambda_{n}, \quad
        \tilde{q}_{n+1} = q_n + \Delta t M^{-1} p_n + \frac{\Delta t^2}{2} M^{-1} F(q_n)

        p_{n+1} - \tilde{p}_{n+1} &= \mathcal{G}(q_{n+1})^T \mu_{n+1}, \quad
        \tilde{p}_{n+1} = p_n + \frac{\Delta t}{2} (F(q_n) + F(q_{n+1})) + \mathcal{G}(q_{n})^T \lambda_{n}

    in the least-squares sense, and the residuals are checked to be small.

    Args:
        coord_const: InternalCoordinates defining the ``K`` constraints,
            the same as used in :class:`pyscivr.integrators.ShakeRattle`.
        posi: ``(T, N, 3)`` or ``(T, 3N)`` positions.
        mome: ``(T, N, 3)`` or ``(T, 3N)`` momenta.
        force: ``(T, N, 3)`` or ``(T, 3N)`` forces.
        mass: ``(N,)`` atomic masses.
        dt: Time step.
        atol: Absolute tolerance for checking the residuals.

    Returns:
        ``(lam, mu)``, where

        - ``lam``: ``(T, K)`` Lagrange multipliers for position constraints.
        - ``mu``: ``(T, K)`` Lagrange multipliers for momentum constraints.

    Raises:
        AssertionError: If the residuals exceed ``atol``, i.e., the trajectory is
            not consistent with SHAKE/RATTLE under ``coord_const``.
    """
    n_step = len(posi)
    mass_inv = 1.0 / np.repeat(np.asarray(mass), 3)
    posi = np.asarray(posi, dtype=float).reshape(n_step, -1)  # (T, 3N)
    mome = np.asarray(mome, dtype=float).reshape(n_step, -1)  # (T, 3N)
    force = np.asarray(force, dtype=float).reshape(n_step, -1)  # (T, 3N)

    G = np.asarray(coord_const.get_const_grad_batch(posi))  # (T, K, 3N)
    inv_metric = G @ (mass_inv[:, None] * G.transpose(0, 2, 1))  # (T, K, K)

    n_const = G.shape[1]

    posi_pred = posi[:-1] + dt * mass_inv * mome[:-1] + 0.5 * dt**2 * mass_inv * force[:-1]
    delta_posi = posi[1:] - posi_pred
    lam = np.linalg.solve(inv_metric[:-1], np.einsum("tki,ti->tk", G[:-1], delta_posi)[..., None])[..., 0] / dt

    mome_pred = mome[:-1] + 0.5 * dt * (force[:-1] + force[1:]) + np.einsum("tki,tk->ti", G[:-1], lam)
    delta_mome = mome[1:] - mome_pred
    mu = np.linalg.solve(inv_metric[1:], np.einsum("tki,ti->tk", G[1:], mass_inv * delta_mome)[..., None])[..., 0]

    lam = np.concatenate([lam, np.zeros((1, n_const))], axis=0)
    mu = np.concatenate([np.zeros((1, n_const)), mu], axis=0)

    # evaluate residuals (q_{n+1} - q~_{n+1} - dt M^{-1} G_{n}^T lam_{n}, p_{n+1} - p~_{n+1} - G_{n+1}^T mu_{n+1})
    res_posi = delta_posi - dt * mass_inv * np.einsum("tki,tk->ti", G[:-1], lam[:-1])
    res_mome = delta_mome - np.einsum("tki,tk->ti", G[1:], mu[1:])
    assert np.allclose(res_posi, 0.0, atol=atol), (
        f"Position residuals are not close to zero. max residual {np.max(np.abs(res_posi))}"
    )
    assert np.allclose(res_mome, 0.0, atol=atol), (
        f"Momentum residuals are not close to zero. max residual {np.max(np.abs(res_mome))}"
    )
    return lam, mu


def load_trajectory(fpath: str, coord_const: InternalCoordinates | None = None) -> dict[str, Any]:
    """Load a trajectory saved by :func:`save_trajectory`.

    Args:
        fpath: Path to the saved trajectory.
        coord_const: InternalCoordinates defining the constraints,
            the same as used in :class:`pyscivr.integrators.ShakeRattle`.
            If provided, the Lagrange multipliers ``lam`` and ``mu`` are restored by :func:`restore_multipliers`.

    Returns:
        Trajectory with keys of ``(posi, mome, force, potential_energy, total_energy, dt, mass)``,
        and additional data saved as ``meta``.
        If ``coord_const`` is provided, the Lagrange multipliers ``lam`` and ``mu`` are also included.
    """
    with np.load(fpath) as f:
        traj = {key: f[key] for key in f.files}
    traj["dt"] = float(traj["dt"])

    if coord_const is not None:
        lam, mu = restore_multipliers(coord_const, traj["posi"], traj["mome"], traj["force"], traj["mass"], traj["dt"])
        traj["lam"], traj["mu"] = lam, mu

    return traj


##################################################################################################
# Hessian
##################################################################################################


def save_hessian(
    fpath: str,
    steps: ArrayLike,
    hess: ArrayLike,
    posi: ArrayLike,
    coord_all: InternalCoordinates,
    active_idx: ArrayLike,
    **meta: Any,
) -> None:
    r"""Save the Hessians or projected Hessians in a compressed form.

    To save the storage, the ``(S, 3N, 3N)`` full Hessians :math:`K` are saved as
    ``K A_A`` of shape ``(S, 3N, m)`` (see also :func:`pyscivr.hessian.eval_hess_active`).
    The projected Hessians ``\mathfrak{P}^T K \mathfrak{P}`` can be restored by :func:`load_hessian`.

    Args:
        fpath: Path to save the Hessians.
        steps: ``(S,)`` indices of the steps where the Hessians are evaluated.
        hess: ``(S, 3N, 3N)`` Hessians or projected Hessians.
        posi: ``(T, N, 3)`` or ``(T, 3N)`` positions of the trajectory.
        coord_all: InternalCoordinates for the full system. It must define ``3N``
            coordinates for coordinate transformation.
        active_idx: ``(m,)`` row indices of the active coordinates in ``coord_all``,
            in the row order after :meth:`build()`.
        **meta: Additional metadata to save with the Hessians.

    Note:
        The file contains the keys ``steps``, ``hess`` and ``active_idx``,
        in addition to ``meta``.
    """
    data = dict(steps=np.asarray(steps))
    active_idx = np.asarray(active_idx, dtype=int)
    posi = np.asarray(posi, dtype=float).reshape(len(posi), -1)
    B = np.asarray(coord_all.get_const_grad_batch(posi[steps]))  # (S, 3N, 3N)
    unit_active = np.eye(B.shape[1])[:, active_idx]  # (3N, m)
    A_active = np.linalg.solve(B, unit_active)  # (S, 3N, m)

    hess_to_save = hess @ A_active  # (S, 3N, m)
    data["hess"] = hess_to_save
    data["active_idx"] = active_idx

    np.savez(fpath, **data, **meta)


def load_hessian(
    fpath: str, coord_all: InternalCoordinates, posi: ArrayLike, n_step=None, symmetrize=True
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    r"""Restore the projected Hessians saved by :func:`save_hessian`.

    The projected Hessians are restored as :math:`\mathfrak{P}^T K \mathfrak{P}`, where :math:`\mathfrak{P} = A_A B_A`.
    Since :math:`\mathfrak{P}` is idempotent, the same Hessians can be restored
    whether the Hessians are projected or not.

    Args:
        fpath: Path to the saved Hessians.
        coord_all: InternalCoordinates for the full system. It must define ``3N``
            coordinates for coordinate transformation.
        posi: ``(T, N, 3)`` or ``(T, 3N)`` positions of the trajectory.
        n_step: If given, the Hessians are interpolated (cubic spline) to all steps.
            ``(0, 1, ..., n_step-1)`` (see :func:`pyscivr.hessian.interpolate_hessian`).
        symmetrize: If True, the Hessians are symmetrized.

    Returns:
        ``(steps, hess)``, where

        - ``steps``: ``(S,)`` indices of the steps where the Hessians are evaluated.
        - ``hess``: ``(S, 3N, 3N)`` projected Hessians, or ``(n_step, 3N, 3N)`` if ``n_step`` is given.
    """
    with np.load(fpath) as f:
        steps = f["steps"]
        hess = f["hess"]
        active_idx = f["active_idx"]

    posi = np.asarray(posi, dtype=float).reshape(len(posi), -1)
    B = np.asarray(coord_all.get_const_grad_batch(posi[steps]))  # (S, 3N, 3N)

    unit_active = np.eye(B.shape[1])[:, active_idx]  # (3N, m)
    A_active = np.linalg.solve(B, unit_active)  # (S, 3N, m)
    B_active = B[:, active_idx, :]  # (S, m, 3N)

    hess = projected_hessian(hess, A_active, B_active, symmetrize=symmetrize)  # (S, 3N, 3N)

    if n_step is not None:
        hess = interpolate_hessian(steps, hess, n_step)  # (n_step, 3N, 3N)
    return steps, hess
