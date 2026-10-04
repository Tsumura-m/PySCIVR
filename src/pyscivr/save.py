import numpy as np

from .hessian import interpolate_hessian, projected_hessian

TRAJ_KEYS = ["posi", "mome", "force", "potential_energy", "total_energy"]

##################################################################################################
# trajectory save/load
##################################################################################################

def save_trajectory(fpath, traj, **meta):
    """
    Save a trajectory.

    fpath : path to save the trajectory
    traj : dict with keys of TRAJ_KEYS"""
    data = {key: traj[key] for key in TRAJ_KEYS if key in traj}
    data["dt"] = np.asarray(traj["dt"])
    data["mass"] = np.asarray(traj["mass"])
    np.savez(fpath, **data, **meta)

def restore_multipliers(coord_const, posi, mome, force, mass, dt, atol=1e-10):
    """Restore the Lagrange multipliers (lam, mu) from the trajectory.

    q_{n+1} - q~_{n+1} = dt M^{-1} G_{n}^T lam_{n}
    (q~_{n+1} = q_n + dt M^{-1} p_n + 0.5 dt^2 M^{-1} F_n)

    p_{n+1} - p~_{n+1} = G_{n+1}^T mu_{n+1}
    (p~_{n+1} = p_n + 0.5 dt (F_n + F_{n+1}) + G_{n}^T lam_{n})

    coord_const : InternalCoordinates for constraints
    posi    : (T, 3N) positions
    mome    : (T, 3N) momenta
    force   : (T, 3N) forces
    mass    : (N,) atomic masses
    dt      : time step
    atol    : tolerance for checking the residuals

    returns:
        lam : (T, K) restored Lagrange multipliers for position constraints
        mu  : (T, K) restored Lagrange multipliers for momentum constraints
    """

    n_step = posi.shape[0]
    mass_inv = 1.0 / np.repeat(np.asarray(mass), 3)
    posi = np.asarray(posi, dtype=float).reshape(n_step, -1)  # (T, 3N)
    mome = np.asarray(mome, dtype=float).reshape(n_step, -1)  # (T, 3N)
    force = np.asarray(force, dtype=float).reshape(n_step, -1) # (T, 3N)

    G = np.asarray(coord_const.get_const_grad_batch(posi))  # (T, K, 3N)
    inv_metric = G @ (mass_inv[:, None] * G.transpose(0, 2, 1))  # (T, K, K)

    n_const = G.shape[1]

    posi_pred = posi[:-1] + dt * mass_inv * mome[:-1] + 0.5 * dt**2 * mass_inv * force[:-1]
    delta_posi = posi[1:] - posi_pred
    lam = np.linalg.solve(inv_metric[:-1], np.einsum("tki,ti->tk", G[:-1], delta_posi)[...,None])[...,0] / dt

    mome_pred = mome[:-1] + 0.5 * dt * (force[:-1] + force[1:]) + np.einsum("tki,tk->ti", G[:-1], lam)
    delta_mome = mome[1:] - mome_pred
    mu = np.linalg.solve(inv_metric[1:], np.einsum("tki,ti->tk", G[1:], mass_inv * delta_mome)[...,None])[...,0]

    lam = np.concatenate([lam, np.zeros((1, n_const))], axis=0)
    mu = np.concatenate([np.zeros((1, n_const)), mu], axis=0)

    # evaluate residuals (q_{n+1} - q~_{n+1} - dt M^{-1} G_{n}^T lam_{n}, p_{n+1} - p~_{n+1} - G_{n+1}^T mu_{n+1})
    res_posi = delta_posi - dt * mass_inv * np.einsum("tki,tk->ti", G[:-1], lam[:-1])
    res_mome = delta_mome - np.einsum("tki,tk->ti", G[1:], mu[1:])
    assert np.allclose(res_posi, 0.0, atol=atol), \
        f"Position residuals are not close to zero. max residual {np.max(np.abs(res_posi))}"
    assert np.allclose(res_mome, 0.0, atol=atol), \
        f"Momentum residuals are not close to zero. max residual {np.max(np.abs(res_mome))}"
    return lam, mu

def load_trajectory(fpath, coord_const=None):
    """Load a trajectory. lam and mu are restored if coord_const is provided.

    returns dict with keys of TRAJ_KEYS + dt, mass, (+ lam, mu)
    """

    with np.load(fpath) as f:
        traj = {key: f[key] for key in f.files}
    traj["dt"] = float(traj["dt"])

    if coord_const is not None:
        lam, mu = restore_multipliers(coord_const, traj["posi"], traj["mome"],
                                      traj["force"], traj["mass"], traj["dt"])
        traj["lam"], traj["mu"] = lam, mu

    return traj

##################################################################################################
# Hessian
##################################################################################################

def save_hessian(fpath, steps, hess, posi, coord_all, active_idx, **meta):
    """Save the projected Hessians or Hessians. (H or P^T H P (P = A_A B_A) (3N, 3N))
    To save the memory, the (projected) Hessians are saved in the form of H A_A (3N, K)
    The projected Hessian can be restored by P^T H P = P (H A_A) B_A using load_hessian().

    steps : (S,) indices of the steps where the Hessians are evaluated
    hess  : (S, 3N, 3N) Hessians or projected Hessians
    posi  : (T, 3N) all positions
    coord_all : InternalCoordinates for coordinate transformation
    active_idx : (m,) indices of the active coordinates

    returns:
        saves the Hessians to fpath in npz format with keys of steps, hess, active_idx
    """

    data = dict(steps=np.asarray(steps))
    active_idx = np.asarray(active_idx, dtype=int)
    posi = np.asarray(posi, dtype=float).reshape(len(posi), -1)
    B = np.asarray(coord_all.get_const_grad_batch(posi[steps]))   # (S, 3N, 3N)
    unit_active = np.eye(B.shape[1])[:, active_idx]  # (3N, m)
    A_active = np.linalg.solve(B, unit_active)  # (S, 3N, m)

    hess_to_save = hess @ A_active  # (S, 3N, m)
    data["hess"] = hess_to_save
    data["active_idx"] = active_idx

    np.savez(fpath, **data, **meta)


def load_hessian(fpath, coord_all, posi, n_step=None, symmetrize=True):
    """Restore the projected Hessians from the saved file.
    P^T H P = P^T (H A_A) B_A  (P = A_A B_A)
    Due to the idempotency of P, the same Hessian can be restored whether the Hessian is projected or not.

    coord_all : InternalCoordinates for coordinate transformation
    posi    : (T, 3N) all positions
    n_step  : if given, the Hessian is interpolated (cubic spline) to all steps (0, 1, ..., n_step-1)
    symmetrize : if True, the Hessian is symmetrized

    return steps (S,), hess (S, 3N, 3N) or (n_step, 3N, 3N)
    """

    with np.load(fpath) as f:
        steps = f["steps"]
        hess = f["hess"]
        active_idx = f["active_idx"]

    posi = np.asarray(posi, dtype=float).reshape(len(posi), -1)
    B = np.asarray(coord_all.get_const_grad_batch(posi[steps]))   # (S, 3N, 3N)

    unit_active = np.eye(B.shape[1])[:, active_idx]  # (3N, m)
    A_active = np.linalg.solve(B, unit_active)  # (S, 3N, m)
    B_active = B[:, active_idx, :]  # (S, m, 3N)

    hess = projected_hessian(hess, A_active, B_active, symmetrize=symmetrize)  # (S, 3N, 3N)

    if n_step is not None:
        hess = interpolate_hessian(steps, hess, n_step)  # (n_step, 3N, 3N)
    return steps, hess
