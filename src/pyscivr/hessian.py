import numpy as np
from scipy.interpolate import CubicSpline


def eval_hess_active(calculator, coord_all, active_idx, posi, steps, *, eps=1e-4, chunk_size=100, rcond=1e-10):
    """
    Evaluate directional derivative of the gradient at selected steps of a trajectory.

    H A_A = d^2V A_A (A_A = dq/dQ_A)

    calculator   : Calculator (energy_forces accepts (B, N, 3))
    coord_all    : InternalCoordinates for coordinate transformation
    active_idx   : (m,) row indices of the active coordinates in coord_all, in the row order
                   after build() (e.g. coord_all.index[name] or coord_all.inv_argsort[active_idx_original]).
    posi         : (T, N, 3) trajectory
    steps        : (S,) indices of the steps to evaluate
    eps          : displacement [Angs]
    chunk_size   : number of geometries passed to the calculator at once (None : all)
    rcond        : relative cutoff for pseudo-inverse

    returns tuple of (HA, AA, BA)
        HA    : (S, 3N, m)   d^2V A_A
        AA    : (S, 3N, m)   A_A
        BA    : (S, m, 3N)   B_A
    """


    steps = np.asarray(steps, dtype=int)
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

    HQ = ((grad[0] - grad[1]) / (2 * eps)).transpose(0, 2, 1)  # (S, 3N, m)
    HA = HQ @ R
    return HA, AA, BA

def projected_hessian(HA, AA, BA, symmetrize=True):
    """
    P^T d^2 V P = P^T (d^2 V A_A) B_A, P = A_A B_A
    """
    P = AA @ BA
    H = P.transpose(0, 2, 1) @ HA @ BA

    if symmetrize:
        H = (H + H.transpose(0, 2, 1)) * 0.5

    return H

def interpolate_hessian(steps, hess, n_step):
    """Cubic spline interpolation of the Hessian along the trajectory."""
    return CubicSpline(steps, hess, axis=0)(np.arange(n_step))
