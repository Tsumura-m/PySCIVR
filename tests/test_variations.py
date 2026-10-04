import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _utils import get_S, h2_initial_momenta

from pyscivr.coordinates import InternalCoordinates
from pyscivr.integrators import ImplicitVerlet, ShakeRattle, quaternion_metric
from pyscivr.potentials import H2SplinePotential
from pyscivr.save import load_trajectory, save_trajectory
from pyscivr.units import CmInv, FemtoSec
from pyscivr.utils import get_inertia
from pyscivr.variations import (
    make_jacobi_func,
    make_vari_propagator,
    make_vari_propagator_implicit,
    propagate_variation,
    propagate_variation_implicit,
    to_internal,
)

@pytest.mark.slow
def test_shake_rattle_monodromy_matches_finite_difference(h2, tmp_path):
    mass, posi_ref = h2["mass"], h2["posi_ref"]
    calc = H2SplinePotential()

    coord_const = InternalCoordinates(posi_ref, mass)
    coord_const.add_translation()
    coord_const.build()

    coord_all = InternalCoordinates(posi_ref, mass)
    coord_all.add_translation()
    coord_all.add_bond(0, 1)
    coord_all.add_polar(0, 1)
    coord_all.add_azimuthal(0, 1)
    coord_all.build()
    active_idx = coord_all.inv_argsort[np.array([3, 4, 5])]  # bond, polar, azimuthal

    # theta = pi/3 (do not pass any singularities)
    n_check = 500
    dt = 0.1 * FemtoSec
    dyn = ShakeRattle(calc, posi_ref, h2_initial_momenta(mass, 2000, 2000, np.pi / 3), mass, dt, coord_const)
    fpath = tmp_path / "h2_rovib.npz"
    save_trajectory(fpath, dyn.run(n_check))
    traj = load_trajectory(fpath, coord_const)
    posi, mome, lam, mu, dt = traj["posi"], traj["mome"], traj["lam"], traj["mu"], traj["dt"]

    hess = calc.hessian(posi)
    jacobi_func, jacobi_rev_func, internal_func, monodromy_func = make_jacobi_func(coord_all, active_idx)
    J_rev_init = jacobi_rev_func(posi[0], mome[0])
    vari_propagator = make_vari_propagator(coord_const, dt)
    vari = propagate_variation(vari_propagator, posi, mome, lam, mu, hess, vari_init=J_rev_init)
    _, _, mono_internal, symplectic_err = to_internal(monodromy_func, posi, mome, vari)

    assert np.all(np.isfinite(symplectic_err))
    assert np.max(symplectic_err) < 1e-4

    # finite difference
    eps = 1e-6
    n3 = posi.shape[1] * 3

    def run_internal(x0):
        tr = ShakeRattle(calc, x0[:n3].reshape(-1, 3), x0[n3:].reshape(-1, 3), mass, dt, coord_const).run(n_check)
        q, p = jax.vmap(internal_func)(tr["posi"].reshape(n_check + 1, -1), tr["mome"].reshape(n_check + 1, -1))
        return np.concatenate([np.asarray(q), np.asarray(p)], axis=1)          # (T, 2m)

    x0 = np.concatenate([posi[0].reshape(-1), mome[0].reshape(-1)])
    J_rev0 = np.asarray(J_rev_init)
    cols = []
    for j in range(J_rev0.shape[1]):
        d = run_internal(x0 + eps * J_rev0[:, j]) - run_internal(x0 - eps * J_rev0[:, j])
        d[:, 2] = (d[:, 2] + np.pi) % (2 * np.pi) - np.pi                     # azimuthal : wrap
        cols.append(d / (2 * eps))
    mono_fd = np.stack(cols, axis=-1)                                          # (T, 2m, 2m)

    mono = np.asarray(mono_internal[:n_check + 1])
    diff = np.max(np.abs(mono_fd - mono), axis=(1, 2))
    scale = np.max(np.abs(mono), axis=(1, 2))
    assert np.max(diff / scale) < 1e-4

def test_implicit_monodromy_matches_finite_difference(ch4, rng):
    inertia = np.diag(get_inertia(ch4["posi_ref"], ch4["mass"]))
    metric_func = quaternion_metric(inertia)
    dt = 0.1 * FemtoSec
    n_step = 2000

    q_init = rng.normal(size=4)
    q_init /= np.linalg.norm(q_init)
    L_dir = rng.normal(size=3)
    L_dir /= np.linalg.norm(L_dir)
    E_rot = 1000 * CmInv
    L_body = np.sqrt(2 * E_rot / np.sum(L_dir**2 / inertia)) * L_dir
    p_init = 2 * get_S(q_init).T @ L_body

    traj = ImplicitVerlet(q_init, p_init, dt, metric_func, normalize=True).run(n_step)
    vari_propagator = make_vari_propagator_implicit(metric_func, dt, normalize=True)
    mono = propagate_variation_implicit(vari_propagator, jnp.asarray(traj["posi"]), jnp.asarray(traj["mome"]),
                                        jnp.asarray(traj["mome_half"]), jnp.eye(8))      # (T, 8, 8)

    q0, p0 = traj["posi"][0], traj["mome"][0]
    dq = rng.normal(size=4)
    dq -= (q0 @ dq) * q0                # tangent
    dp = rng.normal(size=4)
    dp -= (q0 @ dp + p0 @ dq) * q0      # keeps q^T p = 0
    eps = 1e-6
    tr_p = ImplicitVerlet(q0 + eps * dq, p0 + eps * dp, dt, metric_func, normalize=True).run(n_step)
    tr_m = ImplicitVerlet(q0 - eps * dq, p0 - eps * dp, dt, metric_func, normalize=True).run(n_step)
    fd = np.concatenate([tr_p["posi"] - tr_m["posi"], tr_p["mome"] - tr_m["mome"]], axis=1) / (2 * eps)

    pred = np.asarray(mono) @ np.concatenate([dq, dp])
    assert np.max(np.abs(pred - fd)) / np.max(np.abs(pred)) < 1e-5
