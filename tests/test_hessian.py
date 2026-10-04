"""hydrogen_rot.ipynb, hydrogen_rovib.ipynb から移植したテスト."""

import numpy as np
import pytest
from _utils import h2_initial_momenta

from pyscivr.coordinates import InternalCoordinates
from pyscivr.hessian import eval_hess_active, interpolate_hessian, projected_hessian
from pyscivr.integrators import ShakeRattle
from pyscivr.potentials import H2SplinePotential
from pyscivr.save import load_hessian, save_hessian
from pyscivr.units import FemtoSec


@pytest.fixture(scope="module")
def h2_rot_traj(h2):
    """Trajectory of rotated hydrogen passing singularities (pi/2)"""
    mass, posi_ref = h2["mass"], h2["posi_ref"]
    calc = H2SplinePotential()

    coord_const = InternalCoordinates(posi_ref, mass)
    coord_const.add_translation()
    coord_const.add_bond(0, 1)
    coord_const.build()
    dyn = ShakeRattle(calc, posi_ref, h2_initial_momenta(mass, 0, 2000, np.pi / 2), mass, 0.1 * FemtoSec,
                      coord_const)
    traj = dyn.run(2000)

    coord_all = InternalCoordinates(posi_ref, mass)
    coord_all.add_translation()
    coord_all.add_polar(0, 1)
    coord_all.add_azimuthal(0, 1)
    coord_all.add_bond(0, 1)
    coord_all.build()
    active_idx = coord_all.inv_argsort[np.array([3, 4])]  # polar, azimuthal

    return calc, coord_all, active_idx, traj["posi"]

def select_steps(coord_all, posi_all, threshold=1e-2):
    B = np.asarray(coord_all.get_const_grad_batch(posi_all))
    s = np.linalg.svd(B, compute_uv=False)
    ratio = s[:, -1] / s[:, 0]
    steps_all = np.arange(len(posi_all))
    return steps_all[ratio > threshold], steps_all[ratio <= threshold]

def test_removed_steps_are_near_poles(h2_rot_traj):
    _, coord_all, _, posi_all = h2_rot_traj
    _, removed = select_steps(coord_all, posi_all)
    assert len(removed) > 0
    polar_idx = coord_all.names.index("polar(0, 1)")
    polar = np.degrees(coord_all.get_value_batch(posi_all[removed])[:, polar_idx])
    assert np.all(np.minimum(polar, 180.0 - polar) < 10.0)

def test_projected_hessian_matches_analytical(h2_rot_traj, tmp_path):
    calc, coord_all, active_idx, posi_all = h2_rot_traj
    steps, _ = select_steps(coord_all, posi_all)
    steps = steps[::20]

    HA, AA, BA = eval_hess_active(calc, coord_all, active_idx, posi_all, steps)
    proj_H = projected_hessian(HA, AA, BA)

    P = AA @ BA
    hess_ana = calc.hessian(posi_all[steps])
    proj_hess_ana = P.transpose(0, 2, 1) @ hess_ana @ P
    np.testing.assert_allclose(proj_H, proj_hess_ana, atol=1e-2)

    fpath = tmp_path / "hess.npz"
    save_hessian(fpath, steps, proj_H, posi_all, coord_all, active_idx)
    steps_load, hess_load = load_hessian(fpath, coord_all, posi_all)
    np.testing.assert_array_equal(steps_load, steps)
    np.testing.assert_allclose(hess_load, proj_H)

def test_interpolate_hessian_exact_for_cubic():
    steps = np.array([0, 5, 10, 15, 20])
    coef = np.random.default_rng(0).standard_normal((4, 2, 2))

    def f(t):
        t = np.asarray(t, dtype=float)[:, None, None]
        return coef[0] + coef[1] * t + coef[2] * t**2 + coef[3] * t**3

    np.testing.assert_allclose(interpolate_hessian(steps, f(steps), 21), f(np.arange(21)), rtol=1e-10, atol=1e-10)
