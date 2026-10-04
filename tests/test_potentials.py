import numpy as np
import pytest

from pyscivr.potentials import H2SplinePotential


def h2_positions(r):
    return np.array([[(-x / 2, 0, 0), (x / 2, 0, 0)] for x in r])


def test_h2_minimum_at_r_e(h2):
    pot = H2SplinePotential()
    r = np.linspace(0.6, 0.9, 301)
    ener = pot.energy(h2_positions(r))
    assert abs(r[np.argmin(ener)] - h2["r_e"]) < 2e-3


def test_h2_forces_match_finite_difference(h2, rng):
    pot = H2SplinePotential()
    posi = h2["posi_ref"] * 1.1 + 0.02 * rng.standard_normal((2, 3))
    h = 1e-5
    disp = h * np.eye(6).reshape(6, 2, 3)
    e_p = pot.energy(posi[None] + disp)
    e_m = pot.energy(posi[None] - disp)
    force_fd = -((e_p - e_m) / (2 * h)).reshape(2, 3)
    force = np.asarray(pot.forces(posi[None]))[0]
    np.testing.assert_allclose(force, force_fd, atol=1e-6 * np.max(np.abs(force)))


def test_h2_hessian_matches_finite_difference(h2, rng):
    pot = H2SplinePotential()
    posi = h2["posi_ref"] * 1.1 + 0.02 * rng.standard_normal((2, 3))
    h = 1e-5
    disp = h * np.eye(6).reshape(6, 2, 3)
    f_p = np.asarray(pot.forces(posi[None] + disp)).reshape(6, 6)
    f_m = np.asarray(pot.forces(posi[None] - disp)).reshape(6, 6)
    hess_fd = -(f_p - f_m) / (2 * h)
    hess = np.asarray(pot.hessian(posi[None]))[0]
    np.testing.assert_allclose(hess, hess_fd, atol=1e-5 * np.max(np.abs(hess)))
    np.testing.assert_allclose(hess, hess.T, atol=1e-12 * np.max(np.abs(hess)))


@pytest.mark.mace
def test_mace_butane(butane):
    pytest.importorskip("mace")
    from pyscivr.potentials import mace_anicc_potential

    pot = mace_anicc_potential(butane["symbols"])
    posi = butane["posi_ref"]
    ener, force = pot.energy_forces(posi)
    assert np.all(np.isfinite(ener))
    assert np.asarray(force).reshape(-1, 3).shape == posi.shape

    # translational invariance
    ener_shift, _ = pot.energy_forces(posi + np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(ener_shift, ener, atol=1e-6)
