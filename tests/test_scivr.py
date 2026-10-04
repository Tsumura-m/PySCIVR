import numpy as np
import pytest

from pyscivr.basis import get_overlap, overlap_1d
from pyscivr.scivr import (
    get_action,
    get_lagrangian,
    get_prefactor,
    get_ta_spectrum_fft,
    get_ta_spectrum_naive,
)
from pyscivr.units import HBar

# ---------------------------------------------------------------- action / Lagrangian

def test_action_constant_lagrangian():
    dt, n = 0.1, 11
    t = dt * np.arange(n)
    np.testing.assert_allclose(get_action(np.full(n, 2.5), dt), 2.5 * t, atol=1e-12)

@pytest.mark.parametrize("n", [11, 12])
def test_action_exact_for_cubic_at_even_steps(n):
    dt = 0.1
    t = dt * np.arange(n)
    lag = 1.0 - 2.0 * t + 3.0 * t**2 - 4.0 * t**3
    exact = t - t**2 + t**3 - t**4
    np.testing.assert_allclose(get_action(lag, dt)[::2], exact[::2], atol=1e-12)

def test_lagrangian():
    ener_tot = np.array([1.0, 1.0])
    ener_pot = np.array([0.3, 0.6])
    np.testing.assert_allclose(get_lagrangian(ener_tot, ener_pot, 0.2), [0.6, 0.0], atol=1e-12)

# ---------------------------------------------------------------- prefactor

@pytest.mark.parametrize("m", [1, 3])
def test_prefactor_identity_monodromy(m):
    mono = np.tile(np.eye(2 * m), (5, 1, 1))
    gamma = np.linspace(0.5, 2.0, m)
    np.testing.assert_allclose(get_prefactor(mono, gamma), 1.0)

def test_prefactor_invalid_steps_are_nan():
    mono = np.tile(np.eye(2), (5, 1, 1))
    valid = np.array([True, True, False, True, True])
    pref = get_prefactor(mono, np.array([1.0]), valid)
    assert np.isnan(pref[2])
    np.testing.assert_allclose(pref[valid], 1.0)

# ---------------------------------------------------------------- spectrum

def test_spectrum_peak_position():
    dt, n_step, n_fft = 1.0, 2001, 2**14
    t = dt * np.arange(n_step)
    ener_grid, spec = get_ta_spectrum_fft(np.ones(n_step), np.ones(n_step), np.zeros(n_step), dt, n_fft)
    E0 = ener_grid[200]
    _, spec = get_ta_spectrum_fft(np.ones(n_step), np.ones(n_step), -E0 * t, dt, n_fft)
    assert np.argmax(spec) == 200

def test_spectrum_fft_matches_naive():
    rng = np.random.default_rng(0)
    dt, n_step, n_fft = 1.0, 501, 2**12
    pref = np.exp(1j * rng.uniform(0, 2 * np.pi, n_step))
    ovlp = rng.standard_normal(n_step) + 1j * rng.standard_normal(n_step)
    action = rng.standard_normal(n_step)
    valid = rng.uniform(size=n_step) > 0.1

    ener_grid, spec_fft = get_ta_spectrum_fft(pref, ovlp, action, dt, n_fft, valid)
    spec_naive = get_ta_spectrum_naive(pref, ovlp, action, dt, ener_grid[:100], valid)
    np.testing.assert_allclose(spec_naive * (n_step - 1), spec_fft[:100] * n_step, rtol=1e-8)

# ---------------------------------------------------------------- overlap

def test_overlap_r_at_same_point():
    np.testing.assert_allclose(overlap_1d("R", 0.3, 1.2, np.array([0.3]), np.array([1.2]), 2.0), 1.0)

def test_overlap_s1_periodic():
    q = np.linspace(-1.0, 1.0, 5)
    p = np.linspace(-0.5, 0.5, 5) * HBar
    a = overlap_1d("S1", 0.2, 0.1 * HBar, q, p, 5.0)
    b = overlap_1d("S1", 0.2, 0.1 * HBar, q + 2 * np.pi, p, 5.0)
    np.testing.assert_allclose(a, b, atol=1e-12)

@pytest.mark.parametrize("topology, sign", [("I-", -1), ("I+", 1), ("R+", -1)])
def test_overlap_mirror_symmetry(topology, sign):
    """The sign is reversed for I- and R+, when (q_ref, p_ref) -> (-q_ref, -p_ref) """
    q = np.linspace(0.1, 2.0, 5)
    p = np.linspace(-0.5, 0.5, 5) * HBar
    a = overlap_1d(topology, 0.7, 0.3 * HBar, q, p, 3.0)
    b = overlap_1d(topology, -0.7, -0.3 * HBar, q, p, 3.0)
    np.testing.assert_allclose(b, sign * a, atol=1e-12)

def test_get_overlap_is_product():
    rng = np.random.default_rng(0)
    topologies = ["R+", "I-", "S1"]
    posi_ref, mome_ref = np.array([0.7, 1.0, 0.5]), np.array([0.0, 0.2, -0.1]) * HBar
    posi = posi_ref + 0.1 * rng.standard_normal((4, 3))
    mome = mome_ref + 0.1 * HBar * rng.standard_normal((4, 3))
    gamma = np.array([10.0, 3.0, 3.0])
    expected = np.prod([overlap_1d(topologies[k], posi_ref[k], mome_ref[k], posi[:, k], mome[:, k], gamma[k])
                        for k in range(3)], axis=0)
    np.testing.assert_allclose(get_overlap(topologies, posi_ref, mome_ref, posi, mome, gamma), expected)
