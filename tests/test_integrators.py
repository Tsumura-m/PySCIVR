import jax.numpy as jnp
import numpy as np
import pytest
from _utils import get_S, h2_initial_momenta, quat_to_mat, rot_axis
 
from pyscivr.coordinates import InternalCoordinates
from pyscivr.integrators import ImplicitVerlet, ShakeRattle, quaternion_metric
from pyscivr.potentials import H2SplinePotential
from pyscivr.units import CmInv, FemtoSec
from pyscivr.utils import get_inertia, get_mass_center

# ---------------------------------------------------------------- SHAKE/RATTLE

def run_h2(h2, dt_fs, n_step, K_vib, K_rot, theta=np.pi / 2, constrain_bond=False):
    mass, posi_ref = h2["mass"], h2["posi_ref"]
    coord_const = InternalCoordinates(posi_ref, mass)
    coord_const.add_translation()
    if constrain_bond:
        coord_const.add_bond(0, 1)
    coord_const.build()
    dyn = ShakeRattle(H2SplinePotential(), posi_ref, h2_initial_momenta(mass, K_vib, K_rot, theta), mass,
                      dt_fs * FemtoSec, coord_const)
    return dyn.run(n_step)

@pytest.mark.parametrize("K_vib, K_rot, constrain_bond", [
    (2000, 0, False),
    (0, 5000, False),
    (2000, 5000, False),
    (0, 2000, True),
])
def test_shake_rattle_conservation(h2, K_vib, K_rot, constrain_bond):
    traj = run_h2(h2, dt_fs=0.1, n_step=2000, K_vib=K_vib, K_rot=K_rot, constrain_bond=constrain_bond)
    mass = h2["mass"]

    e_tot = traj["total_energy"]
    assert np.max(np.abs(e_tot - e_tot[0])) / CmInv < 5.0  # cm^-1

    com = np.sum(mass[None, :, None] * traj["posi"], axis=1)
    assert np.max(np.abs(com - com[0])) < 1e-10
    assert np.max(np.abs(traj["mome"].sum(axis=1))) < 1e-10

    if constrain_bond:
        r = np.linalg.norm(traj["posi"][:, 1] - traj["posi"][:, 0], axis=1)
        assert np.max(np.abs(r - h2["r_e"])) < 1e-8
 
def test_shake_rattle_second_order(h2):
    """Energy drift should be reduced to approximately 1/4 when dt is halved"""
    T_fs = 100.0
    errs = []
    for dt_fs in (0.1, 0.05):
        traj = run_h2(h2, dt_fs, int(round(T_fs / dt_fs)), K_vib=2000, K_rot=5000)
        e_tot = traj["total_energy"]
        errs.append(np.max(np.abs(e_tot - e_tot[0])))
    assert 3.0 < errs[0] / errs[1] < 5.0
 
# ---------------------------------------------------------------- ImplicitVerlet (spherical coordinates)
 
 
@pytest.mark.parametrize("alpha_deg", [0.0, 30.0, 60.0])
def test_free_rotor_spherical(h2_geom, alpha_deg):
    """compare H2 pure rotation with analytical solution"""
    posi_ref, mass = h2_geom["posi_ref"], h2_geom["mass"]
    r_e = np.linalg.norm(posi_ref[1] - posi_ref[0])
    inertia = 0.5 * mass[0] * r_e**2
 
    def metric(posi):  # [theta, phi]
        theta = posi[0]
        return jnp.diag(jnp.array([1.0, 1.0 / jnp.sin(theta)**2])) / inertia
 
    alpha = np.radians(alpha_deg)
    dt = 0.1 * FemtoSec
    E_rot = 1000 * CmInv
    L = np.sqrt(2.0 * inertia * E_rot)
    period = 2 * np.pi * inertia / L
    n_step = int(3 * period / dt)
 
    posi = np.array([np.pi / 2, 0.0])
    mome = np.array([-L * np.sin(alpha), L * np.cos(alpha)])
    traj = ImplicitVerlet(posi, mome, dt, metric).run(n_step)
 
    # analytical solution
    t = np.arange(len(traj["posi"])) * dt
    omega = L / inertia
    n = np.cos(omega * t)[:, None] * np.array([1.0, 0.0, 0.0]) \
        + np.sin(omega * t)[:, None] * np.array([0.0, np.cos(alpha), np.sin(alpha)])
    theta_ex, phi_ex = np.arccos(np.clip(n[:, 2], -1, 1)), np.arctan2(n[:, 1], n[:, 0])
 
    E = traj["total_energy"]
    assert np.max(np.abs(E - E[0])) / E_rot < 1e-4
    assert np.max(np.abs(traj["mome"][:, 1] - mome[1])) < 1e-10 * L  # p_phi should be conseved
    assert np.max(np.abs(traj["posi"][:, 0] - theta_ex)) < 1e-3
    assert np.max(np.abs((traj["posi"][:, 1] - phi_ex + np.pi) % (2 * np.pi) - np.pi)) < 1e-3

# ---------------------------------------------------------------- ImplicitVerlet (quaternion)

def test_spherical_top_quaternion(ch4, rng):
    """Spherical top molecule (methane)"""
    posi_ref, mass = ch4["posi_ref"], ch4["mass"]
    assert np.allclose(get_mass_center(posi_ref, mass), 0.0)
    inertia_tensor = get_inertia(posi_ref, mass)
    assert np.allclose(inertia_tensor, np.diag(np.diag(inertia_tensor)))
    inertia = np.diag(inertia_tensor)
    metric_func = quaternion_metric(inertia)

    q_init = rng.normal(size=4)
    q_init /= np.linalg.norm(q_init)
    L_dir = rng.normal(size=3)
    L_dir /= np.linalg.norm(L_dir)
    E_rot = 1000 * CmInv
    L_body = np.sqrt(2 * E_rot / np.sum(L_dir**2 / inertia)) * L_dir   # E_rot = L^T I^{-1} L / 2
    p_init = 2 * get_S(q_init).T @ L_body                               # p = 2 S^T L_body

    omega_body = L_body / inertia
    dt = 0.1 * FemtoSec
    period = 2 * np.pi / np.linalg.norm(omega_body)
    n_step = int(2 * period / dt)

    traj = ImplicitVerlet(q_init, p_init, dt, metric_func, normalize=True).run(n_step)
    q, p = traj["posi"], traj["mome"]
    t = np.arange(len(q)) * dt

    C = np.stack([quat_to_mat(x) for x in q])                                 # (T, 3, 3)
    L_b = 0.5 * np.einsum("tij,tj->ti", np.stack([get_S(x) for x in q]), p)   # L_body = S p / 2
    L_s = np.einsum("tij,tj->ti", C, L_b)                                     # L_space (conserved)
    E = traj["total_energy"]
 
    omega_space = C[0] @ omega_body
    w = np.linalg.norm(omega_space)
    C_ex = np.stack([rot_axis(omega_space / w, w * s) @ C[0] for s in t])

    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-10)
    assert np.max(np.abs(E - E[0])) / E_rot < 1e-4
    assert np.max(np.abs(L_s - L_s[0])) / np.linalg.norm(L_body) < 1e-4
    assert np.max(np.abs(C - C_ex)) < 1e-3
