"""
Solve the same system (pure rotation of hydrogen and methane) using two independent methods (implicit integration in internal coordinates / SHAKE/RATTLE in Cartesian coordinates) and verify that the trajectories and monodromy matrices match.
"""

import jax.numpy as jnp
import numpy as np
from _utils import get_S

from pyscivr.coordinates import InternalCoordinates, QuaternionCoordinates, make_converters, quaternion_metric
from pyscivr.integrators import ImplicitVerlet, ShakeRattle
from pyscivr.potentials import ZeroPotential
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

DT = 0.1 * FemtoSec
N_STEP = 10000

def skew(n_dim):
    return np.block([[np.zeros((n_dim, n_dim)), np.eye(n_dim)],
                     [-np.eye(n_dim), np.zeros((n_dim, n_dim))]])


def test_h2_rigid_rotor_implicit_vs_shake_rattle(h2_geom):
    posi_ref, mass = h2_geom["posi_ref"], h2_geom["mass"]
    r_e = np.linalg.norm(posi_ref[1] - posi_ref[0])
    inertia = 0.5 * mass[0] * r_e**2

    def metric_h2(posi):  # [theta, phi]
        theta = posi[0]
        return jnp.diag(jnp.array([1.0, 1.0 / jnp.sin(theta)**2])) / inertia

    coord_const = InternalCoordinates(posi_ref, mass)
    coord_const.add_translation()
    coord_const.add_bond(0, 1)
    coord_const.build()

    coord_all = InternalCoordinates(posi_ref, mass)
    coord_all.add_translation()
    coord_all.add_bond(0, 1)
    coord_all.add_polar(0, 1)
    coord_all.add_azimuthal(0, 1)
    coord_all.build()
    active_idx = coord_all.inv_argsort[[4, 5]]  # polar, azimuthal

    # set initial momentum
    rng = np.random.default_rng(10202)
    K_rot = rng.uniform(10.0, 10000.0)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    r = posi_ref - mass @ posi_ref / mass.sum()
    g_rot = np.cos(angle) * np.cross([0.0, 1.0, 0.0], r).reshape(-1) \
        + np.sin(angle) * np.cross([0.0, 0.0, 1.0], r).reshape(-1)
    unit_ener_rot = 0.5 * np.sum(g_rot**2 / np.repeat(mass, 3))
    mome_cart = (g_rot * np.sqrt(K_rot * CmInv / unit_ener_rot)).reshape(-1, 3)
    posi_cart = posi_ref.copy()

    posi_int = coord_all.get_value(posi_cart)[active_idx]
    A = np.linalg.inv(coord_all.get_const_grad(posi_cart))
    mome_int = A[:, active_idx].T @ mome_cart.reshape(-1)  # P = A^T p

    # (A) implicit integration (theta, phi)
    traj_int = ImplicitVerlet(posi_int, mome_int, DT, metric_h2, normalize=False).run(N_STEP)
    vari_propagator = make_vari_propagator_implicit(metric_h2, DT, normalize=False)
    mono_int = propagate_variation_implicit(vari_propagator, jnp.asarray(traj_int["posi"]), jnp.asarray(traj_int["mome"]),
                                            jnp.asarray(traj_int["mome_half"]), jnp.eye(4))
    J = skew(2)
    symp_err_int = np.max(np.abs(mono_int.transpose(0, 2, 1) @ J @ mono_int - J), axis=(1, 2))

    # (B) SHAKE/RATTLE (Cartesian)
    pot = ZeroPotential()
    _, jacobi_rev_func, _, monodromy_func = make_jacobi_func(coord_all, active_idx)
    traj_cart = ShakeRattle(pot, posi_cart, mome_cart, mass, DT, coord_const).run(N_STEP)
    J_rev_init = jacobi_rev_func(traj_cart["posi"][0], traj_cart["mome"][0])
    vari_cart = propagate_variation(make_vari_propagator(coord_const, DT), traj_cart["posi"], traj_cart["mome"],
                                    traj_cart["lam"], traj_cart["mu"], pot.hessian(traj_cart["posi"]),
                                    vari_init=J_rev_init)
    posi_ref_int, mome_ref_int, mono_ref, symp_err_cart = to_internal(monodromy_func, traj_cart["posi"],
                                                                      traj_cart["mome"], vari_cart)

    assert np.max(symp_err_int) < 1e-10
    assert np.max(symp_err_cart) < 1e-8
    dphi = (traj_int["posi"][:, 1] - posi_ref_int[:, 1] + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(traj_int["posi"][:, 0] - posi_ref_int[:, 0])) < 5e-3
    assert np.max(np.abs(dphi)) < 1.5e-2
    assert np.max(np.abs(traj_int["mome"] - mome_ref_int)) < 3e-3
    rel_err = np.linalg.norm(mono_int - mono_ref, axis=(1, 2)) / np.linalg.norm(mono_ref, axis=(1, 2))
    assert np.max(rel_err) < 5e-3


def test_quaternion_momentum_conversion(ch4):
    """P_yaw = L_z, P_pitch = L_y, P_roll = L_x for no rotated states (q_0 = 1)"""
    n_sample = 100
    inertia = np.diag(get_inertia(ch4["posi_ref"], ch4["mass"]))
    rng = np.random.default_rng(10202)
    L_dir = rng.normal(size=(n_sample, 3))
    L_dir /= np.linalg.norm(L_dir, axis=1, keepdims=True)
    K_rot = rng.uniform(10.0, 1000.0, size=n_sample)
    L_body = np.sqrt(2.0 * K_rot[:, None] * CmInv / np.sum(L_dir**2 / inertia, axis=1, keepdims=True)) * L_dir

    q_unit = np.array([1.0, 0.0, 0.0, 0.0])
    p = 2.0 * L_body @ get_S(q_unit)
    q = np.repeat(q_unit[None, :], n_sample, axis=0)

    _, to_physical_mome, to_redundant_mome, _ = make_converters(QuaternionCoordinates(), active_idx=np.array([0, 1, 2]))
    P = to_physical_mome(q, p)
    np.testing.assert_allclose(P, L_body[:, ::-1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(to_redundant_mome(q, P), p, rtol=1e-10, atol=1e-12)


def test_ch4_spherical_top_quaternion_vs_shake_rattle(ch4):
    posi_ref, mass = ch4["posi_ref"], ch4["mass"]
    inertia = np.diag(get_inertia(posi_ref, mass))
    metric_func = quaternion_metric(inertia)

    K_rot = 500.0 * CmInv
    L_dir = np.array([0.4, 0.2, -0.3])
    L_body = np.sqrt(2.0 * K_rot / np.sum(L_dir**2 / inertia)) * L_dir
    omega = L_body / inertia  # for q_0 = 1

    posi_quat = np.array([1.0, 0.0, 0.0, 0.0])
    mome_quat = 2.0 * L_body @ get_S(posi_quat)
    posi_cart = posi_ref.copy()
    mome_cart = mass[:, None] * np.cross(omega, posi_ref)  # p_a = m_a omega x r_a

    # (A) implicit integration (quaternion)
    active_quat = np.array([0, 1, 2])  # yaw, pitch, roll
    _, jacobi_rev_quat, _, monodromy_quat = make_jacobi_func(QuaternionCoordinates(), active_quat)
    traj_quat = ImplicitVerlet(posi_quat, mome_quat, DT, metric_func, normalize=True).run(N_STEP)
    J_rev_init_quat = jacobi_rev_quat(traj_quat["posi"][0], traj_quat["mome"][0])
    vari_quat = propagate_variation_implicit(make_vari_propagator_implicit(metric_func, DT, normalize=True),
                                             jnp.asarray(traj_quat["posi"]), jnp.asarray(traj_quat["mome"]),
                                             jnp.asarray(traj_quat["mome_half"]), vari_init=J_rev_init_quat)
    posi_euler, mome_euler, mono_euler, symp_err_quat = to_internal(monodromy_quat, traj_quat["posi"],
                                                                    traj_quat["mome"], vari_quat)

    # (B) SHAKE/RATTLE (Cartesian)
    bond_indices = [(0, i) for i in range(1, 5)]
    angle_indices = [(0, 1, i) for i in range(2, 5)]
    dihedral_indices = [(0, 1, 2, i) for i in range(3, 5)]

    coord_const = InternalCoordinates(posi_ref, mass)
    coord_const.add_translation()
    for i in bond_indices:
        coord_const.add_bond(*i)
    for i in angle_indices:
        coord_const.add_angle(*i)
    for i in dihedral_indices:
        coord_const.add_dihedral(*i)
    coord_const.build()

    coord_all = InternalCoordinates(posi_ref, mass)
    coord_all.add_translation()
    coord_all.add_euler()
    for i in bond_indices:
        coord_all.add_bond(*i)
    for i in angle_indices:
        coord_all.add_angle(*i)
    for i in dihedral_indices:
        coord_all.add_dihedral(*i)
    coord_all.build()
    active_cart = coord_all.inv_argsort[[3, 4, 5]]  # yaw, pitch, roll

    pot = ZeroPotential()
    _, jacobi_rev_cart, _, monodromy_cart = make_jacobi_func(coord_all, active_cart)
    traj_cart = ShakeRattle(pot, posi_cart, mome_cart, mass, DT, coord_const).run(N_STEP)
    J_rev_init_cart = jacobi_rev_cart(traj_cart["posi"][0], traj_cart["mome"][0])
    vari_cart = propagate_variation(make_vari_propagator(coord_const, DT), traj_cart["posi"], traj_cart["mome"],
                                    traj_cart["lam"], traj_cart["mu"], pot.hessian(traj_cart["posi"]),
                                    vari_init=J_rev_init_cart)
    posi_euler_ref, mome_euler_ref, mono_euler_ref, symp_err_cart = to_internal(monodromy_cart, traj_cart["posi"],
                                                                                traj_cart["mome"], vari_cart)

    assert np.max(symp_err_quat) < 1e-11
    assert np.max(symp_err_cart) < 1e-9
    d_posi = (posi_euler - posi_euler_ref + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(d_posi)) < 1.5e-4
    assert np.max(np.abs(mome_euler - mome_euler_ref)) < 6e-5
    rel_err = np.linalg.norm(mono_euler - mono_euler_ref, axis=(1, 2)) / np.linalg.norm(mono_euler_ref, axis=(1, 2))
    assert np.max(rel_err) < 1.5e-4
