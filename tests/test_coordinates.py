import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _utils import quat_to_mat, rot_zyx
from jax.test_util import check_grads
 
from pyscivr.coordinates import InternalCoordinates, get_euler_angles, get_rotation_matrix

def build_butane(butane, move_reference=False, eckart=True):
    c = InternalCoordinates(butane["posi_ref"], butane["mass"], move_reference=move_reference)
    for idx in butane["bonds"]:
        c.add_bond(*idx)
    for idx in butane["angles"]:
        c.add_angle(*idx)
    for idx in butane["dihedrals"]:
        c.add_dihedral(*idx)
    if eckart:
        c.add_eckart()
    else:
        c.add_translation()
    c.build()
    return c

def perturbed_geometries(posi_ref, rng, n=3, noise=0.05):
    return [posi_ref + noise * rng.standard_normal(posi_ref.shape) for _ in range(n)]

def center(p, mass):
    return p - mass @ p / mass.sum()

# ---------------------------------------------------------------- InternalCoordinates

@pytest.mark.parametrize("move_reference", [False, True])
@pytest.mark.parametrize("eckart", [False, True])
def test_derivatives_match_autodiff(butane, rng, move_reference, eckart):
    coord = build_butane(butane, move_reference, eckart)
    hess_ad = jax.jacfwd(jax.jacrev(coord.get_const))  # (K, 3N, 3N)
 
    for posi in perturbed_geometries(butane["posi_ref"], rng):
        x = jnp.asarray(posi.reshape(-1))
        B = np.asarray(coord.get_const_grad(x))
        np.testing.assert_allclose(B, jax.jacrev(coord.get_value)(x), rtol=1e-6, atol=1e-8)
 
        w = rng.standard_normal(B.shape[0])      # (K,)
        u = rng.standard_normal(x.size)          # (3N,)
        V = rng.standard_normal((x.size, 2))     # (3N, X)
        H = hess_ad(x)
        np.testing.assert_allclose(coord.get_const_hess_kj_contract(x, w, V),
                                   jnp.einsum("k,kij,ja->ia", w, H, V), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(coord.get_const_hess_ij_contract(x, u, V),
                                   jnp.einsum("kij,i,ja->ka", H, u, V), rtol=1e-6, atol=1e-8)

def test_batch_matches_single(butane, rng):
    coord = build_butane(butane)
    X = np.stack([p.reshape(-1) for p in perturbed_geometries(butane["posi_ref"], rng)])
    np.testing.assert_allclose(coord.get_value_batch(X), np.stack([coord.get_value(x) for x in X]))
    np.testing.assert_allclose(coord.get_const_grad_batch(X), np.stack([coord.get_const_grad(x) for x in X]))

def test_internal_values_invariant_under_rigid_motion(butane, rng):
    coord = InternalCoordinates(butane["posi_ref"], butane["mass"])
    for idx in butane["bonds"]:
        coord.add_bond(*idx)
    for idx in butane["angles"]:
        coord.add_angle(*idx)
    for idx in butane["dihedrals"]:
        coord.add_dihedral(*idx)
    coord.build()
 
    posi = butane["posi_ref"]
    q = rng.normal(size=4)
    posi_moved = posi @ quat_to_mat(q / np.linalg.norm(q)).T + np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(coord.get_value(posi_moved.reshape(-1)), coord.get_value(posi.reshape(-1)),
                               atol=1e-10)

# ---------------------------------------------------------------- rotation matrix / Euler angles

def test_rotation_matrix_recovers_quaternion(ch4, rng):
    mass, posi_ref = ch4["mass"], ch4["posi_ref"]
    ref_c = center(posi_ref, mass)
    #random quaternion
    qs = [rng.normal(size=4) for _ in range(5)] + [np.array([0.0, 1.0, 2.0, 3.0])]
    for q in qs:
        #rotation matrix from quaternion
        Q = quat_to_mat(q / np.linalg.norm(q))
        #rotation matrix from Cartesian
        C = get_rotation_matrix(center(posi_ref @ Q.T, mass), ref_c, mass)
        np.testing.assert_allclose(C, Q, atol=1e-8)

def test_euler_angles_inverse_of_rot_zyx(rng):
    for _ in range(5):
        e = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(0.1, np.pi - 0.1), rng.uniform(-np.pi, np.pi)])
        np.testing.assert_allclose(get_euler_angles(rot_zyx(*e)), e, atol=1e-10)

def test_rotation_matrix_derivatives(ch4, rng):
    mass, posi_ref = ch4["mass"], ch4["posi_ref"]
    ref_c = center(posi_ref, mass)

    def f(x):
        return get_rotation_matrix(x - mass @ x / mass.sum(), ref_c, mass).ravel()

    x_rot = posi_ref @ rot_zyx(0.3, 1.0, -2.0).T + 0.01 * rng.normal(size=posi_ref.shape)
    for x in [posi_ref, x_rot]:
        assert not bool(jnp.isnan(jax.jacrev(f)(x)).any())
        check_grads(f, (x,), order=2, modes=["fwd", "rev"])

@pytest.fixture
def ch4_euler(ch4):
    coord = InternalCoordinates(ch4["posi_ref"], ch4["mass"])
    coord.add_translation()
    coord.add_euler()
    for h in range(1, 5):
        coord.add_bond(0, h)
    for (i, k) in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]:
        coord.add_angle(i, 0, k)
    coord.build()
    return coord

def test_euler_coordinates(ch4, ch4_euler, rng):
    coord = ch4_euler
    # euler angle
    e = np.array([0.3, 1.0, -2.0])
    x = (ch4["posi_ref"] @ rot_zyx(*e).T).ravel()
    i_e = [coord.index[f"euler_{d}"] for d in ("yaw", "pitch", "roll")]

    np.testing.assert_allclose(np.asarray(coord.get_value(x))[i_e], e, atol=1e-8)
    np.testing.assert_allclose(np.delete(np.asarray(coord.get_const(x)), i_e), 0.0, atol=1e-8)
 
    # B matrix (analytical vs finite difference)
    h = 1e-5
    B = np.asarray(coord.get_const_grad(x))
    B_fd = np.stack([(coord.get_value(x + h * u) - coord.get_value(x - h * u)) / (2 * h)
                     for u in np.eye(x.size)], axis=1)
    np.testing.assert_allclose(B, B_fd, atol=1e-6)
 
    # Hessian contraction
    w = rng.normal(size=B.shape[0])
    w_i = rng.normal(size=x.size)
    V = rng.normal(size=(x.size, 2))
 
    def dB(v):
        return (np.asarray(coord.get_const_grad(x + h * v)) - np.asarray(coord.get_const_grad(x - h * v))) / (2 * h)
 
    kj_fd = np.stack([dB(V[:, a]).T @ w for a in range(V.shape[1])], axis=1)
    ij_fd = np.stack([dB(V[:, a]) @ w_i for a in range(V.shape[1])], axis=1)
    np.testing.assert_allclose(coord.get_const_hess_kj_contract(x, w, V), kj_fd, atol=1e-5)
    np.testing.assert_allclose(coord.get_const_hess_ij_contract(x, w_i, V), ij_fd, atol=1e-5)
 
def test_gimbal_lock_condition_number(ch4, ch4_euler):
    conds = []
    for p in [1.0, 0.1, 0.01, 0.001]:
        x = (ch4["posi_ref"] @ rot_zyx(0.3, p, -2.0).T).ravel()
        conds.append(np.linalg.cond(np.asarray(ch4_euler.get_const_grad(x))))
    assert np.all(np.diff(conds) > 0)
