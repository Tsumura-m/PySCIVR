import jax
import jax.numpy as jnp
import numpy as np

from .coordinates import make_converters

jax.config.update("jax_enable_x64", True)

def make_vari_propagator(coord_const, dt):
    """
    Make a function to propagate the variation of the trajectory.

    coord_const: InternalCoordinates for constraints (K rows)
    dt      : time step

    Returns:
        vari_propagator(posi, mome, lam, mu, hess, vari_init) : (T, 6N, 2m) propagated variation of the trajectory
    """

    const_grad_func = coord_const.get_const_grad            # (3N) -> (K, 3N)
    const_hess_kj_func = coord_const.get_const_hess_kj_contract   # (3N), (K,), (3N, 2m) -> (3N, 2m)
    const_hess_ij_func = coord_const.get_const_hess_ij_contract   # (3N), (3N), (3N, 2m) -> (K, 2m)
    mass_inv_flat = jnp.asarray(coord_const.mass_inv_flat, dtype=float)

    def oblique_proj(v, G_prev, G_new):
        """Obique projection of v : P_{n+1} v = (I - M^{-1} G_{n}^T (G_{n+1} M^{-1} G_{n}^T)^{-1} G_{n+1}) v
        """
        metric_obl = G_new @ (mass_inv_flat[:, None] * G_prev.T) # G_{n+1} M^{-1} G_{n}^T
        return v - mass_inv_flat[:, None] * G_prev.T @ jnp.linalg.solve(metric_obl, G_new @ v)

    def ortho_proj(v, G, metric_ortho):
        """Orthogonal projection of v : P_{n} v = (I - G_{n}^T (G_{n} M^{-1} G_{n}^T)^{-1} G_{n} M^{-1}) v
        """
        return v - G.T @ jnp.linalg.solve(metric_ortho, G @ (mass_inv_flat[:, None] * v))

    def step(carry, inputs):
        posi_vari_prev, mome_vari_prev, G_prev = carry
        posi_prev, posi_new, mome_new, lam_prev, mu_new, hess_prev, hess_new = inputs
        G_new = const_grad_func(posi_new)  # (K, 3N)

        # position variation
        posi_vari_pred = posi_vari_prev + dt * mass_inv_flat[:, None] * mome_vari_prev \
                         - 0.5 * dt**2 * mass_inv_flat[:, None] * (hess_prev @ posi_vari_prev)
        hess_lam = const_hess_kj_func(posi_prev, lam_prev, posi_vari_prev)  # H^{lambda} delta q_n
        posi_vari_pred += dt * mass_inv_flat[:, None] * hess_lam
        posi_vari_new = oblique_proj(posi_vari_pred, G_prev, G_new)

        # momentum variation
        mome_vari_pred = (posi_vari_new - posi_vari_prev) / (dt * mass_inv_flat[:, None]) \
                         - 0.5 * dt * (hess_new @ posi_vari_new)
        mome_vari_pred += const_hess_kj_func(posi_new, mu_new, posi_vari_new)  # H^{mu} delta q_{n+1}
        metric_ortho = G_new @ (mass_inv_flat[:, None] * G_new.T)  # G_{n+1} M^{-1} G_{n+1}^T
        mome_vari_tangent = ortho_proj(mome_vari_pred, G_new, metric_ortho)
        hess_ij = const_hess_ij_func(posi_new, mass_inv_flat * mome_new, posi_vari_new) # H_{ijk} (M^{-1} p_{n+1})_k
        mome_vari_normal = -G_new.T @ jnp.linalg.solve(metric_ortho, hess_ij)
        mome_vari_new = mome_vari_tangent + mome_vari_normal

        return (posi_vari_new, mome_vari_new, G_new), (posi_vari_new, mome_vari_new)

    @jax.jit
    def run(posi, mome, lam, mu, hess, vari_init):
        n_atom3 = posi.shape[1]
        posi_vari_init = vari_init[:n_atom3]
        mome_vari_init = vari_init[n_atom3:]
        inputs = (posi[:-1], posi[1:], mome[1:], lam[:-1], mu[1:], hess[:-1], hess[1:])
        carry_init = (posi_vari_init, mome_vari_init, const_grad_func(posi[0]))
        _, (posi_vari, mome_vari) = jax.lax.scan(step, carry_init, inputs)
        posi_vari = jnp.concatenate([posi_vari_init[None], posi_vari])
        mome_vari = jnp.concatenate([mome_vari_init[None], mome_vari])
        return jnp.concatenate([posi_vari, mome_vari], axis=1)  # (T, 6N, 2m)

    return run

def make_vari_propagator_implicit(metric_func, dt, *, potential_func=None, normalize=False):
    r"""
    Make a function to propagate the variation of the trajectory using implicit integration.

        H(Q, P) = 1/2 P^T G(Q) P + V(Q),  X(Q, P) = d(G(Q) P) / dQ, Y(Q, P) = d^2 H(Q) / dQ^2

        (I + dt/2 X(Q_n, P_{n+1/2})^T) delta P_{n+1/2} = dP_{n} - dt/2 Y(Q_n, P_{n+1/2}) delta Q_n
        (I - dt/2 X(Q_{n+1}, P_{n+1/2})) delta Q_{n+1} = (I + dt/2 X(Q_n, P_{n+1/2})) delta Q_n
                                                         + dt/2 (G_n + G_{n+1}) dP_{n+1/2}
        dP_{n+1} = (I - dt/2 X(Q_{n+1}, P_{n+1/2})^T) dP_{n+1/2} - dt/2 Y(Q_{n+1}, P_{n+1/2}) delta Q_{n+1}


    metric_func : function to compute the metric G(Q)
    dt          : time step

    potential_func : function to compute the potential V(Q). If None, the potential is assumed to be zero.
    normalize      : whether to normalize the position to unit norm.
                     This is intended to be used for the quaternion dynamics. (q^T q = 1, p^T q = 0)

    returns:
        vari_propagator(posi, mome, mome_half, vari_init) : (T, 2m, 2m) propagated variation of the trajectory
    """

    if potential_func is None:
        potential_func = lambda q: 0.0

    def hamiltonian(posi, mome):
        return 0.5 * mome @ (metric_func(posi) @ mome) + potential_func(posi)

    get_velo_grad = jax.jacrev(lambda q, p: metric_func(q) @ p, argnums=0)  # X = d(G(Q) P) / dQ (m, m)
    get_hess = jax.hessian(hamiltonian, argnums=0)  # Y = d^2 H(Q) / dQ^2 (m, m)

    def step(carry, inputs):
        posi_vari_prev, mome_vari_prev, metric_prev = carry
        posi_prev, posi_new, mome_new, mome_half = inputs
        eye = jnp.eye(len(posi_prev))

        X_prev = get_velo_grad(posi_prev, mome_half)
        X_new = get_velo_grad(posi_new, mome_half)
        Y_prev = get_hess(posi_prev, mome_half)
        Y_new = get_hess(posi_new, mome_half)
        metric_new = metric_func(posi_new)

        # momentum variation (n -> n+1/2)
        mome_vari_half = jnp.linalg.solve(eye + 0.5 * dt * X_prev.T,
                                          mome_vari_prev - 0.5 * dt * Y_prev @ posi_vari_prev)

        # position variation (n -> n+1)
        posi_vari_new = jnp.linalg.solve(eye - 0.5 * dt * X_new, (eye + 0.5 * dt * X_prev) @ posi_vari_prev
                                         + 0.5 * dt * (metric_prev + metric_new) @ mome_vari_half)

        # momentum variation (n+1/2 -> n+1)
        mome_vari_new = (eye - 0.5 * dt * X_new.T) @ mome_vari_half - 0.5 * dt * Y_new @ posi_vari_new

        if normalize:
            posi_vari_new -= jnp.outer(posi_new, posi_new @ posi_vari_new)
            mome_vari_new -= jnp.outer(posi_new, posi_new @ mome_vari_new + mome_new @ posi_vari_new)

        return (posi_vari_new, mome_vari_new, metric_new), (posi_vari_new, mome_vari_new)

    @jax.jit
    def run(posi, mome, mome_half, vari_init):
        n_dim = posi.shape[1]
        posi_vari_init = vari_init[:n_dim]
        mome_vari_init = vari_init[n_dim:]
        inputs = (posi[:-1], posi[1:], mome[1:], mome_half[:-1])
        _, (posi_vari, mome_vari) = jax.lax.scan(step, (posi_vari_init, mome_vari_init, metric_func(posi[0])), inputs)
        posi_vari = jnp.concatenate([posi_vari_init[None], posi_vari])
        mome_vari = jnp.concatenate([mome_vari_init[None], mome_vari])
        return jnp.concatenate([posi_vari, mome_vari], axis=1) # (T, 2m, 2m)

    return run


def propagate_variation(vari_propagator, posi, mome, lam, mu, hess, vari_init):
    """
    Propagate the variation of the trajectory using the linearized equations of motion.

    vari_propagator : function that propagates the variation of the trajectory
    posi    : (T, N, 3) trajectory positions
    mome    : (T, N, 3) trajectory momenta
    lam     : (T, K) Lagrange multipliers for positional constraints
    mu      : (T, K) Lagrange multipliers for momentum constraints
    hess    : (T, 3N, 3N) (projected) Hessian of the potential energy at each time step
    vari_init: (6N, 2m) initial variation, e.g. J_rev(0)

    returns:
        vari : (T, 6N, 2m) propagated variation of the trajectory
    """
    n_step = len(posi)
    vari = vari_propagator(jnp.asarray(np.asarray(posi, dtype=np.float64)).reshape(n_step, -1),
                           jnp.asarray(np.asarray(mome, dtype=np.float64)).reshape(n_step, -1),
                           jnp.asarray(lam), jnp.asarray(mu), jnp.asarray(hess), jnp.asarray(vari_init))
    return np.asarray(vari)

def propagate_variation_implicit(vari_propagator, posi, mome, mome_half, vari_init):
    """
    Propagate the variation of the trajectory using the linearized equations of motion with implicit integration.

    vari_propagator : function that propagates the variation of the trajectory
    posi    : (T, m) trajectory positions
    mome    : (T, m) trajectory momenta
    mome_half : (T, m) trajectory momenta at half time steps
    vari_init: (2m, 2m) initial variation, e.g. I_{2m}

    returns:
        vari : (T, 2m, 2m) propagated variation of the trajectory
    """
    n_step = len(posi)
    vari = vari_propagator(jnp.asarray(np.asarray(posi, dtype=np.float64)).reshape(n_step, -1),
                           jnp.asarray(np.asarray(mome, dtype=np.float64)).reshape(n_step, -1),
                           jnp.asarray(np.asarray(mome_half, dtype=np.float64)).reshape(n_step, -1),
                           jnp.asarray(vari_init))
    return np.asarray(vari)

def make_jacobi_func(coord_all, active_idx):
    """
    Make Jacobian function to transform the variation from the redundant space to the physical space.

    coord_all : InternalCoordinates for the full system (3N rows)
    active_idx : (m,) row indices of the active coordinates in coord_all (order after build())
    mass : (N,) atomic masses

    returns:
        get_jacobi(posi, mome)     : (2m, 6N) redundant -> physical  J
        get_jacobi_rev(posi, mome) : (6N, 2m) physical -> redundant  J_rev
        get_internal(posi, mome)   : position and momentum of the internal coordinates (m,), (m,)
        get_monodromy(posi, mome, vari) : batched over T (jitted)
            (T, 3N), (T, 3N), (T, 6N, 2m) ->
            posi_internal (T, m), mome_internal (T, m), mono_internal (T, 2m, 2m), symplectic_err (T,)
    """

    active_idx = np.asarray(active_idx, dtype=int)
    n_active = len(active_idx)
    skew = jnp.block([[jnp.zeros((n_active, n_active)), jnp.eye(n_active)],
                      [-jnp.eye(n_active), jnp.zeros((n_active, n_active))]])
    get_const_grad = coord_all.get_const_grad
    get_value = coord_all.get_value
    get_hess_kj = coord_all.get_const_hess_kj_contract
    unit_active = jnp.eye(len(coord_all.coords))[:, active_idx]  # (3N, m)
    _, _, _, _get_R_active = make_converters(coord_all, active_idx)  # R_A

    def _get_jacobi_and_internal(posi, mome):
        B = get_const_grad(posi)
        A_active = jnp.linalg.solve(B, unit_active)  # (3N, m)
        mome_all = jnp.linalg.solve(B.T, mome)  # (3N,)  (P = A^T p)
        AP_grad = -get_hess_kj(posi, mome_all, A_active).T  # d(A_A^T p)/dq, (m, 3N)
        upper = jnp.concatenate([B[active_idx], jnp.zeros_like(B[active_idx])], axis=1)  # (m, 6N)
        lower = jnp.concatenate([AP_grad, A_active.T], axis=1)  # (m, 6N)
        jacobi = jnp.concatenate([upper, lower], axis=0)  # (2m, 6N)
        return jacobi, get_value(posi)[active_idx], mome_all[active_idx]

    @jax.jit
    def get_jacobi(posi, mome):
        return _get_jacobi_and_internal(posi.reshape(-1), mome.reshape(-1))[0]

    @jax.jit
    def get_jacobi_rev(posi, mome):
        posi = posi.reshape(-1)
        mome = mome.reshape(-1)
        A_active = jnp.linalg.solve(get_const_grad(posi), unit_active)  # (3N, m)
        R_active = _get_R_active(posi)  # (3N, m)
        mome_internal = A_active.T @ mome  # (m,)
        RP_grad = jax.vmap(lambda a : jax.jvp(lambda x : _get_R_active(x) @ mome_internal, (posi,), (a,))[1],
                           in_axes=1, out_axes=1)(A_active)  # (3N, m)
        upper = jnp.concatenate([A_active, jnp.zeros_like(A_active)], axis=1)  # (3N, 2m)
        lower = jnp.concatenate([RP_grad, R_active], axis=1)  # (3N, 2m)

        return jnp.concatenate([upper, lower], axis=0)  # (6N, 2m)

    @jax.jit
    def get_internal(posi, mome):
        _, posi_internal, mome_internal = _get_jacobi_and_internal(posi.reshape(-1), mome.reshape(-1))
        return posi_internal, mome_internal  # (m,), (m,)

    @jax.jit
    def get_monodromy(posi_flat, mome_flat, vari):
        """
        posi_flat : (T, 3N) positions in the redundant space
        mome_flat : (T, 3N) momenta in the redundant space
        vari      : (T, 6N, 2m) variation in the redundant space
        """
        jacobi, posi_internal, mome_internal = jax.vmap(_get_jacobi_and_internal)(posi_flat, mome_flat)
        mono_internal = jacobi @ vari
        symplectic_err = jnp.max(jnp.abs(mono_internal.transpose(0, 2, 1) @ skew @ mono_internal - skew), axis=(1, 2))
        return posi_internal, mome_internal, mono_internal, symplectic_err

    return get_jacobi, get_jacobi_rev, get_internal, get_monodromy

def to_internal(monodromy_func, posi, mome, vari):
    """Transform the position, momentum, and variation from the redundant space to the physical space.
    The symplectic errors are also returned.

    monodromy_func : get_monodromy return
    posi : (T, N, 3) positions in the redundant space
    mome : (T, N, 3) momenta in the redundant space
    vari : (T, 6N, 2m) variation in the redundant space

    returns:
        posi_internal : (T, m) positions in the physical space
        mome_internal : (T, m) momenta in the physical space
        mono_internal : (T, 2m, 2m) monodromy matrix in the physical space (M^{phys} = J (M^{red} J_rev))
        symplectic_err : (T,) symplectic errors of the variation
    """

    n_step = posi.shape[0]
    posi_flat = jnp.asarray(np.asarray(posi, dtype=float).reshape(n_step, -1))
    mome_flat = jnp.asarray(np.asarray(mome, dtype=float).reshape(n_step, -1))
    posi_internal, mome_internal, mono_internal, symplectic_err = \
        monodromy_func(posi_flat, mome_flat, jnp.asarray(vari))

    return np.asarray(posi_internal), np.asarray(mome_internal), \
        np.asarray(mono_internal), np.asarray(symplectic_err)
