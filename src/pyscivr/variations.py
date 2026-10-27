"""Propagation of the variations (monodromy matrices) along classical trajectories.

See section II C and appendices E and J 3 of the paper for details.

Notation:
    - ``N``: number of atoms.
    - ``T``: number of time steps of the trajectory.
    - ``K``: number of constraints.
    - ``m``: number of DOFs of the active coordinates.
    - Redundant space: ``(3N,)``-dimensional space of Cartesian coordinates with ``K`` constraints.
    - Physical space: ``(m,)``-dimensional space of the active coordinates.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .coordinates import make_converters

if TYPE_CHECKING:
    from .coordinates import InternalCoordinates

jax.config.update("jax_enable_x64", True)

##################################################################################################
# SHAKE/RATTLE variation propagator
##################################################################################################


def make_vari_propagator(coord_const: InternalCoordinates, dt: float) -> Callable[..., jax.Array]:
    r"""Make a function to propagate the variation along a SHAKE/RATTLE trajectory.

    The linearized SHAKE/RATTLE equations (Eqs. (31)-(37) in the paper) are
    integrated along the trajectory.

    Args:
        coord_const: InternalCoordinates defining the ``K`` constraints,
            the same as used in :class:`pyscivr.integrators.ShakeRattle`.
        dt: time step of the trajectory.

    Returns:
        Jitted function ``vari_propagator(posi, mome, lam, mu, hess, vari_init)``
        returning the ``(T, 6N, 2m)`` propagated variation of the trajectory.
        Use it through :func:`propagate_variation`.
    """
    const_grad_func = coord_const.get_const_grad  # (3N) -> (K, 3N)
    const_hess_kj_func = coord_const.get_const_hess_kj_contract  # (3N), (K,), (3N, 2m) -> (3N, 2m)
    const_hess_ij_func = coord_const.get_const_hess_ij_contract  # (3N), (3N), (3N, 2m) -> (K, 2m)
    mass_inv_flat = jnp.asarray(coord_const.mass_inv_flat, dtype=float)

    def oblique_proj(v: jax.Array, G_prev: jax.Array, G_new: jax.Array) -> jax.Array:  # Eq. (33)
        """Oblique projection of v: P_{n+1} v = (I - M^{-1} G_{n}^T (G_{n+1} M^{-1} G_{n}^T)^{-1} G_{n+1}) v."""
        metric_obl = G_new @ (mass_inv_flat[:, None] * G_prev.T)  # G_{n+1} M^{-1} G_{n}^T
        return v - mass_inv_flat[:, None] * G_prev.T @ jnp.linalg.solve(metric_obl, G_new @ v)

    def ortho_proj(v: jax.Array, G: jax.Array, metric_ortho: jax.Array) -> jax.Array:  # Eq. (31)
        """Orthogonal projection of v: P_{n} v = (I - G_{n}^T (G_{n} M^{-1} G_{n}^T)^{-1} G_{n} M^{-1}) v."""
        return v - G.T @ jnp.linalg.solve(metric_ortho, G @ (mass_inv_flat[:, None] * v))

    def step(
        carry: tuple[jax.Array, ...], inputs: tuple[jax.Array, ...]
    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        posi_vari_prev, mome_vari_prev, G_prev = carry
        posi_prev, posi_new, mome_new, lam_prev, mu_new, hess_prev, hess_new = inputs
        G_new = const_grad_func(posi_new)  # (K, 3N)

        # position variation
        # Eq. (35)
        posi_vari_pred = (
            posi_vari_prev
            + dt * mass_inv_flat[:, None] * mome_vari_prev
            - 0.5 * dt**2 * mass_inv_flat[:, None] * (hess_prev @ posi_vari_prev)
        )
        # Eq. (34)
        hess_lam = const_hess_kj_func(posi_prev, lam_prev, posi_vari_prev)  # H^{\lambda} \delta q_n
        posi_vari_pred += dt * mass_inv_flat[:, None] * hess_lam
        posi_vari_new = oblique_proj(posi_vari_pred, G_prev, G_new)

        # momentum variation
        mome_vari_pred = (posi_vari_new - posi_vari_prev) / (dt * mass_inv_flat[:, None]) - 0.5 * dt * (
            hess_new @ posi_vari_new
        )
        # Eq. (37)
        mome_vari_pred += const_hess_kj_func(posi_new, mu_new, posi_vari_new)  # H^{\mu} \delta q_{n+1}
        # Eq. (36)
        metric_ortho = G_new @ (mass_inv_flat[:, None] * G_new.T)  # G_{n+1} M^{-1} G_{n+1}^T
        mome_vari_tangent = ortho_proj(mome_vari_pred, G_new, metric_ortho)
        hess_ij = const_hess_ij_func(posi_new, mass_inv_flat * mome_new, posi_vari_new)  # H_{ijk} (M^{-1} p_{n+1})_k
        mome_vari_normal = -G_new.T @ jnp.linalg.solve(metric_ortho, hess_ij)
        mome_vari_new = mome_vari_tangent + mome_vari_normal

        return (posi_vari_new, mome_vari_new, G_new), (posi_vari_new, mome_vari_new)

    @jax.jit
    def run(
        posi: jax.Array, mome: jax.Array, lam: jax.Array, mu: jax.Array, hess: jax.Array, vari_init: jax.Array
    ) -> jax.Array:
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


def propagate_variation(
    vari_propagator: Callable[..., jax.Array],
    posi: ArrayLike,
    mome: ArrayLike,
    lam: ArrayLike,
    mu: ArrayLike,
    hess: ArrayLike,
    vari_init: ArrayLike,
) -> NDArray[np.float64]:
    r"""Propagate the variation along a SHAKE/RATTLE trajectory by the linearized equations of motion.

    Args:
        vari_propagator: Function made by :func:`make_vari_propagator`.
        posi: ``(T, N, 3)`` positions of the trajectory.
        mome: ``(T, N, 3)`` momenta of the trajectory.
        lam: ``(T, K)`` Lagrange multipliers for positional constraints.
        mu: ``(T, K)`` Lagrange multipliers for momentum constraints.
        hess: ``(T, 3N, 3N)`` (projected) Hessian of the potential energy at each time step.
        vari_init: ``(6N, 2m)`` initial variation, e.g. ``get_jacobi_rev(posi[0], mome[0])``
            from :func:`make_jacobi_func`.

    Returns:
        ``(T, 6N, 2m)`` propagated variation of the trajectory.

    Note:
        Instead of the ``(6N, 6N)`` full monodromy matrix :math:`M_t^{\text{red}}` (Eq. (32)),
        the variation in the redundant space is retained as the ``(6N, 2m)`` matrix product
        :math:`M_t^{\text{red}} J_{\text{rev}}(0)` (Eq. (42)).
        The variation in the physical space can be obtained by
        :math:`J(t) (M_t^{\text{red}} J_{\text{rev}}(0))` (Eq. (44)),
        and the initial variation in the redundant space is :math:`J_{\text{rev}}(0)`,
        since :math:`J_{\text{rev}}(0) J(0) J_{\text{rev}}(0) = J_{\text{rev}}(0)` (Eq. (47)).
    """
    n_step = len(posi)
    vari = vari_propagator(
        jnp.asarray(np.asarray(posi, dtype=np.float64)).reshape(n_step, -1),
        jnp.asarray(np.asarray(mome, dtype=np.float64)).reshape(n_step, -1),
        jnp.asarray(lam),
        jnp.asarray(mu),
        jnp.asarray(hess),
        jnp.asarray(vari_init),
    )
    return np.asarray(vari)


##################################################################################################
# Implicit Verlet variation propagator
##################################################################################################


def make_vari_propagator_implicit(
    metric_func: Callable[[jax.Array], jax.Array],
    dt: float,
    *,
    potential_func: Callable[[jax.Array], jax.Array] | None = None,
    normalize: bool = False,
) -> Callable[..., jax.Array]:
    r"""Make a function to propagate the variation along an implicit Verlet trajectory.

    For :math:`H(Q, P) = \frac{1}{2} P^T G(Q) P + V(Q)`, with

    .. math::

        X(Q, P) = \frac{\partial (G(Q) P)}{\partial Q}, \quad Y(Q, P) = \frac{\partial^2 H}{\partial Q^2}

    the linearized implicit Verlet equations (Eqs. (J13)-(J14) in the paper) are

    .. math::

        &(I + \frac{\Delta t}{2} X(Q_n, P_{n+\frac{1}{2}})^T) \delta P_{n+\frac{1}{2}} =
        \delta P_{n} - \frac{\Delta t}{2} Y(Q_n, P_{n+\frac{1}{2}}) \delta Q_n

        &(I - \frac{\Delta t}{2} X(Q_{n+1}, P_{n+\frac{1}{2}})) \delta Q_{n+1} =
        (I + \frac{\Delta t}{2} X(Q_n, P_{n+\frac{1}{2}})) \delta Q_n
        + \frac{\Delta t}{2} (G_n + G_{n+1}) \delta P_{n+\frac{1}{2}}

        &\delta P_{n+1} = (I - \frac{\Delta t}{2} X(Q_{n+1}, P_{n+\frac{1}{2}})^T) \delta P_{n+\frac{1}{2}}
        - \frac{\Delta t}{2} Y(Q_{n+1}, P_{n+\frac{1}{2}}) \delta Q_{n+1}

    Args:
        metric_func: Function to compute the metric :math:`G(Q)`, ``(m,)`` -> ``(m, m)``,
            the same as used in :class:`pyscivr.integrators.ImplicitVerlet`.
        dt: Time step.
        potential_func: Function to compute the potential :math:`V(Q)`, ``(m,)`` -> scalar.
            If ``None``, the potential is assumed to be zero.
        normalize: If ``True``, the variations are projected at every step to satisfy the
            linearized constraints ``q^T dq = 0`` and ``q^T dp + p^T dq = 0`` of
            ``q^T q = 1`` and ``p^T q = 0``. This is intended for the quaternion dynamics.

    Returns:
        Jitted function ``vari_propagator(posi, mome, mome_half, vari_init)``
        returning the ``(T, 2m, 2m)`` propagated variation of the trajectory.
        Use it through :func:`propagate_variation_implicit`.
    """
    if potential_func is None:
        potential_func = lambda q: 0.0

    def hamiltonian(posi: jax.Array, mome: jax.Array) -> jax.Array:
        return 0.5 * mome @ (metric_func(posi) @ mome) + potential_func(posi)

    get_velo_grad = jax.jacrev(lambda q, p: metric_func(q) @ p, argnums=0)  # X = d(G(Q) P) / dQ (m, m)
    get_hess = jax.hessian(hamiltonian, argnums=0)  # Y = d^2 H(Q) / dQ^2 (m, m)

    def step(
        carry: tuple[jax.Array, ...], inputs: tuple[jax.Array, ...]
    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        posi_vari_prev, mome_vari_prev, metric_prev = carry
        posi_prev, posi_new, mome_new, mome_half = inputs
        eye = jnp.eye(len(posi_prev))

        X_prev = get_velo_grad(posi_prev, mome_half)
        X_new = get_velo_grad(posi_new, mome_half)
        Y_prev = get_hess(posi_prev, mome_half)
        Y_new = get_hess(posi_new, mome_half)
        metric_new = metric_func(posi_new)

        # momentum variation (n -> n+1/2)
        mome_vari_half = jnp.linalg.solve(
            eye + 0.5 * dt * X_prev.T, mome_vari_prev - 0.5 * dt * Y_prev @ posi_vari_prev
        )

        # position variation (n -> n+1)
        posi_vari_new = jnp.linalg.solve(
            eye - 0.5 * dt * X_new,
            (eye + 0.5 * dt * X_prev) @ posi_vari_prev + 0.5 * dt * (metric_prev + metric_new) @ mome_vari_half,
        )

        # momentum variation (n+1/2 -> n+1)
        mome_vari_new = (eye - 0.5 * dt * X_new.T) @ mome_vari_half - 0.5 * dt * Y_new @ posi_vari_new

        if normalize:
            posi_vari_new -= jnp.outer(posi_new, posi_new @ posi_vari_new)
            mome_vari_new -= jnp.outer(posi_new, posi_new @ mome_vari_new + mome_new @ posi_vari_new)

        return (posi_vari_new, mome_vari_new, metric_new), (posi_vari_new, mome_vari_new)

    @jax.jit
    def run(posi: jax.Array, mome: jax.Array, mome_half: jax.Array, vari_init: jax.Array) -> jax.Array:
        n_dim = posi.shape[1]
        posi_vari_init = vari_init[:n_dim]
        mome_vari_init = vari_init[n_dim:]
        inputs = (posi[:-1], posi[1:], mome[1:], mome_half[:-1])
        _, (posi_vari, mome_vari) = jax.lax.scan(step, (posi_vari_init, mome_vari_init, metric_func(posi[0])), inputs)
        posi_vari = jnp.concatenate([posi_vari_init[None], posi_vari])
        mome_vari = jnp.concatenate([mome_vari_init[None], mome_vari])
        return jnp.concatenate([posi_vari, mome_vari], axis=1)  # (T, 2m, 2m)

    return run


def propagate_variation_implicit(
    vari_propagator: Callable[..., jax.Array],
    posi: ArrayLike,
    mome: ArrayLike,
    mome_half: ArrayLike,
    vari_init: ArrayLike,
) -> NDArray[np.float64]:
    """Propagate the variation along an implicit Verlet trajectory by the linearized equations of motion.

    Args:
        vari_propagator: Function made by :func:`make_vari_propagator_implicit`.
        posi: ``(T, m)`` positions of the trajectory.
        mome: ``(T, m)`` momenta of the trajectory.
        mome_half: ``(T, m)`` momenta at half time steps of the trajectory.
        vari_init: ``(2m, 2m)`` initial variation, e.g. the identity matrix ``I_{2m}``.

    Returns:
        ``(T, 2m, 2m)`` propagated variation of the trajectory (monodromy matrix).
    """
    n_step = len(posi)
    vari = vari_propagator(
        jnp.asarray(np.asarray(posi, dtype=np.float64)).reshape(n_step, -1),
        jnp.asarray(np.asarray(mome, dtype=np.float64)).reshape(n_step, -1),
        jnp.asarray(np.asarray(mome_half, dtype=np.float64)).reshape(n_step, -1),
        jnp.asarray(vari_init),
    )
    return np.asarray(vari)


def make_jacobi_func(
    coord_all: InternalCoordinates, active_idx: ArrayLike
) -> tuple[
    Callable[..., jax.Array],
    Callable[..., jax.Array],
    Callable[..., tuple[jax.Array, jax.Array]],
    Callable[..., tuple[jax.Array, ...]],
]:
    r"""Make Jacobian functions between the redundant and physical spaces (Eq. (42) in the paper).

    Args:
        coord_all: InternalCoordinates for the full system. It must define ``3N``
            coordinates for coordinate transformation.
        active_idx: ``(m,)`` row indices of the active coordinates in ``coord_all``,
            in the row order after :meth:`build()`.

    Returns:
        ``(get_jacobi, get_jacobi_rev, get_internal, get_monodromy)``, where
        all functions are jitted:

        - ``get_jacobi(posi, mome)``: ``(2m, 6N)`` Jacobian ``J`` from the
          redundant space to the physical space.
        - ``get_jacobi_rev(posi, mome)``: ``(6N, 2m)`` Jacobian ``J_rev`` from the
          physical space to the redundant space.
        - ``get_internal(posi, mome)``: ``(m,)`` positions, ``(m,)`` momenta
          in the physical space.
        - ``get_monodromy(posi_flat, mome_flat, vari)``: batched over ``T``.
          Takes ``(T, 3N)`` positions, ``(T, 3N)`` momenta, and ``(T, 6N, 2m)`` variations
          in the redundant space, and returns ``(T, m)`` positions,
          ``(T, m)`` momenta, ``(T, 2m, 2m)`` monodromy matrices,
          and ``(T,)`` symplectic errors in the physical space.
          Use it through :func:`to_internal`.
    """
    active_idx = np.asarray(active_idx, dtype=int)
    n_active = len(active_idx)
    skew = jnp.block(
        [[jnp.zeros((n_active, n_active)), jnp.eye(n_active)], [-jnp.eye(n_active), jnp.zeros((n_active, n_active))]]
    )
    get_const_grad = coord_all.get_const_grad
    get_value = coord_all.get_value
    get_hess_kj = coord_all.get_const_hess_kj_contract
    unit_active = jnp.eye(len(coord_all))[:, active_idx]  # (3N, m)
    _, _, _, _get_R_active = make_converters(coord_all, active_idx)  # R_A

    def _get_jacobi_and_internal(posi: jax.Array, mome: jax.Array) -> tuple[jax.Array, ...]:
        B = get_const_grad(posi)
        A_active = jnp.linalg.solve(B, unit_active)  # (3N, m)
        mome_all = jnp.linalg.solve(B.T, mome)  # (3N,)  (P = A^T p)
        AP_grad = -get_hess_kj(posi, mome_all, A_active).T  # d(A_A^T p)/dq, (m, 3N)
        upper = jnp.concatenate([B[active_idx], jnp.zeros_like(B[active_idx])], axis=1)  # (m, 6N)
        lower = jnp.concatenate([AP_grad, A_active.T], axis=1)  # (m, 6N)
        jacobi = jnp.concatenate([upper, lower], axis=0)  # (2m, 6N)
        return jacobi, get_value(posi)[active_idx], mome_all[active_idx]

    @jax.jit
    def get_jacobi(posi: jax.Array, mome: jax.Array) -> jax.Array:
        return _get_jacobi_and_internal(posi.reshape(-1), mome.reshape(-1))[0]

    @jax.jit
    def get_jacobi_rev(posi: jax.Array, mome: jax.Array) -> jax.Array:
        posi = posi.reshape(-1)
        mome = mome.reshape(-1)
        A_active = jnp.linalg.solve(get_const_grad(posi), unit_active)  # (3N, m)
        R_active = _get_R_active(posi)  # (3N, m)
        mome_internal = A_active.T @ mome  # (m,)
        RP_grad = jax.vmap(
            lambda a: jax.jvp(lambda x: _get_R_active(x) @ mome_internal, (posi,), (a,))[1], in_axes=1, out_axes=1
        )(A_active)  # (3N, m)
        upper = jnp.concatenate([A_active, jnp.zeros_like(A_active)], axis=1)  # (3N, 2m)
        lower = jnp.concatenate([RP_grad, R_active], axis=1)  # (3N, 2m)

        return jnp.concatenate([upper, lower], axis=0)  # (6N, 2m)

    @jax.jit
    def get_internal(posi: jax.Array, mome: jax.Array) -> tuple[jax.Array, jax.Array]:
        _, posi_internal, mome_internal = _get_jacobi_and_internal(posi.reshape(-1), mome.reshape(-1))
        return posi_internal, mome_internal  # (m,), (m,)

    @jax.jit
    def get_monodromy(posi_flat: jax.Array, mome_flat: jax.Array, vari: jax.Array) -> tuple[jax.Array, ...]:
        jacobi, posi_internal, mome_internal = jax.vmap(_get_jacobi_and_internal)(posi_flat, mome_flat)
        mono_internal = jacobi @ vari
        symplectic_err = jnp.max(jnp.abs(mono_internal.transpose(0, 2, 1) @ skew @ mono_internal - skew), axis=(1, 2))
        return posi_internal, mome_internal, mono_internal, symplectic_err

    return get_jacobi, get_jacobi_rev, get_internal, get_monodromy


def to_internal(
    monodromy_func: Callable[..., tuple[jax.Array, ...]], posi: ArrayLike, mome: ArrayLike, vari: ArrayLike
) -> tuple[NDArray[np.float64], ...]:
    r"""Transform the position, momentum, and variations from the redundant space to the physical space.

    The monodromy matrix in the physical space is obtained as
    :math:`M^{\text{phys}} = J(t) (M_t^{\text{red}} J_{\text{rev}}(0))`.

    Args:
        monodromy_func: ``get_monodromy`` function made by :func:`make_jacobi_func`.
        posi: ``(T, N, 3)`` positions in the redundant space.
        mome: ``(T, N, 3)`` momenta in the redundant space.
        vari: ``(T, 6N, 2m)`` variations in the redundant space.

    Returns:
        ``(posi_internal, mome_internal, mono_internal, symplectic_err)``, where

        - ``posi_internal``: ``(T, m)`` positions in the physical space.
        - ``mome_internal``: ``(T, m)`` momenta in the physical space.
        - ``mono_internal``: ``(T, 2m, 2m)`` monodromy matrix in the physical space.
        - ``symplectic_err``: ``(T,)`` symplectic errors
          :math:`\epsilon_t = \max |M^{\text{phys}}(t)^T \Omega M^{\text{phys}}(t) - \Omega|`,
          where :math:`\Omega \equiv \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}` is
          the symplectic matrix (Eq. (82)).
    """
    n_step = len(posi)
    posi_flat = jnp.asarray(np.asarray(posi, dtype=float).reshape(n_step, -1))
    mome_flat = jnp.asarray(np.asarray(mome, dtype=float).reshape(n_step, -1))
    posi_internal, mome_internal, mono_internal, symplectic_err = monodromy_func(
        posi_flat, mome_flat, jnp.asarray(vari)
    )

    return np.asarray(posi_internal), np.asarray(mome_internal), np.asarray(mono_internal), np.asarray(symplectic_err)
