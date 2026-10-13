"""Integrators for classical trajectories in Cartesian and curvilinear coordinates.

See sections II B 2 and appendices J 2 and L of the paper for details.

Notation:
    - ``N``: number of atoms.
    - ``T``: number of time steps of the trajectory.
    - ``K``: number of constraints.
    - ``m``: number of DOFs of the curvilinear coordinates
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .coordinates import InternalCoordinates
    from .potentials import Calculator

jax.config.update("jax_enable_x64", True)


##################################################################################################
# base class
##################################################################################################

class Integrator(ABC):
    r"""Base class for integrators in Cartesian coordinates.

    The trajectory returned by :meth:`run` and :meth:`get_trajectory` is a dictionary with the following keys,
    in addition to the extra keys of each subclass:

    - ``posi``: ``(T, N, 3)`` positions.
    - ``mome``: ``(T, N, 3)`` momenta.
    - ``potential_energy``: ``(T,)`` potential energies.
    - ``force``: ``(T, N, 3)`` forces.
    - ``total_energy``: ``(T,)`` total energies.
    - ``dt``: Time step.
    - ``mass``: ``(N,)`` atomic masses.

    Args:
        calculator: Calculator whose ``energy_forces`` accepts ``(N, 3)`` positions and returns ``(energy, forces)``.
        posi: ``(N, 3)`` initial positions.
        mome: ``(N, 3)`` initial momenta.
        mass: ``(N,)`` atomic masses.
        dt: Time step.
        i_step: Initial step index.
    """

    def __init__(self,
            calculator: Calculator,
            posi: ArrayLike,
            mome: ArrayLike,
            mass: ArrayLike,
            dt: float,
            *,
            i_step: int = 0) -> None:
        self.calculator = calculator
        self.posi = np.asarray(posi, dtype=float).reshape(-1, 3)
        self.mome = np.asarray(mome, dtype=float).reshape(-1, 3)
        self.mass = np.asarray(mass, dtype=float)

        self.i_step = i_step
        self.dt = float(dt)
        self.potential_energy, self.force = self.calculator.energy_forces(self.posi)
        self.records = {}

    @abstractmethod
    def step(self) -> dict[str, Any]:
        """Advance the state by one step.

        Returns:
            Extra records for the current step.
        """
        pass

    def get_kinetic_energy(self) -> float:
        """Get the kinetic energy of the current state."""
        return 0.5 * np.sum(self.mome**2 / self.mass[:, None])

    def get_potential_energy(self) -> float:
        """Get the potential energy of the current state."""
        return self.potential_energy

    def get_total_energy(self) -> float:
        """Get the total energy of the current state."""
        return self.get_kinetic_energy() + self.potential_energy

    def _initial_extra(self) -> dict[str, Any]:
        """Extra records for the initial step."""
        return {}


    def _record(self, extra: dict[str, Any]) -> None:
        """Record the current state."""
        rec = dict(posi=self.posi, mome=self.mome, potential_energy=self.potential_energy,
                   force=self.force, total_energy=self.get_total_energy(), **extra)
        for key, val in rec.items():
            self.records.setdefault(key, []).append(val)


    def run(self, n_step: int) -> dict[str, Any]:
        """Run the integrator for ``n_step`` steps.

        The initial state is recorded at the first call.

        Args:
            n_step: Number of steps to run.

        Returns:
            Trajectory from the initial step.
        """
        if not self.records:
            self._record(self._initial_extra())
        for _ in range(n_step):
            extra = self.step()
            self._record(extra)
            self.i_step += 1
        return self.get_trajectory()

    def get_trajectory(self) -> dict[str, Any]:
        """Get the trajectory as a dictionary of arrays."""
        traj = {key: np.stack([np.asarray(v) for v in val]) for key, val in self.records.items()}
        traj["dt"] = self.dt
        traj["mass"] = self.mass
        return traj


class Verlet(Integrator):
    """Velocity Verlet integrator.

    Args:
        calculator: Calculator whose ``energy_forces`` accepts ``(N, 3)`` positions and returns ``(energy, forces)``.
        posi: ``(N, 3)`` initial positions.
        mome: ``(N, 3)`` initial momenta.
        mass: ``(N,)`` atomic masses.
        dt: Time step.
        i_step: Initial step index.
    """

    def step(self) -> dict[str, Any]:
        """Advance the state by one step using the velocity Verlet algorithm."""
        self.mome += 0.5 * self.dt * self.force
        self.posi += self.dt * self.mome / self.mass[:, None]
        self.potential_energy, self.force = self.calculator.energy_forces(self.posi)
        self.mome += 0.5 * self.dt * self.force
        return {}


##################################################################################################
# SHAKE/RATTLE integrator
##################################################################################################

def _pinv(G: jax.Array, rcond: float) -> jax.Array:
    """Compute the pseudo-inverse of G using SVD."""
    U, S, Vt = jnp.linalg.svd(G, full_matrices=False)
    S_inv = jnp.where(S > rcond * S[0], 1.0 / S, 0.0)
    return (Vt.T * S_inv[None, :]) @ U.T  # (3N, K)

def _build_shake_rattle(
        coord_const: InternalCoordinates,
        dt: float,
        mass: tuple[float, ...],
        max_iter_shake: int,
        tol_shake: float,
        rcond: float) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], jax.Array]:
    """Build the jitted functions of SHAKE/RATTLE integrator."""
    sqrt_mass_flat = jnp.sqrt(np.repeat(mass, 3))
    const_func = coord_const.get_const
    const_grad_func = coord_const.get_const_grad

    def get_G_mwc(posi_mwc: jax.Array) -> jax.Array:
        """Get the gradient of the constraints in the mass-weighted coordinates."""
        return const_grad_func(posi_mwc / sqrt_mass_flat) / sqrt_mass_flat[None, :]

    @jax.jit
    def initialize(posi_mwc: jax.Array, mome_mwc: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Initialize the SHAKE/RATTLE integrator."""
        G_mwc = get_G_mwc(posi_mwc)
        G_pinv_mwc = _pinv(G_mwc, rcond)
        mome_mwc -= G_pinv_mwc @ (G_mwc @ mome_mwc)
        return G_mwc, G_pinv_mwc, mome_mwc

    @jax.jit
    def shake_step(
            posi_mwc: jax.Array,
            mome_mwc: jax.Array,
            force: jax.Array,
            G_pinv_mwc: jax.Array) -> tuple[jax.Array, ...]: # Eqs. (25)-(26)
        """Perform a SHAKE step."""
        force_mwc = force.reshape(-1) / sqrt_mass_flat
        mome_mwc_half = mome_mwc + 0.5 * dt * force_mwc
        posi_mwc_pred = posi_mwc + dt * mome_mwc_half

        def cond(carry):
            i, _, f = carry
            return (i < max_iter_shake) & (jnp.max(jnp.abs(f)) > tol_shake)

        def body_func(carry):
            i, x, f = carry   # i: iteration count, x: current position, f: residual of constraints
            x -= G_pinv_mwc @ f
            return i + 1, x, const_func(x / sqrt_mass_flat)

        f_init = const_func(posi_mwc_pred / sqrt_mass_flat)
        # Eq. (28)
        n_iter, posi_mwc_new, f_final = jax.lax.while_loop(cond, body_func, (0, posi_mwc_pred, f_init))

        delta_posi_mwc = posi_mwc_new - posi_mwc_pred
        lam = G_pinv_mwc.T @ delta_posi_mwc / dt
        mome_mwc_half += delta_posi_mwc / dt

        return posi_mwc_new, posi_mwc_new / sqrt_mass_flat, mome_mwc_half, \
                lam, n_iter, jnp.max(jnp.abs(f_final))

    @jax.jit
    def rattle_step(posi_mwc_new: jax.Array,
                    mome_mwc_half: jax.Array,
                    force_new: jax.Array) -> tuple[jax.Array, ...]: # Eq. (27)
        """Perform a RATTLE step."""
        force_mwc_new = force_new.reshape(-1) / sqrt_mass_flat
        G_mwc = get_G_mwc(posi_mwc_new)
        G_pinv_mwc = _pinv(G_mwc, rcond)
        mome_mwc_pred = mome_mwc_half + 0.5 * dt * force_mwc_new
        # Eq. (30)
        mu = -G_pinv_mwc.T @ mome_mwc_pred
        mome_mwc_new = mome_mwc_pred + G_mwc.T @ mu

        return mome_mwc_new, mome_mwc_new * sqrt_mass_flat, mu, G_pinv_mwc

    return initialize, shake_step, rattle_step, sqrt_mass_flat

class ShakeRattle(Integrator):
    r"""SHAKE/RATTLE integrator with holonomic constraints :math:`F(q) = 0`.

    The integration is performed in the mass-weighted coordinate (Eqs. (25)-(31) in the paper).
    The initial momenta are projected to the tangent space of the constraints.

    .. math::

        p_{n+\frac{1}{2}} &= p_n - \frac{\Delta t}{2}\nabla V(q_n)  + \mathcal{G}(q_n)^T \lambda

        q_{n+1} &= q_n +  \Delta tM^{-1}p_{n+\frac{1}{2}}

        p_{n+1} &= p_{n+\frac{1}{2}}-\frac{\Delta t}{2}\nabla V(q_{n+1}) +\mathcal{G}(q_{n+1})^T \mu

    In addition to the keys of :class:`Integrator`, the trajectory contains:

    - ``lam``: ``(T, K)`` Lagrange multipliers for position constraints.
    - ``mu``: ``(T, K)`` Lagrange multipliers for momentum constraints.
    - ``n_iter_shake``: ``(T,)`` number of iterations for SHAKE.
    - ``err_shake``: ``(T,)`` final ``max|F(q)|`` of SHAKE.

    Args:
        calculator: Calculator whose ``energy_forces`` accepts ``(N, 3)`` positions
            and returns ``(energy, forces)``.
        posi: ``(N, 3)`` initial positions, which must satisfy the constraints within ``tol_init``.
        mome: ``(N, 3)`` initial momenta.
        mass: ``(N,)`` atomic masses.
        dt: Time step.
        coord_const: InternalCoordinates whose residuals (``get_const``) define the ``K`` holonomic constraints.
        max_iter_shake: Maximum number of iterations for SHAKE.
        tol_shake: Tolerance for SHAKE convergence.
        rcond: Cutoff for pseudo-inverse computation.
        tol_init: Tolerance for initial constraint satisfaction.
        i_step: Initial step index.

    Raises:
        AssertionError: If the initial positions do not satisfy the constraints within ``tol_init``.
    """

    def __init__(self,
            calculator: Calculator,
            posi: ArrayLike,
            mome: ArrayLike,
            mass: ArrayLike,
            dt: float,
            coord_const: InternalCoordinates,
            *,
            max_iter_shake: int = 50,
            tol_shake: float = 1e-10,
            rcond: float = 1e-10,
            tol_init: float = 1e-8,
            i_step: int = 0):

        super().__init__(calculator, posi, mome, mass, dt, i_step=i_step)
        initialize, self._shake_step, self._rattle_step, self._sqrt_mass_flat = \
            _build_shake_rattle(coord_const, self.dt, tuple(self.mass.tolist()), max_iter_shake, tol_shake, rcond)

        # initialize
        err = np.max(np.abs(np.asarray(coord_const.get_const(self.posi.reshape(-1)))))
        assert err < tol_init, f"Initial positions do not satisfy constraints: max|F(q)| = {err}"

        self._posi_mwc = jnp.asarray(self.posi.reshape(-1) * self._sqrt_mass_flat)
        self._mome_mwc = jnp.asarray(self.mome.reshape(-1) / self._sqrt_mass_flat)
        _, self._G_pinv_mwc, self._mome_mwc = initialize(self._posi_mwc, self._mome_mwc)
        self.mome = np.asarray(self._mome_mwc * self._sqrt_mass_flat).reshape(-1, 3)
        self.n_const = int(self._G_pinv_mwc.shape[1])

    def _initial_extra(self) -> dict[str, Any]:
        zeros = np.zeros((self.n_const))
        return dict(lam=zeros, mu=zeros, n_iter_shake=0, err_shake=0.0)

    def step(self) -> dict[str, Any]:
        """Advance the state by one step using the SHAKE/RATTLE algorithm.

        Returns:
            Extra records for the current step, including ``lam``, ``mu``, ``n_iter_shake``, and ``err_shake``.
        """
        posi_mwc_new, posi_new, mome_mwc_half, lam, n_iter, err = self._shake_step(
            self._posi_mwc, self._mome_mwc, self.force, self._G_pinv_mwc)

        # potential energy and force at new positions
        self.posi = np.asarray(posi_new).reshape(-1, 3)
        self.potential_energy, self.force = self.calculator.energy_forces(self.posi)
        mome_mwc_new, mome_new, mu, G_pinv_mwc_new = self._rattle_step(posi_mwc_new, mome_mwc_half, self.force)

        self._posi_mwc, self._mome_mwc, self._G_pinv_mwc = posi_mwc_new, mome_mwc_new, G_pinv_mwc_new
        self.mome = np.asarray(mome_new).reshape(-1, 3)

        # lambda belong to the previous step
        self.records["lam"][-1] = lam
        return dict(lam=np.zeros_like(lam), mu=mu, n_iter_shake=n_iter, err_shake=err)


##################################################################################################
# Implicit Verlet integrator
##################################################################################################

@lru_cache(maxsize=128)
def _build_implicit_verlet(
        metric_func: Callable[[jax.Array], jax.Array],
        potential_func: Callable[[jax.Array], float] | None,
        dt: float,
        normalize: bool,
        max_iter_newton: int,
        tol_newton: float) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Build the jitted functions of ImplicitVerlet integrator."""
    if potential_func is None:
        potential_func = lambda q: 0.0

    def hamiltonian(posi: jax.Array, mome: jax.Array) -> jax.Array:
        return 0.5 * mome @ (metric_func(posi) @ mome) + potential_func(posi)

    dHdq = jax.jit(jax.grad(hamiltonian, argnums=0))

    def newton(residual_func: Callable[[jax.Array], jax.Array],
               x_init: jax.Array) -> tuple[jax.Array, int, float]:
        """Solve residual_func(x) = 0 using Newton's method."""
        jacobian_func = jax.jacrev(residual_func)

        def cond(carry):
            i, _, f = carry
            return (i < max_iter_newton) & (f > tol_newton)

        def body_func(carry):
            i, x, _ = carry
            J = jacobian_func(x)
            delta_x = jnp.linalg.solve(J, -residual_func(x))
            return i + 1, x + delta_x, jnp.max(jnp.abs(delta_x))

        n_iter, x_new, err = jax.lax.while_loop(cond, body_func, (0, x_init, jnp.inf))
        return x_new, n_iter, err

    def step_func(posi: jax.Array, mome: jax.Array) -> tuple[jax.Array, ...]:
        """Perform one step of the implicit Verlet integrator."""
        # P (n -> n+1/2)  (Eq. (J4))
        mome_half, n_iter_mome, err_mome = newton(lambda x: x - mome + 0.5 * dt * dHdq(posi, x), mome)

        # Q (n -> n+1)  (Eq. (J5))
        G = metric_func(posi)
        posi_new, n_iter_posi, err_posi = newton(lambda x: x - posi - 0.5 * dt * (G + metric_func(x)) @ mome_half, posi)

        if normalize:
            posi_new /= jnp.linalg.norm(posi_new)

        # P (n+1/2 -> n+1)  (Eq. (J6))
        mome_new = mome_half - 0.5 * dt * dHdq(posi_new, mome_half)
        if normalize:
            mome_new -= (mome_new @ posi_new) * posi_new  # project to the tangent space

        return posi_new, mome_new, mome_half, jnp.array([n_iter_mome, n_iter_posi]), jnp.array([err_mome, err_posi])

    @partial(jax.jit, static_argnums=2)
    def run_scan(posi: jax.Array, mome: jax.Array, n_step: int) -> tuple[jax.Array, ...]:
        """Run the implicit Verlet integrator for n_step steps."""
        def body_func(carry, _):
            posi, mome = carry
            posi_new, mome_new, mome_half, n_iter, err = step_func(posi, mome)
            return (posi_new, mome_new), (posi_new, mome_new, mome_half, n_iter, err)

        _, traj = jax.lax.scan(body_func, (posi, mome), None, length=n_step)
        return traj  # posi, mome, mome_half, n_iter, err (n_step, ...)

    @jax.jit
    def _energies(posi: jax.Array, mome: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute the potential and kinetic energies."""
        pot = jax.vmap(potential_func)(posi)
        kin = jax.vmap(lambda p, q: 0.5 * p @ (metric_func(q) @ p))(mome, posi)
        return pot, pot + kin

    return run_scan, _energies

class ImplicitVerlet:
    r"""Implicit Verlet integrator for :math:`H(Q, P) = \frac{1}{2} P^T G(Q) P + V(Q)`.

    One step of the integrator (Eqs. (J4)-(J6) in the paper) is given by:

    .. math::

        P_{n+\frac{1}{2}} &= P_n - \frac{\Delta t}{2} \frac{\partial H}{\partial Q}(Q_n, P_{n+\frac{1}{2}})

        Q_{n+1} &= Q_n + \frac{\Delta t}{2} (G(Q_n) + G(Q_{n+1})) P_{n+\frac{1}{2}}

        P_{n+1} &= P_{n+\frac{1}{2}} - \frac{\Delta t}{2} \frac{\partial H}{\partial Q}(Q_{n+1}, P_{n+\frac{1}{2}})

    where the implicit equations are solved by Newton's method.

    The trajectory returned by :meth:`run` and :meth:`get_trajectory` is a dictionary with the following keys:

    - ``posi``: ``(T, m)`` positions.
    - ``mome``: ``(T, m)`` momenta.
    - ``mome_half``: ``(T, m)`` momenta at half step.
    - ``potential_energy``: ``(T,)`` potential energies.
    - ``total_energy``: ``(T,)`` total energies.
    - ``n_iter_mome``: ``(T,)`` number of iterations for the implicit equation of momenta.
    - ``n_iter_posi``: ``(T,)`` number of iterations for the implicit equation of positions.
    - ``err_mome``: ``(T,)`` final error of the implicit equation of momenta.
    - ``err_posi``: ``(T,)`` final error of the implicit equation of positions.
    - ``dt``: Time step.

    Args:
        posi: ``(m,)`` initial positions.
        mome: ``(m,)`` initial momenta.
        dt: Time step.
        metric_func: Function to compute the metric :math:`G(Q)`, ``(m,)`` -> ``(m, m)``.
        potential_func: Function to compute the potential :math:`V(Q)`, ``(m,)`` -> scalar.
            If ``None``, the potential is assumed to be zero.
        normalize: If ``True``, the positions are normalized to unit norm and the momenta are the projected onto
            the tangent space at every step. (``q^T q = 1, p^T q = 0``). This is intended for the quaternion dynamics.
        max_iter_newton: Maximum number of iterations for Newton's method.
        tol_newton: Tolerance for Newton's method convergence.
        i_step: Initial step index.

    Note:
        The jitted functions are cached by
        ``(metric_func, potential_func, dt, normalize, max_iter_newton, tol_newton)``.
        Passing the same function objects reuses the compiled functions.

    Warnings:
        The integrator is vulnerable to the singularity of the metric :math:`G(Q)`.
        Check the condition number of :math:`G(Q)` before use.
    """

    def __init__(self,
            posi: ArrayLike,
            mome: ArrayLike,
            dt: float,
            metric_func: Callable[[jax.Array], jax.Array],
            *,
            potential_func: Callable[[jax.Array], float] | None = None,
            normalize: bool = False,
            max_iter_newton: int = 50,
            tol_newton: float = 1e-12,
            i_step: int = 0) -> None:

        self.posi = np.asarray(posi, dtype=float).reshape(-1)
        self.mome = np.asarray(mome, dtype=float).reshape(-1)
        self.n_dim = self.posi.shape[0]
        self.i_step = i_step
        self.dt = float(dt)
        self.records = {}

        dt = self.dt

        if normalize:
            self.posi /= np.linalg.norm(self.posi)
            self.mome -= (self.mome @ self.posi) * self.posi

        self._run_scan, self._energies = _build_implicit_verlet(metric_func, potential_func, dt,
                                                                normalize, max_iter_newton, tol_newton)

        self._append(self.posi[None], self.mome[None], np.zeros((1, self.n_dim), dtype=float),
                     np.zeros((1, 2), dtype=int), np.zeros((1, 2), dtype=float))

    def _append(self, posi: ArrayLike, mome: ArrayLike,
                mome_half: ArrayLike, n_iter: ArrayLike, err: ArrayLike) -> None:
        """Append the current state to the records."""
        pot, tot = self._energies(posi, mome)
        rec = dict(posi=posi, mome=mome, mome_half=mome_half, potential_energy=pot, total_energy=tot,
                   n_iter_mome=n_iter[:, 0], n_iter_posi=n_iter[:, 1], err_mome=err[:, 0], err_posi=err[:, 1])

        for key, val in rec.items():
            self.records.setdefault(key, []).append(np.asarray(val))

    def step(self) -> None:
        """Advance the state by one step using the implicit Verlet algorithm."""
        return self.run(1, return_trajectory=False)

    def run(self, n_step: int, return_trajectory: bool = True) -> dict[str, np.ndarray] | None:
        """Run the implicit Verlet integrator for ``n_step`` steps.

        Args:
            n_step: Number of steps to run.
            return_trajectory: If ``True``, return the trajectory from the initial step.

        Returns:
            Trajectory from the initial step if ``return_trajectory`` is ``True``, otherwise ``None``.
        """
        posi, mome, mome_half, n_iter, err = self._run_scan(jnp.asarray(self.posi), jnp.asarray(self.mome), n_step)

        # mome_half should belong to the previous step
        self.records["mome_half"][-1][-1] = np.asarray(mome_half[0])
        mome_half = np.concatenate([np.asarray(mome_half[1:]), np.zeros((1, self.n_dim), dtype=float)])

        self._append(np.asarray(posi), np.asarray(mome), mome_half, np.asarray(n_iter), np.asarray(err))
        self.posi = np.asarray(posi[-1])
        self.mome = np.asarray(mome[-1])
        self.i_step += n_step
        if return_trajectory:
            return self.get_trajectory()

    def get_total_energy(self) -> float:
        """Get the total energy of the last step."""
        return float(self.records["total_energy"][-1][-1])

    def get_trajectory(self) -> dict[str, np.ndarray]:
        """Get the trajectory as a dictionary of arrays."""
        traj = {key: np.concatenate(val) for key, val in self.records.items()}
        traj["dt"] = self.dt
        return traj
