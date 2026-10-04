from abc import ABC, abstractmethod
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


##################################################################################################
# base class
##################################################################################################

class Integrator(ABC):
    """Base class for integrators.

    calculator : energy/force calculator
    posi       : (N, 3) initial positions
    mome       : (N, 3) initial momenta
    mass       : (N,) atomic masses
    dt         : time step
    i_step     : initial step index (default: 0)

    traj keys:
        posi    : (T, N, 3) trajectory positions
        mome    : (T, N, 3) trajectory momenta
        potential_energy : (T,) potential energy
        force   : (T, N, 3) forces
        total_energy : (T,) total energy
    """

    def __init__(self, calculator, posi, mome, mass, dt, *, i_step=0):
        self.calculator = calculator
        self.posi = np.asarray(posi, dtype=float).reshape(-1, 3)
        self.mome = np.asarray(mome, dtype=float).reshape(-1, 3)
        self.mass = np.asarray(mass, dtype=float)

        self.i_step = i_step
        self.dt = float(dt)
        self.potential_energy, self.force = self.calculator.energy_forces(self.posi)
        self.records = {}

    @abstractmethod
    def step(self):
        pass

    def get_kinetic_energy(self):
        return 0.5 * np.sum(self.mome**2 / self.mass[:, None])

    def get_total_energy(self):
        return self.get_kinetic_energy() + self.potential_energy

    def _initial_extra(self):
        """Extra records for the initial step."""
        return {}

    def _record(self, extra):
        """Record the current state."""
        rec = dict(posi=self.posi, mome=self.mome, potential_energy=self.potential_energy,
                   force=self.force, total_energy=self.get_total_energy(), **extra)
        for key, val in rec.items():
            self.records.setdefault(key, []).append(val)

    def run(self, n_step):
        """Run the integrator for n_step steps."""
        if not self.records:
            self._record(self._initial_extra())
        for _ in range(n_step):
            extra = self.step()
            self._record(extra)
            self.i_step += 1
        return self.get_trajectory()

    def get_trajectory(self):
        """Get the trajectory as a dictionary of arrays."""
        traj = {key: np.stack([np.asarray(v) for v in val]) for key, val in self.records.items()}
        traj["dt"] = self.dt
        traj["mass"] = self.mass
        return traj




##################################################################################################
# SHAKE/RATTLE integrator
##################################################################################################

def _pinv(G, rcond):
    """Compute the pseudo-inverse of G using SVD."""
    U, S, Vt = jnp.linalg.svd(G, full_matrices=False)
    S_inv = jnp.where(S > rcond * S[0], 1.0 / S, 0.0)
    return (Vt.T * S_inv[None, :]) @ U.T  # (3N, K)

def _build_shake_rattle(coord_const, dt, mass, max_iter_shake, tol_shake, rcond):
    """Build the jitted functions of SHAKE/RATTLE integrator."""
    sqrt_mass_flat = jnp.sqrt(np.repeat(mass, 3))
    const_func = coord_const.get_const
    const_grad_func = coord_const.get_const_grad

    def get_G_mwc(posi_mwc):
        return const_grad_func(posi_mwc / sqrt_mass_flat) / sqrt_mass_flat[None, :]

    @jax.jit
    def initialize(posi_mwc, mome_mwc):
        """Initialize the SHAKE/RATTLE integrator."""
        G_mwc = get_G_mwc(posi_mwc)
        G_pinv_mwc = _pinv(G_mwc, rcond)
        mome_mwc -= G_pinv_mwc @ (G_mwc @ mome_mwc)
        return G_mwc, G_pinv_mwc, mome_mwc

    @jax.jit
    def shake_step(posi_mwc, mome_mwc, force, G_pinv_mwc):
        """Perform a SHAKE step."""

        force_mwc = force.reshape(-1) / sqrt_mass_flat
        mome_mwc_half = mome_mwc + 0.5 * dt * force_mwc
        posi_mwc_pred = posi_mwc + dt * mome_mwc_half

        def cond(carry):
            i, _, f = carry
            return (i < max_iter_shake) & (jnp.max(jnp.abs(f)) > tol_shake)

        def body_func(carry):
            i, x, f = carry   # i: iteration count, x: current position, f: constraint value
            x -= G_pinv_mwc @ f
            return i + 1, x, const_func(x / sqrt_mass_flat)

        f_init = const_func(posi_mwc_pred / sqrt_mass_flat)
        n_iter, posi_mwc_new, f_final = jax.lax.while_loop(cond, body_func, (0, posi_mwc_pred, f_init))

        delta_posi_mwc = posi_mwc_new - posi_mwc_pred
        lam = G_pinv_mwc.T @ delta_posi_mwc / dt
        mome_mwc_half += delta_posi_mwc / dt

        return posi_mwc_new, posi_mwc_new / sqrt_mass_flat, mome_mwc_half, \
                lam, n_iter, jnp.max(jnp.abs(f_final))

    @jax.jit
    def rattle_step(posi_mwc_new, mome_mwc_half, force_new):
        """Perform a RATTLE step."""

        force_mwc_new = force_new.reshape(-1) / sqrt_mass_flat
        G_mwc = get_G_mwc(posi_mwc_new)
        G_pinv_mwc = _pinv(G_mwc, rcond)
        mome_mwc_pred = mome_mwc_half + 0.5 * dt * force_mwc_new
        mu = -G_pinv_mwc.T @ mome_mwc_pred
        mome_mwc_new = mome_mwc_pred + G_mwc.T @ mu

        return mome_mwc_new, mome_mwc_new * sqrt_mass_flat, mu, G_pinv_mwc

    return shake_step, rattle_step, initialize, sqrt_mass_flat


class ShakeRattle(Integrator):
    """SHAKE/RATTLE integrator with holonomic constraints F(q) = 0.

    calculator : energy/force calculator
    posi       : (N, 3) initial positions
    mome       : (N, 3) initial momenta
    mass       : (N,) atomic masses
    dt         : time step
    coord_const : InternalCoordinates for constraints

    max_iter_shake : maximum number of iterations for SHAKE
    tol_shake      : tolerance for SHAKE convergence
    rcond          : cutoff for pseudo-inverse computation
    tol_init       : tolerance for initial constraint satisfaction
    i_step         : initial step index (default: 0)
    """
    def __init__(self, calculator, posi, mome, mass, dt, coord_const, *,
                 max_iter_shake=50, tol_shake=1e-10, rcond=1e-10, tol_init=1e-8, i_step=0):

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

    def _initial_extra(self):
        zeros = np.zeros((self.n_const))
        return dict(lam=zeros, mu=zeros, n_iter_shake=0, err_shake=0.0)

    def step(self):
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

@lru_cache(maxsize=128)
def _build_implicit_verlet(metric_func, potential_func, dt, normalize, max_iter_newton, tol_newton):
    """Build the jitted functions of ImplicitVerlet integrator."""
    if potential_func is None:
        potential_func = lambda q: 0.0

    def hamiltonian(posi, mome):
        return 0.5 * mome @ (metric_func(posi) @ mome) + potential_func(posi)

    dHdq = jax.jit(jax.grad(hamiltonian, argnums=0))

    def newton(residual_func, x_init):
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

    def step_func(posi, mome):
        """Perform one step of the implicit Verlet integrator."""

        # P (n -> n+1/2)
        mome_half, n_iter_mome, err_mome = newton(lambda x: x - mome + 0.5 * dt * dHdq(posi, x), mome)

        # Q (n -> n+1)
        G = metric_func(posi)
        posi_new, n_iter_posi, err_posi = newton(lambda x: x - posi - 0.5 * dt * (G + metric_func(x)) @ mome_half, posi)

        if normalize:
            posi_new /= jnp.linalg.norm(posi_new)

        # P (n+1/2 -> n+1)
        mome_new = mome_half - 0.5 * dt * dHdq(posi_new, mome_half)
        if normalize:
            mome_new -= (mome_new @ posi_new) * posi_new  # project to the tangent space

        return posi_new, mome_new, mome_half, jnp.array([n_iter_mome, n_iter_posi]), jnp.array([err_mome, err_posi])

    @partial(jax.jit, static_argnums=2)
    def run_scan(posi, mome, n_step):
        """Run the implicit Verlet integrator for n_step steps."""
        def body_func(carry, _):
            posi, mome = carry
            posi_new, mome_new, mome_half, n_iter, err = step_func(posi, mome)
            return (posi_new, mome_new), (posi_new, mome_new, mome_half, n_iter, err)

        _, traj = jax.lax.scan(body_func, (posi, mome), None, length=n_step)
        return traj  # posi, mome, mome_half, n_iter, err (n_step, ...)

    @jax.jit
    def _energies(posi, mome):
        """Compute the potential and kinetic energies."""
        pot = jax.vmap(potential_func)(posi)
        kin = jax.vmap(lambda p, q: 0.5 * p @ (metric_func(q) @ p))(mome, posi)
        return pot, pot + kin

    return run_scan, _energies

class ImplicitVerlet:
    """Implicit Verlet integrator for H(Q, P) = 0.5 P^T G(Q) P + V(Q).

        P_{n+1/2} = P_n - dt/2 dH/dq(Q_n, P_{n+1/2})
        Q_{n+1} = Q_n + dt/2 (G(Q_n) + G(Q_{n+1})) P_{n+1/2}
        P_{n+1} = P_{n+1/2} - dt/2 dH/dq(Q_{n+1}, P_{n+1/2})

    Note that this propagator is vulnerable to the singularity of the metric G(Q).
    You should check the condition number of G(Q) before using this integrator.

    posi       : (m,) initial positions
    mome       : (m,) initial momenta
    dt         : time step
    metric_func : function to compute the metric G(Q)

    potential_func : function to compute the potential V(Q). If None, the potential is assumed to be zero.
    normalize      : whether to normalize the position to unit norm.
                     This is intended to be used for the quaternion dynamics. (q^T q = 1, p^T q = 0)
    max_iter_newton : maximum number of iterations for Newton's method
    tol_newton      : tolerance for Newton's method convergence
    i_step          : initial step index (default: 0)
    """

    def __init__(self, posi, mome, dt, metric_func, *,
                 potential_func=None, normalize=False,
                 max_iter_newton=50, tol_newton=1e-12, i_step=0):

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

    def _append(self, posi, mome, mome_half, n_iter, err):
        """Append the current state to the records."""
        pot, tot = self._energies(posi, mome)
        rec = dict(posi=posi, mome=mome, mome_half=mome_half, potential_energy=pot, total_energy=tot,
                   n_iter_mome=n_iter[:, 0], n_iter_posi=n_iter[:, 1], err_mome=err[:, 0], err_posi=err[:, 1])

        for key, val in rec.items():
            self.records.setdefault(key, []).append(np.asarray(val))

    def step(self):
        return self.run(1, return_trajectory=False)

    def run(self, n_step, return_trajectory=True):
        """Run the implicit Verlet integrator for n_step steps."""
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

    def get_total_energy(self):
        """Get the total energy of the last step."""
        return float(self.records["total_energy"][-1][-1])

    def get_trajectory(self):
        """Get the trajectory as a dictionary of arrays."""
        traj = {key: np.concatenate(val) for key, val in self.records.items()}
        traj["dt"] = self.dt
        return traj

def quaternion_metric(inertia):
    """Metric for quaternion dynamics.

    inertia : (3,) principal moments of inertia

    returns:
        metric_func : (4,) quaternion -> (4, 4) metric matrix
    """
    inertia = jnp.asarray(inertia, dtype=float)

    def metric_func(q):
        q0, q1, q2, q3 = q
        S = jnp.array([
            [-q1, q0, q3, -q2],
            [-q2, -q3, q0, q1],
            [-q3, q2, -q1, q0],
        ])
        return 0.25 * S.T @ (S / inertia[:, None])  # (4, 4)

    return metric_func
