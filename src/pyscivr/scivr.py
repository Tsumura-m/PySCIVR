import numpy as np

from .units import HBar


def _simpson_weights(n_points, dt):
    """Quadrature weights of the running Simpson rule for the integral over all n_point points.

    at 2n (n = 1,2,3,..., (n_points-1)/2)
    w(2n-2) += 1/3 * dt, w(2n-1) += 4/3 * dt, w(2n) += 1/3 * dt (simpson's rule)
    at n_points-1 (if n_points-1 is odd):
    w(n_points-2) += 1/2 * dt, w(n_points-1) += 1/2 * dt (trapezoidal rule)

    n_points : number of points
    dt       : time step
    """

    weights = np.zeros(n_points)
    if n_points == 1:
        return weights
    n_simp = n_points if (n_points - 1) % 2 == 0 else n_points - 1  # points overed by Simpson's rule
    if n_simp >= 3:
        w = np.ones(n_simp)
        w[1:-1:2] = 4
        w[2:-1:2] = 2
        weights[:n_simp] = w * dt / 3
    if n_simp < n_points:
        weights[n_simp - 1] += 0.5 * dt
        weights[n_simp] += 0.5 * dt
    return weights

def get_action(lagrangian, dt):
    """Compute the action integral S = int dt L.

    Even steps : Simpson's rule, Odd steps : Simpson's rule + trapezoidal rule
    """

    lag = np.asarray(lagrangian, dtype=float)
    action = np.zeros_like(lag)
    blocks = (lag[0:-2:2] + 4 * lag[1:-1:2] + lag[2::2]) * dt / 3  # Δt/3 * (L_n + 4 L_{n+1} + L_{n+2})
    # Even steps: S_{2n+2} = S_{2n} + Δt/3 * (L_{2n} + 4 L_{2n+1} + L_{n+2})
    action[2::2] = np.cumsum(blocks)
    n_odd = len(action[1::2])
    # Odd steps: S_{2n+1} = S_{2n} + Δt/2 * (L_{2n} + L_{2n+1})
    action[1::2] = action[0::2][:n_odd] + (lag[0::2][:n_odd] + lag[1::2]) * dt * 0.5
    return action

def get_lagrangian(ener_tot, ener_pot, ener_ref=0.0):
    """Compute the Lagrangian L = T - V = E_tot - 2V + V_ref.

    ener_tot : (T,) total energy of the trajectory
    ener_pot : (T,) potential energy of the trajectory
    ener_ref : reference potential energy for the Lagrangian
    """

    ener_tot = np.asarray(ener_tot, dtype=float)
    ener_pot = np.asarray(ener_pot, dtype=float)
    return ener_tot - 2 * ener_pot + ener_ref

def get_prefactor(mono, gamma, valid=None):
    """Herman-Kluk pre-factor C_t with branch tracking of the complex square root.

    mono  : (T, 2m, 2m) monodromy matrix along the trajectory
    gamma : (m,) width parameter of the coherent state basis
    valid : (T,) bool. Invalid steps (near singularity) are skipped in the branch tracking and returned as NaN
    """

    mono = np.asarray(mono, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    n_step = mono.shape[0]
    m = gamma.shape[0]
    if valid is None:
        valid = np.ones(n_step, dtype=bool)

    qq, qp = mono[:, :m, :m], mono[:, :m, m:]
    pq, pp = mono[:, m:, :m], mono[:, m:, m:]
    #g, g_inv = np.diag(gamma), np.diag(1.0 / gamma)
    #mat = qq + g_inv @ pp @ g - 1j * HBar * qp @ g + 1j / HBar * g_inv @ pq
    g, g_inv = gamma, 1.0 / gamma
    mat = qq + pp * g_inv[None, :, None] * g[None, None, :] - 1j * HBar * qp * g[None, None, :] \
          + 1j / HBar * pq * g_inv[None, :, None]
    det = np.linalg.det(mat) * 0.5 ** m

    prefactor = np.full(n_step, np.nan, dtype=complex)
    d = det[valid]
    phase = 0.5 * np.unwrap(np.angle(d))
    prefactor [valid] = np.sqrt(np.abs(d)) * np.exp(1j * phase)
    return prefactor

def get_ta_spectrum_naive(prefactor, overlap, action, dt, ener_grid, valid=None, chunk_size=1000):
    """TA-SC-IVR spectrum of a single trajectory.
    O(n_E n_step) where n_E is the number of energy grid points and n_step is the number of time steps.
    A(E) = |int_0^T dt <Psi(0)|Q_t, P_t> C_t exp(i (S_t + E t + phi_t) / hbar)|^2/T

    prefactor : (T,) Herman-Kluk pre-factor C_t
    overlap   : (T,) overlap <Psi(0)| q_t, p_t>
    action    : (T,) action integral S_t
    dt        : time step
    ener_grid : (n_E,) energy grid [eV] for the spectrum
    valid     : (T,) bool. Invalid steps (near singularity) are skipped in the amplitude and contribute nothing.
    chunk_size: number of points to compute the spectrumat once to save memory

    returrns:
        spec : (n_E,) spectrum |A(E)|^2 / T
    """

    prefactor = np.asarray(prefactor, dtype=complex)
    overlap = np.asarray(overlap, dtype=complex)
    action = np.asarray(action, dtype=float)
    ener_grid = np.asarray(ener_grid, dtype=float)
    n_step = prefactor.shape[0]

    if valid is None:
        valid = np.ones(n_step, dtype=bool)

    t = dt * np.arange(n_step)
    with np.errstate(invalid="ignore"):
        val = np.exp(1j * np.angle(prefactor) + 1j * action / HBar) * overlap
    val = np.where(valid, val, 0.0) * _simpson_weights(n_step, dt)

    amp = np.empty((len(ener_grid),), dtype=complex)
    for s in range(0, len(ener_grid), chunk_size):
        e = ener_grid[s:s + chunk_size]
        amp[s:s + chunk_size] = np.exp(1j * np.outer(e, t) / HBar) @ val      # sum_j val_j exp(+2 pi i k j / n_fft)

    return np.abs(amp) ** 2 / ((n_step-1) * dt)

def get_ta_spectrum_fft(prefactor, overlap, action, dt, n_fft, valid=None):
    """TA-SC-IVR spectrum of a single trajectory by zero-padded FFT. O(n_fft log n_fft)

    prefactor : (T,) Herman-Kluk pre-factor C_t
    overlap   : (T,) overlap <Psi(0)| q_t, p_t>
    action    : (T,) action integral S_t
    dt        : time step
    n_fft     : number of points for the FFT. Should be a power of 2
    valid     : (T,) bool. Invalid steps (near singularity) are skipped in the amplitude and contribute nothing.

    returns:
        ener_grid : (n_fft,) energy grid [eV] on the grid E_k = 2 pi hbar k / (n_fft dt)
        spec      : (n_fft,) spectrum |A(E)|^2 / T
    """
    prefactor = np.asarray(prefactor, dtype=complex)
    n_step = prefactor.shape[0]
    if valid is None:
        valid = np.ones(n_step, dtype=bool)
    with np.errstate(invalid="ignore"):
        val = np.exp(1j * np.angle(prefactor) + 1j * np.asarray(action) / HBar) * overlap
    val = np.where(valid, val, 0.0) * _simpson_weights(n_step, dt)

    amp = np.fft.ifft(val, n=n_fft) * n_fft                 # sum_j val_j exp(+2 pi i k j / n_fft)
    ener_grid = 2.0 * np.pi * HBar * np.arange(n_fft) / (n_fft * dt)
    return ener_grid, np.abs(amp) ** 2 / (n_step * dt)
