"""Semiclassical quantities and TA-SC-IVR spectra along classical trajectories.

See section II A for details.

Notation:
    - ``T``: number of time steps of the trajectory.
    - ``m``: number of active coordinates.
    - ``n_E``: number of energy grid points.
"""

from __future__ import annotations

from warnings import deprecated

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .units import HBar


def _simpson_weights(n_points: int, dt: float) -> NDArray[np.float64]:
    r"""Quadrature weights of Simpson's rule for the integral over all ``n_points`` points.

    Simpson's rule is applied to the first odd number of points, and the
    trapezoidal rule to the last interval if ``n_points-1`` is odd.

    - at ``2n (n = 0,1,2,..., (n_points-3)/2)``:
      ``w(2n) += 1/3 * dt, w(2n+1) += 4/3 * dt, w(2n+2) += 1/3 * dt`` (Simpson's rule)
    - at ``n_points-1`` (if ``n_points-1`` is odd):
      ``w(n_points-2) += 1/2 * dt, w(n_points-1) += 1/2 * dt`` (trapezoidal rule)

    Args:
        n_points: Number of points to integrate over. ``(0, 1, 2, ..., n_points-1)``.
        dt: Time step.

    Returns:
        ``(n_points,)`` quadrature weights.
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


def get_action(lagrangian: ArrayLike, dt: float) -> NDArray[np.float64]:
    r"""Compute the running action integral (Eq. (3) in the paper).

    .. math::

        S_t = \int_0^t L(Q_\tau, P_\tau) d\tau

    Args:
        lagrangian: ``(T,)`` Lagrangian along the trajectory.
        dt: Time step.

    Returns:
        ``(T,)`` running action integral along the trajectory, with ``S_0 = 0``.

    Note:
        At even steps, the integral is computed by Simpson's rule,
        At odd steps, the last interval is added by the trapezoidal rule to
        the value at the previous even step.
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


def get_lagrangian(ener_tot: ArrayLike, ener_pot: ArrayLike, *, ener_ref: float = 0.0) -> NDArray[np.float64]:
    r"""Compute the Lagrangian from the total and potential energies.

    .. math::
        L = K - (V - V_{\text{ref}}) = E_{\text{tot}} - 2 V + V_{\text{ref}}

    where :math:`K` is the kinetic energy, :math:`V` is the potential energy.

    Args:
        ener_tot: ``(T,)`` total energy of the trajectory.
        ener_pot: ``(T,)`` potential energy of the trajectory.
        ener_ref: Reference potential energy for the Lagrangian.

    Returns:
        ``(T,)`` Lagrangian along the trajectory.
    """
    ener_tot = np.asarray(ener_tot, dtype=float)
    ener_pot = np.asarray(ener_pot, dtype=float)
    return ener_tot - 2 * ener_pot + ener_ref


def get_prefactor(
    mono: ArrayLike, gamma: ArrayLike, valid: ArrayLike | None = None
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    r"""Compute the Herman-Kluk pre-factor with branch tracking of the complex square root (Eq. (6) in the paper).

    .. math::

        C_t = \sqrt{\det \left(\frac{1}{2} \left( M_{qq} + \Gamma^{-1} M_{pp} \Gamma
        - i \hbar M_{qp} \Gamma + \frac{i}{\hbar} \Gamma^{-1} M_{pq} \right)\right)}

    where :math:`M` is the monodromy matrix and :math:`\Gamma = \text{diag}(\gamma)`
    is the width parameter of the coherent state basis.
    The branch of the square root is chosen so that the phase of :math:`C_t` is continuous along the trajectory.

    Args:
        mono: ``(T, 2m, 2m)`` monodromy matrices along the trajectory.
        gamma: ``(m,)`` width parameters of the coherent state basis.
        valid: ``(T,)`` bool mask. Invalid steps (e.g. near singularity) are
            skipped in the branch tracking. If ``None``, all steps are considered valid.

    Returns:
        ``(T,)`` Herman-Kluk pre-factor :math:`C_t` along the trajectory, with ``NaN`` at invalid steps.
    """
    mono = np.asarray(mono, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    n_step = mono.shape[0]
    m = gamma.shape[0]
    if valid is None:
        valid = np.ones(n_step, dtype=bool)

    qq, qp = mono[:, :m, :m], mono[:, :m, m:]
    pq, pp = mono[:, m:, :m], mono[:, m:, m:]
    g, g_inv = gamma, 1.0 / gamma
    mat = (
        qq
        + pp * g_inv[None, :, None] * g[None, None, :]
        - 1j * HBar * qp * g[None, None, :]
        + 1j / HBar * pq * g_inv[None, :, None]
    )
    det = np.linalg.det(mat) * 0.5**m

    prefactor = np.full(n_step, np.nan, dtype=complex)
    d = det[valid]
    phase = 0.5 * np.unwrap(np.angle(d))
    prefactor[valid] = np.sqrt(np.abs(d)) * np.exp(1j * phase)
    return prefactor


@deprecated("get_ta_spectrum_naive is deprecated. Use get_ta_spectrum_fft instead.")
def get_ta_spectrum_naive(
    prefactor: ArrayLike,
    overlap: ArrayLike,
    action: ArrayLike,
    dt: float,
    ener_grid: ArrayLike,
    valid: ArrayLike | None = None,
    chunk_size: int = 1000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Compute the TA-SC-IVR spectrum of a single trajectory by direct summation.

    .. deprecated:: 0.1.0
        Use :func:`get_ta_spectrum_fft` instead.

    Computes the same spectrum as :func:`get_ta_spectrum_fft` on an arbitrary energy grid,
    at the cost of :math:`O(n_E T)`.

    Args:
        prefactor: ``(T,)`` Herman-Kluk pre-factor :math:`C_t`.
        overlap: ``(T,)`` overlap :math:`\langle \Psi(0) | Q_t, P_t \rangle`.
        action: ``(T,)`` action integral :math:`S_t`.
        dt: Time step.
        ener_grid: ``(n_E,)`` energy grid [eV] for the spectrum
        valid: ``(T,)`` bool. Invalid steps (near singularity) are skipped in
            the amplitude and contribute nothing. If ``None``, all steps are considered valid.
        chunk_size: Number of energy points to compute at once to save memory.

    Returns:
        ``(ener_grid, spectrum)``, where

        - ``ener_grid``: ``(n_E,)`` energy grid [eV] for the spectrum.
        - ``spectrum``: ``(n_E,)`` spectrum, normalized by the total time `t_tot = T dt`.
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
        e = ener_grid[s : s + chunk_size]
        amp[s : s + chunk_size] = np.exp(1j * np.outer(e, t) / HBar) @ val  # sum_j val_j exp(+2 pi i k j / n_fft)

    return ener_grid, np.abs(amp) ** 2 / (n_step * dt)


def get_ta_spectrum_fft(
    prefactor: ArrayLike, overlap: ArrayLike, action: ArrayLike, dt: float, n_fft: int, valid: ArrayLike | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Compute the TA-SC-IVR spectrum of a single trajectory by zero-padded FFT (Eq. (12) in the paper).

    .. math::

        I(E) = \frac{1}{t_{\text{tot}}} \left| \int_0^{t_{\text{tot}}} dt
        \langle \Psi(0) | Q_t, P_t \rangle e^{i \left(\arg C_t + \frac{ S_t + E t}{\hbar}\right)}  \right|^2

    The integral is evaluated by Simpson's rule (see :func:`get_action` for details),
    at the cost of :math:`O(n_{\text{fft}} \log n_{\text{fft}})`.

    Args:
        prefactor: ``(T,)`` Herman-Kluk pre-factor :math:`C_t`.
        overlap: ``(T,)`` overlap :math:`\langle \Psi(0) | Q_t, P_t \rangle`.
        action: ``(T,)`` action integral :math:`S_t`.
        dt: Time step.
        n_fft: Number of points for the FFT. Must be at least ``T``, and should be a power of 2.
        valid: ``(T,)`` bool. Invalid steps (near singularity) are skipped in the
            amplitude and contribute nothing. If ``None``, all steps are considered valid.

    Returns:
        ``(ener_grid, spectrum)``, where

        - ``ener_grid``: ``(n_fft,)`` energy grid [eV] for the spectrum.
          The grid is :math:`E_k = \frac{2 \pi \hbar k}{n_{\text{fft}} dt}`.
        - ``spectrum``: ``(n_fft,)`` spectrum, normalized by the total time ``t_tot = T dt``.

    Raises:
        ValueError: If ``n_fft < T``.

    Note:
        The grid spacing is :math:`\Delta E = \frac{2 \pi \hbar}{n_{\text{fft}} dt}`,
        and the maximum energy is :math:`E_{\text{max}} = \frac{2 \pi \hbar}{dt}`.
        Increasing ``n_fft`` refines the grid by interpolation, but the intrinsic
        resolution :math:`\frac{2 \pi \hbar}{t_{\text{tot}}}` is not improved.
    """
    if n_fft < len(prefactor):
        raise ValueError(f"n_fft ({n_fft}) must be at least the number of steps ({len(prefactor)})")

    prefactor = np.asarray(prefactor, dtype=complex)
    n_step = prefactor.shape[0]
    if valid is None:
        valid = np.ones(n_step, dtype=bool)
    with np.errstate(invalid="ignore"):
        val = np.exp(1j * np.angle(prefactor) + 1j * np.asarray(action) / HBar) * overlap
    val = np.where(valid, val, 0.0) * _simpson_weights(n_step, dt)

    amp = np.fft.ifft(val, n=n_fft) * n_fft  # sum_j val_j exp(+2 pi i k j / n_fft)
    ener_grid = 2.0 * np.pi * HBar * np.arange(n_fft) / (n_fft * dt)
    return ener_grid, np.abs(amp) ** 2 / (n_step * dt)
