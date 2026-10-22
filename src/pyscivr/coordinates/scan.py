"""1D scan."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .core import InternalCoordinates

jax.config.update("jax_enable_x64", True)

def rigid_scan_1d(
        coord_all: InternalCoordinates,
        grid: ArrayLike,
        scan_idx: int,
        *,
        max_iter_newton: int = 20,
        tol_newton: float = 1e-12,
        ds_min: float = 1e-4,
        ) -> NDArray[np.float64]:
    """Generate Cartesian structures by scanning a single internal coordinate while keeping the others fixed.

    Args:
        coord_all: InternalCoordinates for the full system. It must define ``3N`` coordinates.
        grid: 1D ascending array of target values for the scanned coordinate.
        scan_idx: Index of the scanned coordinate in ``coord_all``.
        max_iter_newton: Maximum number of iterations for Newton's method.
        tol_newton: Tolerance for convergence in Newton's method.
        ds_min: Minimum step size for the scan.

    Returns:
        ``(len(grid), N, 3)`` Cartesian coordinates for the scanned structures.
    """
    assert len(coord_all) == len(coord_all.posi_ref) * 3, "coord_all must define 3N coordinates."
    assert 0 <= scan_idx < len(coord_all), "scan_idx must be a valid index in coord_all."

    periodic = np.array([c.topology == "S1" for c in coord_all.coords], dtype=bool)
    values_ref = np.array([c.value_ref for c in coord_all.coords], dtype=float)

    def residual_func(x: jax.Array, target: jax.Array) -> jax.Array:
        d = coord_all.get_value(x) - target
        return jnp.where(periodic, (d + jnp.pi) % (2 * jnp.pi) - jnp.pi, d)

    jacobian_func = coord_all.get_const_grad

    @jax.jit
    def _newton(x_init: jax.Array, target: jax.Array) -> tuple[jax.Array, int, float]:
        """Solve residual_func(x) = target using Newton's method."""

        def cond(carry):
            i, _, f = carry
            return (i < max_iter_newton) & (f > tol_newton)

        def body_func(carry):
            i, x, _ = carry
            J = jacobian_func(x)
            delta_x = jnp.linalg.solve(J, -residual_func(x, target))
            return i + 1, x + delta_x, jnp.max(jnp.abs(delta_x))

        n_iter, x_new, err = jax.lax.while_loop(cond, body_func, (0, x_init, jnp.inf))
        return x_new, n_iter, err

    def _scan(x, sub_grid):
        res = []
        for v in sub_grid:
            values_new = values_ref.copy()
            values_new[scan_idx] = v
            values_prev = np.asarray(coord_all.get_value(x))
            dv = -np.asarray(residual_func(x, values_new))
            s, ds = 0.0, 1.0
            while s < 1.0:
                step = min(ds, 1.0 - s)
                x_new, _, err = _newton(x, values_prev + (s + step) * dv)
                if not (np.isfinite(err) and err <= tol_newton):
                    ds *= 0.5
                    if ds < ds_min:
                        raise RuntimeError(f"Newton's method failed to converge (err={err}).")
                    continue
                x, s, ds = x_new, s + step, ds * 2.0
            res.append(np.asarray(x).reshape(-1, 3))
        return res

    grid = np.asarray(grid, dtype=float)
    assert np.all(np.diff(grid) > 0), "Grid points must be sorted in ascending order."
    x0 = jnp.asarray(coord_all.posi_ref.reshape(-1), dtype=float)
    v0 = values_ref[scan_idx]
    if periodic[scan_idx]:
        v0 = grid[0] + (v0 - grid[0]) % (2 * np.pi)
    k = np.searchsorted(grid, v0)
    left = _scan(x0, grid[:k][::-1])
    right = _scan(x0, grid[k:])
    return np.array(left[::-1] + right)

