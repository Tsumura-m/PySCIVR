"""PySCIVR: semiclassical initial value representation in internal coordinates.

Modules:
    - :mod:`~pyscivr.coordinates`: internal coordinates and coordinate conversions.
    - :mod:`~pyscivr.integrators`: integrators for classical trajectories.
    - :mod:`~pyscivr.hessian`: Hessians in the active internal coordinates.
    - :mod:`~pyscivr.variations`: propagation of the monodromy matrices.
    - :mod:`~pyscivr.basis`: coherent state overlaps for various topologies.
    - :mod:`~pyscivr.scivr`: semiclassical prefactors, actions and spectra.
    - :mod:`~pyscivr.potentials`: potential energy calculators.
    - :mod:`~pyscivr.save`: saving and loading of trajectories and Hessians.
    - :mod:`~pyscivr.units`: physical constants and unit conversion.
    - :mod:`~pyscivr.utils`: utilities for molecular geometries.
"""

from .coordinates import InternalCoordinates, QuaternionCoordinates
from .hessian import eval_hess_active, interpolate_hessian, projected_hessian
from .integrators import ImplicitVerlet, ShakeRattle
from .scivr import get_action, get_lagrangian, get_prefactor, get_ta_spectrum_fft
from .utils import get_geom, get_inertia, get_mass_center
from .variations import (
    make_jacobi_func,
    make_vari_propagator,
    make_vari_propagator_implicit,
    propagate_variation,
    propagate_variation_implicit,
    to_internal,
)
