"""Coordinate transformations for internal coordinates and quaternion coordinates.

See sections II B 1, II C, III B, III C, and appendix L of the paper for details.

Notation:
    - ``N``: number of atoms.
    - ``K``: number of all coordinates (rows of the coordinate system).
    - ``m``: number of active coordinates. (``m <= K``)
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

jax.config.update("jax_enable_x64", True)

# topologies (section II E 2)
R = "R"  # full line (-inf, inf)
R_PLUS = "R+"  # half line [0, inf): bond length
S1 = "S1"  # circle [x, x+2pi) (x is any real number): dihedral angle, yaw & roll angle
I_MINUS = "I-"  # semi-circle [0, pi]: bond angle, pitch angle


##################################################################################################
# Local internal coodinates (coordinate defined by a small number of atoms)
##################################################################################################


def get_bond(q: ArrayLike) -> jax.Array:
    """Compute the bond length between two atoms.

    Args:
        q: ``(2, 3)`` positions of the two atoms.

    Returns:
        Distance between ``q[0]`` and ``q[1]``.
    """
    return jnp.linalg.norm(q[1] - q[0])


def get_bond_angle(q: ArrayLike) -> jax.Array:
    """Compute the bond angle at ``q[1]``.

    Args:
        q: ``(3, 3)`` positions of the three atoms.

    Returns:
        Angle between ``q[0]-q[1]`` and ``q[2]-q[1]`` in [0, pi].
    """
    u, v = q[0] - q[1], q[2] - q[1]
    x = jnp.dot(u, v)  # |u||v|cos(theta)
    y = jnp.linalg.norm(jnp.cross(u, v))  # |u||v|sin(theta)
    return jnp.arctan2(y, x)  # theta


def get_dihedral_angle(q: ArrayLike) -> jax.Array:
    """Compute the dihedral ``q[0]-q[1]-q[2]-q[3]``.

    Args:
        q: ``(4, 3)`` positions of the four atoms.

    Returns:
        Dihedral angle in [-pi, pi).
    """
    b1, b2, b3 = q[1] - q[0], q[2] - q[1], q[3] - q[2]
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    m1 = jnp.cross(n1, b2 / jnp.linalg.norm(b2))
    x = jnp.dot(n1, n2)  # |n1||n2|cos(phi)
    y = jnp.dot(m1, n2)  # |n1||n2|sin(phi)
    return jnp.arctan2(y, x)  # phi


def get_polar_angle(q: ArrayLike) -> jax.Array:
    """Compute the polar angle of the vector ``q[1]-q[0]`` from the z-axis.

    Args:
        q: ``(2, 3)`` positions of the two atoms.

    Returns:
        Polar angle in [0, pi].
    """
    d = q[1] - q[0]
    return jnp.arctan2(jnp.sqrt(d[0] ** 2 + d[1] ** 2), d[2])  # theta in [0, pi]


def get_azimuthal_angle(q: ArrayLike) -> jax.Array:
    """Compute the azimuthal angle of the vector ``q[1]-q[0]`` from the x-axis in the xy-plane.

    Args:
        q: ``(2, 3)`` positions of the two atoms.

    Returns:
        Azimuthal angle in [-pi, pi).
    """
    d = q[1] - q[0]
    return jnp.arctan2(d[1], d[0])  # phi in [-pi, pi)


# kind -> (function, number of atoms, topology)
LOCAL_KINDS = {
    "bond": (get_bond, 2, R_PLUS),
    "angle": (get_bond_angle, 3, I_MINUS),
    "dihedral": (get_dihedral_angle, 4, S1),
    "polar": (get_polar_angle, 2, I_MINUS),
    "azimuthal": (get_azimuthal_angle, 2, S1),
}


@jax.custom_jvp
def _top_eigvec(K: jax.Array) -> jax.Array:
    """Top eigenvector of a symmetric matrix K."""
    return jnp.linalg.eigh(K)[1][:, -1]


@_top_eigvec.defjvp
def _top_eigvec_jvp(primals: tuple[jax.Array], tangents: tuple[jax.Array]) -> tuple[jax.Array, jax.Array]:
    """Derivative of the top eigenvector of a symmetric matrix K."""
    (K,) = primals
    (dK,) = tangents
    q = _top_eigvec(K)
    lam = q @ K @ q
    r = dK @ q - (q @ dK @ q) * q
    dq = jnp.linalg.solve((lam * jnp.eye(K.shape[0]) - K + jnp.outer(q, q)), r)
    return q, dq


def quat_to_rotmat(q: ArrayLike) -> jax.Array:
    """Convert a quaternion to a rotation matrix (Eq. (L6) in the paper).

    Args:
        q: ``(4,)`` quaternion ``(q0, q1, q2, q3)``.

    Returns:
        ``(3, 3)`` rotation matrix ``C``.
    """
    q0, q1, q2, q3 = q
    C = jnp.array(
        [
            [1.0 - 2.0 * (q2**2 + q3**2), 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
            [2.0 * (q1 * q2 + q0 * q3), 1.0 - 2.0 * (q1**2 + q3**2), 2.0 * (q2 * q3 - q0 * q1)],
            [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), 1.0 - 2.0 * (q1**2 + q2**2)],
        ]
    )
    return C


def get_rotation_matrix(posi_c: ArrayLike, posi_ref_c: ArrayLike, mass: ArrayLike) -> jax.Array:
    r"""Compute the rotation matrix minimizing the mass-weighted RMSD by Horn's method.

    The rotation matrix between ``posi_c`` and ``posi_ref_c`` is computed
    following Horn, J. Opt. Soc. Am. A 4, 629-642 (1987).

    Solves ``min_C (posi_ref_c C^T - posi_c)^T M (posi_ref_c C^T - posi_c)``
    with ``M = diag(mass)``. ``C`` rotates ``posi_ref_c`` to ``posi_c``
    (``posi_ref_c C^T ~ posi_c``).

    Args:
        posi_c: ``(N, 3)`` mass-centered coordinates.
        posi_ref_c: ``(N, 3)`` mass-centered reference coordinates.
        mass: ``(N,)`` atomic masses.

    Returns:
        ``(3, 3)`` rotation matrix ``C``.

    Note:
        :math:`C` is obtained from the eigenvector :math:`q` of the largest eigenvalue
        :math:`\lambda` of the symmetric (4, 4) matrix :math:`K`. The derivative of :math:`q`
        is implemented by a custom JVP rule (``_top_eigvec``),

        .. math::

            (\lambda I - K + q q^T) \dot{q} = \dot{K} q - (q^T \dot{K} q) q

        where the term :math:`q q^T` makes the matrix nonsingular as long as the
        largest eigenvalue is nondegenerate. The resulting :math:`\dot{q}` is
        orthogonal to :math:`q`, so the normalization of :math:`q` is preserved.

        The default derivative of ``jnp.linalg.eigh`` involves
        :math:`1 / (\lambda_i - \lambda_j)` for all pairs of eigenvalues, and yields ``NaN``
        when any two eigenvalues are degenerate (e.g., near the reference
        geometry of a spherical top molecule such as CH4), even though only
        the top eigenvector is needed here. The custom rule uses only the top
        eigenpair and remains finite in such cases. Since the rule itself is
        differentiable, reverse-mode and higher-order derivatives (used in
        the Hessian contractions) are also available.
    """
    A = jnp.einsum("i,ij,ik->jk", mass, posi_ref_c, posi_c)
    tr = jnp.trace(A)
    D = A - A.T
    delta = jnp.array([D[1, 2], D[2, 0], D[0, 1]])
    K = jnp.block([[tr, delta[None, :]], [delta[:, None], A + A.T - tr * jnp.eye(3)]])

    C = quat_to_rotmat(_top_eigvec(K))

    return C


def get_euler_angles(C: ArrayLike) -> jax.Array:
    """Compute the Euler angles (yaw, pitch, roll) in ZYX convention from the rotation matrix (Eq. (L7) in the paper).

    Args:
        C: ``(3, 3)`` rotation matrix (from reference to current)

    Returns:
        ``(3,)`` Euler angles ``(yaw, pitch, roll)``, where

        - yaw: rotation around z-axis in [-pi, pi).
        - pitch: rotation around y-axis, shifted to [0, pi].
        - roll: rotation around x-axis in [-pi, pi).
    """
    yaw = jnp.arctan2(C[1, 0], C[0, 0])  # [-pi, pi)
    pitch = jnp.arctan2(jnp.sqrt(C[0, 0] ** 2 + C[1, 0] ** 2), C[2, 0])  # [0, pi]
    roll = jnp.arctan2(C[2, 1], C[2, 2])  # [-pi, pi)
    return jnp.array([yaw, pitch, roll])


KIND_ORDER = ("translation", "rotation", "euler", "normal", "bond", "angle", "dihedral", "polar", "azimuthal")


@dataclass
class Coordinate:
    """Definition of a single coordinate."""

    #: Kind of the coordinate.
    kind: str
    #: Indices of the atoms defining the coordinate (empty for global and constant coordinates).
    atom_indices: tuple[int, ...]
    #: Value at the reference geometry.
    value_ref: float
    #: Name of the coordinate.
    name: str
    #: Topology of the coordinate.
    topology: str


def _skew(c: jax.Array) -> jax.Array:
    """Skew-symmetric matrix for cross product.

    c(N, 3) -> c(N, 3, 3)
    """
    z = jnp.zeros(c.shape[0])
    return jnp.stack(
        [
            jnp.stack([z, -c[:, 2], c[:, 1]], axis=-1),
            jnp.stack([c[:, 2], z, -c[:, 0]], axis=-1),
            jnp.stack([-c[:, 1], c[:, 0], z], axis=-1),
        ],
        axis=1,
    )


##################################################################################################
# InternalCoordinates class (Cartesian -> Internal coordinates)
##################################################################################################


class InternalCoordinates:
    """Internal coordinates defined from Cartesian coordinates.

    The coordinate system is set up as follows:

    1. Add coordinates using :meth:`add_bond`, :meth:`add_angle`, etc.
    2. Call :meth:`build` to finalize the coordinate system.
    3. Pass the object to other functions to run dynamics.

    Args:
        posi_ref: ``(N, 3)`` reference positions of the atoms.
        mass: ``(N,)`` atomic masses.
        move_reference: If ``True``, the reference positions are updated to the
            current positions at every step (time-dependent Eckart conditions Eq. (85) in the paper).
            The rotational constraints then do not affect the dynamics, and the angular momentum is conserved.
            They are used only to remove the rotational components in the coordinate transformation, assuming
            zero angular momentum (L = 0). See section III C in the paper.
            If ``False``, ``posi_ref`` is used as a fixed reference. (Eckart conditions Eq. (84) in the paper)
    """

    def __init__(self, posi_ref: ArrayLike, mass: ArrayLike, move_reference: bool = False) -> None:
        self.posi_ref = np.asarray(posi_ref, dtype=float).reshape(-1, 3)
        self.mass = np.asarray(mass, dtype=float)
        self.mass_inv_flat = 1.0 / np.repeat(self.mass, 3)
        self.n_atom = len(self.posi_ref)
        self.move_reference = move_reference
        self.coords = []  # list of Coordinate objects
        self.remove_translation = False
        self.remove_rotation = False
        self.remove_euler = False
        self.normal_vectors = []
        self._built = False
        self.local_kinds = dict(LOCAL_KINDS)
        self.kind_order = list(KIND_ORDER)

    def __repr__(self) -> str:
        lines = ["InternalCoordinates:"]
        for i, c in enumerate(self.coords):
            lines.append(f"  {i:3d}: {c.kind:15s} {c.atom_indices} value_ref={c.value_ref:.6f}")
        return "\n".join(lines)

    def __len__(self) -> int:
        """Return the number of coordinates."""
        return len(self.coords)

    def add_local(self, kind: str, atoms: tuple[int, ...], name: str | None = None) -> None:
        """Add a local coordinate defined by a small number of atoms.

        Args:
            kind: Kind of the coordinate.
            atoms: Indices of the atoms defining the coordinate, in the same
                order as required by the coordinate function.
            name: Name of the coordinate.
                If ``None``, a default name is generated based on ``kind`` and ``atoms``.
        """
        self._add_local(kind, atoms, name)

    def _add_local(self, kind: str, atoms: tuple[int, ...], name: str | None = None) -> None:  # local
        assert kind in self.local_kinds, f"Unknown local coordinate kind: {kind}"
        func, n, topology = self.local_kinds[kind]
        atoms = tuple(int(a) for a in atoms)
        assert len(atoms) == n and len(set(atoms)) == n, f"Invalid atoms for {kind}: {atoms}"
        assert all(0 <= a < self.n_atom for a in atoms), f"Atom indices out of range: {atoms}"
        value_ref = func(self.posi_ref[np.array(atoms)])
        if name is None:
            name = f"{kind}{atoms}"
        self.coords.append(Coordinate(kind, atoms, value_ref, name, topology))

    def add_bond(self, i: int, j: int, name: str | None = None) -> None:  # local
        """Add a bond length between atoms ``i`` and ``j``.

        Args:
            i: Index of the first atom.
            j: Index of the second atom.
            name: Name of the coordinate.
                If ``None``, ``bond(i, j)`` is used as the default name.
        """
        self._add_local("bond", (i, j), name)

    def add_angle(self, i: int, j: int, k: int, name: str | None = None) -> None:  # local
        """Add a bond angle ``i-j-k`` at atom ``j``.

        Args:
            i: Index of the first atom.
            j: Index of the second atom (central atom of the angle).
            k: Index of the third atom.
            name: Name of the coordinate.
                If ``None``, ``angle(i, j, k)`` is used as the default name.
        """
        self._add_local("angle", (i, j, k), name)

    def add_dihedral(self, i: int, j: int, k: int, l: int, name: str | None = None) -> None:  # local
        """Add a dihedral angle ``i-j-k-l``.

        Args:
            i: Index of the first atom.
            j: Index of the second atom.
            k: Index of the third atom.
            l: Index of the fourth atom.
            name: Name of the coordinate.
                If ``None``, ``dihedral(i, j, k, l)`` is used as the default name.
        """
        self._add_local("dihedral", (i, j, k, l), name)

    def add_polar(self, i: int, j: int, name: str | None = None) -> None:  # local
        """Add a polar angle of the vector ``i-j`` from the z-axis.

        Args:
            i: Index of the first atom.
            j: Index of the second atom.
            name: Name of the coordinate.
                If ``None``, ``polar(i, j)`` is used as the default name.
        """
        self._add_local("polar", (i, j), name)

    def add_azimuthal(self, i: int, j: int, name: str | None = None) -> None:  # local
        """Add an azimuthal angle of the vector ``i-j`` from the x-axis in the xy-plane.

        Args:
            i: Index of the first atom.
            j: Index of the second atom.
            name: Name of the coordinate.
                If ``None``, ``azimuthal(i, j)`` is used as the default name.
        """
        self._add_local("azimuthal", (i, j), name)

    def add_translation(self) -> None:  # constant
        r"""Add three (Eckart) translational coordinates (x, y, z) of the center of mass.

        .. math::

            \sum_i m_i (\mathbf{r}_i - \mathbf{R}_i)

        where :math:`\mathbf{R}_i` is the reference position.
        """
        if self.remove_translation:
            return
        self.remove_translation = True
        for d in "xyz":
            self.coords.append(Coordinate("translation", (), 0.0, f"translation_{d}", R))

    def add_rotation(self) -> None:  # constant
        r"""Add three (Eckart) rotational coordinates (x, y, z).

        .. math::

            \sum_i m_i (\mathbf{R}_i - \mathbf{R}_{\text{com}}) \times (\mathbf{r}_i - \mathbf{r}_{\text{com}})

        where :math:`\mathbf{R}_i` is the reference position, and
        :math:`\mathbf{R}_{\text{com}}` and :math:`\mathbf{r}_{\text{com}}` are the
        centers of mass of the reference and current positions.
        """
        if self.remove_rotation:
            return
        assert not self.remove_euler, "Cannot add rotation when Euler angles are removed"
        self.remove_rotation = True
        for d in "xyz":
            self.coords.append(Coordinate("rotation", (), 0.0, f"rotation_{d}", R))

    def add_eckart(self) -> None:  # constant
        """Add Eckart translational and rotational coordinates."""
        self.add_translation()
        self.add_rotation()

    def add_euler(self) -> None:  # global
        """Add three Euler angles (yaw, pitch, roll) of the overall rotation."""
        if self.remove_euler:
            return
        assert not self.remove_rotation, "Cannot add Euler angles when rotation is removed"
        self.remove_euler = True
        values_ref = get_euler_angles(
            get_rotation_matrix(self._centered(self.posi_ref), self._centered(self.posi_ref), self.mass)
        )
        for d, t, topology in zip("yaw pitch roll".split(), values_ref, (S1, I_MINUS, S1)):
            self.coords.append(Coordinate("euler", (), float(t), f"euler_{d}", topology))

    def add_normal_mode(self, mode_vector: ArrayLike, name: str | None = None) -> None:  # global
        """Not implemented yet."""
        raise NotImplementedError("Normal mode coordinates are not implemented yet.")
        mode_vector = np.asarray(mode_vector, dtype=float).reshape(-1)
        assert mode_vector.shape == (3 * self.n_atom,), f"Invalid mode vector shape: {mode_vector.shape}"
        value_ref = 0.0
        if name is None:
            name = f"normal{len(self.normal_vectors)}"
        self.normal_vectors.append(mode_vector)
        self.coords.append(Coordinate("normal", (), float(value_ref), name, R))

    def _centered(self, p: ArrayLike) -> jax.Array:
        return p - jnp.sum(p * self.mass[:, None], axis=0) / jnp.sum(self.mass)

    def build(self) -> None:
        """Finalize the coordinate system.

        The coordinates are sorted in the order of ``KIND_ORDER``, so the row
        order after ``build()`` can differ from the order in which they were added.
        Use :meth:`find_coordinate` or ``inv_argsort`` to find the new row indices.

        After the call, the following attributes are available (``x`` is the
        flattened ``(3N,)`` positions):

        - ``names``, ``index``, ``topology``, ``atom_indices``: coordinate information.
        - ``argsort``, ``inv_argsort``: mapping between the original and sorted orders.
        - ``get_value(x)``: ``(K,)`` raw coordinate values.
        - ``get_const(x)``: ``(K,)`` residuals ``value - value_ref``, wrapped to
          [-pi, pi) for S1 topology.
        - ``get_const_grad(x)``: ``(K, 3N)`` gradient ``B`` of the residuals (Eq. (15)).
        - ``get_const_hess_kj_contract(x, w, V)``: ``(3N, X)`` contraction of the
          second derivatives (Eq. (34)).
        - ``get_const_hess_ij_contract(x, w, V)``: ``(K, X)`` contraction of the
          second derivatives (Eq. (36)).
        - ``get_value_batch``, ``get_const_batch``, ``get_const_grad_batch``:
          batched versions over the first axis.
        """
        if self._built:
            warnings.warn("InternalCoordinates already built. Ignoring build() call.")
            return
        self._built = True
        for c in self.coords:
            assert c.kind in self.kind_order, f"Unknown kind: {c.kind}"

        self.argsort = sorted(range(len(self.coords)), key=lambda i: self.kind_order.index(self.coords[i].kind))
        self.coords = [self.coords[i] for i in self.argsort]
        self.inv_argsort = np.empty(len(self.argsort), dtype=int)
        self.inv_argsort[self.argsort] = np.arange(len(self.argsort))
        coords = self.coords

        self.names = [c.name for c in coords]
        self.index = {c.name: i for i, c in enumerate(coords)}
        self.topology = [c.topology for c in coords]
        self.atom_indices = [c.atom_indices for c in coords]

        n_atom = self.n_atom
        n_trans = 3 if self.remove_translation else 0
        n_rot = 3 if self.remove_rotation else 0
        n_euler = 3 if self.remove_euler else 0
        n_normal = len(self.normal_vectors)
        n_global = n_euler + n_normal
        normal_vectors = jnp.array(self.normal_vectors, dtype=float).reshape(n_normal, 3 * n_atom)
        sqrt_mass = jnp.repeat(jnp.sqrt(self.mass), 3)

        values_ref = jnp.array([c.value_ref for c in coords], dtype=float)
        # only S1 topology is regarded as periodic
        periodic = jnp.array([c.topology == S1 for c in coords], dtype=bool)
        mass = jnp.array(self.mass, dtype=float)
        posi_ref_default = jnp.array(self.posi_ref, dtype=float)
        move_reference = self.move_reference

        # local groups: (function, atom_indices, row slice)
        groups = []
        for kind in self.kind_order:
            if kind not in self.local_kinds:
                continue
            func, n, _ = self.local_kinds[kind]
            rows = [i for i, c in enumerate(coords) if c.kind == kind]
            if not rows:
                continue
            idx = jnp.array([coords[i].atom_indices for i in rows], dtype=int).reshape(-1, n)
            groups.append((func, idx, slice(rows[0], rows[-1] + 1)))

        def reference(x: jax.Array) -> jax.Array:
            if move_reference:
                return jax.lax.stop_gradient(x)
            return posi_ref_default

        def mass_centered(p: jax.Array) -> jax.Array:
            return p - jnp.sum(p * mass[:, None], axis=0) / jnp.sum(mass)

        def evaluate_global_f(x: jax.Array) -> jax.Array:
            """Evaluate the global coordinate values.

            x: (N, 3) -> (n_global,)
            """
            ref_c = mass_centered(reference(x))
            x_c = mass_centered(x)
            rot_mat = get_rotation_matrix(x_c, ref_c, mass)
            res = []
            if n_euler:
                res.append(get_euler_angles(rot_mat))
            if n_normal:
                res.append(normal_vectors @ (sqrt_mass * (x_c @ rot_mat - ref_c).reshape(-1)))
            return jnp.concatenate(res)

        def evaluate_all_f(x: jax.Array) -> jax.Array:
            """Evaluate all coordinate values.

            x: (N, 3) -> (K,)
            """
            ref = reference(x)
            res = []

            if n_trans:
                res.append(jnp.sum(mass[:, None] * (x - ref), axis=0))
            if n_rot:
                res.append(jnp.sum(mass[:, None] * jnp.cross(mass_centered(ref), x), axis=0))
            if n_global:
                res.append(evaluate_global_f(x))
            for func, idx, _ in groups:
                res.append(jax.vmap(func)(x[idx]))
            return jnp.concatenate(res)

        def evaluate_residual(x: jax.Array) -> jax.Array:
            """Evaluate the residuals F(x) = value - value_ref, wrapped to [-pi, pi) for S1 topology.

            x: (N, 3) -> (K,)
            """
            d = evaluate_all_f(x) - values_ref
            return jnp.where(periodic, (d + jnp.pi) % (2 * jnp.pi) - jnp.pi, d)

        def evaluate_gradient(x: jax.Array) -> jax.Array:
            """Evaluate the gradient of the residual function (Eq. (15)).

            x: (N, 3) -> (K, 3N)
            """
            ref = reference(x)
            res = []
            if n_trans:
                # d/dx_i sum_j m_j (x_j - c) = m_i * I
                B_trans = mass[None, :, None] * jnp.eye(3)[:, None, :]  # (3, N, 3)
                res.append(B_trans.reshape(3, -1))
            if n_rot:
                # d/dx_i sum_j m_j (c_j x x_j) = m_i (r_i x I) = m_i * skew(c_i)
                B_rot = (mass[:, None, None] * _skew(mass_centered(ref))).transpose(1, 0, 2)  # (3, N, 3)
                res.append(B_rot.reshape(3, -1))
            if n_global:
                res.append(jax.jacrev(evaluate_global_f)(x).reshape(n_global, -1))  # (n_global, 3N)
            for func, idx, _ in groups:  # for each local coordinate group
                g = jax.vmap(jax.grad(func))(x[idx])  # (K_i, N_i, 3) (coordinates size, local atoms, xyz)
                m = idx.shape[0]
                B = jnp.zeros((m, n_atom, 3)).at[jnp.arange(m)[:, None], idx, :].set(g)  # (K_i, N, 3)
                res.append(B.reshape(m, -1))
            return jnp.concatenate(res)  # (K, 3N)

        def evaluate_hess_kj_contract(x: jax.Array, w: jax.Array, V: jax.Array) -> jax.Array:
            """Contract the second derivatives of the residual function over k and j (Eq. (34)).

            x: (N, 3), w: (K,), V: (3N, X)
            -> (wHV)_(i,a) = sum_j sum_k w_k d^2/dx_i dx_j F_k(x) V_(j,a) : (3N, X)
            """
            grad_wF = jax.grad(lambda x: jnp.sum(w * evaluate_all_f(x)))  # (N, 3)
            jvp_1d = lambda v: jax.jvp(grad_wF, (x,), (v.reshape(-1, 3),))[1].reshape(-1)  # (3N,)
            return jax.vmap(jvp_1d, in_axes=1, out_axes=1)(V)  # (3N, X)

        def evaluate_hess_ij_contract(x: jax.Array, w: jax.Array, V: jax.Array) -> jax.Array:
            """Contract the second derivatives of the residual function over i and j (Eq. (36)).

            x: (N, 3), w: (3N,), V: (3N, X)
            -> (wHV)_(k,a) = sum_i sum_j w_i d^2/dx_i dx_j F_k(x) V_(j,a) : (K, X)
            """
            jvp_w = lambda y: jax.jvp(evaluate_all_f, (y,), (w.reshape(-1, 3),))[1]  # (K,)
            jvp_wv = lambda v: jax.jvp(jvp_w, (x,), (v.reshape(-1, 3),))[1]  # (K,)
            return jax.vmap(jvp_wv, in_axes=1, out_axes=1)(V)  # (K, X)

        def _flat(f: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
            return lambda x_flat, *args: f(x_flat.reshape(-1, 3), *args)

        # single geometry
        self.get_value = jax.jit(_flat(evaluate_all_f))
        self.get_const = jax.jit(_flat(evaluate_residual))
        self.get_const_grad = jax.jit(_flat(evaluate_gradient))
        self.get_const_hess_kj_contract = jax.jit(_flat(evaluate_hess_kj_contract))
        self.get_const_hess_ij_contract = jax.jit(_flat(evaluate_hess_ij_contract))

        # multiple geometries
        self.get_value_batch = jax.jit(jax.vmap(self.get_value, in_axes=0, out_axes=0))
        self.get_const_batch = jax.jit(jax.vmap(self.get_const, in_axes=0, out_axes=0))
        self.get_const_grad_batch = jax.jit(jax.vmap(self.get_const_grad, in_axes=0, out_axes=0))

        print(f"Build complete: {len(self)} coordinates")

    def find_coordinate(self, kind: str, atoms: Sequence[int]) -> tuple[int, float, str, str]:
        """Find a local coordinate after :meth:`build` from its kind and atom indices.

        Args:
            kind: Kind of the coordinate (e.g. ``"bond"``, ``"dihedral"``).
            atoms: Indices of the atoms defining the coordinate, in the same
                order as given when it was added.

        Returns:
            ``(index, value_ref, name, topology)``, where ``index`` is the row index
            after :meth:`build`, which can be used in ``active_idx``.

        Raises:
            RuntimeError: If :meth:`build` has not been called.
            KeyError: If the coordinate is not found.
            ValueError: If more than one coordinate matches.
        """
        if not hasattr(self, "index"):
            raise RuntimeError("Call build() first.")
        atoms = tuple(int(a) for a in atoms)
        matches = [i for i, c in enumerate(self.coords) if c.kind == kind and c.atom_indices == atoms]
        if not matches:
            available = [(c.kind, c.atom_indices) for c in self.coords if c.kind in self.local_kinds]
            raise KeyError(f"Coordinate {kind}{atoms} not found. Available: {available}")
        if len(matches) > 1:
            raise ValueError(f"Coordinate {kind}{atoms} is not unique (rows {matches}).")
        i = matches[0]
        c = self.coords[i]
        return i, float(c.value_ref), c.name, c.topology

    def register_kind(self, kind: str, func: Callable[[jax.Array], jax.Array], n_atoms: int, topology: str) -> None:
        """Register a new kind of local coordinate.

        Args:
            kind: Kind of the coordinate (e.g. ``"bond"``, ``"dihedral"``).
            func: Function that computes the coordinate value from the positions of the atoms.
                It should take an array of shape ``(n_atoms, 3)`` and return a scalar.
            n_atoms: Number of atoms defining the coordinate.
            topology: Topology of the coordinate (e.g. ``S1``, ``R``, etc.).

        Examples:
            For example, to fix the difference of two bond lengths,
            one can define a new coordinate kind ``"bond_diff"`` as follows:

            >>> def bond_diff(p: jax.Array) -> jax.Array:
            ...     return jnp.linalg.norm(p[1] - p[0]) - jnp.linalg.norm(p[2] - p[0])
            >>> coord = InternalCoordinates(posi_ref, mass)
            >>> coord.register_kind("bond_diff", bond_diff, 3, R)
            >>> coord.add_local("bond_diff", (0, 1, 2))
        """
        assert kind not in self.local_kinds, f"Kind {kind} is already registered."
        assert topology in (R, R_PLUS, S1, I_MINUS), f"Invalid topology: {topology}"
        self.local_kinds[kind] = (func, n_atoms, topology)
        self.kind_order.append(kind)


##################################################################################################
# QuaternionCoordinates class (Quaternion -> Euler angles)
##################################################################################################


class QuaternionCoordinates:
    """Euler angles defined from quaternion coordinates.

    This object can be used in place of :class:`InternalCoordinates` to define the Euler angles.

    The coordinates are ``(yaw, pitch, roll, norm)``, where ``norm`` is the
    normalization condition ``(q^T q - 1)/2`` of the quaternion.
    The functions take the ``(4,)`` quaternion in place of the ``(3N,)`` positions.
    """

    def __init__(self) -> None:
        self.names = ["yaw", "pitch", "roll", "norm"]
        self.coords = self.names
        self.index = {name: i for i, name in enumerate(self.names)}
        self.topology = [S1, I_MINUS, S1, R]
        self.mass_inv_flat = jnp.ones(4, dtype=float)  # dummy mass for quaternion coordinates

        def evaluate_all_f(q: jax.Array) -> jax.Array:
            """Evaluate the Euler angles and norm condition.

            q: (4,) -> [yaw, pitch, roll, (q^T q - 1)/2 (norm condition)]: (4,)
            """
            return jnp.concatenate([get_euler_angles(quat_to_rotmat(q)), jnp.array([0.5 * (q @ q - 1.0)])])

        def evaluate_hess_kj_contract(q: jax.Array, w: jax.Array, V: jax.Array) -> jax.Array:
            """Contract the second derivatives over k and j (Eq. (34) in the paper).

            q: (4,), w: (4,), V: (4, X)
            -> (wHV)_(i,a) = sum_j sum_k w_k d^2/dq_i dq_j F_k(q) V_(j,a) : (4, X)
            """
            grad_wF = jax.grad(lambda q: jnp.sum(w * evaluate_all_f(q)))  # (4,)
            jvp_1d = lambda v: jax.jvp(grad_wF, (q,), (v.reshape(-1),))[1].reshape(-1)  # (4,)
            return jax.vmap(jvp_1d, in_axes=1, out_axes=1)(V)  # (4, X)

        self.get_value = jax.jit(evaluate_all_f)
        self.get_const_grad = jax.jit(jax.jacrev(evaluate_all_f))
        self.get_const_hess_kj_contract = jax.jit(evaluate_hess_kj_contract)

    def __repr__(self) -> str:
        lines = ["QuaternionCoordinate:"]
        for i, name in enumerate(self.names):
            lines.append(f"  {i:3d}: {name:15s}")
        return "\n".join(lines)

    def __len__(self) -> int:
        """Return the number of coordinates."""
        return len(self.coords)

    def build(self) -> None:
        """Do nothing, since the coordinate system is already defined."""
        print("Build is not required for QuaternionCoordinate, as it is already defined.")
        pass


def make_converters(
    coord_all: InternalCoordinates | QuaternionCoordinates, active_idx: ArrayLike | None = None
) -> tuple[
    Callable[[jax.Array], jax.Array],
    Callable[[jax.Array, jax.Array], jax.Array],
    Callable[[jax.Array, jax.Array], jax.Array],
    Callable[[jax.Array], jax.Array],
]:
    """Make coordinate converters for the active (physical) coordinates.

    Args:
        coord_all: InternalCoordinates for the full system. It must define ``3N``
            coordinates for coordinate transformation.
        active_idx: (m,) row indices of the active coordinates.
            If ``None``, all coordinates are regarded as active.

    Returns:
        ``(to_physical_posi, to_physical_mome, to_redundant_mome, get_R_active)``,
        where the first three are jitted and batched over the first axis:

        - ``to_physical_posi(posi)``: ``(n_batch, 3N)`` -> ``(n_batch, m)``.
        - ``to_physical_mome(posi, mome)``: ``(n_batch, 3N)``, ``(n_batch, 3N)`` -> ``(n_batch, m)``,
          ``P_A = A_A^T p``.
        - ``to_redundant_mome(posi, mome_int)``: ``(n_batch, 3N)``, ``(n_batch, m)`` -> ``(n_batch, 3N)``,
          ``p = R_A P_A``.
        - ``get_R_active(x)``: ``(3N,)`` -> ``(3N, m)`` (Eq. (41)), not batched.

        For :class:`QuaternionCoordinates`, 3N is replaced by 4.
    """
    if active_idx is None:
        active_idx = np.arange(len(coord_all))
    const_idx = np.setdiff1d(np.arange(len(coord_all)), active_idx)
    mass_inv_flat = jnp.asarray(coord_all.mass_inv_flat, dtype=float)
    get_value, get_grad = coord_all.get_value, coord_all.get_const_grad

    def get_R_active(x: jax.Array) -> jax.Array:
        B = get_grad(x)
        B_A, B_C = B[active_idx], B[const_idx]
        G_CC = B_C @ (mass_inv_flat[:, None] * B_C.T)
        G_CA = B_C @ (mass_inv_flat[:, None] * B_A.T)
        return B_A.T - B_C.T @ jnp.linalg.solve(G_CC, G_CA)  # (3N, m)

    to_physical_posi = jax.jit(jax.vmap(lambda x: get_value(x)[active_idx]))
    to_physical_mome = jax.jit(jax.vmap(lambda x, p: jnp.linalg.solve(get_grad(x).T, p)[active_idx]))
    to_redundant_mome = jax.jit(jax.vmap(lambda x, P: get_R_active(x) @ P))
    return to_physical_posi, to_physical_mome, to_redundant_mome, get_R_active


def quaternion_metric(inertia: ArrayLike) -> Callable[[jax.Array], jax.Array]:
    """Make the metric function for the quaternion dynamics (Eq. (L5) in the paper).

    Args:
        inertia: ``(3,)`` principal moments of inertia.

    Returns:
        Function to compute the metric, ``(4,)`` quaternion -> ``(4, 4)`` matrix.
        It can be passed to :class:`pyscivr.integrators.ImplicitVerlet` as ``metric_func``.
    """
    inertia = jnp.asarray(inertia, dtype=float)

    def metric_func(q: jax.Array) -> jax.Array:
        q0, q1, q2, q3 = q
        S = jnp.array(
            [
                [-q1, q0, q3, -q2],
                [-q2, -q3, q0, q1],
                [-q3, q2, -q1, q0],
            ]
        )
        return 0.25 * S.T @ (S / inertia[:, None])  # (4, 4)

    return metric_func
