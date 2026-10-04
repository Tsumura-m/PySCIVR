from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# topologies (section II E 2)
R = "R"          # full line (-inf, inf)
R_PLUS = "R+"    # half line [0, inf) : bond length
S1 = "S1"        # circle [x, x+2pi) (x is any real number) : dihedral angle, yaw & roll angle
I_MINUS = "I-"   # semi-circle [0, pi] : bond angle, pitch angle


##################################################################################################
# Local internal coodinates (coordinate defined by a small number of atoms)
##################################################################################################

def get_bond(p):
    return jnp.linalg.norm(p[1] - p[0])

def get_bond_angle(p):
    u, v = p[0] - p[1], p[2] - p[1]
    x = jnp.dot(u, v)                         # |u||v|cos(theta)
    y = jnp.linalg.norm(jnp.cross(u, v))      # |u||v|sin(theta)
    return jnp.arctan2(y, x)                   # theta

def get_dihedral_angle(p):
    b1, b2, b3 = p[1] - p[0], p[2] - p[1], p[3] - p[2]
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    m1 = jnp.cross(n1, b2 / jnp.linalg.norm(b2))
    x = jnp.dot(n1, n2)          # |n1||n2|cos(phi)
    y = jnp.dot(m1, n2)          # |n1||n2|sin(phi)
    return jnp.arctan2(y, x)

def get_polar_angle(p):
    d = p[1] - p[0]
    return jnp.arctan2(jnp.sqrt(d[0]**2 + d[1]**2), d[2])  # theta in [0, pi]

def get_azimuthal_angle(p):
    d = p[1] - p[0]
    return jnp.arctan2(d[1], d[0])  # phi in [-pi, pi)

# kind -> (function, number of atoms, topology)
LOCAL_KINDS = {
    "bond" : (get_bond, 2, R_PLUS),
    "angle" : (get_bond_angle, 3, I_MINUS),
    "dihedral" : (get_dihedral_angle, 4, S1),
    "polar" : (get_polar_angle, 2, I_MINUS),
    "azimuthal" : (get_azimuthal_angle, 2, S1),
}

# top eigenvector q of the symmetric matrix K (K q = lam q)
@jax.custom_jvp
def _top_eigvec(K):
    return jnp.linalg.eigh(K)[1][:, -1]

# derivative of the top eigenvector q of the symmetric matrix K
@_top_eigvec.defjvp
def _top_eigvec_jvp(primals, tangents): # primals = K, tangents = dK
    K, = primals
    dK, = tangents
    q = _top_eigvec(K)
    lam = q @ K @ q
    r = dK @ q - (q @ dK @ q) * q
    dq = jnp.linalg.solve((lam * jnp.eye(K.shape[0]) - K + jnp.outer(q, q)), r)
    return q, dq

def quat_to_rotmat(q):
    """
    Convert a quaternion to a rotation matrix.
    q : (4,) quaternion (q0, q1, q2, q3)
    returns:
        C : (3, 3) rotation matrix
    """
    q0, q1, q2, q3 = q
    C = jnp.array([
        [1.0-2.0*(q2**2 + q3**2), 2.0*(q1*q2 - q0*q3), 2.0*(q1*q3 + q0*q2)],
        [2.0*(q1*q2 + q0*q3), 1.0-2.0*(q1**2 + q3**2), 2.0*(q2*q3 - q0*q1)],
        [2.0*(q1*q3 - q0*q2), 2.0*(q2*q3 + q0*q1), 1.0-2.0*(q1**2 + q2**2)]
    ])
    return C

def get_rotation_matrix(posi_c, posi_ref_c, mass):
    """
    Compute the rotation matrix C that minimizes the mass-weighted RMSD
    between posi_c and posi_ref_c with Horn's method.

    min_C (posi_ref_c C^T - posi_c)^T M (posi_ref_c C^T - posi_c)  (M = diag(mass))
    C rotates posi_ref_c to posi_c (posi_ref_c C^T ~ posi_c)

    posi_c : (N, 3) mass-centered coordinates
    posi_ref_c : (N, 3) mass-centered reference coordinates
    mass : (N,) atomic masses
    returns:
        C : (3, 3) rotation matrix
    """
    A = jnp.einsum("i,ij,ik->jk", mass, posi_ref_c, posi_c)
    tr = jnp.trace(A)
    D = A - A.T
    delta = jnp.array([D[1, 2], D[2, 0], D[0, 1]])
    K = jnp.block([[tr, delta[None, :]], [delta[:, None], A + A.T - tr * jnp.eye(3)]])

    C = quat_to_rotmat(_top_eigvec(K))

    return C

def get_euler_angles(C):
    """
    Compute the Euler angles (yaw, pitch, roll) in ZYX convention from the rotation matrix C.
    C : (3, 3) rotation matrix (from reference to current)
    returns:
        yaw : rotation around z-axis
        pitch : rotation around y-axis  (shifted to [0, pi])
        roll : rotation around x-axis
    """
    yaw = jnp.arctan2(C[1, 0], C[0, 0]) # [-pi, pi)
    pitch = jnp.arctan2(jnp.sqrt(C[0, 0]**2 + C[1, 0]**2), C[2, 0]) # [0, pi]
    roll = jnp.arctan2(C[2, 1], C[2, 2]) # [-pi, pi)
    return jnp.array([yaw, pitch, roll])


KIND_ORDER = ("translation", "rotation", "euler", "normal",
              "bond", "angle", "dihedral", "polar", "azimuthal")


@dataclass
class Coordinate:
    kind: str
    atom_indices: tuple
    target: float
    name : str
    topology : str

def _skew(c):
    """Skew-symmetric matrix for cross product.
    c(N, 3) -> c(N, 3, 3)
    """

    z = jnp.zeros(c.shape[0])
    return jnp.stack([jnp.stack([z, -c[:, 2], c[:, 1]], axis=-1),
                      jnp.stack([c[:, 2], z, -c[:, 0]], axis=-1),
                      jnp.stack([-c[:, 1], c[:, 0], z], axis=-1)], axis=1)


##################################################################################################
# InternalCoordinates class (Cartesian -> Internal coordinates)
##################################################################################################

class InternalCoordinates:
    """
    InternalCoordinates class for defining internal coordinates from Cartesian coordinates.

    Usage:
        1, add coordinates using add_bond, add_angle, etc.
        2, call build() to finalize the coordinate system
        3, pass the InternalCoordinates object to other functions to run dynamics

    posi_ref : (N, 3) reference positions
    mass : (N,) atomic masses

    move_reference : whether to move the reference positions during the dynamics (default: False)
    """

    def __init__(self, posi_ref, mass, move_reference=False):
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

    def __repr__(self):
        lines = ["InternalCoordinates:"]
        for i, c in enumerate(self.coords):
            lines.append(f"  {i:3d} : {c.kind:15s} {c.atom_indices} target={c.target:.6f}")
        return "\n".join(lines)

    def _add_local(self, kind, atoms, target, name): # local
        func, n, topology = LOCAL_KINDS[kind]
        atoms = tuple(int(a) for a in atoms)
        assert len(atoms) == n and len(set(atoms)) == n, f"Invalid atoms for {kind}: {atoms}"
        assert all(0 <= a < self.n_atom for a in atoms), f"Atom indices out of range: {atoms}"
        if target is None:
            target = func(self.posi_ref[np.array(atoms)])
        if name is None:
            name = f"{kind}{atoms}"
        self.coords.append(Coordinate(kind, atoms, target, name, topology))

    def add_bond(self, i, j, target=None, name=None): # local
        self._add_local("bond", (i, j), target, name)

    def add_angle(self, i, j, k, target=None, name=None): # local
        self._add_local("angle", (i, j, k), target, name)

    def add_dihedral(self, i, j, k, l, target=None, name=None): # local
        self._add_local("dihedral", (i, j, k, l), target, name)

    def add_polar(self, i, j, target=None, name=None): # local
        self._add_local("polar", (i, j), target, name)

    def add_azimuthal(self, i, j, target=None, name=None): # local
        self._add_local("azimuthal", (i, j), target, name)

    def add_translation(self): # constant
        if self.remove_translation:
            return
        self.remove_translation = True
        for d in "xyz":
            self.coords.append(Coordinate("translation", (), 0.0, f"translation_{d}", R))

    def add_rotation(self): # constant
        if self.remove_rotation:
            return
        assert not self.remove_euler, "Cannot add rotation when Euler angles are removed"
        self.remove_rotation = True
        for d in "xyz":
            self.coords.append(Coordinate("rotation", (), 0.0, f"rotation_{d}", R))

    def add_eckart(self): # constant
        self.add_translation()
        self.add_rotation()

    def add_euler(self): # global
        if self.remove_euler:
            return
        assert not self.remove_rotation, "Cannot add Euler angles when rotation is removed"
        self.remove_euler = True
        targets = get_euler_angles(get_rotation_matrix(self._centered(self.posi_ref),
                                                       self._centered(self.posi_ref), self.mass))
        for d, t, topology in zip("yaw pitch roll".split(), targets, (S1, I_MINUS, S1)):
            self.coords.append(Coordinate("euler", (), float(t), f"euler_{d}", topology))

    def add_normal_mode(self, mode_vector, target=None, name=None): # global
        mode_vector = np.asarray(mode_vector, dtype=float).reshape(-1)
        assert mode_vector.shape == (3 * self.n_atom,), \
               f"Invalid mode vector shape: {mode_vector.shape}"
        if target is None:
            target = 0.0
        if name is None:
            name = f"normal{len(self.normal_vectors)}"
        self.normal_vectors.append(mode_vector)
        self.coords.append(Coordinate("normal", (), float(target), name, R))

    def _centered(self, p):
        return p - jnp.sum(p * self.mass[:, None], axis=0) / jnp.sum(self.mass)

    def build(self):
        for c in self.coords:
            assert c.kind in KIND_ORDER, f"Unknown kind: {c.kind}"

        self.argsort = sorted(range(len(self.coords)),
                              key=lambda i: KIND_ORDER.index(self.coords[i].kind))
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

        targets = jnp.array([c.target for c in coords], dtype=float)
        # only S1 topology is regarded as periodic
        periodic = jnp.array([c.topology == S1 for c in coords], dtype=bool)
        mass = jnp.array(self.mass, dtype=float)
        posi_ref_default = jnp.array(self.posi_ref, dtype=float)
        move_reference = self.move_reference

        # local groups : (function, atom_indices, row slice)
        groups = []
        for kind, (func, n, _) in LOCAL_KINDS.items():
            rows = [i for i, c in enumerate(coords) if c.kind == kind]
            if not rows:
                continue
            idx = jnp.array([coords[i].atom_indices for i in rows], dtype=int).reshape(-1, n)
            groups.append((func, idx, slice(rows[0], rows[-1] + 1)))

        def reference(x):
            if move_reference:
                return jax.lax.stop_gradient(x)
            return posi_ref_default

        def mass_centered(p):
            return p - jnp.sum(p * mass[:, None], axis=0) / jnp.sum(mass)

        def evaluate_global_f(x):
            """
            x : (N, 3) -> global coordinate values (K,)
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

        def evaluate_all_f(x):
            """
            x : (N, 3) -> raw coordinate values (K,)
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

        def evaluate_residual(x):
            """
            x : (N, 3) -> F(x) = value - target, wrapped to [-pi, pi) for S1 topology
            """
            d = evaluate_all_f(x) - targets
            return jnp.where(periodic, (d + jnp.pi) % (2 * jnp.pi) - jnp.pi, d)

        def evaluate_gradient(x):
            """
            x : (N, 3) -> B_(k,i) = dF_k(x)/dx_i : (K, 3N)
            """
            ref = reference(x)
            res = []
            if n_trans:
                # d/dx_i sum_j m_j (x_j - c) = m_i * I
                B_trans = mass[None, :, None] * jnp.eye(3)[:, None, :] # (3, N, 3)
                res.append(B_trans.reshape(3, -1))
            if n_rot:
                # d/dx_i sum_j m_j (c_j x x_j) = m_i (r_i x I) = m_i * skew(c_i)
                B_rot = (mass[:, None, None] * _skew(mass_centered(ref))).transpose(1, 0, 2) # (3, N, 3)
                res.append(B_rot.reshape(3, -1))
            if n_global:
                res.append(jax.jacrev(evaluate_global_f)(x).reshape(n_global, -1))  # (n_global, 3N)
            for func, idx, _ in groups: # for each local coordinate group
                g = jax.vmap(jax.grad(func))(x[idx])  # (K_i, N_i, 3) (coordinates size, local atoms, xyz)
                m = idx.shape[0]
                B = jnp.zeros((m, n_atom, 3)).at[jnp.arange(m)[:, None], idx, :].set(g) # (K_i, N, 3)
                res.append(B.reshape(m, -1))
            return jnp.concatenate(res) # (K, 3N)

        def evaluate_hess_kj_contract(x, w, V):
            """
            x : (N, 3), w : (K,), V : (3N, X)
            -> (wHV)_(i,a) = sum_j sum_k w_k d^2/dx_i dx_j F_k(x) V_(j,a)  : (3N, X)
            """
            grad_wF = jax.grad(lambda x : jnp.sum(w * evaluate_all_f(x)))  # (N, 3)
            jvp_1d = lambda v : jax.jvp(grad_wF, (x,), (v.reshape(-1, 3),))[1].reshape(-1) # (3N,)
            return jax.vmap(jvp_1d, in_axes=1, out_axes=1)(V)  # (3N, X)

        def evaluate_hess_ij_contract(x, w, V):
            """
            x : (N, 3), w : (3N,), V : (3N, X)
            -> (wHV)_(k,a) = sum_i sum_j w_i d^2/dx_i dx_j F_k(x) V_(j,a)  : (K, X)
            """
            jvp_w = lambda y : jax.jvp(evaluate_all_f, (y,), (w.reshape(-1, 3),))[1] # (K,)
            jvp_wv = lambda v : jax.jvp(jvp_w, (x,), (v.reshape(-1, 3),))[1] # (K,)
            return jax.vmap(jvp_wv, in_axes=1, out_axes=1)(V)  # (K, X)


        def _flat(f):
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

        print(f"Build complete: {len(coords)} coordinates")



##################################################################################################
# QuaternionCoordinates class (Quaternion -> Euler angles)
##################################################################################################

class QuaternionCoordinates:
    """
    QuaternionCoordinate class for defining Euler angles from quaternion coordinates.
    """

    def __init__(self):
        self.names = ["yaw", "pitch", "roll", "norm"]
        self.coords = self.names
        self.index = {name: i for i, name in enumerate(self.names)}
        self.topology = [S1, I_MINUS, S1, R]
        self.mass_inv_flat = jnp.ones(4, dtype=float)  # dummy mass for quaternion coordinates

        def evaluate_all_f(q):
            """
            q : quaternion (4,) -> [ yaw, pitch, roll, (q^T q - 1)/2 (norm condition) ] (4,)
            """
            return jnp.concatenate([get_euler_angles(quat_to_rotmat(q)), jnp.array([0.5 * (q @ q - 1.0)])])

        def evaluate_hess_kj_contract(q, w, V):
            """
            q : (4,), w : (4,), V : (4, X)
            -> (wHV)_(i,a) = sum_j sum_k w_k d^2/dq_i dq_j F_k(q) V_(j,a)  : (4, X)
            """
            grad_wF = jax.grad(lambda q : jnp.sum(w * evaluate_all_f(q)))  # (4,)
            jvp_1d = lambda v : jax.jvp(grad_wF, (q,), (v.reshape(-1),))[1].reshape(-1) # (4,)
            return jax.vmap(jvp_1d, in_axes=1, out_axes=1)(V)  # (4, X)

        self.get_value = jax.jit(evaluate_all_f)
        self.get_const_grad = jax.jit(jax.jacrev(evaluate_all_f))
        self.get_const_hess_kj_contract = jax.jit(evaluate_hess_kj_contract)

    def __repr__(self):
        lines = ["QuaternionCoordinate:"]
        for i, name in enumerate(self.names):
            lines.append(f"  {i:3d} : {name:15s}")
        return "\n".join(lines)

    def build(self):
        print("Build is not required for QuaternionCoordinate, as it is already defined.")
        pass

def make_converters(coord_all, active_idx=None):
    """Coordinate conversions for the active (physical) coordinates (jitted, batched over the first axis).

    coord_all : Coordinates class for coordinate transformation (InternalCoordinates or QuaternionCoordinates)
    active_idx : (m,) row indices of the active coordinates

    returns:
        to_physical_posi(posi)           : (n_batch, 3N) -> (n_batch, m)
        to_physical_mome(posi, mome)     : (n_batch, 3N), (n_batch, 3N) -> (n_batch, m)   P_A = A_A^T p
        to_redundant_mome(posi, mome_int): (n_batch, 3N), (n_batch, m) -> (n_batch, 3N)   p = R_A P_A
        get_R_active(x)                  : (3N,) -> (3N, m)
    """
    if active_idx is None:
        active_idx = np.arange(len(coord_all.coords))
    const_idx = np.setdiff1d(np.arange(len(coord_all.coords)), active_idx)
    mass_inv_flat = jnp.asarray(coord_all.mass_inv_flat, dtype=float)
    get_value, get_grad = coord_all.get_value, coord_all.get_const_grad

    def get_R_active(x):
        B = get_grad(x)
        B_A, B_C = B[active_idx], B[const_idx]
        G_CC = B_C @ (mass_inv_flat[:, None] * B_C.T)
        G_CA = B_C @ (mass_inv_flat[:, None] * B_A.T)
        return B_A.T - B_C.T @ jnp.linalg.solve(G_CC, G_CA)  # (3N, m)

    to_physical_posi = jax.jit(jax.vmap(lambda x: get_value(x)[active_idx]))
    to_physical_mome = jax.jit(jax.vmap(lambda x, p: jnp.linalg.solve(get_grad(x).T, p)[active_idx]))
    to_redundant_mome = jax.jit(jax.vmap(lambda x, P: get_R_active(x) @ P))
    return to_physical_posi, to_physical_mome, to_redundant_mome, get_R_active
