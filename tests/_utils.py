import numpy as np
 
from pyscivr.units import CmInv
  
def quat_to_mat(q):
    """C(q) : reference -> current, posi = posi_ref C^T"""
    q0, q1, q2, q3 = q
    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]])
 
def rot_zyx(yaw, pitch, roll):  # pitch : shifted to [0, pi]
    p = pitch - np.pi / 2
    cz, sz, cy, sy, cx, sx = np.cos(yaw), np.sin(yaw), np.cos(p), np.sin(p), np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx
 
 
def rot_axis(axis, angle):
    """Rodrigues : rotation by angle around the unit vector axis"""
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
 
 
def get_S(q):
    """omega_body = 2 S(q) dq/dt"""
    q0, q1, q2, q3 = q
    return np.array([[-q1, q0, q3, -q2],
                     [-q2, -q3, q0, q1],
                     [-q3, q2, -q1, q0]])
 
 
def h2_initial_momenta(mass, K_vib, K_rot, theta):
    """rotation : +-y, +-z,  vibration : +-x (along the bond),  total momentum = 0"""
    p_vib = np.sqrt(K_vib * CmInv * mass[0])
    p_rot = np.sqrt(K_rot * CmInv * mass[0])
    p_rot_y = p_rot * np.cos(theta)
    p_rot_z = p_rot * np.sin(theta)
    return np.array([[-p_vib, -p_rot_y, -p_rot_z], [p_vib, p_rot_y, p_rot_z]])
