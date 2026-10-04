import numpy as np

from .units import HBar

TOPOLOGIES = ("R", "R+", "S1", "I-", "I+")

def _overlap_r1(posi_ref, mome_ref, posi, mome, gamma):
    dq = posi_ref - posi
    dp = mome_ref - mome
    return np.exp(-0.25 * gamma * dq**2 - 0.25 / (gamma * HBar**2) * dp**2
                  + 0.5j / HBar * (mome_ref + mome) * dq)

def overlap_1d(topology, posi_ref, mome_ref, posi, mome, gamma, n_image=3):
    """<q_ref, p_ref | q, p> for 1D topology

    topology : topology of the 1D coordinate
    posi_ref : reference position
    mome_ref : reference momentum
    posi     : position
    mome     : momentum
    gamma    : width parameter of the coherent state basis
    n_image  : number of images to consider for periodic boundary conditions
    """
    posi = np.asarray(posi, dtype=float)
    if topology == "R":
        return _overlap_r1(posi_ref, mome_ref, posi, mome, gamma)
    if topology == "R+":
        return _overlap_r1(posi_ref, mome_ref, posi, mome, gamma) \
               - _overlap_r1(-posi_ref, -mome_ref, posi, mome, gamma)

    shifts = 2.0 * np.pi * np.arange(-n_image, n_image + 1).reshape((-1,) + (1,) * posi.ndim)
    direct = np.sum(_overlap_r1(posi_ref + shifts, mome_ref, posi, mome, gamma), axis=0)

    if topology == "S1":
        return direct

    mirror = np.sum(_overlap_r1(-posi_ref + shifts, -mome_ref, posi, mome, gamma), axis=0)
    if topology == "I-":
        return direct - mirror
    if topology == "I+":
        return direct + mirror

    raise ValueError(f"Unknown topology: {topology}")

def get_overlap(topologies, posi_ref, mome_ref, posi, mome, gammas, n_image=3):
    """<q_ref, p_ref | q, p> for multiple 1D topologies

    topologies : list of topologies of the 1D coordinates
    posi_ref   : reference positions (m,)
    mome_ref   : reference momenta (m,)
    posi       : positions (T, m)
    mome       : momenta (T, m)
    gammas      : width parameter of the coherent state basis (m,)
    n_image    : number of images to consider for periodic boundary conditions
    """

    posi = np.asarray(posi, dtype=float)
    mome = np.asarray(mome, dtype=float)
    res = np.ones(posi.shape[0], dtype=complex)

    for i, topology in enumerate(topologies):
        res *= overlap_1d(topology, posi_ref[i], mome_ref[i], posi[:, i], mome[:, i], gammas[i], n_image)

    return res
