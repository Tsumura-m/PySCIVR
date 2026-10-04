"""Fixtures and molecular structures for tests"""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pyscivr.utils import get_geom

# equilibrium geometry of butane
BUTANE_XYZ = """
C -1.782165997551355 0.7046770832233177 -0.35714805624899937
C -0.307337646395355 0.6970983276793177 0.02906807235300065
C 0.307337377499645 -0.6970984889146823 -0.029067048691999352
C 1.782166248497645 -0.7046769327506822 0.35714710091900065
H -2.360922424464355 0.0645086840323177 0.3111270081550006
H -2.205694947563355 1.7082196088783177 -0.31033219262899936
H -1.918495941808355 0.32979988334831767 -1.3731438030169993
H 0.252693249063645 1.3629886221333176 -0.6342421534349993
H -0.18693161525735502 1.0993804742503177 1.0393632762480005
H -0.252692582778355 -1.3629884147086824 0.6342443302950007
H 0.18693000317364497 -1.0993812487946824 -1.0393618525409993
H 2.2056947311936446 -1.7082197386196825 0.3103330163240006
H 1.918497721703645 -0.32979741827468234 1.3731417938860007
H 2.360922020616645 -0.0645103239996823 -0.3111302374959994
"""

BUTANE_BONDS = [(0,1), (1,2), (2,3), (0,4), (0,5), (0,6), (1,7), (1,8), (2,9), (2,10), (3,11), (3,12), (3,13)]
BUTANE_ANGLES = [(0,1,2), (1,2,3), (1,0,4), (1,0,5), (1,0,6), (0,1,7), (0,1,8), (3,2,9), (3,2,10), (2,3,11), (2,3,12), (2,3,13)]
BUTANE_DIHEDRALS = [(2,1,0,4), (2,1,0,5), (2,1,0,6), (4,0,1,7), (4,0,1,8), (11,3,2,9), (11,3,2,10), (1,2,3,11), (1,2,3,12), (1,2,3,13), (0,1,2,3)]

# equilibrium geometry of methane
CH4_XYZ = """
C        -0.000000000000       0.000000000000      -0.000000000000
H         0.627615334142       0.627615334142       0.627615334142
H        -0.627615334142      -0.627615334142       0.627615334142
H         0.627615334142      -0.627615334142      -0.627615334142
H        -0.627615334142       0.627615334142      -0.627615334142
"""

# equilibrium geometry of hydrogen
H2_XYZ = """
H        -0.370801958573       0.000000000000       0.000000000000
H         0.370801958573       0.000000000000       0.000000000000
"""

@pytest.fixture
def rng():
    return np.random.default_rng(0)

@pytest.fixture(scope="session")
def h2():
    mass = np.array([1.008, 1.008])
    r_e = 0.370801958573 * 2  # equilibrium bond length in Angstrom
    posi_ref = np.array([[-r_e / 2, 0.0, 0.0], [r_e / 2, 0.0, 0.0]])
    return {"mass": mass, "r_e": r_e, "posi_ref": posi_ref}

@pytest.fixture(scope="session")
def h2_geom():
    symbols, posi_ref, mass = get_geom(H2_XYZ)
    return {"symbols": symbols, "posi_ref": posi_ref, "mass": mass}

@pytest.fixture(scope="session")
def butane():
    symbols, posi_ref, mass = get_geom(BUTANE_XYZ)
    return {"symbols": symbols, "posi_ref": posi_ref, "mass": mass,
            "bonds": BUTANE_BONDS, "angles": BUTANE_ANGLES, "dihedrals": BUTANE_DIHEDRALS}

@pytest.fixture(scope="session")
def ch4():
    symbols, posi_ref, mass = get_geom(CH4_XYZ)
    return {"symbols": symbols, "posi_ref": posi_ref, "mass": mass}
