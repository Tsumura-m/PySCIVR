import math

import scipy.constants as sc
from scipy.constants import physical_constants as pc

# __length__ = 'Bohr'
# __energy__ = 'Hartree'
# __mass__ = 'me'
# __charge__ = 'e'
# __temperature__ = 'K'
# __mol__ = 'mol'

my_units = {"length": "Angs", "energy": "eV", "mass": "amu", "charge": "e", "temperature": "Kelvin", "mol": "mol"}

__length__ = my_units["length"]
__energy__ = my_units["energy"]
__mass__ = my_units["mass"]
__charge__ = my_units["charge"]
__temperature__ = my_units["temperature"]
__mol__ = my_units["mol"]

###################################################################################################

"""
#symbol of units
E...energy
L...Length
C...charge
M...mass
K...temperature
N...amount of substance

T...time (=√(M/E)*L)
I...Electric current (=C/T)
"""


"""physical constants(SI base unit)"""
_const_dict = {}
_const_dict["MPro_SI"] = (sc.m_p, "M/")  #  proton mass
_const_dict["MNeu_SI"] = (sc.m_n, "M/")  #  neutron mass
_const_dict["MEle_SI"] = (sc.m_e, "M/")  #  electron mass
_const_dict["HPlank_SI"] = (sc.Planck, "ET/")  #  Plank constant
_const_dict["HBar_SI"] = (sc.hbar, "ET/")  #  Dirac constant
_const_dict["KB_SI"] = (sc.k, "E/K")  #  Boltzman constant
_const_dict["SpeedOfLight_SI"] = (sc.c, "L/T")  #  speed of light
_const_dict["NAvo_SI"] = (sc.Avogadro, "/N")  #  Avogadro Number
_const_dict["Mu0_SI"] = (sc.mu_0, "ML/TTII")  #  Magnetic constant
_const_dict["Eps0_SI"] = (sc.epsilon_0, "TTTTII/MLLL")  #  Electric constant
_const_dict["E_SI"] = (sc.e, "C/")  #  elementary charge
_const_dict["Sec_SI"] = (1.0, "T/")  #  second
_const_dict["NanoSec_SI"] = (sc.nano, "T/")  #  femtosecond
_const_dict["PicoSec_SI"] = (sc.pico, "T/")  #  femtosecond
_const_dict["FemtoSec_SI"] = (sc.femto, "T/")  #  femtosecond
_const_dict["Debye_SI"] = (1e-21 / sc.c, "CL/")  #  debye

"""Length unit(SI base unit)"""
_length_dict = {}
_length_dict["Metre"] = 1.0  #  meter
_length_dict["nm"] = sc.nano  #  nanometer
_length_dict["cm"] = sc.centi  #  centimeter
_length_dict["Angs"] = sc.angstrom  #  Angstrom
_length_dict["Bohr"] = pc["Bohr radius"][0]  #  Bohr radius

"""Energy unit(SI base unit)"""
_energy_dict = {}
_energy_dict["Joule"] = 1.0  #  Joule
_energy_dict["eV"] = sc.electron_volt  #  electron volt
_energy_dict["Hartree"] = pc["Hartree energy"][0]  #  Hartree energy
_energy_dict["Ry"] = pc["Rydberg constant times hc in J"][0]  #  Rydberg energy
_energy_dict["CmInv"] = pc["inverse meter-joule relationship"][0] * 100  #  inverse centimeter
_energy_dict["Hz"] = pc["hertz-joule relationship"][0]  #  Hz
_energy_dict["kcal"] = sc.calorie * 1000

"""Mass unit(SI base unit)"""
_mass_dict = {}
_mass_dict["kg"] = 1.0  #  kilogram
_mass_dict["gram"] = sc.gram  #  gram
_mass_dict["amu"] = sc.atomic_mass  #  atomic mass unit
_mass_dict["me"] = sc.m_e  #  electron mass

"""Charge unit(SI base unit)"""
_charge_dict = {}
_charge_dict["C"] = 1.0  #  Coulomb
_charge_dict["e"] = pc["elementary charge"][0]  #  elementary charge

"""temperatue unit(SI base unit)"""
_temper_dict = {}
_temper_dict["Kelvin"] = 1.0  #  Kelvin

"""amount of substance unit(SI base unit)"""
_mol_dict = {}
_mol_dict["mol"] = 1.0  #  mol


def _construct_unit():
    length_base = _length_dict[__length__]
    energy_base = _energy_dict[__energy__]
    mass_base = _mass_dict[__mass__]
    charge_base = _charge_dict[__charge__]

    for key in _length_dict.keys():
        _length_dict[key] /= length_base
    for key in _energy_dict.keys():
        _energy_dict[key] /= energy_base
    for key in _mass_dict.keys():
        _mass_dict[key] /= mass_base
    for key in _charge_dict.keys():
        _charge_dict[key] /= charge_base

    globals().update(_length_dict)
    globals().update(_energy_dict)
    globals().update(_mass_dict)
    globals().update(_charge_dict)
    globals().update(_temper_dict)
    globals().update(_mol_dict)

    symbol = dict()
    symbol["E"] = _energy_dict["Joule"]
    symbol["L"] = _length_dict["Metre"]
    symbol["C"] = _charge_dict["C"]
    symbol["M"] = _mass_dict["kg"]
    symbol["K"] = _temper_dict["Kelvin"]
    symbol["N"] = _mol_dict["mol"]

    symbol["T"] = math.sqrt(symbol["M"] / symbol["E"]) * symbol["L"]
    symbol["I"] = symbol["C"] / symbol["T"]

    for key, val in _const_dict.items():
        value = val[0]
        units = val[1].split("/")

        globals().update({key: value})

        for char in units[0]:
            value *= symbol[char]
        for char in units[1]:
            value /= symbol[char]
        assert key[-3:] == "_SI", f"key = {key} is not ended with '_SI'"
        globals().update({key[:-3]: value})


_construct_unit()
