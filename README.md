# PySCIVR

PySCIVR is a Python package for wavepacket dynamics with the semiclassical
Herman–Kluk propagator [1] in curvilinear coordinates, based on Ref. [2].

## Features

- **On-the-fly dynamics**: the time evolution is evaluated without constructing
  the potential energy surface or the kinetic energy operator in advance.
- **Robust to coordinate singularities**: dynamics is propagated in redundant
  Cartesian coordinates with constraints, and transformed to curvilinear
  coordinates only when needed.
- **General curvilinear coordinates**: bond lengths, bond angles, torsions,
  Euler angles, normal modes, etc. can be combined freely, and reduced-dimensional
  models are easily constructed by fixing some of them.
- **Power spectra**: time-averaged SC-IVR [3] spectra can be computed for initial
  wavepackets given as direct products of one-dimensional coherent states.

## Examples

The tutorials in [`docs/notebooks`](https://github.com/Tsumura-m/PySCIVR/tree/main/docs/notebooks) reproduce the calculations in Ref. [2]
and more, which are the practical way to learn how to use the package:

- Pure rotation and rovibration of H2 (linear molecule)
- Pure rotation of CH4 (nonlinear molecule)
- Torsional dynamics of butane

## Installation

```bash
git clone https://github.com/Tsumura-m/PySCIVR.git
cd PySCIVR
uv sync
```

The dependencies and the virtual environment are managed by [uv](https://docs.astral.sh/uv/).
Python 3.13 or later is required.

Scripts are run with `uv run`:

```bash
uv run python your_script.py
```

The installation is successful if the following commands give:

```console
$ uv run python --version
Python 3.13.11
$ uv run which python
path/to/PySCIVR/.venv/bin/python
$ uv run python -c "import pyscivr"
$
```

The last command prints nothing if the package is installed correctly.

To change the Python version (3.13 or later), run:

```bash
uv python pin 3.14
uv sync
```

### Jupyter Lab and VS Code

The virtual environment can also be used from Jupyter Lab or VS Code.
To start Jupyter Lab in the environment, run:

```bash
uv run --with jupyterlab jupyter lab
```

In VS Code, select `.venv/bin/python` as the Python interpreter or the notebook kernel.

## Documentation

The API reference and the executed tutorials are available at
[https://tsumura-m.github.io/PySCIVR](https://tsumura-m.github.io/PySCIVR).

The documentation can also be built locally:

```bash
uv run --group docs sphinx-build -b html docs docs/_build/html
```

and opened with `open docs/_build/html/index.html` (macOS) or
`xdg-open docs/_build/html/index.html` (Linux).

## References

[1] M. F. Herman and E. Kluk, "A semiclasical justification for the use of
non-spreading wavepackets in dynamics calculations,"
*Chem. Phys.* **91**, 27–34 (1984).

[2] M. Tsumura and Y. Kurashige, "Semiclassical initial value representation in
curvilinear coordinates," *J. Chem. Phys.* (Under Review);
ChemRxiv, https://doi.org/10.26434/chemrxiv.15008503

[3] Y. Elran and K. G. Kay, "Improving the efficiency of the Herman–Kluk propagator
by time integration," *J. Chem. Phys.* **110**, 3653–3659 (1999).;
A. L. Kaledin and W. H. Miller, "Time averaging the semiclassical
initial value representation for the calculation of vibrational
energy levels. II. Application to H2CO, NH3, CH4, CH2D2,"
*J. Chem. Phys.* **119**, 3078–3084 (2003). 
