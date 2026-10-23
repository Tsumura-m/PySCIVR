.. PySCIVR documentation master file, created by
   sphinx-quickstart on Fri Oct  9 16:43:16 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PySCIVR
=======

PySCIVR is a Python package for wavepacket dynamics with the semiclassical Herman–Kluk propagator
in curvilinear coordinates. The dynamics is propagated on the fly in redundant Cartesian coordinates
with constraints, which is robust to coordinate singularities, and power spectra are computed
by the time-averaged SC-IVR.

Getting started
---------------

1. Install the package following the :doc:`README`.
2. Learn the usage through the :doc:`tutorials`, from the basic procedure to the spectra of actual molecules.
3. See the :doc:`modules` for the details of each function.

The precomputed data for the tutorials are available at Zenodo (https://doi.org/10.5281/zenodo.22639777).

Citation
--------

If you use PySCIVR in your research, please cite:

   M. Tsumura and Y. Kurashige, "On-The-Fly Dynamics of Multiconfigurational Wave Packets by a Semiclassical Method,"
   *J. Chem. Theory Comput.* **21**, 11401-11414 (2025).

   M. Tsumura and Y. Kurashige, "Semiclassical initial value representation in curvilinear coordinates,"
   ChemRxiv (2026), https://doi.org/10.26434/chemrxiv.15008503

.. code-block:: bibtex

   @article{tsumura2025fly,
      title={On-The-Fly Dynamics of Multiconfigurational Wave Packets by a Semiclassical Method},
      author={Tsumura, Masaya and Kurashige, Yuki},
      journal={Journal of Chemical Theory and Computation},
      volume={21},
      number={22},
      pages={11401--11414},
      year={2025},
      publisher={ACS Publications}
   }

   @article{tsumura2026semiclassical,
     author  = {Tsumura, Masaya and Kurashige, Yuki},
     title   = {Semiclassical initial value representation in curvilinear coordinates},
     journal = {ChemRxiv},
     year    = {2026},
     doi     = {10.26434/chemrxiv.15008503},
   }

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   README
   tutorials
   modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
