Tutorials
=========

These tutorials show the practical usage of PySCIVR step by step.

The first five tutorials, from ``Single Point Calculation`` to ``Spectrum Calculation``, go through the whole procedure of
an SC-IVR calculation: potentials, constrained dynamics, Hessians and monodromy matrices, direct dynamics
in curvilinear coordinates, and spectra. They use simple examples that run in a short time.

The following tutorials, from ``Rotational Spectra of Hydrogen``, compute the spectra of actual molecules with the phase-space sampling,
reproducing the calculations in the paper.


Running the tutorials
---------------------

The tutorials are Jupyter notebooks in ``docs/notebooks``. Move to this directory  and start Jupyter Lab in the virtual environment of PySCIVR:

.. code-block:: bash

   cd docs/notebooks
   uv run --with jupyterlab jupyter lab

Alternatively, open the notebooks in VS Code and select ``.venv/bin/python`` in the repository as the kernel.

Precomputed data, such as the trajectories of butane and the spectra computed in the tutorials,
are available at Zenodo (https://doi.org/10.5281/zenodo.22639777).
Download them as needed and place them in ``docs/notebooks``.


.. toctree::
   :maxdepth: 1
   :caption: Basics

   notebooks/single_point
   notebooks/constrained_dynamics
   notebooks/hessian_and_variation
   notebooks/implicit_verlet
   notebooks/spectrum_calculation

.. toctree::
   :maxdepth: 1
   :caption: Examples

   notebooks/hydrogen_pure_rot
   notebooks/hydrogen_rovib
   notebooks/methane_pure_rot
   notebooks/butane_torsion
