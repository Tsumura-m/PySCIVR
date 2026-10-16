# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PySCIVR'
copyright = '2026, Masaya Tsumura'
author = 'Masaya Tsumura'
release = '0.1.0'

import os, sys
sys.path.insert(0, os.path.abspath('../src'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',   # for automatic documentation generation
    'sphinx.ext.napoleon',  # for Google style docstrings
    'sphinx.ext.mathjax',   # for rendering math equations
    'sphinx.ext.viewcode',  # for adding links to source code
    'sphinx.ext.intersphinx',  # for linking to external documentation
    'myst_nb',  # for Markdown and Jupyter notebooks
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# -- Options for autodoc -------------------------------------------------
autodoc_member_order = 'bysource'  # Order members by the order they appear
autodoc_typehints = 'description'  # Show type hints in the description
autodoc_type_aliases = {
    'ArrayLike': 'numpy.typing.ArrayLike',
    'NDArray': 'numpy.typing.NDArray',
}

# -- Options for myst_nb -------------------------------------------------
nb_execution_mode = "off"
myst_enable_extensions = [
    "dollarmath",  # $...$ and $$...$$
    "amsmath",     # \begin{align} etc.
]
