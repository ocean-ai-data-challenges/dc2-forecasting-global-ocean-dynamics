# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Make the dc2 package importable without installing it.
# On ReadTheDocs the full dependency stack is not available, so we rely on
# autodoc_mock_imports below instead of a full `pip install .`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dc2'
copyright = '2025, Kamel Ait Mohand, Guillermo Cossio'
author = 'Kamel Ait Mohand, Guillermo Cossio'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

# Mock heavy / unavailable packages so autodoc can import dc2 on ReadTheDocs
# without needing the full scientific stack installed.
autodoc_mock_imports = [
    'dctools',
    'argopy',
    'cartopy',
    'copernicusmarine',
    'dask',
    'distributed',
    'geopandas',
    'h5py',
    'netCDF4',
    'numpy',
    'pandas',
    'pangeo_forge_recipes',
    'pyinterp',
    's3fs',
    'scipy',
    'shapely',
    'torch',
    'torchvision',
    'torchgeo',
    'xarray',
    'xbatcher',
    'xskillscore',
    'zarr',
]

# Autodocs/Autosummary config
autodoc_typehints = 'description'

# Stolen from weatherbench2:
# https://stackoverflow.com/a/66295922/809705
autosummary_generate = True

# MyST Options
# https://myst-parser.readthedocs.io/en/latest/configuration.html

myst_heading_anchors = 2
myst_links_external_new_tab = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# html_logo = "_static/wb2-logo-wide.png" # TODO: draw a logo for the DCs
