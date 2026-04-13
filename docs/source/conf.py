# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path
from typing import Union

# ---------------------------------------------------------------------------
# Patch Sphinx _MockObject so that ``MockedType | None`` (PEP 604 union
# syntax) works at runtime.  Without this, dctools modules that use
# ``xr.Dataset | None`` in annotations (without ``from __future__ import
# annotations``) crash during autosummary import.
# ---------------------------------------------------------------------------
from sphinx.ext.autodoc.mock import _MockObject  # noqa: E402

def _mock_or(self, other):
    return Union[self, other] if other is not None else self

_MockObject.__or__ = _mock_or
_MockObject.__ror__ = lambda self, other: Union[other, self]

# Make the dc2 package importable without installing it.
# On ReadTheDocs the full dependency stack is not available, so we rely on
# autodoc_mock_imports below instead of a full `pip install .`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

# Also make dctools importable so autodoc can extract its docstrings.
# Locally it is installed in dc-env; on ReadTheDocs it is pip-installed
# without deps (see .readthedocs.yaml), so heavy transitive dependencies
# are handled via autodoc_mock_imports below.
_DCTOOLS_SRC = _PROJECT_ROOT.parent / 'dc-tools'
if _DCTOOLS_SRC.is_dir():
    sys.path.insert(0, str(_DCTOOLS_SRC))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dc2'
copyright = '2025, Kamel Ait Mohand, Guillermo Cossio'
author = 'Kamel Ait Mohand, Guillermo Cossio'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

myst_enable_extensions = [
    'colon_fence',
]

# myst-nb: do not execute notebooks at build time (they need the full
# scientific stack which is not available on ReadTheDocs).
nb_execution_mode = 'off'

templates_path = ['_templates']
exclude_patterns = []

# Mock heavy / unavailable packages so autodoc can import dc2 on ReadTheDocs
# without needing the full scientific stack installed.
autodoc_mock_imports = [
    # Heavy / unavailable packages mocked for both dc2 and dctools autodoc.
    'argopy',
    'cartopy',
    'cftime',
    'copernicusmarine',
    'dask',
    'dcleaderboard',
    'distributed',
    'fsspec',
    'geopandas',
    'h5py',
    'loguru',
    'matplotlib',
    'netCDF4',
    'numpy',
    'oceanbench',
    'pandas',
    'pangeo_forge_recipes',
    'psutil',
    'pyinterp',
    'pyproj',
    'rich',
    's3fs',
    'scipy',
    'shapely',
    'torch',
    'torchvision',
    'torchgeo',
    'tqdm',
    'ujson',
    'xarray',
    'xbatcher',
    'xskillscore',
    'yaml',
    'zarr',
    'zstandard',
]

# Autodocs/Autosummary config
autodoc_typehints = 'description'

# Stolen from weatherbench2:
# https://stackoverflow.com/a/66295922/809705
autosummary_generate = True

# Always overwrite previously generated stubs so that stale stubs for
# removed dctools submodules do not accumulate across builds.
autosummary_generate_overwrite = True

# Suppress docutils warnings from third-party docstrings (dctools) that we
# cannot fix in this repo.
suppress_warnings = ['docutils']

# ---------------------------------------------------------------------------
# Patch autosummary to tolerate ImportErrors in stale / missing modules.
# dctools is an external package that evolves independently; when a module is
# removed or renamed the corresponding stub becomes stale and crashes the
# build.  This wrapper catches those import errors so the build continues.
# ---------------------------------------------------------------------------
def _patch_autosummary_generate():
    import sphinx.ext.autosummary.generate as _asg
    _orig_generate = _asg.generate_autosummary_docs

    def _tolerant_generate(*args, **kwargs):
        try:
            return _orig_generate(*args, **kwargs)
        except Exception as exc:                       # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "autosummary generation skipped (non-fatal): %s", exc,
            )
    _asg.generate_autosummary_docs = _tolerant_generate

_patch_autosummary_generate()

# MyST Options
# https://myst-parser.readthedocs.io/en/latest/configuration.html

myst_heading_anchors = 2
myst_links_external_new_tab = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# Copy the standalone leaderboard HTML/CSS/JS to the build output so that
# the real leaderboard is served alongside the Sphinx documentation.
html_extra_path = ['_extra']
# html_logo = "_static/wb2-logo-wide.png" # TODO: draw a logo for the DCs
