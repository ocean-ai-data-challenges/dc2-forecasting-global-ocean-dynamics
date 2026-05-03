# Notebooks

Interactive Jupyter notebooks are provided to help you get started with the DC2
evaluation pipeline. They can be run locally, on Google Colab, or on Binder.

---

## Evaluation Quickstart

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics/blob/main/notebooks/evaluation_quickstart.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics/main?labpath=notebooks/evaluation_quickstart.ipynb)

A quick tour of the DC2 evaluation pipeline. This notebook walks you through:

1. Generating a **sample submission** (random noise conforming to the DC2 grid)
2. **Inspecting** the dataset structure with xarray
3. **Validating** the format against the DC2 specification
4. **Visualizing** existing results (GloNet baseline RMSD by lead time and depth)

Best for newcomers who want to understand the DC2 pipeline before
preparing their own model predictions.

→ {doc}`Full notebook <../notebooks/evaluation_quickstart>`

---

## Submission Workflow

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics/blob/main/notebooks/submit.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics/main?labpath=notebooks/submit.ipynb)

An end-to-end notebook for evaluating **your own model** and submitting results to
the leaderboard. It covers:

1. **Preparing predictions** — a customizable data-loading block with examples
   (local Zarr, single NetCDF, S3 download, Copernicus Marine)
2. **Inspecting** the generated forecast files
3. **Validating** against the DC2 grid specification (coordinates, variables,
   NaN fraction)
4. **Running the full evaluation** — creates a custom YAML config and launches
   `dc2/evaluate.py` against all five reference datasets (GLORYS12, Argo,
   SARAL, Jason-3, SWOT)
5. **Analyzing results** — RMSD tables, lead-time curves, depth profiles,
   cross-reference comparison plots
6. **Submitting to the leaderboard** — checks which files are ready and
   provides step-by-step PR instructions

→ {doc}`Full notebook <../notebooks/submit>`

---

## Running notebooks locally

```bash
# From the repository root (with dc-env activated)
cd notebooks
jupyter lab
```

The notebooks assume the working directory is `notebooks/` and that the project
root is the parent directory. All paths are resolved automatically.
