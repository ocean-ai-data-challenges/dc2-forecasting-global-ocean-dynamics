# DC2 — Forecasting Global Ocean Dynamics

[![Documentation](https://readthedocs.org/projects/dc2-forecasting-global-ocean-dynamics/badge/?version=latest)](https://dc2-forecasting-global-ocean-dynamics.readthedocs.io/en/latest/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Data Challenge 2 (DC2)** is an open benchmark for **probabilistic short-term forecasting of
global ocean dynamics**. Participants produce 10-day forecasts of the upper ocean state on a
0.25° global grid, and predictions are evaluated against independent in-situ and satellite
observations over the period **January 2024 – January 2025**.

DC2 is part of the [PPR Océan & Climat](https://www.ocean-climat.fr/) (*Projet Prioritaire de
Recherche*), a national programme launched by the French government and managed by CNRS and
Ifremer.

---

## Scientific overview

### Task

Given any set of input data (reanalysis, satellite observations, in-situ profiles…), produce
daily global ocean state forecasts for lead times 0–9 days. Five physical variables must be
predicted:

| Variable | Description | Dimensions |
|---|---|---|
| `zos` | Sea surface height | 2-D (surface) |
| `thetao` | Ocean temperature | 3-D (21 depth levels, 0.5 m → 5 275 m) |
| `so` | Ocean salinity | 3-D |
| `uo` | Eastward current | 3-D |
| `vo` | Northward current | 3-D |

### Evaluation data

Forecasts are evaluated against five independent reference datasets:

- **GLORYS12** — Global ocean reanalysis (3-D ground truth)
- **SARAL/AltiKa** — Ka-band along-track altimetry (SSH)
- **Jason-3** — Ku-band along-track altimetry (SSH)
- **SWOT** — Wide-swath 2-D radar interferometry (SSH)
- **Argo floats** — In-situ temperature & salinity profiles (3-D)

### Metrics

- **RMSD** (Root Mean Square Deviation) against all reference datasets
- **Geostrophic current RMSD** derived from the SSH gradient
- **Mixed Layer Depth RMSD** diagnosed from T/S profiles
- **Lagrangian trajectory deviation** (virtual particle advection)
- **Class 4 intercomparison** (GODAE/Copernicus framework: RMSE, bias, MAE…)

### Reference model

The baseline is **GloNet** (Global Neural Ocean Forecasting System), developed by Mercator Ocean
International within the PPR Océan & Climat framework.

> For a detailed description of the task, data, and metrics, see the
> [scientific documentation](https://github.com/ocean-ai-data-challenges/data-challenges-info/tree/main/DC2).

---

## Getting started

### Option A — Local installation (conda + Poetry)

**Prerequisites:** [Conda](https://docs.conda.io/) (or Mamba/Micromamba) and
[Poetry](https://python-poetry.org/) (install via
[pipx](https://pipx.pypa.io/): `pipx install poetry`).

```bash
# 1. Clone the repository
git clone https://github.com/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics.git
cd dc2-forecasting-global-ocean-dynamics

# 2. Create a conda environment and install ESMF (not supported by Poetry)
conda create --name dc2 python=3.11
conda activate dc2
conda install -c conda-forge esmf esmpy

# 3. Install project dependencies with Poetry
poetry lock
poetry install

# 4. (Optional) Install dev dependencies (pytest, ruff, mypy, poethepoet…)
poetry install --with dev
```

**Quick test:**

```bash
poetry run python -c "import dc2; print('dc2 installed successfully')"
```

### Option B — Docker

A pre-built Docker image includes all dependencies (see [`docker/`](docker/) for
build instructions):

```bash
# Console mode
docker run -it --rm --name dc2 \
    ghcr.io/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics:latest bash

# JupyterLab mode
docker run --rm -p 8888:8888 --name dc2-lab \
    ghcr.io/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics:latest
```

Then open the JupyterLab URL printed in the terminal and use the built-in terminal.

### Option C — EDITO Datalab

A ready-to-use environment is available on the **EDITO Datalab** platform (no local
installation required):

> <https://datalab.dive.edito.eu/launcher/service-playground/dc2-forecasting-global-ocean-dynamics>

Open a terminal inside the service and run the evaluation commands described below.

For more details on all installation options, see the
[Hands-on Demo Manual](https://oceanbench2025.sciencesconf.org/resource/page/id/2).

---

## Evaluating a new model

### 1. Generate or prepare your forecast

Your forecast must conform to the DC2 grid (0.25° × 0.25°, 21 depth levels, 10 lead-time
days). The recommended format is **one Zarr store per initialisation date** (Layout A):

```
my_model/
    2024-01-03.zarr
    2024-01-10.zarr
    ...
    2024-12-25.zarr
```

To generate a compliant sample submission for testing:

```bash
python scripts/create_sample_submission.py \
    --output /tmp/sample_model \
    --variables zos thetao so uo vo \
    --seed 42
```

### 2. Validate the submission format

```bash
python dc2/submit.py validate /path/to/my_model --model-name MyModel
```

This checks variable presence, grid conformity, NaN fraction (< 10%), temporal coverage,
and CF metadata. Add `--quick` for a fast check on the first few dates only.

### 3. Run the full evaluation

```bash
python dc2/evaluate.py --model-name MyModel
```

Or, equivalently, via the submit CLI (validate → evaluate → leaderboard):

```bash
python dc2/submit.py run /path/to/my_model --model-name MyModel \
    -d ./dc2_output \
    --team "My Team" \
    --description "Short description of the model"
```

The pipeline will:
1. Download observation catalogues (SARAL, Jason-3, SWOT, Argo, GLORYS12)
2. Interpolate forecast fields to observation positions
3. Compute all metrics (RMSD, geostrophic currents, MLD, Lagrangian, Class 4)
4. Write results to `dc2_output/results/results_<MODEL_NAME>.json` and
   `dc2_output/results/results_<MODEL_NAME>_per_bins.jsonl.gz` (spatial maps)

### 4. Inspect the DC2 specification

```bash
python dc2/submit.py info --config dc2
```

---

## Submitting to the leaderboard

To appear on the official DC2 leaderboard, send the following to the organisers:

1. The **`results_<MODEL_NAME>.json`** file (aggregated scores)
2. The **`results_<MODEL_NAME>_per_bins.jsonl.gz`** file (spatial bin maps, used to
   generate the interactive leaderboard maps)
3. A brief description of the model and training data
4. A reference (paper, preprint, or code repository)

Both result files are produced by the evaluation pipeline in
`dc2_output/results/`.

Open a [GitHub issue](https://github.com/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics/issues)
for any questions about submission.

---

## Notebooks

Interactive Jupyter notebooks are provided in [`notebooks/`](notebooks/) to help you get
started:

| Notebook | Description |
|---|---|
| **[Evaluation Quickstart](notebooks/evaluation_quickstart.ipynb)** | Generate a sample submission, inspect the dataset, validate the format, and visualise existing results (GloNet baseline). Ideal for a first look at the DC2 pipeline. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics/blob/main/notebooks/evaluation_quickstart.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics/main?labpath=notebooks/evaluation_quickstart.ipynb) |
| **[Submission Workflow](notebooks/submit.ipynb)** | End-to-end submission pipeline: prepare predictions (customisable data-loading block with examples), validate, run the full evaluation, analyse results, and submit to the leaderboard. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics/blob/main/notebooks/submit.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics/main?labpath=notebooks/submit.ipynb) |

---

## Documentation

- **Full technical documentation** (task, data, metrics, API reference):
  [dc2-forecasting-global-ocean-dynamics.readthedocs.io](https://dc2-forecasting-global-ocean-dynamics.readthedocs.io)
- **Hands-on Demo Manual** (step-by-step installation & run):
  [oceanbench2025.sciencesconf.org — Demo Manual](https://oceanbench2025.sciencesconf.org/resource/page/id/2)
- **Workshop demo video**: [YouTube](https://www.youtube.com/watch?v=FypzJ_osAp0)

---

## Project structure

```
dc2/                  # Core package
  config/             # YAML configurations (Wasabi S3, EDITO)
  evaluation/         # DC2-specific evaluation logic
  evaluate.py         # CLI: run evaluation
  submit.py           # CLI: validate & submit
scripts/              # Utility scripts (sample submission, etc.)
docker/               # Dockerfile & conda environment
docs/                 # Sphinx documentation (readthedocs)
notebooks/            # Jupyter notebooks (quickstart, submission workflow)
```

---

## What is the PPR Océan & Climat?

A *Priority Research Project* launched by the French government and managed by CNRS and Ifremer,
aiming to structure French research efforts to improve understanding of the ocean and climate.
See the [project website](https://www.ocean-climat.fr/) (in French) for more details.

---

## License

This project is licensed under the [GPL-3.0](LICENSE) license.
