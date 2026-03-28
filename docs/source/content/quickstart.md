# Quickstart

This page gets you from zero to a validated evaluation run in a few minutes.

---

## 1. Installation

Choose one of three options:

### Option A — Local installation (conda + Poetry)

Install [Poetry](https://python-poetry.org/) (e.g. via [pipx](https://pipx.pypa.io/):
`pipx install poetry`), then:

```bash
# Clone the repository
git clone https://github.com/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics.git
cd dc2-forecasting-global-ocean-dynamics

# Create a conda environment and install ESMF (not supported by Poetry)
conda create --name dc2 python=3.11
conda activate dc2
conda install -c conda-forge esmf esmpy

# Install project dependencies
poetry lock
poetry install

# (Optional) Install dev dependencies (pytest, ruff, mypy, poethepoet…)
poetry install --with dev
```

### Option B — Docker

A pre-built Docker image includes all dependencies:

```bash
# Console mode
docker run -it --rm --name dc2 \
    ghcr.io/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics:latest bash

# JupyterLab mode
docker run --rm -p 8888:8888 --name dc2-lab \
    ghcr.io/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics:latest
```

See [`docker/`](https://github.com/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics/tree/main/docker)
for build and publish instructions.

### Option C — EDITO Datalab

A ready-to-use environment is available on the EDITO Datalab platform (no local
installation required):

> <https://datalab.dive.edito.eu/launcher/service-playground/dc2-forecasting-global-ocean-dynamics>

Open a terminal inside the service and run the commands described below.

For a step-by-step walkthrough, see the
[Hands-on Demo Manual](https://oceanbench2025.sciencesconf.org/resource/page/id/2).

---

## 2. Generate a sample submission

The script `scripts/create_sample_submission.py` creates a compliant dataset filled with
random noise, useful for testing the pipeline before having a real model:

```bash
python scripts/create_sample_submission.py \
    --output /tmp/sample_model \
    --variables zos thetao so uo vo \
    --seed 42
```

This generates the 52 Zarr files corresponding to the evaluation period 2024-01-01 →
2025-01-01 (one per week). Each store conforms to the DC2 grid described in
{doc}`submission`.

---

## 3. Validate the format

Before running the full evaluation, verify that the submission is correctly formatted:

```bash
python dc2/submit.py validate /tmp/sample_model --model-name my_sample
```

A successful run prints a conformance summary. See {doc}`submission` for all validation
options and the full grid specification.

---

## 4. Run the evaluation

Launch the complete evaluation pipeline:

```bash
python dc2/submit.py run /tmp/sample_model --model-name my_sample
```

This will:
1. Download observation catalogues from the DC2 S3 bucket.
2. Interpolate predictions to observation positions.
3. Compute all metrics (RMSD, geostrophic currents, MLD, Lagrangian, Class 4).
4. Write results to `dc2_output/results/results_my_sample.json`.
5. Generate leaderboard HTML pages.

See {doc}`evaluation` for detailed options and pipeline internals.

---

## 5. Inspect the configuration

The `info` subcommand displays the full specification (grid, variables, metrics,
accepted formats) without running any computation:

```bash
python dc2/submit.py info --config dc2
```

---

## Next steps

- {doc}`evaluation` — detailed evaluation options and pipeline steps
- {doc}`submission` — submission format, file layouts, and leaderboard participation
- {doc}`notebooks` — interactive Jupyter notebooks (quickstart, submission workflow)
- {doc}`../content/metrics` — explanation of every metric
- {doc}`../content/data` — datasets used for evaluation
