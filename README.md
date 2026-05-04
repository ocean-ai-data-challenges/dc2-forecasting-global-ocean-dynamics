# DC2 — Forecasting Global Ocean Dynamics

[![Documentation](https://readthedocs.org/projects/dc2-forecasting-global-ocean-dynamics/badge/?version=latest)](https://dc2-forecasting-global-ocean-dynamics.readthedocs.io/en/latest/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Data Challenge 2 (DC2)** is an open benchmark for **probabilistic short-term forecasting of
global ocean dynamics**. Participants submit 10-day global forecasts (0.25° grid), evaluated
against independent in-situ and satellite observations.

DC2 is part of the [PPR Océan & Climat](https://www.ocean-climat.fr/) (*Projet Prioritaire de
Recherche*), coordinated by CNRS and Ifremer.

---

## Forecast Scope

Required forecast variables:

| Variable | Description | Dimensions |
|---|---|---|
| `zos` | Sea surface height | 2-D (surface) |
| `thetao` | Ocean temperature | 3-D (21 depth levels) |
| `so` | Ocean salinity | 3-D |
| `uo` | Eastward current | 3-D |
| `vo` | Northward current | 3-D |

Main reference datasets used for evaluation:

- GLORYS12
- SARAL/AltiKa
- Jason-3
- SWOT
- Argo profiles

---

## Installation

### Option A — Local (Conda + Poetry)

Prerequisites:

- Python `>=3.11,<3.14`
- [Conda](https://docs.conda.io/) or Mamba/Micromamba
- [Poetry](https://python-poetry.org/) (recommended install with `pipx install poetry`)

```bash
git clone https://github.com/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics.git
cd dc2-forecasting-global-ocean-dynamics

# Create environment and install ESMF stack (handled outside Poetry)
conda create --name dc2 python=3.11
conda activate dc2
conda install -c conda-forge esmf esmpy

# Install project dependencies
poetry install

# Optional dev/docs extras
poetry install --with dev,docs
```

Sanity check:

```bash
poetry run python dc2/submit.py info --config dc2
```

### Option B — Docker

```bash
docker run -it --rm --name dc2 \
  ghcr.io/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics:latest bash
```

See [docker/](docker/) for the Dockerfile and environment details.

### Option C — EDITO Datalab

Run directly in the hosted environment:

<https://datalab.dive.edito.eu/launcher/service-playground/dc2-forecasting-global-ocean-dynamics>

---

## Quick Usage

### 1. Generate a sample submission

```bash
poetry run python scripts/create_sample_submission.py \
  --output /tmp/sample_model \
  --variables zos thetao so uo vo \
  --seed 42
```

This script is intended for smoke-testing the pipeline and currently generates a short test period.

### 2. Validate format

```bash
poetry run python dc2/submit.py validate /tmp/sample_model --model-name MyModel
```

Useful validation flags:

- `--quick`
- `--save-report /path/report.json`
- `--max-nan-fraction 0.10`
- `--variables zos thetao` (partial submission)

### 3. Run full pipeline (validate -> evaluate -> leaderboard)

```bash
poetry run python dc2/submit.py run /tmp/sample_model \
  --model-name MyModel \
  --data-directory ./dc2_output \
  --team "My Team" \
  --description "Short model description"
```

Useful run flags:

- `--skip-validation`
- `--quick-validation`
- `--force`

### 4. Optional: run `evaluate.py` directly

`dc2/evaluate.py` is an advanced entrypoint using `dctools` argument parsing.
By default it writes to `dc2_output/` and logs to `dc2_output/logs/dc2.log`.

```bash
poetry run python dc2/evaluate.py --model-name MyModel
```

To switch profile, pass `--config_name dc2_edito` (default is `dc2_wasabi`).

---

## Configuration Profiles

Two DC2 YAML profiles are provided in `dc2/config/`:

| File | Backend | Notes |
|---|---|---|
| `dc2_wasabi.yaml` | Wasabi S3 | Default profile |
| `dc2_edito.yaml` | EDITO S3 (public) | Good default on EDITO platform |

Both define the same challenge grid and metrics, with tunable parallelism and memory settings:

- `parallelism_presets`
- `voluminous_parallelism_presets`
- `restart_workers_per_batch`
- `cleanup_between_batches`
- `resume`

---

## Output Files

Main outputs in `dc2_output/results/`:

- `results_<MODEL_NAME>.json` (aggregated metrics)
- `results_<MODEL_NAME>_per_bins.jsonl.gz` (spatial bins used by leaderboard maps)

The output directory also contains caches and logs (for resumable runs and debugging).

---

## Leaderboard Submission

To submit to the official leaderboard, share with organizers:

1. `results_<MODEL_NAME>.json`
2. `results_<MODEL_NAME>_per_bins.jsonl.gz`
3. Short model description and training data summary
4. A paper/preprint/repository reference

Questions? Open an issue in this repository.

---

## Documentation

- Full docs: <https://dc2-forecasting-global-ocean-dynamics.readthedocs.io>
- Notebooks: [notebooks/](notebooks/)
- Build docs locally:

```bash
poetry install --with docs
poetry run sphinx-build -b html docs/source docs/build/html
```

---

## Repository Layout

```text
dc2/
  config/             # YAML profiles + leaderboard config
  evaluation/         # DC2-specific evaluation logic
  evaluate.py         # Advanced evaluation entrypoint
  submit.py           # validate/run/info CLI entrypoint
scripts/              # helper scripts (sample submission, reproductions)
docs/                 # Sphinx + MyST documentation
notebooks/            # interactive quickstart/submission notebooks
docker/               # Docker setup
```

## What is the PPR Océan & Climat?

A *Priority Research Project* launched by the French government and managed by CNRS and Ifremer,
aiming to structure French research efforts to improve understanding of the ocean and climate.
See the [project website](https://www.ocean-climat.fr/) (in French) for more details.

---

## License

This project is licensed under the [GPL-3.0](LICENSE) license.
