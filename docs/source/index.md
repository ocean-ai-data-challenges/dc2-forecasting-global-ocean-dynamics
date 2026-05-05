# DC 2: Forecasting Global Ocean Dynamics

**DC2** is an open benchmark for **probabilistic short-term forecasting of global ocean dynamics**,
part of the [PPR Océan & Climat](https://www.ocean-climat.fr/) national research programme
(CNRS / Ifremer).

Participants train a model on historical ocean data and submit **10-day forecasts** of the upper
ocean state (SSH, temperature, salinity, currents) at 0.25° resolution. Predictions are evaluated
against independent satellite altimetry (SARAL, Jason-3, SWOT), Argo profiles, and the GLORYS12
reanalysis over the period **January 2024 – January 2025**.

::::{grid} 2
:::{grid-item-card} 🌊 Scientific context
:link: content/task
:link-type: doc

Task definition, variables, evaluation setup and reference model (GloNet).
:::

:::{grid-item-card} 📊 Datasets
:link: content/data
:link-type: doc

Training data (free choice), evaluation observations, and the GLORYS12 reference.
:::

:::{grid-item-card} 📐 Metrics
:link: content/metrics
:link-type: doc

RMSD, geostrophic current RMSD, MLD RMSD, Lagrangian deviation, Class 4 scores.
:::

:::{grid-item-card} 🏆 Leaderboard
:link: content/leaderboard
:link-type: doc

Live leaderboard with interactive maps comparing submitted models.
:::
::::

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics.git
cd dc2-forecasting-global-ocean-dynamics
conda create --name dc2 python=3.11 && conda activate dc2
conda install -c conda-forge esmf esmpy
poetry install

# 2. Generate a sample submission
poetry run python scripts/create_sample_submission.py --output /tmp/sample_model

# 3. Validate and evaluate
poetry run python dc2/submit.py validate /tmp/sample_model --model-name my_model
poetry run python dc2/submit.py run /tmp/sample_model --model-name my_model --data-directory ./dc2_output
```

For detailed installation options (Docker, EDITO Datalab), see the
{doc}`Quickstart guide <content/quickstart>`.

```{toctree}
:maxdepth: 2
:caption: Getting started

content/quickstart.md
content/evaluation.md
content/submission.md
```

```{toctree}
:maxdepth: 2
:caption: Notebooks

content/notebooks.md
notebooks/evaluation_quickstart
notebooks/submit
```

```{toctree}
:maxdepth: 2
:caption: DC2 Challenge

content/task.md
content/data.md
content/metrics.md
content/leaderboard.md
content/references.md
```

```{toctree}
:maxdepth: 2
:caption: API Reference

content/api
content/dctools_api
```
