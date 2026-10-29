# DC 2: Forecasting Global Ocean Dynamics

**DC2** is an open benchmark for **probabilistic short-term forecasting of global ocean dynamics**,
part of the [PPR Océan & Climat](https://www.ocean-climat.fr/) national research programme
(CNRS / Ifremer).

Participants train a model on historical ocean data and submit **10-day forecasts** of the
ocean state (SSH, temperature, salinity, currents, on 21 levels) at 0.25° resolution,
deterministic or as an ensemble. Forecasts are evaluated over **2023–2024** against independent
observations — the along-track altimetry of seven nadir missions, SWOT, satellite SST and SSS,
Argo profiles and float velocities, drogued drifters — and against the GLORYS12 reanalysis.
Two baselines: **GloNet** (neural, deterministic) and **GLO4ENS** (Mercator's 50-member ensemble).
What a forecast may use is set by the {doc}`data rules <content/rules>`.

::::{grid} 2
:::{grid-item-card} 🌊 Scientific context
:link: content/task
:link-type: doc

Task definition, variables, evaluation setup and the baselines (GloNet, GLO4ENS).
:::

:::{grid-item-card} 📊 Datasets
:link: content/data
:link-type: doc

Training data (free choice), evaluation observations, and the GLORYS12 reference.
:::

:::{grid-item-card} 📐 Metrics
:link: content/metrics
:link-type: doc

Class 4 scores against the observations, RMSD against GLORYS12, CRPS and spread-skill for
ensembles, energy by band of scales.
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
copernicusmarine login   # the SST reference is read through the Copernicus Marine API

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
content/rules.md
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
