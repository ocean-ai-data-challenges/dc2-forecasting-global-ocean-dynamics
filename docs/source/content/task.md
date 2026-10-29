# Task description

## Overview

Data Challenge 2 (DC2) is an open benchmark for **probabilistic short-term forecasting of global ocean
dynamics**. Participants train a model on historical ocean data and submit 10-day forecasts of the
ocean state, deterministic or as an ensemble. Forecasts are evaluated over 2023–2024 against
independent in-situ and satellite observations and against the GLORYS12 reanalysis; the
{doc}`rules` say which data a forecast may use.

DC2 is part of the [PPR Océan & Climat](https://www.ocean-climat.fr/) (*Projet Prioritaire de
Recherche* in French), a national research program launched by the French government and managed
by CNRS and Ifremer to improve understanding of the ocean and climate.

## Goal

Given any input data sources (e.g. reanalysis fields, satellite observations, in-situ profiles),
produce daily global ocean state forecasts at 0.25 ° × 0.25 ° horizontal resolution for lead times
$t = 0, 1, \ldots, H - 1$ days ($H$ = `n_days_forecast`, 10). Five physical variables must be
predicted:

| CF standard name | Short name | Description |
|---|---|---|
| `sea_surface_height_above_geoid` | `zos` | Sea surface height |
| `sea_water_potential_temperature` | `thetao` | Ocean temperature |
| `sea_water_salinity` | `so` | Ocean salinity |
| `eastward_sea_water_velocity` | `uo` | Eastward current |
| `northward_sea_water_velocity` | `vo` | Northward current |

3-D variables (`thetao`, `so`, `uo`, `vo`) must be provided on 21 of the GLORYS12 depth
levels, from ~0.5 m to ~5 275 m. The 2-D variable `zos` is surface-only. An ensemble adds a
`member` dimension (see {doc}`submission`).

## Evaluation setup

Predictions are launched every **7 days** (evaluation interval) throughout the configured period.
Each forecast covers **10 days** of lead time. The evaluation pipeline:

1. Downloads or reads the submitted forecast for each initialization date.
2. Interpolates predicted fields to the space-time locations of each observation dataset.
3. Computes the scores (see {doc}`metrics`): RMSD, bias and centred RMSD, and for an ensemble
   CRPS, spread and spread-skill ratio.
4. Aggregates scores per variable, depth level, and lead time and publishes them on the
   [leaderboard](leaderboard.md).

## Spatial domain

The target grid covers the global ocean:

- **Latitude:** −78 ° to +90 ° (step 0.25 °, 672 points)
- **Longitude:** −180 ° to +180 ° (step 0.25 °, 1 440 points)
- **Depth levels (21):** 0.49, 47.4, 92.3, 155.9, 222.5, 318.1, 380.2, 453.9, 541.1, 643.6,
  763.3, 902.3, 1 245.3, 1 684.3, 2 225.1, 3 220.8, 3 597.0, 3 992.5, 4 405.2, 4 833.3,
  5 274.8 m

## Baselines — GloNet and GLO4ENS

- **GloNet** (*Global Neural Ocean Forecasting System*, El Aouni et al. 2025), a deep-learning
  model of Mercator Ocean International: daily deterministic forecasts at 0.25 °. Its training had
  no data of 2024 (it saw 2023): it is in the 2024 track, the leaderboard's reference for that
  year.
- **GLO4ENS**, Mercator Ocean's operational ensemble forecasts (NEMO at 1/4 °, 50 members),
  over 2023–2024: the probabilistic baseline, and the reference of the 2023–2024 view of the
  leaderboard.

See {doc}`data` for their declarations.
