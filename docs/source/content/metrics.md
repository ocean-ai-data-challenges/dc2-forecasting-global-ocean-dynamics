# Evaluation metrics

All metrics are computed by the `dctools` library
([`dctools.metrics`](https://github.com/ocean-ai-data-challenges/dc-tools)), which relies on
the [OceanBench](https://github.com/jejjohnson/oceanbench) backend from Mercator Ocean. The
orchestrating class is `DC2Evaluation` (`dc2/evaluation/evaluation.py`), inherited from
`BaseDCEvaluation` in `dctools`.

---

## Evaluation pipeline

For each initialisation date in the period 2024-01-01 → 2025-01-01 (one forecast every
7 days, i.e. 52 cycles):

1. The submitted model is loaded; its fields are spatially and temporally interpolated to
   the exact positions of each reference dataset using **`pyinterp`** (bilinear interpolation).
2. The **temporal matching window** is ±12 hours around each observation.
3. Metrics are computed per variable, per depth level, and per forecast lead time
   (lead-time 0 to 9 days).
4. Results are aggregated by initialisation date and then published on the leaderboard.

### DC2 variable ↔ OceanBench internal name mapping

| DC2 variable | CF standard name | OceanBench identifier |
|---|---|---|
| `zos` | `sea_surface_height_above_geoid` | `SEA_SURFACE_HEIGHT_ABOVE_GEOID` |
| `thetao` | `sea_water_potential_temperature` | `SEA_WATER_POTENTIAL_TEMPERATURE` |
| `so` | `sea_water_salinity` | `SEA_WATER_SALINITY` |
| `uo` | `eastward_sea_water_velocity` | `EASTWARD_SEA_WATER_VELOCITY` |
| `vo` | `northward_sea_water_velocity` | `NORTHWARD_SEA_WATER_VELOCITY` |

---

## Metrics per reference dataset

The following table summarises the metrics assigned in `dc2/config/dc2_wasabi.yaml`:

| Reference dataset | Applied metric(s) | Evaluated variables |
|---|---|---|
| SARAL/AltiKa | `rmsd` | `zos` (SSH anomaly) |
| Jason-3 | `rmsd` | `zos` (SSH anomaly) |
| SWOT (KaRIn + nadir) | `rmsd` | `zos` (filtered SSH) |
| Argo (profiles `thetao`/`so`) | `rmsd` + `class4` | `thetao`, `so` |
| Argo (velocities `uo`/`vo`) | `rmsd` | `uo`, `vo` |
| GLORYS12 (ground truth) | `rmsd` + `lagrangian` + `rmsd_geostrophic_currents` + `rmsd_mld` | `zos`, `thetao`, `so`, `uo`, `vo` |

---

## 1. RMSD — Root Mean Square Deviation

The central metric of DC2. For each *(forecast, observation)* pair, the predicted field is
interpolated to the observation positions; the RMSD is then:

$$
\text{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \left( \hat{x}_i - x_i \right)^2}
$$

where $\hat{x}_i$ is the predicted value at position $i$ and $x_i$ the observed value.

Two variants coexist in `dctools.metrics.oceanbench_metrics`:

| Case | Function used |
|---|---|
| Reference available (real-time obs) | `func_with_ref: rmsd` (oceanbench) |
| No reference (GLORYS12 comparison) | `func_no_ref: rmsd_of_variables_compared_to_glorys` |

### Spatial RMSD maps by bins

In addition to the global score, the pipeline computes **per-cell RMSD maps** at a configurable
resolution (default `bin_resolution = 4°`). These maps are published on the leaderboard as
interactive visualisations, enabling regional error diagnosis.

---

## 2. Geostrophic current RMSD

Surface geostrophic currents $(u_g, v_g)$ are derived from the SSH field $\eta$ using the
geostrophic balance relations:

$$
u_g = -\frac{g}{f} \frac{\partial \eta}{\partial y}, \qquad
v_g = \frac{g}{f} \frac{\partial \eta}{\partial x}
$$

with $g = 9.81\,\text{m s}^{-2}$ and $f = 2\Omega\sin\phi$ the Coriolis parameter.

This metric is applied to the GLORYS12 reference dataset. The preprocessing function
`preprocess_ref: add_geostrophic_currents` is called before the RMSD computation
(`func_with_ref: rmsd`). When GLORYS12 is not available as a direct reference,
`func_no_ref: rmsd_of_geostrophic_currents_compared_to_glorys` is used.

> **Advantage**: this metric evaluates the quality of the SSH gradient independently of any
> absolute altitude offset, making it sensitive to mesoscale features (eddies, fronts).

---

## 3. Mixed Layer Depth (MLD) RMSD

The mixed layer depth is diagnosed from the predicted temperature and salinity profiles using
a potential density criterion:

$$
\sigma_\theta(z_{\text{MLD}}) - \sigma_\theta(10\,\text{m}) = \Delta\sigma_\theta
= 0.03\,\text{kg m}^{-3}
$$

The function `preprocess_ref: add_mixed_layer_depth` performs this computation before
evaluation (`func_with_ref: rmsd`). In no-reference mode:
`func_no_ref: rmsd_of_mixed_layer_depth_compared_to_glorys`.

This metric tests the model's ability to reproduce the vertical stratification of the ocean,
which is critical for air-sea exchanges, primary production, and cyclone forecasting.

---

## 4. Lagrangian trajectory deviation

Virtual particles are advected by the predicted velocity field $(u, v)$ over the entire
forecast horizon. The Lagrangian deviation is defined as the mean spatial displacement
(in km) between the particle positions from the evaluated model and those from the
GLORYS12 reference:

$$
\delta_L = \frac{1}{N_p} \sum_{p=1}^{N_p} \left\| \mathbf{r}_p^{\text{pred}}(T)
          - \mathbf{r}_p^{\text{ref}}(T) \right\|_2
$$

**Implementation parameters** (from `oceanbench_metrics.py`):

| Parameter | Value |
|---|---|
| Spatial domain (`ZoneCoordinates`) | lat −90 → +90°, lon −180 → +180° (global) |
| Required dimension | `depth` (the method applies only to 3-D velocity fields) |
| Function (with reference) | `deviation_of_lagrangian_trajectories` |
| Function (without reference) | `deviation_of_lagrangian_trajectories_compared_to_glorys` |

> **Applications**: search and rescue at sea, pollutant tracking, ecological connectivity,
> ice drift.

---

## 5. Class 4 score (Argo in-situ)

The **Class 4** metric is a standard in operational ocean forecasting (GODAE / Copernicus
Marine). It directly compares predicted profiles to in-situ Argo profiles using the
`Class4Evaluator` from OceanBench:

```python
ClassEvaluator.run(model_ds, obs_ds, variables=["thetao", "so"])
```

Argo profiles are loaded from the Wasabi S3 catalogue and matched in space-time with
forecasts (window: ±12 h, horizontal distance minimised by `pyinterp`).

The results distinguish:
- **bias** (systematic error);
- **unbiased RMSD** (ubRMSD);
- depth-resolved decomposition (0 – 2 000 m).

---

## Aggregation and leaderboard

Scores are computed for each initialisation date, then aggregated (mean ± standard deviation)
over the entire 2024–2025 period. The leaderboard publishes:

- a global score per metric and per variable;
- spatial RMSD maps per bin (4° × 4° resolution) for each forecast lead time;
- a depth decomposition for `thetao`, `so`, `uo`, `vo`.

See the [leaderboard](leaderboard.md) for current scores and interactive maps.
