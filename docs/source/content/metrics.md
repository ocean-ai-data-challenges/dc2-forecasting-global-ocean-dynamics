# Evaluation metrics

All metrics are computed by the `dctools` library
([`dctools.metrics`](https://github.com/ocean-ai-data-challenges/dc-tools)), which relies on
the [OceanBench](https://github.com/mercator-ocean/oceanbench) backend from Mercator Ocean. The
orchestrating class is `DC2Evaluation` (`dc2/evaluation/dc2.py`), inherited from
`BaseDCEvaluation` in `dctools`.

---

## Evaluation pipeline

For each initialization date of the configured period (one forecast every `n_days_interval` =
7 days):

1. The submitted forecast is loaded for each lead time $0 \ldots$ `n_days_forecast` $- 1$ (the
   horizon is set by `n_days_forecast`; `target_time_values: auto` derives the lead times from it).
2. Its fields are interpolated to the positions of each reference dataset with **`pyinterp`**
   (bilinear), within ±12 hours of each observation.
3. Scores are computed per reference, variable, depth level and lead time.
4. They are written to `results_<MODEL>.json` and published on the leaderboard.

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

What `dc2/config/dc2_*.yaml` and `DC2Evaluation.dataset_references` evaluate by default:

| Reference dataset | Scores | Evaluated variables |
|---|---|---|
| Nadir altimetry L3 (7 missions), SWOT KaRIn | Class 4: `rmse`, `mse`, `mae`, `bias`, **`crmse`**, `count` | `zos` vs absolute dynamic topography (SLA + MDT) |
| Argo profiles | Class 4, per depth level | `thetao` (potential temperature), `so` |
| Argo velocities | Class 4 at the point's depth (`1000m`) | `uo`, `vo` |
| Drifters | Class 4 at 15 m (`15m`) | `uo`, `vo` |
| SST L3S, SSS SMOS L3 | Class 4 (`Surface`) | `thetao`, `so` at the first level |
| GLORYS12 | gridded `rmsd`, area-weighted | `zos`, `thetao`, `so`, `uo`, `vo` |

The mixed layer depth is **not** evaluated: no reference for it was found reliable enough.
Geostrophic-current and Lagrangian metrics exist in `dctools` but are not active in DC2.

---

## 1. RMSD — Root Mean Square Deviation

For each *(forecast, reference)* pair, the predicted field is interpolated to the reference
positions; then

$$
\text{RMSD} = \sqrt{\frac{\sum_i w_i \left( \hat{x}_i - x_i \right)^2}{\sum_i w_i}}
$$

where $\hat{x}_i$ is the predicted value at position $i$, $x_i$ the reference value and
$w_i = \cos(\text{lat}_i)$ the area a point stands for (on a regular grid, a cell at 60° N is
half the area of one at the equator). The same weights are used against observations and against
GLORYS12, as in `oceanml3d-eval`. Two fields on different grids are an error: they are never
truncated to a common size.

### Pooling over bins

Observation scores are first accumulated per spatial/temporal bin (count, sums of the weighted
errors, squared and absolute errors). The score of a variable is that of **all its points at
once**: the bins' mean squared errors are averaged with each bin's weight, then the root is taken.
It is not the mean of the bins' RMSE, in which a bin of one point weighs as much as a bin of ten
thousand. `count` is the total number of points scored.

### Centred RMSD (`crmse`)

$$
\text{cRMSD} = \sqrt{\text{MSE} - \text{bias}^2}
$$

the error once the mean offset is removed. It is the score to read for the sea surface height: the
model's `zos` and the observations' mean dynamic topography are not referenced to exactly the same
surface, and that constant goes to `bias`, not to the forecast error.

### Spatial maps by bins

The pipeline also writes per-cell RMSD maps at `per_bins_resolution` (2° by default), published on
the leaderboard.

---

## 2. Sea surface height against altimetry

Altimeters measure a sea level **anomaly** (SLA, above a mean sea surface); a model's `zos` is a
height above the geoid. Before scoring, the observation's mean dynamic topography is added back:
ADT = SLA + MDT (`mdt`, `mean_topography` or `data_01__mean_dynamic_topography` must be kept in
the source's `keep_variables`; the evaluation stops otherwise). The references are edited
products (DUACS L3 along-track, SWOT L3): their editing removed the land, ice and rain values. As
a guard, anomalies beyond `max_abs_sla` (2 m by default, a threshold on the observation alone)
are dropped first; it was needed by the former, unedited L2 references.

---

## 3. Argo profiles

Profiles are read from the Argo GDAC in Argo's **standard mode**, with the same rules as
`oceanml3d-eval` (`dctools_core.argo_qc`):

- in delayed or adjusted mode (`DATA_MODE` D or A), the `*_ADJUSTED` values and flags are used;
- a profile with a bad position or date flag is dropped; a level with a bad pressure flag is
  dropped; a bad temperature (salinity) flag blanks that level's temperature (salinity) only —
  flags 1 and 2 are good;
- depth is computed from pressure (TEOS-10, about 1 % shallower: 20 m at 2000 dbar);
- in-situ temperature is converted to potential temperature (TEOS-10), the quantity of `thetao`
  (0.16 °C apart at 2000 m);
- profiles are interpolated onto the model's depth levels; a float's shallowest measurement, when
  at most 10 m deep, stands for the levels above it.

Each depth level is scored separately and reported as `Surface`, `50m`, `200m` and `550m` (the
GLORYS12 levels 0.49, 47.4, 222.5 and 541.1 m), like the gridded scores.

---

## 4. Class 4 metrics (model-vs-observation intercomparison)

**Class 4** is a family of verification metrics defined by the
[GODAE OceanView](https://www.godae-oceanview.org/) / Copernicus Marine intercomparison
framework: gridded model forecasts are evaluated directly against point-wise observations,
the model being interpolated to the observation positions in space and time.

In DC2 it is handled by the `Class4Evaluator` from
[OceanBench](https://github.com/mercator-ocean/oceanbench), driven by `dctools`:

1. **Matching** — model fields are interpolated to each observation location (lat, lon, depth,
   time); a time tolerance (±12 h) controls the matching window.
2. **QC** — see the dataset sections above.
3. **Scores** — accumulated per bin, then pooled (section 1).

---

## 5. Probabilistic (ensemble) forecasts

A forecast with a `member` dimension is an ensemble. It is scored twice, against every reference:

- its **ensemble mean** gets all the scores above (rmsd, bias, crmse…), comparable with the
  deterministic models;
- its **members** get three more scores, pooled over bins like the others:

| Score | Definition | Reads |
|---|---|---|
| `crps` | fair CRPS (Ferro 2014): mean over the points of mean_i \|x_i − y\| − Σ_ij \|x_i − x_j\| / (2M(M−1)) | lower is better; the MAE for one member; unbiased for the ensemble size M |
| `spread` | root of the mean ensemble variance (ddof 1) | the forecast's own error estimate |
| `ssr` | spread-skill ratio √((M+1)/M) · spread / rmse(ensemble mean) (Fortin et al. 2014) | 1 for a reliable ensemble, < 1 under-dispersive, > 1 over-dispersive |
| `crps_c` | `crps` of the members minus the bias | the CRPS once the bias is removed, as `crmse` is the rmse once the bias is removed |
| `ssr_c` | √((M+1)/M) · spread / crmse | the spread-skill ratio once the bias is removed |

The points are weighted by cos(latitude), as in the deterministic scores. Against observations,
each member is interpolated to the observation points; against GLORYS, the scores are computed
at the gridded levels (surface, 50, 200, 550 m). The members are read one level at a time: an
ensemble's size is not bounded by a worker's memory. These are the ensemble scores of
WeatherBench 2 (CRPS, spread-skill), on DC2's observations.

The bias of `crps_c` and `ssr_c` is the `bias` published with the scores: that of the ensemble
mean over the points the scores pool (one forecast, lead time, reference, variable and level),
cos(latitude)-weighted. Against observations it takes a second pass over the points (each
member interpolated again), ~1.5 times the time of the observation scores. All the variables
get both versions; SSH reads the centred ones, the model's and the observations' reference
surfaces differing by a constant (−0.5 m for GloNet and GLO4ENS alike): GLO4ENS against the
L3 along-track, `crps` 0.505 m and `crps_c` 0.023 m, `ssr` 0.05 and `ssr_c` 0.57.

The per-bin maps carry `crps`, `crps_c`, `spread`, `ssr` and `ssr_c` too, against the
observations and against GLORYS (at the four levels of the gridded scores); the centred ratio of
a bin removes the level's bias, as `crps_c` does.

---

## 6. Energy by band of scales (spectra)

The point scores cannot tell a smooth forecast from a sharp one slightly off (the double penalty),
nor realistic members from blurred ones. As WeatherBench 2 with its energy spectra, and
oceanml3d-eval with its bands, the variance of each field is split by scale along longitude,
against GLORYS, at the levels of the gridded scores (`dctools.metrics.spectra`):

1. each latitude row is cut into segments of 64 grid points that are ocean in GLORYS and in every
   member (1780 km at the equator, 890 km at 60°);
2. each segment is detrended, Hann-windowed and Fourier-transformed; its power at each
   wavenumber goes to the band of the wavelength it stands for at its latitude;
3. the band variances are summed over the segments.

| Score (per band: `gt500km`, `200_500km`, `100_200km`, `lt100km`) | Reads |
|---|---|
| `energy_ratio_<band>` | variance of the members (mean over them) over GLORYS's: 1 realistic, < 1 blurred, > 1 noisy. A deterministic forecast is its one member |
| `energy_ratio_mean_<band>` | the same for the ensemble mean (ensembles only); < 1 is expected, the mean smooths what the members disagree on |
| `err_ratio_<band>` | variance of the error of the ensemble mean (of the forecast) over GLORYS's: 0 perfect, 1 no skill at those scales |

On 2024-01-04 (lead 0), surface height, bands > 500 / 200–500 / 100–200 / < 100 km: GLO4ENS
(2 members) energy 1.04 / 1.00 / 0.84 / 0.36, its mean 1.03 / 0.97 / 0.75 / 0.28; GloNet
0.95 / 0.95 / 1.03 / 1.94 (energy GLORYS does not have below 100 km). Error over signal:
GLO4ENS 0.15 / 0.27 / 0.89 / 1.17, GloNet 0.14 / 0.32 / 1.26 / 2.81.

---

## Aggregation and leaderboard

Each `(initialization date, lead time, reference)` gives one set of scores. The leaderboard
averages them over the initialization dates of the period, per lead day, in two views: 2024
(every baseline; GloNet's training had no data of 2024) and 2023–2024 (the models whose forecasts
cover it). Within a view the models are compared over their common period. It publishes:

- a global score per metric and per variable;
- spatial maps of the scores per bin for each lead time and depth level;
- a depth decomposition for `thetao`, `so`, `uo`, `vo`.

See the [leaderboard](leaderboard.md) for current scores and interactive maps.
