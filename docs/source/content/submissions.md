# Submitting a model

This page describes the complete procedure for formatting a forecast, validating it locally,
and submitting results to DC2.

---

## 1. Prerequisites and installation

Clone the repository and install the package in *editable* mode:

```bash
git clone https://github.com/ppr-ocean-ia/dc2-forecasting-global-ocean-dynamics.git
cd dc2-forecasting-global-ocean-dynamics
pip install -e .
```

The installation provides the `dc-submit` CLI command (also callable via
`python -m dc.submit`).

---

## 2. Required submission format

### 2.1 DC2 grid

Every forecast must be provided on the DC2 global grid:

| Dimension | Values | No. of points |
|---|---|---|
| `lat` | −78 to +90 °, step 0.25 ° | 672 |
| `lon` | −180 to +180 °, step 0.25 ° | 1 440 |
| `depth` (levels) | 0.494 / 47.374 / 92.327 / 155.851 / 222.475 / 318.127 / 380.213 / 453.938 / 541.089 / 643.567 / 763.333 / 902.339 / 1 245.292 / 1 684.284 / 2 225.078 / 3 220.820 / 3 597.032 / 3 992.484 / 4 405.225 / 4 833.291 / 5 274.784 m | 21 |
| `lead_time` | 0, 1, 2, …, 9 (days after initialisation) | 10 |

### 2.2 Required variables

| Variable | Dimensions | Shape | Unit | Description |
|---|---|---|---|---|
| `zos` | `(time, lat, lon)` | (10, 672, 1 440) | m | Sea surface height |
| `thetao` | `(time, depth, lat, lon)` | (10, 21, 672, 1 440) | °C | Potential temperature |
| `so` | `(time, depth, lat, lon)` | (10, 21, 672, 1 440) | PSU | Salinity |
| `uo` | `(time, depth, lat, lon)` | (10, 21, 672, 1 440) | m s⁻¹ | Zonal current |
| `vo` | `(time, depth, lat, lon)` | (10, 21, 672, 1 440) | m s⁻¹ | Meridional current |

> The `time` dimension encodes **valid dates** (initialisation date + lead-time), not indices.
> CF metadata (`units`, `long_name`) are mandatory for each coordinate.

### 2.3 Accepted variable names (aliases)

The validation pipeline accepts common aliases:

| Coordinate | Accepted names |
|---|---|
| latitude | `lat`, `latitude` |
| longitude | `lon`, `longitude` |
| depth | `depth`, `lev` |
| time | `time` |
| SSH | `zos`, `ssh`, `ssha` |
| Temperature | `thetao`, `temperature` |
| Salinity | `so`, `salinity` |
| Zonal current | `uo`, `u` |
| Meridional current | `vo`, `v` |

---

## 3. Accepted file formats

The `dc-submit info` command lists supported formats. Four layouts are recognised:

| Layout | Description | Example |
|---|---|---|
| **A** — directory of Zarr stores per date | *Recommended.* One `.zarr` per initialisation date in a directory | `model/20240103.zarr`, `model/20240110.zarr`, … |
| **B** — single Zarr store | A single Zarr store covering the entire period | `model/all_forecasts.zarr` |
| **C** — single NetCDF file | A single `.nc` or `.nc4` file | `model/forecasts.nc` |
| **D** — glob of NetCDF files | Any path accepted by `glob` | `/data/model/*.nc` |

Layout A is recommended for large submissions as it enables lazy loading via Dask and
better fault tolerance.

### Layout A structure (directory of Zarr stores per date)

```
my_model/
    2024-01-03.zarr
    2024-01-10.zarr
    2024-01-17.zarr
    ...
    2024-12-25.zarr
```

---

## 4. Generating a sample submission

The script `scripts/create_sample_submission.py` creates a compliant dataset filled with
random noise, useful for testing the pipeline before having a real model:

```bash
python scripts/create_sample_submission.py \
    --output /tmp/sample_model \
    --variables zos thetao so uo vo \
    --seed 42
```

This script generates the 52 Zarr files corresponding to the evaluation period 2024-01-01 →
2025-01-01 (one per week). Each store conforms to the DC2 grid described above.

---

## 5. Validating the submission

Before running the full evaluation, verify locally that the format is correct:

```bash
dc-submit validate <data_path> --model-name <MODEL_NAME> [options]
```

### Validation options

| Option | Description |
|---|---|
| `--model-name NAME` | Model identifier *(required)* |
| `--quick` | Validate only the first few dates (quick test) |
| `--save-report PATH` | Save the validation report to a JSON file |
| `--max-nan-fraction F` | Maximum allowed NaN fraction (default: `0.10`, i.e. 10 %) |
| `--variables V [V …]` | Restrict validation to specific variables |
| `--config {dc2,…}` | Configuration profile (default: `dc2`) |

### What the validation checks

1. **Variable presence**: `zos`, `thetao`, `so`, `uo`, `vo` (or a subset if `--variables`
   is specified).
2. **Grid conformity**: lat, lon, depth, and lead_time match the DC2 specification.
3. **NaN fraction**: no variable exceeds `max_nan_fraction` (10 % by default).
4. **Temporal coverage**: the expected initialisation dates are present.
5. **Types and units**: arrays are floating-point and CF units are provided.

---

## 6. Running the full evaluation

```bash
dc-submit run <data_path> --model-name <MODEL_NAME> [options]
```

### Execution options

| Option | Description |
|---|---|
| `-d DIR`, `--data-directory DIR` | Output directory for results and catalogues |
| `--force` | Overwrite existing results without confirmation |
| `--skip-validation` | Skip initial validation (not recommended) |
| `--quick-validation` | Run a quick validation before evaluation |
| `--description TEXT` | Short model description |
| `--team TEXT` | Team name |
| `--email TEXT` | Contact email |
| `--url TEXT` | Model URL (paper, code, …) |

### Pipeline steps

1. **Catalogue download**: observation catalogues (SARAL, Jason-3, SWOT, Argo, GLORYS12)
   are downloaded from the DC2 Wasabi S3 bucket.
2. **Interpolation**: forecast fields are spatially and temporally interpolated to the
   positions of each reference dataset (`pyinterp`, bilinear, ±12 h window).
3. **Metric computation**: RMSD, geostrophic current RMSD, MLD RMSD, Lagrangian deviation,
   Class 4 score (see [metrics](metrics.md)).
4. **Output**: results are written to `<data_directory>/results/results_<NAME>.json`.
5. **Leaderboard**: leaderboard HTML pages are rebuilt in `<data_directory>/leaderboard/`.

---

## 7. Inspecting the specification

The `dc-submit info` subcommand displays the full configuration (grid, variables, metrics,
accepted formats) without running any evaluation:

```bash
dc-submit info --config dc2
```

---

## 8. Participating in the public leaderboard

To appear on the official leaderboard, contact the DC2 organisers providing:

- the `results_<NAME>.json` file generated by `dc-submit run`;
- a brief description of the model and training data used;
- a reference (paper, preprint, GitHub repository).

> **Note**: the `dctools.submission` module (remote submission backend) is under development.
> The current procedure involves running `dc-submit run` locally and manually sending the
> results to the organisers. Open a
> [GitHub issue](https://github.com/ppr-ocean-ia/dc2-forecasting-global-ocean-dynamics/issues)
> for any questions about submission.

Voir aussi [`dc2/submit.py`](https://github.com/ppr-ocean-ia/dc2-forecasting-global-ocean-dynamics/blob/main/dc2/submit.py)
pour le code complet de l'interface CLI.
