# Submission format

This page specifies the required format for a DC2 submission.

---

## DC2 grid

Every forecast must be provided on the DC2 global grid:

| Dimension | Values | No. of points |
|---|---|---|
| `lat` | −78 to +90 °, step 0.25 ° | 672 |
| `lon` | −180 to +180 °, step 0.25 ° | 1 440 |
| `depth` (levels) | 0.494 / 47.374 / 92.327 / 155.851 / 222.475 / 318.127 / 380.213 / 453.938 / 541.089 / 643.567 / 763.333 / 902.339 / 1 245.292 / 1 684.284 / 2 225.078 / 3 220.820 / 3 597.032 / 3 992.484 / 4 405.225 / 4 833.291 / 5 274.784 m | 21 |
| `lead_time` | 0, 1, 2, …, 9 (days after initialization) | 10 |

---

## Required variables

| Variable | Dimensions | Shape | Unit | Description |
|---|---|---|---|---|
| `zos` | `(time, lat, lon)` | (10, 672, 1 440) | m | Sea surface height |
| `thetao` | `(time, depth, lat, lon)` | (10, 21, 672, 1 440) | °C | Potential temperature |
| `so` | `(time, depth, lat, lon)` | (10, 21, 672, 1 440) | PSU | Salinity |
| `uo` | `(time, depth, lat, lon)` | (10, 21, 672, 1 440) | m s⁻¹ | Zonal current |
| `vo` | `(time, depth, lat, lon)` | (10, 21, 672, 1 440) | m s⁻¹ | Meridional current |

> The `time` dimension encodes **valid dates** (initialization date + lead-time), not indices.
> CF metadata (`units`, `long_name`) are mandatory for each coordinate.

---

## Accepted variable names (aliases)

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

## Accepted file formats

The `poetry run python dc2/submit.py info` command lists supported formats. Four layouts are recognized:

| Layout | Description | Example |
|---|---|---|
| **A** — directory of Zarr stores per date | *Recommended.* One `.zarr` per initialization date in a directory | `model/20240103.zarr`, `model/20240110.zarr`, … |
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

## Validation

Before running the full evaluation, verify locally that your format is correct:

```bash
poetry run python dc2/submit.py validate <data_path> --model-name <MODEL_NAME> [options]
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
4. **Temporal coverage**: the expected initialization dates are present.
5. **Types and units**: arrays are floating-point and CF units are provided.

---

## Participating in the public leaderboard

To appear on the official leaderboard, contact the DC2 organizers providing:

- the `results_<NAME>.json` file (aggregated scores);
- the `results_<NAME>_per_bins.jsonl.gz` file (spatial bin maps used to render the
  interactive leaderboard maps);
- a brief description of the model and training data used;
- a reference (paper, preprint, GitHub repository).

> **Note**: the `dctools.submission` module (remote submission backend) is under development.
> The current procedure involves running the evaluation locally (recommended:
> `poetry run python dc2/submit.py run`) and manually sending the results to the organizers. Open a
> [GitHub issue](https://github.com/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics/issues)
> for any questions about submission.
