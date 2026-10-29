# Evaluation

This page describes the evaluation pipeline in detail: what it does, how to configure it,
and how to interpret the results.

---

## Running the evaluation

### Via the submit CLI (recommended)

```bash
poetry run python dc2/submit.py run <data_path> --model-name <MODEL_NAME> [options]
```

This wraps validation, evaluation, and leaderboard generation in a single command.

### Via evaluate.py (advanced)

```bash
poetry run python dc2/evaluate.py -c <profile or path/to/config.yaml> [-d <output directory>]
```

`evaluate.py` is the lower-level entrypoint: it scores the models of the configuration's
`dataset_references` (GloNet in `dc2_edito`) against their references. By default:

- output directory is `./dc2_output`
- logfile is `./dc2_output/logs/dc2.log`
- config profile is `dc2_edito` (the legacy `dc2_wasabi` with `-c dc2_wasabi`; a YAML path is
  accepted too)

The baselines are re-evaluated with the jobs of `scripts/datarmor/` (see
`dc2/leaderboard_results/README.md`).

---

## Execution options (`submit.py run`)

| Option | Description |
|---|---|
| `-d DIR`, `--data-directory DIR` | Output directory for results and catalogs |
| `--force` | Continue even if validation fails |
| `--skip-validation` | Skip initial validation (not recommended) |
| `--quick-validation` | Run a quick validation before evaluation |
| `--description TEXT` | Short model description |
| `--team TEXT` | Team name |
| `--email TEXT` | Contact email |
| `--url TEXT` | Model URL (paper, code, …) |

---

## Pipeline steps

The evaluation pipeline runs through the following stages:

### 1. Catalog download

Observation catalogs (along-track L3, SWOT, Argo, drifters, SST, SSS, GLORYS12) are downloaded
from the S3 bucket of the profile (EDITO `project-ppr-ocean-ia` by default). These catalogs
specify the space-time positions of all observations used for scoring. The Argo profiles are read
from the Argo GDAC and the SST through the Copernicus Marine API (an account is needed).

### 2. Interpolation

Forecast fields are spatially and temporally interpolated to the positions of each reference
dataset using **`pyinterp`** (bilinear interpolation, ±12 h temporal window).

### 3. Metric computation

The following metrics are computed (see {doc}`metrics` for full details):

| Metric | Reference datasets | Variables |
|---|---|---|
| Class 4 (`rmse`, `bias`, `crmse`, `mae`) | Nadir altimetry L3 (7 missions), SWOT KaRIn | `zos` vs SLA + MDT |
| Class 4, per depth | Argo profiles (standard mode, QC 1-2) | `thetao` (potential), `so` |
| Class 4 | Argo velocities (parking depth), drifters (15 m) | `uo`, `vo` |
| Class 4 (surface) | SST L3S, SSS SMOS L3 | `thetao`, `so` at the first level |
| Area-weighted RMSD, energy by band of scales | GLORYS12 | `zos`, `thetao`, `so`, `uo`, `vo` |
| `crps`, `spread`, `ssr` (and their bias-free `crps_c`, `ssr_c`) | all of the above, ensembles only | all |

### 4. Output files

Results are written to the output directory (default `dc2_output/`):

| File | Content |
|---|---|
| `results/results_<NAME>.json` | Aggregated scores per variable, depth, and lead time |
| `results/results_<NAME>_per_bins.jsonl.gz` | Spatial bin maps (configurable resolution, default 2°) for leaderboard visualization |
| `docs/source/_extra/leaderboard/*.html` (of the repository) | Rebuilt leaderboard pages, with the baselines of `dc2/leaderboard_results/` |

### 5. Leaderboard

Leaderboard HTML pages are automatically rebuilt from the results files. They include
interactive maps of the scores per spatial bin and tables comparing the models, in two views:
2024 (every baseline, GloNet the reference) and 2023–2024 (the models that cover it, GLO4ENS the
reference).

---

## Configuration profiles

Two YAML configuration files ship with the project, under `dc2/config/`:

| File | S3 backend | Use case |
|---|---|---|
| `dc2_wasabi.yaml` | Wasabi (`s3.eu-west-2.wasabisys.com`) | Legacy: 2024, L2 along-track references |
| `dc2_edito.yaml` | EDITO (`minio.dive.edito.eu`) | **Default**: public, credential-free; 2023–2024 |

Both files share the same grid, variables, depth levels, and metric definitions.
They differ in S3 connection details, parallelism tuning, and currently configured
time window.

Current configured period:

- `dc2_wasabi.yaml`: 1 January 2024 – 1 January 2025
- `dc2_edito.yaml`: 1 January 2023 – 31 December 2024 (the evaluation period)

---

## Performance tuning

### Parallelism presets

The YAML configuration defines **two groups of presets**, each with three levels
(`low`, `medium`, `high`). The active level is set via a YAML anchor (`&PARALLEL` or
`&PARALLEL_VOLUMINOUS`) and can be switched by moving the anchor:

```yaml
# Standard datasets (SARAL, Jason-3, Argo, …)
parallelism_presets:
  medium: &PARALLEL                     # ◄ active level
    obs_batch_size: 30                  # observations per evaluation batch
    n_parallel_workers: 6               # Dask workers
    nthreads_per_worker: 2              # threads per worker
    memory_limit_per_worker: "3GB"      # per-worker memory cap
    download_workers: 16                # concurrent prefetch threads

# Heavy datasets (GLORYS gridded, SWOT wide-swath)
voluminous_parallelism_presets:
  medium: &PARALLEL_VOLUMINOUS          # ◄ active level
    obs_batch_size: 24
    n_parallel_workers: 4
    nthreads_per_worker: 2
    memory_limit_per_worker: "4GB"
    download_workers: 4
    gridded_batch_size: 6               # gridded files per batch
```

Each dataset source merges one of these presets via `<<: *PARALLEL` or
`<<: *PARALLEL_VOLUMINOUS` and may override individual keys.

### Dataset-specific overrides

Some datasets need specific tuning because of their data volume:

| Dataset | Key overrides | Rationale |
|---|---|---|
| GLORYS | `nthreads_per_worker: 1`, `download_workers: 6` | Each zarr is ~1.5 GB; two concurrent tasks exceed 4 GB |
| SWOT | `n_parallel_workers: 3`, `nthreads_per_worker: 1`, `memory_limit_per_worker: "6GB"`, `c_lib_threads: 2` | SWOT tasks use 2–3 GB unmanaged RAM (pyinterp/BLAS) |
| Argo profiles | `obs_batch_size: 50` | 520 entries → 11 batches instead of 18 |

### Memory management

Several flags control memory safety:

| Key | Default | Description |
|---|---|---|
| `reduce_precision` | `true` | Store intermediate results in float32 to halve memory |
| `restart_workers_per_batch` | `true` | Restart Dask workers between batches to reclaim leaked memory |
| `cleanup_between_batches` | `true` | Delete prefetched files after each batch to free disk space |
| `max_worker_memory_fraction` | `0.65` | Trigger worker restart when managed memory exceeds this fraction |

### Resuming interrupted runs

Setting `resume: true` enables **checkpoint/resume**: the pipeline checks for
already-completed batch result files on disk and skips them on restart. This is
essential for long-running evaluations (8+ hours) that may be interrupted by OOM
kills or transient network errors.

```yaml
resume: true  # skip already-completed batches on restart
```

### Cluster lifecycle

The Dask cluster is **shut down automatically** after all evaluation batches complete
and before post-processing (results consolidation + leaderboard generation). This
frees worker RAM (typically 18+ GB) so that the driver has enough memory to
decompress and process the `per_bins` results file.

---

## Evaluation period and temporal setup

| Parameter | Value |
|---|---|
| Evaluation period | 1 Jan 2023 – 31 Dec 2024 (`dc2_edito`; the legacy `dc2_wasabi`: 2024) |
| Initialization frequency | Every 7 days |
| Forecast horizon | `n_days_forecast` days (10: lead times 0–9) |
| Temporal matching tolerance | ±12 hours |

---

## Interpreting results

The `results_<NAME>.json` file contains scores structured by:

- **Variable** (`ssh`, `temperature`, `salinity`, `u_current`, `v_current`, `sst`, `sss`)
- **Depth level** (`Surface`, `50m`, `200m`, `550m` for the gridded scores and the Argo profiles;
  `15m` and `1000m` for the drifters and the Argo velocities)
- **Lead time** (0–9 days)
- **Reference dataset** (`nadir_l3`, `swot`, `SST_fields`, `SSS_fields`, `argo_profiles`,
  `argo_velocities`, `drifters_15m`, `glorys`)

An error (rmse, crmse, mae, crps) is better lower; a bias is better closer to 0, a spread-skill or
energy ratio closer to 1. Lead time 0 is reported but does not rank forecasts (rule 3 of the
{doc}`rules`). A forecast is compared with the baselines variable by variable and lead time by
lead time, over the same period: no single score sums it up.
