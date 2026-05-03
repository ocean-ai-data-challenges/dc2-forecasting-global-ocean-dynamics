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
poetry run python dc2/evaluate.py --model-name <MODEL_NAME>
```

`evaluate.py` is the lower-level entrypoint. By default:

- output directory is `./dc2_output`
- logfile is `./dc2_output/logs/dc2.log`
- config profile is `dc2_wasabi` (override with `--config_name dc2_edito`)

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

Observation catalogs (SARAL, Jason-3, SWOT, Argo, GLORYS12) are downloaded from the
DC2 Wasabi S3 bucket (`ppr-ocean-climat`). These catalogs specify the space-time positions
of all observations used for scoring.

### 2. Interpolation

Forecast fields are spatially and temporally interpolated to the positions of each reference
dataset using **`pyinterp`** (bilinear interpolation, ±12 h temporal window).

### 3. Metric computation

The following metrics are computed (see {doc}`metrics` for full details):

| Metric | Reference datasets | Variables |
|---|---|---|
| RMSD | SARAL, Jason-3, SWOT, Argo, GLORYS12 | all |
| Geostrophic current RMSD | GLORYS12 | `zos` (SSH) |
| Mixed Layer Depth RMSD | GLORYS12 | `thetao`, `so` |
| Lagrangian deviation | GLORYS12 | `uo`, `vo` |
| Class 4 scores | Argo | `thetao`, `so` |

### 4. Output files

Results are written to the output directory (default `dc2_output/`):

| File | Content |
|---|---|
| `results/results_<NAME>.json` | Aggregated scores per variable, depth, and lead time |
| `results/results_<NAME>_per_bins.jsonl.gz` | Spatial bin maps (configurable resolution, default 2°) for leaderboard visualization |
| `leaderboard/*.html` | Rebuilt leaderboard HTML pages |

### 5. Leaderboard

Leaderboard HTML pages are automatically rebuilt from the results files. They include
interactive maps showing spatial RMSD distributions and summary tables comparing all
submitted models against the GloNet baseline.

---

## Configuration profiles

Two YAML configuration files ship with the project, under `dc2/config/`:

| File | S3 backend | Use case |
|---|---|---|
| `dc2_wasabi.yaml` | Wasabi (`s3.eu-west-2.wasabisys.com`) | Default — fast S3 access with credentials |
| `dc2_edito.yaml` | EDITO (`minio.dive.edito.eu`) | Public, credential-free access on the EDITO platform |

Both files share the same grid, variables, depth levels, and metric definitions.
They differ only in S3 connection details and (optionally) parallelism values tuned
for different machines.

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
| Evaluation period | 1 January 2024 – 1 January 2025 |
| Initialization frequency | Every 7 days (52 forecasts) |
| Forecast horizon | 10 days (lead times 0–9) |
| Temporal matching tolerance | ±12 hours |

---

## Interpreting results

The `results_<NAME>.json` file contains scores structured by:

- **Variable** (`zos`, `thetao`, `so`, `uo`, `vo`)
- **Depth level** (21 GLORYS12 standard levels)
- **Lead time** (0–9 days)
- **Reference dataset** (SARAL, Jason-3, SWOT, Argo, GLORYS12)

Lower RMSD values indicate better performance. A submission improves on the baseline if it
achieves lower scores than GloNet for at least one metric/variable combination.
