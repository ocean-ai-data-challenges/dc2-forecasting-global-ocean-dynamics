#!/usr/bin/env python3
"""Minimal reproduction of the SWOT Class4 metric hang.

Runs the Class4 metric computation synchronously (no Dask distributed)
for the batch_0 observation data with the highest obs-count time window
(Jan 3, ~19M observations). Times each step and prints progress.

Usage:
    python scripts/repro_hang.py [--chunk-limit N]

If --chunk-limit N is given, process only the first N observation chunks
(each ~500K points) to keep runtime tractable.
"""
import sys
import os
import time
import signal
import argparse
import numpy as np
import pandas as pd
import xarray as xr

# ── Parse args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--chunk-limit", type=int, default=0,
                    help="Max obs chunks to process (0 = all)")
args = parser.parse_args()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 1. Load observation data (shared zarr, lazily) ──────────────────
obs_zarr = os.path.join(BASE, "dc2_output/results_batches/obs_batch_shared/swot/batch_0/batch_shared.zarr")
time_npy = os.path.join(BASE, "dc2_output/results_batches/obs_batch_shared/swot/batch_0/time_index.npy")

print("[1/5] Opening observation zarr (lazy)…")
obs_ds = xr.open_zarr(obs_zarr, chunks={})
print(f"       Full obs: {obs_ds.sizes}")

# Filter for Jan 3 (the largest time window)
t_vals = np.load(time_npy, mmap_mode="r")
t0 = np.datetime64("2024-01-03T00:00:00")
t1 = np.datetime64("2024-01-04T00:00:00")
mask = (t_vals >= t0) & (t_vals <= t1)
indices = np.where(mask)[0]
i0, i1 = int(indices[0]), int(indices[-1]) + 1
print(f"       Jan 3 window: indices [{i0}:{i1}] -> {i1-i0} points "
      f"(mask matches {mask.sum()} points)")

obs_slice = obs_ds.isel(n_points=slice(i0, i1))
# Keep it dask-backed — same as real evaluator
print(f"       obs_slice: {dict(obs_slice.sizes)}, dask-backed: "
      f"{any(hasattr(obs_slice[v].data, 'dask') for v in obs_slice.data_vars)}")

# ── 2. Load prediction data ────────────────────────────────────────
pred_zarr = os.path.join(BASE, "dc2_output/results_batches/pred_prefetch_cache/glonet/20240103.zarr")
print("[2/5] Loading prediction zarr…")
pred_ds = xr.open_zarr(pred_zarr)
print(f"       pred: {dict(pred_ds.sizes)}, vars={list(pred_ds.data_vars)}")

# Select only zos (SWOT only needs SSH)
if "zos" in pred_ds.data_vars:
    pred_ds = pred_ds[["zos"]]
    print(f"       slimmed to zos only: {dict(pred_ds.sizes)}")

# Select the Jan 3 time step
valid_time = np.datetime64("2024-01-03T12:00:00")
pred_slice = pred_ds.sel(time=valid_time, method="nearest").expand_dims("time")
# Materialize (same as evaluator does for predictions)
t_compute = time.time()
pred_np = pred_slice.compute()
print(f"       pred_np computed in {time.time()-t_compute:.1f}s: "
      f"{dict(pred_np.sizes)}, dtype={pred_np['zos'].dtype}")

# ── 3. Setup Class4Evaluator ───────────────────────────────────────
print("[3/5] Creating Class4Evaluator…")
from oceanbench.core.class4_metrics.class4_evaluator import (
    Class4Evaluator,
    xr_to_obs_dataframe,
    interpolate_model_on_obs,
    apply_binning,
)
evaluator = Class4Evaluator(
    metrics=["rmsd"],
    interpolation_method="pyinterp",
    delta_t=pd.Timedelta("12h"),
    bin_specs={"time": "1D", "lat": 4, "lon": 4},
    apply_qc=False,
)
print("       OK")

# ── 4. Run the metric computation (streaming, synchronous) ─────────
print("[4/5] Running metric computation (synchronous, no Dask)…")
print(f"       chunk_limit={args.chunk_limit or 'all'}")

import dask

var = "ssha_filtered"
# Rename to match prediction variable name (zos)
# Actually, let's check what the evaluator expects
# The real evaluator renames obs vars to match pred vars.
# pred has "zos", obs has "ssha_filtered". The evaluator
# normalizes via get_standardized_var_name. For reproduction,
# just rename.
obs_renamed = obs_slice.rename({"ssha_filtered": "zos"})
pred_model = pred_np

print(f"       obs var: zos, model var: zos")
print(f"       obs points: {obs_renamed.sizes.get('n_points', '?')}")

obs_da = obs_renamed["zos"]
model_da = pred_model["zos"]

# Check if model_da is numpy (should be after .compute())
print(f"       model_da dask: {hasattr(model_da.data, 'dask')}, "
      f"shape={model_da.shape}, dims={model_da.dims}")

t_total_start = time.time()

# Stream chunks — same as Class4Evaluator.run()
chunk_gen = xr_to_obs_dataframe(obs_da, include_geometry=False, yield_chunks=True)

interp_cache = {}
n_chunks = 0
total_interp_s = 0.0
total_chunk_s = 0.0

with dask.config.set(scheduler="synchronous"):
    for i, chunk_df in enumerate(chunk_gen):
        t_chunk = time.time()
        if chunk_df.empty:
            print(f"  chunk {i}: empty, skipping")
            continue

        n_rows = len(chunk_df)
        print(f"  chunk {i}: {n_rows} rows", end="", flush=True)

        # Binning
        chunk_df = chunk_df.copy()
        chunk_df, groupby_cols = apply_binning(chunk_df, evaluator.bin_specs)

        # Drop NaN obs
        if "zos" not in chunk_df.columns:
            for col in ["value", "variable"]:
                if col in chunk_df.columns:
                    chunk_df = chunk_df.rename(columns={col: "zos"})
                    break
        if "zos" not in chunk_df.columns:
            print(f" -> no zos column, skipping")
            continue
        chunk_df = chunk_df.dropna(subset=["zos"])
        if chunk_df.empty:
            print(f" -> all NaN, skipping")
            continue

        # Interpolation (this is the suspected hang point)
        t_interp = time.time()
        print(f", interp...", end="", flush=True)

        # Set a 60s alarm to detect hang
        def _alarm(signum, frame):
            print(f"\n  *** ALARM: interpolation hung after 60s on chunk {i}! ***")
            # Print thread state
            import traceback
            for tid, stack in sys._current_frames().items():
                print(f"\n  Thread {tid}:")
                traceback.print_stack(stack, limit=10)
            os._exit(1)
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(60)

        chunk_df = interpolate_model_on_obs(
            model_da, chunk_df, "zos",
            method="pyinterp", cache=interp_cache,
        )

        signal.alarm(0)  # cancel alarm

        dt_interp = time.time() - t_interp
        total_interp_s += dt_interp

        # Drop NaN model values
        chunk_df = chunk_df.dropna(subset=["zos_model", "zos"])
        dt_total = time.time() - t_chunk
        total_chunk_s += dt_total

        print(f" interp={dt_interp:.1f}s, total={dt_total:.1f}s, "
              f"valid={len(chunk_df)}")

        if len(interp_cache) > 2:
            interp_cache.clear()

        n_chunks += 1
        if args.chunk_limit and n_chunks >= args.chunk_limit:
            print(f"\n  Reached chunk limit ({args.chunk_limit}), stopping.")
            break

dt_overall = time.time() - t_total_start
print(f"\n[5/5] Done: {n_chunks} chunks processed in {dt_overall:.1f}s")
print(f"       Total interp time: {total_interp_s:.1f}s")
print(f"       Total chunk time: {total_chunk_s:.1f}s")
print(f"       Avg interp per chunk: {total_interp_s/max(n_chunks,1):.2f}s")
