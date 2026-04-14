#!/usr/bin/env python3
"""Test: run the Class4 metric inside a Dask distributed worker thread.

Same as repro_hang.py but submits the metric computation as a Dask task
to reproduce the exact execution context of the real evaluator.
"""
import os
import sys
import time
import signal
import numpy as np
import pandas as pd
import xarray as xr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Setup data (in driver) ──────────────────────────────────────────
print("[1] Loading data…")
obs_zarr = os.path.join(BASE, "dc2_output/results_batches/obs_batch_shared/swot/batch_0/batch_shared.zarr")
time_npy = os.path.join(BASE, "dc2_output/results_batches/obs_batch_shared/swot/batch_0/time_index.npy")
pred_zarr = os.path.join(BASE, "dc2_output/results_batches/pred_prefetch_cache/glonet/20240103.zarr")

# Use a SMALL time window to keep it fast — Jan 6 (680K pts)
t_vals = np.load(time_npy, mmap_mode="r")
t0 = np.datetime64("2024-01-06T00:00:00")
t1 = np.datetime64("2024-01-07T00:00:00")
mask = (t_vals >= t0) & (t_vals <= t1)
indices = np.where(mask)[0]
i0, i1 = int(indices[0]), int(indices[-1]) + 1
print(f"    Jan 6 window: {i1-i0} obs points")

# ── Create Dask cluster (same config as evaluator) ──────────────────
print("[2] Creating LocalCluster (processes=False, 5 workers × 4 threads)…")
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(
    n_workers=5,
    threads_per_worker=4,
    processes=False,
    memory_limit="5GB",
)
client = Client(cluster)
print(f"    Dashboard: {client.dashboard_link}")


def metric_task(obs_zarr_path, time_npy_path, pred_zarr_path,
                i0, i1, chunk_limit=3):
    """This runs inside a Dask worker thread — same as compute_metric."""
    import threading
    import time as _time
    import numpy as np
    import pandas as pd
    import xarray as xr
    import dask
    from loguru import logger

    tid = threading.current_thread().name
    logger.info(f"[worker {tid}] Starting metric task…")

    # Open obs zarr (lazy)
    obs_ds = xr.open_zarr(obs_zarr_path, chunks={})
    obs_slice = obs_ds.isel(n_points=slice(i0, i1))
    obs_renamed = obs_slice.rename({"ssha_filtered": "zos"})

    # Open pred (numpy)
    pred_ds = xr.open_zarr(pred_zarr_path)
    pred_ds = pred_ds[["zos"]]
    valid_time = np.datetime64("2024-01-06T12:00:00")
    pred_slice = pred_ds.sel(time=valid_time, method="nearest").expand_dims("time")

    t0_compute = _time.time()
    with dask.config.set(scheduler="synchronous"):
        pred_np = pred_slice.compute()
    logger.info(f"[worker {tid}] pred computed in {_time.time()-t0_compute:.1f}s")

    from oceanbench.core.class4_metrics.class4_evaluator import (
        xr_to_obs_dataframe,
        interpolate_model_on_obs,
        apply_binning,
    )

    obs_da = obs_renamed["zos"]
    model_da = pred_np["zos"]

    logger.info(f"[worker {tid}] obs: {obs_da.sizes}, model: {model_da.shape}")

    # Stream chunks
    t_start = _time.time()
    n_done = 0

    with dask.config.set(scheduler="synchronous"):
        chunk_gen = xr_to_obs_dataframe(obs_da, include_geometry=False, yield_chunks=True)
        for i, chunk_df in enumerate(chunk_gen):
            if chunk_df.empty:
                continue
            t_chunk = _time.time()
            logger.info(f"[worker {tid}] chunk {i}: {len(chunk_df)} rows, starting interp…")

            chunk_df = chunk_df.copy()
            if "zos" not in chunk_df.columns:
                for col in ["value", "variable"]:
                    if col in chunk_df.columns:
                        chunk_df = chunk_df.rename(columns={col: "zos"})
                        break
            chunk_df = chunk_df.dropna(subset=["zos"])
            if chunk_df.empty:
                continue

            chunk_df = interpolate_model_on_obs(model_da, chunk_df, "zos", method="pyinterp")
            dt = _time.time() - t_chunk
            logger.info(f"[worker {tid}] chunk {i}: done in {dt:.1f}s")

            n_done += 1
            if chunk_limit and n_done >= chunk_limit:
                break

    total = _time.time() - t_start
    return f"OK: {n_done} chunks in {total:.1f}s"


# ── Submit ONE task ─────────────────────────────────────────────────
print("[3] Submitting single metric task to worker…")
future = client.submit(
    metric_task,
    obs_zarr, time_npy, pred_zarr,
    i0, i1, 3,
    pure=False,
)

print("[4] Waiting for result (timeout=120s)…")
try:
    result = future.result(timeout=120)
    print(f"    Result: {result}")
except Exception as e:
    print(f"    FAILED: {e!r}")

# ── Submit 5 tasks in parallel ──────────────────────────────────────
print("\n[5] Submitting 5 tasks in parallel (same data, 3 chunks each)…")
futures = []
for j in range(5):
    f = client.submit(
        metric_task,
        obs_zarr, time_npy, pred_zarr,
        i0, i1, 3,
        pure=False,
    )
    futures.append(f)

print("[6] Waiting for all 5 tasks (timeout=300s)…")
t_par = time.time()
try:
    from dask.distributed import wait
    done, not_done = wait(futures, timeout=300)
    dt = time.time() - t_par
    print(f"    All 5 completed in {dt:.1f}s")
    for j, f in enumerate(futures):
        print(f"    Task {j}: {f.result()}")
except Exception as e:
    dt = time.time() - t_par
    print(f"    FAILED after {dt:.1f}s: {e!r}")
    # Show which completed and which didn't
    for j, f in enumerate(futures):
        print(f"    Task {j}: status={f.status}")

client.close()
cluster.close()
print("\nDone.")
