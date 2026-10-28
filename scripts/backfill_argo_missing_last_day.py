#!/usr/bin/env python3
"""Backfill the missing last-calendar-day ARGO profiles into existing monthly
index caches under ``dc2_output/catalogs/argo_index/``.

Root cause (see /memories/debugging.md "DC2 argo_profiles ... missing day"
entry): ``dctools/data/connection/argo_data.py::ArgoInterface.
_get_files_for_month()`` used to compute its query window as
``[month_start, MonthEnd(0)]`` (i.e. midnight of the month's own last
calendar day). argopy's ``IndexFetcher(...).region(box)`` treats that upper
bound as EXCLUSIVE, so every profile actually dated on the last day of the
month was silently dropped from every monthly index ever built before the
fix. The fix itself (``end = month_start + DateOffset(months=1)``) only
affects *future* index builds -- every ``dc2_output/catalogs/argo_index/
YYYY_MM.json.zst`` already on disk (built before the fix) is missing that
one day's profiles.

Rather than rebuilding each month from scratch (which would re-download and
re-index ~13-14k profiles/month), this script queries ONLY the missing
last-day window per month (~400-500 profiles/month), downloads + Kerchunk-
indexes just those, and merges them into the existing cached index in
place. ``master_index.json`` is refreshed for every touched month.

Usage:
    python3 scripts/backfill_argo_missing_last_day.py --dry-run
    python3 scripts/backfill_argo_missing_last_day.py --months 2023_01
    python3 scripts/backfill_argo_missing_last_day.py --workers 8
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ujson
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dctools.data.connection.argo_data import ArgoInterface, IndexFetcher  # noqa: E402
import xarray as xr  # noqa: E402
import fsspec  # noqa: E402

INDEX_DIR = Path(__file__).resolve().parents[1] / "dc2_output" / "catalogs" / "argo_index"
VARIABLES = ["PSAL", "TEMP", "PRES"]
MONTH_RE = re.compile(r"^(\d{4})_(\d{2})\.json\.zst$")


def get_missing_day_files(year: int, month: int) -> tuple[str, str, list[str]]:
    """Query argopy for the (previously excluded) last calendar day of the month."""
    month_start = pd.Timestamp(year=year, month=month, day=1)
    next_month_start = month_start + pd.DateOffset(months=1)
    last_day_start = next_month_start - pd.Timedelta(days=1)
    start = last_day_start.strftime("%Y-%m-%d")
    end = next_month_start.strftime("%Y-%m-%d")

    candidate_boxes = [
        [-180, 180, -90, 90, start, end],
        [-180, 180, -90, 90, 0, 6000, start, end],
    ]
    last_exc = None
    idx = None
    for box in candidate_boxes:
        try:
            idx = IndexFetcher(src="gdac").region(box).to_dataframe()
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    if idx is None:
        raise RuntimeError(f"Could not query ARGO index for {start}..{end}: {last_exc}")
    if "file" not in idx.columns:
        return start, end, []
    return start, end, idx["file"].dropna().unique().tolist()


def _extract_time(ref: dict):
    """Mirror ArgoInterface.build_month's inline _extract_time logic."""
    try:
        fs_ref = fsspec.filesystem("reference", fo=ref)
        mapper = fs_ref.get_mapper("")
        ds_tmp = xr.open_dataset(mapper, engine="zarr", consolidated=False)
        time_var = None
        for name in ("TIME", "JULD", "time", "juld"):
            if name in ds_tmp.variables:
                time_var = ds_tmp[name]
                break
        if time_var is None:
            for var in ds_tmp.variables.values():
                if str(var.attrs.get("standard_name", "")).lower() == "time":
                    time_var = var
                    break
        if time_var is None:
            ds_tmp.close()
            return pd.NaT

        values = time_var.values
        if np.issubdtype(getattr(values, "dtype", None), np.datetime64):
            t = pd.to_datetime(values)
        else:
            units = time_var.attrs.get("units")
            calendar = time_var.attrs.get("calendar", "standard")
            if units:
                try:
                    decoded = xr.coding.times.decode_cf_datetime(values, units, calendar)
                    t = pd.to_datetime(decoded)
                except Exception:
                    t = pd.to_datetime(values, errors="coerce")
            else:
                t = pd.to_datetime(values, errors="coerce")
        ds_tmp.close()
        t_flat = np.ravel(t)
        if t_flat.size == 0:
            return pd.NaT
        return t_flat[0]
    except Exception:
        return pd.NaT


def backfill_month(argo: ArgoInterface, year: int, month: int, temp_dir: Path, workers: int, dry_run: bool) -> str:
    key = f"{year}_{month:02d}"
    month_path = INDEX_DIR / f"{key}.json.zst"
    if not month_path.exists():
        return f"SKIP {key}: no existing index file"

    start, end, files = get_missing_day_files(year, month)
    if not files:
        return f"OK {key}: 0 missing-day files found for [{start}, {end}) -- nothing to do"

    if dry_run:
        return f"DRY-RUN {key}: would backfill {len(files)} profiles for [{start}, {end})"

    ref_dir = temp_dir / key
    ref_dir.mkdir(parents=True, exist_ok=True)
    nc_cache_dir = ref_dir / "nc_cache"

    download_map = argo._batch_download_profiles(
        files, nc_cache_dir, n_workers=workers, desc=f"Download ARGO {key} (last day)"
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    new_refs = []
    n_failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(argo._build_single_ref, f, ref_dir, download_map.get(str(f))): f
            for f in files
        }
        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result is not None:
                    new_refs.append(result)
                else:
                    n_failed += 1
            except Exception:
                n_failed += 1

    import shutil

    shutil.rmtree(nc_cache_dir, ignore_errors=True)

    if not new_refs:
        return f"ERROR {key}: 0/{len(files)} new refs built (all failed)"

    new_times = [_extract_time(r) for r in new_refs]
    valid = [(r, t) for r, t in zip(new_refs, new_times) if t is not None and not pd.isna(t)]
    if not valid:
        return f"ERROR {key}: no valid TIME extracted from {len(new_refs)} new refs"
    new_refs, new_times = zip(*valid, strict=False)

    # --- load + decompress existing month index ---
    with open(month_path, "rb") as f:
        existing = ujson.loads(zstd.ZstdDecompressor().decompress(f.read()))
    old_refs = existing["profile_refs"]
    old_epoch = existing["temporal_index"]["sorted_times_epoch"]
    old_n = len(old_refs)

    # Deduplicate against already-present profiles (by exact timestamp, ns
    # resolution) so this script is safely re-runnable/idempotent -- a
    # profile whose timestamp already exists in the cached index is treated
    # as already backfilled and is not re-added.
    old_epoch_set = set(int(e) for e in old_epoch)
    new_epoch_all = pd.DatetimeIndex(new_times).values.view(np.int64)
    dedup_refs = []
    dedup_times = []
    n_duplicate = 0
    for r, t, e in zip(new_refs, new_times, new_epoch_all):
        if int(e) in old_epoch_set:
            n_duplicate += 1
            continue
        dedup_refs.append(r)
        dedup_times.append(t)

    if not dedup_refs:
        return (
            f"OK {key}: 0 new profiles after dedup "
            f"({len(new_refs)} downloaded, all already present) -- nothing to merge"
        )

    all_refs = list(old_refs) + dedup_refs
    all_times = pd.DatetimeIndex(pd.to_datetime(old_epoch)).append(pd.DatetimeIndex(dedup_times))
    order = np.argsort(all_times.values)
    sorted_refs = [all_refs[i] for i in order]
    sorted_epoch = all_times.values[order].view(np.int64)

    merged = {
        "profile_refs": sorted_refs,
        "temporal_index": {"sorted_times_epoch": sorted_epoch.tolist()},
        "metadata": {
            "year": year,
            "month": month,
            "n_prof": len(sorted_refs),
            "variables": VARIABLES,
            "start": int(sorted_epoch[0]),
            "end": int(sorted_epoch[-1]),
        },
    }

    cctx = zstd.ZstdCompressor()
    with open(month_path, "wb") as f:
        f.write(cctx.compress(ujson.dumps(merged).encode()))

    argo._update_master_index(year, month, merged["metadata"])

    new_end = pd.Timestamp(merged["metadata"]["end"])
    return (
        f"OK {key}: added {len(dedup_refs)}/{len(files)} profiles "
        f"({old_n} -> {len(sorted_refs)} total), n_failed={n_failed}, "
        f"n_duplicate={n_duplicate}, new_end={new_end}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--months", help="comma-separated YYYY_MM to process (default: all files in argo_index/)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temp-dir", default="/tmp/argo_backfill_refs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.months:
        keys = [m.strip() for m in args.months.split(",")]
    else:
        keys = sorted(
            m.group(0)[:-9]
            for m in (MONTH_RE.match(p.name) for p in INDEX_DIR.glob("*.json.zst"))
            if m
        )

    print(f"Months to process: {len(keys)}")

    argo = ArgoInterface(base_path=str(INDEX_DIR), variables=VARIABLES, s3_storage_options={})
    temp_dir = Path(args.temp_dir)

    t0 = time.time()
    n_ok = n_err = 0
    for i, key in enumerate(keys, 1):
        year, month = int(key[:4]), int(key[5:7])
        try:
            result = backfill_month(argo, year, month, temp_dir, args.workers, args.dry_run)
        except Exception as exc:  # noqa: BLE001
            result = f"ERROR {key}: {exc!r}"
        print(f"[{i}/{len(keys)}] {result}", flush=True)
        if result.startswith("OK") or result.startswith("DRY-RUN"):
            n_ok += 1
        else:
            n_err += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s: {n_ok} ok, {n_err} errors")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
