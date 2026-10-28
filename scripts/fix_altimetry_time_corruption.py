#!/usr/bin/env python3
"""Repair corrupted ``time`` coordinates in jason3 / saral altimetry zarr stores.

Root cause
----------
Each jason3/saral zarr store encodes its ``time`` coordinate as CF
``"nanoseconds since <per-file reference timestamp>"`` with the raw offsets
stored as **float32**. For most files the reference timestamp matches the
file's own real start time (embedded in the filename), so the raw offsets
stay small (0 to a few hundred seconds worth of nanoseconds) and float32's
~7-significant-digit precision is enough to preserve ~1 Hz sample timing.

For a subset of files (1665/2345 sampled jason3 Oct-Dec 2024 files; an
unknown-but-nonzero fraction of saral Nov-Dec 2024 files), the reference
timestamp is instead "stuck" at a fixed sentinel value roughly two years in
the past (jason3: 2022-12-19 23:09:43.764911232; saral: 2022-12-19
23:34:18.116631936). Because the real per-point offsets from that far-away
epoch are enormous (~6e16 ns), float32's precision at that magnitude is only
accurate to within several seconds -- completely swamping the ~1s real
sample spacing. The on-disk raw values end up quantized into a handful of
nearly-identical buckets (observed: a 3316-point file's raw values spanning
only ~3392 ns total, i.e. collapsed to noise). This means the true
sub-second per-point timing is **unrecoverable** for affected files -- it
was destroyed at write time, not merely mislabeled.

Repair strategy
----------------
Since real per-point timing can't be recovered, reconstruct a synthetic but
accurate ``time`` array by linearly interpolating ``n_points`` timestamps
between the file's filename-embedded start and end datetimes (both are
exact to the second). This is a good approximation for altimeter along-track
data (near-constant sample rate along a single pass) and is far more correct
than the current wrong-by-~2-years or wrong-by-~54-years (1970 epoch)
values.

Only the ``time`` array (and its ``.zattrs``) is rewritten in place -- all
other variables/chunks in the store are left completely untouched, and the
store is re-consolidated (``zarr.consolidate_metadata``) afterwards.

Credentials are read the same way as ``convert_glorys_2023_to_2024_schema.py``
(env vars first, then the local ``mc`` alias config).

Usage examples
--------------
    # Dry-run against an explicit list of bad store paths (one per line)
    python3 scripts/fix_altimetry_time_corruption.py --bad-list /tmp/jason3_bad.txt --dry-run

    # Auto-discover + repair all jason3 stores with the stuck epoch signature
    python3 scripts/fix_altimetry_time_corruption.py --dataset jason3 --scan --workers 8

    # Repair a single store (for manual testing)
    python3 scripts/fix_altimetry_time_corruption.py --paths "project-ppr-ocean-ia/DC2/ZARR/Jason3/JA3_..._.zarr"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

import boto3
import numpy as np
import pandas as pd
import s3fs
import zarr

BUCKET = "project-ppr-ocean-ia"
ENDPOINT_URL = "https://minio.dive.edito.eu"

DATASET_PREFIXES = {
    "jason3": "DC2/ZARR/Jason3",
    "saral": "DC2/ZARR/Saral",
    "swot": "DC2/ZARR/Swot",
}

# Known "stuck" sentinel reference epochs that indicate corruption.
# (swot's corruption signature isn't a stuck-epoch string -- both good and
# bad swot files share the same "nanoseconds since 1970-01-01" units -- the
# real tell is the on-disk `time` array dtype: good files store int64,
# corrupted ones store float32. See is_corrupted()/--bad-list usage for
# swot instead of --scan.)
STUCK_EPOCH_SIGNATURES = {
    "jason3": "2022-12-19",
    "saral": "2022-12-19",
}

# jason3/saral filenames: ..._20241027_232615_20241028_002228...
# swot filenames:         ..._20240111T232120_20240112T001246...
FILENAME_TS_RE = re.compile(r"(\d{8})_(\d{6})_(\d{8})_(\d{6})")
FILENAME_TS_RE_T = re.compile(r"(\d{8})T(\d{6})_(\d{8})T(\d{6})")


def get_credentials(mc_config: Path, mc_alias: str) -> tuple[str, str, str | None]:
    key = os.environ.get("GLORYS_S3_ACCESS_KEY")
    secret = os.environ.get("GLORYS_S3_SECRET_KEY")
    token = os.environ.get("GLORYS_S3_SESSION_TOKEN")
    if key and secret:
        return key, secret, token
    cfg = json.loads(mc_config.read_text())
    alias = cfg["aliases"][mc_alias]
    return alias["accessKey"], alias["secretKey"], alias.get("sessiontoken")


def parse_filename_start_end(store_path: str) -> tuple[datetime, datetime]:
    """Extract (start, end) datetimes embedded in the filename.

    Matches patterns like ``..._20241027_232615_20241028_002228...`` used by
    both jason3 and saral filenames.
    """
    name = store_path.rstrip("/").split("/")[-1]
    m = FILENAME_TS_RE.search(name) or FILENAME_TS_RE_T.search(name)
    if not m:
        raise ValueError(f"could not parse start/end timestamps from filename: {name}")
    start_date, start_time, end_date, end_time = m.groups()
    start_dt = datetime.strptime(start_date + start_time, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(end_date + end_time, "%Y%m%d%H%M%S")
    if end_dt < start_dt:
        raise ValueError(f"parsed end < start for {name}: {start_dt} > {end_dt}")
    return start_dt, end_dt


def get_time_zattrs(fs: s3fs.S3FileSystem, store_key: str) -> dict:
    raw = fs.cat(f"{store_key}/.zmetadata")
    meta = json.loads(raw)
    return meta["metadata"].get("time/.zattrs", {})


def is_corrupted(fs: s3fs.S3FileSystem, store_key: str, dataset: str) -> bool:
    try:
        attrs = get_time_zattrs(fs, store_key)
    except Exception:
        return False
    units = attrs.get("units", "")
    return STUCK_EPOCH_SIGNATURES[dataset] in units


def discover_bad_stores(fs: s3fs.S3FileSystem, dataset: str, months: list[str]) -> list[str]:
    prefix = DATASET_PREFIXES[dataset]
    entries = fs.ls(f"{BUCKET}/{prefix}")
    cands = [e for e in entries if any(f"_{m}" in e for m in months)]
    bad = []
    for e in cands:
        if is_corrupted(fs, e, dataset):
            bad.append(e)
    return bad


def _delete_existing_store_objects(fs_write: s3fs.S3FileSystem, prefix_key: str) -> int:
    """Single-object deletes to work around this MinIO endpoint's rejection
    of botocore's batched DeleteObjects call (MissingContentMD5)."""
    client = boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=fs_write.key,
        aws_secret_access_key=fs_write.secret,
        aws_session_token=fs_write.token,
    )
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix_key):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    for k in keys:
        client.delete_object(Bucket=BUCKET, Key=k)
    fs_write.invalidate_cache()
    return len(keys)


def repair_one(fs_read: s3fs.S3FileSystem, fs_write: s3fs.S3FileSystem, store_path: str, dry_run: bool, verify: bool) -> str:
    """Rewrite only the ``time`` array of ``store_path`` in place."""
    store_key = store_path if store_path.startswith(BUCKET) else f"{BUCKET}/{store_path}"

    try:
        start_dt, end_dt = parse_filename_start_end(store_key)
    except ValueError as exc:
        return f"ERROR {store_key}: {exc}"

    read_store = s3fs.S3Map(root=store_key, s3=fs_read, check=False)
    g = zarr.open_group(read_store, mode="r")
    if "time" not in g:
        return f"ERROR {store_key}: no 'time' array in store"
    time_arr = g["time"]
    n = time_arr.shape[0]
    old_attrs = dict(time_arr.attrs)
    old_units = old_attrs.get("units")
    on_disk_dtype = time_arr.dtype

    new_times = pd.date_range(start=start_dt, end=end_dt, periods=n)
    # Use a reference epoch 1 second BEFORE the file's own start time
    # (matching the small-offset convention already used by "good" files in
    # this dataset), so raw offsets stay small and survive the array's
    # existing float32 on-disk dtype without precision loss. Writing large
    # absolute "nanoseconds since 1970" offsets here would recreate the
    # exact same float32-precision-collapse bug this script is fixing.
    # The 1s shift (rather than epoch == start_dt) is deliberate: the zarr
    # array's storage-level fill_value is 0.0, and xarray's mask_and_scale
    # decoding treats any raw value exactly equal to the fill_value as
    # missing (NaN/NaT) REGARDLESS of decode_times -- an offset of exactly
    # 0.0 for the first real point would be silently masked out otherwise.
    epoch = start_dt - timedelta(seconds=1)
    offsets_ns = (new_times - epoch) / np.timedelta64(1, "ns")
    new_values = np.asarray(offsets_ns, dtype=np.float64).astype(on_disk_dtype)
    new_units = f"nanoseconds since {epoch.isoformat(sep=' ')}"

    if dry_run:
        return (
            f"DRY-RUN {store_key}: n={n} old_units={old_units!r} "
            f"new_units={new_units!r} new_range=({new_times[0]} .. {new_times[-1]})"
        )

    write_store = s3fs.S3Map(root=store_key, s3=fs_write, check=False)
    gw = zarr.open_group(write_store, mode="a")
    gw["time"][:] = new_values
    gw["time"].attrs["units"] = new_units
    if "calendar" in old_attrs:
        gw["time"].attrs["calendar"] = old_attrs["calendar"]
    # Some stores (e.g. swot's misplaced "Expert" reprocessed patch files)
    # also carry stale root-level ACDD time_coverage_start/end attrs
    # (leftover from an unrelated template/granule) that dctools'
    # get_time_bound_values() falls back to when the time *array* itself
    # looks corrupted -- fix those too so a fresh catalog build gets the
    # correct date even before picking up the repaired array.
    if "time_coverage_start" in gw.attrs or "time_coverage_end" in gw.attrs:
        gw.attrs["time_coverage_start"] = new_times[0].isoformat() + "Z"
        gw.attrs["time_coverage_end"] = new_times[-1].isoformat() + "Z"
    zarr.consolidate_metadata(write_store)
    fs_write.invalidate_cache()

    note = ""
    if verify:
        import xarray as xr

        check_store = s3fs.S3Map(root=store_key, s3=fs_read, check=False)
        ds = xr.open_zarr(check_store, consolidated=True)
        t0 = pd.Timestamp(ds["time"].values[0])
        t1 = pd.Timestamp(ds["time"].values[-1])
        n_nat = int(pd.isna(ds["time"].values).sum())
        ds.close()
        # Small tolerance for float32 rounding (observed up to ~a few hundred
        # microseconds at these offset magnitudes) -- a real bug (wrong
        # epoch/masking) would be off by seconds/years, not microseconds.
        tol = pd.Timedelta(seconds=2)
        if n_nat or not (start_dt - tol <= t0 <= end_dt + tol and start_dt - tol <= t1 <= end_dt + tol):
            return f"ERROR {store_key}: post-write verification out of range (t0={t0}, t1={t1}, n_nat={n_nat}, expected [{start_dt},{end_dt}])"
        note = f" verified_range=({t0} .. {t1})"

    return f"OK {store_key}: n={n} old_units={old_units!r} -> new_units={new_units!r}{note}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASET_PREFIXES), help="dataset to scan/repair (used with --scan)")
    parser.add_argument("--scan", action="store_true", help="auto-discover corrupted stores for --dataset via the stuck-epoch signature")
    parser.add_argument("--months", default="202410,202411,202412", help="comma-separated YYYYMM substrings to scan (default covers jason3's known-affected window)")
    parser.add_argument("--bad-list", help="path to a text file with one store path per line to repair")
    parser.add_argument("--paths", nargs="*", help="explicit store path(s) to repair")
    parser.add_argument("--dry-run", action="store_true", help="don't write anything, just print the plan")
    parser.add_argument("--no-verify", dest="verify", action="store_false", help="skip the post-write read-back verification")
    parser.add_argument("--mc-config", default=str(Path.home() / ".mc" / "config.json"))
    parser.add_argument("--mc-alias", default="edito")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    fs_read = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": ENDPOINT_URL})

    if args.scan:
        if not args.dataset:
            parser.error("--scan requires --dataset")
        months = [m.strip() for m in args.months.split(",")]
        print(f"Scanning {args.dataset} for months {months}...", flush=True)
        stores = discover_bad_stores(fs_read, args.dataset, months)
        print(f"Found {len(stores)} corrupted stores.", flush=True)
    elif args.bad_list:
        stores = [line.strip() for line in Path(args.bad_list).read_text().splitlines() if line.strip()]
    elif args.paths:
        stores = args.paths
    else:
        parser.error("must supply one of --scan, --bad-list, or --paths")
        return 2

    if not stores:
        print("Nothing to do.")
        return 0

    key, secret, token = get_credentials(Path(args.mc_config), args.mc_alias)
    fs_write = s3fs.S3FileSystem(key=key, secret=secret, token=token, client_kwargs={"endpoint_url": ENDPOINT_URL})

    n_ok = n_err = 0
    n_done = 0
    print_lock = Lock()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(repair_one, fs_read, fs_write, store, args.dry_run, args.verify): store
            for store in stores
        }
        for fut in as_completed(futures):
            store = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001 - report and keep going
                result = f"ERROR {store}: {exc!r}"
            with print_lock:
                n_done += 1
                print(f"[{n_done}/{len(stores)}] {result}", flush=True)
                if result.startswith("OK") or result.startswith("DRY-RUN"):
                    n_ok += 1
                else:
                    n_err += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s: {n_ok} ok, {n_err} errors")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
