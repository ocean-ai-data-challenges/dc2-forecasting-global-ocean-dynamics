#!/usr/bin/env python3
"""Convert legacy 2023 Glorys zarr stores to the 2024 schema.

The 2023 files on S3 (``DC2/ZARR/Glorys/glorys_YYYY-MM-DD.zarr``) use a
different, older layout than the 2024 files (``DC2/ZARR/Glorys/YYYYMMDD.zarr``):

  2023 (legacy)                          2024 (target schema)
  ----------------------------------     ----------------------------------
  dims: depth, latitude, longitude       dims: depth, time, lat, lon
  no "time" coordinate (date only in     real "time" coordinate
  the global attr ``date``)
  vars: so, thetao, uo, vo, zos, mlotst  vars: salinity, ssh, temperature,
                                          u_current, v_current
  dtype: float32                         dtype: float64

Because the 2023 files have no usable time coordinate, dctools'
``get_time_bound_values()`` returns (None, None) for them, so every 2023
Glorys catalog entry gets filtered out ("empty after filtering") -- the
whole year silently disappears from the DC2 evaluation.

This script re-writes each 2023 file to the 2024 schema/naming
(``YYYYMMDD.zarr``) so a fresh catalog rebuild picks it up exactly like a
2024 file, with no special-casing needed in dctools.  Original
``glorys_YYYY-MM-DD.zarr`` files are left untouched (new files are written
alongside them, no deletion/overwrite of existing data).

Credentials are NEVER hardcoded here: they are read at runtime either from
the GLORYS_S3_ACCESS_KEY / GLORYS_S3_SECRET_KEY environment variables, or
(by default) from the local ``mc`` alias config (``~/.mc/config.json``).

Usage examples
--------------
    # Dry-run a single date (no S3 write, just prints the planned transform)
    python3 scripts/convert_glorys_2023_to_2024_schema.py --dates 2023-01-01 --dry-run

    # Convert + verify a single date, writing to S3
    python3 scripts/convert_glorys_2023_to_2024_schema.py --dates 2023-01-01

    # Convert the whole of 2023 (resumable: already-converted dates are skipped)
    python3 scripts/convert_glorys_2023_to_2024_schema.py --start-date 2023-01-01 --end-date 2023-12-31
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from threading import Lock

import boto3
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

BUCKET = "project-ppr-ocean-ia"
PREFIX = "DC2/ZARR/Glorys"
ENDPOINT_URL = "https://minio.dive.edito.eu"

RENAME_DIMS = {"latitude": "lat", "longitude": "lon"}
RENAME_VARS = {"so": "salinity", "thetao": "temperature", "uo": "u_current", "vo": "v_current", "zos": "ssh"}
DROP_VARS = ["mlotst"]  # not present in the 2024 files / not used by dc2_edito.yaml
VAR_DIM_ORDER = {
    "salinity": ("depth", "time", "lat", "lon"),
    "temperature": ("depth", "time", "lat", "lon"),
    "u_current": ("depth", "time", "lat", "lon"),
    "v_current": ("depth", "time", "lat", "lon"),
    "ssh": ("time", "lat", "lon"),
}
EXPECTED_2024_VARS = set(VAR_DIM_ORDER)


def get_credentials(mc_config: Path, mc_alias: str) -> tuple[str, str, str | None]:
    """Read S3 credentials from env vars, falling back to the local mc alias config.

    The EDITO minio endpoint hands out temporary STS credentials (SSO/Onyxia),
    so a session token must be sent along with the key/secret or every
    authenticated request is rejected (InvalidTokenId / Forbidden).
    """
    key = os.environ.get("GLORYS_S3_ACCESS_KEY")
    secret = os.environ.get("GLORYS_S3_SECRET_KEY")
    token = os.environ.get("GLORYS_S3_SESSION_TOKEN")
    if key and secret:
        return key, secret, token
    cfg = json.loads(mc_config.read_text())
    alias = cfg["aliases"][mc_alias]
    return alias["accessKey"], alias["secretKey"], alias.get("sessiontoken")


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _delete_existing_store_objects(fs_write: s3fs.S3FileSystem, dst_key: str) -> int:
    """Delete all objects under an existing zarr store one key at a time.

    This MinIO endpoint rejects botocore's batched ``DeleteObjects`` call
    (used internally by zarr/s3fs when overwriting an existing store) with
    ``MissingContentMD5`` -- a known incompatibility between recent
    botocore versions (flexible checksums) and this backend. Deleting
    objects one at a time via single ``DeleteObject`` calls avoids the
    batched API entirely and works reliably.
    """
    client = boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=fs_write.key,
        aws_secret_access_key=fs_write.secret,
        aws_session_token=fs_write.token,
    )
    prefix_key = dst_key.split("/", 1)[1] + "/"
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix_key):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    for k in keys:
        client.delete_object(Bucket=BUCKET, Key=k)
    fs_write.invalidate_cache()
    return len(keys)


def _repair_coord_nans(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Repair isolated NaN entries in an otherwise-regular 1D coordinate array.

    The legacy 2023 Glorys source files carry a single stray NaN coordinate
    label at exactly lat=0.0 / lon=0.0 (confirmed: the underlying data at that
    grid line is unaffected, only the coordinate value itself is corrupted).
    Reconstruct any NaN coordinate value via linear interpolation from its
    neighbours -- correct for a regular grid, and self-correcting even if a
    file has the corrupted index somewhere other than the observed lat/lon==0
    position.

    Returns the (possibly repaired) array and the number of NaNs that were fixed.
    """
    nan_mask = np.isnan(values)
    n_nan = int(nan_mask.sum())
    if n_nan == 0:
        return values, 0
    idx = np.arange(len(values))
    good = ~nan_mask
    if good.sum() < 2:
        return values, 0  # not enough valid points to interpolate from
    fixed = values.copy()
    fixed[nan_mask] = np.interp(idx[nan_mask], idx[good], values[good])
    return fixed, n_nan


def convert_one(fs_read: s3fs.S3FileSystem, fs_write: s3fs.S3FileSystem, day: date, dry_run: bool, overwrite: bool, verify: bool) -> str:
    src_key = f"{BUCKET}/{PREFIX}/glorys_{day.strftime('%Y-%m-%d')}.zarr"
    dst_key = f"{BUCKET}/{PREFIX}/{day.strftime('%Y%m%d')}.zarr"

    if not fs_read.exists(f"{src_key}/.zmetadata"):
        return f"SKIP {day}: source not found ({src_key})"

    if not overwrite and fs_read.exists(f"{dst_key}/.zmetadata"):
        return f"SKIP {day}: destination already converted ({dst_key})"

    src_store = s3fs.S3Map(root=src_key, s3=fs_read, check=False)
    ds = xr.open_zarr(src_store, consolidated=True)

    ds = ds.rename({k: v for k, v in RENAME_DIMS.items() if k in ds.dims or k in ds.coords})
    ds = ds.drop_vars([v for v in DROP_VARS if v in ds.data_vars])
    ds = ds.rename_vars({k: v for k, v in RENAME_VARS.items() if k in ds.data_vars})

    ts = pd.Timestamp(day.isoformat())
    ds = ds.expand_dims(time=[ts])

    for var, dim_order in VAR_DIM_ORDER.items():
        if var in ds.data_vars:
            ds[var] = ds[var].transpose(*dim_order).astype("float64")

    n_nan_repaired = {}
    for coord in ("depth", "lat", "lon"):
        if coord in ds.coords:
            ds[coord] = ds[coord].astype("float64")
            if coord in ("lat", "lon"):
                fixed_values, n_nan = _repair_coord_nans(np.asarray(ds[coord].values))
                if n_nan:
                    ds = ds.assign_coords({coord: (ds[coord].dims, fixed_values)})
                    n_nan_repaired[coord] = n_nan

    ds.attrs = {}
    for v in ds.data_vars:
        ds[v].attrs = {}
    for c in ds.coords:
        ds[c].attrs = {}

    missing = EXPECTED_2024_VARS - set(ds.data_vars)
    extra = set(ds.data_vars) - EXPECTED_2024_VARS
    if missing or extra:
        return f"ERROR {day}: schema mismatch after conversion (missing={missing}, extra={extra})"

    chunks = {"depth": 1, "time": 1, "lat": ds.sizes["lat"], "lon": ds.sizes["lon"]}
    ds = ds.chunk(chunks)

    repair_note = f" repaired_nan={n_nan_repaired}" if n_nan_repaired else ""

    if dry_run:
        return f"DRY-RUN {day}: would write {dst_key} vars={sorted(ds.data_vars)} dims={dict(ds.sizes)}{repair_note}"

    if overwrite:
        # Pre-clear any existing destination store one key at a time -- zarr's
        # own overwrite (mode="w") logic would otherwise issue a batched
        # DeleteObjects call, which this MinIO endpoint rejects. Do this
        # unconditionally (not just when ".zmetadata" exists): a store that
        # was interrupted mid-write (e.g. by a crash/reboot) can have leftover
        # data-variable objects with no ".zmetadata" yet, and skipping the
        # pre-clear in that case falls through to zarr's own batched delete
        # and fails the same way. Listing+deleting zero objects is a cheap
        # no-op when the destination doesn't exist yet.
        _delete_existing_store_objects(fs_write, dst_key)

    dst_store = s3fs.S3Map(root=dst_key, s3=fs_write, check=False)
    ds.to_zarr(dst_store, mode="w", consolidated=True)

    if verify:
        check_ds = xr.open_zarr(s3fs.S3Map(root=dst_key, s3=fs_read, check=False), consolidated=True)
        if set(check_ds.data_vars) != EXPECTED_2024_VARS:
            return f"ERROR {day}: post-write verification failed (vars={sorted(check_ds.data_vars)})"
        if check_ds["time"].values[0] != np.datetime64(ts):
            return f"ERROR {day}: post-write time mismatch ({check_ds['time'].values[0]})"
        sample = check_ds["temperature"].isel(depth=0, time=0, lat=100, lon=100).values.item()
        orig_sample = ds["temperature"].isel(depth=0, time=0, lat=100, lon=100).values.item()
        if not np.isclose(sample, orig_sample, equal_nan=True):
            return f"ERROR {day}: post-write value mismatch (got {sample}, expected {orig_sample})"
        lat_nan = int(np.isnan(np.asarray(check_ds["lat"].values)).sum())
        lon_nan = int(np.isnan(np.asarray(check_ds["lon"].values)).sum())
        if lat_nan or lon_nan:
            return f"ERROR {day}: post-write coordinate NaN remains (lat_nan={lat_nan}, lon_nan={lon_nan})"

    return f"OK {day}: wrote {dst_key}{repair_note}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dates", help="comma-separated list of YYYY-MM-DD dates to convert")
    parser.add_argument("--start-date", help="start of a date range (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="end of a date range (YYYY-MM-DD), inclusive")
    parser.add_argument("--dry-run", action="store_true", help="don't write anything, just print the plan")
    parser.add_argument("--overwrite", action="store_true", help="re-convert dates that already have a destination file")
    parser.add_argument("--no-verify", dest="verify", action="store_false", help="skip the post-write read-back verification")
    parser.add_argument("--mc-config", default=str(Path.home() / ".mc" / "config.json"))
    parser.add_argument("--mc-alias", default="edito")
    parser.add_argument("--workers", type=int, default=4, help="number of days to convert concurrently (default: 4)")
    args = parser.parse_args()

    if args.dates:
        days = [date.fromisoformat(d.strip()) for d in args.dates.split(",")]
    elif args.start_date and args.end_date:
        days = list(daterange(date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)))
    else:
        parser.error("must supply either --dates or --start-date/--end-date")
        return 2

    key, secret, token = get_credentials(Path(args.mc_config), args.mc_alias)
    fs_read = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": ENDPOINT_URL})
    fs_write = s3fs.S3FileSystem(key=key, secret=secret, token=token, client_kwargs={"endpoint_url": ENDPOINT_URL})

    n_ok = n_skip = n_err = 0
    n_done = 0
    print_lock = Lock()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(convert_one, fs_read, fs_write, day, args.dry_run, args.overwrite, args.verify): day
            for day in days
        }
        for fut in as_completed(futures):
            day = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001 - report and keep going
                result = f"ERROR {day}: {exc!r}"
            with print_lock:
                n_done += 1
                print(f"[{n_done}/{len(days)}] {result}", flush=True)
                if result.startswith("OK") or result.startswith("DRY-RUN"):
                    n_ok += 1
                elif result.startswith("SKIP"):
                    n_skip += 1
                else:
                    n_err += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s: {n_ok} ok, {n_skip} skipped, {n_err} errors")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
