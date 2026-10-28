"""Regenerate Argo_velocities.nc with a proper `time` coordinate.

The source file published on S3
(``project-ppr-ocean-ia/DC2/Argo_velocities/Argo_velocities.nc``) is a single
monolithic file with 537 156 points (dims: ``n_points``) and variables
``latitude``, ``longitude``, ``depth``, ``U``, ``V``, ``ID`` -- but it is
missing a ``time`` variable entirely, which breaks catalog building in dc2
(the single catalog entry gets null date bounds and is filtered out by
``filter_catalog_by_date``, silently producing an "empty after filtering"
dataset with 0 evaluated points).

The companion ``Argo_velocities.csv`` (the file the .nc was generated from)
does contain a real ``time`` column, with two rows per point: one
``parameter == "EWCT"`` (eastward, -> U) and one ``parameter == "NSCT"``
(northward, -> V). This script verifies -- across the *entire* dataset, not
just a sample -- that the EWCT/NSCT row order matches the .nc file's
``n_points`` order exactly (lat/lon/depth/value/platformId all match with 0
diff, and EWCT/NSCT timestamps agree for 100% of rows), then reattaches the
recovered ``time`` variable (dim ``n_points``, same pattern as the existing
``latitude``/``longitude``/``depth`` coordinate variables) and writes a
corrected local copy of the NetCDF file.

This script does NOT upload anything to S3. Review the output file, then
upload it yourself (you'll need write access to the
``project-ppr-ocean-ia`` bucket on ``minio.dive.edito.eu``), replacing:

    DC2/Argo_velocities/Argo_velocities.nc

Usage:
    python scripts/fix_argo_velocities_time.py \\
        --nc /path/to/Argo_velocities.nc \\
        --csv /path/to/Argo_velocities.csv \\
        --output /path/to/Argo_velocities.fixed.nc
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr


def build_fixed_dataset(nc_path: str, csv_path: str) -> xr.Dataset:
    ds = xr.open_dataset(nc_path)
    n_points = ds.sizes["n_points"]

    df = pd.read_csv(csv_path, comment="#")

    ewct = df[df["parameter"] == "EWCT"].reset_index(drop=True)
    nsct = df[df["parameter"] == "NSCT"].reset_index(drop=True)

    if len(ewct) != n_points or len(nsct) != n_points:
        raise ValueError(
            f"CSV row counts (EWCT={len(ewct)}, NSCT={len(nsct)}) do not match "
            f"n_points={n_points}; cannot safely recover time by position."
        )

    checks = {
        "latitude (EWCT)": np.nanmax(np.abs(ewct["latitude"].values - ds["latitude"].values)),
        "longitude (EWCT)": np.nanmax(np.abs(ewct["longitude"].values - ds["longitude"].values)),
        "depth (EWCT)": np.nanmax(np.abs(ewct["depth"].values - ds["depth"].values)),
        "U vs EWCT value": np.nanmax(np.abs(ewct["value"].values - ds["U"].values)),
        "V vs NSCT value": np.nanmax(np.abs(nsct["value"].values - ds["V"].values)),
        "ID vs EWCT platformId": np.nanmax(np.abs(ewct["platformId"].values - ds["ID"].values)),
        "ID vs NSCT platformId": np.nanmax(np.abs(nsct["platformId"].values - ds["ID"].values)),
    }
    for name, max_diff in checks.items():
        if max_diff > 0:
            raise ValueError(
                f"Alignment check failed for {name}: max diff = {max_diff} "
                "(expected exactly 0). Refusing to reattach time positionally."
            )

    t_ewct = pd.to_datetime(ewct["time"]).values
    t_nsct = pd.to_datetime(nsct["time"]).values
    time_match_frac = float((t_ewct == t_nsct).mean())
    if time_match_frac < 1.0:
        raise ValueError(
            f"EWCT/NSCT timestamps only agree for {time_match_frac:.4%} of rows "
            "(expected 100%). Refusing to reattach time positionally."
        )

    fixed = ds.copy()
    fixed = fixed.assign_coords(time=("n_points", t_ewct))
    fixed["time"].attrs.clear()
    fixed.attrs["TimeFix"] = (
        "The 'time' coordinate was reconstructed post-hoc from the companion "
        "Argo_velocities.csv (EWCT rows, positionally aligned with n_points; "
        "verified exact match on latitude/longitude/depth/U/V/ID across the "
        "full dataset). The original published .nc file had no time variable."
    )
    return fixed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nc", required=True, help="Path to the local Argo_velocities.nc")
    parser.add_argument("--csv", required=True, help="Path to the local Argo_velocities.csv")
    parser.add_argument("--output", required=True, help="Path to write the corrected .nc file")
    args = parser.parse_args()

    fixed = build_fixed_dataset(args.nc, args.csv)
    fixed.to_netcdf(args.output)

    time_vals = pd.to_datetime(fixed["time"].values)
    print(f"Wrote corrected file to: {args.output}")
    print(f"n_points: {fixed.sizes['n_points']}")
    print(f"time range: {time_vals.min()} -> {time_vals.max()}")
    print(
        "\nNext step: upload this file to S3, replacing "
        "DC2/Argo_velocities/Argo_velocities.nc in the project-ppr-ocean-ia bucket."
    )


if __name__ == "__main__":
    sys.exit(main())
