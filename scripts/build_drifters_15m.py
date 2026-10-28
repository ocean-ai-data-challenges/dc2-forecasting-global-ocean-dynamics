#!/usr/bin/env python3
"""Build DC2's surface-current reference: daily 15 m velocities of drogued drifters (CMEMS, delayed mode).

Source: ``cmems_obs-ins_glo_phy-cur_my_drifter_PT1H`` (INSITU_GLO_PHY_UV_DISCRETE_MY_013_044), hourly
velocities of the Global Drifter Program and others, at 15 m (drogued) or 0.5 m (drogue lost: the
wind slip makes them a different quantity, left out). A point is kept when both components have
``value_qc`` 1 (good). Hourly values are averaged per drifter and day (at least ``min_hours`` of them),
the inertial and tidal motions the daily fields of a model do not hold being averaged out: the
"daily 15 m currents vs drogued drifters" setting of oceanml3d-eval's surface_currents_15m benchmark.

One request per month through ``copernicusmarine.subset`` (sparse service, no account needed), one
Zarr store per day:

    <out>/DRIFTERS15M_<YYYYMMDD>.zarr     dims (n_points), coords time (the day, 00:00), lat, lon,
                                           depth (15 m); vars U, V (m/s), ID, n_hours

    python scripts/build_drifters_15m.py --out /data/dc2_drifters --first 2023-01-01 --last 2024-12-31
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DATASET_ID = "cmems_obs-ins_glo_phy-cur_my_drifter_PT1H"
DROGUE_DEPTH_M = 15.0
GOOD_QC = 1


def months(first: str, last: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    a, b = pd.Timestamp(first), pd.Timestamp(last)
    out = []
    while a <= b:
        end = min(b, a + pd.offsets.MonthEnd(0))
        out.append((a, end))
        a = end + pd.Timedelta(days=1)
    return out


def store_path(out: Path, day: pd.Timestamp) -> Path:
    return out / f"DRIFTERS15M_{day:%Y%m%d}.zarr"


def daily_means(df: pd.DataFrame, min_hours: int = 12) -> pd.DataFrame:
    """The sparse CSV -> one row per drifter and day: mean position and velocity of its good hours."""
    df = df[(df["depth"] == DROGUE_DEPTH_M) & (df["value_qc"] == GOOD_QC)]
    wide = df.pivot_table(index=["platform_id", "time", "latitude", "longitude"], columns="variable",
                          values="value", aggfunc="first").reset_index()
    wide = wide.dropna(subset=["EWCT", "NSCT"])
    wide["time"] = pd.to_datetime(wide["time"]).dt.tz_localize(None)
    wide["day"] = wide["time"].dt.floor("D")
    # mean of longitudes across the date line: through the unit vector
    lon = np.deg2rad(wide["longitude"].to_numpy())
    wide["cx"], wide["cy"] = np.cos(lon), np.sin(lon)
    g = wide.groupby(["platform_id", "day"])
    out = g.agg(lat=("latitude", "mean"), cx=("cx", "mean"), cy=("cy", "mean"),
                U=("EWCT", "mean"), V=("NSCT", "mean"), n_hours=("EWCT", "size")).reset_index()
    out = out[out["n_hours"] >= min_hours]
    out["lon"] = np.rad2deg(np.arctan2(out["cy"], out["cx"]))
    return out[["platform_id", "day", "lat", "lon", "U", "V", "n_hours"]]


def to_store(day_table: pd.DataFrame, path: Path, day: pd.Timestamp) -> None:
    n = len(day_table)
    ds = xr.Dataset(
        {"U": ("n_points", day_table["U"].to_numpy(np.float32), {"units": "m s-1", "long_name": "eastward velocity, daily mean"}),
         "V": ("n_points", day_table["V"].to_numpy(np.float32), {"units": "m s-1", "long_name": "northward velocity, daily mean"}),
         "ID": ("n_points", pd.to_numeric(day_table["platform_id"], errors="coerce").to_numpy(np.float64)),
         "n_hours": ("n_points", day_table["n_hours"].to_numpy(np.int16))},
        # stamped at 00:00 of the day, as the daily fields of DC2 (GloNet, GLORYS) are: the +-12 h
        # window of a lead time then holds that day's means only
        coords={"time": ("n_points", np.full(n, day.to_datetime64())),
                "lat": ("n_points", day_table["lat"].to_numpy(np.float32)),
                "lon": ("n_points", day_table["lon"].to_numpy(np.float32)),
                "depth": ("n_points", np.full(n, DROGUE_DEPTH_M, np.float32))},
        attrs={"source": f"CMEMS INSITU_GLO_PHY_UV_DISCRETE_MY_013_044 {DATASET_ID}",
               "comment": "drogued (15 m) drifters, value_qc 1, daily means per drifter"},
    )
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(tmp, mode="w", consolidated=True)
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def build(out: Path, first: str, last: str, min_hours: int = 12, subset=None) -> int:
    if subset is None:
        import copernicusmarine
        subset = copernicusmarine.subset
    written = 0
    for start, end in months(first, last):
        todo = [d for d in pd.date_range(start, end, freq="D") if not store_path(out, d).exists()]
        if not todo:
            continue
        with tempfile.TemporaryDirectory(prefix="drifters_") as tmp:
            subset(dataset_id=DATASET_ID, variables=["EWCT", "NSCT"],
                   start_datetime=f"{todo[0]:%Y-%m-%d}T00:00:00", end_datetime=f"{todo[-1]:%Y-%m-%d}T23:59:59",
                   output_directory=tmp, output_filename="month.csv", disable_progress_bar=True)
            raws = sorted(Path(tmp).glob("month*.csv"))
            if not raws:
                print(f"[drifters] {start:%Y-%m}: no data")
                continue
            table = daily_means(pd.concat([pd.read_csv(r) for r in raws], ignore_index=True), min_hours)
        for day in todo:
            sel = table[table["day"] == day]
            if len(sel):
                to_store(sel, store_path(out, day), day)
                written += 1
        print(f"[drifters] {start:%Y-%m}: {len(table)} drifter-days, {written} stores so far", flush=True)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--first", default="2023-01-01")
    ap.add_argument("--last", default="2024-12-31")
    ap.add_argument("--min-hours", type=int, default=12)
    a = ap.parse_args()
    print(f"[drifters] {build(a.out, a.first, a.last, a.min_hours)} daily stores under {a.out}")


if __name__ == "__main__":
    main()
