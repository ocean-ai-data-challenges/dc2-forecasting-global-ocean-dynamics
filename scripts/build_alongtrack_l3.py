#!/usr/bin/env python3
"""Build DC2's along-track SSH references from the edited CMEMS L3 product (SEALEVEL_GLO_PHY_L3_MY_008_062).

The along-track references DC2 used so far are the missions' L2 GDR files converted to Zarr with only
``ssha`` and the mean topography kept: no editing flag survived, and a day of SARAL holds thousands of
land, ice and rain values of tens of metres. The DUACS L3 product is the same measurements edited,
corrected and noise-filtered (``sla_filtered``, ~65 km), with the MDT the absolute height is built from
(ADT = SLA + MDT, done by dctools). It is also the along-track reference of oceanml3d-eval
(``fetch-cmems --preset saral_l3``), so both benchmarks score against the same points.

One request per mission and month through ``copernicusmarine.subset`` (the sparse service: no account
needed), cut into one Zarr store per mission and day:

    <out>/<mission>/<MISSION>_L3_<YYYYMMDD>.zarr     dims (time), coords time/lat/lon,
                                                      vars sla_filtered, sla_unfiltered, mdt

Days already on disk are skipped, so an interrupted build resumes. ``value_qc`` is not used: the L3
points are edited already, and its value is constant per period (1 over 2023, 0 over 2024 in the
202411 version), not a quality flag.

    python scripts/build_alongtrack_l3.py --out /data/dc2_l3 --first 2023-01-01 --last 2024-12-31
    python scripts/build_alongtrack_l3.py --out /data/dc2_l3 --missions alg --first 2024-01-01 --last 2024-01-31

Then upload ``<out>/<mission>/`` to ``s3://project-ppr-ocean-ia/DC2/ZARR/L3/<mission>/`` and rebuild the
catalogs (scripts/rebuild_missing_catalog.py).
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PRODUCT_ID = "SEALEVEL_GLO_PHY_L3_MY_008_062"
VARIABLES = ("sla_filtered", "sla_unfiltered", "mdt")
# CMEMS mission codes of the product -> the store name prefix. alg: SARAL/AltiKa (drifting phase);
# j3n: Jason-3 on its interleaved orbit (from 2022-04); swon: SWOT's nadir altimeter (from 2023-07).
MISSIONS = {
    "alg": "SARAL", "j3n": "JASON3", "s3a": "SENTINEL3A", "s3b": "SENTINEL3B",
    "s6a-lr": "SENTINEL6A", "h2b": "HY2B", "swon": "SWOTNADIR",
}


def dataset_id(mission: str) -> str:
    return f"cmems_obs-sl_glo_phy-ssh_my_{mission}-l3-duacs_PT1S"


def with_retries(call, attempts: int = 5, wait_s: float = 30.0):
    """``call()``, again after a pause when the server times out or drops the connection."""
    import time
    for k in range(attempts):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 -- requests/arcosparse network errors
            if k == attempts - 1:
                raise
            print(f"[l3] {type(exc).__name__}: {exc}; retry {k + 1}/{attempts - 1} in {wait_s:.0f} s", flush=True)
            time.sleep(wait_s * (k + 1))


def months(first: str, last: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """``[first, last]`` cut at month boundaries, ends inclusive."""
    a, b = pd.Timestamp(first), pd.Timestamp(last)
    out = []
    while a <= b:
        end = min(b, a + pd.offsets.MonthEnd(0))
        out.append((a, end))
        a = end + pd.Timedelta(days=1)
    return out


def store_path(out: Path, mission: str, day: pd.Timestamp) -> Path:
    return out / mission / f"{MISSIONS[mission]}_L3_{day:%Y%m%d}.zarr"


def pivot(df: pd.DataFrame) -> pd.DataFrame:
    """The sparse CSV (one row per variable and point) -> one row per point: time, lat, lon, VARIABLES."""
    wide = df.pivot_table(index=["time", "longitude", "latitude"], columns="variable", values="value",
                          aggfunc="first").reset_index()
    wide = wide.rename(columns={"longitude": "lon", "latitude": "lat"})
    wide["time"] = pd.to_datetime(wide["time"]).dt.tz_localize(None)
    wide["lon"] = ((wide["lon"] + 180.0) % 360.0) - 180.0
    for v in VARIABLES:
        if v not in wide:
            wide[v] = np.nan
    wide = wide.dropna(subset=["sla_filtered", "sla_unfiltered"], how="all")
    return wide[["time", "lat", "lon", *VARIABLES]].sort_values("time").drop_duplicates("time")


def to_store(day_table: pd.DataFrame, path: Path, mission: str) -> None:
    """One day of one mission as a Zarr store, written to a temporary name then moved in place."""
    ds = xr.Dataset(
        {v: ("time", day_table[v].to_numpy(np.float32), {"units": "m"}) for v in VARIABLES},
        coords={"time": day_table["time"].to_numpy(),
                "lat": ("time", day_table["lat"].to_numpy(np.float32)),
                "lon": ("time", day_table["lon"].to_numpy(np.float32))},
        attrs={"source": f"CMEMS {PRODUCT_ID} {dataset_id(mission)}", "mission": MISSIONS[mission],
               "comment": "edited DUACS L3 along-track; ADT = sla_filtered + mdt"},
    )
    ds["sla_filtered"].attrs["long_name"] = "sea level anomaly, noise-filtered (~65 km)"
    ds["mdt"].attrs["long_name"] = "mean dynamic topography (CNES-CLS)"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    ds.to_zarr(tmp, mode="w", consolidated=True)
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def build(out: Path, mission: str, first: str, last: str, subset=None) -> int:
    """The daily stores of one mission over ``[first, last]``; returns the number written."""
    if subset is None:
        import copernicusmarine
        subset = copernicusmarine.subset
    written = 0
    for start, end in months(first, last):
        days = pd.date_range(start, end, freq="D")
        todo = [d for d in days if not store_path(out, mission, d).exists()]
        if not todo:
            continue
        with tempfile.TemporaryDirectory(prefix=f"l3_{mission}_") as tmp:
            with_retries(lambda: subset(dataset_id=dataset_id(mission), variables=list(VARIABLES),
                   start_datetime=f"{todo[0]:%Y-%m-%d}T00:00:00",
                   end_datetime=f"{todo[-1]:%Y-%m-%d}T23:59:59",
                   output_directory=tmp, output_filename="month.csv", disable_progress_bar=True))
            raws = sorted(Path(tmp).glob("month*.csv"))
            if not raws:
                print(f"[l3] {mission} {start:%Y-%m}: no data")
                continue
            table = pivot(pd.concat([pd.read_csv(r) for r in raws], ignore_index=True))
        for day in todo:
            sel = table[(table["time"] >= day) & (table["time"] < day + pd.Timedelta(days=1))]
            if len(sel):
                to_store(sel, store_path(out, mission, day), mission)
                written += 1
        print(f"[l3] {mission} {start:%Y-%m}: {len(table)} points, {written} stores so far", flush=True)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--first", default="2023-01-01")
    ap.add_argument("--last", default="2024-12-31")
    ap.add_argument("--missions", nargs="+", default=list(MISSIONS), choices=list(MISSIONS))
    a = ap.parse_args()
    for m in a.missions:
        n = build(a.out, m, a.first, a.last)
        print(f"[l3] {m}: {n} daily stores written under {a.out / m}")


if __name__ == "__main__":
    main()
