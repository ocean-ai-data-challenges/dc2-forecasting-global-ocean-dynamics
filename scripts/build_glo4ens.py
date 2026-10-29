#!/usr/bin/env python3
"""Build DC2's probabilistic baseline: Mercator's GLO4ENS ensemble forecasts on DC2's grid.

Source: the weekly forecasts of the Mercator Ocean global ensemble (``source: MOI GLO4ENS``), 50
members, 28 daily means, native ORCA025 grid (1049 x 1440, curvilinear) and 75 levels, in the public
CloudFerro bucket ``MOISICEEF`` (anonymous read), one Zarr store per cycle and file kind:

    glo4-ens50_ng_1d-m_<start>-<end>_<kind>_fcst_R<cycle>.zarr
    kind: 3DT-thetao, 3DT-so, 3DU-uo, 3DV-vo, 2DT-oce (zos, ssh_ib, tos), 2DT-siconc

Cycles are on Thursdays; the 3-D files cover 2022-12-29 .. 2024-12-26 (the 2-D ones 2020-12 ..
today).

What is done to them, the way GloNet's forecasts were put on DC2's grid:

- velocities: NEMO's C grid stores ``uo`` at the U point east of its T point and ``vo`` at the V
  point north of it, along the grid's i/j axes (rotated north of ~20N). They are averaged to the T
  points (a land face counts as a zero velocity, the model's wall condition), then rotated to
  east/north with the local direction of the i axis;
- vertical: linear interpolation from the 75 levels to DC2's 21 (the first level above 0.5058 m);
- horizontal: bilinear (xESMF) from ORCA025 to DC2's 0.25-degree grid, NaN-aware (a target point
  is ocean when at least half of its source weight is);
- ``zos``: minus ``ssh_ib``, its inverse-barometer part (the altimetry and GLORYS have none);
- land: NaN (the 3-D files have it; ``zos`` holds 0 there and takes the surface T mask);
- time: daily means stamped at 00:00 of their day (the source has 12:00); lead 0 is the cycle day;
- ``ens`` -> ``member`` (1..50). Values stay packed in int16 with the source's scale factors.

One Zarr store per cycle, the layout of DC2's GloNet stores plus the member dimension:

    <out>/<YYYYMMDD>.zarr    dims (time, member, depth, lat, lon); vars thetao, so, uo, vo, zos

A cycle is cut in (lead day, member) units, each one read, converted and written into its region of
the store; finished units are recorded in ``<store>.units_done/`` so an interrupted build resumes.
A full cycle at 10 days and 50 members reads ~150 GB from the bucket: run it next to the data
(EDITO, Datarmor), not over a home link.

    python scripts/build_glo4ens.py --out /data/glo4ens --first 2024-01-01 --last 2024-01-31
    python scripts/build_glo4ens.py --out /tmp/g4 --cycles 20240104 --members 1-5 --days 3

Then upload ``<out>/`` to ``s3://project-ppr-ocean-ia/DC2/ZARR/GLO4ENS/`` and rebuild the catalog
(scripts/rebuild_missing_catalog.py glo4ens).
"""
from __future__ import annotations

import argparse
import multiprocessing
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ENDPOINT = "https://s3.waw3-1.cloudferro.com"
BUCKET = "MOISICEEF"
NAME = re.compile(r"glo4-ens50_ng_1d-m_(\d{8})-(\d{8})_(\w+-\w+)_fcst_R(\d{8})\.zarr$")
KINDS = {"thetao": "3DT-thetao", "so": "3DT-so", "uo": "3DU-uo", "vo": "3DV-vo", "zos": "2DT-oce"}
VARIABLES_3D = ("thetao", "so", "uo", "vo")
# the source's packing, kept (int16, fill -32767)
PACKING = {"thetao": (0.001, 17.5), "so": (0.0008, 25.0), "uo": (0.00015, 0.0),
           "vo": (0.00015, 0.0), "zos": (0.001, 0.0)}
ATTRS = {
    "thetao": {"standard_name": "sea_water_potential_temperature", "units": "degC"},
    "so": {"standard_name": "sea_water_salinity", "units": "1e-3"},
    "uo": {"standard_name": "eastward_sea_water_velocity", "units": "m s-1"},
    "vo": {"standard_name": "northward_sea_water_velocity", "units": "m s-1"},
    "zos": {"standard_name": "sea_surface_height_above_geoid", "units": "m"},
}
DC2_CONFIG = Path(__file__).resolve().parents[1] / "dc2" / "config" / "dc2_edito.yaml"


def filesystem():
    import s3fs
    return s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": ENDPOINT})


def inventory(fs) -> dict[str, dict[str, str]]:
    """{cycle YYYYMMDD: {variable: store path}} for the cycles that have all the variables."""
    found: dict[str, dict[str, str]] = {}
    for path in fs.ls(BUCKET):
        m = NAME.search(path)
        if not m:
            continue
        for var, kind in KINDS.items():
            if m.group(3) == kind:
                found.setdefault(m.group(4), {})[var] = path
    return {c: v for c, v in sorted(found.items()) if len(v) == len(KINDS)}


def target_grid(config: Path = DC2_CONFIG) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DC2's lat, lon and depth axes (``target_dimensions`` of its config)."""
    import yaml
    dims = yaml.safe_load(config.read_text())["target_dimensions"]
    axis = {k: np.arange(dims[k]["start"], dims[k]["stop"], dims[k]["step"])
            for k in ("lat", "lon")}
    return axis["lat"], axis["lon"], np.asarray(dims["depth"], dtype=np.float64)


# --------------------------------------------------------------------------- grid operations

def i_axis_angle(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Angle (radians, counter-clockwise from east) of the grid's i axis at the T points.

    The chord from T(i-1) to T(i+1) projected on the local east and north at T(i) (a great
    circle's initial bearing would be off by ~0.2 degrees along a parallel); ORCA's i axis is
    periodic (1440 columns, no halo here).
    """
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    xyz = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])
    chord = np.roll(xyz, -1, axis=-1) - np.roll(xyz, 1, axis=-1)
    east = np.stack([-np.sin(lam), np.cos(lam), np.zeros_like(lam)])
    north = np.stack([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)])
    return np.arctan2((chord * north).sum(0), (chord * east).sum(0))


def velocity_at_t(u: np.ndarray, v: np.ndarray, t_mask: np.ndarray, angle: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """C-grid ``(u, v)`` along the i/j axes -> eastward and northward velocities at the T points.

    ``u[..., j, i]`` is the U point between T(i) and T(i+1), ``v[..., j, i]`` the V point between
    T(j) and T(j+1). Land faces are zero velocities; T points on land are NaN.
    """
    u0, v0 = np.nan_to_num(u), np.nan_to_num(v)
    ut = 0.5 * (u0 + np.roll(u0, 1, axis=-1))
    vt = 0.5 * v0
    vt[..., 1:, :] += 0.5 * v0[..., :-1, :]
    cos, sin = np.cos(angle), np.sin(angle)
    east = np.where(t_mask, ut * cos - vt * sin, np.nan)
    north = np.where(t_mask, ut * sin + vt * cos, np.nan)
    return east.astype(np.float32), north.astype(np.float32)


def vertical_weights(src: np.ndarray, target: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Levels and weights of a linear interpolation in depth; above the first level, the first."""
    k1 = np.clip(np.searchsorted(src, target), 1, len(src) - 1)
    k0 = k1 - 1
    w = (target - src[k0]) / (src[k1] - src[k0])
    above = target <= src[0]
    k0[above], k1[above], w[above] = 0, 0, 0.0
    return k0, k1, w.astype(np.float32)


def to_levels(field: np.ndarray, weights) -> np.ndarray:
    """(75, y, x) -> (n_target, y, x); a level below the bottom is NaN."""
    k0, k1, w = weights
    return field[k0] * (1 - w)[:, None, None] + field[k1] * w[:, None, None]


def regridder(src_lat, src_lon, lat, lon, weights_file: Path | None = None):
    """Bilinear ORCA025 -> regular grid (xESMF), its weights cached in ``weights_file``."""
    import xesmf
    src = xr.Dataset(coords={"lat": (("y", "x"), src_lat), "lon": (("y", "x"), src_lon)})
    dst = xr.Dataset(coords={"lat": ("lat", lat), "lon": ("lon", lon)})
    reuse = weights_file is not None and Path(weights_file).exists()
    return xesmf.Regridder(src, dst, "bilinear", periodic=True, ignore_degenerate=True,
                           unmapped_to_nan=True, reuse_weights=reuse,
                           filename=str(weights_file) if weights_file else None)


def regrid(rg, field: np.ndarray) -> np.ndarray:
    """(..., y, x) on ORCA -> (..., lat, lon), a target point kept when half its weight is ocean."""
    dims = tuple(f"d{k}" for k in range(field.ndim - 2)) + ("y", "x")
    out = rg(xr.DataArray(field, dims=dims), skipna=True, na_thres=0.5)
    return np.asarray(out.values, dtype=np.float32)


# --------------------------------------------------------------------------- one cycle

def store_path(out: Path, cycle: str) -> Path:
    return out / f"{cycle}.zarr"


def _done_dir(store: Path) -> Path:
    return store.with_name(store.name + ".units_done")


def create_store(store: Path, cycle: str, times, members, lat, lon, depth) -> None:
    """The empty store (metadata only): one chunk per (lead day, member, level)."""
    import dask.array as da
    shape3 = (len(times), len(members), len(depth), len(lat), len(lon))
    chunks3 = (1, 1, 1, len(lat), len(lon))
    data_vars = {}
    for v in VARIABLES_3D:
        data_vars[v] = (("time", "member", "depth", "lat", "lon"),
                        da.zeros(shape3, chunks=chunks3, dtype=np.float32), ATTRS[v])
    data_vars["zos"] = (("time", "member", "lat", "lon"),
                        da.zeros(shape3[:2] + shape3[3:], chunks=(1, 1, len(lat), len(lon)),
                                 dtype=np.float32), ATTRS["zos"])
    ds = xr.Dataset(
        data_vars,
        coords={"time": ("time", times), "member": ("member", np.asarray(members, np.int32)),
                "depth": ("depth", depth.astype(np.float32), {"units": "m", "positive": "down"}),
                "lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})},
        attrs={"source": f"Mercator Ocean GLO4ENS ensemble forecast (s3://{BUCKET} at {ENDPOINT})",
               "forecast_reference_time": f"{pd.Timestamp(cycle):%Y-%m-%d}",
               "comment": "ORCA025 -> DC2 0.25-degree grid: zos - ssh_ib; velocities averaged "
                          "to T points and rotated to east/north, linear in depth, bilinear "
                          "(NaN-aware) in latitude/longitude; daily means stamped at 00:00",
               "regrid_method": "bilinear"},
    )
    encoding = {v: {"dtype": "int16", "scale_factor": PACKING[v][0], "add_offset": PACKING[v][1],
                    "_FillValue": -32767} for v in (*VARIABLES_3D, "zos")}
    tmp = store.with_name(store.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    ds.to_zarr(tmp, mode="w", compute=False, encoding=encoding, consolidated=True)
    tmp.rename(store)
    _done_dir(store).mkdir(exist_ok=True)


_WORKER: dict = {}


def _init_worker(sources, lat, lon, depth, weights_file):
    fs = filesystem()
    opened = {v: xr.open_zarr(fs.get_mapper(p), chunks=None, consolidated=True)
              for v, p in sources.items()}
    ll = opened["zos"]
    src_lat, src_lon = ll["latitude"].values, ll["longitude"].values
    src_depth = opened["thetao"]["deptht"].values.astype(np.float64)
    _WORKER.update(
        opened=opened, angle=i_axis_angle(src_lat, src_lon),
        vweights=vertical_weights(src_depth, depth),
        rg=regridder(src_lat, src_lon, lat, lon, weights_file),
    )


def _read(var: str, t: int, ens: int, attempts: int = 5, name: str | None = None) -> np.ndarray:
    """``name`` (default ``var``) of the store of ``var``, one lead day of one member."""
    import time
    ds = _WORKER["opened"][var]
    for k in range(attempts):
        try:
            return np.asarray(ds[name or var].isel(time=t, ens=ens).values, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 -- network errors of s3fs/aiohttp
            if k == attempts - 1:
                raise
            print(f"[glo4ens] read {var} t={t} ens={ens}: {type(exc).__name__}; retry {k + 1}",
                  flush=True)
            time.sleep(20 * (k + 1))
    raise AssertionError


def convert_unit(t: int, ens_index: int) -> dict[str, np.ndarray]:
    """One lead day of one member, all variables, on DC2's grid."""
    w, rg = _WORKER["vweights"], _WORKER["rg"]
    so = _read("so", t, ens_index)
    t_mask = np.isfinite(so)                       # (75, y, x), True on ocean
    out = {"so": regrid(rg, to_levels(so, w))}
    del so
    out["thetao"] = regrid(rg, to_levels(_read("thetao", t, ens_index), w))
    east, north = velocity_at_t(_read("uo", t, ens_index), _read("vo", t, ens_index), t_mask,
                                _WORKER["angle"])
    out["uo"] = regrid(rg, to_levels(east, w))
    out["vo"] = regrid(rg, to_levels(north, w))
    del east, north
    # zos holds the inverse-barometer response to the atmospheric pressure (ssh_ib): removed, as
    # in the altimetry (DUACS SLA are corrected of it) and GLORYS (not forced by the pressure).
    # Against the L3 along-track of 2024-01-04: crmse 0.12-0.14 m with it, 0.045-0.050 m without.
    zos = _read("zos", t, ens_index) - _read("zos", t, ens_index, name="ssh_ib")
    zos = np.where(t_mask[0], zos, np.nan).astype(np.float32)
    out["zos"] = regrid(rg, zos)
    return out


def _run_unit(store: str, t: int, m: int, ens_index: int) -> tuple[int, int]:
    fields = convert_unit(t, ens_index)
    ds = xr.Dataset({v: (("time", "member", "depth", "lat", "lon"), fields[v][None, None])
                     for v in VARIABLES_3D})
    ds["zos"] = (("time", "member", "lat", "lon"), fields["zos"][None, None])
    ds.to_zarr(store, region={"time": slice(t, t + 1), "member": slice(m, m + 1)}, mode="r+")
    (_done_dir(Path(store)) / f"{t}_{m}").touch()
    return t, m


def build_cycle(out: Path, cycle: str, sources: dict[str, str], members: list[int], days: int,
                workers: int, weights_file: Path) -> Path:
    lat, lon, depth = target_grid()
    store = store_path(out, cycle)
    first = pd.Timestamp(cycle)
    times = pd.date_range(first, periods=days, freq="D").to_numpy()
    if not store.exists():
        out.mkdir(parents=True, exist_ok=True)
        create_store(store, cycle, times, members, lat, lon, depth)
    done = {p.name for p in _done_dir(store).iterdir()}
    todo = [(t, m, member - 1) for t in range(days) for m, member in enumerate(members)
            if f"{t}_{m}" not in done]
    print(f"[glo4ens] {cycle}: {len(todo)} of {days * len(members)} (day, member) units to do",
          flush=True)
    if todo:
        # the weights once, before the workers reuse them
        if not Path(weights_file).exists():
            _init_worker(sources, lat, lon, depth, weights_file)
        # spawn, not fork: s3fs's event loop thread does not survive a fork (the workers hang)
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker,
                                 initargs=(sources, lat, lon, depth, weights_file)) as ex:
            futures = [ex.submit(_run_unit, str(store), t, m, e) for t, m, e in todo]
            for k, f in enumerate(as_completed(futures), 1):
                t, m = f.result()
                if k % 10 == 0 or k == len(futures):
                    print(f"[glo4ens] {cycle}: {k}/{len(futures)} units", flush=True)
    xr.open_zarr(store)   # readable
    return store


def parse_members(text: str) -> list[int]:
    """'1-50', '1,5,9' or '10' (the first 10)."""
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    if "," in text:
        return [int(x) for x in text.split(",")]
    return list(range(1, int(text) + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--first", default="2023-01-01", help="first cycle date (with --last)")
    ap.add_argument("--last", default="2024-12-31")
    ap.add_argument("--cycles", nargs="+", help="cycle dates YYYYMMDD (instead of --first/--last)")
    ap.add_argument("--members", default="1-50", help="'1-50', '1,2,3' or a count (default all 50)")
    ap.add_argument("--days", type=int, default=10, help="lead days kept (DC2's n_days_forecast)")
    ap.add_argument("--workers", type=int, default=2, help="processes (~2.5 GB each)")
    ap.add_argument("--weights", type=Path, default=None,
                    help="xESMF weights file (default <out>/orca025_to_dc2_bilinear.nc)")
    a = ap.parse_args()
    fs = filesystem()
    cycles = inventory(fs)
    if a.cycles:
        chosen = a.cycles
    else:
        lo, hi = pd.Timestamp(a.first), pd.Timestamp(a.last)
        chosen = [c for c in cycles if lo <= pd.Timestamp(c) <= hi]
    missing = [c for c in chosen if c not in cycles]
    if missing:
        raise SystemExit(f"no complete GLO4ENS cycle for {missing} "
                         "(3-D files: 20221229 .. 20241226)")
    weights = a.weights or a.out / "orca025_to_dc2_bilinear.nc"
    a.out.mkdir(parents=True, exist_ok=True)
    for c in chosen:
        build_cycle(a.out, c, cycles[c], parse_members(a.members), a.days, a.workers, weights)
        print(f"[glo4ens] {c}: {store_path(a.out, c)}", flush=True)


if __name__ == "__main__":
    main()
