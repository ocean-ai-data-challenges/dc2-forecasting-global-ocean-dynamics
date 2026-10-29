#!/usr/bin/env python3
r"""Score DC2's probabilistic baseline, GLO4ENS, straight from its bucket: no copy of the forecasts.

The GLO4ENS forecasts stay in the public CloudFerro bucket ``MOISICEEF``; only their scores are
kept. One cycle at a time:

1. the cycle is put on DC2's grid (``build_glo4ens.build_cycle``) into ``<work>/tmp/``;
2. DC2 scores it against the references of the config (``dc2/evaluate.py``, its ``glo4ens``
   source pointed to ``<work>/tmp``), over the cycle's own window, into ``<work>/run``;
3. its batch result files are tagged with the cycle (``results_glo4ens_batch_c<YYYYMMDD>_<i>``:
   the next cycle's run does not take them for its own, the consolidation reads them all);
4. the converted cycle, the observation/reference download caches and the temporary files are
   deleted; ``<work>/cycles_done/<YYYYMMDD>`` records the cycle.

The disk holds one converted cycle at a time (~31 GB at 50 members and 10 days, 3 GB at 5), the
reference catalogs and the scores. An interrupted run resumes: finished cycles are skipped, an
unfinished one resumes from its saved batches. ``<work>/run/results/results_glo4ens.json`` holds
the scores of all the cycles done (and the leaderboard pages beside it).

    python scripts/evaluate_glo4ens.py --work $SCRATCH/glo4ens --first 2023-01-01 --last 2024-12-31
    python scripts/evaluate_glo4ens.py --work /tmp/g4 --cycles 20240104 --members 2 \\
        --references nadir_l3 glorys
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import build_glo4ens  # noqa: E402

MODEL = "glo4ens"
BASELINE = "glonet"   # glo4ens is scored against the references of the deterministic baseline
# download caches and temporary files of a run, deleted after each cycle
CACHES = ["results_batches/obs_prefetch_cache", "results_batches/ref_prefetch_cache",
          "results_batches/obs_batch_shared", "results/_npy_cache", "results/dask-report.html"]


def cycle_config(base: dict, cycle: str, tmp: Path, references: list[str]) -> dict:
    """The base config restricted to one cycle's window, its glo4ens source on ``tmp``."""
    cfg = dict(base)
    n_days = int(cfg["n_days_forecast"])
    first = pd.Timestamp(cycle)
    cfg["start_time"] = f"{first:%Y-%m-%d}"
    # one day past the last lead day: the catalogs are filtered on [start, end)
    cfg["end_time"] = f"{first + pd.Timedelta(days=n_days):%Y-%m-%d}"
    sources = []
    for src in cfg["sources"]:
        if src.get("dataset") == MODEL:
            src = {**src, "config": "local", "root_path": str(tmp), "file_pattern": f"{cycle}.zarr"}
        sources.append(src)
    cfg["sources"] = sources
    cfg["dataset_references"] = {MODEL: references}
    cfg["resume"] = True
    return cfg


def tag_batches(run: Path, cycle: str) -> int:
    """results_glo4ens_batch_<i>.json.gz -> results_glo4ens_batch_c<cycle>_<i>.json.gz."""
    n = 0
    for f in (run / "results_batches").glob(f"results_{MODEL}_batch_*.json.gz"):
        index = f.name[len(f"results_{MODEL}_batch_"):-len(".json.gz")]
        if index.isdigit():
            f.rename(f.with_name(f"results_{MODEL}_batch_c{cycle}_{index}.json.gz"))
            n += 1
    return n


def clean(run: Path, tmp: Path, cycle: str) -> None:
    shutil.rmtree(build_glo4ens.store_path(tmp, cycle), ignore_errors=True)
    shutil.rmtree(build_glo4ens._done_dir(build_glo4ens.store_path(tmp, cycle)), ignore_errors=True)
    for c in CACHES:
        p = run / c
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    # the catalog of the deleted cycle: rebuilt for the next one
    (run / "catalogs" / f"{MODEL}.json").unlink(missing_ok=True)


def evaluate_cycle(work: Path, base: dict, cycle: str, sources: dict, members: list[int],
                   workers: int, references: list[str], python: str = sys.executable) -> None:
    run, tmp = work / "run", work / "tmp"
    done = work / "cycles_done" / cycle
    if done.exists():
        print(f"[glo4ens-eval] {cycle}: done already", flush=True)
        return
    n_days = int(base["n_days_forecast"])
    build_glo4ens.build_cycle(tmp, cycle, sources, members, n_days, workers,
                              work / "orca025_to_dc2_bilinear.nc")
    (run / "catalogs" / f"{MODEL}.json").unlink(missing_ok=True)
    cfg_path = work / "configs" / f"{cycle}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cycle_config(base, cycle, tmp, references), sort_keys=False))
    log = work / "logs" / f"{cycle}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as out:
        rc = subprocess.run([python, str(ROOT / "dc2" / "evaluate.py"), "-d", str(run),
                             "-c", str(cfg_path), "-l", str(log.with_suffix(".dc2.log"))],
                            stdout=out, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    if rc != 0:
        # the converted cycle is kept: a rerun resumes from the saved batches
        raise SystemExit(f"[glo4ens-eval] {cycle}: evaluation failed (exit {rc}), see {log}")
    n = tag_batches(run, cycle)
    clean(run, tmp, cycle)
    done.parent.mkdir(parents=True, exist_ok=True)
    done.touch()
    print(f"[glo4ens-eval] {cycle}: {n} batch file(s) kept, converted cycle deleted", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True, type=Path, help="work directory (scratch)")
    ap.add_argument("--config", default=str(ROOT / "dc2" / "config" / "dc2_edito.yaml"))
    ap.add_argument("--first", default="2023-01-01")
    ap.add_argument("--last", default="2024-12-31")
    ap.add_argument("--cycles", nargs="+", help="cycle dates YYYYMMDD (instead of --first/--last)")
    ap.add_argument("--members", default="1-50")
    ap.add_argument("--workers", type=int, default=2, help="conversion processes (~2.5 GB each)")
    ap.add_argument("--references", nargs="+", default=None,
                    help="default: the config's dataset_references of glonet")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter of the DC2 evaluation (default: this one; the conversion "
                         "needs xESMF, the evaluation dctools)")
    a = ap.parse_args()
    base = yaml.safe_load(Path(a.config).read_text())
    if not any(s.get("dataset") == MODEL for s in base["sources"]):
        raise SystemExit(f"no '{MODEL}' source in {a.config}")
    if a.references is None:
        a.references = list((base.get("dataset_references") or {}).get(BASELINE) or [])
        if not a.references:
            raise SystemExit(f"no dataset_references of '{BASELINE}' in {a.config}: "
                             "give --references")
    cycles = build_glo4ens.inventory(build_glo4ens.filesystem())
    if a.cycles:
        chosen = a.cycles
    else:
        lo, hi = pd.Timestamp(a.first), pd.Timestamp(a.last)
        n_days = int(base["n_days_forecast"])
        # the forecast's last lead day within the period
        chosen = [c for c in cycles if lo <= pd.Timestamp(c)
                  and pd.Timestamp(c) + pd.Timedelta(days=n_days - 1) <= hi]
    missing = [c for c in chosen if c not in cycles]
    if missing:
        raise SystemExit(f"no complete GLO4ENS cycle for {missing}")
    members = build_glo4ens.parse_members(a.members)
    print(f"[glo4ens-eval] {len(chosen)} cycle(s), {len(members)} member(s), "
          f"{base['n_days_forecast']} lead days, references {a.references}", flush=True)
    for c in chosen:
        evaluate_cycle(a.work, base, c, cycles[c], members, a.workers, a.references, a.python)


if __name__ == "__main__":
    main()
