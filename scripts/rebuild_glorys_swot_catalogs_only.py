#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Rebuild ONLY the glorys and swot catalogs from a real S3 listing.

Unlike ``dc2/evaluate.py`` (which runs the full evaluation pipeline for all
10 datasets) or ``scripts/rebuild_edito_catalogs.py`` (which just copies an
existing catalog snapshot from Wasabi), this script:

1. Loads the dc2_edito.yaml config exactly like the real pipeline
   (``load_args_and_config``), then filters ``args.sources`` down to only
   the ``glorys`` and ``swot`` entries.
2. Calls ``dctools.data.datasets.dataset.get_dataset_from_config`` directly
   for each of those two datasets, WITHOUT going through
   ``BaseDCEvaluation.setup_dataset_manager()``/``get_catalog()`` first.

   IMPORTANT: ``setup_dataset_manager()`` normally calls
   ``BaseDCEvaluation.get_catalog()`` before building each dataset, which
   silently DOWNLOADS a pre-built catalog snapshot from
   ``s3://project-ppr-ocean-ia/catalogs/<dataset>.json`` whenever the local
   catalog file is missing -- masking any real rebuild. That remote catalog
   snapshot was confirmed stale (dated 2026-08-10, predates both the Glorys
   2023->2024 schema conversion and the SWOT corrupted-timestamp fix). This
   script bypasses that shortcut entirely so the catalog is genuinely
   rebuilt from a fresh S3 file listing + per-file metadata extraction.
3. Does NOT call ``.run()`` / anything metrics-related -- construction of
   the ``RemoteDataset`` alone triggers ``list_files_with_metadata()`` +
   ``build_catalog()`` + saves the catalog JSON to
   ``dc2_output/catalogs/<dataset>.json``, then the script exits.

Usage::

    conda run -n dc-env python3 scripts/rebuild_glorys_swot_catalogs_only.py
"""

import os as _os
import sys
from pathlib import Path

# Cap BLAS thread pools before any heavy import (mirrors dc2/evaluate.py).
for _var in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "BLOSC_NTHREADS",
):
    _os.environ.setdefault(_var, "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DC2_CONFIG_DIR = PROJECT_ROOT / "dc2" / "config"
CONFIG_PATH = DC2_CONFIG_DIR / "dc2_edito.yaml"
DATA_DIR = PROJECT_ROOT / "dc2_output"
LOG_FILE = DATA_DIR / "logs" / "rebuild_glorys_swot_catalogs_only.log"

TARGET_DATASETS = {"glorys", "swot"}

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

from dctools.utilities.args_config import parse_arguments, load_args_and_config
from dctools.data.datasets.dataset import get_dataset_from_config
from dctools.data.coordinates import get_target_depth_values
from dctools.utilities.file_utils import FileCacheManager
from dc2.evaluation.dc2 import DC2Evaluation


def main() -> int:
    sys.argv = [
        "rebuild_glorys_swot_catalogs_only.py",
        "--data_directory", str(DATA_DIR),
        "--logfile", str(LOG_FILE),
    ]
    cli_args = parse_arguments()

    args = load_args_and_config(str(CONFIG_PATH), args=cli_args)
    if args is None:
        print("Config loading failed.")
        return 1

    vars(args)["regridder_weights"] = _os.path.join(args.data_directory, "weights")
    vars(args)["catalog_dir"] = _os.path.join(args.data_directory, "catalogs")
    vars(args)["result_dir"] = _os.path.join(args.data_directory, "results")
    vars(args)["logs_dir"] = _os.path.join(args.data_directory, "logs")
    _os.makedirs(args.catalog_dir, exist_ok=True)
    _os.makedirs(args.result_dir, exist_ok=True)
    _os.makedirs(args.logs_dir, exist_ok=True)

    if not getattr(args, "leaderboard_config", None):
        vars(args)["leaderboard_config"] = str(DC2_CONFIG_DIR / "leaderboard_config.yaml")

    raw_sources = getattr(args, "sources", []) or []
    filtered_sources = [
        s for s in raw_sources
        if isinstance(s, dict) and s.get("dataset") in TARGET_DATASETS
    ]
    found = {s["dataset"] for s in filtered_sources}
    missing = TARGET_DATASETS - found
    if missing:
        print(f"WARNING: could not find source config for: {missing}")
    print(f"Targeting datasets: {sorted(found)}")
    # Keep args.sources filtered so DC2Evaluation's own bookkeeping (all_datasets
    # etc.) doesn't try to touch anything else -- but note all_datasets is
    # actually derived from a hardcoded dataset_references dict, independent
    # of args.sources, so this filtering only affects what WE iterate below.
    vars(args)["sources"] = filtered_sources

    evaluator = DC2Evaluation(args)
    try:
        file_cache = FileCacheManager(getattr(args, "max_cache_files", 100))
        for source in filtered_sources:
            name = source["dataset"]
            print(f"\n=== Rebuilding catalog for '{name}' (bypassing remote catalog cache) ===")
            kwargs = {
                "source": source,
                "root_data_folder": args.data_directory,
                "root_catalog_folder": args.catalog_dir,
                "dataset_processor": evaluator.dataset_processor,
                "max_samples": args.max_samples,
                "file_cache": file_cache,
                "target_depth_values": get_target_depth_values(args),
                "filter_values": {
                    "start_time": args.start_time,
                    "end_time": args.end_time,
                    "min_lon": args.min_lon,
                    "max_lon": args.max_lon,
                    "min_lat": args.min_lat,
                    "max_lat": args.max_lat,
                },
            }
            dataset = get_dataset_from_config(**kwargs)
            catalog = dataset.get_catalog()
            n_entries = len(catalog.gdf) if catalog is not None else 0
            print(f"=== '{name}': catalog rebuilt with {n_entries} entries ===")
    finally:
        evaluator.close()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
