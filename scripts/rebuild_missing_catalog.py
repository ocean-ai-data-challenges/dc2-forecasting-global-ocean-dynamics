#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Rebuild the local catalog JSON for ONE dataset alias, without running the
full dc2 evaluation pipeline (no obs/pred prefetch, no metrics computation).

This reuses the exact same code path as the real pipeline
(``get_dataset_from_config`` -> ``RemoteDataset.__init__`` ->
``list_files_with_metadata()`` -> ``catalog.to_json()``), just scoped to a
single dataset and without ever entering ``run_eval()``'s batch loop.

Usage:
    python scripts/rebuild_missing_catalog.py <dataset_alias> [<dataset_alias> ...]

Example:
    python scripts/rebuild_missing_catalog.py glorys
    python scripts/rebuild_missing_catalog.py saral jason3 swot
"""
import os
import sys
from pathlib import Path

# Cap BLAS thread pools before numpy/scipy import (same as evaluate.py).
for _var in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "GOTO_NUM_THREADS",
    "BLOSC_NTHREADS",
):
    os.environ.setdefault(_var, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger  # noqa: E402
from dctools.utilities.args_config import parse_arguments, load_args_and_config  # noqa: E402
from dctools.data.datasets.dataset import get_dataset_from_config  # noqa: E402
from dctools.data.coordinates import get_target_depth_values  # noqa: E402
from oceanbench.core.distributed import DatasetProcessor  # noqa: E402

DC2_CONFIG_DIR = PROJECT_ROOT / "dc2" / "config"
DEFAULT_CONFIG_NAME = "dc2_edito"


def build_dataset_processor(source: dict, fallback_cfg: dict) -> DatasetProcessor:
    """Build a DatasetProcessor sized per this dataset's own YAML overrides."""
    n_workers = int(source.get("n_parallel_workers", fallback_cfg.get("n_parallel_workers", 2)))
    threads_per_worker = int(
        source.get("nthreads_per_worker", fallback_cfg.get("nthreads_per_worker", 1))
    )
    memory_limit = source.get("memory_limit_per_worker", fallback_cfg.get("memory_limit_per_worker", "4GB"))

    logger.info(
        f"Building DatasetProcessor: n_workers={n_workers}, "
        f"threads_per_worker={threads_per_worker}, memory_limit={memory_limit}"
    )
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

    return DatasetProcessor(
        distributed=True,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
    )


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <dataset_alias> [<dataset_alias> ...]")
        return 1
    target_aliases = sys.argv[1:]

    default_output_dir = PROJECT_ROOT / "dc2_output"
    default_log_dir = default_output_dir / "logs"
    argv = [
        "-d", str(default_output_dir),
        "-l", str(default_log_dir / "rebuild_catalog.log"),
    ]
    cli_args = parse_arguments(argv)

    config_path = DC2_CONFIG_DIR / f"{DEFAULT_CONFIG_NAME}.yaml"
    args = load_args_and_config(str(config_path), args=cli_args)
    if args is None:
        print("Config loading failed.")
        return 1

    vars(args)["catalog_dir"] = os.path.join(args.data_directory, "catalogs")
    os.makedirs(args.catalog_dir, exist_ok=True)

    sources = getattr(args, "sources", []) or []
    by_name = {s.get("dataset"): s for s in sources if isinstance(s, dict)}

    for alias in target_aliases:
        if alias not in by_name:
            logger.error(f"Dataset '{alias}' not found in {config_path}. Skipping.")
            continue

        source = by_name[alias]
        catalog_path = os.path.join(args.catalog_dir, f"{alias}.json")
        if os.path.exists(catalog_path):
            logger.warning(
                f"Local catalog already exists at {catalog_path} — it will be "
                "REUSED (not rebuilt) unless you remove it first. Skipping."
            )
            continue

        logger.info(f"=== Rebuilding catalog for '{alias}' ===")
        processor = build_dataset_processor(source, fallback_cfg={})
        try:
            kwargs = {
                "source": source,
                "root_data_folder": args.data_directory,
                "root_catalog_folder": args.catalog_dir,
                "dataset_processor": processor,
                "max_samples": args.max_samples,
                "file_cache": None,
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
            n_entries = len(dataset.list_paths())
            logger.info(f"=== '{alias}' catalog rebuilt with {n_entries} entries -> {catalog_path} ===")
        except Exception:
            logger.exception(f"Failed to rebuild catalog for '{alias}'")
        finally:
            try:
                processor.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
