#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Evaluation of a model against a given reference."""

import sys
from pathlib import Path

# Ensure the repository root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dctools.processing.runner import run_from_cli  # noqa: E402


def _has_arg(argv: list[str], *names: str) -> bool:
    for i, arg in enumerate(argv):
        if arg in names:
            return True
        for name in names:
            if arg.startswith(f"{name}="):
                return True
    return False


def _inject_default_paths(argv: list[str]) -> None:
    default_output_dir = PROJECT_ROOT / "dc2_output"
    default_log_dir = default_output_dir / "logs"
    default_logfile = default_log_dir / "dc2.log"

    if not _has_arg(argv, "-d", "--data_directory"):
        argv.extend(["--data_directory", str(default_output_dir)])

    if not _has_arg(argv, "-l", "--logfile"):
        default_log_dir.mkdir(parents=True, exist_ok=True)
        argv.extend(["--logfile", str(default_logfile)])


if __name__ == "__main__":
    _inject_default_paths(sys.argv)
    sys.exit(run_from_cli(default_config_name="dc2"))
