#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Evaluation of a model against a given reference."""

import sys
from pathlib import Path

# Ensure the repository root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dc2.evaluation.evaluation import DC2Evaluation  # noqa: E402
from dctools.processing.runner import run_from_config  # noqa: E402
from dctools.utilities.args_config import parse_arguments  # noqa: E402

# Directory that holds the DC2-specific YAML configs shipped in this repo.
DC2_CONFIG_DIR = PROJECT_ROOT / "dc2" / "config"

# Default config file name (without .yaml).
# Switch to "dc2_edito" if running against the EDITO infrastructure.
DEFAULT_CONFIG_NAME = "dc2_wasabi"

# Absolute path to the leaderboard display config shipped with this repo.
_LEADERBOARD_TEXTS_YAML = DC2_CONFIG_DIR / "leaderboard_texts.yaml"


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


def _resolve_config_path(cli_args) -> Path:
    """Return the Path to the YAML config, resolved relative to this repo.

    Priority:
    1. ``-c / --config_name`` CLI flag  →  dc2/config/<name>.yaml
    2. DEFAULT_CONFIG_NAME              →  dc2/config/dc2_wasabi.yaml
    """
    config_name = getattr(cli_args, "config_name", None) or DEFAULT_CONFIG_NAME
    return DC2_CONFIG_DIR / f"{config_name}.yaml"


if __name__ == "__main__":
    _inject_default_paths(sys.argv)
    cli_args = parse_arguments()
    # Inject the leaderboard texts path so DC2Evaluation can find it without
    # relying on a relative path baked into evaluation.py.
    if not getattr(cli_args, "leaderboard_config", None):
        vars(cli_args)["leaderboard_config"] = str(_LEADERBOARD_TEXTS_YAML)
    config_path = _resolve_config_path(cli_args)
    sys.exit(run_from_config(config_path, evaluation_cls=DC2Evaluation, cli_args=cli_args))
