#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Evaluation of a model against a given reference."""

import sys
from pathlib import Path

# Ensure the repository root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dc2.evaluation.dc2 import DC2Evaluation  # noqa: E402
from dctools.processing.runner import run_from_config  # noqa: E402
from dctools.utilities.args_config import parse_arguments  # noqa: E402

# Directory that holds the DC2-specific YAML configs shipped in this repo.
DC2_CONFIG_DIR = PROJECT_ROOT / "dc2" / "config"

# Default config file name (without .yaml).
# Switch to "dc2_edito" if running against the EDITO infrastructure.
DEFAULT_CONFIG_NAME = "dc2_wasabi"

# Absolute path to the leaderboard display config shipped with this repo.
_LEADERBOARD_CONFIG_YAML = DC2_CONFIG_DIR / "leaderboard_config.yaml"


def _has_arg(argv: list[str], short: str, long: str) -> bool:
    """Check whether *short* or *long* flag already appears in *argv*."""
    return any(a == short or a == long or a.startswith(f"{long}=") for a in argv)


def _resolve_dc2_config(cli_args) -> Path:
    """Resolve the DC2 YAML config path from CLI args or default."""
    config_name = getattr(cli_args, "config_name", None) or DEFAULT_CONFIG_NAME
    return DC2_CONFIG_DIR / f"{config_name}.yaml"


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
    cli_args = parse_arguments()
    # Inject the leaderboard config path so DC2Evaluation can find it without
    # relying on a relative path baked into dc2.py.
    if not getattr(cli_args, "leaderboard_config", None):
        vars(cli_args)["leaderboard_config"] = str(_LEADERBOARD_CONFIG_YAML)
    config_path = _resolve_dc2_config(cli_args)
    sys.exit(run_from_config(config_path, evaluation_cls=DC2Evaluation, cli_args=cli_args))
