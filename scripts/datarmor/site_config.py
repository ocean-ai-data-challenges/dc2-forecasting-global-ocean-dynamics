#!/usr/bin/env python3
"""Write a full DC2 config: a site overlay (``base:`` and per-dataset ``sources:`` overrides)
applied to its base config, top-level ``key=value`` settings (strings) last.

    python scripts/datarmor/site_config.py dc2/config/dc2_datarmor.yaml out.yaml start_time=2024-01-01
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


def site_config(overlay_path: Path, settings: list[str] = ()) -> dict:
    overlay = yaml.safe_load(overlay_path.read_text())
    # safe_load resolves the base's anchors and merge keys: each source is a plain mapping
    cfg = yaml.safe_load((overlay_path.parent / overlay.pop("base")).read_text())
    overrides = overlay.pop("sources", {}) or {}
    by_name = {s["dataset"]: s for s in cfg["sources"]}
    unknown = set(overrides) - set(by_name)
    if unknown:
        raise SystemExit(f"{overlay_path}: no source {sorted(unknown)} in the base config")
    for name, fields in overrides.items():
        by_name[name].update(fields)
    cfg.update(overlay)
    for s in settings:
        key, _, value = s.partition("=")
        cfg[key] = value                     # kept a string: dates as the base writes them
    return cfg


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    cfg = site_config(Path(sys.argv[1]), sys.argv[3:])
    Path(sys.argv[2]).write_text(yaml.safe_dump(cfg, sort_keys=False))
