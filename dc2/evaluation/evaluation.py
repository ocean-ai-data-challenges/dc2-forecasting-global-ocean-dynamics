#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""DC2 evaluation class – DC2-specific wiring only.

All generic evaluation logic lives in :class:`.base.BaseDCEvaluation`.
"""

from argparse import Namespace
from pathlib import Path

import yaml
from loguru import logger

from dctools.processing.base import BaseDCEvaluation
from dctools.utilities.misc_utils import display_width


class DC2Evaluation(BaseDCEvaluation):
    """Class that manages evaluation of Data Challenge 2."""

    def __init__(self, arguments: Namespace) -> None:
        """Init class.

        Args:
            arguments (Namespace): Namespace with config.
        """
        super().__init__(arguments)

        # Load leaderboard display config — path injected by evaluate.py into args
        # (falls back to the YAML key from the DC config file if present).
        _lb_config_path = getattr(arguments, "leaderboard_config", None)
        if _lb_config_path:
            _lb_yaml = Path(_lb_config_path)
            if not _lb_yaml.is_absolute():
                # Treat relative paths as relative to the project root (cwd).
                _lb_yaml = Path.cwd() / _lb_yaml
            if _lb_yaml.is_file():
                try:
                    self.leaderboard_custom_config = (
                        yaml.safe_load(_lb_yaml.read_text(encoding="utf-8")) or {}
                    )
                except Exception:  # noqa: BLE001
                    pass  # leave self.leaderboard_custom_config as None (base default)

        self.dataset_references = {
            "glonet": [
                "argo_profiles", "glorys", "jason3", "saral", "swot", # "argo_velocities",
                # "SSS_fields", "SST_fields",
            ],
        }
        self.all_datasets = list(
            set(
                list(self.dataset_references.keys())
                + [item for sublist in self.dataset_references.values() for item in sublist]
            )
        )
        self._init_cluster()

    # ------------------------------------------------------------------
    # Pretty evaluation summary
    # ------------------------------------------------------------------
    def _box_line(self, text: str, width: int, center: bool = False) -> str:
        """Format *text* inside ``║ … ║`` with exact *width* inner cols."""
        vis = display_width(text)
        if center:
            total_pad = width - vis
            left = total_pad // 2
            right = total_pad - left
            return f"║{' ' * left}{text}{' ' * right}║"
        pad = width - 2 - vis  # 2 leading spaces
        return f"║  {text}{' ' * max(pad, 0)}║"

    def _print_eval_summary(self) -> None:
        """Print a formatted summary of the upcoming evaluation run."""
        W = 72  # inner width (between box borders)
        TOP = f"╔{'═' * W}╗"
        BOT = f"╚{'═' * W}╝"
        SEP = f"╟{'─' * W}╢"
        BLANK = f"║{' ' * W}║"

        ln = lambda text="", center=False: self._box_line(text, W, center)  # noqa: E731

        rows: list[str] = [TOP, BLANK]
        rows.append(ln("🌊  DC2 — EVALUATION SUMMARY", center=True))
        rows.append(BLANK)
        rows.append(SEP)

        # Time window
        start = getattr(self.args, "start_time", "?")
        end = getattr(self.args, "end_time", "?")
        forecast = getattr(self.args, "n_days_forecast", "?")
        interval = getattr(self.args, "n_days_interval", "?")
        rows.append(ln(f"📅  Period          {start}  →  {end}"))
        rows.append(ln(f"    Forecast        {forecast} days   ·   Interval  {interval} days"))
        rows.append(SEP)

        # Models to evaluate
        models = list(self.dataset_references.keys())
        rows.append(ln(f"🔬  Models to evaluate ({len(models)})"))
        for m in models:
            refs = self.dataset_references[m]
            rows.append(ln(f"    ▸ {m.upper()}"))
            rows.append(ln(f"      vs {len(refs)} reference(s): {', '.join(refs)}"))
        rows.append(SEP)

        # All datasets
        rows.append(ln(f"📦  Total datasets loaded: {len(self.all_datasets)}"))
        # Wrap dataset list into lines that fit the box
        max_text = W - 6  # 2 leading spaces + "    " prefix
        ds_str = ", ".join(sorted(self.all_datasets))
        while ds_str:
            chunk, ds_str = ds_str[:max_text], ds_str[max_text:]
            if ds_str and not ds_str.startswith(",") and not chunk.endswith(","):
                last_comma = chunk.rfind(",")
                if last_comma > 0:
                    ds_str = chunk[last_comma + 1:] + ds_str
                    chunk = chunk[:last_comma + 1]
            rows.append(ln(f"    {chunk.strip()}"))
        rows.append(SEP)

        # Parallelization
        batch = getattr(self.args, "batch_size", "?")
        obs_batch = getattr(self.args, "obs_batch_size", None)
        rows.append(ln(f"⚙️   Batch size       {batch}" + (f"   ·   Obs batch  {obs_batch}" if obs_batch else "")))
        rows.append(BLANK)
        rows.append(BOT)

        banner = "\n".join(rows)
        logger.opt(colors=True).info(f"\n<bold>{banner}</bold>")

    # ------------------------------------------------------------------
    # Override run_eval to display summary first
    # ------------------------------------------------------------------
    def run_eval(self) -> None:
        """Run the full evaluation pipeline with an initial summary."""
        self._print_eval_summary()
        super().run_eval()
