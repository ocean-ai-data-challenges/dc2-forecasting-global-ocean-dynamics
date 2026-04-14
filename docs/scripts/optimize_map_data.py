#!/usr/bin/env python3
"""Optimize map_data for ReadTheDocs: remove temporal snapshots & reduce precision.

1. Deletes all .js files containing a date pattern (2024-MM-DD) in their name.
   These are weekly temporal snapshots; the aggregated (dateless) files per
   lead-day are preserved.
2. Rounds all numeric values to 3 decimal places in the remaining .js files.

Usage
-----
    python docs/scripts/optimize_map_data.py            # apply optimizations
    python docs/scripts/optimize_map_data.py --dry-run   # show what would happen
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

MAP_DATA_DIR = (
    Path(__file__).resolve().parents[1] / "source" / "_extra" / "leaderboard" / "map_data"
)

# Matches files with a date like 2024-01-03 in their name
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")

# Matches numeric literals (int or float) in JSONP content
NUMBER_RE = re.compile(r"-?\d+\.\d{4,}")


def round_match(m: re.Match) -> str:
    return f"{float(m.group()):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize map_data for RTD")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without modifying files")
    args = parser.parse_args()

    if not MAP_DATA_DIR.is_dir():
        print(f"ERROR: {MAP_DATA_DIR} does not exist.", file=sys.stderr)
        sys.exit(1)

    all_js = sorted(MAP_DATA_DIR.glob("*.js"))
    dated = [f for f in all_js if DATE_PATTERN.search(f.name)]
    keep = [f for f in all_js if not DATE_PATTERN.search(f.name)]

    dated_size = sum(f.stat().st_size for f in dated)
    keep_size = sum(f.stat().st_size for f in keep)

    print(f"Total .js files:    {len(all_js)}")
    print(f"  Dated (to remove): {len(dated)}  ({dated_size / 1e9:.2f} GB)")
    print(f"  Aggregated (keep): {len(keep)}  ({keep_size / 1e6:.1f} MB)")

    if args.dry_run:
        print("\n[dry-run] No files modified.")
        return

    # Step 1: Remove dated files
    print(f"\nRemoving {len(dated)} dated files ...")
    for f in dated:
        f.unlink()
    print("  Done.")

    # Step 2: Reduce precision in remaining files
    print(f"Reducing precision in {len(keep)} files ...")
    saved_bytes = 0
    for f in keep:
        original = f.read_text()
        optimized = NUMBER_RE.sub(round_match, original)
        if optimized != original:
            saved_bytes += len(original) - len(optimized)
            f.write_text(optimized)
    print(f"  Done. Saved {saved_bytes / 1e6:.1f} MB from precision reduction.")

    # Final stats
    remaining = sorted(MAP_DATA_DIR.glob("*.js"))
    final_size = sum(f.stat().st_size for f in remaining)
    print(f"\nFinal: {len(remaining)} files, {final_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
