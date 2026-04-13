#!/usr/bin/env python3
"""Pack docs/source/_extra/leaderboard/map_data/ into a tar.gz archive.

Usage
-----
    python docs/scripts/pack_map_data.py            # create archive
    python docs/scripts/pack_map_data.py --info      # show stats only

The archive is written to:
    docs/source/_extra/leaderboard/map_data.tar.gz

It is then uploaded to a GitHub Release via the GitHub Actions workflow
(.github/workflows/docs.yml) and downloaded by ReadTheDocs during its
pre-build step (see .readthedocs.yaml).
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

LEADERBOARD_DIR = Path(__file__).resolve().parents[1] / "source" / "_extra" / "leaderboard"
MAP_DATA_DIR = LEADERBOARD_DIR / "map_data"
ARCHIVE_PATH = LEADERBOARD_DIR / "map_data.tar.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack map_data into tar.gz")
    parser.add_argument("--info", action="store_true", help="Print stats without creating archive")
    args = parser.parse_args()

    if not MAP_DATA_DIR.is_dir():
        print(f"ERROR: {MAP_DATA_DIR} does not exist.", file=sys.stderr)
        sys.exit(1)

    files = sorted(p for p in MAP_DATA_DIR.rglob("*") if p.is_file())
    total_size = sum(f.stat().st_size for f in files)
    print(f"Files:    {len(files)}")
    print(f"Raw size: {total_size / 1e9:.2f} GB")

    if args.info:
        if ARCHIVE_PATH.exists():
            print(f"Archive:  {ARCHIVE_PATH.stat().st_size / 1e6:.1f} MB  ({ARCHIVE_PATH})")
        else:
            print("Archive:  not yet created")
        return

    print(f"Compressing to {ARCHIVE_PATH} ...")
    with tarfile.open(ARCHIVE_PATH, "w:gz", compresslevel=6) as tar:
        for i, f in enumerate(files, 1):
            arcname = f"map_data/{f.relative_to(MAP_DATA_DIR).as_posix()}"
            tar.add(f, arcname=arcname)
            if i % 2000 == 0:
                print(f"  {i}/{len(files)} ...")

    archive_size = ARCHIVE_PATH.stat().st_size
    ratio = total_size / archive_size if archive_size else 0
    print(f"Done: {archive_size / 1e6:.1f} MB  (ratio {ratio:.1f}x)")
    print(f"\nNext steps:")
    print(f"  1. Upload to GitHub Release:")
    print(f"     gh release create leaderboard-data --title 'Leaderboard map data' --latest=false || true")
    print(f"     gh release upload leaderboard-data {ARCHIVE_PATH} --clobber")
    print(f"  2. Or trigger the GitHub Actions workflow with 'Upload map_data.tar.gz' checked.")


if __name__ == "__main__":
    main()
