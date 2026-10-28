#!/usr/bin/env python3
"""Rebuild Edito catalogs from Wasabi catalogs.

Downloads all catalog JSON files from the Wasabi S3 bucket, replaces the
Wasabi bucket name (``ppr-ocean-climat``) with the Edito bucket name
(``project-ppr-ocean-ia``) in every ``path`` field, then saves the results
locally under ``dc2_output/catalogs/``.

The argo_index monthly files (``*.json.zst``) are copied as-is because they
contain Argo profile identifiers, not S3 URIs.

Usage::

    python scripts/rebuild_edito_catalogs.py [--output-dir dc2_output/catalogs]

After running this script, re-run the evaluation with ``dc2_edito.yaml``.
The local catalog files will be picked up automatically (the loader checks
for local copies before downloading from S3).
"""

import argparse
import json
import os
from pathlib import Path

import s3fs

# ---------------------------------------------------------------------------
# Wasabi connection parameters (read-only, as defined in dc2_wasabi.yaml)
# ---------------------------------------------------------------------------
WASABI_KEY = "A10PTBP92GOXQW8OPE8G"
WASABI_SECRET = "1C1Iscst7rfDnXp5eM8DtL7AkFiUAycINzctMEGQ"
WASABI_ENDPOINT = "https://s3.eu-west-2.wasabisys.com"
WASABI_BUCKET = "ppr-ocean-climat"
WASABI_CATALOG_FOLDER = "catalogs/Wasabi"

# ---------------------------------------------------------------------------
# Edito path substitution
# ---------------------------------------------------------------------------
EDITO_BUCKET = "project-ppr-ocean-ia"

# Prefix to replace in every catalog ``path`` value
WASABI_PATH_PREFIX = f"s3://{WASABI_BUCKET}/"
EDITO_PATH_PREFIX = f"s3://{EDITO_BUCKET}/"


def replace_paths_in_catalog(data: dict) -> tuple[dict, int]:
    """Return a copy of *data* with Wasabi S3 paths replaced by Edito paths.

    Returns the modified catalog dict and the number of replacements made.
    """
    replacements = 0
    features = data.get("features", [])
    for feature in features:
        props = feature.get("properties", {})
        path = props.get("path", "")
        if isinstance(path, str) and path.startswith(WASABI_PATH_PREFIX):
            props["path"] = path.replace(WASABI_PATH_PREFIX, EDITO_PATH_PREFIX, 1)
            replacements += 1
    return data, replacements


def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    argo_index_dir = output_dir / "argo_index"
    argo_index_dir.mkdir(parents=True, exist_ok=True)

    fs = s3fs.S3FileSystem(
        key=WASABI_KEY,
        secret=WASABI_SECRET,
        endpoint_url=WASABI_ENDPOINT,
        anon=False,
    )

    remote_catalog_root = f"{WASABI_BUCKET}/{WASABI_CATALOG_FOLDER}"

    # ------------------------------------------------------------------
    # 1. List all files in the Wasabi catalog folder (non-recursive)
    # ------------------------------------------------------------------
    all_items = fs.ls(remote_catalog_root, detail=True)

    json_files = [
        item for item in all_items
        if item["type"] == "file" and item["name"].endswith(".json")
    ]
    subdirs = [
        item for item in all_items
        if item["type"] == "directory"
    ]

    # ------------------------------------------------------------------
    # 2. Process JSON catalog files
    # ------------------------------------------------------------------
    for item in json_files:
        remote_path = item["name"]
        filename = Path(remote_path).name
        local_path = output_dir / filename

        print(f"Downloading  s3://{remote_path} ...", end=" ", flush=True)
        with fs.open(remote_path, "rb") as fh:
            data = json.load(fh)

        features_before = len(data.get("features", []))
        data, n_replaced = replace_paths_in_catalog(data)

        with open(local_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

        print(
            f"saved -> {local_path}  "
            f"({features_before} features, {n_replaced} paths replaced)"
        )

    # ------------------------------------------------------------------
    # 3. Copy argo_index files (no path substitution needed)
    # ------------------------------------------------------------------
    for subdir in subdirs:
        subdir_name = Path(subdir["name"]).name  # e.g. "argo_index"
        local_subdir = output_dir / subdir_name
        local_subdir.mkdir(parents=True, exist_ok=True)

        sub_files = fs.ls(subdir["name"], detail=True)
        for sub_item in sub_files:
            if sub_item["type"] != "file":
                continue
            remote_sub = sub_item["name"]
            # Skip S3 "directory marker" objects (zero-byte objects whose name
            # ends with "/" or matches the folder name exactly).
            if remote_sub.rstrip("/") == subdir["name"].rstrip("/"):
                continue
            sub_filename = Path(remote_sub).name
            local_sub_path = local_subdir / sub_filename
            size_kb = sub_item.get("size", 0) // 1024

            print(
                f"Copying      s3://{remote_sub} ({size_kb} KB) ...",
                end=" ",
                flush=True,
            )
            with fs.open(remote_sub, "rb") as fh:
                content = fh.read()
            with open(local_sub_path, "wb") as fh:
                fh.write(content)
            print(f"saved -> {local_sub_path}")

    print("\nDone. Edito catalogs written to:", output_dir)
    print(
        f"\nPath substitution applied:  {WASABI_PATH_PREFIX!r}"
        f"  ->  {EDITO_PATH_PREFIX!r}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "dc2_output" / "catalogs",
        help="Local directory where catalog files will be written "
             "(default: dc2_output/catalogs relative to project root)",
    )
    args = parser.parse_args()
    main(args.output_dir)
