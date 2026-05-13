"""
Fetch source datasets, raw GPS, and the discriminator checkpoint from the
project's public HuggingFace dataset:

    https://huggingface.co/datasets/nthPerson/famail-temporal-data

The dataset is public — no token required.

Usage:
    python -m famail_temporal.fetch_data
    python -m famail_temporal.fetch_data --skip-raw   # 200 MB instead of 600 MB
    python -m famail_temporal.fetch_data --repo-id other/repo   # override default

The HuggingFace dataset has this layout:

    <repo>/
      source_data/                       -> famail_temporal/source_data/
      discriminator_checkpoints/         -> famail_temporal/discriminator_checkpoints/
      raw_data/                          -> <repo-root>/raw_data/
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from famail_temporal import config

DEFAULT_REPO_ID = "nthPerson/famail-temporal-data"


def _move_tree(src: Path, dst: Path) -> int:
    if not src.exists():
        print(f"[fetch_data] '{src.name}/' not in dataset; skipping.", flush=True)
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(target))
        n += 1
    print(f"[fetch_data] Placed {n} file(s) under {dst}", flush=True)
    return n


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download famail_temporal data from a HuggingFace dataset.",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("FAMAIL_DATA_REPO", DEFAULT_REPO_ID),
        help=f"HF dataset repo (default: {DEFAULT_REPO_ID}; override with "
             "$FAMAIL_DATA_REPO or this flag).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF access token (or set $HF_TOKEN). Not required for the public "
             "dataset, but useful if you hit anonymous rate limits.",
    )
    parser.add_argument(
        "--raw-data-dir",
        default=str(config.PACKAGE_ROOT.parent / "raw_data"),
        help="Where to put raw GPS files (default: <repo-root>/raw_data).",
    )
    parser.add_argument(
        "--skip-raw", action="store_true",
        help="Skip raw_data/ download (~418 MB). Only fetch source_data + "
             "discriminator checkpoint.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[fetch_data] ERROR: huggingface_hub is not installed. Run:\n"
            "    pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    source_data_dir = config.SOURCE_DATA_DIR
    checkpoint_dir = config.DISCRIMINATOR_CHECKPOINT_DIR
    raw_data_dir = Path(args.raw_data_dir)

    allow_patterns = ["source_data/*", "discriminator_checkpoints/**"]
    if not args.skip_raw:
        allow_patterns.append("raw_data/*")

    staging = config.PACKAGE_ROOT / ".fetch_staging"
    if staging.exists():
        shutil.rmtree(staging)

    print(f"[fetch_data] Downloading from {args.repo_id} ...", flush=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        token=args.token,
        local_dir=str(staging),
        allow_patterns=allow_patterns,
    )

    _move_tree(staging / "source_data", source_data_dir)
    _move_tree(staging / "discriminator_checkpoints", checkpoint_dir)
    if not args.skip_raw:
        _move_tree(staging / "raw_data", raw_data_dir)

    shutil.rmtree(staging, ignore_errors=True)
    print("[fetch_data] Done. Run `python -m famail_temporal.preprocess` next.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
