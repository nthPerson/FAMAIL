"""E31 — Dataset cleanup summary: dirty vs clean removal statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from famail_temporal.analysis import _io


def dataset_summary(dirty_meta: dict, clean_meta: dict) -> dict:
    """Pure transform: compare dirty vs clean processing_metadata dicts.

    Returns a dict with keys 'dirty', 'clean', 'delta' each containing
    the relevant removal statistics.
    """
    dirty_rs = dirty_meta["removal_summary"]
    clean_rs = clean_meta["removal_summary"]
    sinks = clean_meta.get("stuck_gps_sinks", {})

    flagged = sinks.get("flagged_cells", [])
    phantom = sinks.get("n_pickups_removed", 0)

    dirty_block = {
        "n_removed": dirty_rs["n_removed"],
        "removal_rate": dirty_rs["removal_rate"],
        "total_seeking_extracted": dirty_rs["total_seeking_extracted"],
    }
    clean_block = {
        "n_removed": clean_rs["n_removed"],
        "removal_rate": clean_rs["removal_rate"],
        "total_seeking_extracted": clean_rs["total_seeking_extracted"],
        "n_sink_cells": len(flagged),
        "phantom_pickups_removed": phantom,
    }
    delta_block = {
        "n_removed": clean_rs["n_removed"] - dirty_rs["n_removed"],
        "removal_rate": round(clean_rs["removal_rate"] - dirty_rs["removal_rate"], 4),
        "total_seeking_extracted": (
            clean_rs["total_seeking_extracted"] - dirty_rs["total_seeking_extracted"]
        ),
    }
    return {"dirty": dirty_block, "clean": clean_block, "delta": delta_block}


def write_dataset_summary(
    dirty_source_dir: str | Path,
    clean_source_dir: str | Path,
    out_dir: str | Path,
) -> Path:
    """Load real on-disk metadata, compute summary, write JSON + Markdown."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirty_meta = _io.processing_metadata(dirty_source_dir)
    clean_meta = _io.processing_metadata(clean_source_dir)

    summary = dataset_summary(dirty_meta, clean_meta)

    json_path = out_dir / "dataset_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    md_path = out_dir / "dataset_summary.md"
    d = summary["dirty"]
    c = summary["clean"]
    delta = summary["delta"]
    md = (
        "# Dataset Cleanup Summary (E31)\n\n"
        "| Metric | Dirty | Clean | Delta |\n"
        "|--------|-------|-------|-------|\n"
        f"| n_removed | {d['n_removed']:,} | {c['n_removed']:,} | {delta['n_removed']:,} |\n"
        f"| removal_rate | {d['removal_rate']:.4f} | {c['removal_rate']:.4f} | {delta['removal_rate']:.4f} |\n"
        f"| total_seeking_extracted | {d['total_seeking_extracted']:,} | {c['total_seeking_extracted']:,} | {delta['total_seeking_extracted']:,} |\n"
        f"| n_sink_cells | — | {c['n_sink_cells']} | — |\n"
        f"| phantom_pickups_removed | — | {c['phantom_pickups_removed']:,} | — |\n"
    )
    md_path.write_text(md)

    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="E31: dataset cleanup summary")
    parser.add_argument("--dirty-source", required=True, help="Path to source_data_dirty/")
    parser.add_argument("--clean-source", required=True, help="Path to source_data/")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    out = write_dataset_summary(args.dirty_source, args.clean_source, args.out_dir)
    print(f"Written: {out.parent}")

    # Print the summary for inspection
    summary = json.loads(out.read_text())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
