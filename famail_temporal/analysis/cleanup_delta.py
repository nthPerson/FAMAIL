"""E22 (editor-level) — Editor dirty-vs-clean fairness delta."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from famail_temporal.analysis import _io

_METRICS = ["f_spatial", "f_causal", "gini_dsr", "gini_asr"]


def editor_delta(dirty_metrics: dict, clean_metrics: dict) -> dict:
    """Pure transform: compare two editor metrics.json dicts.

    For each metric, returns:
      - dirty_before, dirty_after, dirty_edit_delta
      - clean_before, clean_after, clean_edit_delta
      - baseline_shift_dirty_to_clean  (clean.before - dirty.before)
      - after_shift                    (clean.after  - dirty.after)
      - edit_delta_shift               (clean.delta  - dirty.delta)
    """
    result = {}
    for metric in _METRICS:
        db = dirty_metrics["metrics_before"].get(metric)
        da = dirty_metrics["metrics_after"].get(metric)
        dd = dirty_metrics["deltas"].get(metric)
        cb = clean_metrics["metrics_before"].get(metric)
        ca = clean_metrics["metrics_after"].get(metric)
        cd = clean_metrics["deltas"].get(metric)

        if any(v is None for v in [db, da, dd, cb, ca, cd]):
            continue

        result[metric] = {
            "dirty_before": db,
            "dirty_after": da,
            "dirty_edit_delta": dd,
            "clean_before": cb,
            "clean_after": ca,
            "clean_edit_delta": cd,
            "baseline_shift_dirty_to_clean": cb - db,
            "after_shift": ca - da,
            "edit_delta_shift": cd - dd,
        }
    return result


def write_editor_cleanup_delta(
    dirty_run_dir: str | Path,
    clean_run_dir: str | Path,
    out_dir: str | Path,
) -> Path:
    """Load editor metrics from both runs, compute delta, write CSV + JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirty_metrics = _io.editor_metrics(dirty_run_dir)
    clean_metrics = _io.editor_metrics(clean_run_dir)

    delta = editor_delta(dirty_metrics, clean_metrics)

    json_path = out_dir / "cleanup_delta_editor.json"
    json_path.write_text(json.dumps(delta, indent=2))

    csv_path = out_dir / "cleanup_delta_editor.csv"
    fieldnames = [
        "metric",
        "dirty_before", "dirty_after", "dirty_edit_delta",
        "clean_before", "clean_after", "clean_edit_delta",
        "baseline_shift_dirty_to_clean", "after_shift", "edit_delta_shift",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for metric, row in delta.items():
            writer.writerow({"metric": metric, **row})

    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="E22 editor-level cleanup delta")
    parser.add_argument("--dirty-run", required=True, help="Path to dirty editor run dir")
    parser.add_argument("--clean-run", required=True, help="Path to clean editor run dir")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    out = write_editor_cleanup_delta(args.dirty_run, args.clean_run, args.out_dir)
    print(f"Written: {out.parent}")

    delta = json.loads(out.read_text())
    print(json.dumps(delta, indent=2))


if __name__ == "__main__":
    main()
