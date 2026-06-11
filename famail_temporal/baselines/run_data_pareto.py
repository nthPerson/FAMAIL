"""CLI: compute the data-level fairness x retention Pareto.

Loads the full corpus bundle, computes the raw point and a filtered@K sweep
(no GAN), optionally runs the existing one-shot editing pipeline for the
FAMAIL point, and writes pareto_points.json + pareto.png.

Example:
    python -m famail_temporal.baselines.run_data_pareto \
        --k-levels 100 500 1000 5000 --with-edit --edit-k 1000
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.baselines.pareto import (
    ParetoPoint, raw_point, filtered_points, edited_point, points_to_json,
)
from famail_temporal.baselines.figure import plot_pareto


def edited_point_from_result(result) -> ParetoPoint:
    """Adapt an ExperimentResult's post-edit metrics into the edit ParetoPoint."""
    return edited_point(
        f_spatial=result.f_spatial_after, f_causal=result.f_causal_after,
        gini_dsr=result.gini_dsr_after, gini_asr=result.gini_asr_after,
    )


def edited_point_from_dir(edit_dir: Path) -> ParetoPoint:
    """Build the FAMAIL edit point from a persisted editing run's metrics.json.

    Avoids re-running the editing pipeline: uses the exact post-edit metrics
    the run persisted (e.g. the canonical no-dedup k=10000 source,
    DeltaF_causal=+0.0128), so the figure and the paper quote one number.
    """
    meta = json.loads((Path(edit_dir) / "metrics.json").read_text())
    after = meta["metrics_after"]
    return edited_point(
        f_spatial=after["f_spatial"], f_causal=after["f_causal"],
        gini_dsr=after["gini_dsr"], gini_asr=after["gini_asr"],
    )


def _run_edit(edit_k: int) -> ParetoPoint:
    """Run the existing editing pipeline once for the FAMAIL point.

    Uses the validated strongest config: causal-emphasis alpha=(0.2, 0.7, 0.1)
    with unit-distinct selection (--max-per-unit 1), which achieved
    DeltaF_causal=+0.0087 at k=1000 (run
    2026-05-27T22-29-57_1000k_causal_emphasis_dedup) -- a balanced
    multi-objective that matches the pure-causal gain without gaming a
    single metric.
    """
    from famail_temporal.evaluation.runner import run_experiment
    result = run_experiment(
        config_overrides={
            "ALPHA_SPATIAL": 0.2, "ALPHA_CAUSAL": 0.7, "ALPHA_FIDELITY": 0.1,
        },
        name="data-pareto-edit",
        k=edit_k,
        max_per_unit=1,
        device="auto",
    )
    return edited_point_from_result(result)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="famail_temporal.baselines.run_data_pareto")
    ap.add_argument("--k-levels", type=int, nargs="+",
                    default=[100, 500, 1000, 5000])
    ap.add_argument("--with-edit", action="store_true",
                    help="Also run the editing pipeline for the FAMAIL point.")
    ap.add_argument("--edit-k", type=int, default=1000)
    ap.add_argument("--edit-from-dir", type=Path, default=None,
                    help="Build the FAMAIL point from a persisted editing "
                         "run's metrics.json instead of re-running the "
                         "pipeline (takes precedence over --with-edit)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(config.PACKAGE_ROOT) / "results" / "data_pareto")
    args = ap.parse_args(argv)

    bundle = DataBundle.load()
    points: List[ParetoPoint] = [raw_point(bundle)]
    points.extend(filtered_points(bundle, args.k_levels))
    if args.edit_from_dir is not None:
        points.append(edited_point_from_dir(args.edit_from_dir))
    elif args.with_edit:
        points.append(_run_edit(args.edit_k))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pareto_points.json").write_text(points_to_json(points))
    plot_pareto(points, args.out_dir / "pareto.png", metric="f_causal")
    print(f"wrote {args.out_dir / 'pareto_points.json'}")
    print(f"wrote {args.out_dir / 'pareto.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
