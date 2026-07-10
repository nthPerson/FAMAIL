"""Runner CLI for the Demographic Oversampling baseline (Mission-3 4th arm).

Additive semantics end-to-end: sample -> phantoms -> (D', S') -> fairness
rescore (data_level_fairness on a supply-substituted bundle — the
supply_recount-validated pattern) + external metrics
(run_external_fairness.assemble_results on additive Y vectors) -> arm dir.

The arm dir deliberately writes duplicates.pkl, NOT histories.pkl: the
substitution-semantics CLIs (run_external_fairness, supply_recount) must
fail loudly on this dir rather than silently mis-score an additive corpus.

Module seams (_load_bundle, _selected_grid, _rescore_fairness, _external)
keep the CLI unit-testable on a synthetic bundle without the real dataset
(pattern: run_stifgsm_baseline.py).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from famail_temporal import config
from famail_temporal.baselines.demographic_oversampling import (
    PLACEBO, additive_demand, additive_supply, disadvantaged_cell_masks,
    eligible_pools, escape_fractions, make_phantom, sample_duplicates,
)
from famail_temporal.baselines.stifgsm_baseline import adjacency_violation_rate


# --------------------------------------------------------------- seams --------
def _load_bundle():
    from famail_temporal.data.loader import DataBundle
    return DataBundle.load()


def _selected_grid():
    # Same in-package artifact read the external-fairness harness performs
    # (cell_demographics.pkl -> enriched EQUITY_AXES cell values).
    from famail_temporal.baselines import external_fairness_io as efio
    return efio._enriched_selected_grid()


def _rescore_fairness(bundle, D_after, S_after):
    """{f_spatial/f_causal before/after + deltas} under the additive grids.

    Supply substitution via dataclasses.replace — the exact pattern
    analysis/supply_recount.py:381-392 validated against the editing pipeline.
    """
    from dataclasses import replace
    from famail_temporal.baselines.metrics import data_level_fairness

    before = data_level_fairness(bundle)
    bundle_after = replace(
        bundle, active_taxis_3d=S_after.astype(bundle.active_taxis_3d.dtype),
    )
    after = data_level_fairness(bundle_after, pickup_3d=D_after)
    return {
        "f_spatial_before": float(before["f_spatial"]),
        "f_spatial_after": float(after["f_spatial"]),
        "f_causal_before": float(before["f_causal"]),
        "f_causal_after": float(after["f_causal"]),
        "deltas": {
            "f_spatial": float(after["f_spatial"] - before["f_spatial"]),
            "f_causal": float(after["f_causal"] - before["f_causal"]),
        },
    }


def _external(bundle, D_after, S_after, arm_dir, meta, seed, B):
    """External fairness metrics (DP/DI/SDR/Theil + bootstrap CIs) on the
    additive Y vectors, written in the harness's standard schema."""
    from famail_temporal.baselines import external_fairness_io as efio
    from famail_temporal.baselines import run_external_fairness as ref

    Y_before = efio.service_ratio_Y(bundle.pickup_3d, bundle)
    Y_after = efio.service_ratio_Y(D_after, bundle, supply_3d=S_after)
    demo = efio.per_unit_demographics(bundle)
    result = ref.assemble_results(Y_before, Y_after, demo, seed=seed, B=B)
    out = Path(arm_dir) / "external_fairness"
    ref.write_json(result, out, meta)
    (out / "report.md").write_text(ref.render_markdown(result, meta))
    return result


# ----------------------------------------------------------------- CLI --------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="famail_temporal.baselines.run_demographic_oversampling")
    ap.add_argument("--variant", choices=["targeted", PLACEBO])
    ap.add_argument("--dose", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out-root", type=Path,
                    default=Path(config.PACKAGE_ROOT) / "results")
    ap.add_argument("--summarize", nargs="+", type=Path, default=None,
                    help="Arm dirs to summarize into a dose-response table+figure")
    ap.add_argument("--out", type=Path, default=None,
                    help="--summarize output dir")
    args = ap.parse_args(argv)
    if args.summarize is None and (args.variant is None or args.dose is None):
        ap.error("--variant and --dose are required (unless --summarize)")
    return args


def run(args) -> Path:
    t0 = time.monotonic()
    bundle = _load_bundle()
    n_corpus = len(bundle.trajectories)

    masks = disadvantaged_cell_masks(_selected_grid())
    pools = eligible_pools(bundle.trajectories, masks)
    specs = sample_duplicates(pools, n_corpus, args.dose, args.seed,
                              variant=args.variant)
    n_wr = sum(1 for s in specs if s.with_replacement)
    if n_wr:
        print(f"[demo_oversample] WARNING: {n_wr} draws fell back to "
              f"with-replacement (stratum pool smaller than quota)",
              file=sys.stderr, flush=True)

    # Clip phantoms to the BUNDLE's own grid (== config.GRID_DIMS for the real
    # dataset, so production behavior is unchanged) — this is the grid
    # additive_demand/additive_supply index into, so a phantom shifted past the
    # edge must land on the boundary, not out of bounds.
    grid_dims = tuple(bundle.pickup_3d.shape[:2])
    phantoms, n_clipped = [], 0
    for spec in specs:
        ph, nc = make_phantom(bundle.trajectories[spec.source_index], spec,
                              grid_dims=grid_dims)
        phantoms.append(ph)
        n_clipped += nc

    D_after = additive_demand(bundle, phantoms)
    S_after = additive_supply(bundle, phantoms)

    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    arm_dir = (Path(args.out_root)
               / f"{ts}_baseline_demo_oversample_{args.variant}"
                 f"_d{args.dose}_s{args.seed}_{config.CITY}")
    arm_dir.mkdir(parents=True, exist_ok=True)

    with open(arm_dir / "duplicates.pkl", "wb") as f:
        pickle.dump({"specs": specs, "phantoms": phantoms}, f)

    arm = {
        "mode": f"oversample-{args.variant}-d{args.dose}",
        "variant": args.variant,
        "dose": args.dose,
        "seed": args.seed,
        "n_edited": len(phantoms),
        "n_corpus": n_corpus,
        "corpus_inflation": (args.dose / n_corpus) if n_corpus else 0.0,
        "adjacency_violation_rate": adjacency_violation_rate(phantoms),
        "per_stratum_draws": dict(Counter(s.stratum for s in specs)),
        "n_multi_axis_sources": sum(1 for s in specs if len(s.eligible_axes) > 1),
        "n_with_replacement": sum(1 for s in specs if s.with_replacement),
        "n_clipped_states": n_clipped,
        **escape_fractions(specs, phantoms, masks),
    }
    fairness = _rescore_fairness(bundle, D_after, S_after)

    meta = {"dataset": f"demo-oversample-{args.variant}-d{args.dose}-s{args.seed}",
            "city": config.CITY, "edit_dir": str(arm_dir),
            "seed": args.seed, "B": args.bootstrap}
    _external(bundle, D_after, S_after, arm_dir, meta, args.seed, args.bootstrap)

    (arm_dir / "metrics.json").write_text(json.dumps(
        {"arm": arm, "fairness": fairness,
         "runtime_s": time.monotonic() - t0}, indent=2, default=float))
    print(f"[demo_oversample] wrote {arm_dir}", flush=True)
    return arm_dir


def _arm_row(arm_dir: Path) -> dict:
    meta = json.loads((arm_dir / "metrics.json").read_text())
    ext = json.loads(
        (arm_dir / "external_fairness" / "external_fairness.json").read_text())
    mig = ext["metrics"]["MigrantRatio"]["district_extremes"]
    return {
        "mode": meta["arm"]["mode"],
        "variant": meta["arm"]["variant"],
        "dose": meta["arm"]["dose"],
        "seed": meta["arm"]["seed"],
        "corpus_inflation": meta["arm"].get("corpus_inflation"),
        "d_f_causal": meta["fairness"]["deltas"]["f_causal"],
        "d_f_spatial": meta["fairness"]["deltas"]["f_spatial"],
        "d_dp_migrant": mig["demographic_parity"]["delta"],
        "d_di_migrant": mig["disparate_impact"]["delta"],
        "d_theil": ext["theil"]["delta"],
    }


def summarize_arms(arm_dirs) -> str:
    rows = sorted((_arm_row(Path(d)) for d in arm_dirs),
                  key=lambda r: (r["variant"], r["dose"], r["seed"]))
    lines = [
        "# Demographic Oversampling — dose-response summary", "",
        "| Arm | seed | inflation | ΔF_causal | ΔF_spatial | ΔDP (migrant/extremes) "
        "| ΔDI (migrant/extremes) | ΔTheil |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['seed']} | {r['corpus_inflation']:.3f} "
            f"| {r['d_f_causal']:+.4f} | {r['d_f_spatial']:+.4f} "
            f"| {r['d_dp_migrant']:+.4f} | {r['d_di_migrant']:+.4f} "
            f"| {r['d_theil']:+.4f} |")
    return "\n".join(lines)


def _dose_figure(arm_dirs, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [_arm_row(Path(d)) for d in arm_dirs]
    fig, ax = plt.subplots(figsize=(6, 4))
    for variant, marker in (("targeted", "o"), (PLACEBO, "s")):
        pts = sorted((r for r in rows if r["variant"] == variant),
                     key=lambda r: r["dose"])
        if pts:
            ax.plot([r["dose"] for r in pts], [r["d_f_causal"] for r in pts],
                    marker=marker, label=f"{variant} ΔF_causal")
            ax.plot([r["dose"] for r in pts], [r["d_dp_migrant"] for r in pts],
                    marker=marker, ls="--", label=f"{variant} ΔDP migrant")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("dose (duplicates)")
    ax.set_ylabel("Δ (after − before)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.summarize:
        out = args.out or Path(config.PACKAGE_ROOT) / "baselines" / \
            "demographic_oversampling_results"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.md").write_text(summarize_arms(args.summarize))
        _dose_figure(args.summarize, out / "dose_response.png")
        print(f"[demo_oversample] wrote {out / 'summary.md'}", flush=True)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
