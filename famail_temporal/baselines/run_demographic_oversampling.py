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
    ap.add_argument("--variant", choices=["targeted", PLACEBO], required=True)
    ap.add_argument("--dose", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out-root", type=Path,
                    default=Path(config.PACKAGE_ROOT) / "results")
    return ap.parse_args(argv)


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


def main(argv=None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
