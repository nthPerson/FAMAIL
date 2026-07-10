"""Runner CLI for the Mission-3 baseline editors (vanilla ST-iFGSM/FGSM/random).

Wires Tasks 1-2 (`attack_trajectories` + `package_arm`) into an end-to-end run:
load the frozen HuMID discriminator and the city bundle, take the headline
edit set's trajectory ids from an `--edit-dir` `histories.pkl`, attack the
matching bundle trajectories, package the arm, then RESCORE data-level fairness
before vs after using the SAME path the editing pipeline uses
(`metrics.data_level_fairness` + `external_fairness_io.build_edited_pickup_3d`),
writing the deltas into the arm's `metrics.json["fairness"]`.

The four module-level seams (`_load_bundle`, `_load_disc`, `_driver_profiles`,
`_rescore`) isolate the heavy real loads so the CLI is unit-testable on a
synthetic bundle without touching a GPU, a real checkpoint, or the dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from famail_temporal import config
from famail_temporal.baselines.stifgsm_baseline import (
    attack_trajectories, package_arm,
)


# --------------------------------------------------------------- seams --------
def _load_bundle():
    from famail_temporal.data.loader import DataBundle
    return DataBundle.load()


def _load_disc(device):
    from famail_temporal.fidelity.checkpoint import load_discriminator
    path = config.DISCRIMINATOR_CHECKPOINT_DIR / config.DISCRIMINATOR_CHECKPOINT_FILENAME
    return load_discriminator(path).to(device)


def _driver_profiles(bundle):
    """{driver_id: np.ndarray (11,)} from the SAME source HuMID's identity
    branches use.

    `fidelity_eval.build_identity_branch` (fidelity_eval.py:361) takes the
    profile as an injected argument; its callers
    (`run_level1_table_v2.py:396`, `run_level2_table.py:371`) source it from
    `bundle.multi_stream.profile_features` — the per-driver 11-dim z-scored
    profile artifact HuMID was trained on, keyed by original driver id. We
    return that mapping directly (dict copy so callers can't mutate the
    bundle's frozen artifact)."""
    return dict(bundle.multi_stream.profile_features)


def _rescore(bundle, arm_dir):
    """Data-level fairness before vs after the arm's edits.

    Uses the established pickup-grid -> fairness path (the one `run_data_pareto`
    / `run_level1_table_v2` use): `metrics.data_level_fairness(bundle)` on the
    bundle's own demand grid (before), and on the after-edit demand grid from
    `external_fairness_io.build_edited_pickup_3d(bundle, arm_dir)` (after).
    Returns the flat {f_spatial_before/after, f_causal_before/after} dict the
    CLI writes into metrics.json["fairness"]."""
    from famail_temporal.baselines.external_fairness_io import build_edited_pickup_3d
    from famail_temporal.baselines.metrics import data_level_fairness

    before = data_level_fairness(bundle)
    after = data_level_fairness(bundle, pickup_3d=build_edited_pickup_3d(bundle, arm_dir))
    return {
        "f_spatial_before": float(before["f_spatial"]),
        "f_spatial_after": float(after["f_spatial"]),
        "f_causal_before": float(before["f_causal"]),
        "f_causal_after": float(after["f_causal"]),
    }


# ----------------------------------------------------------- edit-set io ------
def _edit_trajectory_ids(edit_dir):
    """Ordered, deduped `h.original.trajectory_id` from an edit-dir histories.

    Reads an internal Mission-3 pipeline artifact (a `List[ModificationHistory]`
    pickle produced by this codebase's own editor / Task-2 `package_arm`), not
    untrusted external data."""
    with open(Path(edit_dir) / "histories.pkl", "rb") as f:
        histories = pickle.load(f)
    ids, seen = [], set()
    for h in histories:
        tid = h.original.trajectory_id
        if tid not in seen:
            seen.add(tid)
            ids.append(tid)
    return ids


def _select_trajectories(bundle, edit_ids):
    """Bundle trajectories matching `edit_ids`, in edit-set order."""
    by_id = {t.trajectory_id: t for t in bundle.trajectories}
    return [by_id[tid] for tid in edit_ids if tid in by_id]


# ------------------------------------------------------------------ cli -------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run a Mission-3 baseline editor (ifgsm/fgsm/random) + rescore.")
    p.add_argument("--edit-dir", required=True,
                   help="Headline edit dir; its histories.pkl supplies the trajectory ids.")
    p.add_argument("--mode", required=True, choices=("ifgsm", "fgsm", "random"))
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--step", type=float, default=None)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out-root", default=str(config.RESULTS_DIR)
                   if hasattr(config, "RESULTS_DIR") else "famail_temporal/results")
    p.add_argument("--limit", type=int, default=None,
                   help="Truncate the edit set to the first N trajectories (smoke runs).")
    p.add_argument("--no-random-start", dest="random_start", action="store_false",
                   help="Use textbook-vanilla iFGSM/FGSM init (delta=0) instead of "
                        "PGD-style random start; ignored by mode=random.")
    p.set_defaults(random_start=True)
    return p.parse_args(argv)


def run_baseline(args) -> Path:
    city = os.environ.get("FAMAIL_CITY", "shenzhen")

    bundle = _load_bundle()
    disc = _load_disc(args.device)
    profiles = _driver_profiles(bundle)

    edit_ids = _edit_trajectory_ids(args.edit_dir)
    if args.limit is not None:
        edit_ids = edit_ids[: args.limit]
    trajs = _select_trajectories(bundle, edit_ids)

    outcomes = attack_trajectories(
        trajs, disc, profiles, args.mode,
        epsilon=args.epsilon, step=args.step, max_iterations=args.max_iterations,
        seed=args.seed, device=args.device, batch_size=args.batch_size,
        random_start=args.random_start,
    )

    ts = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
    arm_dir = Path(args.out_root) / f"{ts}_baseline_{args.mode}_{city}"
    arm_config = {
        "mode": args.mode,
        "city": city,
        "epsilon": args.epsilon,
        "step": args.step,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "random_start": args.random_start,
    }
    package_arm(trajs, outcomes, arm_dir, arm_config)

    # Rescore data-level fairness before vs after and fold into metrics.json.
    fairness = _rescore(bundle, arm_dir)
    fairness["deltas"] = {
        "f_spatial": fairness["f_spatial_after"] - fairness["f_spatial_before"],
        "f_causal": fairness["f_causal_after"] - fairness["f_causal_before"],
    }
    meta_path = arm_dir / "metrics.json"
    meta = json.loads(meta_path.read_text())
    meta["fairness"] = fairness
    meta_path.write_text(json.dumps(meta, indent=2))

    d_causal = fairness["deltas"]["f_causal"]
    d_spatial = fairness["deltas"]["f_spatial"]
    adj = meta["arm"]["adjacency_violation_rate"]
    mean_p = meta["arm"]["mean_final_p"]
    print(f"[baseline] mode={args.mode} n={len(trajs)} "
          f"dF_causal={d_causal:+.4f} dF_spatial={d_spatial:+.4f} "
          f"adjacency_rate={adj:.3f} mean_final_p={mean_p:.4f}")
    return arm_dir


def main(argv=None):
    return run_baseline(parse_args(argv))


if __name__ == "__main__":
    main()
