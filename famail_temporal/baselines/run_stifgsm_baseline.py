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
import torch

from famail_temporal import config
from famail_temporal.baselines.stifgsm_baseline import (
    attack_trajectories, package_arm,
)


# --------------------------------------------------------------- seams --------
def _resolve_device(device: str) -> str:
    """Resolve the ``auto`` sentinel to a concrete torch device string.

    ``torch.device("auto")`` raises, so the documented ``--device auto`` run-book
    would crash; resolve it here to ``cuda`` when a GPU is visible else ``cpu``.
    Any explicit value (``cpu``/``cuda``/``cuda:0``/...) passes through unchanged.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


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


# ------------------------------------------------------------ fidelity --------
def score_fidelity(arm_dir, disc, bundle, *, device, seed: int = 0,
                   pairs_per_driver: int = 20, batch_size: int = 64) -> dict:
    """Fidelity-A/B for a packaged arm dir; writes metrics.json["fidelity"].

    READ-THEN-REUSE: mirrors the established Level-1 v2 protocol
    (`run_level1_table_v2.py:508-534` pairing pass + `:536-552` gate/A and
    `:570-608` Fidelity-B), reusing its helpers verbatim-in-shape:

    - matched pairs  = (original branch, edited branch), SAME driver, built by
      `fidelity_eval.build_identity_branch` via `_build_source_pairs`
      (run_level1_table_v2.py:158) with the driver's real-context pool + profile;
    - mismatched     = original branches across DIFFERENT drivers (partner
      d' = next driver, v2's modulo protocol) — the real-anchored gate low;
    - gate matched   = (original-of-d, disjoint second real sample of d) — the
      v2 raw-source convention (run_level1_table_v2.py:481-493), falling back
      to with-replacement sampling when the pool is too small;
    - Fidelity-B     = 5-key distributional JS (`trajectory_statistics` +
      `stat_ranges` + `distributional_fidelity`, keys=_STAT_KEYS_V2) plus
      `terminal_cell_distribution_js`, edited (modified) vs original, with the
      aggregate = mean over all 6 components (v2's `_b_component`).

    The arm's originals ARE real trajectories, so both the context pool and the
    gate anchor come from `bundle.trajectories` grouped by driver (originals as
    fallback for drivers absent from the bundle). All discriminator use is
    frozen/forward-only through fidelity_eval.
    """
    import random

    from famail_temporal.baselines import fidelity_eval as fe
    from famail_temporal.baselines.gan.drivers import group_by_driver
    from famail_temporal.baselines.run_level1_table_v2 import (
        _build_source_pairs, _real_context_tensors, _terminal_pickups_from_trajs,
    )

    arm_dir = Path(arm_dir)
    # Internal Mission-3 artifact written by our own package_arm (trusted).
    with open(arm_dir / "histories.pkl", "rb") as f:
        histories = pickle.load(f)
    if not histories:
        raise ValueError(f"empty histories.pkl in {arm_dir}")

    hist_by_driver = {}
    for h in histories:
        hist_by_driver.setdefault(int(h.modified.driver_id), []).append(h)
    drivers = sorted(hist_by_driver)

    groups = group_by_driver(bundle.trajectories)
    profiles = _driver_profiles(bundle)
    zeros11 = np.zeros(11, dtype=np.float32)
    rng = random.Random(seed)

    # ---- Pass 1: per-driver slot-0 sets + context (v2 lines 456-506) ----
    real_slot0_by_d, raw_slot0_by_d, edited_slot0_by_d = {}, {}, {}
    real_context_by_d, prof_by_d = {}, {}
    for d in drivers:
        hs = hist_by_driver[d][:pairs_per_driver]
        real_pool = groups.get(d) or [h.original for h in hist_by_driver[d]]
        real_context_by_d[d] = _real_context_tensors(real_pool)
        prof = profiles.get(d)
        prof_by_d[d] = zeros11 if prof is None else prof
        real_slot0_by_d[d] = [fe.real_to_disc_tensor(h.original) for h in hs]
        edited_slot0_by_d[d] = [fe.real_to_disc_tensor(h.modified) for h in hs]
        # Gate matched partner: DISJOINT second real sample of d (v2:481-493);
        # too few disjoint -> sample WITH REPLACEMENT (degrades to overlap).
        orig_ids = {h.original.trajectory_id for h in hist_by_driver[d]}
        disjoint = [t for t in real_pool if t.trajectory_id not in orig_ids]
        n_take = len(real_slot0_by_d[d])
        if len(disjoint) >= n_take:
            raw_slot0_by_d[d] = [fe.real_to_disc_tensor(t) for t in disjoint[:n_take]]
        else:
            raw_slot0_by_d[d] = [
                fe.real_to_disc_tensor(real_pool[rng.randrange(len(real_pool))])
                for _ in range(n_take)
            ]

    # ---- Pass 2: pair d against mismatch partner d' = next driver (v2:508-534).
    # "raw" source anchors the gate; source_slot0_other for raw is d''s ORIGINAL
    # branches (originals are real), per the arm protocol's gate-anchor spec.
    matched = {"raw": [], "edited": []}
    mismatched = {"raw": [], "edited": []}
    slot0 = {"raw": raw_slot0_by_d, "edited": edited_slot0_by_d}
    other = {"raw": real_slot0_by_d, "edited": edited_slot0_by_d}
    for k, d in enumerate(drivers):
        dprime = drivers[(k + 1) % len(drivers)]
        if not real_slot0_by_d[d]:
            continue
        for name in ("raw", "edited"):
            m, mm = _build_source_pairs(
                real_slot0=real_slot0_by_d[d],
                source_slot0=slot0[name][d],
                source_slot0_other=other[name][dprime],
                real_context=real_context_by_d[d],
                source_context_other=real_context_by_d[dprime],
                profile_d=prof_by_d[d], profile_dp=prof_by_d[dprime], rng=rng,
            )
            matched[name].extend(m)
            mismatched[name].extend(mm)

    # ---- gate (real-anchored) + edited Fidelity-A (v2:536-552) ----
    gate = fe.identity_validation_gate(
        disc, matched_pairs=matched["raw"], mismatched_pairs=mismatched["raw"],
        batch_size=batch_size, device=device,
    )
    a_match = fe.humid_identity_fidelity(
        disc, matched["edited"], batch_size=batch_size, device=device)
    a_mis = fe.humid_identity_fidelity(
        disc, mismatched["edited"], batch_size=batch_size, device=device)
    fidelity_a = {
        "mean": float(a_match["mean"]),
        "std": float(a_match["std"]),
        "n": int(a_match["n"]),
        "separation": float(a_match["mean"] - a_mis["mean"]),
        "trusted": bool(gate["passed"]),
    }

    # ---- Fidelity-B: edited vs original (v2:570-608) ----
    originals = [h.original for h in histories]
    modifieds = [h.modified for h in histories]
    raw_stats = [fe.trajectory_statistics(t) for t in originals]
    edited_stats = [fe.trajectory_statistics(t) for t in modifieds]
    ranges = fe.stat_ranges([raw_stats, edited_stats], keys=fe._STAT_KEYS_V2)
    dist = fe.distributional_fidelity(
        edited_stats, raw_stats, ranges=ranges, keys=fe._STAT_KEYS_V2)
    tjs = fe.terminal_cell_distribution_js(
        _terminal_pickups_from_trajs(modifieds),
        _terminal_pickups_from_trajs(originals),
    )
    per_stat = {k: float(v) for k, v in dist["per_stat"].items()}
    fidelity_b = {
        "per_stat": per_stat,
        "terminal_cell_js": float(tjs),
        # v2's _b_component aggregate: mean over the 5 stat JS + terminal-cell JS.
        "aggregate": float(np.mean(list(per_stat.values()) + [float(tjs)])),
    }

    fidelity = {"fidelity_a": fidelity_a, "gate": gate, "fidelity_b": fidelity_b}
    meta_path = arm_dir / "metrics.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["fidelity"] = fidelity
    meta_path.write_text(json.dumps(meta, indent=2))
    return fidelity


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
    p.add_argument("--score-fidelity", action="store_true",
                   help="Also score identity Fidelity-A (+ real-anchored gate) and "
                        "discriminator-free Fidelity-B on the packaged arm, writing "
                        "metrics.json['fidelity'].")
    p.set_defaults(random_start=True)
    return p.parse_args(argv)


def run_baseline(args) -> Path:
    city = os.environ.get("FAMAIL_CITY", "shenzhen")
    device = _resolve_device(args.device)

    bundle = _load_bundle()
    disc = _load_disc(device)
    profiles = _driver_profiles(bundle)

    edit_ids = _edit_trajectory_ids(args.edit_dir)
    if args.limit is not None:
        edit_ids = edit_ids[: args.limit]
    trajs = _select_trajectories(bundle, edit_ids)

    outcomes = attack_trajectories(
        trajs, disc, profiles, args.mode,
        epsilon=args.epsilon, step=args.step, max_iterations=args.max_iterations,
        seed=args.seed, device=device, batch_size=args.batch_size,
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

    if args.score_fidelity:
        fidelity = score_fidelity(arm_dir, disc, bundle,
                                  device=device, seed=args.seed)
        print(f"[baseline] fidelity_a={fidelity['fidelity_a']['mean']:.4f} "
              f"gate_passed={fidelity['gate']['passed']} "
              f"fidelity_b={fidelity['fidelity_b']['aggregate']:.4f}")

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
