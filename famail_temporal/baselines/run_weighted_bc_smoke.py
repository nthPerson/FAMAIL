"""Weighted-BC sweep: does upweighting the edited demonstrations during BC let
the data-level fairness edge survive training -- without wrecking fidelity?

Level-2 found the edited-data F_causal edge does NOT transfer through a
driver-conditioned BC policy (paired edited-raw F_causal -0.0022, 5/5 seeds).
The 1-seed pilot (run_weighted_bc_smoke at w=30) flipped that to +0.0268,
implying the bottleneck is BC's flat per-token MLE mean averaging away the
~3.6% edited trajectories -- NOT the 1/N metric wall. This script confirms it
at scale WITH fidelity guardrails: it trains the same TrajectoryLSTM on the
edited corpus with the edited trajectories' per-sequence loss upweighted
(train_mle sample_weights), across paired seeds and weight doses, and scores
every policy on the full Level-1 axes (F_causal, F_spatial, identity
Fidelity-A with the real-anchored gate, enriched Fidelity-B) by REUSING the
Level-2 evaluator.

Arms: raw, edited (w=1, must ~reproduce the L2 -0.0022 baseline), and one
edited_wK arm per --weights dose. Editor, metrics, and the locked L2 table are
untouched; this lives entirely in the BC trainer + a parallel runner.
"""
from __future__ import annotations
import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.utils.seeding import set_all_seeds
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.drivers import build_driver_index, group_by_driver
from famail_temporal.baselines.gan.train_mle import train_mle
from famail_temporal.fidelity.checkpoint import load_discriminator
from famail_temporal.baselines import fidelity_eval as fe
from famail_temporal.baselines._manifest import write_run_manifest, append_timing, sha256_file
from famail_temporal.baselines.run_level2_table import (
    build_edited_corpus, traj_training_data, _evaluate_policy,
)
from famail_temporal.baselines import _enrich
from famail_temporal.baselines.run_level1_table_v2 import (
    _select_eval_drivers, _real_context_tensors, _build_source_pairs,
    _terminal_pickups_from_trajs,
)

_METRICS = ("f_causal", "f_spatial", "fidelity_a", "fidelity_b")


def edited_ids(histories) -> set:
    """trajectory_ids of the trajectories the editor actually modified."""
    return {int(h.original.trajectory_id) for h in histories}


def weight_vector(trajs, ids: set, w: float) -> List[float]:
    """w for trajectories whose id was edited, 1.0 for the rest (index-aligned)."""
    return [float(w) if int(t.trajectory_id) in ids else 1.0 for t in trajs]


def random_subset_weight_vector(
    trajs, edited_id_set: set, w: float, *, k: int | None = None, seed: int = 12345,
) -> List[float]:
    """Placebo control: w on a random, size-matched subset of the NON-edited
    trajectories, 1.0 everywhere else (index-aligned).

    This is the decisive control for the weighted-BC fairness claim. The edited
    arm upweights the ~3.6% EDITED trajectories and F_causal rises; a reviewer
    can object that upweighting *any* small subset reshapes the effective
    training distribution and can move a 1/N global metric. Applying this vector
    to the RAW corpus tests exactly that: if F_causal does NOT rise, the gain is
    edit-specific (not an oversampling artifact).

    ``k`` defaults to ``len(edited_id_set)`` so the placebo subset is the same
    size as the edited set. The subset is drawn with an INDEPENDENT RNG
    (``random.Random(seed)``) so it never touches the global torch/numpy/random
    state that ``set_all_seeds`` controls -- otherwise it would perturb the
    paired-seed training determinism and the raw/edited arms would stop
    reproducing the locked Level-2 baseline. Build it ONCE before the seed loop.
    """
    k = len(edited_id_set) if k is None else k
    non_edited = [
        i for i, t in enumerate(trajs) if int(t.trajectory_id) not in edited_id_set
    ]
    if k > len(non_edited):
        raise ValueError(
            f"requested random subset of {k} > {len(non_edited)} non-edited trajs"
        )
    chosen = set(random.Random(seed).sample(non_edited, k))
    return [float(w) if i in chosen else 1.0 for i in range(len(trajs))]


def _paired_vs_raw(per_seed: Dict[str, List[float]], arm: str) -> dict:
    """Paired per-seed (arm - raw) stats for one metric."""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    diffs = [float(a - r) for a, r in zip(per_seed[arm], per_seed["raw"])]
    n = len(diffs)
    mean = float(np.mean(diffs)) if n else float("nan")
    std = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    p = None
    if wilcoxon is not None and n >= 1 and any(d != 0.0 for d in diffs):
        try:
            p = float(wilcoxon(diffs).pvalue)
        except Exception:
            p = None
    return {"diffs": [round(d, 4) for d in diffs], "mean": mean, "std": std,
            "n": n, "wilcoxon_p": p}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="famail_temporal.baselines.run_weighted_bc_smoke")
    ap.add_argument(
        "--edit-dir", type=str,
        default="famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup",
    )
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--weights", type=str, default="10,30",
                    help="Comma-separated upweight doses for the edited arms (w=1 always run).")
    ap.add_argument("--placebo", type=str, default="",
                    help="Comma-separated doses for the random-subset PLACEBO arms: "
                         "upweight a random, size-matched NON-edited subset of the RAW "
                         "corpus. Decisive control -- if F_causal does NOT rise here, the "
                         "edited arms' gain is edit-specific, not an oversampling artifact. "
                         "Empty (default) = no placebo arms (preserves prior sweeps).")
    ap.add_argument("--placebo-seed", type=int, default=12345,
                    help="Fixed RNG seed for the placebo subset, INDEPENDENT of the per-seed "
                         "training RNG: one fixed subset reused across all training seeds, so "
                         "the placebo arm differs from raw only by the fixed random weighting.")
    ap.add_argument("--mle-epochs", type=int, default=20)
    ap.add_argument("--max-eval-drivers", type=int, default=50)
    ap.add_argument("--pairs-per-driver", type=int, default=20)
    ap.add_argument("--min-driver-trajs", type=int, default=6)
    ap.add_argument("--fidelity-sample-size", type=int, default=5000)
    ap.add_argument("--gen-batch-size", type=int, default=gc.GEN_BATCH_SIZE)
    ap.add_argument("--max-batch-tokens", type=int,
                    default=gc.MLE_BATCH_SIZE * gc.MAX_TRAIN_TOKENS)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    t0_run = time.time()

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    up_weights = [float(w) for w in str(args.weights).split(",") if w.strip()]
    placebo_weights = [float(w) for w in str(args.placebo).split(",") if w.strip()]
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    max_len = gc.MAX_GEN_LEN
    fss = args.fidelity_sample_size

    # ---- discriminator (Fidelity-A) checkpoint guard before any expensive work ----
    ckpt = Path(config.PACKAGE_ROOT) / "discriminator_checkpoints" / "default" / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"discriminator checkpoint not found: {ckpt}")
    disc = load_discriminator(ckpt).to(device)

    print(f"[wbc] loading bundle (device={device})", flush=True)
    bundle = DataBundle.load()
    # histories.pkl is a trusted in-repo artifact from FAMAIL's editing runner
    # (algorithm/persistence.py), not external input -- pickle.load is safe here
    # (same artifact loaded by run_level2_table.py).
    with open(Path(args.edit_dir) / "histories.pkl", "rb") as f:
        histories = pickle.load(f)

    raw_trajs = bundle.trajectories
    driver_to_idx = build_driver_index(raw_trajs)
    n_drivers = len(driver_to_idx)
    groups = group_by_driver(raw_trajs)
    profiles = bundle.multi_stream.profile_features
    zeros11 = np.zeros(11, dtype=np.float32)
    edited_corpus = build_edited_corpus(raw_trajs, histories)
    eids = edited_ids(histories)
    print(f"[wbc] {len(eids)} edited of {len(raw_trajs)} "
          f"({100.0 * len(eids) / len(raw_trajs):.1f}%)", flush=True)

    D_raw = traj_training_data(raw_trajs, driver_to_idx)
    D_edited = traj_training_data(edited_corpus, driver_to_idx)

    # Arms: (name, training-data, sample_weights)
    arms: List = [("raw", D_raw, None), ("edited", D_edited, None)]
    for w in up_weights:
        if w == 1.0:
            continue
        arms.append((f"edited_w{int(w)}", D_edited, weight_vector(edited_corpus, eids, w)))
    # Placebo arms: upweight a random, size-matched NON-edited subset of the RAW
    # corpus. Built ONCE here (independent RNG) so it never perturbs the per-seed
    # training determinism the paired design depends on.
    for w in placebo_weights:
        if w == 1.0:
            continue
        arms.append((
            f"random_w{int(w)}", D_raw,
            random_subset_weight_vector(raw_trajs, eids, w, seed=args.placebo_seed),
        ))
    arm_names = [a[0] for a in arms]
    if placebo_weights:
        n_pl = len(eids)
        print(f"[wbc] placebo: upweighting a fixed random {n_pl}-traj NON-edited "
              f"subset of RAW (seed={args.placebo_seed}) at doses {placebo_weights}",
              flush=True)

    # ---- Evaluation fixtures (policy-independent), mirroring run_level2_table.main ----
    rng = random.Random(seeds[0])
    drivers = _select_eval_drivers(
        groups, min_trajs=args.min_driver_trajs, max_drivers=args.max_eval_drivers,
    )
    if not drivers:
        raise SystemExit(f"no driver has >= {args.min_driver_trajs} real trajectories")
    ppd = args.pairs_per_driver
    real_slot0_by_d: Dict[int, list] = {}
    real_context_by_d: Dict[int, list] = {}
    prof_by_d: Dict[int, object] = {}
    for d in drivers:
        real_d = groups[d]
        real_context_by_d[d] = _real_context_tensors(real_d)
        prof_d = profiles.get(d)
        if prof_d is None:
            prof_d = zeros11
        prof_by_d[d] = prof_d
        real_slot0_by_d[d] = [fe.real_to_disc_tensor(t) for t in real_d[:ppd]]
    eval_fixtures = {
        "drivers": drivers, "ppd": ppd, "real_slot0": real_slot0_by_d,
        "real_context": real_context_by_d, "prof": prof_by_d,
    }
    raw_stats = [fe.trajectory_statistics(t) for t in raw_trajs[:fss]]
    raw_pickups = _terminal_pickups_from_trajs(raw_trajs[:fss])

    # ---- Real-anchored validation gate (policy-independent, computed ONCE) ----
    print(f"[wbc] validation gate over {len(drivers)} eval drivers", flush=True)
    raw_matched: list = []
    raw_mismatched: list = []
    for k, d in enumerate(drivers):
        dprime = drivers[(k + 1) % len(drivers)]
        real_slot0 = real_slot0_by_d[d]
        if not real_slot0:
            continue
        real_d = groups[d]
        n_take = len(real_slot0)
        n_avail = len(real_d)
        if n_avail >= 2 * n_take:
            raw_slot0_d = [fe.real_to_disc_tensor(t) for t in real_d[n_take:2 * n_take]]
        else:
            raw_slot0_d = [
                fe.real_to_disc_tensor(real_d[rng.randrange(n_avail)])
                for _ in range(n_take)
            ]
        m, mm = _build_source_pairs(
            real_slot0=real_slot0, source_slot0=raw_slot0_d,
            source_slot0_other=real_slot0_by_d[dprime],
            real_context=real_context_by_d[d],
            source_context_other=real_context_by_d[dprime],
            profile_d=prof_by_d[d], profile_dp=prof_by_d[dprime], rng=rng,
        )
        raw_matched.extend(m)
        raw_mismatched.extend(mm)
    gate = fe.identity_validation_gate(
        disc, matched_pairs=raw_matched, mismatched_pairs=raw_mismatched, device=device,
    )
    print(f"[wbc] gate {'PASSED' if gate['passed'] else 'FAILED'} "
          f"(matched {gate['high_matched']:.3f} vs mismatched {gate['low_mismatched']:.3f})",
          flush=True)

    # ---- Paired loop: seed x arm (evaluate-and-discard each policy) ----
    per_seed: Dict[str, Dict[str, List[float]]] = {
        m: {a: [] for a in arm_names} for m in _METRICS
    }
    per_arm_empty: Dict[str, List[int]] = {a: [] for a in arm_names}
    _FB_COMPONENTS = ("length", "mean_displacement", "coverage",
                      "radius_of_gyration", "net_displacement", "terminal_cell")
    _DEGEN_KEYS = ("terminal_cell_entropy_bits", "mean_trip_length", "std_trip_length")
    per_arm_fb: Dict[str, Dict[str, List[float]]] = {
        a: {c: [] for c in _FB_COMPONENTS} for a in arm_names
    }
    per_arm_sep: Dict[str, List[float]] = {a: [] for a in arm_names}
    per_arm_degen: Dict[str, Dict[str, List[float]]] = {
        a: {k: [] for k in _DEGEN_KEYS} for a in arm_names
    }
    for s in seeds:
        for name, D, sw in arms:
            t0 = time.time()
            print(f"[wbc] seed={s} arm={name}: train + evaluate", flush=True)
            # Identical init + batch order across arms; arms differ only in
            # training data + per-sequence weights.
            set_all_seeds(s)
            model = TrajectoryLSTM(n_drivers=n_drivers).to(device)
            train_mle(
                model, D["sequences"], D["contexts"],
                epochs=args.mle_epochs, lr=gc.MLE_LR, batch_size=gc.MLE_BATCH_SIZE,
                device=device, driver_idxs=D["driver_idxs"],
                max_batch_tokens=args.max_batch_tokens, sample_weights=sw,
            )
            m = _evaluate_policy(
                model, driver_idxs=D["driver_idxs"], contexts=D["contexts"],
                filtered_train=D["trajs"], bundle=bundle, eval_drivers=eval_fixtures,
                driver_to_idx=driver_to_idx, groups=groups, profiles=profiles,
                zeros11=zeros11, raw_stats=raw_stats, raw_pickups=raw_pickups,
                disc=disc, rng=rng, fss=fss, max_len=max_len, device=device,
                gen_batch_size=args.gen_batch_size,
            )
            for metric in _METRICS:
                per_seed[metric][name].append(float(m[metric]))
            per_arm_empty[name].append(int(m["n_empty"]))
            for c in _FB_COMPONENTS:
                per_arm_fb[name][c].append(float(m["fidelity_b_per_component"][c]))
            per_arm_sep[name].append(float(m["fidelity_a_separation"]))
            for k in _DEGEN_KEYS:
                per_arm_degen[name][k].append(float(m[k]))
            print(f"[wbc]   {name}: f_causal={m['f_causal']:.4f} "
                  f"f_spatial={m['f_spatial']:.4f} fid_a={m['fidelity_a']:.4f} "
                  f"fid_b={m['fidelity_b']:.4f} ({round(time.time() - t0, 1)}s)", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- aggregate ----
    def _ms(vals: List[float]) -> dict:
        return {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": [round(float(v), 4) for v in vals],
        }

    per_arm = {
        a: (
            {m: _ms(per_seed[m][a]) for m in _METRICS}
            | {"n_empty": per_arm_empty[a]}
            | {"fidelity_b_per_component": {
                c: {"values": [round(float(v), 4) for v in per_arm_fb[a][c]]}
                for c in _FB_COMPONENTS
            }}
            | {"fidelity_a_separation": {"values": [round(float(v), 4) for v in per_arm_sep[a]]}}
            | {k: {"values": [round(float(v), 4) for v in per_arm_degen[a][k]]}
               for k in _DEGEN_KEYS}
        )
        for a in arm_names
    }
    paired = {
        m: {a: _paired_vs_raw(per_seed[m], a) for a in arm_names if a != "raw"}
        for m in _METRICS
    }

    result = {
        "edit_dir": args.edit_dir, "seeds": seeds, "weights": up_weights,
        "placebo_weights": placebo_weights, "placebo_seed": args.placebo_seed,
        "mle_epochs": args.mle_epochs, "n_edited": len(eids), "n_corpus": len(raw_trajs),
        "n_eval_drivers": len(drivers), "gate": gate, "trusted": bool(gate["passed"]),
        "per_arm": per_arm, "paired_vs_raw": paired,
        "effective_edited_fraction": {
            str(int(w)): _enrich.effective_edited_fraction(len(eids), len(raw_trajs), w)
            for w in ([1.0] + up_weights)
        },
    }

    out_dir = args.out_dir or (
        Path(config.PACKAGE_ROOT) / "results" / "weighted_bc_sweep"
        / time.strftime("%Y-%m-%dT%H-%M-%S")
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(json.dumps(result, indent=2, default=float))

    # ---- E10 dose-response table ----
    (out_dir / "dose_response.json").write_text(json.dumps(
        _enrich.dose_response_table(per_arm, paired, up_weights), indent=2, default=float))

    # ---- E26 paired_stats with t_ci ----
    import copy
    paired_ci = copy.deepcopy(paired)
    for metric, by_arm in paired_ci.items():
        for arm, leaf in by_arm.items():
            if isinstance(leaf, dict) and "diffs" in leaf:
                leaf["t_ci"] = list(_enrich.t_ci(leaf["diffs"]))
    (out_dir / "paired_stats.json").write_text(json.dumps(paired_ci, indent=2, default=float))

    # ---- E27 chosen id sets (edited + placebo per weight) ----
    raw_ids = [int(t.trajectory_id) for t in raw_trajs]
    chosen = {"edited_ids": sorted(int(i) for i in eids)}
    for w in placebo_weights:
        chosen[f"random_w{int(w)}"] = _enrich.chosen_placebo_ids(raw_ids, eids, args.placebo_seed)
    (out_dir / "chosen_ids.json").write_text(json.dumps(chosen, indent=2))

    # ---- provenance ----
    _gate_extra = {
        "discriminator_sha256": sha256_file(ckpt),
        "gate_matched": float(gate["high_matched"]),
        "gate_mismatched": float(gate["low_mismatched"]),
        "gate_passed": bool(gate["passed"]),
    }
    write_run_manifest(out_dir, argv=sys.argv, seeds=seeds, edit_dir=str(args.edit_dir),
                       extra=_gate_extra)
    append_timing(out_dir / "timings.jsonl", "weighted_bc", time.time() - t0_run)

    # ---- summary ----
    print(f"\n[wbc] ===== SWEEP SUMMARY ({len(seeds)} seeds, {args.mle_epochs} epochs, "
          f"gate {'PASSED' if gate['passed'] else 'FAILED'}) =====", flush=True)
    print(f"{'arm':<13}{'F_causal':>16}{'F_spatial':>16}{'Fid-A↑':>16}{'Fid-B↓':>16}",
          flush=True)
    for a in arm_names:
        pa = per_arm[a]
        def c(m):
            return f"{pa[m]['mean']:.4f}±{pa[m]['std']:.4f}"
        print(f"{a:<13}{c('f_causal'):>16}{c('f_spatial'):>16}"
              f"{c('fidelity_a'):>16}{c('fidelity_b'):>16}", flush=True)
    print("\n[wbc] paired Δ vs raw (mean ± std, Wilcoxon p):", flush=True)
    for a in arm_names:
        if a == "raw":
            continue
        fc = paired["f_causal"][a]
        fs = paired["f_spatial"][a]
        fb = paired["fidelity_b"][a]
        pstr = "n/a" if fc["wilcoxon_p"] is None else f"{fc['wilcoxon_p']:.3f}"
        print(f"  {a:<12} ΔF_causal={fc['mean']:+.4f}±{fc['std']:.4f} (p={pstr})  "
              f"ΔF_spatial={fs['mean']:+.4f}  ΔFid-B={fb['mean']:+.4f}", flush=True)
    print(f"\n[wbc] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
