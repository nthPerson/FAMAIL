"""CLI: Level-2 usability table (fairness transfer).

Train a driver-conditioned BC policy on each of four matched, full-corpus data
sources -- raw, FAM-AIL edited, BC-generated, GAN-generated -- across paired
seeds, then evaluate each trained policy's generated demand on the Level-1 axes
(causal/spatial fairness, identity Fidelity-A with the real-anchored gate,
enriched Fidelity-B). Reports paired per-seed differences (edited vs raw
headline; edited vs BC-gen/GAN-gen secondary). HuMID is frozen, read-only.

See docs/superpowers/specs/2026-06-18-level2-usability-fairness-transfer-design.md
and docs/superpowers/plans/2026-06-18-level2-usability-fairness-transfer.md.
"""
from __future__ import annotations
import argparse
import json
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.utils.seeding import set_all_seeds
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.drivers import (
    build_driver_index, group_by_driver, driver_idxs_for,
)
from famail_temporal.baselines.gan.sequences import (
    trajectory_context, trajectory_to_tokens, flat_cell,
)
from famail_temporal.baselines.gan.rollout import (
    generate_trajectories, generate_pickups, pickups_to_pickup_3d,
)
from famail_temporal.baselines.gan.train_mle import train_mle
from famail_temporal.baselines.metrics import data_level_fairness
from famail_temporal.fidelity.checkpoint import load_discriminator
from famail_temporal.baselines import fidelity_eval as fe
from famail_temporal.baselines.run_level1_table_v2 import (
    _select_eval_drivers, _real_context_tensors, _build_source_pairs,
    _train_and_generate_cond, _gen_cond_slot0, _gen_fidelity_full,
    _terminal_pickups_from_cells, _terminal_pickups_from_trajs,
    _edited_fairness_from_metrics, _curves_for_source,
)

_SOURCE_ORDER = ["raw", "edited", "bcgen", "gangen"]


def build_edited_corpus(raw_trajs, histories) -> list:
    """Full corpus with each modified trajectory swapped in by trajectory_id.

    Same length/order as raw_trajs; the 3,773 edited trajectories replace their
    originals, all others are kept (so D_edited is index-aligned to D_raw).
    """
    mod_by_id = {int(h.original.trajectory_id): h.modified for h in histories}
    return [mod_by_id.get(int(t.trajectory_id), t) for t in raw_trajs]


def traj_training_data(trajs, driver_to_idx) -> dict:
    """Token sequences + contexts + embedding indices for a list of Trajectories."""
    return {
        "sequences": [trajectory_to_tokens(t) for t in trajs],
        "contexts": [trajectory_context(t) for t in trajs],
        "driver_idxs": driver_idxs_for(trajs, driver_to_idx),
        "trajs": trajs,
    }


def gen_training_data(model, raw_trajs, driver_to_idx, *, max_len, device,
                      gen_batch_size) -> dict:
    """Driver-conditioned generated training set: one rollout per real seed.

    Each generated trajectory inherits its seed's driver + start-context.
    Empty rollouts fall back to [BOS, start_cell, EOS] (counted in n_empty) so
    the set stays index-aligned and full-corpus-sized.
    """
    contexts = [trajectory_context(t) for t in raw_trajs]
    driver_idxs = driver_idxs_for(raw_trajs, driver_to_idx)
    gen_cells = generate_trajectories(
        model, contexts, max_len=max_len, device=device,
        gen_batch_size=gen_batch_size, driver_idxs=driver_idxs, progress=False,
    )
    sequences = []
    n_empty = 0
    for cells, (start_cell, _t) in zip(gen_cells, contexts):
        if cells:
            sequences.append([gc.BOS] + list(cells) + [gc.EOS])
        else:
            n_empty += 1
            sequences.append([gc.BOS, start_cell, gc.EOS])
    return {
        "sequences": sequences, "contexts": contexts,
        "driver_idxs": driver_idxs, "trajs": raw_trajs, "n_empty": n_empty,
    }


def _paired_diff_stats(per_seed: Dict[str, List[float]], *, baseline: str = "edited") -> dict:
    """Paired per-seed differences baseline - other, per other source.

    Returns {other: {diffs, mean, std, n, wilcoxon_p}}. wilcoxon_p is None when
    SciPy is unavailable, n < 1, or all differences are zero (no signed-rank
    test is defined).
    """
    try:
        from scipy.stats import wilcoxon  # optional dependency
    except Exception:
        wilcoxon = None
    base = per_seed[baseline]
    out: Dict[str, dict] = {}
    for other, vals in per_seed.items():
        if other == baseline:
            continue
        diffs = [float(b - o) for b, o in zip(base, vals)]
        n = len(diffs)
        mean = float(np.mean(diffs)) if n else float("nan")
        std = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        p = None
        if wilcoxon is not None and n >= 1 and any(d != 0.0 for d in diffs):
            try:
                p = float(wilcoxon(diffs).pvalue)
            except Exception:
                p = None
        out[other] = {"diffs": diffs, "mean": mean, "std": std, "n": n, "wilcoxon_p": p}
    return out


def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2, default=float)


def render_level2_table(result: dict) -> str:
    """Render the Level-2 usability table as markdown.

    Per-source mean ± std across seeds for {F_causal, F_spatial, Fidelity-A,
    Fidelity-B}, the real-anchored gate verdict, and the paired fairness-transfer
    block (F_causal headline edited−raw + secondary edited−bcgen/gangen).
    """
    g = result["gate"]
    gate_line = (
        f"Validation gate (real-anchored): **{'PASSED' if g['passed'] else 'FAILED'}** "
        f"(matched {g['high_matched']:.3f} vs mismatched {g['low_mismatched']:.3f}, "
        f"margin {g['margin']:.2f})"
    )

    def cell(src, m):
        d = result["per_source"][src][m]
        return f"{d['mean']:.4f} ± {d['std']:.4f}"

    rows = []
    for s in _SOURCE_ORDER:
        rows.append(
            f"| {s} | {cell(s,'f_causal')} | {cell(s,'f_spatial')} "
            f"| {cell(s,'fidelity_a')} | {cell(s,'fidelity_b')} |"
        )
    pj = result["paired"]["f_causal"]

    def pline(o):
        d = pj[o]
        p = "n/a" if d["wilcoxon_p"] is None else f"{d['wilcoxon_p']:.3f}"
        return f"| edited − {o} | {d['mean']:+.4f} ± {d['std']:.4f} | {d['n']} | {p} |"

    paired_block = (
        "\n\n## Paired fairness transfer (F_causal, by seed)\n\n"
        "| Comparison | mean Δ ± std | n seeds | Wilcoxon p |\n|---|---:|---:|---:|\n"
        + pline("raw") + "\n" + pline("bcgen") + "\n" + pline("gangen") + "\n"
    )
    return (
        "# Level-2 Usability Table (fairness transfer)\n\n"
        f"Edit source: `{result['edit_dir']}`\n\nSeeds: {result['seeds']} | "
        f"Eval drivers: {result['n_eval_drivers']}\n\n{gate_line}\n\n"
        "Each cell is mean ± std across seeds (driver-conditioned BC trained on that source).\n\n"
        "| Source (training data) | F_causal | F_spatial | Fidelity-A ↑ | Fidelity-B ↓ |\n"
        "|---|---:|---:|---:|---:|\n" + "\n".join(rows) + paired_block
    )


# --------------------------------------------------- per-policy evaluation ----

def _evaluate_policy(
    model, *, driver_idxs, contexts, filtered_train, bundle, eval_drivers,
    driver_to_idx, groups, profiles, zeros11, raw_stats, raw_pickups, disc,
    rng, fss, max_len, device, gen_batch_size,
) -> dict:
    """Score ONE trained driver-conditioned policy on the Level-1 axes.

    Mirrors run_level1_table_v2's per-source scoring, but everything is generated
    from this single policy:

    * Fairness: corpus-scale driver-conditioned demand grid -> data_level_fairness.
    * Fidelity-A: identity-aware, two-pass over eval drivers. Pass 1 builds each
      eval driver's GENERATED slot-0 set; pass 2 pairs driver d (matched) against
      d' = next eval driver (mismatched), honoring _build_source_pairs's
      ``source_slot0_other`` (the OTHER driver's GENERATED slot-0 -- the fixed L1
      bug). The gate is policy-independent and computed once in ``main``.
    * Fidelity-B: enriched 5-key distributional JS + terminal-cell JS vs raw,
      on a per-policy-vs-raw pooled grid (ranges=None) for a consistent guardrail.

    ``real_slot0_by_d`` / ``real_context_by_d`` / ``prof_by_d`` are precomputed
    once in ``main`` and reused here for the anchor branch.
    Returns {f_causal, f_spatial, fidelity_a, fidelity_a_separation, fidelity_b,
    fidelity_b_per_component, n_empty}.
    """
    real_slot0_by_d, real_context_by_d, prof_by_d = (
        eval_drivers["real_slot0"], eval_drivers["real_context"],
        eval_drivers["prof"],
    )
    drivers = eval_drivers["drivers"]
    ppd = eval_drivers["ppd"]

    # ---- Fairness (corpus-scale, driver-conditioned) ----
    pickups = generate_pickups(
        model, contexts, max_len=max_len, device=device,
        gen_batch_size=gen_batch_size, progress=False, driver_idxs=driver_idxs,
    )
    fair = data_level_fairness(
        bundle, pickup_3d=pickups_to_pickup_3d(bundle, pickups),
    )

    # ---- Fidelity-A pass 1: this policy's generated slot-0 set per eval driver ----
    gen_slot0_by_d: Dict[int, list] = {}
    for d in drivers:
        gen_slot0_by_d[d] = _gen_cond_slot0(
            model, groups[d], driver_to_idx[d], pairs_per_driver=ppd,
            max_len=max_len, device=device, gen_batch_size=gen_batch_size,
        )

    # ---- Fidelity-A pass 2: d (matched) vs d' = next driver (mismatched) ----
    matched: list = []
    mismatched: list = []
    for k, d in enumerate(drivers):
        dprime = drivers[(k + 1) % len(drivers)]
        real_slot0 = real_slot0_by_d[d]
        if not real_slot0:
            continue
        m, mm = _build_source_pairs(
            real_slot0=real_slot0,
            source_slot0=gen_slot0_by_d[d],
            source_slot0_other=gen_slot0_by_d[dprime],
            real_context=real_context_by_d[d],
            source_context_other=real_context_by_d[dprime],
            profile_d=prof_by_d[d], profile_dp=prof_by_d[dprime], rng=rng,
        )
        matched.extend(m)
        mismatched.extend(mm)
    a_match = fe.humid_identity_fidelity(disc, matched, device=device)
    a_mis = fe.humid_identity_fidelity(disc, mismatched, device=device)
    fidelity_a = float(a_match["mean"])
    fidelity_a_separation = float(a_match["mean"] - a_mis["mean"])

    # ---- Fidelity-B (enriched distributional + terminal-cell JS vs raw) ----
    fb_n = min(fss, len(contexts))
    gen_cells, n_empty = _gen_fidelity_full(
        model, filtered_train, contexts, driver_idxs, n=fb_n,
        max_len=max_len, device=device, gen_batch_size=gen_batch_size,
    )
    src_stats = [fe.trajectory_statistics(c) for c in gen_cells if c]
    per = fe.distributional_fidelity(
        src_stats, raw_stats, keys=fe._STAT_KEYS_V2,
    )["per_stat"]
    tj = fe.terminal_cell_distribution_js(
        _terminal_pickups_from_cells(gen_cells), raw_pickups,
    )
    fidelity_b = float(np.mean(list(per.values()) + [tj]))
    per_component = {**per, "terminal_cell": float(tj)}

    return {
        "f_causal": float(fair["f_causal"]),
        "f_spatial": float(fair["f_spatial"]),
        "fidelity_a": fidelity_a,
        "fidelity_a_separation": fidelity_a_separation,
        "fidelity_b": fidelity_b,
        "fidelity_b_per_component": per_component,
        "n_empty": int(n_empty),
    }


# ---------------------------------------------------------------- assembly ----

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.baselines.run_level2_table",
        description="Assemble the Level-2 usability table (fairness transfer): "
                    "train a driver-conditioned BC policy on each of four "
                    "matched full-corpus sources (raw/edited/bcgen/gangen) "
                    "across paired seeds, score each policy on the Level-1 axes, "
                    "and report paired per-seed differences.",
    )
    ap.add_argument(
        "--edit-dir", type=str,
        default="famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup",
        help="Persisted editing run (provides edited trajectories + histories.pkl).",
    )
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4",
                    help="Comma-separated paired seeds; all four arms share each seed.")
    ap.add_argument("--mle-epochs", type=int, default=20)
    ap.add_argument("--max-eval-drivers", type=int, default=50,
                    help="Cap on the number of eval drivers for identity Fidelity-A.")
    ap.add_argument("--pairs-per-driver", type=int, default=20,
                    help="Identity-branch slot-0 pairs sampled per eval driver.")
    ap.add_argument("--min-driver-trajs", type=int, default=6,
                    help="Min real trajectories for a driver to be eligible.")
    ap.add_argument("--fidelity-sample-size", type=int, default=5000)
    ap.add_argument("--gan-loss", type=str, default="wgan-gp",
                    choices=["bce", "wgan-gp"],
                    help="Loss for the GAN generator that produces the gangen source.")
    ap.add_argument("--adv-epochs", type=int, default=3,
                    help="GAN-generator adversarial epochs (gangen source only).")
    ap.add_argument("--n-critic", type=int, default=5)
    ap.add_argument("--gen-batch-size", type=int, default=gc.GEN_BATCH_SIZE)
    ap.add_argument("--max-batch-tokens", type=int,
                    default=gc.MLE_BATCH_SIZE * gc.MAX_TRAIN_TOKENS,
                    help="Token-budget cap per MLE minibatch for full-corpus training.")
    ap.add_argument("--gen-max-tokens", type=int, default=gc.MAX_TRAIN_TOKENS,
                    help="Max token length for the BC/GAN GENERATOR pretraining "
                         "corpus (matches Level-1 v2; filters the ~0.7%% long "
                         "outliers the adversarial trainer cannot batch). The "
                         "downstream policies still train on the full corpus.")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    if not seeds:
        raise SystemExit("--seeds must list at least one integer seed")

    def _log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    max_len = gc.MAX_GEN_LEN
    fss = args.fidelity_sample_size

    # ---- checkpoint guard (before any expensive work) ----
    ckpt = Path(config.PACKAGE_ROOT) / "discriminator_checkpoints" / "default" / "best.pt"
    if not ckpt.exists():
        raise SystemExit(
            f"discriminator checkpoint not found: {ckpt}\n"
            "Level-2 Fidelity-A (HuMID) requires the trained discriminator."
        )
    # load_discriminator maps weights to CPU; move to the run device so the CUDA
    # input tensors built in fidelity_eval meet on-device weights.
    disc = load_discriminator(ckpt).to(device)

    # ---- data ----
    _log(f"[level2] loading bundle (device={device})")
    bundle = DataBundle.load()
    # histories.pkl is a trusted in-repo artifact from FAMAIL's editing runner
    # (algorithm/persistence.py), not external input -- pickle.load is safe here.
    with open(Path(args.edit_dir) / "histories.pkl", "rb") as f:
        histories = pickle.load(f)

    raw_trajs = bundle.trajectories
    driver_to_idx = build_driver_index(raw_trajs)
    groups = group_by_driver(raw_trajs)
    profiles = bundle.multi_stream.profile_features
    zeros11 = np.zeros(11, dtype=np.float32)

    # ---- Build the L1 generators ONCE (token-capped, matching Level-1 v2) ----
    # These supply the bcgen / gangen TRAINING data (one rollout per real seed),
    # not the policies under test. They are trained on the first seed so the
    # generated training corpora are fixed across the paired loop.
    #
    # The generators are pretrained on the --gen-max-tokens-capped corpus (256,
    # identical to Level-1 v2) -- NOT the full corpus. The ~0.7% long outliers
    # (up to 1654 tokens) cannot be batched by the adversarial GAN trainer (no
    # token-budget guard there) and the generators emit at most MAX_GEN_LEN=64
    # tokens regardless, so capping is lossless and keeps these generators
    # consistent with the validated Level-1 table. The full-corpus property the
    # fairness claim rests on lives on the DOWNSTREAM training data: D_bcgen /
    # D_gangen are still full-corpus-SIZED (one rollout per real seed across all
    # 105,401 trajectories) and trained with the --max-batch-tokens budget.
    _log(f"[level2] training BC generator (MLE-only, {args.mle_epochs} epochs, "
         f"n_drivers={len(driver_to_idx)})")
    bc = _train_and_generate_cond(
        raw_trajs, driver_to_idx, adv_epochs=0, gan_loss="bce", n_critic=1,
        mle_epochs=args.mle_epochs, max_len=max_len, max_tokens=args.gen_max_tokens,
        device=device, seed=seeds[0],
    )
    _log(f"[level2] training GAN generator ({args.gan_loss}, mle={args.mle_epochs}, "
         f"adv={args.adv_epochs}, n_critic={args.n_critic})")
    gan = _train_and_generate_cond(
        raw_trajs, driver_to_idx, adv_epochs=args.adv_epochs, gan_loss=args.gan_loss,
        n_critic=args.n_critic, mle_epochs=args.mle_epochs, max_len=max_len,
        max_tokens=args.gen_max_tokens, device=device, seed=seeds[0],
    )

    # ---- Build the four matched, full-corpus TRAINING datasets ----
    _log("[level2] building the four matched full-corpus training datasets")
    D_raw = traj_training_data(raw_trajs, driver_to_idx)
    D_edited = traj_training_data(build_edited_corpus(raw_trajs, histories), driver_to_idx)
    D_bcgen = gen_training_data(
        bc["model"], raw_trajs, driver_to_idx, max_len=max_len, device=device,
        gen_batch_size=args.gen_batch_size,
    )
    D_gangen = gen_training_data(
        gan["model"], raw_trajs, driver_to_idx, max_len=max_len, device=device,
        gen_batch_size=args.gen_batch_size,
    )
    D = {"raw": D_raw, "edited": D_edited, "bcgen": D_bcgen, "gangen": D_gangen}

    # ---- Precompute evaluation fixtures ONCE (policy-independent) ----
    # A single rng drives ALL identity-branch sampling for determinism.
    rng = random.Random(seeds[0])
    drivers = _select_eval_drivers(
        groups, min_trajs=args.min_driver_trajs, max_drivers=args.max_eval_drivers,
    )
    if not drivers:
        raise SystemExit(
            f"no driver has >= {args.min_driver_trajs} real trajectories; "
            "cannot build identity Fidelity-A pairs."
        )
    _log(f"[level2] identity Fidelity-A over {len(drivers)} eval drivers")
    ppd = args.pairs_per_driver

    real_slot0_by_d: Dict[int, list] = {}
    real_context_by_d: Dict[int, list] = {}
    prof_by_d: Dict[int, object] = {}
    for d in drivers:
        real_d = groups[d]
        real_context_by_d[d] = _real_context_tensors(real_d)
        prof_d = profiles.get(d)
        if prof_d is None:
            _log(f"[level2] WARN driver {d} has no profile -> zero profile")
            prof_d = zeros11
        prof_by_d[d] = prof_d
        real_slot0_by_d[d] = [fe.real_to_disc_tensor(t) for t in real_d[:ppd]]

    eval_fixtures = {
        "drivers": drivers, "ppd": ppd,
        "real_slot0": real_slot0_by_d, "real_context": real_context_by_d,
        "prof": prof_by_d,
    }

    raw_stats = [fe.trajectory_statistics(t) for t in raw_trajs[:fss]]
    raw_pickups = _terminal_pickups_from_trajs(raw_trajs[:fss])

    # ---- Gate (real-anchored, policy-independent, computed ONCE) ----
    # Build raw matched/mismatched from the real slot-0 sets: matched = real-d vs
    # real-d (a disjoint/replacement second sample), mismatched = real-d vs
    # real-d'. d' = next eval driver, mirroring run_level1_table_v2's gate.
    _log("[level2] running real-anchored validation gate (real-vs-real)")
    raw_matched: list = []
    raw_mismatched: list = []
    for k, d in enumerate(drivers):
        dprime = drivers[(k + 1) % len(drivers)]
        real_slot0 = real_slot0_by_d[d]
        if not real_slot0:
            continue
        # A disjoint second sample of d's real trajectories for the matched arm
        # (real-d vs another real-d); fall back to sampling with replacement when
        # the driver has too few trajectories for a disjoint half.
        real_d = groups[d]
        n_take = len(real_slot0)
        n_avail = len(real_d)
        if n_avail >= 2 * n_take:
            raw_slot0_d = [
                fe.real_to_disc_tensor(t) for t in real_d[n_take:2 * n_take]
            ]
        else:
            raw_slot0_d = [
                fe.real_to_disc_tensor(real_d[rng.randrange(n_avail)])
                for _ in range(n_take)
            ]
        raw_slot0_dp = real_slot0_by_d[dprime]
        m, mm = _build_source_pairs(
            real_slot0=real_slot0,
            source_slot0=raw_slot0_d,
            source_slot0_other=raw_slot0_dp,
            real_context=real_context_by_d[d],
            source_context_other=real_context_by_d[dprime],
            profile_d=prof_by_d[d], profile_dp=prof_by_d[dprime], rng=rng,
        )
        raw_matched.extend(m)
        raw_mismatched.extend(mm)
    gate = fe.identity_validation_gate(
        disc, matched_pairs=raw_matched, mismatched_pairs=raw_mismatched,
        device=device,
    )
    trusted = bool(gate["passed"])
    if not gate["passed"]:
        _log("[level2] Validation gate FAILED -> Fidelity-A is UNTRUSTED; "
             "Fidelity-B (distributional divergence) is the PRIMARY fidelity metric.")

    # ---- Paired loop: seed x source (evaluate-and-discard each policy) ----
    per_seed_metric: Dict[str, Dict[str, List[float]]] = {
        m: {src: [] for src in _SOURCE_ORDER}
        for m in ("f_causal", "f_spatial", "fidelity_a", "fidelity_b")
    }
    per_source_empty: Dict[str, List[int]] = {src: [] for src in _SOURCE_ORDER}
    for s in seeds:
        for src in _SOURCE_ORDER:
            _log(f"[level2] seed={s} source={src}: train + evaluate policy")
            # Pairing guarantee: seed immediately before BOTH model construction
            # AND train_mle so init + the randperm minibatch sequence match across
            # all four arms for this seed; arms differ ONLY in training data.
            set_all_seeds(s)
            model = TrajectoryLSTM(n_drivers=len(driver_to_idx)).to(device)
            train_mle(
                model, D[src]["sequences"], D[src]["contexts"],
                epochs=args.mle_epochs, lr=gc.MLE_LR, batch_size=gc.MLE_BATCH_SIZE,
                device=device, driver_idxs=D[src]["driver_idxs"],
                max_batch_tokens=args.max_batch_tokens,
            )
            m = _evaluate_policy(
                model, driver_idxs=D[src]["driver_idxs"],
                contexts=D[src]["contexts"], filtered_train=D[src]["trajs"],
                bundle=bundle, eval_drivers=eval_fixtures,
                driver_to_idx=driver_to_idx, groups=groups, profiles=profiles,
                zeros11=zeros11, raw_stats=raw_stats, raw_pickups=raw_pickups,
                disc=disc, rng=rng, fss=fss, max_len=max_len, device=device,
                gen_batch_size=args.gen_batch_size,
            )
            for metric in ("f_causal", "f_spatial", "fidelity_a", "fidelity_b"):
                per_seed_metric[metric][src].append(float(m[metric]))
            per_source_empty[src].append(int(m["n_empty"]))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- Stats: per-source mean/std + paired per-seed differences ----
    def _mean_std(vals: List[float]) -> dict:
        return {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": [float(v) for v in vals],
        }

    per_source: Dict[str, dict] = {}
    for src in _SOURCE_ORDER:
        per_source[src] = {
            metric: _mean_std(per_seed_metric[metric][src])
            for metric in ("f_causal", "f_spatial", "fidelity_a", "fidelity_b")
        }
        per_source[src]["n_empty"] = [int(x) for x in per_source_empty[src]]

    paired = {
        metric: _paired_diff_stats(per_seed_metric[metric], baseline="edited")
        for metric in ("f_causal", "f_spatial", "fidelity_a", "fidelity_b")
    }

    result = {
        "edit_dir": args.edit_dir,
        "seeds": seeds,
        "gate": gate,
        "n_eval_drivers": len(drivers),
        "trusted": trusted,
        "gen_max_tokens": args.gen_max_tokens,
        "max_batch_tokens": args.max_batch_tokens,
        "mle_epochs": args.mle_epochs,
        "per_source": per_source,
        "paired": paired,
    }

    # ---- persistence (default out-dir lives under results/, which is gitignored) ----
    out_dir = args.out_dir
    if out_dir is None:
        stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = Path(config.PACKAGE_ROOT) / "results" / "level2_table" / stamp
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "level2_metrics.json").write_text(result_to_json(result))
    (out_dir / "level2_table.md").write_text(render_level2_table(result))
    (out_dir / "driver_index.json").write_text(
        json.dumps({str(k): v for k, v in driver_to_idx.items()}, indent=2)
    )

    # ---- summary ----
    _log("")
    _log(render_level2_table(result))
    head = paired["f_causal"]["raw"]
    _log(
        f"[level2] headline Δ(edited−raw) F_causal = {head['mean']:+.4f} ± "
        f"{head['std']:.4f} (n={head['n']} seeds), Wilcoxon p="
        + ("n/a" if head["wilcoxon_p"] is None else f"{head['wilcoxon_p']:.3f}")
    )
    # If the paired CI crosses zero, the per-trajectory transfer is a null; the
    # corpus-scale interpretation is the relevant one (scale-to-10 note).
    ci_crosses_zero = abs(head["mean"]) <= head["std"]
    if ci_crosses_zero:
        _log(
            "[level2] paired CI crosses zero -> per-policy fairness transfer is a "
            "null at this scale; the corpus-scale (scale-to-10) reading is the "
            "relevant interpretation of the edit's F_causal advantage."
        )
    _log(f"[level2] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
