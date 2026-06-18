"""CLI: Level-1 data-quality table v2 (driver-conditioned generation +
identity-aware Fidelity-A + enriched Fidelity-B).

Four sources -- raw, FAM-AIL edited, BC (driver-conditioned), GAN
(driver-conditioned) -- scored on causal fairness, spatial fairness, an
identity-aware HuMID Fidelity-A (real-anchored matched-vs-mismatched gate),
and an enriched discriminator-free Fidelity-B. HuMID is consumed frozen,
read-only, forward-only. See
docs/superpowers/plans/2026-06-17-driver-conditioned-fidelity.md and
docs/superpowers/specs/2026-06-17-driver-conditioned-fidelity-design.md.

Example:
    python -m famail_temporal.baselines.run_level1_table_v2 \
        --edit-dir famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup \
        --mle-epochs 20 --device auto
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

# gc MUST be imported before the argparse defaults reference gc.MAX_TRAIN_TOKENS
# / gc.GEN_BATCH_SIZE (else --help fails).
from famail_temporal.baselines.gan import config as gc
from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.utils.seeding import set_all_seeds
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.drivers import (
    build_driver_index, group_by_driver, driver_idxs_for,
)
from famail_temporal.baselines.gan.sequences import (
    trajectory_context, trajectory_to_tokens,
)
from famail_temporal.baselines.gan.rollout import (
    generate_trajectories, generate_pickups, pickups_to_pickup_3d,
)
from famail_temporal.baselines.gan.train_mle import train_mle
from famail_temporal.baselines.gan.train_adversarial import adversarial_finetune
from famail_temporal.baselines.metrics import data_level_fairness
from famail_temporal.fidelity.checkpoint import load_discriminator
from famail_temporal.baselines import fidelity_eval as fe

_SOURCE_ORDER = ["raw", "edited", "bc", "gan"]


def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2, default=float)


def render_table_v2(result: dict) -> str:
    """Render the Level-1 v2 table + real-anchored gate verdict as markdown."""
    g = result["gate"]
    gate_line = (
        f"Validation gate (real-anchored): **{'PASSED' if g['passed'] else 'FAILED'}** "
        f"(matched real-d/real-d {g['high_matched']:.3f} vs mismatched real-d/real-d' "
        f"{g['low_mismatched']:.3f}, margin {g['margin']:.2f})"
    )
    rows = []
    for key in _SOURCE_ORDER:
        s = result["sources"][key]
        a = f"{s['fidelity_a']:.3f}" + ("" if s["fidelity_a_trusted"] else " (untrusted)")
        sep = s.get("fidelity_a_separation")
        sep_str = "n/a" if sep is None else f"{sep:+.3f}"
        rows.append(
            f"| {key} | {s['f_causal']:.4f} | {s['f_spatial']:.4f} "
            f"| {a} | {sep_str} | {s['fidelity_b']:.4f} |"
        )
    return (
        "# Level-1 Data-Quality Table v2 (driver-conditioned)\n\n"
        f"Edit source: `{result['edit_dir']}`\n\n"
        f"Eval drivers: {result['n_eval_drivers']}\n\n"
        f"{gate_line}\n\n"
        "| Source | F_causal | F_spatial | Fidelity-A (identity, higher=better) "
        "| A separation (matched-mismatched) | Fidelity-B (divergence, lower=better) |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows) + "\n"
    )


# --------------------------------------------------- v1-copied fairness/curve ----
# Copied VERBATIM from run_level1_table.py (unchanged in v2): the edited-source
# fairness comes from the edit pipeline's persisted metrics_after, and the
# training-curve flattener is identical.

def _edited_fairness_from_metrics(edit_dir: Path) -> dict:
    """Edited-source fairness = the edit pipeline's authoritative metrics_after.

    The edit relocates only k_modified pickups WITHIN the full corpus; its
    after-grid fairness is already computed and persisted in metrics.json. We
    read it directly rather than recomputing from the modified subset (which
    would be a sparse, non-comparable grid). This is on the same basis as raw:
    the edit's metrics_before.f_causal equals data_level_fairness(bundle).
    Returns {"f_causal": float, "f_spatial": float}.
    """
    mpath = edit_dir / "metrics.json"
    if not mpath.exists():
        raise SystemExit(f"edit metrics.json not found: {mpath}")
    after = json.loads(mpath.read_text()).get("metrics_after")
    if not after or "f_causal" not in after or "f_spatial" not in after:
        raise SystemExit(f"{mpath} missing metrics_after.f_causal/f_spatial")
    return {"f_causal": float(after["f_causal"]), "f_spatial": float(after["f_spatial"])}


def _curves_for_source(src: dict) -> dict:
    """Flatten one source's captured training curves into a JSON-ready dict.

    ``src`` is a ``_train_and_generate_cond`` result. BC has ``adv_curve=None``
    (pure MLE), so its ``adv`` entry is null; the GAN source carries both phases.
    """
    mle = src["mle_curve"]
    out = {
        "mle_epoch_losses": mle["epoch_losses"],
        "mle_batch_losses": mle["batch_losses"],
        "adv": None,
    }
    adv = src.get("adv_curve")
    if adv is not None:
        out["adv"] = {
            "g_epoch_losses": adv["g_losses"],
            "d_epoch_losses": adv["d_losses"],
            "g_batch_losses": adv["g_batch_losses"],
            "d_batch_losses": adv["d_batch_losses"],
        }
    return out


# ------------------------------------------------------- pure Fidelity-A helpers ----

def _select_eval_drivers(groups, *, min_trajs: int, max_drivers: int) -> List[int]:
    """Drivers (sorted) with >= min_trajs real trajectories, capped to max_drivers."""
    eligible = sorted(d for d, ts in groups.items() if len(ts) >= min_trajs)
    return eligible[:max_drivers]


def _real_context_tensors(real_trajs) -> List[torch.Tensor]:
    """real_to_disc_tensor for each real trajectory (the driver's context pool)."""
    return [fe.real_to_disc_tensor(t) for t in real_trajs]


def _build_source_pairs(
    *, real_slot0: List[torch.Tensor], source_slot0: List[torch.Tensor],
    real_context: List[torch.Tensor], source_context_other: List[torch.Tensor],
    profile_d, profile_dp, rng: random.Random,
) -> Tuple[list, list]:
    """Build matched + mismatched identity-branch pair lists for one driver.

    matched[i] = ( branch(real_slot0[i], real_context, prof d),
                   branch(source_slot0[i], real_context, prof d) )    # same driver
    mismatched[i] = ( branch(real_slot0[i], real_context, prof d),
                      branch(source_slot0[i], source_context_other, prof d') )  # diff driver

    For raw, source_slot0 are other real-d trajectories. For edited/bc/gan,
    source_slot0 are edited/generated-for-d trajectories. ``source_context_other``
    is the OTHER driver d''s real context (used only in the mismatched branch).
    Pure given pre-built tensors + an injected rng.
    """
    matched, mismatched = [], []
    for i in range(min(len(real_slot0), len(source_slot0))):
        real_branch = fe.build_identity_branch(real_slot0[i], real_context, rng=rng)
        src_branch_d = fe.build_identity_branch(source_slot0[i], real_context, rng=rng)
        matched.append((
            (real_branch[0], real_branch[1], profile_d),
            (src_branch_d[0], src_branch_d[1], profile_d),
        ))
        src_branch_dp = fe.build_identity_branch(
            source_slot0[i], source_context_other, rng=rng,
        )
        mismatched.append((
            (real_branch[0], real_branch[1], profile_d),
            (src_branch_dp[0], src_branch_dp[1], profile_dp),
        ))
    return matched, mismatched


# --------------------------------------------------------- train + generate ----

def _train_and_generate_cond(
    train_trajectories, driver_to_idx, *,
    adv_epochs, gan_loss, n_critic, mle_epochs, max_len, max_tokens, device, seed,
) -> dict:
    """Train a driver-conditioned generator and return the model + aligned data.

    Like v1's ``_train_and_generate`` but (a) builds ``driver_idxs`` aligned with
    the surviving ``filtered_train`` list (the alignment guarantee: the i-th
    sequence/context/driver_idx all describe ``filtered_train[i]``), and (b)
    constructs the model with ``n_drivers=len(driver_to_idx)`` so the additive
    driver embedding is sized to the training corpus. ``adv_epochs == 0`` is the
    pure-MLE "BC" source; the GAN source passes ``adv_epochs > 0``.
    """
    set_all_seeds(seed)
    filtered_train = [
        t for t in train_trajectories
        if max_tokens is None or len(trajectory_to_tokens(t)) <= max_tokens
    ]
    if not filtered_train:
        raise ValueError(f"no training trajectories remain after max_tokens={max_tokens}")
    sequences = [trajectory_to_tokens(t) for t in filtered_train]
    contexts = [trajectory_context(t) for t in filtered_train]
    driver_idxs = driver_idxs_for(filtered_train, driver_to_idx)
    model = TrajectoryLSTM(n_drivers=len(driver_to_idx)).to(device)
    mle_curve = train_mle(
        model, sequences, contexts, epochs=mle_epochs, lr=gc.MLE_LR,
        batch_size=gc.MLE_BATCH_SIZE, device=device, progress=False,
        driver_idxs=driver_idxs,
    )
    adv_curve = None
    if adv_epochs > 0:
        adv_curve = adversarial_finetune(
            model, sequences, contexts, epochs=adv_epochs, lr_g=gc.ADV_LR_G,
            lr_d=gc.ADV_LR_D, batch_size=gc.ADV_BATCH_SIZE, max_len=max_len,
            tau_start=gc.GUMBEL_TAU_START, tau_end=gc.GUMBEL_TAU_END,
            d_update_every=gc.D_UPDATE_EVERY, mle_lambda=gc.ADV_MLE_LAMBDA,
            gan_loss=gan_loss, gp_lambda=gc.WGAN_GP_LAMBDA, n_critic=n_critic,
            device=device, progress=False, driver_idxs=driver_idxs,
        )
    return {
        "model": model, "filtered_train": filtered_train, "contexts": contexts,
        "driver_idxs": driver_idxs, "mle_curve": mle_curve, "adv_curve": adv_curve,
    }


# ------------------------------------------ generated slot-0 sets per driver ----

def _gen_cond_slot0(model, real_d, driver_idx, *, pairs_per_driver, max_len,
                    device, gen_batch_size) -> List[torch.Tensor]:
    """Driver-conditioned slot-0 tensors for one eval driver.

    Take up to ``pairs_per_driver`` of driver ``d``'s real contexts, roll out
    full trajectories conditioned on ``driver_idx``, and convert each non-empty
    rollout to a discriminator tensor. The synthesized (time_bucket, day_index)
    is the paired real context's first state (same domain-matching convention as
    v1's fidelity pairs). Empty rollouts are skipped.
    """
    real_subset = real_d[:pairs_per_driver]
    ctxs_d = [trajectory_context(t) for t in real_subset]
    if not ctxs_d:
        return []
    gen_cells = generate_trajectories(
        model, ctxs_d, max_len=max_len, device=device,
        gen_batch_size=gen_batch_size, driver_idxs=[driver_idx] * len(ctxs_d),
    )
    out: List[torch.Tensor] = []
    for i, cells in enumerate(gen_cells):
        if not cells:
            continue
        s0 = real_subset[i].states[0]
        out.append(fe.generated_to_disc_tensor(
            cells, time_bucket=s0.time_bucket, day_index=s0.day_index,
        ))
    return out


# ------------------------------------------------------------ fidelity-B pairs ----

def _gen_fidelity_full(model, filtered_train, contexts, driver_idxs, *, n,
                       max_len, device, gen_batch_size):
    """Generate full driver-conditioned trajectories over the first ``n`` contexts.

    Returns ``(gen_cells, n_empty)`` where ``gen_cells`` is index-aligned with
    ``filtered_train[:n]`` (driver-conditioned via ``driver_idxs[:n]``) and
    ``n_empty`` counts empty rollouts. Mirrors v1's ``_gen_fidelity_pairs`` but
    keeps only the cell sequences (Fidelity-B is discriminator-free).
    """
    gen_cells = generate_trajectories(
        model, contexts[:n], max_len=max_len, device=device,
        gen_batch_size=gen_batch_size, progress=False,
        driver_idxs=driver_idxs[:n],
    )
    n_empty = sum(1 for c in gen_cells if not c)
    return gen_cells, n_empty


def _terminal_pickups_from_cells(gen_cells) -> List[Tuple[int, int, int]]:
    """(x, y, 0) of the LAST cell of each non-empty generated trajectory."""
    out: List[Tuple[int, int, int]] = []
    for cells in gen_cells:
        if not cells:
            continue
        x, y = int(cells[-1]) // gc.GY, int(cells[-1]) % gc.GY
        out.append((x, y, 0))
    return out


def _terminal_pickups_from_trajs(trajs) -> List[Tuple[int, int, int]]:
    """(x, y, 0) of the LAST state of each real/edited trajectory."""
    out: List[Tuple[int, int, int]] = []
    for t in trajs:
        s = t.states[-1]
        out.append((int(s.x_grid), int(s.y_grid), 0))
    return out


# ---------------------------------------------------------------- assembly ----

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.baselines.run_level1_table_v2",
        description="Assemble the Level-1 v2 data-quality table "
                    "(driver-conditioned; raw/edited/BC/GAN).",
    )
    ap.add_argument(
        "--edit-dir", type=str,
        default="famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup",
        help="Persisted editing run (provides edited trajectories + histories.pkl).",
    )
    ap.add_argument("--mle-epochs", type=int, default=20)
    ap.add_argument("--adv-epochs", type=int, default=3,
                    help="GAN source ONLY; BC is always pure MLE (adv_epochs=0).")
    ap.add_argument("--gan-loss", type=str, default="wgan-gp",
                    choices=["bce", "wgan-gp"])
    ap.add_argument("--n-critic", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=gc.MAX_TRAIN_TOKENS)
    ap.add_argument("--fidelity-sample-size", type=int, default=5000)
    ap.add_argument("--gen-batch-size", type=int, default=gc.GEN_BATCH_SIZE)
    # ---- v2-specific identity Fidelity-A controls ----
    ap.add_argument("--max-eval-drivers", type=int, default=50,
                    help="Cap on the number of eval drivers for identity Fidelity-A.")
    ap.add_argument("--pairs-per-driver", type=int, default=20,
                    help="Identity-branch slot-0 pairs sampled per eval driver.")
    ap.add_argument("--min-driver-trajs", type=int, default=6,
                    help="Min real trajectories for a driver to be eligible.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

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
            "Level-1 Fidelity-A (HuMID) requires the trained discriminator."
        )
    # load_discriminator maps weights to CPU; move to the run device so the
    # CUDA input tensors built in fidelity_eval meet on-device weights (else a
    # device-mismatch RuntimeError on the first forward — the validation gate).
    disc = load_discriminator(ckpt).to(device)

    # ---- data ----
    _log(f"[level1-v2] loading bundle (device={device})")
    bundle = DataBundle.load()
    # histories.pkl is produced locally by FAMAIL's own editing runner (see
    # algorithm/persistence.py); it is a trusted in-repo artifact, not external
    # input -- pickle.load is safe here (mirrors gan/variants.py).
    with open(Path(args.edit_dir) / "histories.pkl", "rb") as f:
        histories = pickle.load(f)

    raw_trajs = bundle.trajectories
    driver_to_idx = build_driver_index(raw_trajs)
    groups = group_by_driver(raw_trajs)
    profiles = bundle.multi_stream.profile_features
    zeros11 = np.zeros(11, dtype=np.float32)

    # ---- train the two driver-conditioned generative sources ----
    _log(f"[level1-v2] training BC (MLE-only, {args.mle_epochs} epochs, "
         f"n_drivers={len(driver_to_idx)})")
    bc = _train_and_generate_cond(
        raw_trajs, driver_to_idx, adv_epochs=0, gan_loss="bce", n_critic=1,
        mle_epochs=args.mle_epochs, max_len=max_len, max_tokens=args.max_tokens,
        device=device, seed=args.seed,
    )
    _log(f"[level1-v2] training GAN ({args.gan_loss}, mle={args.mle_epochs}, "
         f"adv={args.adv_epochs}, n_critic={args.n_critic})")
    gan = _train_and_generate_cond(
        raw_trajs, driver_to_idx, adv_epochs=args.adv_epochs, gan_loss=args.gan_loss,
        n_critic=args.n_critic, mle_epochs=args.mle_epochs, max_len=max_len,
        max_tokens=args.max_tokens, device=device, seed=args.seed,
    )

    # ====================================================================
    # Fidelity-A: identity-aware, real-anchored gate + per-source matched
    # ====================================================================
    # A single rng drives ALL identity-branch sampling for determinism.
    rng = random.Random(args.seed)

    eval_drivers = _select_eval_drivers(
        groups, min_trajs=args.min_driver_trajs, max_drivers=args.max_eval_drivers,
    )
    if not eval_drivers:
        raise SystemExit(
            f"no driver has >= {args.min_driver_trajs} real trajectories; "
            "cannot build identity Fidelity-A pairs."
        )
    _log(f"[level1-v2] identity Fidelity-A over {len(eval_drivers)} eval drivers")
    ppd = args.pairs_per_driver

    # Group edited histories by their (preserved) driver_id so each driver's
    # edited slot-0 set draws from its own modified trajectories.
    edited_by_driver: Dict[int, list] = {}
    for h in histories:
        edited_by_driver.setdefault(int(h.modified.driver_id), []).append(h.modified)

    # Accumulate matched + mismatched pairs per source across eval drivers.
    matched: Dict[str, list] = {k: [] for k in _SOURCE_ORDER}
    mismatched: Dict[str, list] = {k: [] for k in _SOURCE_ORDER}

    for k, d in enumerate(eval_drivers):
        dprime = eval_drivers[(k + 1) % len(eval_drivers)]  # mismatch partner
        real_d = groups[d]
        real_dp = groups[dprime]
        # Context pools: ALL of each driver's real trajectories (build_identity_
        # branch samples slots 1..N-1 from these).
        real_context_d = _real_context_tensors(real_d)
        real_context_dp = _real_context_tensors(real_dp)
        prof_d = profiles.get(d)
        prof_dp = profiles.get(dprime)
        if prof_d is None:
            _log(f"[level1-v2] WARN driver {d} has no profile -> zero profile")
            prof_d = zeros11
        if prof_dp is None:
            _log(f"[level1-v2] WARN driver {dprime} has no profile -> zero profile")
            prof_dp = zeros11

        # REAL branch slot-0: up to ppd of d's real trajectories (the
        # trajectory-under-test for the REAL/anchor branch).
        real_slot0 = [fe.real_to_disc_tensor(t) for t in real_d[:ppd]]
        if not real_slot0:
            continue

        # ---- raw source: source_slot0 = a DISJOINT sample of d's real trajs
        # (real-d vs another real-d). If d has too few for a disjoint second
        # half, sample WITH REPLACEMENT (note: degrades to overlapping pairs).
        n_avail = len(real_d)
        if n_avail >= 2 * len(real_slot0):
            raw_slot0 = [
                fe.real_to_disc_tensor(t)
                for t in real_d[len(real_slot0):len(real_slot0) + len(real_slot0)]
            ]
        else:
            # too few for disjoint -> sample with replacement
            raw_slot0 = [
                fe.real_to_disc_tensor(real_d[rng.randrange(n_avail)])
                for _ in range(len(real_slot0))
            ]
        m, mm = _build_source_pairs(
            real_slot0=real_slot0, source_slot0=raw_slot0,
            real_context=real_context_d, source_context_other=real_context_dp,
            profile_d=prof_d, profile_dp=prof_dp, rng=rng,
        )
        matched["raw"].extend(m)
        mismatched["raw"].extend(mm)

        # ---- edited source: source_slot0 = edited-d modified trajectories
        ed_d = edited_by_driver.get(d, [])
        if ed_d:
            edited_slot0 = [fe.real_to_disc_tensor(t) for t in ed_d[:ppd]]
            m, mm = _build_source_pairs(
                real_slot0=real_slot0, source_slot0=edited_slot0,
                real_context=real_context_d, source_context_other=real_context_dp,
                profile_d=prof_d, profile_dp=prof_dp, rng=rng,
            )
            matched["edited"].extend(m)
            mismatched["edited"].extend(mm)
        # (driver with no edited trajectories is skipped for the edited source only)

        # ---- bc / gan sources: driver-conditioned generated slot-0
        for name, src in (("bc", bc), ("gan", gan)):
            gen_slot0 = _gen_cond_slot0(
                src["model"], real_d, driver_to_idx[d],
                pairs_per_driver=ppd, max_len=max_len, device=device,
                gen_batch_size=args.gen_batch_size,
            )
            if not gen_slot0:
                continue
            m, mm = _build_source_pairs(
                real_slot0=real_slot0, source_slot0=gen_slot0,
                real_context=real_context_d, source_context_other=real_context_dp,
                profile_d=prof_d, profile_dp=prof_dp, rng=rng,
            )
            matched[name].extend(m)
            mismatched[name].extend(mm)

    # ---- gate (real-anchored): raw matched vs raw mismatched ----
    _log("[level1-v2] running real-anchored validation gate")
    gate = fe.identity_validation_gate(
        disc, matched_pairs=matched["raw"], mismatched_pairs=mismatched["raw"],
        device=device,
    )
    trusted = bool(gate["passed"])

    # ---- per-source Fidelity-A + separation ----
    _log("[level1-v2] scoring per-source identity Fidelity-A")
    fidelity_a: Dict[str, float] = {}
    separation: Dict[str, float] = {}
    for key in _SOURCE_ORDER:
        a_match = fe.humid_identity_fidelity(disc, matched[key], device=device)
        a_mis = fe.humid_identity_fidelity(disc, mismatched[key], device=device)
        fidelity_a[key] = float(a_match["mean"])
        separation[key] = float(a_match["mean"] - a_mis["mean"])

    # ====================================================================
    # Fidelity-B: enriched 5-key distributional + terminal-cell JS
    # ====================================================================
    bc_n = min(fss, len(bc["contexts"]))
    gan_n = min(fss, len(gan["contexts"]))
    _log(f"[level1-v2] generating BC Fidelity-B rollouts (N={bc_n})")
    bc_cells, bc_empty = _gen_fidelity_full(
        bc["model"], bc["filtered_train"], bc["contexts"], bc["driver_idxs"],
        n=bc_n, max_len=max_len, device=device, gen_batch_size=args.gen_batch_size,
    )
    _log(f"[level1-v2] generating GAN Fidelity-B rollouts (N={gan_n})")
    gan_cells, gan_empty = _gen_fidelity_full(
        gan["model"], gan["filtered_train"], gan["contexts"], gan["driver_idxs"],
        n=gan_n, max_len=max_len, device=device, gen_batch_size=args.gen_batch_size,
    )

    _log("[level1-v2] scoring Fidelity-B (5-key distributional + terminal-cell JS)")
    raw_sample = raw_trajs[:fss]
    edited_sample = histories[:fss]
    raw_stats = [fe.trajectory_statistics(t) for t in raw_sample]
    edited_stats = [fe.trajectory_statistics(h.modified) for h in edited_sample]
    bc_stats = [fe.trajectory_statistics(c) for c in bc_cells if c]
    gan_stats = [fe.trajectory_statistics(c) for c in gan_cells if c]
    ranges = fe.stat_ranges(
        [raw_stats, edited_stats, bc_stats, gan_stats], keys=fe._STAT_KEYS_V2,
    )
    b_edited = fe.distributional_fidelity(
        edited_stats, raw_stats, ranges=ranges, keys=fe._STAT_KEYS_V2,
    )
    b_bc = fe.distributional_fidelity(
        bc_stats, raw_stats, ranges=ranges, keys=fe._STAT_KEYS_V2,
    )
    b_gan = fe.distributional_fidelity(
        gan_stats, raw_stats, ranges=ranges, keys=fe._STAT_KEYS_V2,
    )

    # ---- terminal-cell JS per source (vs raw terminal cells) ----
    raw_pickups = _terminal_pickups_from_trajs(raw_sample)
    edited_pickups = _terminal_pickups_from_trajs([h.modified for h in edited_sample])
    bc_pickups_term = _terminal_pickups_from_cells(bc_cells)
    gan_pickups_term = _terminal_pickups_from_cells(gan_cells)
    tjs_edited = fe.terminal_cell_distribution_js(edited_pickups, raw_pickups)
    tjs_bc = fe.terminal_cell_distribution_js(bc_pickups_term, raw_pickups)
    tjs_gan = fe.terminal_cell_distribution_js(gan_pickups_term, raw_pickups)

    def _b_component(dist: dict, terminal_js: float) -> Tuple[dict, float]:
        """per-component dict (5 stat JS + terminal_cell) + aggregate mean."""
        comp = dict(dist["per_stat"])
        comp["terminal_cell"] = float(terminal_js)
        agg = float(np.mean(list(dist["per_stat"].values()) + [terminal_js]))
        return comp, agg

    comp_edited, fb_edited = _b_component(b_edited, tjs_edited)
    comp_bc, fb_bc = _b_component(b_bc, tjs_bc)
    comp_gan, fb_gan = _b_component(b_gan, tjs_gan)
    # Raw vs raw is 0.0 by definition; avoid a degenerate self-call.
    comp_raw = {k: 0.0 for k in fe._STAT_KEYS_V2}
    comp_raw["terminal_cell"] = 0.0

    # ====================================================================
    # Fairness: identical to v1 (driver-conditioned pickups for bc/gan)
    # ====================================================================
    # Fairness is a CORPUS-SCALE demand-grid metric, so BC/GAN fairness rollouts
    # deliberately cover the FULL filtered corpus (not the fidelity sample): raw/
    # edited fairness already use every trajectory, and a sub-sampled demand grid
    # would not be comparable. This is the run's largest single compute cost.
    _log("[level1-v2] scoring single-seed fairness")
    f_raw = data_level_fairness(bundle)
    f_edited = _edited_fairness_from_metrics(Path(args.edit_dir))
    bc_pickups = generate_pickups(
        bc["model"], bc["contexts"], max_len=max_len, device=device,
        gen_batch_size=args.gen_batch_size, progress=False,
        driver_idxs=bc["driver_idxs"],
    )
    f_bc = data_level_fairness(bundle, pickup_3d=pickups_to_pickup_3d(bundle, bc_pickups))
    gan_pickups = generate_pickups(
        gan["model"], gan["contexts"], max_len=max_len, device=device,
        gen_batch_size=args.gen_batch_size, progress=False,
        driver_idxs=gan["driver_idxs"],
    )
    f_gan = data_level_fairness(bundle, pickup_3d=pickups_to_pickup_3d(bundle, gan_pickups))

    # ---- assemble result ----
    result = {
        "edit_dir": args.edit_dir,
        "gate": gate,
        "n_eval_drivers": len(eval_drivers),
        "sources": {
            "raw": {
                "f_causal": f_raw["f_causal"], "f_spatial": f_raw["f_spatial"],
                # Raw is the anchor: Fidelity-A is the gate's matched mean, the
                # separation is the gate high-low (the well-posedness margin).
                "fidelity_a": float(gate["high_matched"]),
                "fidelity_a_separation": float(gate["high_matched"] - gate["low_mismatched"]),
                "fidelity_a_trusted": trusted,
                "fidelity_b": 0.0, "fidelity_b_per_component": comp_raw,
                "n_empty": 0,
            },
            "edited": {
                "f_causal": f_edited["f_causal"], "f_spatial": f_edited["f_spatial"],
                "fidelity_a": fidelity_a["edited"],
                "fidelity_a_separation": separation["edited"],
                "fidelity_a_trusted": trusted,
                "fidelity_b": fb_edited, "fidelity_b_per_component": comp_edited,
                "n_empty": 0,
            },
            "bc": {
                "f_causal": f_bc["f_causal"], "f_spatial": f_bc["f_spatial"],
                "fidelity_a": fidelity_a["bc"],
                "fidelity_a_separation": separation["bc"],
                "fidelity_a_trusted": trusted,
                "fidelity_b": fb_bc, "fidelity_b_per_component": comp_bc,
                "n_empty": bc_empty,
            },
            "gan": {
                "f_causal": f_gan["f_causal"], "f_spatial": f_gan["f_spatial"],
                "fidelity_a": fidelity_a["gan"],
                "fidelity_a_separation": separation["gan"],
                "fidelity_a_trusted": trusted,
                "fidelity_b": fb_gan, "fidelity_b_per_component": comp_gan,
                "n_empty": gan_empty,
            },
        },
    }

    # ---- persistence ----
    out_dir = args.out_dir
    if out_dir is None:
        stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = Path(config.PACKAGE_ROOT) / "results" / "level1_table_v2" / stamp
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "level1_v2_metrics.json").write_text(result_to_json(result))
    (out_dir / "level1_v2_table.md").write_text(render_table_v2(result))

    def _stat_arr(stats):
        keys = fe._STAT_KEYS_V2
        if not stats:
            return np.zeros((0, len(keys)), dtype=np.float64)
        return np.asarray([[s[k] for k in keys] for s in stats], dtype=np.float64)

    np.savez(
        out_dir / "trajectory_stats.npz",
        raw=_stat_arr(raw_stats), edited=_stat_arr(edited_stats),
        bc=_stat_arr(bc_stats), gan=_stat_arr(gan_stats),
    )
    training_curves = {"bc": _curves_for_source(bc), "gan": _curves_for_source(gan)}
    (out_dir / "training_curves.json").write_text(
        json.dumps(training_curves, indent=2, default=float)
    )
    (out_dir / "driver_index.json").write_text(
        json.dumps({str(k): v for k, v in driver_to_idx.items()}, indent=2)
    )

    # ---- summary ----
    _log("")
    _log(render_table_v2(result))
    if not gate["passed"]:
        _log("[level1-v2] Validation gate FAILED -> Fidelity-A is UNTRUSTED; "
             "Fidelity-B (distributional divergence) is the PRIMARY fidelity metric.")
    _log(f"[level1-v2] training curves: "
         f"BC mle final={training_curves['bc']['mle_epoch_losses'][-1]:.3f}; "
         f"GAN mle final={training_curves['gan']['mle_epoch_losses'][-1]:.3f}")
    _log(f"[level1-v2] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
