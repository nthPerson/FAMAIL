"""CLI: assemble the Level-1 data-quality table (Two-Level Argument, Level 1).

Compares four data sources -- raw, FAM-AIL edited, BC-generated, GAN-generated
-- on causal fairness, spatial fairness, and two fidelity metrics (HuMID paired
[gated] + discriminator-free distributional). See
docs/superpowers/specs/2026-06-17-level1-data-quality-table-design.md.

Example:
    python -m famail_temporal.baselines.run_level1_table \
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
from typing import List

import numpy as np
import torch

# gc MUST be imported before the argparse defaults reference gc.MAX_TRAIN_TOKENS
# / gc.GEN_BATCH_SIZE (else --help fails).
from famail_temporal.baselines.gan import config as gc
from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.utils.seeding import set_all_seeds
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
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


def render_table(result: dict) -> str:
    """Render the Level-1 table + gate verdict as markdown."""
    g = result["gate"]
    gate_line = (
        f"Validation gate: **{'PASSED' if g['passed'] else 'FAILED'}** "
        f"(real-real {g['high_real_real']:.3f} vs collapsed {g['low_collapsed']:.3f} / "
        f"shuffled {g['low_shuffled']:.3f}, margin {g['margin']:.2f})"
    )
    rows = []
    for key in _SOURCE_ORDER:
        s = result["sources"][key]
        a = f"{s['fidelity_a']:.3f}" + ("" if s["fidelity_a_trusted"] else " (untrusted)")
        rows.append(
            f"| {key} | {s['f_causal']:.4f} | {s['f_spatial']:.4f} "
            f"| {a} | {s['fidelity_b']:.4f} |"
        )
    return (
        "# Level-1 Data-Quality Table\n\n"
        f"Edit source: `{result['edit_dir']}`\n\n"
        f"{gate_line}\n\n"
        "_Fairness columns are single-seed (this table's internal coherence); the "
        "authoritative multi-seed fairness figures are the variance-suite 5-seed "
        "mean ± std._\n\n"
        "| Source | F_causal (single-seed) | F_spatial (single-seed) "
        "| Fidelity-A (HuMID, higher=better) "
        "| Fidelity-B (divergence, lower=better) |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(rows) + "\n"
    )


# --------------------------------------------------------- train + generate ----

def _train_and_generate(
    train_trajectories, *,
    adv_epochs, gan_loss, n_critic, mle_epochs, max_len, max_tokens,
    device, seed,
) -> dict:
    """Train a generator (MLE + optional adversarial) and return the model plus
    the index-aligned filtered training trajectories and their contexts.

    Mirrors ``model_level.fit_and_evaluate`` but (a) returns the trained model
    (which ``fit_and_evaluate`` does not) and (b) builds ONE ``filtered_train``
    list (the surviving Trajectory OBJECTS) from which BOTH ``sequences`` and
    ``contexts`` are derived -- the alignment guarantee (spec §10): the i-th
    generated trajectory must pair with ``filtered_train[i]``.

    ``adv_epochs == 0`` skips adversarial fine-tuning entirely (the pure-MLE B0
    "BC" source); only the GAN source passes ``adv_epochs > 0``.
    """
    set_all_seeds(seed)
    filtered_train = [
        t for t in train_trajectories
        if max_tokens is None or len(trajectory_to_tokens(t)) <= max_tokens
    ]
    if not filtered_train:
        raise ValueError(
            f"no training trajectories remain after the max_tokens={max_tokens} filter"
        )
    sequences = [trajectory_to_tokens(t) for t in filtered_train]
    contexts = [trajectory_context(t) for t in filtered_train]

    model = TrajectoryLSTM().to(device)
    mle_curve = train_mle(
        model, sequences, contexts,
        epochs=mle_epochs, lr=gc.MLE_LR, batch_size=gc.MLE_BATCH_SIZE,
        device=device, progress=False,
    )
    adv_curve = None
    if adv_epochs > 0:
        adv_curve = adversarial_finetune(
            model, sequences, contexts,
            epochs=adv_epochs, lr_g=gc.ADV_LR_G, lr_d=gc.ADV_LR_D,
            batch_size=gc.ADV_BATCH_SIZE, max_len=max_len,
            tau_start=gc.GUMBEL_TAU_START, tau_end=gc.GUMBEL_TAU_END,
            d_update_every=gc.D_UPDATE_EVERY, mle_lambda=gc.ADV_MLE_LAMBDA,
            gan_loss=gan_loss, gp_lambda=gc.WGAN_GP_LAMBDA, n_critic=n_critic,
            device=device, progress=False,
        )
    return {
        "model": model, "filtered_train": filtered_train, "contexts": contexts,
        "mle_curve": mle_curve, "adv_curve": adv_curve,
    }


# ------------------------------------------------------------- fairness rows ----

def _fairness_from_pickups(bundle, pickups) -> dict:
    grid = pickups_to_pickup_3d(bundle, pickups)
    return data_level_fairness(bundle, pickup_3d=grid)


def _edited_pickups(histories) -> List[tuple]:
    """Terminal-state pickups of each edited (.modified) trajectory.

    t_block reuses the modified trajectory's context block
    (``trajectory_context(h.modified)[1]``) -- the same block the editor used.
    """
    pickups = []
    for h in histories:
        s = h.modified.states[-1]
        t_block = trajectory_context(h.modified)[1]
        pickups.append((int(s.x_grid), int(s.y_grid), t_block))
    return pickups


def _curves_for_source(src: dict) -> dict:
    """Flatten one source's captured training curves into a JSON-ready dict.

    ``src`` is a ``_train_and_generate`` result. BC has ``adv_curve=None`` (pure
    MLE), so its ``adv`` entry is null; the GAN source carries both phases.
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


# ---------------------------------------------------------------- assembly ----

def _gen_fidelity_pairs(model, filtered_train, contexts, *, n, max_len, device,
                        gen_batch_size):
    """BC/GAN fidelity pairs over the first ``n`` contexts (+ the gen cells).

    Returns ``(pairs, gen_cells, n_empty)`` where ``gen_cells`` is the full list
    of generated cell-id sequences (index-aligned with ``filtered_train[:n]``)
    and ``pairs`` skips empty rollouts (counted in ``n_empty``).
    """
    gen_cells = generate_trajectories(
        model, contexts[:n], max_len=max_len, device=device,
        gen_batch_size=gen_batch_size, progress=False,
    )
    pairs = []
    n_empty = 0
    for i in range(len(gen_cells)):
        if not gen_cells[i]:
            n_empty += 1
            continue
        real = filtered_train[i]
        # Synthesize the generated trajectory's time/day from the paired real
        # seed's first state. We pass the raw `time_bucket` (domain ~[1,288]),
        # NOT the coarse context `t_block`: the discriminator's FeatureNormalizer
        # encodes time as 2*pi*time_bucket/288 and the real branch feeds raw
        # buckets, so real and generated must meet in the SAME domain. Do not
        # "fix" this toward spec §3.4's looser "time block" wording.
        pairs.append((
            fe.real_to_disc_tensor(real),
            fe.generated_to_disc_tensor(
                gen_cells[i],
                time_bucket=real.states[0].time_bucket,
                day_index=real.states[0].day_index,
            ),
        ))
    return pairs, gen_cells, n_empty


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.baselines.run_level1_table",
        description="Assemble the Level-1 data-quality table (raw/edited/BC/GAN).",
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
    _log(f"[level1] loading bundle (device={device})")
    bundle = DataBundle.load()
    # histories.pkl is produced locally by FAMAIL's own editing runner (see
    # algorithm/persistence.py); it is a trusted in-repo artifact, not external
    # input -- pickle.load is safe here (mirrors gan/variants.py).
    with open(Path(args.edit_dir) / "histories.pkl", "rb") as f:
        histories = pickle.load(f)

    raw_trajs = bundle.trajectories
    raw_sample = raw_trajs[:fss]

    # ---- train the two generative sources ----
    _log(f"[level1] training BC (MLE-only, {args.mle_epochs} epochs)")
    bc = _train_and_generate(
        raw_trajs, adv_epochs=0, gan_loss="bce", n_critic=1,
        mle_epochs=args.mle_epochs, max_len=max_len, max_tokens=args.max_tokens,
        device=device, seed=args.seed,
    )
    _log(f"[level1] training GAN ({args.gan_loss}, mle={args.mle_epochs}, "
         f"adv={args.adv_epochs}, n_critic={args.n_critic})")
    gan = _train_and_generate(
        raw_trajs, adv_epochs=args.adv_epochs, gan_loss=args.gan_loss,
        n_critic=args.n_critic, mle_epochs=args.mle_epochs, max_len=max_len,
        max_tokens=args.max_tokens, device=device, seed=args.seed,
    )

    # ---- BC/GAN fidelity pairs + gen cells (full-trajectory, first N) ----
    bc_n = min(fss, len(bc["contexts"]))
    gan_n = min(fss, len(gan["contexts"]))
    _log(f"[level1] generating BC fidelity rollouts (N={bc_n})")
    bc_pairs, bc_cells, bc_empty = _gen_fidelity_pairs(
        bc["model"], bc["filtered_train"], bc["contexts"], n=bc_n,
        max_len=max_len, device=device, gen_batch_size=args.gen_batch_size,
    )
    _log(f"[level1] generating GAN fidelity rollouts (N={gan_n})")
    gan_pairs, gan_cells, gan_empty = _gen_fidelity_pairs(
        gan["model"], gan["filtered_train"], gan["contexts"], n=gan_n,
        max_len=max_len, device=device, gen_batch_size=args.gen_batch_size,
    )

    # ---- validation gate (collapsed = GAN's longest rollouts) ----
    _log("[level1] running validation gate")
    real_pairs = [
        (fe.real_to_disc_tensor(t), fe.real_to_disc_tensor(t)) for t in raw_sample
    ]
    gan_max_len = max((len(c) for c in gan_cells), default=0)
    K = min(500, gan_n)
    longest = sorted(range(len(gan_cells)),
                     key=lambda i: len(gan_cells[i]), reverse=True)[:K]
    collapsed_pairs = []
    for i in longest:
        if not gan_cells[i]:
            continue
        real = gan["filtered_train"][i]
        collapsed_pairs.append((
            fe.real_to_disc_tensor(real),
            fe.generated_to_disc_tensor(
                gan_cells[i],
                time_bucket=real.states[0].time_bucket,
                day_index=real.states[0].day_index,
            ),
        ))
    # shuffled: each real trajectory's own flat cells, randomly permuted.
    rng = random.Random(args.seed)
    shuffled_pairs = []
    for t in raw_sample:
        cells = [int(s.x_grid) * gc.GY + int(s.y_grid) for s in t.states]
        if not cells:
            continue
        shuffled_cells = rng.sample(cells, len(cells))
        shuffled_pairs.append((
            fe.real_to_disc_tensor(t),
            fe.generated_to_disc_tensor(
                shuffled_cells,
                time_bucket=t.states[0].time_bucket,
                day_index=t.states[0].day_index,
            ),
        ))
    gate = fe.validation_gate(
        disc, real_pairs=real_pairs, collapsed_pairs=collapsed_pairs,
        shuffled_pairs=shuffled_pairs, device=device,
    )

    # ---- Fidelity-A (HuMID paired) per source ----
    _log("[level1] scoring Fidelity-A (HuMID paired)")
    edited_sample = histories[:fss]
    edited_pairs = [
        (fe.real_to_disc_tensor(h.original), fe.real_to_disc_tensor(h.modified))
        for h in edited_sample
    ]
    a_edited = fe.humid_paired_fidelity(disc, edited_pairs, device=device)
    a_bc = fe.humid_paired_fidelity(disc, bc_pairs, device=device)
    a_gan = fe.humid_paired_fidelity(disc, gan_pairs, device=device)

    # ---- Fidelity-B (shared distributional grid) per source ----
    _log("[level1] scoring Fidelity-B (distributional, shared grid)")
    raw_stats = [fe.trajectory_statistics(t) for t in raw_sample]
    edited_stats = [fe.trajectory_statistics(h.modified) for h in edited_sample]
    bc_stats = [fe.trajectory_statistics(c) for c in bc_cells if c]
    gan_stats = [fe.trajectory_statistics(c) for c in gan_cells if c]
    ranges = fe.stat_ranges([raw_stats, edited_stats, bc_stats, gan_stats])
    b_edited = fe.distributional_fidelity(edited_stats, raw_stats, ranges=ranges)
    b_bc = fe.distributional_fidelity(bc_stats, raw_stats, ranges=ranges)
    b_gan = fe.distributional_fidelity(gan_stats, raw_stats, ranges=ranges)
    # Raw vs raw is 0.0 by definition; avoid a degenerate self-call.
    b_raw_per_stat = {k: 0.0 for k in ("length", "mean_displacement", "coverage")}

    # ---- single-seed fairness per source ----
    # Fairness is a CORPUS-SCALE demand-grid metric, so BC/GAN fairness rollouts
    # deliberately cover the FULL filtered corpus (not the fidelity sample `fss`):
    # raw/edited fairness already use every trajectory, and a sub-sampled demand
    # grid would not be comparable. This is the run's largest single compute cost.
    _log("[level1] scoring single-seed fairness")
    f_raw = data_level_fairness(bundle)
    f_edited = _fairness_from_pickups(bundle, _edited_pickups(histories))
    bc_pickups = generate_pickups(
        bc["model"], bc["contexts"], max_len=max_len, device=device,
        gen_batch_size=args.gen_batch_size, progress=False,
    )
    f_bc = _fairness_from_pickups(bundle, bc_pickups)
    gan_pickups = generate_pickups(
        gan["model"], gan["contexts"], max_len=max_len, device=device,
        gen_batch_size=args.gen_batch_size, progress=False,
    )
    f_gan = _fairness_from_pickups(bundle, gan_pickups)

    trusted = bool(gate["passed"])
    result = {
        "edit_dir": args.edit_dir,
        "gate": gate,
        "gan_max_len": int(gan_max_len),
        "sources": {
            "raw": {
                "f_causal": f_raw["f_causal"], "f_spatial": f_raw["f_spatial"],
                "fidelity_a": float(gate["high_real_real"]),
                # Raw is the anchor: fidelity_a is the gate's real-vs-real mean,
                # which carries no std (validation_gate returns means only), so
                # 0.0 is a placeholder, not a measured dispersion.
                "fidelity_a_std": 0.0, "fidelity_a_n": len(real_pairs),
                "fidelity_a_trusted": trusted,
                "fidelity_b": 0.0, "fidelity_b_per_stat": b_raw_per_stat,
                "n_empty": 0,
            },
            "edited": {
                "f_causal": f_edited["f_causal"], "f_spatial": f_edited["f_spatial"],
                "fidelity_a": a_edited["mean"], "fidelity_a_std": a_edited["std"],
                "fidelity_a_n": a_edited["n"], "fidelity_a_trusted": trusted,
                "fidelity_b": b_edited["aggregate"],
                "fidelity_b_per_stat": b_edited["per_stat"], "n_empty": 0,
            },
            "bc": {
                "f_causal": f_bc["f_causal"], "f_spatial": f_bc["f_spatial"],
                "fidelity_a": a_bc["mean"], "fidelity_a_std": a_bc["std"],
                "fidelity_a_n": a_bc["n"], "fidelity_a_trusted": trusted,
                "fidelity_b": b_bc["aggregate"],
                "fidelity_b_per_stat": b_bc["per_stat"], "n_empty": bc_empty,
            },
            "gan": {
                "f_causal": f_gan["f_causal"], "f_spatial": f_gan["f_spatial"],
                "fidelity_a": a_gan["mean"], "fidelity_a_std": a_gan["std"],
                "fidelity_a_n": a_gan["n"], "fidelity_a_trusted": trusted,
                "fidelity_b": b_gan["aggregate"],
                "fidelity_b_per_stat": b_gan["per_stat"], "n_empty": gan_empty,
            },
        },
    }

    # ---- persistence ----
    out_dir = args.out_dir
    if out_dir is None:
        stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = Path(config.PACKAGE_ROOT) / "results" / "level1_table" / stamp
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "level1_metrics.json").write_text(result_to_json(result))
    (out_dir / "level1_table.md").write_text(render_table(result))

    def _stat_arr(stats):
        keys = ("length", "mean_displacement", "coverage")
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

    # ---- summary ----
    _log("")
    _log(render_table(result))
    _log(f"[level1] gan_max_len={gan_max_len} (real mean ~18)")
    if gan_max_len <= 22:
        _log("[level1] CAVEAT: collapsed sample is not meaningfully longer than "
             "the real mean -- it is not a true degraded case (gate still runs).")
    if not gate["passed"]:
        _log("[level1] Validation gate FAILED -> Fidelity-A is UNTRUSTED; "
             "Fidelity-B (distributional divergence) is the PRIMARY fidelity metric.")
    _log(f"[level1] training curves: BC mle final={training_curves['bc']['mle_epoch_losses'][-1]:.3f}; "
         f"GAN mle final={training_curves['gan']['mle_epoch_losses'][-1]:.3f}")
    _log(f"[level1] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
