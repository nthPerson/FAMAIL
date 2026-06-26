"""CLI: multi-seed paired B0-vs-FAMAIL variance suite (Meeting 37 action item).

Trains N paired (B0 raw-corpus, FAMAIL edited-corpus) MLE-only generators —
the SAME seed within each pair, so the only difference inside a pair is the
training data — then reports:
  - mean +/- std (sample std, ddof=1) of production F_spatial / F_causal,
    DI (both Y conventions), and localized F_causal, per variant;
  - paired deltas (FAMAIL - B0) mean +/- std;
  - the JS noise floor: within-variant pairwise JS across seeds, vs the
    cross-variant paired JS (the "transmitted signal" of the 2026-06-08
    metric-hardening report, now with error bars).

Per-seed artifacts persist INCREMENTALLY (seed_<k>.json after each pair), so a
crash mid-suite preserves completed seeds.

Example:
    python -m famail_temporal.baselines.run_variance_suite \\
        --seeds 0,1,2,3,4 --mle-epochs 5 --device auto
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.model_level import fit_and_evaluate
from famail_temporal.baselines.gan.variants import load_edited_trajectories
from famail_temporal.baselines.gan.rollout import pickups_to_pickup_3d
from famail_temporal.baselines.transmission import (
    terminal_cell_histogram, trajectories_terminal_histogram,
    jensen_shannon_divergence,
)
from famail_temporal.baselines.district_metrics import di_from_bundle_and_pickup_grid
from famail_temporal.baselines.localized_metrics import (
    edited_units_from_histories, localized_f_causal,
)
from famail_temporal.baselines._manifest import write_run_manifest, append_timing


DEFAULT_EDIT_DIR = (
    Path(config.PACKAGE_ROOT) / "results"
    / "2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup"
)

# Scalar metrics tracked per (variant, seed) for aggregation.
METRIC_KEYS = [
    "f_spatial", "f_causal",
    "di_primary", "di_supplementary",
    "f_causal_localized", "f_causal_global_mi",
]


# ---------------- pure helpers (unit-tested) ----------------

def adv_curve_or_none(result: dict) -> dict | None:
    """The adversarial training curve for a fit_and_evaluate result, or None.

    Returns None when no adversarial phase ran (empty loss lists) -- the
    variance suite uses adv_epochs=0, so B0/FAMAIL are pure-MLE and this is
    None. Kept so a future adv_epochs>0 run persists g/d curves automatically.
    """
    adv = result.get("adv_losses") or {}
    if not adv.get("g_losses") and not adv.get("d_losses"):
        return None
    return {
        "g_epoch_losses": [float(x) for x in adv.get("g_losses", [])],
        "d_epoch_losses": [float(x) for x in adv.get("d_losses", [])],
        "g_batch_losses": [float(x) for x in adv.get("g_batch_losses", [])],
        "d_batch_losses": [float(x) for x in adv.get("d_batch_losses", [])],
    }


def mean_std(values: List[float]) -> Dict[str, float]:
    """Sample statistics (ddof=1; std=0 for a single value; NaN for none).

    The empty case matters for single-seed runs: with one seed there are no
    within-variant JS pairs, and an unguarded arr.min() on a zero-size array
    raises — which would discard a completed (GPU-expensive) suite at the
    final aggregation step.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def paired_delta_stats(*, b0: List[float], famail: List[float]) -> Dict[str, float]:
    """Stats of per-seed deltas (FAMAIL - B0); pairs are index-aligned."""
    return mean_std([f - b for b, f in zip(b0, famail)])


def pairwise_js_stats(histograms: List[np.ndarray]) -> Dict[str, float]:
    """Stats over all C(n,2) within-group JS pairs — the seed noise floor."""
    vals = [
        jensen_shannon_divergence(histograms[i], histograms[j])
        for i in range(len(histograms))
        for j in range(i + 1, len(histograms))
    ]
    return mean_std(vals)


def cross_js_values(a: List[np.ndarray], b: List[np.ndarray]) -> List[float]:
    """Index-paired JS between two histogram lists (B0_i vs FAMAIL_i)."""
    return [jensen_shannon_divergence(x, y) for x, y in zip(a, b)]


def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2, default=float)


# ---------------- orchestration ----------------

def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _seed_metrics(bundle, result, edited_units) -> Dict[str, float]:
    """Extract the tracked scalars + histogram from one fit_and_evaluate result."""
    grid = pickups_to_pickup_3d(bundle, result["pickups"])
    di = di_from_bundle_and_pickup_grid(bundle, grid)
    loc = localized_f_causal(bundle, grid, edited_units)
    return {
        "f_spatial": float(result["generated"]["f_spatial"]),
        "f_causal": float(result["generated"]["f_causal"]),
        "di_primary": float(di["di_primary"]),
        "di_supplementary": float(di["di_supplementary"]),
        "f_causal_localized": float(loc["f_causal_localized"]),
        "f_causal_global_mi": float(loc["f_causal_global"]),
        "final_mle_loss": float(result["mle_losses"][-1]),
        # Full per-epoch curve: the convergence evidence behind the
        # strengthened (20-epoch) pretraining choice (2026-06-10 direction).
        "mle_losses": [float(x) for x in result["mle_losses"]],
        "mle_batch_losses": [float(x) for x in result.get("mle_batch_losses", [])],
        "adv_curve": adv_curve_or_none(result),
    }


def _write_report(out_dir: Path, agg: dict, seeds: List[int]) -> None:
    rows_variant = []
    for key in METRIC_KEYS:
        b = agg["b0"][key]
        f = agg["famail"][key]
        d = agg["paired_delta"][key]
        rows_variant.append(
            f"| {key} | {b['mean']:.4f} +/- {b['std']:.4f} "
            f"| {f['mean']:.4f} +/- {f['std']:.4f} "
            f"| **{d['mean']:+.4f} +/- {d['std']:.4f}** |"
        )
    js = agg["js"]
    report = f"""# Variance suite report (seeds {seeds})

Paired B0 (raw corpus) vs FAMAIL (edited corpus), MLE-only, same seed within
each pair. Sample std (ddof=1), n={len(seeds)}.

## Definitions

- **B0** — LSTM grid-cell-sequence generator trained with teacher-forced MLE
  only (behavioral cloning) on the RAW corpus.
- **FAMAIL** — identical architecture/recipe/seed, trained on the EDITED
  corpus (the persisted editing run supplied via --edit-dir).
- **Paired (by seed)** — B0 and FAMAIL share the RNG seed within a pair, so
  the only difference is the training corpus; paired deltas (FAMAIL - B0)
  cancel seed-level variance.
- **f_spatial / f_causal** — production fairness metrics (higher = fairer):
  geographic evenness of pickups / 1 - fraction of the supply-demand residual
  explained by district demographics.
- **f_causal_localized** — M=I f_causal restricted to the active units the
  edit relocated pickups out of; **f_causal_global_mi** — the same M=I
  formula on ALL active units (the paired global comparator).
- **di_primary / di_supplementary** — district disparate-impact ratio
  (top-3 vs bottom-3 districts by hukou ratio) under Y = supply/demand
  (f_causal-aligned) / Y = demand/supply (robustness lens); 1.0 = parity.
- **JS** — Jensen-Shannon divergence between terminal-pickup-cell histograms,
  in bits (0 = identical, 1 = disjoint). Terminal cells are the channel any
  model-level fairness effect must pass through. The within-variant pairwise
  JS across seeds is the seed NOISE FLOOR; the cross-variant paired JS is the
  SIGNAL; JS(p_raw, p_edited) is the data-level TARGET the edit created.

## Fairness metrics, mean +/- std

| Metric | B0 | FAMAIL | paired Delta (FAMAIL - B0) |
|---|---:|---:|---:|
{chr(10).join(rows_variant)}

## JS noise floor vs transmitted signal (terminal-cell histograms, bits)

| Quantity | mean +/- std | n |
|---|---:|---:|
| within-B0 pairwise JS (seed noise floor) | {js['within_b0']['mean']:.5f} +/- {js['within_b0']['std']:.5f} | {js['within_b0']['n']} |
| within-FAMAIL pairwise JS | {js['within_famail']['mean']:.5f} +/- {js['within_famail']['std']:.5f} | {js['within_famail']['n']} |
| cross-variant paired JS (signal) | {js['cross_paired']['mean']:.5f} +/- {js['cross_paired']['std']:.5f} | {js['cross_paired']['n']} |
| JS(p_raw, p_edited) (data-level target) | {js['js_target']:.5f} | 1 |
| transmission ratio (cross / target) | {js['transmission_ratio']['mean']:.3f} +/- {js['transmission_ratio']['std']:.3f} | {js['transmission_ratio']['n']} |

Reading: the cross-variant JS is a real distributional signal only if it
clears the within-variant noise floor. If cross ~ within, the generated
distributions of B0 and FAMAIL differ no more than two B0 re-trainings do.
"""
    (out_dir / "report.md").write_text(report)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.baselines.run_variance_suite",
    )
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="comma-separated seed list; one paired run per seed")
    ap.add_argument("--mle-epochs", type=int, default=gc.MLE_EPOCHS)
    ap.add_argument("--max-len", type=int, default=gc.MAX_GEN_LEN)
    ap.add_argument("--mle-batch-size", type=int, default=gc.MLE_BATCH_SIZE)
    ap.add_argument("--max-tokens", type=int, default=gc.MAX_TRAIN_TOKENS)
    ap.add_argument("--gen-batch-size", type=int, default=gc.GEN_BATCH_SIZE)
    ap.add_argument("--edit-dir", type=Path, default=DEFAULT_EDIT_DIR)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    if not seeds:
        raise SystemExit("--seeds must name at least one seed")
    device = _resolve_device(args.device)

    if args.out_dir is None:
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = (
            Path(config.PACKAGE_ROOT) / "baselines" / "variance_suite"
            / "results" / f"{timestamp}_seeds{seeds[0]}-{seeds[-1]}"
        )
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = DataBundle.load()
    edited_trajs = load_edited_trajectories(bundle, args.edit_dir)
    edited_units = edited_units_from_histories(args.edit_dir)
    p_raw = trajectories_terminal_histogram(bundle.trajectories)
    p_edited = trajectories_terminal_histogram(edited_trajs)
    js_target = jensen_shannon_divergence(p_raw, p_edited)

    common = dict(
        mle_epochs=args.mle_epochs, adv_epochs=0, max_len=args.max_len,
        mle_batch_size=args.mle_batch_size,
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
        gen_batch_size=args.gen_batch_size,
        device=device, progress=not args.quiet,
    )

    per_seed: List[dict] = []
    hists_b0: List[np.ndarray] = []
    hists_fam: List[np.ndarray] = []
    for k, seed in enumerate(seeds):
        print(f"\n=== seed {seed} ({k + 1}/{len(seeds)}) ===", flush=True)
        b0 = fit_and_evaluate(bundle, seed=seed, **common)
        fam = fit_and_evaluate(
            bundle, train_trajectories=edited_trajs, seed=seed, **common,
        )
        hb = terminal_cell_histogram(b0["pickups"])
        hf = terminal_cell_histogram(fam["pickups"])
        hists_b0.append(hb)
        hists_fam.append(hf)
        entry = {
            "seed": seed,
            "b0": _seed_metrics(bundle, b0, edited_units),
            "famail": _seed_metrics(bundle, fam, edited_units),
            "js_cross": float(jensen_shannon_divergence(hb, hf)),
        }
        per_seed.append(entry)
        # Incremental persistence: a crash preserves completed seeds.
        (out_dir / f"seed_{seed}.json").write_text(result_to_json(entry))
        print(
            f"[seed {seed}] B0 f_causal={entry['b0']['f_causal']:.4f}  "
            f"FAMAIL f_causal={entry['famail']['f_causal']:.4f}  "
            f"delta={entry['famail']['f_causal'] - entry['b0']['f_causal']:+.4f}  "
            f"js_cross={entry['js_cross']:.5f}",
            flush=True,
        )

    cross_vals = cross_js_values(hists_b0, hists_fam)
    agg = {
        "seeds": seeds,
        "b0": {k: mean_std([e["b0"][k] for e in per_seed]) for k in METRIC_KEYS},
        "famail": {k: mean_std([e["famail"][k] for e in per_seed]) for k in METRIC_KEYS},
        "paired_delta": {
            k: paired_delta_stats(
                b0=[e["b0"][k] for e in per_seed],
                famail=[e["famail"][k] for e in per_seed],
            ) for k in METRIC_KEYS
        },
        "js": {
            "within_b0": pairwise_js_stats(hists_b0),
            "within_famail": pairwise_js_stats(hists_fam),
            "cross_paired": mean_std(cross_vals),
            "js_target": float(js_target),
            "transmission_ratio": mean_std(
                [v / js_target for v in cross_vals]
            ) if js_target > 0 else {"mean": float("nan"), "std": float("nan"),
                                     "min": float("nan"), "max": float("nan"),
                                     "n": len(cross_vals)},
        },
        "config": {
            "mle_epochs": args.mle_epochs, "max_len": args.max_len,
            "mle_batch_size": args.mle_batch_size, "max_tokens": args.max_tokens,
            "edit_dir": str(args.edit_dir),
        },
        "per_seed_values": {
            arm: {k: [float(e[arm][k]) for e in per_seed] for k in METRIC_KEYS}
            for arm in ("b0", "famail")
        },
    }
    (out_dir / "aggregate.json").write_text(result_to_json(agg))
    np.savez(
        out_dir / "terminal_cell_histograms.npz",
        p_raw=p_raw, p_edited=p_edited,
        **{f"p_b0_seed{s}": h for s, h in zip(seeds, hists_b0)},
        **{f"p_famail_seed{s}": h for s, h in zip(seeds, hists_fam)},
    )
    _write_report(out_dir, agg, seeds)

    # ---- provenance ----
    write_run_manifest(out_dir, argv=sys.argv, seeds=seeds, edit_dir=str(args.edit_dir),
                       extra={})
    append_timing(out_dir / "timings.jsonl", "variance_suite", time.time() - t0)

    print("\n=== Variance suite summary ===")
    for key in ("f_causal", "f_spatial"):
        d = agg["paired_delta"][key]
        print(
            f"paired delta {key}: {d['mean']:+.4f} +/- {d['std']:.4f} "
            f"(min {d['min']:+.4f}, max {d['max']:+.4f}, n={d['n']})"
        )
    js = agg["js"]
    print(
        f"JS noise floor (within-B0): {js['within_b0']['mean']:.5f} "
        f"+/- {js['within_b0']['std']:.5f}; "
        f"cross signal: {js['cross_paired']['mean']:.5f} "
        f"+/- {js['cross_paired']['std']:.5f}"
    )
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
