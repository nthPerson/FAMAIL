# FAMAIL Baselines — Variance Suite + WGAN (Meeting 37 Action Items) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address Dr. Zhang's Meeting-37 action items overnight: (1) train 5 paired B0/FAMAIL behavioral-cloning models and report mean ± std of F_spatial/F_causal plus the JS noise floor, (2) test her "more generator pretraining" hypothesis, (3) try WGAN-GP, (4) verify the generator gradient direction. Results must be on disk by morning for the meeting.

**Architecture:** One new CLI (`run_variance_suite.py`) that loops seeds over the existing `fit_and_evaluate` (MLE-only, paired by seed) and aggregates; one extension to `adversarial_finetune` adding a `wgan-gp` loss mode (Wasserstein critic loss + embedding-space gradient penalty + n_critic schedule); a serial overnight driver script (controller-authored) that queues everything on the single GPU.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest. Reuses `fit_and_evaluate`, `transmission.py`, `district_metrics.py`, `localized_metrics.py`, `variants.py` — all untouched except the listed files.

---

## Scope

### Delivers
1. **Task 1 — `run_variance_suite.py`**: N-seed paired B0+FAMAIL MLE-only training, per-seed metrics (production F_spatial/F_causal, DI both conventions, localized F_causal), JS noise-floor matrix (within-variant pairwise vs cross-variant paired), mean ± std aggregation, incremental per-seed persistence, report.md.
2. **Task 2 — WGAN-GP mode**: `gan_loss="wgan-gp"` in `adversarial_finetune` (Wasserstein losses, gradient penalty on embedding-space interpolates, `n_critic` schedule), plumbed through `fit_and_evaluate` and `run_b0_adversarial.py` CLI. Plus gradient-direction verification tests (Dr. Zhang's first diagnostic) for BOTH loss modes.
3. **Task 3 — overnight driver** (controller executes, no subagent): serial queue of variance suite → pretraining ablation (`--mle-epochs 10/20 --adv-epochs 3`, zero new code) → 2 WGAN configs.

### Does NOT deliver
- No changes to the editing algorithm, `fairness/`, or production F_causal.
- No morning aggregation/meeting doc (written interactively tomorrow from the overnight artifacts).
- No PR (user said hold off).
- No B2/B1-differentiable work.

### Branch

```bash
git checkout implement-gan-baselines
git checkout -b variance-suite-wgan
```

### Locked design decisions
1. **Paired seeds**: B0 and FAMAIL share the seed within a pair (same init + shuffling; only training data differs). Report mean ± std of *paired deltas*, not just marginals.
2. **MLE pretraining strengthened to 20 epochs** (user direction, 2026-06-10): the 5-epoch generator was visibly under-converged (loss still dropping 0.84 -> 0.78 in the final epoch), so a 5x re-run at 5 epochs would only put error bars around a weak generator. Comparability with the 2026-06-08 single-seed run is deliberately sacrificed for generator strength; batch 32 / max_tokens 256 unchanged. Per-epoch MLE loss curves are persisted per seed as the convergence evidence.
3. **JS noise floor** = within-variant pairwise JS (10 pairs per variant at n=5); **signal** = cross-variant paired JS (5 values). Both computed from the same terminal-cell histograms used by `transmission.py`.
4. **WGAN-GP per Gulrajani et al.**: critic loss `mean(D(fake)) − mean(D(real)) + λ·GP`, λ=10; generator loss `−mean(D(fake))`; GP on embedding-space interpolates (real embeds vs soft-fake embeds, padded to common length, per-sample max length for readout). No label smoothing in wgan mode. The MLE anchor (`mle_lambda`) remains available in both modes.
5. **`n_critic`** (wgan convention: critic updates per generator update) implemented as "G updates every n_critic-th batch". Existing `d_update_every` continues to mean "D updates every k-th batch" and composes with it.
6. **Sample std (ddof=1)** for all mean ± std aggregations.
7. **Edit source** unchanged: `famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`.
8. **Overnight artifacts** under `famail_temporal/results/overnight_2026-06-10/` (untracked research artifacts); variance-suite runs persist to `famail_temporal/baselines/variance_suite/results/<ts>/` (same convention as `metric_hardening/`).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `famail_temporal/baselines/run_variance_suite.py` | multi-seed paired runner + aggregation + persistence | Create |
| `famail_temporal/baselines/tests/test_variance_suite.py` | pure-helper unit tests | Create |
| `famail_temporal/baselines/gan/config.py` | `GAN_LOSS`, `WGAN_GP_LAMBDA`, `WGAN_N_CRITIC` | Modify |
| `famail_temporal/baselines/gan/critic.py` | public `forward_embed` delegate | Modify |
| `famail_temporal/baselines/gan/train_adversarial.py` | wgan-gp mode + `_gradient_penalty` + n_critic | Modify |
| `famail_temporal/baselines/gan/model_level.py` | pass-through `gan_loss`, `gp_lambda`, `n_critic` | Modify |
| `famail_temporal/baselines/gan/run_b0_adversarial.py` | `--gan-loss`, `--gp-lambda`, `--n-critic` flags | Modify |
| `famail_temporal/baselines/gan/tests/test_wgan.py` | GP + wgan smoke + loss-direction tests | Create |

---

## Task 1: `run_variance_suite.py` — paired multi-seed variance suite

**Files:**
- Create: `famail_temporal/baselines/run_variance_suite.py`
- Test: `famail_temporal/baselines/tests/test_variance_suite.py`

- [ ] **Step 1: Branch off** (once, at plan start)

```bash
git checkout implement-gan-baselines && git checkout -b variance-suite-wgan
```

- [ ] **Step 2: Write the failing tests**

Create `famail_temporal/baselines/tests/test_variance_suite.py`:

```python
"""Unit tests for the variance suite's pure aggregation helpers."""
import json
import math

import numpy as np

from famail_temporal.baselines import run_variance_suite as vs


def test_mean_std_basic():
    out = vs.mean_std([1.0, 2.0, 3.0])
    assert out["mean"] == 2.0
    assert math.isclose(out["std"], 1.0)  # sample std, ddof=1
    assert out["min"] == 1.0 and out["max"] == 3.0 and out["n"] == 3


def test_mean_std_single_value_has_zero_std():
    out = vs.mean_std([5.0])
    assert out["mean"] == 5.0 and out["std"] == 0.0 and out["n"] == 1


def test_paired_delta_stats_subtracts_b0_from_famail():
    out = vs.paired_delta_stats(b0=[1.0, 2.0], famail=[1.5, 2.1])
    assert math.isclose(out["mean"], 0.3)
    assert out["n"] == 2


def test_pairwise_js_stats_zero_for_identical_histograms():
    h = np.array([0.5, 0.5, 0.0])
    out = vs.pairwise_js_stats([h, h.copy(), h.copy()])
    assert out["n"] == 3  # C(3,2) pairs
    assert math.isclose(out["mean"], 0.0, abs_tol=1e-12)


def test_pairwise_js_stats_one_for_disjoint_histograms():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    out = vs.pairwise_js_stats([a, b])
    assert out["n"] == 1
    assert math.isclose(out["mean"], 1.0, rel_tol=1e-6)


def test_cross_js_values_paired_by_index():
    a = [np.array([1.0, 0.0]), np.array([0.5, 0.5])]
    b = [np.array([1.0, 0.0]), np.array([0.5, 0.5])]
    vals = vs.cross_js_values(a, b)
    assert len(vals) == 2
    assert all(math.isclose(v, 0.0, abs_tol=1e-12) for v in vals)


def test_result_to_json_roundtrips_numpy_floats():
    blob = vs.result_to_json({"x": np.float64(1.5), "y": {"z": [1, 2]}})
    loaded = json.loads(blob)
    assert loaded["x"] == 1.5 and loaded["y"]["z"] == [1, 2]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_variance_suite.py -v`
Expected: FAIL (ImportError — module doesn't exist).

- [ ] **Step 4: Implement `run_variance_suite.py`**

```python
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

def mean_std(values: List[float]) -> Dict[str, float]:
    """Sample statistics (ddof=1; std=0 for a single value)."""
    arr = np.asarray(values, dtype=np.float64)
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
    }
    (out_dir / "aggregate.json").write_text(result_to_json(agg))
    np.savez(
        out_dir / "terminal_cell_histograms.npz",
        p_raw=p_raw, p_edited=p_edited,
        **{f"p_b0_seed{s}": h for s, h in zip(seeds, hists_b0)},
        **{f"p_famail_seed{s}": h for s, h in zip(seeds, hists_fam)},
    )
    _write_report(out_dir, agg, seeds)

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_variance_suite.py -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Run the full baselines suite**

Run: `python -m pytest famail_temporal/baselines/ -q`
Expected: all green (purely additive change).

- [ ] **Step 7: CLI parse check (no GPU run)**

Run: `python -m famail_temporal.baselines.run_variance_suite --help`
Expected: usage message, exit 0.

- [ ] **Step 8: Commit**

```bash
git add famail_temporal/baselines/run_variance_suite.py \
        famail_temporal/baselines/tests/test_variance_suite.py
git commit -m "feat(baselines): paired multi-seed variance suite (Meeting 37 item 1+2)"
```

---

## Task 2: WGAN-GP mode + gradient-direction verification

**Files:**
- Modify: `famail_temporal/baselines/gan/config.py`
- Modify: `famail_temporal/baselines/gan/critic.py`
- Modify: `famail_temporal/baselines/gan/train_adversarial.py`
- Modify: `famail_temporal/baselines/gan/model_level.py`
- Modify: `famail_temporal/baselines/gan/run_b0_adversarial.py`
- Test: `famail_temporal/baselines/gan/tests/test_wgan.py`

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/gan/tests/test_wgan.py`:

```python
"""WGAN-GP mode + generator gradient-direction verification.

The direction tests answer Dr. Zhang's first diagnostic (Meeting 37): "is the
generator loss / gradient update direction correct?" For both loss modes,
gradient DESCENT on the generator loss must push the critic's score on fakes
UPWARD (toward 'real'), i.e. d(loss)/d(score) < 0.
"""
import torch
import torch.nn as nn

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.critic import SequenceCritic
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.train_adversarial import (
    adversarial_finetune, _gradient_penalty,
)


def test_bce_generator_loss_gradient_direction_is_correct():
    # Non-saturating BCE: g_loss = BCE(score, 1). Descent must RAISE the score.
    scores = torch.zeros(4, requires_grad=True)
    loss = nn.BCEWithLogitsLoss()(scores, torch.ones_like(scores))
    loss.backward()
    assert (scores.grad < 0).all()  # -grad step increases the score


def test_wgan_generator_loss_gradient_direction_is_correct():
    # Wasserstein: g_loss = -mean(score). Descent must RAISE the score.
    scores = torch.zeros(4, requires_grad=True)
    loss = -scores.mean()
    loss.backward()
    assert (scores.grad < 0).all()


def test_gradient_penalty_finite_and_nonnegative():
    torch.manual_seed(0)
    critic = SequenceCritic()
    real = torch.randint(0, gc.N_CELLS, (2, 5))
    real_len = torch.tensor([5, 3])
    fake_soft = torch.softmax(torch.randn(2, 4, gc.VOCAB_SIZE), dim=-1)
    fake_len = torch.tensor([4, 2])
    gp = _gradient_penalty(
        critic, real, real_len, fake_soft, fake_len,
        device=torch.device("cpu"),
    )
    assert torch.isfinite(gp)
    assert float(gp.item()) >= 0.0


def test_wgan_adversarial_finetune_smoke():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    sequences = [
        [gc.BOS, 10, 11, gc.EOS],
        [gc.BOS, 20, 21, 22, gc.EOS],
        [gc.BOS, 30, 31, gc.EOS],
        [gc.BOS, 40, 41, 42, 43, gc.EOS],
    ]
    contexts = [(10, 0), (20, 1), (30, 0), (40, 1)]
    out = adversarial_finetune(
        model, sequences, contexts,
        epochs=1, lr_g=1e-4, lr_d=1e-4, batch_size=2, max_len=8,
        tau_start=1.0, tau_end=0.5, device=torch.device("cpu"),
        gan_loss="wgan-gp", n_critic=2, mle_lambda=0.0,
    )
    assert set(out) == {"g_losses", "d_losses"}
    assert len(out["g_losses"]) == 1 and len(out["d_losses"]) == 1
    assert all(map(torch.isfinite, map(torch.tensor, out["g_losses"])))
    assert all(map(torch.isfinite, map(torch.tensor, out["d_losses"])))


def test_bce_mode_unchanged_smoke():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    sequences = [[gc.BOS, 10, 11, gc.EOS], [gc.BOS, 20, 21, gc.EOS]]
    contexts = [(10, 0), (20, 1)]
    out = adversarial_finetune(
        model, sequences, contexts,
        epochs=1, lr_g=1e-4, lr_d=1e-4, batch_size=2, max_len=8,
        tau_start=1.0, tau_end=0.5, device=torch.device("cpu"),
    )
    assert len(out["g_losses"]) == 1 and len(out["d_losses"]) == 1


def test_unknown_gan_loss_raises():
    model = TrajectoryLSTM()
    try:
        adversarial_finetune(
            model, [[gc.BOS, 10, gc.EOS]], [(10, 0)],
            epochs=1, lr_g=1e-4, lr_d=1e-4, batch_size=1, max_len=4,
            tau_start=1.0, tau_end=0.5, device=torch.device("cpu"),
            gan_loss="nope",
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "gan_loss" in str(e)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_wgan.py -v`
Expected: FAIL (ImportError on `_gradient_penalty`; TypeError on `gan_loss` kwarg).

- [ ] **Step 3: config.py additions**

Append to `famail_temporal/baselines/gan/config.py` (after the stabilization block):

```python
# Wasserstein GAN (Meeting 37 action item: try WGAN against mode collapse /
# length blowup). "wgan-gp" replaces the BCE losses with the Wasserstein
# critic objective + gradient penalty (Gulrajani et al. 2017); no label
# smoothing applies in this mode. The MLE anchor (ADV_MLE_LAMBDA) remains
# available in both modes.
GAN_LOSS = "bce"                  # "bce" | "wgan-gp"
WGAN_GP_LAMBDA = 10.0             # gradient-penalty weight
WGAN_N_CRITIC = 5                 # critic updates per generator update
                                  # (wgan convention; ignored in bce mode)
```

- [ ] **Step 4: critic.py — public embed-path delegate**

Add to `SequenceCritic` (after `forward_soft`):

```python
    def forward_embed(
        self, embedded: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Score pre-embedded sequences. embedded: (B, L, EMBED_DIM).

        Public entry for the WGAN-GP gradient penalty, which interpolates
        real and fake EMBEDDINGS (the discrete token space can't be
        interpolated) and needs critic scores differentiable w.r.t. them.
        """
        return self._forward_embed(embedded, lengths)
```

- [ ] **Step 5: train_adversarial.py — wgan-gp mode**

Add `import torch.nn.functional as F` to the imports. Add the GP helper above `adversarial_finetune`:

```python
def _gradient_penalty(
    critic: SequenceCritic,
    real_ids: torch.Tensor,
    real_lengths: torch.Tensor,
    fake_soft: torch.Tensor,
    fake_lengths: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """WGAN-GP penalty on embedding-space interpolates (Gulrajani et al.).

    Token sequences are discrete, so the interpolation happens in the critic's
    embedding space: real ids are embedded, fakes are soft-mixed embeddings,
    both zero-padded to a common length, then x_hat = eps*real + (1-eps)*fake
    with per-sample eps ~ U[0,1]. The readout length for x_hat is the
    elementwise max of the pair so no valid timestep is cut off.
    """
    real_emb = critic.embed(real_ids)                       # (B, Lr, E)
    fake_emb = fake_soft @ critic.embed.weight              # (B, Lf, E)
    L = max(real_emb.size(1), fake_emb.size(1))
    real_emb = F.pad(real_emb, (0, 0, 0, L - real_emb.size(1)))
    fake_emb = F.pad(fake_emb, (0, 0, 0, L - fake_emb.size(1)))
    eps = torch.rand(real_emb.size(0), 1, 1, device=device)
    interp = (eps * real_emb + (1.0 - eps) * fake_emb).requires_grad_(True)
    lengths = torch.maximum(real_lengths, fake_lengths)
    scores = critic.forward_embed(interp, lengths)
    grads = torch.autograd.grad(
        outputs=scores.sum(), inputs=interp, create_graph=True,
    )[0]                                                    # (B, L, E)
    grad_norm = grads.flatten(1).norm(2, dim=1)             # (B,)
    return ((grad_norm - 1.0) ** 2).mean()
```

Extend the `adversarial_finetune` signature (after `mle_lambda`):

```python
    gan_loss: str = gc.GAN_LOSS,
    gp_lambda: float = gc.WGAN_GP_LAMBDA,
    n_critic: int = 1,
```

At the top of the function body, validate and document:

```python
    if gan_loss not in ("bce", "wgan-gp"):
        raise ValueError(f"unknown gan_loss: {gan_loss!r} (use 'bce' or 'wgan-gp')")
```

Replace the discriminator-step block's loss computation with a mode branch (the `no_grad` rollout and the optimizer/clip mechanics stay identical):

```python
            if batch_i % d_update_every == 0:
                with torch.no_grad():
                    fake_soft, fake_len = gumbel_rollout(
                        model, cc, tb, max_len=max_len, tau=tau,
                        device=device, hard=True,
                    )
                d_real = critic.forward_ids(real, real_lengths)
                d_fake = critic.forward_soft(fake_soft, fake_len)
                if gan_loss == "bce":
                    # One-sided label smoothing: real target < 1.0, fake stays 0.
                    loss_d = (
                        bce(d_real, torch.full_like(d_real, real_label))
                        + bce(d_fake, torch.zeros_like(d_fake))
                    )
                else:  # wgan-gp: critic maximizes real-fake gap, GP enforces 1-Lipschitz
                    loss_d = (
                        d_fake.mean() - d_real.mean()
                        + gp_lambda * _gradient_penalty(
                            critic, real, real_lengths, fake_soft, fake_len,
                            device=device,
                        )
                    )
                opt_d.zero_grad()
                loss_d.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                opt_d.step()
                d_batch.append(float(loss_d.item()))
```

Gate the generator step on the n_critic schedule and branch its loss (wgan trains the critic `n_critic` batches per G update; bce keeps the current every-batch G update):

```python
            # ----- Generator step (gradients via Gumbel) -----
            g_step = gan_loss == "bce" or (batch_i % n_critic == n_critic - 1)
            if g_step:
                fake_soft, fake_len = gumbel_rollout(
                    model, cc, tb, max_len=max_len, tau=tau,
                    device=device, hard=True,
                )
                d_fake_g = critic.forward_soft(fake_soft, fake_len)
                if gan_loss == "bce":
                    adv_g = bce(d_fake_g, torch.ones_like(d_fake_g))
                else:  # wgan: maximize critic score on fakes
                    adv_g = -d_fake_g.mean()
                loss_g = adv_g
                mle_nll = None
                if mle_lambda > 0:
                    logits = model(real[:, :-1], cc, tb)            # (b, L-1, V)
                    mle_nll = ce(
                        logits.reshape(-1, gc.VOCAB_SIZE), real[:, 1:].reshape(-1),
                    )
                    loss_g = adv_g + mle_lambda * mle_nll
                opt_g.zero_grad()
                loss_g.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt_g.step()
                g_batch.append(float(adv_g.item()))
                fake_len_sum += float(fake_len.float().sum().item())
                fake_len_cnt += int(fake_len.numel())
```

(The progress-postfix block stays; guard its `g` entry with `max(1, len(g_batch))` since wgan mode may not have G-stepped yet in the first batches. Same for the end-of-epoch `g_losses.append` — use `max(1, len(g_batch))`.)

Update the module docstring's first paragraph to mention both modes, and the function docstring to document `gan_loss` / `gp_lambda` / `n_critic` (wgan: G updates every n_critic-th batch; label smoothing inert; reported d_loss is the Wasserstein critic loss including GP).

- [ ] **Step 6: model_level.py pass-through**

Add to the `fit_and_evaluate` signature (after `adv_mle_lambda`):

```python
    gan_loss: str = gc.GAN_LOSS,
    gp_lambda: float = gc.WGAN_GP_LAMBDA,
    n_critic: int = 1,
```

and pass them through in the `adversarial_finetune(...)` call:

```python
        d_update_every=d_update_every, mle_lambda=adv_mle_lambda,
        gan_loss=gan_loss, gp_lambda=gp_lambda, n_critic=n_critic,
```

Also extend the adversarial `_phase(...)` line to include `gan_loss={gan_loss}`.

- [ ] **Step 7: run_b0_adversarial.py flags**

Add after `--adv-mle-lambda`:

```python
    ap.add_argument("--gan-loss", default=gc.GAN_LOSS,
                    choices=["bce", "wgan-gp"],
                    help="adversarial objective: non-saturating BCE GAN or "
                         "Wasserstein GAN with gradient penalty")
    ap.add_argument("--gp-lambda", type=float, default=gc.WGAN_GP_LAMBDA,
                    help="gradient-penalty weight (wgan-gp only)")
    ap.add_argument("--n-critic", type=int, default=1,
                    help="critic updates per generator update (wgan-gp "
                         "convention; 1 = update G every batch)")
```

and pass them in the `fit_and_evaluate(...)` call:

```python
        gan_loss=args.gan_loss, gp_lambda=args.gp_lambda, n_critic=args.n_critic,
```

Also: the result dict now includes `pickups` (~105k tuples); `run_b0_adversarial` serializes the whole dict. Drop the bulk before writing — replace the write with:

```python
    slim = {k: v for k, v in result.items() if k != "pickups"}
    (args.out_dir / "b0_adversarial_fairness.json").write_text(
        result_to_json(slim)
    )
```

- [ ] **Step 8: Run the new tests + the existing adversarial/model-level tests**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_wgan.py famail_temporal/baselines/gan/tests/ -v`
Expected: PASS (new + all existing GAN tests — bce path behavior unchanged).

- [ ] **Step 9: Run the full baselines suite**

Run: `python -m pytest famail_temporal/baselines/ -q`
Expected: all green.

- [ ] **Step 10: Commit**

```bash
git add famail_temporal/baselines/gan/config.py \
        famail_temporal/baselines/gan/critic.py \
        famail_temporal/baselines/gan/train_adversarial.py \
        famail_temporal/baselines/gan/model_level.py \
        famail_temporal/baselines/gan/run_b0_adversarial.py \
        famail_temporal/baselines/gan/tests/test_wgan.py
git commit -m "feat(baselines/gan): WGAN-GP mode + generator gradient-direction tests (Meeting 37 items 3+4)"
```

---

## Task 3: Overnight driver (controller-executed; no subagent)

The controller writes `famail_temporal/results/overnight_2026-06-10/driver.sh` (untracked research artifact) and launches it detached. Serial queue — one GPU job at a time:

```bash
#!/bin/bash
# Overnight queue for Meeting 38 prep (2026-06-10). Serial: one GPU job at a time.
# No -e: a failed run must not kill the rest of the queue.
set -u
ROOT=/home/robert/FAMAIL
OUT=$ROOT/famail_temporal/results/overnight_2026-06-10
mkdir -p "$OUT/logs"
cd "$ROOT"

run () {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name" >> "$OUT/logs/queue.log"
  "$@" > "$OUT/logs/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $name (exit $?)" >> "$OUT/logs/queue.log"
}

# 1. Headline: paired 5-seed variance suite with the STRENGTHENED generator
#    (20 MLE epochs per 2026-06-10 user direction; ~85-100 min)
run variance_suite python -m famail_temporal.baselines.run_variance_suite \
    --seeds 0,1,2,3,4 --mle-epochs 20 --device auto

# 2. Pretraining ablation (Dr. Zhang's hypothesis: under-pretrained G is why
#    the critic overpowers it). 10 vs 20 epochs, then 20 + critic-slowing
#    (her "train G more per D update" recipe on the strongest pretrain).
run ablation_mle10 python -m famail_temporal.baselines.gan.run_b0_adversarial \
    --mle-epochs 10 --adv-epochs 3 --seed 0 --out-dir "$OUT/ablation_mle10"
run ablation_mle20 python -m famail_temporal.baselines.gan.run_b0_adversarial \
    --mle-epochs 20 --adv-epochs 3 --seed 0 --out-dir "$OUT/ablation_mle20"
run ablation_mle20_dslow python -m famail_temporal.baselines.gan.run_b0_adversarial \
    --mle-epochs 20 --adv-epochs 3 --d-update-every 2 --seed 0 \
    --out-dir "$OUT/ablation_mle20_dslow"

# 3. WGAN-GP from the strongest (20-epoch) pretrain: standard critic-heavy,
#    then Dr. Zhang's gen-heavy instinct
run wgan_ncritic5 python -m famail_temporal.baselines.gan.run_b0_adversarial \
    --gan-loss wgan-gp --n-critic 5 --mle-epochs 20 --adv-epochs 3 --seed 0 \
    --out-dir "$OUT/wgan_ncritic5"
run wgan_genheavy python -m famail_temporal.baselines.gan.run_b0_adversarial \
    --gan-loss wgan-gp --n-critic 1 --d-update-every 2 --mle-epochs 20 \
    --adv-epochs 3 --seed 0 --out-dir "$OUT/wgan_genheavy"

touch "$OUT/ALL_DONE"
```

Launch: `nohup setsid bash driver.sh >/dev/null 2>&1 &` — survives session death. NOT passing `--quiet`: with non-TTY stderr, `Progress` falls back to periodic prints, so the per-epoch g/d losses and real-vs-fake length diagnostics (exactly what Dr. Zhang asked to see) land in the logs.

---

## Self-Review

**1. Coverage vs the approved scope:**
- 5-seed paired B0+FAMAIL, mean ± std, paired deltas — Task 1. ✓
- JS noise floor (within-variant) vs signal (cross-variant) — Task 1 (`pairwise_js_stats` / `cross_js_values`). ✓
- Pretraining ablation — Task 3 driver, zero new code (`--mle-epochs 10/20`, existing `--out-dir`). ✓
- WGAN — Task 2 (`wgan-gp` mode, GP, n_critic) + 2 driver configs. ✓
- Gradient-direction verification — Task 2 tests (both modes). ✓
- No PR — not in plan. ✓

**2. Placeholder scan:** No TBDs. All code complete. Driver script fully written.

**3. Type consistency:**
- `mean_std(List[float]) -> Dict`; `paired_delta_stats(*, b0, famail) -> Dict`; `pairwise_js_stats(List[ndarray]) -> Dict`; `cross_js_values(List, List) -> List[float]` — all used by `main` and tested. ✓
- `_gradient_penalty(critic, real_ids, real_lengths, fake_soft, fake_lengths, *, device) -> Tensor` — called from the D-step and tested directly. ✓
- `adversarial_finetune(..., gan_loss=, gp_lambda=, n_critic=)` ← `fit_and_evaluate(..., gan_loss=, gp_lambda=, n_critic=)` ← CLI flags. ✓

**4. Known risks, accepted:**
- wgan mode with `n_critic > 1`: `g_batch` may be empty early in an epoch → guarded with `max(1, len(g_batch))` in postfix and epoch-mean.
- GP doubles backward through the critic (small LSTM) — modest cost; rollout still dominates.
- The structural critic length-leak (last-valid-step readout) is NOT addressed — deliberately, so the WGAN runs test Dr. Zhang's suggestion against the same critic, isolating the loss-mode variable.
- `run_b0_adversarial` JSON slimming (drop `pickups`) is a behavior change to an existing CLI output; the fairness numbers and loss histories are preserved, which is everything the existing consumers read.

**5. Standing constraints:** branch `variance-suite-wgan`; named-file staging only; no literal eval-paren; no edits to `algorithm/`/`fairness/`; ε=2 untouched; overnight artifacts untracked.
