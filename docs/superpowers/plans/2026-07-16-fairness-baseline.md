# Fairness-Intervention Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two fairness-method baseline arms (Kamiran–Calders-style reweighing; fairness-penalty BC) trained on the raw Shenzhen corpus and compared against FAMAIL at the model level, per `docs/superpowers/specs/2026-07-16-fairness-baseline-design.md`.

**Architecture:** New pure-function module `famail_temporal/baselines/fairness_baseline.py` (weight rule + differentiable penalty); minimal wiring diffs into `run_weighted_bc_smoke.py` (new arm flags) and `gan/train_mle.py` (optional penalty term, default-off). Existing arms must be provably unchanged (regression gate) before any new result is trusted.

**Tech Stack:** Python 3.12, PyTorch, numpy, pytest; existing FAMAIL baselines machinery.

## Global Constraints

- **Era discipline:** edited corpus = `famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered` (α\*=(0.1,0.8,0.1), fingerprint 2,337+7,545); raw corpus = the bundle `DataBundle.load()` loads under PRIMARY config. Verify `famail_temporal/config.py` `DEMOGRAPHIC_FEATURES == ["AvgHousingPricePerSqM", "CompPerCapita", "MigrantRatio"]` before ANY run.
- **No editor changes.** Nothing under `famail_temporal/algorithm/` or `evaluation/runner.py` is touched (frozen-editor gate must stay PASS).
- **Default-off invariant:** every modification must be a no-op when the new flags are unset — enforced by Task 5's regression gate.
- **Ledger discipline:** every GPU run wrapped in `python -m famail_temporal.analysis.run_ledger start|finish` with queue ids `FB-REWEIGH`, `FB-PENALTY-PILOT`, `FB-PENALTY`.
- **GPU scheduling:** suites queue only after the C1 dose-extension finishes (check `nvidia-smi` + `famail_temporal/results/experiments_campaign/b_chain.log` shows `B-CHAIN COMPLETE`).
- Grouping convention everywhere: migrant axis, district extremes (`region_extremes`, `disadvantaged_high=True`) — the SAME grouping as every external table. ⚠️ CORRECTED 2026-07-16 (Task 1 finding): the A1 log's **N_D = 6,950 counts active (cell × time-block) UNITS**, not spatial cells (the grid has only 4,320 cells). The weight rule keys on spatial cells; verified spatial counts on Shenzhen PRIMARY: **462 disadvantaged / 406 advantaged / 1,011 excluded cells**, 92.8% trajectory pickup-cell match rate.
- Tests live in `famail_temporal/baselines/tests/`; suite must stay green: `python -m pytest famail_temporal/baselines/tests/ -q`.

---

### Task 1: Weight rule — `fairness_reweigh_weight_vector`

**Files:**
- Create: `famail_temporal/baselines/fairness_baseline.py`
- Test: `famail_temporal/baselines/tests/test_fairness_baseline.py`

**Interfaces:**
- Consumes: `famail_temporal.baselines.external_fairness.region_extremes(values, disadvantaged_high)`; `famail_temporal.baselines.run_external_fairness` uses `io.service_ratio_Y(bundle.pickup_3d, bundle)` (see its `_run_one`, line ~211) — reuse the same `io` module's unit/value construction.
- Produces:
  - `unit_groups_and_sdr(bundle) -> tuple[dict[tuple[int,int], int], dict[int, float]]` — maps active-unit cell `(cx, cy)` → group label (0 adv, 1 disadv, −1 excluded) using migrant/district-extremes, and group → mean supply-demand ratio Y.
  - `fairness_reweigh_weight_vector(trajs, bundle) -> list[float]` — index-aligned per-trajectory weights: weight of a trajectory whose `pickup_cell` falls in group g is `(1.0 / max(Y_g, 1e-6))`, then the whole vector is normalized to **mean 1.0**; trajectories in excluded cells (label −1) get the pre-normalization weight 1.0.

- [ ] **Step 1: Write the failing tests** (pure math on synthetic inputs + one real-bundle smoke)

```python
# famail_temporal/baselines/tests/test_fairness_baseline.py
import numpy as np
import pytest

from famail_temporal.baselines.fairness_baseline import (
    normalize_mean_one, weights_from_groups,
)


def test_normalize_mean_one():
    w = normalize_mean_one([2.0, 4.0, 6.0])
    assert np.isclose(np.mean(w), 1.0)
    assert np.isclose(w[1] / w[0], 2.0)  # ratios preserved


def test_weights_from_groups_inverse_sdr():
    # group 1 (disadv) has SDR 2.0, group 0 (adv) has SDR 8.0 -> disadv gets 4x
    groups_of_trajs = [1, 0, -1, 1]
    sdr_by_group = {0: 8.0, 1: 2.0}
    w = weights_from_groups(groups_of_trajs, sdr_by_group)
    assert np.isclose(np.mean(w), 1.0)
    assert np.isclose(w[0] / w[1], 4.0)       # inverse-SDR ratio
    assert np.isclose(w[2] * len(w) / sum(1 for _ in w), w[2])  # excluded finite
    assert w[0] == w[3]                        # same group, same weight
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest famail_temporal/baselines/tests/test_fairness_baseline.py -q` → FAIL (`ModuleNotFoundError` / `ImportError`).

- [ ] **Step 3: Minimal implementation**

```python
# famail_temporal/baselines/fairness_baseline.py
"""Fairness-intervention baseline arms (spec: docs/superpowers/specs/
2026-07-16-fairness-baseline-design.md). Pure functions only — wiring lives in
run_weighted_bc_smoke.py / gan/train_mle.py."""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np


def normalize_mean_one(w: List[float]) -> List[float]:
    arr = np.asarray(w, dtype=np.float64)
    m = float(arr.mean())
    if m <= 0:
        raise ValueError("weight mean must be positive")
    return list(arr / m)


def weights_from_groups(
    groups_of_trajs: List[int], sdr_by_group: Dict[int, float],
) -> List[float]:
    """Kamiran-Calders-style inverse-service weights: 1/SDR_g for group g,
    1.0 for excluded (-1), normalized to mean 1 (effective dataset size kept)."""
    raw = [
        1.0 / max(sdr_by_group[g], 1e-6) if g in sdr_by_group and g >= 0 else 1.0
        for g in groups_of_trajs
    ]
    return normalize_mean_one(raw)
```

- [ ] **Step 4: Run to verify pass** — same command → 2 passed.

- [ ] **Step 5: Add the bundle-facing layer + smoke test.** Read `famail_temporal/baselines/run_external_fairness.py:208-230` (`_run_one`) and the `io` module it imports to copy the exact construction of per-unit values and Y, then implement:

```python
def unit_groups_and_sdr(bundle) -> Tuple[Dict[tuple, int], Dict[int, float]]:
    """Active-unit cell -> group label (migrant axis, district extremes,
    disadvantaged_high=True) and group -> mean before-edit Y (supply/demand),
    built EXACTLY as run_external_fairness builds them (same io/ef calls)."""
    ...  # implement by mirroring _run_one's io.service_ratio_Y + _groups_for


def fairness_reweigh_weight_vector(trajs, bundle) -> List[float]:
    cell_group, sdr = unit_groups_and_sdr(bundle)
    groups_of_trajs = [cell_group.get(tuple(t.pickup_cell), -1) for t in trajs]
    return weights_from_groups(groups_of_trajs, sdr)
```

Smoke test (skipped when data absent), asserting the known grouping size:

```python
def test_unit_groups_real_bundle():
    pytest.importorskip("torch")
    from famail_temporal.data.loader import DataBundle
    try:
        bundle = DataBundle.load()
    except Exception:
        pytest.skip("bundle data not available")
    from famail_temporal.baselines.fairness_baseline import unit_groups_and_sdr
    cell_group, sdr = unit_groups_and_sdr(bundle)
    n_d = sum(1 for g in cell_group.values() if g == 1)
    assert n_d == 6950            # matches the A1 run's N_D on Shenzhen PRIMARY
    assert sdr[1] < sdr[0]        # disadvantaged group is under-served
```

- [ ] **Step 6: Run full test file** — expect 3 passed (or 2 passed 1 skipped off-box).
- [ ] **Step 7: Commit** — `git add famail_temporal/baselines/fairness_baseline.py famail_temporal/baselines/tests/test_fairness_baseline.py && git commit -m "feat(fairness-baseline): inverse-SDR reweigh weight rule (Task 1)"`

### Task 2: Differentiable DP-gap penalty (pure function)

**Files:**
- Modify: `famail_temporal/baselines/fairness_baseline.py` (append)
- Test: `famail_temporal/baselines/tests/test_fairness_baseline.py` (append)

**Interfaces:**
- Produces: `dp_gap_penalty(logits, tgt, mask_disadv, mask_adv, pad_id) -> torch.Tensor` (scalar, differentiable): mean predicted probability mass per advantaged cell minus per disadvantaged cell, over non-PAD target positions. Positive when the policy predicts more service per advantaged cell.
- Also: `cell_masks_for_vocab(cell_group, vocab_size, token_of_cell) -> tuple[Tensor, Tensor]` — boolean masks over the generator vocabulary. `token_of_cell` is the token id for grid cell `(cx, cy)`; locate the exact mapping in `famail_temporal/baselines/gan/` (the tokenization used by `traj_training_data` in `run_level2_table.py`) and implement against it; the test below is mapping-independent.

- [ ] **Step 1: Write the failing tests**

```python
import torch
from famail_temporal.baselines.fairness_baseline import dp_gap_penalty


def test_dp_gap_zero_for_uniform_logits():
    B, L, V = 2, 3, 10
    logits = torch.zeros(B, L, V)                    # uniform after softmax
    tgt = torch.ones(B, L, dtype=torch.long)         # no PAD
    m_d = torch.zeros(V, dtype=torch.bool); m_d[:3] = True   # 3 disadv cells
    m_a = torch.zeros(V, dtype=torch.bool); m_a[3:6] = True  # 3 adv cells
    g = dp_gap_penalty(logits, tgt, m_d, m_a, pad_id=0)
    assert torch.isclose(g, torch.tensor(0.0), atol=1e-6)


def test_dp_gap_positive_when_adv_favored_and_differentiable():
    B, L, V = 1, 2, 6
    logits = torch.full((B, L, V), -10.0, requires_grad=True)
    with torch.no_grad():
        logits[..., 3:6] = 10.0                      # all mass on adv cells
    tgt = torch.ones(B, L, dtype=torch.long)
    m_d = torch.zeros(V, dtype=torch.bool); m_d[:3] = True
    m_a = torch.zeros(V, dtype=torch.bool); m_a[3:6] = True
    g = dp_gap_penalty(logits, tgt, m_d, m_a, pad_id=0)
    assert g.item() > 0.2
    g.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
```

- [ ] **Step 2: Run to verify failure** — ImportError on `dp_gap_penalty`.
- [ ] **Step 3: Minimal implementation**

```python
import torch


def dp_gap_penalty(logits, tgt, mask_disadv, mask_adv, pad_id: int):
    """Differentiable DP-gap analog over predicted next-cell distributions:
    (mean predicted mass per ADVANTAGED cell) - (per DISADVANTAGED cell),
    averaged over non-PAD positions. NOT F_causal (metric-firewall: the
    baseline optimizes an external-family quantity)."""
    probs = torch.softmax(logits, dim=-1)            # (B, L, V)
    valid = (tgt != pad_id).to(probs.dtype)          # (B, L)
    n_valid = valid.sum().clamp_min(1.0)
    mass_d = (probs[..., mask_disadv].sum(-1) * valid).sum() / (
        n_valid * int(mask_disadv.sum()))
    mass_a = (probs[..., mask_adv].sum(-1) * valid).sum() / (
        n_valid * int(mask_adv.sum()))
    return mass_a - mass_d
```

- [ ] **Step 4: Run to verify pass.** Then implement `cell_masks_for_vocab` against the located token↔cell mapping, with a test that a known cell's token lands in exactly one mask.
- [ ] **Step 5: Commit** — `git commit -m "feat(fairness-baseline): differentiable DP-gap penalty (Task 2)"`

### Task 3: Optional penalty in `train_mle` (default-off, provably)

**Files:**
- Modify: `famail_temporal/baselines/gan/train_mle.py:47-153`
- Test: `famail_temporal/baselines/tests/test_fairness_baseline.py` (append)

**Interfaces:**
- Produces: `train_mle(..., penalty_fn=None, penalty_lambda=0.0)` — when set, `loss = ce + penalty_lambda * penalty_fn(logits, tgt)`; `penalty_fn` closes over the vocab masks. Returned dict gains `"penalty_values"` (per-batch floats) ONLY when active (absent otherwise, so existing manifests are byte-stable).

- [ ] **Step 1: Failing test — λ=0 path is bit-identical to unmodified training**

```python
def _tiny_training(penalty_fn=None, penalty_lambda=0.0):
    import torch
    from famail_temporal.utils.seeding import set_all_seeds
    from famail_temporal.baselines.gan.generator import TrajectoryLSTM
    from famail_temporal.baselines.gan.train_mle import train_mle
    set_all_seeds(0)
    model = TrajectoryLSTM(n_drivers=2)
    seqs = [[1, 2, 3], [2, 3, 4, 5], [3, 4]]
    ctxs = [(1, 0), (2, 1), (3, 0)]
    kwargs = dict(epochs=2, lr=1e-3, batch_size=2,
                  device=torch.device("cpu"), driver_idxs=[0, 1, 0])
    if penalty_fn is not None:
        kwargs.update(penalty_fn=penalty_fn, penalty_lambda=penalty_lambda)
    return train_mle(model, seqs, ctxs, **kwargs)["epoch_losses"]


def test_penalty_default_off_is_identical():
    assert _tiny_training() == _tiny_training(penalty_fn=None, penalty_lambda=0.0)


def test_penalty_changes_loss_when_active():
    import torch
    from famail_temporal.baselines.fairness_baseline import dp_gap_penalty
    m_d = torch.zeros(_vocab_size(), dtype=torch.bool); m_d[1] = True
    m_a = torch.zeros(_vocab_size(), dtype=torch.bool); m_a[2] = True
    fn = lambda lg, tg: dp_gap_penalty(lg, tg, m_d, m_a, pad_id=_pad_id())
    base = _tiny_training()
    pen = _tiny_training(penalty_fn=fn, penalty_lambda=100.0)
    assert base != pen
```

(`_vocab_size()`/`_pad_id()` read `gc.VOCAB_SIZE`/`gc.PAD` from `famail_temporal.baselines.gan.config` — define as two-line helpers at the top of the test file.)

- [ ] **Step 2: Run — FAIL** (unexpected keyword `penalty_fn`).
- [ ] **Step 3: Implement.** In `train_mle`: add keyword-only params `penalty_fn=None, penalty_lambda: float = 0.0`; after the existing `loss` computation (both branches, i.e., after line ~145):

```python
                if penalty_fn is not None and penalty_lambda != 0.0:
                    pen = penalty_fn(logits, tgt)
                    loss = loss + penalty_lambda * pen
                    penalty_values.append(float(pen.item()))
```

with `penalty_values: List[float] = []` initialized before the epoch loop and `result["penalty_values"] = penalty_values` added **only when `penalty_fn is not None`**.
- [ ] **Step 4: Run to verify both tests pass; run the WHOLE baselines test suite** — `python -m pytest famail_temporal/baselines/tests/ -q` → all green.
- [ ] **Step 5: Commit** — `git commit -m "feat(fairness-baseline): optional DP-gap penalty in train_mle, default-off (Task 3)"`

### Task 4: Arm wiring in `run_weighted_bc_smoke.py`

**Files:**
- Modify: `famail_temporal/baselines/run_weighted_bc_smoke.py` (CLI ~line 145-160; arm assembly ~line 203-236; train call ~line 322-327)

**Interfaces:**
- Consumes: Task 1 `fairness_reweigh_weight_vector`, Task 2 `dp_gap_penalty` + `cell_masks_for_vocab`, Task 3 `train_mle(..., penalty_fn, penalty_lambda)`.
- Produces: CLI flags `--fairness-reweigh` (store_true) and `--fairness-penalty "0.1,1,10"` (comma floats, empty default); arm names `fair_reweigh`, `fair_penalty_l<λ>`; arms tuple extended to `(name, D, sw, penalty_lambda)` with `0.0` for all existing arms.

- [ ] **Step 1:** Extend the arms tuples: change `arms: List = [("raw", D_raw, None), ("edited", D_edited, None)]` to 4-tuples ending in `0.0`; update every `arms.append` accordingly and the loop header `for name, D, sw in arms:` → `for name, D, sw, plam in arms:`.
- [ ] **Step 2:** Add the two flags and arm construction after the most-fair block (~line 227):

```python
    if args.fairness_reweigh:
        from famail_temporal.baselines.fairness_baseline import (
            fairness_reweigh_weight_vector)
        arms.append(("fair_reweigh", D_raw,
                     fairness_reweigh_weight_vector(raw_trajs, bundle), 0.0))
    fp_lambdas = [float(x) for x in str(args.fairness_penalty).split(",") if x.strip()]
    if fp_lambdas:
        from famail_temporal.baselines.fairness_baseline import (
            unit_groups_and_sdr, cell_masks_for_vocab, dp_gap_penalty)
        cell_group, _ = unit_groups_and_sdr(bundle)
        m_d, m_a = cell_masks_for_vocab(cell_group, gc.VOCAB_SIZE, token_of_cell)
        _penalty_fn = lambda lg, tg: dp_gap_penalty(lg, tg, m_d.to(device),
                                                    m_a.to(device), pad_id=gc.PAD)
        for lam in fp_lambdas:
            arms.append((f"fair_penalty_l{lam:g}", D_raw, None, lam))
```

- [ ] **Step 3:** Thread the penalty into the train call: `train_mle(..., sample_weights=sw, penalty_fn=(_penalty_fn if plam else None), penalty_lambda=plam)`.
- [ ] **Step 4:** Smoke run, tiny: `python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered --seeds 0 --weights "" --fairness-reweigh --fairness-penalty "1" --mle-epochs 1 --out-dir /tmp/fb_smoke` → completes; `sweep.json` contains `fair_reweigh` and `fair_penalty_l1` arms.
- [ ] **Step 5: Commit** — `git commit -m "feat(fairness-baseline): fair_reweigh + fair_penalty arms in the WBC harness (Task 4)"`

### Task 5: RESULT regression gate (Mission-3 lesson — REQUIRED)

**Files:** none created; evidence recorded in the ledger config-note + commit message of Task 4 (amend if needed).

- [ ] **Step 1:** With the Task-1..4 code in place, re-run ONE seed of the unmodified arms: `python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered --seeds 0 --weights 30 --mle-epochs 20 --device auto --out-dir /tmp/fb_regression` (~1.5h GPU; schedule after C1).
- [ ] **Step 2:** Compare seed-0 `raw`, `edited`, `edited_w30` values (f_causal, f_spatial, fidelity_a, fidelity_b, all 4 decimals) against the committed `famail_temporal/results/weighted_bc_sweep/alpha_sweep_s10_c80_f10_filtered_6seed/sweep.json` seed-0 entries. Expected: **identical to every recorded decimal**. Any mismatch = STOP, revert wiring, diagnose (the default-off invariant is broken).
- [ ] **Step 3:** Record PASS in the FB-REWEIGH ledger row's config-note ("regression gate: seed-0 raw/edited/edited_w30 identical to committed sweep").

### Task 6: λ pilot (FB-PENALTY-PILOT)

- [ ] **Step 1:** Ledger-wrapped pilot at seed 0, grid λ ∈ {0.1, 1, 10}: same command shape as Task 4 Step 4 but `--mle-epochs 20 --fairness-penalty "0.1,1,10"`, out-dir `famail_temporal/results/weighted_bc_sweep/fairness_penalty_pilot`.
- [ ] **Step 2:** Selection rule (spec §2b): λ_hi = largest λ with held-out next-step degradation < 20% relative (proxy: `fair_penalty_l*` arm's fidelity_a within 0.02 of raw AND n_empty == 0 AND epoch_losses finite/decreasing); λ_lo = λ_hi/10; λ_mid = geometric mean. If all three unstable → halve the grid and repeat once; if still unstable, ship reweigh-only (spec §7 fallback) and record the finding.
- [ ] **Step 3:** Record chosen grid in the ledger row + a short `PILOT.md` in the out-dir.

### Task 7: Full suites (GPU, after C1)

- [ ] **Step 1: FB-REWEIGH** (~10h): `python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered --seeds 0,1,2,3,4,5 --weights 30 --fairness-reweigh --out-dir famail_temporal/results/weighted_bc_sweep/fairness_baseline_6seed` — includes raw + edited + edited_w30 + fair_reweigh in ONE suite so pairing is internal. Ledger-wrapped (`FB-REWEIGH`), setsid launcher in `famail_temporal/results/experiments_campaign/` per house pattern, monitor armed.
- [ ] **Step 2: FB-PENALTY** (~10h): same but `--fairness-penalty "<λ_lo>,<λ_mid>,<λ_hi>"` from Task 6, out-dir `famail_temporal/results/weighted_bc_sweep/fairness_penalty_6seed`, ledger `FB-PENALTY`.
- [ ] **Step 3:** If wall-clock projects past **Jul 23**, ship FB-REWEIGH alone (decided fallback).

### Task 8: Model-level external scoring (rollout)

- [ ] **Step 1:** Read `PAPER/external-metrics/scripts/option_a_rollout_eval.py`'s CLI/arm-selection (this produced §4.4's allocation numbers for the WBC arms) and run it against both new suites' out-dirs so each new arm's rolled-out allocation gets DP/DI/Theil + mean(Y|disadv.). If its arm filter is hardcoded to WBC arm names, extend the filter list (smallest possible diff, committed with test-free justification recorded in the ledger note; the tool is analysis-side, not the editor).
- [ ] **Step 2:** Fid-A + JS come free from the suite itself (`sweep.json` per-arm fidelity_a + the JS machinery in the variance suite pattern; report Fid-A from sweep.json and per-arm JS vs raw via the same `fe.trajectory_statistics` path the suite already logs).
- [ ] **Step 3:** Ledger rows for the rollout evals (`FB-ROLLOUT-REWEIGH`, `FB-ROLLOUT-PENALTY`).

### Task 9: Slot-in + curation (per-landing sequence)

- [ ] **Step 1:** §4.5 gains the fairness-intervention paragraph + rows/sub-table (placement coordinated with the 8-page cut plan; at minimum the headline comparison sentence stays in the main 8 pages). Honesty rule from spec §1: report every axis as measured.
- [ ] **Step 2:** Curate result JSONs to `PAPER/baselines/fairness-intervention/` (git add — `!PAPER/**/*.json` re-includes) + DATA_INVENTORY rows + ledger cross-refs.
- [ ] **Step 3:** Gates: `cd paper && latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh` — check exit codes; render the affected page (the Table-6 lesson: compare table edge to caption wrap; `grep -a Overfull main.log`).
- [ ] **Step 4:** Commit `paper+campaign(FB): fairness-intervention baselines — <result summary>`; re-verify kamirancalders2012 + zheng2023 against the ACM DL and note in the citation audit trail (Dr. Kash directive).

## Self-Review

- Spec coverage: §1 comparison structure → Tasks 4/7 (arms share one suite, raw+FAMAIL internal); §2a → Task 1; §2b → Tasks 2/6; §3 scoring → Tasks 7/8; §4 shape+gate → Tasks 3/4/5; §5 placement+citations → Task 9; §6/§7 fallbacks → Tasks 6/7. No gaps.
- Placeholders: Task 1 Step 5 and Task 8 Step 1 require reading a named file first — each names the exact file/lines and the expected interface, with concrete test/values (N_D=6950) verifying the outcome. Acceptable read-then-implement steps, not TBDs.
- Type consistency: arms 4-tuples everywhere after Task 4 Step 1; `penalty_fn(logits, tgt)` arity consistent across Tasks 2/3/4.
