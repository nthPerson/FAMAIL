# FAMAIL Baselines — Metric Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the model-level metrics needed to defensibly evaluate whether the data-level editing signal (ΔF_causal = +0.0128, confirmed intrinsic at the +0.0128 ceiling per methodology §8.7–§8.8) actually survives the LSTM → multinomial → terminal-cell-only generation pipeline — plus paper-ready persistence so the results drop directly into the paper.

**Architecture:** Three new metric modules + one orchestrator CLI, built on top of the smallest possible Phase-4 prerequisite (a `variants.py` to load the edited corpus and a `train_trajectories` keyword on `fit_and_evaluate` so we can train a FAMAIL generator on the edited data). Each run produces a per-run dir (metrics.json + npz arrays + report.md) and updates a single canonical `RESULTS.md` that the paper can quote directly.

**Tech Stack:** Python 3.12, PyTorch (existing MLE-only generator path), NumPy, pandas (for the hukou CSV), pytest. Reuses Phase 3's `fit_and_evaluate`, Phase 1's `data_level_fairness` / `build_fairness_grid`, the existing editing pipeline's `histories.pkl` as the FAMAIL edit source, and `bundle.g0_func` for the F_causal residual.

---

## Scope: what this plan delivers (and what it doesn't)

### Delivers

1. **Phase-4 prerequisites (Tasks 1 + 2):** the smallest pieces of the committed Phase-4 plan so we can train an MLE-only FAMAIL generator on the edited corpus. Tasks 3-4 of that plan (full suite orchestrator and CLI) remain deferred.
2. **JS terminal-cell transmission check (Task 3):** does the ~1% data-level edit signal survive the LSTM → multinomial sampling → terminal-cell-only pipeline?
3. **DI ratio (Task 4):** disparate-impact at the district level, computed under BOTH `Y = active_taxis / pickup_mass` (primary, aligned with F_causal) AND `Y = pickup_mass / active_taxis` (supplementary robustness lens).
4. **Localized F_causal (Task 5):** F_causal restricted to the active units the edit actually touches — concentrates rather than dilutes the signal.
5. **Orchestrator + persistence (Task 6):** one CLI, per-run artifact dir, and a single paper-ready summary writeup.
6. **Documentation (Task 7):** a `MODEL_LEVEL_METRICS.md` methods doc, `RESULTS.md` paper-ready summary, and STATUS.md update.

### Does NOT deliver

- The full Phase-4 `model_suite.py` + `run_model_suite.py` (deferred — these add B2 sweep orchestration which we don't need to answer the transmission question).
- B1 (deferred to Phase 5).
- Multi-seed / paired statistics. Single seed for now; multi-seed is the right follow-up *if* the transmission check shows the model-level headline is defensible.
- Any change to F_causal's `Y` (explicitly out of scope; `Y = supply/demand` stays).

### Branch

```bash
git checkout implement-gan-baselines
git checkout -b metric-hardening
```

Do this once at the start. Subsequent commits land on `metric-hardening`.

### Design decisions (locked, no veto needed)

1. **DI's `Y` is aligned with F_causal's `Y` = `active_taxis / pickup_mass`.** A supplementary DI under the flipped `pickup_mass / active_taxis` is also reported as a robustness lens. F_causal itself is NOT changed.
2. **Two-level averaging for DI** (mean of district-means → group-mean ratio): normalizes for both within-district size and between-district population differences, and treats each district as one observation.
3. **Restrict to active units** (`mask_3d=True`) throughout — same convention as F_causal.
4. **Clamp the demand denominator with `config.DEMAND_FLOOR`** in the primary DI (matches the existing F_causal clamp); clamp the supply denominator with a comparable floor in the supplementary DI.
5. **FAMAIL edit source = `famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`** (the shipped no-dedup k=10000 edit, +0.0128 / +0.0003, per methodology §8.7).
6. **Persistence pattern**: every run writes a timestamped per-run dir under `famail_temporal/baselines/metric_hardening/results/`, AND appends/updates the single `RESULTS.md` summary.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `famail_temporal/baselines/gan/variants.py` | `apply_edits`, `load_edited_trajectories`, `filtered_trajectories` | Create |
| `famail_temporal/baselines/gan/model_level.py` | + `train_trajectories` kwarg; return `pickups` in result dict | Modify |
| `famail_temporal/baselines/transmission.py` | terminal-cell histograms + JS divergence + transmission metrics | Create |
| `famail_temporal/baselines/district_metrics.py` | DI ratio (both `Y` conventions) | Create |
| `famail_temporal/baselines/localized_metrics.py` | F_causal restricted to edited active units | Create |
| `famail_temporal/baselines/run_metric_hardening.py` | CLI orchestrator + persistence | Create |
| `famail_temporal/baselines/gan/tests/test_variants.py` | edit-swap + filter mechanics | Create |
| `famail_temporal/baselines/gan/tests/test_model_level_variants.py` | `train_trajectories` behavior | Create |
| `famail_temporal/baselines/tests/test_transmission.py` | histogram + JS + transmission_metrics | Create |
| `famail_temporal/baselines/tests/test_district_metrics.py` | DI under both `Y` conventions on synthetic | Create |
| `famail_temporal/baselines/tests/test_localized_metrics.py` | localized F_causal on synthetic | Create |
| `famail_temporal/baselines/tests/test_run_metric_hardening.py` | result serialization | Create |
| `famail_temporal/baselines/metric_hardening/RESULTS.md` | paper-ready summary (skeleton + filled by Task 7) | Create |
| `famail_temporal/docs/MODEL_LEVEL_METRICS.md` | methodology doc for the three new metrics | Create |
| `famail_temporal/baselines/STATUS.md` | new "Phase 4 — model-level metrics" section | Modify |

---

## Task 1: `variants.py` — edit-swap + filter builders

**Files:**
- Create: `famail_temporal/baselines/gan/variants.py`
- Test: `famail_temporal/baselines/gan/tests/test_variants.py`

This is Task 1 from the committed Phase-4 plan (`docs/superpowers/plans/2026-05-28-famail-gan-baselines-phase4-famail-b2-model-level.md`), reproduced here so this plan is self-contained.

- [ ] **Step 1: Branch off**

```bash
git checkout implement-gan-baselines
git checkout -b metric-hardening
```

- [ ] **Step 2: Write the failing tests**

Create `famail_temporal/baselines/gan/tests/test_variants.py`:

```python
"""Unit tests for gan.variants."""
import pickle
from types import SimpleNamespace

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import active_units, make_traj_at
from famail_temporal.baselines.datasets import rank_unfair_trajectory_indices
from famail_temporal.baselines.gan import variants


def test_apply_edits_swaps_by_trajectory_id_preserving_order():
    raw = [make_traj_at(1, 1, 0, traj_id=10),
           make_traj_at(2, 2, 0, traj_id=11),
           make_traj_at(3, 3, 0, traj_id=12)]
    edited_11 = make_traj_at(5, 5, 0, traj_id=11)
    out = variants.apply_edits(raw, {11: edited_11})
    assert [t.trajectory_id for t in out] == [10, 11, 12]
    assert out[1] is edited_11
    assert out[0] is raw[0] and out[2] is raw[2]


def test_load_edited_trajectories_reads_histories_pkl(tmp_path):
    bundle = _make_synthetic_bundle()
    bundle.trajectories.extend([
        make_traj_at(2, 2, 0, traj_id=100),
        make_traj_at(3, 3, 0, traj_id=101),
    ])
    edited_100 = make_traj_at(4, 4, 0, traj_id=100)
    histories = [SimpleNamespace(modified=edited_100)]
    (tmp_path / "histories.pkl").write_bytes(pickle.dumps(histories))

    out = variants.load_edited_trajectories(bundle, tmp_path)
    assert len(out) == len(bundle.trajectories)
    by_id = {t.trajectory_id: t for t in out}
    assert by_id[100] is edited_100
    assert by_id[101] in bundle.trajectories


def test_filtered_trajectories_drops_top_ranked():
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 20)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    ranked = rank_unfair_trajectory_indices(bundle)
    n = min(2, len(ranked))
    out = variants.filtered_trajectories(bundle, n)
    assert len(out) == len(bundle.trajectories) - n
    removed_ids = {bundle.trajectories[i].trajectory_id for i in ranked[:n]}
    kept_ids = {t.trajectory_id for t in out}
    assert kept_ids.isdisjoint(removed_ids)


def test_filtered_trajectories_zero_remove_is_full_corpus():
    bundle = _make_synthetic_bundle()
    out = variants.filtered_trajectories(bundle, 0)
    assert len(out) == len(bundle.trajectories)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_variants.py -v`
Expected: FAIL (ModuleNotFoundError on `gan.variants`).

- [ ] **Step 4: Implement `variants.py`**

Create `famail_temporal/baselines/gan/variants.py`:

```python
"""Training-corpus variant builders for the model-level baselines.

FAMAIL trains the shared generator on the EDITED corpus (pickups relocated by a
persisted ST-iFGSM editing run, ε=2); B2 trains on a FILTERED corpus (top-K
most-unfair trajectories removed). Both reuse the same DataBundle for fairness
scoring (scoring reads pickup_3d/mask_3d/hat_matrices, never the trajectory
list), so only the *training* trajectory list changes.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Union

from famail_temporal.data.loader import DataBundle
from famail_temporal.utils.trajectory import Trajectory
from famail_temporal.baselines.datasets import rank_unfair_trajectory_indices


def apply_edits(
    trajectories: List[Trajectory], modified_by_tid: Dict[int, Trajectory],
) -> List[Trajectory]:
    """Swap edited trajectories in by trajectory_id, preserving length/order.

    Mirrors the editing runner's trajs_after reconstruction: an entry is
    replaced iff its trajectory_id appears in modified_by_tid.
    """
    return [modified_by_tid.get(t.trajectory_id, t) for t in trajectories]


def load_edited_trajectories(
    bundle: DataBundle, edit_dir: Union[str, Path],
) -> List[Trajectory]:
    """Build the FAMAIL edited corpus from a persisted editing run.

    Reads <edit_dir>/histories.pkl (each element exposes `.modified` carrying
    the relocated pickup and its trajectory_id) and swaps those into
    bundle.trajectories. Returns a list the same length/order as
    bundle.trajectories.
    """
    with open(Path(edit_dir) / "histories.pkl", "rb") as f:
        histories = pickle.load(f)
    modified_by_tid = {h.modified.trajectory_id: h.modified for h in histories}
    return apply_edits(bundle.trajectories, modified_by_tid)


def filtered_trajectories(bundle: DataBundle, n_remove: int) -> List[Trajectory]:
    """bundle.trajectories with the top-`n_remove` most-unfair removed."""
    if n_remove <= 0:
        return list(bundle.trajectories)
    removed = set(rank_unfair_trajectory_indices(bundle)[:n_remove])
    return [t for i, t in enumerate(bundle.trajectories) if i not in removed]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_variants.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/gan/variants.py famail_temporal/baselines/gan/tests/test_variants.py
git commit -m "feat(baselines/gan): FAMAIL/B2 training-corpus variant builders"
```

---

## Task 2: `fit_and_evaluate` gains `train_trajectories` + exposes `pickups`

**Files:**
- Modify: `famail_temporal/baselines/gan/model_level.py`
- Test: `famail_temporal/baselines/gan/tests/test_model_level_variants.py`

Two additive changes: a new `train_trajectories` keyword (default `None` → `bundle.trajectories`) so FAMAIL trains on the edited corpus, AND `pickups` added to the result dict so the orchestrator can build terminal-cell histograms without a second generation pass.

- [ ] **Step 1: Write the failing test**

Create `famail_temporal/baselines/gan/tests/test_model_level_variants.py`:

```python
"""fit_and_evaluate train_trajectories param + pickups exposure."""
import torch

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import active_units, make_traj_at
from famail_temporal.baselines.gan import model_level


def test_train_trajectories_controls_generation_count():
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 20)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    subset = bundle.trajectories[:10]
    out = model_level.fit_and_evaluate(
        bundle, train_trajectories=subset,
        mle_epochs=2, adv_epochs=0, max_len=8,
        device=torch.device("cpu"), seed=0,
    )
    assert out["n_generated"] == len(subset)
    assert set(out["corpus"]) == {"f_spatial", "f_causal", "gini_dsr", "gini_asr"}


def test_default_train_trajectories_is_full_corpus_and_exposes_pickups():
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 12)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    out = model_level.fit_and_evaluate(
        bundle, mle_epochs=2, adv_epochs=0, max_len=8,
        device=torch.device("cpu"), seed=0,
    )
    assert out["n_generated"] == len(bundle.trajectories)
    # pickups exposed for downstream metric work (transmission, DI, etc.)
    assert "pickups" in out
    assert len(out["pickups"]) == out["n_generated"]
    assert all(len(p) == 3 for p in out["pickups"])  # (x, y, t_block)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_model_level_variants.py -v`
Expected: FAIL — first test errors with `TypeError: ... unexpected keyword argument 'train_trajectories'`.

- [ ] **Step 3: Add `train_trajectories` + `pickups`**

In `famail_temporal/baselines/gan/model_level.py`, locate the current signature:

```python
def fit_and_evaluate(
    bundle: DataBundle, *,
    mle_epochs: int = gc.MLE_EPOCHS,
    adv_epochs: int = gc.ADV_EPOCHS,
    max_len: int = gc.MAX_GEN_LEN,
    mle_batch_size: int = gc.MLE_BATCH_SIZE,
    adv_batch_size: int = gc.ADV_BATCH_SIZE,
    adv_lr_g: float = gc.ADV_LR_G,
    adv_lr_d: float = gc.ADV_LR_D,
    d_update_every: int = gc.D_UPDATE_EVERY,
    adv_mle_lambda: float = gc.ADV_MLE_LAMBDA,
    adv_max_len: int | None = None,
    gen_batch_size: int = gc.GEN_BATCH_SIZE,
    max_tokens: int | None = gc.MAX_TRAIN_TOKENS,
    device: torch.device | None = None,
    seed: int = 0,
    progress: bool = False,
) -> dict:
```

Insert `train_trajectories: list | None = None,` as the first keyword-only param (right after the opening `*,`). Then locate the corpus selection lines:

```python
    pairs = [
        (trajectory_to_tokens(t), trajectory_context(t))
        for t in bundle.trajectories
    ]
```

Replace with:

```python
    train_trajectories = (
        bundle.trajectories if train_trajectories is None else train_trajectories
    )
    if not train_trajectories:
        raise ValueError("fit_and_evaluate requires a non-empty training corpus")
    pairs = [
        (trajectory_to_tokens(t), trajectory_context(t))
        for t in train_trajectories
    ]
```

(Move the `if not bundle.trajectories` check above this, or remove it — `train_trajectories` defaults to `bundle.trajectories` so the new guard subsumes it.)

Then locate the result-dict construction near the end. Add `"pickups": pickups` to it:

```python
    result = {
        "generated": data_level_fairness(bundle, pickup_3d=gen_grid),
        "corpus": data_level_fairness(bundle),
        "n_generated": len(pickups),
        "pickups": pickups,
        "mle_losses": mle_losses,
        "adv_losses": adv_losses,
    }
```

Update the existing `test_model_level.py` so the strict equality assertion becomes a subset check (since `pickups` is now also returned):

```python
    # In test_fit_and_evaluate_returns_fairness_and_histories, change:
    #     assert set(out) == {
    #         "generated", "corpus", "n_generated", "mle_losses", "adv_losses",
    #     }
    # To:
    assert {
        "generated", "corpus", "n_generated", "mle_losses", "adv_losses",
    } <= set(out)
```

- [ ] **Step 4: Run the new test AND the existing model_level tests**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_model_level_variants.py famail_temporal/baselines/gan/tests/test_model_level.py -v`
Expected: PASS (the new behavior + B0 unchanged).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/model_level.py \
        famail_temporal/baselines/gan/tests/test_model_level_variants.py \
        famail_temporal/baselines/gan/tests/test_model_level.py
git commit -m "feat(baselines/gan): fit_and_evaluate accepts train_trajectories + exposes pickups"
```

---

## Task 3: `transmission.py` — terminal-cell JS check

**Files:**
- Create: `famail_temporal/baselines/transmission.py`
- Test: `famail_temporal/baselines/tests/test_transmission.py`

The load-bearing check: does the data-level edit signal survive the generator?

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/tests/test_transmission.py`:

```python
"""Unit tests for the terminal-cell transmission check."""
import math

import numpy as np

from famail_temporal.baselines import transmission as tr


def test_terminal_cell_histogram_normalized_and_one_hot_on_single_pickup():
    h = tr.terminal_cell_histogram([(2, 3, 0)], n_cells=100)
    assert h.shape == (100,)
    flat = 2 * 90 + 3  # gc.GY = 90; flat_cell(2, 3) = 2*90 + 3 = 183
    # In this test n_cells=100 < flat=183, so the out-of-range guard drops it.
    # Use a different example below.
    h2 = tr.terminal_cell_histogram([(0, 5, 0), (0, 5, 1)], n_cells=100)
    assert math.isclose(h2.sum(), 1.0, rel_tol=1e-12)
    assert h2[5] == 1.0  # flat_cell(0, 5) = 5; both pickups land there


def test_terminal_cell_histogram_handles_empty_input():
    h = tr.terminal_cell_histogram([], n_cells=100)
    assert h.shape == (100,) and h.sum() == 0.0


def test_jensen_shannon_zero_for_identical_distributions():
    p = np.array([0.25, 0.25, 0.5])
    js = tr.jensen_shannon_divergence(p, p)
    assert math.isclose(js, 0.0, abs_tol=1e-12)


def test_jensen_shannon_one_for_disjoint_distributions_in_bits():
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])
    js = tr.jensen_shannon_divergence(p, q)
    # JS(disjoint) = log2(2) / 2 + log2(2) / 2 = 1.0 in bits
    assert math.isclose(js, 1.0, rel_tol=1e-6)


def test_jensen_shannon_symmetric():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.4, 0.5])
    assert math.isclose(
        tr.jensen_shannon_divergence(p, q),
        tr.jensen_shannon_divergence(q, p),
        rel_tol=1e-12,
    )


def test_transmission_metrics_bundle_has_expected_keys():
    p_raw = np.array([0.5, 0.5, 0.0])
    p_edited = np.array([0.3, 0.7, 0.0])
    p_gen_b0 = np.array([0.5, 0.5, 0.0])
    p_gen_famail = np.array([0.4, 0.6, 0.0])
    out = tr.transmission_metrics(p_raw, p_edited, p_gen_b0, p_gen_famail)
    assert set(out) == {
        "js_target", "js_generated", "transmission_ratio",
        "js_b0_vs_raw", "js_famail_vs_edited",
    }
    # Target shift is real, generated shift is positive and smaller than target here.
    assert out["js_target"] > 0
    assert 0 < out["js_generated"] < out["js_target"]
    assert 0 < out["transmission_ratio"] < 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_transmission.py -v`
Expected: FAIL (`ModuleNotFoundError: ... transmission`).

- [ ] **Step 3: Implement `transmission.py`**

Create `famail_temporal/baselines/transmission.py`:

```python
"""Terminal-cell transmission check.

The fairness metric (F_causal, F_spatial) depends only on each rollout's
terminal pickup cell, so the model-level B0-vs-FAMAIL test reduces to: does
the LSTM reproduce a ~1% shift in the marginal distribution of one token?
This module measures that *before* the headline number is trusted.

Reported metrics (all JS in bits, bounded in [0, 1]):
- js_target           = JS(p_raw, p_edited)         — the signal we WANT to transmit
- js_generated        = JS(p_gen_B0, p_gen_FAMAIL)   — the signal that DID transmit
- transmission_ratio  = js_generated / js_target     — ≈1 = faithful, ≪1 = washed out
- js_b0_vs_raw, js_famail_vs_edited                  — per-variant fidelity to its own target
"""
from __future__ import annotations
from typing import Iterable, Tuple

import numpy as np

from famail_temporal.baselines.gan import config as gc


def terminal_cell_histogram(
    pickups: Iterable[Tuple[int, int, int]],
    n_cells: int = gc.N_CELLS,
) -> np.ndarray:
    """Build a normalized histogram over flat cell ids from pickup tuples.

    Each pickup is (x, y, t_block); only (x, y) is used (the metric is
    length/time-block-blind by design). Returns a length-`n_cells` array that
    sums to 1 (or all zeros if the input is empty). Out-of-vocab cells are
    dropped (no-op in production; matters for the small synthetic bundle).
    """
    h = np.zeros(n_cells, dtype=np.float64)
    for (x, y, _) in pickups:
        flat = int(x) * gc.GY + int(y)
        if 0 <= flat < n_cells:
            h[flat] += 1.0
    total = h.sum()
    return h / total if total > 0 else h


def trajectories_terminal_histogram(
    trajectories: Iterable, n_cells: int = gc.N_CELLS,
) -> np.ndarray:
    """Same as terminal_cell_histogram but reads from Trajectory.states[-1]."""
    h = np.zeros(n_cells, dtype=np.float64)
    for traj in trajectories:
        s = traj.states[-1]
        flat = int(s.x_grid) * gc.GY + int(s.y_grid)
        if 0 <= flat < n_cells:
            h[flat] += 1.0
    total = h.sum()
    return h / total if total > 0 else h


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """JS divergence in bits (log base 2). Symmetric, in [0, 1].

    Implemented as 0.5 KL(p || m) + 0.5 KL(q || m), m = 0.5 (p + q),
    with an eps-clip to avoid log(0). 0 if p == q; 1 if disjoint support.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl_bits(a: np.ndarray, b: np.ndarray) -> float:
        # Only nonzero rows of a contribute to KL(a||b).
        mask = a > 0
        return float(np.sum(
            a[mask] * (np.log2(a[mask] + eps) - np.log2(b[mask] + eps))
        ))

    return 0.5 * _kl_bits(p, m) + 0.5 * _kl_bits(q, m)


def transmission_metrics(
    p_raw: np.ndarray,
    p_edited: np.ndarray,
    p_gen_b0: np.ndarray,
    p_gen_famail: np.ndarray,
) -> dict:
    """Compute the full transmission bundle from four terminal-cell histograms."""
    js_target = jensen_shannon_divergence(p_raw, p_edited)
    js_generated = jensen_shannon_divergence(p_gen_b0, p_gen_famail)
    transmission_ratio = (
        js_generated / js_target if js_target > 0 else float("nan")
    )
    return {
        "js_target": float(js_target),
        "js_generated": float(js_generated),
        "transmission_ratio": float(transmission_ratio),
        "js_b0_vs_raw": float(jensen_shannon_divergence(p_gen_b0, p_raw)),
        "js_famail_vs_edited": float(jensen_shannon_divergence(p_gen_famail, p_edited)),
    }
```

Note: the failing-on-first-write `test_terminal_cell_histogram_normalized_and_one_hot_on_single_pickup` test's first assertion (n_cells=100, flat=183) is intentionally checking the out-of-range guard's behavior; only the second part validates a successful in-range histogram.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_transmission.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/transmission.py \
        famail_temporal/baselines/tests/test_transmission.py
git commit -m "feat(baselines): terminal-cell JS transmission check"
```

---

## Task 4: `district_metrics.py` — DI ratio (both Y conventions)

**Files:**
- Create: `famail_temporal/baselines/district_metrics.py`
- Test: `famail_temporal/baselines/tests/test_district_metrics.py`

DI under `Y_primary = active_taxis / pickup_mass` (aligned with F_causal) and `Y_supplementary = pickup_mass / active_taxis` (demand-pressure lens). Two-level averaging (mean of district-means → group ratio). Active-units only. Both numerator+denominator clamped at `config.DEMAND_FLOOR` to match F_causal's clamp convention.

- [ ] **Step 1: Inspect the data sources first (do not skip)**

Before writing the test, the implementer MUST inspect `source_data/grid_to_district_mapping.pkl` to determine its actual format (dict? array? mapping schema?). Run:

```bash
python -c "
import pickle
from famail_temporal import config
with open(config.SOURCE_DATA_DIR + '/grid_to_district_mapping.pkl', 'rb') as f:
    m = pickle.load(f)
print(type(m).__name__)
if isinstance(m, dict):
    k = next(iter(m))
    print('key sample:', repr(k), 'value sample:', repr(m[k]))
    print('n keys:', len(m))
else:
    print('shape/len:', getattr(m, 'shape', None) or len(m))
"
```

Also inspect the demographics CSV:

```bash
head -3 famail_temporal/source_data/all_demographics_by_district.csv
```

Adapt the `_load_district_grid` and `_load_hukou` helpers in Step 3 to the actual schema. If `NonRegisteredRatio` is the literal column name, use it; if not (e.g. `non_registered_ratio` or computed from `Registered`/`Total`), document the choice and use that.

- [ ] **Step 2: Write the failing tests**

Create `famail_temporal/baselines/tests/test_district_metrics.py`:

```python
"""Unit tests for district-level DI on synthetic district + grid setup."""
import numpy as np
import pytest

from famail_temporal.baselines import district_metrics as dm


def _synthetic_inputs():
    """3 districts × 4 cells/t_blocks each = 12 active units. Hukou ratios
    chosen so that district 0 = top-3 hukou, district 2 = bottom-3 hukou
    (with only 3 districts each is both top-3 and bottom-3, so we use 6
    districts to exercise the grouping cleanly)."""
    n_districts = 6
    # Hukou ratios increasing: districts 3,4,5 are top-3 (high hukou);
    # districts 0,1,2 are bottom-3 (low hukou).
    hukou_ratios = np.array([0.10, 0.15, 0.20, 0.60, 0.70, 0.80])
    # 2 active units per district -> 12 units total.
    # district_of_unit: which district each active unit belongs to.
    district_of_unit = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    # demand_N, supply_N per active unit (12 vector each)
    demand_N = np.array([1.0]*6 + [2.0]*6)        # higher demand in top-3 hukou
    supply_N = np.array([5.0]*6 + [5.0]*6)        # equal supply
    return n_districts, hukou_ratios, district_of_unit, demand_N, supply_N


def test_di_primary_below_one_when_high_hukou_has_lower_supply_demand_ratio():
    # supply/demand: low-hukou districts get 5/1=5, top-hukou get 5/2=2.5.
    # DI_primary = mean(top-hukou Y) / mean(low-hukou Y) = 2.5 / 5.0 = 0.5
    n_d, hukou, district_of_unit, demand_N, supply_N = _synthetic_inputs()
    out = dm.compute_di(
        demand_N=demand_N, supply_N=supply_N,
        district_of_unit=district_of_unit,
        hukou_ratios=hukou,
        n_top=3, n_bottom=3, demand_floor=1e-3, supply_floor=1e-3,
    )
    assert out["di_primary"] == pytest.approx(0.5, rel=1e-6)


def test_di_supplementary_is_inverse_of_primary_under_equal_supply():
    # demand/supply: top-hukou get 2/5=0.4, low-hukou get 1/5=0.2.
    # DI_supplementary = 0.4 / 0.2 = 2.0 (the inverse of 0.5)
    n_d, hukou, district_of_unit, demand_N, supply_N = _synthetic_inputs()
    out = dm.compute_di(
        demand_N=demand_N, supply_N=supply_N,
        district_of_unit=district_of_unit,
        hukou_ratios=hukou,
        n_top=3, n_bottom=3, demand_floor=1e-3, supply_floor=1e-3,
    )
    assert out["di_supplementary"] == pytest.approx(2.0, rel=1e-6)


def test_di_returns_per_district_means_for_traceability():
    n_d, hukou, district_of_unit, demand_N, supply_N = _synthetic_inputs()
    out = dm.compute_di(
        demand_N=demand_N, supply_N=supply_N,
        district_of_unit=district_of_unit,
        hukou_ratios=hukou,
        n_top=3, n_bottom=3, demand_floor=1e-3, supply_floor=1e-3,
    )
    assert "per_district_y_primary" in out
    assert out["per_district_y_primary"].shape == (6,)
    # district 0 (low-hukou): supply/demand = 5/1 = 5
    assert out["per_district_y_primary"][0] == pytest.approx(5.0, rel=1e-6)
    # district 3 (high-hukou): supply/demand = 5/2 = 2.5
    assert out["per_district_y_primary"][3] == pytest.approx(2.5, rel=1e-6)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_district_metrics.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 4: Implement `district_metrics.py`**

Create `famail_temporal/baselines/district_metrics.py`. The synthetic test exercises the pure `compute_di` core; the data loaders (`_load_grid_to_district`, `_load_hukou`, `district_of_unit_from_bundle`) wrap real-data IO and are exercised end-to-end by the orchestrator (Task 6).

```python
"""District-level Disparate Impact (DI) ratio under both Y conventions.

For each district d, restrict to its active units (mask_3d=True), compute:
- Y_primary(d)       = mean(supply_N / max(demand_N, DEMAND_FLOOR))  (aligned w/ F_causal)
- Y_supplementary(d) = mean(demand_N / max(supply_N, supply_floor))  (demand pressure)

Then DI = mean(district_means in top-n_top hukou) / mean(district_means in bottom-n_bottom hukou).

Two-level averaging normalizes for both within-district size differences and
between-district population differences, and treats each district as one unit
of analysis (matches the spec's "top-3 vs bottom-3" framing). Districts that
have zero active units are dropped before grouping.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from famail_temporal import config
from famail_temporal.data.loader import DataBundle


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else float("nan")


def compute_di(
    *,
    demand_N: np.ndarray,           # (N,) demand per active unit
    supply_N: np.ndarray,           # (N,) supply per active unit
    district_of_unit: np.ndarray,   # (N,) int — district id per active unit
    hukou_ratios: np.ndarray,       # (n_districts,) NonRegisteredRatio per district
    n_top: int = 3,
    n_bottom: int = 3,
    demand_floor: float = config.DEMAND_FLOOR,
    supply_floor: float | None = None,
) -> dict:
    """Two-level DI ratio under both Y conventions.

    Returns:
        di_primary             — mean(Y_primary in top-hukou) / mean(... bottom)
        di_supplementary       — same for Y_supplementary
        per_district_y_primary — (n_districts,) per-district mean Y_primary
        per_district_y_supplementary — (n_districts,) per-district mean Y_supplementary
        top_district_ids, bottom_district_ids — which districts entered each group
        n_active_per_district  — sanity check (which districts have zero coverage)
    """
    if supply_floor is None:
        # Match the demand-floor convention; supply rarely hits zero in practice.
        supply_floor = demand_floor

    demand_N = np.asarray(demand_N, dtype=np.float64)
    supply_N = np.asarray(supply_N, dtype=np.float64)
    district_of_unit = np.asarray(district_of_unit, dtype=np.int64)
    n_districts = len(hukou_ratios)

    y_primary_per_unit = supply_N / np.maximum(demand_N, demand_floor)
    y_supplementary_per_unit = demand_N / np.maximum(supply_N, supply_floor)

    per_district_y_primary = np.full(n_districts, np.nan, dtype=np.float64)
    per_district_y_supplementary = np.full(n_districts, np.nan, dtype=np.float64)
    n_active_per_district = np.zeros(n_districts, dtype=np.int64)
    for d in range(n_districts):
        mask = district_of_unit == d
        n_active_per_district[d] = int(mask.sum())
        if mask.any():
            per_district_y_primary[d] = _safe_mean(y_primary_per_unit[mask])
            per_district_y_supplementary[d] = _safe_mean(y_supplementary_per_unit[mask])

    # Drop districts with zero active units before ranking by hukou.
    has_coverage = ~np.isnan(per_district_y_primary)
    covered_ids = np.where(has_coverage)[0]
    if len(covered_ids) < n_top + n_bottom:
        raise ValueError(
            f"Need at least n_top + n_bottom = {n_top + n_bottom} covered "
            f"districts; have {len(covered_ids)}."
        )
    # Sort covered districts by hukou ratio ascending (low hukou first).
    order = covered_ids[np.argsort(hukou_ratios[covered_ids])]
    bottom_ids = order[:n_bottom]      # lowest hukou ratio
    top_ids = order[-n_top:]           # highest hukou ratio

    def _group_mean(district_y: np.ndarray, group: np.ndarray) -> float:
        return _safe_mean(district_y[group])

    di_primary = (
        _group_mean(per_district_y_primary, top_ids)
        / _group_mean(per_district_y_primary, bottom_ids)
    )
    di_supplementary = (
        _group_mean(per_district_y_supplementary, top_ids)
        / _group_mean(per_district_y_supplementary, bottom_ids)
    )

    return {
        "di_primary": float(di_primary),
        "di_supplementary": float(di_supplementary),
        "per_district_y_primary": per_district_y_primary,
        "per_district_y_supplementary": per_district_y_supplementary,
        "top_district_ids": top_ids.tolist(),
        "bottom_district_ids": bottom_ids.tolist(),
        "n_active_per_district": n_active_per_district.tolist(),
    }


def _load_grid_to_district() -> np.ndarray:
    """Return (GX, GY) int array mapping each grid cell to a district id.

    Adapt this function in Step 1 to the actual pkl format. Returns -1 for
    cells with no district assignment (those units are dropped from DI).
    """
    path = Path(config.SOURCE_DATA_DIR) / "grid_to_district_mapping.pkl"
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    # Implementer: adapt based on the inspection output in Step 1.
    # The common shape is either (GX, GY) array directly, or a dict
    # {(x, y): district_id}. Build a (GX, GY) array initialized to -1.
    GX, GY = config.GRID_DIMS
    grid = np.full((GX, GY), -1, dtype=np.int64)
    if isinstance(mapping, np.ndarray):
        grid = mapping.astype(np.int64)
    elif isinstance(mapping, dict):
        for (x, y), did in mapping.items():
            if 0 <= int(x) < GX and 0 <= int(y) < GY:
                grid[int(x), int(y)] = int(did)
    else:
        raise ValueError(
            f"Unexpected grid_to_district_mapping.pkl format: {type(mapping).__name__}"
        )
    return grid


def _load_hukou() -> Tuple[np.ndarray, list[str]]:
    """Return (hukou_ratios (n_districts,), district_names) from the demographics CSV."""
    path = Path(config.SOURCE_DATA_DIR) / "all_demographics_by_district.csv"
    df = pd.read_csv(path)
    # The exact column name comes from the data dictionary; the canonical
    # name is `NonRegisteredRatio`. Adjust in Step 1 if the CSV header differs.
    if "NonRegisteredRatio" not in df.columns:
        raise KeyError(
            "Expected 'NonRegisteredRatio' column in all_demographics_by_district.csv; "
            f"got columns {list(df.columns)}. Adjust _load_hukou to compute it from "
            "raw counts if the CSV uses different column names."
        )
    return df["NonRegisteredRatio"].to_numpy(dtype=np.float64), df.iloc[:, 0].tolist()


def district_of_active_units(bundle: DataBundle) -> np.ndarray:
    """For each active unit (i in the flat active-unit vector), return its district id.

    Active units are unique (cell, t_block) tuples flagged by mask_3d. We
    aggregate to (cell, _) — the district is a property of the cell, not the
    t_block — so unit i's district = grid_to_district[unit_xy[i]].
    """
    grid_to_dist = _load_grid_to_district()
    mask = bundle.mask_3d  # (GX, GY, T)
    GX, GY, T = mask.shape
    # Active units in flat-vector order: numpy where iterates in C order over
    # the bundle.mask_3d, matching how X_demo etc. were built. Use same order.
    xx, yy, tt = np.where(mask)
    return grid_to_dist[xx, yy]


def di_from_bundle_and_pickup_grid(
    bundle: DataBundle,
    pickup_3d: np.ndarray,
    *,
    n_top: int = 3,
    n_bottom: int = 3,
) -> dict:
    """End-to-end DI from a bundle + a pickup demand grid (raw or generated).

    Reads supply (active_taxis) from the bundle, demand from the provided
    pickup_3d, aggregates to active units, and calls compute_di.
    """
    mask = bundle.mask_3d
    demand_N = pickup_3d[mask].astype(np.float64)
    supply_N = bundle.active_taxis_3d[mask].astype(np.float64)
    district_of_unit = district_of_active_units(bundle)
    hukou_ratios, _ = _load_hukou()
    return compute_di(
        demand_N=demand_N, supply_N=supply_N,
        district_of_unit=district_of_unit,
        hukou_ratios=hukou_ratios,
        n_top=n_top, n_bottom=n_bottom,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_district_metrics.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/district_metrics.py \
        famail_temporal/baselines/tests/test_district_metrics.py
git commit -m "feat(baselines): district-level DI ratio (both Y conventions)"
```

---

## Task 5: `localized_metrics.py` — F_causal restricted to edited units

**Files:**
- Create: `famail_temporal/baselines/localized_metrics.py`
- Test: `famail_temporal/baselines/tests/test_localized_metrics.py`

Restrict the F_causal regression to the active units the edit actually touches. With M=I (uniform weighting), localized F_causal = 1 − R'(I − H_demo)R / R'R = 1 − r²_demo on the touched subset. Uses the bundle's frozen `g0_func` for the residual to keep `R` consistent across conditions.

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/tests/test_localized_metrics.py`:

```python
"""Unit tests for localized F_causal."""
import numpy as np
import pytest

from famail_temporal.baselines import localized_metrics as lm


def test_localized_f_causal_one_when_residual_orthogonal_to_demo():
    # Residual R is orthogonal to the demographic columns -> r²=0, F_causal=1.
    rng = np.random.default_rng(0)
    n = 50
    X_demo = rng.standard_normal((n, 3))
    # Construct R orthogonal to X_demo by removing X_demo's projection.
    R = rng.standard_normal(n)
    H = X_demo @ np.linalg.pinv(X_demo.T @ X_demo) @ X_demo.T
    R_orth = R - H @ R
    f = lm.f_causal_orthogonality(R_orth, X_demo)
    assert f == pytest.approx(1.0, abs=1e-8)


def test_localized_f_causal_zero_when_residual_fully_in_demo_span():
    rng = np.random.default_rng(1)
    n = 50
    X_demo = rng.standard_normal((n, 3))
    # R is a linear combination of X_demo -> r²=1, F_causal=0.
    beta = np.array([1.5, -0.7, 2.3])
    R = X_demo @ beta
    f = lm.f_causal_orthogonality(R, X_demo)
    assert f == pytest.approx(0.0, abs=1e-8)


def test_localized_f_causal_zero_safe_on_zero_residual():
    # Degenerate: R = 0. Define F_causal = 1.0 (no residual to explain).
    n = 10
    X_demo = np.random.default_rng(0).standard_normal((n, 3))
    R = np.zeros(n)
    assert lm.f_causal_orthogonality(R, X_demo) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_localized_metrics.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `localized_metrics.py`**

Create `famail_temporal/baselines/localized_metrics.py`:

```python
"""F_causal restricted to the active units the edit touches.

The global F_causal dilutes the editing signal across ~34,524 active units;
locally (the ~1,186-3,773 units the edit relocates pickups *from*) the effect
is concentrated. Localized F_causal uses the same residual definition as the
global metric — R = Y − g_0(D), Y = supply/demand, g_0 from the bundle — but
restricts the orthogonality computation to the touched units.

With M = I (uniform weighting), F_causal_localized = 1 − R'(I−H_demo)R / R'R,
which is 1 − r²_demo on the touched subset.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.loader import DataBundle


def edited_units_from_histories(
    edit_dir: str | Path,
) -> List[Tuple[int, int, int]]:
    """Return (x, y, t_block) of each edited trajectory's ORIGINAL pickup unit.

    Reads <edit_dir>/histories.pkl. We use the *original* pickup unit (not the
    modified one) because the editing moves the pickup OUT OF that unit's
    demand — that's the unit where the change is concentrated.
    """
    with open(Path(edit_dir) / "histories.pkl", "rb") as f:
        histories = pickle.load(f)
    out = []
    for h in histories:
        s = h.original.states[-1]
        # Identify t_block from the original pickup state's time bucket.
        # Use the same convention as Phase-2 sequences: t_block = hour_to_block(time_bucket_to_hour(...))
        from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
        t_block = hour_to_block_index(time_bucket_to_hour(s.time_bucket))
        out.append((int(s.x_grid), int(s.y_grid), int(t_block)))
    return out


def active_unit_index_of(
    bundle: DataBundle, units: Iterable[Tuple[int, int, int]],
) -> np.ndarray:
    """Map (x, y, t_block) units to their flat active-unit index.

    Returns a 1-D int array of indices into the N-vector ordering used by
    pickup_N, supply_N, X_demo, etc. (i.e., C-order traversal of mask_3d).
    Drops units that fall outside mask_3d (inactive cells).
    """
    mask = bundle.mask_3d  # (GX, GY, T)
    # Build a (GX, GY, T) lookup table: -1 for inactive, else its flat index.
    flat_index = np.full(mask.shape, -1, dtype=np.int64)
    flat_index[mask] = np.arange(int(mask.sum()))
    seen = set()
    out: list[int] = []
    for (x, y, t) in units:
        idx = int(flat_index[int(x), int(y), int(t)])
        if idx >= 0 and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return np.array(out, dtype=np.int64)


def residual_and_demo(
    bundle: DataBundle, pickup_3d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute R = supply/demand − g_0(demand) and X_demo over ALL active units.

    Uses bundle.g0_func (frozen) on the clamped demand, exactly as
    FAMAILObjective does (so the residual is comparable to the global F_causal).
    Returns (R, X_demo) both as numpy arrays.
    """
    mask = bundle.mask_3d
    demand_N = pickup_3d[mask].astype(np.float64)
    supply_N = bundle.active_taxis_3d[mask].astype(np.float64)
    Y = supply_N / np.maximum(demand_N, config.DEMAND_FLOOR)
    D_clamped = np.maximum(demand_N, config.DEMAND_FLOOR)
    # g0_func.eval_torch lives on tensors; convert in/out.
    D_t = torch.from_numpy(D_clamped).float()
    g0 = bundle.g0_func.eval_torch(D_t).cpu().numpy().astype(np.float64)
    R = Y - g0
    # X_demo is a torch tensor stored as a buffer on FAMAILObjective in the
    # gradient path; the bundle's hat_matrices dict carries the raw numpy.
    X_demo = np.asarray(bundle.hat_matrices["X_demo"], dtype=np.float64)
    return R, X_demo


def f_causal_orthogonality(R: np.ndarray, X_demo: np.ndarray) -> float:
    """F_causal under M = I: 1 − R'(I−H_demo)R / R'R = 1 − r²_demo.

    Degenerate cases:
    - R has zero norm -> return 1.0 (no residual variance to explain).
    - X_demo has zero columns / rank 0 -> H_demo = 0, F_causal = 0.
    """
    R = np.asarray(R, dtype=np.float64).ravel()
    n = R.shape[0]
    if n == 0:
        return float("nan")
    rr = float(R @ R)
    if rr <= 0.0:
        return 1.0
    if X_demo.size == 0 or X_demo.shape[1] == 0:
        return 0.0
    # H_demo = X (X'X)^-1 X'  (pinv for numerical safety)
    XtX = X_demo.T @ X_demo
    XtX_inv = np.linalg.pinv(XtX)
    H_demo = X_demo @ XtX_inv @ X_demo.T
    res = R - H_demo @ R               # (I − H_demo) R
    return float(1.0 - (R @ res) / rr)


def localized_f_causal(
    bundle: DataBundle, pickup_3d: np.ndarray, edited_units: Iterable[Tuple[int, int, int]],
) -> dict:
    """Return localized + global F_causal for traceability."""
    R, X_demo = residual_and_demo(bundle, pickup_3d)
    f_global = f_causal_orthogonality(R, X_demo)
    idx = active_unit_index_of(bundle, edited_units)
    if idx.size == 0:
        return {
            "f_causal_localized": float("nan"),
            "f_causal_global": float(f_global),
            "n_edited_active_units": 0,
        }
    f_local = f_causal_orthogonality(R[idx], X_demo[idx])
    return {
        "f_causal_localized": float(f_local),
        "f_causal_global": float(f_global),
        "n_edited_active_units": int(idx.size),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_localized_metrics.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/localized_metrics.py \
        famail_temporal/baselines/tests/test_localized_metrics.py
git commit -m "feat(baselines): localized F_causal restricted to edited units"
```

---

## Task 6: `run_metric_hardening.py` — orchestrator + persistence

**Files:**
- Create: `famail_temporal/baselines/run_metric_hardening.py`
- Test: `famail_temporal/baselines/tests/test_run_metric_hardening.py`

One CLI: load corpus + edited corpus, train MLE-only B0 + MLE-only FAMAIL, compute all three metric bundles, write a per-run dir + append the canonical `RESULTS.md`.

- [ ] **Step 1: Write the failing test** (only the pure JSON helpers are unit-tested; the full run is manual)

Create `famail_temporal/baselines/tests/test_run_metric_hardening.py`:

```python
"""Unit tests for the metric-hardening CLI's pure helpers."""
import json

from famail_temporal.baselines import run_metric_hardening as r


def test_result_to_json_roundtrips():
    result = {
        "transmission": {
            "js_target": 0.02, "js_generated": 0.014,
            "transmission_ratio": 0.70,
            "js_b0_vs_raw": 0.001, "js_famail_vs_edited": 0.005,
        },
        "di_b0": {"di_primary": 1.02, "di_supplementary": 0.98},
        "di_famail": {"di_primary": 1.07, "di_supplementary": 0.93},
        "localized_b0": {"f_causal_localized": 0.42, "f_causal_global": 0.808, "n_edited_active_units": 3773},
        "localized_famail": {"f_causal_localized": 0.46, "f_causal_global": 0.812, "n_edited_active_units": 3773},
        "edit_dir": "famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup",
    }
    blob = r.result_to_json(result)
    loaded = json.loads(blob)
    assert loaded["transmission"]["transmission_ratio"] == 0.70
    assert loaded["di_famail"]["di_primary"] == 1.07
    assert loaded["localized_b0"]["n_edited_active_units"] == 3773
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_metric_hardening.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `run_metric_hardening.py`**

Create `famail_temporal/baselines/run_metric_hardening.py`:

```python
"""CLI: model-level metric hardening for the FAMAIL paper headline.

Trains MLE-only B0 (full corpus) and MLE-only FAMAIL (edited corpus), then:
  1. JS terminal-cell transmission check.
  2. Disparate impact (DI) under both Y conventions.
  3. Localized F_causal restricted to the edited active units.

Per-run artifacts: <out-dir>/metrics.json, terminal_cell_histograms.npz,
report.md. The canonical summary at famail_temporal/baselines/metric_hardening/
RESULTS.md is updated by Task 7 once the first real-data run lands.

Example:
    python -m famail_temporal.baselines.run_metric_hardening \\
        --edit-dir famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup \\
        --mle-epochs 5 --device auto --seed 0
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.model_level import fit_and_evaluate
from famail_temporal.baselines.gan.variants import load_edited_trajectories
from famail_temporal.baselines.gan.rollout import pickups_to_pickup_3d
from famail_temporal.baselines.transmission import (
    terminal_cell_histogram, trajectories_terminal_histogram, transmission_metrics,
)
from famail_temporal.baselines.district_metrics import di_from_bundle_and_pickup_grid
from famail_temporal.baselines.localized_metrics import (
    edited_units_from_histories, localized_f_causal,
)


DEFAULT_EDIT_DIR = (
    Path(config.PACKAGE_ROOT) / "results"
    / "2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup"
)


def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2, default=float)


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _write_report(out_dir: Path, result: dict, cmd: List[str]) -> None:
    """Per-run human-readable report (markdown). Paper-ready prose."""
    t = result["transmission"]
    di_b0 = result["di_b0"]
    di_fam = result["di_famail"]
    loc_b0 = result["localized_b0"]
    loc_fam = result["localized_famail"]
    report = f"""# Metric hardening run report

**Command**: `{' '.join(cmd)}`
**Edit source**: `{result['edit_dir']}`

## Transmission (does the data-level signal survive the LSTM?)

| Quantity | Value |
|---|---|
| JS(p_raw, p_edited) — *target* shift | **{t['js_target']:.5f}** bits |
| JS(p_gen_B0, p_gen_FAMAIL) — *transmitted* shift | **{t['js_generated']:.5f}** bits |
| **Transmission ratio** (transmitted / target) | **{t['transmission_ratio']:.3f}** |
| JS(p_gen_B0, p_raw) — B0 fidelity to raw target | {t['js_b0_vs_raw']:.5f} |
| JS(p_gen_FAMAIL, p_edited) — FAMAIL fidelity to edited target | {t['js_famail_vs_edited']:.5f} |

Reading: transmission_ratio ≈ 1.0 means the generator faithfully transmitted
the edit; ≪ 1 means MLE smoothing + multinomial sampling washed it out.

## Disparate impact (DI) — both Y conventions

|       | Y = supply/demand (primary; F_causal-aligned) | Y = demand/supply (supplementary) |
|---|---:|---:|
| B0     | {di_b0['di_primary']:.4f} | {di_b0['di_supplementary']:.4f} |
| FAMAIL | {di_fam['di_primary']:.4f} | {di_fam['di_supplementary']:.4f} |
| ΔDI    | **{di_fam['di_primary'] - di_b0['di_primary']:+.4f}** | **{di_fam['di_supplementary'] - di_b0['di_supplementary']:+.4f}** |

Top-3 hukou districts: {di_b0.get('top_district_ids', [])}; bottom-3: {di_b0.get('bottom_district_ids', [])}.
Both DIs should move in the *same* direction under FAMAIL editing (robustness).

## Localized F_causal (restricted to {loc_b0['n_edited_active_units']} edited active units)

|       | F_causal_global | F_causal_localized |
|---|---:|---:|
| B0     | {loc_b0['f_causal_global']:.4f} | {loc_b0['f_causal_localized']:.4f} |
| FAMAIL | {loc_fam['f_causal_global']:.4f} | {loc_fam['f_causal_localized']:.4f} |
| Δ      | {loc_fam['f_causal_global'] - loc_b0['f_causal_global']:+.4f} | **{loc_fam['f_causal_localized'] - loc_b0['f_causal_localized']:+.4f}** |

Reading: localized Δ should be substantially larger than global Δ because the
edit's effect concentrates in the touched units. If localized Δ is also small,
the headline is fragile and the data-level Pareto is the more honest framing.
"""
    (out_dir / "report.md").write_text(report)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.baselines.run_metric_hardening",
    )
    ap.add_argument("--mle-epochs", type=int, default=gc.MLE_EPOCHS)
    ap.add_argument("--max-len", type=int, default=gc.MAX_GEN_LEN)
    ap.add_argument("--mle-batch-size", type=int, default=gc.MLE_BATCH_SIZE)
    ap.add_argument("--max-tokens", type=int, default=gc.MAX_TRAIN_TOKENS)
    ap.add_argument("--gen-batch-size", type=int, default=gc.GEN_BATCH_SIZE)
    ap.add_argument("--edit-dir", type=Path, default=DEFAULT_EDIT_DIR,
                    help="Results dir with histories.pkl for the FAMAIL edit")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    device = _resolve_device(args.device)
    bundle = DataBundle.load()
    edited_trajs = load_edited_trajectories(bundle, args.edit_dir)

    # --- Train B0 (MLE-only, full corpus) ---
    b0 = fit_and_evaluate(
        bundle,
        mle_epochs=args.mle_epochs, adv_epochs=0, max_len=args.max_len,
        mle_batch_size=args.mle_batch_size,
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
        gen_batch_size=args.gen_batch_size,
        device=device, seed=args.seed, progress=not args.quiet,
    )

    # --- Train FAMAIL (MLE-only, edited corpus) ---
    famail = fit_and_evaluate(
        bundle, train_trajectories=edited_trajs,
        mle_epochs=args.mle_epochs, adv_epochs=0, max_len=args.max_len,
        mle_batch_size=args.mle_batch_size,
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
        gen_batch_size=args.gen_batch_size,
        device=device, seed=args.seed, progress=not args.quiet,
    )

    # --- Build histograms (terminal cells only — what the metric sees) ---
    p_raw = trajectories_terminal_histogram(bundle.trajectories)
    p_edited = trajectories_terminal_histogram(edited_trajs)
    p_gen_b0 = terminal_cell_histogram(b0["pickups"])
    p_gen_famail = terminal_cell_histogram(famail["pickups"])

    transmission = transmission_metrics(p_raw, p_edited, p_gen_b0, p_gen_famail)

    # --- DI on generated grids (uses each variant's pickup grid) ---
    b0_grid = pickups_to_pickup_3d(bundle, b0["pickups"])
    famail_grid = pickups_to_pickup_3d(bundle, famail["pickups"])
    di_b0 = di_from_bundle_and_pickup_grid(bundle, b0_grid)
    di_famail = di_from_bundle_and_pickup_grid(bundle, famail_grid)

    # --- Localized F_causal on the same grids, restricted to edited units ---
    edited_units = edited_units_from_histories(args.edit_dir)
    loc_b0 = localized_f_causal(bundle, b0_grid, edited_units)
    loc_famail = localized_f_causal(bundle, famail_grid, edited_units)

    result = {
        "transmission": transmission,
        "di_b0": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                  for k, v in di_b0.items()},
        "di_famail": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                      for k, v in di_famail.items()},
        "localized_b0": loc_b0,
        "localized_famail": loc_famail,
        "edit_dir": str(args.edit_dir),
        "b0_fairness": b0["generated"],
        "famail_fairness": famail["generated"],
        "corpus_fairness": b0["corpus"],
        "n_generated": b0["n_generated"],
        "mle_losses_b0": b0["mle_losses"],
        "mle_losses_famail": famail["mle_losses"],
    }

    # --- Persist ---
    if args.out_dir is None:
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = (
            Path(config.PACKAGE_ROOT) / "baselines" / "metric_hardening"
            / "results" / f"{timestamp}_metric_hardening"
        )
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(result_to_json(result))
    np.savez(
        out_dir / "terminal_cell_histograms.npz",
        p_raw=p_raw, p_edited=p_edited,
        p_gen_b0=p_gen_b0, p_gen_famail=p_gen_famail,
    )
    cmd = ["python", "-m", "famail_temporal.baselines.run_metric_hardening"]
    if argv:
        cmd += argv
    _write_report(out_dir, result, cmd)

    print(f"\\n=== Metric hardening summary ===")
    print(f"Transmission ratio: {transmission['transmission_ratio']:.3f} "
          f"(JS_target={transmission['js_target']:.5f}, "
          f"JS_generated={transmission['js_generated']:.5f})")
    print(f"DI_primary       B0={di_b0['di_primary']:.4f}  "
          f"FAMAIL={di_famail['di_primary']:.4f}  "
          f"ΔDI={di_famail['di_primary'] - di_b0['di_primary']:+.4f}")
    print(f"F_causal local   B0={loc_b0['f_causal_localized']:.4f}  "
          f"FAMAIL={loc_famail['f_causal_localized']:.4f}  "
          f"Δ={loc_famail['f_causal_localized'] - loc_b0['f_causal_localized']:+.4f}")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run unit test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_metric_hardening.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full baselines suite**

Run: `python -m pytest famail_temporal/baselines/ -q`
Expected: PASS (all old + new tests).

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/run_metric_hardening.py \
        famail_temporal/baselines/tests/test_run_metric_hardening.py
git commit -m "feat(baselines): metric-hardening orchestrator (transmission + DI + localized)"
```

- [ ] **Step 7: Real-data smoke (manual; GPU; ~5 min)**

Run:

```bash
python -m famail_temporal.baselines.run_metric_hardening \
    --edit-dir famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup \
    --mle-epochs 5 --device auto --seed 0
```

Expected: writes a timestamped per-run dir under `famail_temporal/baselines/metric_hardening/results/`, prints the three-line summary, and exits 0. Inspect `report.md` and `metrics.json` in that dir.

**Do NOT commit the result artifacts** (the `results/` subdir under `metric_hardening/` is a research artifact directory, same as the other `famail_temporal/results/`). They are referenced from `RESULTS.md` and `STATUS.md` (Task 7) so paper-writing can find them.

---

## Task 7: Documentation — methods doc + RESULTS.md + STATUS update

**Files:**
- Create: `famail_temporal/docs/MODEL_LEVEL_METRICS.md`
- Create: `famail_temporal/baselines/metric_hardening/RESULTS.md`
- Modify: `famail_temporal/baselines/STATUS.md`

- [ ] **Step 1: Create the methods doc**

Write `famail_temporal/docs/MODEL_LEVEL_METRICS.md` with three sections, one per metric, each containing:

1. **Motivation** — what failure mode this metric protects against (transmission: terminal-cell-blindness of F_causal; DI: r²'s small dynamic range; localized: dilution across 34k units).
2. **Formula** — verbatim from the code:
   - JS in bits, formula and properties.
   - DI primary: `Y = active_taxis / max(pickup_mass, DEMAND_FLOOR)`; two-level mean of district means; ratio top-3-hukou / bottom-3-hukou. DI supplementary: flipped `Y`.
   - Localized F_causal: same `R = Y − g_0(D)` as global F_causal, but the orthogonality computation `1 − R'(I−H_demo)R / R'R` is restricted to the active units in `histories.pkl::original`.
3. **Reading rules** — what does a healthy / red-flag number look like, with explicit thresholds (e.g., "transmission_ratio < 0.3 means model-level headline is fragile; lead with data-level Pareto").
4. **Reproduction** — exact CLI command + edit-dir + edge-case notes.

Include a link to this doc in `famail_temporal/docs/TRAJECTORY_EDITING_METHODOLOGY.md`'s introduction (one-line "see also").

- [ ] **Step 2: Create the paper-ready RESULTS.md**

Write `famail_temporal/baselines/metric_hardening/RESULTS.md` with:

1. **TL;DR table** — one row per (seed, edit_dir) configuration:
   `| edit_dir | seed | transmission_ratio | ΔDI_primary | ΔDI_supplementary | ΔF_causal_localized | verdict |`
   where `verdict ∈ {"model-level defensible", "fragile — lead with data-level Pareto"}`.
2. **Headline numbers from the real-data run** (copy from Task 6 Step 7's per-run `metrics.json`):
   - Transmission: `js_target`, `js_generated`, `transmission_ratio`.
   - DI: B0/FAMAIL under both conventions.
   - Localized F_causal: B0/FAMAIL global vs localized.
3. **Interpretation paragraphs** — one per metric, drop-in for the paper.
4. **Reproduction** — exact command, edit-dir, and a pointer to the per-run dir.

- [ ] **Step 3: Update STATUS.md**

Append a new section under the existing Phase headings:

```markdown
## Phase 4 — model-level (MLE-only B0/FAMAIL) — METRIC HARDENING DONE

Plan: [`docs/superpowers/plans/2026-06-06-metric-hardening.md`](../../docs/superpowers/plans/2026-06-06-metric-hardening.md).

Adds the model-level transmission + dynamic-range metrics (`baselines/transmission.py`,
`district_metrics.py`, `localized_metrics.py`, `run_metric_hardening.py`). Both
generators (B0 + FAMAIL) train MLE-only via `fit_and_evaluate(..., adv_epochs=0,
train_trajectories=...)`. The collapsing adversarial GAN remains an opt-in
"amplification" ablation per the `B0_DECISION_BRIEF.md` pivot.

### Results — first real-data run (<timestamp>)

| Metric | B0 | FAMAIL | Δ |
|---|---:|---:|---:|
| Transmission ratio (JS_generated / JS_target) | — | — | **<filled>** |
| DI_primary (supply/demand, F_causal-aligned)   | <filled> | <filled> | <filled> |
| DI_supplementary (demand/supply)               | <filled> | <filled> | <filled> |
| F_causal_localized (~3,773 edited units)        | <filled> | <filled> | <filled> |
| F_causal_global                                 | <filled> | <filled> | <filled> |

Reading: **<paper-ready one-sentence verdict>**. Full writeup:
[`baselines/metric_hardening/RESULTS.md`](metric_hardening/RESULTS.md);
methodology: [`docs/MODEL_LEVEL_METRICS.md`](../docs/MODEL_LEVEL_METRICS.md).
```

(Fill `<filled>` and `<paper-ready one-sentence verdict>` from Task 6 Step 7's output.)

- [ ] **Step 4: Commit the docs**

```bash
git add famail_temporal/docs/MODEL_LEVEL_METRICS.md \
        famail_temporal/baselines/metric_hardening/RESULTS.md \
        famail_temporal/baselines/STATUS.md
git commit -m "docs(baselines): model-level metric hardening — methodology + results"
```

---

## Self-Review

**1. Spec coverage:**
- Phase-4 prerequisites (variants + train_trajectories) — Tasks 1, 2. ✓
- JS terminal-cell transmission check — Task 3. ✓ (matches earlier design: `js_target`, `js_generated`, `transmission_ratio`, per-variant fidelity).
- DI under primary (supply/demand, F_causal-aligned) AND supplementary (demand/supply, robustness) — Task 4. ✓
- Localized F_causal on edited active units — Task 5. ✓
- Orchestrator + per-run persistence + canonical RESULTS.md — Task 6 + 7. ✓
- Methods doc — Task 7. ✓
- Branch off `implement-gan-baselines` — Task 1 Step 1. ✓
- F_causal `Y = supply/demand` unchanged — explicitly out of scope.
- DEMAND_FLOOR clamp matches F_causal convention — Task 4 / Task 5 use `config.DEMAND_FLOOR`. ✓

**2. Placeholder scan:**
- Task 4 Step 1 has a "if the column name differs from `NonRegisteredRatio`" branch — this is a real-data inspection step, not a placeholder; the implementer fills it from the actual CSV header.
- Task 4 Step 3's `_load_grid_to_district` has two format branches (np.ndarray vs dict) — both implemented; the inspection step picks which is real.
- No "TBD" / "similar to Task N" / "add error handling" without code.

**3. Type consistency:**
- `apply_edits(trajectories, modified_by_tid) -> List[Trajectory]`; `load_edited_trajectories(bundle, edit_dir) -> List[Trajectory]`; `filtered_trajectories(bundle, n_remove) -> List[Trajectory]`. Used by `run_metric_hardening.py`. ✓
- `fit_and_evaluate(bundle, *, train_trajectories=None, ...) -> dict` with `pickups` key added. ✓
- `terminal_cell_histogram(pickups, n_cells) -> np.ndarray`; `trajectories_terminal_histogram(trajectories, n_cells) -> np.ndarray`; `jensen_shannon_divergence(p, q) -> float`; `transmission_metrics(p_raw, p_edited, p_gen_b0, p_gen_famail) -> dict`. ✓
- `compute_di(demand_N, supply_N, district_of_unit, hukou_ratios, *, n_top, n_bottom, demand_floor, supply_floor) -> dict`; `di_from_bundle_and_pickup_grid(bundle, pickup_3d, *, n_top, n_bottom) -> dict`. ✓
- `edited_units_from_histories(edit_dir) -> List[(x, y, t_block)]`; `localized_f_causal(bundle, pickup_3d, edited_units) -> dict`. ✓

**4. Ambiguity:**
- DI's `Y_primary` is `supply/demand` (matches F_causal's residual numerator structure); `Y_supplementary` is `demand/supply` (robustness). Both clamp the denominator with `config.DEMAND_FLOOR` (primary) / `supply_floor=DEMAND_FLOOR` default (supplementary). Stated in Task 4 design notes + code comments.
- Localized F_causal uses M=I (uniform weighting) so it reduces to `1 − r²_demo` on the subset — stated in the docstring + tests.
- The pure-function DI/localized cores are synthetic-tested; the real-data IO wrappers are end-to-end exercised by Task 6 Step 7's smoke. No `bundle` mocking is required for the unit tests.

**5. Standing-constraint check:**
- No change to the trajectory-editing algorithm — all new code is in `baselines/` and reads existing `histories.pkl` / bundle artifacts. ε=2 / editing intermediates untouched. ✓
- F_causal's `Y = supply/demand` unchanged (and the new metrics align with it). ✓
- `git add` stages only named files in every commit. No `git add -A`. ✓
- Branch `metric-hardening` keeps everything separate from the `implement-gan-baselines` work the PI brief is still pending on. ✓
- `model.train(False)` used in any inference paths (rollouts go through the existing `generate_pickups` which already uses this idiom). No `.eval()` introduced.

---
