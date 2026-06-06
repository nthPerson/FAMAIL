# FAMAIL GAN Baselines — Phase 4: FAMAIL + B2 Model-Level Dataset Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the model-level **headline** — train the shared Phase-3 generator (MLE → adversarial) on the **FAMAIL-edited** corpus and on **B2-filtered** corpora, and compare their rollout fairness against the raw-data **B0**, all scored on the same `DataBundle`. This earns the central claim: *a generator trained on edited data produces fairer rollouts than one trained on raw data, at full retention, and fairer-per-retained-trajectory than filtering.*

**Architecture:** Phase 3 proved that scoring is trajectory-independent (it reads only `pickup_3d/mask_3d/hat_matrices`), so every baseline reuses the **same bundle** for fairness and varies only the *training* trajectory list. This phase adds (1) a `variants.py` module that builds the edited and filtered training corpora, (2) a one-parameter generalization of `fit_and_evaluate` to accept an explicit `train_trajectories` list, and (3) a suite orchestrator + CLI that runs B0 / FAMAIL / B2-levels with a shared seed and reports them side by side. The FAMAIL edit is loaded from a persisted editing run via a configurable `--edit-dir` flag (no slow re-edit), defaulting to the validated no-dedup k=10000 run.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest. Reuses Phase 3's `fit_and_evaluate`, Phase 1's `rank_unfair_trajectory_indices`, `famail_temporal.utils.trajectory.Trajectory`, `famail_temporal.data.loader.DataBundle`, and the editing run's `histories.pkl` artifact.

---

## Scope: FAMAIL + B2 model-level only (B1 deferred to Phase 5)

Per the design decision (confirmed 2026-05-28), this plan builds the **headline** (B0 vs FAMAIL) plus **B2** (generate-then-filter at several retention levels). It does **not** include:

- **B1 differentiable fairness loss** — deferred to **Phase 5**. The reuse seam is documented in the Phase-3 plan (`FAMAILObjective.forward(soft_pickup_3d)` + a differentiable soft-terminal-cell scatter modeled on `famail_temporal/algorithm/soft_cell_assignment.py::inject_soft_counts_into_3d`).
- **Pure-GAN ablation** and **multi-seed paired scale-up** — Phase 5.
- **Eval-time Siamese realism critic / JS-divergence utility** (spec §5) — Phase 5.

### Design decisions (flagged — veto if you disagree)

1. **FAMAIL edited data is LOADED from a persisted editing run, not re-edited live.** A `--edit-dir` flag points at a results directory; the loader reads `<edit_dir>/histories.pkl` and swaps each `ModificationHistory.modified` trajectory into `bundle.trajectories` by `trajectory_id` — exactly how `famail_temporal/evaluation/runner.py` reconstructs `trajs_after`. Default: `famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup` (the validated no-dedup k=10000 run: 3,773 edits, ΔF_causal=+0.0128, ΔF_spatial=+0.0003). Rationale: reuses the exact validated edit, is fast and deterministic, and the flag lets you point at future edits without code changes.
2. **B0 and FAMAIL are paired by construction.** Editing only relocates the *terminal pickup* cell; it never changes a trajectory's *start* cell or time-block. So `trajectory_context` is identical between the raw and edited corpora, and with a shared `seed` (same init) B0 and FAMAIL generate from the identical context set — the rollout-fairness difference is pure learned-behavior difference. No separate paired-context plumbing is needed.
3. **`fit_and_evaluate` gains one keyword-only param `train_trajectories` (default `None` → `bundle.trajectories`).** Training sequences AND generation contexts both derive from `train_trajectories`; fairness is always scored on `bundle` (the raw reference) and the generated grid. This is backward-compatible — B0 is `fit_and_evaluate(bundle)` unchanged.
4. **Edited terminal cells are float-valued** (the perturbation is continuous within ε=2, e.g. `(13.2, 33.0)`). The existing `trajectory_to_tokens` → `flat_cell` already `int()`-truncates coordinates to a grid cell, so edited trajectories tokenize correctly with **no special handling** — the same path raw trajectories use.
5. **B2 generation reflects the data loss.** A filtered variant trains on fewer trajectories and therefore generates from fewer contexts (the retained ones) → a lower-mass demand grid. That reduced coverage *is* the model-level cost of filtering we want to measure; we do not back-fill the removed contexts.

### What this phase reuses unchanged

`fit_and_evaluate`'s internals (MLE + adversarial + rollout + grid + fairness), `data_level_fairness`, `rank_unfair_trajectory_indices`, the generator/critic/gumbel modules. The trajectory-editing algorithm is untouched (we only *read* its persisted output). ε=2 is not involved here.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `famail_temporal/baselines/gan/variants.py` | `apply_edits`, `load_edited_trajectories`, `filtered_trajectories` — build training-corpus variants | Create |
| `famail_temporal/baselines/gan/model_level.py` | + `train_trajectories` keyword param on `fit_and_evaluate` | Modify |
| `famail_temporal/baselines/gan/model_suite.py` | `run_suite` — B0 + FAMAIL + B2-levels with a shared seed | Create |
| `famail_temporal/baselines/gan/run_model_suite.py` | CLI: `--edit-dir`, `--b2-remove`, epochs, device, seed → JSON | Create |
| `famail_temporal/baselines/gan/tests/test_variants.py` | edit-swap + filter mechanics | Create |
| `famail_temporal/baselines/gan/tests/test_model_level_variants.py` | `fit_and_evaluate(train_trajectories=...)` | Create |
| `famail_temporal/baselines/gan/tests/test_model_suite.py` | end-to-end suite on a tiny bundle + a written histories.pkl | Create |
| `famail_temporal/baselines/gan/tests/test_run_model_suite.py` | result serialization | Create |

---

## Task 1: `variants.py` — edit-swap + filter builders

**Files:**
- Create: `famail_temporal/baselines/gan/variants.py`
- Test: `famail_temporal/baselines/gan/tests/test_variants.py`

- [ ] **Step 1: Write the failing tests**

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
    assert [t.trajectory_id for t in out] == [10, 11, 12]   # order/length preserved
    assert out[1] is edited_11                               # the matching id swapped
    assert out[0] is raw[0] and out[2] is raw[2]             # others untouched


def test_load_edited_trajectories_reads_histories_pkl(tmp_path):
    bundle = _make_synthetic_bundle()
    bundle.trajectories.extend([
        make_traj_at(2, 2, 0, traj_id=100),
        make_traj_at(3, 3, 0, traj_id=101),
    ])
    # A persisted edit: trajectory 100's pickup relocated. ModificationHistory
    # is duck-typed here as an object exposing `.modified` (a Trajectory).
    edited_100 = make_traj_at(4, 4, 0, traj_id=100)
    histories = [SimpleNamespace(modified=edited_100)]
    (tmp_path / "histories.pkl").write_bytes(pickle.dumps(histories))

    out = variants.load_edited_trajectories(bundle, tmp_path)
    assert len(out) == len(bundle.trajectories)
    by_id = {t.trajectory_id: t for t in out}
    assert by_id[100] is edited_100        # swapped
    assert by_id[101] in bundle.trajectories  # untouched


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

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_variants.py -v`
Expected: FAIL (ModuleNotFoundError on `gan.variants`).

- [ ] **Step 3: Implement `variants.py`**

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

    Mirrors the editing runner's own trajs_after reconstruction: an entry is
    replaced iff its trajectory_id appears in modified_by_tid.
    """
    return [modified_by_tid.get(t.trajectory_id, t) for t in trajectories]


def load_edited_trajectories(
    bundle: DataBundle, edit_dir: Union[str, Path],
) -> List[Trajectory]:
    """Build the FAMAIL edited corpus from a persisted editing run.

    Reads <edit_dir>/histories.pkl (a list whose elements expose a `.modified`
    Trajectory carrying the relocated pickup and its trajectory_id) and swaps
    those into bundle.trajectories. Returns a list the same length/order as
    bundle.trajectories. Editing relocates only the terminal pickup cell, so the
    edited trajectories share their start cell / time-block with the originals.
    """
    with open(Path(edit_dir) / "histories.pkl", "rb") as f:
        histories = pickle.load(f)
    modified_by_tid = {h.modified.trajectory_id: h.modified for h in histories}
    return apply_edits(bundle.trajectories, modified_by_tid)


def filtered_trajectories(bundle: DataBundle, n_remove: int) -> List[Trajectory]:
    """bundle.trajectories with the top-`n_remove` most-unfair removed.

    Ranking via rank_unfair_trajectory_indices (positionally-aligned indices,
    most-unfair first; only strictly-negative-attribution trajectories are
    rankable, so the number actually removed is min(n_remove, len(ranked))).
    n_remove <= 0 returns the full corpus.
    """
    if n_remove <= 0:
        return list(bundle.trajectories)
    removed = set(rank_unfair_trajectory_indices(bundle)[:n_remove])
    return [t for i, t in enumerate(bundle.trajectories) if i not in removed]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_variants.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/variants.py famail_temporal/baselines/gan/tests/test_variants.py
git commit -m "feat(baselines/gan): FAMAIL/B2 training-corpus variant builders"
```

---

## Task 2: Generalize `fit_and_evaluate` with `train_trajectories`

**Files:**
- Modify: `famail_temporal/baselines/gan/model_level.py`
- Test: `famail_temporal/baselines/gan/tests/test_model_level_variants.py`

`fit_and_evaluate` already builds `sequences`/`contexts` from `bundle.trajectories` (as `pairs`, then applies the `max_tokens` length filter from the Phase-3 OOM hotfix) and exposes `mle_batch_size`/`adv_batch_size`/`max_tokens`. This task adds a keyword-only `train_trajectories` (default `None` → `bundle.trajectories`) so FAMAIL/B2 train on a variant corpus while fairness is still scored on `bundle`. The existing `max_tokens` filter then applies to the *variant* corpus, so all baselines exclude the same long-tail trajectories (editing preserves length, so B0/FAMAIL exclude identical ids — pairing intact).

- [ ] **Step 1: Write the failing test**

Create `famail_temporal/baselines/gan/tests/test_model_level_variants.py`:

```python
"""fit_and_evaluate trains on a provided train_trajectories list."""
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
        mle_epochs=2, adv_epochs=2, max_len=8,
        device=torch.device("cpu"), seed=0,
    )
    # Generation is one rollout per training context.
    assert out["n_generated"] == len(subset)
    # Fairness is still scored on the full bundle (the raw reference).
    assert set(out["corpus"]) == {"f_spatial", "f_causal", "gini_dsr", "gini_asr"}


def test_default_train_trajectories_is_full_corpus():
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 12)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    out = model_level.fit_and_evaluate(
        bundle, mle_epochs=2, adv_epochs=2, max_len=8,
        device=torch.device("cpu"), seed=0,
    )
    assert out["n_generated"] == len(bundle.trajectories)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_model_level_variants.py -v`
Expected: FAIL — `test_train_trajectories_controls_generation_count` errors with `TypeError: fit_and_evaluate() got an unexpected keyword argument 'train_trajectories'`.

- [ ] **Step 3: Add the `train_trajectories` parameter**

In `famail_temporal/baselines/gan/model_level.py`, add `train_trajectories` as the first keyword-only parameter (everything else in the signature stays). Change the signature from:

```python
def fit_and_evaluate(
    bundle: DataBundle, *,
    mle_epochs: int = gc.MLE_EPOCHS,
    adv_epochs: int = gc.ADV_EPOCHS,
    max_len: int = gc.MAX_GEN_LEN,
    mle_batch_size: int = gc.MLE_BATCH_SIZE,
    adv_batch_size: int = gc.ADV_BATCH_SIZE,
    max_tokens: int | None = gc.MAX_TRAIN_TOKENS,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict:
```

to (add only the `train_trajectories` line):

```python
def fit_and_evaluate(
    bundle: DataBundle, *,
    train_trajectories: list | None = None,
    mle_epochs: int = gc.MLE_EPOCHS,
    adv_epochs: int = gc.ADV_EPOCHS,
    max_len: int = gc.MAX_GEN_LEN,
    mle_batch_size: int = gc.MLE_BATCH_SIZE,
    adv_batch_size: int = gc.ADV_BATCH_SIZE,
    max_tokens: int | None = gc.MAX_TRAIN_TOKENS,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict:
```

Extend the docstring's first paragraph to note the new behavior (keep the existing `max_tokens`/batch-size paragraph):

```python
    """Train (MLE + adversarial) on `train_trajectories` (default
    bundle.trajectories), generate one rollout per training context, and return
    generated-vs-corpus fairness + loss histories. Fairness is always scored on
    `bundle` (the raw reference) and the generated grid, so variants (edited /
    filtered) reuse the same bundle and change only the training corpus.
    ...
    """
```

Then change the empty-corpus guard and the `pairs` source. Replace:

```python
    if not bundle.trajectories:
        raise ValueError(
            "fit_and_evaluate requires a non-empty corpus (bundle.trajectories)"
        )
    set_all_seeds(seed)

    pairs = [
        (trajectory_to_tokens(t), trajectory_context(t))
        for t in bundle.trajectories
    ]
```

with:

```python
    train_trajectories = (
        bundle.trajectories if train_trajectories is None else train_trajectories
    )
    if not train_trajectories:
        raise ValueError("fit_and_evaluate requires a non-empty training corpus")
    set_all_seeds(seed)

    pairs = [
        (trajectory_to_tokens(t), trajectory_context(t))
        for t in train_trajectories
    ]
```

Leave everything else — the `max_tokens` filter on `pairs`, training (which already uses `mle_batch_size`/`adv_batch_size`), rollout, grid, and the return dict with `data_level_fairness(bundle, ...)` — unchanged.

- [ ] **Step 4: Run the new test AND the existing model_level test**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_model_level_variants.py famail_temporal/baselines/gan/tests/test_model_level.py -v`
Expected: PASS (new param behavior + the Phase-3 end-to-end test still green — proves B0 is unchanged).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/model_level.py famail_temporal/baselines/gan/tests/test_model_level_variants.py
git commit -m "feat(baselines/gan): fit_and_evaluate accepts an explicit train_trajectories corpus"
```

---

## Task 3: `model_suite.py` — B0 / FAMAIL / B2 suite

**Files:**
- Create: `famail_temporal/baselines/gan/model_suite.py`
- Test: `famail_temporal/baselines/gan/tests/test_model_suite.py`

- [ ] **Step 1: Write the failing test**

Create `famail_temporal/baselines/gan/tests/test_model_suite.py`:

```python
"""End-to-end model-level suite (B0 + FAMAIL + B2) on a tiny bundle."""
import pickle
from types import SimpleNamespace

import torch

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import active_units, make_traj_at
from famail_temporal.baselines.gan import model_suite


def _bundle_with_trajs(n=20):
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, n)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    return bundle


def test_run_suite_returns_b0_famail_and_b2(tmp_path):
    bundle = _bundle_with_trajs(20)
    # Persisted edit: relocate the pickup of the first two trajectories.
    histories = [
        SimpleNamespace(modified=make_traj_at(4, 4, 0, traj_id=t.trajectory_id))
        for t in bundle.trajectories[:2]
    ]
    (tmp_path / "histories.pkl").write_bytes(pickle.dumps(histories))

    out = model_suite.run_suite(
        bundle, edit_dir=tmp_path, b2_remove_levels=[1, 2],
        mle_epochs=2, adv_epochs=2, max_len=8,
        device=torch.device("cpu"), seed=0,
    )
    assert set(out) == {"B0", "FAMAIL", "B2", "edit_dir"}
    for variant in (out["B0"], out["FAMAIL"]):
        assert set(variant["generated"]) == {"f_spatial", "f_causal", "gini_dsr", "gini_asr"}
    # FAMAIL trains on the full (edited) corpus -> same generation count as B0.
    assert out["FAMAIL"]["n_generated"] == out["B0"]["n_generated"]
    # B2 keyed by removal level; each removes that many -> fewer generations.
    assert set(out["B2"]) == {1, 2}
    assert out["B2"][2]["n_generated"] == out["B0"]["n_generated"] - 2
    assert out["edit_dir"] == str(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_model_suite.py -v`
Expected: FAIL (ModuleNotFoundError on `gan.model_suite`).

- [ ] **Step 3: Implement `model_suite.py`**

Create `famail_temporal/baselines/gan/model_suite.py`:

```python
"""Model-level baseline suite: B0 (raw) vs FAMAIL (edited) vs B2 (filtered).

All variants share one generator architecture and one seed (paired init), and
are scored on the same DataBundle. B0 and FAMAIL are paired by construction:
editing relocates only the terminal pickup, so both train/generate from the
identical (start cell, time-block) context set, making the rollout-fairness
difference a pure learned-behavior difference.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Union

import torch

from famail_temporal.data.loader import DataBundle
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.model_level import fit_and_evaluate
from famail_temporal.baselines.gan.variants import (
    load_edited_trajectories, filtered_trajectories,
)


def run_suite(
    bundle: DataBundle, *,
    edit_dir: Union[str, Path],
    b2_remove_levels: List[int],
    mle_epochs: int = gc.MLE_EPOCHS,
    adv_epochs: int = gc.ADV_EPOCHS,
    max_len: int = gc.MAX_GEN_LEN,
    mle_batch_size: int = gc.MLE_BATCH_SIZE,
    adv_batch_size: int = gc.ADV_BATCH_SIZE,
    max_tokens: int | None = gc.MAX_TRAIN_TOKENS,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict:
    """Run B0 / FAMAIL / B2-levels with a shared seed; return their results."""
    common = dict(
        mle_epochs=mle_epochs, adv_epochs=adv_epochs, max_len=max_len,
        mle_batch_size=mle_batch_size, adv_batch_size=adv_batch_size,
        max_tokens=max_tokens, device=device, seed=seed,
    )

    b0 = fit_and_evaluate(bundle, **common)

    edited = load_edited_trajectories(bundle, edit_dir)
    famail = fit_and_evaluate(bundle, train_trajectories=edited, **common)

    b2 = {}
    for n_remove in b2_remove_levels:
        kept = filtered_trajectories(bundle, n_remove)
        b2[n_remove] = fit_and_evaluate(bundle, train_trajectories=kept, **common)

    return {"B0": b0, "FAMAIL": famail, "B2": b2, "edit_dir": str(edit_dir)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_model_suite.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full gan suite**

Run: `python -m pytest famail_temporal/baselines/gan/ -q`
Expected: PASS (Phase 2 + Phase 3 + Phase 4).

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/gan/model_suite.py famail_temporal/baselines/gan/tests/test_model_suite.py
git commit -m "feat(baselines/gan): B0/FAMAIL/B2 model-level suite orchestrator"
```

---

## Task 4: `run_model_suite.py` — CLI + real-data smoke

**Files:**
- Create: `famail_temporal/baselines/gan/run_model_suite.py`
- Test: `famail_temporal/baselines/gan/tests/test_run_model_suite.py`

- [ ] **Step 1: Write the failing test**

Create `famail_temporal/baselines/gan/tests/test_run_model_suite.py`:

```python
"""Unit test for run_model_suite result serialization."""
import json

from famail_temporal.baselines.gan import run_model_suite as r


def test_result_to_json_roundtrips():
    f = {"f_spatial": 0.08, "f_causal": 0.81, "gini_dsr": 0.9, "gini_asr": 0.9}
    result = {
        "B0": {"generated": f, "corpus": f, "n_generated": 105401,
               "mle_losses": [3.1], "adv_losses": {"g_losses": [0.7], "d_losses": [1.3]}},
        "FAMAIL": {"generated": f, "corpus": f, "n_generated": 105401,
                   "mle_losses": [3.0], "adv_losses": {"g_losses": [0.7], "d_losses": [1.3]}},
        "B2": {1000: {"generated": f, "corpus": f, "n_generated": 104401,
                      "mle_losses": [3.0], "adv_losses": {"g_losses": [0.7], "d_losses": [1.3]}}},
        "edit_dir": "famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup",
    }
    blob = r.result_to_json(result)
    loaded = json.loads(blob)
    assert loaded["FAMAIL"]["generated"]["f_causal"] == 0.81
    # JSON object keys are strings: the B2 level key round-trips as "1000".
    assert "1000" in loaded["B2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_run_model_suite.py -v`
Expected: FAIL (ModuleNotFoundError on `gan.run_model_suite`).

- [ ] **Step 3: Implement `run_model_suite.py`**

Create `famail_temporal/baselines/gan/run_model_suite.py`:

```python
"""CLI: run the model-level baseline suite (B0 vs FAMAIL vs B2) on the real
corpus and report each variant's generated-vs-corpus fairness.

Example:
    python -m famail_temporal.baselines.gan.run_model_suite \
        --mle-epochs 5 --adv-epochs 3 \
        --edit-dir famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup \
        --b2-remove 1000 5000 20000 --device auto
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.model_suite import run_suite

DEFAULT_EDIT_DIR = (
    Path(config.PACKAGE_ROOT) / "results"
    / "2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup"
)


def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2)


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.baselines.gan.run_model_suite",
    )
    ap.add_argument("--mle-epochs", type=int, default=gc.MLE_EPOCHS)
    ap.add_argument("--adv-epochs", type=int, default=gc.ADV_EPOCHS)
    ap.add_argument("--max-len", type=int, default=gc.MAX_GEN_LEN)
    ap.add_argument("--mle-batch-size", type=int, default=gc.MLE_BATCH_SIZE)
    ap.add_argument("--adv-batch-size", type=int, default=gc.ADV_BATCH_SIZE)
    ap.add_argument("--max-tokens", type=int, default=gc.MAX_TRAIN_TOKENS,
                    help="exclude trajectories longer than this from training "
                         "(<=0 disables the filter)")
    ap.add_argument("--edit-dir", type=Path, default=DEFAULT_EDIT_DIR,
                    help="results dir with histories.pkl for the FAMAIL edit")
    ap.add_argument("--b2-remove", type=int, nargs="*",
                    default=[1000, 5000, 20000],
                    help="B2 removal levels (top-K most-unfair trajectories)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(config.PACKAGE_ROOT) / "results" / "model_suite")
    args = ap.parse_args(argv)

    bundle = DataBundle.load()
    result = run_suite(
        bundle, edit_dir=args.edit_dir, b2_remove_levels=args.b2_remove,
        mle_epochs=args.mle_epochs, adv_epochs=args.adv_epochs,
        max_len=args.max_len, mle_batch_size=args.mle_batch_size,
        adv_batch_size=args.adv_batch_size,
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
        device=_resolve_device(args.device), seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "model_suite_fairness.json").write_text(result_to_json(result))
    print(f"B0     generated F_causal={result['B0']['generated']['f_causal']:.4f}")
    print(f"FAMAIL generated F_causal={result['FAMAIL']['generated']['f_causal']:.4f}")
    for level, res in result["B2"].items():
        print(f"B2(-{level}) generated F_causal={res['generated']['f_causal']:.4f}"
              f"  (n_gen={res['n_generated']})")
    print(f"wrote {args.out_dir / 'model_suite_fairness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_run_model_suite.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/run_model_suite.py famail_temporal/baselines/gan/tests/test_run_model_suite.py
git commit -m "feat(baselines/gan): model-level suite CLI (B0/FAMAIL/B2)"
```

- [ ] **Step 6: Real-data smoke (manual; needs cache + GPU; long-running)**

Run: `python -m famail_temporal.baselines.gan.run_model_suite --mle-epochs 5 --adv-epochs 3 --device auto`
Expected: writes `famail_temporal/results/model_suite/model_suite_fairness.json`. **Interpretation:** the headline is `FAMAIL.generated.f_causal > B0.generated.f_causal` (edited-data model fairer at full retention). B2 levels show how much data must be discarded to match FAMAIL's fairness. If FAMAIL ≈ B0 (no model-level transfer), that's a finding to record and discuss — **not** something to patch by changing fairness/editing code (the data-level edit is +0.0128; a washed-out model-level signal is itself a result). Per the standing protocol, flag any surprise rather than "fixing" it. Note this CLI runs `fit_and_evaluate` once per variant (B0 + FAMAIL + each B2 level), so wall-clock ≈ (2 + #b2-levels) × a single adversarial-B0 run — start with a single `--b2-remove` level to bound cost, and consider the batched-generation optimization noted in the Phase-3 wrap-up.

---

## Self-Review

**1. Spec coverage (FAMAIL + B2 model-level):**
- FAMAIL: train shared generator on edited corpus, score rollout fairness vs raw B0 (spec §4.3 FAMAIL row; headline claim §1) — Tasks 1, 2, 3. ✓
- B2: generate-then-filter at several retention levels (spec §4.3 B2 row; §5 Pareto) — Tasks 1, 3. ✓
- Single shared architecture / vary-the-data, paired seed (spec §2 decision 1, §3.1) — Task 3 (shared `fit_and_evaluate` + one seed; B0/FAMAIL context-paired by construction). ✓
- Edited-data source configurable (user requirement 2026-05-28) — Task 4 `--edit-dir`. ✓
- **Deferred (stated up front):** B1 fairness loss, pure-GAN ablation, multi-seed scale-up, eval-time Siamese realism critic / JS-divergence utility — Phase 5.

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step is complete; every test step has assertions + an exact command + expected outcome.

**3. Type consistency:**
- `apply_edits(trajectories, modified_by_tid) -> List[Trajectory]`; `load_edited_trajectories(bundle, edit_dir) -> List[Trajectory]` (calls `apply_edits`); `filtered_trajectories(bundle, n_remove) -> List[Trajectory]`. ✓
- `fit_and_evaluate(bundle, *, train_trajectories=None, mle_epochs, adv_epochs, max_len, device, seed)` — additive keyword; B0 call sites unchanged. Returns the same 5-key dict. ✓
- `run_suite(bundle, *, edit_dir, b2_remove_levels, mle_epochs, adv_epochs, max_len, device, seed) -> {"B0", "FAMAIL", "B2": {level: ...}, "edit_dir"}`. Consumed by `run_model_suite` + asserted in `test_model_suite`. ✓
- `rank_unfair_trajectory_indices(bundle) -> List[int]` (Phase 1, positionally aligned) reused by `filtered_trajectories`. ✓

**4. Ambiguity:** Edit source = persisted `histories.pkl` swapped by `trajectory_id` (decisions 1, 2). Float edited coords tokenize via the existing `int()`-truncating `flat_cell` (decision 4). B2's reduced generation count is intentional (decision 5). B0/FAMAIL paired by construction (decision 2). All explicit.

**5. Standing-constraint check:** No change to the trajectory-editing algorithm — Phase 4 only *reads* a persisted edit (`histories.pkl`) and subsets/swaps trajectory lists. `fit_and_evaluate`'s change is an additive, backward-compatible keyword (verified by re-running the Phase-3 `test_model_level`). ε=2 is not involved. No `git add -A` — every commit stages named files only.

---
