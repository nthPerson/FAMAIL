# Level-2 Usability: Fairness Transfer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the edited data's fairness advantage survives downstream behavior-cloning training — train a driver-conditioned BC policy on each of four matched, full-corpus data sources (raw, FAM-AIL edited, BC-generated, GAN-generated) across paired seeds, then evaluate each trained policy's generated demand on the Level-1 axes and report paired fairness-transfer statistics.

**Architecture:** A new orchestrator `run_level2_table.py` builds four content-matched training datasets over the full corpus (raw = `bundle.trajectories`; edited = the corpus with the 3,773 modified trajectories swapped in by `trajectory_id`; BC-gen / GAN-gen = one driver-conditioned rollout per real seed from the Level-1 generators), trains a fresh driver-conditioned `TrajectoryLSTM` per (source, seed) with paired seeds, evaluates each policy by **reusing the Level-1 v2 scoring helpers** (identity Fidelity-A + gate, enriched Fidelity-B, `data_level_fairness`), and computes paired per-seed differences (Wilcoxon). `train_mle` gains an optional token-budgeted batching path so the full corpus (incl. ~0.7% long outliers) trains without OOM.

**Tech Stack:** Python 3, PyTorch, NumPy, SciPy (Wilcoxon; fall back to a paired-sign summary if SciPy is absent), pytest. Reuses `famail_temporal.baselines.{gan,fidelity_eval,metrics}`, `run_level1_table_v2` helpers, and `famail_temporal.fidelity` (frozen HuMID, read-only).

## Global Constraints

- **Branch:** `level-2-usability` (already created off `main`). Stage **named files only** — never `git add -A`/`.`. Do not commit anything under any `results/` directory (gitignored run artifacts).
- **Frozen / read-only:** no edits to `famail_temporal/algorithm/`, `famail_temporal/fairness/`, `famail_temporal/fidelity/`. HuMID is forward-only under `torch.no_grad()`; use `model.train(False)` (the literal `eval` immediately followed by `(` is blocked by a repo security hook — never write it).
- **Do NOT modify** `run_level1_table_v2.py`, `run_level1_table.py`, or `fidelity_eval.py` (they are merged on `main` and reused **by import**). The only shared file this plan modifies is `train_mle.py` (purely additive, backward-compatible).
- **Backward compatibility:** `train_mle`'s new `max_batch_tokens` defaults to `None` → the existing fixed-`batch_size` path, numerically identical (regression-tested). Level-1 and the variance suite must be unaffected.
- **Full corpus:** all training and evaluation use the entire corpus (105,401 trajectories) — no `max_tokens` filtering. `max_batch_tokens` bounds per-batch memory.
- **Paired seeds:** for a given seed `s`, `set_all_seeds(s)` is called immediately before constructing AND training each arm's policy, so all four arms share weight-init and minibatch ordering for that seed; arms differ only in training data.
- **HuMID forward** (verified): `forward(x1, x2, mask1, mask2, *, profile_1, profile_2)` accepts 4D `[B,N,L,4]` seeking + `[B,11]` profiles; driving omitted → symmetric zero default. `disc = load_discriminator(ckpt).to(device)`.
- **Verified data facts:** `Trajectory` has `.trajectory_id` (corpus load index) and `.driver_id` (int 0–49). `histories.pkl` = 3,773 `History(original, modified)` with matching `trajectory_id`/`driver_id`; every `.original` matches the corpus-by-id content; all 3,773 `.modified` differ. Corpus token lengths: median 9, p99 213, max 1654; 763 (0.72%) exceed 256 tokens.
- **Config constants:** `gc.MLE_BATCH_SIZE=32`, `gc.MAX_TRAIN_TOKENS=256`, `gc.MLE_LR`, `gc.MLE_EPOCHS`, `gc.MAX_GEN_LEN=64`, `gc.GEN_BATCH_SIZE=512`, `gc.BOS/EOS/PAD/GY/N_CELLS`. Token budget default = `MLE_BATCH_SIZE * MAX_TRAIN_TOKENS = 8192` (caps per-batch padded logits at the Level-1 worst-case-per-batch).

---

## Reused Level-1 v2 module-level helpers (import, do not duplicate)

From `famail_temporal.baselines.run_level1_table_v2`:
- `_select_eval_drivers(groups, *, min_trajs, max_drivers) -> List[int]`
- `_real_context_tensors(real_trajs) -> List[Tensor]`
- `_build_source_pairs(*, real_slot0, source_slot0, source_slot0_other, real_context, source_context_other, profile_d, profile_dp, rng) -> (matched, mismatched)`
- `_train_and_generate_cond(train_trajectories, driver_to_idx, *, adv_epochs, gan_loss, n_critic, mle_epochs, max_len, max_tokens, device, seed) -> {model, filtered_train, contexts, driver_idxs, mle_curve, adv_curve}`
- `_gen_cond_slot0(model, real_d, driver_idx, *, pairs_per_driver, max_len, device, gen_batch_size) -> List[Tensor]`
- `_gen_fidelity_full(model, filtered_train, contexts, driver_idxs, *, n, max_len, device, gen_batch_size) -> (pairs, gen_cells, n_empty)` (we use `gen_cells`)
- `_terminal_pickups_from_cells(gen_cells)`, `_terminal_pickups_from_trajs(trajs)`
- `_edited_fairness_from_metrics(edit_dir)`, `_curves_for_source(src)`

From `famail_temporal.baselines.fidelity_eval` (as `fe`): `humid_identity_fidelity`, `identity_validation_gate`, `build_identity_branch`, `trajectory_statistics`, `distributional_fidelity`, `stat_ranges`, `terminal_cell_distribution_js`, `real_to_disc_tensor`, `generated_to_disc_tensor`, `N_TRAJS_PER_BRANCH`, `_STAT_KEYS_V2`, `GATE_MARGIN`.

From `famail_temporal.baselines.gan.drivers`: `build_driver_index`, `group_by_driver`, `driver_idxs_for`. From `gan.sequences`: `trajectory_context`, `trajectory_to_tokens`, `flat_cell`. From `gan.rollout`: `generate_pickups`, `pickups_to_pickup_3d`. From `gan.train_mle`: `train_mle`. From `metrics`: `data_level_fairness`. From `gan.generator`: `TrajectoryLSTM`.

---

## File Structure

- **Modify** `famail_temporal/baselines/gan/train_mle.py` — optional `max_batch_tokens` token-budgeted batching (additive, `None` = current behavior).
- **Create** `famail_temporal/baselines/run_level2_table.py` — Level-2 orchestrator: matched-dataset builders, per-policy evaluation, paired-seed loop, paired statistics, render, persist, CLI.
- **Create** tests: `famail_temporal/baselines/tests/test_train_mle_token_budget.py`, `test_level2_datasets.py`, `test_level2_stats.py`, `test_run_level2_table.py`.
- **Create (manual phase)** `famail_temporal/baselines/LEVEL2_RESULTS.md`; update `docs/two_level_argument.md` Level-2 status.

---

### Task 1: Token-budgeted batching in `train_mle`

**Files:**
- Modify: `famail_temporal/baselines/gan/train_mle.py`
- Test: `famail_temporal/baselines/tests/test_train_mle_token_budget.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `train_mle(model, sequences, contexts, *, epochs, lr, batch_size, device, progress=False, driver_idxs=None, max_batch_tokens: int | None = None)`. When `max_batch_tokens` is set, minibatches are formed greedily from the (seeded) permutation so that `len(batch) * max_len_in_batch <= max_batch_tokens` (a single over-budget trajectory forms its own batch); `batch_size` remains an upper cap on count. When `None`, the existing fixed-`batch_size` slicing is used unchanged.

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/tests/test_train_mle_token_budget.py`:

```python
import torch

from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.train_mle import train_mle, _token_budget_batches

DEV = torch.device("cpu")


def test_token_budget_batches_splits_long_trajectory():
    # lengths: three short (2) + one long (10); budget 8 tokens, cap 32 count.
    perm = [0, 1, 2, 3]
    lengths = [2, 2, 2, 10]
    batches = list(_token_budget_batches(perm, lengths, batch_size=32, max_batch_tokens=8))
    # short ones group while count*maxlen <= 8: [0,1,2] -> 3*2=6 ok, adding 3 (len10) -> 4*10>8 stop
    assert batches[0] == [0, 1, 2]
    # the long one forms its own batch (10 > 8 alone is allowed)
    assert batches[1] == [3]


def test_token_budget_batches_respects_count_cap():
    perm = list(range(10))
    lengths = [1] * 10
    batches = list(_token_budget_batches(perm, lengths, batch_size=4, max_batch_tokens=10_000))
    assert all(len(b) <= 4 for b in batches)
    assert sum(len(b) for b in batches) == 10


def test_none_path_unchanged():
    """max_batch_tokens=None trains via the original fixed-batch path."""
    torch.manual_seed(0)
    m = TrajectoryLSTM().to(DEV)
    seqs = [[gc.BOS, 0, 1, gc.EOS], [gc.BOS, 2, 3, gc.EOS], [gc.BOS, 1, 0, gc.EOS]]
    ctx = [(0, 0), (2, 1), (1, 0)]
    out = train_mle(m, seqs, ctx, epochs=1, lr=1e-3, batch_size=2, device=DEV)
    assert len(out["epoch_losses"]) == 1


def test_budget_path_trains_full_corpus_shapes():
    torch.manual_seed(0)
    m = TrajectoryLSTM(n_drivers=2).to(DEV)
    seqs = [[gc.BOS, 0, 1, gc.EOS]] * 3 + [[gc.BOS] + list(range(8)) + [gc.EOS]]
    ctx = [(0, 0)] * 4
    out = train_mle(
        m, seqs, ctx, epochs=1, lr=1e-3, batch_size=32, device=DEV,
        driver_idxs=[0, 1, 0, 1], max_batch_tokens=8,
    )
    assert len(out["epoch_losses"]) == 1 and len(out["batch_losses"]) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_train_mle_token_budget.py -q`
Expected: FAIL (`_token_budget_batches` undefined; `max_batch_tokens` unexpected kwarg).

- [ ] **Step 3: Implement the token-budgeted batcher + wire it in**

In `famail_temporal/baselines/gan/train_mle.py`, add the helper above `train_mle`:

```python
def _token_budget_batches(perm, lengths, *, batch_size, max_batch_tokens):
    """Yield index batches from `perm` (a list of indices) so that, per batch,
    len(batch) * max(lengths in batch) <= max_batch_tokens, with at most
    batch_size indices per batch. A single trajectory longer than the budget
    forms its own batch (never dropped). `lengths[i]` is the token length of
    sequence i. Deterministic given `perm`.
    """
    i = 0
    n = len(perm)
    while i < n:
        batch = []
        cur_max = 0
        while i < n and len(batch) < batch_size:
            cand = perm[i]
            new_max = max(cur_max, lengths[cand])
            if batch and new_max * (len(batch) + 1) > max_batch_tokens:
                break
            batch.append(cand)
            cur_max = new_max
            i += 1
        yield batch
```

Then add the parameter to `train_mle` and branch the batch loop. Change the signature to include `max_batch_tokens: int | None = None`, precompute `lengths = [len(s) for s in sequences]`, and replace the inner `for start in range(0, n, batch_size):` loop with a unified iterator over batches of indices:

```python
    lengths = [len(s) for s in sequences]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        batch_losses: List[float] = []
        if max_batch_tokens is None:
            perm_list = perm.tolist()
            batches = [
                perm_list[start : start + batch_size]
                for start in range(0, n, batch_size)
            ]
        else:
            batches = list(_token_budget_batches(
                perm.tolist(), lengths,
                batch_size=batch_size, max_batch_tokens=max_batch_tokens,
            ))
        with Progress(len(batches), f"MLE epoch {epoch + 1}/{epochs}", enabled=progress) as bar:
            for idx in batches:
                batch = _pad_batch([sequences[i] for i in idx], device)
                ctx_cell = torch.tensor(
                    [contexts[i][0] for i in idx], dtype=torch.long, device=device,
                )
                ctx_tblock = torch.tensor(
                    [contexts[i][1] for i in idx], dtype=torch.long, device=device,
                )
                di = (
                    torch.tensor([driver_idxs[i] for i in idx], dtype=torch.long, device=device)
                    if driver_idxs is not None else None
                )
                inp = batch[:, :-1]
                tgt = batch[:, 1:]
                logits = model(inp, ctx_cell, ctx_tblock, driver_idx=di)
                loss = loss_fn(logits.reshape(-1, gc.VOCAB_SIZE), tgt.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                batch_losses.append(float(loss.item()))
                all_batch_losses.append(float(loss.item()))
                bar.update(1, loss=f"{sum(batch_losses) / len(batch_losses):.3f}")
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
```

Keep the rest of `train_mle` (docstring — add a `max_batch_tokens` note; `model.to(device).train()`; optimizer; return dict) unchanged. **Verify the `None` path is identical:** `perm_list[start:start+batch_size]` over `range(0, n, batch_size)` reproduces the original `perm[start:start+batch_size].tolist()` batches exactly (same perm tensor, same slices).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_train_mle_token_budget.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Regression — Level-1 + variance suite use train_mle**

Run: `python -m pytest famail_temporal/baselines/tests/ -q -k "smoke or level1 or variance or train_rollout or datasets"`
Expected: PASS (None-path unchanged).

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/gan/train_mle.py famail_temporal/baselines/tests/test_train_mle_token_budget.py
git commit -m "feat(baselines/gan): optional token-budgeted batching in train_mle (full-corpus memory guard)"
```
(append the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer)

---

### Task 2: Matched full-corpus training-set builders

**Files:**
- Create: `famail_temporal/baselines/run_level2_table.py` (start the module: header, imports, the builders)
- Test: `famail_temporal/baselines/tests/test_level2_datasets.py`

**Interfaces:**
- Consumes: `histories.pkl`, `bundle.trajectories`, the Level-1 generators, `trajectory_to_tokens`/`trajectory_context`, `flat_cell`, `driver_idxs_for`.
- Produces:
  - `build_edited_corpus(raw_trajs, histories) -> List[Trajectory]` — the full corpus with each modified trajectory swapped in by `trajectory_id`; same length/order as `raw_trajs`.
  - `traj_training_data(trajs, driver_to_idx) -> dict` → `{"sequences", "contexts", "driver_idxs", "trajs"}` (token sequences, contexts, embedding indices; all index-aligned).
  - `gen_training_data(model, raw_trajs, driver_to_idx, *, max_len, device, gen_batch_size) -> dict` — generate one driver-conditioned rollout per real seed; build token sequences `[BOS] + cells + [EOS]` (empty rollout → `[BOS, start_cell, EOS]` fallback, counted), with the real seed's context + driver. Returns the same dict shape plus `"n_empty"`.

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/tests/test_level2_datasets.py`:

```python
import torch

from famail_temporal.baselines import run_level2_table as r2
from famail_temporal.baselines.gan import config as gc


class _State:
    def __init__(self, x, y, t=10, d=1):
        self.x_grid, self.y_grid, self.time_bucket, self.day_index = x, y, t, d


class _Traj:
    def __init__(self, tid, did, cells):
        self.trajectory_id, self.driver_id = tid, did
        self.states = [_State(x, y) for (x, y) in cells]


class _Hist:
    def __init__(self, original, modified):
        self.original, self.modified = original, modified


def test_build_edited_corpus_swaps_by_id():
    raw = [_Traj(0, 0, [(0, 0), (1, 1)]), _Traj(1, 0, [(2, 2)]), _Traj(2, 1, [(3, 3)])]
    mod = _Traj(1, 0, [(9, 9)])
    histories = [_Hist(raw[1], mod)]
    edited = r2.build_edited_corpus(raw, histories)
    assert len(edited) == 3
    assert edited[0] is raw[0] and edited[2] is raw[2]      # unchanged kept
    assert edited[1] is mod                                  # modified swapped in
    assert [(s.x_grid, s.y_grid) for s in edited[1].states] == [(9, 9)]


def test_traj_training_data_aligned():
    raw = [_Traj(0, 5, [(0, 0), (1, 1)]), _Traj(1, 7, [(2, 2)])]
    d2i = {5: 0, 7: 1}
    out = r2.traj_training_data(raw, d2i)
    assert len(out["sequences"]) == len(out["contexts"]) == len(out["driver_idxs"]) == 2
    assert out["driver_idxs"] == [0, 1]
    assert out["sequences"][0][0] == gc.BOS and out["sequences"][0][-1] == gc.EOS


def test_gen_training_data_empty_fallback(monkeypatch):
    raw = [_Traj(0, 5, [(4, 4), (5, 5)]), _Traj(1, 7, [(6, 6)])]
    d2i = {5: 0, 7: 1}
    # stub generate_trajectories: first rollout empty, second non-empty
    monkeypatch.setattr(r2, "generate_trajectories", lambda *a, **k: [[], [12]])
    out = r2.gen_training_data(object(), raw, d2i, max_len=8, device=torch.device("cpu"),
                               gen_batch_size=4)
    assert len(out["sequences"]) == 2 and out["n_empty"] == 1
    # empty -> [BOS, start_cell, EOS]; start cell = flat_cell(4,4)
    from famail_temporal.baselines.gan.sequences import flat_cell
    assert out["sequences"][0] == [gc.BOS, flat_cell(4, 4), gc.EOS]
    assert out["sequences"][1] == [gc.BOS, 12, gc.EOS]
    assert out["driver_idxs"] == [0, 1]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest famail_temporal/baselines/tests/test_level2_datasets.py -q`
Expected: FAIL (`run_level2_table` missing).

- [ ] **Step 3: Implement the module header + builders**

Create `famail_temporal/baselines/run_level2_table.py` with the header/imports and the three builders:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_level2_datasets.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Import smoke**

Run: `python -c "import famail_temporal.baselines.run_level2_table"`
Expected: no error (all Level-1 v2 imports resolve).

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/run_level2_table.py famail_temporal/baselines/tests/test_level2_datasets.py
git commit -m "feat(baselines): Level-2 matched full-corpus training-set builders (edited swap + gen sets)"
```

---

### Task 3: Paired-difference statistics helper

**Files:**
- Modify: `famail_temporal/baselines/run_level2_table.py` (add `_paired_diff_stats`)
- Test: `famail_temporal/baselines/tests/test_level2_stats.py`

**Interfaces:**
- Produces: `_paired_diff_stats(per_seed: Dict[str, List[float]], *, baseline: str = "edited") -> Dict[str, dict]` — given per-source lists of a metric across seeds (index = seed), compute, for each other source `o`, the paired differences `baseline[s] - o[s]`, their `mean`, `std`, `n`, and a paired Wilcoxon p-value (`None` if SciPy unavailable or n < 1 or all-zero). Returns `{o: {"diffs", "mean", "std", "n", "wilcoxon_p"}}`.

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/tests/test_level2_stats.py`:

```python
from famail_temporal.baselines import run_level2_table as r2


def test_paired_diff_stats_basic():
    per_seed = {
        "edited": [0.82, 0.83, 0.81, 0.84, 0.82],
        "raw":    [0.80, 0.81, 0.80, 0.81, 0.80],
        "bcgen":  [0.80, 0.80, 0.81, 0.80, 0.81],
    }
    out = r2._paired_diff_stats(per_seed, baseline="edited")
    assert set(out) == {"raw", "bcgen"}
    raw = out["raw"]
    assert raw["n"] == 5
    assert abs(raw["mean"] - sum(e - r for e, r in
               zip(per_seed["edited"], per_seed["raw"])) / 5) < 1e-9
    assert len(raw["diffs"]) == 5
    # wilcoxon_p present (float) or None if scipy missing
    assert raw["wilcoxon_p"] is None or 0.0 <= raw["wilcoxon_p"] <= 1.0


def test_paired_diff_stats_handles_constant_and_missing_scipy():
    per_seed = {"edited": [0.5, 0.5], "raw": [0.5, 0.5]}
    out = r2._paired_diff_stats(per_seed, baseline="edited")
    assert out["raw"]["mean"] == 0.0
    assert out["raw"]["wilcoxon_p"] is None   # all-zero diffs -> no test
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest famail_temporal/baselines/tests/test_level2_stats.py -q`
Expected: FAIL (`_paired_diff_stats` undefined).

- [ ] **Step 3: Implement**

Append to `run_level2_table.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_level2_stats.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/run_level2_table.py famail_temporal/baselines/tests/test_level2_stats.py
git commit -m "feat(baselines): Level-2 paired-difference statistics helper (Wilcoxon)"
```

---

### Task 4: Per-policy evaluation, orchestrator, render, persistence

**Files:**
- Modify: `famail_temporal/baselines/run_level2_table.py` (add `_evaluate_policy`, `render_level2_table`, `result_to_json`, `main`)
- Test: `famail_temporal/baselines/tests/test_run_level2_table.py`

**Interfaces:**
- Produces:
  - `result_to_json(result) -> str`
  - `render_level2_table(result) -> str` — markdown: per-source mean ± std for {F_causal, F_spatial, Fidelity-A, Fidelity-B}, the gate line, and the paired headline/secondary block.
  - `_evaluate_policy(model, *, driver_idxs, contexts, filtered_train, bundle, eval_drivers, driver_to_idx, groups, profiles, zeros11, raw_stats, raw_pickups, disc, rng, fss, max_len, device, gen_batch_size) -> dict` → `{f_causal, f_spatial, fidelity_a, fidelity_a_separation, fidelity_b, fidelity_b_per_component, n_empty}` (the policy's own matched/mismatched Fidelity-A built two-pass over eval drivers exactly as Level-1 v2; gate is computed once outside).
  - `main(argv) -> int`.

**`_evaluate_policy` flow** (per trained policy, mirrors Level-1 v2's per-source scoring, all gen driver-conditioned via `driver_idxs`):
1. **Fairness:** `generate_pickups(model, contexts, max_len=max_len, device=device, gen_batch_size=gen_batch_size, driver_idxs=driver_idxs)` → `pickups_to_pickup_3d(bundle, pickups)` → `data_level_fairness(bundle, pickup_3d=grid)`.
2. **Fidelity-A:** for each eval driver `d`, `gen_slot0_by_d[d] = _gen_cond_slot0(model, groups[d], driver_to_idx[d], pairs_per_driver=ppd, max_len=max_len, device=device, gen_batch_size=gen_batch_size)`. Then two-pass pair (d vs next eval driver d′): `matched/mismatched += _build_source_pairs(real_slot0=real_slot0_by_d[d], source_slot0=gen_slot0_by_d[d], source_slot0_other=gen_slot0_by_d[dprime], real_context=real_context_by_d[d], source_context_other=real_context_by_d[dprime], profile_d=prof_by_d[d], profile_dp=prof_by_d[dprime], rng=rng)`. `fidelity_a = humid_identity_fidelity(disc, matched, device=device)["mean"]`; `separation = fidelity_a - humid_identity_fidelity(disc, mismatched, device=device)["mean"]`. (`real_slot0_by_d`, `real_context_by_d`, `prof_by_d` precomputed once in `main` and passed in.)
3. **Fidelity-B:** `_, gen_cells, n_empty = _gen_fidelity_full(model, filtered_train, contexts, driver_idxs, n=min(fss, len(contexts)), max_len=max_len, device=device, gen_batch_size=gen_batch_size)`; `src_stats = [fe.trajectory_statistics(c) for c in gen_cells if c]`; `per = fe.distributional_fidelity(src_stats, raw_stats, keys=fe._STAT_KEYS_V2)["per_stat"]` (ranges=None → pooled src+raw grid, a vs-raw guardrail consistent across policies); `tj = fe.terminal_cell_distribution_js(_terminal_pickups_from_cells(gen_cells), raw_pickups)`; `fidelity_b = mean(list(per.values()) + [tj])`; `per_component = {**per, "terminal_cell": tj}`.

**`main` flow:**
1. argparse: `--edit-dir` (default = the committed no-dedup causal-emphasis edit), `--seeds` (default `"0,1,2,3,4"`), `--mle-epochs` (20), `--max-eval-drivers` (50), `--pairs-per-driver` (20), `--min-driver-trajs` (6), `--fidelity-sample-size` (5000), `--gan-loss` (wgan-gp), `--adv-epochs` (3), `--n-critic` (5), `--gen-batch-size` (gc.GEN_BATCH_SIZE), `--max-batch-tokens` (default `gc.MLE_BATCH_SIZE * gc.MAX_TRAIN_TOKENS`), `--device` (auto), `--out-dir`, `--quiet`.
2. Device; checkpoint guard; `disc = load_discriminator(ckpt).to(device)`; `bundle = DataBundle.load()`; load `histories.pkl`.
3. `raw_trajs = bundle.trajectories`; `driver_to_idx = build_driver_index(raw_trajs)`; `groups = group_by_driver(raw_trajs)`; `profiles = bundle.multi_stream.profile_features`; `zeros11 = np.zeros(11, np.float32)`.
4. **Build the L1 generators once** (full corpus, `max_tokens=None`): `bc = _train_and_generate_cond(raw_trajs, driver_to_idx, adv_epochs=0, gan_loss="bce", n_critic=1, mle_epochs=args.mle_epochs, max_len=gc.MAX_GEN_LEN, max_tokens=None, device=device, seed=args.seeds[0])`; `gan = _train_and_generate_cond(raw_trajs, driver_to_idx, adv_epochs=args.adv_epochs, gan_loss=args.gan_loss, n_critic=args.n_critic, mle_epochs=args.mle_epochs, max_len=gc.MAX_GEN_LEN, max_tokens=None, device=device, seed=args.seeds[0])`. *(These produce the BC-gen/GAN-gen training data; `_train_and_generate_cond` already accepts `max_tokens` — pass `None` for full corpus.)*
5. **Build the four training datasets** (full corpus): `D_raw = traj_training_data(raw_trajs, driver_to_idx)`; `D_edited = traj_training_data(build_edited_corpus(raw_trajs, histories), driver_to_idx)`; `D_bcgen = gen_training_data(bc["model"], raw_trajs, driver_to_idx, max_len=gc.MAX_GEN_LEN, device=device, gen_batch_size=args.gen_batch_size)`; `D_gangen = gen_training_data(gan["model"], ...)`. Map `{"raw": D_raw, "edited": D_edited, "bcgen": D_bcgen, "gangen": D_gangen}`.
6. **Precompute evaluation fixtures once:** `eval_drivers = _select_eval_drivers(groups, min_trajs=args.min_driver_trajs, max_drivers=args.max_eval_drivers)`; per-d `real_slot0_by_d` (up to ppd real `real_to_disc_tensor`), `real_context_by_d` (`_real_context_tensors`), `prof_by_d` (`profiles.get(d, zeros11)`); `raw_stats = [fe.trajectory_statistics(t) for t in raw_trajs[:fss]]`; `raw_pickups = _terminal_pickups_from_trajs(raw_trajs[:fss])`.
7. **Gate (once):** build raw matched/mismatched from `real_slot0_by_d` (d vs d′, real-vs-real) and `identity_validation_gate(disc, matched_pairs=raw_matched, mismatched_pairs=raw_mismatched, device=device)`; `trusted = gate["passed"]`.
8. **Paired loop:** `per_seed_metric = {m: {src: [] for src in _SOURCE_ORDER} for m in ("f_causal","f_spatial","fidelity_a","fidelity_b")}`. For each `seed s` in `args.seeds`: for each `src` in `_SOURCE_ORDER`: `set_all_seeds(s)`; `model = TrajectoryLSTM(n_drivers=len(driver_to_idx)).to(device)`; `train_mle(model, D[src]["sequences"], D[src]["contexts"], epochs=args.mle_epochs, lr=gc.MLE_LR, batch_size=gc.MLE_BATCH_SIZE, device=device, driver_idxs=D[src]["driver_idxs"], max_batch_tokens=args.max_batch_tokens)`; `m = _evaluate_policy(model, driver_idxs=D[src]["driver_idxs"], contexts=D[src]["contexts"], filtered_train=D[src]["trajs"], ..., disc=disc, rng=rng, fss=fss, max_len=gc.MAX_GEN_LEN, device=device, gen_batch_size=args.gen_batch_size)`; append each metric; `del model; torch.cuda.empty_cache()` (evaluate-and-discard).
9. **Stats:** for each metric, `paired = _paired_diff_stats(per_seed_metric[metric], baseline="edited")`. Build `result`: `{edit_dir, seeds, gate, n_eval_drivers, trusted, per_source: {src: {metric: {mean, std, values}}}, paired: {metric: paired}}`. F_causal paired is the headline.
10. **Persist** (default out-dir `config.PACKAGE_ROOT/"results"/"level2_table"/<stamp>`): `level2_metrics.json` (`result_to_json`), `level2_table.md` (`render_level2_table`), `driver_index.json`. Summary log: the table + the headline paired `Δ(edited−raw) F_causal` mean±std + Wilcoxon p + the scale-to-10 note if the CI crosses zero.

- [ ] **Step 1: Write the failing test** (pure helpers + render only; the GPU pipeline is the manual phase)

Create `famail_temporal/baselines/tests/test_run_level2_table.py`:

```python
import json
from famail_temporal.baselines import run_level2_table as r2


def _result():
    return {
        "edit_dir": "x", "seeds": [0, 1],
        "gate": {"high_matched": 0.84, "low_mismatched": 0.17, "margin": 0.2,
                 "passed": True, "n_matched": 10, "n_mismatched": 10},
        "n_eval_drivers": 5, "trusted": True,
        "per_source": {
            s: {"f_causal": {"mean": 0.81, "std": 0.004, "values": [0.81, 0.81]},
                "f_spatial": {"mean": 0.08, "std": 0.001, "values": [0.08, 0.08]},
                "fidelity_a": {"mean": 0.84, "std": 0.003, "values": [0.84, 0.84]},
                "fidelity_b": {"mean": 0.05, "std": 0.002, "values": [0.05, 0.05]}}
            for s in ("raw", "edited", "bcgen", "gangen")
        },
        "paired": {"f_causal": {
            "raw": {"diffs": [0.01, 0.012], "mean": 0.011, "std": 0.0014, "n": 2, "wilcoxon_p": 0.5},
            "bcgen": {"diffs": [0.01, 0.01], "mean": 0.01, "std": 0.0, "n": 2, "wilcoxon_p": None},
            "gangen": {"diffs": [0.0, 0.0], "mean": 0.0, "std": 0.0, "n": 2, "wilcoxon_p": None},
        }},
    }


def test_render_level2_table_has_sources_gate_and_paired():
    md = r2.render_level2_table(_result())
    for s in ("raw", "edited", "bcgen", "gangen"):
        assert s in md
    assert "PASSED" in md
    assert "edited" in md and "raw" in md
    assert "0.011" in md or "+0.011" in md   # headline paired mean appears


def test_result_to_json_round_trips():
    assert json.loads(r2.result_to_json(_result()))["paired"]["f_causal"]["raw"]["mean"] == 0.011
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_level2_table.py -q`
Expected: FAIL (`render_level2_table` / `result_to_json` undefined).

- [ ] **Step 3: Implement `result_to_json`, `render_level2_table`, `_evaluate_policy`, `main`**

Add `result_to_json`:

```python
def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2, default=float)
```

Add `render_level2_table` (per-source mean±std + gate + paired headline/secondary):

```python
def render_level2_table(result: dict) -> str:
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
```

Then implement `_evaluate_policy` (per the documented flow above) and `main` (per the documented flow above), mirroring `run_level1_table_v2`'s structure for the precompute fixtures, the gate, and persistence. Use a single `rng = random.Random(seeds[0])` for pair sampling; `set_all_seeds(s)` immediately before each arm's model construction + training (the pairing guarantee). Evaluate-and-discard each policy (`del model`; `torch.cuda.empty_cache()` when CUDA).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_level2_table.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: `--help` + full-suite regression**

Run: `python -m famail_temporal.baselines.run_level2_table --help` (prints usage incl. `--seeds`, `--max-batch-tokens`); then `python -m pytest famail_temporal/baselines/tests/ -q`
Expected: usage prints cleanly; full suite PASSES.

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/run_level2_table.py famail_temporal/baselines/tests/test_run_level2_table.py
git commit -m "feat(baselines): Level-2 orchestrator (per-policy eval + paired fairness-transfer table)"
```

---

## Manual experiment + documentation phase (controller-executed)

After Tasks 1–4 pass review:

- [ ] **GPU smoke (cheap) BEFORE the long run.** Confirm full-corpus training does not OOM with the longest trajectories: run `--seeds 0,1 --mle-epochs 2 --adv-epochs 1 --max-eval-drivers 4 --pairs-per-driver 3 --fidelity-sample-size 200 --device cuda --out-dir ~/level2_smoke`. Verify a populated paired table + gate verdict + no OOM. (If OOM on the longest trajectory, lower `--max-batch-tokens`.)
- [ ] **Full run** (background `nohup`, log OUTSIDE `/tmp`, harness-tracked waiter): `--seeds 0,1,2,3,4 --mle-epochs 20 --adv-epochs 3 --gan-loss wgan-gp --n-critic 5 --device cuda`. Expect ~1–2 hr (20 BC trainings + the L1 generators + 20 evaluations).
- [ ] **Scale-to-10 trigger:** if the headline `Δ(edited−raw) F_causal` paired 95% CI crosses zero, rerun with `--seeds 0,1,...,9` before concluding.
- [ ] **Write `famail_temporal/baselines/LEVEL2_RESULTS.md`** — the paired table, the gate verdict, the headline fairness-transfer result (mean Δ ± std, Wilcoxon p, per-seed diffs), the fidelity guardrail check, the BC-gen self-distillation + GAN-gen collapse-propagation readings, and an honest transfer / no-transfer verdict (state the seed count + the proximity to the noise floor).
- [ ] **Update `docs/two_level_argument.md`** Level-2 section status → done + headline; update memory ([[famail-paper-argument]], a new level-2 pickup or extend [[level1-v2-pickup]]).
- [ ] **Final code review** (subagent) over the whole Level-2 diff; then `superpowers:finishing-a-development-branch` (PR + merge after user approval).

---

## Self-Review (against the spec)

**Spec coverage:** §3.1 driver-conditioned BC → Task 4 (`TrajectoryLSTM(n_drivers)` + `train_mle`). §3.2 matched-per-seed datasets → Task 2 + Task 4 step 5. §3.3 full corpus + token budget → Task 1 + `max_tokens=None` throughout. §3.4 5 paired seeds + scale trigger → Task 4 paired loop + manual phase. §3.5 fairness-transfer outcome, fidelity guardrail, hypothesis test → Task 3 + Task 4 (`_paired_diff_stats`, render). §6 reuse L1 v2 scoring → imports in Task 2/4. §7 statistics → Task 3. §10 edge cases (empty rollout fallback, raw/edited alignment, paired integrity, gate-may-fail, memory) → Tasks 1/2/4. §11 testing → one test file per task.

**Placeholder scan:** none — Tasks 1–3 have complete code; Task 4's `_evaluate_policy`/`main` are integration glue specified by the documented ordered flow + the imported L1 v2 helpers they mirror (intentionally not duplicated line-by-line to avoid drift from the merged L1 v2 code).

**Type consistency:** training datasets are `{sequences: List[List[int]], contexts: List[Tuple[int,int]], driver_idxs: List[int], trajs: List[Trajectory]}`; `_paired_diff_stats` consumes `Dict[str, List[float]]`; `_evaluate_policy` returns the per-source metric dict the render/stats read; `max_batch_tokens` is `int | None` everywhere; gate keys match `identity_validation_gate`.
