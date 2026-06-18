# Level-1 Data-Quality Table + Fidelity Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Level-1 data-quality table (4 data sources × {F_causal, F_spatial, Fidelity-A HuMID, Fidelity-B distributional}) that answers whether FAM-AIL's edited data is higher quality than raw / BC-generated / GAN-generated data, with the HuMID fidelity numbers guarded by a validation gate.

**Architecture:** Three new units under `famail_temporal/baselines/` — full-trajectory generation (`gan/rollout.py` addition), a discriminator-free + discriminator-based fidelity-evaluation module (`fidelity_eval.py`), and an orchestrator CLI (`run_level1_table.py`) — plus a paper-ready results doc. The HuMID discriminator (`famail_temporal/fidelity/`) is reused frozen and inference-only. No change to the editing algorithm, fairness formulas, or ε.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest. Reuses `data_level_fairness`, `pickups_to_pickup_3d`, `load_edited_trajectories`, `load_discriminator`, `transmission.jensen_shannon_divergence`, `sample_trajectory_cells`, and the `Progress` helper.

**Spec:** `docs/superpowers/specs/2026-06-17-level1-data-quality-table-design.md`.

## Global Constraints

- **Branch:** all work on `two-level-paper` (already current). Do NOT work on `main`.
- **Staging:** stage only the named files per task — never `git add -A` / `git add .` (untracked research-artifact dirs live under `baselines/`).
- **Security hook:** never write the literal token `eval` immediately followed by `(`. Use `model.train(False)` for inference mode; `load_discriminator()` already sets the discriminator to inference mode + frozen.
- **No editing-algorithm changes:** touch only `famail_temporal/baselines/` (+ its tests) and docs. Do NOT modify `famail_temporal/algorithm/`, `famail_temporal/fairness/`, or `famail_temporal/fidelity/` (reuse the last read-only).
- **Fidelity scoring is forward-only:** run discriminator calls under `torch.no_grad()`; the editing-time cuDNN-backward workaround is unnecessary here.
- **Coordinate convention:** pipeline coords are 0-indexed `[0-47, 0-89]`; the discriminator expects 1-indexed `[1-48, 1-90]` → add +1 to x,y in every discriminator tensor builder.
- **Flat-cell convention:** a generated token is a flat cell id; `x = cell // gc.GY`, `y = cell % gc.GY` (inverse of `flat_cell(x, y) = x * gc.GY + y`).
- **Fairness convention:** higher = fairer (F_causal, F_spatial). Fidelity-A: higher = more realistic. Fidelity-B: a divergence, **lower = more faithful**.
- **Edit source default:** `famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup` (the no-dedup k=10000 edit, ΔF_causal +0.0128).
- **Discriminator checkpoint:** `famail_temporal/discriminator_checkpoints/default/best.pt`.

> **Noted refinement of spec §3.4 (flag for reviewers):** the spec says generated per-step `time_bucket` = the generation context's time block. The generation context carries `t_block` (a coarse model-level block from `hour_to_block_index`), whereas the discriminator was trained on raw `time_bucket` (30-min index) — a units mismatch. To avoid feeding the discriminator an out-of-units value, the converter takes an explicit `time_bucket` and `day_index`, and the orchestrator supplies the **paired real seed trajectory's first-state `time_bucket` and `day_index`**. This is more realistic than fabricating from `t_block` and keeps the generated trajectory's temporal context in-distribution. Same intent as the spec (a fixed synthesized temporal context per generated trajectory); cleaner units.

---

## Task 1: Full-trajectory generation (`generate_trajectories`)

**Files:**
- Modify: `famail_temporal/baselines/gan/rollout.py`
- Test: `famail_temporal/baselines/gan/tests/test_generate_trajectories.py`

**Interfaces:**
- Consumes: `sample_trajectory_cells` (existing), `model.step` (`(prev[B], cc[B], tb[B], hidden) -> (logits[B,V], hidden)`), `gc.BOS`, `gc.EOS`, `gc.N_CELLS`, `Progress`.
- Produces: `generate_trajectories(model, contexts, *, max_len, device, gen_batch_size=512, temperature=1.0, progress=False) -> List[List[int]]` — one full cell-id sequence per context, **index-aligned with `contexts`**; specials stripped.

- [ ] **Step 1: Write the failing test**

Create `famail_temporal/baselines/gan/tests/test_generate_trajectories.py`:

```python
"""generate_trajectories: full cell-sequence capture, index-aligned with contexts."""
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.rollout import generate_trajectories


def test_generate_trajectories_one_per_context_indexed_and_clean():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    contexts = [(0, 0), (5, 1), (10, 0), (20, 1), (30, 0)]
    out = generate_trajectories(
        model, contexts, max_len=8, device=torch.device("cpu"),
        gen_batch_size=2,  # exercises multi-batch path
    )
    assert isinstance(out, list) and len(out) == len(contexts)
    for seq in out:
        assert isinstance(seq, list)
        assert len(seq) <= 8
        # only in-vocabulary cell ids; no BOS/EOS/PAD
        assert all(0 <= c < gc.N_CELLS for c in seq)


def test_generate_trajectories_empty_contexts():
    model = TrajectoryLSTM()
    out = generate_trajectories(
        model, [], max_len=8, device=torch.device("cpu"), gen_batch_size=4,
    )
    assert out == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_generate_trajectories.py -v`
Expected: FAIL (`ImportError: cannot import name 'generate_trajectories'`).

- [ ] **Step 3: Implement `generate_trajectories`**

In `famail_temporal/baselines/gan/rollout.py`, add (after `sample_terminal_cells_batched`, reusing the same imports already present — `torch`, `gc`, `Progress`, `TrajectoryLSTM`, `List`/`Tuple` from typing):

```python
def generate_trajectories(
    model: "TrajectoryLSTM",
    contexts: List[Tuple[int, int]],
    *,
    max_len: int,
    device: torch.device,
    gen_batch_size: int = 512,
    temperature: float = 1.0,
    progress: bool = False,
) -> List[List[int]]:
    """One FULL cell-id sequence per context, index-aligned with ``contexts``.

    Unlike ``generate_pickups`` (which keeps only the terminal cell), this
    retains the entire rollout so downstream fidelity scoring can compare full
    trajectories. Each context is ``(start flat-cell, start time-block)`` — the
    tuple produced by ``sequences.trajectory_context``. Specials (BOS/EOS/PAD)
    are stripped; only in-vocabulary cell ids (< N_CELLS) are kept. Batched
    autoregressive decode mirrors ``sample_terminal_cells_batched``.
    """
    model.to(device).train(False)
    results: List[List[int]] = []
    bar = Progress(len(contexts), "generating trajectories", enabled=progress)
    for start in range(0, len(contexts), gen_batch_size):
        chunk = contexts[start : start + gen_batch_size]
        b = len(chunk)
        cc = torch.tensor([c for c, _ in chunk], dtype=torch.long, device=device)
        tb = torch.tensor([t for _, t in chunk], dtype=torch.long, device=device)
        prev = torch.full((b,), gc.BOS, dtype=torch.long, device=device)
        hidden = None
        done = torch.zeros(b, dtype=torch.bool, device=device)
        seqs: List[List[int]] = [[] for _ in range(b)]
        with torch.no_grad():
            for _ in range(max_len):
                logits, hidden = model.step(prev, cc, tb, hidden)   # (b, V)
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(1)         # (b,)
                nxt_cpu = nxt.tolist()
                done_cpu = done.tolist()
                for i in range(b):
                    if done_cpu[i]:
                        continue
                    tok = nxt_cpu[i]
                    if tok == gc.EOS:
                        done[i] = True
                    elif tok < gc.N_CELLS:
                        seqs[i].append(tok)
                prev = nxt
                if bool(done.all()):
                    break
        results.extend(seqs)
        bar.update(b)
    bar.close()
    return results
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_generate_trajectories.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full GAN suite (no regression)**

Run: `python -m pytest famail_temporal/baselines/gan/tests/ -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/gan/rollout.py \
        famail_temporal/baselines/gan/tests/test_generate_trajectories.py
git commit -m "feat(baselines/gan): full-trajectory generation (generate_trajectories)"
```

---

## Task 2: Tensor builders + trajectory statistics (`fidelity_eval.py` part 1)

**Files:**
- Create: `famail_temporal/baselines/fidelity_eval.py`
- Test: `famail_temporal/baselines/tests/test_fidelity_eval_builders.py`

**Interfaces:**
- Consumes: `gc.GY` (grid width for un-flattening), `Trajectory` (duck-typed: `.states[i].x_grid/.y_grid/.time_bucket/.day_index`).
- Produces:
  - `real_to_disc_tensor(traj) -> torch.Tensor` shape `[L, 4]` float32, rows `(x+1, y+1, time_bucket, day_index)`.
  - `generated_to_disc_tensor(cells, time_bucket, day_index) -> torch.Tensor` shape `[L, 4]` float32 (cells are flat ids; un-flatten then +1).
  - `trajectory_statistics(traj_or_cells) -> dict{"length": int, "mean_displacement": float, "coverage": int}` (accepts a `Trajectory` or a list of flat cell ids).

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/tests/test_fidelity_eval_builders.py`:

```python
"""fidelity_eval: discriminator tensor builders + trajectory statistics."""
import math
from types import SimpleNamespace

import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines import fidelity_eval as fe


def _state(x, y, t, d):
    return SimpleNamespace(x_grid=float(x), y_grid=float(y), time_bucket=t, day_index=d)


def _traj(states):
    return SimpleNamespace(states=states, driver_id=0)


def test_real_to_disc_tensor_adds_one_to_coords():
    traj = _traj([_state(0, 0, 5, 2), _state(3, 7, 5, 2)])
    out = fe.real_to_disc_tensor(traj)
    assert out.shape == (2, 4)
    # +1 coord conversion; time/day preserved
    assert out[0].tolist() == [1.0, 1.0, 5.0, 2.0]
    assert out[1].tolist() == [4.0, 8.0, 5.0, 2.0]


def test_generated_to_disc_tensor_unflattens_and_adds_one():
    # flat cell c -> (c // GY, c % GY); then +1
    c0 = 0                      # (0, 0) -> (1, 1)
    c1 = 2 * gc.GY + 3         # (2, 3) -> (3, 4)
    out = fe.generated_to_disc_tensor([c0, c1], time_bucket=9, day_index=4)
    assert out.shape == (2, 4)
    assert out[0].tolist() == [1.0, 1.0, 9.0, 4.0]
    assert out[1].tolist() == [3.0, 4.0, 9.0, 4.0]


def test_trajectory_statistics_from_cells():
    # cells (0,0) -> (0,1) -> (0,3): length 3, coverage 3,
    # displacements: |(0,1)-(0,0)|=1, |(0,3)-(0,1)|=2 -> mean 1.5
    cells = [0, 1, 3]
    s = fe.trajectory_statistics(cells)
    assert s["length"] == 3
    assert s["coverage"] == 3
    assert math.isclose(s["mean_displacement"], 1.5, rel_tol=1e-9)


def test_trajectory_statistics_from_trajectory_and_short_len():
    traj = _traj([_state(2, 2, 0, 0)])           # single state
    s = fe.trajectory_statistics(traj)
    assert s["length"] == 1
    assert s["coverage"] == 1
    assert s["mean_displacement"] == 0.0          # len < 2 -> 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_fidelity_eval_builders.py -v`
Expected: FAIL (`ModuleNotFoundError: ... fidelity_eval`).

- [ ] **Step 3: Implement the builders + statistics**

Create `famail_temporal/baselines/fidelity_eval.py`:

```python
"""Level-1 fidelity evaluation: discriminator (HuMID) + discriminator-free.

All functions are inference/analysis only — no training, no global state. The
HuMID discriminator (famail_temporal/fidelity) is consumed read-only and
forward-only (under torch.no_grad). See
docs/superpowers/specs/2026-06-17-level1-data-quality-table-design.md.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.transmission import jensen_shannon_divergence


# ---------------------------------------------------------------- builders ----

def _xy_from_cells(cells: Sequence[int]) -> List[Tuple[int, int]]:
    """Flat cell ids -> [(x, y), ...] via x = c // GY, y = c % GY."""
    return [(int(c) // gc.GY, int(c) % gc.GY) for c in cells]


def _xy_from_traj(traj) -> List[Tuple[int, int]]:
    return [(int(s.x_grid), int(s.y_grid)) for s in traj.states]


def real_to_disc_tensor(traj) -> torch.Tensor:
    """Trajectory -> discriminator input [L, 4]: (x+1, y+1, time_bucket, day)."""
    rows = [
        [float(s.x_grid) + 1.0, float(s.y_grid) + 1.0,
         float(s.time_bucket), float(s.day_index)]
        for s in traj.states
    ]
    return torch.tensor(rows, dtype=torch.float32)


def generated_to_disc_tensor(
    cells: Sequence[int], time_bucket: int, day_index: int,
) -> torch.Tensor:
    """Generated flat cells -> discriminator input [L, 4].

    Un-flattens each cell to (x, y), adds +1 (1-indexed), and synthesizes a
    constant per-step (time_bucket, day_index) supplied by the caller (the
    paired real seed's temporal context; see plan Global Constraints note).
    """
    rows = [
        [float(x) + 1.0, float(y) + 1.0, float(time_bucket), float(day_index)]
        for (x, y) in _xy_from_cells(cells)
    ]
    return torch.tensor(rows, dtype=torch.float32)


# ------------------------------------------------------------- statistics ----

def trajectory_statistics(
    traj_or_cells: Union[object, Sequence[int]],
) -> Dict[str, float]:
    """{'length', 'mean_displacement', 'coverage'} for a Trajectory or cell list.

    - length: number of steps.
    - mean_displacement: mean Euclidean distance between consecutive (x, y)
      cells (0.0 if length < 2).
    - coverage: count of unique (x, y) cells visited.
    """
    if hasattr(traj_or_cells, "states"):
        xy = _xy_from_traj(traj_or_cells)
    else:
        xy = _xy_from_cells(traj_or_cells)
    length = len(xy)
    coverage = len(set(xy))
    if length < 2:
        mean_disp = 0.0
    else:
        dists = [
            float(np.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1]))
            for i in range(length - 1)
        ]
        mean_disp = float(np.mean(dists))
    return {"length": length, "mean_displacement": mean_disp, "coverage": coverage}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_fidelity_eval_builders.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/fidelity_eval.py \
        famail_temporal/baselines/tests/test_fidelity_eval_builders.py
git commit -m "feat(baselines): fidelity tensor builders + trajectory statistics"
```

---

## Task 3: Discriminator-free distributional fidelity (`fidelity_eval.py` part 2)

**Files:**
- Modify: `famail_temporal/baselines/fidelity_eval.py`
- Test: `famail_temporal/baselines/tests/test_fidelity_eval_distributional.py`

**Interfaces:**
- Consumes: `trajectory_statistics` (Task 2), `jensen_shannon_divergence` (existing).
- Produces: `distributional_fidelity(source_stats, raw_stats, *, bins=50) -> dict{"per_stat": {"length", "mean_displacement", "coverage"}, "aggregate": float}` where each value is the JS divergence (bits, lower=better) between the source's and raw's histogram of that statistic; `aggregate` = mean of the three.

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/tests/test_fidelity_eval_distributional.py`:

```python
"""fidelity_eval.distributional_fidelity: histogram + JS over trajectory stats."""
import math

from famail_temporal.baselines import fidelity_eval as fe


def _stats(lengths, disps, covs):
    return [
        {"length": l, "mean_displacement": d, "coverage": c}
        for l, d, c in zip(lengths, disps, covs)
    ]


def test_identical_distributions_have_zero_divergence():
    s = _stats([10, 12, 14, 16], [1.0, 1.1, 0.9, 1.0], [8, 9, 10, 11])
    out = fe.distributional_fidelity(s, list(s), bins=10)
    assert math.isclose(out["per_stat"]["length"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["per_stat"]["mean_displacement"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["per_stat"]["coverage"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["aggregate"], 0.0, abs_tol=1e-9)


def test_disjoint_length_distributions_have_high_divergence():
    raw = _stats([10, 11, 12, 13], [1.0]*4, [8, 8, 8, 8])
    gen = _stats([50, 52, 54, 56], [1.0]*4, [8, 8, 8, 8])   # collapsed-like lengths
    out = fe.distributional_fidelity(gen, raw, bins=20)
    # length distributions are disjoint -> JS near 1 bit; coverage identical -> ~0
    assert out["per_stat"]["length"] > 0.9
    assert math.isclose(out["per_stat"]["coverage"], 0.0, abs_tol=1e-9)
    assert out["aggregate"] > 0.0


def test_aggregate_is_mean_of_three():
    raw = _stats([10, 20], [1.0, 2.0], [5, 6])
    gen = _stats([10, 20], [1.0, 2.0], [5, 6])
    out = fe.distributional_fidelity(gen, raw, bins=8)
    ps = out["per_stat"]
    assert math.isclose(
        out["aggregate"],
        (ps["length"] + ps["mean_displacement"] + ps["coverage"]) / 3.0,
        rel_tol=1e-9,
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_fidelity_eval_distributional.py -v`
Expected: FAIL (`AttributeError: ... distributional_fidelity`).

- [ ] **Step 3: Implement `distributional_fidelity`**

Append to `famail_temporal/baselines/fidelity_eval.py`:

```python
# -------------------------------------------------- distributional fidelity ----

_STAT_KEYS = ("length", "mean_displacement", "coverage")


def _hist(values: List[float], lo: float, hi: float, bins: int) -> np.ndarray:
    """Normalized histogram over [lo, hi]; uniform if the range is degenerate."""
    arr = np.asarray(values, dtype=np.float64)
    if hi <= lo:
        h = np.ones(bins, dtype=np.float64)
        return h / h.sum()
    counts, _ = np.histogram(arr, bins=bins, range=(lo, hi))
    total = counts.sum()
    if total == 0:
        return np.zeros(bins, dtype=np.float64)
    return counts.astype(np.float64) / total


def distributional_fidelity(
    source_stats: List[Dict[str, float]],
    raw_stats: List[Dict[str, float]],
    *,
    bins: int = 50,
) -> Dict[str, object]:
    """Per-statistic JS divergence (bits, lower=better) of source vs raw.

    For each of {length, mean_displacement, coverage}, histogram the source
    values and the raw values on a SHARED bin grid (pooled min..max of both),
    then take the Jensen-Shannon divergence. aggregate = mean of the three.
    """
    per_stat: Dict[str, float] = {}
    for key in _STAT_KEYS:
        src = [float(s[key]) for s in source_stats]
        raw = [float(s[key]) for s in raw_stats]
        pooled = src + raw
        lo, hi = (min(pooled), max(pooled)) if pooled else (0.0, 0.0)
        p = _hist(src, lo, hi, bins)
        q = _hist(raw, lo, hi, bins)
        per_stat[key] = float(jensen_shannon_divergence(p, q))
    aggregate = float(np.mean([per_stat[k] for k in _STAT_KEYS]))
    return {"per_stat": per_stat, "aggregate": aggregate}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_fidelity_eval_distributional.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/fidelity_eval.py \
        famail_temporal/baselines/tests/test_fidelity_eval_distributional.py
git commit -m "feat(baselines): discriminator-free distributional fidelity (Fidelity-B)"
```

---

## Task 4: HuMID paired fidelity + validation gate (`fidelity_eval.py` part 3)

**Files:**
- Modify: `famail_temporal/baselines/fidelity_eval.py`
- Test: `famail_temporal/baselines/tests/test_fidelity_eval_humid.py`

**Interfaces:**
- Consumes: a discriminator module with `forward(x1, x2) -> Tensor[B, 1]` in [0,1] (the real `MultiStreamSiameseDiscriminator`, or a stub in tests). Pairs are `(left[L,4], right[L',4])` tensors (variable length; padded internally per batch).
- Produces:
  - `humid_paired_fidelity(discriminator, pairs, *, batch_size=64, device=None) -> dict{"mean": float, "std": float, "n": int}` — mean same-agent probability over pairs (forward-only).
  - `validation_gate(discriminator, *, real_pairs, collapsed_pairs, shuffled_pairs, batch_size=64, device=None, margin=GATE_MARGIN) -> dict{"high_real_real", "low_collapsed", "low_shuffled", "margin", "passed"}`.
  - module constant `GATE_MARGIN = 0.2`.

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/tests/test_fidelity_eval_humid.py`:

```python
"""fidelity_eval: HuMID paired fidelity + validation gate (stub discriminator)."""
import math

import torch
import torch.nn as nn

from famail_temporal.baselines import fidelity_eval as fe


class _ConstDiscriminator(nn.Module):
    """Returns a fixed same-agent probability for every pair."""
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def forward(self, x1, x2, **kwargs):
        b = x1.shape[0]
        return torch.full((b, 1), self.prob)


class _LengthSimDiscriminator(nn.Module):
    """High prob when the two trajectories have similar length, else low.

    Stands in for 'realistic' scoring: real-vs-real (similar lengths) -> high;
    real-vs-collapsed (very different lengths) -> low.
    """
    def forward(self, x1, x2, **kwargs):
        # x1, x2: [B, L, 4] padded; recover per-row nonzero length from coords.
        l1 = (x1[..., 0] > 0).sum(dim=1).float()
        l2 = (x2[..., 0] > 0).sum(dim=1).float()
        diff = (l1 - l2).abs()
        prob = torch.exp(-diff / 5.0).unsqueeze(1)   # close lengths -> ~1
        return prob


def _pair(len_a, len_b):
    a = torch.ones(len_a, 4)
    b = torch.ones(len_b, 4)
    return (a, b)


def test_humid_paired_fidelity_mean_over_pairs():
    disc = _ConstDiscriminator(0.8)
    pairs = [_pair(5, 5), _pair(6, 6), _pair(7, 7)]
    out = fe.humid_paired_fidelity(disc, pairs, batch_size=2)  # multi-batch
    assert out["n"] == 3
    assert math.isclose(out["mean"], 0.8, rel_tol=1e-6)
    assert math.isclose(out["std"], 0.0, abs_tol=1e-6)


def test_validation_gate_passes_with_clear_separation():
    disc = _LengthSimDiscriminator()
    real_pairs = [_pair(18, 18) for _ in range(8)]        # similar -> high
    collapsed_pairs = [_pair(18, 52) for _ in range(8)]   # mismatch -> low
    shuffled_pairs = [_pair(18, 50) for _ in range(8)]    # mismatch -> low
    out = fe.validation_gate(
        disc, real_pairs=real_pairs, collapsed_pairs=collapsed_pairs,
        shuffled_pairs=shuffled_pairs, batch_size=4,
    )
    assert out["high_real_real"] > out["low_collapsed"]
    assert out["high_real_real"] > out["low_shuffled"]
    assert out["passed"] is True


def test_validation_gate_fails_without_separation():
    disc = _ConstDiscriminator(0.5)   # cannot tell real from garbage
    pairs = [_pair(18, 18) for _ in range(4)]
    out = fe.validation_gate(
        disc, real_pairs=pairs, collapsed_pairs=pairs, shuffled_pairs=pairs,
        batch_size=4,
    )
    assert out["passed"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_fidelity_eval_humid.py -v`
Expected: FAIL (`AttributeError: ... humid_paired_fidelity`).

- [ ] **Step 3: Implement paired fidelity + gate**

Append to `famail_temporal/baselines/fidelity_eval.py`:

```python
# ------------------------------------------------- HuMID paired fidelity ----

GATE_MARGIN = 0.2   # min (high_real_real - max low) for the gate to pass


def _pad_pairs_to_batch(
    pairs: List[Tuple[torch.Tensor, torch.Tensor]], device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack a list of (left[L,4], right[L',4]) into padded [B, Lmax, 4] tensors."""
    lefts = [p[0] for p in pairs]
    rights = [p[1] for p in pairs]
    lmax = max(t.shape[0] for t in lefts)
    rmax = max(t.shape[0] for t in rights)

    def _stack(tensors: List[torch.Tensor], lmax: int) -> torch.Tensor:
        out = torch.zeros(len(tensors), lmax, 4, dtype=torch.float32)
        for i, t in enumerate(tensors):
            out[i, : t.shape[0]] = t
        return out

    return _stack(lefts, lmax).to(device), _stack(rights, rmax).to(device)


def _score_pairs(
    discriminator: torch.nn.Module,
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Forward the discriminator over all pairs; return per-pair probs (N,)."""
    probs: List[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            x1, x2 = _pad_pairs_to_batch(chunk, device)
            out = discriminator(x1, x2)
            probs.extend(out.reshape(-1).detach().cpu().tolist())
    return np.asarray(probs, dtype=np.float64)


def humid_paired_fidelity(
    discriminator: torch.nn.Module,
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> Dict[str, float]:
    """Mean same-agent probability over (left, right) trajectory-tensor pairs."""
    device = device or torch.device("cpu")
    if not pairs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    probs = _score_pairs(discriminator, pairs, batch_size=batch_size, device=device)
    return {
        "mean": float(probs.mean()),
        "std": float(probs.std(ddof=1)) if probs.size > 1 else 0.0,
        "n": int(probs.size),
    }


def validation_gate(
    discriminator: torch.nn.Module,
    *,
    real_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    collapsed_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    shuffled_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 64,
    device: torch.device | None = None,
    margin: float = GATE_MARGIN,
) -> Dict[str, object]:
    """Does the discriminator rank real-vs-real above real-vs-garbage?

    Passes iff high_real_real - max(low_collapsed, low_shuffled) >= margin AND
    high_real_real exceeds both lows. All three means are returned regardless.
    """
    device = device or torch.device("cpu")
    high = humid_paired_fidelity(discriminator, real_pairs, batch_size=batch_size, device=device)["mean"]
    low_c = humid_paired_fidelity(discriminator, collapsed_pairs, batch_size=batch_size, device=device)["mean"]
    low_s = humid_paired_fidelity(discriminator, shuffled_pairs, batch_size=batch_size, device=device)["mean"]
    worst_low = max(low_c, low_s)
    passed = bool((high - worst_low) >= margin and high > low_c and high > low_s)
    return {
        "high_real_real": float(high),
        "low_collapsed": float(low_c),
        "low_shuffled": float(low_s),
        "margin": float(margin),
        "passed": passed,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_fidelity_eval_humid.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full fidelity_eval test set**

Run: `python -m pytest famail_temporal/baselines/tests/test_fidelity_eval_builders.py famail_temporal/baselines/tests/test_fidelity_eval_distributional.py famail_temporal/baselines/tests/test_fidelity_eval_humid.py -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/fidelity_eval.py \
        famail_temporal/baselines/tests/test_fidelity_eval_humid.py
git commit -m "feat(baselines): HuMID paired fidelity + validation gate (Fidelity-A)"
```

---

## Task 5: Orchestrator CLI (`run_level1_table.py`)

**Files:**
- Create: `famail_temporal/baselines/run_level1_table.py`
- Test: `famail_temporal/baselines/tests/test_run_level1_table.py`

**Interfaces:**
- Consumes: `DataBundle.load`, `load_edited_trajectories`, `fit_and_evaluate` (for re-training the BC + GAN generators), `generate_trajectories` (Task 1), `pickups_to_pickup_3d`, `data_level_fairness`, `trajectory_context`, `load_discriminator`, and all of `fidelity_eval` (Tasks 2-4).
- Produces: `run_level1_table.py` CLI; unit-testable pure helpers `result_to_json(result) -> str` and `render_table(result) -> str`.

**Design notes for the implementer (the full run is GPU + manual; only the pure helpers are unit-tested here):**
- **Sources & trajectories:** raw = `bundle.trajectories`; edited = `load_edited_trajectories(bundle, edit_dir)`; BC + GAN = re-train via `fit_and_evaluate(...)` then re-build the model is not returned by `fit_and_evaluate`, so generation must happen INSIDE a dedicated train+generate path. Use this approach: call a new private `_train_and_generate(bundle, train_trajectories, *, gan_loss, n_critic, mle_epochs, adv_epochs, max_len, device, seed, fidelity_sample_size)` that mirrors `fit_and_evaluate`'s training (MLE [+ adversarial for GAN]) but returns the trained `model` so the orchestrator can call `generate_trajectories`. To avoid duplicating training logic, import and reuse the training functions `train_mle` and `adversarial_finetune` directly (both are public in `gan/train_mle.py` / `gan/train_adversarial.py`), exactly as `model_level.fit_and_evaluate` does.
- **Fidelity sample:** full-trajectory generation for fidelity uses a context SAMPLE of size `fidelity_sample_size` (default 5000) — the first N filtered contexts — for tractability (looped/batched full decode over all ~105k is unnecessary for a mean realism estimate; the fairness columns still use full terminal-cell generation via `generate_pickups`). This is a v1 performance choice consistent with spec §3.3 (single-seed, coarse fidelity); document it in the results doc.
- **Pairing:** for BC/GAN, `contexts = [trajectory_context(t) for t in filtered_train][:N]` and `real_pairs_source = filtered_train[:N]`; `gen_cells = generate_trajectories(model, contexts, ...)`; pair `(real_to_disc_tensor(filtered_train[i]), generated_to_disc_tensor(gen_cells[i], time_bucket=filtered_train[i].states[0].time_bucket, day_index=filtered_train[i].states[0].day_index))`. Skip pairs whose `gen_cells[i]` is empty (count as `n_empty`). `filtered_train` = trajectories surviving the same `max_tokens` filter `fit_and_evaluate` applies (replicate it: `[t for t in train if len(trajectory_to_tokens(t)) <= max_tokens]`).
- **Edited pairing:** read `histories.pkl`; pair `(real_to_disc_tensor(h.original), real_to_disc_tensor(h.modified))` over a sample.
- **Validation gate inputs:** `real_pairs` = `(real_to_disc_tensor(t_i), real_to_disc_tensor(t_i))` self-pairs over a raw sample; `collapsed_pairs` = `(real_to_disc_tensor(real_i), generated_to_disc_tensor(gan_cells_i, ...))` drawn from the GAN source's longest rollouts; `shuffled_pairs` = `(real_to_disc_tensor(real_i), generated_to_disc_tensor(shuffle(real_i cells), ...))` where the real cells are randomly permuted.
- **Fairness columns:** per source, terminal cells → `pickups_to_pickup_3d` → `data_level_fairness`. Raw uses `data_level_fairness(bundle)`; edited uses the edit's terminal cells; BC/GAN use their full generation's terminal cells (last cell of each rollout via `generate_pickups`, kept separate from the fidelity sample). Label all as single-seed.
- **Gate → trust:** set each source's `fidelity_a_trusted = gate["passed"]`.
- **Persistence:** per-run dir `Path(config.PACKAGE_ROOT)/"results"/"level1_table"/<timestamp>/` with `level1_metrics.json`, `level1_table.md`, `trajectory_stats.npz`; print a summary.

- [ ] **Step 1: Write the failing test (pure helpers only)**

Create `famail_temporal/baselines/tests/test_run_level1_table.py`:

```python
"""run_level1_table pure helpers: JSON round-trip + table rendering."""
import json

from famail_temporal.baselines import run_level1_table as r


def _fake_result():
    return {
        "gate": {"high_real_real": 0.82, "low_collapsed": 0.41,
                 "low_shuffled": 0.39, "margin": 0.2, "passed": True},
        "sources": {
            "raw":    {"f_causal": 0.8052, "f_spatial": 0.0822,
                       "fidelity_a": 1.0, "fidelity_a_trusted": True,
                       "fidelity_b": 0.0},
            "edited": {"f_causal": 0.8180, "f_spatial": 0.0824,
                       "fidelity_a": 0.79, "fidelity_a_trusted": True,
                       "fidelity_b": 0.03},
            "bc":     {"f_causal": 0.8062, "f_spatial": 0.0828,
                       "fidelity_a": 0.71, "fidelity_a_trusted": True,
                       "fidelity_b": 0.05},
            "gan":    {"f_causal": 0.8198, "f_spatial": 0.0843,
                       "fidelity_a": 0.22, "fidelity_a_trusted": True,
                       "fidelity_b": 0.61},
        },
        "edit_dir": "famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup",
    }


def test_result_to_json_roundtrips():
    blob = r.result_to_json(_fake_result())
    loaded = json.loads(blob)
    assert loaded["sources"]["edited"]["f_causal"] == 0.8180
    assert loaded["gate"]["passed"] is True


def test_render_table_contains_sources_and_gate_verdict():
    md = r.render_table(_fake_result())
    assert "Fidelity-A" in md and "Fidelity-B" in md
    for label in ("raw", "edited", "bc", "gan"):
        assert label in md
    assert "0.8180" in md          # edited f_causal rendered
    assert "PASSED" in md or "passed" in md
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_level1_table.py -v`
Expected: FAIL (`ModuleNotFoundError: ... run_level1_table`).

- [ ] **Step 3: Implement `run_level1_table.py`**

Create `famail_temporal/baselines/run_level1_table.py`. Implement: `result_to_json`, `render_table`, the private `_train_and_generate`, source assembly, gate, fidelity, persistence, and `main(argv)`. Pure helpers (tested) are:

```python
"""CLI: assemble the Level-1 data-quality table (Two-Level Argument, Level 1).

Compares four data sources -- raw, FAM-AIL edited, BC-generated, GAN-generated
-- on causal fairness, spatial fairness, and two fidelity metrics (HuMID paired
[gated] + discriminator-free distributional). See
docs/superpowers/specs/2026-06-17-level1-data-quality-table-design.md.

Example:
    python -m famail_temporal.baselines.run_level1_table \\
        --edit-dir famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup \\
        --mle-epochs 20 --device auto
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
        "| Source | F_causal | F_spatial | Fidelity-A (HuMID, higher=better) "
        "| Fidelity-B (divergence, lower=better) |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(rows) + "\n"
    )


# ... main() and helpers (_train_and_generate, source assembly, gate, persist) ...
```

The implementer completes `main(argv)` and the private helpers per the design notes above. `main` must: parse args (`--edit-dir`, `--mle-epochs` default 20, `--adv-epochs` default 3, `--gan-loss` default `wgan-gp`, `--n-critic` default 5, `--max-tokens` default `gc.MAX_TRAIN_TOKENS`, `--fidelity-sample-size` default 5000, `--gen-batch-size`, `--seed` default 0, `--device` default `auto`, `--out-dir`, `--quiet`); build the four sources; run the gate; compute Fidelity-A/B + fairness per source; assemble `result` with the exact key shape the tests expect (`gate`, `sources[<key>]` with `f_causal/f_spatial/fidelity_a/fidelity_a_trusted/fidelity_b`, `edit_dir`); persist `level1_metrics.json` + `level1_table.md` (via `render_table`) + `trajectory_stats.npz`; print a summary. Resolve the discriminator path as `Path(config.PACKAGE_ROOT)/"discriminator_checkpoints"/"default"/"best.pt"` and load via `load_discriminator(path)`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_level1_table.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: CLI parse check + full baselines suite**

Run: `python -m famail_temporal.baselines.run_level1_table --help`
Expected: usage with all flags, exit 0.
Run: `python -m pytest famail_temporal/baselines/ -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/run_level1_table.py \
        famail_temporal/baselines/tests/test_run_level1_table.py
git commit -m "feat(baselines): Level-1 data-quality table orchestrator"
```

---

## Task 6: Real-data smoke + paper-ready results doc

**Files:**
- Create: `famail_temporal/baselines/LEVEL1_RESULTS.md`
- (research artifacts written to `famail_temporal/results/level1_table/<ts>/`, gitignored — NOT committed)

- [ ] **Step 1: Run the real-data smoke (GPU, ~40 min)**

Run:
```bash
python -m famail_temporal.baselines.run_level1_table \
    --edit-dir famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup \
    --mle-epochs 20 --device auto --seed 0
```
Expected: writes a timestamped dir under `famail_temporal/results/level1_table/`, prints the table + gate verdict, exits 0. Inspect `level1_table.md` and `level1_metrics.json`.

- [ ] **Step 2: Write `LEVEL1_RESULTS.md`**

Create `famail_temporal/baselines/LEVEL1_RESULTS.md` (paper-ready) with: the filled 4×4 table (copied from the run's `level1_metrics.json`); the validation-gate verdict and what it implies (if failed, lead with Fidelity-B and say so); one interpretation paragraph per metric; the headline finding (does FAM-AIL edited data win on causal fairness among faithful sources, and does fidelity disqualify the collapsed GAN?); a "single-seed v1" caveat; and the exact reproduction command + per-run dir path. Cross-link the spec and `MEETING_38_PREP.md`.

- [ ] **Step 3: Commit the results doc**

```bash
git add famail_temporal/baselines/LEVEL1_RESULTS.md
git commit -m "docs(baselines): Level-1 data-quality results (first real-data run)"
```

---

## Self-Review

**1. Spec coverage:**
- §4 four data sources (raw/edited/BC/GAN) — Task 5 source assembly. ✓
- §5 Fidelity-A HuMID paired, reduced-mode, +1 coords, no_grad — Task 4 (`humid_paired_fidelity`) + Task 2 builders + Task 5 pairing. ✓
- §6 validation gate (3 categories, margin 0.2, non-fatal fail, persisted) — Task 4 (`validation_gate`, `GATE_MARGIN`) + Task 5 wiring. ✓
- §7 Fidelity-B distributional (length/displacement/coverage, JS, lower=better, shared bins) — Task 3. ✓
- §8.1 `fidelity_eval.py` all six functions — Tasks 2-4. ✓
- §8.2 `generate_trajectories` additive in rollout.py — Task 1. ✓
- §8.3 orchestrator CLI with the listed flags — Task 5. ✓
- §8.4 persistence (per-run dir + npz + canonical results doc) — Task 5 + Task 6. ✓
- §10 error handling (missing checkpoint, gate fail non-fatal, empty trajectory, len<2, coord conversion, max_tokens alignment) — covered across Task 2 (len<2), Task 4 (gate), Task 5 design notes (checkpoint/empty/alignment). ✓
- §11 testing (unit per function + manual smoke) — Tasks 1-5 tests + Task 6 smoke. ✓
- §3.3 single-seed; §3.4 time/day synthesis (refined to paired-real seed, flagged); §3.5 three statistics; §6 margin — all reflected. ✓

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Task 5's `main` body is described prose-plus-tested-helpers because the full run is GPU-manual and only the pure helpers are unit-tested — the design notes give exact function names, pairing expressions, key shapes, and flag defaults (not a placeholder; an explicit, complete contract). The result-dict key shape is pinned by the Task-5 test.

**3. Type consistency:**
- `generate_trajectories(...) -> List[List[int]]` (Task 1) consumed by Task 5. ✓
- `real_to_disc_tensor/generated_to_disc_tensor -> Tensor[L,4]` (Task 2) feed `humid_paired_fidelity` pairs + `validation_gate` (Task 4) + Task 5. ✓
- `trajectory_statistics -> dict{length,mean_displacement,coverage}` (Task 2) feeds `distributional_fidelity` (Task 3). ✓
- `distributional_fidelity -> {per_stat, aggregate}`; orchestrator reads `aggregate` as each source's `fidelity_b`. ✓
- `humid_paired_fidelity -> {mean,std,n}`; orchestrator reads `mean` as `fidelity_a`. ✓
- `validation_gate -> {high_real_real, low_collapsed, low_shuffled, margin, passed}`; matches Task 5 test's `gate` shape. ✓
- `result` dict shape (`gate`, `sources[key]{f_causal,f_spatial,fidelity_a,fidelity_a_trusted,fidelity_b}`, `edit_dir`) consistent between Task 5 impl, its test, and `render_table`. ✓

**4. Standing constraints:** branch `two-level-paper`; named-file staging; no `eval(` literal; baselines-only (no algorithm/fairness/fidelity edits); forward-only `no_grad`; +1 coords; flat-cell un-flatten; edit-source + checkpoint paths pinned in Global Constraints. ✓
