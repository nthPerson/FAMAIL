# Driver-Conditioned Generation + Identity-Aware Fidelity (Level-1 v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the BC/GAN baselines driver-conditioned so generated trajectories carry a driver identity, then score Fidelity-A with the frozen HuMID discriminator used near its trained regime (a real-anchored matched-vs-mismatched gate), and enrich the discriminator-free Fidelity-B — producing a trustworthy Level-1 v2 data-quality table.

**Architecture:** An *optional, additive* driver embedding is folded into `TrajectoryLSTM`'s existing additive context (`cell + tblock + driver`); `driver_idx=None` preserves today's numerics bit-for-bit. Fidelity-A builds HuMID inputs that mirror `fidelity/context.py` exactly (slot-0 trajectory-under-test + 4 real same-driver context trajectories + the driver's real 11-dim profile, driving omitted symmetrically), and a **real-anchored** gate (real-d vs real-d = matched; real-d vs real-d′ = mismatched) establishes that HuMID is well-posed in our regime. Per-source matched scores fill the table; per-source matched-minus-mismatched separations (including the spec's gen-for-d vs gen-for-d′) are persisted as diagnostics. Fidelity-B gains radius-of-gyration, net-displacement, and a terminal-cell distribution JS.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest. Reuses the existing `famail_temporal.baselines.gan` generator/training/rollout stack, `famail_temporal.fidelity` (frozen HuMID, read-only), and `famail_temporal.baselines.fidelity_eval` / `metrics` / `transmission`.

## Global Constraints

- **Branch:** `two-level-paper`. Stage **named files only** — never `git add -A` / `git add .`. The untracked `famail_temporal/baselines/*/results/` directories are gitignored run artifacts; do not commit PNG/CSV/JSON under any `results/` dir.
- **Frozen read-only modules:** No edits to `famail_temporal/algorithm/`, `famail_temporal/fairness/`, or `famail_temporal/fidelity/`. HuMID is consumed forward-only under `torch.no_grad()`; use `model.train(False)` (the literal `eval` immediately followed by `(` is blocked by a repo security hook — never write it).
- **Backward compatibility is a hard requirement.** Every `driver_idx` / `driver_idxs` parameter defaults to `None`, and the `None` path must be numerically identical to the pre-change code (regression-tested). Existing tests in `famail_temporal/baselines/tests/` and `famail_temporal/tests/` must continue to pass unchanged.
- **Coordinate convention:** HuMID expects **1-indexed** coords `[1-48, 1-90]`. Always build discriminator inputs through `fidelity_eval.real_to_disc_tensor` / `generated_to_disc_tensor` (they add +1). Never feed raw `Trajectory.to_tensor()`.
- **Mask convention:** boolean, `True = valid step` (mirrors `fidelity/context.py`).
- **HuMID forward signature** (verified): `forward(x1, x2, mask1=None, mask2=None, *, driving_1=None, driving_2=None, mask_d1=None, mask_d2=None, profile_1=None, profile_2=None) -> probs [B,1]`. Trajectory tensors may be 3D `[B,L,4]` (legacy, auto-expands to N=1 then zero-pads) or 4D `[B,N,L,4]` (Ren-aligned). The internal `FeatureNormalizer` maps the 4 raw features `[x,y,time_bucket,day_index]` → 6 normalized features; **inputs are 4-dim, not 6-dim.** When `driving_1`/`driving_2` are `None`, the model uses a fixed zero `driving_default_embedding` for both branches (symmetric graceful degradation). `profile_*` are `[B, 11]`.
- **Checkpoint:** `famail_temporal/discriminator_checkpoints/default/best.pt`, loaded via `fidelity.checkpoint.load_discriminator(path)` → frozen model; **move it to the run device with `.to(device)`** before any forward (v1 device bug).
- **Single representative seed** (v1 stands). `seed=0` default.
- **Data facts:** `n_drivers ≈ 50`; `Trajectory.driver_id` is an int in `[0, 49]`; `bundle.multi_stream.profile_features: Dict[int, np.ndarray]` (driver_id → 11-dim normalized); `bundle.multi_stream.seeking_trajs: Dict[int, List]` (driver_id → 1-indexed `[x,y,t,d]` trajectories). The real branches in Fidelity-A use `bundle.trajectories` grouped by `driver_id` (the same corpus the generators are trained on), converted via `real_to_disc_tensor` (+1).
- **gan/config.py constants** (verified): `VOCAB_SIZE`, `N_CELLS`, `BOS`, `EOS`, `PAD`, `N_TBLOCKS`, `EMBED_DIM=64`, `HIDDEN_DIM=128`, `N_LAYERS=1`, `MLE_BATCH_SIZE=32`, `MAX_GEN_LEN=64`, `MAX_TRAIN_TOKENS=256`, `GEN_BATCH_SIZE=512`, `GY`, `D_UPDATE_EVERY=1`, `WGAN_N_CRITIC=5`, etc.

---

## Design decisions resolved at plan time (user-confirmed 2026-06-17)

1. **Gate anchor = real-anchored (superset).** The headline `trusted` verdict comes from real-d vs real-d′ separation (HuMID well-posedness in our regime). Per-source matched/mismatched separations — including the spec §3.4 gen-for-d vs gen-for-d′ — are **also** computed and persisted as diagnostics. Every raw mean is saved so the gate can be reinterpreted post-run. *(Deviation from spec §3.4 literal wording; strict superset; documented in results.)*
2. **Branch construction mirrors `fidelity/context.py` exactly.** A HuMID branch = slot 0 (the trajectory under test) + slots 1..N-1 (the driver's **real** context trajectories, sampled with replacement if fewer than N-1) + the driver's **real** 11-dim profile. Streams = **seeking + profile**; **driving = None** for both branches (symmetric). N = 5.
3. **Per-source Fidelity-A** = mean matched same-agent probability:
   - `raw`: (real-d slot0_a + real-d context) vs (real-d slot0_b + real-d context) — the anchor/ceiling.
   - `edited`: (real-d + context) vs (edited-d slot0 + real-d context).
   - `bc` / `gan`: (real-d + context) vs (gen-cond-d slot0 + real-d context).
4. **Gate** (real-anchored): `high_matched` = Fidelity-A(raw); `low_mismatched` = mean over (d, d′≠d) of HuMID( (real-d + context d, profile d) vs (real-d′ slot0 + context d′, profile d′) ). `passed = (high_matched - low_mismatched) >= MARGIN and high_matched > low_mismatched`. `MARGIN = 0.2` (reuse `GATE_MARGIN`).
5. **Per-source separation diagnostic** for edited/bc/gan: `matched(S) - mismatched(S)` where `mismatched(S)` pairs (real-d + context d, profile d) against (S-of-d′ slot0 + context d′, profile d′). For bc/gan this is the spec's "did the generator capture driver style" test.
6. **Packaging = new sibling module** `run_level1_table_v2.py` (v1's `run_level1_table.py` stays frozen). Reuse `fidelity_eval`, `metrics.data_level_fairness`, the edited-fairness-from-`metrics_after` fix, and the curve-capture machinery.
7. **Enriched Fidelity-B:** `trajectory_statistics` gains `radius_of_gyration` + `net_displacement` (0.0 if length < 2). `distributional_fidelity` / `stat_ranges` become **key-parameterized** (default = the original 3 keys → v1 behavior unchanged); v2 passes the 5-key set. A new `terminal_cell_distribution_js` adds a corpus-level terminal-cell JS. v2 aggregate = mean of the 5 per-trajectory JS + the terminal-cell JS.
8. **Unknown driver_id** at generation → raise `KeyError` with a clear message. **Driver with < N trajectories** for context/sets → sample with replacement (logged note). **Profile missing** for a driver → zero/default profile (HuMID already supports it), logged.

---

## File Structure

- **Modify** `famail_temporal/baselines/gan/generator.py` — optional `n_drivers` + `driver_embed`; `driver_idx` kw through `forward`/`step`/`step_embed`.
- **Create** `famail_temporal/baselines/gan/drivers.py` — `build_driver_index`, `group_by_driver`.
- **Modify** `famail_temporal/baselines/gan/train_mle.py` — optional `driver_idxs`.
- **Modify** `famail_temporal/baselines/gan/gumbel.py` — optional `driver_idx`.
- **Modify** `famail_temporal/baselines/gan/train_adversarial.py` — optional `driver_idxs` (→ gumbel + MLE-anchor forward).
- **Modify** `famail_temporal/baselines/gan/rollout.py` — optional `driver_idx` through `sample_trajectory_cells`, `sample_terminal_cells_batched`, `generate_trajectories`, `generate_pickups`.
- **Modify** `famail_temporal/baselines/fidelity_eval.py` — N-set identity builder + `humid_identity_fidelity` + `identity_validation_gate`; RoG/net-disp in `trajectory_statistics`; key-parameterized `stat_ranges`/`distributional_fidelity`; `terminal_cell_distribution_js`.
- **Create** `famail_temporal/baselines/run_level1_table_v2.py` — the v2 orchestrator + CLI.
- **Create** tests: `test_generator_driver_cond.py`, `test_drivers.py`, `test_train_rollout_driver_cond.py`, `test_fidelity_eval_identity.py`, `test_fidelity_eval_enriched.py`, `test_run_level1_table_v2.py` (all in `famail_temporal/baselines/tests/`).
- **Create (deliverables, manual phase)** `famail_temporal/baselines/LEVEL1_V2_RESULTS.md`, `famail_temporal/baselines/LEVEL1_V2_METHODOLOGY.md`.

---

### Task 1: Optional driver embedding in `TrajectoryLSTM`

**Files:**
- Modify: `famail_temporal/baselines/gan/generator.py`
- Test: `famail_temporal/baselines/tests/test_generator_driver_cond.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `TrajectoryLSTM(__init__)` gains keyword `n_drivers: int | None = None`. When set, `self.driver_embed = nn.Embedding(n_drivers, embed_dim)`.
  - `forward(tokens, ctx_cell, ctx_tblock, driver_idx=None)`
  - `step(token, ctx_cell, ctx_tblock, hidden=None, driver_idx=None)`
  - `step_embed(input_embed, ctx_cell, ctx_tblock, hidden=None, driver_idx=None)`
  - When `driver_idx is not None`: `ctx = cell_embed(ctx_cell) + tblock_embed(ctx_tblock) + driver_embed(driver_idx)`. `driver_idx` is a `(B,)` long tensor.

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/tests/test_generator_driver_cond.py`:

```python
import torch

from famail_temporal.baselines.gan.generator import TrajectoryLSTM


def _tiny(n_drivers=None):
    return TrajectoryLSTM(
        vocab_size=20, n_tblocks=4, embed_dim=8, hidden_dim=8, n_layers=1,
    ) if n_drivers is None else TrajectoryLSTM(
        vocab_size=20, n_tblocks=4, embed_dim=8, hidden_dim=8, n_layers=1,
        n_drivers=n_drivers,
    )


def test_driver_idx_none_matches_unconditioned():
    """driver_idx=None on a driver-aware model == an unconditioned model with
    the same shared weights (the driver embedding contributes nothing)."""
    torch.manual_seed(0)
    cond = _tiny(n_drivers=3)
    plain = _tiny()
    # Copy the shared (non-driver) weights so the two models are identical
    # except for the unused driver embedding.
    plain.load_state_dict(
        {k: v for k, v in cond.state_dict().items() if k != "driver_embed.weight"}
    )
    tokens = torch.randint(0, 20, (2, 5))
    cc = torch.tensor([1, 2])
    tb = torch.tensor([0, 3])
    out_cond = cond(tokens, cc, tb, driver_idx=None)
    out_plain = plain(tokens, cc, tb)
    assert torch.allclose(out_cond, out_plain, atol=1e-6)


def test_driver_idx_changes_logits():
    torch.manual_seed(0)
    cond = _tiny(n_drivers=3)
    tokens = torch.randint(0, 20, (2, 5))
    cc = torch.tensor([1, 2])
    tb = torch.tensor([0, 3])
    a = cond(tokens, cc, tb, driver_idx=torch.tensor([0, 0]))
    b = cond(tokens, cc, tb, driver_idx=torch.tensor([1, 2]))
    assert a.shape == (2, 5, 20)
    assert not torch.allclose(a, b)


def test_step_and_step_embed_accept_driver_idx():
    torch.manual_seed(0)
    cond = _tiny(n_drivers=3)
    tok = torch.tensor([1, 2])
    cc = torch.tensor([1, 2])
    tb = torch.tensor([0, 3])
    di = torch.tensor([0, 1])
    logits_step, h = cond.step(tok, cc, tb, None, driver_idx=di)
    assert logits_step.shape == (2, 20)
    emb = cond.cell_embed(tok)
    logits_se, _ = cond.step_embed(emb, cc, tb, None, driver_idx=di)
    assert logits_se.shape == (2, 20)
    # step delegates to step_embed: same driver_idx -> same logits at step 0
    assert torch.allclose(logits_step, logits_se, atol=1e-6)


def test_unconditioned_model_unchanged_regression():
    """A model built without n_drivers has no driver_embed and behaves exactly
    as before (positional call still works)."""
    torch.manual_seed(0)
    m = _tiny()
    assert not hasattr(m, "driver_embed")
    tokens = torch.randint(0, 20, (2, 5))
    out = m(tokens, torch.tensor([1, 2]), torch.tensor([0, 3]))
    assert out.shape == (2, 5, 20)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_generator_driver_cond.py -q`
Expected: FAIL (`__init__` rejects `n_drivers`; `forward`/`step`/`step_embed` reject `driver_idx`).

- [ ] **Step 3: Implement the optional driver embedding**

Edit `famail_temporal/baselines/gan/generator.py`. Update `__init__` to accept `n_drivers` and conditionally create the embedding; add a private `_ctx` helper and thread `driver_idx` through all three decode methods:

```python
    def __init__(
        self,
        vocab_size: int = gc.VOCAB_SIZE,
        n_tblocks: int = gc.N_TBLOCKS,
        embed_dim: int = gc.EMBED_DIM,
        hidden_dim: int = gc.HIDDEN_DIM,
        n_layers: int = gc.N_LAYERS,
        n_drivers: int | None = None,
    ):
        super().__init__()
        self.cell_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=gc.PAD)
        self.tblock_embed = nn.Embedding(n_tblocks, embed_dim)
        # Optional additive driver conditioning. Absent (n_drivers=None) -> the
        # model is bit-for-bit the original unconditioned generator.
        self.driver_embed = (
            nn.Embedding(n_drivers, embed_dim) if n_drivers is not None else None
        )
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers, batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def _ctx(self, ctx_cell, ctx_tblock, driver_idx=None):
        """Additive conditioning context (B, E): cell + tblock [+ driver]."""
        ctx = self.cell_embed(ctx_cell) + self.tblock_embed(ctx_tblock)
        if driver_idx is not None:
            if self.driver_embed is None:
                raise ValueError(
                    "driver_idx was supplied but this TrajectoryLSTM was built "
                    "without n_drivers (no driver embedding exists)."
                )
            ctx = ctx + self.driver_embed(driver_idx)
        return ctx
```

Then rewrite the three decode methods to use `_ctx` and accept `driver_idx`:

```python
    def forward(
        self,
        tokens: torch.Tensor,      # (B, L) long input token ids
        ctx_cell: torch.Tensor,    # (B,) long start-cell ids
        ctx_tblock: torch.Tensor,  # (B,) long start time-block ids
        driver_idx: torch.Tensor | None = None,  # (B,) long driver ids
    ) -> torch.Tensor:
        x = self.cell_embed(tokens)                                   # (B, L, E)
        ctx = self._ctx(ctx_cell, ctx_tblock, driver_idx)            # (B, E)
        x = x + ctx.unsqueeze(1)                                      # broadcast
        out, _ = self.lstm(x)                                         # (B, L, H)
        return self.head(out)                                        # (B, L, V)

    def step_embed(
        self,
        input_embed: torch.Tensor,
        ctx_cell: torch.Tensor,
        ctx_tblock: torch.Tensor,
        hidden=None,
        driver_idx: torch.Tensor | None = None,
    ):
        ctx = self._ctx(ctx_cell, ctx_tblock, driver_idx)            # (B, E)
        x = (input_embed + ctx).unsqueeze(1)                          # (B, 1, E)
        out, hidden = self.lstm(x, hidden)                            # (B, 1, H)
        return self.head(out[:, -1]), hidden                          # (B, V), state

    def step(
        self,
        token: torch.Tensor,
        ctx_cell: torch.Tensor,
        ctx_tblock: torch.Tensor,
        hidden=None,
        driver_idx: torch.Tensor | None = None,
    ):
        return self.step_embed(
            self.cell_embed(token), ctx_cell, ctx_tblock, hidden,
            driver_idx=driver_idx,
        )
```

Keep the existing module docstring/comments; only the signatures and the context construction change. Preserve the explanatory comments on `step`/`step_embed`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_generator_driver_cond.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the existing generator-dependent tests (regression)**

Run: `python -m pytest famail_temporal/baselines/tests/test_smoke.py famail_temporal/baselines/tests/test_run_level1_table.py -q`
Expected: PASS (no behavior change for unconditioned models).

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/gan/generator.py famail_temporal/baselines/tests/test_generator_driver_cond.py
git commit -m "feat(baselines/gan): optional additive driver embedding in TrajectoryLSTM"
```

---

### Task 2: Driver index map + grouping helper

**Files:**
- Create: `famail_temporal/baselines/gan/drivers.py`
- Test: `famail_temporal/baselines/tests/test_drivers.py`

**Interfaces:**
- Consumes: `Trajectory.driver_id` (int).
- Produces:
  - `build_driver_index(trajectories) -> Dict[int, int]` — maps each distinct `driver_id` to a contiguous embedding index `[0, n_drivers)`, ordered by **sorted driver_id** (deterministic). Use this `n_drivers = len(map)` to size `TrajectoryLSTM(n_drivers=...)`.
  - `invert_driver_index(driver_to_idx) -> Dict[int, int]` — embedding index → driver_id.
  - `group_by_driver(trajectories) -> Dict[int, List[Trajectory]]` — keyed by **original driver_id**, insertion order preserved per driver.
  - `driver_idxs_for(trajectories, driver_to_idx) -> List[int]` — index-aligned embedding indices for a trajectory list (raises `KeyError` on an unknown driver_id with a clear message).

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/tests/test_drivers.py`:

```python
import pytest

from famail_temporal.baselines.gan.drivers import (
    build_driver_index, invert_driver_index, group_by_driver, driver_idxs_for,
)


class _Stub:
    def __init__(self, driver_id):
        self.driver_id = driver_id


def test_build_driver_index_is_sorted_and_contiguous():
    trajs = [_Stub(7), _Stub(2), _Stub(7), _Stub(5)]
    m = build_driver_index(trajs)
    assert m == {2: 0, 5: 1, 7: 2}          # sorted driver_id -> contiguous idx


def test_invert_driver_index():
    m = {2: 0, 5: 1, 7: 2}
    assert invert_driver_index(m) == {0: 2, 1: 5, 2: 7}


def test_group_by_driver_counts():
    trajs = [_Stub(7), _Stub(2), _Stub(7), _Stub(5)]
    g = group_by_driver(trajs)
    assert set(g) == {2, 5, 7}
    assert len(g[7]) == 2 and len(g[2]) == 1 and len(g[5]) == 1


def test_driver_idxs_for_aligned():
    trajs = [_Stub(7), _Stub(2), _Stub(7)]
    m = build_driver_index(trajs)
    assert driver_idxs_for(trajs, m) == [2, 0, 2]


def test_driver_idxs_for_unknown_raises():
    m = {2: 0}
    with pytest.raises(KeyError):
        driver_idxs_for([_Stub(99)], m)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_drivers.py -q`
Expected: FAIL (`ModuleNotFoundError: drivers`).

- [ ] **Step 3: Implement the helper**

Create `famail_temporal/baselines/gan/drivers.py`:

```python
"""Driver-index map and per-driver grouping for driver-conditioned generation.

`Trajectory.driver_id` is an int in [0, 49]. The generator's driver embedding
is sized by the number of DISTINCT drivers in the training corpus, so we map
each driver_id to a contiguous embedding index (sorted for determinism). The
map is persisted with a run so conditioned generation is reproducible.
"""
from __future__ import annotations
from typing import Dict, List


def build_driver_index(trajectories) -> Dict[int, int]:
    """{driver_id -> contiguous embedding idx}, ordered by sorted driver_id."""
    ids = sorted({int(t.driver_id) for t in trajectories})
    return {did: i for i, did in enumerate(ids)}


def invert_driver_index(driver_to_idx: Dict[int, int]) -> Dict[int, int]:
    """{embedding idx -> driver_id}."""
    return {idx: did for did, idx in driver_to_idx.items()}


def group_by_driver(trajectories) -> Dict[int, List]:
    """{driver_id -> [Trajectory, ...]} (insertion order preserved)."""
    groups: Dict[int, List] = {}
    for t in trajectories:
        groups.setdefault(int(t.driver_id), []).append(t)
    return groups


def driver_idxs_for(trajectories, driver_to_idx: Dict[int, int]) -> List[int]:
    """Index-aligned embedding indices for `trajectories`.

    Raises KeyError (clear message) if a trajectory's driver_id is absent from
    the map — that signals the map was built from a different corpus.
    """
    out: List[int] = []
    for t in trajectories:
        did = int(t.driver_id)
        if did not in driver_to_idx:
            raise KeyError(
                f"driver_id {did} not in driver_to_idx (built from a different "
                f"corpus?); known ids: {sorted(driver_to_idx)[:5]}..."
            )
        out.append(driver_to_idx[did])
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_drivers.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/drivers.py famail_temporal/baselines/tests/test_drivers.py
git commit -m "feat(baselines/gan): driver-index map + per-driver grouping helper"
```

---

### Task 3: Thread `driver_idxs` through training + rollout

**Files:**
- Modify: `famail_temporal/baselines/gan/train_mle.py`
- Modify: `famail_temporal/baselines/gan/gumbel.py`
- Modify: `famail_temporal/baselines/gan/train_adversarial.py`
- Modify: `famail_temporal/baselines/gan/rollout.py`
- Test: `famail_temporal/baselines/tests/test_train_rollout_driver_cond.py`

**Interfaces:**
- Consumes: `TrajectoryLSTM.forward/step/step_embed(..., driver_idx=...)` (Task 1).
- Produces (all new params default `None` → unchanged behavior):
  - `train_mle(model, sequences, contexts, *, epochs, lr, batch_size, device, progress=False, driver_idxs: List[int] | None = None)`
  - `gumbel_rollout(model, ctx_cell, ctx_tblock, *, max_len, tau, device, hard=True, driver_idx: torch.Tensor | None = None)`
  - `adversarial_finetune(..., driver_idxs: List[int] | None = None)`
  - `sample_trajectory_cells(..., driver_idx: int | None = None)`
  - `sample_terminal_cells_batched(..., driver_idx: torch.Tensor | None = None)`
  - `generate_trajectories(model, contexts, *, max_len, device, gen_batch_size=512, temperature=1.0, progress=False, driver_idxs: List[int] | None = None)`
  - `generate_pickups(model, contexts, *, max_len, device, gen_batch_size=gc.GEN_BATCH_SIZE, progress=False, driver_idxs: List[int] | None = None)`

**Index alignment contract:** `driver_idxs` is index-aligned with `sequences`/`contexts`. In `generate_trajectories`/`generate_pickups`, the i-th `driver_idxs` entry conditions the i-th context; per-batch we slice the same `[start:start+gen_batch_size]` window.

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/tests/test_train_rollout_driver_cond.py`:

```python
import torch

from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.train_mle import train_mle
from famail_temporal.baselines.gan.gumbel import gumbel_rollout
from famail_temporal.baselines.gan.rollout import (
    generate_trajectories, generate_pickups,
)

DEV = torch.device("cpu")


def _seqs():
    # tiny in-vocab sequences: [BOS, cell, cell, EOS]
    return [[gc.BOS, 0, 1, gc.EOS], [gc.BOS, 2, 3, gc.EOS], [gc.BOS, 1, 0, gc.EOS]]


def _ctx():
    return [(0, 0), (2, 1), (1, 0)]


def test_train_mle_driver_idxs_runs_and_returns_curves():
    torch.manual_seed(0)
    m = TrajectoryLSTM(n_drivers=2).to(DEV)
    out = train_mle(
        m, _seqs(), _ctx(), epochs=1, lr=1e-3, batch_size=2, device=DEV,
        driver_idxs=[0, 1, 0],
    )
    assert "epoch_losses" in out and "batch_losses" in out
    assert len(out["epoch_losses"]) == 1


def test_train_mle_none_path_unchanged():
    """driver_idxs=None on an unconditioned model trains as before."""
    torch.manual_seed(0)
    m = TrajectoryLSTM().to(DEV)
    out = train_mle(m, _seqs(), _ctx(), epochs=1, lr=1e-3, batch_size=2, device=DEV)
    assert len(out["epoch_losses"]) == 1


def test_gumbel_rollout_accepts_driver_idx():
    torch.manual_seed(0)
    m = TrajectoryLSTM(n_drivers=2).to(DEV)
    cc = torch.tensor([0, 2])
    tb = torch.tensor([0, 1])
    soft, lengths = gumbel_rollout(
        m, cc, tb, max_len=5, tau=1.0, device=DEV,
        driver_idx=torch.tensor([0, 1]),
    )
    assert soft.shape == (2, 5, gc.VOCAB_SIZE)
    assert lengths.shape == (2,)


def test_generate_trajectories_driver_idxs_aligned():
    torch.manual_seed(0)
    m = TrajectoryLSTM(n_drivers=2).to(DEV)
    ctxs = _ctx()
    out = generate_trajectories(
        m, ctxs, max_len=5, device=DEV, gen_batch_size=2, driver_idxs=[0, 1, 0],
    )
    assert len(out) == len(ctxs)


def test_generate_pickups_driver_idxs_aligned():
    torch.manual_seed(0)
    m = TrajectoryLSTM(n_drivers=2).to(DEV)
    ctxs = _ctx()
    out = generate_pickups(
        m, ctxs, max_len=5, device=DEV, gen_batch_size=2, driver_idxs=[0, 1, 0],
    )
    assert len(out) == len(ctxs)


def test_generate_trajectories_none_path_unchanged():
    torch.manual_seed(0)
    m = TrajectoryLSTM().to(DEV)
    ctxs = _ctx()
    out = generate_trajectories(m, ctxs, max_len=5, device=DEV, gen_batch_size=2)
    assert len(out) == len(ctxs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_train_rollout_driver_cond.py -q`
Expected: FAIL (unexpected keyword `driver_idxs` / `driver_idx`).

- [ ] **Step 3a: `train_mle.py`**

Add the parameter and build a per-batch driver tensor. In `train_mle`, add `driver_idxs: List[int] | None = None` to the signature (keyword), and inside the batch loop build the driver tensor for the batch and pass it to the model:

```python
                ctx_tblock = torch.tensor(
                    [contexts[i][1] for i in idx], dtype=torch.long, device=device,
                )
                di = (
                    torch.tensor(
                        [driver_idxs[i] for i in idx], dtype=torch.long, device=device,
                    )
                    if driver_idxs is not None else None
                )
                inp = batch[:, :-1]
                tgt = batch[:, 1:]
                logits = model(inp, ctx_cell, ctx_tblock, driver_idx=di)  # (B, L-1, V)
```

- [ ] **Step 3b: `gumbel.py`**

Add `driver_idx: torch.Tensor | None = None` to `gumbel_rollout` and pass it into `step_embed`:

```python
    for t in range(max_len):
        logits, hidden = model.step_embed(
            prev_embed, cc, tb, hidden, driver_idx=driver_idx,
        )                                                              # (B, V)
```

(`driver_idx` is already on `device` when supplied by callers; it indexes the embedding directly. Add a one-line note in the docstring.)

- [ ] **Step 3c: `train_adversarial.py`**

Add `driver_idxs: List[int] | None = None` to `adversarial_finetune`. Inside the batch loop, after building `cc`/`tb`, build the batch driver tensor and pass it into BOTH gumbel rollouts and the MLE-anchor forward:

```python
            tb = torch.tensor(
                [contexts[i][1] for i in idx], dtype=torch.long, device=device,
            )
            di = (
                torch.tensor(
                    [driver_idxs[i] for i in idx], dtype=torch.long, device=device,
                )
                if driver_idxs is not None else None
            )
```

Then update the three call sites:
- the discriminator-step rollout: `gumbel_rollout(model, cc, tb, max_len=max_len, tau=tau, device=device, hard=True, driver_idx=di)`
- the generator-step rollout: same added `driver_idx=di`
- the MLE-anchor forward: `logits = model(real[:, :-1], cc, tb, driver_idx=di)`

- [ ] **Step 3d: `rollout.py`**

Add `driver_idx` plumbing to the four decoders. For the batched decoders, the caller passes a per-row driver tensor; for `generate_trajectories`/`generate_pickups` the caller passes a `driver_idxs` list that is sliced per batch.

`sample_trajectory_cells` — add `driver_idx: int | None = None`; build `di = torch.tensor([driver_idx], ...)` if set, pass `model.step(tok, cc, tb, hidden, driver_idx=di)`.

`sample_terminal_cells_batched` — add `driver_idx: torch.Tensor | None = None`; pass it straight into `model.step(prev, cc, tb, hidden, driver_idx=driver_idx)`.

`generate_trajectories` — add `driver_idxs: List[int] | None = None`. Per batch window `start:start+gen_batch_size`, build:
```python
        di = (
            torch.tensor(driver_idxs[start : start + gen_batch_size],
                         dtype=torch.long, device=device)
            if driver_idxs is not None else None
        )
```
and call `model.step(prev, cc, tb, hidden, driver_idx=di)`.

`generate_pickups` — add `driver_idxs: List[int] | None = None`. Per batch window build the same `di` and pass it to `sample_terminal_cells_batched(model, cc, tb, max_len=max_len, device=device, driver_idx=di)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_train_rollout_driver_cond.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Regression — existing gan/baseline tests**

Run: `python -m pytest famail_temporal/baselines/tests/ -q -k "smoke or level1 or variance or datasets"`
Expected: PASS (None-path unchanged).

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/gan/train_mle.py famail_temporal/baselines/gan/gumbel.py famail_temporal/baselines/gan/train_adversarial.py famail_temporal/baselines/gan/rollout.py famail_temporal/baselines/tests/test_train_rollout_driver_cond.py
git commit -m "feat(baselines/gan): thread optional driver_idxs through training + rollout"
```

---

### Task 4: Identity-aware Fidelity-A (N-set builder + gate)

**Files:**
- Modify: `famail_temporal/baselines/fidelity_eval.py`
- Test: `famail_temporal/baselines/tests/test_fidelity_eval_identity.py`

**Interfaces:**
- Consumes: HuMID `forward(x1, x2, mask1, mask2, *, profile_1, profile_2)` with 4D `[B,N,L,4]` inputs; `real_to_disc_tensor` / `generated_to_disc_tensor` (existing).
- Produces:
  - `N_TRAJS_PER_BRANCH = 5` (module constant).
  - `build_identity_branch(slot0: Tensor[L,4], context: List[Tensor[Li,4]], *, n=N_TRAJS_PER_BRANCH, rng) -> (set_tensor [N,L*,4], mask [N,L*])` — slot 0 = `slot0`; slots 1..n-1 sampled from `context` (with replacement if `len(context) < n-1`; zero-filled single step if `context` empty), each padded to the branch's max length. Pure (rng injected).
  - `_pad_sets_to_batch(branches_left, branches_right, profiles_left, profiles_right, device) -> (x1[B,N,Lmax,4], x2, m1[B,N,Lmax], m2, p1[B,11], p2)` — pads N-set branches across the batch.
  - `humid_identity_fidelity(disc, pairs, *, batch_size=64, device=None) -> {mean, std, n}` where each `pair` is `((set_l, mask_l, prof_l), (set_r, mask_r, prof_r))`; forward-only, `disc.train(False)`, `profile_*` passed, driving=None.
  - `identity_validation_gate(disc, *, matched_pairs, mismatched_pairs, batch_size=64, device=None, margin=GATE_MARGIN) -> {high_matched, low_mismatched, margin, passed, n_matched, n_mismatched}`.

- [ ] **Step 1: Write the failing tests** (stub discriminator; no GPU)

Create `famail_temporal/baselines/tests/test_fidelity_eval_identity.py`:

```python
import random
import numpy as np
import torch

from famail_temporal.baselines import fidelity_eval as fe


def _traj_tensor(L, base):
    # [L,4] with distinguishable coords; +1 already applied by convention
    return torch.tensor(
        [[base + i + 1.0, base + i + 1.0, 10.0, 1.0] for i in range(L)],
        dtype=torch.float32,
    )


def _profile(v):
    return np.full(11, float(v), dtype=np.float32)


class _ProfileSameStub(torch.nn.Module):
    """Returns high prob iff the two branches' profiles are (near) equal.

    Stands in for an identity discriminator: same driver (same profile) -> ~1,
    different driver -> ~0. Ignores trajectories; exercises the plumbing + gate.
    """
    def forward(self, x1, x2, mask1=None, mask2=None, *, profile_1=None,
                profile_2=None, **kw):
        b = x1.shape[0]
        if profile_1 is None or profile_2 is None:
            return torch.full((b, 1), 0.5)
        same = (profile_1 - profile_2).abs().sum(dim=-1) < 1e-6
        return torch.where(same, torch.full((b,), 0.95),
                           torch.full((b,), 0.05)).unsqueeze(-1)


def _branch(slot0_base, ctx_bases, rng):
    slot0 = _traj_tensor(4, slot0_base)
    ctx = [_traj_tensor(3, b) for b in ctx_bases]
    return fe.build_identity_branch(slot0, ctx, rng=rng)


def test_build_identity_branch_shapes_and_slot0():
    rng = random.Random(0)
    s, m = _branch(0, [10, 20, 30, 40], rng)
    assert s.shape[0] == fe.N_TRAJS_PER_BRANCH
    assert m.shape[0] == fe.N_TRAJS_PER_BRANCH
    # slot 0's first real step keeps its identity coord (0+1)
    assert float(s[0, 0, 0]) == 1.0
    assert bool(m[0, 0])


def test_build_identity_branch_samples_with_replacement_when_short():
    rng = random.Random(0)
    s, m = _branch(0, [10], rng)   # only 1 context, needs n-1
    assert s.shape[0] == fe.N_TRAJS_PER_BRANCH


def test_identity_fidelity_high_for_same_profile():
    rng = random.Random(0)
    disc = _ProfileSameStub()
    sl, ml = _branch(0, [10, 20, 30, 40], rng)
    sr, mr = _branch(5, [10, 20, 30, 40], rng)
    p = _profile(1)
    pairs = [((sl, ml, p), (sr, mr, p))]   # same profile
    out = fe.humid_identity_fidelity(disc, pairs)
    assert out["mean"] > 0.9 and out["n"] == 1


def test_identity_gate_passes_when_matched_above_mismatched():
    rng = random.Random(0)
    disc = _ProfileSameStub()
    sl, ml = _branch(0, [10, 20, 30, 40], rng)
    sr, mr = _branch(5, [10, 20, 30, 40], rng)
    p_d, p_dp = _profile(1), _profile(2)
    matched = [((sl, ml, p_d), (sr, mr, p_d))]        # same driver
    mismatched = [((sl, ml, p_d), (sr, mr, p_dp))]    # different driver
    gate = fe.identity_validation_gate(
        disc, matched_pairs=matched, mismatched_pairs=mismatched,
    )
    assert gate["passed"] is True
    assert gate["high_matched"] > gate["low_mismatched"]


def test_identity_gate_fails_for_constant_discriminator():
    class _Const(torch.nn.Module):
        def forward(self, x1, x2, mask1=None, mask2=None, **kw):
            return torch.full((x1.shape[0], 1), 0.7)
    rng = random.Random(0)
    sl, ml = _branch(0, [10, 20, 30, 40], rng)
    sr, mr = _branch(5, [10, 20, 30, 40], rng)
    p = _profile(1)
    matched = [((sl, ml, p), (sr, mr, p))]
    mismatched = [((sl, ml, p), (sr, mr, p))]
    gate = fe.identity_validation_gate(
        _Const(), matched_pairs=matched, mismatched_pairs=mismatched,
    )
    assert gate["passed"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_fidelity_eval_identity.py -q`
Expected: FAIL (`build_identity_branch` / `humid_identity_fidelity` / `identity_validation_gate` undefined).

- [ ] **Step 3: Implement the identity Fidelity-A block**

Append to `famail_temporal/baselines/fidelity_eval.py` (after the existing HuMID section). Add `import random` at the top if not present (it is not — add `import random` to the stdlib imports).

```python
# --------------------------------------------- identity (N-set) Fidelity-A ----
# Mirrors fidelity/context.py: a HuMID branch is N=5 seeking trajectories ---
# slot 0 = the trajectory under test, slots 1..N-1 = the SAME driver's real
# context trajectories --- plus that driver's real 11-dim profile. Driving is
# omitted from BOTH branches (symmetric graceful degradation: the model falls
# back to its fixed zero driving_default_embedding). This is exactly how HuMID
# scores fidelity inside the editing algorithm, so generated/edited trajectories
# are evaluated near the discriminator's trained regime (unlike v1's single
# seeking-only trajectory, which was deeply OOD and failed the gate).

N_TRAJS_PER_BRANCH = 5


def _pad_one_set(trajs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of [Li,4] trajectories to ([N, Lmax, 4], mask [N, Lmax])."""
    n = len(trajs)
    lmax = max(t.shape[0] for t in trajs)
    out = torch.zeros(n, lmax, 4, dtype=torch.float32)
    mask = torch.zeros(n, lmax, dtype=torch.bool)
    for i, t in enumerate(trajs):
        li = t.shape[0]
        out[i, :li] = t
        mask[i, :li] = True
    return out, mask


def build_identity_branch(
    slot0: torch.Tensor,
    context: List[torch.Tensor],
    *,
    n: int = N_TRAJS_PER_BRANCH,
    rng: "random.Random",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build one N-trajectory HuMID branch: ([N, Lmax, 4], mask [N, Lmax]).

    slot 0 = ``slot0`` (the trajectory under test); slots 1..n-1 are sampled
    from ``context`` (the driver's real trajectories). If fewer than n-1
    context trajectories are available they are sampled WITH REPLACEMENT; if
    ``context`` is empty a single zero step is used (the model masks it out).
    ``rng`` is injected for determinism. Coordinate convention is the caller's:
    pass tensors already built via real_to_disc_tensor / generated_to_disc_tensor.
    """
    need = n - 1
    if len(context) == 0:
        ctx = [torch.zeros(1, 4, dtype=torch.float32) for _ in range(need)]
    elif len(context) >= need:
        ctx = rng.sample(context, need)
    else:
        ctx = [context[rng.randrange(len(context))] for _ in range(need)]
    return _pad_one_set([slot0] + ctx)


def _pad_sets_to_batch(pairs, device):
    """Stack identity-branch pairs into batched 4D HuMID inputs + profiles.

    Each pair is ((set_l [N,Ll,4], mask_l [N,Ll], prof_l [11]),
                  (set_r [N,Lr,4], mask_r [N,Lr], prof_r [11])).
    Returns (x1 [B,N,Lmax,4], x2, m1 [B,N,Lmax], m2, p1 [B,11], p2).
    """
    n = pairs[0][0][0].shape[0]
    lmax = max(
        max(p[0][0].shape[1] for p in pairs),
        max(p[1][0].shape[1] for p in pairs),
    )
    b = len(pairs)
    x1 = torch.zeros(b, n, lmax, 4, dtype=torch.float32)
    x2 = torch.zeros(b, n, lmax, 4, dtype=torch.float32)
    m1 = torch.zeros(b, n, lmax, dtype=torch.bool)
    m2 = torch.zeros(b, n, lmax, dtype=torch.bool)
    p1 = torch.zeros(b, 11, dtype=torch.float32)
    p2 = torch.zeros(b, 11, dtype=torch.float32)
    for i, ((sl, ml, pl), (sr, mr, pr)) in enumerate(pairs):
        x1[i, :, : sl.shape[1]] = sl
        m1[i, :, : ml.shape[1]] = ml
        x2[i, :, : sr.shape[1]] = sr
        m2[i, :, : mr.shape[1]] = mr
        p1[i] = torch.as_tensor(pl, dtype=torch.float32)
        p2[i] = torch.as_tensor(pr, dtype=torch.float32)
    return (x1.to(device), x2.to(device), m1.to(device), m2.to(device),
            p1.to(device), p2.to(device))


def _score_identity_pairs(disc, pairs, *, batch_size, device) -> np.ndarray:
    probs: List[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            x1, x2, m1, m2, p1, p2 = _pad_sets_to_batch(chunk, device)
            out = disc(x1, x2, mask1=m1, mask2=m2, profile_1=p1, profile_2=p2)
            probs.extend(out.reshape(-1).detach().cpu().tolist())
    return np.asarray(probs, dtype=np.float64)


def humid_identity_fidelity(
    disc, pairs, *, batch_size: int = 64, device: torch.device | None = None,
) -> Dict[str, float]:
    """Mean same-agent probability over N-set identity-branch pairs."""
    device = device or torch.device("cpu")
    if not pairs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    disc.train(False)   # defensive inference mode (see humid_paired_fidelity)
    probs = _score_identity_pairs(disc, pairs, batch_size=batch_size, device=device)
    return {
        "mean": float(probs.mean()),
        "std": float(probs.std(ddof=1)) if probs.size > 1 else 0.0,
        "n": int(probs.size),
    }


def identity_validation_gate(
    disc, *, matched_pairs, mismatched_pairs,
    batch_size: int = 64, device: torch.device | None = None,
    margin: float = GATE_MARGIN,
) -> Dict[str, object]:
    """Real-anchored well-posedness gate for identity Fidelity-A.

    Passes iff high_matched - low_mismatched >= margin AND high_matched >
    low_mismatched. ``matched_pairs`` are same-driver branch pairs (expected
    high); ``mismatched_pairs`` are different-driver pairs (expected low). All
    means are returned regardless (an empty list -> NaN mean -> passed=False).
    """
    device = device or torch.device("cpu")
    high = humid_identity_fidelity(disc, matched_pairs, batch_size=batch_size, device=device)
    low = humid_identity_fidelity(disc, mismatched_pairs, batch_size=batch_size, device=device)
    passed = bool((high["mean"] - low["mean"]) >= margin and high["mean"] > low["mean"])
    return {
        "high_matched": float(high["mean"]),
        "low_mismatched": float(low["mean"]),
        "margin": float(margin),
        "passed": passed,
        "n_matched": high["n"],
        "n_mismatched": low["n"],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_fidelity_eval_identity.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Regression — existing fidelity_eval tests**

Run: `python -m pytest famail_temporal/baselines/tests/test_fidelity_eval_humid.py famail_temporal/baselines/tests/test_fidelity_eval_builders.py -q`
Expected: PASS (no change to v1 functions).

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/fidelity_eval.py famail_temporal/baselines/tests/test_fidelity_eval_identity.py
git commit -m "feat(baselines): identity-aware Fidelity-A (N-set branch builder + real-anchored gate)"
```

---

### Task 5: Enriched Fidelity-B (RoG, net-displacement, terminal-cell JS)

**Files:**
- Modify: `famail_temporal/baselines/fidelity_eval.py`
- Test: `famail_temporal/baselines/tests/test_fidelity_eval_enriched.py`

**Interfaces:**
- Consumes: `transmission.terminal_cell_histogram`, `transmission.jensen_shannon_divergence` (existing).
- Produces:
  - `trajectory_statistics(...)` now returns 5 keys: existing `{length, mean_displacement, coverage}` + `radius_of_gyration` + `net_displacement` (both 0.0 if length < 2).
  - `_STAT_KEYS_V2 = ("length", "mean_displacement", "coverage", "radius_of_gyration", "net_displacement")`.
  - `stat_ranges(stat_lists, *, keys=_STAT_KEYS)` — key-parameterized (default = original 3 → v1 unchanged).
  - `distributional_fidelity(source_stats, raw_stats, *, bins=BINS, ranges=None, keys=_STAT_KEYS)` — aggregate over `keys` (default 3 → v1 unchanged).
  - `terminal_cell_distribution_js(source_pickups, raw_pickups, *, n_cells=gc.N_CELLS) -> float` — JS (bits) between two terminal-cell histograms; each input is an iterable of `(x, y, t_block)` pickup tuples.

**Backward-compat note:** `_STAT_KEYS` stays the original 3 so existing v1 callers/tests (`test_fidelity_eval_distributional.py`, `run_level1_table.py`) are unaffected. The two new statistics are additive keys on the returned dict.

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/tests/test_fidelity_eval_enriched.py`:

```python
import numpy as np

from famail_temporal.baselines import fidelity_eval as fe


def test_trajectory_statistics_has_new_keys():
    # straight line (0,0)->(3,0): RoG of x-coords {0,1,2,3} about mean 1.5,
    # y all 0 -> RoG = sqrt(mean((x-1.5)^2)) = sqrt(1.25); net disp = 3.0
    cells = [fe.gc.GY * 0 + 0, fe.gc.GY * 1 + 0, fe.gc.GY * 2 + 0, fe.gc.GY * 3 + 0]
    s = fe.trajectory_statistics(cells)
    assert set(s) >= {"length", "mean_displacement", "coverage",
                      "radius_of_gyration", "net_displacement"}
    assert abs(s["net_displacement"] - 3.0) < 1e-6
    assert abs(s["radius_of_gyration"] - np.sqrt(1.25)) < 1e-6


def test_short_trajectory_zero_rog_and_netdisp():
    s = fe.trajectory_statistics([fe.gc.GY * 5 + 5])   # length 1
    assert s["radius_of_gyration"] == 0.0
    assert s["net_displacement"] == 0.0


def test_distributional_fidelity_default_keys_unchanged():
    """Default (3-key) aggregate is unchanged (v1 backward-compat)."""
    src = [{"length": 2, "mean_displacement": 1.0, "coverage": 2,
            "radius_of_gyration": 0.5, "net_displacement": 1.0}]
    raw = [{"length": 2, "mean_displacement": 1.0, "coverage": 2,
            "radius_of_gyration": 0.5, "net_displacement": 1.0}]
    out = fe.distributional_fidelity(src, raw)
    assert set(out["per_stat"]) == set(fe._STAT_KEYS)   # only the original 3
    assert out["aggregate"] == 0.0                       # identical -> 0


def test_distributional_fidelity_v2_keys():
    src = [{"length": 2, "mean_displacement": 1.0, "coverage": 2,
            "radius_of_gyration": 0.5, "net_displacement": 1.0}]
    raw = [{"length": 9, "mean_displacement": 4.0, "coverage": 9,
            "radius_of_gyration": 3.0, "net_displacement": 8.0}]
    out = fe.distributional_fidelity(src, raw, keys=fe._STAT_KEYS_V2)
    assert set(out["per_stat"]) == set(fe._STAT_KEYS_V2)  # all 5
    assert out["aggregate"] > 0.0


def test_terminal_cell_distribution_js_identical_zero():
    pk = [(1, 2, 0), (3, 4, 1), (1, 2, 2)]
    assert fe.terminal_cell_distribution_js(pk, pk) == 0.0


def test_terminal_cell_distribution_js_disjoint_high():
    a = [(1, 2, 0), (1, 2, 1)]
    b = [(10, 20, 0), (10, 20, 1)]
    js = fe.terminal_cell_distribution_js(a, b)
    assert js > 0.9   # disjoint support -> ~1 bit
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_fidelity_eval_enriched.py -q`
Expected: FAIL (new keys / `_STAT_KEYS_V2` / `terminal_cell_distribution_js` missing).

- [ ] **Step 3a: Extend `trajectory_statistics`**

In `fidelity_eval.py`, extend the function to compute the two new statistics. Replace the body's return so it adds the keys (keep the existing length/coverage/mean_disp logic):

```python
    length = len(xy)
    coverage = len(set(xy))
    if length < 2:
        mean_disp = 0.0
        rog = 0.0
        net_disp = 0.0
    else:
        dists = [
            float(np.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1]))
            for i in range(length - 1)
        ]
        mean_disp = float(np.mean(dists))
        pts = np.asarray(xy, dtype=np.float64)            # [L, 2]
        cm = pts.mean(axis=0)
        rog = float(np.sqrt(np.mean(np.sum((pts - cm) ** 2, axis=1))))
        net_disp = float(np.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1]))
    return {
        "length": length,
        "mean_displacement": mean_disp,
        "coverage": coverage,
        "radius_of_gyration": rog,
        "net_displacement": net_disp,
    }
```

Update the docstring to mention the two new statistics.

- [ ] **Step 3b: Key-parameterize ranges + distributional fidelity**

Add the v2 key tuple next to `_STAT_KEYS`:

```python
_STAT_KEYS = ("length", "mean_displacement", "coverage")
_STAT_KEYS_V2 = (
    "length", "mean_displacement", "coverage",
    "radius_of_gyration", "net_displacement",
)
```

Change `stat_ranges` to accept `keys`:

```python
def stat_ranges(
    stat_lists: List[List[Dict[str, float]]], *, keys: tuple = _STAT_KEYS,
) -> Dict[str, tuple]:
    ranges: Dict[str, tuple] = {}
    for key in keys:
        vals = [float(s[key]) for stats in stat_lists for s in stats]
        ranges[key] = (min(vals), max(vals)) if vals else (0.0, 0.0)
    return ranges
```

Change `distributional_fidelity` to accept `keys` and iterate over it (replace the two `for key in _STAT_KEYS:` / aggregate lines):

```python
def distributional_fidelity(
    source_stats: List[Dict[str, float]],
    raw_stats: List[Dict[str, float]],
    *,
    bins: int = BINS,
    ranges: Dict[str, tuple] | None = None,
    keys: tuple = _STAT_KEYS,
) -> Dict[str, object]:
    ...
    per_stat: Dict[str, float] = {}
    for key in keys:
        src = [float(s[key]) for s in source_stats]
        raw = [float(s[key]) for s in raw_stats]
        if ranges is not None:
            lo, hi = ranges[key]
        else:
            pooled = src + raw
            lo, hi = (min(pooled), max(pooled)) if pooled else (0.0, 0.0)
        p = _hist(src, lo, hi, bins)
        q = _hist(raw, lo, hi, bins)
        per_stat[key] = float(jensen_shannon_divergence(p, q))
    aggregate = float(np.mean([per_stat[k] for k in keys]))
    return {"per_stat": per_stat, "aggregate": aggregate}
```

Keep the existing non-empty guard and docstring (update the docstring to note `keys`).

- [ ] **Step 3c: Add `terminal_cell_distribution_js`**

Add the import at the top of the module (next to the existing transmission import):

```python
from famail_temporal.baselines.transmission import (
    jensen_shannon_divergence, terminal_cell_histogram,
)
```

Then add the function (after `distributional_fidelity`):

```python
def terminal_cell_distribution_js(
    source_pickups, raw_pickups, *, n_cells: int = gc.N_CELLS,
) -> float:
    """JS divergence (bits) between two terminal-cell (pickup) distributions.

    Each input is an iterable of (x, y, t_block) pickup tuples (only (x, y) is
    used). Reuses transmission.terminal_cell_histogram + jensen_shannon_
    divergence. 0.0 if the two distributions are identical; ~1.0 if disjoint.
    """
    p = terminal_cell_histogram(source_pickups, n_cells=n_cells)
    q = terminal_cell_histogram(raw_pickups, n_cells=n_cells)
    return float(jensen_shannon_divergence(p, q))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_fidelity_eval_enriched.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Regression — v1 distributional tests**

Run: `python -m pytest famail_temporal/baselines/tests/test_fidelity_eval_distributional.py -q`
Expected: PASS (default-key path unchanged).

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/fidelity_eval.py famail_temporal/baselines/tests/test_fidelity_eval_enriched.py
git commit -m "feat(baselines): enriched Fidelity-B (radius-of-gyration, net-displacement, terminal-cell JS)"
```

---

### Task 6: v2 orchestrator `run_level1_table_v2.py`

**Files:**
- Create: `famail_temporal/baselines/run_level1_table_v2.py`
- Test: `famail_temporal/baselines/tests/test_run_level1_table_v2.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5; `DataBundle.load()`; `data_level_fairness`; `load_discriminator`; `trajectory_context`/`trajectory_to_tokens`; `set_all_seeds`.
- Produces (pure, unit-testable helpers + a `main(argv)` CLI):
  - `result_to_json(result) -> str`, `render_table_v2(result) -> str`.
  - `_train_and_generate_cond(train_trajectories, driver_to_idx, *, adv_epochs, gan_loss, n_critic, mle_epochs, max_len, max_tokens, device, seed) -> dict` — like v1's `_train_and_generate` but builds `driver_idxs` aligned with `filtered_train` and constructs the model with `n_drivers=len(driver_to_idx)`; returns `{model, filtered_train, contexts, driver_idxs, mle_curve, adv_curve}`.
  - `_edited_fairness_from_metrics(edit_dir)` — copy of the v1 fix.
  - `_curves_for_source(src)` — copy of the v1 helper.
  - `_select_eval_drivers(group_by_driver_map, *, min_trajs, max_drivers) -> List[int]` — drivers with enough real trajectories, deterministic order, capped.
  - `_real_context_tensors(real_trajs) -> List[Tensor]` — `real_to_disc_tensor` for each.
  - `_build_source_pairs(...)` — assemble matched + mismatched identity-branch pairs per source (pure given pre-built tensors + rng).

**The orchestrator's Fidelity-A flow** (documented inline):
1. `driver_to_idx = build_driver_index(raw_trajs)`, `groups = group_by_driver(raw_trajs)`, `profiles = bundle.multi_stream.profile_features`.
2. `eval_drivers = _select_eval_drivers(groups, min_trajs=6, max_drivers=args.max_eval_drivers)`. For each `d`, pick `d' = next eval driver (wrap-around)`.
3. Per driver `d`: build real context tensors from `groups[d]`; sample `pairs_per_driver` matched slot0 pairs; for raw, slot0_a/slot0_b are two distinct real-d trajectories; for edited, slot0_b is an edited-d trajectory; for bc/gan, slot0_b is a gen-cond-d trajectory.
4. Conditioned generation for the Fidelity-A sets: for each eval driver, take up to `pairs_per_driver` of `d`'s real contexts and `generate_trajectories(model, ctxs_d, ..., driver_idxs=[idx_d]*len)` → gen-cond-d slot0 tensors (via `generated_to_disc_tensor`, time/day from the paired real context's first state).
5. Matched pairs use profile `d` on both branches; mismatched pairs put driver `d` on the real branch and driver `d'` on the source branch (slot0 from source-of-d', context d', profile d').
6. Gate = `identity_validation_gate(disc, matched=raw_matched, mismatched=raw_mismatched)`. `trusted = gate.passed`.
7. Per source `S`: `fidelity_a(S) = humid_identity_fidelity(disc, matched_pairs[S]).mean`; `separation(S) = fidelity_a(S) - humid_identity_fidelity(disc, mismatched_pairs[S]).mean`.

**Fidelity-B flow:** generate full trajectories + pickups per source over the fidelity sample (driver-conditioned via the aligned `driver_idxs[:n]`); compute 5-key `distributional_fidelity(keys=_STAT_KEYS_V2)` on a shared grid (`stat_ranges(keys=_STAT_KEYS_V2)`); compute `terminal_cell_distribution_js(source_pickups, raw_pickups)`; `fidelity_b = mean(list(per_stat.values()) + [terminal_cell_js])`.

**Fairness flow:** identical to v1 (driver-conditioned pickups for bc/gan via `generate_pickups(..., driver_idxs=...)`, edited from `metrics_after`, raw from `data_level_fairness(bundle)`).

- [ ] **Step 1: Write the failing test** (pure helpers + render + a stubbed end-to-end on the synthetic bundle is out of scope; test the pure pieces)

Create `famail_temporal/baselines/tests/test_run_level1_table_v2.py`:

```python
import json
import random

import torch

from famail_temporal.baselines import run_level1_table_v2 as r2
from famail_temporal.baselines import fidelity_eval as fe


def test_render_table_v2_contains_all_sources_and_gate():
    result = {
        "edit_dir": "x",
        "gate": {"high_matched": 0.9, "low_mismatched": 0.3, "margin": 0.2,
                 "passed": True, "n_matched": 10, "n_mismatched": 10},
        "n_eval_drivers": 5,
        "sources": {
            k: {
                "f_causal": 0.8, "f_spatial": 0.08,
                "fidelity_a": 0.7, "fidelity_a_separation": 0.4,
                "fidelity_a_trusted": True,
                "fidelity_b": 0.05,
                "fidelity_b_per_component": {"length": 0.01, "terminal_cell": 0.02},
                "n_empty": 0,
            } for k in ("raw", "edited", "bc", "gan")
        },
    }
    md = r2.render_table_v2(result)
    assert "PASSED" in md
    for k in ("raw", "edited", "bc", "gan"):
        assert k in md
    # round-trips as JSON
    assert json.loads(r2.result_to_json(result))["gate"]["passed"] is True


def test_select_eval_drivers_filters_and_caps():
    class _T:
        def __init__(self, d): self.driver_id = d
    groups = {0: [_T(0)] * 10, 1: [_T(1)] * 3, 2: [_T(2)] * 8, 3: [_T(3)] * 7}
    out = r2._select_eval_drivers(groups, min_trajs=6, max_drivers=2)
    assert out == [0, 2]          # driver 1 (only 3) excluded; sorted; capped to 2


def test_build_source_pairs_alignment_smoke():
    """matched/mismatched pair lists are equal length and well-formed."""
    rng = random.Random(0)
    def _tt(base, L=4):
        return torch.tensor(
            [[base + i + 1.0, base + i + 1.0, 10.0, 1.0] for i in range(L)],
            dtype=torch.float32,
        )
    import numpy as np
    real_ctx = [_tt(10), _tt(20), _tt(30), _tt(40)]
    prof_d = np.zeros(11, dtype=np.float32)
    prof_dp = np.ones(11, dtype=np.float32)
    matched, mismatched = r2._build_source_pairs(
        real_slot0=[_tt(0), _tt(1)],
        source_slot0=[_tt(5), _tt(6)],
        real_context=real_ctx,
        source_context_other=real_ctx,
        profile_d=prof_d, profile_dp=prof_dp, rng=rng,
    )
    assert len(matched) == 2 and len(mismatched) == 2
    # each pair is ((set,mask,prof),(set,mask,prof))
    (sl, ml, pl), (sr, mr, pr) = matched[0]
    assert sl.shape[0] == fe.N_TRAJS_PER_BRANCH
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_level1_table_v2.py -q`
Expected: FAIL (`run_level1_table_v2` missing).

- [ ] **Step 3: Implement the v2 orchestrator**

Create `famail_temporal/baselines/run_level1_table_v2.py`. This is the longest task; build it from the v1 orchestrator's shape. Key blocks below — the implementer fills the `main()` glue following the documented flow.

Header + imports:

```python
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
```

`result_to_json`, `render_table_v2`:

```python
def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2, default=float)


def render_table_v2(result: dict) -> str:
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
```

The v1-copied helpers `_edited_fairness_from_metrics` and `_curves_for_source` — **copy them verbatim** from `run_level1_table.py` (lines 144-183).

Pure Fidelity-A helpers:

```python
def _select_eval_drivers(groups, *, min_trajs: int, max_drivers: int) -> List[int]:
    """Drivers (sorted) with >= min_trajs real trajectories, capped to max_drivers."""
    eligible = sorted(d for d, ts in groups.items() if len(ts) >= min_trajs)
    return eligible[:max_drivers]


def _real_context_tensors(real_trajs) -> List[torch.Tensor]:
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
```

`_train_and_generate_cond` — based on v1's `_train_and_generate`, with driver conditioning:

```python
def _train_and_generate_cond(
    train_trajectories, driver_to_idx, *,
    adv_epochs, gan_loss, n_critic, mle_epochs, max_len, max_tokens, device, seed,
) -> dict:
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
```

The `main(argv)` glue assembles, in order: device + checkpoint guard (`disc = load_discriminator(ckpt).to(device)`); `DataBundle.load()`; load `histories.pkl`; `driver_to_idx`, `groups`, `profiles`; train BC + GAN conditioned (`_train_and_generate_cond`); generate per-source gen-cond-d slot0 sets for eval drivers; build matched/mismatched pairs per source via `_build_source_pairs`; gate from raw; per-source `fidelity_a` + `separation`; Fidelity-B (5-key distributional + terminal-cell JS over the fidelity sample); fairness (raw/edited/bc/gan); assemble `result`; persist `level1_v2_metrics.json`, `level1_v2_table.md`, `training_curves.json`, `trajectory_stats.npz`, and `driver_index.json`. Mirror v1's persistence/CLI-args block (add `--max-eval-drivers` default 50, `--pairs-per-driver` default 20, `--min-driver-trajs` default 6).

Persist the driver map:
```python
    (out_dir / "driver_index.json").write_text(
        json.dumps({str(k): v for k, v in driver_to_idx.items()}, indent=2)
    )
```

CLI args block: copy v1's argparse, add the three new args, keep `--edit-dir` default, `--mle-epochs` 20, `--adv-epochs` 3, `--gan-loss wgan-gp`, `--n-critic 5`, `--fidelity-sample-size 5000`, `--seed 0`, `--device auto`, `--out-dir`, `--quiet`. The persistence default dir = `config.PACKAGE_ROOT / "results" / "level1_table_v2" / <stamp>`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_level1_table_v2.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Import + arg-parse smoke**

Run: `python -m famail_temporal.baselines.run_level1_table_v2 --help`
Expected: prints usage including `--max-eval-drivers`, `--pairs-per-driver`, `--min-driver-trajs` (no import errors).

- [ ] **Step 6: Full baselines suite (regression)**

Run: `python -m pytest famail_temporal/baselines/tests/ -q`
Expected: PASS (all v1 + v2 unit tests).

- [ ] **Step 7: Commit**

```bash
git add famail_temporal/baselines/run_level1_table_v2.py famail_temporal/baselines/tests/test_run_level1_table_v2.py
git commit -m "feat(baselines): Level-1 v2 orchestrator (driver-conditioned + identity Fidelity-A + enriched Fidelity-B)"
```

---

## Manual experiment + documentation phase (controller-executed, not a TDD subagent task)

After all six tasks pass review:

- [ ] **GPU smoke (cheap) BEFORE the long run.** Verify the HuMID multi-stream forward (seeking N=5 + profile, driving=None) on the real checkpoint with a tiny matched/mismatched set on `cuda`, and a 1-epoch driver-conditioned BC train + a handful of conditioned rollouts. (Catches device/shape bugs like v1's `.to(device)`.) Run a 2-epoch, `--max-eval-drivers 4 --pairs-per-driver 3 --fidelity-sample-size 200 --mle-epochs 2 --adv-epochs 1` invocation end-to-end on GPU; confirm a populated table + gate verdict.
- [ ] **Full run** (background `nohup`, log OUTSIDE `/tmp` e.g. `~/level1_v2_run.log`, harness-tracked `tail --pid` waiter): `--mle-epochs 20 --adv-epochs 3 --gan-loss wgan-gp --n-critic 5 --device auto`. Expect ~40+ min.
- [ ] **Plot curves** via `plot_training_curves.py` then `replot_training_curves.py` (legible PNGs) pointed at the v2 `training_curves.json`.
- [ ] **Write `famail_temporal/baselines/LEVEL1_V2_RESULTS.md`** — the v2 table, the gate verdict (did real-anchored Fidelity-A become trustworthy?), per-source separation diagnostics, Fidelity-B per-component breakdown, comparison to v1 (raw 0.8052 / edited 0.8180 / BC 0.8064 / GAN 0.8212; v1 gate FAILED), and any empirical surprises (surfaced, not silently fixed).
- [ ] **Write `famail_temporal/baselines/LEVEL1_V2_METHODOLOGY.md`** — paper-ready prose: the driver-conditioned generation architecture (additive embedding), the identity-aware Fidelity-A construction (slot-0 + real context + profile, mirroring HuMID's editing-algorithm usage; the real-anchored gate; the matched/mismatched separation; the gate-anchor deviation from spec §3.4 and why), and the enriched Fidelity-B statistics (definitions of RoG, net-displacement, terminal-cell JS).
- [ ] **Update memory** `project_paper_argument.md` with v2 headline; mark the v2-pickup task COMPLETED.
- [ ] **Final code review** (subagent) over the whole v2 diff; then `superpowers:finishing-a-development-branch`.

---

## Self-Review (against the spec)

**Spec coverage:**
- §4 Driver-conditioned generation → Tasks 1 (generator), 2 (driver map), 3 (training/rollout plumbing). ✓
- §5 Identity-aware Fidelity-A (N-set seeking+profile, driving omitted, matched-vs-mismatched gate) → Task 4 + orchestrator flow in Task 6. ✓ (Branch construction mirrors context.py per user confirmation; gate is real-anchored superset per user confirmation.)
- §5.3 single-trajectory fallback → not needed: the N-set build is implemented and verified; `build_identity_branch` already degrades gracefully (zero context). Documented.
- §6 Enriched Fidelity-B (RoG, net-disp, terminal-cell JS, aggregate) → Task 5 + orchestrator. ✓
- §7 Integration / v2 table + persistence → Task 6. ✓
- §10 Error handling (unknown driver raises; <N trajs sample-with-replacement; missing profile → zeros; driver_idx=None regression; empty rollout excluded; length<2 → 0.0; gate may fail non-fatally) → covered across Tasks 1–6. ✓
- §11 Testing (generator regression, driver map, identity Fidelity-A + gate with stub, enriched stats, orchestrator helpers) → one test file per task. ✓

**Deviations (user-confirmed):** gate anchor = real-anchored superset (vs spec §3.4 literal); branch = context.py-mirrored slot0+real-context (vs all-N-from-source). Both persist all raw means so the alternative interpretation is recoverable. Documented in the methodology doc.

**Placeholder scan:** none — every code step shows complete code; the only "fill-in" is `main()`'s glue, which is fully specified by the documented ordered flow + the v1 orchestrator it mirrors (a single integration task, intentionally not duplicated line-by-line to avoid drift from v1's persistence block).

**Type consistency:** `driver_idx` is a `(B,)` long tensor everywhere; `driver_idxs` is a `List[int]`; identity pairs are `((set[N,L,4], mask[N,L], profile[11]), (...))`; `_STAT_KEYS` (3) vs `_STAT_KEYS_V2` (5) are distinct and used consistently (v1 default vs v2 explicit).
