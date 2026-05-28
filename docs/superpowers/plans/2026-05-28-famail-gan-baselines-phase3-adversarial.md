# FAMAIL GAN Baselines — Phase 3: Adversarial Training Subsystem + Standard-Adversarial B0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the adversarial training stage to the MLE generator from Phase 2 — a differentiable Gumbel-softmax rollout, a real-vs-fake sequence critic, and a non-saturating GAN fine-tune loop — then wire them into a `fit_and_evaluate` orchestrator that completes the spec's *standard-adversarial* B0 (MLE pretrain → adversarial fine-tune → generate → grid → fairness).

**Architecture:** Three new modules in the existing `famail_temporal/baselines/gan/` subpackage plus one generator extension. The generator gains a `step_embed` method that decodes from a precomputed input embedding, so a straight-through Gumbel-softmax rollout can feed `soft_onehot @ cell_embed.weight` back as the next input (gradients flow across the discrete sequence). A purpose-built `SequenceCritic` — an LSTM over the *same* cell-token vocabulary as the generator — scores realism of both hard real sequences (`forward_ids`) and soft generated sequences (`forward_soft`). `adversarial_finetune` alternates a discriminator step and a non-saturating generator step with an annealed Gumbel temperature, starting from MLE-pretrained weights. `fit_and_evaluate` chains MLE → adversarial → the existing Phase-2 rollout→grid→fairness seam.

**Tech Stack:** Python 3.12, PyTorch (`nn.Embedding`/`nn.LSTM`/`nn.Linear`, Adam, `F.gumbel_softmax`, `BCEWithLogitsLoss`), NumPy, pytest. Reuses Phase 2's `TrajectoryLSTM`, `train_mle`, `generate_pickups`, `pickups_to_pickup_3d`, `trajectory_to_tokens`, `trajectory_context`, `_pad_batch`; Phase 1's `data_level_fairness`; and `famail_temporal.utils.seeding.set_all_seeds`, `famail_temporal.data.loader.DataBundle`.

---

## Scope: this plan = the adversarial subsystem + standard-adversarial B0

Phase 2 delivered the MLE generator and explicitly deferred "the adversarial discriminator + Gumbel-softmax fine-tune" to Phase 3. This plan delivers exactly that, and uses it to complete the spec's B0 (which is defined as *standard adversarial* training, spec §4.3). It does **not** include:

- **B1 differentiable fairness loss** — deferred to **Phase 4**. The hard part (a differentiable `soft_pickup_3d` scattered from Gumbel rollouts, fed to `FAMAILObjective`) is the suite's heaviest engineering (spec §8) and is cleanly separable. The reuse seam is documented below so Phase 4 can pick it up without rework.
- **FAMAIL / B2 model-level dataset swaps** — deferred to **Phase 4**. Once `fit_and_evaluate` exists, these are "build the edited/filtered *trajectory* dataset, then call `fit_and_evaluate` on it." The dataset-variant builders (edited/filtered trajectory sequences, not just demand grids) are their own plumbing task.
- **Pure-GAN ablation** (skip MLE pretrain) and **multi-seed paired scale-up** — deferred to **Phase 4/5**.

This mirrors the spec's de-risked build order (§7): item 2 (B0 end-to-end, now completed with the adversarial stage) before item 3 (FAMAIL+B2) and item 4 (B1). Each phase stays a working, independently testable unit.

### Design decisions (flagged — veto if you disagree)

1. **The real-vs-fake critic is a fresh LSTM over the cell-token vocabulary** (`SequenceCritic`), mirroring the Siamese encoder *design* (spec decision #8: "reuses the Siamese encoder **design** retasked real-vs-fake"), **not** a literal reuse of the trained `SiameseLSTMDiscriminator` module. Reason: the generator emits sequences over a 4323-symbol cell vocabulary via `nn.Embedding`; Gumbel-softmax adversarial training needs the critic to consume a soft distribution `y_soft` over that vocab (`y_soft @ embedding.weight`). The Siamese module consumes raw `[x, y, time_bucket, day_index]` features through a `FeatureNormalizer` and a *paired same-agent* head — retasking it to single-sequence real-vs-fake over soft cell distributions would require more surgery than a purpose-built critic and would not share the generator's representation. The **trained** same-agent Siamese discriminator remains reserved, unmodified, for the **eval-time realism critic** (spec decision #9) — a separate, later concern.
2. **The critic is unconditioned** (scores a bare token sequence; no `(start cell, time-block)` input). Realism of the *sequence shape* is what separates real from generated; conditioning the critic adds parameters for little gain. Simplification; revisit if D collapses.
3. **Fixed-length differentiable rollout.** The Gumbel rollout always decodes `max_len` steps (no early `break` at EOS) so the batch keeps a static `(B, max_len, V)` shape and gradients flow through every step. The first sampled EOS is recorded in a `lengths` vector for downstream masking. (The non-differentiable Phase-2 `generate_pickups` keeps its early-stop behavior; only the adversarial path uses the fixed-length rollout.)
4. **Straight-through hard Gumbel** (`hard=True`): the forward pass uses discrete one-hots (faithful to real generation) while gradients use the soft relaxation. The next-step input embedding is `y_soft @ cell_embed.weight` so the recurrence is differentiable.
5. **`b0.py` (Phase 2, MLE-only) is kept as-is.** The adversarial B0 lives in a new `model_level.py::fit_and_evaluate`; the two coexist so the cheap MLE keystone stays runnable and the adversarial run is additive.

### Reuse seam reserved for Phase 4 (B1) — documented, not built

B1 adds `λ·(1 − F_causal)` to the generator loss. The differentiable F path already exists: `famail_temporal.algorithm.objective.FAMAILObjective(bundle).forward(soft_pickup_3d)` returns `(total, terms)` with a differentiable `terms["f_causal"]`. Phase 4 will add a `soft_pickups_to_pickup_3d` that scatters each rollout's **terminal-step soft distribution** (`soft_onehots[b, lengths[b]-1, :N_CELLS]`) times `pickup_mass(bundle, t_block)` into a `(GX, GY, T)` tensor with gradient flow, then add `λ·(1 − f_causal)` to `loss_g`. Nothing in Phase 3 needs to anticipate this beyond returning `soft_onehots` + `lengths` from the rollout (which it does).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `famail_temporal/baselines/gan/config.py` | + adversarial hyperparameters (epochs, LRs, Gumbel temps, critic hidden dim) | Modify |
| `famail_temporal/baselines/gan/generator.py` | + `step_embed` (decode from a precomputed input embedding); refactor `step` to use it | Modify |
| `famail_temporal/baselines/gan/gumbel.py` | `gumbel_rollout` — differentiable straight-through batched rollout | Create |
| `famail_temporal/baselines/gan/critic.py` | `SequenceCritic` — real-vs-fake LSTM over the cell vocabulary | Create |
| `famail_temporal/baselines/gan/train_adversarial.py` | `adversarial_finetune` — non-saturating GAN fine-tune loop | Create |
| `famail_temporal/baselines/gan/model_level.py` | `fit_and_evaluate` — MLE → adversarial → generate → grid → fairness | Create |
| `famail_temporal/baselines/gan/run_b0_adversarial.py` | CLI entry point for standard-adversarial B0 | Create |
| `famail_temporal/baselines/gan/tests/test_adv_config.py` | adversarial config sanity | Create |
| `famail_temporal/baselines/gan/tests/test_generator_step_embed.py` | `step_embed` ≡ `step` equivalence | Create |
| `famail_temporal/baselines/gan/tests/test_gumbel.py` | rollout shapes, one-hotness, length bounds, gradient flow | Create |
| `famail_temporal/baselines/gan/tests/test_critic.py` | shapes, soft≡hard equivalence, overfit-separable-batch | Create |
| `famail_temporal/baselines/gan/tests/test_train_adversarial.py` | loop runs, losses finite, params change | Create |
| `famail_temporal/baselines/gan/tests/test_model_level.py` | end-to-end on a tiny synthetic bundle | Create |
| `famail_temporal/baselines/gan/tests/test_run_b0_adversarial.py` | result serialization | Create |

---

## Task 1: Adversarial hyperparameters in `config.py`

**Files:**
- Modify: `famail_temporal/baselines/gan/config.py`
- Test: `famail_temporal/baselines/gan/tests/test_adv_config.py`

- [ ] **Step 1: Write the failing test**

Create `famail_temporal/baselines/gan/tests/test_adv_config.py`:

```python
"""Sanity checks for the Phase-3 adversarial hyperparameters."""
from famail_temporal.baselines.gan import config as gc


def test_adversarial_constants_present_and_sane():
    assert gc.ADV_EPOCHS >= 1
    assert gc.ADV_LR_G > 0 and gc.ADV_LR_D > 0
    assert gc.ADV_BATCH_SIZE >= 1
    # Temperature is annealed downward toward (but never to) zero.
    assert gc.GUMBEL_TAU_START >= gc.GUMBEL_TAU_END > 0
    assert gc.D_HIDDEN_DIM >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_adv_config.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'ADV_EPOCHS'`

- [ ] **Step 3: Add the constants**

Append to `famail_temporal/baselines/gan/config.py` (after the `MAX_GEN_LEN` line):

```python

# Adversarial fine-tune (Phase 3)
ADV_EPOCHS = 3
ADV_LR_G = 1e-4                   # generator LR during fine-tune (small: don't undo MLE)
ADV_LR_D = 1e-4                   # critic LR
ADV_BATCH_SIZE = 256
GUMBEL_TAU_START = 1.0            # Gumbel-softmax temperature, annealed start
GUMBEL_TAU_END = 0.5             #   -> end (sharper, closer to discrete)
D_HIDDEN_DIM = 128               # critic LSTM hidden size
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_adv_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/config.py famail_temporal/baselines/gan/tests/test_adv_config.py
git commit -m "feat(baselines/gan): adversarial fine-tune hyperparameters"
```

---

## Task 2: Generator `step_embed` (decode from a precomputed input embedding)

**Files:**
- Modify: `famail_temporal/baselines/gan/generator.py`
- Test: `famail_temporal/baselines/gan/tests/test_generator_step_embed.py`

This adds the hook the Gumbel rollout needs: a single-step decode that takes an already-computed `(B, E)` input embedding instead of a token id, so the next input can be a differentiable soft embedding. `step` is refactored to call it (behavior unchanged).

- [ ] **Step 1: Write the failing test** (the new method must match `step` when fed the hard-token embedding)

Create `famail_temporal/baselines/gan/tests/test_generator_step_embed.py`:

```python
"""step_embed must reproduce step() when fed the hard-token embedding."""
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM


def test_step_embed_matches_step():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    model.train(False)
    B = 3
    token = torch.randint(0, gc.N_CELLS, (B,))
    cc = torch.randint(0, gc.N_CELLS, (B,))
    tb = torch.randint(0, gc.N_TBLOCKS, (B,))

    logits_step, h_step = model.step(token, cc, tb, None)
    embed = model.cell_embed(token)                      # (B, E)
    logits_embed, h_embed = model.step_embed(embed, cc, tb, None)

    assert logits_embed.shape == (B, gc.VOCAB_SIZE)
    assert torch.allclose(logits_step, logits_embed, atol=1e-6)
    assert torch.allclose(h_step[0], h_embed[0], atol=1e-6)
    assert torch.allclose(h_step[1], h_embed[1], atol=1e-6)


def test_step_embed_passes_gradient_to_input():
    model = TrajectoryLSTM()
    embed = torch.randn(2, gc.EMBED_DIM, requires_grad=True)
    cc = torch.zeros(2, dtype=torch.long)
    tb = torch.zeros(2, dtype=torch.long)
    logits, _ = model.step_embed(embed, cc, tb, None)
    logits.sum().backward()
    assert embed.grad is not None and embed.grad.abs().sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_generator_step_embed.py -v`
Expected: FAIL with `AttributeError: 'TrajectoryLSTM' object has no attribute 'step_embed'`

- [ ] **Step 3: Add `step_embed` and refactor `step`**

In `famail_temporal/baselines/gan/generator.py`, replace the existing `step` method with the following two methods (the new `step` is numerically identical — it just delegates):

```python
    def step_embed(
        self,
        input_embed: torch.Tensor,  # (B, E) precomputed input-token embedding
        ctx_cell: torch.Tensor,     # (B,) long start-cell ids
        ctx_tblock: torch.Tensor,   # (B,) long start time-block ids
        hidden=None,                # (h, c) LSTM state from the previous step
    ):
        """Single-step decode from a precomputed input embedding.

        Used by the Gumbel-softmax rollout, where the next input is a
        differentiable soft embedding (soft_onehot @ cell_embed.weight) rather
        than a hard token id. Carries the recurrent state for O(L) decode.
        """
        ctx = self.cell_embed(ctx_cell) + self.tblock_embed(ctx_tblock)  # (B, E)
        x = (input_embed + ctx).unsqueeze(1)                          # (B, 1, E)
        out, hidden = self.lstm(x, hidden)                            # (B, 1, H)
        return self.head(out[:, -1]), hidden                          # (B, V), state

    def step(
        self,
        token: torch.Tensor,       # (B,) long current token id
        ctx_cell: torch.Tensor,    # (B,) long start-cell ids
        ctx_tblock: torch.Tensor,  # (B,) long start time-block ids
        hidden=None,               # (h, c) LSTM state from the previous step
    ):
        """Single-step decode from a token id: next-token logits + LSTM state.

        Equivalent to slicing the last position of forward() over the full
        prefix (LSTM is a recurrence), with the same per-step additive
        conditioning. Delegates to step_embed after embedding the token.
        """
        return self.step_embed(self.cell_embed(token), ctx_cell, ctx_tblock, hidden)
```

- [ ] **Step 4: Run the new test AND the existing generator/rollout tests (the refactor must not change `step`)**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_generator_step_embed.py famail_temporal/baselines/gan/tests/test_generator.py famail_temporal/baselines/gan/tests/test_rollout.py -v`
Expected: PASS (new equivalence + gradient tests, and all pre-existing generator/rollout tests still green)

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/generator.py famail_temporal/baselines/gan/tests/test_generator_step_embed.py
git commit -m "feat(baselines/gan): generator step_embed for differentiable decode"
```

---

## Task 3: `gumbel.py` — differentiable straight-through rollout

**Files:**
- Create: `famail_temporal/baselines/gan/gumbel.py`
- Test: `famail_temporal/baselines/gan/tests/test_gumbel.py`

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/gan/tests/test_gumbel.py`:

```python
"""Unit tests for gan.gumbel.gumbel_rollout."""
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.gumbel import gumbel_rollout


def _ctx(B):
    cc = torch.randint(0, gc.N_CELLS, (B,))
    tb = torch.randint(0, gc.N_TBLOCKS, (B,))
    return cc, tb


def test_rollout_shapes_and_one_hot():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    B, max_len = 4, 10
    cc, tb = _ctx(B)
    soft, lengths = gumbel_rollout(
        model, cc, tb, max_len=max_len, tau=1.0,
        device=torch.device("cpu"), hard=True,
    )
    assert soft.shape == (B, max_len, gc.VOCAB_SIZE)
    # hard=True -> each step is a one-hot: sums to 1, max is 1.
    assert torch.allclose(soft.sum(dim=-1), torch.ones(B, max_len), atol=1e-5)
    assert torch.allclose(soft.max(dim=-1).values, torch.ones(B, max_len), atol=1e-5)
    assert lengths.shape == (B,)
    assert int(lengths.min()) >= 1 and int(lengths.max()) <= max_len


def test_rollout_gradient_flows_to_model():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    cc, tb = _ctx(2)
    soft, _ = gumbel_rollout(
        model, cc, tb, max_len=6, tau=1.0,
        device=torch.device("cpu"), hard=True,
    )
    soft.sum().backward()
    grad_total = sum(
        p.grad.abs().sum() for p in model.parameters() if p.grad is not None
    )
    assert grad_total > 0


def test_rollout_seed_deterministic():
    model = TrajectoryLSTM()
    cc, tb = _ctx(3)
    torch.manual_seed(7)
    a, la = gumbel_rollout(model, cc, tb, max_len=8, tau=1.0,
                           device=torch.device("cpu"), hard=True)
    torch.manual_seed(7)
    b, lb = gumbel_rollout(model, cc, tb, max_len=8, tau=1.0,
                           device=torch.device("cpu"), hard=True)
    assert torch.equal(a, b) and torch.equal(la, lb)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_gumbel.py -v`
Expected: FAIL (ModuleNotFoundError on `gan.gumbel`)

- [ ] **Step 3: Implement `gumbel.py`**

Create `famail_temporal/baselines/gan/gumbel.py`:

```python
"""Differentiable straight-through Gumbel-softmax rollout.

Decodes a fixed number of steps (max_len) so the batch keeps a static
(B, max_len, V) shape and gradients flow through every step; the first
sampled EOS per row is recorded in `lengths` for downstream masking. The
next-step input is a differentiable soft embedding (y @ cell_embed.weight),
so the recurrence is end-to-end differentiable wrt the generator's params.
"""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM


def gumbel_rollout(
    model: TrajectoryLSTM,
    ctx_cell: torch.Tensor,     # (B,) long start-cell ids
    ctx_tblock: torch.Tensor,   # (B,) long start time-block ids
    *,
    max_len: int,
    tau: float,
    device: torch.device,
    hard: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (soft_onehots, lengths).

    soft_onehots: (B, max_len, VOCAB_SIZE) per-step straight-through one-hots,
        differentiable wrt model parameters.
    lengths: (B,) long — 1-based index of the first sampled EOS, or max_len.
    """
    cc = ctx_cell.to(device)
    tb = ctx_tblock.to(device)
    B = cc.shape[0]

    prev_embed = model.cell_embed(
        torch.full((B,), gc.BOS, dtype=torch.long, device=device)
    )                                                       # (B, E)
    hidden = None
    steps = []
    ended = torch.zeros(B, dtype=torch.bool, device=device)
    lengths = torch.full((B,), max_len, dtype=torch.long, device=device)

    for t in range(max_len):
        logits, hidden = model.step_embed(prev_embed, cc, tb, hidden)   # (B, V)
        y = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)        # (B, V)
        steps.append(y)
        nxt = y.argmax(dim=-1)                                          # (B,)
        newly_ended = (~ended) & (nxt == gc.EOS)
        lengths = torch.where(
            newly_ended, torch.full_like(lengths, t + 1), lengths,
        )
        ended = ended | (nxt == gc.EOS)
        prev_embed = y @ model.cell_embed.weight                       # (B, E)

    soft_onehots = torch.stack(steps, dim=1)                           # (B, L, V)
    return soft_onehots, lengths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_gumbel.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/gumbel.py famail_temporal/baselines/gan/tests/test_gumbel.py
git commit -m "feat(baselines/gan): differentiable Gumbel-softmax rollout"
```

---

## Task 4: `critic.py` — real-vs-fake sequence critic

**Files:**
- Create: `famail_temporal/baselines/gan/critic.py`
- Test: `famail_temporal/baselines/gan/tests/test_critic.py`

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/gan/tests/test_critic.py`:

```python
"""Unit tests for gan.critic.SequenceCritic."""
import torch
import torch.nn.functional as F

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.critic import SequenceCritic


def test_forward_ids_returns_per_sequence_logit():
    torch.manual_seed(0)
    critic = SequenceCritic()
    B, L = 5, 7
    ids = torch.randint(0, gc.N_CELLS, (B, L))
    lengths = torch.full((B,), L, dtype=torch.long)
    out = critic.forward_ids(ids, lengths)
    assert out.shape == (B,)


def test_soft_matches_hard_onehot():
    """forward_soft on a hard one-hot equals forward_ids on the same ids."""
    torch.manual_seed(0)
    critic = SequenceCritic()
    critic.train(False)
    B, L = 4, 6
    ids = torch.randint(0, gc.N_CELLS, (B, L))
    lengths = torch.full((B,), L, dtype=torch.long)
    onehot = F.one_hot(ids, num_classes=gc.VOCAB_SIZE).float()
    a = critic.forward_ids(ids, lengths)
    b = critic.forward_soft(onehot, lengths)
    assert torch.allclose(a, b, atol=1e-5)


def test_critic_can_separate_trivial_real_vs_fake():
    """A few D-steps should push real logits up and fake logits down on a
    trivially separable batch (real = low cell ids, fake = high cell ids)."""
    torch.manual_seed(0)
    critic = SequenceCritic()
    opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
    bce = torch.nn.BCEWithLogitsLoss()
    L = 5
    real = torch.zeros(8, L, dtype=torch.long)               # all cell 0
    fake = torch.full((8, L), gc.N_CELLS - 1, dtype=torch.long)  # all last cell
    lengths = torch.full((8,), L, dtype=torch.long)
    for _ in range(50):
        d_real = critic.forward_ids(real, lengths)
        d_fake = critic.forward_ids(fake, lengths)
        loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        opt.zero_grad(); loss.backward(); opt.step()
    assert critic.forward_ids(real, lengths).mean() > critic.forward_ids(fake, lengths).mean()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_critic.py -v`
Expected: FAIL (ModuleNotFoundError on `gan.critic`)

- [ ] **Step 3: Implement `critic.py`**

Create `famail_temporal/baselines/gan/critic.py`:

```python
"""Real-vs-fake LSTM critic over the grid-cell vocabulary.

Mirrors the generator's representation (its own nn.Embedding over the same
VOCAB) so it can score BOTH hard real token sequences (forward_ids) and
differentiable Gumbel-softmax fake sequences (forward_soft, via
soft_onehot @ embed.weight). One realism logit per sequence, read off the
last valid timestep (BCEWithLogits convention: real = 1, fake = 0).
Unconditioned (Phase-3 simplification).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from famail_temporal.baselines.gan import config as gc


class SequenceCritic(nn.Module):
    def __init__(
        self,
        vocab_size: int = gc.VOCAB_SIZE,
        embed_dim: int = gc.EMBED_DIM,
        hidden_dim: int = gc.D_HIDDEN_DIM,
        n_layers: int = gc.N_LAYERS,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=gc.PAD)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers, batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def _forward_embed(
        self, embedded: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self.lstm(embedded)                          # (B, L, H)
        idx = (lengths - 1).clamp(min=0)                      # last valid step
        last = out[torch.arange(out.size(0), device=out.device), idx]  # (B, H)
        return self.head(last).squeeze(-1)                    # (B,)

    def forward_ids(
        self, token_ids: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Score hard real sequences. token_ids: (B, L) long."""
        return self._forward_embed(self.embed(token_ids), lengths)

    def forward_soft(
        self, soft_onehots: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Score soft generated sequences. soft_onehots: (B, L, VOCAB_SIZE)."""
        embedded = soft_onehots @ self.embed.weight           # (B, L, E)
        return self._forward_embed(embedded, lengths)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_critic.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/critic.py famail_temporal/baselines/gan/tests/test_critic.py
git commit -m "feat(baselines/gan): real-vs-fake sequence critic"
```

---

## Task 5: `train_adversarial.py` — non-saturating GAN fine-tune loop

**Files:**
- Create: `famail_temporal/baselines/gan/train_adversarial.py`
- Test: `famail_temporal/baselines/gan/tests/test_train_adversarial.py`

- [ ] **Step 1: Write the failing test**

Create `famail_temporal/baselines/gan/tests/test_train_adversarial.py`:

```python
"""Smoke test for gan.train_adversarial.adversarial_finetune."""
import copy
import math

import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.train_adversarial import adversarial_finetune


def test_finetune_runs_and_updates_generator():
    torch.manual_seed(0)
    sequences = [
        [gc.BOS, 10, 11, 12, gc.EOS],
        [gc.BOS, 40, 41, gc.EOS],
        [gc.BOS, 5, 6, 7, 8, gc.EOS],
        [gc.BOS, 20, 21, gc.EOS],
    ]
    contexts = [(10, 0), (40, 1), (5, 0), (20, 2)]
    model = TrajectoryLSTM()
    before = copy.deepcopy(model.state_dict())

    history = adversarial_finetune(
        model, sequences, contexts,
        epochs=2, lr_g=1e-3, lr_d=1e-3, batch_size=2,
        max_len=8, tau_start=1.0, tau_end=0.5,
        device=torch.device("cpu"),
    )

    assert set(history) == {"g_losses", "d_losses"}
    assert len(history["g_losses"]) == 2 and len(history["d_losses"]) == 2
    assert all(math.isfinite(x) for x in history["g_losses"] + history["d_losses"])
    # The generator's parameters moved (fine-tune actually stepped G).
    after = model.state_dict()
    assert any(
        not torch.allclose(before[k], after[k]) for k in before
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_train_adversarial.py -v`
Expected: FAIL (ModuleNotFoundError on `gan.train_adversarial`)

- [ ] **Step 3: Implement `train_adversarial.py`**

Create `famail_temporal/baselines/gan/train_adversarial.py`:

```python
"""Gumbel-softmax adversarial fine-tune of an MLE-pretrained generator.

Non-saturating GAN: the discriminator maximizes log D(real) + log(1 - D(fake));
the generator maximizes log D(fake). The fake batch is a differentiable
straight-through Gumbel-softmax rollout, so generator gradients flow through
the discrete sequence. The Gumbel temperature is annealed across epochs. A
fresh SequenceCritic is created and trained alongside (the trained Siamese
discriminator is reserved for eval-time realism, not used here).
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.critic import SequenceCritic
from famail_temporal.baselines.gan.gumbel import gumbel_rollout
from famail_temporal.baselines.gan.train_mle import _pad_batch


def _anneal(epoch: int, n_epochs: int, start: float, end: float) -> float:
    if n_epochs <= 1:
        return end
    return start + (end - start) * (epoch / (n_epochs - 1))


def adversarial_finetune(
    model: TrajectoryLSTM,
    sequences: List[List[int]],
    contexts: List[Tuple[int, int]],
    *,
    epochs: int,
    lr_g: float,
    lr_d: float,
    batch_size: int,
    max_len: int,
    tau_start: float,
    tau_end: float,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Fine-tune `model` (in place) against a fresh critic. Returns per-epoch
    mean generator and discriminator losses."""
    model.to(device).train()
    critic = SequenceCritic().to(device).train()
    opt_g = torch.optim.Adam(model.parameters(), lr=lr_g)
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr_d)
    bce = nn.BCEWithLogitsLoss()
    n = len(sequences)
    g_losses: List[float] = []
    d_losses: List[float] = []

    for epoch in range(epochs):
        tau = _anneal(epoch, epochs, tau_start, tau_end)
        perm = torch.randperm(n)
        g_batch: List[float] = []
        d_batch: List[float] = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size].tolist()
            real = _pad_batch([sequences[i] for i in idx], device)      # (b, Lr)
            real_lengths = torch.tensor(
                [len(sequences[i]) for i in idx], dtype=torch.long, device=device,
            )
            cc = torch.tensor(
                [contexts[i][0] for i in idx], dtype=torch.long, device=device,
            )
            tb = torch.tensor(
                [contexts[i][1] for i in idx], dtype=torch.long, device=device,
            )

            # ----- Discriminator step (generator fixed) -----
            with torch.no_grad():
                fake_soft, fake_len = gumbel_rollout(
                    model, cc, tb, max_len=max_len, tau=tau,
                    device=device, hard=True,
                )
            d_real = critic.forward_ids(real, real_lengths)
            d_fake = critic.forward_soft(fake_soft, fake_len)
            loss_d = (
                bce(d_real, torch.ones_like(d_real))
                + bce(d_fake, torch.zeros_like(d_fake))
            )
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ----- Generator step (non-saturating; gradients via Gumbel) -----
            fake_soft, fake_len = gumbel_rollout(
                model, cc, tb, max_len=max_len, tau=tau,
                device=device, hard=True,
            )
            d_fake_g = critic.forward_soft(fake_soft, fake_len)
            loss_g = bce(d_fake_g, torch.ones_like(d_fake_g))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            g_batch.append(float(loss_g.item()))
            d_batch.append(float(loss_d.item()))

        g_losses.append(sum(g_batch) / len(g_batch))
        d_losses.append(sum(d_batch) / len(d_batch))

    return {"g_losses": g_losses, "d_losses": d_losses}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_train_adversarial.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/train_adversarial.py famail_temporal/baselines/gan/tests/test_train_adversarial.py
git commit -m "feat(baselines/gan): Gumbel-softmax adversarial fine-tune loop"
```

---

## Task 6: `model_level.py` — `fit_and_evaluate` orchestrator (standard-adversarial B0)

**Files:**
- Create: `famail_temporal/baselines/gan/model_level.py`
- Test: `famail_temporal/baselines/gan/tests/test_model_level.py`

- [ ] **Step 1: Write the failing test** (end-to-end on a tiny synthetic bundle with real trajectories)

Create `famail_temporal/baselines/gan/tests/test_model_level.py`:

```python
"""End-to-end MLE -> adversarial -> generate -> grid -> fairness on a tiny bundle."""
import torch

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import active_units, make_traj_at
from famail_temporal.baselines.gan import model_level


def test_fit_and_evaluate_returns_fairness_and_histories():
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 20)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    out = model_level.fit_and_evaluate(
        bundle, mle_epochs=2, adv_epochs=2, max_len=8,
        device=torch.device("cpu"), seed=0,
    )
    assert set(out) == {
        "generated", "corpus", "n_generated", "mle_losses", "adv_losses",
    }
    for key in ("generated", "corpus"):
        m = out[key]
        assert set(m) == {"f_spatial", "f_causal", "gini_dsr", "gini_asr"}
        assert 0.0 <= m["f_causal"] <= 1.0
    assert out["n_generated"] == len(bundle.trajectories)
    assert len(out["mle_losses"]) == 2
    assert set(out["adv_losses"]) == {"g_losses", "d_losses"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_model_level.py -v`
Expected: FAIL (ModuleNotFoundError on `gan.model_level`)

- [ ] **Step 3: Implement `model_level.py`**

Create `famail_temporal/baselines/gan/model_level.py`:

```python
"""Standard-adversarial training paradigm (spec B0): MLE pretrain -> Gumbel
adversarial fine-tune -> generate -> demand grid -> data-level fairness.

This is the spec's "B0 end-to-end" with the adversarial stage Phase 2 deferred.
FAMAIL and B2 reuse this verbatim by passing an edited / filtered bundle
(Phase 4); only the training data changes.
"""
from __future__ import annotations
import torch

from famail_temporal.data.loader import DataBundle
from famail_temporal.utils.seeding import set_all_seeds
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.sequences import (
    trajectory_to_tokens, trajectory_context,
)
from famail_temporal.baselines.gan.train_mle import train_mle
from famail_temporal.baselines.gan.train_adversarial import adversarial_finetune
from famail_temporal.baselines.gan.rollout import (
    generate_pickups, pickups_to_pickup_3d,
)
from famail_temporal.baselines.metrics import data_level_fairness


def fit_and_evaluate(
    bundle: DataBundle, *,
    mle_epochs: int = gc.MLE_EPOCHS,
    adv_epochs: int = gc.ADV_EPOCHS,
    max_len: int = gc.MAX_GEN_LEN,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict:
    """Train (MLE + adversarial) on bundle.trajectories, generate one rollout
    per real context, and return generated-vs-corpus fairness + loss histories.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not bundle.trajectories:
        raise ValueError(
            "fit_and_evaluate requires a non-empty corpus (bundle.trajectories)"
        )
    set_all_seeds(seed)

    sequences = [trajectory_to_tokens(t) for t in bundle.trajectories]
    contexts = [trajectory_context(t) for t in bundle.trajectories]

    model = TrajectoryLSTM().to(device)
    mle_losses = train_mle(
        model, sequences, contexts,
        epochs=mle_epochs, lr=gc.MLE_LR, batch_size=gc.MLE_BATCH_SIZE,
        device=device,
    )
    adv_losses = adversarial_finetune(
        model, sequences, contexts,
        epochs=adv_epochs, lr_g=gc.ADV_LR_G, lr_d=gc.ADV_LR_D,
        batch_size=gc.ADV_BATCH_SIZE, max_len=max_len,
        tau_start=gc.GUMBEL_TAU_START, tau_end=gc.GUMBEL_TAU_END,
        device=device,
    )

    pickups = generate_pickups(model, contexts, max_len=max_len, device=device)
    gen_grid = pickups_to_pickup_3d(bundle, pickups)

    return {
        "generated": data_level_fairness(bundle, pickup_3d=gen_grid),
        "corpus": data_level_fairness(bundle),
        "n_generated": len(pickups),
        "mle_losses": mle_losses,
        "adv_losses": adv_losses,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_model_level.py -v`
Expected: PASS

- [ ] **Step 5: Run the full gan test suite**

Run: `python -m pytest famail_temporal/baselines/gan/ -v`
Expected: PASS (all Phase 2 + Phase 3 tests)

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/gan/model_level.py famail_temporal/baselines/gan/tests/test_model_level.py
git commit -m "feat(baselines/gan): MLE+adversarial fit_and_evaluate orchestrator (B0)"
```

---

## Task 7: `run_b0_adversarial.py` — CLI + real-data smoke

**Files:**
- Create: `famail_temporal/baselines/gan/run_b0_adversarial.py`
- Test: `famail_temporal/baselines/gan/tests/test_run_b0_adversarial.py`

- [ ] **Step 1: Write the failing test** (the JSON-serialization helper is the only pure logic to unit-test)

Create `famail_temporal/baselines/gan/tests/test_run_b0_adversarial.py`:

```python
"""Unit test for run_b0_adversarial result serialization."""
import json

from famail_temporal.baselines.gan import run_b0_adversarial as r


def test_result_to_json_roundtrips():
    result = {
        "generated": {"f_spatial": 0.08, "f_causal": 0.79,
                      "gini_dsr": 0.9, "gini_asr": 0.9},
        "corpus": {"f_spatial": 0.082, "f_causal": 0.805,
                   "gini_dsr": 0.94, "gini_asr": 0.9},
        "n_generated": 105401,
        "mle_losses": [3.1, 2.4],
        "adv_losses": {"g_losses": [0.71, 0.69], "d_losses": [1.30, 1.32]},
    }
    blob = r.result_to_json(result)
    loaded = json.loads(blob)
    assert loaded["n_generated"] == 105401
    assert loaded["corpus"]["f_causal"] == 0.805
    assert loaded["adv_losses"]["g_losses"] == [0.71, 0.69]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_run_b0_adversarial.py -v`
Expected: FAIL (ModuleNotFoundError on `gan.run_b0_adversarial`)

- [ ] **Step 3: Implement `run_b0_adversarial.py`**

Create `famail_temporal/baselines/gan/run_b0_adversarial.py`:

```python
"""CLI: train the standard-adversarial B0 (MLE + Gumbel adversarial fine-tune)
on the real corpus and report generated-vs-corpus fairness.

Example:
    python -m famail_temporal.baselines.gan.run_b0_adversarial \
        --mle-epochs 5 --adv-epochs 3 --device auto
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
from famail_temporal.baselines.gan.model_level import fit_and_evaluate


def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2)


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.baselines.gan.run_b0_adversarial",
    )
    ap.add_argument("--mle-epochs", type=int, default=gc.MLE_EPOCHS)
    ap.add_argument("--adv-epochs", type=int, default=gc.ADV_EPOCHS)
    ap.add_argument("--max-len", type=int, default=gc.MAX_GEN_LEN)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(config.PACKAGE_ROOT) / "results" / "b0_adversarial")
    args = ap.parse_args(argv)

    bundle = DataBundle.load()
    result = fit_and_evaluate(
        bundle, mle_epochs=args.mle_epochs, adv_epochs=args.adv_epochs,
        max_len=args.max_len, device=_resolve_device(args.device), seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "b0_adversarial_fairness.json").write_text(
        result_to_json(result)
    )
    print(f"corpus    F_causal={result['corpus']['f_causal']:.4f}")
    print(f"generated F_causal={result['generated']['f_causal']:.4f}")
    print(f"wrote {args.out_dir / 'b0_adversarial_fairness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_run_b0_adversarial.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/run_b0_adversarial.py famail_temporal/baselines/gan/tests/test_run_b0_adversarial.py
git commit -m "feat(baselines/gan): standard-adversarial B0 CLI entry point"
```

- [ ] **Step 6: Real-data smoke (manual; needs cache + GPU recommended)**

Run: `python -m famail_temporal.baselines.gan.run_b0_adversarial --mle-epochs 5 --adv-epochs 3 --device auto`
Expected: writes `famail_temporal/results/b0_adversarial/b0_adversarial_fairness.json`. Inspect: `corpus.f_causal ≈ 0.805`; `generated.f_causal` should land *near* the corpus value (bias reproduced/propagated through the adversarial model). Watch the loss histories: if `d_losses` collapses toward 0 while `g_losses` diverges, the critic has won (mode collapse) — and `generated.f_causal` may drift *below* corpus, which is the **amplification** sub-claim (spec §1 "bonus claim"), a finding to record, **not** a bug to fix by changing fairness code. Flag any large gap for discussion per the standing protocol.

---

## Self-Review

**1. Spec coverage (Phase 3 = adversarial subsystem + standard-adversarial B0):**
- Gumbel-softmax differentiable discrete generation (spec §4.1, decision #4) — Tasks 2, 3. ✓
- Real-vs-fake critic mirroring the Siamese encoder design (spec §4.1, decision #8) — Task 4. ✓
- MLE pretrain → adversarial fine-tune (spec §4.2 "primary"; SeqGAN recipe) — Tasks 5, 6. ✓
- B0 = standard adversarial on raw data, rollout → grid → fairness (spec §4.3, §4.4) — Tasks 6, 7. ✓
- Temperature anneal 1.0 → 0.5 (spec §9) — Tasks 1, 5. ✓
- **Deferred by design (stated up front):** B1 fairness loss (spec §4.3 / build-order item 4), FAMAIL/B2 model-level swaps (item 3), pure-GAN ablation + multi-seed (item 5), eval-time Siamese realism critic + JS-divergence utility (spec §5). The B1 reuse seam (`FAMAILObjective` + a terminal-soft-pickup scatter) is documented so Phase 4 needs no rework.

**2. Placeholder scan:** No "TBD" / "add error handling" / "similar to Task N". Every code step is complete; every test step has assertions + an exact command + expected outcome. ✓

**3. Type consistency:**
- `TrajectoryLSTM.step_embed(input_embed, ctx_cell, ctx_tblock, hidden=None) -> (logits, hidden)`; `step` delegates to it. Used by `gumbel_rollout`. ✓
- `gumbel_rollout(model, ctx_cell, ctx_tblock, *, max_len, tau, device, hard=True) -> (soft_onehots (B,L,V), lengths (B,))`. Consumed by `adversarial_finetune` (`critic.forward_soft(fake_soft, fake_len)`). ✓
- `SequenceCritic.forward_ids(token_ids, lengths) -> (B,)` and `forward_soft(soft_onehots, lengths) -> (B,)`. Both go through `_forward_embed(embedded, lengths)`. ✓
- `adversarial_finetune(model, sequences, contexts, *, epochs, lr_g, lr_d, batch_size, max_len, tau_start, tau_end, device) -> {"g_losses", "d_losses"}`. Called by `fit_and_evaluate` with `gc.ADV_*` + `gc.GUMBEL_TAU_*`. ✓
- `fit_and_evaluate(bundle, *, mle_epochs, adv_epochs, max_len, device, seed) -> {"generated","corpus","n_generated","mle_losses","adv_losses"}`. Asserted in test_model_level; consumed by `run_b0_adversarial`. ✓
- Reused Phase-2 signatures unchanged: `train_mle(model, sequences, contexts, *, epochs, lr, batch_size, device)`, `_pad_batch(seqs, device)`, `generate_pickups(model, contexts, *, max_len, device)`, `pickups_to_pickup_3d(bundle, pickups)`, `data_level_fairness(bundle, pickup_3d=...)`, `trajectory_to_tokens`, `trajectory_context`. ✓
- New config constants `ADV_EPOCHS / ADV_LR_G / ADV_LR_D / ADV_BATCH_SIZE / GUMBEL_TAU_START / GUMBEL_TAU_END / D_HIDDEN_DIM` defined once in `gan/config.py`, imported via `gc.`. ✓

**4. Ambiguity:** Fixed-length rollout with EOS recorded in `lengths` (not early-break) — stated in design decision #3, the `gumbel.py` docstring, and asserted in `test_gumbel`. Straight-through hard Gumbel (decision #4). Critic unconditioned (decision #2). `lengths-1` clamped at 0 so an immediate-EOS row still indexes a valid step. Fresh critic per fine-tune (the trained Siamese critic is reserved for eval-time realism). All explicit.

**5. Standing-constraint check:** No change to the trajectory-editing algorithm or its intermediate calculations — all new code is GAN-side (generator/critic/loop/orchestrator). The `step` refactor is numerically identical (verified by `test_step_embed_matches_step` and re-running the existing generator/rollout suites). ε=2 is untouched (editing not invoked here). No `git add -A` — every commit stages named files only. ✓

---
