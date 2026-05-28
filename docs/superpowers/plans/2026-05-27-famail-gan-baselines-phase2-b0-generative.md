# FAMAIL GAN Baselines — Phase 2: B0 Generative Baseline (MLE keystone) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the model-level keystone — train an autoregressive trajectory generator on the corpus by maximum likelihood, sample rollouts, aggregate their pickups into a demand grid, and measure the generations' fairness — establishing the "bias in → bias out" B0 baseline and the reusable train→generate→grid→fairness pipeline.

**Architecture:** A new `famail_temporal/baselines/gan/` subpackage. Trajectories are encoded as sequences of flat grid-cell tokens with a `(start cell, start time-block)` conditioning context. A small conditional LSTM language-model over the cell vocabulary is trained with next-token cross-entropy (MLE). Rollouts are sampled autoregressively; each rollout's terminal cell is its pickup, aggregated into a `pickup_3d` demand grid (reusing Phase 1's `pickup_mass`), then scored with Phase 1's `data_level_fairness`. Supply/mask/hat-matrices come unchanged from the `DataBundle`, so only the demand channel reflects the generator.

**Tech Stack:** Python 3.12, PyTorch (nn.Embedding/LSTM/Linear, Adam, cross-entropy), NumPy, pytest. Reuses `famail_temporal.baselines.datasets.pickup_mass`, `famail_temporal.baselines.metrics.data_level_fairness`, `famail_temporal.data.aggregation`, `famail_temporal.utils.trajectory.Trajectory`, `famail_temporal.config`.

---

## Scope: this plan = spec Phase 2 MINUS adversarial training

The spec (§6) lists Phase 2 as "B0 end-to-end: G + D + MLE pretrain + rollout→grid + fairness eval." This plan delivers everything **except the adversarial discriminator + Gumbel-softmax fine-tune**, which move to a new **Phase 3** plan. Rationale:

- The MLE generator is **step 1 of the spec's own training paradigm** (§4.2) and is required regardless.
- It is **stable and fully testable** (overfit-a-batch, shape/contract, grid-aggregation invariants), unlike discrete-sequence adversarial training.
- It already establishes the **model-level B0 claim** (a generative model trained on biased data reproduces the bias in its generations) and the **reusable train→generate→grid→fairness pipeline** that B1/B2/FAMAIL and the adversarial layer all build on.

**Deferred to Phase 3 (documented, not stubbed):** real-vs-fake discriminator (reusing the Siamese encoder), Gumbel-softmax adversarial fine-tune, B1 differentiable fairness loss, B2/FAMAIL dataset swaps at the model level, pure-GAN ablation, multi-seed paired scale-up. The generator's `forward` is written so a Gumbel-softmax sampling path can be added later without changing its interface.

### Design decisions (flagged — veto if you disagree)
1. **Full-sequence generation** (model all cells, take the terminal as the pickup), not pickup-only — keeps the "trajectory generator" framing.
2. **Conditioning** = `(start cell, start time-block)`, injected by adding a context embedding to every input token embedding (simple, robust). Matches spec §4.1.
3. **Pickup time-block = the conditioning start time-block** (Phase-2 simplification — the generator models cells, not time). Acceptable because trajectories are short and the conditioning set is the real corpus's start contexts, so the generated pickup-time distribution ≈ the real one. Modeling time explicitly is deferred.
4. **Corpus-matched generation**: one rollout per real trajectory's context, so the generated demand grid is directly comparable in scale to `bundle.pickup_3d`.

---

## File Structure

| File | Responsibility |
|---|---|
| `famail_temporal/baselines/gan/__init__.py` | Package marker |
| `famail_temporal/baselines/gan/config.py` | Vocabulary constants + generator/training hyperparameters |
| `famail_temporal/baselines/gan/sequences.py` | Trajectory ↔ cell-token sequence, conditioning-context extraction |
| `famail_temporal/baselines/gan/generator.py` | `TrajectoryLSTM` conditional LM |
| `famail_temporal/baselines/gan/train_mle.py` | MLE training loop + padded batching |
| `famail_temporal/baselines/gan/rollout.py` | Autoregressive sampling → pickups → demand grid |
| `famail_temporal/baselines/gan/b0.py` | Orchestrate B0: train→generate→grid→fairness |
| `famail_temporal/baselines/gan/run_b0.py` | CLI entry point |
| `famail_temporal/baselines/gan/tests/__init__.py` | Test package marker |
| `famail_temporal/baselines/gan/tests/test_sequences.py` | Encoding round-trips, context |
| `famail_temporal/baselines/gan/tests/test_generator.py` | Output shapes, context wiring |
| `famail_temporal/baselines/gan/tests/test_train_mle.py` | Overfit-a-batch |
| `famail_temporal/baselines/gan/tests/test_rollout.py` | Sampling bounds, grid scale, seed-determinism |
| `famail_temporal/baselines/gan/tests/test_b0.py` | End-to-end on a tiny synthetic bundle |

---

## Task 1: Scaffold `gan` subpackage + config

**Files:**
- Create: `famail_temporal/baselines/gan/__init__.py`, `famail_temporal/baselines/gan/tests/__init__.py`, `famail_temporal/baselines/gan/config.py`
- Test: `famail_temporal/baselines/gan/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `famail_temporal/baselines/gan/tests/test_config.py`:

```python
"""Vocabulary/config sanity for the GAN baselines."""
from famail_temporal import config as root_config
from famail_temporal.baselines.gan import config as gc


def test_vocab_layout():
    gx, gy = root_config.GRID_DIMS
    assert gc.N_CELLS == gx * gy
    # Three special tokens above the cell ids, all distinct and contiguous.
    assert gc.BOS == gc.N_CELLS
    assert gc.EOS == gc.N_CELLS + 1
    assert gc.PAD == gc.N_CELLS + 2
    assert gc.VOCAB_SIZE == gc.N_CELLS + 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'famail_temporal.baselines.gan'`

- [ ] **Step 3: Create the package + config**

Create `famail_temporal/baselines/gan/__init__.py`:

```python
"""B0 generative baseline (MLE keystone) for the FAMAIL GAN baselines."""
```

Create `famail_temporal/baselines/gan/tests/__init__.py` (empty):

```python
```

Create `famail_temporal/baselines/gan/config.py`:

```python
"""Vocabulary constants and hyperparameters for the trajectory generator."""
from famail_temporal import config as _root

GX, GY = _root.GRID_DIMS          # (48, 90)
N_CELLS = GX * GY                 # 4320 flat cell ids: 0 .. N_CELLS-1
BOS = N_CELLS                     # begin-of-sequence
EOS = N_CELLS + 1                 # end-of-sequence
PAD = N_CELLS + 2                 # padding (ignored by the loss)
VOCAB_SIZE = N_CELLS + 3
N_TBLOCKS = _root.T               # conditioning time-block cardinality

# Generator
EMBED_DIM = 64
HIDDEN_DIM = 128
N_LAYERS = 1

# Training
MLE_EPOCHS = 5
MLE_LR = 1e-3
MLE_BATCH_SIZE = 256

# Generation
MAX_GEN_LEN = 64                  # hard cap on rollout length (cells)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/__init__.py famail_temporal/baselines/gan/tests/__init__.py famail_temporal/baselines/gan/config.py famail_temporal/baselines/gan/tests/test_config.py
git commit -m "feat(baselines/gan): scaffold gan subpackage + vocabulary config"
```

---

## Task 2: `sequences.py` — trajectory ↔ cell-token encoding + context

**Files:**
- Create: `famail_temporal/baselines/gan/sequences.py`
- Test: `famail_temporal/baselines/gan/tests/test_sequences.py`

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/gan/tests/test_sequences.py`:

```python
"""Unit tests for gan.sequences."""
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan import sequences as sq
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour


def _traj():
    # start at cell (2,3) in hour 8 (time_bucket 8*12+1=97); ends at (5,7)
    states = [
        TrajectoryState(x_grid=2.0, y_grid=3.0, time_bucket=97, day_index=1),
        TrajectoryState(x_grid=4.0, y_grid=6.0, time_bucket=97, day_index=1),
        TrajectoryState(x_grid=5.0, y_grid=7.0, time_bucket=97, day_index=1),
    ]
    return Trajectory(trajectory_id=0, driver_id=0, states=states)


def test_flat_cell_round_trip():
    for (x, y) in [(0, 0), (2, 3), (47, 89)]:
        assert sq.unflat_cell(sq.flat_cell(x, y)) == (x, y)


def test_trajectory_to_tokens_brackets_with_bos_eos():
    toks = sq.trajectory_to_tokens(_traj())
    assert toks[0] == gc.BOS
    assert toks[-1] == gc.EOS
    # Interior tokens are the three states' flat cells.
    assert toks[1:-1] == [sq.flat_cell(2, 3), sq.flat_cell(4, 6), sq.flat_cell(5, 7)]
    assert all(0 <= t < gc.VOCAB_SIZE for t in toks)


def test_trajectory_context_is_start_cell_and_block():
    cell, tblock = sq.trajectory_context(_traj())
    assert cell == sq.flat_cell(2, 3)
    assert tblock == hour_to_block_index(time_bucket_to_hour(97))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_sequences.py -v`
Expected: FAIL (ModuleNotFoundError on `gan.sequences`)

- [ ] **Step 3: Implement `sequences.py`**

Create `famail_temporal/baselines/gan/sequences.py`:

```python
"""Trajectory <-> cell-token sequence encoding and conditioning context."""
from __future__ import annotations
from typing import List, Tuple

from famail_temporal.baselines.gan import config as gc
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
from famail_temporal.utils.trajectory import Trajectory


def flat_cell(x: int, y: int) -> int:
    return int(x) * gc.GY + int(y)


def unflat_cell(idx: int) -> Tuple[int, int]:
    return divmod(int(idx), gc.GY)


def trajectory_to_tokens(traj: Trajectory) -> List[int]:
    """[BOS, cell_0, ..., cell_{L-1}, EOS] of flat cell ids."""
    cells = [flat_cell(s.x_grid, s.y_grid) for s in traj.states]
    return [gc.BOS] + cells + [gc.EOS]


def trajectory_context(traj: Trajectory) -> Tuple[int, int]:
    """(start flat-cell, start time-block) for conditioning."""
    s0 = traj.states[0]
    t_block = hour_to_block_index(time_bucket_to_hour(s0.time_bucket))
    return flat_cell(s0.x_grid, s0.y_grid), t_block
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_sequences.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/sequences.py famail_temporal/baselines/gan/tests/test_sequences.py
git commit -m "feat(baselines/gan): trajectory<->cell-token encoding + context"
```

---

## Task 3: `generator.py` — conditional LSTM language model

**Files:**
- Create: `famail_temporal/baselines/gan/generator.py`
- Test: `famail_temporal/baselines/gan/tests/test_generator.py`

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/gan/tests/test_generator.py`:

```python
"""Unit tests for gan.generator.TrajectoryLSTM."""
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM


def test_forward_returns_vocab_logits():
    model = TrajectoryLSTM()
    B, L = 4, 7
    tokens = torch.randint(0, gc.N_CELLS, (B, L))
    ctx_cell = torch.randint(0, gc.N_CELLS, (B,))
    ctx_tblock = torch.randint(0, gc.N_TBLOCKS, (B,))
    logits = model(tokens, ctx_cell, ctx_tblock)
    assert logits.shape == (B, L, gc.VOCAB_SIZE)


def test_context_changes_logits():
    """Different conditioning context must change the output distribution."""
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    tokens = torch.randint(0, gc.N_CELLS, (1, 5))
    c0 = torch.tensor([0]); c1 = torch.tensor([gc.N_CELLS - 1])
    tb = torch.tensor([0])
    out0 = model(tokens, c0, tb)
    out1 = model(tokens, c1, tb)
    assert not torch.allclose(out0, out1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_generator.py -v`
Expected: FAIL (ImportError on `gan.generator`)

- [ ] **Step 3: Implement `generator.py`**

Create `famail_temporal/baselines/gan/generator.py`:

```python
"""Conditional autoregressive LSTM over the grid-cell vocabulary.

forward() returns next-token logits for teacher-forced MLE training. The
conditioning context (start cell + start time-block) is injected by adding a
context embedding to every input-token embedding. A Gumbel-softmax sampling
path can be layered on later (Phase 3) without changing this interface.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from famail_temporal.baselines.gan import config as gc


class TrajectoryLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int = gc.VOCAB_SIZE,
        n_tblocks: int = gc.N_TBLOCKS,
        embed_dim: int = gc.EMBED_DIM,
        hidden_dim: int = gc.HIDDEN_DIM,
        n_layers: int = gc.N_LAYERS,
    ):
        super().__init__()
        self.cell_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=gc.PAD)
        self.tblock_embed = nn.Embedding(n_tblocks, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers, batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,      # (B, L) long input token ids
        ctx_cell: torch.Tensor,    # (B,) long start-cell ids
        ctx_tblock: torch.Tensor,  # (B,) long start time-block ids
    ) -> torch.Tensor:
        x = self.cell_embed(tokens)                                   # (B, L, E)
        ctx = self.cell_embed(ctx_cell) + self.tblock_embed(ctx_tblock)  # (B, E)
        x = x + ctx.unsqueeze(1)                                      # broadcast
        out, _ = self.lstm(x)                                         # (B, L, H)
        return self.head(out)                                        # (B, L, V)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_generator.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/generator.py famail_temporal/baselines/gan/tests/test_generator.py
git commit -m "feat(baselines/gan): conditional LSTM trajectory language model"
```

---

## Task 4: `train_mle.py` — MLE training loop

**Files:**
- Create: `famail_temporal/baselines/gan/train_mle.py`
- Test: `famail_temporal/baselines/gan/tests/test_train_mle.py`

- [ ] **Step 1: Write the failing test** (overfit a tiny batch — the canonical "training actually learns" check)

Create `famail_temporal/baselines/gan/tests/test_train_mle.py`:

```python
"""Unit test for gan.train_mle: the model can overfit a tiny dataset."""
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.train_mle import train_mle


def test_overfits_tiny_dataset():
    torch.manual_seed(0)
    # Two short fixed sequences with fixed contexts.
    sequences = [
        [gc.BOS, 10, 11, 12, gc.EOS],
        [gc.BOS, 40, 41, gc.EOS],
    ]
    contexts = [(10, 0), (40, 1)]
    model = TrajectoryLSTM()
    losses = train_mle(
        model, sequences, contexts,
        epochs=200, lr=1e-2, batch_size=2, device=torch.device("cpu"),
    )
    # Loss should fall substantially as the model memorizes the two sequences.
    assert losses[-1] < losses[0] * 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_train_mle.py -v`
Expected: FAIL (ImportError on `gan.train_mle`)

- [ ] **Step 3: Implement `train_mle.py`**

Create `famail_temporal/baselines/gan/train_mle.py`:

```python
"""Maximum-likelihood (next-token) training for the trajectory LSTM."""
from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM


def _pad_batch(
    seqs: List[List[int]], device: torch.device,
) -> torch.Tensor:
    """Right-pad a list of token sequences to (B, Lmax) with PAD."""
    lmax = max(len(s) for s in seqs)
    out = torch.full((len(seqs), lmax), gc.PAD, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return out


def train_mle(
    model: TrajectoryLSTM,
    sequences: List[List[int]],
    contexts: List[Tuple[int, int]],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> List[float]:
    """Train `model` by next-token cross-entropy. Returns per-epoch mean loss.

    Teacher forcing: predict tokens[1:] from tokens[:-1]. PAD positions are
    ignored by the loss.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=gc.PAD)
    n = len(sequences)
    epoch_losses: List[float] = []

    for _ in range(epochs):
        perm = torch.randperm(n)
        batch_losses: List[float] = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size].tolist()
            batch = _pad_batch([sequences[i] for i in idx], device)
            ctx_cell = torch.tensor(
                [contexts[i][0] for i in idx], dtype=torch.long, device=device,
            )
            ctx_tblock = torch.tensor(
                [contexts[i][1] for i in idx], dtype=torch.long, device=device,
            )
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            logits = model(inp, ctx_cell, ctx_tblock)         # (B, L-1, V)
            loss = loss_fn(
                logits.reshape(-1, gc.VOCAB_SIZE), tgt.reshape(-1),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.item()))
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
    return epoch_losses
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_train_mle.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/train_mle.py famail_temporal/baselines/gan/tests/test_train_mle.py
git commit -m "feat(baselines/gan): MLE next-token training loop"
```

---

## Task 5: `rollout.py` — sampling → pickups → demand grid

**Files:**
- Create: `famail_temporal/baselines/gan/rollout.py`
- Test: `famail_temporal/baselines/gan/tests/test_rollout.py`

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/gan/tests/test_rollout.py`:

```python
"""Unit tests for gan.rollout."""
import numpy as np
import torch

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan import rollout as rl
from famail_temporal.baselines.datasets import pickup_mass


def test_sample_cells_are_valid_and_bounded():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    cells = rl.sample_trajectory_cells(
        model, ctx_cell=5, ctx_tblock=0,
        max_len=16, device=torch.device("cpu"),
    )
    assert 0 <= len(cells) <= 16
    assert all(0 <= c < gc.N_CELLS for c in cells)  # specials stripped


def test_pickups_to_grid_scale_and_placement():
    bundle = _make_synthetic_bundle()
    # Two pickups at distinct units; grid mass equals sum of pickup masses.
    pickups = [(2, 3, 0), (2, 3, 0), (4, 5, 1)]
    grid = rl.pickups_to_pickup_3d(bundle, pickups)
    assert grid.shape == bundle.pickup_3d.shape
    assert grid[2, 3, 0] == np.float32(2 * pickup_mass(bundle, 0))
    assert grid[4, 5, 1] == np.float32(pickup_mass(bundle, 1))
    # Untouched cells are zero.
    assert grid.sum() > 0
    grid[2, 3, 0] = 0.0
    grid[4, 5, 1] = 0.0
    assert np.allclose(grid, 0.0)


def test_generate_pickups_is_seed_deterministic():
    model = TrajectoryLSTM()
    contexts = [(5, 0), (9, 1), (3, 0)]
    torch.manual_seed(7)
    a = rl.generate_pickups(model, contexts, max_len=16, device=torch.device("cpu"))
    torch.manual_seed(7)
    b = rl.generate_pickups(model, contexts, max_len=16, device=torch.device("cpu"))
    assert a == b
    assert len(a) == len(contexts)
    # Each pickup inherits its context's time-block (Phase-2 simplification).
    assert [p[2] for p in a] == [c[1] for c in contexts]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_rollout.py -v`
Expected: FAIL (ImportError on `gan.rollout`)

- [ ] **Step 3: Implement `rollout.py`**

Create `famail_temporal/baselines/gan/rollout.py`:

```python
"""Autoregressive sampling and demand-grid aggregation for generations."""
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.sequences import unflat_cell
from famail_temporal.baselines.datasets import pickup_mass
from famail_temporal.data.loader import DataBundle


@torch.no_grad()
def sample_trajectory_cells(
    model: TrajectoryLSTM, ctx_cell: int, ctx_tblock: int,
    *, max_len: int, device: torch.device, temperature: float = 1.0,
) -> List[int]:
    """Sample one trajectory's cell ids (BOS/EOS/specials stripped).

    Autoregressive multinomial decode from BOS; stops at EOS or max_len.
    Only in-vocabulary *cell* ids (< N_CELLS) are kept.
    """
    model.to(device).train(False)   # inference mode (no dropout/grad)
    cc = torch.tensor([ctx_cell], dtype=torch.long, device=device)
    tb = torch.tensor([ctx_tblock], dtype=torch.long, device=device)
    seq = [gc.BOS]
    cells: List[int] = []
    for _ in range(max_len):
        inp = torch.tensor([seq], dtype=torch.long, device=device)
        logits = model(inp, cc, tb)[0, -1]               # (V,)
        probs = torch.softmax(logits / temperature, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        if nxt == gc.EOS:
            break
        seq.append(nxt)
        if nxt < gc.N_CELLS:                              # ignore stray specials
            cells.append(nxt)
    return cells


def generate_pickups(
    model: TrajectoryLSTM, contexts: List[Tuple[int, int]],
    *, max_len: int, device: torch.device,
) -> List[Tuple[int, int, int]]:
    """One rollout per context; pickup = terminal cell, t_block = context block.

    If a rollout produces no cells, it falls back to the start cell so every
    context yields a pickup (keeps the generated grid corpus-matched).
    """
    out: List[Tuple[int, int, int]] = []
    for (ctx_cell, ctx_tblock) in contexts:
        cells = sample_trajectory_cells(
            model, ctx_cell, ctx_tblock, max_len=max_len, device=device,
        )
        terminal = cells[-1] if cells else ctx_cell
        x, y = unflat_cell(terminal)
        out.append((x, y, ctx_tblock))
    return out


def pickups_to_pickup_3d(
    bundle: DataBundle, pickups: List[Tuple[int, int, int]],
) -> np.ndarray:
    """Aggregate generated pickups into a mean-hourly demand grid.

    Each pickup adds pickup_mass(t_block) at its (cell, t_block), mirroring the
    editing modifier's accounting so the generated grid is scale-comparable to
    bundle.pickup_3d.
    """
    grid = np.zeros_like(bundle.pickup_3d)
    for (x, y, t_block) in pickups:
        grid[x, y, t_block] += pickup_mass(bundle, t_block)
    return grid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_rollout.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/rollout.py famail_temporal/baselines/gan/tests/test_rollout.py
git commit -m "feat(baselines/gan): autoregressive sampling + demand-grid aggregation"
```

---

## Task 6: `b0.py` — orchestrate the B0 baseline

**Files:**
- Create: `famail_temporal/baselines/gan/b0.py`
- Test: `famail_temporal/baselines/gan/tests/test_b0.py`

- [ ] **Step 1: Write the failing test** (end-to-end on a tiny synthetic bundle with real trajectories)

Create `famail_temporal/baselines/gan/tests/test_b0.py`:

```python
"""End-to-end B0 on a tiny synthetic bundle."""
import torch

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import active_units, make_traj_at
from famail_temporal.baselines.gan import b0


def test_run_b0_returns_generated_and_corpus_fairness():
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 20)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    out = b0.run_b0(
        bundle, epochs=3, max_len=8, device=torch.device("cpu"), seed=0,
    )
    assert set(out) == {"generated", "corpus", "n_generated"}
    for key in ("generated", "corpus"):
        m = out[key]
        assert set(m) == {"f_spatial", "f_causal", "gini_dsr", "gini_asr"}
        assert 0.0 <= m["f_causal"] <= 1.0
    assert out["n_generated"] == len(bundle.trajectories)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_b0.py -v`
Expected: FAIL (ImportError on `gan.b0`)

- [ ] **Step 3: Implement `b0.py`**

Create `famail_temporal/baselines/gan/b0.py`:

```python
"""B0 baseline: train an MLE trajectory generator on a dataset, generate
rollouts, and measure the generations' data-level fairness against the corpus.

The B0 claim: a generative model trained on biased data reproduces the bias
in its generations (generated fairness ~ corpus fairness, possibly worse via
mode collapse). This module also IS the reusable train->generate->grid->
fairness pipeline that the filtered/edited variants and the adversarial layer
build on.
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
from famail_temporal.baselines.gan.rollout import (
    generate_pickups, pickups_to_pickup_3d,
)
from famail_temporal.baselines.metrics import data_level_fairness


def run_b0(
    bundle: DataBundle, *,
    epochs: int = gc.MLE_EPOCHS,
    max_len: int = gc.MAX_GEN_LEN,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict:
    """Train on bundle.trajectories, generate one rollout per trajectory's
    context, and return generated vs corpus fairness."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(seed)

    sequences = [trajectory_to_tokens(t) for t in bundle.trajectories]
    contexts = [trajectory_context(t) for t in bundle.trajectories]

    model = TrajectoryLSTM().to(device)
    train_mle(
        model, sequences, contexts,
        epochs=epochs, lr=gc.MLE_LR, batch_size=gc.MLE_BATCH_SIZE, device=device,
    )

    pickups = generate_pickups(model, contexts, max_len=max_len, device=device)
    gen_grid = pickups_to_pickup_3d(bundle, pickups)

    return {
        "generated": data_level_fairness(bundle, pickup_3d=gen_grid),
        "corpus": data_level_fairness(bundle),
        "n_generated": len(pickups),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_b0.py -v`
Expected: PASS

- [ ] **Step 5: Run the full gan test suite**

Run: `python -m pytest famail_temporal/baselines/gan/ -v`
Expected: PASS (all tests from Tasks 1–6)

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/gan/b0.py famail_temporal/baselines/gan/tests/test_b0.py
git commit -m "feat(baselines/gan): B0 train->generate->grid->fairness pipeline"
```

---

## Task 7: `run_b0.py` — CLI + real-data smoke

**Files:**
- Create: `famail_temporal/baselines/gan/run_b0.py`
- Test: `famail_temporal/baselines/gan/tests/test_run_b0.py`

- [ ] **Step 1: Write the failing test** (the JSON-serialization helper is the only pure logic to unit-test)

Create `famail_temporal/baselines/gan/tests/test_run_b0.py`:

```python
"""Unit test for run_b0 result serialization."""
import json

from famail_temporal.baselines.gan import run_b0 as r


def test_result_to_json_roundtrips():
    result = {
        "generated": {"f_spatial": 0.08, "f_causal": 0.80,
                      "gini_dsr": 0.9, "gini_asr": 0.9},
        "corpus": {"f_spatial": 0.082, "f_causal": 0.805,
                   "gini_dsr": 0.94, "gini_asr": 0.9},
        "n_generated": 105401,
    }
    blob = r.result_to_json(result)
    loaded = json.loads(blob)
    assert loaded["n_generated"] == 105401
    assert loaded["corpus"]["f_causal"] == 0.805
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_run_b0.py -v`
Expected: FAIL (ImportError on `gan.run_b0`)

- [ ] **Step 3: Implement `run_b0.py`**

Create `famail_temporal/baselines/gan/run_b0.py`:

```python
"""CLI: train the B0 generative baseline on the real corpus and report
generated-vs-corpus fairness.

Example:
    python -m famail_temporal.baselines.gan.run_b0 --epochs 5 --device auto
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
from famail_temporal.baselines.gan.b0 import run_b0


def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2)


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="famail_temporal.baselines.gan.run_b0")
    ap.add_argument("--epochs", type=int, default=gc.MLE_EPOCHS)
    ap.add_argument("--max-len", type=int, default=gc.MAX_GEN_LEN)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(config.PACKAGE_ROOT) / "results" / "b0")
    args = ap.parse_args(argv)

    bundle = DataBundle.load()
    result = run_b0(
        bundle, epochs=args.epochs, max_len=args.max_len,
        device=_resolve_device(args.device), seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "b0_fairness.json").write_text(result_to_json(result))
    print(f"corpus    F_causal={result['corpus']['f_causal']:.4f}")
    print(f"generated F_causal={result['generated']['f_causal']:.4f}")
    print(f"wrote {args.out_dir / 'b0_fairness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/gan/tests/test_run_b0.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/gan/run_b0.py famail_temporal/baselines/gan/tests/test_run_b0.py
git commit -m "feat(baselines/gan): B0 CLI entry point"
```

- [ ] **Step 6: Real-data smoke (manual; needs cache + GPU recommended)**

Run: `python -m famail_temporal.baselines.gan.run_b0 --epochs 5 --device auto`
Expected: writes `famail_temporal/results/b0/b0_fairness.json`. Inspect: `corpus.f_causal ≈ 0.805`; `generated.f_causal` should land *near* the corpus value (bias reproduced). A large gap in either direction is a finding to record (under-fit generator vs. mode-collapse amplification), not something to "fix" by changing fairness code — flag for discussion.

---

## Self-Review

**1. Spec coverage (Phase 2 MLE-keystone portion):**
- Grid-cell sequence representation + conditioning `(start cell, time-block)` — Tasks 2, 3. ✓ (spec §4.1)
- Autoregressive LSTM generator — Task 3. ✓
- MLE pretraining (spec §4.2 step 1) — Task 4. ✓
- Rollout → pickup/dropoff grid aggregation (the model-level seam) — Task 5. ✓ (spec §4.4)
- B0 = train on raw, generate, measure fairness (spec §4.3) — Tasks 6, 7. ✓
- Adversarial fine-tune, discriminator, B1 loss, B2/FAMAIL model-level, pure-GAN, multi-seed — **deferred to Phase 3 by design** (stated up front).

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step is complete; every test step has assertions + an exact command. ✓

**3. Type consistency:** `TrajectoryLSTM(...)` constructor + `forward(tokens, ctx_cell, ctx_tblock)` signature identical across generator/train/rollout/tests. `train_mle(model, sequences, contexts, *, epochs, lr, batch_size, device)` keyword-only signature matches all callers (test + b0). `generate_pickups(model, contexts, *, max_len, device)` and `pickups_to_pickup_3d(bundle, pickups)` match b0's usage. `run_b0(bundle, *, epochs, max_len, device, seed)` returns `{generated, corpus, n_generated}`, asserted in test_b0 and consumed by run_b0/CLI. Vocabulary constants (`N_CELLS/BOS/EOS/PAD/VOCAB_SIZE/N_TBLOCKS`) defined once in `gan/config.py` and imported everywhere. ✓

**4. Ambiguity:** Pickup time-block = conditioning start block (Phase-2 simplification, stated in scope + rollout docstring + asserted in test_rollout). One rollout per real context → corpus-matched generation. Empty rollouts fall back to the start cell so every context yields a pickup. All explicit.

---
