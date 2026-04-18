# `fidelity/` — Discriminator-based trajectory realism check

## Purpose

Provide `F_fidelity`, a score in [0, 1] measuring whether a modified trajectory remains
indistinguishable from authentic expert trajectories. Implemented by porting the pre-trained
Multi-Stream Siamese discriminator from the parent codebase as an **opaque inference-only
module**: no training code, no dataset classes, just the four classes needed to load and run
the checkpoint.

---

## Files

| File | Role |
|---|---|
| `model.py` | `FeatureNormalizer`, `SiameseLSTMEncoder`, `ProfileEncoder`, `MultiStreamSiameseDiscriminator` — four ported classes |
| `checkpoint.py` | `load_discriminator()` — loads `discriminator_checkpoints/default/best.pt`, sets eval mode, freezes parameters |
| `context.py` | `MultiStreamContextBuilder` + `MultiStreamData` — assembles the five multi-stream inputs for one trajectory |
| `compute.py` | `compute_ffidelity()` — runs the discriminator forward pass with the cuDNN workaround |

---

## Key design choices

### 1. Checkpoint is opaque — no training code ported

The parent codebase (`discriminator/model/model.py`) contains 1,297 lines across 8 classes,
including training loops, dataset classes, and 5 deprecated architectures. Only the 4 classes
needed for inference are ported:

| Ported | Excluded |
|---|---|
| `FeatureNormalizer` | All `Dataset` and `DataLoader` classes |
| `SiameseLSTMEncoder` | All `Trainer` classes |
| `ProfileEncoder` | Deprecated architectures (V1, V2) |
| `MultiStreamSiameseDiscriminator` | Training-mode branches (`model.train()`) |

The checkpoint format includes an `architecture_config` dict (added during one-time
preprocessing; see `discriminator_checkpoints/README.md`). If this key is absent,
`load_discriminator()` raises specifically — partial loads are not silently tolerated.

### 2. The cuDNN backward-in-inference workaround

cuDNN's optimized RNN kernel does not support backward passes when the module is in eval mode.
However, ST-iFGSM requires gradient flow through the discriminator's LSTM while also requiring
inference-mode behavior (dropout disabled). The solution is to wrap the discriminator forward
pass in:

```python
with torch.backends.cudnn.flags(enabled=False):
    similarity = discriminator(x1, x2, ...)
```

This disables the cuDNN RNN kernel for that call, falling back to the pure-PyTorch
implementation which supports backward in any mode. Without this workaround, calling
`loss.backward()` after a discriminator forward pass in eval mode raises a cuDNN
`RuntimeError`. The workaround is preserved verbatim from the V3 codebase.

### 3. Multi-stream context builder decisions (preserved verbatim from V3)

Four implementation decisions from the current codebase are carried forward unchanged:

| Decision | Value | Rationale |
|---|---|---|
| D1 | Both Siamese branches represent the same driver | Cross-driver comparison found empirically worse |
| D2 | Seeking fill strategy = 'sample', N=5, slot 0 is target | Matches V3 training setup |
| D3 | Coordinate convention: +1 offset when injecting into context | V3 discriminator trained on 1-indexed coords; modifier is 0-indexed |
| D4 | Gradient flows through slot 0 of x2 only | Only the modified trajectory's slot carries the perturbation gradient |

These decisions are documented here (not in comments) because they encode non-obvious
cross-component contracts. Changing any of them without retraining the discriminator would
silently corrupt `F_fidelity`.

### 4. `ALPHA_FIDELITY = 0` cleanly skips the entire pathway

If `config.ALPHA_FIDELITY == 0`, `FAMAILObjective.forward()` skips the discriminator call
entirely:

```python
if ALPHA_FIDELITY > 0:
    f_fidelity, _ = compute_ffidelity(discriminator, ...)
else:
    f_fidelity = 0.0
```

This means the discriminator checkpoint does not need to exist, and no GPU memory is consumed
by the model, when running fairness-only experiments. It also means `ALPHA_FIDELITY = 0` is a
clean ablation condition.

---

## API surface

```python
from famail_temporal.fidelity.checkpoint import load_discriminator
from famail_temporal.fidelity.context import MultiStreamContextBuilder, MultiStreamData
from famail_temporal.fidelity.compute import compute_ffidelity

# Load discriminator (called once during DataBundle.load())
discriminator = load_discriminator()  # returns MultiStreamSiameseDiscriminator, eval mode

# Build context for one trajectory (called per trajectory in modifier)
builder = MultiStreamContextBuilder(multi_stream_data)
context_kwargs = builder.build(driver_idx, traj_idx, modified_pickup_tensor)
# Returns dict of tensors ready for discriminator(**context_kwargs)

# Compute fidelity score (differentiable; cuDNN workaround applied internally)
f_fidelity, breakdown = compute_ffidelity(discriminator, anchor_kwargs, modified_kwargs)
# Returns: f_fidelity in [0, 1], breakdown dict with raw similarity score
```

---

## Dependencies

- `config.py` — `ALPHA_FIDELITY`, discriminator checkpoint path
- `utils/trajectory.py` — `Trajectory`, `TrajectoryState` (coordinate access)
- Third-party: `torch`

No imports from `fairness/`, `data/`, or `algorithm/`.

---

## Paper-section hook

This module corresponds to the **"Fidelity Term"** subsection of the Methods section. The cuDNN
workaround rationale and the multi-stream context decisions may appear in a technical appendix
("Implementation Details"). Checkpoint provenance (training date, dataset, performance metrics)
appears in the Supplementary Materials section on discriminator training.
