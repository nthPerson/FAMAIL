# V3 Multi-Stream Siamese Discriminator — Training Results

**Date trained:** 2026-04-24
**Run ID:** `20260424_120508`
**Source checkpoint:** `checkpoints/20260424_120508/best.pt` (in the parent monorepo)
**Installed at:** `famail_temporal/discriminator_checkpoints/default/best.pt`
**Plots directory:** `checkpoints/20260424_120508/plots/` (8 PNGs, in the parent monorepo)
**Training duration:** ~48 minutes (88 epochs via early stopping at ~33 s/epoch)

## Context

Third v3 retrain in this iteration cycle. Changes from the prior
2026-04-23 run:

1. **`T=4` → `T=24` transition** (full hourly time-block resolution) —
   see [`../../docs/F_CAUSAL_METHODOLOGY_NOTES.md`](../../docs/F_CAUSAL_METHODOLOGY_NOTES.md)
   and [`../../docs/FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md`](../../docs/FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md).
   Forced a companion migration of the F_causal hat-matrix representation
   from dense (O(N²), ~19 GB at T=24) to compact FWL form (O(Np), ~1 MB).
2. **Training pipeline optimizations**: `num_workers=4` + `pin_memory=True`
   + `non_blocking=True` + `persistent_workers=True` in DataLoader; batch
   size 32 → 128; learning rate 6e-5 → 1e-4 (balanced scaling rule);
   mixed-precision training via `torch.amp.autocast('cuda')` + GradScaler.
3. **200 max epochs** (was 100) so training could run to convergence
   rather than hitting the budget.

Effect on wall-clock: ~77 s/epoch (prior config) → ~33 s/epoch (new
config). Net: 88 epochs to early-stop in ~48 minutes (prior run: 100
epochs, never early-stopped, in 141 minutes). Did not exhaust the 200
epoch budget — early-stopping at patience=10 fired at epoch 88 because
val loss hadn't improved past epoch 78's 0.1702.

## Source data

Regenerated 2026-04-23 with `action_space_violation` filter + 8-hour
`implausibly_long` threshold:

| Metric | Value |
|---|---|
| Raw trajectories extracted | 393,670 |
| Trajectories after per-trajectory filters | 197,830 |
| Overall removal rate | 49.75% |
| Seeking trajectories retained | 105,401 |
| Driving trajectories retained | 92,429 |
| Unique drivers | 50 |
| Calendar days spanned | 2016-07-01 → 2016-09-30 (66 weekdays) |

Removal breakdown:

| Category | Count |
|---|---:|
| `action_space_violation` | 195,540 |
| `implausibly_long` | 300 |

## Dataset generation

Command:
```
python -m discriminator.multi_stream.dataset_generation \
    --seeking-fixed-length 256 \
    --driving-fixed-length 128
```

| Field | Value |
|---|---|
| Positive pairs | 5000 (500 identical, 4500 same-driver/different-day) |
| Negative pairs | 5000 |
| Total pairs | 10,000 |
| Train / Val / Test split | 7,500 / 1,500 / 1,000 |
| Seeking shape per pair branch | `[5, 256, 4]` |
| Driving shape per pair branch | `[5, 128, 4]` |
| Profile dim | 11 |
| Agent coverage | 50/50 drivers in both positives and negatives |

## Model architecture (v3, concatenation mode)

Unchanged from prior runs:

| Component | Value |
|---|---|
| Model version | v3 (`MultiStreamSiameseDiscriminator`) |
| LSTM hidden dims | (200, 100), 2-layer, bidirectional |
| Dropout | 0.2 |
| Streams | seeking, driving, profile |
| Trajectories per stream per branch | 5 |
| Trajectory projection dim | 48 |
| Profile hidden dims | (64, 32) |
| Profile output dim | 8 |
| Classifier hidden dims | (64, 32, 8) |
| Combination mode | concatenation |
| Total trainable parameters | 1,556,337 |

## Training configuration

| Hyperparameter | Value |
|---|---|
| Epochs (max) | 200 |
| Batch size | **128** (was 32) |
| Optimizer | Adam, lr=**1e-4** (was 6e-5) |
| Weight decay | 1e-4 |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early stopping patience | 10 epochs |
| Mixed precision | **True** (cuda autocast + GradScaler) |
| DataLoader `num_workers` | **4** (was 0) |
| DataLoader `persistent_workers` | True |
| DataLoader `pin_memory` | True |
| `.to(device, non_blocking=True)` | everywhere |
| Seed | 42 |
| Device | CUDA (NVIDIA RTX 3070) |

Training command:
```
python -m discriminator.model.train \
    --model-version v3 \
    --data-dir discriminator/multi_stream/datasets/default \
    --combination-mode concatenation \
    --batch-size 128 --lr 1e-4 --epochs 200 --num-workers 4
```

## Results — best epoch (epoch 78)

| Metric | Value | Prior (2026-04-23) | Delta |
|---|---:|---:|---:|
| `train_loss` | 0.2123 | 0.3304 | **-0.118** |
| `val_loss` | **0.1702** | 0.3217 | **-0.152** |
| `val_accuracy` | **0.9327** | 0.9153 | +0.017 |
| `val_positive_accuracy` | 1.0000 | 0.9987 | +0.001 |
| `val_negative_accuracy` | 0.8666 | 0.8336 | +0.033 |
| `val_f1` | 0.9364 | 0.9212 | +0.015 |
| `val_auc` | **0.9820** | 0.9296 | **+0.052** |
| `val_identical_score` | **0.9260** | 0.6305 | **+0.295** |
| Epoch time | 33.4 s | 49.5 s | -32% |
| Converged | Yes (early-stop @88) | No (hit 100 cap) | — |

### Reading the metrics

- **`val_auc = 0.982`** — near-perfect ranking between same-driver and
  different-driver pairs. The prior run's 0.930 was already solid; this
  is materially stronger and enters "very high discrimination" territory
  for a Siamese model at this scale.
- **`val_identical_score = 0.926`** — the trained representation now
  strongly agrees that "same trajectory twice = same driver" (probability
  0.926 vs 0.631 in the prior run). Well above the 0.5 warning floor
  and approaching saturation. This is the Siamese sanity metric that
  confirms the model has learned a meaningful similarity space, not a
  degenerate one.
- **`val_negative_accuracy = 0.867`** — still the weakest sub-metric
  (as is typical for Siamese training on finite datasets), but +0.033
  over prior. The gap between positive accuracy (1.0) and negative
  accuracy (0.867) is ~13 percentage points, within the trainer's
  normal-range tolerance (warning fires at >30 pp gap).
- **`val_loss = 0.170`** — training converged (val loss hit a true
  minimum then stopped improving, triggering early-stopping) rather than
  hitting the epoch budget. This means 200 epochs is sufficient headroom
  for this recipe.

## Training plots

All plots at `checkpoints/20260424_120508/plots/` in the parent monorepo:

- `loss_curves.png` — train vs val BCE loss per epoch
- `accuracy_curves.png` — overall, positive-class, negative-class accuracy
- `auc_f1_curves.png` — val AUC and F1
- `identical_curve.png` — Siamese identical-pair sanity score
- `learning_rate_curve.png` — LR schedule (log scale, shows the
  ReduceLROnPlateau step(s) kicking in around epoch 85)
- `roc_curve.png` — ROC on val set (best.pt), AUC 0.982
- `precision_recall_curve.png` — PR curve on val set
- `training_summary.png` — 4-panel overview for presentations

## Convergence trajectory (key waypoints)

| Epoch | Train loss | Val loss | Val acc | Val AUC | Identical |
|:-:|--:|--:|--:|--:|--:|
| 1 | 0.709 | 0.710 | 0.495 | – | 0.587 |
| 10 | 0.695 | 0.693 | 0.524 | – | 0.530 |
| 15 | 0.639 | 0.621 | 0.676 | – | 0.575 |
| 20 | 0.425 | 0.389 | 0.895 | – | 0.609 |
| 30 | 0.301 | 0.250 | 0.909 | – | 0.846 |
| 40 | 0.276 | 0.229 | 0.914 | – | 0.866 |
| 50 | 0.256 | 0.202 | 0.927 | – | 0.883 |
| 60 | 0.239 | 0.193 | 0.927 | – | 0.904 |
| 70 | 0.220 | 0.180 | 0.927 | – | 0.926 |
| **78** | **0.212** | **0.170** | **0.933** | **0.982** | **0.926** |
| 88 | 0.205 | 0.177 | 0.929 | – | 0.936 |
|  — | — | *early stop* | — | — | — |

Breakout from random-chance baseline happened between epochs 12–16
(similar to prior runs). Validation loss floor reached epoch 78;
early-stop triggered at epoch 88 (10-epoch patience).

## Verification after installation

- **Loader test:** `load_discriminator(...)` returns `MultiStreamSiameseDiscriminator`
  with 1,556,337 parameters — not the `nn.Identity` fallback.
- **Compact hat-matrix integration:** verified end-to-end with the new
  T=24 cache; the pipeline uses `compute_fcausal_compact` + compact
  `(X_demo, XtX_inv)` representation, not the dense (I−H, M) form.
- **Full famail_temporal test suite** — see end-of-task summary section
  below.

## Reproducibility

Exact reproduction requires:

1. `famail_temporal/source_data/` regenerated after commit `3517ab4`
   (MAX_TRAJECTORY_DURATION_BUCKETS = 96).
2. `famail_temporal/cache/` rebuilt after the T=24 transition (commit
   `be2b339` or later — contains the compact hat-matrix form).
3. `python -m discriminator.multi_stream.dataset_generation --seeking-fixed-length 256 --driving-fixed-length 128`
4. `python -m discriminator.model.train --model-version v3 --data-dir discriminator/multi_stream/datasets/default --combination-mode concatenation --batch-size 128 --lr 1e-4 --epochs 200 --num-workers 4`

Seed is 42 throughout. CUDA nondeterminism (cuDNN) may cause small
numerical differences across machines — the training is designed to be
robust to that.

## Notes and caveats

- **Training plots are in the working-tree checkpoints directory**, not
  in the installed path. They are not gitignored; consider copying them
  into a tracked location (e.g., `docs/training-runs/2026-04-24/plots/`)
  when preparing the paper.
- The `best.pt` weights were frozen at epoch 78. Training for 10 more
  epochs (up to 88) did NOT improve val_loss — this is evidence that
  we're at a genuine minimum for this hyperparameter recipe, not an
  early-stop artifact.
- The compact hat-matrix form is used end-to-end at inference time;
  no N×N matrix is ever materialized. This was verified by running the
  full famail_temporal test suite (including the slow real-data tests)
  at T=24 with the new checkpoint.
