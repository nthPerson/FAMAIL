# V3 Multi-Stream Siamese Discriminator — Training Results

**Date trained:** 2026-04-24
**Run ID:** `20260423_233120`
**Source checkpoint:** `/home/robert/FAMAIL/checkpoints/20260423_233120/best.pt`
**Installed at:** `famail_temporal/discriminator_checkpoints/default/best.pt`
**Training duration:** 141 minutes (100 epochs × ~50 s/epoch + validation overhead)

## Context

This retraining was motivated by three compounding changes landed in the same
development cycle:

1. **Unified source-data generation tool** ([`famail_temporal/data/source_generation/`](../../data/source_generation/))
   replaces three legacy tools. See
   [`docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md`](../../../docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md).

2. **`action_space_violation` per-trajectory invariant** rejects
   trajectories with non-adjacent consecutive-state transitions, enforcing
   9-action agent consistency. See
   [`docs/superpowers/specs/2026-04-21-action-space-violation-filter-design.md`](../../../docs/superpowers/specs/2026-04-21-action-space-violation-filter-design.md).

3. **Tighter `implausibly_long` threshold** — `MAX_TRAJECTORY_DURATION_BUCKETS`
   from 120 (10h) to 96 (8h), reflecting that a single seeking or driving
   episode shouldn't exceed a standard work day. Minor impact on the
   dataset size (90 additional trajectories dropped).

The prior v3 checkpoint (dated 2026-04-21) was trained with the 10-hour
threshold; this retraining aligns the discriminator's training distribution
with the current production pipeline.

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

Removal breakdown (delta from prior 10h-threshold regeneration):

| Category | Count | Δ from prior |
|---|---:|---:|
| `action_space_violation` | 195,540 | no change |
| `implausibly_long` | 300 | +90 (tighter threshold) |

## Dataset generation

Command:
```
python -m discriminator.multi_stream.dataset_generation \
    --seeking-fixed-length 256 \
    --driving-fixed-length 128
```

Pair counts and shapes identical to prior run (10,000 pairs, 7,500/1,500/1,000
train/val/test, seeking [5,256,4] / driving [5,128,4] / profile [11]).
Agent coverage: 50/50 drivers in both positives and negatives.

## Model architecture (v3, concatenation mode)

Unchanged from prior run:

| Component | Value |
|---|---|
| Model version | v3 (`MultiStreamSiameseDiscriminator`) |
| LSTM hidden dims | (200, 100), 2-layer, bidirectional |
| Dropout | 0.2 |
| Streams | seeking, driving, profile |
| Trajectories per stream per branch | 5 |
| Total trainable parameters | 1,556,337 |

## Training configuration

Unchanged from prior run:

| Hyperparameter | Value |
|---|---|
| Epochs (max) | 100 |
| Batch size | 32 |
| Optimizer | Adam, lr=6e-5, weight_decay=1e-4 |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early stopping patience | 10 epochs |
| Seed | 42 |

## Results — best epoch (epoch 100)

Note: training reached the full 100-epoch budget without triggering early
stopping. Val loss was still improving in the final epochs (see convergence
trajectory below). The prior run early-stopped at epoch 33 when val loss
plateaued; this run did not plateau within the 100-epoch budget.

| Metric | Value | Prior (2026-04-21) | Δ |
|---|---:|---:|---:|
| `train_loss` | 0.3304 | 0.2885 | +0.042 |
| `val_loss` | **0.3217** | **0.2115** | **+0.110** |
| `val_accuracy` | **0.9153** | **0.9240** | −0.009 |
| `val_positive_accuracy` | 0.9987 | 0.9974 | +0.001 |
| `val_negative_accuracy` | 0.8336 | 0.8497 | −0.016 |
| `val_f1` | 0.9212 | 0.9296 | −0.008 |
| `val_auc` | **0.9296** | **0.9629** | **−0.033** |
| `val_identical_score` | **0.6305** | **0.8868** | **−0.256** |
| Best epoch | 100 (of 100) | 33 (early-stopped) | — |

### Interpretation

- **val_accuracy and val_AUC are strong** (0.92 and 0.93 respectively) — the
  model cleanly discriminates same-driver from different-driver pairs.
- **val_identical_score dropped from 0.89 to 0.63.** The model is still
  above the 0.5 warning threshold the trainer uses, but noticeably less
  confident about identical-trajectory pairs than the prior run. This
  suggests the trained representation has somewhat less separation between
  the "definitely same driver" end of the spectrum and ambiguous cases.
- **Training did not converge within budget.** Best epoch = final epoch (100)
  with val_loss still trending down. The prior run converged at epoch 33;
  this run had very different loss-curve dynamics. Possible causes:
  - Stochastic variation (same seed but dataset-generation RNG consumes
    different inputs when trajectories differ).
  - The slightly tighter 8-hour filter changed which trajectories entered
    the training set in a way that made the pairing problem harder.
  - The LR scheduler may not have stepped down as aggressively as in the
    prior run.

### Recommendation

The current checkpoint is installed and functional — F_fidelity will
compute nontrivial values and the full pipeline tests pass. However, if
this checkpoint is used for research-quality evaluation (vs. development
smoke testing), it may be worth a follow-up training run with:
- Higher `--epochs` (e.g., 200) to let training complete convergence, OR
- Multiple seeds to assess whether the ~3-point AUC drop is stochastic
  variation or a systematic shift.

The prior checkpoint (epoch-33 run dated 2026-04-21) achieved slightly
stronger metrics and is still available at
`/home/robert/FAMAIL/checkpoints/20260421_145958/best.pt` if a rollback
becomes useful for comparison.

## Convergence trajectory

The loss curve shows steady improvement across all 100 epochs without the
plateau that triggered early stopping in the prior run. Key waypoints:

| Epoch | Train loss | Val loss | Val acc | Val AUC | Val identical |
|:-:|--:|--:|--:|--:|--:|
| 1 | ~0.70 | ~0.70 | 0.50 | 0.52 | 0.57 |
| 25 | ~0.48 | ~0.46 | 0.78 | 0.85 | 0.60 |
| 50 | ~0.41 | ~0.39 | 0.87 | 0.91 | 0.63 |
| 75 | ~0.37 | ~0.35 | 0.90 | 0.92 | 0.63 |
| 100 | 0.3304 | **0.3217** | **0.9153** | **0.9296** | 0.6305 |

(Approximate values at intermediate epochs; exact values recoverable from
the checkpoint's `history` dict.)

## Verification after installation

- **Loader test:** `load_discriminator(...)` returns `MultiStreamSiameseDiscriminator`
  with 1,556,337 parameters — not the `nn.Identity` fallback.
- **DataBundle.load()** passes — loads the new source_data + new checkpoint.
- **Full famail_temporal test suite** — see the Full regression test section
  in the end-of-task summary.

## Reproducibility

Exact reproduction requires:
1. `famail_temporal/source_data/` regenerated after commit `3517ab4` (where
   `MAX_TRAJECTORY_DURATION_BUCKETS = 96` is present).
2. `python -m discriminator.multi_stream.dataset_generation --seeking-fixed-length 256 --driving-fixed-length 128`
3. `python -m discriminator.model.train --model-version v3 --data-dir discriminator/multi_stream/datasets/default --combination-mode concatenation --lr 6e-5`

Seed is 42 throughout. CUDA nondeterminism may cause small numerical
differences across machines.

## Known caveats

- **Not fully converged.** Training did not early-stop. A follow-up longer
  training run is available on request.
- **Weaker identical_score than prior.** 0.63 vs 0.89 — worth watching as
  an indicator of feature-space separation quality.
- **Checkpoint is gitignored.** 19 MB `best.pt` is not tracked by git; this
  markdown is the persistent record.
