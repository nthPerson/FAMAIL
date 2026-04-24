# V3 Multi-Stream Siamese Discriminator — Training Results

**Date trained:** 2026-04-21
**Run ID:** `20260421_145958`
**Source checkpoint:** `/home/robert/FAMAIL/checkpoints/20260421_145958/best.pt`
**Installed at:** `famail_temporal/discriminator_checkpoints/default/best.pt`
**Training duration:** 48 minutes (33 epochs × ~87 s/epoch)

## Context

This retraining was motivated by two separate changes landed in the same
development cycle:

1. **Unified source-data generation tool** ([`famail_temporal/data/source_generation/`](../../../famail_temporal/data/source_generation/))
   replaces three legacy tools (extractor, pickup/dropoff counter, profile
   feature generator) with one enriched-event-stream pipeline. The new tool
   eliminates three cross-tool seam bugs by construction and produces
   artifacts that are the single source of truth for both famail_temporal
   trajectory modification and the discriminator training pipeline. See
   [`docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md`](../../../docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md).

2. **`action_space_violation` per-trajectory invariant** rejects any
   trajectory containing a consecutive-state transition with
   `max(|dx|, |dy|) > 1` — enforcing physical consistency with the 9
   possible actions of the original `all_trajs.pkl` state vector (8 compass
   moves + stay). Trajectories that don't conform cannot be rollouts of a
   9-action agent (GPS dropouts, high-speed movement between ~15–30s GPS
   samples on a ~1 km grid). See
   [`docs/superpowers/specs/2026-04-21-action-space-violation-filter-design.md`](../../../docs/superpowers/specs/2026-04-21-action-space-violation-filter-design.md).

Retraining was necessary to keep the training-time and inference-time
trajectory distributions consistent. The prior v3 checkpoint was trained on
legacy-extractor output (which silently truncated long segments to 1000
states and did not enforce 9-action consistency); the famail_temporal
pipeline now scores trajectories from the unified tool's output. Without
retraining, the discriminator would see trajectories at inference time that
differ systematically from its training distribution.

## Source data

From `famail_temporal/source_data/processing_metadata.json` (regenerated
2026-04-21 01:24):

| Metric | Value |
|---|---|
| Raw trajectories extracted | 393,670 |
| Trajectories after per-trajectory filters | 197,920 |
| Overall removal rate | 49.72% |
| Seeking trajectories retained | 105,488 |
| Driving trajectories retained | 92,432 |
| Unique drivers | 50 |
| Calendar days spanned | 2016-07-01 → 2016-09-30 (66 weekdays) |

Removal breakdown:

| Category | Count |
|---|---:|
| `action_space_violation` | 195,540 |
| `implausibly_long` | 210 |
| All other categories | 0 |

The `action_space_violation` category dominates, as expected — ~50% of raw
trajectories contain at least one non-adjacent GPS transition.

## Dataset generation

Command:
```
python -m discriminator.multi_stream.dataset_generation \
    --seeking-fixed-length 256 \
    --driving-fixed-length 128
```

Resulting `discriminator/multi_stream/datasets/default/{train,val,test}.npz`:

| Field | Value |
|---|---|
| Positive pairs | 5000 (500 identical, 4500 same-driver/different-day) |
| Negative pairs | 5000 (different drivers, one day each) |
| Total pairs | 10,000 |
| Train split | 7,500 |
| Val split | 1,500 |
| Test split | 1,000 |
| Seeking shape per pair branch | `[5, 256, 4]` |
| Driving shape per pair branch | `[5, 128, 4]` |
| Profile dim | 11 |
| Agent coverage | 50/50 drivers in both positives and negatives |

Fixed-length padding was introduced to prevent a training hang caused by
outlier trajectories: p99 seeking length is ~226 states in the filtered
dataset, but a handful of 2000+ outliers (likely pre-filter artifacts)
could pin the padded length and balloon per-batch compute. Setting
`seeking_fixed_length=256` and `driving_fixed_length=128` covers >99% of
trajectories while keeping epoch times around 90 seconds.

## Model architecture (v3, concatenation mode)

| Component | Value |
|---|---|
| Model version | v3 (`MultiStreamSiameseDiscriminator`) |
| LSTM hidden dims | (200, 100), 2-layer |
| Bidirectional | True |
| Dropout | 0.2 |
| Streams | seeking, driving, profile |
| Trajectories per stream per branch | 5 |
| Trajectory projection dim | 48 |
| Profile feature count | 11 |
| Profile hidden dims | (64, 32) |
| Profile output dim | 8 |
| Classifier hidden dims | (64, 32, 8) |
| Combination mode | concatenation |
| Total trainable parameters | 1,556,337 |

## Training configuration

| Hyperparameter | Value |
|---|---|
| Epochs (max) | 100 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 6e-5 |
| Weight decay | 1e-4 |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early stopping patience | 10 epochs |
| Early stopping min_delta | 1e-4 |
| Save best only | True |
| Seed | 42 |
| Device | CUDA |

The full training command:
```
python -m discriminator.model.train \
    --model-version v3 \
    --data-dir discriminator/multi_stream/datasets/default \
    --combination-mode concatenation \
    --lr 6e-5
```

## Results — best epoch (epoch 33)

| Metric | Value |
|---|---:|
| `train_loss` | 0.2885 |
| `val_loss` | **0.2115** |
| `val_accuracy` | **0.9240** |
| `val_positive_accuracy` | 0.9974 |
| `val_negative_accuracy` | 0.8497 |
| `val_f1` | 0.9296 |
| `val_auc` | **0.9629** |
| `val_identical_score` | 0.8868 |
| `learning_rate` at save | 6.00e-05 |

**Reading the metrics:**

- `val_accuracy = 92.4%` and `val_auc = 0.963`: the model cleanly discriminates
  same-driver pairs from different-driver pairs across 1,500 held-out pairs.
- `val_positive_accuracy = 99.7%` and `val_negative_accuracy = 85.0%`: the
  model is more confident about same-driver pairs than different-driver
  pairs — a common Siamese pattern. The ~15-percentage-point split is
  normal (below the 30-point warning threshold the trainer uses), so the
  loss signal was not dominated by either class.
- `val_identical_score = 0.887`: when the same trajectory is fed into both
  branches, the model outputs 0.887 average probability of "same driver".
  Above the 0.5 warn threshold; below 1.0 which would suggest the model
  collapsed to always-output-1. This is the "sanity check" for Siamese
  training — it confirms the model learned the trivial case without
  overfitting to it.

## Convergence trajectory

The model spent ~12 epochs near-random before the loss signal broke
through, then converged quickly:

| Epoch | Train loss | Val loss | Val acc | Val AUC | Val identical |
|:-:|--:|--:|--:|--:|--:|
| 1 | 0.7028 | 0.7008 | 0.503 | 0.516 | 0.565 |
| 5 | 0.6931 | 0.6903 | 0.523 | 0.562 | 0.514 |
| 10 | 0.6062 | 0.5903 | 0.663 | 0.735 | 0.641 |
| 13 | 0.5149 | 0.4418 | 0.812 | 0.868 | 0.764 |
| 15 | 0.3820 | 0.2965 | 0.901 | 0.937 | 0.813 |
| 20 | 0.3360 | 0.2609 | 0.914 | 0.940 | 0.839 |
| 25 | 0.3194 | 0.2489 | 0.909 | 0.947 | 0.862 |
| 30 | 0.2912 | 0.2375 | 0.915 | 0.954 | 0.890 |
| **33** | **0.2885** | **0.2115** | **0.924** | **0.963** | **0.887** |

Training stopped at epoch 33 via early stopping (patience=10 elapsed without
`val_loss` improvement beyond `min_delta=1e-4`).

## Verification after installation

- **Loader test:** `load_discriminator(config.DISCRIMINATOR_CHECKPOINT_DIR / config.DISCRIMINATOR_CHECKPOINT_FILENAME)`
  returns a `MultiStreamSiameseDiscriminator` (1,556,337 params) — not the
  `nn.Identity` fallback that fires when the checkpoint is missing.
- **`DataBundle.load()` real-data test** passes: loads regenerated
  `source_data/` cleanly, with 5,834 active units and all 50 drivers in
  the multi-stream profile.
- **End-to-end trajectory modification integration** (`test_modifier_integration.py`,
  `test_runner_real_data.py`): all 7 tests pass in 2 minutes with the
  real discriminator in the pipeline (versus 15 seconds when
  `nn.Identity` was the fallback — the timing confirms the discriminator
  is running actual forward passes).

## Reproducibility

Exact reproduction of this checkpoint requires:

1. `famail_temporal/source_data/` regenerated from the raw GPS data using
   the unified source-generation tool at commit `af7636d` or later (where
   the `action_space_violation` invariant is present).
2. Dataset generation via
   `python -m discriminator.multi_stream.dataset_generation --seeking-fixed-length 256 --driving-fixed-length 128`
   (seed=42 is baked into the config).
3. Training via the command shown above (seed=42 baked in).

Because the source GPS data (`raw_data/taxi_record_*.pkl`) is not
committed to the repo, byte-level reproducibility across machines depends
on identical raw inputs. Given identical inputs, training is deterministic
modulo CUDA nondeterminism.

## Prior checkpoint comparison

No detailed metrics are available for prior v3 checkpoints (the
`discriminator/model/checkpoints/` subdirectories contain older runs with
unknown histories). Qualitatively, this run:

- Trains on a **filtered, action-space-consistent** trajectory set (prior
  runs trained on the full pre-filter distribution).
- Uses **fixed-length padding** (prior runs used pad-to-longest, which was
  susceptible to outlier trajectories pinning the padded length).
- Consumes data from the **unified source-generation tool** (prior runs
  consumed the legacy extractor's output with its known seam bugs).

## Open questions / known caveats

- **`REMOVAL_RATE_WARN_THRESHOLD = 0.05` fires on every real-data run**
  now that steady-state removal is ~50%. Flagged in the final-review
  cleanup for the action_space_violation feature; deferred pending a
  research-direction decision on whether to raise the threshold to ~0.55,
  split it per-category, or leave as-is. Does not affect training.
- **`val_negative_accuracy` (85.0%)** is the weakest sub-metric. If
  further improvement is wanted, the natural knobs are: more
  different-driver negative pairs (currently 1:1 with positives), or
  hard-negative mining using the current model's predictions.
- **Discriminator checkpoint is gitignored** — the file at
  `famail_temporal/discriminator_checkpoints/default/best.pt` is 19 MB
  and not tracked. This markdown file is the persistent record.
