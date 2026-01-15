# Discriminator Model Improvement Implementation Summary

## Date: January 2025

## Overview

This document summarizes the improvements implemented to fix the discriminator model's inability to handle identical trajectory inputs correctly.

---

## Problem Identified

The original discriminator model (`SiameseLSTMDiscriminator`) produced probability ~0.0 when comparing a trajectory to itself, instead of ~1.0. Root cause: concatenation-based architecture trained without identical pairs.

| Test Mode | Expected | Original Model |
|-----------|----------|----------------|
| Identical Trajectories | ~1.0 | 0.27 ❌ |
| Same Driver, Same Day | >0.5 | 0.30 ❌ |
| Same Driver, Different Days | >0.5 | 0.52 ✅ |
| Different Drivers | <0.5 | 0.52 ❌ |

---

## Implemented Solutions

### 1. Dataset Generation: Identical Pair Support

**File**: `discriminator/dataset_generation_tool/generation.py`

**Changes**:
- Added `identical_pair_ratio` parameter to `GenerationConfig` (default: 0.0)
- Created `sample_identical_pairs()` function
- Updated `assemble_dataset()` to include identical pairs
- Added `pair_type` tracking: "identical", "same_agent", "different_agent"

**Usage**:
```python
config = GenerationConfig(
    data_path=Path("source_data/all_trajs.pkl"),
    positive_pairs=5000,
    negative_pairs=5000,
    identical_pair_ratio=0.15,  # 15% of positive pairs will be identical
)

# Result: 750 identical + 4250 same_agent + 5000 different_agent pairs
```

---

### 2. New Model Architecture: SiameseLSTMDiscriminatorV2

**File**: `discriminator/model/model.py`

The V2 model uses **distance-based** embedding combination instead of concatenation:

| Mode | Combination Formula | Use Case |
|------|---------------------|----------|
| `difference` | `\|emb1 - emb2\|` | Default, handles identical inputs |
| `distance` | `\|emb1 - emb2\|` + cosine + euclidean | Richer features |
| `hybrid` | `concat(emb1, emb2)` + difference + metrics | Backward compatible |

**Why this works for identical inputs**:
- V1: `concat(emb, emb)` → classifier learned this pattern = NOT same agent
- V2: `|emb - emb| = 0` → classifier learns zero difference = same agent

**Usage**:
```python
from discriminator.model import SiameseLSTMDiscriminatorV2

model = SiameseLSTMDiscriminatorV2(
    hidden_dim=128,
    num_layers=2,
    combination_mode="difference"
)
```

---

### 3. Training Script Updates

**File**: `discriminator/model/train.py`

**New arguments**:
```bash
--model-version v1|v2          # Select model architecture
--combination-mode MODE        # V2 only: difference|distance|hybrid
```

**Example**:
```bash
# Train V2 model with distance-based combination
python train.py --data dataset.npz \
    --model-version v2 \
    --combination-mode difference \
    --epochs 100
```

---

### 4. Identical Trajectory Validation Metric

**File**: `discriminator/model/trainer.py`

Training now validates model behavior on identical inputs every epoch:

```
Epoch   1/100 | Train: 0.623 | Val: 0.512 | Acc: 0.750 | AUC: 0.823 | Identical: 0.950 | 2.3s *
Epoch   2/100 | Train: 0.521 | Val: 0.489 | Acc: 0.782 | AUC: 0.856 | Identical: 0.962 | 2.1s *
...
Epoch  50/100 | Train: 0.312 | Val: 0.345 | Acc: 0.891 | AUC: 0.924 | Identical: 0.235 ⚠️ LOW | 2.0s
```

If "Identical" score drops below 0.5, a warning is shown.

---

### 5. Dataset Generation UI

**File**: `discriminator/dataset_generation_tool/app.py`

Added "Identical Pair Settings" section:
- Slider for `identical_pair_ratio` (0.0 to 0.5, default 0.15)
- Shows estimated identical pair count
- Helpful tooltip explaining why identical pairs matter

---

## Next Steps

1. **Generate new dataset** with identical pairs:
   ```bash
   # Run the Streamlit app
   streamlit run discriminator/dataset_generation_tool/app.py
   
   # Set identical_pair_ratio = 0.15 (15%)
   # Generate dataset
   ```

2. **Train V2 model**:
   ```bash
   python discriminator/model/train.py \
       --data-dir discriminator/datasets/new_dataset \
       --model-version v2 \
       --combination-mode difference \
       --epochs 100
   ```

3. **Verify improvements** using the Fidelity Dashboard:
   ```bash
   streamlit run objective_function/spatial_fairness/fidelity/dashboard.py
   # Use "Discriminator Tests" tab
   ```

---

## Expected Results After Re-training

| Test Mode | Current | Target |
|-----------|---------|--------|
| Identical Trajectories | 0.27 | >0.95 |
| Same Driver, Same Day | 0.30 | >0.70 |
| Same Driver, Different Days | 0.52 | >0.70 |
| Different Drivers | 0.52 | <0.30 |

---

## Files Modified

| File | Change |
|------|--------|
| `discriminator/dataset_generation_tool/generation.py` | Added identical pair support |
| `discriminator/model/model.py` | Added SiameseLSTMDiscriminatorV2 |
| `discriminator/model/train.py` | Added V2 model support |
| `discriminator/model/trainer.py` | Added identical validation metric |
| `discriminator/model/__init__.py` | Exported V2 model |
| `discriminator/dataset_generation_tool/app.py` | Added UI for identical ratio |
| `discriminator/DISCRIMINATOR_ANALYSIS_AND_IMPROVEMENT_PLAN.md` | Original analysis |
