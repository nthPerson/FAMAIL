# Discriminator Model Analysis and Improvement Plan

## Executive Summary

Analysis of the ST-SiameseNet discriminator model reveals a **critical architectural issue**: the model fails to recognize identical trajectory inputs as belonging to the same agent. When the same trajectory is fed as both inputs, the model outputs probability ~0.0 instead of the expected ~1.0.

**Key Finding**: The model has 89% validation accuracy on the training distribution but **generalizes poorly** to the identical-trajectory case because such pairs were never part of training data.

---

## Problem Statement

### Test Results Summary

| Test Mode | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Identical Trajectories | ~1.0 | 0.27 | ❌ FAIL |
| Same Driver, Same Day | >0.5 | 0.30 | ❌ FAIL |
| Same Driver, Different Days | >0.5 | 0.52 | ✅ PASS |
| Different Drivers | <0.5 | 0.52 | ❌ FAIL |

### Root Cause Analysis

Direct model testing revealed:

```python
# Identical trajectory test
traj = [[15.0, 25.0, 100.0, 3.0]] * 50
model(traj, traj)  # Output: 0.0000 (expected ~1.0)

# Training data positive pair
model(x1, x2)      # Output: 0.7240 (expected >0.5) ✓
model(x1, x1)      # Output: 0.0000 (expected ~1.0) ✗
```

**The classifier produces logits of -12.45 for identical inputs, resulting in probability 0.0000.**

---

## Technical Analysis

### 1. Dataset Generation Issue

The training dataset (`traj_pair_5000pos_5000neg_80-20_split`) was generated with:

```json
{
  "positive_pairs": 5000,
  "negative_pairs": 5000,
  "positive_strategy": "random",
  "negative_strategy": "random"
}
```

**Problem**: Positive pairs are always **different trajectories** from the same agent. The model never sees the case where both inputs are **identical**.

### 2. Model Architecture

```
SiameseLSTMDiscriminator:
├── FeatureNormalizer (4 → 6 features)
├── SiameseLSTMEncoder (shared weights)
│   └── LSTM: hidden_dim=128, num_layers=2
├── Concatenation: [emb1, emb2] → 256 dims
└── Classifier: Linear(256→128) → ReLU → Linear(128→64) → ReLU → Linear(64→1)
```

**Issue**: The concatenation-based classifier learns patterns specific to the **difference** between embeddings, not their **similarity**. When embeddings are identical, the learned pattern doesn't match.

### 3. Training Distribution vs. Inference Distribution

| Scenario | Training | Inference (Fidelity Term) |
|----------|----------|---------------------------|
| Identical trajectories | Never seen | Common (edited ≈ original) |
| Same agent, different days | Common | Common |
| Different agents | Common | N/A for fidelity |

The model was not trained on the distribution it needs to handle.

---

## Proposed Solutions

### Solution 1: Augment Training Data with Identical Pairs (Quick Fix)

Add identical trajectory pairs (same trajectory as both x1 and x2) to the training dataset:

```python
# In dataset generation, add:
for agent_id, trajs in trajectories.items():
    for traj in trajs:
        # Add pair where x1 == x2 with label=1
        identical_pairs.append((traj, traj, label=1))
```

**Recommendation**: Add 10-20% identical pairs to training data.

### Solution 2: Use Distance-Based Similarity (Recommended)

Replace concatenation with a distance-based combination:

```python
# Current (problematic):
combined = torch.cat([emb1, emb2], dim=-1)  # [batch, 256]

# Better options:
# Option A: Absolute difference
combined = torch.abs(emb1 - emb2)  # [batch, 128]

# Option B: Element-wise product + difference
combined = torch.cat([
    emb1 * emb2,           # similarity signal
    torch.abs(emb1 - emb2) # difference signal
], dim=-1)  # [batch, 256]

# Option C: Cosine similarity + L2 distance
cos_sim = F.cosine_similarity(emb1, emb2, dim=-1, keepdim=True)
l2_dist = torch.norm(emb1 - emb2, dim=-1, keepdim=True)
combined = torch.cat([cos_sim, l2_dist, emb1, emb2], dim=-1)
```

### Solution 3: Add Contrastive Loss Component

Add a contrastive loss term to explicitly learn that identical embeddings → high similarity:

```python
def contrastive_loss(emb1, emb2, label, margin=1.0):
    distance = F.pairwise_distance(emb1, emb2)
    loss = label * distance.pow(2) + \
           (1 - label) * F.relu(margin - distance).pow(2)
    return loss.mean()

total_loss = bce_loss + lambda_contrastive * contrastive_loss
```

### Solution 4: Follow ST-iFGSM Architecture More Closely

The original ST-iFGSM paper uses:
- LSTM: 200 and 100 hidden units (2 layers)
- Similarity learner: [64, 32, 8, 1] FC layers with ReLU
- **Key**: They likely use a distance-based similarity metric

---

## Implementation Priority

### Phase 1: Quick Fixes (1-2 days)

1. **Add identical pairs to training data**
   - Modify `generation.py` to include identical trajectory pairs
   - Re-generate dataset with 10-20% identical pairs
   - Re-train model

2. **Add validation tests to training**
   - Include identical-trajectory test in validation metrics
   - Early stopping should consider this metric

### Phase 2: Architecture Improvements (3-5 days)

1. **Implement distance-based similarity**
   - Create `SiameseLSTMDiscriminatorV2` with improved combination
   - Compare multiple combination methods

2. **Add contrastive loss**
   - Implement contrastive loss component
   - Tune hyperparameters (margin, lambda)

### Phase 3: Comprehensive Evaluation (2-3 days)

1. **Create comprehensive test suite**
   - Identical trajectories
   - Same agent, same day
   - Same agent, different days
   - Different agents
   - Perturbed trajectories (small edits)

2. **Benchmark against baselines**
   - Simple L2 distance classifier
   - Cosine similarity threshold
   - Current architecture

---

## Recommended Dataset Configuration

```json
{
  "positive_pairs": 6000,
  "negative_pairs": 5000,
  "identical_pairs": 1000,
  "days": 2,
  "positive_strategy": "random",
  "negative_strategy": "random",
  "augmentations": {
    "include_identical": true,
    "identical_ratio": 0.15,
    "noise_augmentation": false
  }
}
```

---

## Expected Outcomes After Fixes

| Test Mode | Current | Expected After Fix |
|-----------|---------|-------------------|
| Identical Trajectories | 0.27 | >0.95 |
| Same Driver, Same Day | 0.30 | >0.70 |
| Same Driver, Different Days | 0.52 | >0.70 |
| Different Drivers | 0.52 | <0.30 |

---

## Files to Modify

1. **`discriminator/dataset_generation_tool/generation.py`**
   - Add `include_identical_pairs` option
   - Add `identical_pair_ratio` parameter

2. **`discriminator/model/model.py`**
   - Create `SiameseLSTMDiscriminatorV2` with distance-based combination
   - Add optional contrastive loss method

3. **`discriminator/model/train.py`**
   - Add identical-trajectory validation metric
   - Update early stopping criteria

4. **`discriminator/model/trainer.py`**
   - Implement contrastive loss option
   - Add comprehensive validation tests

---

## Conclusion

The current discriminator model has a fundamental generalization issue that makes it unsuitable for the Fidelity Term's requirements. The model works on its training distribution (different trajectories from same/different agents) but fails catastrophically on identical trajectories.

**Immediate Action**: Implement Solution 1 (add identical pairs) as a quick fix, then proceed with Solution 2 (distance-based similarity) for a more robust long-term solution.

---

*Document created: January 14, 2026*
*Analysis by: FAMAIL Development Team*
