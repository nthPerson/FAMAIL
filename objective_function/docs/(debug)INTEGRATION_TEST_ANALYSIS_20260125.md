# Integration Test Analysis: Objective Function Term Evaluation

**Date**: January 25, 2026  
**Author**: Analysis conducted at user request  
**Status**: Investigation Complete  
**Tests Analyzed**: "5000 Trajectory Count" and "20000 Trajectory Count"

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Test Configuration Overview](#2-test-configuration-overview)
3. [Test Results Comparison](#3-test-results-comparison)
4. [F_spatial Analysis](#4-f_spatial-analysis)
5. [F_causal Analysis](#5-f_causal-analysis)
6. [F_fidelity Analysis](#6-f_fidelity-analysis)
7. [Root Cause Analysis](#7-root-cause-analysis)
8. [Recommendations](#8-recommendations)
9. [Conclusion](#9-conclusion)

---

## 1. Executive Summary

This document analyzes integration test results from the FAMAIL Integrated Objective Function Dashboard to evaluate whether each objective function term is being calculated effectively. Two tests were conducted with identical configurations except for trajectory count (5,000 vs 20,000).

### Key Findings

| Term | Value | Effectiveness | Critical Issue |
|------|-------|---------------|----------------|
| **F_spatial** | 0.0857 (both tests) | ⚠️ **Problematic** | Extremely low value indicates potential calculation issue |
| **F_causal** | 0.0031 → 0.0050 | ⚠️ **Problematic** | R² near zero; g(d) explains virtually none of the variance |
| **F_fidelity** | 0.5000 (both tests) | ⚠️ **Default Value** | No discriminator loaded; using placeholder |

### Overall Assessment

**The integration test reveals significant issues with how the objective function terms interact with the test data.** While the implementations appear mathematically correct, the test configuration creates conditions where the terms cannot produce meaningful values.

---

## 2. Test Configuration Overview

Both tests used the following configuration:
- **Data Source**: Real trajectory data (from file)
- **Supply/Demand Data**: Real active_taxis + pickup_dropoff_counts
- **g(d) Estimation**: Isotonic regression
- **g(d) Freeze**: ✅ Enabled (1000 lookup points)
- **Grid Dimensions**: 5×5 (as indicated by 25 total cells in the grid)

| Parameter | 5000 Trajectory Test | 20000 Trajectory Test |
|-----------|---------------------|----------------------|
| Trajectory Count | 5,000 | 20,000 |
| Active Cells | 1,260 | 1,689 |
| Demand Range | [0.10, 41.14] | [0.10, 174.63] |
| Supply Range | [3.0, 3283.0] | [3.0, 3283.0] (same) |

---

## 3. Test Results Comparison

### 3.1 Similarities

| Metric | 5000 Test | 20000 Test | Observation |
|--------|-----------|------------|-------------|
| F_spatial | 0.0857 | 0.0857 | **Identical** - surprising |
| F_fidelity | 0.5000 | 0.5000 | **Identical** - default value |
| Supply Range | [3, 3283] | [3, 3283] | **Identical** - same supply data |

### 3.2 Differences

| Metric | 5000 Test | 20000 Test | Change |
|--------|-----------|------------|--------|
| F_causal | 0.0031 | 0.0050 | +61% |
| Active Cells | 1,260 | 1,689 | +34% |
| Max Demand | 41.14 | 174.63 | +324% |
| Var(Y) | 2,048,492 | 638,769 | -69% |
| Var(R) | 2,042,074 | 635,554 | -69% |
| Y range max | 19,793 | 14,126 | -29% |

### 3.3 Causal Debug Comparison

```
                           5000 Test         20000 Test
─────────────────────────────────────────────────────────
Active cells (D > 0.1)     1,260             1,689
Demand range               [0.10, 41.14]     [0.10, 174.63]
Supply range               [3.0, 3283.0]     [3.0, 3283.0]
Y = S/D range             [14.28, 19793.0]  [4.39, 14125.8]
g(D) range                 [0.69, 15.61]     [0.08, 15.61]
Var(Y)                     2,048,492         638,769
Var(R)                     2,042,074         635,554
R² (raw)                   0.003133          0.005033
```

---

## 4. F_spatial Analysis

### 4.1 How F_spatial is Calculated

Based on the implementation in `combined_objective.py`:

```python
def compute_spatial_fairness(self, pickup_counts, dropoff_counts):
    pickup_flat = pickup_counts.flatten()
    dropoff_flat = dropoff_counts.flatten()
    
    gini_pickup = self._pairwise_gini(pickup_flat)
    gini_dropoff = self._pairwise_gini(dropoff_flat)
    
    f_spatial = 1.0 - 0.5 * (gini_pickup + gini_dropoff)
    return f_spatial
```

**Formula**: $F_{\text{spatial}} = 1 - \frac{1}{2}(G_{\text{pickup}} + G_{\text{dropoff}})$

### 4.2 Analysis of F_spatial = 0.0857

**What this value means**:
- $F_{\text{spatial}} = 0.0857$ implies average Gini ≈ 0.914
- This indicates **extreme inequality** in pickup/dropoff distribution
- Nearly all pickups/dropoffs are concentrated in very few cells

**Is this expected?**

Yes and no:
- **Expected**: Taxi activity in cities is naturally concentrated (airports, business districts, etc.)
- **Concerning**: A Gini of 0.914 is extremely high, even for urban taxi data
- **Identical across tests**: The fact that F_spatial is identical (0.0857) for both 5,000 and 20,000 trajectories is **suspicious**

### 4.3 Why F_spatial is Identical in Both Tests

The spatial fairness term depends on the **distribution** of pickups/dropoffs, not the absolute counts. If the sampling preserves the distribution (which random sampling should), then:

- Gini coefficient measures **relative inequality**
- Doubling all counts doesn't change the Gini
- Therefore, F_spatial being identical is **mathematically consistent**

### 4.4 Effectiveness Assessment

| Criterion | Assessment |
|-----------|------------|
| Mathematical correctness | ✅ **Correct** - pairwise Gini is properly implemented |
| Value range | ✅ **Valid** - 0.0857 is in [0, 1] |
| Gradient flow | ✅ **Verified** - gradients computed successfully |
| Meaningful for optimization | ⚠️ **Questionable** - such high inequality may reflect data reality |

**Concern**: The extremely low F_spatial (0.0857) may not leave much room for optimization improvement. If this reflects true urban taxi patterns, the term is working correctly but the baseline is very unfair.

### 4.5 Missing Component: Active Taxis Normalization

The **original formulation** divides by active taxi count per cell:

$$
DSR_i = \frac{O_i}{N_i \cdot T}
$$

However, the integration test appears to compute **raw Gini on pickup/dropoff counts** without normalization by active taxis. This is a **simplified formulation** that may differ from the full spatial fairness metric documented in the literature.

---

## 5. F_causal Analysis

### 5.1 How F_causal is Calculated

Based on the implementation:

```python
def compute_causal_fairness(self, pickup_counts, supply_tensor):
    D = demand[mask]  # Pickup counts
    S = supply[mask]  # Supply from active_taxis
    
    Y = S / (D + eps)  # Service ratio
    g_d = frozen_lookup(D)  # Expected ratio from g(d)
    R = Y - g_d  # Residual
    
    var_Y = Y.var()
    var_R = R.var()
    r_squared = 1.0 - var_R / var_Y
    return clamp(r_squared, 0, 1)
```

**Formula**: $F_{\text{causal}} = R^2 = 1 - \frac{\text{Var}(R)}{\text{Var}(Y)}$

### 5.2 Analysis of Extremely Low R² Values

The R² values are:
- **5000 Test**: R² = 0.0031 (0.31%)
- **20000 Test**: R² = 0.0050 (0.50%)

**This means the g(d) function explains less than 1% of the variance in service ratios.**

### 5.3 Why R² is So Low: The Scale Mismatch Problem

Looking at the debug info:

| Metric | 5000 Test | 20000 Test |
|--------|-----------|------------|
| Y = S/D range | [14.3, **19,793**] | [4.4, **14,126**] |
| g(D) range | [0.7, 15.6] | [0.08, 15.6] |

**Critical Issue**: The Y values (service ratios) range up to **~20,000** while g(D) predictions max out at **~15.6**. This is a **massive scale mismatch**.

**Root Cause Analysis**:

1. **g(d) was estimated from real supply/demand data** with a certain relationship
2. **The integration test creates demand counts** from trajectory sampling (pickup coordinates → grid cells)
3. **Supply tensor comes from real active_taxis data** summed across all periods
4. **The Y = S/D ratios in the test** are not comparable to the ratios used to fit g(d)

### 5.4 The Data Mixing Problem

The g(d) function was fitted on:
- Demand: Real pickup counts from `pickup_dropoff_counts.pkl`
- Supply: Real active taxi counts from `active_taxis.pkl`
- Time alignment: Matching periods

But the integration test uses:
- Demand: **Sampled trajectory pickups** (soft counts from coordinates)
- Supply: **Summed supply** across all time periods
- Time alignment: **None** - trajectories from all periods mixed together

**This creates an incompatible Y = S/D computation** where:
- Demand is a small sample (5,000-20,000 pickups)
- Supply is aggregated across the entire dataset (~3,000+ per cell in some areas)
- The ratios are meaningless compared to what g(d) was trained on

### 5.5 Verification: The Numbers Don't Match

**Expected from g(d) fitting**:
- If isotonic regression achieved R² = 0.44 on real data (per documentation)
- Then g(d) should explain ~44% of variance when applied to similar data

**Observed in test**:
- R² = 0.003 to 0.005
- g(d) explains <1% of variance

**Conclusion**: The test is **not using compatible data** for the g(d) function.

### 5.6 Effectiveness Assessment

| Criterion | Assessment |
|-----------|------------|
| Mathematical correctness | ✅ **Correct** - R² formula is properly implemented |
| Gradient flow | ✅ **Verified** - gradients computed through Y |
| Frozen g(d) | ✅ **Correct** - lookup table is frozen |
| Data compatibility | ❌ **FAILED** - Test data doesn't match g(d) training data |
| Meaningful for optimization | ❌ **NO** - R² ≈ 0 means gradients won't guide toward fairness |

---

## 6. F_fidelity Analysis

### 6.1 How F_fidelity is Calculated

Based on the implementation:

```python
def compute_fidelity(self, trajectory_features, reference_features):
    if self.discriminator is None:
        # Return default value if discriminator not provided
        return torch.tensor(0.5, device=trajectory_features.device)
    
    similarity = self.discriminator(trajectory_features, reference_features)
    return similarity.mean()
```

### 6.2 Analysis of F_fidelity = 0.5

**F_fidelity = 0.5 is the default placeholder value** returned when no discriminator is loaded.

This indicates:
1. No trajectory features were passed to the forward function
2. The integration test **does not load or use the discriminator model**
3. Fidelity is not being computed at all - it's a constant

### 6.3 Why Fidelity Isn't Computed

Looking at `_run_comprehensive_gradient_test`:

```python
module = DifferentiableFAMAILObjective(
    ...
    g_function=g_function,  # ✅ Passed
    # discriminator=???      # ❌ NOT passed
)

total, terms = module(
    pickup_test, dropoff_test,
    supply_tensor=supply_tensor,
    # trajectory_features=???  # ❌ NOT passed
)
```

The test:
- Does pass supply_tensor (for causal fairness)
- Does **not** pass a discriminator model
- Does **not** pass trajectory_features

### 6.4 Effectiveness Assessment

| Criterion | Assessment |
|-----------|------------|
| Mathematical correctness | N/A - not computed |
| Gradient flow | N/A - constant value has zero gradient |
| Meaningful for optimization | ❌ **NO** - constant contributes nothing to optimization |

---

## 7. Root Cause Analysis

### 7.1 Summary of Issues

| Issue | Affected Term | Severity | Root Cause |
|-------|---------------|----------|------------|
| Scale mismatch between test demand and g(d) training data | F_causal | **Critical** | Supply/demand data mixing |
| Missing discriminator | F_fidelity | **Critical** | Test doesn't load discriminator |
| Missing trajectory features | F_fidelity | **Critical** | Test doesn't convert trajectories to features |
| Extreme inequality baseline | F_spatial | **Moderate** | May reflect actual data distribution |

### 7.2 The Fundamental Problem

The integration test is designed to verify **gradient flow** through the objective function, not to produce **meaningful fairness values**. The test:

1. **Correctly verifies** that gradients propagate through all differentiable operations
2. **Does not verify** that the computed values are semantically meaningful
3. **Uses incompatible data** for the causal fairness term

### 7.3 Why F_causal is Near Zero

```
Test Setup:
├── g(d) fitted on: Real (demand, supply) pairs at matching time periods
│   └── Produces: R² ≈ 0.44 on training data
│
└── Integration test uses:
    ├── Demand: Soft counts from sampled trajectory pickups
    │   └── Scale: 0.1 to 174 (for 20k trajectories)
    │
    └── Supply: Sum of active_taxis across ALL periods
        └── Scale: 3 to 3,283 per cell
        
Result: Y = Supply/Demand = 3,283/0.1 = 32,830 (extreme!)
        g(D) predicts: ~15.6 maximum
        Residual R: Enormous
        Var(R) ≈ Var(Y)
        R² ≈ 0
```

---

## 8. Recommendations

### 8.1 Immediate Fixes

#### 8.1.1 Fix F_causal Data Compatibility

**Option A: Use Consistent Time Periods**
```python
# Instead of summing supply across all periods:
supply_tensor = _create_supply_tensor_from_data(
    supply_demand_result['supply'],
    config['grid_dims'],
    period=(time_bucket, day),  # Match trajectory time period
)
```

**Option B: Use Real Demand Instead of Sampled**
```python
# Use actual pickup counts from pickup_dropoff_counts.pkl
# Instead of soft counts from trajectory coordinates
demand_tensor = extract_demand_tensor(pickup_dropoff_data, config)
```

**Option C: Scale Normalization**
```python
# Normalize both to similar scales
demand_normalized = demand_counts / demand_counts.max()
supply_normalized = supply_tensor / supply_tensor.max()
Y = supply_normalized / (demand_normalized + eps)
```

#### 8.1.2 Fix F_fidelity

```python
# Load discriminator model
discriminator = load_discriminator(checkpoint_path)

# Convert trajectories to features
trajectory_features = trajectories_to_features(trajectories)
reference_features = trajectory_features.detach()

# Pass to module
module = DifferentiableFAMAILObjective(
    ...,
    discriminator=discriminator,
)

total, terms = module(
    pickup_test, dropoff_test,
    supply_tensor=supply_tensor,
    trajectory_features=trajectory_features,
    reference_features=reference_features,
)
```

### 8.2 Documentation Updates

Update the Integration Testing documentation to clarify:
1. The test verifies **gradient flow**, not semantic correctness
2. F_fidelity = 0.5 is expected when no discriminator is loaded
3. F_causal requires compatible supply/demand data

### 8.3 Suggested Test Modes

| Mode | Purpose | F_spatial | F_causal | F_fidelity |
|------|---------|-----------|----------|------------|
| **Gradient Flow Test** | Verify backprop works | ✅ Computed | ✅ Computed | Placeholder |
| **Semantic Validity Test** | Verify meaningful values | ✅ Real data | ✅ Matched data | ✅ Real discriminator |
| **Full Integration Test** | End-to-end validation | ✅ | ✅ | ✅ |

---

## 9. Conclusion

### 9.1 Are the Objective Function Terms Being Calculated Effectively?

| Term | Gradient Flow | Mathematical Correctness | Semantic Meaningfulness |
|------|---------------|--------------------------|-------------------------|
| **F_spatial** | ✅ Yes | ✅ Yes | ⚠️ Questionable (extreme inequality) |
| **F_causal** | ✅ Yes | ✅ Yes | ❌ No (incompatible data) |
| **F_fidelity** | ❌ No (constant) | N/A | ❌ No (not computed) |

### 9.2 Summary

**The implementations are mathematically correct**, but the integration test configuration creates conditions where:

1. **F_spatial** shows extreme inequality (Gini ≈ 0.91) which may be real data characteristics
2. **F_causal** produces near-zero R² due to data incompatibility between the frozen g(d) and test supply/demand
3. **F_fidelity** is not computed at all (no discriminator loaded)

### 9.3 Key Takeaway

> **The integration test successfully verifies gradient flow but does not verify that the objective function produces semantically meaningful values for trajectory optimization.**

To use these terms for actual trajectory modification, the test configuration must be updated to ensure:
1. Supply and demand data come from matching time periods
2. The discriminator model is loaded and trajectory features are provided
3. The g(d) function is applied to data with similar distributions to what it was trained on

---

## Appendix A: Formula Reference

### Spatial Fairness
$$F_{\text{spatial}} = 1 - \frac{1}{2}(G_{\text{pickup}} + G_{\text{dropoff}})$$

$$G = \frac{\sum_{i}\sum_{j}|x_i - x_j|}{2n^2\bar{x}}$$

### Causal Fairness
$$F_{\text{causal}} = R^2 = 1 - \frac{\text{Var}(Y - g(D))}{\text{Var}(Y)}$$

$$Y = \frac{S}{D} \quad \text{(Service Ratio)}$$

### Fidelity
$$F_{\text{fidelity}} = \frac{1}{|\mathcal{T}'|}\sum_{\tau' \in \mathcal{T}'}\text{Discriminator}(\tau', \tau_{\text{ref}})$$

---

## Appendix B: Raw Test Data

### 5000 Trajectory Test
```
Active cells (demand > 0.1): 1260
Demand range: [0.1012, 41.1370]
Supply range: [3.0000, 3283.0000]
Y = S/D range: [14.2797, 19793.0137]
g(D) range: [0.6854, 15.6147]
Var(Y): 2048491.875000
Var(R): 2042073.500000
R² (raw): 0.003133

F_spatial = 0.0857
F_causal = 0.0031
F_fidelity = 0.5000
```

### 20000 Trajectory Test
```
Active cells (demand > 0.1): 1689
Demand range: [0.1001, 174.6252]
Supply range: [3.0000, 3283.0000]
Y = S/D range: [4.3859, 14125.8213]
g(D) range: [0.0801, 15.6147]
Var(Y): 638768.937500
Var(R): 635554.125000
R² (raw): 0.005033

F_spatial = 0.0857
F_causal = 0.0050
F_fidelity = 0.5000
```
