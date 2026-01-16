# ST-iFGSM Algorithm Reference Documentation

> **Purpose**: This document provides a detailed technical reference for the Spatial-Temporal Iterative Fast Gradient Sign Method (ST-iFGSM) algorithm. It serves as foundational documentation for designing FAMAIL's trajectory editing algorithm.

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Formulation](#problem-formulation)
3. [Algorithm Components](#algorithm-components)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Algorithm Pseudocode](#algorithm-pseudocode)
6. [Detailed Algorithm Walkthrough](#detailed-algorithm-walkthrough)
7. [Intuitive Understanding](#intuitive-understanding)
8. [Key Design Decisions](#key-design-decisions)

---

## Overview

### Background

ST-iFGSM is an adversarial attack algorithm designed for Human Mobility Identification (HuMID) models operating on spatial-temporal trajectory data. The algorithm generates adversarial perturbations that can fool trajectory-based identification models while maintaining realistic trajectory characteristics.

### Core Objective

The algorithm aims to **minimally perturb** a trajectory to cause misclassification, subject to:
- **L∞-norm constraint**: Each GPS point edit must be within ε distance from original
- **L₀-norm constraint**: Minimize the number of GPS points modified

### Key Innovation

Unlike standard FGSM which perturbs all input dimensions, ST-iFGSM introduces:
1. **Iterative refinement** for attack success
2. **Redundancy elimination** to minimize edits via gradient-based pruning

---

## Problem Formulation

### Original Optimization Problem

The HuMID attack objective is formulated as:

$$
\max_{\tau'} \ell(f(\tau, \tau'), y)
$$

Subject to:
$$
\|\tau - \tau'\|_\infty \leq \epsilon, \quad \|\tau - \tau'\|_0 \leq \eta
$$

Where:
- $\tau$ = original trajectory
- $\tau'$ = perturbed trajectory  
- $f$ = HuMID discriminator model
- $y$ = true label (same-driver/different-driver)
- $\epsilon$ = maximum per-GPS-point perturbation distance
- $\eta$ = maximum number of GPS points to modify

### Reformulated Optimization Problem

Since trajectory lengths vary, the problem is reformulated to minimize perturbation count:

$$
\min_{\tau'} \|\tau - \tau'\|_0
$$

Subject to:
$$
\|\tau - \tau'\|_\infty \leq \epsilon, \quad f(\tau') \neq y
$$

This reformulation says: **find the minimum number of GPS point edits needed to successfully fool the model**, where each edit is bounded by ε.

---

## Algorithm Components

### Object Definitions

| Symbol | Type | Description |
|--------|------|-------------|
| $\tau$ | Trajectory | Original (clean) input trajectory |
| $\tau'$ | Trajectory | Current perturbed/attacked trajectory |
| $\delta$ | Vector | Per-step perturbation computed via FGSM-style update |
| $\mathbf{v}$ | Binary Vector | Perturbation mask ($\mathbf{v} \in \{0,1\}^n$) indicating which GPS points are "active" for perturbation |
| $\Delta$ | Vector | Cumulative perturbation after N inner FGSM steps |
| $\alpha$ | Scalar | Step size (learning rate) for each FGSM iteration |
| $\epsilon$ | Scalar | Maximum per-dimension perturbation magnitude (L∞ bound) |
| $N$ | Integer | Number of inner FGSM iterations |
| $\mathbf{g}$ | Vector | Gradient of loss with respect to trajectory |
| $\ell(f(\tau, \tau'), y)$ | Scalar | Loss function measuring model error |

### Loss Function

The loss function $\ell(f(\tau, \tau'), y)$ is the **binary cross-entropy** between:
- The HuMID model's predicted similarity probability $f(\tau, \tau')$
- The true same-driver/different-driver label $y$

**Interpretation**: This loss measures how wrong the model is about whether the two trajectories belong to the same driver.

**Attack Goal**: Maximize this loss to force:
- **False Positive**: Model thinks different drivers are the same
- **False Negative**: Model thinks the same driver is different

---

## Mathematical Foundations

### Standard FGSM

The Fast Gradient Sign Method computes a one-step perturbation:

$$
\tau' = \tau + \epsilon \cdot \text{sign}[\nabla_\tau \ell(f(\tau), y)]
$$

**Limitation**: Single-step attacks often fail to achieve misclassification.

### Iterative FGSM (I-FGSM)

Applies multiple smaller steps for more reliable attacks:

$$
\tau'_0 = \tau
$$

$$
\tau'_{j+1} = \text{clip}\{\tau'_j + \alpha \cdot \text{sign}[\nabla_{\tau'_j} \ell(f(\tau'_j), y)]\}
$$

for $j = 0, 1, \ldots, N-1$

**Key Insight**: Taking many small steps allows the attack to adjust direction based on the evolving loss landscape.

### Gradient-Based Importance

The gradient magnitude $|g_i|$ at GPS point $i$ indicates how much perturbing that point contributes to increasing the loss:
- **Large $|g_i|$**: GPS point $i$ is important for the attack
- **Small $|g_i|$**: GPS point $i$ contributes little; perturbation is redundant

---

## Algorithm Pseudocode

```
Algorithm 1: ST-iFGSM Attack

Input:
  - τ: target trajectory (sequence of n GPS points)
  - f: HuMID model (discriminator)
  - ε: L∞-norm bound (max perturbation distance per GPS point)
  - α: learning rate (step size)
  - N: number of inner FGSM iterations

Output:
  - τ': adversarial trajectory with minimal perturbations

Initialize:
  - v ← (1, 1, ..., 1) ∈ {0,1}^n  // all GPS points active
  - τ' ← τ
  - τ'_prev ← τ

REPEAT:
  // === INNER LOOP: Iterative FGSM (N steps) ===
  τ' ← τ
  FOR j = 0 TO N-1:
    // Step 1: Compute gradient at current τ'
    g^(j) ← ∇_{τ'} ℓ(f(τ'), y)
    
    // Step 2: Compute clipped perturbation
    δ^(j) ← clip(α · sign(g^(j)), -ε, ε)
    
    // Step 3: Apply masked perturbation
    τ' ← τ' + δ^(j) ⊙ v
  END FOR
  
  // === POST INNER LOOP: Compute cumulative perturbation ===
  Δ ← τ' - τ
  
  // === REDUNDANCY ELIMINATION ===
  // Step 4: Compute gradient at final τ'
  g ← ∇_{τ'} ℓ(f(τ'), y)
  
  // Step 5: Find least important GPS points
  i* ← argmin_i(|g_i|) where v_i = 1
  
  // Step 6: Prune weak dimensions from mask
  v_{i*} ← 0
  
  // Step 7: Reapply masked perturbation
  τ'_prev ← τ'  // store successful attack before pruning
  τ' ← τ + Δ ⊙ v

UNTIL f(τ') = f(τ)  // attack no longer successful

RETURN τ'_prev  // return last successful adversarial trajectory
```

---

## Detailed Algorithm Walkthrough

### Phase 1: Initialization

```
v ← (1, 1, ..., 1)
τ' ← τ
```

- **Mask vector v**: Initialized with all 1s, meaning every GPS point is initially eligible for perturbation
- **Perturbed trajectory τ'**: Starts as a copy of the original trajectory

### Phase 2: Inner Loop (Iterative FGSM)

For a fixed mask $\mathbf{v}$, the inner loop runs $N$ iterations to build up a successful attack.

#### Step 2.1: Gradient Computation

$$
\mathbf{g}^{(j)} = \nabla_{\tau'} \ell(f(\tau'), y)
$$

**What this computes**: The gradient tells us, for each GPS coordinate, how changing that coordinate affects the loss.
- Positive gradient: Increasing this coordinate increases the loss
- Negative gradient: Decreasing this coordinate increases the loss

#### Step 2.2: Perturbation Calculation

$$
\delta^{(j)} = \text{clip}(\alpha \cdot \text{sign}(\mathbf{g}^{(j)}), -\epsilon, \epsilon)
$$

**Breakdown**:
1. `sign(g^(j))`: Returns +1 for positive gradients, -1 for negative (direction of steepest ascent)
2. `α · sign(...)`: Takes a step of size α in that direction
3. `clip(..., -ε, ε)`: Ensures each component stays within [-ε, ε] bounds

#### Step 2.3: Masked Update

$$
\tau' \leftarrow \tau' + \delta^{(j)} \odot \mathbf{v}
$$

**Hadamard product** $\delta^{(j)} \odot \mathbf{v}$:
- Where $v_i = 1$: Apply the perturbation $\delta^{(j)}_i$
- Where $v_i = 0$: Zero out the perturbation (GPS point unchanged)

**Result**: τ' is updated only at "active" GPS points selected by the mask.

#### Inner Loop Summary

After $N$ steps:
- $\tau'$ has accumulated $N$ masked FGSM updates
- Each step recalculates the gradient based on the current (evolving) $\tau'$
- The attack direction adapts as the trajectory changes

### Phase 3: Cumulative Perturbation

$$
\Delta = \tau' - \tau
$$

**Purpose**: Record the total perturbation applied during the inner loop. This captures the full "attack recipe" before pruning.

### Phase 4: Redundancy Elimination (Pruning)

This is the key innovation of ST-iFGSM over standard I-FGSM.

#### Step 4.1: Final Gradient Evaluation

$$
\mathbf{g} = \nabla_{\tau'} \ell(f(\tau'), y)
$$

Compute the gradient at the final attacked trajectory to assess which perturbations are most important.

#### Step 4.2: Identify Least Important GPS Points

$$
i^* = \arg\min_{i: v_i = 1}(|g_i|)
$$

**Intuition**: GPS points with small gradient magnitude contribute least to the attack success. Perturbing them is "redundant."

#### Step 4.3: Update Mask

$$
v_{i^*} \leftarrow 0
$$

Turn off the least important GPS points—they will no longer be perturbed in subsequent iterations.

#### Step 4.4: Rebuild Perturbed Trajectory

$$
\tau' = \tau + \Delta \odot \mathbf{v}
$$

**Critical Step**: 
- Take the cumulative perturbation $\Delta$ from the inner loop
- Mask it with the updated $\mathbf{v}$
- This "compresses" the attack by removing unnecessary perturbations

### Phase 5: Stopping Condition

```
UNTIL f(τ') = f(τ)
```

**Interpretation**: Keep pruning GPS points until the attack fails. When $f(\tau') = f(\tau)$, we've pruned too much—the perturbations on remaining GPS points are insufficient for misclassification.

**Return Value**: Return $\tau'_{prev}$—the last trajectory that successfully fooled the model with the minimum number of perturbed GPS points.

---

## Intuitive Understanding

### The Two-Loop Structure

| Loop | Purpose | What Changes |
|------|---------|--------------|
| **Inner Loop** | Build a successful attack | $\tau'$ accumulates perturbations via iterative FGSM |
| **Outer Loop** | Minimize attack footprint | Mask $\mathbf{v}$ gets pruned; only essential perturbations remain |

### Analogy: Sculpture Carving

Think of the algorithm as sculpting:

1. **Inner Loop (Adding Clay)**: Build up a full perturbation that successfully attacks the model—this may modify many GPS points.

2. **Outer Loop (Carving Away)**: Chisel away unnecessary modifications. After each carving step, test if the sculpture still "works" (attack succeeds). Stop when removing any more would break it.

### Gradient as "Importance Score"

The gradient $|g_i|$ serves as an importance score:

```
High |g_i| → GPS point i is crucial for attack → Keep perturbing it
Low  |g_i| → GPS point i is redundant     → Stop perturbing it
```

This is analogous to **saliency-based feature selection** in interpretability methods.

### Evolution of Key Variables

```
Outer Iteration 1:
  v = [1, 1, 1, 1, 1]  (all active)
  Inner loop builds τ' with all GPS points
  Prune: remove GPS point 3 (smallest |g_3|)
  v = [1, 1, 0, 1, 1]

Outer Iteration 2:
  Inner loop builds τ' (point 3 frozen)
  Prune: remove GPS point 1 (smallest |g_1|)
  v = [0, 1, 0, 1, 1]

Outer Iteration 3:
  Inner loop builds τ' (points 1,3 frozen)
  Prune: remove GPS point 5 (smallest |g_5|)
  v = [0, 1, 0, 1, 0]
  Attack fails! f(τ') = f(τ)

Return: τ' from Iteration 2 (last successful attack)
```

---

## Key Design Decisions

### 1. Why Iterative Instead of Single-Step?

**Problem**: Single-step FGSM often fails because:
- The linear approximation (gradient) is only locally accurate
- Large steps can overshoot optimal perturbation

**Solution**: Multiple small steps allow:
- Re-evaluation of gradient at each step
- Adaptation to the changing loss landscape
- Higher attack success rate

### 2. Why Prune by Gradient Magnitude?

**Reasoning**: If $|g_i| \approx 0$, then:
- Changing GPS point $i$ has minimal effect on the loss
- The perturbation at point $i$ is "carried along" by other points
- Removing it shouldn't break the attack

### 3. Why Rebuild τ' After Pruning?

The step $\tau' = \tau + \Delta \odot \mathbf{v}$ ensures:
- Previously pruned dimensions stay at their original values
- The trajectory is consistent with the current mask
- Fresh start for the next inner loop iteration

### 4. Why Not Just Set Perturbations to Zero?

Simply zeroing out $\delta$ at pruned points during the inner loop isn't enough because:
- The cumulative effect from previous iterations remains
- Other GPS points' gradients are computed assuming the pruned points were perturbed
- A clean rebuild from $\tau + \Delta \odot \mathbf{v}$ gives a coherent attacked trajectory

### 5. Temporal Features Are Not Perturbed

**Design Choice**: Only spatial (GPS coordinate) features are perturbed, not temporal features.

**Rationale**: Perturbing timestamps would create:
- Unrealistic back-and-forth movements
- Physically impossible travel patterns
- Easily detectable anomalies

---

## Summary Table

| Stage | Input | Operation | Output |
|-------|-------|-----------|--------|
| Init | $\tau$ | Copy trajectory, set all mask to 1 | $\tau' = \tau$, $\mathbf{v} = \mathbf{1}$ |
| Inner Loop | $\tau'$, $\mathbf{v}$ | N steps of masked FGSM | Updated $\tau'$ |
| Cumulative | $\tau$, $\tau'$ | Compute difference | $\Delta = \tau' - \tau$ |
| Prune | $\tau'$ | Gradient analysis, update mask | Sparser $\mathbf{v}$ |
| Rebuild | $\tau$, $\Delta$, $\mathbf{v}$ | Apply masked perturbation | New $\tau' = \tau + \Delta \odot \mathbf{v}$ |
| Check | $\tau'$ | Compare $f(\tau')$ vs $f(\tau)$ | Continue or return |

---

## References

- Goodfellow, I. J., et al. "Explaining and Harnessing Adversarial Examples." ICLR 2015. (Original FGSM)
- Kurakin, A., et al. "Adversarial Examples in the Physical World." ICLR 2017. (I-FGSM)
- Source Paper: "ST-iFGSM: Spatial-Temporal Iterative Fast Gradient Sign Method for Adversarial Attacks on Human Mobility Identification." KDD 2023.

---

*Document created for FAMAIL project reference. Last updated: January 2026.*
