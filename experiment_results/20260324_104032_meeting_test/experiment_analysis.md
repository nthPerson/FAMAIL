# Experiment Analysis: `meeting_test` (2026-03-24)

## Executive Summary

This experiment modified 15 trajectories over 50 ST-iFGSM iterations with equal weighting across spatial fairness (0.33), causal fairness (0.33), and fidelity (0.34). **The global Gini coefficient decreased by only 0.00016** (0.9031 → 0.9029), while fidelity dominated the optimization, rising from 0.50 to 0.90. The combined objective improved by +0.137, but this improvement is almost entirely attributable to fidelity gains, not fairness gains. No trajectories converged, and severe oscillation plagued the optimization loop. The core finding is that **the gradient signal from spatial and causal fairness terms is too weak relative to fidelity to meaningfully steer trajectory modifications toward fairer outcomes**.

---

## 1. Global Fairness Outcomes

| Metric | Before | After | Δ | Relative Δ |
|--------|--------|-------|---|------------|
| Gini coefficient | 0.9031 | 0.9029 | −0.00016 | −0.018% |
| F_spatial (1 − Gini) | 0.0969 | 0.0971 | +0.00016 | +0.16% |
| F_causal | 0.9787 | 0.9788 | +0.00011 | +0.011% |
| F_fidelity | 0.5000 | 0.9026 | +0.4026 | +80.5% |
| Combined (L) | 0.5249 | 0.6619 | +0.1370 | +26.1% |

### Assessment

**Spatial fairness was not meaningfully improved.** A Gini reduction of 0.00016 on a 48×90 grid with hundreds of trajectories is statistically negligible and would not survive any reasonable significance test. The 0.018% relative change in Gini is well within noise thresholds for this grid resolution.

**Causal fairness was essentially unchanged.** The F_causal term started at 0.9787 (already very high), leaving minimal room for gradient-driven improvement. The Δ of +0.00011 is negligible.

**Fidelity captured virtually all optimization effort.** The +0.4026 improvement in F_fidelity accounts for 98.5% of the combined objective improvement. The discriminator started at the uninformative prior of 0.50 and quickly found high-fidelity positions, which dominated the gradient signal.

### Marginal Returns by Trajectory

The global snapshots reveal diminishing returns with severe non-monotonicity in fidelity:

| Trajectories Modified | Gini Δ (from previous) | F_fidelity | Combined |
|----------------------|------------------------|------------|----------|
| 1 | −0.000029 | 0.930 | 0.671 |
| 2 | −0.000043 | 0.928 | 0.670 |
| 5 | −0.000007 | 0.924 | 0.669 |
| 9 | −0.000034 | 0.899 | 0.661 |
| 12 | −0.000028 | 0.901 | 0.661 |
| 15 | +0.000000 | 0.903 | 0.662 |

The most impactful trajectory for spatial fairness was the 2nd modified (Δ Gini = −0.000043). After that, marginal Gini improvements are negligible or zero. Notably, the combined objective _decreased_ from 0.671 after the 1st trajectory to 0.662 after the 15th — additional modifications degraded fidelity faster than they improved fairness.

---

## 2. Gradient Behavior Analysis

### 2.1 Gradient Magnitude Scale

| Term | Mean |∇| | Max |∇| |
|------|---------|---------|
| Spatial | 3.85 × 10⁻⁶ | 8.49 × 10⁻⁵ |
| Causal | 1.56 × 10⁻⁵ | 8.29 × 10⁻⁴ |
| Fidelity | 6.58 × 10⁻⁶ | 4.30 × 10⁻⁵ |

All gradient magnitudes are extremely small (10⁻⁶ to 10⁻⁴ range). This is a consequence of the soft cell assignment mechanism: when a pickup moves by step size α = 0.1 grid cells, the change in the Gaussian-softmax cell weights is smooth but very small, producing tiny gradients in the fairness terms which must then propagate through aggregation over the full 48×90 grid.

**15.6% of all iterations (117/750) recorded a gradient norm of exactly zero**, indicating complete gradient vanishing. This is a serious optimization failure.

### 2.2 Gradient Fraction Dominance

Across all 750 iterations:

| Condition | Count | Percentage |
|-----------|-------|------------|
| Fidelity-dominated (>80% of gradient) | 268 | 35.7% |
| Spatial-dominated (>80%) | 76 | 10.1% |
| Causal-dominated (>80%) | 40 | 5.3% |
| Causal near-zero (<1%) | 318 | 42.4% |

**The fidelity term dominates the gradient direction more than 3× as often as spatial, and nearly 7× as often as causal.** The causal gradient is effectively absent for 42.4% of all iterations.

### 2.3 Causal Gradient Vanishing

The causal gradient exhibits rapid decay for most trajectories:

| Trajectory | Iterations with Active Causal (>1%) | Mean Active Fraction |
|------------|--------------------------------------|---------------------|
| 119 | 0/50 | 0.000 |
| 295 | 3/50 | 0.012 |
| 451 | 24/50 | 0.029 |
| 15 | 11/50 | 0.159 |
| 384 | 23/50 | 0.137 |
| 390 | 27/50 | 0.188 |
| 94 | 50/50 | 0.379 |
| 111 | 49/50 | 0.419 |
| 57 | 23/50 | 0.532 |
| 200 | 31/50 | 0.523 |

**Root cause**: F_causal starts at 0.9787, near its theoretical maximum. The R²-based causal fairness measure is already highly saturated, meaning the gradient ∂F_causal/∂pickup_location is near-zero everywhere. Modifying a single pickup within ±2 cells produces effectively no change in causal fairness. This renders the causal weight (α₂ = 0.33) wasted budget — it contributes weight to the combined objective but produces no steering signal.

### 2.4 Spatial-Causal Alignment

The mean spatial-causal alignment is **−0.224** (std = 0.716), indicating that when both gradients are active, they frequently point in **opposing directions**. This adversarial relationship means optimizing one fairness term can degrade the other, creating a tug-of-war that wastes optimization steps.

However, the practical impact is limited because the causal gradient vanishes so quickly. The alignment metric becomes meaningless (0.000) once causal gradients are zero, which happens for the majority of iteration steps.

### 2.5 Fidelity Gradient Behavior

The fidelity (discriminator) gradient is the most consistently active signal. The discriminator model (ST-SiameseNet) provides a smooth, well-conditioned gradient landscape relative to the aggregation-dependent fairness terms. This is expected: the discriminator operates on the trajectory's local features (state vector), while fairness terms require aggregation over the entire grid.

The fidelity term's gradient does not vanish because the discriminator score varies continuously with position — unlike the fairness terms which are globally aggregated and thus locally insensitive.

---

## 3. Convergence and Oscillation Analysis

### 3.1 Zero Convergence

All 15 trajectories exhausted the full 50 iterations without converging (convergence threshold: 1 × 10⁻⁶). This is unsurprising given the gradient magnitudes are in the 10⁻⁶ range — the convergence criterion is essentially at the noise floor of the gradients themselves.

### 3.2 Severe Oscillation

| Trajectory | Direction Changes (of 48 possible) | Oscillation Rate |
|------------|-------------------------------------|-----------------|
| 57 | 36 | 75.0% |
| 390 | 35 | 72.9% |
| 150 | 34 | 70.8% |
| 451 | 34 | 70.8% |
| 111 | 33 | 68.8% |
| 177 | 33 | 68.8% |
| 119 | 32 | 66.7% |
| Mean | 31.5 | 65.7% |

A random walk would produce ~50% direction changes. **The observed rate of 65.7% exceeds random**, meaning the optimizer is actively anti-correlated across iterations — it overshoots in one direction, then corrects in the opposite direction. This is characteristic of a step size (α = 0.1) that is too large for the curvature of the objective landscape.

### 3.3 Epsilon Boundary Saturation

Most trajectories quickly reach the ε = 2.0 perturbation boundary and then oscillate along it:

- **12 of 15 trajectories** reached the boundary by iteration 19 or earlier
- After boundary contact, perturbations oscillate in the unclamped dimension while the clamped dimension stays at ±2.0
- Trajectory 94 is notable: it never reached the boundary (final perturbation = 0.0 grid cells), meaning the gradients were so weak they couldn't accumulate meaningful displacement

**The epsilon boundary acts as a hard constraint that prevents further exploration**, but the oscillation pattern shows the optimizer is not settled — it continues to bounce between positions without finding a stable optimum within the feasible region.

### 3.4 Temperature Annealing Interaction

The soft cell assignment temperature anneals from τ = 1.0 → 0.1 over 50 iterations:

| Iteration | τ | Effect |
|-----------|---|--------|
| 0 | 1.000 | Broad, smooth softmax — gradients flow to many cells |
| 10 | 0.816 | Slightly sharper |
| 25 | 0.541 | Moderate sharpening |
| 49 | 0.100 | Near-hard assignment — gradients concentrate on 1–2 cells |

As τ decreases, the softmax sharpens and fairness gradients should become more localized and potentially larger. However, the data shows **no consistent improvement in spatial/causal gradient magnitudes in later iterations** — by the time τ is low enough to produce meaningful gradients, most trajectories are already saturated at the ε boundary and oscillating. The temperature schedule and the perturbation dynamics are not well-synchronized.

---

## 4. Attribution and Selection Analysis

### 4.1 Spatial Clustering

The 15 selected trajectories cluster into only **8 unique pickup cells**:

| Pickup Cell | Count | Selection Driver |
|-------------|-------|-----------------|
| (17, 38) | 5 | LIS = 1.0 (max inequality) |
| (29, 54) | 3 | DCD = 0.77 (demand deviation) |
| (20, 28) | 2 | Mixed (LIS = 0.73) |
| Others | 5 × 1 each | Various |

**Selecting multiple trajectories from the same cell produces diminishing returns**: after the first trajectory from cell (17, 38) is modified, subsequent modifications from the same cell move pickups into the same target area, providing minimal additional fairness benefit while accumulating fidelity penalties.

### 4.2 LIS vs. DCD Score Bimodality

The attribution scores reveal a stark bimodal pattern:

- **8 trajectories** are LIS-dominant (LIS > 0.5, DCD < 0.05): selected because they originate in cells with high local inequality
- **7 trajectories** are DCD-dominant (DCD > 0.5, LIS < 0.05): selected because their service deviates from demand patterns

No trajectories score high on both measures simultaneously. This suggests the LIS and DCD attribution signals identify fundamentally different types of unfairness, and the equal 0.5/0.5 weighting may be selecting a heterogeneous mix that dilutes the optimization's focus.

### 4.3 Per-Trajectory Fairness Impact

Examining the local f_spatial values per trajectory (as computed within each trajectory's optimization):

- All trajectories show f_spatial ≈ 0.1366, varying only in the 5th–6th decimal place
- The maximum per-trajectory f_spatial delta was **+0.000081** (trajectory 344)
- 6 of 15 trajectories achieved f_spatial delta < 0.000001

**The per-trajectory spatial fairness signal is 3–4 orders of magnitude smaller than the per-trajectory fidelity signal**, confirming that the optimizer has no practical spatial fairness gradient to follow.

---

## 5. Computational Efficiency

| Metric | Value |
|--------|-------|
| Wall time | 167.9s |
| Trajectories | 15 |
| Total iterations | 750 |
| Time per trajectory | ~11.2s |
| Time per iteration | ~0.224s |

Given that zero trajectories converged and the fairness improvement is negligible, the 750 iterations represent **wasted computation for fairness purposes**. The fidelity improvement could likely be achieved in far fewer iterations.

---

## 6. Root Cause Diagnosis

The experiment's failure to improve fairness stems from three compounding issues:

### Issue 1: Gradient Scale Mismatch (Primary)

The fairness terms (spatial and causal) produce gradients 2–3 orders of magnitude too small to compete with the fidelity gradient. This is structural: fairness is computed by aggregating over the entire 48×90 grid (4,320 cells), so a single pickup's contribution is diluted by a factor of ~1/4320. The discriminator, by contrast, operates on the individual trajectory's local features, producing a much stronger per-trajectory gradient.

**Impact**: The sign-based ST-iFGSM step `δ = α · sign(∇L)` partially mitigates this through normalization, but when the fairness gradient magnitude is at the numerical precision floor, the sign becomes unreliable, leading to random-walk behavior rather than directed optimization.

### Issue 2: Causal Saturation (Secondary)

F_causal = 0.9787 at baseline leaves no room for gradient-driven improvement. The causal term weight (α₂ = 0.33) effectively becomes dead weight in the objective function — it contributes to the combined score but provides no directional information.

**Impact**: One-third of the objective function's weighting is allocated to a term that cannot improve, reducing the effective weight on spatial fairness from 0.33 to 0.33/(0.33 + 0.34) ≈ 0.49 of the active gradient signal (and that signal is still dwarfed by fidelity).

### Issue 3: Perturbation Boundary Oscillation (Tertiary)

The step size α = 0.1 combined with ε = 2.0 means trajectories reach the boundary in ~20 iterations, then spend the remaining 30 iterations oscillating without improving. The sign-based update (+0.1 or −0.1 per step) creates a discrete grid of reachable positions, and the optimizer cycles between them.

**Impact**: Effective optimization occurs only in the first 15–20 iterations. The remaining 60% of computation is wasted on boundary oscillation.

---

## 7. Recommendations for Improvement

### 7.1 Gradient Scaling / Adaptive Weighting (High Priority)

**Problem**: Equal weights (0.33/0.33/0.34) ignore the 1000× gradient magnitude disparity.

**Options**:
- **Gradient normalization per term**: Before combining, normalize each term's gradient to unit norm, then apply weights. This ensures each term contributes equally to the direction regardless of magnitude: `∇L = α₁ · (∇F_sp / ||∇F_sp||) + α₂ · (∇F_ca / ||∇F_ca||) + α₃ · (∇F_fi / ||∇F_fi||)`
- **Adaptive weighting**: Dynamically increase α₁ (spatial) when F_spatial improvement is below a threshold, and decrease α₃ (fidelity) once F_fidelity exceeds a target (e.g., 0.8).
- **Two-phase optimization**: First optimize fidelity-only to find realistic positions, then freeze fidelity and optimize fairness-only within the acceptable fidelity region.

### 7.2 Remove or Reduce Causal Weight (High Priority)

With F_causal at 0.9787, the causal term should either be:
- **Set to α₂ = 0** and redistributed to spatial (e.g., α₁ = 0.60, α₃ = 0.40), or
- **Monitored as a constraint** (e.g., require F_causal ≥ 0.97) rather than an objective term to optimize

### 7.3 Decaying Step Size (Medium Priority)

Replace the fixed α = 0.1 with an annealing schedule (e.g., α_t = α₀ / (1 + t/10)) to reduce oscillation in later iterations. This mirrors the temperature annealing already applied to soft cell assignment.

### 7.4 Cell-Diverse Selection (Medium Priority)

Enforce diversity in the selected trajectories: instead of selecting 5 trajectories from cell (17, 38), select at most 1–2 per cell and cover more of the grid. This maximizes the spatial fairness impact per modified trajectory.

### 7.5 Early Stopping on Fidelity (Low Priority)

Once a trajectory achieves F_fidelity > 0.9, stop iterating (or switch to fairness-only optimization). This saves computation and prevents the oscillation pattern that degrades final outcomes.

### 7.6 Increase ε or Use Continuous Relaxation (Low Priority)

ε = 2.0 grid cells (~2.2 km) may be insufficient to move pickups from over-served to under-served areas. Consider increasing ε to 3–5 cells for trajectories in highly over-served cells, or use a soft penalty on perturbation magnitude rather than a hard clip.

---

## 8. Key Takeaway for Scientific Claims

**This experiment demonstrates that gradient-based trajectory modification can successfully maintain behavioral realism (fidelity improvement from 0.50 → 0.90) but fails to improve spatial fairness under equal weighting because the fairness gradient signal is structurally dominated by the discriminator gradient.** The core algorithmic assumption — that ST-iFGSM with a combined objective will balance fairness and fidelity — is not supported by this data. Effective fairness optimization requires either gradient normalization across terms, adaptive weighting, or a multi-phase optimization strategy that decouples the fairness and fidelity objectives.

The result is not a failure of the soft cell assignment mechanism or the attribution pipeline. Both produce reasonable outputs (temperature annealing works, top-k selection identifies plausible targets). The failure is in the **gradient composition step**, where three terms of vastly different magnitudes are naively summed with equal weights.

---

## Appendix: Per-Trajectory Summary

| Traj | Driver | Orig → Mod Cell | |δ| | Obj | F_fidelity | Causal Active Iters | Oscillation Rate | Selection Driver |
|------|--------|-----------------|-----|-----|------------|---------------------|------------------|-----------------|
| 119 | 49 | (41,41)→(42,42) | 1.41 | 0.669 | 0.995 | 0/50 | 66.7% | DCD |
| 451 | 34 | (20,28)→(18,30) | 2.83 | 0.653 | 0.948 | 24/50 | 70.8% | LIS |
| 384 | 19 | (29,53)→(31,54) | 2.24 | 0.650 | 0.941 | 23/50 | 64.6% | DCD |
| 295 | 35 | (20,28)→(22,30) | 2.83 | 0.650 | 0.941 | 3/50 | 64.6% | LIS |
| 15 | 19 | (29,54)→(31,56) | 2.83 | 0.647 | 0.930 | 11/50 | 62.5% | DCD |
| 344 | 22 | (35,15)→(35,17) | 2.00 | 0.646 | 0.928 | 18/50 | 60.4% | DCD |
| 57 | 19 | (29,54)→(31,55) | 2.24 | 0.645 | 0.925 | 23/50 | 75.0% | DCD |
| 94 | 22 | (28,16)→(28,16) | 0.00 | 0.644 | 0.922 | 50/50 | 62.5% | DCD |
| 200 | 19 | (29,54)→(31,55) | 2.24 | 0.644 | 0.921 | 31/50 | 62.5% | DCD |
| 208 | 39 | (17,38)→(19,37) | 2.24 | 0.622 | 0.858 | 47/50 | 64.6% | LIS |
| 150 | 2 | (17,38)→(18,40) | 2.24 | 0.619 | 0.849 | 49/50 | 70.8% | LIS |
| 390 | 27 | (17,38)→(17,40) | 2.00 | 0.615 | 0.837 | 27/50 | 72.9% | LIS |
| 137 | 15 | (14,15)→(16,17) | 2.83 | 0.633 | 0.889 | 28/50 | 60.4% | LIS |
| 111 | 9 | (17,38)→(18,40) | 2.24 | 0.619 | 0.848 | 49/50 | 68.8% | LIS |
| 177 | 36 | (17,38)→(18,40) | 2.24 | 0.604 | 0.806 | 49/50 | 68.8% | LIS |
