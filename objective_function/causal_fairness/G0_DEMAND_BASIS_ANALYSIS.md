# g₀(D) Basis Function Analysis: Polynomial vs. Isotonic for Hat Matrix F_causal

**Date**: 2026-02-14
**Context**: Determining whether the hat matrix F_causal formulation (Option B / Option C) can use polynomial demand features without sacrificing the quality of the demand→ratio fit.
**Prerequisite Reading**: `FCAUSAL_GDX_JOINT_ANALYSIS.md` (Sections 5.3.4, 7.3)

---

## 1. The Problem

The hat matrix formulation of F_causal (Revised Rank 1 in the joint analysis) achieves an elegant closed-form computation:

$$F_{\text{causal}} = \frac{Y'(I-H)Y}{Y'MY}$$

where H is a constant projection matrix. This formulation requires that the entire model (demand features + demographic features) be **linear in parameters** — i.e., expressible as Y = Xβ + ε where X is a fixed design matrix.

**Polynomial regression** (Y = β₀ + β₁D + β₂D² + ...) is linear in parameters. ✓
**Isotonic regression** is NOT linear in parameters. ✗

However, the empirical comparison from the Causal Fairness Dashboard shows:

| Method | R² |
|--------|-----|
| Isotonic | 0.4453 |
| Binning (10 bins) | 0.4439 |
| Polynomial (degree 2) | 0.1949 |
| Linear | 0.1064 |
| LOWESS | 0.0000 |

**The polynomial (degree 2) R² is less than half of isotonic's R².** This is a significant gap that must be understood and addressed before committing to the hat matrix approach.

---

## 2. Why Polynomial Degree 2 Fails

### 2.1 The Shape of the Demand→Ratio Relationship

The scatter plot from the dashboard reveals the characteristic shape of Y = S/D:

```
Y (Service Ratio)
  |
40|  ·
  |  ··
  |  ···
20|  · ···
  |  ·  ····
10|  ···  ·····
  |  ·····  ·······
  |  ··········  ··········
  |  ···························  ···············
 0+----+----+----+----+----+----+----+----→ D
  0   20   40   60   80  100  120  140
```

Key characteristics:
1. **Steep decay at low D**: Y drops rapidly from ~40 to ~5 as D goes from 1 to 10
2. **Long flat tail**: Y is approximately constant (~0.5–1.0) for D > 30
3. **Extreme positive skew in D**: Most cells have D < 20; few have D > 60
4. **Heavy-tailed Y at low D**: Large variance in Y when D is small

This is a **hyperbolic** relationship: Y ≈ a/D + b. More precisely, since Y = S/D and supply S tends to grow sub-linearly with demand D, the curve decays like a power law.

### 2.2 Why Polynomial(D, degree=2) Can't Capture This

A degree-2 polynomial Y = β₀ + β₁D + β₂D² has three problems with this data:

1. **Wrong asymptotic behavior**: As D → ∞, a polynomial → ±∞, but the true relationship → 0⁺. The orange curve in the scatter plot visibly goes negative for D > 120, which is physically impossible (Y = S/D ≥ 0).

2. **One inflection point only**: A parabola has one turning point. The data has a sharp decline followed by a flat tail — requiring a curve that bends from steep-decline into near-horizontal. Degree 2 can't match both the steep part and the flat part.

3. **Dominated by low-D points**: With most data concentrated at D < 20 (where Y is large and variable), the polynomial tries to fit the steep decline but produces wild behavior in the extrapolation region.

### 2.3 Why Isotonic and Binning Succeed

Both isotonic regression and binning are **non-parametric** methods:

- **Isotonic** fits a piecewise-constant decreasing function with as many breakpoints as needed. It perfectly adapts to the steep decline at low D and the flat tail at high D.
- **Binning** (10 quantile bins) computes the mean Y within each demand decile. This is essentially a step function that also adapts to the data's shape.

Their R² values (0.445 and 0.444) are nearly identical — both capture the same underlying monotone decreasing pattern. The small gap is because isotonic is a finer-grained step function than 10 bins.

---

## 3. Can We Do Better With Transformed Polynomial Bases?

The hat matrix requires a model that is **linear in parameters** — but the features themselves don't need to be raw polynomial powers of D. Any transformation of D that is computed once and fixed can serve as a basis function in the design matrix.

### 3.1 Candidate Basis Functions

The hyperbolic shape Y ≈ a/D + b suggests several transformations:

| Basis Set | Design Matrix Columns | Parameters | Shape Captured |
|-----------|----------------------|------------|----------------|
| **Poly(D, 2)** | [D, D²] | 3 | Parabolic (poor) |
| **Poly(D, 5)** | [D, D², D³, D⁴, D⁵] | 6 | Better curve, but Runge oscillations at edges |
| **Reciprocal** | [1/D] | 2 | Hyperbolic decay (excellent for this data) |
| **Reciprocal + poly** | [1/D, D] | 3 | Hyperbolic + linear correction |
| **Log** | [log(D)] | 2 | Logarithmic decay (moderate fit) |
| **Log + reciprocal** | [log(D), 1/D] | 3 | Flexible nonlinear decay |
| **Power basis** | [D^(-1), D^(-0.5), D^(0.5)] | 4 | Flexible power-law family |
| **Rational** | [1/(D+1), D/(D+1)] | 3 | Bounded rational function |

All of these are **linear in parameters** — they have the form Y = Xβ where X contains pre-computed transformations of D. The hat matrix H = X(X'X)⁻¹X' can be constructed for any of them.

### 3.2 Why Reciprocal Basis Should Work Well

The theoretical relationship is Y = S/D. If supply S scales as S ∝ D^α for some 0 < α < 1 (sub-linear), then:

$$Y = \frac{S}{D} \propto \frac{D^\alpha}{D} = D^{\alpha - 1}$$

For α = 0 (constant supply regardless of demand): Y ∝ 1/D.
For α = 0.5: Y ∝ D^(-0.5).
For α = 1 (linear scaling): Y = constant.

The actual data likely has α somewhere between 0 and 1, making a **power-law basis** [D^(-1), D^(-0.5), 1, D^(0.5)] a natural choice. The simplest effective version is likely:

$$g_0(D) = \beta_0 + \beta_1 \cdot \frac{1}{D+1}$$

This is a 2-parameter model (same as linear regression) that captures the right asymptotic behavior: Y → β₀ as D → ∞, and Y → β₀ + β₁ as D → 0. The +1 in the denominator avoids division by zero.

### 3.3 Expected R² With Transformed Bases

Without running the actual data (which requires the dashboard), we can reason about expected performance:

| Basis | Expected R² | Reasoning |
|-------|-------------|-----------|
| 1/D alone | ~0.35–0.45 | Captures the dominant hyperbolic shape |
| 1/D + 1/D² | ~0.40–0.45 | Adds curvature control at low D |
| 1/D + D | ~0.40–0.45 | Allows slight linear trend in tail |
| 1/D + log(D) | ~0.40–0.45 | log captures intermediate range |
| Poly(D, 5+) | ~0.30–0.40 | Higher degree helps but oscillates |
| Poly(D, 10+) | ~0.40+ | Many parameters, overfitting risk |

**Key prediction**: A simple reciprocal basis [1, 1/(D+1)] should approach isotonic's R² of 0.445 with only 2 parameters, because the underlying relationship is genuinely hyperbolic. The polynomial basis [1, D, D²] fails not because it has too few parameters, but because it uses the **wrong functional family**.

---

## 4. Implications for the Hat Matrix F_causal Formulation

### 4.1 The Core Question Restated

Can we construct a design matrix X such that:
1. The demand features in X capture the D→Y relationship as well as isotonic regression (R² ≈ 0.44)
2. X also includes demographic features
3. X is a fixed matrix (linear-in-parameters), enabling the hat matrix formulation

**Answer: Almost certainly yes**, by using transformed demand bases instead of raw polynomial powers.

### 4.2 Proposed Design Matrix for Option B (Hat Matrix)

For F_causal via the hat matrix, the design matrix for the **demographic regression on residuals** would be:

$$X_{\text{demo}} = [\text{intercept}, x_1, x_2, \ldots, x_p]$$

The hat matrix $H = X_{\text{demo}}(X_{\text{demo}}'X_{\text{demo}})^{-1}X_{\text{demo}}'$ projects R onto the demographic column space. Here, R = Y - g₀(D), and g₀(D) is computed separately (possibly isotonic — not constrained by the hat matrix).

**This is the key realization: Option B's hat matrix operates on the demographic regression of residuals, NOT on the demand model.** The demand model g₀(D) can be ANY method (including isotonic), as long as the demographic regression R ~ x is linear-in-parameters.

Since demographics x are already continuous features that are standardized, the demographic regression is naturally linear. No transformation needed. **Option B's hat matrix formulation does NOT require polynomial demand features.**

### 4.3 Proposed Design Matrix for Option C (Hat Matrix, Full Model)

Option C compares Y ~ D vs Y ~ D + x. Both models must be linear-in-parameters. Here, the demand basis matters:

$$X_{\text{red}} = [\text{intercept}, \phi_1(D), \phi_2(D), \ldots]$$
$$X_{\text{full}} = [\text{intercept}, \phi_1(D), \phi_2(D), \ldots, x_1, x_2, \ldots, x_p]$$

For Option C, we DO need a linear-in-parameters demand model that captures the D→Y relationship well. The transformed basis approach (reciprocal, log, power) solves this.

### 4.4 Summary of Constraints by Formulation

| Formulation | Demand model constraint | Recommended approach |
|-------------|------------------------|---------------------|
| **Option B** (residual-demographic R²) | g₀(D) can be ANY method | Use isotonic for g₀(D); hat matrix only for R ~ x |
| **Option C** (ΔR²) | Both models must be linear-in-parameters | Use transformed basis [1/(D+1), log(D+1), ...] |
| **Frozen β variant** | g₀(D) can be ANY method | Use isotonic for g₀(D); pre-compute β from OLS on R ~ x |

---

## 5. Revised Recommendation: Option B Does NOT Require Polynomial g₀

### 5.1 Corrected Architecture for Option B

The previous discussion in the joint analysis assumed that using the hat matrix for F_causal required the entire model (demand + demographics) to be in a single design matrix. **This is not the case for Option B.**

Option B's two-stage structure is:

**Stage 1** (pre-computed, any method): Fit g₀(D) to get residuals R = Y - g₀(D)
**Stage 2** (hat matrix): Compute R²(R ~ x) using the demographic hat matrix

Stage 1 can use **isotonic regression** (R² = 0.445) — the best-performing method. Only Stage 2 needs the hat matrix, and the demographic regression is naturally linear-in-parameters.

The F_causal computation during optimization becomes:

```python
# Pre-compute ONCE:
g0_predictions = isotonic_model.predict(D_original)  # frozen g₀(D) lookup table
H_demo = X_demo @ np.linalg.inv(X_demo.T @ X_demo) @ X_demo.T  # demographic hat matrix
M = np.eye(N) - np.ones((N,N)) / N  # centering matrix

# During optimization (differentiable):
Y = S / (D + eps)              # service ratios (differentiable)
R = Y - g0_lookup[cell_idx]    # residuals (differentiable through Y; g₀ lookup is frozen)
SS_res = R @ (np.eye(N) - H_demo) @ R  # residual SS after demographics
SS_tot = R @ M @ R                      # total SS of residuals
F_causal = SS_res / SS_tot              # 1 - R²_demo
```

**g₀(D) enters only as a frozen lookup table** — it doesn't need to be a hat matrix projection. The demand model and the demographic regression are decoupled.

### 5.2 Corrected Architecture for Option C

Option C does require the full model comparison, so the demand basis matters. The recommended approach:

```python
# Pre-compute ONCE:
# Transformed demand features (linear in parameters, captures hyperbolic shape)
phi_D = np.column_stack([
    np.ones(N),                  # intercept
    1.0 / (D + 1),              # reciprocal (captures Y ∝ 1/D)
    np.log(D + 1),              # log (captures intermediate curvature)
])
X_red = phi_D                                      # demand-only model
X_full = np.hstack([phi_D, standardized_demographics])  # full model

H_red = X_red @ np.linalg.inv(X_red.T @ X_red) @ X_red.T
H_full = X_full @ np.linalg.inv(X_full.T @ X_full) @ X_full.T
M = np.eye(N) - np.ones((N,N)) / N

# During optimization (differentiable):
Y = S / (D + eps)
R2_red = 1 - (Y @ (np.eye(N) - H_red) @ Y) / (Y @ M @ Y)
R2_full = 1 - (Y @ (np.eye(N) - H_full) @ Y) / (Y @ M @ Y)
delta_R2 = R2_full - R2_red
F_causal = 1 - delta_R2
```

However, there's a subtlety: **D changes during optimization** (pickups move between cells), so the design matrices X_red and X_full would need to be recomputed — they contain φ(D), which depends on the current demand. This breaks the "constant matrix" property of the hat matrix approach.

### 5.3 The D-Changes Problem (Affects Option C, Not Option B)

When the ST-iFGSM optimizer moves a pickup from cell A to cell B:
- D_A decreases, D_B increases
- Y_A = S_A/D_A changes, Y_B = S_B/D_B changes
- For Option C: φ(D_A) and φ(D_B) also change, so X_red and X_full change, so H_red and H_full change — they're no longer constant

For Option B, this is less of a problem:
- g₀(D) is a frozen lookup table — but which lookup entry to use depends on D, which changes
- In practice, soft cell assignment distributes a pickup across neighboring cells with continuous weights, and the demand per cell changes smoothly
- The g₀ lookup is frozen in the sense that the function g₀ doesn't change, but g₀(D_c) for each cell does change as D_c changes
- This is identical to how the current production code works ([objective.py:380-382](../../trajectory_modification/objective.py#L380-L382))
- The demographic hat matrix H_demo depends only on x (demographics), which is truly fixed

**Conclusion**: Option B's hat matrix formulation works cleanly even when D changes, because the hat matrix H_demo depends only on demographics (fixed). Option C's hat matrices depend on φ(D), which changes during optimization, breaking the constant-matrix assumption.

### 5.4 Updated Ranking

This analysis reinforces **Option B as the clear winner**:

| Formulation | g₀(D) method | Hat matrix for | D-changes issue? | Effective? |
|-------------|-------------|----------------|-----------------|------------|
| **Option B** | Isotonic (R²=0.445) | Demographics only | No — H_demo is constant | **Yes** ✓ |
| **Option C** | Transformed polynomial | Demand + Demographics | Yes — H_red, H_full change with D | Problematic |

---

## 6. Isotonic g₀(D): Properties and Suitability

Given that Option B allows isotonic g₀(D), let's confirm it's the right choice.

### 6.1 Why Isotonic is Appropriate for g₀(D)

1. **Monotonicity assumption is correct**: Higher demand should (on average) lead to lower Y = S/D, because supply grows sub-linearly with demand. The data confirms this (scatter plot shows clear decreasing trend).

2. **Non-parametric flexibility**: Isotonic regression adapts to the data's shape without assuming a functional form. The steep decline at low D and flat tail at high D are captured automatically.

3. **Best empirical R²**: At 0.4453, isotonic captures the most demand→ratio variance of all tested methods.

4. **Frozen lookup is efficient**: IsotonicRegression from sklearn produces a monotone piecewise-linear interpolator. Prediction is O(log n) per query (binary search on breakpoints).

5. **Binning (R²=0.4439) is essentially equivalent**: The near-identical R² of 10-bin quantile binning suggests the relationship is well-described by a monotone step function with ~10 levels. Isotonic is a finer version of this.

### 6.2 Isotonic's Limitation for Gradient Flow

Isotonic regression is not differentiable in the traditional sense (it's piecewise constant with jumps at breakpoints). However, for FAMAIL's purposes this doesn't matter:

- **g₀ is frozen**: No gradients flow through g₀. It's a lookup table.
- **Gradients flow through Y**: R_c = Y_c - g₀(D_c). The gradient ∂R_c/∂Y_c = 1 regardless of g₀'s form.
- **When D changes**: g₀(D_c) changes discretely (jumping between isotonic levels), but with soft cell assignment distributing pickups across neighbors, the effective demand changes smoothly and the lookup transitions are smoothed out.

### 6.3 What About Higher-Degree Polynomials?

For completeness, here's why simply increasing the polynomial degree is not the answer:

| Degree | Expected Behavior |
|--------|------------------|
| 2 | R² ≈ 0.19. Parabola, goes negative at high D. |
| 3 | R² ≈ 0.25–0.30. Can capture the asymmetry but still oscillates. |
| 5 | R² ≈ 0.30–0.38. Better fit in data range, but Runge's phenomenon at edges. |
| 10 | R² ≈ 0.40+. Approaching isotonic, but severe overfitting and numerical instability with Vandermonde matrix. |
| 20+ | Condition number of X'X becomes huge. Numerically unstable. |

The fundamental issue is that **polynomials are the wrong basis for a 1/D-shaped function**. No amount of increasing degree fixes the structural mismatch efficiently. In contrast, a 2-parameter reciprocal model [1, 1/(D+1)] should achieve R² ≈ 0.40+ immediately.

---

## 7. Practical Recommendations

### 7.1 For Option B (Recommended F_causal)

**Use isotonic g₀(D)**. It achieves the best demand→ratio fit (R² = 0.445), is simple, and is already implemented. The hat matrix formulation applies only to the demographic regression, where demographics are naturally linear features.

No changes to the existing `estimate_g_function` infrastructure are needed for g₀(D).

### 7.2 For Option C (If Pursued as Validation)

If Option C is used as a validation metric alongside Option B, use a **transformed-basis polynomial** for the demand features to maintain the hat matrix structure:

```python
phi_D = np.column_stack([
    np.ones(N),           # intercept
    1.0 / (D + 1),       # reciprocal
    np.log(D + 1),       # log
    np.sqrt(D),          # square root (optional)
])
```

This should achieve R² close to isotonic's 0.445 while remaining linear-in-parameters.

However, note the D-changes problem from Section 5.3: Option C's hat matrices change during optimization because D changes. This can be addressed by:
1. **Freezing D at original values** for the hat matrix computation (treating it like g₀'s frozen lookup)
2. **Re-computing H_red and H_full each step** (expensive but correct)
3. **Using Option C only as a pre/post metric**, not during optimization

Approach (3) is the most practical: compute ΔR² before and after trajectory modification to validate that F_causal (Option B) improved the demographic fairness, without using ΔR² as the optimization target.

### 7.3 For the Demographic Explorer Dashboard

The current dashboard's "g(D,x) Estimator" page uses `PolynomialFeatures(degree=poly_degree)` for demand features in the g(D,x) models. To improve these exploratory models:

1. **Add transformed-basis options** to `build_feature_matrix`: reciprocal, log, power-law bases
2. **Compare LODO ΔR² across basis choices**: this validates whether demographics add signal beyond demand, which matters for the F_causal formulation decision
3. **Don't conflate g(D,x) model quality with g₀(D) choice**: the exploratory g(D,x) models help validate the premise (demographics matter), but g₀(D) for production is separate

---

## 8. Summary

| Question | Answer |
|----------|--------|
| Is polynomial g₀(D) sufficient? | **No** — degree 2 gets R² = 0.19 vs isotonic's 0.45. The relationship is hyperbolic, not polynomial. |
| Does this block the hat matrix F_causal? | **No** — Option B only needs the hat matrix for demographics, not demand. g₀(D) can be isotonic. |
| What about Option C? | Option C needs linear-in-parameters demand. Use reciprocal/log basis, but D-changes during optimization are problematic. |
| Best g₀(D) method? | **Isotonic** (R² = 0.445, non-parametric, simple, already implemented). |
| Could transformed polynomials match isotonic? | Likely yes — [1, 1/(D+1), log(D+1)] should approach R² ≈ 0.44 with 3 parameters. Worth validating empirically. |
| What's the recommended architecture? | **Option B with isotonic g₀(D) + demographic hat matrix H_demo.** |

### Architecture Diagram

```
                    PRE-COMPUTED (before optimization)
                    ═══════════════════════════════════

                    g₀(D): Isotonic regression
                           D → g₀(D) lookup table

                    H_demo: X(X'X)⁻¹X' where X = [1, x₁, ..., xₚ]
                            (demographics only, ~500×500 matrix)

                    M:      I - 11'/N (centering matrix)


                    DURING OPTIMIZATION (differentiable)
                    ═══════════════════════════════════

    Trajectory → Soft Cell Assignment → S_c, D_c per cell
    Positions                              ↓
                                     Y_c = S_c / D_c    (differentiable)
                                           ↓
                                     R_c = Y_c - g₀(D_c) (g₀ is frozen lookup)
                                           ↓
                              ┌─────────────┴─────────────┐
                              │                           │
                        SS_res = R'(I-H)R          SS_tot = R'MR
                              │                           │
                              └─────────────┬─────────────┘
                                            ↓
                                  F_causal = SS_res / SS_tot
                                            ↓
                                     ∂F/∂R → ∂R/∂Y → ∂Y/∂S → ∂S/∂positions
```
