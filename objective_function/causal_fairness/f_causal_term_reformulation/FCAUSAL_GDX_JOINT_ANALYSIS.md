# Joint Analysis: F_causal Formulation, g(D,x) Model Selection, and Evaluation Metrics

**Date**: 2026-02-14 (revised 2026-02-14)
**Context**: FAMAIL Causal Fairness Term Reformulation
**Prerequisite Reading**: `G_FUNCTION_REFORMULATION_PLAN.md`

> **Revision Note (2026-02-14)**: Sections 5.3.1–5.3.3 and the rankings in Section 7
> contained an error in the differentiability analysis of Option B. The error was
> confusing the variance-ratio identity R² = Var(ŷ)/Var(y) (which only holds when ŷ
> is the OLS fit of the *current* y) with the general sum-of-squares R² formula
> (which has both numerator and denominator depending on R). The original analysis
> incorrectly concluded that Option B had a "frozen numerator" problem requiring
> workaround formulations. In fact, the direct R²(R ~ x) formulation is fully
> differentiable via two approaches (frozen β with SS form, and hat matrix). See
> **Section 5.3.4** for the corrected analysis, and **Section 7.3** for revised
> rankings.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [The Circular Dependency](#2-the-circular-dependency)
3. [System Architecture Constraints](#3-system-architecture-constraints)
4. [Analysis Framework](#4-analysis-framework)
5. [Formulation-by-Formulation Analysis](#5-formulation-by-formulation-analysis)
6. [Evaluation Metric Analysis](#6-evaluation-metric-analysis)
7. [Ranked Recommendations](#7-ranked-recommendations)
8. [Implementation Path](#8-implementation-path)
9. [Open Questions and Risks](#9-open-questions-and-risks)

---

## 1. Problem Statement

The causal fairness term F_causal in the FAMAIL objective function serves a dual purpose:

1. **Measurement**: Quantify how much demographic context (income, housing prices, etc.) influences the taxi service ratio beyond what demand alone would predict.
2. **Optimization signal**: Provide a differentiable gradient that guides the ST-iFGSM trajectory modifier to reduce demographic-driven service inequality.

Three interdependent design choices must be made simultaneously:

| Decision | Question |
|----------|----------|
| **F_causal formulation** | How do we compute the causal fairness score from Y, D, and x? |
| **g(D,x) model architecture** | What functional form should g take, and what features should it use? |
| **Model selection metric** | How do we evaluate candidate g(D,x) models to pick the best one? |

The difficulty is that these three decisions are not independent — each constrains the others.

---

## 2. The Circular Dependency

### 2.1 The "Chicken or Egg" Problem

The dependency structure is:

```
F_causal formulation
    ↕ determines what g must compute
g(D,x) model role
    ↕ determines what "good" means for g
Model selection metric
    ↕ determines which g gets selected
Selected g(D,x) model
    ↕ determines F_causal's behavior
F_causal behavior during optimization
```

Concretely:
- **If F_causal = 1 - Var[g(D,x) - g(D,x̄)] / Var[Y]** (Option A1), then g must be an accurate conditional model, and the selection metric should favor prediction accuracy (R²).
- **If F_causal = 1 - R²(R ~ x)** (Option B), then g is merely g₀(D) — a demand-only baseline — and g(D,x) isn't even needed during optimization. The selection metric is irrelevant to g(D,x) because the metric operates on g₀(D) residuals.
- **If F_causal = 1 - ΔR²** (Option C), then g(D,x) and g₀(D) are both needed as a comparison pair, and the metric should capture the incremental demographic contribution.

### 2.2 Breaking the Circularity

The key insight that breaks this circularity is recognizing that **F_causal's formulation is the upstream decision**. Here's why:

1. F_causal defines what the optimizer will try to maximize. This is a normative choice about what "fair" means — it cannot be determined empirically.
2. Once F_causal is fixed, the role of g (or g₀, or both) becomes clear — it's determined by the mathematical structure of F_causal.
3. Once g's role is clear, the right evaluation metric follows — it should measure how well g fulfills that specific role.

**Decision order: F_causal first → g's role follows → metric follows.**

The reason it felt like a chicken-or-egg problem is that during the *exploration phase* (the demographic explorer dashboard), we were using LODO R² to evaluate g(D,x) models *before* deciding on F_causal. This is actually fine for exploration — LODO R² answers "can demographics predict service ratios?" — but it should not be the final model selection criterion until we know what g is being selected *for*.

---

## 3. System Architecture Constraints

Before analyzing formulations, we must establish the non-negotiable constraints of the FAMAIL optimization pipeline.

### 3.1 Gradient Flow Architecture

The ST-iFGSM optimizer computes:

```
δ = clip(α · sign[∇_pickup L], -ε, ε)
```

where L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity.

The gradient ∇F_causal must flow through:

```
F_causal → Y = S/D → S (supply counts per cell) → soft cell assignment → pickup locations
```

Critical constraints:
- **g is frozen** during optimization (pre-computed, no gradient)
- **x (demographics) are fixed** per cell — they never change during optimization
- **D (demand) changes** when pickups move between cells via soft cell assignment
- **S (supply) changes** correspondingly
- Only S and D are differentiable; g and x are constants

### 3.2 What F_causal Can Differentiate Through

Any F_causal formulation must ultimately depend on Y = S/D (or equivalently S and D separately), because that's the only path through which gradients reach the trajectory parameters.

This means:
- F_causal **can** use Y, S, D as differentiable quantities
- F_causal **cannot** differentiate through g (frozen lookup), x (fixed), or any model fitted on the current data

### 3.3 Data Granularity Reality

- 10 Shenzhen districts, ~100-400 cells each
- Demographics are constant within a district (district-level data)
- Effective demographic degrees of freedom: 10 (one per district)
- With ~300-500 active cells (demand > threshold), about 30-50 cells per district on average
- LODO cross-validation has 6-10 folds (one per district with sufficient data)

---

## 4. Analysis Framework

For each F_causal formulation option, we evaluate:

| Criterion | Description |
|-----------|-------------|
| **Causal validity** | Does it actually measure demographic influence on service, or something else? |
| **Optimization signal quality** | Does the gradient point the optimizer in the right direction? |
| **g(D,x) role clarity** | Is g's purpose unambiguous? Does it determine a clear model selection criterion? |
| **Robustness to data limitations** | Does it work with 10 districts and district-level demographics? |
| **Differentiability** | Can gradients flow correctly through the computation? |
| **Interpretability** | Can we explain what the score means to stakeholders? |

---

## 5. Formulation-by-Formulation Analysis

### 5.1 Option A1: Demographic Attribution

**Formula**:
$$F_{\text{causal}} = 1 - \frac{\text{Var}[g(D_c, x_c) - g(D_c, \bar{x})]}{{\text{Var}[Y_c]}}$$

**What it measures**: The fraction of predicted service ratio variation attributable to demographics rather than demand.

**Role of g(D,x)**: Must be an accurate conditional predictor of Y from both D and x. The better g fits the data, the more meaningful the attribution.

**Implied model selection metric**: LODO R² is *partially* appropriate — g must predict well — but the metric should also ensure that g's demographic coefficients are stable across folds (since the attribution depends on g(D,x) - g(D,x̄), which requires stable demographic coefficients).

**Analysis**:

| Criterion | Assessment |
|-----------|------------|
| Causal validity | **Moderate**. Measures the model's learned sensitivity to demographics, not the true causal effect. If g is misspecified, the attribution is wrong. If g overfits to demographic patterns, it inflates the unfairness estimate. |
| Optimization signal | **Problematic**. During optimization, g is frozen. F_causal would be: 1 - Var[g(D,x) - g(D,x̄)] / Var[Y]. The numerator is constant (g is frozen, x is fixed). The denominator Var[Y] changes as trajectories are modified. So the optimizer would maximize F_causal by *increasing Var[Y]* — making service ratios more spread out — which is the opposite of fairness. The gradient signal is perverse. |
| g(D,x) role | Clear: accurate conditional predictor |
| Robustness | **Weak**. With only 10 district profiles, g(D,x) may overfit demographics, making the attribution unreliable. |
| Differentiability | **Partial**. The numerator is constant (frozen g, fixed x). Only Var[Y] in the denominator carries gradient. This is a degenerate gradient signal. |
| Interpretability | Good in principle ("X% of predicted service variation comes from demographics"), but misleading if g is misspecified. |

**Verdict**: **Not recommended for production F_causal**. The frozen-g constraint makes the optimization signal perverse. Useful as an exploratory diagnostic only.

---

### 5.2 Option A2: Conditional R² (g(D,x) as the full model)

**Formula**:
$$F_{\text{causal}} = R^2 = 1 - \frac{\text{Var}[Y - g(D, x)]}{\text{Var}[Y]}$$

**What it measures**: How well g(D,x) predicts the observed service ratio. Higher R² means the model explains more variance.

**Role of g(D,x)**: Direct predictor of Y. The better g fits, the higher F_causal.

**Implied model selection metric**: LODO R² is directly appropriate — maximize prediction accuracy.

**Analysis**:

| Criterion | Assessment |
|-----------|------------|
| Causal validity | **Very weak**. This is the same problem as the original g(D) formulation, just with more features. A high R² means the model (including demographics) explains observed patterns — but those patterns *include* the unfairness. This doesn't measure demographic bias; it measures model fit quality. |
| Optimization signal | **Problematic (same as original)**. Maximizing R² = 1 - Var(Y - g(D,x))/Var(Y) with frozen g means minimizing Var(residuals)/Var(Y). The optimizer pushes Y toward g's predictions — but g was trained on unfair data, so the optimizer would push toward the *existing unfair pattern*. This perpetuates the circularity identified in the reformulation meeting. |
| g(D,x) role | Clear: predict Y as accurately as possible |
| Robustness | Moderate — same as any regression with 10 district profiles |
| Differentiability | Yes — same gradient structure as current g(D) formulation |
| Interpretability | Misleading: "98% of service variation is explained" sounds good but doesn't address fairness |

**Verdict**: **Not recommended**. This is the current formulation with more features — it suffers from the same circular fairness baseline problem identified in the reformulation meeting. Including demographics in g doesn't fix the fundamental issue; it potentially makes it worse by learning *how* demographics create bias and then optimizing to maintain that pattern.

---

### 5.3 Option B: Demographic Disparity Score (Residual-Demographic Regression)

**Formula**:
$$F_{\text{causal}} = 1 - \rho^2(R, x) = 1 - R^2_{\text{demo}}$$

where $R_c = Y_c - g_0(D_c)$ are residuals from a demand-only model, and $R^2_{\text{demo}}$ is the R² of regressing those residuals on demographic features x.

**What it measures**: After removing the expected effect of demand, how much do demographics explain the remaining service variation? If $\rho^2 = 0$, demographics don't predict the unexplained part — service is fair. If $\rho^2$ is high, demographics systematically predict which cells get over/under-served relative to demand — service is unfair.

**Role of g₀(D)**: Demand-only baseline. Strips out the demand effect so we can isolate demographics.

**Role of g(D,x)**: **Not used during optimization**. g(D,x) is only relevant during the model selection / exploration phase to understand the demand-demographics-service relationship. During optimization, only g₀(D) is needed (frozen), and the demographic regression is pre-computed as fixed coefficients.

**Implied model selection metric for g₀(D)**: Standard g(D) fitting quality (binning/isotonic/polynomial R²). We want g₀ to capture the demand→ratio relationship well so that residuals genuinely reflect the "unexplained" part.

**Implied model selection metric for g(D,x)**: If g(D,x) is used at all, it's for the exploratory phase only. The relevant metric would be LODO Residual-Demographic Independence (RDI): does including demographics in g reduce the correlation between residuals and demographics in held-out districts?

**Analysis**:

| Criterion | Assessment |
|-----------|------------|
| Causal validity | **Strong**. Directly operationalizes "do demographics predict service deviations from demand-expected levels?" This is the closest to the counterfactual question: "if we held demand constant, would demographics still predict service?" |
| Optimization signal | **Requires careful design**. The challenge is making $\rho^2(R, x)$ differentiable. R = Y - g₀(D) is differentiable (Y depends on supply). But $\rho^2(R, x)$ involves fitting a regression of R on x — this regression would need to be either (a) pre-computed with frozen coefficients, or (b) computed differentiably. Option (a) is simpler and consistent with the frozen-g principle. See Section 5.3.1 below. |
| g's role | Clear: g₀(D) removes demand effect; the demographic regression measures remaining bias |
| Robustness | **Moderate**. The demographic regression has only 10 effective data points (district means of residuals). With cell-level data (~300-500 cells), there's more signal, but all cells in a district share the same x. This is a real limitation but a data constraint, not a formulation problem. |
| Differentiability | Yes — with pre-computed demographic regression coefficients (see 5.3.1) |
| Interpretability | **Excellent**. "X% of service ratio deviations from demand-expected levels are explained by demographics." Direct, actionable. |

#### 5.3.1 Differentiable Implementation of Option B

The demographic regression $R \sim x$ must be handled carefully for gradient flow.

**Approach: Frozen demographic coefficients (β)**

Pre-compute the demographic regression coefficients β from the original (unmodified) data:

```
β = (X'X)^{-1} X'R₀    (where R₀ = Y₀ - g₀(D₀) from original data)
```

During optimization, compute:

```
R_c = Y_c - g₀(D_c)           ← differentiable through Y_c
predicted_R_c = β^T x_c        ← constant (β frozen, x fixed)
ρ² = Var(predicted_R) / Var(R)  ← differentiable through Var(R)
F_causal = 1 - ρ²             ← differentiable
```

Wait — this has the same issue as Option A1. If β and x are frozen, then predicted_R = β^T x is constant per cell. So $\rho^2 = \text{Var}(\text{constant}) / \text{Var}(R)$. The numerator is constant, and the optimizer would again maximize F_causal by increasing Var(R) — making residuals more spread out.

**This is a fundamental problem for all formulations that divide by Var(R) or Var(Y) while keeping the numerator frozen.**

**Resolution: Reformulate ρ² to avoid the degenerate gradient.**

Instead of the variance-ratio form, use the **sum-of-squares** form:

$$F_{\text{causal}} = 1 - \frac{\sum_c (R_c - \bar{R})(\hat{R}_c - \bar{\hat{R}})}{\sqrt{\sum_c (R_c - \bar{R})^2 \cdot \sum_c (\hat{R}_c - \bar{\hat{R}})^2}}$$

No — this is the correlation coefficient, not R². And the gradient is complex.

**Better resolution**: Use a direct measure that doesn't involve variance ratios:

$$F_{\text{causal}} = 1 - \frac{1}{\text{Var}(Y)} \sum_c w_c \cdot (\beta^T x_c)^2$$

where $w_c = 1/N$ and β are frozen coefficients. This is essentially $1 - \text{Var}(\hat{R}) / \text{Var}(Y)$, where:
- Numerator $\text{Var}(\hat{R})$ = constant (frozen β, fixed x)
- Denominator $\text{Var}(Y)$ = differentiable

This still has the perverse "increase Var(Y)" gradient.

**The core issue**: Any formulation of the form $1 - \frac{\text{constant}}{\text{Var}(\text{differentiable})}$ has a degenerate gradient that increases the denominator rather than decreasing the numerator.

#### 5.3.2 Resolving the Gradient Problem for Option B

The way to fix this is to **not normalize by Var(Y)**. Instead, use an absolute measure of demographic influence:

**Variant B-abs**: Sum of squared demographic-predicted residuals

$$F_{\text{causal}} = -\sum_c (\beta^T x_c \cdot (Y_c - g_0(D_c)))$$

This doesn't work either because β^T x_c is a constant and Y_c - g₀(D_c) is the residual — we want to minimize the product, but β^T x_c can be negative, creating sign issues.

**Variant B-cov**: Minimize the covariance between residuals and demographic predictions

$$F_{\text{causal}} = 1 - |\text{Cov}(R, \hat{R})| / \sigma$$

where $R = Y - g_0(D)$, $\hat{R} = \beta^T x$ (frozen). Cov(R, R̂) is differentiable through R, and its gradient pushes R to be uncorrelated with R̂ — exactly what we want. But the scaling factor σ reintroduces the normalization problem.

**Variant B-direct (Recommended)**: Use the mean squared demographic-predicted component of the residual, scaled by a fixed reference:

$$F_{\text{causal}} = 1 - \frac{\sum_c (\beta^T x_c) \cdot R_c}{\text{Var}_0(Y)}$$

where $\text{Var}_0(Y)$ is the variance of Y from the **original unmodified data** (a constant, pre-computed). This gives a clean gradient:

- $\nabla F_{\text{causal}} \propto -\sum_c (\beta^T x_c) \cdot \nabla R_c$
- Since $R_c = Y_c - g_0(D_c)$ and $g_0$ is frozen, $\nabla R_c = \nabla Y_c$
- The gradient pushes Y_c *down* in cells where $\beta^T x_c > 0$ (cells where demographics predict over-service) and *up* where $\beta^T x_c < 0$ (under-service)
- This is **exactly the right signal**: redistribute service to counteract demographic bias

But there's still a sign issue: $\sum (\beta^T x_c) \cdot R_c$ can be positive or negative. We need an absolute or squared version.

**Variant B-mse (Recommended)**: Minimize the mean squared covariance-weighted residual:

$$F_{\text{causal}} = 1 - \frac{\text{MSE}(\hat{R}, R)}{\text{Var}_0(Y)}$$

where $\hat{R}_c = \beta^T x_c$ (frozen demographic prediction), $R_c = Y_c - g_0(D_c)$ (differentiable residual), and $\text{Var}_0(Y)$ is a frozen constant from original data.

$\text{MSE}(\hat{R}, R) = \frac{1}{N}\sum_c (\hat{R}_c - R_c)^2$

Gradient: $\nabla F_{\text{causal}} \propto \sum_c (\hat{R}_c - R_c) \cdot \nabla R_c$

This pushes R_c toward $\hat{R}_c$ — wait, that's wrong. We want residuals to be *independent* of demographics, not to match the demographic prediction.

**The correct formulation needs to minimize the alignment between R and x, not minimize the distance between R and β^T x.**

**Variant B-corr (Final Recommended Form)**:

$$F_{\text{causal}} = 1 - \frac{1}{\text{Var}_0(Y)} \cdot \frac{1}{N} \sum_c \left(\sum_j \beta_j \cdot x_{cj}\right)^2 \cdot \text{sign\_match}(R_c, \hat{R}_c)$$

This is getting overly complex. Let us step back and reconsider.

#### 5.3.3 The Fundamental Gradient Design Principle

The optimizer changes Y_c by redistributing supply (moving trajectory pickups between cells). We want F_causal's gradient to encode:

> "Move supply toward under-served-relative-to-demographics cells and away from over-served-relative-to-demographics cells"

The simplest differentiable formulation that achieves this is:

$$F_{\text{causal}} = -\frac{1}{N \cdot \sigma_0} \sum_c \sum_j \beta_j \cdot \tilde{x}_{cj} \cdot R_c$$

where:
- $R_c = Y_c - g_0(D_c)$ is the demand-adjusted residual (differentiable through Y)
- $\beta_j$ are frozen coefficients from regressing R on x in original data
- $\tilde{x}_{cj}$ are standardized demographic features (fixed per cell)
- $\sigma_0$ is a normalizing constant from original data
- The negative sign means we want to minimize the weighted sum (high F_causal = low demographic correlation)

The gradient: $\nabla_{Y_c} F_{\text{causal}} = -\frac{1}{N \cdot \sigma_0} \sum_j \beta_j \cdot \tilde{x}_{cj}$

This is a **constant per cell** — it tells the optimizer exactly how much to increase or decrease Y_c based on that cell's demographic profile. Cells where demographics predict over-service ($\sum \beta_j x_{cj} > 0$) get pushed down; under-service cells get pushed up.

However, this linear form doesn't naturally live in [0, 1]. We can transform it:

$$F_{\text{causal}} = 1 - \frac{|\text{Cov}(R, \hat{R})|}{\text{Var}_0(Y)}$$

where $\text{Cov}(R, \hat{R}) = \frac{1}{N}\sum_c (R_c - \bar{R})(\hat{R}_c - \overline{\hat{R}})$ and $\hat{R}_c = \sum_j \beta_j \tilde{x}_{cj}$.

Since $\hat{R}$ is constant (frozen β, fixed x), $\overline{\hat{R}}$ is constant. So:

$\text{Cov}(R, \hat{R}) = \frac{1}{N}\sum_c (R_c - \bar{R}) \cdot c_c$

where $c_c = \hat{R}_c - \overline{\hat{R}}$ is a per-cell constant. This is differentiable through R_c, and its gradient is:

$\nabla_{R_c} |\text{Cov}| = \text{sign}(\text{Cov}) \cdot \frac{c_c}{N}$

This pushes $R_c$ to make Cov(R, R̂) = 0, which is **exactly** what we want: residuals uncorrelated with demographic predictions.

**This is the correct differentiable form of Option B.**

#### 5.3.4 Corrected Analysis: Direct R²(R ~ x) IS Differentiable (Revision)

> **This section supersedes Sections 5.3.1–5.3.3 above.** The earlier analysis
> incorrectly concluded that the standard R² formulation had a degenerate gradient.

The original Option B formulation $F_{\text{causal}} = 1 - R^2(R \sim x)$ **is directly differentiable** and does NOT suffer from the frozen-numerator problem. The error in Sections 5.3.1–5.3.3 was using the identity $R^2 = \text{Var}(\hat{y})/\text{Var}(y)$, which only holds when $\hat{y}$ is the OLS fit of the *current* y. With frozen $\beta$, $\hat{R}$ is not the OLS fit of current R, so this identity doesn't apply.

The correct formulation uses the **sum-of-squares** R²:

$$R^2_{\text{demo}} = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_c (R_c - \hat{R}_c)^2}{\sum_c (R_c - \bar{R})^2}$$

$$F_{\text{causal}} = 1 - R^2_{\text{demo}} = \frac{SS_{\text{res}}}{SS_{\text{tot}}} = \frac{\sum_c (R_c - \hat{R}_c)^2}{\sum_c (R_c - \bar{R})^2}$$

**Both SS_res and SS_tot depend on R_c**, which is differentiable through Y_c. Even with frozen $\hat{R}_c = \beta^T x_c$, the squared difference $(R_c - \hat{R}_c)^2$ changes as $R_c$ changes. There is no frozen-numerator / Var(Y) issue.

There are two viable implementation approaches:

##### Approach A: Frozen β with Sum-of-Squares R²

Pre-compute $\beta$ from original data. During optimization, $\hat{R}_c = \beta^T x_c$ is constant per cell, but SS_res and SS_tot both change as R changes.

**Gradient**:

$$\frac{\partial F}{\partial R_c} = \frac{2}{SS_{\text{tot}}} \left[(R_c - \hat{R}_c) - F \cdot (R_c - \bar{R})\right]$$

**Direction check** — Rich-district cell ($\hat{R}_c > 0$, over-served, $R_c \approx \hat{R}_c > \bar{R}$):
- $(R_c - \hat{R}_c) \approx 0$
- $-F \cdot (R_c - \bar{R}) < 0$ (since $R_c > \bar{R}$)
- Net gradient: **negative** → pushes $R_c$ down → reduces over-service ✓

**Direction check** — Poor-district cell ($\hat{R}_c < 0$, under-served, $R_c \approx \hat{R}_c < \bar{R}$):
- $(R_c - \hat{R}_c) \approx 0$
- $-F \cdot (R_c - \bar{R}) > 0$ (since $R_c < \bar{R}$)
- Net gradient: **positive** → pushes $R_c$ up → reduces under-service ✓

**Trade-off**: This is a pseudo-R² (stale $\beta$ may drift from the OLS fit as the optimizer changes R). In practice this may be minor since ST-iFGSM uses bounded perturbations.

##### Approach B: Hat Matrix (Analytical Re-fitting)

Express OLS analytically so $\beta$ is implicitly re-computed at each step. Since demographics X are fixed, $P = (X'X)^{-1}X'$ is a constant matrix, and $\beta^* = PR$ is linear in R (differentiable).

The hat matrix $H = X(X'X)^{-1}X'$ projects R onto the column space of X:

$$R^2_{\text{demo}} = 1 - \frac{R'(I-H)R}{R'MR} \quad \text{where } M = I - \frac{11'}{N}$$

$$F_{\text{causal}} = \frac{R'(I-H)R}{R'MR}$$

Both $(I-H)$ and $M$ are constant symmetric matrices. This is a ratio of quadratic forms in R — smoothly differentiable.

**Gradient**:

$$\frac{\partial F}{\partial R} = \frac{2}{R'MR} \left[(I-H)R - F \cdot MR\right]$$

**Key advantages**:
- Computes the **true R²** of regressing current R on x at every optimization step
- $\beta$ stays calibrated as the optimizer changes the service distribution
- Mathematically clean (no stale-coefficient concern)
- The projection (I-H) identifies the component of R that is orthogonal to demographics, which is precisely what we want to maximize

##### Approach Comparison

| Aspect | Frozen β (A) | Hat Matrix (B) |
|--------|-------------|----------------|
| **R² accuracy** | Pseudo-R² (stale β) | True R² (re-fitted β) |
| **Implementation** | Simpler (vector ops) | Requires H matrix (~500×500, computed once) |
| **Coefficient interpretability** | β visible, fixed | β implicit in H |
| **Gradient correctness** | Correct direction, may drift | Always correct |
| **Computational cost** | O(N) per step | O(N²) per step (matrix-vector) |
| **Recommended when** | ST-iFGSM (small perturbations) | Large modification budgets |

For FAMAIL's ST-iFGSM pipeline with bounded ε perturbations, **Approach A (frozen β)** is likely sufficient and simpler to implement. Approach B is the theoretically superior option if implementation cost is acceptable.

##### Value Range and Semantics

$F_{\text{causal}} = SS_{\text{res}} / SS_{\text{tot}}$ lives in $[0, 1]$ with clear semantics:
- $F_{\text{causal}} = 1$: demographics explain none of the residual variance → perfectly fair
- $F_{\text{causal}} = 0$: demographics explain all residual variance → maximally unfair
- Direct interpretation: "the fraction of demand-adjusted service variance that is NOT explained by demographics"

This is identical to Option B as originally proposed in the reformulation plan, without any workaround formulations needed.

---

### 5.4 Option C: Partial R² / ΔR²

**Formula**:
$$F_{\text{causal}} = 1 - \Delta R^2 = 1 - (R^2_{\text{full}} - R^2_{\text{reduced}})$$

where $R^2_{\text{full}}$ is from Y ~ D + x and $R^2_{\text{reduced}}$ is from Y ~ D.

**What it measures**: The incremental explanatory power of demographics beyond demand. If demographics add nothing ($\Delta R^2 = 0$), service is fair.

**Role of g**: Both g₀(D) and g(D,x) are needed — the formulation compares them.

**Analysis**:

| Criterion | Assessment |
|-----------|------------|
| Causal validity | **Strong in theory**. ΔR² is a well-established statistical test for the significance of additional regressors. It directly asks "do demographics add explanatory power beyond demand?" |
| Optimization signal | **Degenerate**. Both models are frozen during optimization. $R^2_{\text{full}}$ and $R^2_{\text{reduced}}$ are both of the form $1 - \text{Var}(\text{residual})/\text{Var}(Y)$. The residuals from both models have frozen prediction components, so $\Delta R^2$ = (frozen difference) / Var(Y), and the gradient again just increases Var(Y). |
| g's role | Clear but dual: need both demand-only and full models |
| Robustness | Same district-level limitations |
| Differentiability | Degenerate gradient (same structural issue as A1) |
| Interpretability | **Excellent**. "Demographics explain an additional X% of service variation beyond demand." |

**Verdict**: **Good diagnostic metric, problematic optimization target**. Like Option A1, the frozen-models-divided-by-Var(Y) structure creates a degenerate gradient. However, ΔR² is valuable as a *model selection metric* — it directly measures whether demographics matter.

---

### 5.5 Option D: Group Fairness (Demand-Stratified Comparison)

**Formula**:
$$F_{\text{causal}} = 1 - \frac{1}{|B|} \sum_b \max_{k_1, k_2} |\bar{Y}_{b,k_1} - \bar{Y}_{b,k_2}|$$

**What it measures**: At each demand level, the maximum difference in average service ratio between demographic groups.

**Analysis**:

| Criterion | Assessment |
|-----------|------------|
| Causal validity | **Excellent**. Directly measures the gap the meeting identified: "at the same demand level, do different income groups get different service?" |
| Optimization signal | **Good** if implemented with soft binning/continuous relaxation. The gradient pushes supply toward the under-served group at each demand level. |
| g's role | **None** — no g function needed. Demand binning replaces it. |
| Robustness | **Weak**. With 10 districts split into 2-3 income groups and 10 demand bins, each bin×group may have 5-15 cells. Very noisy. |
| Differentiability | **Challenging**. Hard binning (np.digitize) is not differentiable. Would need soft binning (Gaussian kernel). The max over group pairs is also non-smooth (would need softmax approximation). |
| Interpretability | **Best of all options**. "In similar-demand areas, rich and poor neighborhoods differ in service ratio by X." |

**Verdict**: **Excellent conceptual clarity, poor practical feasibility** at current data granularity. Worth revisiting if sub-district demographic data becomes available.

---

### 5.6 Synthesis: The Gradient Problem (Revised)

> **Revised**: The original synthesis incorrectly included Option B in the set of
> formulations with degenerate gradients. Option B's R²(R ~ x) formulation is
> fully differentiable (see Section 5.3.4). The problem is narrower than originally
> stated.

Options A1, A2, and the current production formulation share a structural problem:

> When g, β, and x are all frozen during optimization, any F_causal that takes the form $1 - \frac{\text{frozen numerator}}{\text{Var}(Y)}$ has a degenerate gradient that increases Var(Y) rather than addressing fairness.

The current production F_causal (R² = 1 - Var(R)/Var(Y) with frozen g₀(D)) also has this issue, but it's less severe because Var(R) is NOT fully frozen — R = Y - g₀(D), and while g₀(D_c) is frozen as a lookup, D itself changes during optimization (pickups move between cells), which changes which g₀ values are looked up. So both numerator and denominator change, and the gradient isn't purely degenerate. The optimizer does push Y toward g₀(D), though the direction is influenced by both the R² improvement and Var(Y) effects.

**The formulations that avoid this problem**:
1. **Option B** (Section 5.3.4): Uses R²(R ~ x) with sum-of-squares form. Both SS_res and SS_tot depend on R, so neither is frozen. Gradient flows correctly through both.
2. **Option B-corr** (Section 5.3.3): Uses Cov(R, R̂) / Var₀(Y) where Var₀ is a frozen constant from original data. Gradient flows through Cov. (However, this is now superseded by the direct R² form — see Section 5.3.4.)
3. **Option C** (with hat matrix formulation): ΔR² can be computed differentiably using hat matrices for both the reduced and full models. Both R² values change as Y changes.
4. **Option D with soft binning**: Gradient flows through group means directly.

---

## 6. Evaluation Metric Analysis

Now that we've analyzed the F_causal formulations, we can evaluate which model selection metric is appropriate for each.

### 6.1 Metrics Under Consideration

| Metric | Formula | Measures |
|--------|---------|----------|
| **LODO R²** | $1 - \text{Var}(Y - \hat{Y}_{\text{oof}}) / \text{Var}(Y)$ | Overall out-of-fold prediction accuracy |
| **LODO ΔR²** | LODO R²(full) - LODO R²(demand-only) | Demographic contribution to prediction |
| **LODO RDI** | $1 - R^2(\text{oof\_residuals} \sim x)$ | Residual independence from demographics |
| **Worst-District R²** | $\min_d R^2_d$ | Worst-case spatial prediction |
| **LODO MAE** | $\text{mean}(|Y - \hat{Y}_{\text{oof}}|)$ | Absolute prediction error |
| **Composite** | $\beta \cdot \text{LODO\_R}^2 + (1-\beta) \cdot \text{RDI}$ | Balanced accuracy + fairness |

### 6.2 Metric Appropriateness by Formulation

| F_causal Option | Primary Metric | Reasoning |
|-----------------|---------------|-----------|
| **A1** (Attribution) | LODO R² + Coefficient Stability | g must predict accurately, and β_demo must be stable |
| **A2** (Conditional R²) | LODO R² | Direct prediction accuracy |
| **B** (Residual-Demo) | g₀(D) fit quality + LODO ΔR² | g₀ must remove demand effect well; ΔR² confirms demographics matter |
| **B-corr** (Production) | g₀(D) fit quality + LODO RDI | g₀ must remove demand; RDI confirms the formulation will produce useful gradients |
| **C** (Partial R²) | LODO ΔR² | Directly measures what C computes |
| **D** (Group Fairness) | N/A (no g needed) | — |

### 6.3 The Case Against LODO R² as Primary Metric

LODO R² is the wrong primary metric for all formulations except A2 (which itself is not recommended). The reasons:

1. **For Option B/B-corr**: g₀(D) is demand-only — LODO R² of g(D,x) is irrelevant because g(D,x) isn't used during optimization. The relevant quality measure is how well g₀(D) captures the demand→ratio relationship, which is evaluated by standard R² or MAE on the demand-only model.

2. **LODO R² conflates demand and demographic effects**: A model with LODO R² = 0.40 might get 0.38 from demand and 0.02 from demographics. LODO R² can't distinguish these contributions.

3. **LODO R² rewards learning the unfair pattern**: If demographics truly cause service bias, a model that accurately predicts this bias (high LODO R²) has learned the unfair pattern. Using it to select g(D,x) and then using g(D,x) in F_causal creates a circularity where we select models that best encode the bias we're trying to eliminate.

### 6.4 Recommended Metrics by Use Case

**For g₀(D) selection** (demand-only baseline):
- **Primary**: In-sample R² of g₀(D) on the demand→ratio relationship
- **Secondary**: Visual inspection of g₀(D) fit (scatter plot with curve)
- **Rationale**: g₀(D) is fitted on all data and frozen. We want it to capture the demand effect well. Cross-validation is less critical because it's a simple 1D function with well-understood estimation methods (isotonic, polynomial).

**For g(D,x) model selection** (exploratory / understanding the data):
- **Primary**: LODO ΔR² (does adding demographics help predict service ratios in unseen districts?)
- **Secondary**: LODO RDI (are residuals independent of demographics in held-out districts?)
- **Tertiary**: Worst-District R² (does the model have spatial blind spots?)
- **Diagnostic**: Overfit Gap (Train R² - LODO R²), per-district R² variance
- **Rationale**: During exploration, we want to understand whether demographics matter (ΔR²) and whether g(D,x) successfully absorbs the demographic effect (RDI). LODO R² itself should be tracked but not used as the selection criterion.

**For β (demographic regression coefficients) selection** (production F_causal with Option B-corr):
- **Primary**: Stability of β across LODO folds (coefficient variation)
- **Secondary**: Statistical significance of β (p-values from OLS diagnostics)
- **Rationale**: The frozen β coefficients define the gradient direction for the optimizer. Unstable β means the gradient direction is unreliable. We want β that's consistent regardless of which district is held out.

---

## 7. Ranked Recommendations (revised in Section 7.3)

### 7.1 Recommended F_causal + g Configuration

Based on the analysis above, considering causal validity, gradient quality, data constraints, and practical feasibility:

---

#### Original Rank 1: Option B-corr — Covariance-based Demographic Residual Independence (recently revised)

**F_causal**:
$$F_{\text{causal}} = 1 - \frac{|\text{Cov}(R, \hat{R})|}{\text{Var}_0(Y)}$$

where:
- $R_c = Y_c - g_0(D_c)$ — differentiable residual (Y changes during optimization, g₀ frozen)
- $\hat{R}_c = \sum_j \beta_j \tilde{x}_{cj}$ — demographic prediction of residual (frozen β, fixed x)
- $\text{Var}_0(Y)$ — variance of Y from original unmodified data (frozen constant for normalization)

**g₀(D)**: Isotonic regression or polynomial (degree 2-3) on demand→ratio relationship. Pre-computed, frozen.

**β coefficients**: OLS regression of original residuals $R_0 = Y_0 - g_0(D_0)$ on standardized demographic features. Pre-computed, frozen.

**g(D,x)**: Not needed during optimization. Used only during exploration to validate that demographics carry meaningful signal.

**Model selection metric**: For β, use coefficient stability across LODO folds. For g₀, use standard fit quality metrics.

**Why Rank 1**:
- Clean gradient signal: pushes residuals to be uncorrelated with demographics
- Separates the demand effect (via g₀) from the demographic effect (via β)
- No degenerate Var(Y) gradient issue (Var₀ is frozen)
- Directly measures and optimizes against demographic influence
- Interpretable: "the covariance between demand-adjusted service and demographic predictions"
- Minimal change from current architecture (g₀ already exists; add frozen β lookup)

**Risks**:
- β may be unstable with only 10 district profiles
- Cov(R, R̂) could be very small, making F_causal ≈ 1 always (weak signal)
- |Cov| is not smooth at 0 (use sqrt(Cov² + ε) instead)

---

#### Rank 2: Option B-classic — R²(R ~ x) with Frozen Var₀ Normalization

**F_causal**:
$$F_{\text{causal}} = 1 - \frac{\text{Var}(\hat{R})}{\text{Var}_0(Y)}$$

where $\hat{R}_c = \beta^T x_c$ is fully frozen. This is a constant! F_causal doesn't change during optimization.

Wait — this doesn't work. If F_causal is constant, there's no gradient signal at all.

**Revision**: We need the residual to appear in the formulation in a way that's differentiable:

$$F_{\text{causal}} = 1 - \frac{\sum_c (\hat{R}_c - R_c)^2 - \sum_c (\bar{R} - R_c)^2}{N \cdot \text{Var}_0(Y)}$$

No, this is getting algebraically unwieldy and it's essentially trying to compute R²(R ~ x) in a differentiable way.

**Revised Rank 2**: The covariance form (Rank 1) is actually the cleanest differentiable form of Option B. Instead, offer a variant:

#### Rank 2: Option B-corr with Squared Covariance (Smooth)

$$F_{\text{causal}} = 1 - \frac{\text{Cov}(R, \hat{R})^2}{\text{Var}_0(Y) \cdot \text{Var}(\hat{R})}$$

This is actually the squared Pearson correlation $r^2(R, \hat{R})$, but with $\text{Var}(\hat{R})$ as a frozen constant (since $\hat{R}$ is frozen). So:

$$F_{\text{causal}} = 1 - \frac{\text{Cov}(R, \hat{R})^2}{C}$$

where C = Var₀(Y) · Var(R̂) is a constant. This is smooth at 0 (unlike |Cov|), differentiable, and the gradient pushes Cov toward 0.

**Why Rank 2**: Same merits as Rank 1, but uses squared correlation instead of absolute covariance. Smoother gradient near the optimum. Slightly less interpretable (squared units). The Var(R̂) normalization is also frozen, which is cleaner.

---

#### Rank 3: Option C — Partial R² (as diagnostic + selection metric, not optimization target)

**Recommended use**: Use ΔR² as the **model selection metric** during the exploration phase, not as the optimization target F_causal.

**Why**: ΔR² directly answers "should we care about demographics?" — if ΔR² ≈ 0 across all models, demographics don't matter and we can simplify. If ΔR² > 0, it validates the need for Option B-corr.

**Pair with**: Option B-corr for actual optimization.

---

#### Rank 4: Option A1 — Demographic Attribution (exploratory diagnostic only)

**Recommended use**: Compute Option A1's attribution (g(D,x) - g(D,x̄)) as a **diagnostic visualization** in the dashboard. It provides an intuitive per-cell "demographic influence" map.

**Do not use as F_causal** during optimization (degenerate gradient).

---

#### Rank 5: Option D — Group Fairness (reporting metric only)

**Recommended use**: Compute demand-stratified group comparisons as a **reporting metric** for papers and presentations. It's the most intuitive measure of fairness.

**Do not use as F_causal** during optimization (data sparsity in bins × groups, differentiability challenges).

---

#### Not Recommended: Option A2 — Conditional R²

**Reason**: Perpetuates the circular fairness baseline problem. Including demographics in g and then measuring R² of g(D,x) optimizes toward the existing (unfair) pattern.

---

### 7.2 Summary Table

| Rank | F_causal Formulation | g Function | Model Selection Metric | Use |
|------|---------------------|------------|----------------------|-----|
| **1** | B-corr: 1 - \|Cov(R, R̂)\| / Var₀ | g₀(D) isotonic + frozen β | β stability + g₀ fit quality | **Production optimization** |
| **2** | B-corr²: 1 - Cov²/(C) | g₀(D) isotonic + frozen β | β stability + g₀ fit quality | **Production (smoother)** |
| **3** | C: ΔR² | g₀(D) + g(D,x) comparison | LODO ΔR² | **Model selection metric** |
| **4** | A1: Attribution | g(D,x) conditional model | LODO R² + coeff stability | **Exploratory diagnostic** |
| **5** | D: Group Fairness | None (binning) | N/A | **Reporting metric** |
| ✗ | A2: Conditional R² | g(D,x) | LODO R² | **Not recommended** |

### 7.3 Revised Rankings (2026-02-14 Correction)

> The corrected differentiability analysis in Section 5.3.4 changes the rankings
> significantly. Option B's direct R² formulation is now the top recommendation,
> superseding the Cov-based workaround.

#### Revised Rank 1: Option B — Direct R²(R ~ x) with Hat Matrix

**F_causal**:
$$F_{\text{causal}} = 1 - R^2_{\text{demo}} = \frac{R'(I-H)R}{R'MR}$$

where:
- $R_c = Y_c - g_0(D_c)$ — differentiable residual
- $H = X(X'X)^{-1}X'$ — constant hat matrix (demographics projection)
- $M = I - 11'/N$ — constant centering matrix

**g₀(D)**: Isotonic or polynomial (degree 2-3). Pre-computed, frozen.

**g(D,x)**: NOT needed during optimization. The hat matrix H implicitly performs the demographic regression at each step.

**Model selection metric**: g₀(D) fit quality (in-sample R²). For exploratory validation, LODO ΔR² confirms demographics carry signal.

**Why Rank 1**:
- **Direct, interpretable formulation**: "fraction of demand-adjusted service variance not explained by demographics" — identical to the original Option B proposal
- **No workaround needed**: The sum-of-squares R² is directly differentiable; no Cov-based or Var₀-based reformulation required
- **True R² at every step** (hat matrix approach): β is implicitly re-fitted, so the signal stays calibrated
- **Clean gradient**: pushes over-served-relative-to-demographics cells down, under-served cells up
- **Natural [0, 1] range** with clear semantics
- **Minimal new infrastructure**: only need to pre-compute H matrix once (from demographic features X)

**Risks**:
- Same 10-district limitation as all demographic formulations
- H matrix is N×N (~500×500) — small enough to be practical
- If demographic signal is weak (R²_demo ≈ 0), F_causal ≈ 1 always and gradient is weak

**Simpler variant**: Use Approach A (frozen β) instead of the hat matrix if implementation simplicity is preferred. Gradient direction is identical; only the magnitude adaptation differs.

---

#### Revised Rank 2: Option C — Partial ΔR² (Differentiable via Hat Matrices)

**F_causal**:
$$F_{\text{causal}} = 1 - \Delta R^2 = 1 - \left(\frac{R'(I-H_{\text{red}})R}{R'MR} - \frac{R'(I-H_{\text{full}})R}{R'MR}\right)$$

where $H_{\text{red}}$ is the hat matrix for the demand-only model (Y ~ D) and $H_{\text{full}}$ is for the full model (Y ~ D + x). Note: this operates on Y directly, not residuals R, since we're comparing two models.

> **Note**: Upon reflection, Option C with hat matrices is mathematically equivalent to comparing Option B's R² (demographics explain residuals from g₀) against a baseline of zero. If g₀(D) is fitted via a method captured by $H_{\text{red}}$, then ΔR² = R²_full - R²_red, and the gradient is a combination of the two hat matrix gradients. This is strictly more complex than Option B for equivalent information.

**Why Rank 2**: Provides the clearest statistical answer ("do demographics add explanatory power beyond demand?") but is more complex to implement than Option B for essentially the same optimization signal. Best used as a validation metric alongside Option B.

---

#### Revised Rank 3: Option B-corr — Covariance-based (Superseded but Valid)

The Cov-based formulation from the original Rank 1 remains a valid fallback. It's simpler to implement than the hat matrix approach but has weaker interpretability (Cov magnitude isn't a proportion) and requires the Var₀(Y) normalization hack. Use this only if the hat matrix approach proves impractical.

---

#### Ranks 4–5 and Not Recommended: Unchanged

Options A1 (diagnostic), D (reporting), and A2 (not recommended) retain their original roles.

---

### 7.4 Revised Summary Table

| Rank | F_causal Formulation | g Function | Model Selection Metric | Use |
|------|---------------------|------------|----------------------|-----|
| **1** | B-direct: $SS_{\text{res}}/SS_{\text{tot}}$ via hat matrix | g₀(D) isotonic/poly | g₀ fit quality + LODO ΔR² | **Production optimization** |
| **1-alt** | B-direct: frozen β SS form | g₀(D) isotonic/poly + frozen β | g₀ fit quality + β stability | **Production (simpler)** |
| **2** | C: ΔR² via hat matrices | g₀(D) + demographics | LODO ΔR² | **Validation metric** |
| **3** | B-corr: Cov-based | g₀(D) + frozen β | β stability | **Fallback** |
| **4** | A1: Attribution | g(D,x) | LODO R² + coeff stability | **Exploratory diagnostic** |
| **5** | D: Group Fairness | None (binning) | N/A | **Reporting metric** |
| ✗ | A2: Conditional R² | g(D,x) | LODO R² | **Not recommended** |

---

## 8. Implementation Path

Given the ranked recommendations, the implementation path is:

### Phase 1: Validate the Premise (Current Dashboard Work)

Use the demographic explorer dashboard to confirm that demographics carry signal:
1. Fit g(D,x) models with various architectures
2. Compute **LODO ΔR²** (not LODO R²) as the primary metric → confirms demographics add predictive power
3. Compute **per-district residual-demographic correlations** → identifies which demographic features drive bias
4. **If LODO ΔR² ≈ 0 for all models**: demographics don't help at district-level granularity. Pursue sub-district data or accept that causal fairness at current resolution is limited.
5. **If LODO ΔR² > 0**: proceed to Phase 2.

### Phase 2: Implement B-corr

1. Fit g₀(D) using best demand-only method (isotonic or polynomial)
2. Compute residuals R₀ = Y₀ - g₀(D₀) on original data
3. Regress R₀ on standardized demographics x to get β coefficients
4. Validate β stability across LODO folds
5. Implement the differentiable F_causal computation in `FAMAILObjective`
6. Verify gradient flow (adapt existing `verify_causal_fairness_gradient`)

### Phase 3: Validate Optimization Behavior

1. Run ST-iFGSM with new F_causal on a small batch
2. Check that the optimizer moves supply in the expected direction (toward under-served-relative-to-demographics cells)
3. Compare F_causal before/after modification
4. Compute Option D (group fairness) as a validation metric — did the demand-stratified income gap decrease?

### Phase 4: Full Pipeline Integration

1. Add β coefficients to DataBundle
2. Update trajectory modification dashboard to show new F_causal
3. Run ablation: compare old g(D)-only F_causal vs. new B-corr F_causal
4. Report Option D group fairness improvements for paper

---

## 9. Open Questions and Risks

### 9.1 The 10-District Limitation

All formulations involving demographics are limited by having only 10 distinct demographic profiles. This means:
- Regression β has at most 10 effective data points for demographic effects
- LODO has at most 10 folds, with high variance in per-fold estimates
- Overfitting to district-specific effects (not truly demographic) is a real risk

**Mitigation**: Use regularized regression (Ridge) for β estimation; validate β stability across folds; use only 2-3 demographic features (not all 13+7 enriched).

### 9.2 Confounding

District-level demographics correlate with many things: road network quality, distance to city center, land use patterns, taxi driver familiarity. The β coefficients may capture these confounds rather than true demographic effects.

**Mitigation**: This is acknowledged as a limitation. The term should perhaps be renamed "Demographic Fairness" rather than "Causal Fairness" (as noted in the reformulation plan, Question 3).

### 9.3 Gradient Signal Strength

If the demographic effect is small (β coefficients are tiny), Cov(R, R̂) will be near zero, and F_causal ≈ 1 already. The gradient signal for the optimizer would be very weak compared to F_spatial and F_fidelity, making α₂ effectively zero regardless of its weight.

**Mitigation**: Monitor F_causal's gradient magnitude relative to other terms. If too weak, consider increasing α₂ or using a different scaling.

### 9.4 Interaction Between F_causal and F_spatial

F_spatial (Gini coefficient) already pushes toward equitable service distribution. If demographic inequality is correlated with spatial inequality (rich districts are well-served AND spatially concentrated), F_spatial may already address some of the demographic bias. The unique contribution of F_causal is to direct the optimization specifically toward *demographic* equity, not just *spatial* equity.

**Investigation needed**: Compute the correlation between F_spatial's gradient direction and F_causal's gradient direction. If they're highly correlated, F_causal may be redundant. If they're independent or opposed (e.g., F_spatial wants to equalize all cells but F_causal wants to specifically boost poor-district cells), they provide complementary signals.

### 9.5 The g₀(D) Circularity Remains

Even with Option B-corr, g₀(D) is fitted on observed (potentially unfair) data. If service is systematically lower in low-demand areas *because of demographic bias* (low-income areas have both low demand and poor service), g₀(D) will absorb some of this bias — it will predict low service for low demand, and the residuals will be smaller than the true demographic effect.

**Mitigation**: This is a known limitation but is less severe than the original formulation. g₀(D) captures the *average* demand→service relationship across all districts. The residual R = Y - g₀(D) captures deviations from this average. If low-income districts consistently fall below the average demand-service curve, β will pick this up. The bias in g₀(D) attenuates the signal but doesn't eliminate it.

---

## Appendix A: Notation Reference

| Symbol | Meaning |
|--------|---------|
| $Y_c$ | Service ratio at cell c: $S_c / D_c$ |
| $D_c$ | Demand (pickup counts) at cell c |
| $S_c$ | Supply (active taxis or dropoff counts) at cell c |
| $x_c$ | Demographic feature vector for cell c (from district) |
| $\bar{x}$ | Mean demographic vector across all cells |
| $g_0(D)$ | Demand-only predictor of service ratio |
| $g(D, x)$ | Demand + demographics predictor of service ratio |
| $R_c$ | Residual: $Y_c - g_0(D_c)$ |
| $\hat{R}_c$ | Demographic prediction of residual: $\beta^T x_c$ |
| $\beta$ | Regression coefficients: R ~ x |
| $\text{Var}_0(Y)$ | Variance of Y from original (unmodified) data |
| $F_{\text{causal}}$ | Causal fairness score in [0, 1], higher = fairer |

## Appendix B: Differentiability Audit for Option B (Revised)

### B.1 Hat Matrix Approach (Revised Rank 1)

```
F_causal = R'(I-H)R / R'MR

where:
  H = X(X'X)⁻¹X'  = constant projection matrix (demographics)
  M = I - 11'/N     = constant centering matrix
  R_c = Y_c - g₀(D_c)  → differentiable through Y_c
  Y_c = S_c / D_c       → differentiable through S_c
  S_c = Σ_τ soft_assign(τ)  → differentiable through trajectory positions

Numerator: R'(I-H)R = Σ_{i,j} R_i (I-H)_{ij} R_j  ← quadratic in R ✓
Denominator: R'MR = Σ_{i,j} R_i M_{ij} R_j          ← quadratic in R ✓

∂F/∂R = (2/R'MR) · [(I-H)R - F · MR]

Per cell: ∂F/∂R_c = (2/SS_tot) · [((I-H)R)_c - F · (R_c - R̄)]

Gradient flow:
  F_causal → R (via quadratic form) → Y → S/D → soft cell assignment → pickups ✓

Direction check:
  Over-served rich cell (R_c > R̄, demographics explain it):
    (I-H)R_c ≈ 0 (well-predicted by demographics)
    -F·(R_c - R̄) < 0
    Net: negative → pushes R_c down → reduces over-service ✓

  Under-served poor cell (R_c < R̄, demographics explain it):
    (I-H)R_c ≈ 0
    -F·(R_c - R̄) > 0
    Net: positive → pushes R_c up → reduces under-service ✓

  Anomalous cell (R_c deviates but NOT due to demographics):
    (I-H)R_c is large (not captured by H projection)
    This term dominates → gradient doesn't try to "fix" non-demographic deviations ✓
```

### B.2 Frozen β Approach (Revised Rank 1-alt)

```
F_causal = SS_res / SS_tot = Σ(R_c - R̂_c)² / Σ(R_c - R̄)²

where R̂_c = β^T x_c is frozen constant per cell.

SS_res = Σ_c (R_c - R̂_c)²  ← changes as R_c changes (R̂_c fixed, but diff changes) ✓
SS_tot = Σ_c (R_c - R̄)²    ← changes as R_c changes ✓

∂F/∂R_c = (2/SS_tot) · [(R_c - R̂_c) - F · (R_c - R̄)]

Same gradient direction as hat matrix approach, but R̂_c uses stale β rather
than the true current projection HR. In practice, for bounded ε perturbations,
the difference is negligible.
```

### B.3 Original Cov-based Approach (Rank 3 Fallback)

```
F_causal = 1 - |Cov(R, R̂)| / Var₀(Y)

∂F/∂R_c = -sign(Cov) · c_c / (N · Var₀(Y))
  where c_c = R̂_c - R̂̄ (constant per cell)

This is a valid formulation but superseded by the direct R² approach,
which has clearer interpretation, natural [0,1] range without ad-hoc
normalization, and equivalent gradient direction.
```
