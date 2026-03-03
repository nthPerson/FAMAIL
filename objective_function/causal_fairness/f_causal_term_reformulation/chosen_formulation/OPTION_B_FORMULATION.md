# Option B: Demographic Residual Independence via Hat Matrix

## The Chosen F_causal Formulation for FAMAIL

**Date**: 2026-02-25
**Status**: Chosen formulation (default in implementation)
**Prerequisite Reading**: `../FCAUSAL_GDX_JOINT_ANALYSIS.md`, `../G0_DEMAND_BASIS_ANALYSIS.md`, `../FCAUSAL_FORMULATIONS.md`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation: Why the Original F_causal Needed Reformulation](#2-motivation-why-the-original-fcausal-needed-reformulation)
3. [The Path to Option B: From g(D) to g(D,x) to Hat Matrix](#3-the-path-to-option-b-from-gd-to-gdx-to-hat-matrix)
4. [The Option B Formulation](#4-the-option-b-formulation)
5. [Matrix Algebra Walkthrough](#5-matrix-algebra-walkthrough)
6. [The Role of g₀(D): Power Basis Demand Model](#6-the-role-of-g₀d-power-basis-demand-model)
7. [Causal Analysis Foundations](#7-causal-analysis-foundations)
8. [Why Option B is Appropriate for FAMAIL](#8-why-option-b-is-appropriate-for-famail)
9. [Controlled Direct Effect (CDE) Evaluation](#9-controlled-direct-effect-cde-evaluation)
10. [Implementation Summary](#10-implementation-summary)
11. [Addressing Open Questions from Project Meetings](#11-addressing-open-questions-from-project-meetings)
12. [References and Related Literature](#12-references-and-related-literature)

---

## 1. Executive Summary

The causal fairness term F_causal in the FAMAIL objective function has been reformulated from a simple demand-alignment measure (R² of service ratios against demand) to a **demographic residual independence** measure that directly tests whether demographic factors like income, GDP, and housing prices predict taxi service levels after controlling for demand.

**The chosen formulation (Option B):**

$$F_{\text{causal}} = \frac{R'(I - H)R}{R'MR}$$

where:
- $R = Y - g_0(D)$: residuals after removing the demand effect
- $H = X(X'X)^{-1}X'$: the hat matrix that projects onto the demographic feature space
- $M = I - \frac{11'}{N}$: the centering matrix
- $I$: identity matrix

**Interpretation**: F_causal measures the fraction of demand-adjusted service variation that is **not** explained by demographics. Higher F_causal (closer to 1) means service distribution is more independent of demographics — more fair.

This formulation:
- Is grounded in **counterfactual fairness** and **Controlled Direct Effect** (CDE) from causal inference theory
- Uses the **hat matrix** to analytically perform the demographic regression at every optimization step, eliminating stale coefficients
- Is **fully differentiable**, enabling end-to-end gradient-based trajectory modification
- Provides **clean, directionally correct gradients** that push over-served wealthy cells down and under-served poor cells up

---

## 2. Motivation: Why the Original F_causal Needed Reformulation

### 2.1 The Original Formulation

The original causal fairness term was:

$$F_{\text{causal}} = R^2 = 1 - \frac{\text{Var}(Y - g_0(D))}{\text{Var}(Y)}$$

where $g_0(D)$ was an isotonic regression of the service ratio $Y = S/D$ on demand $D$, fitted from **observed data**. The R² measured how well demand alone explains variation in service ratios.

### 2.2 Three Critical Problems

These problems were identified during the **Formulation Verification Meeting** (February 6, 2026) and refined through Meetings 22–24:

**Problem 1: Circular fairness baseline.** The function $g_0(D)$ was estimated from the observed data, which already encodes discriminatory patterns. The "expected" service ratio $g_0(d)$ was learned from a world where low-income areas receive less service — so the model treats that as "normal." It cannot represent a truly fair baseline because it was trained on unfair outcomes.

**Problem 2: Demand-only model is a misnomer for "causal" fairness.** The original formulation measured *model fit error* — how well demand predicts supply — not whether *sensitive attributes* (income, demographics) are driving the supply distribution. This is better described as "unexplained inequality" rather than "causal fairness." As noted in the Formulation Verification Meeting: the term was "essentially a misnomer."

**Problem 3: No mechanism to detect demographic bias.** If two cells have the same demand but are in different districts (one wealthy, one poor), the original $g_0(D)$ predicts the same service ratio for both. It has no way to detect or penalize the systematic service gap between low-income and high-income areas *at the same demand level*. The formulation was blind to the very inequality it was supposed to measure.

### 2.3 The Core Insight

The key insight from the Formulation Verification Meeting: the causal fairness term must directly measure and penalize the **influence of demographic factors on service distribution**, not merely how well demand explains supply. After trajectory modification, the difference in service ratio between low-income and high-income areas — *holding demand constant* — should be minimized.

This reframing moved the question from:
- Old: "Does supply track demand?" (demand alignment)
- New: **"After accounting for demand, do demographics still predict service?"** (demographic independence)

---

## 3. The Path to Option B: From g(D) to g(D,x) to Hat Matrix

### 3.1 First Attempt: Fit g(D, x) with Demographic Features

The initial plan (documented in `G_FUNCTION_REFORMULATION_PLAN.md`) was to fit a conditional model $g(\text{Supply} \mid \text{Demand}, \text{Income})$ that includes demographic features alongside demand. The idea: measure how much the model's predictions change when demographics are varied.

**Why this was abandoned** (documented in `FCAUSAL_GDX_JOINT_ANALYSIS.md`):

1. **Circular dependency**: The "right" g(D,x) model depends on which F_causal formulation uses it, and which F_causal formulation works best depends on how good g is — a chicken-and-egg problem.

2. **LODO R² is the wrong metric**: Leave-One-District-Out cross-validation measures *predictive accuracy*, not whether the demographic signal represents *unfairness*. A model that predicts well might simply be learning the unfair patterns more accurately.

3. **Model selection complexity**: OLS, Ridge, Random Forest, Neural Network — each gives different R², and none directly answers the fairness question. Testing at Meeting 24 showed neural network achieving R² ≈ 0.6, but even a good model doesn't resolve whether the demographic predictions represent bias.

4. **Overfitting risk**: With only 10 Shenzhen districts providing demographic variation, complex models can memorize district-specific patterns rather than learning generalizable relationships.

### 3.2 The Breakthrough: Two-Stage Approach

The breakthrough insight (documented in `FCAUSAL_GDX_JOINT_ANALYSIS.md`, Section 5.3) was to separate the problem into two stages:

**Stage 1**: Fit a demand-only model $g_0(D)$ to capture the expected demand→service relationship. Compute residuals $R = Y - g_0(D)$. These residuals represent the service variation that demand *does not explain*.

**Stage 2**: Test whether demographics predict these residuals. If they do (R²_demo > 0), then demographic factors are influencing service beyond what demand would dictate — that's unfairness. If they don't (R²_demo ≈ 0), service deviations from demand expectations are unrelated to demographics — that's fair.

This elegantly avoids the circular dependency: g₀(D) only needs to capture the demand effect (no demographic features involved), and the fairness test is a separate, well-defined regression of residuals on demographics.

### 3.3 The Hat Matrix Discovery

The final innovation, discovered by Meeting 23 (February 17, 2026), was realizing that the demographic regression in Stage 2 can be expressed **analytically** via the hat matrix:

$$H = X(X'X)^{-1}X'$$

Because demographics $X$ are **fixed** properties of each grid cell (they never change during optimization), the hat matrix $H$ is a **constant** that can be pre-computed once. During optimization, the residual vector $R$ changes as the optimizer modifies trajectories, but the projection $HR$ always gives the *current* OLS-best-fit of $R$ onto demographics — automatically, without re-running any regression.

This eliminated:
- The need to fit a separate demographic model (no neural network, no Ridge, no model selection)
- The risk of stale coefficients (the hat matrix implicitly re-fits at every step)
- Differentiability concerns (the computation is a ratio of smooth quadratic forms in $R$)

As noted in Meeting 23: "Hat matrix approach is fully differentiable and provides huge computational efficiency gains."

---

## 4. The Option B Formulation

### 4.1 Complete Mathematical Statement

Given:
- $N$ active grid cells (cells with sufficient demand and valid demographic data)
- $Y_c = S_c / D_c$: the observed service ratio at cell $c$
- $g_0(D_c)$: the demand-only expected service ratio (frozen power basis model)
- $X \in \mathbb{R}^{N \times (p+1)}$: the design matrix of standardized demographic features with intercept
- $R_c = Y_c - g_0(D_c)$: the demand-adjusted residual at cell $c$

The causal fairness score is:

$$\boxed{F_{\text{causal}} = \frac{R'(I - H)R}{R'MR} = 1 - R^2_{\text{demo}}}$$

where:
- $H = X(X'X)^{-1}X'$ is the hat matrix (demographic projection)
- $M = I - \frac{11'}{N}$ is the centering matrix
- $R^2_{\text{demo}} = 1 - \frac{R'(I-H)R}{R'MR}$ is the R² of regressing residuals on demographics

### 4.2 Component Interpretation

| Symbol | Meaning | Constant? |
|--------|---------|:---------:|
| $R$ | Demand-adjusted residuals: "How much more/less service does this cell get than demand predicts?" | No — changes as optimizer moves pickups |
| $H$ | Demographic hat matrix: "What would the OLS regression of R on demographics predict?" | **Yes** — demographics are fixed |
| $(I-H)R$ | Residual after removing demographic signal: "What's left after demographics can't explain it?" | No — changes with R |
| $R'(I-H)R$ | $SS_{\text{res}}$: Total squared residual variance *not* explained by demographics | No |
| $R'MR$ | $SS_{\text{tot}}$: Total centered variance of residuals | No |
| $M$ | Centering matrix: subtracts the mean from any vector | **Yes** |

### 4.3 Score Interpretation

- **F_causal = 1.0**: Demographics explain *none* of the demand-adjusted service variation → perfectly fair
- **F_causal = 0.0**: Demographics explain *all* of the demand-adjusted service variation → maximally unfair
- **F_causal ≈ 0.95–0.98** (typical range with Shenzhen data): Weak but nonzero demographic influence on residuals

### 4.4 Gradient

The gradient of F_causal with respect to the residual vector R is:

$$\frac{\partial F}{\partial R_c} = \frac{2}{R'MR} \left[((I-H)R)_c - F \cdot (R_c - \bar{R})\right]$$

**Directional correctness verification:**

| Cell Type | Example | Gradient Direction | Effect |
|-----------|---------|-------------------|--------|
| Over-served wealthy cell | $R_c > \bar{R}$, demographics predict over-service | **Negative** | Push service down |
| Under-served poor cell | $R_c < \bar{R}$, demographics predict under-service | **Positive** | Push service up |
| Anomalous cell | Deviation NOT explained by demographics | $(I-H)R_c$ dominates | Gradient doesn't interfere |

The gradient correctly redistributes service to counteract demographic bias while leaving demand-driven and random variation alone.

---

## 5. Matrix Algebra Walkthrough

This section provides a step-by-step walkthrough of the hat matrix formulation to aid in understanding and mathematical validation. This addresses the action item from Meeting 24: *"Validate mathematical formulation of hat matrix expressions."*

### 5.1 Setup: The OLS Regression R ~ X

We want to measure how well demographics predict demand-adjusted residuals. In classical OLS:

$$\hat{R} = X\hat{\beta}, \quad \hat{\beta} = (X'X)^{-1}X'R$$

where $\hat{R}$ is the vector of fitted values (what demographics predict the residuals should be) and $\hat{\beta}$ are the regression coefficients.

### 5.2 The Hat Matrix as a Projection

Substituting the OLS solution back:

$$\hat{R} = X(X'X)^{-1}X'R = HR$$

The hat matrix $H$ **projects** any vector onto the column space of $X$ (the space spanned by demographic features). This is why it's called the "hat" matrix — it puts the hat on $R$ to get $\hat{R}$.

**Key properties of H:**
- $H$ is **symmetric**: $H = H'$
- $H$ is **idempotent**: $HH = H$ (projecting twice is the same as projecting once)
- $H$ has **rank** $p + 1$ (number of demographic features + intercept)
- $(I - H)$ is also symmetric and idempotent: it projects onto the space **orthogonal** to demographics

### 5.3 Decomposing R into Demographic and Non-Demographic Components

Using $H$, we decompose the residual vector into two orthogonal parts:

$$R = \underbrace{HR}_{\hat{R}} + \underbrace{(I-H)R}_{e}$$

where:
- $\hat{R} = HR$: the part of $R$ that lives in the demographic subspace — the part *explained by* demographics
- $e = (I-H)R$: the part of $R$ orthogonal to demographics — the part demographics *cannot explain*

Because $H$ and $(I-H)$ are complementary projections onto orthogonal subspaces, these components are perpendicular: $\hat{R}'e = 0$.

### 5.4 The Pythagorean Decomposition of Variance

By the Pythagorean theorem for orthogonal decompositions:

$$\|R - \bar{R}\|^2 = \|\hat{R} - \bar{\hat{R}}\|^2 + \|e - \bar{e}\|^2$$

$$R'MR = (HR)'M(HR) + ((I-H)R)'M((I-H)R)$$

$$SS_{\text{tot}} = SS_{\text{explained}} + SS_{\text{residual}}$$

Note: because $X$ includes an intercept column, $\bar{\hat{R}} = \bar{R}$, so the mean of the fitted values equals the mean of the actual values, and the decomposition is exact.

### 5.5 R² and F_causal

The R² of the demographic regression is:

$$R^2_{\text{demo}} = \frac{SS_{\text{explained}}}{SS_{\text{tot}}} = 1 - \frac{SS_{\text{residual}}}{SS_{\text{tot}}} = 1 - \frac{R'(I-H)R}{R'MR}$$

Therefore:

$$F_{\text{causal}} = 1 - R^2_{\text{demo}} = \frac{R'(I-H)R}{R'MR} = \frac{SS_{\text{residual}}}{SS_{\text{tot}}}$$

This is the fraction of residual variance that demographics **cannot** explain — the "fair" component.

### 5.6 Why the Hat Matrix Eliminates Stale Coefficients

A common concern is: "If $H$ is pre-computed but $R$ changes during optimization, doesn't the regression become outdated?"

No. The hat matrix $H$ depends **only** on $X$ (demographics), which is constant. The regression coefficients $\hat{\beta} = (X'X)^{-1}X'R$ are **linear functions of R**. As $R$ changes, $\hat{\beta}$ changes automatically:

$$\hat{\beta}_{\text{new}} = (X'X)^{-1}X'R_{\text{new}}$$

The hat matrix captures this implicitly:

$$\hat{R}_{\text{new}} = HR_{\text{new}} = X(X'X)^{-1}X'R_{\text{new}} = X\hat{\beta}_{\text{new}}$$

At every optimization step, $HR$ gives the **exact** OLS fit of the *current* residual vector on demographics. No re-fitting is needed because the projection is a single matrix-vector multiplication.

### 5.7 Numerical Example

Consider 4 cells, 1 demographic feature (income), and demand-adjusted residuals $R$:

$$X = \begin{bmatrix} 1 & -1.2 \\ 1 & -0.5 \\ 1 & 0.6 \\ 1 & 1.1 \end{bmatrix}, \quad R = \begin{bmatrix} -0.3 \\ -0.1 \\ 0.2 \\ 0.4 \end{bmatrix}$$

Step 1: $X'X = \begin{bmatrix} 4 & 0 \\ 0 & 3.06 \end{bmatrix}$

Step 2: $(X'X)^{-1} = \begin{bmatrix} 0.25 & 0 \\ 0 & 0.327 \end{bmatrix}$

Step 3: $X'R = \begin{bmatrix} 0.2 \\ 0.9 \end{bmatrix}$

Step 4: $\hat{\beta} = (X'X)^{-1}X'R = \begin{bmatrix} 0.05 \\ 0.294 \end{bmatrix}$

Step 5: $H = X(X'X)^{-1}X'$ (a 4×4 constant matrix)

Step 6: $HR = \hat{R} = \begin{bmatrix} -0.303 \\ -0.097 \\ 0.226 \\ 0.373 \end{bmatrix}$

Step 7: $SS_{\text{res}} = R'(I-H)R = 0.0014$, $SS_{\text{tot}} = R'MR = 0.305$

Step 8: $R^2_{\text{demo}} = 1 - 0.0014/0.305 = 0.995$

Step 9: $F_{\text{causal}} = 0.005$

In this example, income strongly predicts the service residuals — highly unfair. The optimizer would push $R$ toward a configuration where the rich/poor gradient disappears.

### 5.8 The Centering Matrix M

The centering matrix $M = I - \frac{11'}{N}$ subtracts the mean from any vector:

$$Mv = v - \bar{v}$$

In the denominator $R'MR$:

$$R'MR = (R - \bar{R})'(R - \bar{R}) = \sum_c (R_c - \bar{R})^2 = N \cdot \text{Var}(R)$$

This is simply $N$ times the sample variance of the residuals — the total sum of squares ($SS_{\text{tot}}$).

---

## 6. The Role of g₀(D): Power Basis Demand Model

### 6.1 Does the Hat Matrix Replace g₀(D)?

**No.** The hat matrix and g₀(D) serve different, complementary roles:

| Component | Role | During Optimization |
|-----------|------|:-------------------:|
| $g_0(D)$ | Remove the demand effect from service ratios to produce residuals $R = Y - g_0(D)$ | Frozen lookup (pre-computed) |
| $H$ (hat matrix) | Project residuals onto demographic space to measure demographic influence | Constant matrix (pre-computed) |

The hat matrix operates on the **residuals** after demand has been removed. Without g₀(D), the residuals would be the raw service ratios $Y$, which contain both demand-driven and demographic-driven variation. The demand signal would dominate, making the demographic R² unreliable.

g₀(D) is essential for Option B because it isolates the "unexplained" service variation that the demographic regression then analyzes.

### 6.2 Why Power Basis Instead of Isotonic Regression?

The demand→service ratio relationship is fundamentally **hyperbolic**: when demand $D$ is small, $Y = S/D$ is large and variable; as $D$ grows, $Y$ approaches a constant. Empirical comparison on the Shenzhen data:

| Method | R² | Hat-Matrix Compatible |
|--------|:---:|:---------------------:|
| Isotonic | 0.4453 | No |
| **Power Basis** | **0.4450** | **Yes** |
| Reciprocal | 0.4444 | Yes |
| Binning (10 bins) | 0.4439 | No |
| Log | 0.3610 | Yes |
| Polynomial (deg 2) | 0.1949 | Yes |
| Linear | 0.1064 | Yes |

The power basis model:

$$g_0(D) = \beta_0 + \frac{\beta_1}{D+1} + \frac{\beta_2}{\sqrt{D+1}} + \beta_3 \cdot \sqrt{D+1}$$

matches isotonic's R² within **0.0003** while being **linear in parameters**. Each term captures a different aspect of the decay:
- $1/(D+1)$: Rapid hyperbolic decay (dominant when $D$ is small)
- $1/\sqrt{D+1}$: Moderate decay rate
- $\sqrt{D+1}$: Slow sub-linear growth (captures the tail behavior)

### 6.3 Can g₀(D) Be Further Optimized?

The power basis g₀(D) achieves R² = 0.4450 vs. isotonic's R² = 0.4453 — a gap of only 0.07%. This is effectively at the ceiling of what a monotone demand→ratio function can capture. The remaining ~55% of variance in $Y$ is driven by:

- Spatiotemporal heterogeneity (time of day, day of week)
- Individual driver behavior
- Stochastic effects (small sample sizes per cell)
- **Demographic factors** (the signal we measure with Option B)

Further optimization of g₀(D) — for example, adding more basis functions or using a different functional form — would yield diminishing returns. The critical question is not whether g₀(D) is perfect, but whether it removes *enough* of the demand signal that the residuals cleanly reflect the demographic influence.

### 6.4 Why Not Isotonic for Option B?

Option B's hat matrix approach does not *require* g₀ to be linear in parameters — the hat matrix operates on residuals, not on g₀ itself. In principle, isotonic regression could be used for g₀ in Option B.

However, using power basis provides **consistency with Option C** (which does require linear-in-parameters demand features) and enables the **Frisch-Waugh-Lovell equivalence** between Options B and C, which serves as a mathematical validation check. Power basis is chosen for Option B not out of necessity, but for theoretical consistency and practical validation.

The **baseline formulation** (kept for backward compatibility per advisor's request) retains isotonic g₀ to preserve exact historical comparability.

---

## 7. Causal Analysis Foundations

This section establishes how the Option B formulation relates to established concepts in causal inference and algorithmic fairness. This addresses the research task: *"Develop an explanation/story for how the F_causal term extends Counterfactual Fairness or whatever causal fairness analysis concept is most appropriate."*

### 7.1 The Causal Graph for Taxi Service Fairness

The FAMAIL taxi service system can be represented as a causal directed acyclic graph (DAG):

```
                    Demographic
                    Features (X)
                   /     |       \
                  /      |        \
                 v       v         v
              Demand   [Direct   Infrastructure,
              (D)     Effect]    POI Density, etc.
                 \       |        /
                  \      |       /
                   v     v      v
                 Service Ratio (Y = S/D)
```

The key causal pathways from demographics to service ratio:

1. **Mediated path** (acceptable): X → D → Y. Wealthy areas may have higher demand, leading to different service ratios. This is demand-driven and acceptable.

2. **Direct path** (potentially unfair): X → Y. Demographics influence service *beyond* what demand explains. Drivers may prefer wealthy neighborhoods for higher tips, better road conditions, or perceived safety — creating a direct demographic effect on service that is not mediated through demand.

Option B isolates pathway (2) by first removing pathway (1) via $g_0(D)$, then testing whether residuals $R = Y - g_0(D)$ correlate with $X$.

### 7.2 Counterfactual Fairness

**Definition** (Kusner et al., 2017): A prediction $\hat{Y}$ is counterfactually fair with respect to sensitive attribute $A$ if, for any individual:

$$P(\hat{Y}_{A \leftarrow a} = y \mid X = x, A = a) = P(\hat{Y}_{A \leftarrow a'} = y \mid X = x, A = a)$$

In plain language: the prediction would remain the same in a *counterfactual world* where the individual's sensitive attribute were different.

**Connection to Option B**: Consider a grid cell $c$ in a wealthy district. Counterfactual fairness asks: "If this cell were in a poor district instead (everything else being equal), would its service ratio change?"

- If $R^2_{\text{demo}} = 0$ (F_causal = 1): The residuals $R$ are independent of demographics. Counterfactually changing a cell's district affiliation would not change its expected residual. Service (after demand adjustment) is counterfactually fair.

- If $R^2_{\text{demo}} > 0$ (F_causal < 1): Demographics predict residuals. A wealthy cell's positive residual would become smaller (or negative) if it were counterfactually placed in a poor district. Service is counterfactually unfair.

Option B's F_causal = 1 − R²_demo directly operationalizes counterfactual fairness at the **aggregate level**: it measures the degree to which the service distribution would change under counterfactual demographic reassignment. Maximizing F_causal drives the system toward a state where such counterfactual changes would have no effect.

### 7.3 Controlled Direct Effect (CDE)

**Definition** (Pearl, 2001): The Controlled Direct Effect of treatment $X$ on outcome $Y$, controlling for mediator $Z$, is:

$$\text{CDE}(z) = E[Y \mid do(X = x_1), Z = z] - E[Y \mid do(X = x_0), Z = z]$$

This measures the effect of changing $X$ from $x_0$ to $x_1$ while **holding the mediator $Z$ constant** at value $z$.

**Connection to Option B**: In FAMAIL:
- $X$ = demographic features (treatment / sensitive attribute)
- $Y$ = service ratio (outcome)
- $Z = D$ = demand (mediator, controlled for via $g_0(D)$)

The CDE asks: "If we change a cell's demographics from low-income to high-income while holding demand constant, how much does the expected service ratio change?"

$$\text{CDE}(d) = E[Y \mid \text{high income}, D = d] - E[Y \mid \text{low income}, D = d]$$

The residuals $R = Y - g_0(D)$ are precisely the demand-controlled service values. By regressing $R$ on demographics $X$, Option B estimates the CDE structure across all cells:

$$\hat{R}_c = \hat{\beta}'x_c = \text{predicted service deviation based on demographics}$$

The regression coefficient $\hat{\beta}_j$ for demographic feature $j$ estimates how much a one-unit increase in that feature changes the expected service ratio, **holding demand constant**. The R²_demo measures how much of the service variation is attributable to the CDE.

The action item from Meeting 23 — *"Need to add Y terms (Y1, Y2, Y3) to correctly calculate Controlled Direct Effect for evaluation purposes"* — is addressed here: the Y terms correspond to:
- **Y1** = $E[Y]$: the overall expected service ratio (the unconditional mean)
- **Y2** = $E[Y \mid D = d]$ = $g_0(D)$: the demand-conditional expected service (what g₀ estimates)
- **Y3** = $E[Y \mid D = d, X = x]$ = $g_0(D) + \hat{\beta}'x$: the demand-and-demographic-conditional expected service

The CDE for a one-unit change in demographic feature $j$ is then:

$$\text{CDE}_j = Y_3(x_j + 1) - Y_3(x_j) = \hat{\beta}_j$$

And the total demographic influence is captured by R²_demo, which measures how much the Y3 predictions differ from the Y2 predictions across cells.

### 7.4 Mediation Analysis and Direct vs. Indirect Effects

**Framework** (Baron & Kenny, 1986; Pearl, 2001): In mediation analysis, the total effect of $X$ on $Y$ is decomposed into:

- **Indirect effect** (mediated through $Z$): $X \to D \to Y$
- **Direct effect** (not through $Z$): $X \to Y$ (controlling for $D$)

$$\text{Total Effect} = \text{Direct Effect} + \text{Indirect Effect}$$

Option B isolates the **direct effect** of demographics on service. By conditioning on demand through $g_0(D)$, we partial out the indirect path and measure only the direct path from demographics to service.

In the FAMAIL context:
- **Indirect effect** (acceptable): Wealthy areas generate more demand → higher demand may lead to different service ratios. This is a market mechanism and is not inherently unfair.
- **Direct effect** (potentially unfair): Wealthy areas receive better service *even at the same demand level*. This could reflect driver preference for affluent areas, better road infrastructure enabling more efficient service, or other forms of structural advantage.

Option B's F_causal = 1 − R²_demo measures the strength of the direct effect. The optimization target is to **eliminate the direct path** while preserving the indirect (demand-mediated) path.

### 7.5 Partial R² and the Frisch-Waugh-Lovell Theorem

**Partial R²** (also called the coefficient of partial determination) measures the incremental explanatory power of a set of variables after controlling for other variables. Option B's R²_demo is precisely the partial R² of demographics controlling for demand.

The **Frisch-Waugh-Lovell (FWL) theorem** provides a formal equivalence: regressing $Y$ on $[D, X]$ and extracting the partial R² of $X$ is equivalent to:
1. Regressing $Y$ on $D$ to get residuals $R = Y - \hat{Y}_{D}$
2. Regressing $R$ on $X$ to get R²_demo

This is **exactly** the two-stage procedure in Option B. The FWL theorem guarantees that our two-stage approach gives the **same result** as a single regression of $Y$ on both demand and demographics — as long as the demand model is linear in parameters (which is true for the power basis).

This equivalence provides important validation: the Option B procedure is not an approximation or heuristic. It is **mathematically equivalent** to the gold-standard multiple regression approach for partial R².

### 7.6 Fairness Through Unawareness vs. Fairness Through Awareness

Option B connects to two contrasting fairness paradigms:

- **Fairness through unawareness**: Simply ignore sensitive attributes in the model. This is the baseline approach — g₀(D) uses only demand, ignoring demographics entirely. The problem: ignoring demographics doesn't mean demographics aren't influencing outcomes.

- **Fairness through awareness** (Dwork et al., 2012): Explicitly account for sensitive attributes to identify and correct disparities. This is Option B's approach — it explicitly measures demographic influence on residuals and optimizes to eliminate it.

Option B is a **fairness through awareness** method: it uses demographic information not to make predictions, but to *audit* the service distribution and guide corrections. The demographic features appear only in the hat matrix (the auditing tool), never in the trajectory modification directly.

### 7.7 Relationship to Statistical Parity and Conditional Statistical Parity

- **Statistical parity**: $P(Y > t \mid X = x_1) = P(Y > t \mid X = x_2)$ for all demographic groups. This is a strong requirement that ignores legitimate differences in demand.

- **Conditional statistical parity** (controlling for a legitimate factor $Z$): $P(Y > t \mid X = x_1, Z = z) = P(Y > t \mid X = x_2, Z = z)$. This allows differences driven by the legitimate factor (demand) while requiring equality across demographic groups at the same demand level.

Option B approximates **conditional statistical parity** at the mean level: it tests whether $E[Y \mid X, D] = E[Y \mid D]$ — whether the expected service ratio depends on demographics after controlling for demand. F_causal = 1 corresponds to exact conditional statistical parity (at the mean).

---

## 8. Why Option B is Appropriate for FAMAIL

### 8.1 Comparison with Alternative Formulations

Three F_causal formulations were implemented and analyzed. Option B was chosen as the default for production optimization:

| Criterion | Baseline | Option B | Option C |
|-----------|----------|----------|----------|
| **Question** | Does supply follow demand? | After demand, do demographics predict service? | Do demographics add predictive power beyond demand? |
| **Causal validity** | Weak (conflates demand and demographics) | **Strong** (isolates demographic direct effect) | Strong (same signal as B) |
| **Gradient** | Partially degenerate (Var(Y) inflation risk) | **Clean, directionally correct** | Clean (same direction as B) |
| **Hat matrix constant during optimization?** | N/A | **Yes** (H depends only on demographics) | **No** (H_red depends on demand features φ(D), which change) |
| **Re-fits β each step?** | N/A (no β) | **Yes** (implicitly via H) | Yes (implicitly) |
| **Interpretation** | Demand alignment | **Demographic independence of residuals** | Marginal demographic contribution |
| **Computational cost** | O(N) | O(N²) matrix-vector | O(N²) × 2 matrix-vector products |

### 8.2 Why Not Baseline?

The baseline is kept for backward compatibility but has fundamental limitations:
1. It doesn't measure demographic influence — only demand alignment
2. Its gradient has a known "degenerate" component that can incentivize increasing Var(Y) rather than improving fairness
3. It was identified as a misnomer for "causal" fairness

### 8.3 Why Not Option C?

Option C (partial ΔR²) measures a similar concept to Option B but with a practical limitation: its hat matrices $H_{\text{red}}$ and $H_{\text{full}}$ contain demand features φ(D), and **demand changes during optimization** as pickups move between cells. This means Option C's projection matrices are no longer constant, breaking the clean pre-computation story.

Option B avoids this because its hat matrix depends **only on demographics** (truly fixed). The demand effect is removed separately through g₀(D). This clean separation of concerns is a key architectural advantage.

### 8.4 Gradient Quality

Option B's gradient pushes the optimizer in exactly the right direction:

**Over-served wealthy cell** (positive residual $R_c > \bar{R}$, demographics predict the over-service):
- The gradient $\partial F / \partial R_c < 0$ → Push $R_c$ down → Reduce service to this cell

**Under-served poor cell** (negative residual $R_c < \bar{R}$, demographics predict the under-service):
- The gradient $\partial F / \partial R_c > 0$ → Push $R_c$ up → Increase service to this cell

**Anomalous cell** (deviation not explained by demographics):
- The $(I-H)R$ term dominates → Gradient is small → Optimizer leaves it alone

This is exactly the redistribution pattern needed: move service from demographically-advantaged cells toward demographically-disadvantaged cells, while preserving variation that isn't related to demographics.

### 8.5 Compatibility with FAMAIL Architecture

Option B integrates cleanly with the existing FAMAIL pipeline:

1. **Soft cell assignment** makes $S_c$ and $D_c$ differentiable → $Y_c = S_c/D_c$ carries gradients
2. **Frozen g₀(D)** provides a constant demand baseline → $R_c = Y_c - g_0(D_c)$ is differentiable through $Y_c$
3. **Constant hat matrix** means only $R$ carries gradients in $F = R'(I-H)R / R'MR$
4. **Gradient flows** end-to-end: $F_{\text{causal}} \to R \to Y \to S/D \to \text{soft cell counts} \to \text{pickup positions}$

---

## 9. Controlled Direct Effect (CDE) Evaluation

This section directly addresses the Meeting 23 action item: *"Need to add Y terms (Y1, Y2, Y3) to correctly calculate Controlled Direct Effect for evaluation."*

### 9.1 Y Terms for CDE Calculation

The three Y terms represent nested models of increasing complexity:

| Term | Definition | Model | What It Captures |
|------|-----------|-------|-----------------|
| $Y_1$ | $\bar{Y}$ | Intercept only | Overall average service ratio |
| $Y_2$ | $g_0(D_c)$ | Demand model | Expected service given demand |
| $Y_3$ | $g_0(D_c) + \hat{\beta}'x_c$ | Demand + demographics | Expected service given demand AND demographics |

### 9.2 Computing CDE from These Terms

The CDE for a unit increase in demographic feature $j$ is:

$$\text{CDE}_j = \hat{\beta}_j = [(X'X)^{-1}X'R]_j$$

This is the $j$-th regression coefficient from regressing residuals on demographics.

For the full CDE between two demographic profiles (e.g., highest-income vs. lowest-income district):

$$\text{CDE}_{\text{total}} = Y_3(\text{high income}) - Y_3(\text{low income}) = \hat{\beta}'(x_{\text{high}} - x_{\text{low}})$$

### 9.3 Beyond Unit Change: Full Causal Validation

Meeting 23 noted that the current implementation *"only captures derivative/unit change; full causal validation requires modeling how demographic changes affect outcomes."*

Option B's hat matrix approach addresses this concern more fully than a simple derivative:

1. **The regression coefficients $\hat{\beta}$** give the CDE per unit change in each demographic feature
2. **The R²_demo** gives the fraction of service variation attributable to demographics — this is a *global* measure of the CDE's aggregate strength
3. **The vector $(I-H)R$** decomposes each cell's residual into fair (orthogonal to demographics) and unfair (along demographics) components
4. **Pre- and post-optimization comparison** of these quantities provides full CDE evaluation:
   - Before modification: $R^2_{\text{demo,before}}$, $\hat{\beta}_{\text{before}}$
   - After modification: $R^2_{\text{demo,after}}$, $\hat{\beta}_{\text{after}}$
   - Success metric: $R^2_{\text{demo,after}} < R^2_{\text{demo,before}}$ (demographics explain less after editing)

---

## 10. Implementation Summary

### 10.1 Pre-computation (Once, During Data Loading)

```
1. Load demographic data for each grid cell (from cell_demographics.pkl)
2. Identify active cells: demand ≥ threshold AND valid demographics AND no NaN
3. Extract demographic features X for active cells
4. Standardize X (zero mean, unit variance) — preserves relative relationships
5. Add intercept column: X_design = [1 | X_standardized]
6. Compute hat matrix: H = X_design @ pinv(X_design)
7. Store (I-H) and M = I - 11'/N as constant matrices
8. Fit g₀(D) using power basis on active cells → frozen prediction function
```

### 10.2 During Optimization (Every ST-iFGSM Step)

```
1. Compute soft cell counts → S_c, D_c per cell (differentiable)
2. Compute Y_c = S_c / D_c (differentiable)
3. Extract Y for active cells
4. Compute R = Y - g₀(D) using frozen power basis lookup (differentiable through Y)
5. Compute F_causal = R'(I-H)R / R'MR (differentiable — quadratic forms in R)
6. Backpropagate: ∂F/∂R → ∂R/∂Y → ∂Y/∂S → ∂S/∂positions
```

### 10.3 Key Files

| File | Purpose |
|------|---------|
| `objective_function/causal_fairness/utils.py` | Core math: `build_hat_matrix()`, `compute_fcausal_option_b_torch()`, `precompute_hat_matrices()` |
| `objective_function/causal_fairness/config.py` | `CausalFairnessConfig` with `formulation="option_b"` |
| `trajectory_modification/objective.py` | `FAMAILObjective._compute_option_b_causal()` — production optimization |
| `trajectory_modification/data_loader.py` | `DataBundle` with `hat_matrices`, `g0_power_basis_func`, `active_cell_indices` |
| `trajectory_modification/dashboard.py` | Formulation selector in sidebar (baseline / option_b / option_c) |

### 10.4 Demographic Features Used

| Feature | Description | Source |
|---------|-------------|--------|
| AvgHousingPricePerSqM | Average housing price per m² | District statistics |
| GDPperCapita | GDP per capita (derived: GDP / population) | Computed |
| CompPerCapita | Employee compensation per capita (derived) | Computed |

These three features were selected through exploratory analysis in the demographic explorer dashboard as the most informative proxies for neighborhood socioeconomic status. The feature set is configurable via `CausalFairnessConfig.demographic_features`.

---

## 11. Addressing Open Questions from Project Meetings

### 11.1 "Do we still need to model G(d) despite the hat matrix approach?" (Meeting 24)

**Yes, but the hat matrix dramatically simplifies its role.** The hat matrix replaces the need for a separate *demographic* model (no neural network or random forest needed). But $g_0(D)$ — the *demand-only* model — is still needed to produce the residuals that the hat matrix operates on.

The good news: the power basis g₀(D) is simple, well-understood, and performs within 0.0003 R² of the best non-parametric estimator. There is no model selection problem for g₀ — it's a solved problem.

### 11.2 "Both G(d) and G(d,x) functions can be represented in hat matrix form" (Meeting 24)

**Correct.** If g₀(D) uses the power basis (linear in parameters), then:
- $g_0(D) = X_D \hat{\beta}_D$ where $X_D$ is the power basis design matrix → $H_D = X_D(X_D'X_D)^{-1}X_D'$ is the demand hat matrix
- $g(D, x) = X_{\text{full}} \hat{\beta}_{\text{full}}$ where $X_{\text{full}} = [X_D \mid X_{\text{demo}}]$ → $H_{\text{full}}$ is the combined hat matrix

This is the basis for Option C's formulation. For Option B, only $H_{\text{demo}}$ (the demographic hat matrix operating on residuals) is needed.

### 11.3 "Hat matrix math needs mathematical validation" (Meeting 24)

The matrix algebra walkthrough in Section 5 provides this validation. The key mathematical facts that underpin Option B:

1. **$H$ is idempotent and symmetric**: Follows from $H = X(X'X)^{-1}X'$
2. **The Pythagorean decomposition is exact**: Because $X$ includes an intercept, $\hat{R}'e = 0$
3. **R² equals the quadratic form ratio**: Standard OLS result, no approximation
4. **FWL equivalence**: Two-stage procedure gives same R² as single regression (proven in `FCAUSAL_FORMULATIONS.md`, Appendix A)
5. **Gradient is correct**: Verified numerically with synthetic data — over-served wealthy cells get negative gradient, under-served poor cells get positive gradient
6. **NumPy and PyTorch implementations agree**: Tested to within machine precision (difference < 1e-8)

### 11.4 "Neural network recommended for G function" (Meeting 24)

**This recommendation has been superseded** by the hat matrix approach. The neural network was recommended for fitting a g(D,x) model when the plan was to directly model the demand-demographic-service relationship. With Option B's two-stage approach:
- Stage 1 (g₀) uses the power basis — no neural network needed
- Stage 2 (demographic regression) uses the hat matrix — an analytical OLS solution, no neural network needed

The hat matrix is strictly superior to a neural network for this purpose because it:
- Gives the exact OLS solution (no training, no convergence issues)
- Is fully differentiable by construction
- Never overfits (it's linear regression with 3-4 features on ~400 cells)
- Implicitly re-fits at every optimization step

### 11.5 "May need to adjust objective weights due to different scale of new causal component" (Meeting 23)

**Confirmed.** The baseline F_causal typically ranges 0.40–0.50 (demand R²), while Option B's F_causal ranges 0.95–0.99 (demographic R² of residuals). The scales are different because they measure different things. Displaying percentages and normalizing term scales (Meeting 24 action item) will help compare contributions of the three objective terms.

### 11.6 "Why 'editing' and not 'generation'?" (Meeting 24 — Paper Contribution Context)

The Option B formulation strengthens the case for trajectory *editing* over *generation* because:
1. **Attribution is meaningful**: LIS and DCD scores identify specific trajectories contributing to demographic unfairness, enabling targeted minimal edits
2. **The hat matrix provides interpretable gradients**: Each cell's gradient tells us exactly how much to adjust service based on its demographic profile — this fine-grained control is possible with editing, not with wholesale generation
3. **Fidelity preservation**: Editing preserves the driver's original behavior except for the fairness correction; generation must learn behavior from scratch

---

## 12. References and Related Literature

### 12.1 Causal Inference Foundations
- **Pearl, J.** (2001). *Direct and Indirect Effects.* Proceedings of UAI. — Defines Controlled Direct Effect (CDE) and Natural Direct Effect (NDE).
- **Pearl, J.** (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. — Comprehensive framework for causal reasoning with DAGs.

### 12.2 Algorithmic Fairness
- **Kusner, M. J., Loftus, J., Russell, C., & Silva, R.** (2017). *Counterfactual Fairness.* NeurIPS. — Defines counterfactual fairness using structural causal models.
- **Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R.** (2012). *Fairness Through Awareness.* ITCS. — Proposes fairness through awareness of sensitive attributes.
- **Kilbertus, N., Carulla, M. R., Parascandolo, G., Hardt, M., Janzing, D., & Scholkopf, B.** (2017). *Avoiding Discrimination through Causal Reasoning.* NeurIPS. — Connects causal reasoning to algorithmic fairness.

### 12.3 Statistical Methods
- **Frisch, R. & Waugh, F. V.** (1933). *Partial Time Regressions as Compared with Individual Trends.* Econometrica. — Establishes the FWL theorem for partial regression.
- **Baron, R. M. & Kenny, D. A.** (1986). *The Moderator-Mediator Variable Distinction.* Journal of Personality and Social Psychology. — Foundation for mediation analysis.

### 12.4 FAMAIL Project Documents
- `FCAUSAL_GDX_JOINT_ANALYSIS.md` — Comprehensive analysis of F_causal formulation options A1, A2, B, C, and D
- `G0_DEMAND_BASIS_ANALYSIS.md` — Analysis of g₀(D) basis functions and hat matrix compatibility
- `FCAUSAL_FORMULATIONS.md` — Implementation design reference for all three formulations
- `G_FUNCTION_REFORMULATION_PLAN.md` — Original reformulation plan from Verification Meeting
