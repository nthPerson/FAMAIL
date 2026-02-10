# Demographic Explorer: Usage Guide

## Overview

The Demographic Explorer dashboard helps you find the best g(D, x) estimator for the causal fairness term in FAMAIL's multi-objective function. The current g(D) model only uses demand to predict expected service ratios. Extending it with demographic features x — creating g(D, x) — can capture whether service patterns correlate with demographic characteristics like income, population density, or migrant ratios.

**Why this matters**: If service ratios systematically differ across demographic groups even after controlling for demand, the fairness term should account for this. A good g(D, x) model separates the demand-driven component of service from the demographic-driven component, enabling more targeted fairness corrections.

**Key constraint**: Shenzhen's demographic data is at the **district level** — only **10 unique profiles** across ~28,000 cell-period observations. This limits model complexity and makes rigorous cross-validation essential.

```
streamlit run objective_function/causal_fairness/demographic_explorer.py
```

---

## Recommended Workflow

### Step 1: Explore Features (Tab 5: Feature Engineering)

Start here to understand the available demographic features and identify redundancy.

1. Review the **Feature Summary Table** — note which features are raw (13) vs derived (7).
2. Check the **Correlation Matrix** — look for pairwise |r| > 0.85, which indicates redundancy. Including highly correlated features inflates variance and destabilizes coefficients.
3. Look at **Feature-Residual Scatters** — if a feature has no clear relationship with residuals across districts, it may not be useful.

**Action**: Uncheck features in the sidebar that are redundant, irrelevant, or too noisy.

### Step 2: Explore Spatial Patterns (Tab 1: Spatial Maps)

Compare demographic maps to service metrics to build intuition.

1. Set the left map to a demographic variable (e.g., GDPperCapita) and the right map to MeanResidual.
2. Look for **spatial alignment** — do under-served areas (negative residuals, shown in red) overlap with low-income areas?
3. Check the Pearson correlation at the bottom — this gives a quick quantitative answer.

**What to look for**: If demographic maps and residual maps show clear spatial overlap, demographics may be predictive. If they look unrelated, the signal is weak.

### Step 3: Review District Patterns (Tab 2: District Summaries)

Check whether districts actually differ in service outcomes.

1. The **Service Ratio vs Demand** scatter should show whether districts cluster differently above/below the g(D) line.
2. **Mean Residual by District** shows which districts are systematically under/over-served.
3. **Residual Box Plots** show the spread — if all districts have similar distributions centered near zero, demographics may not add much.

**What to look for**: Districts that consistently sit above or below zero in the residual chart. These are the districts where g(D) alone fails to capture service patterns.

### Step 4: Fit and Compare Models (Tab 3: Model Comparison)

This is the core step. Test multiple model architectures with the same feature set.

1. Select models to fit (default: OLS, Ridge, Lasso).
2. Click "Fit Models" — this runs both full-data fitting and LODO cross-validation.
3. Sort the comparison table by **LODO R²** (primary metric).
4. Check the **Overfit Gap** column — a gap > 0.1 signals overfitting.

**What to look for**: The model with the highest LODO R² that also has a reasonable overfit gap (< 0.05).

### Step 5: Validate (Tabs 4 & 6)

Deep-dive into your best model.

1. **Tab 4 (Diagnostics)**: Check coefficient p-values (significant features should have p < 0.05), VIF (all features should be < 10), and the Q-Q plot (residuals should approximately follow the diagonal).
2. **Tab 6 (CV Detail)**: Check per-district LODO R² — are any districts dramatically worse? The OOF residual spatial map shows where the model systematically fails.

---

## What Good Results Look Like

| Metric | Ideal | Concerning | Interpretation |
|--------|-------|------------|----------------|
| **LODO R²** | > 0.3 | < 0.1 | Model generalizes to unseen districts |
| **Train R² - LODO R²** | < 0.05 | > 0.15 | Large gap = overfitting |
| **VIF (all features)** | < 5 | > 10 | High VIF = multicollinearity, remove features |
| **Coefficient p-values** | < 0.05 | > 0.1 | Non-significant features add noise |
| **Breusch-Pagan p** | > 0.05 | < 0.01 | Low p = heteroscedasticity (unequal variance) |
| **Durbin-Watson** | ~2.0 | < 1.5 or > 2.5 | Departure from 2 = autocorrelation |
| **AIC/BIC** | Lower is better | — | Compare across models on the same data |
| **Permutation importance** | Few features dominate | All features ~equal | Clear winners = robust signal |

### Interpreting Low LODO R²

A LODO R² near 0 doesn't necessarily mean demographics are irrelevant — it means the model **can't generalize to an unseen district**. With only 10 districts, each district has unique characteristics that are hard to extrapolate from the other 9. This is an inherent limitation of district-level data.

If LODO R² is low but Train R² is moderate (~0.03-0.10), the model captures real within-district patterns but can't predict across districts. This is still useful information — it means a different modeling approach or finer-grained demographic data may be needed.

---

## Feature Selection Guidance

### Recommended Starting Features

1. **GDPperCapita** (derived) — GDP normalized by population. More interpretable than raw GDP.
2. **AvgHousingPricePerSqM** (raw) — Direct proxy for area wealth. High variance across districts.
3. **CompPerCapita** (derived) — Employee compensation per worker. Captures income levels.
4. **MigrantRatio** (derived) — Non-registered population fraction. Captures socioeconomic dynamics.

### Features to Avoid

- **AreaKm2** — Geographic constant, not a demographic predictor. Correlates with number of cells per district, creating mechanical bias.
- **Raw population counts** (YearEndPermanentPop10k, etc.) — Confounded by district size. Use per-capita versions instead.
- **Log transform + raw version together** — Creates perfect multicollinearity. Choose one or the other.

### Rule of Thumb

With 10 districts, use **3-5 features maximum**. More features risk overfitting — the model has fewer "degrees of freedom" to distinguish signal from noise. If Lasso zeroes out a feature, that's a strong signal to remove it.

---

## Model Selection Guidance

| Model | Best For | Watch Out For |
|-------|----------|---------------|
| **Ridge** | Default choice. Handles collinearity well. | May retain irrelevant features (doesn't zero them out). |
| **Lasso** | Automatic feature selection. Use when unsure which features matter. | Can be unstable with collinear features (may arbitrarily pick one). |
| **ElasticNet** | Compromise between Ridge and Lasso. | Requires tuning both alpha and L1 ratio. |
| **OLS** | Maximum interpretability (statsmodels gives p-values). | Sensitive to collinearity; standard errors inflate. |
| **OLS + Interactions** | Testing if demographic effects change with demand level. | Doubles the number of parameters; overfitting risk. |
| **Random Forest** | Detecting non-linear patterns. | Expects overfitting with 10 districts. Use for insight, not production. |
| **Gradient Boosting** | Similar to RF, often better for tabular data. | Same overfitting concern as RF. |

### Tuning Alpha (Regularization)

- **Higher alpha** → stronger regularization → simpler model → lower train R² but potentially better LODO R²
- **Lower alpha** → weaker regularization → closer to OLS → higher train R² but potential overfitting
- Use the alpha slider and re-run "Fit Models" to find the sweet spot where LODO R² is maximized

---

## Interpreting Fairness Implications (Tab 6)

Tab 6's fairness section shows how your g(D, x) model choice affects the causal fairness score.

- **If g(D, x) LODO R² >> g(D) R²**: Demographics meaningfully predict service beyond demand. This supports using Options A1/B/C in the causal fairness term — the extended model captures real demographic bias.
- **If g(D, x) LODO R² ≈ g(D) R²**: Demographics add little predictive power. The current demand-only model may be sufficient. The low Option A2 score (~0.26 in current data) likely reflects inherent noise, not missing demographic variables.
- **Per-district R² comparison**: Shows whether demographics help uniformly or only for specific districts. If only 1-2 districts improve, the signal may be idiosyncratic rather than systematic.

---

## Common Pitfalls

1. **Overfitting**: With only 10 unique demographic profiles, any model with >10 parameters can memorize the training data. **Always check LODO R²** — if it's much lower than Train R², the model is overfitting.

2. **Ecological fallacy**: District-level demographics describe the average, not individual cells. A high-GDP district may contain cells with very different local conditions. Treat results as suggestive, not definitive.

3. **Simpson's paradox**: An overall correlation between demographics and residuals may reverse within individual demand bins. The district scatter plot (Tab 2) can reveal this — look for districts that are under-served at high demand but over-served at low demand.

4. **Feature leakage**: Avoid features that mechanically relate to the grid structure. For example, AreaKm2 correlates with the number of valid cells per district, which affects how supply is distributed and aggregated.

5. **Multiple testing**: When comparing many models and feature combinations, some will look good by chance. Focus on LODO R² (which is inherently conservative) rather than train-set metrics.
