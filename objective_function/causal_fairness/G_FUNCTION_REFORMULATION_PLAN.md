# g(D) Function Reformulation Plan

**Demographically-Aware Causal Fairness Estimator**

*Created: 2026-02-06*
*Context: FAMAIL Formulation Verification Meeting (2026-02-06)*
*Author: FAMAIL Research Team*

---

## 1. Background & Motivation

### 1.1 Current g(D) Formulation

The current causal fairness term uses:

$$F_{\text{causal}} = \max(0,\; R^2), \quad R^2 = 1 - \frac{\text{Var}(Y - g(D))}{\text{Var}(Y)}$$

where:
- $Y_c = S_c / D_c$ is the observed supply-to-demand ratio at cell $c$
- $g(D_c)$ is an isotonic regression fitted on observed data: $g(d) = \hat{\mathbb{E}}[Y \mid D = d]$
- $R^2$ measures how well demand alone explains variance in service ratios

**The function $g(d)$ takes only demand $D$ as input and returns the expected service ratio.**

### 1.2 Problem Identified in Meeting

Three critical issues were identified during the Formulation Verification Meeting:

1. **Circular fairness baseline**: The current $g(d)$ is estimated *from* the observed (unfair) data. It learns the existing demand-supply relationship, which already encodes discriminatory patterns. It cannot represent a "perfectly fair" baseline because it was trained on unfair outcomes.

2. **Demand-only model is a misnomer for "causal" fairness**: The current formulation measures *model fit error* (how well demand predicts supply), not whether *sensitive attributes* (income, demographics) are driving the supply distribution. This is unexplained-inequality, not causal fairness.

3. **No mechanism to detect demographic bias**: If two areas have the same demand but different income levels, the current $g(d)$ predicts the same service ratio for both. It has no way to detect or penalize the fact that low-income areas may systematically receive less service than high-income areas at the same demand level.

### 1.3 Proposed Direction

Revise $g$ to be a **conditional model** that includes demographic features:

$$g(\text{Supply} \mid \text{Demand}, \text{Income/Demographics})$$

The key insight: after trajectory modification, the difference in predicted service ratio between low-income and high-income areas *at the same demand level* should be minimized. The causal fairness term should measure — and the optimization should reduce — the influence of demographic factors on service distribution.

---

## 2. Available Data Inventory

### 2.1 Demographic Data

**Source**: `data/demographic_data/all_demographics_by_district.csv`

10 Shenzhen districts with the following features:

| Feature | Column Name | Description | Potential Relevance |
|---------|------------|-------------|-------------------|
| GDP | `GDPin10000Yuan` | Gross domestic product (10k yuan) | **Primary** — direct income proxy |
| Housing Price | `AvgHousingPricePerSqM` | Average housing price per m² | **Primary** — strong income/wealth indicator |
| Population Density | `PopDensityPerKm2` | Population per km² | **Secondary** — demand driver, controls for density |
| Total Population | `YearEndPermanentPop10k` | Year-end permanent population (10k) | Secondary — demand driver |
| Employee Compensation | `EmployeeCompensation100MYuan` | Total employee compensation (100M yuan) | **Primary** — direct income metric |
| Employment Count | `AvgEmployedPersons` | Average employed persons | Secondary — economic activity |
| Area | `AreaKm2` | District area in km² | Control variable |
| Registered Population | `RegisteredPermanentPop10k` | Registered permanent pop (10k) | Secondary — relates to urban stability |
| Non-Registered Population | `NonRegisteredPermanentPop10k` | Non-registered pop (10k) | Secondary — migrant worker population |

### 2.2 Grid-to-District Mapping

**Source**: `data/visualization/streamlit__LOCAL_BACKUP_20260112_142441/app_data/grid_to_district_ArcGIS_table.csv`

- Maps each grid cell (row, col) to its majority Shenzhen district
- Contains 4,500 entries covering the 48×90 grid (with 1-indexed row/col that maps to the 50×90 GIS grid)
- Includes overlap percentage for cells on district boundaries
- Some cells fall outside all districts (ocean/non-Shenzhen area) — these have empty district fields

**Note**: An updated, code-accessible version of this mapping should be created in the `data/geo_data/` or `source_data/` directory.

### 2.3 Demand & Supply Data

| Dataset | File | Granularity | Usage |
|---------|------|-------------|-------|
| Pickup/Dropoff Counts | `source_data/pickup_dropoff_counts.pkl` | Per cell × time bucket × day | Demand $D_c$ (mean hourly pickups) |
| Active Taxis | `source_data/active_taxis_5x5_hourly.pkl` | Per cell × hour × day | Supply $S_c$ (mean hourly active taxis) |
| Trajectories | `source_data/passenger_seeking_trajs_45-800.pkl` | Per trajectory (sequence of states) | Trajectories to modify |

### 2.4 Data Granularity Challenge

- **Grid cells**: 48 × 90 = 4,320 cells
- **Districts**: 10 (demographic data resolution)
- **Cells per district**: Approximately 100–400 cells per district
- **Consequence**: Demographic features are constant within a district. All cells in the same district get the same income/GDP/housing price values.
- **Implication**: Demographic features introduce variability only *between* districts (at district boundaries), not *within* districts. This is a limitation but still meaningful — it captures the macro-level income gradients across the city.

---

## 3. Reformulation Options

### 3.1 Notation

Let $\mathbf{x}_c$ denote the demographic feature vector for cell $c$ (inherited from the cell's majority district). Let $D_c$ denote demand at cell $c$, $S_c$ supply, and $Y_c = S_c / D_c$ the service ratio.

### 3.2 Option A: Conditional Regression Model — $g(D, \mathbf{x})$

**Formulation**:

$$g(D_c, \mathbf{x}_c) = \hat{\mathbb{E}}[Y \mid D = D_c, \mathbf{x} = \mathbf{x}_c]$$

Train a regression model that predicts the service ratio $Y$ from both demand $D$ and demographic features $\mathbf{x}$.

**New Causal Fairness Metric**: The loss function changes to measure how much the model's predictions *depend on* the demographic features, controlling for demand:

$$F_{\text{causal}} = 1 - \frac{\text{Var}_c\!\big[g(D_c, \mathbf{x}_c) - g(D_c, \bar{\mathbf{x}})\big]}{\text{Var}_c\!\big[Y_c\big]}$$

where $\bar{\mathbf{x}}$ is the mean demographic vector across all cells. This measures how much of the variation in predictions is attributable to demographics rather than demand.

**Alternatively (simpler)**: Keep R² but now residuals are computed against the demographically-aware model:

$$R_c = Y_c - g(D_c, \mathbf{x}_c)$$

Higher R² means the model (which includes demographics) better explains the observed service ratio — but this reverts to the same issue of fitting observed patterns.

**Pros**:
- Direct extension of the current framework
- The demographic coefficient(s) directly measure income sensitivity
- Multiple model choices available (see Section 4)

**Cons**:
- With only 10 distinct demographic profiles, the regression may overfit demographics
- Still somewhat circular: the model learns the *existing* (potentially unfair) relationship

**Recommended for**: Initial exploration and understanding the demand-demographics-supply relationship.

---

### 3.3 Option B: Demographic Disparity Score (Preferred Approach)

**Core Idea**: Instead of fitting g to predict Y from demand + demographics, measure the *disparity* in service ratio between demographic groups at similar demand levels, and use that disparity as the causal unfairness measure.

**Formulation**:

1. **Fit a demand-only model** $g_0(D)$ as before (isotonic regression on demand).
2. **Compute residuals** $R_c = Y_c - g_0(D_c)$ for each cell.
3. **Measure how residuals correlate with demographics**:

$$F_{\text{causal}} = 1 - \rho^2(R, \mathbf{x})$$

where $\rho^2(R, \mathbf{x})$ is the coefficient of determination of regressing residuals $R$ on demographic features $\mathbf{x}$.

**Interpretation**: After removing the effect of demand (via $g_0$), if residuals still correlate with income/demographics, that indicates demographic-driven unfairness. A perfect score ($F_{\text{causal}} = 1$) means residuals are independent of demographics — service deviations from demand-expected levels are *not* driven by income.

**Pros**:
- Cleanly separates the demand effect from the demographic effect
- Does not require g to "learn" fairness — it explicitly measures demographic dependency
- Directly operationalizes the meeting's goal: "after revision, the difference in service ratio between low-income and high-income areas (given same demand) should be minimized"
- Keeps $g_0(D)$ as before (isotonic regression), which is well understood
- R²-based, so fits naturally into the existing framework

**Cons**:
- Two-stage estimation (g₀ then demographic regression) may lose some efficiency
- The demographic regression may be noisy with only 10 distinct district profiles

**Recommended for**: Primary production implementation. This directly encodes what the team wants to measure.

---

### 3.4 Option C: Partial R² / Conditional Independence Test

**Formulation**: Directly test whether supply is conditionally independent of demographics given demand:

$$F_{\text{causal}} = 1 - \Delta R^2$$

where:
- $R^2_{\text{full}}$ = R² of model $Y \sim D + \mathbf{x}$ (demand + demographics)
- $R^2_{\text{reduced}}$ = R² of model $Y \sim D$ (demand only)
- $\Delta R^2 = R^2_{\text{full}} - R^2_{\text{reduced}}$ = incremental R² from demographics

**Interpretation**: $\Delta R^2$ measures the additional variance in service ratios explained by demographics *beyond* what demand already explains. If $\Delta R^2 = 0$, demographics add nothing → perfect causal fairness. If $\Delta R^2$ is large, demographics significantly influence supply beyond demand.

**Pros**:
- Well-grounded in statistical theory (partial correlation / Type II sums of squares)
- Clear interpretation: "how much additional explanatory power do demographics provide over demand alone?"
- Easy to implement: fit two linear models, compare R² values

**Cons**:
- Assumes linear relationships (can be mitigated with non-linear models)
- May be sensitive to the functional form chosen for the demand model

**Recommended for**: Strong alternative if Option B proves unstable. Also useful as a validation metric.

---

### 3.5 Option D: Group Fairness via Demand-Stratified Comparison

**Formulation**: Partition cells into demographic groups (e.g., high-income vs. low-income districts) and compare their mean service ratios at similar demand levels.

For demand bin $b$ and demographic group $k$:

$$\text{Disparity}(b) = \max_{k_1, k_2} \left| \bar{Y}_{b, k_1} - \bar{Y}_{b, k_2} \right|$$

$$F_{\text{causal}} = 1 - \frac{1}{|B|} \sum_{b \in B} \text{Disparity}(b)$$

**Pros**:
- Very interpretable: "at the same demand level, do high-income and low-income areas get different service?"
- Non-parametric — no assumptions about functional form
- Aligns perfectly with the meeting discussion

**Cons**:
- Binning loses granularity and requires sufficient data per bin × group
- Not naturally differentiable (would need kernel smoothing or soft binning)
- With 10 districts split into ~2-3 income groups and ~10 demand bins, each bin×group may have very few cells

**Recommended for**: Interpretable reporting and visualization, not as the primary optimization target.

---

## 4. Model Choices for g(D, x) Training

For options that require a regression model with demand + demographics:

### 4.1 Linear Regression

$$g(D, \mathbf{x}) = \beta_0 + \beta_1 D + \boldsymbol{\beta}_x^T \mathbf{x}$$

- **Advantages**: Interpretable coefficients ($\beta_x$ directly measures income sensitivity), fast to fit, well-understood
- **Disadvantages**: May not capture non-linear demand-supply relationship
- **Best for**: Option C (partial R²), initial exploration, interpretability

### 4.2 Polynomial/Interaction Model

$$g(D, \mathbf{x}) = \beta_0 + \beta_1 D + \beta_2 D^2 + \boldsymbol{\beta}_x^T \mathbf{x} + \boldsymbol{\gamma}^T (D \cdot \mathbf{x})$$

- **Advantages**: Captures interaction between demand and demographics (e.g., income matters more at high demand)
- **Disadvantages**: More parameters, risk of overfitting with small dataset
- **Best for**: Exploring demand-income interactions

### 4.3 Gradient Boosted Trees (XGBoost/LightGBM)

$$g(D, \mathbf{x}) = \text{GBT}(D, \mathbf{x})$$

- **Advantages**: Handles non-linearities automatically, robust, feature importance scores
- **Disadvantages**: Not differentiable (but g is frozen during optimization, so this is fine), potential overfitting with ~4,000 cells and 10 unique demographic profiles
- **Best for**: Best overall prediction accuracy for Options A and C

### 4.4 Isotonic Regression with Demographic Residual Decomposition

1. Fit $g_0(D)$ via isotonic regression (current approach)
2. Compute residuals $R_c = Y_c - g_0(D_c)$
3. Regress residuals on demographics: $R_c \approx \boldsymbol{\beta}^T \mathbf{x}_c$

- **Advantages**: Minimal change from current pipeline, clean separation of demand and demographic effects
- **Disadvantages**: Two-stage estimation
- **Best for**: **Option B (recommended approach)** — this is the natural implementation

### 4.5 Gaussian Process Regression

$$g(D, \mathbf{x}) \sim \mathcal{GP}(\mu, k)$$

- **Advantages**: Provides uncertainty estimates, flexible kernel can encode demand and demographic effects separately
- **Disadvantages**: Computational cost, complexity, potentially overkill for this problem
- **Best for**: If uncertainty quantification is needed for research paper

---

## 5. Recommended Training Features

Based on the demographic data available and the meeting discussion:

### 5.1 Primary Income Proxies (use 1–3 of these)

| Feature | Column | Rationale | Risk |
|---------|--------|-----------|------|
| GDP per capita | `GDPin10000Yuan / YearEndPermanentPop10k` | Most direct income measure | Only 10 distinct values |
| Average Housing Price | `AvgHousingPricePerSqM` | Strong income correlate, captures wealth | Only 6 distinct values (some districts share) |
| Employee Compensation per capita | `EmployeeCompensation100MYuan / AvgEmployedPersons` | Wage-level proxy | Only 10 distinct values |

### 5.2 Secondary Features (consider for enrichment)

| Feature | Column | Rationale |
|---------|--------|-----------|
| Population Density | `PopDensityPerKm2` | Controls for urban density (demand driver) |
| Non-Registered Population Ratio | `NonRegisteredPermanentPop10k / YearEndPermanentPop10k` | Migrant worker proxy |
| Employment Count | `AvgEmployedPersons` | Economic activity scale |

### 5.3 Feature Engineering

- **Per-capita normalization**: GDP and compensation should be divided by population to get per-capita measures
- **Log transformation**: Income-related features are typically right-skewed; log transform may normalize distributions
- **Standardization**: All features should be z-score standardized before modeling
- **Interaction term**: $D \times \text{income}$ captures whether income's effect varies with demand level
- **District indicator**: One-hot encoding of district (alternative to continuous features, but uses all 10 degrees of freedom)

---

## 6. Implementation Roadmap

### Phase 1: Data Pipeline (Prerequisite)

1. **Create clean grid-to-district mapping**:
   - Process the ArcGIS grid-to-district table into a clean 48×90 lookup
   - Handle boundary cells (use majority district)
   - Handle cells outside Shenzhen (assign no demographic data, or nearest district)
   - Output: `source_data/grid_to_district_mapping.pkl` — Dict mapping `(x, y)` → `district_name`

2. **Create cell-level demographic feature matrix**:
   - Join grid-to-district mapping with demographic data
   - Compute per-capita derived features
   - Output: `source_data/cell_demographics.pkl` — Dict mapping `(x, y)` → feature vector, or a numpy array of shape `(48, 90, n_features)`

3. **Explore data granularity**:
   - Search for TAZ (Traffic Analysis Zone) level data for Shenzhen
   - If found, this would dramatically improve demographic resolution

### Phase 2: Analysis & Model Selection

4. **Exploratory analysis**:
   - Visualize Y vs. D colored by district/income group
   - Compute correlation between residuals ($Y - g_0(D)$) and demographic features
   - Assess whether demographic bias is statistically detectable at district-level granularity

5. **Model comparison**:
   - Implement Options B, C, and D
   - Compare metrics: partial R², demographic disparity score, group fairness gap
   - Select the formulation that best captures the relationship and is most defensible

### Phase 3: Integration

6. **Implement chosen g(D, x) estimator**:
   - Add new estimation function to `objective_function/causal_fairness/utils.py`
   - Create a new config parameter for demographic feature path
   - Ensure the estimator returns a callable compatible with the existing interface

7. **Update `FAMAILObjective.compute_causal_fairness()`**:
   - Accept demographic feature grid as additional parameter
   - Implement the new F_causal formulation
   - Maintain backward compatibility (old formulation available for ablation)

8. **Update `DataBundle`**:
   - Add demographic data loading to `data_loader.py`
   - Include `cell_demographics` in the DataBundle dataclass

### Phase 4: Validation

9. **Ablation study**:
   - Compare original g(D) formulation vs. new g(D, x) formulation
   - Measure: F_causal score, demographic disparity before/after modification
   - Keep results from current implementation as baseline

10. **Visualization updates**:
    - Add demographic-colored heatmaps to dashboard
    - Show service ratio by income group
    - Implement the heatmap improvements discussed in the meeting (gray for no-data areas, fixed color scale)

---

## 7. Differentiability Considerations

The new g(D, x) function, like the current g(D), will be **frozen** during optimization. Gradients do not flow through g — they flow only through $Y_c = S_c / D_c$.

For **Option B** (recommended):
- $g_0(D)$ remains frozen (isotonic regression, no gradient)
- The demographic regression on residuals is also pre-computed
- $F_{\text{causal}}$ measures correlation between $R_c = Y_c - g_0(D_c)$ and demographics $\mathbf{x}_c$
- Since $\mathbf{x}_c$ is fixed (demographics don't change) and $g_0(D_c)$ is frozen, gradients flow through $Y_c$ → $S_c / D_c$ → pickup counts (via soft cell assignment), exactly as before

**No changes to the gradient flow architecture are needed.** The new formulation changes *what* is measured, not *how* gradients propagate.

---

## 8. Addressing the Data Granularity Limitation

### 8.1 With District-Level Data (Current)

- 10 districts, ~100-400 cells each
- Demographic features are constant within each district
- Effective sample size for demographic analysis: 10 data points (district means)
- **Mitigation**: Use cell-level demand/supply data within each district to increase statistical power. Each district has hundreds of cells with varying demand, providing within-district variation in D and Y even though demographics x are constant.

### 8.2 With TAZ-Level Data (If Available)

- Traffic Analysis Zones are typically smaller than districts (50-200 zones per city)
- Would provide 5-20x more demographic resolution
- **Action item**: Search Chinese government statistical databases (National Bureau of Statistics, Shenzhen Statistical Bureau) for sub-district demographic data

### 8.3 Synthetic Dasymetric Disaggregation (Fallback)

If TAZ data is unavailable, consider disaggregating district-level demographics to cell level using:
- Land use data (residential, commercial, industrial zones)
- POI (Point of Interest) density
- Building footprint data
- Housing price variation within districts (from real estate platforms)

This is a significant research effort but could substantially improve resolution.

---

## 9. Summary of Recommendations

| Aspect | Recommendation |
|--------|---------------|
| **Primary formulation** | Option B: Demographic Disparity Score |
| **g₀(D) model** | Keep isotonic regression (current) |
| **Demographic regression model** | Linear regression on residuals (simple, interpretable) |
| **Primary features** | GDP per capita, Average Housing Price, Employee Compensation per capita |
| **F_causal metric** | $1 - \rho^2(R, \mathbf{x})$ where R = residuals from $g_0(D)$ |
| **Validation approach** | Ablation study comparing old vs. new formulation |
| **Data priority** | Grid-to-district mapping pipeline → cell demographic matrix |
| **Granularity improvement** | Search for TAZ-level data; consider dasymetric methods |
| **Differentiability** | No changes needed — g remains frozen during optimization |

---

## 10. Open Questions

1. **Feature selection**: Should we use a single composite income index or multiple individual features? Multiple features capture different income dimensions (as discussed in the meeting) but increase complexity.

2. **Poisson distribution for soft cell counts**: The meeting raised the question of whether Poisson distributions might be more appropriate than Gaussian for modeling count data in soft cell assignment. This is orthogonal to the g(D) reformulation but should be investigated in parallel.

3. **Causality rigor**: Even with demographics in the model, we are measuring *associational* fairness (correlation between demographics and service), not *interventional* causal fairness. Should we rename the term (e.g., "Demographic Fairness" instead of "Causal Fairness") to be more precise?

4. **Threshold effects**: With only 10 districts, is there a risk the model picks up district-specific effects unrelated to income (e.g., geographic features, road network density)?

5. **Temporal variation**: Should demographic effects be analyzed across time periods (e.g., does income-based bias vary by time of day)?
