# F_causal Formulations: Design Reference

**Date**: 2026-02-14
**Context**: Implementation of three causal fairness formulations for the FAMAIL objective function
**Prerequisite Reading**: `FCAUSAL_GDX_JOINT_ANALYSIS.md`, `G0_DEMAND_BASIS_ANALYSIS.md`

---

## 1. Overview

The FAMAIL objective function includes a causal fairness term F_causal that measures whether taxi service distribution is driven by passenger demand rather than demographic factors (income, population composition, etc.). Three formulations are implemented:

| ID | Name | Formula | Default |
|----|------|---------|:-------:|
| **Baseline** | Historical R² | $R^2 = 1 - \text{Var}(R)/\text{Var}(Y)$ with isotonic $g_0(D)$ | |
| **Option B** | Demographic Residual Independence | $F = R'(I-H)R \;/\; R'MR$ via demographic hat matrix | Yes |
| **Option C** | Partial ΔR² | $F = 1 - (R^2_{\text{full}} - R^2_{\text{red}})$ via dual hat matrices | |

All three share the same semantic: **F_causal ∈ [0, 1], higher = fairer** (service driven by demand, not demographics).

---

## 2. Why Three Formulations?

### 2.1 Why We Moved Beyond the Baseline

The historical (baseline) formulation measures whether service ratios Y = S/D follow the expected demand–ratio curve g₀(D). It answers: *"Does supply track demand?"* This is useful but incomplete — it cannot distinguish between:

- Service variation caused by **demand patterns** (acceptable)
- Service variation caused by **demographic factors** (potentially unfair)

For example, if wealthy neighborhoods have high demand AND high service ratios, the baseline R² is high (demand explains service), but the *reason* wealthy neighborhoods have high demand may itself be a fairness concern. The baseline conflates demand alignment with demographic fairness.

### 2.2 Why We Abandoned g(D, x) Model Fitting

The original plan was to fit a conditional model g(D, x) that predicts service ratios from both demand and demographics, then use the model's performance metrics to assess fairness. This approach had several problems:

1. **Circular dependency**: The "right" g(D, x) model depends on which F_causal formulation uses it, and vice versa.
2. **LODO R² is the wrong metric**: Leave-One-District-Out cross-validation measures predictive accuracy, not whether the demographic signal represents unfairness.
3. **Model selection complexity**: OLS, Ridge, Random Forest, Neural Network — each gives different R², and none directly answers the fairness question.
4. **Overfitting risk**: With only 10 Shenzhen districts providing demographic variation, complex models can memorize district-specific patterns.

The key insight (documented in `FCAUSAL_GDX_JOINT_ANALYSIS.md`, Section 5.3.4) was that the hat matrix formulation eliminates the need for g(D, x) model fitting entirely. The demographic regression is performed analytically at each optimization step via the hat matrix H = X(X'X)⁻¹X', which is a constant projection matrix computed once from the demographic features.

### 2.3 Why We Replaced Isotonic g₀(D) with Power Basis

The demand–service ratio relationship Y = S/D is fundamentally hyperbolic: when D is small, Y is large and variable; when D is large, Y approaches a constant. Empirical comparison (documented in `G0_DEMAND_BASIS_ANALYSIS.md`):

| g₀(D) Method | R² | Hat Matrix Compatible |
|--------------|-----|:---------------------:|
| Isotonic | 0.4453 | No |
| **Power Basis** | **0.4450** | **Yes** |
| Reciprocal | 0.4444 | Yes |
| Binning | 0.4439 | No |
| Log | 0.3610 | Yes |
| Polynomial (deg 2) | 0.1949 | Yes |
| Linear | 0.1064 | Yes |

The power basis `g₀(D) = β₀ + β₁/(D+1) + β₂/√(D+1) + β₃·√(D+1)` matches isotonic's R² within 0.0003 while being **linear in parameters**. This enables:

- **Hat matrix formulation**: g₀ can be expressed as a projection H_D, making Option C fully differentiable
- **Closed-form coefficients**: No iterative fitting needed
- **Interpretable terms**: Each basis function captures a different decay rate (rapid 1/D, moderate 1/√D, slow √D growth)

The baseline formulation retains isotonic regression to preserve exact historical comparability.

---

## 3. Formulation Details

### 3.1 Baseline: Historical R² (Isotonic g₀)

**Purpose**: Backward-compatible baseline for comparison with prior results.

**Formulation**:

$$F_{\text{baseline}} = R^2 = 1 - \frac{\text{Var}(Y - g_0(D))}{\text{Var}(Y)}$$

where $g_0(D)$ is fitted using isotonic regression (monotone decreasing).

**Interpretation**: "What fraction of service ratio variance is explained by demand alone?"

**Implementation**:
- g₀: `estimate_g_isotonic()` → frozen prediction function
- R² computed as `1 - Var(residuals) / Var(Y)`
- Per-period averaging: F_baseline = mean of per-period R² values

**Gradient behavior**: The gradient ∂F/∂Y has a known limitation — because g₀(D) is frozen and the R² formula is Var(R)/Var(Y), the gradient partly incentivizes increasing Var(Y) rather than reducing residuals. This is the "degenerate gradient" problem documented in `FCAUSAL_GDX_JOINT_ANALYSIS.md` Section 5.6.

**When to use**: Historical comparison, ablation studies, backward compatibility.

---

### 3.2 Option B: Demographic Residual Independence (Default)

**Purpose**: Measure whether demand-adjusted service residuals correlate with demographics.

**Formulation** (Hat Matrix Approach):

$$F_{\text{causal}} = \frac{R'(I-H)R}{R'MR}$$

where:
- $Y_c = S_c / D_c$ — service ratio per cell
- $R_c = Y_c - g_0(D_c)$ — residual after removing demand effect (g₀ uses power basis)
- $H = X(X'X)^{-1}X'$ — hat matrix projecting onto demographic feature space
- $M = I - \frac{11'}{N}$ — centering matrix
- $X$ — standardized demographic feature matrix (columns: income, population, etc.)

**Interpretation**: "What fraction of demand-adjusted service variation is NOT explained by demographics?" Equivalently, this is 1 − R²(R ~ demographics).

- F_causal = 1.0: Demographics explain none of the residual → perfectly fair
- F_causal = 0.0: Demographics explain all of the residual → maximally unfair

**Gradient**:

$$\frac{\partial F}{\partial R_c} = \frac{2}{R'MR} \left[((I-H)R)_c - F \cdot (R_c - \bar{R})\right]$$

Direction check:
- **Over-served rich cell** (R_c > R̄, demographics explain it): gradient is negative → pushes R_c down ✓
- **Under-served poor cell** (R_c < R̄, demographics explain it): gradient is positive → pushes R_c up ✓
- **Anomalous cell** (deviation NOT due to demographics): (I-H)R_c dominates → gradient doesn't "fix" it ✓

**Key properties**:
- The hat matrix implicitly re-fits the demographic regression at every optimization step
- No stale coefficients — β is always the true OLS fit of current R on demographics
- (I-H) projects R onto the space orthogonal to demographics, which is exactly the "fair" component
- Natural [0, 1] range with clear semantics

**Pre-computation** (done once during data loading):
1. Identify active cells (demand > threshold)
2. Extract demographic features X for active cells
3. Standardize X (zero mean, unit variance)
4. Add intercept column
5. Compute H = X(X'X)⁻¹X'
6. Store (I-H) and M as constant matrices
7. Fit g₀(D) using power basis on active cells

**When to use**: Default for production optimization. Best gradient behavior, clearest fairness interpretation.

---

### 3.3 Option C: Partial ΔR² (Dual Hat Matrices)

**Purpose**: Measure the incremental explanatory power of demographics beyond demand.

**Formulation**:

$$F_{\text{causal}} = 1 - \Delta R^2 = 1 - (R^2_{\text{full}} - R^2_{\text{red}})$$

where:
- $R^2_{\text{red}} = 1 - \frac{Y'(I-H_{\text{red}})Y}{Y'MY}$ — R² of demand-only model (Y ~ power_basis(D))
- $R^2_{\text{full}} = 1 - \frac{Y'(I-H_{\text{full}})Y}{Y'MY}$ — R² of full model (Y ~ power_basis(D) + demographics)
- $H_{\text{red}} = X_D(X_D'X_D)^{-1}X_D'$ — demand-only hat matrix
- $H_{\text{full}} = X_{D+x}(X_{D+x}'X_{D+x})^{-1}X_{D+x}'$ — full hat matrix

**Interpretation**: "How much additional variance do demographics explain beyond what demand already explains?" The ΔR² is the marginal contribution of demographics.

- F_causal = 1.0: Demographics add no explanatory power → fair
- F_causal = 0.0: Demographics explain everything that demand doesn't → maximally unfair

**Relationship to Option B**: When g₀ is linear-in-parameters (as with power basis), Options B and C are related by the Frisch-Waugh-Lovell theorem, but they report **different scales**:

$$\Delta R^2 = R^2_{\text{demo}} \times (1 - R^2_{\text{demand}})$$

where R²_demo is Option B's value and R²_demand = R²_red. The gradient directions are identical — both push the same cells in the same direction — but the magnitudes differ because:
- Option B: denominator is residual variance (Var(R), smaller)
- Option C: denominator is total variance (Var(Y), larger)

In practice on Shenzhen data (R²_demand ≈ 0.45), this means Option C's ΔR² ≈ 0.55 × R²_demo. The interpretation also differs:
- Option B says: "residuals don't correlate with demographics"
- Option C says: "demographics don't add explanatory power beyond demand"

**Pre-computation** (done once during data loading):
1. Build demand feature matrix X_D: power basis [1/(D+1), 1/√(D+1), √(D+1)] + intercept
2. Build full feature matrix X_full: [X_D | standardized demographics]
3. Compute H_red = X_D(X_D'X_D)⁻¹X_D' and H_full = X_full(X_full'X_full)⁻¹X_full'
4. Store (I-H_red), (I-H_full), and M

**When to use**: Validation and paper reporting. Provides the clearest statistical story: "we tested whether demographics add explanatory power beyond demand, and found ΔR² = X."

---

## 4. Architecture

### 4.1 Data Flow

```
                    Data Loading Phase (Once)
                    ========================

source_data/cell_demographics.pkl ──→ Demographics Grid (48×90×n_features)
source_data/grid_to_district_mapping.pkl ──→ District IDs, Valid Mask
pickup_dropoff_counts.pkl ──→ Demand per cell
active_taxis_*.pkl ──→ Supply per cell

                         │
                         ▼
               ┌─────────────────────┐
               │   Pre-computation   │
               │                     │
               │  1. Filter active   │
               │     cells (D > min) │
               │  2. Compute Y = S/D │
               │  3. Fit g₀(D) with  │
               │     power_basis     │
               │  4. Build X_demo    │
               │  5. Compute hat     │
               │     matrices        │
               │  6. Store in        │
               │     DataBundle      │
               └─────────┬───────────┘
                         │
                         ▼
              Optimization Phase (Per Step)
              ============================

    Baseline:  F = 1 - Var(Y - g₀_iso(D)) / Var(Y)
                     ↑ isotonic lookup (frozen)

    Option B:  R = Y - g₀_pb(D)
               F = R'(I-H_demo)R / R'MR
                     ↑ constant matrices

    Option C:  F = 1 - [Y'(H_full-H_red)Y / Y'MY]
                     ↑ constant matrices
```

### 4.2 Configuration

```python
@dataclass
class CausalFairnessConfig(TermConfig):
    # --- Formulation selection ---
    formulation: str = "option_b"       # "baseline", "option_b", "option_c"

    # --- Baseline g₀(D) method (used by baseline formulation) ---
    estimation_method: str = "isotonic"  # For baseline only

    # --- Demographics ---
    demographic_features: List[str] = None  # Features to include in hat matrix
    demographics_data_path: str = None      # Path to cell_demographics.pkl
    district_mapping_path: str = None       # Path to grid_to_district_mapping.pkl
```

### 4.3 Module Interface

```python
# In FAMAILObjective:

def compute_causal_fairness(self, demand, supply):
    if self.causal_formulation == "baseline":
        # Historical: R² = 1 - Var(R)/Var(Y) with isotonic g₀
        return self._compute_baseline_causal(demand, supply)
    elif self.causal_formulation == "option_b":
        # Hat matrix: F = R'(I-H)R / R'MR
        return self._compute_option_b_causal(demand, supply)
    elif self.causal_formulation == "option_c":
        # Dual hat matrix: F = 1 - ΔR²
        return self._compute_option_c_causal(demand, supply)
```

---

## 5. Gradient Comparison

| Property | Baseline | Option B | Option C |
|----------|----------|----------|----------|
| **Gradient target** | Var(Y) denominator | Demographic projection | Dual projection difference |
| **Degenerate gradient?** | Partially (Var(Y) inflation) | No | No |
| **Direction correctness** | Approximately correct | Exactly correct | Exactly correct |
| **Re-fits β per step?** | N/A (no β) | Yes (via hat matrix) | Yes (via hat matrices) |
| **Computational cost** | O(N) | O(N²) matrix-vector | O(N²) matrix-vector ×2 |
| **Gradient formula** | $\frac{2}{N \cdot \text{Var}(Y)} [g_0(D_c) - \bar{Y}]$ | $\frac{2}{R'MR}[(I-H)R - F \cdot MR]_c$ | Combination of two hat matrix gradients |

---

## 6. Expected Numerical Behavior

With the current Shenzhen dataset (~400-500 active cells, 10 districts):

- **Baseline F_causal**: Expected range 0.40–0.50 (demand explains ~40-50% of service ratio variance)
- **Option B F_causal**: 1 - R²_demo where R²_demo is demographics' share of *residual* variance. If the demographic signal in residuals is weak (R²_demo ≈ 0.02–0.05), then F_causal ≈ 0.95–0.98.
- **Option C F_causal**: 1 - ΔR² where ΔR² is demographics' marginal share of *total* variance. Since ΔR² = R²_demo × (1 - R²_demand), and R²_demand ≈ 0.45, Option C values will be higher: F_causal ≈ 0.97–0.99.

Note: The baseline measures a fundamentally different thing (demand alignment) than Options B/C (demographic independence). A high baseline F_causal does NOT imply high Option B F_causal, or vice versa. They are complementary metrics.

---

## 7. Paper Narrative

The three formulations support a layered story:

1. **Baseline**: "Service ratios broadly follow demand patterns (R² ≈ 0.45)."
2. **Option B**: "After accounting for demand, residual service variation shows [weak/moderate/strong] correlation with demographics (R²_demo ≈ X)."
3. **Option C**: "Demographics add ΔR² = X explanatory power beyond demand alone, confirming that service allocation is [not/partially/substantially] influenced by neighborhood socioeconomic characteristics."

The optimization then targets Option B (or C): the modified trajectories reduce the demographic signal in service residuals while preserving demand alignment (enforced by the power basis g₀ in the residual computation).

---

## Appendix A: Relationship Between Options B and C (FWL)

When g₀(D) is linear-in-parameters (as with power basis), the Frisch-Waugh-Lovell (FWL) theorem relates the two formulations:

$$\Delta R^2 = R^2_{\text{demo}} \times (1 - R^2_{\text{demand}})$$

where R²_demo is Option B's demographic R² on residuals and R²_demand is the demand-only R².

This means:
- **Gradient directions are identical**: both push the same cells up or down
- **Scalar values differ**: Option B reports R² relative to residual variance, Option C relative to total Y variance
- **The relationship is exact** when g₀ uses the same power basis features as the demand model in Option C

**Proof sketch**:
- Let $M_D = I - H_D$ be the residual-maker matrix for the demand model
- FWL says: regressing Y on [D, x] and extracting the x coefficient is equivalent to regressing $M_D Y$ on $M_D x$
- $M_D Y = Y - H_D Y = Y - g_0(D) = R$ (the demand residual)
- $M_D x = x - H_D x$ (demographics residualized on demand)
- The R² of this residualized regression equals ΔR²

This means Options B and C should produce numerically identical results (up to floating-point precision) when both use power basis demand features. The choice between them is about interpretation and diagnostic output, not numerical outcome.

## Appendix B: Demographic Features

The default demographic features for the hat matrix (from `cell_demographics.pkl`):

| Feature | Description | Source |
|---------|-------------|--------|
| AvgHousingPricePerSqM | Average housing price per square meter | District statistics |
| GDPperCapita | GDP per capita (derived: GDP / population) | Computed |
| CompPerCapita | Employee compensation per capita (derived) | Computed |

These are the features identified as most informative by the demographic explorer dashboard. The feature set is configurable via `CausalFairnessConfig.demographic_features`.

---

## Appendix C: File Locations

| File | Purpose |
|------|---------|
| `objective_function/causal_fairness/utils.py` | Core math: hat matrices, g₀ fitting, F_causal computation |
| `objective_function/causal_fairness/config.py` | Configuration with formulation selection |
| `objective_function/causal_fairness/term.py` | CausalFairnessTerm (dashboard evaluation) |
| `trajectory_modification/objective.py` | FAMAILObjective (production optimization) |
| `trajectory_modification/data_loader.py` | DataBundle with demographics and hat matrices |
| `trajectory_modification/dashboard.py` | Trajectory modification testing dashboard |
| `source_data/cell_demographics.pkl` | Demographics grid (48×90×13) |
| `source_data/grid_to_district_mapping.pkl` | District IDs and valid cell mask |
