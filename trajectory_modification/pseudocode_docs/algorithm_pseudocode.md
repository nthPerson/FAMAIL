# FAMAIL Trajectory Editing Algorithm — Pseudocode & Formulations

> **Generated**: 2026-03-09
> **Source files analyzed**: `trajectory_modification/modifier.py`, `objective.py`, `trajectory.py`,
> `data_loader.py`, `metrics.py`, `dashboard.py`; `objective_function/soft_cell_assignment/module.py`;
> `objective_function/causal_fairness/utils.py`

---

## Table of Contents

1. [High-Level Pipeline](#1-high-level-pipeline)
2. [Phase 1 — Attribution (Trajectory Selection)](#2-phase-1--attribution)
3. [Phase 2 — Trajectory Editing (ST-iFGSM)](#3-phase-2--trajectory-editing)
4. [Phase 3 — Evaluation](#4-phase-3--evaluation)
5. [Objective Function Formulations](#5-objective-function-formulations)
6. [Soft Cell Assignment](#6-soft-cell-assignment)
7. [Known Issues](#7-known-issues)

---

## 1. High-Level Pipeline

```
ALGORITHM FAMAIL_TrajectoryEditing(T, DataBundle, params)
────────────────────────────────────────────────────────
Input:
  T             — Set of all expert trajectories
  DataBundle    — Pickup/dropoff counts, active taxis, g(d), demographics, hat matrices
  params        — {α, ε, max_iter, weights (α₁,α₂,α₃), k, causal_formulation, ...}

Output:
  T'            — Modified trajectory set
  metrics       — Before/after fairness metrics

Steps:
  1. INITIALIZE global state (pickup_grid, dropoff_grid, active_taxis, causal grids)
  2. COMPUTE initial fairness snapshot (F_spatial, F_causal, F_fidelity)

  ──── Phase 1: Attribution ────
  3. scores ← COMPUTE_ATTRIBUTION(T, DataBundle)
  4. T_selected ← SELECT_TOP_K(scores, k, method)

  ──── Phase 2: Editing ────
  5. FOR EACH trajectory τ ∈ T_selected (sequentially):
       τ' ← ST_iFGSM_MODIFY(τ, objective_fn, params)
       UPDATE global counts (move pickup from old cell to new cell)
       RECORD modification history

  ──── Phase 3: Evaluation ────
  6. COMPUTE final fairness snapshot
  7. RETURN T' (with T_selected replaced by modified versions), metrics
```

---

## 2. Phase 1 — Attribution

Attribution ranks trajectories by their contribution to global unfairness, using two
complementary scores: one for spatial inequality (LIS) and one for causal inequality (DCD).

### 2.1 Local Inequality Score (LIS)

Measures how far each cell's service count deviates from the global mean.

```
FUNCTION COMPUTE_CELL_LIS(cell_counts)
───────────────────────────────────────
Input:  cell_counts — 2D array [48 × 90], either pickup or dropoff counts
Output: lis — 2D array [48 × 90] of per-cell LIS values ≥ 0

  μ ← mean(cell_counts)                    // Global mean count across all cells
  IF μ ≈ 0: RETURN zeros(48, 90)

  FOR each cell c:
    LIS_c ← |N_c − μ| / μ                 // N_c = count in cell c

  RETURN lis
```

**Trajectory-level LIS**:

```
FUNCTION TRAJECTORY_LIS(τ, pickup_lis, dropoff_lis)
────────────────────────────────────────────────────
  p ← pickup_cell(τ)                       // τ.states[-1] — last state
  d ← dropoff_cell(τ)                      // τ.states[0]  — first state

  LIS_τ ← max(pickup_lis[p], dropoff_lis[d])

  RETURN LIS_τ
```

> **Note**: The max aggregation flags a trajectory if *either* endpoint is in a highly
> unequal cell. Pickup LIS uses the pickup count grid; dropoff LIS uses the dropoff grid.

### 2.2 Demand-Conditional Deviation (DCD)

Measures how much the actual service ratio deviates from the demand-predicted expected ratio.

```
FUNCTION COMPUTE_CELL_DCD(demand, supply, g_function, formulation, hat_matrices, g0_func, active_indices)
─────────────────────────────────────────────────────────────────────────────────────────────────────────
Input:
  demand          — 2D array [48 × 90], pickup counts (demand proxy)
  supply          — 2D array [48 × 90], active taxis or supply proxy
  g_function      — Pre-fitted function mapping demand → expected service ratio
  formulation     — 'baseline' or 'option_b'
  hat_matrices    — Dict with 'H_demo' key (required for Option B)
  g0_func         — Power basis g₀(D) function (required for Option B)
  active_indices  — Indices of cells with valid demographic data (required for Option B)

Output: dcd — 2D array [48 × 90] of per-cell DCD values ≥ 0

  IF formulation = 'option_b' AND hat_matrices, g0_func, active_indices all available:
    // ★ FIXED (7.3) — Option B: demographic residual projection
    D ← demand.flatten()[active_indices]
    S ← supply.flatten()[active_indices]
    Y ← S / (D + ε)
    R ← Y − g₀(D)                            // Demand-controlled residuals
    R̂ ← H_demo @ R                           // Project onto demographic subspace
    DCD[active_indices] ← |R̂|                // Magnitude of demographically-explained residual
  ELSE:
    // Baseline: deviation from expected service ratio
    FOR each cell c WHERE demand_c > ε:
      Y_c ← supply_c / (demand_c + ε)
      DCD_c ← |Y_c − g(demand_c)|

  RETURN dcd
```

**Trajectory-level DCD**:

```
  DCD_τ ← DCD at pickup_cell(τ)            // Only pickup cell (the modifiable endpoint)
```

### 2.3 Combined Attribution Score & Selection

```
FUNCTION COMPUTE_ATTRIBUTION(T, DataBundle)
────────────────────────────────────────────
  pickup_lis  ← COMPUTE_CELL_LIS(pickup_grid)
  dropoff_lis ← COMPUTE_CELL_LIS(dropoff_grid)
  dcd_grid    ← COMPUTE_CELL_DCD(demand_grid, supply_grid, g_function)

  FOR each τ ∈ T:
    LIS_τ ← TRAJECTORY_LIS(τ, pickup_lis, dropoff_lis)
    DCD_τ ← dcd_grid[pickup_cell(τ)]

  // Normalize to [0, 1]
  LIS_norm_τ ← LIS_τ / max_over_all(LIS)
  DCD_norm_τ ← DCD_τ / max_over_all(DCD)

  // Weighted combination (default weights: 0.5, 0.5)
  Score_τ ← w_LIS · LIS_norm_τ + w_DCD · DCD_norm_τ

  RETURN scores for all τ


FUNCTION SELECT_TOP_K(scores, k, method='top_k')
──────────────────────────────────────────────────
  IF method = 'top_k':
    RETURN k trajectories with highest Score_τ

  IF method = 'diverse':
    selected ← []
    cell_count ← {}                         // Tracks how many selected from each cell
    β ← 0.5                                 // Diversity penalty factor

    SORT scores descending
    FOR each trajectory τ in sorted order:
      IF |selected| ≥ k: BREAK
      c ← pickup_cell(τ)
      Score_eff ← Score_τ · (1 − β · cell_count[c])
      IF Score_eff > 0 OR |selected| < k:
        ADD τ to selected
        cell_count[c] += 1

    RETURN selected
```

### 2.4 DCD Reformulation for Option B

The current DCD uses `g(D)` — a global isotonic/power-basis function mapping demand to
expected service ratio. This corresponds to the **baseline** causal fairness formulation.

Under **Option B**, causal fairness is measured by *demographic residual independence*:
how much demographic features explain the residuals `R = Y − g₀(D)`. The relevant
"unfairness signal" is not `|Y − g(D)|` but rather **how much of the residual is
explained by demographics**.

**Proposed Option B–compatible DCD**:

```
  R_c ← Y_c − g₀(D_c)                     // Demand-controlled residual at cell c
  R̂_c ← (H_demo · R)_c                    // Demographic projection of residual vector
  DCD_c^{optB} ← |R̂_c|                    // Magnitude of demographically-explained residual
```

Where `H_demo = X_demo(X_demo'X_demo)⁻¹X_demo'` is the demographic hat matrix.
Cells where demographics explain a large portion of the residual should be prioritized.

> **Status**: IMPLEMENTED (7.3 fix). `compute_cell_dcd_scores` in `dashboard.py` now
> supports both baseline and Option B formulations via the `causal_formulation` parameter.
> When `causal_formulation='option_b'`, the hat matrix from `st.session_state['hat_matrices']`
> is used to project residuals onto the demographic subspace.

---

## 3. Phase 2 — Trajectory Editing

The core editing loop uses a modified ST-iFGSM (Spatiotemporal Iterative Fast Gradient
Sign Method) to perturb pickup locations and maximize the combined objective.

### 3.1 ST-iFGSM Main Loop

```
FUNCTION ST_iFGSM_MODIFY(τ, objective_fn, params)
───────────────────────────────────────────────────
Input:
  τ              — Original trajectory to modify
  objective_fn   — FAMAILObjective (computes L = α₁F_spatial + α₂F_causal + α₃F_fidelity)
  params         — {α, ε, max_iter, convergence_threshold, τ_max, τ_min}

Output:
  ModificationHistory — {original, modified, per-iteration results, convergence info}

  // ──── Initialization ────
  τ' ← clone(τ)                                    // Modified trajectory starts as copy
  cumulative_δ ← [0, 0]                            // Total perturbation from original
  p_orig ← [τ.states[-1].x, τ.states[-1].y]       // Original pickup (continuous coords)
  p_current ← copy(p_orig)
  cell_orig ← floor(p_orig)                        // Original cell (integer)
  L_prev ← −∞

  // Store base counts (global counts minus this trajectory's contribution)
  base_pickup_counts ← global_pickup_counts.clone()

  // ──── Iterative Optimization ────
  FOR t = 0, 1, ..., max_iter − 1:

    // (a) Temperature annealing
    IF annealing_enabled:
      τ_t ← τ_max · (τ_min / τ_max)^(t / (max_iter − 1))
      soft_assign.set_temperature(τ_t)

    // (b) Create differentiable pickup tensor
    p_tensor ← Tensor(p_current, requires_grad=True)

    // (c) Build trajectory features for discriminator
    features_τ  ← τ.to_tensor()    // [seq_len, 4] → [1, seq_len, 4]
    features_τ' ← τ'.to_tensor()   // [seq_len, 4] → [1, seq_len, 4]

    // (d) Compute objective via differentiable soft cell assignment
    // Gradient path: pickup_location → soft_probs → soft_counts → objective
    soft_counts ← COMPUTE_SOFT_PICKUP_COUNTS(p_tensor, cell_orig, base_pickup_counts)
    L, terms ← objective_fn(soft_counts, dropoff_counts, causal_supply,
                             active_taxis, features_τ, features_τ', causal_demand)
    L.backward()
    grad ← p_tensor.grad                           // ∂L/∂p

    // (e) ST-iFGSM perturbation step
    IF ‖grad‖ > 1e-8:
      δ ← α · sign(grad)                           // Step in gradient sign direction
      cumulative_δ ← clip(cumulative_δ + δ, −ε, ε) // ★ Clip to ε-ball

    // (f) Apply perturbation to get new position
    old_cell ← floor(p_current)
    p_new ← clip_to_grid(p_orig + cumulative_δ)    // ★ Offset from ORIGINAL
    new_cell ← floor(p_new)

    // (g) Update global counts if cell changed
    IF old_cell ≠ new_cell:
      pickup_counts[old_cell] −= 1
      pickup_counts[new_cell] += 1

    p_current ← p_new

    // (h) Update modified trajectory (always offset from original, not running copy)
    τ' ← τ.apply_perturbation(cumulative_δ)         // ★ FIXED (7.1) — offset from original τ

    // (i) Record iteration result
    RECORD {τ', L, F_spatial, F_causal, F_fidelity, ‖grad‖, cumulative_δ}

    // (j) Check convergence
    IF |L − L_prev| < convergence_threshold: BREAK
    L_prev ← L

  RETURN ModificationHistory
```

### 3.2 Batch Modification

Trajectories are modified **sequentially** — each modification updates global counts
before the next trajectory is processed. This means later trajectories see the effects
of earlier modifications.

```
FUNCTION MODIFY_BATCH(T_selected, objective_fn)
─────────────────────────────────────────────────
  histories ← []
  FOR each τ ∈ T_selected:
    history ← ST_iFGSM_MODIFY(τ, objective_fn)
    // Global counts are already updated inside ST_iFGSM_MODIFY

    // ★ FIXED (7.2) — Sync base counts so next trajectory sees current world state
    base_pickup_counts ← pickup_counts.clone()
    base_dropoff_counts ← dropoff_counts.clone()

    histories.append(history)
  RETURN histories
```

---

## 4. Phase 3 — Evaluation

After all modifications, global fairness metrics are recomputed.

```
FUNCTION EVALUATE(global_metrics)
──────────────────────────────────
  gini ← PAIRWISE_GINI(DSR)                // DSR = pickups / active_taxis per cell
  r²   ← COMPUTE_R_SQUARED(Y, g_function)  // Y = supply / demand per cell
  fidelity ← mean(fidelity_scores)          // From discriminator during editing

  F_spatial ← 1 − gini                     // Or 1 − 0.5·(Gini_DSR + Gini_ASR)
  F_causal  ← clamp(r², 0, 1)              // Or Option B/C formulation
  F_fidelity ← fidelity

  L ← α₁·F_spatial + α₂·F_causal + α₃·F_fidelity

  RETURN FairnessSnapshot{gini, r², fidelity, F_spatial, F_causal, F_fidelity, L}
```

---

## 5. Objective Function Formulations

The combined objective maximized during editing:

$$L = \alpha_1 \cdot F_{\text{spatial}} + \alpha_2 \cdot F_{\text{causal}} + \alpha_3 \cdot F_{\text{fidelity}}$$

All terms are in $[0, 1]$ where higher = better (fairer / more realistic).

### 5.1 F_spatial — Spatial Fairness

**Formula**: $F_{\text{spatial}} = 1 - \frac{1}{2}\big(\text{Gini}(\text{DSR}) + \text{Gini}(\text{ASR})\big)$

Where:
- $\text{DSR}_c = \text{pickups}_c / \text{active\_taxis}_c$ (Departure Service Rate)
- $\text{ASR}_c = \text{dropoffs}_c / \text{active\_taxis}_c$ (Arrival Service Rate)

**Gini computation** (differentiable pairwise formula):

$$\text{Gini}(\mathbf{x}) = \frac{\sum_i \sum_j |x_i - x_j|}{2 n^2 \bar{x}}$$

```
FUNCTION PAIRWISE_GINI(values)
───────────────────────────────
  n ← length(values)
  μ ← mean(values) + ε
  diff_matrix ← |values_i − values_j| for all pairs (i, j)
  gini ← sum(diff_matrix) / (2 · n² · μ)
  RETURN clamp(gini, 0, 1)
```

> **Implementation**: `objective.py` lines 241-270, 272-343
> Gini is computed over **service-active cells only** (~2000 of 4320), filtered by:
> `service_mask = (pickup_counts > ε) | (dropoff_counts > ε) | (active_taxis > 0.5)`
> This avoids dilution from ~2300 cells with no real taxi activity (7.5 fix).

### 5.2 F_causal — Causal Fairness

Three formulations are implemented. **Option B is the preferred formulation**.

#### 5.2.1 Baseline: R² with Isotonic g₀(D)

$$F_{\text{causal}}^{\text{baseline}} = R^2 = 1 - \frac{\text{Var}(R)}{\text{Var}(Y)}$$

Where:
- $Y_c = S_c / (D_c + \varepsilon)$ is the service ratio at cell $c$
- $g_0(D)$ is an isotonic regression fitted on $(D, Y)$ pairs
- $R_c = Y_c - g_0(D_c)$ is the demand-adjusted residual

Higher R² means demand explains more of the service ratio variance → fairer.

Only cells with $D_c \geq 1.0$ are included.

> **Implementation**: `objective.py` lines 384-441
> `g₀` is fitted using `sklearn.IsotonicRegression(increasing=False)` on hourly-aggregated
> per-cell (D, Y) pairs.

#### 5.2.2 Option B: Demographic Residual Independence (PREFERRED)

$$F_{\text{causal}}^{\text{optB}} = \frac{\mathbf{R}^\top (\mathbf{I} - \mathbf{H}_{\text{demo}}) \mathbf{R}}{\mathbf{R}^\top \mathbf{M} \mathbf{R}}$$

Where:
- $\mathbf{R} = \mathbf{Y} - g_0(\mathbf{D})$ — vector of demand-controlled residuals
- $g_0(\mathbf{D})$ — power basis fit: $g_0(D) = \beta_0 + \beta_1 (D+1)^{-1} + \beta_2 (D+1)^{-0.5} + \beta_3 (D+1)^{0.5}$
- $\mathbf{H}_{\text{demo}} = \mathbf{X}_d (\mathbf{X}_d^\top \mathbf{X}_d)^{-1} \mathbf{X}_d^\top$ — hat matrix projecting onto demographic features
- $\mathbf{X}_d$ — standardized demographic features (default: AvgHousingPricePerSqM, GDPperCapita, CompPerCapita)
- $\mathbf{M} = \mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ — centering matrix
- $n$ — number of active cells (demand ≥ 1, valid demographics, no NaN features)

**Interpretation**: The fraction of residual variance *not* explained by demographics.
$F = 1$ means demographics explain nothing (perfectly fair).
$F = 0$ means demographics fully explain residuals (maximally unfair).

**Gradient flow**: $g_0$ and $\mathbf{H}_{\text{demo}}$ are frozen constants. Gradients flow
through $\mathbf{R}$ via $\mathbf{Y} = \mathbf{S} / (\mathbf{D} + \varepsilon)$, and $\mathbf{S}$
is affected by soft pickup counts.

> **Implementation**: `objective.py` lines 443-517, `utils.py:compute_fcausal_option_b_torch()`
> Hat matrices are pre-computed once in `DataBundle.load_default()` via
> `precompute_hat_matrices()` and stored as numpy arrays, lazily converted to tensors.

#### 5.2.3 Option C: Partial ΔR²

$$F_{\text{causal}}^{\text{optC}} = 1 - \Delta R^2 = 1 - (R^2_{\text{full}} - R^2_{\text{red}})$$

Where:
- $R^2_{\text{red}} = 1 - \frac{\mathbf{Y}^\top (\mathbf{I} - \mathbf{H}_{\text{red}}) \mathbf{Y}}{\mathbf{Y}^\top \mathbf{M} \mathbf{Y}}$ — R² using demand-only model
- $R^2_{\text{full}} = 1 - \frac{\mathbf{Y}^\top (\mathbf{I} - \mathbf{H}_{\text{full}}) \mathbf{Y}}{\mathbf{Y}^\top \mathbf{M} \mathbf{Y}}$ — R² using demand + demographics model
- $\mathbf{H}_{\text{red}}$ — hat matrix for power-basis demand features only
- $\mathbf{H}_{\text{full}}$ — hat matrix for demand + demographics features

**Interpretation**: How much *additional* variance demographics explain beyond demand alone.
$F = 1$ means demographics add no explanatory power → fair.

> **Implementation**: `objective.py` lines 519-587, `utils.py:compute_fcausal_option_c_torch()`

### 5.3 F_fidelity — Behavioral Fidelity

$$F_{\text{fidelity}} = \text{Discriminator}(\tau, \tau')$$

- Uses a pre-trained ST-SiameseNet (LSTM-based) from `discriminator/model/model.py`
- Input: original trajectory features $\tau$ [batch, seq_len, 4] and modified $\tau'$
- Output: similarity score in $[0, 1]$ where 1 = indistinguishable from original
- The discriminator is **frozen** (eval mode, no gradient updates to its parameters)
- Gradients flow through $\tau'$ features for optimization of the pickup location
- If `α₃ = 0`, fidelity computation is skipped entirely (no discriminator needed)

> **Implementation**: `objective.py` lines 589-633
> Feature format: each trajectory state → `[x_grid, y_grid, time_bucket, day_index]`

---

## 6. Soft Cell Assignment

The key differentiable bridge between continuous pickup locations and discrete grid counts.

### 6.1 Gaussian Soft Assignment

For a continuous location $(x, y)$ and its original cell $(c_x, c_y)$, compute
probability distribution over a $(2k+1) \times (2k+1)$ neighborhood (default $k = 2$, so $5 \times 5$):

$$\sigma_{(i,j)}(x, y) = \frac{\exp\!\Big(-\frac{(x - (c_x + i))^2 + (y - (c_y + j))^2}{2\tau^2}\Big)}{\sum_{i',j'} \exp\!\Big(-\frac{(x - (c_x + i'))^2 + (y - (c_y + j'))^2}{2\tau^2}\Big)}$$

for $i, j \in \{-k, \ldots, k\}$.

- **High temperature** ($\tau \to \infty$): uniform distribution → more gradient flow, less precise
- **Low temperature** ($\tau \to 0$): approaches hard assignment → less gradient flow, more precise

### 6.2 Temperature Annealing

Exponential decay from $\tau_{\max} = 1.0$ to $\tau_{\min} = 0.1$:

$$\tau_t = \tau_{\max} \cdot \left(\frac{\tau_{\min}}{\tau_{\max}}\right)^{t / (T-1)}$$

This allows early iterations to explore broadly (soft assignments) and later iterations
to commit to specific cells (hard assignments).

### 6.3 Soft Count Computation

During optimization, the pickup count grid is recomputed differentiably:

```
FUNCTION COMPUTE_SOFT_PICKUP_COUNTS(p_tensor, cell_orig, base_counts)
──────────────────────────────────────────────────────────────────────
  // Start with base counts (global counts MINUS this trajectory's original contribution)
  soft_counts ← base_counts.clone()
  soft_counts[cell_orig] −= 1.0                    // Remove original hard assignment

  // Compute soft assignment probabilities
  probs ← SOFT_CELL_ASSIGN(p_tensor, cell_orig)    // [1, ns, ns]

  // Scatter soft probabilities into count grid
  FOR each (di, dj) in neighborhood:
    cell ← (cell_orig_x + di, cell_orig_y + dj)
    IF in_bounds(cell):
      soft_counts[cell] += probs[di + k, dj + k]   // Differentiable addition

  RETURN soft_counts
  // Gradient chain: soft_counts → probs → p_tensor
```

> **Implementation**: `modifier.py` lines 222-266

---

## 7. Known Issues

### 7.1 ~~CRITICAL: Epsilon Constraint Not Respected~~ — FIXED

**File**: `modifier.py` line 409
**Was**: `modified = modified.apply_perturbation(cumulative_delta)`
**Now**: `modified = trajectory.apply_perturbation(cumulative_delta)`

The `apply_perturbation` method adds the delta to the **current** pickup location. Since
`cumulative_delta` is the total offset from the **original** position, applying it to the
already-shifted `modified` trajectory compounded the perturbation each iteration, causing
the pickup to drift far beyond the ε-ball constraint (up to ~50ε with 50 iterations).

**Fix applied**: Always apply `cumulative_delta` to the original `trajectory` parameter
(which is never mutated), not the running `modified` copy. The `apply_perturbation` method
clones internally, so this is safe.

### 7.2 ~~Inconsistent Count Tracking in Soft Cell Mode~~ — FIXED

**Was**: In batch mode, `_base_pickup_counts` was set once in `set_global_state()` and
never updated between trajectories. This meant the soft count computation for trajectory
N+1 still subtracted from trajectory N's *original* cell position rather than its final
modified position.

**Fix applied**: After each trajectory's ST-iFGSM loop completes, sync base counts:
```python
self._base_pickup_counts = self.pickup_counts.clone()
self._base_dropoff_counts = self.dropoff_counts.clone()
```
This ensures each subsequent trajectory in a batch sees the correct world state.

### 7.3 ~~DCD Attribution Uses Baseline g(D), Not Option B~~ — FIXED

**Was**: `compute_cell_dcd_scores` always used `DCD_c = |Y_c − g(D_c)|` regardless of
which causal formulation was selected.

**Fix applied**: `compute_cell_dcd_scores` now accepts a `causal_formulation` parameter.
When `causal_formulation='option_b'`, it computes `DCD_c = |R̂_c|` where `R̂ = H_demo @ R`
and `R = Y − g₀(D)`. This is propagated through `compute_trajectory_attribution_scores`,
`_compute_cell_fairness_scores`, and the dashboard call sites.

### 7.4 Computational Complexity Analysis

#### Current Algorithm: Attribute Once, Then k Sequential Modifications

**Phase 1 — Attribution** (computed once):
- Cell LIS: O(C) where C = 4320 cells
- Cell DCD (baseline): O(C) — one pass through cells
- Cell DCD (Option B): O(n³ + n·C) — hat matrix precomputation O(n³) where n ≈ 148
  active demographic cells, then projection O(n)
- Trajectory scoring + sorting: O(N log N) where N = total trajectories (~50 drivers × ~100 trips)
- **Total attribution**: O(C + N log N) ≈ O(5,000–10,000) — negligible

**Phase 2 — Modification** (per trajectory, T iterations):
- Soft cell assignment: O(w²) per iteration, w = window size (5×5 = 25 cells) — negligible
- Objective function evaluation, dominated by:
  - **Pairwise Gini** (most expensive): O(n²) per call, called 2× per iteration (DSR + ASR)
    - All cells (pre-7.5 fix): n = 4320 → O(37.3M) per iteration
    - Service-active cells (post-7.5 fix): n ≈ 2000 → O(8M) per iteration
  - F_causal (Option B): O(n_causal²) where n_causal ≈ 148 — negligible vs Gini
  - F_fidelity (discriminator forward pass): O(seq_len × hidden²) ≈ O(50 × 128²) ≈ O(800K) — negligible vs Gini
- Backward pass (autograd): ~2× forward cost
- **Per-iteration cost**: O(n²) dominated by Gini
- **Per-trajectory cost** (T = 50 iterations): O(T × n²)
  - Pre-fix: O(50 × 37.3M) ≈ O(1.87B)
  - Post-fix: O(50 × 8M) ≈ O(400M)
- **Total for k trajectories**: O(k × T × n²)

#### Alternative: Re-attribution After Each Trajectory

Additional cost per trajectory: O(C + N log N) ≈ O(5,000–10,000)

As fraction of modification cost (post-7.5 fix):
```
  re-attribution / modification = 10,000 / 400,000,000 ≈ 0.0025%
```

**Conclusion**: Re-attribution is computationally free relative to modification cost.
The dominant bottleneck is the pairwise Gini computation at O(n²) per iteration.
Re-attributing after each trajectory modification is recommended for future implementation
to ensure each subsequent trajectory selection accounts for the updated world state.

#### Summary Table

| Component | Cost | Notes |
|-----------|------|-------|
| Attribution (all N trajectories) | O(C + N log N) | ~5K–10K ops |
| Per Gini call (post-7.5) | O(n²), n ≈ 2000 | ~4M ops |
| Per ST-iFGSM iteration | O(n²) | Dominated by 2× Gini |
| Per trajectory (T = 50 iters) | O(T × n²) | ~400M ops |
| Full batch (k trajectories) | O(k × T × n²) | ~4B ops for k=10 |
| Re-attribution overhead | O(C + N log N) | 0.0025% of modification |

### 7.5 ~~Spatial Fairness Computed Over All Cells (Including Zeros)~~ — FIXED

**Was**: Gini computed over all 4320 cells. Since `data_loader.py:747` floors
`active_taxis_grid` to 0.1, the existing `active_taxis > eps` mask was True for ALL
cells — it filtered nothing. ~2300 cells with no real taxi activity diluted the metric.

**Fix applied**: Added a **service mask** to filter cells for Gini computation:
```
service_mask = (pickup_counts > ε) | (dropoff_counts > ε) | (active_taxis > 0.5)
```
The 0.5 threshold distinguishes real taxi activity (mean-aggregated from integer counts
≥ 1) from the artificial 0.1 floor. This yields ~2000 service-active cells.

Applied in both `objective.py` (`compute_spatial_fairness`) and `metrics.py`
(`compute_gini`). Side benefit: pairwise Gini drops from O(4320²) ≈ 18.7M to
O(2000²) ≈ 4M operations — a ~4.7× speedup.

---

## Appendix A: Data Flow Summary

```
Raw GPS data
  ↓
pickup_dropoff_counts.pkl → aggregate_to_grid('sum') → pickup_grid, dropoff_grid [48×90]
                          → aggregate_to_grid('mean') → causal_demand_grid [48×90]
  ↓
active_taxis_5x5_hourly.pkl → aggregate_to_grid('mean') → active_taxis_grid [48×90]
                             →                           → causal_supply_grid [48×90]
  ↓
cell_demographics.pkl + grid_to_district_mapping.pkl
  → enrich_demographic_features()
  → precompute_hat_matrices(demands, demo_features)
  → hat_matrices {I_minus_H_demo, M, I_minus_H_red, I_minus_H_full, ...}
  ↓
estimate_g_power_basis(demands, ratios) → g₀(D) power basis function
GFunctionLoader.estimate_from_data()    → g(D) isotonic regression
```

## Appendix B: Key Parameters & Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| α (step size) | 0.1 | ST-iFGSM step magnitude |
| ε (max perturbation) | 2.0 (modifier) / 3.0 (dashboard) | Max offset per dimension from original |
| max_iterations | 50 | Maximum ST-iFGSM iterations per trajectory |
| convergence_threshold | 1e-6 | Stop if |ΔL| < threshold |
| α₁ (spatial weight) | 0.33 | Weight for F_spatial |
| α₂ (causal weight) | 0.33 | Weight for F_causal |
| α₃ (fidelity weight) | 0.34 | Weight for F_fidelity |
| neighborhood_size | 5 | Soft cell assignment window (5×5) |
| τ_max | 1.0 | Initial temperature for annealing |
| τ_min | 0.1 | Final temperature for annealing |
| k (top-k) | User-selected | Number of trajectories to modify |
| w_LIS | 0.5 | Attribution weight for LIS |
| w_DCD | 0.5 | Attribution weight for DCD |
| β (diversity penalty) | 0.5 | Penalty factor for diverse selection |
| DEMAND_THRESHOLD | 1.0 | Minimum demand for cell to be "active" |
| causal_formulation | 'option_b' | 'baseline', 'option_b', or 'option_c' |

## Appendix C: Trajectory Representation

Each trajectory τ is a sequence of states from passenger-seeking start to pickup:

```
τ = [s₀, s₁, ..., s_{n-1}]

where each sᵢ = (x_grid, y_grid, time_bucket, day_index)

  s₀       = start of passenger-seeking path (also treated as "dropoff" for LIS)
  s_{n-1}  = pickup location (THE MODIFIABLE PARAMETER)
```

The pickup is always the **last** state. Only the pickup location (2D continuous coords)
is modified during editing. Time bucket and day index remain unchanged.

For the discriminator, each state is converted to a 4-element feature vector
`[x, y, time, day]`, producing a tensor of shape `[seq_len, 4]`.
