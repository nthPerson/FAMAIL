# FAMAIL: Algorithm Overview

**Fairness-Aware Multi-Agent Imitation Learning — Trajectory Modification Algorithm**

---

## 1. Introduction

### 1.1 Problem Statement

Taxi service distribution in urban environments is inherently unequal. Some areas receive disproportionately high service relative to demand, while other areas — often lower-income or peripheral neighborhoods — are underserved. This inequality arises naturally from expert driver behavior: drivers learn to optimize for personal revenue, gravitating toward high-demand areas and avoiding less profitable zones.

FAMAIL (Fairness-Aware Multi-Agent Imitation Learning) addresses this problem by modifying expert taxi driver trajectories to improve the overall fairness of service distribution. Rather than generating entirely new trajectories, FAMAIL edits existing expert trajectories — specifically, their pickup locations — so that the resulting distribution more equitably serves all areas of the city.

### 1.2 Design Philosophy

The FAMAIL approach prioritizes **trajectory editing over trajectory generation**:

- **Editing** applies small, bounded adjustments to existing expert trajectories (moving pickup locations within a constrained spatial neighborhood).
- **Generation** would create entirely new synthetic trajectories from scratch.

Editing is preferred because:

1. **Preserves expert knowledge**: Driving patterns, route choices, and temporal behaviors are retained.
2. **Maximizes efficiency**: Not all trajectories contribute equally to global unfairness, so a minimal number of trajectories are modified.
3. **Bounded modifications**: Spatial constraints ($\epsilon$) limit how far pickup locations can move, ensuring modifications remain realistic.
4. **Enables fidelity validation**: A discriminator network can verify that a modified trajectory still "looks like" it came from the same driver.

### 1.3 Study Area and Grid

The system operates over a spatial grid covering Shenzhen, China:

- **Grid dimensions**: 48 × 90 cells (latitude × longitude)
- **Coordinate system**: Integer cell indices where each cell represents a geographic zone
- **Temporal resolution**: 5-minute time buckets (288 per day, 1-indexed) or hourly aggregation (24 per day, 0-indexed)
- **Days covered**: Monday through Saturday (1-indexed, 1–6), derived from July 2016 GPS data from 50 expert taxi drivers

---

## 2. System Goals

FAMAIL simultaneously optimizes three objectives:

1. **Spatial Fairness** ($F_{\text{spatial}}$): Equalize the distribution of taxi service (pickups and dropoffs) across all grid cells, normalized by taxi availability. Low Gini coefficient = high fairness.

2. **Causal Fairness** ($F_{\text{causal}}$): Ensure that the supply of taxi service provided to a cell is explained by demand and is not driven by sensitive demographic attributes (income, wealth). In the current demand-only baseline, this is measured as $R^2$ of the relationship $Y = g_0(D)$, where $Y$ is the service ratio and $D$ is demand. The planned revision conditions on demographic features as well: $g(D, \mathbf{x})$, measuring whether service disparities persist after accounting for both demand and demographics. **Note**: The current demand-only formulation is a misnomer for "causal fairness" — it measures unexplained inequality (demand-alignment), not the influence of sensitive attributes. The revised formulation addresses this limitation.

3. **Fidelity** ($F_{\text{fidelity}}$): Ensure that modified trajectories remain behaviorally realistic — that a discriminator network cannot easily distinguish an edited trajectory from the original. High discriminator similarity = high fidelity.

These are combined into a single objective:

$$\mathcal{L} = \alpha_1 \cdot F_{\text{spatial}} + \alpha_2 \cdot F_{\text{causal}} + \alpha_3 \cdot F_{\text{fidelity}}$$

where $\alpha_1 + \alpha_2 + \alpha_3 = 1$ (default: $\alpha_1 = 0.33$, $\alpha_2 = 0.33$, $\alpha_3 = 0.34$).

The objective is **maximized**: higher $\mathcal{L}$ means fairer outcomes with better behavioral fidelity.

---

## 3. Two-Phase Pipeline

The trajectory modification process consists of two distinct phases:

### Phase 1: Attribution (Trajectory Selection)

Not all trajectories contribute equally to global unfairness. The **attribution phase** identifies the trajectories that have the greatest negative impact on fairness and selects them for modification.

Attribution uses two complementary methods:

- **Local Inequality Score (LIS)**: Measures each trajectory's contribution to *spatial* inequality. A trajectory whose pickup or dropoff falls in a cell that deviates significantly from the global mean service rate receives a high LIS score.

- **Demand-Conditional Deviation (DCD)**: Measures each trajectory's contribution to *causal* inequality. A trajectory whose pickup cell shows a large deviation between the actual service ratio $Y = S/D$ and the expected fair ratio $g(D)$ receives a high DCD score.

These scores are combined into a weighted sum:

$$\text{Combined}_\tau = w_{\text{LIS}} \cdot \widetilde{\text{LIS}}_\tau + w_{\text{DCD}} \cdot \widetilde{\text{DCD}}_\tau$$

where tildes denote normalization to $[0, 1]$. The top-$k$ trajectories by combined score are selected for modification.


### Phase 2: Modification (Gradient-Based Editing)

Selected trajectories are individually modified using a variant of the **ST-iFGSM** (Spatio-Temporal iterative Fast Gradient Sign Method) algorithm. For each selected trajectory:

1. Compute the gradient of the combined objective $\mathcal{L}$ with respect to the trajectory's pickup location.
2. Apply a bounded perturbation in the direction of the gradient to improve fairness.
3. Iterate until convergence or a maximum number of iterations is reached.
4. After modifying one trajectory, update global counts and proceed to the next.

---

## 4. The Six-Step Algorithm

The complete trajectory modification loop follows six steps:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAJECTORY MODIFICATION LOOP                     │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 1: Compute attribution scores (LIS + DCD) for all             │
│          trajectories and rank by fairness impact                   │
│      ↓                                                              │
│  STEP 2: Select top-k highest-impact trajectories                   │
│      ↓                                                              │
│  STEP 3: For each selected trajectory, apply ST-iFGSM to            │
│          iteratively modify its pickup location:                    │
│              δ = clip(α · sign[∇L], -ε, ε)                          │
│      ↓                                                              │
│  STEP 4: Validate fidelity via discriminator                        │
│      ↓                                                              │
│  STEP 5: Update global pickup counts to reflect the edit            │
│      ↓                                                              │
│  STEP 6: Check convergence of global fairness → LOOP or EXIT        │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 1: Attribution Scoring

For every trajectory $\tau$ in the dataset:

- Compute $\text{LIS}_\tau$ based on how much the trajectory's pickup and dropoff cells deviate from the mean service rate (see Section 5.1).
- Compute $\text{DCD}_\tau$ based on how much the trajectory's pickup cell deviates from the expected fair service ratio $g(D)$ (see Section 5.2).
- Normalize both scores to $[0, 1]$ across all trajectories.
- Compute $\text{Combined}_\tau = w_{\text{LIS}} \cdot \widetilde{\text{LIS}}_\tau + w_{\text{DCD}} \cdot \widetilde{\text{DCD}}_\tau$.

### Step 2: Top-$k$ Selection

Select the $k$ trajectories with the highest combined attribution scores. Two selection strategies are available:

- **Top-$k$**: Simply take the $k$ highest-scoring trajectories.
- **Diverse**: Apply a spatial penalty to avoid clustering modifications in the same cells. When a trajectory from cell $c$ is selected, future candidates from cell $c$ have their effective score reduced by a penalty factor.

### Step 3: Gradient-Based Modification (ST-iFGSM)

For each selected trajectory $\tau$, the modifier runs the following loop:

1. **Initialize**: $\tau' = \tau$ (modified trajectory starts as original).
2. **For each iteration** $t = 1, \ldots, T$:
   - Compute the pickup location as a differentiable tensor $\mathbf{p}$ with `requires_grad=True`.
   - Using **soft cell assignment**, compute differentiable pickup counts from $\mathbf{p}$.
   - Evaluate the objective: $\mathcal{L}(\mathbf{p}) = \alpha_1 F_{\text{spatial}} + \alpha_2 F_{\text{causal}} + \alpha_3 F_{\text{fidelity}}$.
   - Compute $\nabla_\mathbf{p} \mathcal{L}$ via backpropagation.
   - Compute perturbation: $\delta = \text{clip}(\alpha \cdot \text{sign}(\nabla_\mathbf{p} \mathcal{L}),\ -\epsilon,\ \epsilon)$.
   - Update pickup: $\mathbf{p} \leftarrow \text{project}(\mathbf{p} + \delta)$, where projection clips to valid grid bounds.
3. **Return** the modified trajectory $\tau'$.

**Key parameters**:
- $\alpha$: Step size (default: 0.1 grid cells per iteration)
- $\epsilon$: Maximum cumulative perturbation per dimension (default: 3.0 grid cells)
- $T$: Maximum iterations (default: 50)
- Convergence threshold: $10^{-6}$

### Step 4: Fidelity Validation

After modification, the discriminator evaluates $f(\tau, \tau') \in [0, 1]$:
- $f \approx 1$: Trajectories appear to be from the same driver (good fidelity).
- $f \approx 0$: Trajectories appear different (poor fidelity, modification too aggressive).

The discriminator is a Siamese LSTM network (ST-SiameseNet) trained to distinguish trajectories from the same driver vs. different drivers.

### Step 5: Global Count Update

After each trajectory modification, the global pickup count grid is updated:
- Decrement the pickup count in the original cell.
- Increment the pickup count in the new cell.

This ensures that subsequent modifications account for the changed distribution.

### Step 6: Convergence Check

After all selected trajectories are modified, check whether global fairness metrics have improved sufficiently. If not, return to Step 1 with updated counts for another round of attribution and modification.

---

## 5. Attribution Methods

### 5.1 Local Inequality Score (LIS) — Spatial Fairness

LIS measures how much a grid cell's pickup or dropoff count deviates from the global mean, normalized by the mean:

$$\text{LIS}_c = \frac{|c - \mu|}{\mu}$$

where $c$ is the count in cell $c$ and $\mu$ is the global mean count across all cells.

For a trajectory $\tau$ with pickup cell $p$ and dropoff cell $d$:

$$\text{LIS}_\tau = \max(\text{LIS}_p^{\text{pickup}},\ \text{LIS}_d^{\text{dropoff}})$$

The max aggregation ensures that a trajectory is flagged if *either* its pickup or dropoff location is in a highly unequal cell. **NOTE** This formulation may be revised to consider only the pickup location.

### 5.2 Demand-Conditional Deviation (DCD) — Causal Fairness

DCD measures how much the actual service ratio at a cell deviates from the expected fair ratio predicted by the $g(d)$ function:

$$\text{DCD}_c = |Y_c - g(D_c)|$$

where:
- $Y_c = S_c / D_c$ is the actual service ratio (supply per demand) at cell $c$
- $g(D_c)$ is the expected service ratio given demand $D_c$, estimated from historical data using isotonic regression (current demand-only baseline; planned revision to $g(D_c, \mathbf{x}_c)$ will also condition on district-level demographic features)

For a trajectory $\tau$ with pickup cell $p$:

$$\text{DCD}_\tau = \text{DCD}_p$$

DCD focuses on the pickup cell because that is the location being modified.

### 5.3 Combined Attribution and Selection

The combined attribution score for trajectory $\tau$ is:

$$\text{Combined}_\tau = w_{\text{LIS}} \cdot \widetilde{\text{LIS}}_\tau + w_{\text{DCD}} \cdot \widetilde{\text{DCD}}_\tau$$

where $\widetilde{\cdot}$ denotes normalization to $[0, 1]$ by dividing each score by the maximum score across all trajectories. Default weights are $w_{\text{LIS}} = w_{\text{DCD}} = 0.5$.

The top-$k$ trajectories by combined score are selected for modification. This ensures the algorithm focuses its effort on trajectories that have the greatest potential to improve both spatial and causal fairness simultaneously.

---

## 6. Gradient Flow and Differentiability

### 6.1 The Differentiability Challenge

The objective function $\mathcal{L}$ depends on discrete grid cell counts (e.g., "how many pickups are in cell $(i, j)$?"). Discrete assignment — which assigns a pickup to exactly one grid cell — is not differentiable, preventing gradient-based optimization.

### 6.2 Soft Cell Assignment

The solution is **soft cell assignment**: instead of assigning a pickup to a single cell, the pickup's location is represented as a probability distribution over nearby cells. This is implemented as a Gaussian softmax over a neighborhood:

$$\sigma_c(\mathbf{p}; \tau) = \frac{\exp\!\left(-\|\mathbf{p} - \mathbf{c}\|^2 / (2\tau^2)\right)}{\sum_{c' \in \mathcal{N}} \exp\!\left(-\|\mathbf{p} - \mathbf{c}'\|^2 / (2\tau^2)\right)}$$

where:
- $\mathbf{p} = (x, y)$ is the continuous pickup location
- $\mathbf{c}$ is the center of cell $c$
- $\mathcal{N}$ is the neighborhood of $n \times n$ cells centered on the original cell (default: $5 \times 5$, i.e., $k = 2$)
- $\tau$ is the temperature parameter

This creates a differentiable path: **pickup location → soft probabilities → soft counts → objective → gradient**.

### 6.3 Temperature Annealing

The temperature $\tau$ controls the sharpness of the assignment:
- **High $\tau$** (e.g., 1.0): Soft assignments spread probability mass across many cells (smooth gradients, but less precise).
- **Low $\tau$** (e.g., 0.1): Assignments approach hard (one-hot) assignment (precise, but gradients may vanish).

During optimization, exponential annealing is used:

$$\tau_t = \tau_{\max} \cdot \left(\frac{\tau_{\min}}{\tau_{\max}}\right)^{t/T}$$

This starts with soft assignments (good gradient flow) and transitions to hard assignments (accurate cell counts) as optimization progresses.

### 6.4 Gradient Flow Path

The full gradient flow path during the ST-iFGSM loop is:

```
pickup_location (requires_grad=True)
    ↓
soft cell probabilities σ_c(p; τ)
    ↓
differentiable pickup counts = base_counts + Σ_c σ_c · 1
    ↓
F_spatial (Gini of DSR and ASR)  +  F_causal (R² of Y vs g(D))  +  F_fidelity (discriminator)
    ↓
L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity
    ↓
∇_p L (via backpropagation)
    ↓
δ = clip(α · sign(∇_p L), -ε, ε)
    ↓
p' = clip(p + δ, grid_bounds)
```

### 6.5 Gradient Modes

The trajectory modifier supports two gradient computation modes:

1. **`soft_cell` mode** (preferred): Computes true gradients through the soft cell assignment mechanism as described above. Requires the `SoftCellAssignment` module.

2. **`heuristic` mode** (fallback): Estimates gradient direction by examining the Departure Service Rate (DSR) of neighboring cells. The gradient points toward cells with lower DSR (underserved areas). This mode does not require differentiable counts but provides less precise optimization direction.

---

## 7. Objective Function Terms

### 7.1 Spatial Fairness ($F_{\text{spatial}}$)

Spatial fairness measures the equity of taxi service distribution across all cells using the Gini coefficient:

$$F_{\text{spatial}} = 1 - \frac{1}{2}\left(G_{\text{DSR}} + G_{\text{ASR}}\right)$$

where:
- $\text{DSR}_c = N_c^{\text{pickup}} / A_c$ is the **Departure Service Rate** (pickups normalized by active taxis)
- $\text{ASR}_c = N_c^{\text{dropoff}} / A_c$ is the **Arrival Service Rate** (dropoffs normalized by active taxis)
- $A_c$ is the number of active taxis in cell $c$ (from the `active_taxis` dataset)

The Gini coefficient uses the differentiable pairwise formula:

$$G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |x_i - x_j|}{2n^2 \bar{x}}$$

$F_{\text{spatial}} \in [0, 1]$, where $1$ means perfect equality (no variation in service rates across cells).

### 7.2 Causal Fairness ($F_{\text{causal}}$)

> **Terminology Note (2026-02-06):** The current $F_{\text{causal}}$ is a **misnomer** — it measures *demand-alignment* (unexplained inequality), not true causal fairness. It does not account for the influence of sensitive demographic attributes on service distribution. The planned revision (see mathematical_foundations.md, Section 11.6) will condition on demographic features to measure demographic-driven disparities. The current implementation is preserved as a demand-only baseline for ablation studies.

Causal fairness (in its current demand-only form) measures whether taxi supply is allocated proportionally to demand. It uses the coefficient of determination $R^2$:

$$F_{\text{causal}} = \max\!\left(0,\ R^2\right) = \max\!\left(0,\ 1 - \frac{\text{Var}(R)}{\text{Var}(Y)}\right)$$

where:
- $Y_c = S_c / D_c$ is the service ratio (supply per demand) at cell $c$
- $g(D_c)$ is the expected service ratio given demand $D_c$, estimated by isotonic regression
- $R_c = Y_c - g(D_c)$ is the residual (deviation from expected)

$F_{\text{causal}} \in [0, 1]$, where $1$ means all variance in service ratios is explained by demand alone (no unexplained inequality in the current formulation; no demographic-driven inequality in the planned revision).

**Key implementation detail**: The $g(d)$ function is **frozen** during optimization — it is pre-fitted on historical data and does not receive gradients. The demand threshold for including cells in the $R^2$ computation must match the threshold used when fitting $g(d)$ (default: $D \geq 1.0$).

### 7.3 Fidelity ($F_{\text{fidelity}}$)

Fidelity measures whether the modified trajectory still resembles the original driver's behavior:

$$F_{\text{fidelity}} = f(\tau, \tau')$$

where $f$ is a Siamese LSTM discriminator (ST-SiameseNet) that outputs a similarity score in $[0, 1]$:
- $f \approx 1$: Same-agent behavior
- $f \approx 0$: Different-agent behavior

The discriminator architecture:
- Two shared-weight LSTM branches (hidden dims: 200 → 100)
- Optional bidirectional processing (not used in the current model used in the algorithm)
- Dropout for regularization
- Sigmoid output for similarity

The discriminator is frozen during trajectory modification (no parameter updates), but gradients flow through it to the modified trajectory features.

---

## 8. Data Flow

### 8.1 Input Datasets

The trajectory modification framework uses three primary datasets:

| Dataset | File | Purpose |
|---------|------|---------|
| Passenger-seeking trajectories | `source_data/passenger_seeking_trajs_45-800.pkl` | Trajectories to be analyzed and modified |
| Pickup/dropoff counts | `source_data/pickup_dropoff_counts.pkl` | Historical demand and service data |
| Active taxis | `source_data/active_taxis_5x5_hourly.pkl` | Taxi supply (availability) per cell |

### 8.2 Aggregation Strategy

The raw data uses different temporal granularities. The framework aggregates data into 2D grids ($48 \times 90$) using configured strategies:

| Use Case | Data Source | Aggregation | Reason |
|----------|------------|-------------|--------|
| **Spatial fairness** (DSR, ASR) | Pickup/dropoff counts | **Sum** across all time periods | Total service volume measures overall equity |
| **Spatial fairness** (normalization) | Active taxis | **Mean** across time periods | Average availability reflects typical taxi presence |
| **Causal fairness** ($Y$, $g(d)$) | Pickup counts + Active taxis | **Mean** across time periods | g(d) is fitted on hourly means; Y = S/D must match this scale |

### 8.3 Trajectory Representation

Each trajectory consists of a sequence of states from passenger-seeking to pickup:

```
states[0]   → Starting location (beginning of passenger-seeking)
states[1:-1] → Intermediate passenger-seeking path
states[-1]  → Pickup location (the state that gets modified)
```

Each state contains four elements: `[x_grid, y_grid, time_bucket, day_index]`.

The `passenger_seeking_trajs_45-800.pkl` dataset uses a 4-element state vector, structured identically to the first 4 elements of the 126-element state vector in the `all_trajs.pkl` dataset.

---

## 9. Convergence and Constraints

### 9.1 Spatial Constraint ($\epsilon$)

The maximum perturbation per spatial dimension is bounded:

$$\|\delta\|_\infty \leq \epsilon$$

where $\epsilon$ defaults to 3.0 grid cells. This means a pickup can move at most 3 cells in the x-direction and 3 cells in the y-direction from its original location.

### 9.2 Iterative Convergence

Within a single trajectory's modification loop, convergence is declared when:

$$|\mathcal{L}^{(t)} - \mathcal{L}^{(t-1)}| < \theta$$

where $\theta = 10^{-4}$ is the convergence threshold.

### 9.3 Grid Boundary Enforcement

After each perturbation step, the modified pickup location is projected back to valid grid bounds:

$$x' = \text{clip}(x + \delta_x, 0, 47), \quad y' = \text{clip}(y + \delta_y, 0, 89)$$

---

## 10. Design Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Attribution method** | LIS + DCD (heuristic) | Computationally efficient and interpretable; directly tied to the two fairness terms being optimized |
| **Modification method** | Gradient-based (ST-iFGSM) | Directly optimizes the objective function; adapts perturbation direction to current fairness landscape |
| **Trajectory scope** | Edit pickups only | Pickup relocation has the most direct impact on spatial service distribution |
| **Differentiability** | Soft cell assignment | Bridges the discrete cell assignment gap; enables gradient flow to pickup coordinates |
| **$g(d)$ estimation** | Isotonic regression (demand-only baseline) | Non-parametric, monotone decreasing; fits observed demand-supply relationship without imposing a functional form. Planned revision to $g(d, \mathbf{x})$ with demographic features will require multivariate methods (see mathematical_foundations.md, Section 11.6) |
| **$F_{\text{causal}}$ naming** | Currently a misnomer | Measures unexplained inequality (demand-alignment), not true causal fairness. Current form preserved as baseline for ablation; planned revision with demographic conditioning will address this limitation |
| **$g(d)$ during optimization** | Frozen | Prevents circular optimization where g(d) adapts to modified data |
| **Leave-one-out attribution** | Rejected | Computationally infeasible: ~97 days per iteration for 44,000 trajectories |
| **Spatial mismatch attribution** | Rejected | Not directly tied to the fairness terms in the objective function; poor explainability |
| **Batch modification** | Sequential with global update | Captures interdependence — modifying one trajectory changes the fairness landscape for others |
