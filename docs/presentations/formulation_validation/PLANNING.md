# FAMAIL Formulation Validation Presentation — Planning Document

**Target Audience**: Statistician and mathematician reviewing the mathematical formulation for correctness and rigor.

**Goal**: Present the FAMAIL objective function, its component terms, the differentiability mechanism, and the optimization algorithm with enough mathematical detail for formal validation.

**Color Scheme**:
- Primary (Dark Red): `#A6192E` — headings, accents
- Secondary (Lt Gray): `#CDCDC8` — backgrounds, dividers
- Tertiary (Dark Teal): `#008080` — links, highlights, math annotations
- Charcoal: `#2D2828` — body text
- White: `#FFFFFF` — slide backgrounds
- Black: `#000000` — emphasis

**Image Assets** (relative paths from this directory):
- `../assets/FAMAIL_overview_card.png` — full-slide overview card
- `../assets/FAMAIL_icon.png` — small branding icon (header/corner)

---

## Slide Inventory

### Slide 1 — Title

- **Title**: FAMAIL: Formulation Validation
- **Subtitle**: Fairness-Aware Multi-Agent Imitation Learning
- **Content**: San Diego State University · Computational Transportation Research Group
- **Visual**: `FAMAIL_icon.png` centered or prominent

### Slide 2 — Project Overview Card

- **Full-slide image**: `FAMAIL_overview_card.png`
- **No additional text** — image is self-contained

### Slide 3 — Problem Statement

- 48×90 spatial grid over Shenzhen, China
- 50 expert taxi drivers, July 2016 GPS data
- Taxi service is spatially and causally unequal
- Goal: modify pickup locations to reduce inequality while preserving driver behavior
- Only pickups are modified; dropoffs remain fixed

### Slide 4 — Combined Objective Function

$$\mathcal{L}(\tau') = \alpha_1 \cdot F_{\text{spatial}} + \alpha_2 \cdot F_{\text{causal}} + \alpha_3 \cdot F_{\text{fidelity}}$$

- $\alpha_1 + \alpha_2 + \alpha_3 = 1$, default $(0.33, 0.33, 0.34)$
- Each term $\in [0, 1]$; therefore $\mathcal{L} \in [0, 1]$
- **Maximized**: higher $\mathcal{L}$ = fairer + more faithful
- **Code**: `trajectory_modification/objective.py` → `FAMAILObjective.forward()` (L459)

### Slide 5 — Spatial Fairness ($F_{\text{spatial}}$)

$$F_{\text{spatial}} = 1 - \tfrac{1}{2}\bigl(G_{\text{DSR}} + G_{\text{ASR}}\bigr)$$

- DSR = $N_c^{\text{pickup}} / A_c$, ASR = $N_c^{\text{dropoff}} / A_c$
- Pairwise Gini (differentiable):

$$G(\mathbf{x}) = \frac{\sum_i \sum_j |x_i - x_j|}{2n^2 \bar{x}}$$

- $F_{\text{spatial}} = 1$ ⟹ perfect equality
- Uses **SUM** aggregation for counts, **MEAN** for active taxis
- **Code**: `trajectory_modification/objective.py` → `compute_spatial_fairness()` (L247), `_pairwise_gini()` (L216)

### Slide 6 — Causal Fairness ($F_{\text{causal}}$)

$$F_{\text{causal}} = \max\!\bigl(0,\; R^2\bigr) = \max\!\left(0,\; 1 - \frac{\text{Var}(R)}{\text{Var}(Y)}\right)$$

- $Y_c = S_c / D_c$ (service ratio), $R_c = Y_c - g(D_c)$ (residual)
- $g(d)$: expected service ratio given demand, fitted via **isotonic regression** (monotone decreasing)
- $g(d)$ is **frozen** during optimization — no gradient flow through it
- Uses **MEAN** aggregation (must match $g(d)$ fitting scale)
- $F_{\text{causal}} = 1$ ⟹ all variance in $Y$ explained by demand
- **Code**: `trajectory_modification/objective.py` → `compute_causal_fairness()` (L320)
- **g(d) estimation**: `trajectory_modification/data_loader.py` → `GFunctionLoader.estimate_from_data()` (L370)

### Slide 7 — Fidelity ($F_{\text{fidelity}}$)

$$F_{\text{fidelity}} = f(\tau, \tau')$$

- Siamese LSTM (ST-SiameseNet): two shared-weight branches → similarity in $[0, 1]$
- Hidden dims: 200 → 100, sigmoid output
- Discriminator is **frozen** during modification (no parameter updates)
- Gradients flow *through* discriminator to modified trajectory features
- $f \approx 1$ ⟹ same-agent behavior preserved
- **Code**: `trajectory_modification/discriminator_adapter.py` → `DiscriminatorAdapter`
- **Code**: `trajectory_modification/objective.py` → `compute_fidelity()` (L413)

### Slide 8 — Soft Cell Assignment (Differentiability Bridge)

**Challenge**: Objective depends on discrete cell counts; hard assignment is non-differentiable.

**Solution**: Gaussian softmax over neighborhood:

$$\sigma_c(\mathbf{p};\, \tau) = \frac{\exp\!\bigl(-\|\mathbf{p} - \mathbf{c}\|^2 / 2\tau^2\bigr)}{\sum_{c' \in \mathcal{N}} \exp\!\bigl(-\|\mathbf{p} - \mathbf{c}'\|^2 / 2\tau^2\bigr)}$$

- Neighborhood: $(2k+1) \times (2k+1)$ cells, default $k=2$ (5×5)
- Temperature annealing: $\tau_t = \tau_{\max} \cdot (\tau_{\min} / \tau_{\max})^{t/T}$
- Creates differentiable path: location → soft probabilities → soft counts → objective → gradient
- **Code**: `objective_function/soft_cell_assignment/module.py` → `SoftCellAssignment.forward()` (L90)

### Slide 9 — Attribution: LIS + DCD

**Phase 1** selects which trajectories to modify.

**LIS** (Local Inequality Score — spatial):
$$\text{LIS}_c = \frac{|c - \mu|}{\mu}, \qquad \text{LIS}_\tau = \max(\text{LIS}_p, \text{LIS}_d)$$

**DCD** (Demand-Conditional Deviation — causal):
$$\text{DCD}_c = |Y_c - g(D_c)|, \qquad \text{DCD}_\tau = \text{DCD}_p$$

**Combined**: $\text{Combined}_\tau = w_{\text{LIS}} \cdot \widetilde{\text{LIS}}_\tau + w_{\text{DCD}} \cdot \widetilde{\text{DCD}}_\tau$ (tildes = normalized to $[0,1]$)

- Default weights: $w_{\text{LIS}} = w_{\text{DCD}} = 0.5$
- Top-$k$ selection (with optional spatial diversity penalty)
- **Code**: `trajectory_modification/dashboard.py` → `compute_cell_lis_scores()` (L215), `compute_cell_dcd_scores()` (L239), `compute_trajectory_attribution_scores()` (L297), `select_top_k_by_attribution()` (L411)

### Slide 10 — ST-iFGSM Algorithm

**Phase 2**: gradient-based modification of selected trajectories.

For each selected $\tau$, repeat for $t = 1, \ldots, T$:

1. Set $\mathbf{p}$ as differentiable tensor
2. Compute soft cell probabilities $\sigma_c(\mathbf{p};\, \tau_t)$
3. Compute differentiable pickup counts
4. Evaluate $\mathcal{L} = \alpha_1 F_{\text{spatial}} + \alpha_2 F_{\text{causal}} + \alpha_3 F_{\text{fidelity}}$
5. Backpropagate: $\nabla_\mathbf{p} \mathcal{L}$
6. Perturbation: $\delta = \text{clip}\bigl(\alpha \cdot \text{sign}(\nabla_\mathbf{p} \mathcal{L}),\; -\epsilon,\; \epsilon\bigr)$
7. Update: $\mathbf{p} \leftarrow \text{clip}(\mathbf{p} + \delta,\; \text{grid bounds})$

**Parameters**: $\alpha = 0.1$, $\epsilon = 3.0$, $T = 50$, convergence $\theta = 10^{-4}$

After each trajectory: update global counts, proceed to next.

- **Code**: `trajectory_modification/modifier.py` → `TrajectoryModifier.modify_single()` (L268), `_compute_soft_pickup_counts()` (L222)

### Slide 11 — Gradient Flow Diagram

Visual diagram showing the complete differentiable path:

```
pickup_location (requires_grad=True)
      ↓
soft cell probabilities σ_c(p; τ)
      ↓
differentiable pickup counts
      ↓
F_spatial + F_causal + F_fidelity
      ↓
L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity
      ↓
∇_p L  →  δ = clip(α · sign(∇L), -ε, ε)  →  p' = clip(p + δ, bounds)
```

- Soft cell assignment is the critical bridge enabling gradient flow
- $g(d)$ is frozen (no gradient); discriminator parameters frozen but gradients flow through

### Slide 12 — Summary & Discussion Points

- Three-term objective: spatial fairness, causal fairness, fidelity (all in $[0,1]$)
- Soft cell assignment makes discrete counts differentiable
- LIS + DCD selects high-impact trajectories; ST-iFGSM modifies them
- Key questions for validation:
  - Is the pairwise Gini formulation appropriate for measuring spatial equity?
  - Is $R^2$ with frozen $g(d)$ a sound measure of causal fairness?
  - Does temperature annealing converge correctly from soft to hard assignment?
  - Are the $\epsilon$-bounds sufficient to prevent unrealistic modifications?

---

## Code Cross-Reference Table

| Concept | File | Function/Class | Line |
|---------|------|----------------|------|
| Combined objective | `trajectory_modification/objective.py` | `FAMAILObjective.forward()` | L459 |
| Spatial fairness | `trajectory_modification/objective.py` | `compute_spatial_fairness()` | L247 |
| Pairwise Gini | `trajectory_modification/objective.py` | `_pairwise_gini()` | L216 |
| Causal fairness | `trajectory_modification/objective.py` | `compute_causal_fairness()` | L320 |
| Fidelity | `trajectory_modification/objective.py` | `compute_fidelity()` | L413 |
| Soft cell assignment | `objective_function/soft_cell_assignment/module.py` | `SoftCellAssignment.forward()` | L90 |
| ST-iFGSM loop | `trajectory_modification/modifier.py` | `TrajectoryModifier.modify_single()` | L268 |
| Soft pickup counts | `trajectory_modification/modifier.py` | `_compute_soft_pickup_counts()` | L222 |
| g(d) estimation | `trajectory_modification/data_loader.py` | `GFunctionLoader.estimate_from_data()` | L370 |
| LIS scoring | `trajectory_modification/dashboard.py` | `compute_cell_lis_scores()` | L215 |
| DCD scoring | `trajectory_modification/dashboard.py` | `compute_cell_dcd_scores()` | L239 |
| Trajectory attribution | `trajectory_modification/dashboard.py` | `compute_trajectory_attribution_scores()` | L297 |
| Top-k selection | `trajectory_modification/dashboard.py` | `select_top_k_by_attribution()` | L411 |
| Discriminator adapter | `trajectory_modification/discriminator_adapter.py` | `DiscriminatorAdapter` | — |
