# Whole-Trajectory Editing: Investigation & Design

## 1. Motivation

The current ST-iFGSM algorithm modifies only the **pickup location** (the final state) of each trajectory. However, every intermediate state in the trajectory represents a taxi being physically present in a grid cell during a time period. The `active_taxis` dataset is built from exactly this kind of GPS presence data — any taxi with at least one GPS reading in a cell's neighborhood during a time period is counted as "active" there.

This means that the **entire trajectory path** — not just the pickup — contributes to the spatial distribution of taxi service supply. Modifying intermediate states could:

- **Redistribute supply** more broadly across underserved cells (affecting DSR/ASR in spatial fairness)
- **Increase the gradient signal** by giving the optimizer many more degrees of freedom
- **Better reflect reality**: a driver who reroutes through underserved neighborhoods provides genuine service presence, not just a relocated pickup point

## 2. Current Architecture Summary

### 2.1 What Gets Modified Today

| Component | Current Scope | File | Key Lines |
|-----------|--------------|------|-----------|
| **Perturbation target** | `states[-1]` only (pickup) | `modifier.py` | L302-313 |
| **Perturbation vector** | 2D: `[δx, δy]` | `modifier.py` | L309 |
| **Soft cell assignment** | Single point → neighborhood probabilities | `soft_cell_assignment/module.py` | L90-130 |
| **Pickup count update** | Subtracts old cell, adds soft probs | `modifier.py` | L222-266 |
| **Trajectory tensor** | `[seq_len, 4]` with `[x, y, time, day]` | `trajectory.py` | L88-92 |
| **Discriminator input** | Full trajectory `[batch, seq_len, 4]` | `model.py` | L325-352 |

### 2.2 Gradient Flow Path (Current)

```
pickup_location (requires_grad=True)
    │
    ▼
SoftCellAssignment.forward()  →  soft_probs [1, 5, 5]
    │
    ▼
_compute_soft_pickup_counts()  →  soft_pickup_counts [48, 90]
    │
    ▼
FAMAILObjective.forward()
    ├── compute_spatial_fairness()  →  Gini(DSR), Gini(ASR)
    ├── compute_causal_fairness()  →  R² or hat-matrix formulations
    └── compute_fidelity()         →  discriminator(τ, τ')
    │
    ▼
total.backward()  →  ∂L/∂pickup_location
```

**Key observation**: Gradients currently flow from the objective back to the pickup location via soft cell assignment. Only `pickup_counts` are made differentiable; `active_taxis` and `dropoff_counts` remain fixed tensors.

### 2.3 What Fairness Metrics Use

| Metric | Numerator (modifiable) | Denominator (fixed) |
|--------|----------------------|---------------------|
| **DSR** (Departure Service Rate) | `pickup_counts[x,y]` | `active_taxis[x,y]` |
| **ASR** (Arrival Service Rate) | `dropoff_counts[x,y]` | `active_taxis[x,y]` |
| **Causal Y = S/D** | `supply[x,y]` | `demand[x,y]` |

With whole-trajectory editing, intermediate states could influence **both numerators and denominators** — a taxi's presence in a cell is both "supply" (active taxi) and indirectly affects future pickup potential.

## 3. Design Space for Whole-Trajectory Editing

### 3.1 What "Intermediate State" Means

Each trajectory is a sequence of states `[s₀, s₁, ..., sₙ₋₁, sₙ]` where:
- `s₀` through `sₙ₋₁`: **Passenger-seeking path** — the taxi is driving through cells looking for a fare
- `sₙ`: **Pickup location** — where the taxi picks up a passenger

Each state `sᵢ = (x_grid, y_grid, time_bucket, day_index)`.

A taxi at state `sᵢ` represents service supply in cell `(x_grid, y_grid)` at time `time_bucket`. This directly maps to what `active_taxis` measures: "how many taxis were present in this cell's neighborhood during this time period."

### 3.2 Three Candidate Approaches

#### Approach A: Full Unconstrained Editing

**Concept**: Make all states `s₀, ..., sₙ` independently modifiable with their own perturbation vectors.

**Perturbation vector**: `δ ∈ ℝ^{(n+1) × 2}` — a 2D displacement per state.

```python
# Instead of:
cumulative_delta = np.zeros(2)        # [δx, δy] for pickup only

# We have:
cumulative_delta = np.zeros((seq_len, 2))  # [δx, δy] per state
```

**Pros**:
- Maximum flexibility and gradient signal
- Can redistribute supply across many cells simultaneously

**Cons**:
- **Spatial coherence**: Independent perturbations can create physically impossible trajectories (taxi teleporting between distant cells)
- **Temporal constraint**: Perturbations change the cell a taxi is in, but the time bucket is fixed — a taxi can only move ~1 cell per 5-minute interval at typical speeds
- **Optimization difficulty**: Much higher-dimensional search space (2N vs 2)
- **Discriminator sensitivity**: LSTM-based discriminator processes the full sequence — major path deviations will tank fidelity

#### Approach B: Anchored Backward Propagation

**Concept**: Modify the pickup location as today, then **back-propagate** spatial modifications to earlier states using interpolation, creating a smooth rerouted path.

```
Original:    s₀ → s₁ → s₂ → ... → sₖ → sₖ₊₁ → ... → sₙ
Modified:    s₀ → s₁ → s₂ → ... → sₖ → sₖ₊₁' → ... → sₙ'
                                    ▲           ▲         ▲
                              anchor point   interpolated  pickup (modified)
```

The `Trajectory.interpolate_to_pickup()` method already exists and does exactly this for the last few states. This approach would:
1. Modify `sₙ` (pickup) as usual using gradient-based optimization
2. Choose an anchor point `sₖ` (the earliest state that should remain fixed)
3. Linearly interpolate intermediate states between `sₖ` and `sₙ'`

**Pros**:
- Maintains spatial coherence (smooth path)
- Minimal changes to the optimizer — the gradient target is still 2D
- The existing `interpolate_to_pickup()` method provides a template
- Naturally bounded perturbation — earlier states change less

**Cons**:
- Intermediate state modifications are not directly gradient-guided
- Limited to linear interpolation (real taxi paths are non-linear)
- The anchor point choice is a hyperparameter

#### Approach C: Differentiable Whole-Path Optimization

**Concept**: Make **all** state positions differentiable simultaneously, but add a **path smoothness regularizer** to maintain spatial coherence.

```python
# All states as a differentiable tensor
trajectory_positions = torch.tensor(
    [[s.x_grid, s.y_grid] for s in trajectory.states],
    requires_grad=True
)  # Shape: [seq_len, 2]
```

The objective becomes:

```
L' = α₁·F_spatial(soft_counts_all_states) + α₂·F_causal + α₃·F_fidelity + λ·R_smooth
```

Where `R_smooth` is a path smoothness regularizer:

```
R_smooth = -(1/T) Σᵢ ‖sᵢ' - sᵢ₋₁'‖² / ‖sᵢ - sᵢ₋₁‖²
```

This penalizes deviations from the original inter-state distances, encouraging smooth rerouting rather than teleportation.

**Pros**:
- Fully gradient-guided — every state modification is informed by the objective
- Path coherence is enforced via the regularizer (tunable via λ)
- Soft cell assignment naturally extends to multiple points
- The discriminator already processes the full modified sequence — gradients through fidelity automatically penalize unrealistic modifications

**Cons**:
- Most complex to implement
- Higher computational cost per iteration (soft cell assignment for every state)
- Temperature annealing and convergence dynamics change significantly
- Need to balance the smoothness regularizer weight λ

## 4. Recommended Approach: C (Differentiable Whole-Path Optimization)

Approach C is the strongest candidate because:

1. **It stays true to the project's design philosophy**: FAMAIL already invested in making the pipeline end-to-end differentiable (soft cell assignment, differentiable Gini, differentiable R²). Extending this to all states is the natural next step.

2. **The discriminator provides a built-in coherence check**: The LSTM-based SiameseLSTM discriminator processes the full `[seq_len, 4]` trajectory. If intermediate states are perturbed unrealistically, the fidelity score will drop, automatically constraining the optimizer. This is a more principled guard than ad-hoc interpolation (Approach B).

3. **The soft cell assignment module already supports batched computation**: The `compute_soft_counts()` and `compute_soft_counts_vectorized()` functions in `soft_cell_assignment/module.py` already handle multiple points simultaneously. They accept `[num_traj, ns, ns]` probability tensors and aggregate them to the grid.

4. **Approach B, while simpler, is fundamentally limited**: Back-propagated interpolation is not gradient-informed for intermediate states, so the optimizer cannot discover that routing through a specific underserved cell would improve the objective. It also cannot deviate from the linear path between anchor and pickup — real rerouting might involve detours.

### 4.1 What Changes Are Required

Below is an inventory of every component that needs modification, with specific details on what changes.

#### 4.1.1 `trajectory.py` — Trajectory Representation

**New method: `apply_whole_perturbation()`**

```python
def apply_whole_perturbation(
    self,
    delta: np.ndarray,  # Shape: [seq_len, 2]
    grid_dims: Tuple[int, int] = (48, 90),
) -> 'Trajectory':
    """Apply per-state perturbations to the entire trajectory."""
    modified = self.clone()
    for i, state in enumerate(modified.states):
        new_x = np.clip(state.x_grid + delta[i, 0], 0, grid_dims[0] - 1)
        new_y = np.clip(state.y_grid + delta[i, 1], 0, grid_dims[1] - 1)
        modified.states[i] = TrajectoryState(
            x_grid=new_x, y_grid=new_y,
            time_bucket=state.time_bucket,
            day_index=state.day_index,
        )
    return modified
```

**Note**: Time bucket and day index are **never modified** — they are intrinsic temporal labels, not spatial positions. A taxi's position can be rerouted, but time cannot be rewound.

#### 4.1.2 `modifier.py` — Core Optimization Loop

This is where the most significant changes occur.

**Key changes to `modify_single()`**:

1. **Perturbation vector**: Change from `np.zeros(2)` to `np.zeros((seq_len, 2))`.

2. **Differentiable trajectory tensor**: Instead of creating a single `pickup_tensor` with `requires_grad=True`, create a tensor for all positions:

   ```python
   all_positions = torch.tensor(
       [[s.x_grid, s.y_grid] for s in modified.states],
       dtype=torch.float32, device=device, requires_grad=True
   )  # Shape: [seq_len, 2]
   ```

3. **Soft count computation**: Extend `_compute_soft_pickup_counts()` to handle all states. Each state contributes soft probability mass to the supply grid. The existing `compute_soft_counts_vectorized()` function can be used for this:

   ```python
   # For each state, compute soft assignment probabilities
   original_cells = torch.tensor(
       [[int(s.x_grid), int(s.y_grid)] for s in trajectory.states],
       device=device
   )  # [seq_len, 2]

   # Get soft assignments for all states
   all_probs = self.soft_assign(all_positions, original_cells.float())
   # Shape: [seq_len, ns, ns]

   # Aggregate to grid-level soft supply counts
   soft_supply = compute_soft_counts_vectorized(
       all_probs, original_cells, self.grid_dims, base_supply_counts
   )
   ```

4. **Separate supply vs. pickup counts**: Currently only pickup counts are made differentiable. With whole-trajectory editing, we need to distinguish:
   - **Soft supply counts** (from ALL states) → affects `active_taxis` / supply in fairness
   - **Soft pickup counts** (from final state only) → affects `pickup_counts` in DSR

5. **Path smoothness regularizer**: Add a new term to the objective:

   ```python
   def compute_path_smoothness(positions, original_positions):
       """Penalize deviation from original inter-state distances."""
       # Original consecutive distances
       orig_diffs = original_positions[1:] - original_positions[:-1]
       orig_dists = (orig_diffs ** 2).sum(dim=-1) + eps

       # Modified consecutive distances
       mod_diffs = positions[1:] - positions[:-1]
       mod_dists = (mod_diffs ** 2).sum(dim=-1)

       # Ratio of modified to original distances (1.0 = unchanged)
       ratios = mod_dists / orig_dists
       # Penalize deviation from 1.0
       smoothness = 1.0 - (ratios - 1.0).pow(2).mean()
       return torch.clamp(smoothness, 0.0, 1.0)
   ```

6. **Per-state ε-ball constraint**: Each state should have its own ε-ball centered on its original position, but potentially with a smaller radius for earlier states (reflecting that earlier path segments are less relevant to pickup fairness):

   ```python
   # State-dependent epsilon: smaller for earlier states, larger near pickup
   state_epsilon = epsilon * (0.3 + 0.7 * (i / (seq_len - 1)))
   ```

   This creates an "epsilon cone" — states near the pickup can deviate more than states at the start of the trajectory.

7. **Gradient application**: The sign-gradient step becomes matrix-valued:

   ```python
   # ST-iFGSM for whole trajectory
   if grad_norm > 1e-8:
       delta = alpha * np.sign(grad)  # [seq_len, 2]
       cumulative_delta = np.clip(
           cumulative_delta + delta,
           -state_epsilons[:, None],  # [seq_len, 1]
           state_epsilons[:, None],
       )
   ```

#### 4.1.3 `objective.py` — FAMAILObjective

**New forward signature** to accept soft supply counts:

```python
def forward(
    self,
    pickup_counts: torch.Tensor,
    dropoff_counts: torch.Tensor,
    supply: torch.Tensor,
    active_taxis: torch.Tensor,
    tau_features: Optional[torch.Tensor] = None,
    tau_prime_features: Optional[torch.Tensor] = None,
    causal_demand: Optional[torch.Tensor] = None,
    soft_active_taxis: Optional[torch.Tensor] = None,  # NEW
    path_smoothness: Optional[torch.Tensor] = None,     # NEW
    alpha_smoothness: float = 0.0,                       # NEW
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
```

If `soft_active_taxis` is provided, use it instead of the fixed `active_taxis` for DSR/ASR computation in spatial fairness. This makes the denominator differentiable too:

```python
# In compute_spatial_fairness:
taxis = soft_active_taxis if soft_active_taxis is not None else active_taxis
dsr[active_mask] = pickup_counts[active_mask] / taxis[active_mask]
```

The path smoothness term gets folded into the combined objective:

```
L' = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity + α₄·R_smooth
```

#### 4.1.4 `soft_cell_assignment/module.py` — Batched Soft Assignment

The existing `compute_soft_counts_vectorized()` already handles batched assignment. The main change is that the `SoftCellAssignment.forward()` needs to handle `[seq_len, 2]` inputs efficiently rather than `[1, 2]`.

This already works — the module accepts `[batch, 2]` inputs. We just need to call it with `batch = seq_len` instead of `batch = 1`.

#### 4.1.5 `trajectory_modification/data_loader.py` — DataBundle

**No structural changes required**. The `DataBundle` already carries `active_taxis_grid` which serves as the base supply counts. For whole-trajectory editing, the modifier would:
1. Start with `base_supply = active_taxis_grid.clone()`
2. Subtract the current trajectory's hard supply contribution
3. Add the soft supply from all modified states

However, a new utility method could be useful:

```python
def compute_trajectory_supply_contribution(
    self,
    trajectory: Trajectory,
) -> np.ndarray:
    """Compute this trajectory's contribution to the supply grid."""
    supply = np.zeros(self.grid_dims, dtype=np.float32)
    for state in trajectory.states:
        x, y = int(state.x_grid), int(state.y_grid)
        if 0 <= x < self.grid_dims[0] and 0 <= y < self.grid_dims[1]:
            supply[x, y] += 1
    return supply
```

### 4.2 Interaction with the Discriminator (Fidelity Term)

The discriminator is a **critical natural constraint** for whole-trajectory editing. Here's why it works well:

1. **The discriminator processes full trajectories**: `SiameseLSTMDiscriminator.forward()` takes `[batch, seq_len, 4]` tensors — it sees every state in the trajectory.

2. **The LSTM is sequence-sensitive**: The shared bidirectional LSTM encoder processes states in order, so physically implausible jumps (a taxi teleporting from one side of the city to another) would produce hidden states that differ significantly from the original trajectory's embedding.

3. **Gradients flow back through `tau_prime_features`**: In `compute_fidelity()`, `torch.enable_grad()` is explicitly used. If we make `tau_prime_features` depend on the differentiable `all_positions` tensor, gradients from fidelity will naturally discourage unrealistic modifications.

**Implementation detail**: Currently `tau_prime_features` is built from `modified.to_tensor()`, which converts states to a numpy array and then to a tensor — breaking the gradient chain. For whole-trajectory editing, we need to build `tau_prime_features` directly from the differentiable `all_positions` tensor:

```python
# Build tau_prime_features with gradient connection
time_day = torch.tensor(
    [[s.time_bucket, s.day_index] for s in trajectory.states],
    dtype=torch.float32, device=device
)  # [seq_len, 2] — fixed, no gradients

tau_prime_features = torch.cat([all_positions, time_day], dim=-1)
# [seq_len, 4] — positions carry gradients, time/day are fixed
tau_prime_features = tau_prime_features.unsqueeze(0)  # [1, seq_len, 4]
```

This preserves the gradient chain from objective → discriminator → positions.

### 4.3 Impact on Each Fairness Term

#### Spatial Fairness (F_spatial)

**Current**: Only `pickup_counts` changes as the pickup is moved.

**With whole-trajectory editing**: Two effects:
1. **pickup_counts** still changes via the final state (pickup location)
2. **active_taxis** (the DSR/ASR denominator) becomes differentiable through intermediate states — taxis being "rerouted" through underserved cells increases supply there

This is particularly powerful because the Gini coefficient measures inequality of DSR = pickups/active_taxis. Currently, only the numerator is optimized. With whole-trajectory editing, both numerator and denominator can be jointly optimized, potentially achieving much stronger fairness improvements.

#### Causal Fairness (F_causal)

**Current**: Uses fixed `causal_supply` and `causal_demand` grids.

**With whole-trajectory editing**: The supply grid could be made partially differentiable. When a trajectory's intermediate states shift, the supply in affected cells changes. This allows the optimizer to better align supply with demand (the core of causal fairness) by routing taxis through high-demand areas.

However, the causal term's sensitivity to supply changes may be lower than spatial fairness, because:
- `Y = S/D` is computed per-cell with mean aggregation
- One trajectory's supply contribution is tiny relative to the total
- The g(d) function is frozen (no gradients through demand → expected ratio)

**Recommendation**: Start with spatial fairness only receiving soft supply counts. Add causal supply differentiability as a second-phase enhancement if spatial results are promising.

#### Fidelity (F_fidelity)

**Enhanced**: The discriminator now provides gradients for **all** states, not just the pickup. This means:
- States that the LSTM encoder is particularly sensitive to (e.g., early states that set the hidden state direction) will have stronger gradient signals against modification
- The discriminator naturally acts as a "physical plausibility" constraint for the whole path

## 5. Computational Considerations

### 5.1 Cost per Iteration

| Operation | Current (pickup only) | Whole trajectory |
|-----------|----------------------|-----------------|
| Soft cell assignment | 1 point × 5×5 neighborhood | N points × 5×5 neighborhood |
| Gradient computation | 2 parameters | 2N parameters |
| Count aggregation | 1 scatter | N scatters (or vectorized) |
| Discriminator forward | Same | Same (already processes full seq) |

For a typical trajectory with ~20 states, computational cost per iteration increases roughly 20×. However:
- The vectorized `compute_soft_counts_vectorized()` amortizes the aggregation cost
- The discriminator cost is unchanged (it already processes the full sequence)
- The bottleneck is likely the pairwise Gini computation (O(n²) in number of cells), which is independent of trajectory length

### 5.2 Convergence Implications

With 2N parameters instead of 2, the optimization landscape is higher-dimensional. This may require:
- **More iterations** for convergence (increase `max_iterations`)
- **Smaller step size** `α` to avoid overshooting
- **Lower initial temperature** for soft cell assignment (to keep assignments focused)
- **Gradient clipping** to prevent individual states from dominating the update

### 5.3 Memory

Each trajectory's soft cell computation produces a `[seq_len, ns, ns]` probability tensor. For `ns=5` and `seq_len=20`, this is `20 × 5 × 5 = 500` floats — negligible. The main memory cost is storing the computation graph for backpropagation, which grows linearly with sequence length.

## 6. Implementation Roadmap

### Phase 1: Infrastructure (Minimal Changes)

1. Add `apply_whole_perturbation()` to `Trajectory` class
2. Add `compute_trajectory_supply_contribution()` to `DataBundle`
3. Add path smoothness regularizer as a standalone function
4. Add `soft_active_taxis` parameter to `FAMAILObjective.forward()`

### Phase 2: Modifier Core

5. Extend `TrajectoryModifier` with a `modification_scope` parameter (`'pickup_only'` | `'whole_trajectory'`)
6. Implement `_compute_soft_supply_counts()` for all-state soft assignment
7. Modify `modify_single()` to handle whole-trajectory perturbation vectors
8. Build `tau_prime_features` with gradient chain preserved

### Phase 3: Regularization & Tuning

9. Implement the epsilon cone (state-dependent perturbation bounds)
10. Add path smoothness regularizer with configurable `alpha_smoothness`
11. Tune hyperparameters: `α`, `ε`, `λ`, temperature schedule
12. Add dashboard controls for whole-trajectory editing parameters

### Phase 4: Evaluation

13. Compare pickup-only vs. whole-trajectory editing on fairness metrics
14. Verify fidelity scores remain acceptable
15. Visualize trajectory modifications (before/after paths on grid)
16. Measure computational overhead

## 7. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Unrealistic trajectories (teleportation) | Fidelity drops, trajectories become useless for IL training | Discriminator constraint + path smoothness regularizer + epsilon cone |
| Gradient vanishing through long sequences | Optimization stalls, only near-pickup states get modified | Use bidirectional LSTM gradients (already in place); consider gradient scaling |
| Computational blowup | Iterations become too slow for interactive dashboard | Vectorized soft counts; limit scope to last K states as a compromise |
| Objective landscape becomes too complex | Poor convergence, oscillation | Reduce step size, increase iterations, use momentum |
| Supply count perturbation too small to matter | No measurable fairness improvement from intermediate states | Analyze actual supply contribution of one trajectory vs. global counts |

## 8. Key Open Question: Signal-to-Noise Ratio

The most critical question is whether modifying one trajectory's intermediate states produces a **large enough change** in the supply grid to meaningfully affect fairness metrics.

Consider: The `active_taxis` grid is aggregated over all 50 drivers across all time periods. A single trajectory has ~20 states. Moving those 20 states changes supply in ~20 cells by ±1 each. The typical `active_taxis` value per cell (hourly, with 5×5 neighborhood) is on the order of 5-15 taxis.

So each trajectory modification changes supply by roughly **±7-20%** in affected cells. This is non-trivial and comparable to the effect of moving a pickup, which changes pickup counts by ±1 in a cell that might have 0-5 pickups.

**Conclusion**: The signal should be sufficient, especially for cells with low taxi presence where a ±1 supply change represents a large relative shift.

## 9. Alternative: Scope-Limited Editing (Last K States)

As a pragmatic middle ground, consider modifying only the **last K states** (e.g., K=5-10) rather than the full trajectory. This:
- Focuses optimization on the most relevant part of the path (approach to pickup)
- Reduces computational cost proportionally
- Is easier to keep spatially coherent (fewer states to constrain)
- Still provides richer signal than pickup-only editing

This could be implemented as a `modification_depth` parameter in `TrajectoryModifier`:

```python
class TrajectoryModifier:
    def __init__(self, ..., modification_depth: int = 1):
        """
        modification_depth: Number of states to modify from the end.
            1 = pickup only (current behavior)
            N = last N states
            -1 = all states
        """
```

This provides backward compatibility (depth=1) while enabling incremental exploration.
