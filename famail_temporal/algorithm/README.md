# `algorithm/` — ST-iFGSM trajectory modification orchestration

## Purpose

Orchestrate the three-term objective function and apply the Signed-Temporal Iterative
Fast Gradient Sign Method (ST-iFGSM) to modify taxi pickup locations. This module contains
the single grid-to-unit conversion point, the autograd-safe delta-tensor injection pattern,
the attribution-to-trajectory ranking pipeline, and the per-trajectory perturbation loop.

---

## Files

| File | Role |
|---|---|
| `objective.py` | `FAMAILObjective` — combines F_spatial, F_causal, F_fidelity; contains the sole `(48, 90, T) -> (N,)` conversion |
| `modifier.py` | `TrajectoryModifier` — ST-iFGSM loop over selected trajectories |
| `attribution.py` | `compute_per_unit_attribution`, `rank_trajectories`, `select_top_k` — maps per-unit scores to trajectory ranking |
| `soft_cell_assignment.py` | `SoftCellAssignment` — Gaussian softmax over neighborhood; `inject_soft_counts_into_3d` — delta-tensor injection |

---

## Key design choices

### 1. Single grid-to-unit conversion point

The masking operation `tensor_3d[mask_3d]` — which converts a `(48, 90, T)` grid tensor to an
`(N,)` active-unit vector — happens exactly once, at the top of `FAMAILObjective.forward()`:

```python
pickup_N       = soft_pickup_3d[mask_3d]
dropoff_N      = dropoff_3d[mask_3d]
active_taxis_N = active_taxis_3d[mask_3d]
```

`fairness/` functions never see grid dimensions. `fidelity/` functions never see N-vectors.
This architectural invariant makes each module independently testable and prevents the class of
bugs where a fairness function silently receives a full-grid tensor.

### 2. `pickup_N` carries gradient for both spatial and causal terms

Both `compute_fspatial` and `compute_fcausal_torch` receive `pickup_N` as an argument. Because
`soft_pickup_3d` is constructed via the delta-tensor pattern from a differentiable
`pickup_tensor`, the gradient flows:

```
pickup_tensor (x,y)
    -> SoftCellAssignment
    -> probs_2d
    -> delta tensor [:, :, t*]
    -> base_3d + delta  (autograd graph preserved)
    -> gather via mask_3d
    -> pickup_N
    -> F_spatial + F_causal
    -> weighted sum (total objective)
    -> backward
    -> pickup_tensor.grad
```

`g_0(D_N)` is computed under `torch.no_grad()` so that the demand-response baseline does not
contribute gradient — only the residual `R = Y - g_0(D)` drives the causal gradient.

### 3. Delta-tensor pattern for autograd-safe 3D injection

Direct in-place modification of a tensor breaks autograd. Instead, `SoftCellAssignment` and
`inject_soft_counts_into_3d` use the delta pattern:

```python
delta = torch.zeros_like(base_3d)          # no grad history
# scatter soft counts into delta[:, :, t*] via neighborhood loop
out = base_3d + delta                       # autograd-safe addition
```

Only `delta[:, :, t*]` is non-zero; the other time-block slices are unchanged by construction.
This avoids any in-place operations on tensors with `requires_grad=True`.

### 4. Per-trajectory attribution via pickup inheritance

Per-unit attribution scores `a_i` (from `fairness.causal.per_unit_attribution`) are
`(N,)` — one score per active `(cell, t)` unit. A trajectory inherits its score from the
active unit corresponding to its pickup cell and time block:

```python
cell, t_block = pickup_cell_and_time_block(trajectory)
unit_idx = unit_map.from_cell_time(cell, t_block)
score = per_unit[unit_idx] if unit_idx >= 0 else 0.0
```

Trajectories whose pickup falls in an inactive unit receive score `0.0` and are never selected.
The sum property `sum(a_i) = 1 - F_causal` is inherited by the trajectory scores in aggregate.

### 5. Cross-trajectory ordering semantics via shared `_base_pickup_3d`

`TrajectoryModifier` maintains a mutable `_base_pickup_3d` tensor that accumulates the
contributions of all modified trajectories. When trajectory k is being modified:

1. Trajectory k's original pickup contribution is subtracted from `_base_pickup_3d`
2. The soft assignment places k's pickup mass at the learned location
3. After convergence, `_base_pickup_3d` is updated to reflect k's final location

Trajectories modified earlier in the batch affect the objective seen by later trajectories.
This order-dependence is **intentional**: earlier modifications change the fairness landscape,
and later modifications respond to the updated state. Attribution scores are computed once before
any modifications (from the unmodified `_base_pickup_3d`) so that the selection order is stable.

### 6. Pickup-in-inactive-unit safeguard

At `modify_single()` entry, the soft-assignment neighborhood around the pickup is checked:
if no cell in the `(2k+1) x (2k+1)` window has an active unit in time block `t*`, the
trajectory is skipped with a logged warning. Trajectories selected by `select_top_k()` pass
this check trivially (they have positive attribution scores, which implies their pickup cell is
active). The safeguard handles edge cases from manual or test invocations.

### Mass balance under mean-hourly aggregation

Because `pickup_3d` stores mean-hourly rates (not raw counts), a single trajectory's pickup
contributes:

```
pickup_mass = 1.0 / (n_hours_per_block[t*] * n_days)
```

The modifier subtracts `pickup_mass` at the original cell and adds `pickup_mass * probs_2d`
via the soft distribution. Total mass is conserved: the mean-aggregated tensor's sum is
unchanged after any single-trajectory modification.

---

## Gradient flow diagram

```
pickup_tensor (x, y)  [requires_grad=True]
         |
         v
  SoftCellAssignment
  (Gaussian softmax, temperature tau)
         |
         v
    probs_2d  (2k+1, 2k+1), sums to 1
         |
         v
  inject_soft_counts_into_3d
  delta[:, :, t*] = probs_2d * pickup_mass
         |
         v
  soft_pickup_3d = base_3d + delta
  (autograd graph intact)
         |
         v
  gather via mask_3d
         |
         v
    pickup_N  (N,)
      /        \
     v          v
 F_spatial   F_causal
 (Gini DSR)  (R^2 Option B)
      \        /
       v      v
  alpha_s * F_s + alpha_c * F_c  (+  alpha_f * F_f if ALPHA_FIDELITY > 0)
         |
         v
       total  (scalar)
         |
         v
     backward()
         |
         v
  pickup_tensor.grad
         |
         v
  delta = clip(alpha * sign(grad), -eps, eps)
```

---

## API surface

```python
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.algorithm.modifier import TrajectoryModifier
from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution,
    rank_trajectories,
    select_top_k,
)
from famail_temporal.algorithm.soft_cell_assignment import (
    SoftCellAssignment,
    inject_soft_counts_into_3d,
)

# Build objective
obj = FAMAILObjective(bundle)

# Attribution pipeline
unit_scores = compute_per_unit_attribution(bundle)       # (N,) numpy array
ranked      = rank_trajectories(bundle, unit_scores)     # list of (score, traj_idx)
top_k       = select_top_k(ranked, k=10)                 # list of traj_idx

# Run modifier
modifier = TrajectoryModifier(bundle)
history  = modifier.modify_single(traj_idx, n_iterations=50)
# Returns list of dicts: [{total, f_spatial, f_causal, f_fidelity, delta_norm}, ...]

# Soft cell assignment (used internally by modifier)
sca   = SoftCellAssignment(neighborhood_size=5, tau=1.0)
probs = sca(pickup_x, pickup_y)  # (11, 11) probability distribution
```

---

## Dependencies

- `data/` — `DataBundle`, `UnitIndexMap`
- `fairness/` — `compute_fspatial`, `compute_fcausal_torch`, `per_unit_attribution`
- `fidelity/` — `compute_ffidelity`, `MultiStreamContextBuilder`
- `config.py` — all algorithm hyperparameters
- `utils/` — `set_all_seeds`, `Trajectory`
- Third-party: `torch`, `numpy`

---

## Paper-section hook

This module corresponds to the **"Algorithm"** section of the Methods. The ST-iFGSM loop
(Section 8.3 of the spec) is the primary algorithmic contribution. The attribution pipeline
(Section 8.4) and the pickup-in-inactive-unit safeguard appear in the Results section when
discussing which trajectories were selected for modification and why. The gradient flow diagram
above can be adapted as a figure in the paper.
