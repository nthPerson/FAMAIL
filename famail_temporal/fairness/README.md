# `fairness/` — Pooled fairness metrics over active `(cell, t)` units

## Purpose

Compute `F_spatial` (demand-service equity via Gini coefficient) and `F_causal` (demographic
alignment via Option B R^2 projection) over the N-vector of active `(cell, t)` units. Also
exposes the per-unit attribution decomposition that maps `1 - F_causal` back to individual units,
which drives trajectory selection.

---

## Files

| File | Role |
|---|---|
| `spatial.py` | `pairwise_gini()` + `compute_fspatial()` — differentiable Gini over N-vectors |
| `causal.py` | `compute_fcausal()` (numpy), `compute_fcausal_torch()`, `per_unit_attribution()`, `per_unit_attribution_signed()` |
| `hat_matrices.py` | `precompute_hat_matrices()` — builds demographic hat matrix and centering matrix from active-unit demographics |
| `g0_power_basis.py` | `G0Function` dataclass + `fit_g0()` — power-basis regression and isotonic diagnostic |

---

## Key design choices

### 1. N-vector inputs only — no grid geometry

All functions in this module receive vectors of length N (one element per active unit). The
mapping from the `(48, 90, T)` grid to the N-vector happens exactly once, in
`algorithm/objective.py::forward()`. Keeping fairness code grid-unaware means it can be tested
against synthetic N-vectors without any grid infrastructure, and the same math applies regardless
of how N is composed from spatial and temporal dimensions.

### 2. Pooled Gini (single Gini over all N units)

`F_spatial = 1 - 0.5 * (Gini(DSR) + Gini(ASR))` where `DSR = pickup_N / active_taxis_N` and
`ASR = dropoff_N / active_taxis_N`. The Gini is computed over all N active units simultaneously —
not per time-block and averaged. Pooling ensures that time-blocks with many active units carry
more weight, which reflects their larger contribution to total service exposure. The pairwise
formula `G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mu)` is used because it is differentiable
everywhere except when two units have equal values (a measure-zero event during optimization).

### 3. Option B as the sole causal formulation

Option B (demographic hat-matrix projection) is the only causal formulation in this module.
Earlier versions of the codebase had Options A–D behind a string dispatch; Option B was selected
as the single published formulation because:

- It is grounded in the Frisch-Waugh-Lovell partial regression theorem
- It produces an R^2-style score directly interpretable as proportion of variance explained
- It admits the per-unit attribution sum property (see below)

The formula is:

```
D  = max(demand_N, DEMAND_FLOOR)
Y  = supply_N / D
R  = Y - g_0(D)
F_causal = R' (I - H_demo) R / R' M R
```

where `H_demo = X_demo (X_demo' X_demo)^-1 X_demo'` and `M = I - 11'/N`.

### 4. Per-unit attribution decomposition with sum property

Under Option B, `1 - F_causal = r^2_demo` admits the exact decomposition:

```
r^2_demo = sum_i [(M R)_i^2 - ((I-H) R)_i^2] / R' M R
```

Each term is unit `i`'s contribution to demographic-explained variance. The sum property holds
because both M and (I - H) are idempotent. This is a publishable property: the attribution scores
are not heuristic weights but a mathematically exact partition of the causal fairness deficit.

`per_unit_attribution_signed()` multiplies each score by `sign((H R)_i)` to distinguish units
where service exceeds the demographic prediction (positive, reducing the deficit) from those
where it falls short (negative, worsening the deficit).

### 5. `g_0(D)` — power basis with isotonic diagnostic

The power basis `[1, 1/(D+1), 1/sqrt(D+1), sqrt(D+1)]` is fit by ordinary least squares to
approximate `Y ~ a/D`. This form is used (rather than splines or kernels) because:

- The four parameters are interpretable
- The fitted function is a linear combination of the basis, so `g_0(D_N)` is a numpy dot product
  at inference time — no surrogate model required
- The hat-matrix algebra in `compute_fcausal` requires `g_0` to be computed outside the
  gradient tape (`torch.no_grad()`) to avoid double-counting the demand term

An isotonic (monotone) regression is also fit during preprocessing as a diagnostic. If the two
fits disagree by more than a configurable tolerance, preprocessing raises a warning: the power
basis may not capture the true relationship and the causal score may be unreliable.

---

## API surface

```python
from famail_temporal.fairness.spatial import compute_fspatial, pairwise_gini
from famail_temporal.fairness.causal import (
    compute_fcausal,
    compute_fcausal_torch,
    per_unit_attribution,
    per_unit_attribution_signed,
)
from famail_temporal.fairness.hat_matrices import precompute_hat_matrices
from famail_temporal.fairness.g0_power_basis import G0Function, fit_g0

# Spatial fairness (torch tensors, differentiable)
f_spatial, breakdown = compute_fspatial(pickup_N, dropoff_N, active_taxis_N)
# Returns: f_spatial in [0, 1], breakdown dict with DSR, ASR, Gini values

# Causal fairness (numpy, for preprocessing / attribution)
f_causal, breakdown = compute_fcausal(pickup_N, supply_N, g0_func, I_minus_H, M)

# Causal fairness (torch, for gradient flow during ST-iFGSM)
f_causal_t, _ = compute_fcausal_torch(pickup_N_t, supply_N_t, g0_D_N, I_minus_H_t, M_t)

# Per-unit attribution
scores = per_unit_attribution(R, I_minus_H, M)           # (N,) >= 0, sums to 1 - F_causal
signed = per_unit_attribution_signed(R, H_demo, I_minus_H, M)  # (N,) signed

# Hat-matrix precomputation (called by preprocess.py)
matrices = precompute_hat_matrices(demo_matrix_N_by_F)
# Returns dict with keys: 'I_minus_H_demo', 'M', 'H_demo', 'rank'

# g_0 fitting (called by preprocess.py)
g0 = fit_g0(demand_N, supply_ratio_N)  # returns G0Function
y_hat = g0(demand_N)                   # evaluates on new demand values
```

---

## Dependencies

- `config.py` — `DEMAND_FLOOR`, `EPS`, `DEMOGRAPHIC_FEATURES`
- No other `famail_temporal/` imports

Third-party: `numpy`, `torch`, `scikit-learn` (isotonic regression in `g0_power_basis.py`).

---

## Paper-section hook

This module corresponds to the **"Fairness Metrics"** subsection of the Methods section.
`F_spatial` and `F_causal` are the primary evaluation metrics. The per-unit attribution
decomposition (sum property) will appear in a Results subsection explaining which
neighborhoods/time-blocks drive the fairness deficit. The isotonic diagnostic may appear in
an appendix on model diagnostics.
