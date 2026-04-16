# `tests/` — Test organization, running conventions, and fixture inventory

## Purpose

Verify mathematical correctness, guard against known bug classes, and confirm end-to-end
integration of the trajectory modification pipeline. Tests are written with the same rigor as
production code — "this is research code" is not an excuse for untested invariants.

---

## Files

| File | What it tests |
|---|---|
| `test_math_invariants.py` | Core mathematical properties of Gini, F_causal, hat matrices, attribution |
| `test_spatial_fairness.py` | `compute_fspatial` edge cases and gradient flow |
| `test_causal_fairness.py` | `compute_fcausal`, `compute_fcausal_torch`, attribution sum property |
| `test_hat_matrices.py` | Idempotency of M and (I-H); rank of H_demo |
| `test_g0_power_basis.py` | Power basis fit, isotonic diagnostic agreement |
| `test_aggregation.py` | 5-min bucket → hourly → block-mean aggregation correctness |
| `test_active_mask.py` | UnitIndexMap canonical ordering, roundtrip cell↔unit, inactive filter |
| `test_data_loader.py` | DataBundle.load() shape assertions, frozen dataclass, mass balance |
| `test_soft_cell_assignment.py` | Probs sum to 1, neighborhood boundary clamping, temperature annealing |
| `test_attribution.py` | Per-unit scores sum to 1-F_causal, inactive pickup scores 0 |
| `test_gradient_flow.py` | Non-zero non-NaN gradients, only t* slice carries gradient |
| `test_fidelity_model.py` | Forward pass shape, eval mode, parameter freeze |
| `test_fidelity_checkpoint.py` | Missing architecture_config raises; correct checkpoint loads |
| `test_fidelity_compute.py` | cuDNN workaround: backward succeeds in eval mode |
| `test_ms_context.py` | +1 coordinate offset, slot 0 gradient, seeking fill strategy |
| `test_ms_data.py` | MultiStreamData frozen, driver index access |
| `test_objective.py` | FAMAILObjective: single gather point, ALPHA_FIDELITY=0 skips discriminator |
| `test_modifier.py` | ST-iFGSM: epsilon-ball respected, mass balance, convergence |
| `test_modifier_integration.py` | Fixed-seed 5-iteration run: metrics improve/plateau, no NaN |
| `test_seeding.py` | set_all_seeds produces identical results across two runs |
| `test_trajectory.py` | TrajectoryState field access, coordinate indexing |
| `conftest.py` | `synthetic_bundle` fixture; `seeded` autouse fixture; `--run-slow` flag |
| `synthetic/fixtures.py` | `make_synthetic_bundle()` — in-memory DataBundle with known properties |

---

## Test categories

### Category 1: Mathematical invariants

Tests that verify properties that must hold by construction, regardless of data. Failures here
indicate a formula error — not a data or integration issue.

Examples:
- Equal DSR across all units → `Gini = 0`, `F_spatial = 1`
- One-hot DSR (one unit has all demand-service ratio) → `Gini -> (N-1)/N`
- `R` in span of `X_demo` → `F_causal = 0` (service fully explained by demographics)
- `R` orthogonal to `X_demo` → `F_causal = 1` (service not explained by demographics)
- `sum(per_unit_attribution) == 1 - F_causal` within `EPS`
- `(I-H)^2 == (I-H)` (idempotency)
- `M^2 == M` (idempotency)
- `rank(H_demo) == n_features + 1` (intercept included)

### Category 2: Bug-class regression guards

Tests that directly target the classes of bugs encountered during V2/V3 development. Each test
is a guard against a specific known failure mode.

Examples:
- Gradient flow through full pooled objective: `pickup_tensor.grad` is non-zero and non-NaN
- Gradient only flows through the `t*` slice: other slices of `soft_pickup_3d.grad` are zero
- LSTM backward succeeds in eval mode (cuDNN workaround active)
- Canonical unit ordering stable across two `DataBundle.load()` calls with different seeds
- Pickup in inactive unit scores 0 in attribution (not NaN, not error)
- Hat matrix shape matches unit count: `I_minus_H.shape == (N, N)`
- Missing `architecture_config` in checkpoint raises `ValueError` (not `KeyError` or silent wrong load)
- `+1` coordinate offset present in discriminator context inputs (guard against 0/1-indexed mismatch)

### Category 3: Integration tests

End-to-end tests that use the full `DataBundle` (marked `@pytest.mark.slow`) or a large
synthetic bundle that exercises the complete pipeline.

Examples:
- Fixed-seed 5-iteration convergence: `total` objective is non-decreasing across iterations
- Cross-trajectory baseline update: modifying trajectory A changes the objective seen by trajectory B
- Epsilon-ball respected: all pickup deltas after 50 iterations are within `EPSILON_BALL` grid cells
- Inactive-pickup trajectory skipped: no error, warning logged, history is empty list
- Mass balance: `pickup_3d.sum()` unchanged after any single-trajectory modification

---

## Running the tests

```bash
# Fast tests only (synthetic fixtures, < 10 seconds)
pytest famail_temporal/tests/

# Show which tests are running
pytest famail_temporal/tests/ -v

# Include slow tests (requires real DataBundle.load(), may take 1-2 minutes)
pytest famail_temporal/tests/ --run-slow

# Run a specific category
pytest famail_temporal/tests/ -k "invariant"
pytest famail_temporal/tests/ -k "gradient"
pytest famail_temporal/tests/ -m "slow" --run-slow

# Run a single file
pytest famail_temporal/tests/test_math_invariants.py -v
```

---

## Fixtures

**`conftest.py`** provides two fixtures and the `--run-slow` flag:

| Fixture | Scope | Description |
|---|---|---|
| `synthetic_bundle` | `session` | In-memory `DataBundle` with N=200 synthetic active units, T=4 blocks, 5 synthetic trajectories. No disk I/O. All tests that do not need real data use this fixture. |
| `seeded` | `function` (autouse) | Calls `set_all_seeds(42)` and sets `torch.backends.cudnn.deterministic = True` before every test. Ensures reproducibility without explicit seed management in individual tests. |

**`synthetic/fixtures.py`** defines `make_synthetic_bundle()`:

- Generates pickup/dropoff/active-taxis tensors with known statistical properties (e.g., one
  grid quadrant has systematically higher DSR to produce non-trivial Gini)
- Computes hat matrices and attribution scores analytically so that math invariant tests can
  compare against closed-form expected values
- Creates 5 synthetic `Trajectory` objects with pickups in known (cell, t_block) slots
- Returns a `DataBundle` with `discriminator=None` unless `include_discriminator=True` is
  specified (avoids requiring a checkpoint for fast tests)

Slow tests that need the real discriminator use `DataBundle.load()` directly and are decorated
with `@pytest.mark.slow`.

---

## Paper-section hook

The test suite is described in the **Reproducibility appendix** of the paper. Specific
mathematical invariant tests (Category 1) may be referenced in the Methods section as
"verified properties" alongside the formulas. The fixed-seed integration test provides the
primary evidence that the algorithm is deterministic across runs, which is a requirement for
reported metrics.
