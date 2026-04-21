# `famail_temporal/` — FAMAIL Temporal: Fairness-Aware Trajectory Modification

`famail_temporal/` is a ground-up, self-contained rewrite of the FAMAIL trajectory modification
algorithm with explicit temporal granularity. It extends fairness metrics to the `(cell,
time_block)` level — distinguishing morning-peak underservice from evening-peak underservice in
the same neighborhood — while preserving the soft-cell-assignment gradient flow and the
ST-iFGSM perturbation design. The module is written to serve as the paper-replication codebase:
concise, documented, and free of external dependencies.

---

## Objective function

```
L = alpha_spatial * F_spatial + alpha_causal * F_causal + alpha_fidelity * F_fidelity
```

All three terms are in [0, 1] where **higher is always better**:

| Term | Measures | Formula |
|---|---|---|
| `F_spatial` | Equitable demand-service ratio across `N` active units | `1 - 0.5 * (Gini(DSR) + Gini(ASR))` |
| `F_causal` | Service alignment with demand (not demographics) | `R'(I-H_demo)R / R'MR` — Option B |
| `F_fidelity` | Realism of modified trajectories | Pre-trained Multi-Stream Siamese discriminator |

Default weights: `alpha_spatial = alpha_causal = 0.33`, `alpha_fidelity = 0.34`. Set
`ALPHA_FIDELITY = 0` in `config.py` to run fairness-only experiments without loading the
discriminator checkpoint.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r famail_temporal/requirements.txt

# 2. Provision source_data/ — one of:
#    (a) Run the unified source-generation tool on raw GPS:
#        python -m famail_temporal.data.source_generation \
#            --input-dir raw_data/ --output-dir famail_temporal/source_data/
#    (b) Copy pre-built files in from the repo-root source_data/ directory.
#    See famail_temporal/source_data/README.md for the full file list.
cp source_data/cell_demographics.pkl        famail_temporal/source_data/
cp source_data/grid_to_district_mapping.pkl famail_temporal/source_data/

# 3. Run preprocessing (one-time; writes to cache/)
python -m famail_temporal.preprocess

# 4. Run the fast test suite (should pass in < 10 seconds)
pytest famail_temporal/tests/

# 5. Run all tests including slow integration tests
pytest famail_temporal/tests/ --run-slow
```

---

## Using the API

```python
from famail_temporal.data.loader import DataBundle
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.algorithm.modifier import TrajectoryModifier
from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution,
    rank_trajectories,
    select_top_k,
)
from famail_temporal.utils.seeding import set_all_seeds

# Reproducibility
set_all_seeds(42)

# Load preprocessed data (uses cache; rebuild with force_rebuild_cache=True)
bundle = DataBundle.load()
print(f"Active units: {bundle.unit_map.n_units}")  # expected: 6,000–8,000

# Build the objective
objective = FAMAILObjective(bundle)

# Compute baseline fairness metrics
import torch
base_pickup = torch.tensor(bundle.pickup_3d, dtype=torch.float32)
f_spatial, f_causal, f_fidelity, total = objective.forward(base_pickup)
print(f"Baseline — F_spatial={f_spatial:.4f}, F_causal={f_causal:.4f}")

# Attribution: rank trajectories by contribution to unfairness
unit_scores = compute_per_unit_attribution(bundle)          # (N,) array
ranked = rank_trajectories(bundle, unit_scores)             # sorted by score
top_k = select_top_k(ranked, k=10)                         # list of traj_idx

# Modify trajectories using ST-iFGSM
modifier = TrajectoryModifier(bundle)
for traj_idx in top_k:
    history = modifier.modify_single(traj_idx, n_iterations=50)
    final = history[-1]
    print(f"Traj {traj_idx}: total {history[0]['total']:.4f} -> {final['total']:.4f}")
```

---

## Directory layout

| Path | README | One-sentence description |
|---|---|---|
| `data/` | [README](data/README.md) | Ingest raw files; produce canonical `(48, 90, T)` tensors and active-unit index |
| `fairness/` | [README](fairness/README.md) | Pooled Gini and Option B R^2 fairness metrics; per-unit attribution decomposition |
| `fidelity/` | [README](fidelity/README.md) | Port of the pre-trained Siamese discriminator for trajectory realism scoring |
| `algorithm/` | [README](algorithm/README.md) | ST-iFGSM loop, FAMAILObjective, soft cell assignment, attribution-to-trajectory ranking |
| `utils/` | [README](utils/README.md) | Reproducible seeding and trajectory dataclasses |
| `tests/` | [README](tests/README.md) | Math invariants, bug-class guards, and integration tests |
| `source_data/` | [README](source_data/README.md) | Source datasets (output of `source_generation/`; input to `preprocess.py` and `loader.py`) |
| `cache/` | [README](cache/README.md) | Preprocessed artifacts with config-encoded filenames |
| `discriminator_checkpoints/` | [README](discriminator_checkpoints/README.md) | Canonical fidelity checkpoint and provenance |

Root files:

| File | Role |
|---|---|
| `config.py` | Single source of truth for all hyperparameters |
| `preprocess.py` | One-time preprocessing: `source_data/` → `cache/` |
| `requirements.txt` | `torch`, `numpy`, `scikit-learn`, `pytest` |

---

## Key design commitments

Four architectural invariants that every module in this directory must respect:

1. **One active-unit ordering.** The `(N,)` active-unit vector has a canonical order (cell-major,
   then time-block) fixed at preprocess time. All arrays in R^N use this order. The ordering is
   asserted at every load boundary.

2. **Single grid-to-unit conversion point.** The `(48, 90, T) -> (N,)` gather happens in exactly
   one place: `algorithm/objective.py::forward()`. `fairness/` modules never see grid dimensions;
   `fidelity/` modules never see N-vectors.

3. **Gradient flow only through `pickup_counts`.** The only tensor that varies during ST-iFGSM
   is `soft_pickup_3d`, and only in the `[:, :, t*]` slice where `t*` is the pickup's time block.
   All other inputs are frozen. `g_0(D)` is evaluated under `torch.no_grad()`.

4. **No external dependencies.** No imports from outside `famail_temporal/`. Only `torch`,
   `numpy`, `scikit-learn`, and `pytest`.

---

## Reproducibility

All experiments call `set_all_seeds(seed)` before any stochastic operation. The fast test
suite (`pytest famail_temporal/tests/`) runs in under 10 seconds and uses only synthetic data.
The slow test suite (`pytest famail_temporal/tests/ --run-slow`) exercises the full pipeline
against real data and must pass before any reported results are considered final.

Preprocessing is deterministic given the same raw data and `config.py` values. Cache
filenames encode the config parameters that affect each artifact, so multiple configurations
coexist in `cache/` without invalidating each other.

---

## Design specification

Full design decisions, mathematical derivations, stability safeguards, and identified snags are
documented in:

```
docs/superpowers/specs/2026-04-16-famail-temporal-design.md
```

This README is a navigational entry point. For mathematical details of any component, consult
the component's sub-directory README and the spec.
