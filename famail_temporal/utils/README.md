# `utils/` — Shared utilities with no domain knowledge

## Purpose

Provide two thin utilities — reproducible seeding and trajectory data structures — that are
needed across multiple modules but have no awareness of fairness metrics, the grid, or the
algorithm. Everything in this module is stateless and has no dependencies outside `config.py`.

---

## Files

| File | Role |
|---|---|
| `seeding.py` | `set_all_seeds(seed)` — sets seeds for `random`, `numpy`, `torch`, `torch.cuda`, and the multi-stream context sampler in a single call |
| `trajectory.py` | `Trajectory` and `TrajectoryState` dataclasses — represent a single driver's trajectory and per-timestep state |

---

## Key design choices

### 1. `set_all_seeds` covers every randomness source

Reproducibility in this codebase requires controlling five separate RNG streams:

| Library | Call |
|---|---|
| Python `random` | `random.seed(seed)` |
| NumPy | `np.random.seed(seed)` |
| PyTorch CPU | `torch.manual_seed(seed)` |
| PyTorch CUDA | `torch.cuda.manual_seed_all(seed)` |
| Multi-stream context sampler | `np.random.seed(seed)` (also covers the 'sample' fill strategy in `fidelity/context.py`) |

A single `set_all_seeds(seed)` call at the top of every script or test is sufficient. The
function is idempotent: calling it multiple times with the same seed resets all streams to the
same state.

Note: PyTorch's cuDNN operations may still introduce non-determinism from concurrent kernel
execution even with seeds set. To achieve bit-exact reproducibility on GPU, also set:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

The `conftest.py` seeded autouse fixture handles this for all tests.

### 2. `Trajectory` / `TrajectoryState` ported verbatim from legacy code

The `Trajectory` and `TrajectoryState` dataclasses are lifted directly from the V3 codebase
without modification. They were already well-structured and consistent with the 126-element
state vector schema documented in `CLAUDE.md`. Porting verbatim (rather than redesigning)
avoids introducing incompatibilities with the raw data format.

State vector schema (for reference):

| Indices | Field |
|---|---|
| 0–1 | `x_grid`, `y_grid` (0-indexed) |
| 2 | `time_bucket` |
| 3 | `day_index` |
| 4–24 | POI distances |
| 25–49 | Pickup counts (5×5 window) |
| 50–74 | Traffic volume |
| 75–99 | Speed |
| 100–124 | Wait times |
| 125 | `action_code` |

Coordinates are 0-indexed in the trajectory state vector. The fidelity context builder adds +1
when constructing discriminator inputs (which were trained on 1-indexed coordinates).

---

## API surface

```python
from famail_temporal.utils.seeding import set_all_seeds
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState

# Reproducibility
set_all_seeds(42)

# Trajectory data access
traj: Trajectory = bundle.trajectories[driver_idx]
state: TrajectoryState = traj.states[timestep_idx]
x, y = state.x_grid, state.y_grid       # 0-indexed grid coordinates
t    = state.time_bucket                 # raw time bucket (0–287)
```

---

## Dependencies

- `config.py` — none currently (seeding uses hardcoded library calls)
- Standard library: `random`, `dataclasses`
- Third-party: `numpy`, `torch`

No imports from `data/`, `fairness/`, `fidelity/`, or `algorithm/`.

---

## Paper-section hook

These utilities are briefly mentioned in the **Reproducibility appendix** of the paper.
The `set_all_seeds` function is cited as the mechanism ensuring experiment reproducibility,
and the seed value used for all reported results is recorded there.
