# FAMAIL-Temporal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **MODEL REQUIREMENT — OPUS ONLY:** Every task in this plan must be executed by **Claude Opus 4.6 (1M context)**. Do not dispatch or switch to Sonnet, Haiku, or any other model for any task, step, sub-agent dispatch, or review. This requirement is non-negotiable: the mathematical density and cross-task consistency requirements of this rewrite exceed what smaller models reliably handle. If using subagent-driven-development, pass `model: "opus"` (or the equivalent) when dispatching each subagent. If using executing-plans inline, ensure the current session is on Opus before proceeding.

**Goal:** Build `famail_temporal/` — a standalone, dependency-free reimplementation of the FAMAIL trajectory modification algorithm with temporally-aware fairness metrics (T=4 time blocks, pooled Option B F_causal, per-unit attribution).

**Architecture:** Five core submodules under `famail_temporal/` (`data/`, `fairness/`, `fidelity/`, `algorithm/`, `utils/`) plus `preprocess.py`, `config.py`, and `tests/`. All metrics operate on a single flattened vector of N active `(cell, t)` units produced by a `UnitIndexMap` built once at preprocess time. The discriminator is ported as an opaque pre-trained checkpoint.

**Tech Stack:** Python 3.10, PyTorch ≥2.0, NumPy ≥1.24, scikit-learn ≥1.2, pytest ≥7.0.

**Design spec:** [docs/superpowers/specs/2026-04-16-famail-temporal-design.md](../specs/2026-04-16-famail-temporal-design.md)

**Test conventions:** Every task that creates production code includes a failing-first test. Fast tests (< 10 s) use synthetic fixtures; slow tests are marked `@pytest.mark.slow` and skipped by default.

**Commit conventions:** Each task ends with a commit using conventional commit prefixes (`feat:`, `test:`, `chore:`, `docs:`). Every commit must pass `pytest` (fast tests) — no broken commits.

**Working directory:** All commands assume CWD is the FAMAIL repo root (`/home/robert/FAMAIL`).

**Plan structure:** This plan spans 9 phases and 34 tasks. Phase 1–4 are in this file; Phases 5–9 are in companion files `2026-04-16-famail-temporal-phase5-6.md`, `2026-04-16-famail-temporal-phase7-8.md`, and `2026-04-16-famail-temporal-phase9.md` (execute in order).

---

## Phase 1: Scaffolding (Tasks 1–4)

### Task 1: Create directory tree and empty package files

**Files:**
- Create: `famail_temporal/__init__.py`, `famail_temporal/data/__init__.py`, `famail_temporal/fairness/__init__.py`, `famail_temporal/fidelity/__init__.py`, `famail_temporal/algorithm/__init__.py`, `famail_temporal/utils/__init__.py`, `famail_temporal/tests/__init__.py`, `famail_temporal/tests/synthetic/__init__.py`
- Create: `famail_temporal/raw_data/.gitkeep`, `famail_temporal/cache/.gitkeep`, `famail_temporal/discriminator_checkpoints/.gitkeep`
- Create: `famail_temporal/requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Create the directory tree**

```bash
mkdir -p famail_temporal/{data,fairness,fidelity,algorithm,utils}
mkdir -p famail_temporal/tests/synthetic
mkdir -p famail_temporal/raw_data famail_temporal/cache famail_temporal/discriminator_checkpoints/default
```

- [ ] **Step 2: Create empty `__init__.py` and `.gitkeep` files**

```bash
touch famail_temporal/__init__.py \
      famail_temporal/data/__init__.py \
      famail_temporal/fairness/__init__.py \
      famail_temporal/fidelity/__init__.py \
      famail_temporal/algorithm/__init__.py \
      famail_temporal/utils/__init__.py \
      famail_temporal/tests/__init__.py \
      famail_temporal/tests/synthetic/__init__.py \
      famail_temporal/raw_data/.gitkeep \
      famail_temporal/cache/.gitkeep \
      famail_temporal/discriminator_checkpoints/.gitkeep
```

- [ ] **Step 3: Write `famail_temporal/requirements.txt`**

```
torch>=2.0,<3.0
numpy>=1.24,<2.0
scikit-learn>=1.2,<2.0
pytest>=7.0
```

- [ ] **Step 4: Append to `.gitignore`**

```
# famail_temporal data/cache/checkpoints
famail_temporal/raw_data/*
!famail_temporal/raw_data/.gitkeep
!famail_temporal/raw_data/README.md
famail_temporal/cache/*
!famail_temporal/cache/.gitkeep
!famail_temporal/cache/README.md
famail_temporal/discriminator_checkpoints/*
!famail_temporal/discriminator_checkpoints/.gitkeep
!famail_temporal/discriminator_checkpoints/README.md
!famail_temporal/discriminator_checkpoints/default/
famail_temporal/discriminator_checkpoints/default/*.pt
```

- [ ] **Step 5: Verify**

```bash
find famail_temporal -type f | sort
```

Expected: 12 files.

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/ .gitignore
git commit -m "chore: scaffold famail_temporal directory tree"
```

---

### Task 2: Write `config.py`

**Files:**
- Create: `famail_temporal/config.py`

- [ ] **Step 1: Write the file**

```python
"""
Configuration constants for famail_temporal.

Every reviewer-visible knob lives here. The cache/ filenames encode the values
of this config so multiple configurations can coexist without invalidation.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

# Paths
PACKAGE_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PACKAGE_ROOT / "raw_data"
CACHE_DIR = PACKAGE_ROOT / "cache"
DISCRIMINATOR_CHECKPOINT_DIR = PACKAGE_ROOT / "discriminator_checkpoints"
DISCRIMINATOR_CHECKPOINT_FILENAME = "default/best.pt"

# Grid geometry (fixed by the Shenzhen dataset)
GRID_DIMS: Tuple[int, int] = (48, 90)
N_TIME_BUCKETS: int = 288

# Time blocks — end > 24 encodes wraparound
TIME_BLOCKS: List[Tuple[str, int, int]] = [
    ("morning_peak", 7, 10),
    ("midday",       10, 16),
    ("evening_peak", 16, 20),
    ("night",        20, 31),  # 20 → 07 next day
]
T: int = len(TIME_BLOCKS)

# Active-unit filter
ACTIVE_SUPPLY_THRESHOLD: float = 0.5
DEMAND_FLOOR: float = 0.01
SUPPLY_FLOOR: float = 0.1

# Demographics
DEMOGRAPHIC_FEATURES: List[str] = [
    "AvgHousingPricePerSqM",
    "GDPperCapita",
    "CompPerCapita",
]

# Objective weights
ALPHA_SPATIAL: float = 0.33
ALPHA_CAUSAL: float = 0.33
ALPHA_FIDELITY: float = 0.34

# ST-iFGSM
STEP_SIZE_ALPHA: float = 0.1
EPSILON_BALL: float = 2.0
MAX_ITERATIONS: int = 50
CONVERGENCE_TOL: float = 1e-6

# Soft cell assignment
SOFT_NEIGHBORHOOD_SIZE: int = 5
TAU_MAX: float = 1.0
TAU_MIN: float = 0.1
ANNEAL_TEMPERATURE: bool = True

# Numerical stability
EPS: float = 1e-8
MIN_ACTIVE_UNITS_PER_BLOCK: int = 10
MIN_TOTAL_ACTIVE_UNITS: int = 100

# Reproducibility
DEFAULT_SEED: int = 42


def cache_suffix(include_features: bool = False) -> str:
    """Build the config-encoded filename suffix for cached artifacts."""
    base = f"T{T}_thr{ACTIVE_SUPPLY_THRESHOLD}"
    if include_features:
        tokens = []
        for f in DEMOGRAPHIC_FEATURES:
            token = f.lower().replace("percapita", "").replace("avg", "").replace("price", "")
            token = token.replace("persqm", "").strip("_")
            tokens.append(token)
        base += "_feat-" + "-".join(tokens)
    return base
```

- [ ] **Step 2: Verify import**

```bash
python -c "from famail_temporal import config; print(config.T, config.cache_suffix(True))"
```

Expected: `4 T4_thr0.5_feat-housing-gdp-comp`

- [ ] **Step 3: Commit**

```bash
git add famail_temporal/config.py
git commit -m "feat(config): add config.py single source of truth"
```

---

### Task 3: Write `utils/seeding.py`

**Files:**
- Create: `famail_temporal/utils/seeding.py`
- Create: `famail_temporal/tests/test_seeding.py`

- [ ] **Step 1: Write the failing test**

`famail_temporal/tests/test_seeding.py`:

```python
"""Tests for utils.seeding."""
import numpy as np
import torch

from famail_temporal.utils.seeding import set_all_seeds


def test_numpy_reproducible():
    set_all_seeds(123)
    a = np.random.rand(5)
    set_all_seeds(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_torch_reproducible():
    set_all_seeds(123)
    a = torch.rand(5)
    set_all_seeds(123)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_python_random_reproducible():
    import random
    set_all_seeds(123)
    a = [random.random() for _ in range(5)]
    set_all_seeds(123)
    b = [random.random() for _ in range(5)]
    assert a == b
```

- [ ] **Step 2: Run test (expect failure)**

```bash
pytest famail_temporal/tests/test_seeding.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `famail_temporal/utils/seeding.py`**

```python
"""Unified seed control for reproducibility."""

from __future__ import annotations
import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set random, numpy, torch, and torch.cuda seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest famail_temporal/tests/test_seeding.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/utils/seeding.py famail_temporal/tests/test_seeding.py
git commit -m "feat(utils): add set_all_seeds for reproducibility"
```

---

### Task 4: Write `utils/trajectory.py` (ported from `trajectory_modification/trajectory.py`)

**Files:**
- Source reference (do not modify): `trajectory_modification/trajectory.py`
- Create: `famail_temporal/utils/trajectory.py`
- Create: `famail_temporal/tests/test_trajectory.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for utils.trajectory."""
import numpy as np
import torch

from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


def _make_trajectory(n_states: int = 5) -> Trajectory:
    states = [
        TrajectoryState(x_grid=float(i), y_grid=float(i + 1),
                        time_bucket=100 + i, day_index=1)
        for i in range(n_states)
    ]
    return Trajectory(trajectory_id=0, driver_id=7, states=states)


def test_pickup_cell_is_last_state():
    traj = _make_trajectory(5)
    assert traj.pickup_cell == (4, 5)


def test_to_tensor_shape():
    traj = _make_trajectory(5)
    t = traj.to_tensor()
    assert t.shape == (5, 4)
    assert t.dtype == torch.float32


def test_clone_is_deep():
    traj = _make_trajectory(3)
    clone = traj.clone()
    clone.states[-1].x_grid = 99.0
    assert traj.states[-1].x_grid != 99.0


def test_apply_perturbation_clips_to_grid():
    traj = _make_trajectory(3)
    perturbed = traj.apply_perturbation(np.array([100.0, -100.0]), grid_dims=(48, 90))
    assert perturbed.states[-1].x_grid == 47.0
    assert perturbed.states[-1].y_grid == 0.0
```

- [ ] **Step 2: Run test (expect failure)**

```bash
pytest famail_temporal/tests/test_trajectory.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `famail_temporal/utils/trajectory.py`**

Port from `trajectory_modification/trajectory.py` verbatim, removing any parent-project imports. Full content:

```python
"""Trajectory representation for famail_temporal."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np
import torch


@dataclass
class TrajectoryState:
    x_grid: float
    y_grid: float
    time_bucket: int
    day_index: int

    def to_array(self) -> np.ndarray:
        return np.array([self.x_grid, self.y_grid, self.time_bucket, self.day_index])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "TrajectoryState":
        return cls(
            x_grid=float(arr[0]), y_grid=float(arr[1]),
            time_bucket=int(arr[2]), day_index=int(arr[3]),
        )


@dataclass
class Trajectory:
    trajectory_id: Any
    driver_id: Any
    states: List[TrajectoryState]
    metadata: dict = field(default_factory=dict)

    @property
    def pickup_state(self) -> TrajectoryState:
        return self.states[-1]

    @property
    def pickup_cell(self) -> Tuple[int, int]:
        s = self.pickup_state
        return (int(s.x_grid), int(s.y_grid))

    @property
    def n_states(self) -> int:
        return len(self.states)

    def to_discriminator_format(self) -> np.ndarray:
        return np.array([s.to_array() for s in self.states])

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.to_discriminator_format(), dtype=torch.float32)

    def clone(self) -> "Trajectory":
        return Trajectory(
            trajectory_id=self.trajectory_id,
            driver_id=self.driver_id,
            states=[TrajectoryState(s.x_grid, s.y_grid, s.time_bucket, s.day_index)
                    for s in self.states],
            metadata=self.metadata.copy(),
        )

    def apply_perturbation(self, delta: np.ndarray,
                           grid_dims: Tuple[int, int] = (48, 90)) -> "Trajectory":
        modified = self.clone()
        pickup = modified.states[-1]
        new_x = float(np.clip(pickup.x_grid + delta[0], 0, grid_dims[0] - 1))
        new_y = float(np.clip(pickup.y_grid + delta[1], 0, grid_dims[1] - 1))
        modified.states[-1] = TrajectoryState(
            x_grid=new_x, y_grid=new_y,
            time_bucket=pickup.time_bucket, day_index=pickup.day_index,
        )
        return modified
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest famail_temporal/tests/test_trajectory.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/utils/trajectory.py famail_temporal/tests/test_trajectory.py
git commit -m "feat(utils): port Trajectory and TrajectoryState"
```

---

## Phase 2: Data aggregation (Tasks 5–8)

### Task 5: `hour_to_block_index` helper

**Files:**
- Create: `famail_temporal/data/aggregation.py` (initial)
- Create: `famail_temporal/tests/test_aggregation.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for data.aggregation."""
import pytest

from famail_temporal.data.aggregation import hour_to_block_index


@pytest.mark.parametrize("hour,expected", [
    (7, 0), (9, 0), (10, 1), (15, 1),
    (16, 2), (19, 2), (20, 3), (23, 3),
    (0, 3), (6, 3),
])
def test_hour_to_block_index(hour, expected):
    assert hour_to_block_index(hour) == expected


def test_invalid_hour_raises():
    with pytest.raises(ValueError):
        hour_to_block_index(24)
```

- [ ] **Step 2: Run test (expect failure)**

```bash
pytest famail_temporal/tests/test_aggregation.py -v
```

- [ ] **Step 3: Write `famail_temporal/data/aggregation.py` (initial)**

```python
"""
Aggregation of raw .pkl data into (48, 90, T) tensors.

Handles the time-bucket-to-block mapping with night wraparound, and builds
the three base tensors (pickup_3d, dropoff_3d, active_taxis_3d) using the
unified mean-hourly aggregation rule.
"""

from __future__ import annotations
from famail_temporal import config


def hour_to_block_index(hour: int) -> int:
    """Map hour [0, 24) → time block index [0, T)."""
    if not (0 <= hour < 24):
        raise ValueError(f"Hour must be in [0, 24), got {hour}")
    for i, (_, start, end) in enumerate(config.TIME_BLOCKS):
        if end > 24:
            if hour >= start or hour < (end - 24):
                return i
        else:
            if start <= hour < end:
                return i
    raise ValueError(f"Hour {hour} did not map to any time block")
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest famail_temporal/tests/test_aggregation.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/data/aggregation.py famail_temporal/tests/test_aggregation.py
git commit -m "feat(data): add hour_to_block_index with wraparound handling"
```

---

### Task 6: 3D aggregation functions

**Files:**
- Modify: `famail_temporal/data/aggregation.py` (append)
- Modify: `famail_temporal/tests/test_aggregation.py` (append)

- [ ] **Step 1: Append failing tests**

```python
import numpy as np
from famail_temporal.data.aggregation import (
    aggregate_pickup_dropoff,
    aggregate_active_taxis,
    time_bucket_to_hour,
    block_n_hours,
)


def test_time_bucket_to_hour():
    assert time_bucket_to_hour(1) == 0
    assert time_bucket_to_hour(12) == 0
    assert time_bucket_to_hour(13) == 1
    assert time_bucket_to_hour(288) == 23


def test_block_n_hours():
    assert block_n_hours(0) == 3   # morning_peak
    assert block_n_hours(1) == 6   # midday
    assert block_n_hours(2) == 4   # evening_peak
    assert block_n_hours(3) == 11  # night (wraparound)


def test_aggregate_pickup_dropoff_mean_scale():
    # cell (5, 10) at hour 7 (block 0, 3 hours), day 1: 6 pickups
    raw_data = {(5 + 1, 10 + 1, 85, 1): [6, 0]}
    n_days = 1
    pickup_3d, dropoff_3d = aggregate_pickup_dropoff(raw_data, n_days=n_days)
    assert pickup_3d.shape == (48, 90, 4)
    # mean hourly = 6 / (3 × 1) = 2.0
    assert np.isclose(pickup_3d[5, 10, 0], 2.0)
    assert pickup_3d.sum() == pickup_3d[5, 10, 0]


def test_aggregate_active_taxis_mean():
    raw_data = {
        (5 + 1, 10 + 1, 7, 1): 20,
        (5 + 1, 10 + 1, 8, 1): 10,
    }
    taxis_3d = aggregate_active_taxis(raw_data, n_days=1)
    assert taxis_3d.shape == (48, 90, 4)
    # mean hourly = (20 + 10) / (3 × 1) = 10.0
    assert np.isclose(taxis_3d[5, 10, 0], 10.0)
```

- [ ] **Step 2: Run tests (expect failure)**

```bash
pytest famail_temporal/tests/test_aggregation.py -v
```

- [ ] **Step 3: Append to `data/aggregation.py`**

```python
from typing import Dict, Tuple
import numpy as np


def time_bucket_to_hour(time_bucket: int) -> int:
    """Map 1-indexed time_bucket (1..288, 5-min) to 0-indexed hour (0..23)."""
    return (time_bucket - 1) // 12


def block_n_hours(block_idx: int) -> int:
    """Number of hours covered by block `block_idx`, handling wraparound."""
    _, start, end = config.TIME_BLOCKS[block_idx]
    return end - start


def dataset_n_days(raw_data: Dict[Tuple, object]) -> int:
    """Infer the number of distinct day_index values."""
    days = {key[3] for key in raw_data.keys() if len(key) >= 4}
    return len(days)


def aggregate_pickup_dropoff(
    raw_data: Dict[Tuple[int, int, int, int], object],
    n_days: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate raw counts dict → (48, 90, T) mean-hourly tensors.

    Raw keys use 1-indexed (x, y) and 1-indexed time_bucket.
    Aggregation: sum raw counts per (cell, block, day) combination, then
    divide by uniform n_obs = block_n_hours(t) × n_days.
    """
    pickup_3d = np.zeros((*config.GRID_DIMS, config.T), dtype=np.float32)
    dropoff_3d = np.zeros((*config.GRID_DIMS, config.T), dtype=np.float32)

    for key, counts in raw_data.items():
        if len(key) < 4:
            continue
        x_raw, y_raw, time_bucket, _day = key
        x, y = int(x_raw) - 1, int(y_raw) - 1
        if not (0 <= x < config.GRID_DIMS[0] and 0 <= y < config.GRID_DIMS[1]):
            continue
        hour = time_bucket_to_hour(int(time_bucket))
        t_block = hour_to_block_index(hour)
        if isinstance(counts, (list, tuple)):
            pickup = counts[0] if len(counts) >= 1 else 0
            dropoff = counts[1] if len(counts) >= 2 else 0
        else:
            pickup, dropoff = int(counts), 0
        pickup_3d[x, y, t_block] += pickup
        dropoff_3d[x, y, t_block] += dropoff

    for t in range(config.T):
        divisor = block_n_hours(t) * n_days
        if divisor > 0:
            pickup_3d[:, :, t] /= divisor
            dropoff_3d[:, :, t] /= divisor

    return pickup_3d, dropoff_3d


def aggregate_active_taxis(
    raw_data: Dict[Tuple[int, int, int, int], int],
    n_days: int,
) -> np.ndarray:
    """Aggregate hourly active_taxis → (48, 90, T) mean-hourly tensor.

    Raw keys use 1-indexed (x, y) and 0-indexed hour.
    """
    active_3d = np.zeros((*config.GRID_DIMS, config.T), dtype=np.float32)

    for key, count in raw_data.items():
        if len(key) < 4:
            continue
        x_raw, y_raw, hour, _day = key
        x, y = int(x_raw) - 1, int(y_raw) - 1
        if not (0 <= x < config.GRID_DIMS[0] and 0 <= y < config.GRID_DIMS[1]):
            continue
        if not (0 <= int(hour) < 24):
            continue
        t_block = hour_to_block_index(int(hour))
        active_3d[x, y, t_block] += count

    for t in range(config.T):
        divisor = block_n_hours(t) * n_days
        if divisor > 0:
            active_3d[:, :, t] /= divisor

    active_3d = np.maximum(active_3d, config.SUPPLY_FLOOR)
    return active_3d
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest famail_temporal/tests/test_aggregation.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/data/aggregation.py famail_temporal/tests/test_aggregation.py
git commit -m "feat(data): add 3D aggregation with mean-hourly scale"
```

---

### Task 7: `UnitIndexMap` dataclass

**Files:**
- Create: `famail_temporal/data/active_mask.py`
- Create: `famail_temporal/tests/test_active_mask.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for data.active_mask."""
import numpy as np
import pytest

from famail_temporal.data.active_mask import UnitIndexMap


def _make_small_mask():
    mask = np.zeros((3, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[0, 0, 1] = True
    mask[1, 0, 0] = True
    mask[2, 1, 1] = True
    return mask


def test_canonical_ordering():
    mask = _make_small_mask()
    umap = UnitIndexMap.from_mask(mask, grid_shape=(3, 2))
    assert umap.n_units == 4
    np.testing.assert_array_equal(umap.cell_indices, [0, 0, 2, 5])
    np.testing.assert_array_equal(umap.time_block_indices, [0, 1, 0, 1])


def test_from_cell_time_roundtrip():
    mask = _make_small_mask()
    umap = UnitIndexMap.from_mask(mask, grid_shape=(3, 2))
    assert umap.from_cell_time(0, 0) == 0
    assert umap.from_cell_time(0, 1) == 1
    assert umap.from_cell_time(2, 0) == 2
    assert umap.from_cell_time(5, 1) == 3
    assert umap.from_cell_time(1, 0) == -1  # inactive


def test_to_cell_time():
    mask = _make_small_mask()
    umap = UnitIndexMap.from_mask(mask, grid_shape=(3, 2))
    assert umap.to_cell_time(0) == (0, 0)
    assert umap.to_cell_time(3) == (5, 1)


def test_units_per_block():
    mask = _make_small_mask()
    umap = UnitIndexMap.from_mask(mask, grid_shape=(3, 2))
    np.testing.assert_array_equal(umap.units_per_block, [2, 2])
```

- [ ] **Step 2: Run test (expect failure)**

```bash
pytest famail_temporal/tests/test_active_mask.py -v
```

- [ ] **Step 3: Write `famail_temporal/data/active_mask.py` (UnitIndexMap only)**

```python
"""Active-unit mask and canonical ordering."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class UnitIndexMap:
    """Canonical ordering of active (cell, t) units.

    Ordering rule: cell-major, then time-block within cell.
    """
    cell_indices: np.ndarray
    time_block_indices: np.ndarray
    flat_lookup: np.ndarray
    n_units: int
    n_active_cells: int
    units_per_block: np.ndarray

    @classmethod
    def from_mask(cls, mask_3d: np.ndarray, grid_shape: Tuple[int, int]) -> "UnitIndexMap":
        gx, gy = grid_shape
        t = mask_3d.shape[2]
        assert mask_3d.shape == (gx, gy, t)

        cell_list, block_list = [], []
        for x in range(gx):
            for y in range(gy):
                flat_cell = x * gy + y
                for t_idx in range(t):
                    if mask_3d[x, y, t_idx]:
                        cell_list.append(flat_cell)
                        block_list.append(t_idx)

        cell_indices = np.asarray(cell_list, dtype=np.int32)
        time_block_indices = np.asarray(block_list, dtype=np.int8)
        n_units = len(cell_list)

        flat_lookup = np.full(gx * gy * t, -1, dtype=np.int32)
        for unit_idx, (c, b) in enumerate(zip(cell_list, block_list)):
            flat_lookup[c * t + b] = unit_idx

        units_per_block = np.zeros(t, dtype=np.int64)
        for b in block_list:
            units_per_block[b] += 1

        n_active_cells = len(set(cell_list))

        return cls(
            cell_indices=cell_indices,
            time_block_indices=time_block_indices,
            flat_lookup=flat_lookup,
            n_units=n_units,
            n_active_cells=n_active_cells,
            units_per_block=units_per_block,
        )

    def from_cell_time(self, cell: int, t: int) -> int:
        n_blocks = len(self.units_per_block)
        idx = cell * n_blocks + t
        if idx < 0 or idx >= len(self.flat_lookup):
            return -1
        return int(self.flat_lookup[idx])

    def to_cell_time(self, unit_idx: int) -> Tuple[int, int]:
        return int(self.cell_indices[unit_idx]), int(self.time_block_indices[unit_idx])

    def to_flat_cell(self, unit_idx: int) -> int:
        return int(self.cell_indices[unit_idx])

    def to_time_block(self, unit_idx: int) -> int:
        return int(self.time_block_indices[unit_idx])
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest famail_temporal/tests/test_active_mask.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/data/active_mask.py famail_temporal/tests/test_active_mask.py
git commit -m "feat(data): add UnitIndexMap with cell-major ordering"
```

---

### Task 8: `compute_active_mask` function

**Files:**
- Modify: `famail_temporal/data/active_mask.py` (append)
- Modify: `famail_temporal/tests/test_active_mask.py` (append)

- [ ] **Step 1: Append failing tests**

```python
from famail_temporal.data.active_mask import compute_active_mask


def test_active_mask_supply_threshold():
    active_3d = np.zeros((48, 90, 4), dtype=np.float32)
    active_3d[5, 10, 0] = 1.0
    active_3d[6, 11, 0] = 0.3
    valid_mask = np.ones((48, 90), dtype=bool)
    demographics = np.zeros((48, 90, 3), dtype=np.float32)
    mask = compute_active_mask(active_3d, valid_mask, demographics)
    assert mask.shape == (48, 90, 4)
    assert mask[5, 10, 0]
    assert not mask[6, 11, 0]


def test_active_mask_rejects_nan_demographics():
    active_3d = np.ones((48, 90, 4), dtype=np.float32) * 10.0
    valid_mask = np.ones((48, 90), dtype=bool)
    demographics = np.zeros((48, 90, 3), dtype=np.float32)
    demographics[5, 10, 0] = np.nan
    mask = compute_active_mask(active_3d, valid_mask, demographics)
    assert not mask[5, 10, 0]
    assert not mask[5, 10, 3]
```

- [ ] **Step 2: Run test (expect failure)**

```bash
pytest famail_temporal/tests/test_active_mask.py::test_active_mask_supply_threshold -v
```

- [ ] **Step 3: Append to `data/active_mask.py`**

```python
from famail_temporal import config


def compute_active_mask(
    active_taxis_3d: np.ndarray,
    valid_mask: np.ndarray,
    demographics: np.ndarray,
) -> np.ndarray:
    """A unit (c, t) is active iff:
      1. active_taxis_3d[c, t] > ACTIVE_SUPPLY_THRESHOLD
      2. valid_mask[c] is True
      3. No NaN in any demographic feature for cell c
    """
    gx, gy = valid_mask.shape
    t = active_taxis_3d.shape[2]
    assert active_taxis_3d.shape == (gx, gy, t)
    assert demographics.shape[:2] == (gx, gy)

    cell_finite = np.isfinite(demographics).all(axis=-1)
    cell_valid = valid_mask & cell_finite
    supply_ok = active_taxis_3d > config.ACTIVE_SUPPLY_THRESHOLD
    return supply_ok & cell_valid[:, :, None]
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest famail_temporal/tests/test_active_mask.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/data/active_mask.py famail_temporal/tests/test_active_mask.py
git commit -m "feat(data): add compute_active_mask with validity and NaN checks"
```

---

## Phase 3: Fairness foundations (Tasks 9–12)

### Task 9: `build_power_basis_features` + `G0Function`

**Files:**
- Create: `famail_temporal/fairness/g0_power_basis.py`
- Create: `famail_temporal/tests/test_g0_power_basis.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for fairness.g0_power_basis."""
import numpy as np

from famail_temporal.fairness.g0_power_basis import (
    build_power_basis_features,
    G0Function,
)


def test_power_basis_shape_with_intercept():
    D = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = build_power_basis_features(D, include_intercept=True)
    assert X.shape == (5, 4)
    np.testing.assert_array_equal(X[:, 0], np.ones(5))


def test_power_basis_shape_without_intercept():
    D = np.array([1.0, 2.0, 3.0])
    X = build_power_basis_features(D, include_intercept=False)
    assert X.shape == (3, 3)


def test_g0function_shape():
    coefficients = np.array([0.1, 0.5, 0.2, 0.05])
    g0 = G0Function(coefficients=coefficients, d_min=0.01, d_max=10.0)
    D = np.array([1.0, 2.0, 3.0])
    assert g0(D).shape == (3,)


def test_g0function_clips():
    coefficients = np.array([0.0, 1.0, 0.0, 0.0])   # g0 = 1/(D+1)
    g0 = G0Function(coefficients=coefficients, d_min=1.0, d_max=5.0)
    np.testing.assert_allclose(g0(np.array([100.0])), 1.0 / 6.0)
    np.testing.assert_allclose(g0(np.array([0.0])), 1.0 / 2.0)
```

- [ ] **Step 2: Run tests (expect failure)**

```bash
pytest famail_temporal/tests/test_g0_power_basis.py -v
```

- [ ] **Step 3: Write `famail_temporal/fairness/g0_power_basis.py` (partial)**

```python
"""
Fit g₀(D) using power basis [1, 1/(D+1), 1/√(D+1), √(D+1)].

Captures hyperbolic Y ≈ a/D with four linear parameters.
Fitted once during preprocessing at active-unit block-mean scale.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


def build_power_basis_features(demands: np.ndarray, include_intercept: bool = True) -> np.ndarray:
    """Feature matrix [1, 1/(D+1), 1/√(D+1), √(D+1)] per cell."""
    d_safe = np.asarray(demands, dtype=np.float64) + 1.0
    feats = np.column_stack([
        1.0 / d_safe,
        1.0 / np.sqrt(d_safe),
        np.sqrt(d_safe),
    ])
    if include_intercept:
        feats = np.column_stack([np.ones(len(demands)), feats])
    return feats


@dataclass(frozen=True)
class G0Function:
    """Fitted g₀(D) with power basis coefficients.

    Coefficient order: [intercept, c_{1/(D+1)}, c_{1/√(D+1)}, c_{√(D+1)}]
    """
    coefficients: np.ndarray
    d_min: float
    d_max: float

    def __call__(self, d: np.ndarray) -> np.ndarray:
        d_arr = np.asarray(d, dtype=np.float64)
        d_clipped = np.clip(d_arr, self.d_min, self.d_max)
        X = build_power_basis_features(d_clipped, include_intercept=True)
        return (X @ self.coefficients).astype(np.float64)
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest famail_temporal/tests/test_g0_power_basis.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/fairness/g0_power_basis.py famail_temporal/tests/test_g0_power_basis.py
git commit -m "feat(fairness): G0Function dataclass + power basis builder"
```

---

### Task 10: `fit()` in `g0_power_basis.py`

**Files:**
- Modify: `famail_temporal/fairness/g0_power_basis.py` (append)
- Modify: `famail_temporal/tests/test_g0_power_basis.py` (append)

- [ ] **Step 1: Append failing tests**

```python
from famail_temporal.fairness.g0_power_basis import fit as fit_g0


def test_fit_recovers_hyperbolic():
    rng = np.random.RandomState(42)
    D = np.linspace(0.5, 10.0, 500)
    Y = 2.0 / D + 0.05 * rng.randn(len(D))
    g0, diag = fit_g0(D, Y)
    assert diag['n_points'] == 500
    assert diag['power_r2'] > 0.8


def test_fit_diagnostics():
    D = np.linspace(0.5, 10.0, 100)
    Y = 1.0 / D + 0.01
    _, diag = fit_g0(D, Y)
    assert 'agreement_max_abs_diff' in diag
    assert 'isotonic_r2' in diag
    assert 'power_r2' in diag
```

- [ ] **Step 2: Run tests (expect failure)**

```bash
pytest famail_temporal/tests/test_g0_power_basis.py::test_fit_recovers_hyperbolic -v
```

- [ ] **Step 3: Append to `fairness/g0_power_basis.py`**

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from famail_temporal import config


def fit(demands: np.ndarray, supplies_over_demands: np.ndarray) -> tuple[G0Function, dict]:
    """Fit g₀(D) on (D, Y=S/D) pairs at block-mean scale.

    Returns (g0_func, diagnostics) where diagnostics has:
      'n_points', 'power_r2', 'isotonic_r2', 'agreement_max_abs_diff'.
    """
    D = np.maximum(np.asarray(demands, dtype=np.float64), config.DEMAND_FLOOR)
    Y = np.asarray(supplies_over_demands, dtype=np.float64)

    X = build_power_basis_features(D, include_intercept=True)
    lr = LinearRegression(fit_intercept=False).fit(X, Y)
    g0 = G0Function(
        coefficients=lr.coef_,
        d_min=float(D.min()),
        d_max=float(D.max()),
    )

    iso = IsotonicRegression(increasing=False, out_of_bounds='clip').fit(D, Y)
    y_power = g0(D)
    y_iso = iso.predict(D)
    max_abs_diff = float(np.max(np.abs(y_power - y_iso)))

    y_var = float(np.var(Y)) + 1e-10
    diagnostics = {
        'n_points': int(len(D)),
        'power_r2': float(1.0 - np.var(Y - y_power) / y_var),
        'isotonic_r2': float(1.0 - np.var(Y - y_iso) / y_var),
        'agreement_max_abs_diff': max_abs_diff,
    }
    return g0, diagnostics
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest famail_temporal/tests/test_g0_power_basis.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/fairness/g0_power_basis.py famail_temporal/tests/test_g0_power_basis.py
git commit -m "feat(fairness): fit() with isotonic diagnostic"
```

---

### Task 11: `precompute_hat_matrices`

**Files:**
- Create: `famail_temporal/fairness/hat_matrices.py`
- Create: `famail_temporal/tests/test_hat_matrices.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for fairness.hat_matrices."""
import numpy as np
import pytest

from famail_temporal.fairness.hat_matrices import precompute_hat_matrices


def test_shapes():
    rng = np.random.RandomState(0)
    N = 50
    D = rng.uniform(0.5, 5.0, N)
    demo = rng.randn(N, 3)
    hat = precompute_hat_matrices(D, demo, ["f1", "f2", "f3"])
    assert hat['I_minus_H_demo'].shape == (N, N)
    assert hat['M'].shape == (N, N)
    assert hat['n_units'] == N


def test_I_minus_H_idempotent():
    rng = np.random.RandomState(1)
    N = 40
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    IH = hat['I_minus_H_demo']
    np.testing.assert_allclose(IH @ IH, IH, atol=1e-10)


def test_M_centering():
    rng = np.random.RandomState(2)
    N = 30
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    M = hat['M']
    np.testing.assert_allclose(M @ M, M, atol=1e-10)
    np.testing.assert_allclose(M @ np.ones(N), np.zeros(N), atol=1e-10)


def test_rank_deficient_raises():
    rng = np.random.RandomState(3)
    N = 30
    col1 = rng.randn(N)
    demo = np.column_stack([col1, col1, rng.randn(N)])
    with pytest.raises(AssertionError, match="rank"):
        precompute_hat_matrices(rng.uniform(0.5, 5.0, N), demo, ["a", "b", "c"])
```

- [ ] **Step 2: Run tests (expect failure)**

```bash
pytest famail_temporal/tests/test_hat_matrices.py -v
```

- [ ] **Step 3: Write `famail_temporal/fairness/hat_matrices.py` (partial)**

```python
"""
Pre-compute hat matrices for pooled Option B F_causal.

Inputs are active-unit vectors (length N). Constants during optimization —
only the residual vector R changes across forward passes.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import StandardScaler


def precompute_hat_matrices(
    demands: np.ndarray,
    demographic_features: np.ndarray,
    feature_names: List[str],
) -> Dict[str, np.ndarray]:
    """Build (I - H_demo), M, and diagnostics.

    H_demo projects onto [1, standardized(demographics)] (intercept included).
    M = I - 11'/N is the centering matrix.
    Asserts H_demo has full rank.
    """
    N = len(demands)
    assert demographic_features.shape == (N, len(feature_names)), (
        f"demographic_features shape {demographic_features.shape} "
        f"inconsistent with N={N} and {len(feature_names)} features"
    )

    scaler = StandardScaler()
    X_demo_scaled = scaler.fit_transform(demographic_features)
    X = np.column_stack([np.ones(N), X_demo_scaled])

    H = X @ np.linalg.pinv(X)
    rank_H = int(np.linalg.matrix_rank(H))
    expected_rank = X.shape[1]
    assert rank_H == expected_rank, (
        f"H_demo rank {rank_H}, expected {expected_rank}. "
        "Demographic collinearity — check feature set."
    )

    I_minus_H_demo = np.eye(N) - H
    M = np.eye(N) - np.ones((N, N)) / N

    return {
        'I_minus_H_demo': I_minus_H_demo,
        'M': M,
        'scaler_mean': scaler.mean_,
        'scaler_std': scaler.scale_,
        'n_units': N,
        'n_demo_features': len(feature_names),
        'feature_names': feature_names,
        'rank_H_demo': rank_H,
    }
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest famail_temporal/tests/test_hat_matrices.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/fairness/hat_matrices.py famail_temporal/tests/test_hat_matrices.py
git commit -m "feat(fairness): precompute_hat_matrices with rank assertion"
```

---

### Task 12: `compute_fcausal_torch`

**Files:**
- Modify: `famail_temporal/fairness/hat_matrices.py` (append)
- Modify: `famail_temporal/tests/test_hat_matrices.py` (append)

- [ ] **Step 1: Append failing tests**

```python
import torch
from sklearn.preprocessing import StandardScaler

from famail_temporal.fairness.hat_matrices import compute_fcausal_torch


def test_fcausal_zero_when_R_in_demographic_span():
    N = 50
    rng = np.random.RandomState(4)
    D = rng.uniform(0.5, 5.0, N)
    demo = rng.randn(N, 3)
    hat = precompute_hat_matrices(D, demo, ["f1", "f2", "f3"])
    IH = torch.from_numpy(hat['I_minus_H_demo']).float()
    M = torch.from_numpy(hat['M']).float()
    X_scaled = StandardScaler().fit_transform(demo)
    R = torch.from_numpy(2.0 + 1.5 * X_scaled[:, 0]).float()
    f = compute_fcausal_torch(R, IH, M)
    assert float(f) < 1e-4


def test_fcausal_bounded():
    N = 80
    rng = np.random.RandomState(5)
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    IH = torch.from_numpy(hat['I_minus_H_demo']).float()
    M = torch.from_numpy(hat['M']).float()
    R = torch.randn(N) * 3.0
    f = compute_fcausal_torch(R, IH, M)
    assert 0.0 <= float(f) <= 1.0


def test_fcausal_degenerate_returns_one():
    N = 30
    rng = np.random.RandomState(6)
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    IH = torch.from_numpy(hat['I_minus_H_demo']).float()
    M = torch.from_numpy(hat['M']).float()
    R = torch.full((N,), 0.5)
    f = compute_fcausal_torch(R, IH, M)
    assert float(f) == 1.0
```

- [ ] **Step 2: Run tests (expect failure)**

```bash
pytest famail_temporal/tests/test_hat_matrices.py -v
```

- [ ] **Step 3: Append to `fairness/hat_matrices.py`**

```python
import torch
from famail_temporal import config


def compute_fcausal_torch(
    R: torch.Tensor,
    I_minus_H_demo: torch.Tensor,
    M: torch.Tensor,
    eps: float = config.EPS,
) -> torch.Tensor:
    """Pooled Option B: F_causal = R'(I-H)R / R'MR, clamped to [0, 1]."""
    ss_res_demo = R @ I_minus_H_demo @ R
    ss_tot = R @ M @ R
    f_causal = torch.where(
        ss_tot < eps,
        torch.ones_like(ss_tot),
        ss_res_demo / (ss_tot + eps),
    )
    return torch.clamp(f_causal, 0.0, 1.0)
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest famail_temporal/tests/test_hat_matrices.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/fairness/hat_matrices.py famail_temporal/tests/test_hat_matrices.py
git commit -m "feat(fairness): differentiable compute_fcausal_torch"
```

---

## Phase 4: F_spatial + F_causal + attribution (Tasks 13–16)

### Task 13: `spatial.py` — pooled Gini + F_spatial

**Files:**
- Create: `famail_temporal/fairness/spatial.py`
- Create: `famail_temporal/tests/test_spatial_fairness.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for fairness.spatial."""
import torch

from famail_temporal.fairness.spatial import pairwise_gini, compute_fspatial


def test_gini_equal_values_zero():
    values = torch.full((20,), 3.0)
    assert float(pairwise_gini(values)) < 1e-6


def test_gini_one_hot_approaches_max():
    values = torch.zeros(10)
    values[0] = 100.0
    g = float(pairwise_gini(values))
    assert 0.85 < g <= 0.91


def test_fspatial_perfect_equality():
    N = 30
    pickup = torch.full((N,), 2.0)
    dropoff = torch.full((N,), 2.0)
    active = torch.full((N,), 4.0)
    f, _ = compute_fspatial(pickup, dropoff, active)
    assert float(f) > 0.999


def test_fspatial_bounded():
    N = 50
    torch.manual_seed(42)
    pickup = torch.rand(N) * 5.0
    dropoff = torch.rand(N) * 5.0
    active = torch.rand(N) * 3.0 + 1.0
    f, _ = compute_fspatial(pickup, dropoff, active)
    assert 0.0 <= float(f) <= 1.0
```

- [ ] **Step 2: Run tests (expect failure)**

```bash
pytest famail_temporal/tests/test_spatial_fairness.py -v
```

- [ ] **Step 3: Write `famail_temporal/fairness/spatial.py`**

```python
"""Pooled spatial fairness: one Gini over all active (cell, t) units."""

from __future__ import annotations
from typing import Tuple

import torch

from famail_temporal import config


def pairwise_gini(values: torch.Tensor) -> torch.Tensor:
    """Differentiable pairwise Gini: G = Σᵢ Σⱼ |xᵢ - xⱼ| / (2n²μ)."""
    n = values.numel()
    if n <= 1:
        return torch.tensor(0.0, device=values.device)
    mean_val = values.mean() + config.EPS
    diff = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))
    gini = diff.sum() / (2 * n * n * mean_val)
    return torch.clamp(gini, 0.0, 1.0)


def compute_fspatial(
    pickup_N: torch.Tensor,
    dropoff_N: torch.Tensor,
    active_taxis_N: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """F_spatial = 1 - 0.5·(Gini(DSR) + Gini(ASR))."""
    dsr = pickup_N / (active_taxis_N + config.EPS)
    asr = dropoff_N / (active_taxis_N + config.EPS)
    gini_dsr = pairwise_gini(dsr)
    gini_asr = pairwise_gini(asr)
    f_spatial = 1.0 - 0.5 * (gini_dsr + gini_asr)
    debug = {'gini_dsr': float(gini_dsr), 'gini_asr': float(gini_asr)}
    return f_spatial, debug
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest famail_temporal/tests/test_spatial_fairness.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/fairness/spatial.py famail_temporal/tests/test_spatial_fairness.py
git commit -m "feat(fairness): pooled pairwise Gini + compute_fspatial"
```

---

### Task 14: `causal.py` — F_causal + per-unit attribution

**Files:**
- Create: `famail_temporal/fairness/causal.py`
- Create: `famail_temporal/tests/test_causal_fairness.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for fairness.causal."""
import numpy as np
import torch

from famail_temporal.fairness.causal import compute_fcausal, per_unit_attribution
from famail_temporal.fairness.hat_matrices import precompute_hat_matrices, compute_fcausal_torch


def _make_hat(N, seed):
    rng = np.random.RandomState(seed)
    D = rng.uniform(0.5, 5.0, N)
    demo = rng.randn(N, 3)
    hat = precompute_hat_matrices(D, demo, ["f1", "f2", "f3"])
    IH = torch.from_numpy(hat['I_minus_H_demo']).float()
    M = torch.from_numpy(hat['M']).float()
    return D, demo, IH, M


def test_fcausal_in_unit_interval():
    N = 40
    D, _, IH, M = _make_hat(N, seed=10)
    supply = torch.from_numpy(np.abs(np.random.RandomState(11).randn(N)) * 2.0 + 1.0).float()
    d_t = torch.from_numpy(D).float()
    g0_D = torch.full((N,), 0.5)
    f, _ = compute_fcausal(d_t, supply, g0_D, IH, M)
    assert 0.0 <= float(f) <= 1.0


def test_attribution_sums_to_one_minus_fcausal():
    N = 80
    D, _, IH, M = _make_hat(N, seed=12)
    R = torch.from_numpy(np.random.RandomState(13).randn(N) * 2.0).float()
    f = compute_fcausal_torch(R, IH, M)
    attr = per_unit_attribution(R, IH, M)
    assert abs(float(attr.sum()) - (1.0 - float(f))) < 1e-5


def test_attribution_shape():
    N = 50
    D, _, IH, M = _make_hat(N, seed=14)
    R = torch.randn(N)
    attr = per_unit_attribution(R, IH, M)
    assert attr.shape == (N,)
```

- [ ] **Step 2: Run tests (expect failure)**

```bash
pytest famail_temporal/tests/test_causal_fairness.py -v
```

- [ ] **Step 3: Write `famail_temporal/fairness/causal.py`**

```python
"""
Pooled Option B F_causal + per-unit attribution.

F_causal = R'(I-H_demo)R / R'MR  where R = Y - g₀(D), Y = S/D.

Per-unit attribution decomposes `1 - F_causal = r²_demo`:
    attribution_i = ((MR)_i² - ((I-H)R)_i²) / R'MR
    Σᵢ attribution_i == 1 - F_causal
"""

from __future__ import annotations
from typing import Tuple

import torch

from famail_temporal import config
from famail_temporal.fairness.hat_matrices import compute_fcausal_torch


def compute_fcausal(
    demand_N: torch.Tensor,
    supply_N: torch.Tensor,
    g0_D_N: torch.Tensor,
    I_minus_H_demo: torch.Tensor,
    M: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Gradient flow: demand_N → Y → R → F_causal."""
    D = torch.clamp(demand_N, min=config.DEMAND_FLOOR)
    Y = supply_N / (D + config.EPS)
    R = Y - g0_D_N
    f_causal = compute_fcausal_torch(R, I_minus_H_demo, M)
    debug = {
        'Y_min': float(Y.min()), 'Y_max': float(Y.max()),
        'R_min': float(R.min()), 'R_max': float(R.max()),
        'f_causal': float(f_causal),
    }
    return f_causal, debug


def per_unit_attribution(
    R: torch.Tensor,
    I_minus_H_demo: torch.Tensor,
    M: torch.Tensor,
) -> torch.Tensor:
    """Per-unit contribution to demographic-explained variance."""
    with torch.no_grad():
        MR = M @ R
        IHR = I_minus_H_demo @ R
        ss_tot_vec = MR ** 2
        ss_res_vec = IHR ** 2
        ss_explained_vec = ss_tot_vec - ss_res_vec
        ss_tot_scalar = ss_tot_vec.sum() + config.EPS
        return ss_explained_vec / ss_tot_scalar


def per_unit_attribution_signed(
    R: torch.Tensor,
    I_minus_H_demo: torch.Tensor,
    M: torch.Tensor,
) -> torch.Tensor:
    """Signed attribution: sign of (HR) indicates under/over-service."""
    with torch.no_grad():
        HR = R - I_minus_H_demo @ R
        magnitudes = per_unit_attribution(R, I_minus_H_demo, M)
        return magnitudes * torch.sign(HR)
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest famail_temporal/tests/test_causal_fairness.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/fairness/causal.py famail_temporal/tests/test_causal_fairness.py
git commit -m "feat(fairness): compute_fcausal + per-unit attribution"
```

---

### Task 15: Wire `fairness/__init__.py`

**Files:**
- Modify: `famail_temporal/fairness/__init__.py`

- [ ] **Step 1: Write file**

```python
"""Pooled (cell, t) fairness metrics."""

from famail_temporal.fairness.spatial import (
    pairwise_gini,
    compute_fspatial,
)
from famail_temporal.fairness.causal import (
    compute_fcausal,
    per_unit_attribution,
    per_unit_attribution_signed,
)
from famail_temporal.fairness.hat_matrices import (
    precompute_hat_matrices,
    compute_fcausal_torch,
)
from famail_temporal.fairness.g0_power_basis import (
    G0Function,
    build_power_basis_features,
    fit as fit_g0,
)

__all__ = [
    "pairwise_gini", "compute_fspatial",
    "compute_fcausal", "per_unit_attribution", "per_unit_attribution_signed",
    "precompute_hat_matrices", "compute_fcausal_torch",
    "G0Function", "build_power_basis_features", "fit_g0",
]
```

- [ ] **Step 2: Verify imports**

```bash
python -c "from famail_temporal.fairness import compute_fspatial, compute_fcausal, per_unit_attribution, fit_g0; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add famail_temporal/fairness/__init__.py
git commit -m "chore(fairness): expose public API via __init__"
```

---

### Task 16: Cross-module mathematical invariants

**Files:**
- Create: `famail_temporal/tests/test_math_invariants.py`

- [ ] **Step 1: Write tests**

```python
"""Mathematical invariants across the fairness math stack.

These guard properties that a reviewer might verify by hand from the equations
in the Methods section.
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from famail_temporal.fairness import (
    compute_fcausal_torch,
    per_unit_attribution,
    precompute_hat_matrices,
    pairwise_gini,
    compute_fspatial,
)


def test_I_minus_H_idempotent():
    rng = np.random.RandomState(100)
    N = 60
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    IH = hat['I_minus_H_demo']
    np.testing.assert_allclose(IH @ IH, IH, atol=1e-9)


def test_M_idempotent():
    rng = np.random.RandomState(101)
    N = 60
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    M = hat['M']
    np.testing.assert_allclose(M @ M, M, atol=1e-9)


def test_attribution_sum_property():
    rng = np.random.RandomState(102)
    N = 100
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    IH = torch.from_numpy(hat['I_minus_H_demo']).float()
    M = torch.from_numpy(hat['M']).float()
    for seed in range(5):
        R = torch.from_numpy(np.random.RandomState(seed + 200).randn(N) * 3.0).float()
        f = compute_fcausal_torch(R, IH, M)
        attr = per_unit_attribution(R, IH, M)
        assert abs(float(attr.sum()) - (1.0 - float(f))) < 1e-5


def test_fcausal_zero_when_R_in_demographic_span():
    rng = np.random.RandomState(103)
    N = 70
    demo = rng.randn(N, 3)
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), demo, ["a", "b", "c"])
    IH = torch.from_numpy(hat['I_minus_H_demo']).float()
    M = torch.from_numpy(hat['M']).float()
    X_scaled = StandardScaler().fit_transform(demo)
    R = torch.from_numpy(1.0 + 0.5 * X_scaled[:, 0] + 0.2 * X_scaled[:, 1]).float()
    assert float(compute_fcausal_torch(R, IH, M)) < 1e-4


def test_gini_scale_invariance():
    rng = np.random.RandomState(104)
    x = torch.from_numpy(rng.rand(50) * 10.0).float()
    g1 = float(pairwise_gini(x))
    g2 = float(pairwise_gini(x * 7.3))
    assert abs(g1 - g2) < 1e-5


def test_fspatial_one_when_equal():
    N = 40
    pickup = torch.full((N,), 3.0)
    dropoff = torch.full((N,), 3.0)
    active = torch.full((N,), 5.0)
    f, _ = compute_fspatial(pickup, dropoff, active)
    assert float(f) > 0.999
```

- [ ] **Step 2: Run tests (expect pass)**

```bash
pytest famail_temporal/tests/test_math_invariants.py -v
```

Expected: 6 passed.

- [ ] **Step 3: Commit**

```bash
git add famail_temporal/tests/test_math_invariants.py
git commit -m "test(fairness): cross-module mathematical invariants"
```

---

**End of Phase 1–4 file.** At this checkpoint the fairness math stack is complete and tested. Continue with `2026-04-16-famail-temporal-phase5-6.md` (Data pipeline + Fidelity port), then `2026-04-16-famail-temporal-phase7-8.md` (Algorithm + Integration), then `2026-04-16-famail-temporal-phase9.md` (Documentation).
