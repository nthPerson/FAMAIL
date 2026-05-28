# FAMAIL GAN Baselines — Phase 1: Data-Level Pareto Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data-level fairness × retention Pareto — the no-GAN "fallback" claim that editing dominates filtering on the dataset itself — as the foundation the GAN phases reuse.

**Architecture:** A new `famail_temporal/baselines/` package with pure, unit-tested functions that (a) rank seeking trajectories by unfairness, (b) build the *filtered* demand grid by subtracting each removed trajectory's pickup mass (the same `1/(n_hours_per_block·n_days)` mass the editing modifier uses, so filtering and editing are accounted for consistently), (c) reduce any demand grid to the canonical fairness scalars via the existing `build_fairness_grid` + `_scalar_metrics_from_grid`, and (d) assemble raw / filtered@K-sweep / edited points into a Pareto structure with a figure. The edited point is fetched from the existing `run_experiment`.

**Tech Stack:** Python 3.12, NumPy, PyTorch (transitively, via the fairness grid), pytest, matplotlib. Reuses `famail_temporal.fairness`, `famail_temporal.algorithm.attribution`, `famail_temporal.evaluation.grid`, `famail_temporal.evaluation.runner`.

---

## Scope: this is Phase 1 of 5

The spec (`docs/superpowers/specs/2026-05-27-famail-gan-baselines-design.md` §6) defines a 5-phase build order. This plan covers **Phase 1 only** (data-variant builders + the no-GAN data-level Pareto). It produces working, testable software on its own: the fairness×retention curve that is the data-scarcity argument's fallback. Phases 2–5 (B0 GAN infra, FAMAIL/B2 model-level, B1 fairness-loss, pure-GAN ablation + signal-max scale-up) get their own plans once Phase 1 lands.

**Deferred out of Phase 1 (documented, not stubbed):**
- **District disparate-impact (DI) ratio metric.** It needs the `grid_to_district_mapping.pkl` + hukou demographics and a confirmed per-district supply/demand definition. That definition is an intermediate-calculation decision requiring user sign-off (per the algorithm-change protocol), so it is intentionally NOT written here. It will be added as Phase 1b after sign-off; the `ParetoPoint` dataclass leaves room to extend.
- **Coordinate-descent / iterative re-attribution editing** and **hukou `NonRegisteredRatio` feature** — both gated (spec §3.4). Phase 1 uses the existing one-shot `run_experiment` for the edited point.

---

## File Structure

| File | Responsibility |
|---|---|
| `famail_temporal/baselines/__init__.py` | Package marker (empty) |
| `famail_temporal/baselines/datasets.py` | Trajectory→unit helpers, `pickup_mass`, unfairness ranking, filtered demand-grid builder |
| `famail_temporal/baselines/metrics.py` | `data_level_fairness(bundle, pickup_3d)` → canonical fairness scalars |
| `famail_temporal/baselines/pareto.py` | `ParetoPoint`, `raw_point`, `filtered_points`, `edited_point`, JSON serialize |
| `famail_temporal/baselines/figure.py` | `plot_pareto(points, path)` matplotlib figure |
| `famail_temporal/baselines/run_data_pareto.py` | CLI: load bundle → raw+filtered sweep → optional edited point → JSON + PNG |
| `famail_temporal/baselines/tests/__init__.py` | Test package marker (empty) |
| `famail_temporal/baselines/tests/_helpers.py` | Build synthetic trajectories on active units |
| `famail_temporal/baselines/tests/test_datasets.py` | Unit tests for datasets.py |
| `famail_temporal/baselines/tests/test_metrics.py` | Unit tests for metrics.py |
| `famail_temporal/baselines/tests/test_pareto.py` | Unit tests for pareto.py |
| `famail_temporal/baselines/tests/test_figure.py` | Smoke test for figure.py |

**Conventions reused (do not redefine):**
- Fairness sign: `f_spatial`/`f_causal` ∈ [0,1], **1 = fairest** (`_scalar_metrics_from_grid` docstring in `evaluation/runner.py`).
- Per-trajectory pickup mass: `1/(bundle.n_hours_per_block[t_block] · bundle.n_days)` (`algorithm/modifier.py:304-306`).
- Trajectory→unit: `(cx,cy)=traj.pickup_cell`; `t_block=hour_to_block_index(time_bucket_to_hour(traj.pickup_state.time_bucket))`.
- Unfairness ranking: `compute_per_unit_attribution(bundle)` then `rank_trajectories(...)` (ascending; most-negative = most-unfair first); strictly-negative scores are the only filtering candidates.

---

## Task 1: Scaffold the `baselines` package

**Files:**
- Create: `famail_temporal/baselines/__init__.py`
- Create: `famail_temporal/baselines/tests/__init__.py`
- Test: `famail_temporal/baselines/tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Create `famail_temporal/baselines/tests/test_smoke.py`:

```python
"""Smoke test: the baselines package imports."""


def test_baselines_package_imports():
    import famail_temporal.baselines as b
    assert b is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'famail_temporal.baselines'`

- [ ] **Step 3: Create the package markers**

Create `famail_temporal/baselines/__init__.py` (empty file, single line):

```python
"""FAMAIL GAN baselines: data-level Pareto (Phase 1) and GAN phases (later)."""
```

Create `famail_temporal/baselines/tests/__init__.py` (empty file):

```python
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/__init__.py famail_temporal/baselines/tests/__init__.py famail_temporal/baselines/tests/test_smoke.py
git commit -m "feat(baselines): scaffold baselines package for data-level Pareto"
```

---

## Task 2: Test helper — synthetic trajectories on active units

The synthetic bundle from `test_objective._make_synthetic_bundle()` has `trajectories=[]`. This helper builds 2-state `Trajectory` objects whose terminal pickup lands on a chosen active `(cell, t_block)`, so the datasets/pareto tests have real ranking + filtering inputs.

**Files:**
- Create: `famail_temporal/baselines/tests/_helpers.py`
- Test: `famail_temporal/baselines/tests/test_datasets.py` (created here, exercised in Task 3)

- [ ] **Step 1: Write the helper**

Create `famail_temporal/baselines/tests/_helpers.py`:

```python
"""Test helpers: build synthetic trajectories on a bundle's active units."""
from __future__ import annotations
from typing import List, Tuple

from famail_temporal import config
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


def active_units(bundle, n: int) -> List[Tuple[int, int, int]]:
    """Return up to n active (cx, cy, t_block) triples from the bundle mask."""
    gx, gy, T = bundle.mask_3d.shape
    out: List[Tuple[int, int, int]] = []
    for t in range(T):
        for x in range(gx):
            for y in range(gy):
                if bundle.mask_3d[x, y, t]:
                    out.append((x, y, t))
                    if len(out) >= n:
                        return out
    return out


def time_bucket_for_block(t_block: int) -> int:
    """A 1-indexed 5-min time_bucket whose hour maps back to t_block.

    Uses the block's start hour from config.TIME_BLOCKS; time_bucket =
    start_hour*12 + 1 so time_bucket_to_hour(...) == start_hour and
    hour_to_block_index(start_hour) == t_block.
    """
    start_hour = config.TIME_BLOCKS[t_block][1]
    return start_hour * 12 + 1


def make_traj_at(cx: int, cy: int, t_block: int, traj_id: int) -> Trajectory:
    """A 2-state trajectory whose terminal (pickup) state is at (cx, cy, t_block)."""
    tb = time_bucket_for_block(t_block)
    states = [
        TrajectoryState(x_grid=float(cx), y_grid=float(cy), time_bucket=tb, day_index=1),
        TrajectoryState(x_grid=float(cx), y_grid=float(cy), time_bucket=tb, day_index=1),
    ]
    return Trajectory(trajectory_id=traj_id, driver_id=0, states=states)
```

- [ ] **Step 2: Write a test that the helper produces valid pickup units**

Create `famail_temporal/baselines/tests/test_datasets.py`:

```python
"""Unit tests for famail_temporal.baselines.datasets."""
import numpy as np
import pytest

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import (
    active_units, make_traj_at,
)
from famail_temporal.baselines import datasets as ds


def test_helper_pickup_unit_round_trips():
    bundle = _make_synthetic_bundle()
    (cx, cy, t_block) = active_units(bundle, 1)[0]
    traj = make_traj_at(cx, cy, t_block, traj_id=0)
    assert ds.pickup_unit_of(traj) == (cx, cy, t_block)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_datasets.py::test_helper_pickup_unit_round_trips -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'famail_temporal.baselines.datasets'` (datasets.py not yet created — implemented in Task 3)

- [ ] **Step 4: Commit the helper (test stays red until Task 3)**

```bash
git add famail_temporal/baselines/tests/_helpers.py famail_temporal/baselines/tests/test_datasets.py
git commit -m "test(baselines): add synthetic-trajectory test helper"
```

---

## Task 3: `datasets.py` — unit helpers, ranking, filtered demand grid

**Files:**
- Create: `famail_temporal/baselines/datasets.py`
- Test: `famail_temporal/baselines/tests/test_datasets.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `test_datasets.py`)

```python
def test_pickup_mass_matches_modifier_formula():
    bundle = _make_synthetic_bundle()
    t_block = active_units(bundle, 1)[0][2]
    expected = 1.0 / (int(bundle.n_hours_per_block[t_block]) * bundle.n_days)
    assert ds.pickup_mass(bundle, t_block) == pytest.approx(expected)


def test_build_filtered_subtracts_mass_at_unit_only():
    bundle = _make_synthetic_bundle()
    (cx, cy, t_block) = active_units(bundle, 1)[0]
    traj = make_traj_at(cx, cy, t_block, traj_id=0)
    before = bundle.pickup_3d.copy()
    filtered = ds.build_filtered_pickup_3d(bundle, [traj])
    mass = ds.pickup_mass(bundle, t_block)
    # Target cell dropped by exactly one pickup mass.
    assert filtered[cx, cy, t_block] == pytest.approx(before[cx, cy, t_block] - mass)
    # Everything else identical.
    delta = before - filtered
    delta[cx, cy, t_block] = 0.0
    assert np.allclose(delta, 0.0)
    # Bundle's own grid is untouched (copy semantics).
    assert np.allclose(bundle.pickup_3d, before)


def test_rank_returns_only_negative_scores_most_unfair_first():
    bundle = _make_synthetic_bundle()
    # Put one trajectory on every active unit so ranking has candidates.
    units = active_units(bundle, 25)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    ranked = ds.rank_unfair_trajectory_indices(bundle)
    # All returned indices are valid and unique.
    assert len(set(ranked)) == len(ranked)
    assert all(0 <= i < len(bundle.trajectories) for i in ranked)
    # Recompute scores and confirm every returned idx is strictly negative
    # and the list is ascending (most-negative first).
    from famail_temporal.algorithm.attribution import (
        compute_per_unit_attribution, rank_trajectories,
    )
    attribution = compute_per_unit_attribution(bundle)
    scored = dict(rank_trajectories(bundle.trajectories, attribution, bundle.unit_map))
    returned_scores = [scored[i] for i in ranked]
    assert all(s < 0 for s in returned_scores)
    assert returned_scores == sorted(returned_scores)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_datasets.py -v`
Expected: FAIL (ModuleNotFoundError on `famail_temporal.baselines.datasets`)

- [ ] **Step 3: Implement `datasets.py`**

Create `famail_temporal/baselines/datasets.py`:

```python
"""Dataset-variant builders for the FAMAIL GAN baselines (Phase 1, data-level).

Defines the raw and *filtered* demand-grid variants used by the data-level
fairness x retention Pareto. The filtered variant removes the top-K
most-unfair seeking trajectories and subtracts each removed trajectory's
pickup contribution from the demand grid, using the SAME per-trajectory
pickup mass the editing modifier uses (1/(n_hours_per_block[t_block]*n_days),
see algorithm/modifier.py), so filtering and editing are accounted for
consistently. Supply (active_taxis) is left unchanged — filtering removes a
demand event, not taxi presence, mirroring the editing convention.
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np

from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution, rank_trajectories,
)
from famail_temporal.data.aggregation import (
    hour_to_block_index, time_bucket_to_hour,
)
from famail_temporal.data.loader import DataBundle
from famail_temporal.utils.trajectory import Trajectory


def pickup_unit_of(traj: Trajectory) -> Tuple[int, int, int]:
    """Return (cx, cy, t_block) for a trajectory's terminal pickup."""
    cx, cy = traj.pickup_cell
    t_block = hour_to_block_index(
        time_bucket_to_hour(traj.pickup_state.time_bucket)
    )
    return cx, cy, t_block


def pickup_mass(bundle: DataBundle, t_block: int) -> float:
    """Mean-hourly demand mass of one pickup event in t_block.

    Matches TrajectoryModifier: 1 / (n_hours_per_block[t_block] * n_days).
    """
    n_hours = int(bundle.n_hours_per_block[t_block])
    return 1.0 / (n_hours * bundle.n_days)


def rank_unfair_trajectory_indices(bundle: DataBundle) -> List[int]:
    """Indices into bundle.trajectories ordered most-unfair first.

    Only strictly-negative-attribution trajectories (pickup cells dragging
    fairness below the 1/N baseline) are returned; at/above-baseline
    (score >= 0) and inactive (+inf) trajectories are excluded — they are
    not filtering candidates.
    """
    attribution = compute_per_unit_attribution(bundle)
    scored = rank_trajectories(
        bundle.trajectories, attribution, bundle.unit_map,
    )
    return [idx for idx, score in scored if score < 0]


def build_filtered_pickup_3d(
    bundle: DataBundle, removed_trajs: List[Trajectory],
) -> np.ndarray:
    """Demand grid after removing the given trajectories' pickup events.

    Returns a fresh array (bundle.pickup_3d is not mutated). Cells may go
    below DEMAND_FLOOR here; the downstream fairness grid clamps demand, so
    no clamping is applied at this layer.
    """
    pickup_3d = bundle.pickup_3d.copy()
    for traj in removed_trajs:
        cx, cy, t_block = pickup_unit_of(traj)
        pickup_3d[cx, cy, t_block] -= pickup_mass(bundle, t_block)
    return pickup_3d
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_datasets.py -v`
Expected: PASS (4 tests: round-trip, pickup_mass, filtered-subtract, ranking)

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/datasets.py famail_temporal/baselines/tests/test_datasets.py
git commit -m "feat(baselines): filtered demand-grid builder + unfairness ranking"
```

---

## Task 4: `metrics.py` — data-level fairness of a demand grid

**Files:**
- Create: `famail_temporal/baselines/metrics.py`
- Test: `famail_temporal/baselines/tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `famail_temporal/baselines/tests/test_metrics.py`:

```python
"""Unit tests for famail_temporal.baselines.metrics."""
import numpy as np
import pytest

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines import metrics as m
from famail_temporal.baselines.tests._helpers import active_units, make_traj_at
from famail_temporal.baselines import datasets as ds


def test_data_level_fairness_keys_and_ranges():
    bundle = _make_synthetic_bundle()
    out = m.data_level_fairness(bundle)
    assert set(out) == {"f_spatial", "f_causal", "gini_dsr", "gini_asr"}
    assert 0.0 <= out["f_spatial"] <= 1.0
    assert 0.0 <= out["f_causal"] <= 1.0


def test_data_level_fairness_default_matches_explicit_grid():
    bundle = _make_synthetic_bundle()
    out_default = m.data_level_fairness(bundle)
    out_explicit = m.data_level_fairness(bundle, pickup_3d=bundle.pickup_3d)
    assert out_default == out_explicit


def test_removing_an_unfair_trajectory_does_not_lower_f_causal():
    """Filtering the single most-unfair trajectory should not reduce F_causal."""
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 25)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    ranked = ds.rank_unfair_trajectory_indices(bundle)
    assert ranked, "expected at least one strictly-unfair trajectory"
    removed = [bundle.trajectories[ranked[0]]]
    f_raw = m.data_level_fairness(bundle)["f_causal"]
    f_filt = m.data_level_fairness(
        bundle, pickup_3d=ds.build_filtered_pickup_3d(bundle, removed),
    )["f_causal"]
    assert f_filt >= f_raw - 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_metrics.py -v`
Expected: FAIL (ModuleNotFoundError on `famail_temporal.baselines.metrics`)

- [ ] **Step 3: Implement `metrics.py`**

Create `famail_temporal/baselines/metrics.py`:

```python
"""Data-level fairness metrics for a demand grid (Phase 1).

Reuses the canonical evaluation grid + scalar reduction so values match the
editing pipeline exactly (fairness convention: 1 = fairest).
"""
from __future__ import annotations

import numpy as np

from famail_temporal.data.loader import DataBundle
from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.evaluation.runner import _scalar_metrics_from_grid


def data_level_fairness(
    bundle: DataBundle, pickup_3d: np.ndarray | None = None,
) -> dict:
    """Return {f_spatial, f_causal, gini_dsr, gini_asr} for a demand grid.

    pickup_3d=None evaluates the bundle's own demand grid (the raw variant).
    Pass a filtered/edited demand grid to evaluate a variant.
    """
    grid = build_fairness_grid(bundle, pickup_3d=pickup_3d)
    return _scalar_metrics_from_grid(grid)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_metrics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/metrics.py famail_temporal/baselines/tests/test_metrics.py
git commit -m "feat(baselines): data-level fairness reduction for a demand grid"
```

---

## Task 5: `pareto.py` — ParetoPoint + raw/filtered/edited assembly

**Files:**
- Create: `famail_temporal/baselines/pareto.py`
- Test: `famail_temporal/baselines/tests/test_pareto.py`

- [ ] **Step 1: Write the failing tests**

Create `famail_temporal/baselines/tests/test_pareto.py`:

```python
"""Unit tests for famail_temporal.baselines.pareto."""
import json

import pytest

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import active_units, make_traj_at
from famail_temporal.baselines import pareto as p


def _bundle_with_trajs(n=25):
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, n)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    return bundle


def test_raw_point_has_full_retention():
    bundle = _bundle_with_trajs()
    pt = p.raw_point(bundle)
    assert pt.label == "raw"
    assert pt.retention == 1.0
    assert pt.n_removed == 0


def test_filtered_points_retention_math():
    bundle = _bundle_with_trajs()
    n = len(bundle.trajectories)
    pts = p.filtered_points(bundle, k_levels=[1, 3])
    assert [pt.label for pt in pts] == ["filter@1", "filter@3"]
    assert pts[0].retention == pytest.approx((n - pts[0].n_removed) / n)
    # More filtering => lower retention.
    assert pts[1].retention <= pts[0].retention


def test_filtered_k_capped_at_candidate_count():
    bundle = _bundle_with_trajs()
    huge = 10 ** 9
    pts = p.filtered_points(bundle, k_levels=[huge])
    assert pts[0].n_removed <= len(bundle.trajectories)


def test_edited_point_is_full_retention():
    pt = p.edited_point(
        f_spatial=0.10, f_causal=0.81, gini_dsr=0.9, gini_asr=0.9,
    )
    assert pt.label == "edit"
    assert pt.retention == 1.0
    assert pt.n_removed == 0


def test_points_to_json_roundtrips():
    bundle = _bundle_with_trajs()
    pts = [p.raw_point(bundle)] + p.filtered_points(bundle, [1])
    blob = p.points_to_json(pts)
    loaded = json.loads(blob)
    assert isinstance(loaded, list)
    assert loaded[0]["label"] == "raw"
    assert "f_causal" in loaded[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_pareto.py -v`
Expected: FAIL (ModuleNotFoundError on `famail_temporal.baselines.pareto`)

- [ ] **Step 3: Implement `pareto.py`**

Create `famail_temporal/baselines/pareto.py`:

```python
"""Assemble the data-level fairness x retention Pareto (Phase 1)."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from famail_temporal.data.loader import DataBundle
from famail_temporal.baselines.datasets import (
    rank_unfair_trajectory_indices, build_filtered_pickup_3d,
)
from famail_temporal.baselines.metrics import data_level_fairness


@dataclass(frozen=True)
class ParetoPoint:
    label: str
    retention: float
    f_spatial: float
    f_causal: float
    gini_dsr: float
    gini_asr: float
    n_removed: int


def _point(
    label: str, bundle: DataBundle, pickup_3d: Optional[np.ndarray],
    retention: float, n_removed: int,
) -> ParetoPoint:
    m = data_level_fairness(bundle, pickup_3d=pickup_3d)
    return ParetoPoint(
        label=label, retention=retention,
        f_spatial=m["f_spatial"], f_causal=m["f_causal"],
        gini_dsr=m["gini_dsr"], gini_asr=m["gini_asr"],
        n_removed=n_removed,
    )


def raw_point(bundle: DataBundle) -> ParetoPoint:
    """No intervention: full retention, bundle's own demand grid."""
    return _point("raw", bundle, None, 1.0, 0)


def filtered_points(
    bundle: DataBundle, k_levels: List[int],
) -> List[ParetoPoint]:
    """Generate-then-filter sweep: remove the top-K most-unfair trajectories.

    Ranking is computed once on the raw grid (static generate-then-filter).
    Each k is capped at the number of strictly-unfair candidates.
    """
    n = len(bundle.trajectories)
    if n == 0:
        raise ValueError("bundle has no trajectories to filter")
    ranked = rank_unfair_trajectory_indices(bundle)
    pts: List[ParetoPoint] = []
    for k in k_levels:
        k_eff = min(k, len(ranked))
        removed = [bundle.trajectories[i] for i in ranked[:k_eff]]
        pickup_3d = build_filtered_pickup_3d(bundle, removed)
        retention = (n - k_eff) / n
        pts.append(_point(f"filter@{k}", bundle, pickup_3d, retention, k_eff))
    return pts


def edited_point(
    f_spatial: float, f_causal: float, gini_dsr: float, gini_asr: float,
) -> ParetoPoint:
    """The FAMAIL editing point (full retention) from run_experiment's
    post-edit metrics. Caller passes ExperimentResult.*_after fields."""
    return ParetoPoint(
        label="edit", retention=1.0,
        f_spatial=f_spatial, f_causal=f_causal,
        gini_dsr=gini_dsr, gini_asr=gini_asr, n_removed=0,
    )


def points_to_json(points: List[ParetoPoint]) -> str:
    return json.dumps([asdict(pt) for pt in points], indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_pareto.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/pareto.py famail_temporal/baselines/tests/test_pareto.py
git commit -m "feat(baselines): assemble raw/filtered/edited Pareto points"
```

---

## Task 6: `figure.py` — the Pareto plot

**Files:**
- Create: `famail_temporal/baselines/figure.py`
- Test: `famail_temporal/baselines/tests/test_figure.py`

- [ ] **Step 1: Write the failing test**

Create `famail_temporal/baselines/tests/test_figure.py`:

```python
"""Smoke test for the Pareto figure."""
from famail_temporal.baselines.pareto import ParetoPoint
from famail_temporal.baselines import figure as fig


def test_plot_pareto_writes_png(tmp_path):
    points = [
        ParetoPoint("raw", 1.0, 0.08, 0.805, 0.92, 0.91, 0),
        ParetoPoint("filter@100", 0.99, 0.08, 0.808, 0.92, 0.91, 100),
        ParetoPoint("filter@500", 0.95, 0.08, 0.815, 0.92, 0.91, 500),
        ParetoPoint("edit", 1.0, 0.08, 0.814, 0.92, 0.91, 0),
    ]
    out = tmp_path / "pareto.png"
    fig.plot_pareto(points, out, metric="f_causal")
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_figure.py -v`
Expected: FAIL (ModuleNotFoundError on `famail_temporal.baselines.figure`)

- [ ] **Step 3: Implement `figure.py`**

Create `famail_temporal/baselines/figure.py`:

```python
"""Render the data-level fairness x retention Pareto figure."""
from __future__ import annotations
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt

from famail_temporal.baselines.pareto import ParetoPoint


def plot_pareto(
    points: List[ParetoPoint], path: Path, metric: str = "f_causal",
) -> None:
    """Scatter retention (x) vs fairness `metric` (y), filter points joined
    into a curve, raw and edit drawn as standout markers."""
    fig, ax = plt.subplots(figsize=(7, 5))

    filt = sorted(
        [p for p in points if p.label.startswith("filter@")],
        key=lambda p: p.retention,
    )
    if filt:
        ax.plot(
            [p.retention for p in filt], [getattr(p, metric) for p in filt],
            "-o", color="#dc2626", label="B2 filter", zorder=2,
        )
    for p in points:
        if p.label == "raw":
            ax.scatter([p.retention], [getattr(p, metric)], s=90,
                       color="#1e3a5f", label="B0 raw", zorder=3)
        elif p.label == "edit":
            ax.scatter([p.retention], [getattr(p, metric)], s=140,
                       color="#047857", marker="*", label="FAMAIL edit", zorder=4)

    ax.set_xlabel("Data retention (fraction of corpus kept)")
    ax.set_ylabel(f"{metric}  (1 = fairest)")
    ax.set_title("Data-level fairness x retention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_figure.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/figure.py famail_temporal/baselines/tests/test_figure.py
git commit -m "feat(baselines): data-level Pareto figure"
```

---

## Task 7: `run_data_pareto.py` — CLI wiring (raw + filtered sweep + optional edited)

This is the integration entry point. It loads the real bundle, computes the raw point and the filtered sweep with the new pure functions, optionally fetches the edited point from the existing `run_experiment`, then writes `pareto_points.json` + `pareto.png`. No new science — just wiring tested code.

**Files:**
- Create: `famail_temporal/baselines/run_data_pareto.py`
- Test: `famail_temporal/baselines/tests/test_run_data_pareto.py`

- [ ] **Step 1: Write the failing test** (the edited-point adapter is the only non-trivial logic; test it without touching the GPU)

Create `famail_temporal/baselines/tests/test_run_data_pareto.py`:

```python
"""Unit test for the run_data_pareto edited-point adapter."""
from types import SimpleNamespace

from famail_temporal.baselines import run_data_pareto as rdp


def test_edited_point_from_result_reads_after_fields():
    fake = SimpleNamespace(
        f_spatial_after=0.083, f_causal_after=0.814,
        gini_dsr_after=0.91, gini_asr_after=0.90,
    )
    pt = rdp.edited_point_from_result(fake)
    assert pt.label == "edit"
    assert pt.retention == 1.0
    assert pt.f_causal == 0.814
    assert pt.f_spatial == 0.083
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_data_pareto.py -v`
Expected: FAIL (ModuleNotFoundError on `famail_temporal.baselines.run_data_pareto`)

- [ ] **Step 3: Implement `run_data_pareto.py`**

Create `famail_temporal/baselines/run_data_pareto.py`:

```python
"""CLI: compute the data-level fairness x retention Pareto.

Loads the full corpus bundle, computes the raw point and a filtered@K sweep
(no GAN), optionally runs the existing one-shot editing pipeline for the
FAMAIL point, and writes pareto_points.json + pareto.png.

Example:
    python -m famail_temporal.baselines.run_data_pareto \
        --k-levels 100 500 1000 5000 --with-edit --edit-k 1000
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.baselines.pareto import (
    ParetoPoint, raw_point, filtered_points, edited_point, points_to_json,
)
from famail_temporal.baselines.figure import plot_pareto


def edited_point_from_result(result) -> ParetoPoint:
    """Adapt an ExperimentResult's post-edit metrics into the edit ParetoPoint."""
    return edited_point(
        f_spatial=result.f_spatial_after, f_causal=result.f_causal_after,
        gini_dsr=result.gini_dsr_after, gini_asr=result.gini_asr_after,
    )


def _run_edit(edit_k: int) -> ParetoPoint:
    """Run the existing editing pipeline once for the FAMAIL point.

    Uses the validated strongest config: causal-emphasis alpha=(0.2,0.7,0.1)
    with unit-distinct selection (--max-per-unit 1), which achieved
    ΔF_causal=+0.0087 at k=1000 (run 2026-05-27T22-29-57_1000k_causal_emphasis_dedup)
    — a balanced multi-objective that matches the pure-causal gain without
    gaming a single metric.
    """
    from famail_temporal.evaluation.runner import run_experiment
    result = run_experiment(
        config_overrides={"ALPHA_SPATIAL": 0.2, "ALPHA_CAUSAL": 0.7, "ALPHA_FIDELITY": 0.1},
        name="data-pareto-edit",
        k=edit_k,
        max_per_unit=1,
        device="auto",
    )
    return edited_point_from_result(result)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="famail_temporal.baselines.run_data_pareto")
    ap.add_argument("--k-levels", type=int, nargs="+",
                    default=[100, 500, 1000, 5000])
    ap.add_argument("--with-edit", action="store_true",
                    help="Also run the editing pipeline for the FAMAIL point.")
    ap.add_argument("--edit-k", type=int, default=1000)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(config.PACKAGE_ROOT) / "results" / "data_pareto")
    args = ap.parse_args(argv)

    bundle = DataBundle.load()
    points: List[ParetoPoint] = [raw_point(bundle)]
    points.extend(filtered_points(bundle, args.k_levels))
    if args.with_edit:
        points.append(_run_edit(args.edit_k))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pareto_points.json").write_text(points_to_json(points))
    plot_pareto(points, args.out_dir / "pareto.png", metric="f_causal")
    print(f"wrote {args.out_dir / 'pareto_points.json'}")
    print(f"wrote {args.out_dir / 'pareto.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_run_data_pareto.py -v`
Expected: PASS

- [ ] **Step 5: Run the full baselines test suite**

Run: `python -m pytest famail_temporal/baselines/ -v`
Expected: PASS (all tests from Tasks 1–7)

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/baselines/run_data_pareto.py famail_temporal/baselines/tests/test_run_data_pareto.py
git commit -m "feat(baselines): data-level Pareto CLI (raw + filtered sweep + edited)"
```

---

## Task 8: End-to-end smoke on real data (manual / gated)

This validates the wiring against the real cache. It is **not** a unit test (it needs the preprocessed cache and is slow); run it manually and eyeball the output.

- [ ] **Step 1: Run the no-edit Pareto on the full corpus**

Run: `python -m famail_temporal.baselines.run_data_pareto --k-levels 100 500 1000 5000`
Expected: writes `famail_temporal/results/data_pareto/pareto_points.json` and `pareto.png`. In the JSON, `raw.f_causal ≈ 0.805` (confirms the reuse seam). **Empirically observed (2026-05-28):** `filter@K.f_causal` slightly *decreases* as K grows (0.8052→0.8016 at 3773 removed) and `retention` decreases — i.e., data-level filtering does NOT improve F_causal (see spec §8 "Data-level filtering finding"). This is a real result, not a wiring bug; the raw baseline matching 0.805 confirms correctness.

- [ ] **Step 2: (Optional, heavy) Run with the edited point**

Run: `python -m famail_temporal.baselines.run_data_pareto --k-levels 100 500 1000 5000 --with-edit --edit-k 1000`
Expected: adds an `edit` point at `retention=1.0` with `f_causal ≈ 0.814` (ΔF_causal=+0.0087, above `raw`=0.805). The figure shows the green FAMAIL star at full retention above both raw and the filter curve — editing strictly dominates filtering on both axes at the data level.

- [ ] **Step 3: Sanity-check direction, then stop**

Confirm the edit point sits above raw (it does: 0.814 > 0.805). Do NOT "fix" the filter curve's direction by changing the editing or filtering algorithm — the filtering finding is a real empirical result (see spec §8) and any criterion change requires sign-off.

---

## Self-Review

**1. Spec coverage (Phase 1 portion):**
- Data-variant builders (raw, filtered@K) — Tasks 3, 5. ✓
- Data-level fairness reduction reusing canonical convention — Task 4. ✓
- Fairness × retention Pareto + figure + the "discard X%" geometry — Tasks 5, 6, 7. ✓
- Edited (FAMAIL) point at full retention via existing pipeline — Tasks 5, 7. ✓
- B0/B1/FAMAIL GAN training, rollouts, utility metrics, B1 loss, pure-GAN ablation, signal-max scale-up — **out of Phase 1 by design** (Phases 2–5).
- DI ratio metric — **deferred to Phase 1b (gated; needs district supply/demand definition sign-off)**.

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step has complete code; every test step has assertions and an exact run command. ✓

**3. Type consistency:** `ParetoPoint` fields (`label, retention, f_spatial, f_causal, gini_dsr, gini_asr, n_removed`) are identical across `pareto.py`, the figure, and all tests. `data_level_fairness` returns the same four keys produced by `_scalar_metrics_from_grid`. `build_filtered_pickup_3d` takes `List[Trajectory]` everywhere it is called. `edited_point(...)` keyword args match `edited_point_from_result`'s call. ✓

**4. Ambiguity:** "filter@K" removes the K *most-unfair* (most-negative attribution) trajectories, ranked once on the raw grid; K is capped at the strictly-negative candidate count. Filtering subtracts demand mass only (supply unchanged). All stated explicitly in `datasets.py`/`pareto.py` docstrings.

---
