# Demographic Oversampling Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Mission-3 4th baseline arm — demographic oversampling: duplicate real seeking
trajectories originating in disadvantaged regions under phantom driver IDs, rebuild demand+supply grids
additively, rescore fairness + external metrics, and run the 9-arm CPU experiment matrix.

**Architecture:** One new pure engine module (`demographic_oversampling.py`) + one new CLI runner
(`run_demographic_oversampling.py`) in `famail_temporal/baselines/`, reusing the existing pure scoring
functions (`external_fairness.py`, `run_external_fairness.assemble_results`, `metrics.data_level_fairness`
on a supply-substituted bundle). Zero changes to any existing module. Arm dirs write `duplicates.pkl`
(deliberately NOT `histories.pkl`) + a `metrics.json` in the exact schema `assemble_baseline_table`
already ingests.

**Tech Stack:** Python 3 (repo venv), numpy, dataclasses, pytest; matplotlib (Agg) for the dose figure.

**Spec:** `docs/superpowers/specs/2026-07-09-demographic-oversampling-baseline-design.md` (approved).

## Global Constraints

- Python interpreter: `/home/robert/FAMAIL/.venv/bin/python` (aliased `$PY` below). Work dir: the
  worktree root `/home/robert/FAMAIL/.claude/worktrees/demographic-oversampling`.
- **Frozen-algorithm gate:** `git diff main -- famail_temporal/algorithm/ famail_temporal/evaluation/runner.py`
  must stay EMPTY through the whole branch.
- **New files only** — do not modify any existing Python module. The only existing file edited on this
  branch is `famail_temporal/baselines/STATUS.md` (docs, Task 7).
- City: Shenzhen (do NOT set `FAMAIL_CITY`; default config gives `GRID_DIMS = (48, 90)`).
- **CPU only.** Never touch the GPU — the α-sweep occupies it. Nothing in this plan imports the
  discriminator.
- Headline dose = 10,000 (budget parity with FAMAIL k=10000); `EQUITY_AXES` order is the canonical
  stratum order everywhere.
- Commit after each task. Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

## Pinned upstream interfaces (read-only facts, verified 2026-07-09)

- `Trajectory` (`famail_temporal/utils/trajectory.py:118`): `trajectory_id`, `driver_id`,
  `states: List[TrajectoryState]`, `.clone()` (deep copy), `.pickup_state` = `states[-1]`,
  `.pickup_cell` = int `(x, y)`. `TrajectoryState`: `x_grid: float`, `y_grid: float`,
  `time_bucket: int` (1..288, 5-min; 0 tolerated), `day_index: int`. Coordinates are **0-based**
  bundle-space.
- `pickup_unit_of(traj) -> (cx, cy, t_block)`; `pickup_mass(bundle, t_block) -> float`
  `= 1 / (bundle.n_hours_per_block[t_block] * bundle.n_days)` (`baselines/datasets.py:28,37`).
- `aggregate_active_taxis` normalizer (`data/aggregation.py:120-158`): sums distinct
  (cell, hour, day) counts into blocks, then divides each block by
  `block_n_hours(t) * n_days`, then floors at `config.SUPPLY_FLOOR` (0.1). Raw view counts a
  driver at (cx, cy, hour, day) if ≥1 **empty** ping in the 5×5 neighborhood
  (`data/source_generation/views/active_taxis.py`; `K = data.source_generation.config.NEIGHBORHOOD_K`).
- Terminal seeking state is EXCLUDED from supply (pickup-transition record; demand-only) — the
  convention `analysis/supply_recount.py:175` established.
- `service_ratio_Y(pickup_3d, bundle, supply_3d=None)` (`baselines/external_fairness_io.py:51`):
  Y = supply/demand over `bundle.mask_3d` units, `DEMAND_FLOOR` = 0.5.
- `data_level_fairness(bundle, pickup_3d=None) -> {f_spatial, f_causal, gini_dsr, gini_asr}`
  (`baselines/metrics.py:15`); supply substitution via `dataclasses.replace(bundle,
  active_taxis_3d=S')` — the exact pattern `analysis/supply_recount.py:381-392` validated.
- `run_external_fairness.assemble_results(Y_before, Y_after, demo, seed=0, B=1000) -> dict` is PURE
  and importable; `write_json(result, out_dir, meta)`, `render_markdown(result, meta)` too
  (`baselines/run_external_fairness.py:26,97,106`).
- `assemble_baseline_table._flatten_arm_metrics` reads `arm.mode` (row label), `arm.n_edited` (n),
  `arm.adjacency_violation_rate`, `arm.mean_final_p` (may be absent → em-dash),
  `fairness.{f_causal_before,f_causal_after,f_spatial_before,f_spatial_after}`, optional `fidelity`
  block. Writing this schema means **zero assembler changes**.
- `adjacency_violation_rate(trajs: List[Trajectory]) -> float`
  (`baselines/stifgsm_baseline.py:210`) — reuse for honest reporting (rigid shift preserves the
  source's rate except at boundary clips).
- `external_fairness_io.EQUITY_AXES = ["AvgHousingPricePerSqM", "CompPerCapita", "MigrantRatio"]`,
  `DISADVANTAGED_HIGH = {housing: False, comp: False, migrant: True}`;
  `ef.region_extremes(values, disadvantaged_high, frac=1/3)` (labels: 1=D, 0=A, −1=excluded);
  `efio._enriched_selected_grid() -> (GX, GY, 3)` float64 cell values (in-package artifact read).
- Gitignored data the worktree lacks (Task 6 preflight symlinks them from
  `/home/robert/FAMAIL/famail_temporal/{cache,source_data}/`): the preprocess cache (~1.8G) and
  source_data pickles (~810M). `grid_to_district_mapping.pkl` already copied into
  `famail_temporal/source_data/`.

---

### Task 1: Engine — region selection & quota sampling

**Files:**
- Create: `famail_temporal/baselines/demographic_oversampling.py`
- Test: `famail_temporal/baselines/tests/test_demographic_oversampling.py`

**Interfaces:**
- Consumes: `ef.region_extremes`, `efio.EQUITY_AXES` / `DISADVANTAGED_HIGH` (pinned above).
- Produces (used by Tasks 2-4):
  `REGION_FRAC: float`; `PLACEBO = "placebo"`;
  `disadvantaged_cell_masks(selected_grid: np.ndarray) -> Dict[str, np.ndarray]` ((GX,GY) bool per axis);
  `origin_cell(traj) -> Tuple[int, int]`;
  `eligible_pools(trajectories, masks) -> Dict[str, np.ndarray]` (sorted int64 indices per axis);
  `DuplicateSpec` frozen dataclass with fields
  `source_index: int, stratum: str, eligible_axes: Tuple[str, ...], offset: Tuple[int, int],
  phantom_id: str, with_replacement: bool`;
  `sample_duplicates(pools, n_corpus, dose, seed, variant="targeted") -> List[DuplicateSpec]`.

- [ ] **Step 1: Write the failing tests**

```python
# famail_temporal/baselines/tests/test_demographic_oversampling.py
"""Engine tests for the Demographic Oversampling baseline (Mission-3 4th arm)."""
from types import SimpleNamespace

import numpy as np
import pytest

from famail_temporal.baselines import demographic_oversampling as dov
from famail_temporal.baselines.external_fairness_io import EQUITY_AXES
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


def _traj(cells, traj_id, driver="d0", time_bucket=13, day=0):
    """Trajectory through integer `cells` [(x, y), ...]; last cell is the pickup."""
    states = [TrajectoryState(x_grid=float(x), y_grid=float(y),
                              time_bucket=time_bucket, day_index=day)
              for x, y in cells]
    return Trajectory(trajectory_id=traj_id, driver_id=driver, states=states)


def _selected_grid():
    """(6, 4, 3) cell values: 6 distinct 'district' values along x, same per axis.

    region_extremes(frac=1/3) over 6 distinct values -> k=2 extreme regions per
    pole. For housing/comp (disadvantaged LOW) D = rows {0, 1}; for migrant
    (disadvantaged HIGH) D = rows {4, 5}.
    """
    vals = np.arange(6, dtype=np.float64)          # 0..5, one value per x-row
    grid = np.zeros((6, 4, 3))
    for j in range(3):
        grid[:, :, j] = vals[:, None]
    return grid


def test_disadvantaged_cell_masks_follow_evaluation_convention():
    masks = dov.disadvantaged_cell_masks(_selected_grid())
    assert set(masks) == set(EQUITY_AXES)
    housing = masks["AvgHousingPricePerSqM"]      # disadvantaged LOW -> rows 0, 1
    migrant = masks["MigrantRatio"]               # disadvantaged HIGH -> rows 4, 5
    assert housing.shape == (6, 4) and housing.dtype == bool
    assert housing[0].all() and housing[1].all() and not housing[2:].any()
    assert migrant[4].all() and migrant[5].all() and not migrant[:4].any()


def test_disadvantaged_cell_masks_nan_cells_excluded():
    grid = _selected_grid()
    grid[0, 0, :] = np.nan
    masks = dov.disadvantaged_cell_masks(grid)
    assert not masks["AvgHousingPricePerSqM"][0, 0]


def test_eligible_pools_by_origin_cell():
    masks = dov.disadvantaged_cell_masks(_selected_grid())
    trajs = [
        _traj([(0, 0), (2, 2)], "a"),   # origin row 0 -> housing+comp D
        _traj([(5, 1), (3, 3)], "b"),   # origin row 5 -> migrant D
        _traj([(3, 0), (0, 0)], "c"),   # origin row 3 -> no pool (pickup row ignored)
    ]
    pools = dov.eligible_pools(trajs, masks)
    assert pools["AvgHousingPricePerSqM"].tolist() == [0]
    assert pools["CompPerCapita"].tolist() == [0]
    assert pools["MigrantRatio"].tolist() == [1]


def test_sample_duplicates_quotas_and_dedupe():
    pools = {
        "AvgHousingPricePerSqM": np.array([0, 1, 2, 3]),
        "CompPerCapita": np.array([0, 1, 2, 3]),      # fully overlaps housing
        "MigrantRatio": np.array([10, 11, 12, 13]),
    }
    specs = dov.sample_duplicates(pools, n_corpus=20, dose=7, seed=0)
    assert len(specs) == 7
    # quotas in EQUITY_AXES order: 7 = 3 + 2 + 2
    per = {a: sum(1 for s in specs if s.stratum == a) for a in EQUITY_AXES}
    assert per == {"AvgHousingPricePerSqM": 3, "CompPerCapita": 2, "MigrantRatio": 2}
    # cross-stratum dedupe: housing draws 3 of the shared {0,1,2,3} pool,
    # leaving 1 for comp's quota of 2 -> exactly one flagged fallback draw;
    # distinct sources = 4 (shared pool) + 2 (migrant) = 6 of 7 draws.
    srcs = [s.source_index for s in specs]
    assert sum(1 for s in specs if s.with_replacement) == 1
    assert len(set(srcs)) == 6
    # offsets are rigid, radius-1, never zero
    assert all(max(abs(s.offset[0]), abs(s.offset[1])) == 1 for s in specs)
    # eligible_axes recorded for overlap sources
    housing_specs = [s for s in specs if s.stratum == "AvgHousingPricePerSqM"]
    assert all(set(s.eligible_axes) == {"AvgHousingPricePerSqM", "CompPerCapita"}
               for s in housing_specs)


def test_sample_duplicates_deterministic_and_seed_sensitive():
    pools = {a: np.arange(50) for a in EQUITY_AXES}
    a1 = dov.sample_duplicates(pools, n_corpus=100, dose=12, seed=3)
    a2 = dov.sample_duplicates(pools, n_corpus=100, dose=12, seed=3)
    b = dov.sample_duplicates(pools, n_corpus=100, dose=12, seed=4)
    assert a1 == a2
    assert [s.source_index for s in a1] != [s.source_index for s in b]


def test_sample_duplicates_placebo_uniform_over_corpus():
    pools = {a: np.array([0]) for a in EQUITY_AXES}   # pools must be IGNORED
    specs = dov.sample_duplicates(pools, n_corpus=30, dose=10, seed=0,
                                  variant=dov.PLACEBO)
    assert len(specs) == 10
    assert all(s.stratum == dov.PLACEBO for s in specs)
    assert len({s.source_index for s in specs}) == 10          # without replacement
    assert max(s.source_index for s in specs) < 30


def test_sample_duplicates_empty_pool_hard_error():
    pools = {a: (np.array([], dtype=np.int64) if a == "MigrantRatio"
                 else np.arange(9)) for a in EQUITY_AXES}
    with pytest.raises(ValueError, match="empty pool"):
        dov.sample_duplicates(pools, n_corpus=9, dose=9, seed=0)


def test_sample_duplicates_dose_zero_is_empty():
    pools = {a: np.arange(5) for a in EQUITY_AXES}
    assert dov.sample_duplicates(pools, n_corpus=5, dose=0, seed=0) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest famail_temporal/baselines/tests/test_demographic_oversampling.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'disadvantaged_cell_masks'` (module
doesn't exist yet → ImportError first).

- [ ] **Step 3: Write the implementation**

```python
# famail_temporal/baselines/demographic_oversampling.py
"""Demographic Oversampling baseline engine (Mission-3 4th arm).

Additive RESAMPLING baseline (not perturbation): duplicate real seeking
trajectories originating in demographically disadvantaged regions, under
fresh phantom driver IDs, and rebuild the fairness inputs ADDITIVELY on both
channels — demand (phantom pickups) and tier-2 distinct-count supply
(phantom presence). Spec:
docs/superpowers/specs/2026-07-09-demographic-oversampling-baseline-design.md

Standalone: imports nothing from famail_temporal/algorithm/ or
evaluation/runner.py (frozen-algorithm gate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from famail_temporal import config
from famail_temporal.baselines import external_fairness as ef
from famail_temporal.baselines.datasets import pickup_mass, pickup_unit_of
from famail_temporal.baselines.external_fairness_io import (
    DISADVANTAGED_HIGH, EQUITY_AXES,
)
from famail_temporal.data.aggregation import (
    hour_to_block_index, time_bucket_to_hour,
)
from famail_temporal.data.source_generation import config as sg_config
from famail_temporal.utils.trajectory import Trajectory

# Shared with the evaluation convention: region_extremes' default frac. The
# arm oversamples exactly the group the external metrics call disadvantaged.
REGION_FRAC = 1.0 / 3.0
PLACEBO = "placebo"
# Rigid whole-trajectory offsets: L-inf radius 1, excluding (0, 0).
_OFFSETS: Tuple[Tuple[int, int], ...] = tuple(
    (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)
)


def disadvantaged_cell_masks(selected_grid: np.ndarray) -> Dict[str, np.ndarray]:
    """{axis: (GX, GY) bool} — cells in the axis's bottom-third disadvantaged
    regions, via the SAME rule the external-metrics reporting uses
    (ef.region_extremes over distinct region values, DISADVANTAGED_HIGH poles).
    NaN cells are excluded (never disadvantaged)."""
    masks: Dict[str, np.ndarray] = {}
    for j, axis in enumerate(EQUITY_AXES):
        values = selected_grid[:, :, j].astype(np.float64).ravel()
        groups = ef.region_extremes(
            values, disadvantaged_high=DISADVANTAGED_HIGH[axis], frac=REGION_FRAC,
        )
        masks[axis] = (groups == 1).reshape(selected_grid.shape[:2])
    return masks


def origin_cell(traj: Trajectory) -> Tuple[int, int]:
    """Integer cell of the trajectory's FIRST seeking state (its origin)."""
    s = traj.states[0]
    return int(s.x_grid), int(s.y_grid)


def eligible_pools(
    trajectories: Sequence[Trajectory], masks: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """{axis: sorted int64 indices of trajectories whose origin cell is in the
    axis's disadvantaged mask}. Out-of-grid origins are never eligible."""
    origins = np.array([origin_cell(t) for t in trajectories], dtype=np.int64)
    pools: Dict[str, np.ndarray] = {}
    for axis, mask in masks.items():
        if len(trajectories) == 0:
            pools[axis] = np.array([], dtype=np.int64)
            continue
        inb = ((origins[:, 0] >= 0) & (origins[:, 0] < mask.shape[0])
               & (origins[:, 1] >= 0) & (origins[:, 1] < mask.shape[1]))
        member = np.zeros(len(trajectories), dtype=bool)
        member[inb] = mask[origins[inb, 0], origins[inb, 1]]
        pools[axis] = np.flatnonzero(member).astype(np.int64)
    return pools


@dataclass(frozen=True)
class DuplicateSpec:
    """Provenance of one phantom duplicate."""
    source_index: int                 # index into the sampled trajectory list
    stratum: str                      # axis that drew it, or PLACEBO
    eligible_axes: Tuple[str, ...]    # all axes whose pool contains the source
    offset: Tuple[int, int]           # rigid (dx, dy), L-inf radius 1, != (0,0)
    phantom_id: str                   # fresh driver id (namespaced)
    with_replacement: bool            # True iff drawn by the fallback path


def sample_duplicates(
    pools: Dict[str, np.ndarray], n_corpus: int, dose: int, seed: int,
    variant: str = "targeted",
) -> List[DuplicateSpec]:
    """Draw `dose` DuplicateSpecs.

    targeted: per-axis quotas in EQUITY_AXES order (dose//3 each, remainder to
    the earliest axes); uniform WITHOUT replacement within (axis pool minus
    already-drawn); a stratum whose remaining pool is smaller than its quota
    degrades to WITH-replacement draws from its full pool (flagged). An empty
    pool with a positive quota is a hard error.
    placebo: uniform without replacement over range(n_corpus) (same fallback).
    """
    rng = np.random.default_rng(seed)
    membership = {axis: frozenset(p.tolist()) for axis, p in pools.items()}

    def _axes_of(i: int) -> Tuple[str, ...]:
        return tuple(a for a in EQUITY_AXES if i in membership.get(a, frozenset()))

    specs: List[DuplicateSpec] = []
    drawn: set = set()

    def _draw(stratum: str, pool: np.ndarray, quota: int) -> None:
        if quota <= 0:
            return
        if pool.size == 0:
            raise ValueError(f"empty pool for stratum {stratum!r}")
        avail = np.array(sorted(set(pool.tolist()) - drawn), dtype=np.int64)
        n_wo = min(quota, avail.size)
        picks: List[int] = (
            [int(v) for v in rng.choice(avail, size=n_wo, replace=False)]
            if n_wo else []
        )
        n_wr = quota - n_wo
        if n_wr > 0:
            picks += [int(v) for v in rng.choice(pool, size=n_wr, replace=True)]
        for k, i in enumerate(picks):
            specs.append(DuplicateSpec(
                source_index=i,
                stratum=stratum,
                eligible_axes=_axes_of(i),
                offset=_OFFSETS[int(rng.integers(len(_OFFSETS)))],
                phantom_id=f"phantom_{variant}_s{seed}_{len(specs):06d}",
                with_replacement=(k >= n_wo),
            ))
            drawn.add(i)

    if variant == "targeted":
        base, rem = divmod(dose, len(EQUITY_AXES))
        for j, axis in enumerate(EQUITY_AXES):
            _draw(axis, pools[axis], base + (1 if j < rem else 0))
    elif variant == PLACEBO:
        _draw(PLACEBO, np.arange(n_corpus, dtype=np.int64), dose)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return specs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest famail_temporal/baselines/tests/test_demographic_oversampling.py -q`
Expected: PASS (8 tests). Note: if `test_sample_duplicates_quotas_and_dedupe`'s `n_wr == 1`
assertion fails because the housing draw left ≥2 comp candidates, the test premise is wrong, not
the code — housing draws 3 of {0,1,2,3}, leaving exactly 1 for comp's quota of 2, so exactly 1
fallback draw is forced. Fix the test only if EQUITY_AXES order changed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/demographic_oversampling.py \
        famail_temporal/baselines/tests/test_demographic_oversampling.py
git commit -m "feat(mission3): oversampling engine — region selection + quota sampling

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Engine — phantom materialization + diagnostics

**Files:**
- Modify: `famail_temporal/baselines/demographic_oversampling.py` (append)
- Test: `famail_temporal/baselines/tests/test_demographic_oversampling.py` (append)

**Interfaces:**
- Consumes: Task 1's `DuplicateSpec`, `origin_cell`.
- Produces (used by Tasks 3-4):
  `make_phantom(traj: Trajectory, spec: DuplicateSpec, grid_dims: Tuple[int, int] | None = None)
  -> Tuple[Trajectory, int]` (phantom, n_clipped_states);
  `escape_fractions(specs, phantoms, masks) -> Dict[str, float | None]` with keys
  `origin_escape_frac`, `pickup_outside_frac` (None for placebo-only runs).

- [ ] **Step 1: Write the failing tests (append to the test file)**

```python
def test_make_phantom_rigid_shift_and_identity():
    src = _traj([(3, 3), (4, 3), (4, 4)], "t9", driver="real_driver")
    spec = dov.DuplicateSpec(source_index=0, stratum="MigrantRatio",
                             eligible_axes=("MigrantRatio",), offset=(1, -1),
                             phantom_id="phantom_targeted_s0_000000",
                             with_replacement=False)
    ph, n_clipped = dov.make_phantom(src, spec, grid_dims=(48, 90))
    assert n_clipped == 0
    assert ph.driver_id == "phantom_targeted_s0_000000"
    assert ph.driver_id != src.driver_id
    assert str(src.trajectory_id) in str(ph.trajectory_id)
    # rigid: every state shifted by exactly (1, -1); times/days unchanged
    for s_src, s_ph in zip(src.states, ph.states):
        assert (s_ph.x_grid, s_ph.y_grid) == (s_src.x_grid + 1, s_src.y_grid - 1)
        assert s_ph.time_bucket == s_src.time_bucket
        assert s_ph.day_index == s_src.day_index
    # source untouched (deep copy)
    assert (src.states[0].x_grid, src.states[0].y_grid) == (3.0, 3.0)


def test_make_phantom_clips_at_boundary_and_counts():
    src = _traj([(0, 0), (1, 0)], "t10")
    spec = dov.DuplicateSpec(source_index=0, stratum="CompPerCapita",
                             eligible_axes=("CompPerCapita",), offset=(-1, -1),
                             phantom_id="p", with_replacement=False)
    ph, n_clipped = dov.make_phantom(src, spec, grid_dims=(48, 90))
    assert n_clipped == 2                        # both states clipped in x and/or y
    assert (ph.states[0].x_grid, ph.states[0].y_grid) == (0.0, 0.0)
    assert (ph.states[1].x_grid, ph.states[1].y_grid) == (0.0, 0.0)


def test_adjacency_preserved_without_clipping():
    from famail_temporal.baselines.stifgsm_baseline import adjacency_violation_rate
    src = _traj([(5, 5), (6, 5), (6, 6), (7, 6)], "t11")
    spec = dov.DuplicateSpec(source_index=0, stratum="MigrantRatio",
                             eligible_axes=("MigrantRatio",), offset=(1, 1),
                             phantom_id="p", with_replacement=False)
    ph, n_clipped = dov.make_phantom(src, spec, grid_dims=(48, 90))
    assert n_clipped == 0
    assert adjacency_violation_rate([ph]) == adjacency_violation_rate([src])


def test_escape_fractions():
    masks = dov.disadvantaged_cell_masks(_selected_grid())      # migrant D rows 4-5
    src = _traj([(5, 1), (4, 1), (3, 1)], "t12")                # origin row 5; pickup row 3
    spec = dov.DuplicateSpec(source_index=0, stratum="MigrantRatio",
                             eligible_axes=("MigrantRatio",), offset=(-1, 0),
                             phantom_id="p", with_replacement=False)
    ph, _ = dov.make_phantom(src, spec, grid_dims=(6, 4))
    fr = dov.escape_fractions([spec], [ph], masks)
    # shifted origin = row 4 (still D) -> no escape; shifted pickup = row 2 (outside D)
    assert fr == {"origin_escape_frac": 0.0, "pickup_outside_frac": 1.0}


def test_escape_fractions_placebo_none():
    spec = dov.DuplicateSpec(source_index=0, stratum=dov.PLACEBO,
                             eligible_axes=(), offset=(1, 0),
                             phantom_id="p", with_replacement=False)
    ph, _ = dov.make_phantom(_traj([(2, 2), (3, 2)], "t13"), spec, grid_dims=(6, 4))
    fr = dov.escape_fractions([spec], [ph],
                              dov.disadvantaged_cell_masks(_selected_grid()))
    assert fr == {"origin_escape_frac": None, "pickup_outside_frac": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest famail_temporal/baselines/tests/test_demographic_oversampling.py -q`
Expected: the 5 new tests FAIL with `AttributeError: ... 'make_phantom'`; Task-1 tests still PASS.

- [ ] **Step 3: Write the implementation (append to the module)**

```python
def make_phantom(
    traj: Trajectory, spec: DuplicateSpec,
    grid_dims: Tuple[int, int] | None = None,
) -> Tuple[Trajectory, int]:
    """Rigid-shift deep copy of `traj` under the phantom driver ID.

    Every state is displaced by the SAME (dx, dy) (the "second taxi ran the
    same route one street over" story), clipped to grid bounds; time buckets
    and day indices unchanged. Returns (phantom, n_clipped_states) where a
    state counts as clipped if the clip changed its shifted coordinate.
    """
    gx, gy = grid_dims if grid_dims is not None else config.GRID_DIMS
    dx, dy = spec.offset
    ph = traj.clone()
    ph.trajectory_id = f"{spec.phantom_id}::of::{traj.trajectory_id}"
    ph.driver_id = spec.phantom_id
    n_clipped = 0
    for s in ph.states:
        nx = min(max(s.x_grid + dx, 0.0), float(gx - 1))
        ny = min(max(s.y_grid + dy, 0.0), float(gy - 1))
        if nx != s.x_grid + dx or ny != s.y_grid + dy:
            n_clipped += 1
        s.x_grid, s.y_grid = nx, ny
    return ph, n_clipped


def escape_fractions(
    specs: Sequence[DuplicateSpec], phantoms: Sequence[Trajectory],
    masks: Dict[str, np.ndarray],
) -> Dict[str, float | None]:
    """Post-shift diagnostics over TARGETED duplicates (placebo strata skipped):

    - origin_escape_frac: fraction whose shifted ORIGIN left the drawing
      stratum's disadvantaged region set (pre-shift it was inside by
      construction).
    - pickup_outside_frac: fraction whose shifted PICKUP lies outside that
      set (descriptive — where the fabricated demand lands; pickups may be
      outside even pre-shift).
    """
    n = o_esc = p_out = 0
    for spec, ph in zip(specs, phantoms):
        mask = masks.get(spec.stratum)
        if mask is None:
            continue
        n += 1
        ox, oy = origin_cell(ph)
        px, py = ph.pickup_cell
        o_esc += 0 if mask[ox, oy] else 1
        p_out += 0 if mask[px, py] else 1
    if n == 0:
        return {"origin_escape_frac": None, "pickup_outside_frac": None}
    return {"origin_escape_frac": o_esc / n, "pickup_outside_frac": p_out / n}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest famail_temporal/baselines/tests/test_demographic_oversampling.py -q`
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/demographic_oversampling.py \
        famail_temporal/baselines/tests/test_demographic_oversampling.py
git commit -m "feat(mission3): oversampling engine — phantom materialization + escape diagnostics

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Engine — additive demand & tier-2 supply grids

**Files:**
- Modify: `famail_temporal/baselines/demographic_oversampling.py` (append)
- Test: `famail_temporal/baselines/tests/test_demographic_oversampling.py` (append)

**Interfaces:**
- Consumes: `pickup_mass`, `pickup_unit_of`, `time_bucket_to_hour`, `hour_to_block_index`,
  `sg_config.NEIGHBORHOOD_K` (already imported in Task 1).
- Produces (used by Task 4):
  `additive_demand(bundle, phantoms) -> np.ndarray` (float64, same shape as `bundle.pickup_3d`);
  `additive_supply(bundle, phantoms) -> np.ndarray` (float64, same shape as
  `bundle.active_taxis_3d`). `bundle` needs only: `pickup_3d`, `active_taxis_3d`,
  `n_hours_per_block`, `n_days` (SimpleNamespace-testable).

- [ ] **Step 1: Write the failing tests (append)**

```python
def _stub_bundle(gx=48, gy=90, n_days=2):
    from famail_temporal import config as cfg
    from famail_temporal.data.aggregation import block_n_hours
    T = cfg.T
    return SimpleNamespace(
        pickup_3d=np.zeros((gx, gy, T), dtype=np.float32),
        active_taxis_3d=np.zeros((gx, gy, T), dtype=np.float32),
        n_hours_per_block=np.array([block_n_hours(t) for t in range(T)],
                                   dtype=np.int32),
        n_days=n_days,
    )


def test_additive_demand_mass_conservation_and_placement():
    from famail_temporal.baselines.datasets import pickup_mass, pickup_unit_of
    b = _stub_bundle()
    phantoms = [_traj([(3, 3), (4, 4)], "p1"), _traj([(7, 8), (9, 9)], "p2")]
    D = dov.additive_demand(b, phantoms)
    assert D.dtype == np.float64
    expected = sum(pickup_mass(b, pickup_unit_of(ph)[2]) for ph in phantoms)
    assert np.isclose(D.sum() - np.float64(b.pickup_3d).sum(), expected)
    cx, cy, t = pickup_unit_of(phantoms[0])
    assert D[cx, cy, t] == pytest.approx(pickup_mass(b, t))


def test_additive_supply_distinct_count_semantics():
    b = _stub_bundle()
    # two states of ONE phantom in the same cell/hour -> counted ONCE
    ph = _traj([(10, 10), (10, 10), (11, 10)], "p3", time_bucket=13)
    S = dov.additive_supply(b, [ph])
    from famail_temporal.data.aggregation import (
        hour_to_block_index, time_bucket_to_hour,
    )
    t = hour_to_block_index(time_bucket_to_hour(13))
    unit = 1.0 / (float(b.n_hours_per_block[t]) * b.n_days)
    # states[-1] (the pickup) is EXCLUDED from supply; the two remaining
    # states are both at (10, 10) -> the 5x5 neighborhood around each is
    # identical -> every covered cell gets exactly ONE unit, not two.
    assert S[10, 10, t] == pytest.approx(unit)
    assert S[8, 8, t] == pytest.approx(unit)          # 5x5 reach (K=2)
    assert S[13, 10, t] == 0.0                        # outside the neighborhood
    # two DISTINCT phantoms in the same cell/hour -> counted TWICE
    ph2 = _traj([(10, 10), (11, 10)], "p4", time_bucket=13)
    S2 = dov.additive_supply(b, [ph, ph2])
    assert S2[10, 10, t] == pytest.approx(2 * unit)


def test_additive_supply_matches_production_counter_on_fixture():
    """Pin additive_supply to the PRODUCTION tier-2 convention: the same pings
    pushed through build_active_taxis_counts + aggregate_active_taxis must
    reproduce additive_supply's delta on every cell the phantom covers."""
    import pandas as pd
    from famail_temporal.data.aggregation import aggregate_active_taxis
    from famail_temporal.data.source_generation.views.active_taxis import (
        build_active_taxis_counts,
    )
    from famail_temporal import config as cfg

    b = _stub_bundle(n_days=1)
    ph = _traj([(10, 10), (12, 11), (12, 12)], "pfix", time_bucket=13, day=0)
    S = dov.additive_supply(b, [ph])

    # The production path: raw pings are 1-indexed; the terminal state is the
    # pickup-transition -> give it passenger_indicator=1 so the production
    # counter drops it too (mirrors the supply-only convention).
    hour = 1  # time_bucket 13 -> hour 1 (1-indexed 5-min buckets, 12/hour)
    rows = [
        {"plate_id": "pfix", "x_grid": 10 + 1, "y_grid": 10 + 1,
         "hour": hour, "day_index": 0, "passenger_indicator": 0},
        {"plate_id": "pfix", "x_grid": 12 + 1, "y_grid": 11 + 1,
         "hour": hour, "day_index": 0, "passenger_indicator": 0},
        {"plate_id": "pfix", "x_grid": 12 + 1, "y_grid": 12 + 1,
         "hour": hour, "day_index": 0, "passenger_indicator": 1},   # pickup
    ]
    counts = build_active_taxis_counts(pd.DataFrame(rows))
    agg = aggregate_active_taxis(counts, n_days=1)

    covered = S > 0
    assert covered.any()
    # Where the phantom contributed, production and additive agree exactly.
    assert np.allclose(S[covered], agg[covered])
    # Where it didn't, production shows only its SUPPLY_FLOOR.
    assert np.all(agg[~covered] <= cfg.SUPPLY_FLOOR + 1e-12)


def test_additive_grids_dose_zero_identity():
    b = _stub_bundle()
    b.pickup_3d = np.random.default_rng(0).random(b.pickup_3d.shape).astype(np.float32)
    b.active_taxis_3d = np.random.default_rng(1).random(b.active_taxis_3d.shape).astype(np.float32)
    assert np.array_equal(dov.additive_demand(b, []), np.float64(b.pickup_3d))
    assert np.array_equal(dov.additive_supply(b, []), np.float64(b.active_taxis_3d))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest famail_temporal/baselines/tests/test_demographic_oversampling.py -q`
Expected: 4 new tests FAIL (`AttributeError: ... 'additive_demand'`). NOTE for the fixture test: if
`time_bucket 13 -> hour 1` is wrong, the assertion inside `additive_supply` vs the `hour=1` row will
disagree — verify with
`$PY -c "from famail_temporal.data.aggregation import time_bucket_to_hour; print(time_bucket_to_hour(13))"`
(expected `1`; buckets are 1-indexed 5-min, 12 per hour) and fix the fixture's `hour` to match.

- [ ] **Step 3: Write the implementation (append)**

```python
def additive_demand(bundle, phantoms: Sequence[Trajectory]) -> np.ndarray:
    """D' = bundle.pickup_3d + one pickup-event mass per phantom (float64).

    Existing per-event mass convention (datasets.pickup_mass), ADDED — never
    relocated, never floored (the subtraction floor in the substitution path
    has no additive counterpart)."""
    D = bundle.pickup_3d.astype(np.float64)
    for ph in phantoms:
        cx, cy, t = pickup_unit_of(ph)
        D[cx, cy, t] += pickup_mass(bundle, t)
    return D


def additive_supply(bundle, phantoms: Sequence[Trajectory]) -> np.ndarray:
    """S' = bundle.active_taxis_3d + tier-2 phantom presence (float64).

    Mirrors the production counter exactly (views/active_taxis.py +
    aggregate_active_taxis): a driver counts at (cell, hour, day) if >=1 of
    its empty pings falls in the (2K+1)x(2K+1) neighborhood; distinct per
    (driver, cell, hour, day); mean-hourly normalization divides each block
    by n_hours_per_block[t] * n_days. Phantom driver IDs are fresh, so their
    contributions are independent of the real fleet and purely additive — no
    raw-GPS resegmentation. The terminal (pickup-transition) state is
    EXCLUDED — supply-only, the analysis/supply_recount.py convention."""
    gx, gy = bundle.active_taxis_3d.shape[:2]
    k = sg_config.NEIGHBORHOOD_K
    S = bundle.active_taxis_3d.astype(np.float64)
    for ph in phantoms:
        covered = set()
        for s in ph.states[:-1]:
            x0, y0 = int(s.x_grid), int(s.y_grid)
            hour = time_bucket_to_hour(s.time_bucket)
            for dx in range(-k, k + 1):
                for dy in range(-k, k + 1):
                    x, y = x0 + dx, y0 + dy
                    if 0 <= x < gx and 0 <= y < gy:
                        covered.add((x, y, hour, s.day_index))
        for x, y, hour, _day in covered:
            t = hour_to_block_index(hour)
            S[x, y, t] += 1.0 / (float(bundle.n_hours_per_block[t]) * bundle.n_days)
    return S
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest famail_temporal/baselines/tests/test_demographic_oversampling.py -q`
Expected: PASS (17 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/demographic_oversampling.py \
        famail_temporal/baselines/tests/test_demographic_oversampling.py
git commit -m "feat(mission3): oversampling engine — additive demand + tier-2 phantom supply

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Runner CLI — arm dirs, fairness rescore, external metrics

**Files:**
- Create: `famail_temporal/baselines/run_demographic_oversampling.py`
- Test: `famail_temporal/baselines/tests/test_run_demographic_oversampling.py`

**Interfaces:**
- Consumes: everything Tasks 1-3 produce; `adjacency_violation_rate` from `stifgsm_baseline`;
  `data_level_fairness`; `service_ratio_Y` / `per_unit_demographics`;
  `run_external_fairness.{assemble_results, write_json, render_markdown}`;
  `assemble_baseline_table._flatten_arm_metrics` (test only).
- Produces: CLI `python -m famail_temporal.baselines.run_demographic_oversampling
  --variant targeted|placebo --dose N --seed S [--bootstrap 1000] [--out-root DIR]`;
  arm dir `<out-root>/<ts>_baseline_demo_oversample_<variant>_d<dose>_s<seed>_shenzhen/` containing
  `duplicates.pkl` (`{"specs": [...], "phantoms": [...]}`), `metrics.json`
  (`arm` / `fairness` / meta), `external_fairness/external_fairness.json` + `report.md`.
  Module seams for tests: `_load_bundle()`, `_selected_grid()`,
  `_rescore_fairness(bundle, D_after, S_after)`, `_external(bundle, D_after, S_after, arm_dir,
  meta, seed, B)`; pure `run(args) -> Path` orchestrator.

- [ ] **Step 1: Write the failing tests**

```python
# famail_temporal/baselines/tests/test_run_demographic_oversampling.py
"""CLI-level tests on a synthetic bundle via monkeypatched seams (pattern:
test_run_stifgsm_baseline.py)."""
import json
import pickle
from types import SimpleNamespace

import numpy as np

from famail_temporal.baselines import run_demographic_oversampling as rdo
from famail_temporal.baselines.assemble_baseline_table import _flatten_arm_metrics
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


def _traj(cells, traj_id, driver="d0"):
    states = [TrajectoryState(x_grid=float(x), y_grid=float(y),
                              time_bucket=13, day_index=0) for x, y in cells]
    return Trajectory(trajectory_id=traj_id, driver_id=driver, states=states)


def _selected_grid():
    vals = np.arange(6, dtype=np.float64)
    grid = np.zeros((6, 4, 3))
    for j in range(3):
        grid[:, :, j] = vals[:, None]
    return grid


def _stub_bundle():
    from famail_temporal import config as cfg
    from famail_temporal.data.aggregation import block_n_hours
    T = cfg.T
    # 12 trajectories: 4 originate in housing/comp-D rows (0-1), 4 in
    # migrant-D rows (4-5), 4 in neutral rows.
    trajs = (
        [_traj([(0, j % 4), (2, 2)], f"h{j}", driver=f"dh{j}") for j in range(4)]
        + [_traj([(5, j % 4), (3, 3)], f"m{j}", driver=f"dm{j}") for j in range(4)]
        + [_traj([(3, j % 4), (2, 1)], f"n{j}", driver=f"dn{j}") for j in range(4)]
    )
    return SimpleNamespace(
        trajectories=trajs,
        pickup_3d=np.ones((6, 4, T), dtype=np.float32),
        active_taxis_3d=np.ones((6, 4, T), dtype=np.float32),
        n_hours_per_block=np.array([block_n_hours(t) for t in range(T)],
                                   dtype=np.int32),
        n_days=1,
    )


def _patch_seams(monkeypatch, bundle):
    monkeypatch.setattr(rdo, "_load_bundle", lambda: bundle)
    monkeypatch.setattr(rdo, "_selected_grid", lambda: _selected_grid())
    monkeypatch.setattr(
        rdo, "_rescore_fairness",
        lambda bundle, D, S: {
            "f_spatial_before": 0.1, "f_spatial_after": 0.2,
            "f_causal_before": 0.8, "f_causal_after": 0.9,
            "deltas": {"f_spatial": 0.1, "f_causal": 0.1},
        },
    )
    monkeypatch.setattr(
        rdo, "_external",
        lambda bundle, D, S, arm_dir, meta, seed, B: {"stub": True},
    )


def test_run_targeted_writes_arm_contract(tmp_path, monkeypatch):
    bundle = _stub_bundle()
    _patch_seams(monkeypatch, bundle)
    arm_dir = rdo.run(rdo.parse_args(
        ["--variant", "targeted", "--dose", "6", "--seed", "0",
         "--out-root", str(tmp_path)]))
    assert arm_dir.is_dir()
    assert "demo_oversample_targeted_d6_s0" in arm_dir.name

    # duplicates.pkl round-trips; deliberately NO histories.pkl
    assert not (arm_dir / "histories.pkl").exists()
    with open(arm_dir / "duplicates.pkl", "rb") as f:
        dup = pickle.load(f)
    assert len(dup["specs"]) == len(dup["phantoms"]) == 6
    real_ids = {t.driver_id for t in bundle.trajectories}
    assert all(p.driver_id not in real_ids for p in dup["phantoms"])

    meta = json.loads((arm_dir / "metrics.json").read_text())
    arm = meta["arm"]
    assert arm["mode"] == "oversample-targeted-d6"
    assert arm["variant"] == "targeted" and arm["dose"] == 6 and arm["seed"] == 0
    assert arm["n_edited"] == 6
    assert arm["n_corpus"] == 12
    assert arm["corpus_inflation"] == 6 / 12
    assert set(arm["per_stratum_draws"]) == {
        "AvgHousingPricePerSqM", "CompPerCapita", "MigrantRatio"}
    assert sum(arm["per_stratum_draws"].values()) == 6
    assert isinstance(arm["adjacency_violation_rate"], float)
    assert "origin_escape_frac" in arm and "pickup_outside_frac" in arm
    assert "n_with_replacement" in arm and "n_clipped_states" in arm
    assert meta["fairness"]["f_causal_after"] == 0.9


def test_arm_metrics_ingest_into_baseline_table(tmp_path, monkeypatch):
    _patch_seams(monkeypatch, _stub_bundle())
    arm_dir = rdo.run(rdo.parse_args(
        ["--variant", "targeted", "--dose", "6", "--seed", "0",
         "--out-root", str(tmp_path)]))
    flat = _flatten_arm_metrics(json.loads((arm_dir / "metrics.json").read_text()))
    assert flat["label"] == "oversample-targeted-d6"
    assert flat["n"] == 6
    assert flat["f_causal_before"] == 0.8 and flat["f_causal_after"] == 0.9
    assert flat["fidelity_a"] is None          # not scored: by construction


def test_run_placebo_ignores_pools(tmp_path, monkeypatch):
    _patch_seams(monkeypatch, _stub_bundle())
    arm_dir = rdo.run(rdo.parse_args(
        ["--variant", "placebo", "--dose", "5", "--seed", "1",
         "--out-root", str(tmp_path)]))
    meta = json.loads((arm_dir / "metrics.json").read_text())
    assert meta["arm"]["mode"] == "oversample-placebo-d5"
    assert meta["arm"]["per_stratum_draws"] == {"placebo": 5}
    assert meta["arm"]["origin_escape_frac"] is None


def test_run_dose_zero_grids_identity(tmp_path, monkeypatch):
    """dose=0 end-to-end: the additive grids the runner hands to the scoring
    seams must equal the bundle's own grids exactly."""
    bundle = _stub_bundle()
    captured = {}

    def _capture_rescore(bundle_, D, S):
        captured["D"], captured["S"] = D, S
        return {"f_spatial_before": 0.0, "f_spatial_after": 0.0,
                "f_causal_before": 0.0, "f_causal_after": 0.0,
                "deltas": {"f_spatial": 0.0, "f_causal": 0.0}}

    monkeypatch.setattr(rdo, "_load_bundle", lambda: bundle)
    monkeypatch.setattr(rdo, "_selected_grid", lambda: _selected_grid())
    monkeypatch.setattr(rdo, "_rescore_fairness", _capture_rescore)
    monkeypatch.setattr(rdo, "_external",
                        lambda *a, **k: {"stub": True})
    rdo.run(rdo.parse_args(["--variant", "targeted", "--dose", "0",
                            "--seed", "0", "--out-root", str(tmp_path)]))
    assert np.array_equal(captured["D"], np.float64(bundle.pickup_3d))
    assert np.array_equal(captured["S"], np.float64(bundle.active_taxis_3d))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest famail_temporal/baselines/tests/test_run_demographic_oversampling.py -q`
Expected: FAIL — `ModuleNotFoundError: ... run_demographic_oversampling`.

- [ ] **Step 3: Write the implementation**

```python
# famail_temporal/baselines/run_demographic_oversampling.py
"""Runner CLI for the Demographic Oversampling baseline (Mission-3 4th arm).

Additive semantics end-to-end: sample -> phantoms -> (D', S') -> fairness
rescore (data_level_fairness on a supply-substituted bundle — the
supply_recount-validated pattern) + external metrics
(run_external_fairness.assemble_results on additive Y vectors) -> arm dir.

The arm dir deliberately writes duplicates.pkl, NOT histories.pkl: the
substitution-semantics CLIs (run_external_fairness, supply_recount) must
fail loudly on this dir rather than silently mis-score an additive corpus.

Module seams (_load_bundle, _selected_grid, _rescore_fairness, _external)
keep the CLI unit-testable on a synthetic bundle without the real dataset
(pattern: run_stifgsm_baseline.py).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from famail_temporal import config
from famail_temporal.baselines.demographic_oversampling import (
    PLACEBO, additive_demand, additive_supply, disadvantaged_cell_masks,
    eligible_pools, escape_fractions, make_phantom, sample_duplicates,
)
from famail_temporal.baselines.stifgsm_baseline import adjacency_violation_rate


# --------------------------------------------------------------- seams --------
def _load_bundle():
    from famail_temporal.data.loader import DataBundle
    return DataBundle.load()


def _selected_grid():
    # Same in-package artifact read the external-fairness harness performs
    # (cell_demographics.pkl -> enriched EQUITY_AXES cell values).
    from famail_temporal.baselines import external_fairness_io as efio
    return efio._enriched_selected_grid()


def _rescore_fairness(bundle, D_after, S_after):
    """{f_spatial/f_causal before/after + deltas} under the additive grids.

    Supply substitution via dataclasses.replace — the exact pattern
    analysis/supply_recount.py:381-392 validated against the editing pipeline.
    """
    from dataclasses import replace
    from famail_temporal.baselines.metrics import data_level_fairness

    before = data_level_fairness(bundle)
    bundle_after = replace(
        bundle, active_taxis_3d=S_after.astype(bundle.active_taxis_3d.dtype),
    )
    after = data_level_fairness(bundle_after, pickup_3d=D_after)
    return {
        "f_spatial_before": float(before["f_spatial"]),
        "f_spatial_after": float(after["f_spatial"]),
        "f_causal_before": float(before["f_causal"]),
        "f_causal_after": float(after["f_causal"]),
        "deltas": {
            "f_spatial": float(after["f_spatial"] - before["f_spatial"]),
            "f_causal": float(after["f_causal"] - before["f_causal"]),
        },
    }


def _external(bundle, D_after, S_after, arm_dir, meta, seed, B):
    """External fairness metrics (DP/DI/SDR/Theil + bootstrap CIs) on the
    additive Y vectors, written in the harness's standard schema."""
    from famail_temporal.baselines import external_fairness_io as efio
    from famail_temporal.baselines import run_external_fairness as ref

    Y_before = efio.service_ratio_Y(bundle.pickup_3d, bundle)
    Y_after = efio.service_ratio_Y(D_after, bundle, supply_3d=S_after)
    demo = efio.per_unit_demographics(bundle)
    result = ref.assemble_results(Y_before, Y_after, demo, seed=seed, B=B)
    out = Path(arm_dir) / "external_fairness"
    ref.write_json(result, out, meta)
    (out / "report.md").write_text(ref.render_markdown(result, meta))
    return result


# ----------------------------------------------------------------- CLI --------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="famail_temporal.baselines.run_demographic_oversampling")
    ap.add_argument("--variant", choices=["targeted", PLACEBO], required=True)
    ap.add_argument("--dose", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out-root", type=Path,
                    default=Path(config.PACKAGE_ROOT) / "results")
    return ap.parse_args(argv)


def run(args) -> Path:
    t0 = time.monotonic()
    bundle = _load_bundle()
    n_corpus = len(bundle.trajectories)

    masks = disadvantaged_cell_masks(_selected_grid())
    pools = eligible_pools(bundle.trajectories, masks)
    specs = sample_duplicates(pools, n_corpus, args.dose, args.seed,
                              variant=args.variant)
    n_wr = sum(1 for s in specs if s.with_replacement)
    if n_wr:
        print(f"[demo_oversample] WARNING: {n_wr} draws fell back to "
              f"with-replacement (stratum pool smaller than quota)",
              file=sys.stderr, flush=True)

    phantoms, n_clipped = [], 0
    for spec in specs:
        ph, nc = make_phantom(bundle.trajectories[spec.source_index], spec)
        phantoms.append(ph)
        n_clipped += nc

    D_after = additive_demand(bundle, phantoms)
    S_after = additive_supply(bundle, phantoms)

    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    arm_dir = (Path(args.out_root)
               / f"{ts}_baseline_demo_oversample_{args.variant}"
                 f"_d{args.dose}_s{args.seed}_{config.CITY}")
    arm_dir.mkdir(parents=True, exist_ok=True)

    with open(arm_dir / "duplicates.pkl", "wb") as f:
        pickle.dump({"specs": specs, "phantoms": phantoms}, f)

    arm = {
        "mode": f"oversample-{args.variant}-d{args.dose}",
        "variant": args.variant,
        "dose": args.dose,
        "seed": args.seed,
        "n_edited": len(phantoms),
        "n_corpus": n_corpus,
        "corpus_inflation": (args.dose / n_corpus) if n_corpus else 0.0,
        "adjacency_violation_rate": adjacency_violation_rate(phantoms),
        "per_stratum_draws": dict(Counter(s.stratum for s in specs)),
        "n_multi_axis_sources": sum(1 for s in specs if len(s.eligible_axes) > 1),
        "n_with_replacement": sum(1 for s in specs if s.with_replacement),
        "n_clipped_states": n_clipped,
        **escape_fractions(specs, phantoms, masks),
    }
    fairness = _rescore_fairness(bundle, D_after, S_after)

    meta = {"dataset": f"demo-oversample-{args.variant}-d{args.dose}-s{args.seed}",
            "city": config.CITY, "edit_dir": str(arm_dir),
            "seed": args.seed, "B": args.bootstrap}
    _external(bundle, D_after, S_after, arm_dir, meta, args.seed, args.bootstrap)

    (arm_dir / "metrics.json").write_text(json.dumps(
        {"arm": arm, "fairness": fairness,
         "runtime_s": time.monotonic() - t0}, indent=2, default=float))
    print(f"[demo_oversample] wrote {arm_dir}", flush=True)
    return arm_dir


def main(argv=None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest famail_temporal/baselines/tests/test_run_demographic_oversampling.py -q`
Expected: PASS (4 tests). Also re-run the engine tests together:
`$PY -m pytest famail_temporal/baselines/tests/ -q` — all baseline tests PASS.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/run_demographic_oversampling.py \
        famail_temporal/baselines/tests/test_run_demographic_oversampling.py
git commit -m "feat(mission3): demographic-oversampling runner CLI (arm dirs + additive scoring)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Summary assembler (dose-response table + figure)

**Files:**
- Modify: `famail_temporal/baselines/run_demographic_oversampling.py` (append)
- Test: `famail_temporal/baselines/tests/test_run_demographic_oversampling.py` (append)

**Interfaces:**
- Consumes: arm-dir `metrics.json` (Task 4 schema) + `external_fairness/external_fairness.json`
  (harness schema: `metrics.MigrantRatio.district_extremes.demographic_parity.delta`, etc.).
- Produces: `summarize_arms(arm_dirs: List[Path]) -> str` (markdown) and CLI mode
  `python -m ... run_demographic_oversampling --summarize DIR [DIR ...] --out DIR` writing
  `summary.md` + `dose_response.png`.

- [ ] **Step 1: Write the failing test (append)**

```python
def _fake_arm_dir(tmp_path, variant, dose, seed, d_fc, d_dp):
    d = tmp_path / f"x_baseline_demo_oversample_{variant}_d{dose}_s{seed}_shenzhen"
    (d / "external_fairness").mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "arm": {"mode": f"oversample-{variant}-d{dose}", "variant": variant,
                "dose": dose, "seed": seed, "n_edited": dose,
                "corpus_inflation": dose / 100.0},
        "fairness": {"f_causal_before": 0.8, "f_causal_after": 0.8 + d_fc,
                     "f_spatial_before": 0.1, "f_spatial_after": 0.1,
                     "deltas": {"f_causal": d_fc, "f_spatial": 0.0}},
    }))
    (d / "external_fairness" / "external_fairness.json").write_text(json.dumps({
        "meta": {}, "theil": {"before": 0.2, "after": 0.19, "delta": -0.01,
                              "delta_ci": [-0.02, 0.0], "n_dropped": 0},
        "metrics": {"MigrantRatio": {"district_extremes": {
            "demographic_parity": {"before": 0.5, "after": 0.5 - d_dp,
                                   "delta": -d_dp, "delta_ci": [-d_dp, -d_dp]},
            "disparate_impact": {"before": 0.6, "after": 0.65, "delta": 0.05,
                                 "delta_ci": [0.0, 0.1]},
        }}},
    }))
    return d


def test_summarize_arms(tmp_path):
    dirs = [
        _fake_arm_dir(tmp_path, "targeted", 5, 0, d_fc=0.01, d_dp=0.02),
        _fake_arm_dir(tmp_path, "targeted", 10, 0, d_fc=0.02, d_dp=0.04),
        _fake_arm_dir(tmp_path, "placebo", 10, 0, d_fc=0.001, d_dp=0.001),
    ]
    md = rdo.summarize_arms(dirs)
    assert "oversample-targeted-d10" in md
    assert "+0.0200" in md                      # targeted d10 ΔF_causal
    assert "placebo" in md


def test_summarize_cli_writes_outputs(tmp_path):
    dirs = [_fake_arm_dir(tmp_path, "targeted", 5, 0, 0.01, 0.02),
            _fake_arm_dir(tmp_path, "placebo", 5, 0, 0.0, 0.0)]
    out = tmp_path / "summary_out"
    rc = rdo.main(["--summarize", *map(str, dirs), "--out", str(out)])
    assert rc == 0
    assert (out / "summary.md").exists()
    assert (out / "dose_response.png").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest famail_temporal/baselines/tests/test_run_demographic_oversampling.py -q`
Expected: 2 new tests FAIL (`AttributeError: ... 'summarize_arms'` /
`main` rejects `--summarize`).

- [ ] **Step 3: Write the implementation (append to the runner; adjust `parse_args`/`main`)**

Replace `parse_args` and `main` with:

```python
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="famail_temporal.baselines.run_demographic_oversampling")
    ap.add_argument("--variant", choices=["targeted", PLACEBO])
    ap.add_argument("--dose", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out-root", type=Path,
                    default=Path(config.PACKAGE_ROOT) / "results")
    ap.add_argument("--summarize", nargs="+", type=Path, default=None,
                    help="Arm dirs to summarize into a dose-response table+figure")
    ap.add_argument("--out", type=Path, default=None,
                    help="--summarize output dir")
    args = ap.parse_args(argv)
    if args.summarize is None and (args.variant is None or args.dose is None):
        ap.error("--variant and --dose are required (unless --summarize)")
    return args
```

Append:

```python
def _arm_row(arm_dir: Path) -> dict:
    meta = json.loads((arm_dir / "metrics.json").read_text())
    ext = json.loads(
        (arm_dir / "external_fairness" / "external_fairness.json").read_text())
    mig = ext["metrics"]["MigrantRatio"]["district_extremes"]
    return {
        "mode": meta["arm"]["mode"],
        "variant": meta["arm"]["variant"],
        "dose": meta["arm"]["dose"],
        "seed": meta["arm"]["seed"],
        "corpus_inflation": meta["arm"].get("corpus_inflation"),
        "d_f_causal": meta["fairness"]["deltas"]["f_causal"],
        "d_f_spatial": meta["fairness"]["deltas"]["f_spatial"],
        "d_dp_migrant": mig["demographic_parity"]["delta"],
        "d_di_migrant": mig["disparate_impact"]["delta"],
        "d_theil": ext["theil"]["delta"],
    }


def summarize_arms(arm_dirs) -> str:
    rows = sorted((_arm_row(Path(d)) for d in arm_dirs),
                  key=lambda r: (r["variant"], r["dose"], r["seed"]))
    lines = [
        "# Demographic Oversampling — dose-response summary", "",
        "| Arm | seed | inflation | ΔF_causal | ΔF_spatial | ΔDP (migrant/extremes) "
        "| ΔDI (migrant/extremes) | ΔTheil |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['seed']} | {r['corpus_inflation']:.3f} "
            f"| {r['d_f_causal']:+.4f} | {r['d_f_spatial']:+.4f} "
            f"| {r['d_dp_migrant']:+.4f} | {r['d_di_migrant']:+.4f} "
            f"| {r['d_theil']:+.4f} |")
    return "\n".join(lines)


def _dose_figure(arm_dirs, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [_arm_row(Path(d)) for d in arm_dirs]
    fig, ax = plt.subplots(figsize=(6, 4))
    for variant, marker in (("targeted", "o"), (PLACEBO, "s")):
        pts = sorted((r for r in rows if r["variant"] == variant),
                     key=lambda r: r["dose"])
        if pts:
            ax.plot([r["dose"] for r in pts], [r["d_f_causal"] for r in pts],
                    marker=marker, label=f"{variant} ΔF_causal")
            ax.plot([r["dose"] for r in pts], [r["d_dp_migrant"] for r in pts],
                    marker=marker, ls="--", label=f"{variant} ΔDP migrant")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("dose (duplicates)")
    ax.set_ylabel("Δ (after − before)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.summarize:
        out = args.out or Path(config.PACKAGE_ROOT) / "baselines" / \
            "demographic_oversampling_results"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.md").write_text(summarize_arms(args.summarize))
        _dose_figure(args.summarize, out / "dose_response.png")
        print(f"[demo_oversample] wrote {out / 'summary.md'}", flush=True)
        return 0
    run(args)
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest famail_temporal/baselines/tests/test_run_demographic_oversampling.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/run_demographic_oversampling.py \
        famail_temporal/baselines/tests/test_run_demographic_oversampling.py
git commit -m "feat(mission3): oversampling dose-response summary table + figure

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Data preflight + execute the 9-run experiment matrix

**Files:**
- Create: (no code) — symlinks + result dirs under `famail_temporal/results/`, summary under
  `famail_temporal/baselines/demographic_oversampling_results/`.

**Interfaces:**
- Consumes: Task 4 CLI, Task 5 `--summarize`.
- Produces: 9 arm dirs + `demographic_oversampling_results/{summary.md,dose_response.png}` for
  Task 7's write-up.

- [ ] **Step 1: Symlink the gitignored data into the worktree (read-only reuse, no 2.6 GB copy)**

```bash
WT=/home/robert/FAMAIL/.claude/worktrees/demographic-oversampling
for f in /home/robert/FAMAIL/famail_temporal/cache/*.pkl; do
  ln -s "$f" "$WT/famail_temporal/cache/" 2>/dev/null || true
done
for f in /home/robert/FAMAIL/famail_temporal/source_data/*.pkl \
         /home/robert/FAMAIL/famail_temporal/source_data/processing_metadata.json; do
  ln -s "$f" "$WT/famail_temporal/source_data/" 2>/dev/null || true
done
ls -l "$WT/famail_temporal/cache" | head -5   # expect symlinks
```

(`grid_to_district_mapping.pkl` is already a real copy — the `|| true` skips it.)

- [ ] **Step 2: Bundle smoke check**

Run: `$PY -c "from famail_temporal.data.loader import DataBundle; b = DataBundle.load(); print(len(b.trajectories), b.pickup_3d.shape, b.n_days)"`
Expected: prints the corpus size (≈34,524 trajectories), `(48, 90, 24)`-like shape, and n_days —
no exception. If an artifact is missing, symlink the file it names and re-run.

- [ ] **Step 3: Dose-100 smoke run (end-to-end, ~1-2 min + bootstrap)**

```bash
$PY -m famail_temporal.baselines.run_demographic_oversampling \
  --variant targeted --dose 100 --seed 0 --bootstrap 100
```

Expected: prints `[demo_oversample] wrote famail_temporal/results/<ts>_baseline_demo_oversample_targeted_d100_s0_shenzhen`.
Inspect `metrics.json`: `fairness.deltas` present and small; `arm.per_stratum_draws` ≈
{34, 33, 33}; `arm.origin_escape_frac` small (< 0.15). Delete the smoke dir afterwards
(`rm -r` it) so it doesn't pollute the summary.

- [ ] **Step 4: Run the 9-arm matrix (sequential, CPU; expect minutes per arm)**

```bash
PY=/home/robert/FAMAIL/.venv/bin/python
for spec in "targeted 2500 0" "targeted 5000 0" "targeted 10000 0" \
            "targeted 10000 1" "targeted 10000 2" \
            "placebo 5000 0" "placebo 10000 0" "placebo 10000 1" "placebo 10000 2"; do
  set -- $spec
  $PY -m famail_temporal.baselines.run_demographic_oversampling \
      --variant "$1" --dose "$2" --seed "$3" \
      2>&1 | tee -a famail_temporal/results/demo_oversample_runs.log
done
```

Expected: 9 `[demo_oversample] wrote ...` lines; no tracebacks in the log.

- [ ] **Step 5: Assemble the summary**

```bash
$PY -m famail_temporal.baselines.run_demographic_oversampling \
  --summarize famail_temporal/results/*_baseline_demo_oversample_*_shenzhen \
  --out famail_temporal/baselines/demographic_oversampling_results
```

Expected: `summary.md` (9 rows) + `dose_response.png`. Read `summary.md` and record the headline
numbers (targeted d10000 vs placebo d10000) for Task 7 — REPORT them; do not editorialize beyond
the spec's pre-registered framing.

- [ ] **Step 6: Commit the summary artifacts (arm dirs stay untracked like other result dirs)**

```bash
git add famail_temporal/baselines/demographic_oversampling_results/
git commit -m "results(mission3): demographic-oversampling 9-arm dose-response summary

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: STATUS.md write-up + ship gates

**Files:**
- Modify: `famail_temporal/baselines/STATUS.md` (append to the Mission-3 section, replacing the
  "Planned 4th arm" stub with the built/ran record)

**Interfaces:**
- Consumes: Task 6's `summary.md` numbers; the spec's §1 pre-registered framing.
- Produces: the run-book + results record future sessions rely on.

- [ ] **Step 1: Update STATUS.md**

Replace the `### Planned 4th arm — Demographic Oversampling (new branch)` subsection with a
`### 4th arm — Demographic Oversampling (BUILT + RUN)` subsection containing, in this order:
(1) one-paragraph what/why (resampling baseline for the supply-lift editor; demand-endogeneity
probe; spec link); (2) the exact 9-run command block from Task 6 Step 4 + the `--summarize`
command (the run-book); (3) a transcription of `summary.md`'s table with the headline
targeted-vs-placebo d10000 comparison called out; (4) the disclosure block from the spec §5
verbatim (phantom drivers fabricated; duplicates trivially pass fidelity — not scored, by
construction; corpus inflation = dose); (5) a pointer to
`demographic_oversampling_results/dose_response.png`.

- [ ] **Step 2: Run the ship gates**

```bash
git diff main -- famail_temporal/algorithm/ famail_temporal/evaluation/runner.py | wc -l
```
Expected: `0` (frozen-algorithm gate).

```bash
$PY -m pytest famail_temporal/ -q 2>&1 | tail -3
```
Expected: all tests pass (branch baseline 852 + the ~23 new ≈ 875 passed, 8 skipped, 0 failed).

- [ ] **Step 3: Commit**

```bash
git add famail_temporal/baselines/STATUS.md
git commit -m "docs(mission3): demographic-oversampling run-book + results in STATUS.md

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Report results to the user**

Surface the summary table + dose-response figure and the targeted-vs-placebo contrast for
discussion (feedback protocol: surface findings, don't silently interpret). Merge/PR decisions
happen via superpowers:finishing-a-development-branch AFTER the user reviews the results.
