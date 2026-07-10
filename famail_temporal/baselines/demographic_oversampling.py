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
