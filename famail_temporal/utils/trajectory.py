"""Trajectory representation for famail_temporal."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def taper_weights(l_eff: int) -> Tuple[float, ...]:
    """Linear taper weights for a tail of length ``l_eff``: (1/l_eff, 2/l_eff, ..., 1.0).

    Weight index j (1-indexed) corresponds to the tail state that is j steps
    from the anchor (j=1 is closest to the anchor, j=l_eff is closest to the
    pickup). Equals ``config.TAIL_TAPER`` when ``l_eff == 4``.
    """
    if l_eff <= 0:
        return tuple()
    return tuple(j / l_eff for j in range(1, l_eff + 1))


def _clip_rounded_offset(coord: float, delta_axis: float, dim: int) -> int:
    """Round a continuous per-axis delta and clip so ``coord + offset`` stays in
    ``[0, dim - 1]``."""
    lo = -coord
    hi = dim - 1 - coord
    return int(np.clip(round(delta_axis), lo, hi))


def _repair_axis_offsets(coords: List[float], anchor: int, pickup_idx: int, l_eff: int,
                          pickup_offset: int, delta_axis: float,
                          grid_dim: int) -> Optional[Dict[int, int]]:
    """Backward greedy repair for one axis: returns {tail_index: offset} feasible against
    the ORIGINAL per-axis steps, or None if infeasible for this ``l_eff``.

    Backward DP: R[i] = the interval of offsets at state i from which some chain of
    unit-king-move-compliant offsets can reach the (fixed) pickup offset, intersected
    with the in-grid bound for state i. Built from pickup back to the first tail state
    (a contiguous interval by induction, since each step's feasible-offset window is a
    width-3 interval that slides by 1 as the downstream offset varies by 1). The
    anchor's offset (0, fixed) is then linked in via an intersection at the first tail
    state, and remaining tail offsets are filled forward, each clamped to the closest
    feasible value to its tapered target.
    """
    reach: Dict[int, Tuple[int, int]] = {pickup_idx: (pickup_offset, pickup_offset)}
    for i in range(pickup_idx - 1, anchor, -1):
        lo_next, hi_next = reach[i + 1]
        step = coords[i + 1] - coords[i]
        lo = lo_next + step - 1
        hi = hi_next + step + 1
        g_lo, g_hi = -coords[i], grid_dim - 1 - coords[i]
        lo, hi = max(lo, g_lo), min(hi, g_hi)
        if lo > hi:
            return None
        reach[i] = (lo, hi)

    if l_eff == 0:
        step_anchor = coords[pickup_idx] - coords[anchor]
        if abs(step_anchor + pickup_offset) > 1:
            return None
        return {}

    first = anchor + 1
    lo_r, hi_r = reach[first]
    step_anchor = coords[first] - coords[anchor]
    w_lo, w_hi = -step_anchor - 1, -step_anchor + 1
    lo_final, hi_final = max(lo_r, w_lo), min(hi_r, w_hi)
    if lo_final > hi_final:
        return None

    taper = taper_weights(l_eff)
    offsets: Dict[int, int] = {}
    target = round(taper[0] * delta_axis)
    val = int(min(max(target, lo_final), hi_final))
    offsets[first] = val
    prev, prev_idx = val, first

    for idx in range(first + 1, pickup_idx):
        lo_r2, hi_r2 = reach[idx]
        step_prev = coords[idx] - coords[prev_idx]
        lo_c = max(prev - step_prev - 1, lo_r2)
        hi_c = min(prev - step_prev + 1, hi_r2)
        if lo_c > hi_c:
            return None
        j = idx - anchor
        target = round(taper[j - 1] * delta_axis)
        val = int(min(max(target, lo_c), hi_c))
        offsets[idx] = val
        prev, prev_idx = val, idx

    return offsets


@dataclass
class TrajectoryState:
    """Single state in a trajectory: (x_grid, y_grid, time_bucket, day_index)."""

    x_grid: float
    y_grid: float
    time_bucket: int
    day_index: int

    def to_array(self) -> np.ndarray:
        """Convert to [x, y, time, day] numpy array."""
        return np.array([self.x_grid, self.y_grid, self.time_bucket, self.day_index])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "TrajectoryState":
        """Create from [x, y, time, day] numpy array."""
        return cls(
            x_grid=float(arr[0]), y_grid=float(arr[1]),
            time_bucket=int(arr[2]), day_index=int(arr[3]),
        )


@dataclass
class Trajectory:
    """A taxi trajectory. The pickup is the final state (states[-1])."""

    trajectory_id: Any
    driver_id: Any
    states: List[TrajectoryState]
    metadata: dict = field(default_factory=dict)

    @property
    def pickup_state(self) -> TrajectoryState:
        """The final state — the pickup event."""
        return self.states[-1]

    @property
    def pickup_cell(self) -> Tuple[int, int]:
        """Pickup cell as integer (x, y)."""
        s = self.pickup_state
        return (int(s.x_grid), int(s.y_grid))

    @property
    def n_states(self) -> int:
        """Number of states in this trajectory."""
        return len(self.states)

    def to_discriminator_format(self) -> np.ndarray:
        """Return shape (seq_len, 4) numpy array: rows are [x, y, time, day]."""
        return np.array([s.to_array() for s in self.states])

    def to_tensor(self) -> torch.Tensor:
        """Return shape (seq_len, 4) float32 torch tensor."""
        return torch.tensor(self.to_discriminator_format(), dtype=torch.float32)

    def clone(self) -> "Trajectory":
        """Deep copy: states and metadata are new objects."""
        return Trajectory(
            trajectory_id=self.trajectory_id,
            driver_id=self.driver_id,
            states=[TrajectoryState(s.x_grid, s.y_grid, s.time_bucket, s.day_index)
                    for s in self.states],
            metadata=self.metadata.copy(),
        )

    def apply_perturbation(self, delta: np.ndarray,
                           grid_dims: Tuple[int, int] = (48, 90)) -> "Trajectory":
        """Return a new trajectory with (delta_x, delta_y) applied to the pickup, clipped to grid bounds."""
        modified = self.clone()
        pickup = modified.states[-1]
        new_x = float(np.clip(pickup.x_grid + delta[0], 0, grid_dims[0] - 1))
        new_y = float(np.clip(pickup.y_grid + delta[1], 0, grid_dims[1] - 1))
        modified.states[-1] = TrajectoryState(
            x_grid=new_x, y_grid=new_y,
            time_bucket=pickup.time_bucket, day_index=pickup.day_index,
        )
        return modified

    def apply_tail_perturbation(self, delta: np.ndarray, tail_len: int,
                                grid_dims: Tuple[int, int] = (48, 90)) -> Optional["Trajectory"]:
        """Translate the seeking TAIL (pickup + up to ``tail_len`` preceding states) by a
        linearly tapered, per-axis-independent offset derived from ``delta``, repairing
        king-move adjacency (``max(|dx|, |dy|) <= 1`` between consecutive states) via
        backward greedy assignment (pickup -> anchor).

        The pickup offset is ``round(delta)`` per axis, clipped so the new pickup stays
        in-grid. Tail state j (1-indexed from the anchor side, j=1..L_eff) targets
        ``round(taper_weights(L_eff)[j-1] * delta)``, clamped into the feasible interval
        that preserves adjacency against the ORIGINAL (pre-edit) steps. The anchor
        (index ``len(states) - 2 - L_eff``) is left at offset (0, 0).

        If no compliant assignment exists, ``L_eff`` is deepened (grown toward
        ``len(states) - 2``, i.e. anchor = states[0]) and retried; if it is still
        infeasible at the deepest tail, returns ``None`` (caller should skip the edit).

        Never mutates ``self``; time_bucket/day_index are copied unchanged (spatial-only
        edit).
        """
        n = self.n_states
        if n < 2:
            return None

        xs = [s.x_grid for s in self.states]
        ys = [s.y_grid for s in self.states]
        pickup_idx = n - 1

        pickup_off_x = _clip_rounded_offset(xs[pickup_idx], delta[0], grid_dims[0])
        pickup_off_y = _clip_rounded_offset(ys[pickup_idx], delta[1], grid_dims[1])

        max_l_eff = n - 2
        start_l_eff = min(tail_len, max_l_eff)
        if start_l_eff < 0:
            return None

        for l_eff in range(start_l_eff, max_l_eff + 1):
            anchor = n - 2 - l_eff
            offsets_x = _repair_axis_offsets(xs, anchor, pickup_idx, l_eff,
                                              pickup_off_x, delta[0], grid_dims[0])
            if offsets_x is None:
                continue
            offsets_y = _repair_axis_offsets(ys, anchor, pickup_idx, l_eff,
                                              pickup_off_y, delta[1], grid_dims[1])
            if offsets_y is None:
                continue

            out = self.clone()
            pickup_state = self.states[pickup_idx]
            out.states[pickup_idx] = TrajectoryState(
                x_grid=xs[pickup_idx] + pickup_off_x,
                y_grid=ys[pickup_idx] + pickup_off_y,
                time_bucket=pickup_state.time_bucket, day_index=pickup_state.day_index,
            )
            for idx in range(anchor + 1, pickup_idx):
                orig_state = self.states[idx]
                out.states[idx] = TrajectoryState(
                    x_grid=xs[idx] + offsets_x[idx],
                    y_grid=ys[idx] + offsets_y[idx],
                    time_bucket=orig_state.time_bucket, day_index=orig_state.day_index,
                )
            # anchor (index `anchor`) is left untouched by construction (clone default).
            return out

        return None
