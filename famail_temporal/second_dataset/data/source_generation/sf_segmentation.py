"""Occupancy + gap segmentation of SF cab traces (Task 3.2).

Splits a single driver's time-sorted pings into **seeking** (occupancy 0) and
**driving** (occupancy 1) trajectories — the two streams the F_fidelity
discriminator consumes — breaking a run on an occupancy change or a > `gap_sec`
time gap. Emits 1-indexed `[x, y, time_bucket, day]` states (5-min buckets,
calendar-day ints in SF local time) and the pickup/dropoff transition events.

Day convention (Phase 4): the `day` field of every emitted state is the
**absolute local epoch-day serial** (``local // 86400``), which is exactly what
the pickup/dropoff count path (sf_build → sf_grid_counts) needs so that
`preprocess.dataset_n_days` sees the true calendar-day count. The discriminator
wants day-of-week instead; that remap happens downstream in
`sf_multistream.assemble_multistream` (on copies), leaving this module and the
fairness-count artifacts byte-identical to the pre-Phase-4 build.

`seeking_days` / `driving_days` are **parallel to** `seeking` / `driving`: one
absolute calendar day per trajectory (its start day). This matches the
per-trajectory contract that `discriminator/.../generation.py` enforces
(``len(calendar_days[d]) == len(trajs[d])``); the earlier sorted-distinct form
would make Ren pair-generation raise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from famail_temporal.second_dataset.data.source_generation.sf_config import (
    GridSpec, PDT_OFFSET_SEC,
)

State = List[int]          # [x, y, time_bucket, day]
Trajectory = List[State]


@dataclass(frozen=True)
class SegmentationResult:
    seeking: List[Trajectory]
    driving: List[Trajectory]
    pickups: List[State]
    dropoffs: List[State]
    # Parallel to `seeking` / `driving`: one absolute calendar day (epoch-day
    # serial) per trajectory. len == len(seeking) / len(driving).
    seeking_days: List[int]
    driving_days: List[int]


def segment_driver(
    df_driver,
    grid: GridSpec,
    gap_sec: int = 300,
    tz_offset_sec: int = PDT_OFFSET_SEC,
) -> SegmentationResult:
    """Segment one driver's (time-sorted) pings into seeking/driving + events."""
    n = len(df_driver)
    if n == 0:
        return SegmentationResult([], [], [], [], [], [])

    lat = df_driver["lat"].to_numpy(np.float64)
    lon = df_driver["lon"].to_numpy(np.float64)
    occ = df_driver["occupancy"].to_numpy().astype(int)
    t = df_driver["time_utc"].to_numpy().astype(np.int64)

    # Vectorized grid cell (1-indexed, clipped) + 5-min bucket + local day.
    x = np.clip(np.floor((lat - grid.lat_min) / grid.cell_deg).astype(int),
                0, grid.x_grid_max - 1) + 1
    y = np.clip(np.floor((lon - grid.lon_min) / grid.cell_deg).astype(int),
                0, grid.y_grid_max - 1) + 1
    local = t - tz_offset_sec
    tb = ((local % 86400) // 300 + 1).astype(int)
    day = (local // 86400).astype(int)

    dt = np.diff(t, prepend=t[0])
    seg_break = (dt > gap_sec) | (np.diff(occ, prepend=occ[0]) != 0)
    seg_break[0] = True
    starts = np.flatnonzero(seg_break)
    ends = np.append(starts[1:], n)

    def state(i: int) -> State:
        return [int(x[i]), int(y[i]), int(tb[i]), int(day[i])]

    seeking: List[Trajectory] = []
    driving: List[Trajectory] = []
    # Parallel-to-trajectory calendar days (absolute epoch-day of each
    # trajectory's start state), per the generation.py per-trajectory contract.
    seeking_days: List[int] = []
    driving_days: List[int] = []
    for s, e in zip(starts, ends):
        if e - s < 2:                       # a trajectory needs >= 2 states
            continue
        traj = [state(i) for i in range(s, e)]
        traj_day = int(day[s])
        if occ[s] == 0:
            seeking.append(traj)
            seeking_days.append(traj_day)
        else:
            driving.append(traj)
            driving_days.append(traj_day)

    # Transition events (only across small-gap consecutive pings).
    if n >= 2:
        i = np.arange(1, n)
        small = (t[i] - t[i - 1]) <= gap_sec
        prev, cur = occ[i - 1], occ[i]
        pick_i = i[small & (prev == 0) & (cur == 1)]
        drop_i = i[small & (prev == 1) & (cur == 0)]
    else:
        pick_i = drop_i = np.array([], dtype=int)
    pickups = [state(int(i)) for i in pick_i]
    dropoffs = [state(int(i)) for i in drop_i]

    return SegmentationResult(
        seeking, driving, pickups, dropoffs, seeking_days, driving_days,
    )
