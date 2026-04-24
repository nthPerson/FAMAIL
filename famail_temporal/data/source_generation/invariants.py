"""Per-trajectory + systemic invariant enforcement (see §6 of the design spec)."""
from __future__ import annotations

import numpy as np

from famail_temporal.data.source_generation import config
from famail_temporal.data.source_generation.removal import (
    RemovalRecord,
)
from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult, Trajectory,
)


class SystemicInvariantError(Exception):
    """Raised when an invariant failure cannot be attributed to one trajectory."""


def _validate_single_trajectory(
    traj: Trajectory, kind: str,
    pickup_counts: dict, dropoff_counts: dict,
) -> tuple[bool, int, str, dict]:
    if len(traj) < 2:
        return False, 3, "degenerate_length", {"n_states": len(traj)}
    # Temporal-order check (design spec §6, invariant #4).
    # Catches the one structural violation the upstream sort can't mask:
    # same-day, backward-time_bucket transitions.
    for a, b in zip(traj, traj[1:]):
        ta, da = a[2], a[3]
        tb_, db = b[2], b[3]
        if da == db and tb_ < ta:
            return False, 4, "temporal_order", {
                "day_time_buckets": [(s[3], s[2]) for s in traj],
            }
    # Action-space-violation check (design spec §6, invariant #6). Enforces
    # physical consistency with the 9 possible actions of the original
    # all_trajs.pkl state vector: each consecutive-state transition must
    # satisfy max(|dx|, |dy|) <= 1 (8 compass moves + stay). Trajectories
    # with GPS-dropout or high-speed-movement jumps cannot be rollouts of
    # a 9-action agent and are rejected whole. First violation wins.
    for i, (a, b) in enumerate(zip(traj, traj[1:])):
        max_axis_delta = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        if max_axis_delta > 1:
            return False, 6, "action_space_violation", {
                "from": tuple(a),
                "to": tuple(b),
                "max_axis_delta": max_axis_delta,
                "transition_index": i,
            }
    # Plausibility-of-duration check (design spec §6, invariant #5, research-
    # grounded). A seeking or driving trajectory is a single episode of
    # cruising-between-trips (seeking) or carrying-a-passenger (driving). In
    # urban taxi data these are typically minutes; anything longer than a
    # standard work day (MAX_TRAJECTORY_DURATION_BUCKETS, default 96 buckets
    # = 8 hours) is almost certainly an extraction artifact — a segment that
    # got "stitched" across off-duty time because no passenger-indicator
    # transition occurred in between (e.g., a Friday-evening segment that
    # continues into Monday because Sat+Sun records were filtered out).
    #
    # Rejecting these catches weekend-spanning trajectories by construction
    # (they accumulate ≥48 hours of elapsed time) and any other long-gap
    # artifact, without needing to special-case day-of-week values.
    start_day, start_tb = traj[0][3], traj[0][2]
    end_day, end_tb = traj[-1][3], traj[-1][2]
    if end_day < start_day:
        # Day wrapped backward — must involve ≥1 weekend, so elapsed time is
        # at minimum 2 days. Always implausible for a single episode.
        return False, 5, "implausibly_long", {
            "start": (start_day, start_tb),
            "end": (end_day, end_tb),
            "reason": "day_index wrapped backward (weekend or multi-week gap)",
        }
    if end_day == start_day:
        duration_buckets = end_tb - start_tb
    else:
        days_gap = end_day - start_day
        duration_buckets = (config.TIME_BUCKET_MAX - start_tb) + \
                           (days_gap - 1) * config.TIME_BUCKET_MAX + \
                           end_tb
    if duration_buckets > config.MAX_TRAJECTORY_DURATION_BUCKETS:
        return False, 5, "implausibly_long", {
            "start": (start_day, start_tb),
            "end": (end_day, end_tb),
            "duration_buckets": duration_buckets,
            "duration_minutes": duration_buckets * 5,
            "threshold_buckets": config.MAX_TRAJECTORY_DURATION_BUCKETS,
        }
    for s in traj:
        x, y, tb, day = s
        if not (1 <= x <= config.X_GRID_MAX):
            return False, 2, "out_of_bounds", {"state": s, "axis": "x"}
        if not (1 <= y <= config.Y_GRID_MAX):
            return False, 2, "out_of_bounds", {"state": s, "axis": "y"}
        if not (1 <= tb <= config.TIME_BUCKET_MAX):
            return False, 2, "out_of_bounds", {"state": s, "axis": "time_bucket"}
        if day not in config.WEEKDAY_DAYS:
            return False, 2, "out_of_bounds", {"state": s, "axis": "day"}
    x, y, tb, day = traj[-1]
    key = (x, y, tb, day)
    if kind == "seeking":
        p, _ = pickup_counts.get(key, (0, 0))
        if p < 1:
            return False, 1, "no_matching_count", {"endpoint": key, "pickup_count": p}
    else:
        _, d = dropoff_counts.get(key, (0, 0))
        if d < 1:
            return False, 1, "no_matching_count", {"endpoint": key, "dropoff_count": d}
    return True, -1, "", {}


def apply_per_trajectory_invariants(
    trajs: TrajectoriesResult,
    pickup_counts: dict,
    dropoff_counts: dict,
    plate_to_idx: dict[str, int] | None = None,
) -> tuple[TrajectoriesResult, list[RemovalRecord]]:
    """Validate each trajectory; drop violations and record them."""
    plate_to_idx = plate_to_idx or {}
    kept = TrajectoriesResult()
    removals: list[RemovalRecord] = []

    def process(
        by_plate: dict[str, list[Trajectory]],
        dates_by_plate: dict[str, list[str]],
        kind: str,
    ):
        for plate, traj_list in by_plate.items():
            dates_list = dates_by_plate.get(plate, [])
            keep_list: list[Trajectory] = []
            keep_dates: list[str] = []
            for idx, traj in enumerate(traj_list):
                ok, inv_num, category, fv = _validate_single_trajectory(
                    traj, kind, pickup_counts, dropoff_counts,
                )
                if ok:
                    keep_list.append(traj)
                    if idx < len(dates_list):
                        keep_dates.append(dates_list[idx])
                else:
                    removals.append(RemovalRecord(
                        driver_id=plate,
                        driver_idx=plate_to_idx.get(plate),
                        trajectory_index_within_driver=idx,
                        kind=kind,
                        which_invariant=inv_num,
                        failing_values=fv,
                        n_states_before_removal=len(traj),
                        removal_reason_category=category,
                    ))
            if keep_list:
                if kind == "seeking":
                    kept.seeking_by_plate[plate] = keep_list
                    if keep_dates:
                        kept.seeking_dates_by_plate[plate] = keep_dates
                else:
                    kept.driving_by_plate[plate] = keep_list
                    if keep_dates:
                        kept.driving_dates_by_plate[plate] = keep_dates

    process(trajs.seeking_by_plate, trajs.seeking_dates_by_plate, "seeking")
    process(trajs.driving_by_plate, trajs.driving_dates_by_plate, "driving")
    return kept, removals


def check_systemic_invariants(
    trajs: TrajectoriesResult,
    pickup_counts: dict,
    dropoff_counts: dict,
    profile_matrix: np.ndarray | None,
    n_drivers: int,
    expect_n_drivers: int = config.EXPECTED_N_DRIVERS,
) -> None:
    """Raise SystemicInvariantError on any systemic invariant failure."""
    total_pickups = sum(v[0] for v in pickup_counts.values())
    total_dropoffs = sum(v[1] for v in dropoff_counts.values())
    n_seeking = sum(len(v) for v in trajs.seeking_by_plate.values())
    n_driving = sum(len(v) for v in trajs.driving_by_plate.values())
    if total_pickups != n_seeking:
        raise SystemicInvariantError(
            f"#5: sum(pickup_counts)={total_pickups} != n_seeking={n_seeking}"
        )
    if total_dropoffs != n_driving:
        raise SystemicInvariantError(
            f"#5: sum(dropoff_counts)={total_dropoffs} != n_driving={n_driving}"
        )
    if n_drivers != expect_n_drivers:
        raise SystemicInvariantError(
            f"#6: got {n_drivers} unique drivers; expected {expect_n_drivers}"
        )
    if profile_matrix is not None:
        if profile_matrix.shape != (expect_n_drivers, config.N_PROFILE_FEATURES):
            raise SystemicInvariantError(
                f"#7: profile shape {profile_matrix.shape} != "
                f"({expect_n_drivers}, {config.N_PROFILE_FEATURES})"
            )
        if np.isnan(profile_matrix).any():
            raise SystemicInvariantError("#7: profile contains NaN")
        col_mean = profile_matrix.mean(axis=0)
        col_std = profile_matrix.std(axis=0, ddof=0)
        if not np.allclose(col_mean, 0.0, atol=1e-5):
            raise SystemicInvariantError(
                f"#7: profile column means not ~0: {col_mean}"
            )
        # zscore_normalize preserves constant raw columns at 0 (std_safe=1
        # when raw std < 1e-12), so post-normalization std is either ~0
        # (constant column) or ~1 (varying column). The invariant checks
        # only that non-constant columns normalized correctly.
        varying_cols = col_std > 0.5
        if varying_cols.any() and not np.allclose(
            col_std[varying_cols], 1.0, atol=1e-5,
        ):
            raise SystemicInvariantError(
                f"#7: profile column stds not ~1: {col_std}"
            )
