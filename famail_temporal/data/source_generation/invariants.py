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
    tbs = [s[2] for s in traj]
    for a, b in zip(tbs, tbs[1:]):
        if b < a:
            return False, 4, "temporal_order", {"time_buckets": tbs}
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

    def process(by_plate: dict[str, list[Trajectory]], kind: str):
        for plate, traj_list in by_plate.items():
            keep_list: list[Trajectory] = []
            for idx, traj in enumerate(traj_list):
                ok, inv_num, category, fv = _validate_single_trajectory(
                    traj, kind, pickup_counts, dropoff_counts,
                )
                if ok:
                    keep_list.append(traj)
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
                else:
                    kept.driving_by_plate[plate] = keep_list

    process(trajs.seeking_by_plate, "seeking")
    process(trajs.driving_by_plate, "driving")
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
        if not np.allclose(col_std, 1.0, atol=1e-5):
            raise SystemicInvariantError(
                f"#7: profile column stds not ~1: {col_std}"
            )
