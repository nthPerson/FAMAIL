"""
DataBundle dataclass — immutable container for all data and artifacts needed
to instantiate FAMAILObjective and TrajectoryModifier.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch.nn as nn

from famail_temporal.data.active_mask import UnitIndexMap
from famail_temporal.fairness.g0_power_basis import G0Function
from famail_temporal.fidelity.context import MultiStreamData
from famail_temporal.utils.trajectory import Trajectory


@dataclass(frozen=True, kw_only=True)
class DataBundle:
    """Immutable container for all data and artifacts needed by FAMAILObjective
    and TrajectoryModifier.

    Invariants:
    - pickup_3d, dropoff_3d, active_taxis_3d, and mask_3d all share the same
      (grid_x, grid_y, T) shape.
    - unit_map.n_units == hat_matrices['I_minus_H_demo'].shape[0] (Task 11
      enforces this at precompute time).
    - len(n_hours_per_block) == T (the block axis of the 3D tensors).
    - Fields are rebinding-frozen; contained numpy arrays and dicts may still
      be mutated by reference. Task 11's setflags ensures hat matrices are
      read-only; other tensors follow the same pattern downstream.

    Use ``DataBundle.load()`` to construct from cached artifacts produced by
    ``python -m famail_temporal.preprocess``.
    """

    # Data tensors — all shape (48, 90, T), same spatial/block axes
    pickup_3d: np.ndarray        # (48, 90, T) float32 — mean hourly pickups per (cell, block)
    dropoff_3d: np.ndarray       # (48, 90, T) float32 — mean hourly dropoffs per (cell, block)
    active_taxis_3d: np.ndarray  # (48, 90, T) float32 — mean hourly active taxis per (cell, block)
    mask_3d: np.ndarray          # (48, 90, T) bool — active-unit mask

    # Aggregation metadata
    n_hours_per_block: np.ndarray  # (T,) int32 — hours spanned by each time block
    n_days: int                    # Number of days aggregated over

    # Derived artifacts
    unit_map: UnitIndexMap         # canonical ordering of active (cell, t) units
    g0_func: G0Function            # fitted g_0(D) power-basis function
    hat_matrices: Dict[str, np.ndarray]  # keys: 'I_minus_H_demo', 'M' (and optionally 'scaler_*')

    # Trajectories + multi-stream context
    trajectories: List[Trajectory]
    multi_stream: MultiStreamData

    # Model
    discriminator: nn.Module


# ---------------------------------------------------------------------------
# DataBundle.load() — construct from cached artifacts + raw trajectory data
# ---------------------------------------------------------------------------
import pickle as _pkl
import random as _random

from famail_temporal.data.cache_io import load_artifact
from famail_temporal.data.aggregation import block_n_hours
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


def _parse_trajectory(traj_data, trajectory_id, driver_id):
    """Parse a single trajectory list into a Trajectory, or None if too short."""
    if not isinstance(traj_data, list) or len(traj_data) < 2:
        return None
    states = []
    for state_data in traj_data:
        if len(state_data) >= 4:
            states.append(TrajectoryState(
                x_grid=int(state_data[0]) - 1,   # 1-indexed → 0-indexed
                y_grid=int(state_data[1]) - 1,
                time_bucket=int(state_data[2]),
                day_index=int(state_data[3]),
            ))
    if len(states) < 2:
        return None
    return Trajectory(
        trajectory_id=trajectory_id, driver_id=driver_id, states=states,
    )


def _load_driver_index_mapping() -> dict:
    """Load the plate_id ↔ int driver_idx sidecar produced by source_generation.

    Returns a dict with keys 'plate_to_idx' and 'idx_to_plate'. If the sidecar
    isn't present (e.g., tests or legacy source data), returns {} and callers
    pass plate_id through unchanged.
    """
    from famail_temporal import config
    path = config.SOURCE_DATA_DIR / "driver_index_mapping.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return _pkl.load(f)


def _resolve_driver_id(raw_key, plate_to_idx: dict):
    """Convert a raw plate_id (string) to the integer driver_idx.

    The downstream `ms_*` dicts are keyed by int driver_idx (0..49), and two
    consumers (`evaluation/augment.py`, `fidelity/context.py`) expect every
    `Trajectory.driver_id` to be convertible to int so those lookups succeed.
    Source data from the new source_generation tool is keyed by raw plate_id
    strings; we convert at load time using the sidecar mapping.

    Falls back to returning the raw key unchanged if it's already an int or
    if no mapping is available (tests, legacy). A plate_id that's present in
    the data but missing from the mapping raises with a clear message — that
    indicates a regeneration/mapping drift, not a silent data issue.
    """
    if isinstance(raw_key, int):
        return raw_key
    if not plate_to_idx:
        return raw_key
    if raw_key not in plate_to_idx:
        raise KeyError(
            f"plate_id {raw_key!r} is in passenger_seeking_trajs.pkl but "
            f"missing from driver_index_mapping.pkl. Re-run the source-"
            f"generation tool to regenerate both files together."
        )
    return plate_to_idx[raw_key]


def _load_trajectories(max_trajectories=None, max_drivers=None):
    """Load passenger-seeking trajectories from source_data.

    Converts raw plate_id keys (strings from raw GPS) into integer driver_idx
    values using driver_index_mapping.pkl, so every `Trajectory.driver_id` is
    an int in [0, 49] — matching the int-keyed multi-stream context dicts
    consumed by `fidelity/context.py` and the int-keyed output dict produced
    by `evaluation/augment.py`.
    """
    from famail_temporal import config
    path = config.SOURCE_DATA_DIR / "passenger_seeking_trajs.pkl"
    with open(path, "rb") as f:
        data = _pkl.load(f)
    mapping = _load_driver_index_mapping()
    plate_to_idx = mapping.get("plate_to_idx", {})
    driver_keys = list(data.keys())
    if max_drivers:
        driver_keys = driver_keys[:max_drivers]
    all_trajs = []
    for did in driver_keys:
        resolved_id = _resolve_driver_id(did, plate_to_idx)
        for td in data[did]:
            all_trajs.append((resolved_id, td))
    if max_trajectories and len(all_trajs) > max_trajectories:
        _random.seed(config.DEFAULT_SEED)
        all_trajs = _random.sample(all_trajs, max_trajectories)
    out = []
    for i, (did, td) in enumerate(all_trajs):
        t = _parse_trajectory(td, trajectory_id=i, driver_id=did)
        if t is not None:
            out.append(t)
    return out


def _load_multi_stream():
    """Load the five multi-stream context dicts from source_data."""
    from famail_temporal import config

    def _load(filename):
        path = config.SOURCE_DATA_DIR / filename
        with open(path, "rb") as f:
            return _pkl.load(f)

    driving = {int(k): v for k, v in _load("ms_driving_trajs.pkl").items()}
    seeking = {int(k): v for k, v in _load("ms_seeking_trajs.pkl").items()}
    profile_raw = _load("ms_profile_features.pkl")
    raw_features = profile_raw.get("features_normalized", profile_raw)
    profile = {int(k): v for k, v in raw_features.items()}
    seeking_days = {int(k): v for k, v in _load("ms_seeking_calendar_days.pkl").items()}
    driving_days = {int(k): v for k, v in _load("ms_driving_calendar_days.pkl").items()}
    return MultiStreamData(
        driving_trajs=driving, seeking_trajs=seeking,
        profile_features=profile,
        seeking_days=seeking_days, driving_days=driving_days,
    )


def _load_discriminator_stub():
    """Return a no-op nn.Identity when no checkpoint is available."""
    import torch.nn as _nn
    return _nn.Identity()


def _bundle_load(max_trajectories=None, max_drivers=None):
    """Load cached artifacts + raw trajectories into a DataBundle."""
    from famail_temporal import config

    pickup_3d = load_artifact("pickup_counts")
    dropoff_3d = load_artifact("dropoff_counts")
    active_taxis_3d = load_artifact("active_taxis")
    mask_3d = load_artifact("active_mask")
    unit_map = load_artifact("unit_index_map")
    g0_func = load_artifact("g0_power_basis")
    hat_matrices = load_artifact("hat_matrices", include_features=True)

    # Shape consistency — fail loud with actionable error messages.
    if unit_map.n_units != hat_matrices['I_minus_H_demo'].shape[0]:
        raise ValueError(
            f"unit_map.n_units ({unit_map.n_units}) != hat matrix "
            f"shape[0] ({hat_matrices['I_minus_H_demo'].shape[0]}). "
            f"Regenerate cache with: python -m famail_temporal.preprocess --force"
        )
    if pickup_3d.shape != dropoff_3d.shape or pickup_3d.shape != active_taxis_3d.shape:
        raise ValueError(
            f"3D tensor shape mismatch: pickup {pickup_3d.shape}, "
            f"dropoff {dropoff_3d.shape}, active_taxis {active_taxis_3d.shape}"
        )
    expected_shape = (config.GRID_DIMS[0], config.GRID_DIMS[1], config.T)
    if pickup_3d.shape != expected_shape:
        raise ValueError(
            f"pickup_3d shape {pickup_3d.shape} != expected {expected_shape}"
        )

    n_hours_per_block = np.array(
        [block_n_hours(t) for t in range(config.T)], dtype=np.int32,
    )

    metadata = load_artifact("metadata")
    n_days = metadata['n_days']

    trajectories = _load_trajectories(
        max_trajectories=max_trajectories, max_drivers=max_drivers,
    )
    multi_stream = _load_multi_stream()

    try:
        from famail_temporal.fidelity.checkpoint import load_discriminator
        ckpt_path = (
            config.DISCRIMINATOR_CHECKPOINT_DIR
            / config.DISCRIMINATOR_CHECKPOINT_FILENAME
        )
        if ckpt_path.exists():
            discriminator = load_discriminator(ckpt_path)
        else:
            discriminator = _load_discriminator_stub()
    except (ImportError, ModuleNotFoundError):
        discriminator = _load_discriminator_stub()

    return DataBundle(
        pickup_3d=pickup_3d.copy(),
        dropoff_3d=dropoff_3d.copy(),
        active_taxis_3d=active_taxis_3d.copy(),
        mask_3d=mask_3d.copy(),
        n_hours_per_block=n_hours_per_block,
        n_days=n_days,
        unit_map=unit_map,
        g0_func=g0_func,
        hat_matrices=hat_matrices,
        trajectories=trajectories,
        multi_stream=multi_stream,
        discriminator=discriminator,
    )


def _bundle_load_classmethod(cls, max_trajectories=None, max_drivers=None):
    """Load cached artifacts + raw trajectories into a DataBundle.

    Requires that preprocess.py has been run (python -m famail_temporal.preprocess).
    Uses cache artifacts; falls back to nn.Identity() for the discriminator
    if the checkpoint is not present.
    """
    return _bundle_load(max_trajectories, max_drivers)


DataBundle.load = classmethod(_bundle_load_classmethod)
