"""
Profile feature computation: 11-dimensional driver profile vector with z-score normalization.

Features 0-3 from pickle files, features 4-10 engineered from extracted trajectories.
Targets Ren et al. (KDD 2020) FCN architecture [64, 32, 8] with ReLU activation.
"""

import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from new_all_trajs.config import GlobalBounds
from new_all_trajs.step1_processor import gps_to_grid

from .config import ExtractionConfig

FEATURE_NAMES = [
    "home_x", "home_y",
    "shift_start", "shift_end",
    "freq_grid_x", "freq_grid_y",
    "avg_seeking_dist", "avg_seeking_time",
    "avg_driving_dist", "avg_driving_time",
    "num_trips_per_day",
]

N_FEATURES = 11


def _trajectory_distance(traj: List[List[int]]) -> float:
    """Sum of Euclidean grid-cell hops between consecutive states."""
    dist = 0.0
    for i in range(1, len(traj)):
        dx = traj[i][0] - traj[i - 1][0]
        dy = traj[i][1] - traj[i - 1][1]
        dist += math.sqrt(dx * dx + dy * dy)
    return dist


def _trajectory_time_span(traj: List[List[int]]) -> float:
    """Time span in time-bucket units (each = 5 minutes)."""
    if len(traj) < 2:
        return 0.0
    return abs(traj[-1][2] - traj[0][2])


def _load_home_locations(
    config: ExtractionConfig,
    index_to_plate: Dict[int, str],
    bounds: GlobalBounds,
) -> Dict[int, Tuple[int, int]]:
    """
    Load home locations from home_loc_plates_dict_all.pkl.
    Converts raw GPS coords to grid + offsets.
    Returns {driver_index: (home_x, home_y)} for covered drivers only.
    """
    filepath = config.feature_data_dir / config.home_loc_file
    with open(filepath, "rb") as f:
        home_loc = pickle.load(f)

    plate_to_index = {p: i for i, p in index_to_plate.items()}
    result = {}

    for plate_id, coords in home_loc.items():
        if plate_id not in plate_to_index:
            continue
        idx = plate_to_index[plate_id]
        lat, lon = float(coords[0]), float(coords[1])
        x, y = gps_to_grid(lat, lon, bounds, config.grid_size)
        x += config.x_grid_offset
        y += config.y_grid_offset
        result[idx] = (x, y)

    return result


def _compute_home_fallback(
    seeking_trajs: Dict[int, List],
    missing_drivers: Set[int],
) -> Dict[int, Tuple[int, int]]:
    """
    For drivers missing from home_loc pickle, compute home location
    as the most frequently visited grid cell during seeking periods.
    Approximates Ren's 'longest-staying grid cell'.
    """
    result = {}
    for driver_idx in missing_drivers:
        trajs = seeking_trajs.get(driver_idx, [])
        if not trajs:
            result[driver_idx] = (0, 0)
            continue
        cell_counter: Counter = Counter()
        for traj in trajs:
            for state in traj:
                cell_counter[(state[0], state[1])] += 1
        most_common = cell_counter.most_common(1)[0][0]
        result[driver_idx] = most_common
    return result


def _load_shift_times(
    config: ExtractionConfig,
    index_to_plate: Dict[int, str],
) -> Dict[int, Tuple[float, float]]:
    """
    Load shift start/end from start_finishing_time.pkl.
    Values are time bucket floats in [0, 287] range.
    """
    filepath = config.feature_data_dir / config.start_finish_file
    with open(filepath, "rb") as f:
        start_finish = pickle.load(f)

    plate_to_index = {p: i for i, p in index_to_plate.items()}
    result = {}

    for plate_id, times in start_finish.items():
        if plate_id not in plate_to_index:
            continue
        idx = plate_to_index[plate_id]
        result[idx] = (float(times[0]), float(times[1]))

    return result


def _compute_freq_grid(
    seeking_trajs: Dict[int, List],
    driving_trajs: Dict[int, List],
) -> Dict[int, Tuple[int, int]]:
    """
    Most frequently visited (x, y) grid cell per driver,
    across ALL trajectories (both seeking and driving).
    """
    all_drivers = sorted(set(seeking_trajs.keys()) | set(driving_trajs.keys()))
    result = {}

    for driver_idx in all_drivers:
        cell_counter: Counter = Counter()
        for traj in seeking_trajs.get(driver_idx, []):
            for state in traj:
                cell_counter[(state[0], state[1])] += 1
        for traj in driving_trajs.get(driver_idx, []):
            for state in traj:
                cell_counter[(state[0], state[1])] += 1
        if cell_counter:
            most_common = cell_counter.most_common(1)[0][0]
            result[driver_idx] = most_common
        else:
            result[driver_idx] = (0, 0)

    return result


def _compute_trajectory_metrics(
    trajs: Dict[int, List],
) -> Dict[int, Tuple[float, float]]:
    """
    Compute (avg_distance, avg_time) per driver from trajectories.
    Distance = mean Euclidean grid-cell hops per trajectory.
    Time = mean time-bucket span per trajectory.
    """
    result = {}
    for driver_idx, traj_list in trajs.items():
        if not traj_list:
            result[driver_idx] = (0.0, 0.0)
            continue
        dists = [_trajectory_distance(t) for t in traj_list]
        times = [_trajectory_time_span(t) for t in traj_list]
        result[driver_idx] = (np.mean(dists), np.mean(times))
    return result


def _compute_trips_per_day(
    driving_trajs: Dict[int, List],
    calendar_days_per_driver: Dict[int, int],
) -> Dict[int, float]:
    """
    Average number of driving trips per working day.
    Uses actual unique calendar dates (not day-of-week indices) for division.
    """
    result = {}
    for driver_idx, traj_list in driving_trajs.items():
        if not traj_list:
            result[driver_idx] = 0.0
            continue
        n_days = max(calendar_days_per_driver.get(driver_idx, 1), 1)
        result[driver_idx] = len(traj_list) / n_days
    return result


def _z_score_normalize(
    features: Dict[int, np.ndarray],
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """
    Z-score normalize across all drivers.
    Returns (normalized_features, mean_vector, std_vector).
    std clamped to minimum 1e-8 to avoid division by zero.
    """
    if not features:
        return {}, np.zeros(N_FEATURES), np.ones(N_FEATURES)

    all_vecs = np.array([features[k] for k in sorted(features.keys())])
    mean_vec = all_vecs.mean(axis=0)
    std_vec = all_vecs.std(axis=0)
    std_vec = np.maximum(std_vec, 1e-8)

    normalized = {}
    for driver_idx, vec in features.items():
        normalized[driver_idx] = (vec - mean_vec) / std_vec

    return normalized, mean_vec, std_vec


def compute_profile_features(
    seeking_trajs: Dict[int, List],
    driving_trajs: Dict[int, List],
    index_to_plate: Dict[int, str],
    calendar_days_per_driver: Dict[int, int],
    config: ExtractionConfig,
    bounds: GlobalBounds,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict:
    """
    Compute all 11 profile features for each driver and z-score normalize.

    Args:
        seeking_trajs: {driver_index: [trajectories]} from extraction
        driving_trajs: {driver_index: [trajectories]} from extraction
        index_to_plate: {driver_index: plate_id} mapping
        config: ExtractionConfig with paths to pickle files
        bounds: GlobalBounds for GPS-to-grid conversion
        progress_callback: Optional progress callback

    Returns:
        Dict with keys: features, features_normalized, feature_names,
        normalization, n_features, method, home_loc_coverage
    """
    if progress_callback:
        progress_callback("Computing profile features", 0.0)

    # --- Features 0-1: Home location ---
    home_locs = _load_home_locations(config, index_to_plate, bounds)
    covered = set(home_locs.keys())
    all_drivers = set(index_to_plate.keys())
    missing = all_drivers - covered
    if missing:
        print(f"  Home location: {len(covered)} from pickle, "
              f"{len(missing)} computed from seeking trajectories")
        fallback = _compute_home_fallback(seeking_trajs, missing)
        home_locs.update(fallback)
    else:
        print(f"  Home location: all {len(covered)} from pickle")

    if progress_callback:
        progress_callback("Computing profile features", 0.2)

    # --- Features 2-3: Shift timing ---
    shift_times = _load_shift_times(config, index_to_plate)
    print(f"  Shift times: {len(shift_times)}/{len(index_to_plate)} drivers from pickle")

    if progress_callback:
        progress_callback("Computing profile features", 0.3)

    # --- Features 4-5: Most frequently visited grid ---
    freq_grids = _compute_freq_grid(seeking_trajs, driving_trajs)

    if progress_callback:
        progress_callback("Computing profile features", 0.5)

    # --- Features 6-7: Avg seeking distance/time ---
    seeking_metrics = _compute_trajectory_metrics(seeking_trajs)

    if progress_callback:
        progress_callback("Computing profile features", 0.7)

    # --- Features 8-9: Avg driving distance/time ---
    driving_metrics = _compute_trajectory_metrics(driving_trajs)

    # --- Feature 10: Trips per day ---
    trips_per_day = _compute_trips_per_day(driving_trajs, calendar_days_per_driver)

    if progress_callback:
        progress_callback("Computing profile features", 0.9)

    # --- Assemble 11-feature vectors ---
    features: Dict[int, np.ndarray] = {}
    for driver_idx in sorted(index_to_plate.keys()):
        vec = np.zeros(N_FEATURES, dtype=np.float64)

        hx, hy = home_locs.get(driver_idx, (0, 0))
        vec[0], vec[1] = hx, hy

        st, se = shift_times.get(driver_idx, (0.0, 0.0))
        vec[2], vec[3] = st, se

        fx, fy = freq_grids.get(driver_idx, (0, 0))
        vec[4], vec[5] = fx, fy

        sd, st2 = seeking_metrics.get(driver_idx, (0.0, 0.0))
        vec[6], vec[7] = sd, st2

        dd, dt = driving_metrics.get(driver_idx, (0.0, 0.0))
        vec[8], vec[9] = dd, dt

        vec[10] = trips_per_day.get(driver_idx, 0.0)

        features[driver_idx] = vec

    # --- Z-score normalize ---
    normalized, mean_vec, std_vec = _z_score_normalize(features)

    if progress_callback:
        progress_callback("Computing profile features", 1.0)

    return {
        "features": features,
        "features_normalized": normalized,
        "feature_names": FEATURE_NAMES,
        "normalization": {"mean": mean_vec, "std": std_vec},
        "n_features": N_FEATURES,
        "method": "z-score",
        "home_loc_coverage": {
            "from_pickle": len(covered),
            "from_fallback": len(missing),
            "total": len(home_locs),
        },
    }
