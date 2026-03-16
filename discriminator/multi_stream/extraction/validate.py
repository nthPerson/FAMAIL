"""
Post-extraction validation and summary statistics.

Runs structural checks on extracted trajectories and profile features,
and cross-validates against trip_info_dict_789.pkl.

Usage:
    python -m discriminator.multi_stream.extraction.validate
    python -m discriminator.multi_stream.extraction.validate --data-dir path/to/extracted_data
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))


def validate_trajectories(
    seeking: Dict[int, List],
    driving: Dict[int, List],
    n_expected_drivers: int = 50,
) -> Tuple[bool, List[str]]:
    """
    Structural validation of extracted trajectories.

    Returns (all_passed, list_of_issues).
    """
    issues = []

    # Check driver counts
    seek_drivers = set(seeking.keys())
    drive_drivers = set(driving.keys())
    all_drivers = seek_drivers | drive_drivers

    if len(all_drivers) != n_expected_drivers:
        issues.append(f"Expected {n_expected_drivers} drivers, found {len(all_drivers)}")

    # Check for missing drivers in either dict
    seek_only = seek_drivers - drive_drivers
    drive_only = drive_drivers - seek_drivers
    if seek_only:
        issues.append(f"Drivers in seeking but not driving: {sorted(seek_only)}")
    if drive_only:
        issues.append(f"Drivers in driving but not seeking: {sorted(drive_only)}")

    # Check trajectory structure and value ranges
    n_dedup_violations = 0
    n_range_violations = 0
    n_empty_trajs = 0

    for label, trajs_dict in [("seeking", seeking), ("driving", driving)]:
        for driver_idx, traj_list in trajs_dict.items():
            if not traj_list:
                n_empty_trajs += 1
                continue
            for traj_i, traj in enumerate(traj_list):
                if not traj:
                    issues.append(f"{label} driver {driver_idx} traj {traj_i}: empty")
                    continue

                for state_i, state in enumerate(traj):
                    # Check 4-element structure
                    if len(state) != 4:
                        issues.append(
                            f"{label} driver {driver_idx} traj {traj_i} "
                            f"state {state_i}: expected 4 elements, got {len(state)}"
                        )
                        continue

                    x, y, t, d = state

                    # Value range checks (1-indexed grid)
                    if not (1 <= x <= 48):
                        n_range_violations += 1
                    if not (1 <= y <= 90):
                        n_range_violations += 1
                    if not (0 <= t <= 287):
                        n_range_violations += 1
                    if not (1 <= d <= 5):
                        n_range_violations += 1

                # Check deduplication: no consecutive identical (x, y, t)
                for j in range(1, len(traj)):
                    if (traj[j][0] == traj[j-1][0]
                            and traj[j][1] == traj[j-1][1]
                            and traj[j][2] == traj[j-1][2]):
                        n_dedup_violations += 1

    if n_range_violations > 0:
        issues.append(f"Value range violations: {n_range_violations}")
    if n_dedup_violations > 0:
        issues.append(f"Deduplication violations (consecutive identical x,y,t): {n_dedup_violations}")
    if n_empty_trajs > 0:
        issues.append(f"Drivers with empty trajectory lists: {n_empty_trajs}")

    return len(issues) == 0, issues


def validate_profile_features(
    profile_data: Dict,
    n_expected_drivers: int = 50,
) -> Tuple[bool, List[str]]:
    """
    Validate profile feature structure and normalization.

    Returns (all_passed, list_of_issues).
    """
    issues = []

    features = profile_data.get("features", {})
    normalized = profile_data.get("features_normalized", {})
    n_features = profile_data.get("n_features", 0)
    feature_names = profile_data.get("feature_names", [])
    normalization = profile_data.get("normalization", {})

    # Check driver count
    if len(features) != n_expected_drivers:
        issues.append(f"Expected {n_expected_drivers} drivers, found {len(features)}")

    # Check feature dimensions
    if n_features != 11:
        issues.append(f"Expected 11 features, got {n_features}")
    if len(feature_names) != 11:
        issues.append(f"Expected 11 feature names, got {len(feature_names)}")

    # Check normalization params
    mean = normalization.get("mean", np.array([]))
    std = normalization.get("std", np.array([]))
    if len(mean) != 11:
        issues.append(f"Normalization mean has {len(mean)} elements, expected 11")
    if len(std) != 11:
        issues.append(f"Normalization std has {len(std)} elements, expected 11")

    # Check for zero-std features
    for i, s in enumerate(std):
        if s < 1e-7:
            issues.append(f"Feature {i} ({feature_names[i]}): std ≈ 0 (constant feature)")

    # Check individual feature vectors
    for driver_idx, vec in features.items():
        if len(vec) != 11:
            issues.append(f"Driver {driver_idx}: feature vector has {len(vec)} elements")
        if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
            issues.append(f"Driver {driver_idx}: NaN or Inf in raw features")

    # Check normalized features
    if normalized:
        all_norm = np.array([normalized[k] for k in sorted(normalized.keys())])
        norm_mean = all_norm.mean(axis=0)
        norm_std = all_norm.std(axis=0)
        for i in range(min(11, len(norm_mean))):
            if abs(norm_mean[i]) > 0.1:
                issues.append(
                    f"Normalized feature {i} ({feature_names[i]}): "
                    f"mean={norm_mean[i]:.4f} (expected ~0)"
                )
            if abs(norm_std[i] - 1.0) > 0.1 and std[i] > 1e-7:
                issues.append(
                    f"Normalized feature {i} ({feature_names[i]}): "
                    f"std={norm_std[i]:.4f} (expected ~1)"
                )

    return len(issues) == 0, issues


def cross_validate_with_trip_info(
    driving_trajs: Dict[int, List],
    index_to_plate: Dict[int, str],
    feature_data_dir: Path,
) -> None:
    """
    Cross-validate engineered driving metrics against trip_info_dict_789.pkl.
    Phase 1 found v1=distance (r=1.0), v2=time (r=1.0).
    """
    trip_info_path = feature_data_dir / "trip_info_dict_789.pkl"
    if not trip_info_path.exists():
        print("  trip_info_dict_789.pkl not found, skipping cross-validation")
        return

    with open(trip_info_path, "rb") as f:
        trip_info = pickle.load(f)

    plate_to_index = {p: i for i, p in index_to_plate.items()}
    from .profile_features import _trajectory_distance, _trajectory_time_span

    # Compute per-driver averages from extracted trajectories
    our_dists = {}
    our_times = {}
    for driver_idx, traj_list in driving_trajs.items():
        if traj_list:
            dists = [_trajectory_distance(t) for t in traj_list]
            times = [_trajectory_time_span(t) for t in traj_list]
            our_dists[driver_idx] = np.mean(dists)
            our_times[driver_idx] = np.mean(times)

    # Compute per-driver averages from trip_info (v1=distance, v2=time)
    trip_dists = {}
    trip_times = {}
    for plate_id, months in trip_info.items():
        if plate_id not in plate_to_index:
            continue
        idx = plate_to_index[plate_id]
        v1s, v2s = [], []
        for month in ["07", "08", "09"]:
            if month in months:
                v1s.append(months[month][0])  # distance
                v2s.append(months[month][1])  # time
        if v1s:
            trip_dists[idx] = np.mean(v1s)
            trip_times[idx] = np.mean(v2s)

    # Compute correlations for drivers in both sets
    common = sorted(set(our_dists.keys()) & set(trip_dists.keys()))
    if len(common) < 3:
        print(f"  Only {len(common)} common drivers, too few for correlation")
        return

    from scipy.stats import pearsonr

    our_d = np.array([our_dists[i] for i in common])
    trip_d = np.array([trip_dists[i] for i in common])
    our_t = np.array([our_times[i] for i in common])
    trip_t = np.array([trip_times[i] for i in common])

    r_dist, p_dist = pearsonr(our_d, trip_d)
    r_time, p_time = pearsonr(our_t, trip_t)

    print(f"  Cross-validation with trip_info_dict_789.pkl ({len(common)} drivers):")
    print(f"    avg_driving_dist vs trip_info v1: r={r_dist:.4f} (p={p_dist:.2e})")
    print(f"    avg_driving_time vs trip_info v2: r={r_time:.4f} (p={p_time:.2e})")

    if r_dist < 0.3:
        print("    ⚠ Low distance correlation — review extraction logic")
    if r_time < 0.3:
        print("    ⚠ Low time correlation — review extraction logic")


def main():
    parser = argparse.ArgumentParser(description="Validate extracted trajectories and features")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).resolve().parents[1] / "extracted_data"

    print(f"Validating data in: {data_dir}")
    print("=" * 65)

    # Load data
    seeking_path = data_dir / "seeking_trajs.pkl"
    driving_path = data_dir / "driving_trajs.pkl"
    profile_path = data_dir / "profile_features.pkl"
    metadata_path = data_dir / "extraction_metadata.json"

    if not seeking_path.exists() or not driving_path.exists():
        print("ERROR: seeking_trajs.pkl or driving_trajs.pkl not found")
        return

    with open(seeking_path, "rb") as f:
        seeking = pickle.load(f)
    with open(driving_path, "rb") as f:
        driving = pickle.load(f)

    # Load metadata for driver mapping
    index_to_plate = {}
    feature_data_dir = None
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        index_to_plate = {int(k): v for k, v in metadata.get("driver_mapping", {}).items()}
        feature_data_dir = Path(metadata.get("config", {}).get("feature_data_dir", ""))

    # 1. Trajectory validation
    print("\n1. Trajectory Structural Validation")
    print("─" * 40)
    passed, issues = validate_trajectories(seeking, driving)
    if passed:
        print("  ✓ All structural checks passed")
    else:
        for issue in issues:
            print(f"  ✗ {issue}")

    # Summary stats
    total_seek = sum(len(t) for t in seeking.values())
    total_drive = sum(len(t) for t in driving.values())
    print(f"\n  Total: {total_seek} seeking, {total_drive} driving trajectories")
    print(f"  Drivers with seeking: {sum(1 for t in seeking.values() if t)}")
    print(f"  Drivers with driving: {sum(1 for t in driving.values() if t)}")

    # 2. Profile feature validation
    if profile_path.exists():
        print("\n2. Profile Feature Validation")
        print("─" * 40)
        with open(profile_path, "rb") as f:
            profile_data = pickle.load(f)
        passed, issues = validate_profile_features(profile_data)
        if passed:
            print("  ✓ All profile feature checks passed")
        else:
            for issue in issues:
                print(f"  ✗ {issue}")
    else:
        print("\n2. Profile features not found, skipping")

    # 3. Cross-validation
    if feature_data_dir and feature_data_dir.exists() and index_to_plate:
        print("\n3. Cross-Validation")
        print("─" * 40)
        cross_validate_with_trip_info(driving, index_to_plate, feature_data_dir)
    else:
        print("\n3. Cross-validation skipped (missing metadata or feature dir)")

    print(f"\n{'=' * 65}")
    print("Validation complete.")


if __name__ == "__main__":
    main()
