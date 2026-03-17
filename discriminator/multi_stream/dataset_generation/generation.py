"""Core generation logic for multi-stream trajectory pair datasets.

Implements Ren et al. (KDD 2020) day-based pair sampling:
- Positive: same driver, 2 different calendar days
- Negative: 2 different drivers, 1 day each
- Identical: same driver, same day, same trajectories (both branches)

Each branch produces N independent trajectories per stream (seeking + driving)
plus a profile feature vector.
"""

import json
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import MultiStreamGenerationConfig


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_multi_stream_data(
    extracted_data_dir: Path,
) -> Tuple[
    Dict[int, Dict[int, List]],  # seeking_by_day
    Dict[int, Dict[int, List]],  # driving_by_day
    Dict[int, np.ndarray],       # profile_features
    Dict[int, str],              # cal_day_map
    Dict[int, str],              # index_to_plate
]:
    """Load Phase 2 outputs and build day-indexed structures.

    Returns:
        seeking_by_day:  {driver_idx: {cal_day_idx: [traj, traj, ...]}}
        driving_by_day:  {driver_idx: {cal_day_idx: [traj, traj, ...]}}
        profile_features: {driver_idx: np.ndarray of shape (n_features,)}
        cal_day_map:     {cal_day_idx: "YYYY-MM-DD"}
        index_to_plate:  {driver_idx: plate_id}
    """
    d = Path(extracted_data_dir)

    with open(d / "seeking_trajs.pkl", "rb") as f:
        seeking_trajs = pickle.load(f)  # {driver_idx: [traj, ...]}
    with open(d / "driving_trajs.pkl", "rb") as f:
        driving_trajs = pickle.load(f)
    with open(d / "seeking_calendar_days.pkl", "rb") as f:
        seeking_cal = pickle.load(f)    # {driver_idx: [cal_day_idx, ...]}
    with open(d / "driving_calendar_days.pkl", "rb") as f:
        driving_cal = pickle.load(f)
    with open(d / "calendar_day_map.pkl", "rb") as f:
        cal_day_map = pickle.load(f)    # {cal_day_idx: "YYYY-MM-DD"}
    with open(d / "profile_features.pkl", "rb") as f:
        profile_data = pickle.load(f)   # dict with 'features', 'normalization', etc.

    # Load metadata for driver mapping
    with open(d / "extraction_metadata.json", "r") as f:
        metadata = json.load(f)
    index_to_plate = {int(k): v for k, v in metadata["driver_mapping"].items()}

    # Group trajectories by (driver, calendar_day)
    seeking_by_day: Dict[int, Dict[int, List]] = defaultdict(lambda: defaultdict(list))
    for driver_idx, trajs in seeking_trajs.items():
        days = seeking_cal[driver_idx]
        for traj, day_idx in zip(trajs, days):
            seeking_by_day[driver_idx][day_idx].append(traj)

    driving_by_day: Dict[int, Dict[int, List]] = defaultdict(lambda: defaultdict(list))
    for driver_idx, trajs in driving_trajs.items():
        days = driving_cal[driver_idx]
        for traj, day_idx in zip(trajs, days):
            driving_by_day[driver_idx][day_idx].append(traj)

    # Extract normalized profile features
    profile_features = {}
    norm = profile_data["normalization"]
    mean, std = norm["mean"], norm["std"]
    for driver_idx, feat_vec in profile_data["features"].items():
        # Z-score normalize using pre-computed parameters
        profile_features[driver_idx] = (np.array(feat_vec) - mean) / (std + 1e-8)

    # Convert defaultdicts to regular dicts
    seeking_by_day = {k: dict(v) for k, v in seeking_by_day.items()}
    driving_by_day = {k: dict(v) for k, v in driving_by_day.items()}

    return seeking_by_day, driving_by_day, profile_features, cal_day_map, index_to_plate


# ---------------------------------------------------------------------------
# Usable day filtering
# ---------------------------------------------------------------------------

def find_usable_days(
    seeking_by_day: Dict[int, Dict[int, List]],
    driving_by_day: Dict[int, Dict[int, List]],
    config: MultiStreamGenerationConfig,
) -> Dict[int, List[int]]:
    """Find (driver, day) combos with enough trajectories in BOTH streams.

    A day is usable if the driver has >= min_trajs_per_day seeking AND
    >= min_trajs_per_day driving trajectories on that calendar day.

    Returns:
        {driver_idx: [usable_cal_day_idx, ...]}  (sorted)
    """
    usable: Dict[int, List[int]] = {}
    all_drivers = set(seeking_by_day.keys()) | set(driving_by_day.keys())

    for driver in sorted(all_drivers):
        seek_days = seeking_by_day.get(driver, {})
        drive_days = driving_by_day.get(driver, {})
        common_days = set(seek_days.keys()) & set(drive_days.keys())

        days_ok = []
        for day in sorted(common_days):
            if (len(seek_days[day]) >= config.min_trajs_per_day and
                    len(drive_days[day]) >= config.min_trajs_per_day):
                days_ok.append(day)

        if days_ok:
            usable[driver] = days_ok

    return usable


# ---------------------------------------------------------------------------
# Pair sampling
# ---------------------------------------------------------------------------

def _sample_trajs_for_day(
    trajs: List,
    n: int,
    rng: random.Random,
) -> List:
    """Sample n trajectories from a list, with replacement if needed."""
    if len(trajs) >= n:
        return rng.sample(trajs, n)
    # With replacement when fewer than n available
    return [rng.choice(trajs) for _ in range(n)]


def _pad_trajectory(traj: List[List[int]], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pad a single trajectory to max_len, return (data, mask)."""
    arr = np.array(traj, dtype=np.float32)  # [L, 4]
    L = arr.shape[0]
    if L >= max_len:
        return arr[:max_len], np.ones(max_len, dtype=bool)
    padded = np.zeros((max_len, 4), dtype=np.float32)
    padded[:L] = arr
    mask = np.zeros(max_len, dtype=bool)
    mask[:L] = True
    return padded, mask


def _assemble_branch_trajs(
    trajs: List,
    n_trajs: int,
    fixed_length: Optional[int],
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble N trajectories into padded arrays for one branch of one stream.

    Returns:
        data: [N, L_max, 4]
        mask: [N, L_max]
    """
    sampled = _sample_trajs_for_day(trajs, n_trajs, rng)

    if fixed_length is not None:
        L_max = fixed_length
    else:
        L_max = max(len(t) for t in sampled)

    data = np.zeros((n_trajs, L_max, 4), dtype=np.float32)
    mask = np.zeros((n_trajs, L_max), dtype=bool)

    for i, traj in enumerate(sampled):
        d, m = _pad_trajectory(traj, L_max)
        data[i] = d
        mask[i] = m

    return data, mask


def sample_positive_pairs(
    seeking_by_day: Dict[int, Dict[int, List]],
    driving_by_day: Dict[int, Dict[int, List]],
    profile_features: Dict[int, np.ndarray],
    usable_days: Dict[int, List[int]],
    config: MultiStreamGenerationConfig,
    rng: random.Random,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> List[Dict[str, Any]]:
    """Sample positive pairs (same driver, 2 different days).

    Returns list of pair dicts ready for assembly.
    """
    n_identical = int(config.positive_pairs * config.identical_pair_ratio)
    n_different_day = config.positive_pairs - n_identical

    # Drivers with >= 2 usable days (needed for different-day positives)
    multi_day_drivers = [d for d, days in usable_days.items() if len(days) >= 2]
    all_drivers = list(usable_days.keys())

    pairs = []

    # Coverage pass: one pair per multi-day driver
    if config.ensure_agent_coverage and multi_day_drivers:
        for driver in multi_day_drivers:
            if len(pairs) >= n_different_day:
                break
            day_a, day_b = rng.sample(usable_days[driver], 2)
            pairs.append(_build_pair(
                driver, day_a, driver, day_b,
                seeking_by_day, driving_by_day, profile_features,
                config, rng, label=1
            ))

    # Fill remaining different-day positives
    attempts = 0
    max_attempts = n_different_day * 20
    while len(pairs) < n_different_day and attempts < max_attempts:
        driver = rng.choice(multi_day_drivers) if multi_day_drivers else rng.choice(all_drivers)
        days = usable_days.get(driver, [])
        if len(days) < 2:
            attempts += 1
            continue
        day_a, day_b = rng.sample(days, 2)
        pairs.append(_build_pair(
            driver, day_a, driver, day_b,
            seeking_by_day, driving_by_day, profile_features,
            config, rng, label=1
        ))
        attempts += 1
        if progress_callback and len(pairs) % 500 == 0:
            progress_callback("Positive pairs", len(pairs) / config.positive_pairs)

    # Identical pairs (same driver, same day, same trajectories)
    for _ in range(n_identical):
        driver = rng.choice(all_drivers)
        day = rng.choice(usable_days[driver])
        pairs.append(_build_identical_pair(
            driver, day, seeking_by_day, driving_by_day,
            profile_features, config, rng
        ))

    if progress_callback:
        progress_callback("Positive pairs", 1.0)

    return pairs


def sample_negative_pairs(
    seeking_by_day: Dict[int, Dict[int, List]],
    driving_by_day: Dict[int, Dict[int, List]],
    profile_features: Dict[int, np.ndarray],
    usable_days: Dict[int, List[int]],
    config: MultiStreamGenerationConfig,
    rng: random.Random,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> List[Dict[str, Any]]:
    """Sample negative pairs (2 different drivers, 1 day each)."""
    all_drivers = list(usable_days.keys())
    if len(all_drivers) < 2:
        return []

    pairs = []

    # Coverage pass: each driver paired with a random other
    if config.ensure_agent_coverage:
        for driver_a in all_drivers:
            if len(pairs) >= config.negative_pairs:
                break
            driver_b = driver_a
            while driver_b == driver_a:
                driver_b = rng.choice(all_drivers)
            day_a = rng.choice(usable_days[driver_a])
            day_b = rng.choice(usable_days[driver_b])
            pairs.append(_build_pair(
                driver_a, day_a, driver_b, day_b,
                seeking_by_day, driving_by_day, profile_features,
                config, rng, label=0
            ))

    # Fill remaining
    while len(pairs) < config.negative_pairs:
        driver_a, driver_b = rng.sample(all_drivers, 2)
        day_a = rng.choice(usable_days[driver_a])
        day_b = rng.choice(usable_days[driver_b])
        pairs.append(_build_pair(
            driver_a, day_a, driver_b, day_b,
            seeking_by_day, driving_by_day, profile_features,
            config, rng, label=0
        ))
        if progress_callback and len(pairs) % 500 == 0:
            progress_callback("Negative pairs", len(pairs) / config.negative_pairs)

    if progress_callback:
        progress_callback("Negative pairs", 1.0)

    return pairs[:config.negative_pairs]


# ---------------------------------------------------------------------------
# Pair construction helpers
# ---------------------------------------------------------------------------

def _build_pair(
    driver_a: int, day_a: int,
    driver_b: int, day_b: int,
    seeking_by_day: Dict[int, Dict[int, List]],
    driving_by_day: Dict[int, Dict[int, List]],
    profile_features: Dict[int, np.ndarray],
    config: MultiStreamGenerationConfig,
    rng: random.Random,
    label: int,
) -> Dict[str, Any]:
    """Build one pair dict with N trajs per stream per branch."""
    N = config.n_trajs_per_stream

    # Branch 1: driver_a on day_a
    seek_a, mask_sa = _assemble_branch_trajs(
        seeking_by_day[driver_a][day_a], N,
        config.seeking_fixed_length, rng
    )
    drive_a, mask_da = _assemble_branch_trajs(
        driving_by_day[driver_a][day_a], N,
        config.driving_fixed_length, rng
    )
    prof_a = profile_features[driver_a].copy()

    # Branch 2: driver_b on day_b
    seek_b, mask_sb = _assemble_branch_trajs(
        seeking_by_day[driver_b][day_b], N,
        config.seeking_fixed_length, rng
    )
    drive_b, mask_db = _assemble_branch_trajs(
        driving_by_day[driver_b][day_b], N,
        config.driving_fixed_length, rng
    )
    prof_b = profile_features[driver_b].copy()

    # Add optional profile noise
    if config.profile_noise_std > 0:
        prof_a = prof_a + np.random.randn(*prof_a.shape).astype(np.float32) * config.profile_noise_std
        prof_b = prof_b + np.random.randn(*prof_b.shape).astype(np.float32) * config.profile_noise_std

    return {
        "x1": seek_a, "x2": seek_b,
        "mask1": mask_sa, "mask2": mask_sb,
        "driving_1": drive_a, "driving_2": drive_b,
        "mask_d1": mask_da, "mask_d2": mask_db,
        "profile_1": prof_a.astype(np.float32),
        "profile_2": prof_b.astype(np.float32),
        "label": float(label),
        "driver_a": driver_a, "day_a": day_a,
        "driver_b": driver_b, "day_b": day_b,
    }


def _build_identical_pair(
    driver: int, day: int,
    seeking_by_day: Dict[int, Dict[int, List]],
    driving_by_day: Dict[int, Dict[int, List]],
    profile_features: Dict[int, np.ndarray],
    config: MultiStreamGenerationConfig,
    rng: random.Random,
) -> Dict[str, Any]:
    """Build an identical pair: same driver, same day, same trajectories."""
    N = config.n_trajs_per_stream

    seek, mask_s = _assemble_branch_trajs(
        seeking_by_day[driver][day], N,
        config.seeking_fixed_length, rng
    )
    drive, mask_d = _assemble_branch_trajs(
        driving_by_day[driver][day], N,
        config.driving_fixed_length, rng
    )
    prof = profile_features[driver].copy().astype(np.float32)

    return {
        "x1": seek.copy(), "x2": seek.copy(),
        "mask1": mask_s.copy(), "mask2": mask_s.copy(),
        "driving_1": drive.copy(), "driving_2": drive.copy(),
        "mask_d1": mask_d.copy(), "mask_d2": mask_d.copy(),
        "profile_1": prof.copy(), "profile_2": prof.copy(),
        "label": 1.0,
        "driver_a": driver, "day_a": day,
        "driver_b": driver, "day_b": day,
    }


# ---------------------------------------------------------------------------
# Global padding alignment
# ---------------------------------------------------------------------------

def _global_pad_pairs(
    pairs: List[Dict[str, Any]],
    config: MultiStreamGenerationConfig,
) -> None:
    """Pad all pairs to uniform seeking/driving lengths (in-place).

    After per-pair assembly, trajectories within a pair share the same L
    but different pairs may have different L. This function pads all pairs
    to the global maximum across the dataset.
    """
    if not pairs:
        return

    if config.seeking_padding == "fixed_length" and config.seeking_fixed_length:
        L_s = config.seeking_fixed_length
    else:
        L_s = max(p["x1"].shape[1] for p in pairs)

    if config.driving_padding == "fixed_length" and config.driving_fixed_length:
        L_d = config.driving_fixed_length
    else:
        L_d = max(p["driving_1"].shape[1] for p in pairs)

    N = config.n_trajs_per_stream

    for p in pairs:
        for key, mask_key, target_L in [
            ("x1", "mask1", L_s), ("x2", "mask2", L_s),
            ("driving_1", "mask_d1", L_d), ("driving_2", "mask_d2", L_d),
        ]:
            cur_L = p[key].shape[1]
            if cur_L < target_L:
                pad_L = target_L - cur_L
                p[key] = np.pad(p[key], ((0, 0), (0, pad_L), (0, 0)),
                                mode="constant", constant_values=0)
                p[mask_key] = np.pad(p[mask_key], ((0, 0), (0, pad_L)),
                                     mode="constant", constant_values=False)
            elif cur_L > target_L:
                p[key] = p[key][:, :target_L, :]
                p[mask_key] = p[mask_key][:, :target_L]


# ---------------------------------------------------------------------------
# Assembly and save
# ---------------------------------------------------------------------------

def _pairs_to_arrays(
    pairs: List[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """Stack list of pair dicts into contiguous arrays."""
    return {
        "x1": np.stack([p["x1"] for p in pairs]),
        "x2": np.stack([p["x2"] for p in pairs]),
        "mask1": np.stack([p["mask1"] for p in pairs]),
        "mask2": np.stack([p["mask2"] for p in pairs]),
        "driving_1": np.stack([p["driving_1"] for p in pairs]),
        "driving_2": np.stack([p["driving_2"] for p in pairs]),
        "mask_d1": np.stack([p["mask_d1"] for p in pairs]),
        "mask_d2": np.stack([p["mask_d2"] for p in pairs]),
        "profile_1": np.stack([p["profile_1"] for p in pairs]),
        "profile_2": np.stack([p["profile_2"] for p in pairs]),
        "label": np.array([p["label"] for p in pairs], dtype=np.float32),
    }


def _split_and_save(
    pairs: List[Dict[str, Any]],
    config: MultiStreamGenerationConfig,
    rng: random.Random,
) -> Dict[str, int]:
    """Shuffle, split, pad globally, and save train/val/test .npz files."""
    rng.shuffle(pairs)

    n = len(pairs)
    n_test = int(n * config.test_ratio)
    n_val = int(n * config.val_ratio)
    n_train = n - n_val - n_test

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    counts = {}
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_pairs in splits.items():
        if not split_pairs:
            counts[split_name] = 0
            continue
        _global_pad_pairs(split_pairs, config)
        arrays = _pairs_to_arrays(split_pairs)
        np.savez_compressed(config.output_dir / f"{split_name}.npz", **arrays)
        counts[split_name] = len(split_pairs)

    return counts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_multi_stream_dataset(
    config: MultiStreamGenerationConfig,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Generate a complete multi-stream dataset.

    Returns metadata dict with generation stats.
    """
    t_start = time.time()
    rng = random.Random(config.seed)

    # 1. Load data
    if progress_callback:
        progress_callback("Loading data", 0.0)

    seeking_by_day, driving_by_day, profile_features, cal_day_map, index_to_plate = \
        load_multi_stream_data(config.extracted_data_dir)

    if progress_callback:
        progress_callback("Loading data", 1.0)

    # 2. Find usable days
    usable_days = find_usable_days(seeking_by_day, driving_by_day, config)

    usable_summary = {
        "total_drivers_with_usable_days": len(usable_days),
        "total_usable_driver_days": sum(len(d) for d in usable_days.values()),
        "drivers_with_2plus_days": sum(1 for d in usable_days.values() if len(d) >= 2),
        "per_driver": {str(k): len(v) for k, v in usable_days.items()},
    }

    # 3. Sample pairs
    positive_pairs = sample_positive_pairs(
        seeking_by_day, driving_by_day, profile_features,
        usable_days, config, rng, progress_callback
    )

    negative_pairs = sample_negative_pairs(
        seeking_by_day, driving_by_day, profile_features,
        usable_days, config, rng, progress_callback
    )

    all_pairs = positive_pairs + negative_pairs

    # 4. Compute agent coverage stats
    pos_agents = set()
    neg_agents = set()
    for p in positive_pairs:
        pos_agents.add(p["driver_a"])
    for p in negative_pairs:
        neg_agents.add(p["driver_a"])
        neg_agents.add(p["driver_b"])

    # 5. Split, pad, and save
    if progress_callback:
        progress_callback("Saving splits", 0.5)
    split_counts = _split_and_save(all_pairs, config, rng)
    if progress_callback:
        progress_callback("Saving splits", 1.0)

    # 6. Compute shape info from a sample pair
    shape_info = {}
    if all_pairs:
        sample = all_pairs[0]
        shape_info = {
            "n_trajs_per_stream": config.n_trajs_per_stream,
            "seeking_length": int(sample["x1"].shape[1]),
            "driving_length": int(sample["driving_1"].shape[1]),
            "profile_dim": int(sample["profile_1"].shape[0]),
        }

    elapsed = time.time() - t_start

    # 7. Build and save metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config.to_dict(),
        "generation_time_seconds": elapsed,
        "pair_counts": {
            "positive": len(positive_pairs),
            "negative": len(negative_pairs),
            "identical": int(config.positive_pairs * config.identical_pair_ratio),
            "total": len(all_pairs),
        },
        "split_counts": split_counts,
        "agent_coverage": {
            "positive_agents": len(pos_agents),
            "negative_agents": len(neg_agents),
            "total_available": len(usable_days),
        },
        "usable_days": usable_summary,
        "shape_info": shape_info,
    }

    with open(config.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
