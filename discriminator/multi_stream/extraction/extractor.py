"""
Dual trajectory extraction: seeking (passenger=0) and driving (passenger=1).

Processes raw GPS records in a single pass per driver, detecting passenger
indicator transitions to segment trajectories. Applies deduplication of
consecutive identical quantized states and minimum segment filtering to
address noisy micro-transitions.
"""

import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Add project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from new_all_trajs.config import GlobalBounds
from new_all_trajs.step1_processor import (
    compute_global_bounds,
    load_raw_data,
    timestamp_to_day,
)

from .config import ExtractionConfig, ExtractionStats


def _precompute_bins(bounds: GlobalBounds, grid_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute lat/lon bin edges for fast grid quantization."""
    lat_bins = np.arange(bounds.lat_min, bounds.lat_max + grid_size, grid_size)
    lon_bins = np.arange(bounds.lon_min, bounds.lon_max + grid_size, grid_size)
    return lat_bins, lon_bins


def _gps_to_grid_fast(
    lat: float,
    lon: float,
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
) -> Tuple[int, int]:
    """Convert GPS to grid indices using precomputed bins. 0-indexed."""
    x = int(np.searchsorted(lat_bins, lat, side='right') - 1)
    y = int(np.searchsorted(lon_bins, lon, side='right') - 1)
    x = max(0, min(x, len(lat_bins) - 2))
    y = max(0, min(y, len(lon_bins) - 2))
    return x, y


def _is_within_bbox(lat: float, lon: float, config: ExtractionConfig) -> bool:
    """Check if GPS coordinate falls within Shenzhen bounding box."""
    return (config.lat_min_bound <= lat <= config.lat_max_bound
            and config.lon_min_bound <= lon <= config.lon_max_bound)


def _deduplicate_states(states: List[List[int]]) -> List[List[int]]:
    """
    Collapse consecutive states with identical (x, y, t).
    Keeps the first occurrence of each unique consecutive state.
    """
    if len(states) <= 1:
        return states
    result = [states[0]]
    for s in states[1:]:
        prev = result[-1]
        if s[0] != prev[0] or s[1] != prev[1] or s[2] != prev[2]:
            result.append(s)
    return result


def _finalize_segment(
    states: List[List[int]],
    raw_records: List,
    output_list: List,
    config: ExtractionConfig,
    stats: ExtractionStats,
    seg_type: str,
) -> None:
    """
    Deduplicate, filter, and append a completed segment to the output list.

    Pipeline: dedup → min states check → min duration check → max length truncate → append.
    """
    if not states:
        return

    # Step 1: Deduplicate consecutive identical (x, y, t) states
    original_len = len(states)
    deduped = _deduplicate_states(states)
    stats.states_deduplicated += original_len - len(deduped)

    # Step 2: Check minimum states after deduplication
    if len(deduped) < config.min_segment_states:
        stats.segments_filtered_too_short += 1
        return

    # Step 3: Check minimum duration using raw record timestamps
    if len(raw_records) >= 2:
        duration = abs(float(raw_records[-1][3]) - float(raw_records[0][3]))
        if duration < config.min_segment_duration_sec:
            stats.segments_filtered_too_brief += 1
            return

    # Step 4: Truncate if exceeds max length
    if len(deduped) > config.max_segment_states:
        deduped = deduped[:config.max_segment_states]

    output_list.append(deduped)

    # Update stats
    if seg_type == "seeking":
        stats.seeking_trajectories += 1
        stats.seeking_states_total += len(deduped)
    else:
        stats.driving_trajectories += 1
        stats.driving_states_total += len(deduped)


def extract_dual_trajectories(
    raw_data: Dict[str, List],
    bounds: GlobalBounds,
    config: ExtractionConfig,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, int], ExtractionStats]:
    """
    Single-pass extraction of both seeking and driving trajectories.

    For each driver, processes GPS records chronologically, detecting
    passenger indicator transitions (0→1 = pickup, 1→0 = dropoff) to
    segment trajectories. Day boundaries also finalize open segments.

    Args:
        raw_data: {plate_id: [records]} from load_raw_data(), already flattened
        bounds: GlobalBounds for GPS-to-grid conversion
        config: ExtractionConfig with filtering parameters
        progress_callback: Optional (stage_name, fraction) callback

    Returns:
        (seeking_by_plate, driving_by_plate, calendar_days_per_plate, stats)
        Trajectory dicts: {plate_id: [[[x,y,t,d], ...], ...]}
        calendar_days_per_plate: {plate_id: n_unique_calendar_dates}
    """
    stats = ExtractionStats()

    # Precompute grid bins for performance
    lat_bins, lon_bins = _precompute_bins(bounds, config.grid_size)

    seeking_by_plate: Dict[str, List] = {}
    driving_by_plate: Dict[str, List] = {}
    calendar_days_per_plate: Dict[str, int] = {}

    plates = sorted(raw_data.keys())
    total_plates = len(plates)

    for plate_idx, plate_id in enumerate(plates):
        records = raw_data[plate_id]
        stats.total_raw_records += len(records)

        if progress_callback and plate_idx % 10 == 0:
            progress_callback("Extracting trajectories", plate_idx / total_plates)

        # Sort by timestamp string for chronological order
        try:
            records_sorted = sorted(records, key=lambda r: r[5])
        except (IndexError, TypeError):
            # Fallback: sort by seconds_since_midnight if timestamp unavailable
            records_sorted = sorted(records, key=lambda r: float(r[3]))

        seeking_trajs: List[List[List[int]]] = []
        driving_trajs: List[List[List[int]]] = []
        unique_dates: set = set()  # Track unique calendar dates per driver

        # Accumulators for current segment
        cur_seeking_states: List[List[int]] = []
        cur_seeking_records: List = []
        cur_driving_states: List[List[int]] = []
        cur_driving_records: List = []

        prev_passenger: Optional[int] = None
        prev_day: Optional[int] = None

        for record in records_sorted:
            try:
                lat = float(record[1])
                lon = float(record[2])
                seconds = float(record[3])
                passenger = int(record[4])
                timestamp_str = str(record[5])
            except (IndexError, ValueError, TypeError):
                stats.records_timestamp_invalid += 1
                continue

            # GPS bounding box filter
            if not _is_within_bbox(lat, lon, config):
                stats.records_outside_bbox += 1
                continue

            # Track unique calendar dates (date portion of timestamp)
            unique_dates.add(timestamp_str[:10])

            # Weekend filter
            day = timestamp_to_day(timestamp_str, config.exclude_weekends)
            if day is None:
                stats.records_weekend_filtered += 1
                prev_passenger = passenger
                continue

            # Day boundary detection → finalize any open segments
            if prev_day is not None and day != prev_day:
                _finalize_segment(
                    cur_seeking_states, cur_seeking_records,
                    seeking_trajs, config, stats, "seeking"
                )
                _finalize_segment(
                    cur_driving_states, cur_driving_records,
                    driving_trajs, config, stats, "driving"
                )
                cur_seeking_states = []
                cur_seeking_records = []
                cur_driving_states = []
                cur_driving_records = []
                prev_passenger = None

            # Quantize GPS → grid + offsets
            x, y = _gps_to_grid_fast(lat, lon, lat_bins, lon_bins)
            x += config.x_grid_offset
            y += config.y_grid_offset
            t = int(seconds) // 60 // config.time_interval  # seconds_to_time_bin inline
            state = [x, y, t, day]

            # Transition detection
            if prev_passenger is not None:
                if prev_passenger == 1 and passenger == 0:
                    # Dropoff: finalize driving, start new seeking
                    _finalize_segment(
                        cur_driving_states, cur_driving_records,
                        driving_trajs, config, stats, "driving"
                    )
                    cur_driving_states = []
                    cur_driving_records = []
                    cur_seeking_states = []
                    cur_seeking_records = []
                elif prev_passenger == 0 and passenger == 1:
                    # Pickup: finalize seeking, start new driving
                    _finalize_segment(
                        cur_seeking_states, cur_seeking_records,
                        seeking_trajs, config, stats, "seeking"
                    )
                    cur_seeking_states = []
                    cur_seeking_records = []
                    cur_driving_states = []
                    cur_driving_records = []

            # Append to appropriate accumulator
            if passenger == 0:
                cur_seeking_states.append(state)
                cur_seeking_records.append(record)
            else:
                cur_driving_states.append(state)
                cur_driving_records.append(record)

            prev_passenger = passenger
            prev_day = day

        # Finalize last segments for this driver
        _finalize_segment(
            cur_seeking_states, cur_seeking_records,
            seeking_trajs, config, stats, "seeking"
        )
        _finalize_segment(
            cur_driving_states, cur_driving_records,
            driving_trajs, config, stats, "driving"
        )

        seeking_by_plate[plate_id] = seeking_trajs
        driving_by_plate[plate_id] = driving_trajs
        calendar_days_per_plate[plate_id] = len(unique_dates)
        stats.drivers_processed += 1

    # Compute length distribution stats
    stats.seeking_length_stats = _compute_length_stats(seeking_by_plate)
    stats.driving_length_stats = _compute_length_stats(driving_by_plate)

    if progress_callback:
        progress_callback("Extracting trajectories", 1.0)

    return seeking_by_plate, driving_by_plate, calendar_days_per_plate, stats


def _compute_length_stats(trajs_by_plate: Dict[str, List]) -> Dict[str, float]:
    """Compute min, max, mean, median, p25, p75 of trajectory lengths."""
    lengths = []
    for plate_trajs in trajs_by_plate.values():
        for traj in plate_trajs:
            lengths.append(len(traj))
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "p25": 0, "p75": 0, "count": 0}
    arr = np.array(lengths)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "count": len(lengths),
    }


def _build_driver_index_mapping(
    seeking_by_plate: Dict[str, List],
    driving_by_plate: Dict[str, List],
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Build canonical driver index mapping from plate IDs.
    Uses sorted plate IDs → indices 0-49, matching create_driver_index_mapping().
    """
    all_plates = sorted(set(seeking_by_plate.keys()) | set(driving_by_plate.keys()))
    index_to_plate = {i: p for i, p in enumerate(all_plates)}
    plate_to_index = {p: i for i, p in enumerate(all_plates)}
    return index_to_plate, plate_to_index


def _rekey_by_index(
    trajs_by_plate: Dict[str, List],
    plate_to_index: Dict[str, int],
) -> Dict[int, List]:
    """Re-key trajectory dict from plate_id strings to integer indices."""
    return {plate_to_index[p]: trajs for p, trajs in trajs_by_plate.items()
            if p in plate_to_index}


def run_extraction(
    config: ExtractionConfig,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[Dict[int, List], Dict[int, List], Dict[int, str], Dict[int, int], GlobalBounds, ExtractionStats]:
    """
    Full extraction pipeline: load raw data → compute bounds → extract → index.

    Returns:
        (seeking_indexed, driving_indexed, index_to_plate,
         calendar_days_per_driver, bounds, stats)
    """
    t_start = time.time()

    # 1. Load all raw data files
    raw_data_files = []
    combined_raw: Dict[str, List] = {}
    for filename in config.input_files:
        filepath = config.raw_data_dir / filename
        data, n_records = load_raw_data(filepath)
        raw_data_files.append(data)
        print(f"  Loaded {filename}: {len(data)} drivers, {n_records:,} records")
        # Merge into combined dict (extend records for each plate)
        for plate_id, records in data.items():
            if plate_id in combined_raw:
                combined_raw[plate_id].extend(records)
            else:
                combined_raw[plate_id] = list(records)

    print(f"  Combined: {len(combined_raw)} unique drivers, "
          f"{sum(len(r) for r in combined_raw.values()):,} total records")

    # 2. Compute global bounds from all raw data
    bounds = compute_global_bounds(raw_data_files)
    print(f"  Bounds: lat [{bounds.lat_min:.4f}, {bounds.lat_max:.4f}], "
          f"lon [{bounds.lon_min:.4f}, {bounds.lon_max:.4f}]")

    # 3. Extract dual trajectories
    seeking_by_plate, driving_by_plate, cal_days_plate, stats = extract_dual_trajectories(
        combined_raw, bounds, config, progress_callback
    )

    # 4. Build driver index mapping and re-key
    index_to_plate, plate_to_index = _build_driver_index_mapping(
        seeking_by_plate, driving_by_plate
    )
    seeking_indexed = _rekey_by_index(seeking_by_plate, plate_to_index)
    driving_indexed = _rekey_by_index(driving_by_plate, plate_to_index)

    # Re-key calendar days to driver indices
    calendar_days_indexed = {plate_to_index[p]: d for p, d in cal_days_plate.items()
                             if p in plate_to_index}

    stats.processing_time_seconds = time.time() - t_start

    return seeking_indexed, driving_indexed, index_to_plate, calendar_days_indexed, bounds, stats
