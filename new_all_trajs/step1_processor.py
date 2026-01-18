"""
Step 1: Passenger-Seeking Trajectory Extraction Processor

This module extracts passenger-seeking trajectories from raw taxi GPS data.
A passenger-seeking trajectory begins when the passenger indicator changes 
from 1 to 0 (dropoff) and ends when it changes from 0 to 1 (pickup).

The output is quantized to discrete states using the same quantization logic
as other datasets in the project (pickup_dropoff_counts, active_taxis).
"""

import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict

from config import (
    ProcessingConfig, 
    GlobalBounds, 
    Step1Stats,
)


def load_raw_data(filepath: Path) -> Tuple[Dict[str, List], int]:
    """
    Load raw taxi GPS data from a pickle file.
    
    The raw data structure is:
    {
        plate_id: [record, record, ...]  # Flat list of GPS records for each driver
    }
    
    Each record is a 6-element list:
    [plate_id, latitude, longitude, seconds_since_midnight, passenger_indicator, timestamp_str]
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Tuple of (data dict, total record count)
        
    Raises:
        EOFError: If the file is empty or corrupted
        ValueError: If the data structure is unexpected
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.stat().st_size == 0:
        raise EOFError(f"File is empty: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise EOFError(f"Failed to load pickle file {filepath}: {e}")
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    
    # Flatten nested structure: dict[plate_id] -> list[list[record]] to dict[plate_id] -> list[record]
    # The raw data has structure: plate_id -> list of day-lists -> list of GPS records
    flattened_data = {}
    total_records = 0
    
    for plate_id, day_lists in data.items():
        flattened_records = []
        
        # Handle nested structure (list of lists)
        if day_lists and isinstance(day_lists[0], list) and len(day_lists[0]) > 0 and isinstance(day_lists[0][0], list):
            # It's a list of day-lists containing records
            for day_list in day_lists:
                for record in day_list:
                    if isinstance(record, (list, tuple)) and len(record) >= 6:
                        flattened_records.append(record)
                        total_records += 1
        else:
            # It's already a flat list of records
            for record in day_lists:
                if isinstance(record, (list, tuple)) and len(record) >= 6:
                    flattened_records.append(record)
                    total_records += 1
        
        if flattened_records:
            flattened_data[plate_id] = flattened_records
    
    return flattened_data, total_records


def compute_global_bounds(raw_data_files: List[Dict[str, List]]) -> GlobalBounds:
    """
    Compute global GPS bounds from all raw data files.
    
    CRITICAL: Bounds must be computed from the entire combined dataset
    before any quantization occurs. This ensures grid cell consistency
    across all data.
    
    Args:
        raw_data_files: List of raw data dicts loaded from pickle files
                       (already flattened by load_raw_data)
        
    Returns:
        GlobalBounds object with min/max lat/lon
    """
    all_lats = []
    all_lons = []
    
    for data in raw_data_files:
        for plate_id, records in data.items():
            for record in records:
                # Record format: [plate_id, lat, lon, seconds, passenger, timestamp]
                if isinstance(record, (list, tuple)) and len(record) >= 3:
                    all_lats.append(record[1])  # latitude
                    all_lons.append(record[2])  # longitude
    
    if not all_lats:
        raise ValueError("No valid GPS coordinates found in raw data")
    
    return GlobalBounds(
        lat_min=min(all_lats),
        lat_max=max(all_lats),
        lon_min=min(all_lons),
        lon_max=max(all_lons),
    )


def gps_to_grid(
    lat: float, 
    lon: float, 
    bounds: GlobalBounds, 
    grid_size: float = 0.01
) -> Tuple[int, int]:
    """
    Convert a single GPS coordinate to grid cell indices.
    
    IDENTICAL logic to pickup_dropoff_counts/processor.py::gps_to_grid()
    
    Args:
        lat: Latitude value
        lon: Longitude value
        bounds: GlobalBounds object with coordinate bounds
        grid_size: Size of each grid cell in degrees
        
    Returns:
        Tuple of (x_grid, y_grid) indices (0-indexed before offset)
    """
    lat_bins = np.arange(bounds.lat_min, bounds.lat_max + grid_size, grid_size)
    lon_bins = np.arange(bounds.lon_min, bounds.lon_max + grid_size, grid_size)
    
    x_grid = np.searchsorted(lat_bins, lat, side='right') - 1
    y_grid = np.searchsorted(lon_bins, lon, side='right') - 1
    
    # Clamp to valid range
    x_grid = max(0, min(x_grid, len(lat_bins) - 2))
    y_grid = max(0, min(y_grid, len(lon_bins) - 2))
    
    return int(x_grid), int(y_grid)


def seconds_to_time_bin(seconds: int, time_interval: int = 5) -> int:
    """
    Convert seconds since midnight to time bin.
    
    Args:
        seconds: Seconds since midnight (0-86399)
        time_interval: Size of time bins in minutes
        
    Returns:
        Time bin index [0, 287] for 5-minute intervals
    """
    minutes = seconds // 60
    time_bin = minutes // time_interval
    return int(time_bin)


def timestamp_to_day(timestamp_str: str, exclude_weekends: bool = True) -> Optional[int]:
    """
    Extract day-of-week from timestamp string.
    
    Returns 1-indexed day: Monday = 1, Tuesday = 2, ..., Friday = 5
    Saturday (6) and Sunday (7) return None if exclude_weekends is True.
    
    Note: The raw GPS data only contains weekday data (Monday-Friday),
    so Saturday and Sunday exclusion is effectively a safety filter.
    
    Args:
        timestamp_str: Timestamp string in format 'YYYY-MM-DD HH:MM:SS'
        exclude_weekends: If True, return None for Saturday and Sunday
        
    Returns:
        Day index [1, 5] for weekdays, or None for weekends if excluded
    """
    try:
        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        dow = dt.weekday()  # Monday=0, ..., Saturday=5, Sunday=6
        
        if exclude_weekends and dow >= 5:  # Saturday=5, Sunday=6
            return None
        
        return dow + 1  # Convert to 1-indexed (Monday=1, ..., Friday=5)
    except ValueError:
        return None


def extract_passenger_seeking_trajectories(
    raw_data: Dict[str, List],
    bounds: GlobalBounds,
    config: ProcessingConfig,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Tuple[Dict[str, List[List[List[int]]]], int, int]:
    """
    Extract passenger-seeking trajectories from raw GPS data.
    
    A passenger-seeking trajectory:
    - Begins when passenger_indicator changes from 1 to 0 (dropoff)
    - Ends when passenger_indicator changes from 0 to 1 (pickup)
    - Contains only states where passenger_indicator = 0
    
    Args:
        raw_data: Raw GPS data dict {plate_id: [records]}
        bounds: GlobalBounds for GPS coordinate quantization
        config: Processing configuration
        progress_callback: Optional callback(stage, progress)
        
    Returns:
        Tuple of:
        - trajectories: Dict {plate_id: [[trajectory], ...]}
          Each trajectory is a list of states: [x_grid, y_grid, time_bucket, day]
        - total_states: Total number of states across all trajectories
        - filtered_records: Number of records filtered (Sundays, etc.)
    """
    def update_progress(stage: str, progress: float):
        if progress_callback:
            progress_callback(stage, progress)
    
    trajectories = {}
    total_states = 0
    filtered_records = 0
    
    plate_ids = list(raw_data.keys())
    
    for plate_idx, plate_id in enumerate(plate_ids):
        records = raw_data[plate_id]
        plate_trajectories = []
        current_trajectory = []
        
        # Track previous passenger indicator for transition detection
        prev_passenger = None
        
        for record in records:
            # record: [plate_id, lat, lon, seconds, passenger_indicator, timestamp]
            lat = record[1]
            lon = record[2]
            seconds = record[3]
            passenger = record[4]
            timestamp = record[5]
            
            # Extract day and filter weekends if configured
            day = timestamp_to_day(timestamp, config.exclude_weekends)
            if day is None:
                filtered_records += 1
                prev_passenger = passenger
                continue
            
            # Convert to grid coordinates
            x_grid, y_grid = gps_to_grid(lat, lon, bounds, config.grid_size)
            
            # Apply offsets for alignment
            x_grid += config.x_grid_offset
            y_grid += config.y_grid_offset
            
            # Convert to time bin
            time_bucket = seconds_to_time_bin(seconds, config.time_interval)
            time_bucket += config.time_offset
            
            # Detect transitions and manage trajectories
            if prev_passenger is not None:
                # Transition from passenger to empty (dropoff) - start new trajectory
                if prev_passenger == 1 and passenger == 0:
                    if current_trajectory:
                        # Save previous trajectory if it exists and meets length constraints
                        if len(current_trajectory) >= config.min_trajectory_length and len(current_trajectory) <= config.max_trajectory_length:
                            plate_trajectories.append(current_trajectory)
                            total_states += len(current_trajectory)
                    current_trajectory = []
                
                # Transition from empty to passenger (pickup) - end current trajectory
                elif prev_passenger == 0 and passenger == 1:
                    if current_trajectory and len(current_trajectory) >= config.min_trajectory_length and len(current_trajectory) <= config.max_trajectory_length:
                        plate_trajectories.append(current_trajectory)
                        total_states += len(current_trajectory)
                    current_trajectory = []
            
            # Add state to current trajectory if passenger seeking (passenger = 0)
            if passenger == 0:
                state = [x_grid, y_grid, time_bucket, day]
                current_trajectory.append(state)
            
            prev_passenger = passenger
        
        # Don't forget the last trajectory if it's valid
        if current_trajectory and len(current_trajectory) >= config.min_trajectory_length:
            if len(current_trajectory) <= config.max_trajectory_length:
                plate_trajectories.append(current_trajectory)
                total_states += len(current_trajectory)
        
        # Apply max trajectories per driver limit (keep the longest trajectories)
        if len(plate_trajectories) > config.max_trajectories_per_driver:
            # Sort by trajectory length (descending) and keep the longest N trajectories
            sorted_trajs = sorted(plate_trajectories, key=lambda t: len(t), reverse=True)
            kept_trajs = sorted_trajs[:config.max_trajectories_per_driver]
            removed_trajs = sorted_trajs[config.max_trajectories_per_driver:]
            
            # Recalculate total_states for the removed trajectories
            for traj in removed_trajs:
                total_states -= len(traj)
            
            plate_trajectories = kept_trajs
        
        trajectories[plate_id] = plate_trajectories
        
        # Progress update
        if plate_idx % 10 == 0 or plate_idx == len(plate_ids) - 1:
            progress = (plate_idx + 1) / len(plate_ids)
            update_progress(f"Processing driver {plate_idx + 1}/{len(plate_ids)}", progress)
    
    return trajectories, total_states, filtered_records


def create_driver_index_mapping(plate_trajectories: Dict[str, List]) -> Tuple[Dict[int, List], Dict[int, str], Dict[str, int]]:
    """
    Create integer-indexed driver mapping from plate ID trajectories.
    
    The output all_trajs.pkl uses integer keys (0, 1, 2, ..., 49) instead of plate IDs.
    
    Args:
        plate_trajectories: Dict {plate_id: [[trajectory], ...]}
        
    Returns:
        Tuple of:
        - indexed_trajectories: Dict {driver_index: [[trajectory], ...]}
        - index_to_plate: Dict {driver_index: plate_id}
        - plate_to_index: Dict {plate_id: driver_index}
    """
    indexed_trajectories = {}
    index_to_plate = {}
    plate_to_index = {}
    
    for idx, (plate_id, trajs) in enumerate(sorted(plate_trajectories.items())):
        indexed_trajectories[idx] = trajs
        index_to_plate[idx] = plate_id
        plate_to_index[plate_id] = idx
    
    return indexed_trajectories, index_to_plate, plate_to_index


def process_step1(
    config: ProcessingConfig,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Tuple[Dict[int, List[List[List[int]]]], Step1Stats, Dict[int, str]]:
    """
    Main Step 1 processing function.
    
    Loads raw GPS data and extracts passenger-seeking trajectories.
    
    Args:
        config: Processing configuration
        progress_callback: Optional callback(stage, progress)
        
    Returns:
        Tuple of:
        - trajectories: Dict {driver_index: [[trajectory], ...]}
        - stats: Step1Stats with processing statistics
        - index_to_plate: Dict {driver_index: plate_id}
    """
    start_time = time.time()
    stats = Step1Stats()
    
    def update_progress(stage: str, progress: float):
        if progress_callback:
            progress_callback(stage, progress)
    
    # Stage 1: Load all raw data files
    update_progress("Loading raw data files...", 0.0)
    raw_data_files = []
    combined_data = {}
    
    for i, filename in enumerate(config.input_files):
        filepath = config.raw_data_dir / filename
        if filepath.exists():
            try:
                data, count = load_raw_data(filepath)
                raw_data_files.append(data)
                stats.total_raw_records += count
                
                # Merge into combined data
                for plate_id, records in data.items():
                    if plate_id not in combined_data:
                        combined_data[plate_id] = []
                    combined_data[plate_id].extend(records)
                
                update_progress(f"Loaded {filename}", (i + 1) / len(config.input_files) * 0.2)
            except (EOFError, ValueError) as e:
                update_progress(f"Skipped {filename}: {e}", (i + 1) / len(config.input_files) * 0.2)
    
    if not combined_data:
        raise ValueError("No valid data files found")
    
    stats.unique_drivers = len(combined_data)
    
    # Stage 2: Sort records by timestamp within each driver
    update_progress("Sorting records by timestamp...", 0.25)
    for plate_id in combined_data:
        combined_data[plate_id].sort(key=lambda r: r[5])  # Sort by timestamp string
    
    # Stage 3: Compute global bounds
    update_progress("Computing global bounds...", 0.3)
    bounds = compute_global_bounds(raw_data_files)
    stats.global_bounds = bounds
    
    # Stage 4: Extract passenger-seeking trajectories
    update_progress("Extracting passenger-seeking trajectories...", 0.35)
    plate_trajectories, total_states, filtered_records = extract_passenger_seeking_trajectories(
        combined_data, 
        bounds, 
        config,
        lambda stage, prog: update_progress(stage, 0.35 + prog * 0.55)
    )
    
    stats.records_after_sunday_filter = stats.total_raw_records - filtered_records
    
    # Stage 5: Create driver index mapping
    update_progress("Creating driver index mapping...", 0.92)
    indexed_trajectories, index_to_plate, _ = create_driver_index_mapping(plate_trajectories)
    
    # Stage 6: Compute statistics
    update_progress("Computing statistics...", 0.95)
    all_traj_lengths = []
    for driver_idx, trajs in indexed_trajectories.items():
        for traj in trajs:
            all_traj_lengths.append(len(traj))
    
    stats.total_trajectories = len(all_traj_lengths)
    stats.total_states = sum(all_traj_lengths)
    
    if all_traj_lengths:
        stats.avg_trajectory_length = np.mean(all_traj_lengths)
        stats.min_trajectory_length = min(all_traj_lengths)
        stats.max_trajectory_length = max(all_traj_lengths)
        stats.trajectories_per_driver_avg = stats.total_trajectories / len(indexed_trajectories)
    
    stats.processing_time_seconds = time.time() - start_time
    
    update_progress("Step 1 complete!", 1.0)
    
    return indexed_trajectories, stats, index_to_plate


def save_step1_output(
    trajectories: Dict[int, List[List[List[int]]]],
    output_path: Path,
    index_to_plate: Optional[Dict[int, str]] = None
) -> None:
    """
    Save Step 1 output to pickle file.
    
    Args:
        trajectories: Dict {driver_index: [[trajectory], ...]}
        output_path: Path to save the pickle file
        index_to_plate: Optional driver index to plate ID mapping
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    # Optionally save the index mapping
    if index_to_plate:
        mapping_path = output_path.with_suffix('.mapping.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump(index_to_plate, f)


def load_step1_output(filepath: Path) -> Dict[int, List[List[List[int]]]]:
    """
    Load Step 1 output from pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Dict {driver_index: [[trajectory], ...]}
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Step 1: Extract passenger-seeking trajectories from raw GPS data"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output/passenger_seeking_trajs.pkl",
        help="Output pickle file path"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=2,
        help="Minimum trajectory length"
    )
    args = parser.parse_args()
    
    config = ProcessingConfig()
    config.min_trajectory_length = args.min_length
    
    def progress_callback(stage, progress):
        print(f"[{progress*100:.1f}%] {stage}")
    
    print("Starting Step 1: Passenger-Seeking Trajectory Extraction...")
    print("=" * 60)
    
    trajectories, stats, index_to_plate = process_step1(config, progress_callback)
    
    print("\nProcessing complete!")
    print(f"  Total raw records: {stats.total_raw_records:,}")
    print(f"  Records after Sunday filter: {stats.records_after_sunday_filter:,}")
    print(f"  Unique drivers: {stats.unique_drivers}")
    print(f"  Total trajectories: {stats.total_trajectories:,}")
    print(f"  Total states: {stats.total_states:,}")
    print(f"  Avg trajectory length: {stats.avg_trajectory_length:.2f}")
    print(f"  Min trajectory length: {stats.min_trajectory_length}")
    print(f"  Max trajectory length: {stats.max_trajectory_length}")
    print(f"  Processing time: {stats.processing_time_seconds:.2f}s")
    
    output_path = Path(args.output)
    save_step1_output(trajectories, output_path, index_to_plate)
    print(f"\nSaved to {output_path}")
