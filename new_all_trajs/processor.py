"""
All Trajs Dataset Processor

This module processes raw taxi GPS trajectory data to generate the all_trajs dataset,
using consistent quantization logic aligned with other datasets in the FAMAIL project.

The processing pipeline:
1. Load raw GPS data from taxi_record_*.pkl files
2. Detect passenger-seeking segments (passenger_indicator = 0)
3. Quantize GPS coordinates to grid cells using consistent global bounds
4. Build state vectors with spatial features, temporal features, and action codes
5. Output driver-indexed dictionary of trajectories

Memory Optimization Notes:
- Raw data is processed one month at a time, then cleared
- Global bounds are computed with streaming min/max (no large lists)
- Grid bin arrays are cached to avoid repeated allocations
- Garbage collection is triggered between drivers
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any, Generator
from dataclasses import dataclass
import time
import gc

from config import (
    ProcessingConfig, 
    GlobalBounds, 
    ProcessingStats, 
    NORMALIZATION_CONSTANTS
)


# Module-level cache for grid bins (computed once, reused)
_grid_bins_cache: Dict[str, np.ndarray] = {}


def load_raw_data(filepath: Path) -> Dict[str, List]:
    """
    Load raw taxi GPS data from a pickle file.
    
    The raw data structure is:
    {
        plate_id: [record1, record2, ...]  # List of GPS records
    }
    
    Each record is a 6-element list:
    [plate_id, latitude, longitude, seconds_since_midnight, passenger_indicator, timestamp_str]
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Dictionary keyed by driver plate_id containing list of GPS records
    """
    if filepath.stat().st_size == 0:
        raise EOFError(f"File is empty: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise EOFError(f"Failed to load pickle file {filepath}: {e}")
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    
    return data


def load_feature_data(config: ProcessingConfig) -> Tuple[Dict, Dict, Dict]:
    """
    Load the feature data files (traffic, volume, train_airport).
    
    Args:
        config: ProcessingConfig with paths to data files
        
    Returns:
        Tuple of (traffic_dict, volume_dict, train_airport_dict)
    """
    traffic_path = config.source_data_dir / config.traffic_file
    volume_path = config.source_data_dir / config.volume_file
    train_airport_path = config.source_data_dir / config.train_airport_file
    
    with open(traffic_path, 'rb') as f:
        traffic = pickle.load(f)
    
    with open(volume_path, 'rb') as f:
        volume = pickle.load(f)
    
    with open(train_airport_path, 'rb') as f:
        train_airport = pickle.load(f)
    
    return traffic, volume, train_airport


def compute_global_bounds(all_data: Dict[str, Dict[str, List]]) -> GlobalBounds:
    """
    Compute global GPS bounds from all raw data using streaming min/max.
    
    Memory optimized: Uses streaming approach instead of building large lists.
    
    Args:
        all_data: Dictionary of month data, each containing driver-keyed records
                  Structure: {month: {plate_id: [[records for day1], [records for day2], ...]}}
        
    Returns:
        GlobalBounds object with min/max lat/lon
    """
    lat_min = float('inf')
    lat_max = float('-inf')
    lon_min = float('inf')
    lon_max = float('-inf')
    
    for month_data in all_data.values():
        for days_list in month_data.values():
            # days_list is a list of days, each day is a list of GPS records
            for day_records in days_list:
                for record in day_records:
                    lat = record[1]
                    lon = record[2]
                    if lat < lat_min:
                        lat_min = lat
                    if lat > lat_max:
                        lat_max = lat
                    if lon < lon_min:
                        lon_min = lon
                    if lon > lon_max:
                        lon_max = lon
    
    return GlobalBounds(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


def compute_bounds_from_file(filepath: Path) -> GlobalBounds:
    """
    Compute GPS bounds from a single raw data file.
    
    Memory optimized: Processes file and discards data after getting bounds.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        GlobalBounds object with min/max lat/lon for this file
    """
    lat_min = float('inf')
    lat_max = float('-inf')
    lon_min = float('inf')
    lon_max = float('-inf')
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    for days_list in data.values():
        for day_records in days_list:
            for record in day_records:
                lat = record[1]
                lon = record[2]
                if lat < lat_min:
                    lat_min = lat
                if lat > lat_max:
                    lat_max = lat
                if lon < lon_min:
                    lon_min = lon
                if lon > lon_max:
                    lon_max = lon
    
    # Clear data immediately
    del data
    gc.collect()
    
    return GlobalBounds(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


def merge_bounds(bounds_list: List[GlobalBounds]) -> GlobalBounds:
    """Merge multiple GlobalBounds into one encompassing all."""
    return GlobalBounds(
        lat_min=min(b.lat_min for b in bounds_list),
        lat_max=max(b.lat_max for b in bounds_list),
        lon_min=min(b.lon_min for b in bounds_list),
        lon_max=max(b.lon_max for b in bounds_list),
    )


def initialize_grid_bins(bounds: GlobalBounds, config: ProcessingConfig) -> None:
    """
    Pre-compute and cache grid bin arrays for efficient GPS to grid conversion.
    
    This avoids recreating arrays on every gps_to_grid call.
    
    Args:
        bounds: GlobalBounds object
        config: ProcessingConfig with grid parameters
    """
    global _grid_bins_cache
    _grid_bins_cache['lat_bins'] = np.arange(bounds.lat_min, bounds.lat_max + config.grid_size, config.grid_size)
    _grid_bins_cache['lon_bins'] = np.arange(bounds.lon_min, bounds.lon_max + config.grid_size, config.grid_size)


def gps_to_grid(
    lat: float, 
    lon: float, 
    bounds: GlobalBounds, 
    config: ProcessingConfig
) -> Tuple[int, int]:
    """
    Convert GPS coordinates to grid cell indices.
    
    Uses cached bin arrays for efficiency.
    
    Args:
        lat: Latitude value
        lon: Longitude value
        bounds: GlobalBounds object with coordinate bounds
        config: ProcessingConfig with grid parameters
        
    Returns:
        Tuple of (x_grid, y_grid) indices
    """
    global _grid_bins_cache
    
    # Use cached bins if available, otherwise compute them
    if 'lat_bins' not in _grid_bins_cache:
        initialize_grid_bins(bounds, config)
    
    lat_bins = _grid_bins_cache['lat_bins']
    lon_bins = _grid_bins_cache['lon_bins']
    
    x_grid = np.searchsorted(lat_bins, lat, side='right') - 1 + config.x_grid_offset
    y_grid = np.searchsorted(lon_bins, lon, side='right') - 1 + config.y_grid_offset
    
    return int(x_grid), int(y_grid)


def seconds_to_time_bucket(seconds: int, config: ProcessingConfig) -> int:
    """
    Convert seconds since midnight to time bucket index.
    
    Args:
        seconds: Seconds since midnight (0-86399)
        config: ProcessingConfig with time parameters
        
    Returns:
        Time bucket index [1, 288] (1-indexed)
    """
    time_bucket = seconds // (config.time_interval * 60)  # 5-minute buckets
    return time_bucket + config.time_offset


def timestamp_to_day(timestamp_str: str, config: ProcessingConfig) -> Optional[int]:
    """
    Convert timestamp string to day-of-week index.
    
    Monday = 1, Tuesday = 2, ..., Saturday = 6
    Sunday is excluded (returns None).
    
    Args:
        timestamp_str: Timestamp in format 'YYYY-MM-DD HH:MM:SS'
        config: ProcessingConfig with exclusion settings
        
    Returns:
        Day index [1, 6] or None if Sunday and excluded
    """
    dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    dow = dt.weekday()  # Monday=0, Sunday=6
    
    if config.exclude_sunday and dow == 6:
        return None
    
    return dow + 1


def judging_action(x: int, y: int, nx: int, ny: int) -> int:
    """
    Determine action code based on position change.
    
    This is the exact logic from net-dis-cGAIL.ipynb Cell 6.
    
    Args:
        x: Current x position
        y: Current y position
        nx: Next x position
        ny: Next y position
        
    Returns:
        Action code (0-9)
    """
    if x == 0 and y == 0:
        return 9
    if nx == 0 and ny == 0:
        return 9  # stop
    if x == nx and ny > y:
        return 0  # north
    if x < nx and ny > y:
        return 1  # northeast
    if x < nx and ny == y:
        return 2  # east
    if x < nx and ny < y:
        return 3  # southeast
    if x == nx and ny < y:
        return 4  # south
    if x > nx and ny < y:
        return 5  # southwest
    if x > nx and ny == y:
        return 6  # west
    if x > nx and ny > y:
        return 7  # northwest
    if x == nx and y == ny:
        return 8  # stay
    return 9  # default fallback


def processing_state_features(
    input_state: List,
    volume: Dict,
    train_airport: Dict,
    traffic: Dict
) -> List[float]:
    """
    Generate state features for a given position and time.
    
    This is the exact logic from net-dis-cGAIL.ipynb Cell 9 (processing_state_features).
    
    Args:
        input_state: [x, y, t, day] - position and time
        volume: latest_volume_pickups data
        train_airport: train_airport POI data
        traffic: latest_traffic data
        
    Returns:
        List of state features (121 values: 21 POI distances + 4*25 window features)
    """
    x = int(input_state[0])
    y = int(input_state[1])
    t = int(input_state[2])
    day = int(input_state[3])
    
    # Set up 5x5 grid window centered on (x, y)
    x_range = list(range(x - 2, x + 3))
    y_range = list(range(y - 2, y + 3))
    
    # State features
    n_p = []  # normalized pickup counts from volume (25 values)
    n_v = []  # normalized trip volumes from volume (25 values)
    t_s = []  # normalized traffic speeds from traffic (25 values)
    t_w = []  # normalized traffic waiting times from traffic (25 values)
    
    for i in x_range:
        for j in y_range:
            # Pickup counts and traffic volume from volume dict
            if (i, j, t, day) in volume:
                n_p.append((volume[(i, j, t, day)][0] - NORMALIZATION_CONSTANTS["pickup_mean"]) / NORMALIZATION_CONSTANTS["pickup_std"])
                n_v.append((volume[(i, j, t, day)][1] - NORMALIZATION_CONSTANTS["volume_mean"]) / NORMALIZATION_CONSTANTS["volume_std"])
            else:
                n_p.append(-NORMALIZATION_CONSTANTS["pickup_mean"] / NORMALIZATION_CONSTANTS["pickup_std"])
                n_v.append(-NORMALIZATION_CONSTANTS["volume_mean"] / NORMALIZATION_CONSTANTS["volume_std"])
            
            # Traffic speed and waiting time from traffic dict
            if (i, j, t, day) in traffic:
                t_s.append((traffic[(i, j, t, day)][0] - NORMALIZATION_CONSTANTS["speed_mean"]) / NORMALIZATION_CONSTANTS["speed_std"])
                t_w.append((traffic[(i, j, t, day)][1] - NORMALIZATION_CONSTANTS["wait_mean"]) / NORMALIZATION_CONSTANTS["wait_std"])
            else:
                t_s.append(-NORMALIZATION_CONSTANTS["speed_mean"] / NORMALIZATION_CONSTANTS["speed_std"])
                t_w.append(-NORMALIZATION_CONSTANTS["wait_mean"] / NORMALIZATION_CONSTANTS["wait_std"])
    
    # Manhattan distances to POI locations
    ta = []
    for place in train_airport:
        ta.append(abs(x - train_airport[place][0][0]) + abs(y - train_airport[place][0][1]))
    
    # Assemble complete feature vector
    # ORDER MATCHES ORIGINAL: POI distances, pickup counts, volumes, speeds, waiting times
    whole_step = []
    whole_step.extend(ta)    # indices 0-20: POI distances (21 values)
    whole_step.extend(n_p)   # indices 21-45: pickup counts (25 values)
    whole_step.extend(n_v)   # indices 46-70: traffic volumes (25 values)
    whole_step.extend(t_s)   # indices 71-95: traffic speeds (25 values)
    whole_step.extend(t_w)   # indices 96-120: traffic waiting times (25 values)
    
    return whole_step


def extract_passenger_seeking_segments(
    records: List[List],
    bounds: GlobalBounds,
    config: ProcessingConfig
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Extract passenger-seeking trajectory segments from raw GPS records.
    
    A passenger-seeking segment is a contiguous sequence of records where
    passenger_indicator = 0 (taxi is empty, looking for passengers).
    
    Args:
        records: List of GPS records for a driver
        bounds: GlobalBounds for GPS quantization
        config: ProcessingConfig
        
    Returns:
        List of trajectories, each trajectory is a list of (x, y, t, day) tuples
    """
    trajectories = []
    current_traj = []
    
    for record in records:
        # record format: [plate_id, lat, lon, seconds, passenger_indicator, timestamp]
        lat = record[1]
        lon = record[2]
        seconds = record[3]
        passenger_indicator = record[4]
        timestamp = record[5]
        
        # Get day index
        day = timestamp_to_day(timestamp, config)
        if day is None:  # Sunday excluded
            # End current trajectory if any
            if len(current_traj) >= 2:
                trajectories.append(current_traj)
            current_traj = []
            continue
        
        # Only include passenger-seeking records (empty taxi)
        if passenger_indicator == 0:
            x, y = gps_to_grid(lat, lon, bounds, config)
            t = seconds_to_time_bucket(seconds, config)
            
            current_traj.append((x, y, t, day))
        else:
            # Passenger onboard - end current trajectory
            if len(current_traj) >= 2:  # Minimum 2 states for a valid trajectory
                trajectories.append(current_traj)
            current_traj = []
    
    # Don't forget the last trajectory
    if len(current_traj) >= 2:
        trajectories.append(current_traj)
    
    return trajectories


def deduplicate_trajectory(trajectory: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Remove consecutive duplicate states from a trajectory.
    
    Consecutive states with identical (x, y, t) are merged into one.
    
    Args:
        trajectory: List of (x, y, t, day) tuples
        
    Returns:
        Deduplicated trajectory
    """
    if len(trajectory) <= 1:
        return trajectory
    
    deduped = [trajectory[0]]
    for state in trajectory[1:]:
        prev = deduped[-1]
        # Only add if position or time changed
        if state[:3] != prev[:3]:
            deduped.append(state)
    
    return deduped


def build_state_vectors(
    trajectories: List[List[Tuple[int, int, int, int]]],
    traffic: Dict,
    volume: Dict,
    train_airport: Dict
) -> List[List[List]]:
    """
    Build complete state vectors for all trajectories.
    
    Each state vector has 126 elements:
    - [0]: x_grid
    - [1]: y_grid
    - [2]: time_bucket
    - [3]: day_index
    - [4-24]: POI manhattan distances (21 values)
    - [25-49]: normalized pickup counts (25 values)
    - [50-74]: normalized traffic volumes (25 values)
    - [75-99]: normalized traffic speeds (25 values)
    - [100-124]: normalized traffic waiting times (25 values)
    - [125]: action code
    
    Args:
        trajectories: List of trajectories, each a list of (x, y, t, day) tuples
        traffic: Traffic data dictionary
        volume: Volume data dictionary
        train_airport: POI data dictionary
        
    Returns:
        List of trajectories with full state vectors
    """
    processed_trajectories = []
    
    for traj in trajectories:
        if len(traj) < 2:
            continue
        
        processed_traj = []
        
        for i in range(len(traj) - 1):
            x, y, t, day = traj[i]
            nx, ny, nt, nday = traj[i + 1]
            
            # Build state vector
            state = [x, y, t, day]
            
            # Add state features (POI distances + window features)
            state_features = processing_state_features([x, y, t, day], volume, train_airport, traffic)
            state.extend(state_features)
            
            # Compute action
            action = judging_action(x, y, nx, ny)
            state.append(action)
            
            processed_traj.append(state)
        
        # Add final state with action = 9 (stop)
        x, y, t, day = traj[-1]
        state = [x, y, t, day]
        state_features = processing_state_features([x, y, t, day], volume, train_airport, traffic)
        state.extend(state_features)
        state.append(9)  # Stop action
        processed_traj.append(state)
        
        if len(processed_traj) >= 2:
            processed_trajectories.append(processed_traj)
    
    return processed_trajectories


def process_data(
    config: ProcessingConfig,
    progress_callback: Optional[callable] = None
) -> Tuple[Dict[int, List[List[List]]], ProcessingStats]:
    """
    Main processing function that generates the all_trajs dataset.
    
    Memory optimized version:
    - Computes bounds by streaming through files one at a time
    - Processes each driver's data and immediately clears intermediate data
    - Runs garbage collection between drivers
    - Caches grid bin arrays to avoid repeated allocations
    
    Args:
        config: ProcessingConfig object
        progress_callback: Optional callback function(stage, progress) for progress updates
        
    Returns:
        Tuple of (all_trajs dict, ProcessingStats)
    """
    global _grid_bins_cache
    _grid_bins_cache.clear()  # Reset cache
    
    start_time = time.time()
    stats = ProcessingStats()
    
    def update_progress(stage: str, progress: float):
        if progress_callback:
            progress_callback(stage, progress)
    
    # Stage 1: Compute global bounds WITHOUT loading all data into memory
    update_progress("Computing global bounds...", 0.0)
    bounds_list = []
    valid_files = []
    
    for i, filename in enumerate(config.input_files):
        filepath = config.raw_data_dir / filename
        if filepath.exists():
            try:
                file_bounds = compute_bounds_from_file(filepath)
                bounds_list.append(file_bounds)
                valid_files.append(filename)
                update_progress(f"Scanned {filename} for bounds", (i + 1) / len(config.input_files) * 0.1)
            except (EOFError, ValueError) as e:
                print(f"Warning: Skipped {filename}: {e}")
    
    if not bounds_list:
        raise ValueError("No valid raw data files found")
    
    bounds = merge_bounds(bounds_list)
    stats.global_bounds = bounds
    del bounds_list
    gc.collect()
    
    # Initialize grid bins cache
    initialize_grid_bins(bounds, config)
    
    # Stage 2: Load feature data (these are needed throughout processing)
    update_progress("Loading feature data...", 0.1)
    traffic, volume, train_airport = load_feature_data(config)
    
    # Stage 3: Build driver index by scanning files (without keeping all data)
    update_progress("Building driver index...", 0.15)
    all_plates = set()
    
    for filename in valid_files:
        filepath = config.raw_data_dir / filename
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        all_plates.update(data.keys())
        del data
        gc.collect()
    
    # Create driver ID mapping (consistent ordering)
    sorted_plates = sorted(all_plates, key=lambda x: str(x))
    plate_to_id = {plate: idx for idx, plate in enumerate(sorted_plates)}
    stats.total_drivers = len(sorted_plates)
    
    # Stage 4: Process each driver's data - loading files as needed
    update_progress("Processing trajectories...", 0.2)
    all_trajs = {}  # Don't pre-allocate, add as we go
    total_records = 0
    total_states = 0
    total_trajectories = 0
    
    # Load all raw data into a more efficient structure: plate -> records
    # We need to do this because records for a single driver span multiple files
    # But we'll process and clear as we go
    
    # First pass: collect records per driver from each file
    driver_records_by_file = {plate: [] for plate in sorted_plates}
    
    for file_idx, filename in enumerate(valid_files):
        filepath = config.raw_data_dir / filename
        update_progress(f"Loading {filename}...", 0.2 + (file_idx / len(valid_files)) * 0.1)
        
        with open(filepath, 'rb') as f:
            month_data = pickle.load(f)
        
        for plate in sorted_plates:
            if plate in month_data:
                # Flatten and extend
                for day_records in month_data[plate]:
                    driver_records_by_file[plate].extend(day_records)
        
        # Clear this file's data
        del month_data
        gc.collect()
    
    # Stage 5: Process each driver
    for plate_idx, plate in enumerate(sorted_plates):
        driver_id = plate_to_id[plate]
        
        # Get all records for this driver
        driver_records = driver_records_by_file[plate]
        total_records += len(driver_records)
        
        if len(driver_records) == 0:
            all_trajs[driver_id] = []
            continue
        
        # Sort records by timestamp
        driver_records.sort(key=lambda r: r[5])
        
        # Extract passenger-seeking segments
        raw_trajectories = extract_passenger_seeking_segments(driver_records, bounds, config)
        
        # Clear driver records immediately after extraction
        driver_records_by_file[plate] = None
        
        # Deduplicate each trajectory
        deduped_trajectories = [deduplicate_trajectory(traj) for traj in raw_trajectories]
        deduped_trajectories = [t for t in deduped_trajectories if len(t) >= 2]
        
        # Clear raw trajectories
        del raw_trajectories
        
        # Build state vectors
        processed_trajectories = build_state_vectors(
            deduped_trajectories, traffic, volume, train_airport
        )
        
        # Clear deduped trajectories
        del deduped_trajectories
        
        all_trajs[driver_id] = processed_trajectories
        total_trajectories += len(processed_trajectories)
        total_states += sum(len(t) for t in processed_trajectories)
        
        # Run garbage collection every 5 drivers to reclaim memory
        if plate_idx % 5 == 0:
            gc.collect()
        
        # Update progress
        progress = 0.3 + (plate_idx / len(sorted_plates)) * 0.65
        update_progress(f"Processed driver {plate_idx + 1}/{len(sorted_plates)}", progress)
    
    # Final cleanup
    del driver_records_by_file
    gc.collect()
    
    stats.total_records = total_records
    stats.total_trajectories = total_trajectories
    stats.total_states = total_states
    stats.processing_time_seconds = time.time() - start_time
    
    update_progress("Complete!", 1.0)
    
    return all_trajs, stats


def save_output(data: Dict[int, List[List[List]]], output_path: Path) -> None:
    """
    Save the all_trajs dictionary to a pickle file.
    
    Args:
        data: Dictionary with driver indices as keys and trajectory lists as values
        output_path: Path to save the pickle file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def load_output(filepath: Path) -> Dict[int, List[List[List]]]:
    """
    Load a previously saved all_trajs dictionary.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Dictionary with driver indices as keys and trajectory lists as values
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def process_data_low_memory(
    config: ProcessingConfig,
    progress_callback: Optional[callable] = None
) -> Tuple[Dict[int, List[List[List]]], ProcessingStats]:
    """
    Ultra-low memory version of process_data.
    
    This version processes one driver at a time, loading only the records
    needed for that driver from each file. Much slower but uses significantly
    less memory - suitable for memory-constrained environments like WSL2.
    
    Trade-offs:
    - Re-reads files multiple times (once per driver)
    - Much slower (~50x slower)
    - Uses ~70% less peak memory
    
    Args:
        config: ProcessingConfig object
        progress_callback: Optional callback function(stage, progress) for progress updates
        
    Returns:
        Tuple of (all_trajs dict, ProcessingStats)
    """
    global _grid_bins_cache
    _grid_bins_cache.clear()
    
    start_time = time.time()
    stats = ProcessingStats()
    
    def update_progress(stage: str, progress: float):
        if progress_callback:
            progress_callback(stage, progress)
    
    # Stage 1: Compute bounds and collect driver plates
    update_progress("Scanning files for bounds and drivers...", 0.0)
    bounds_list = []
    valid_files = []
    all_plates = set()
    
    for i, filename in enumerate(config.input_files):
        filepath = config.raw_data_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                # Collect plates
                all_plates.update(data.keys())
                
                # Compute bounds
                lat_min, lat_max = float('inf'), float('-inf')
                lon_min, lon_max = float('inf'), float('-inf')
                
                for days_list in data.values():
                    for day_records in days_list:
                        for record in day_records:
                            lat, lon = record[1], record[2]
                            if lat < lat_min: lat_min = lat
                            if lat > lat_max: lat_max = lat
                            if lon < lon_min: lon_min = lon
                            if lon > lon_max: lon_max = lon
                
                bounds_list.append(GlobalBounds(lat_min, lat_max, lon_min, lon_max))
                valid_files.append(filename)
                
                del data
                gc.collect()
                
                update_progress(f"Scanned {filename}", (i + 1) / len(config.input_files) * 0.1)
            except (EOFError, ValueError) as e:
                print(f"Warning: Skipped {filename}: {e}")
    
    if not bounds_list:
        raise ValueError("No valid raw data files found")
    
    bounds = merge_bounds(bounds_list)
    stats.global_bounds = bounds
    del bounds_list
    gc.collect()
    
    # Initialize grid bins
    initialize_grid_bins(bounds, config)
    
    # Stage 2: Load feature data
    update_progress("Loading feature data...", 0.1)
    traffic, volume, train_airport = load_feature_data(config)
    
    # Create driver mapping
    sorted_plates = sorted(all_plates, key=lambda x: str(x))
    plate_to_id = {plate: idx for idx, plate in enumerate(sorted_plates)}
    stats.total_drivers = len(sorted_plates)
    
    # Stage 3: Process each driver one at a time
    all_trajs = {}
    total_records = 0
    total_states = 0
    total_trajectories = 0
    
    for plate_idx, plate in enumerate(sorted_plates):
        driver_id = plate_to_id[plate]
        
        # Collect records for this driver from all files
        driver_records = []
        for filename in valid_files:
            filepath = config.raw_data_dir / filename
            with open(filepath, 'rb') as f:
                month_data = pickle.load(f)
            
            if plate in month_data:
                for day_records in month_data[plate]:
                    driver_records.extend(day_records)
            
            del month_data
        
        total_records += len(driver_records)
        
        if len(driver_records) == 0:
            all_trajs[driver_id] = []
            gc.collect()
            continue
        
        # Sort and process
        driver_records.sort(key=lambda r: r[5])
        raw_trajectories = extract_passenger_seeking_segments(driver_records, bounds, config)
        del driver_records
        
        deduped_trajectories = [deduplicate_trajectory(traj) for traj in raw_trajectories]
        deduped_trajectories = [t for t in deduped_trajectories if len(t) >= 2]
        del raw_trajectories
        
        processed_trajectories = build_state_vectors(
            deduped_trajectories, traffic, volume, train_airport
        )
        del deduped_trajectories
        
        all_trajs[driver_id] = processed_trajectories
        total_trajectories += len(processed_trajectories)
        total_states += sum(len(t) for t in processed_trajectories)
        
        # Aggressive garbage collection
        gc.collect()
        
        progress = 0.15 + (plate_idx / len(sorted_plates)) * 0.8
        update_progress(f"Processed driver {plate_idx + 1}/{len(sorted_plates)}", progress)
    
    stats.total_records = total_records
    stats.total_trajectories = total_trajectories
    stats.total_states = total_states
    stats.processing_time_seconds = time.time() - start_time
    
    update_progress("Complete!", 1.0)
    
    return all_trajs, stats


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Process raw GPS data to generate all_trajs dataset")
    parser.add_argument("--output", type=str, default="output/new_all_trajs.pkl",
                        help="Output pickle file path")
    args = parser.parse_args()
    
    config = ProcessingConfig()
    
    def progress_callback(stage, progress):
        print(f"[{progress*100:.1f}%] {stage}")
    
    print("Starting data processing...")
    all_trajs, stats = process_data(config, progress_callback)
    
    print(f"\nProcessing complete!")
    print(f"  Total drivers: {stats.total_drivers}")
    print(f"  Total trajectories: {stats.total_trajectories}")
    print(f"  Total states: {stats.total_states}")
    print(f"  Processing time: {stats.processing_time_seconds:.2f}s")
    
    output_path = Path(args.output)
    save_output(all_trajs, output_path)
    print(f"\nSaved to {output_path}")
