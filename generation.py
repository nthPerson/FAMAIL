"""
Active Taxi Count Generation.

This module implements the algorithm for counting active taxis in each
n×n grid neighborhood for each time period.

An "active taxi" is defined as any taxi that was present (had at least one
GPS reading) in the n×n neighborhood during the specified time period.
"""

import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Set, Callable
from collections import defaultdict

try:
    from .config import ActiveTaxisConfig
    from .processor import (
        GlobalBounds,
        ProcessingStats,
        load_raw_data,
        compute_global_bounds,
        gps_to_grid,
        timestamp_to_time_bin,
        timestamp_to_hour,
        timestamp_to_day,
        get_period_key,
        get_neighborhood_cells,
    )
except ImportError:
    from config import ActiveTaxisConfig
    from processor import (
        GlobalBounds,
        ProcessingStats,
        load_raw_data,
        compute_global_bounds,
        gps_to_grid,
        timestamp_to_time_bin,
        timestamp_to_hour,
        timestamp_to_day,
        get_period_key,
        get_neighborhood_cells,
)


def build_taxi_presence_index(
    df: pd.DataFrame,
    period_type: str,
    grid_dims: Tuple[int, int],
    progress_callback: Optional[Callable] = None,
) -> Dict[Tuple, Dict[Tuple[int, int], Set[str]]]:
    """
    Build an index mapping (period) -> (x, y) -> set of taxi plate_ids.
    
    This index tells us which taxis were present in each cell during each period.
    
    Args:
        df: DataFrame with columns: plate_id, x_grid, y_grid, time_bin, hour, day
        period_type: 'time_bucket', 'hourly', 'daily', or 'all'
        grid_dims: (x_max, y_max) grid dimensions
        progress_callback: Optional callback(stage, progress)
        
    Returns:
        Nested dict: period_key -> (x, y) -> set of plate_ids
    """
    # Index structure: {period_key: {(x, y): {plate_id1, plate_id2, ...}}}
    presence_index: Dict[Tuple, Dict[Tuple[int, int], Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    
    total_rows = len(df)
    update_interval = max(1, total_rows // 100)  # Update every 1%
    
    for idx, row in df.iterrows():
        if progress_callback and idx % update_interval == 0:
            progress_callback("Building presence index", 0.3 + 0.3 * (idx / total_rows))
        
        x, y = int(row['x_grid']), int(row['y_grid'])
        plate_id = row['plate_id']
        
        # Determine period key based on period_type
        period_key = get_period_key(
            int(row['time_bin']),
            int(row['hour']),
            int(row['day']),
            period_type
        )
        
        # Record this taxi as present in this cell during this period
        presence_index[period_key][(x, y)].add(plate_id)
    
    return dict(presence_index)


def count_active_taxis_in_neighborhoods(
    presence_index: Dict[Tuple, Dict[Tuple[int, int], Set[str]]],
    neighborhood_size: int,
    grid_dims: Tuple[int, int],
    progress_callback: Optional[Callable] = None,
) -> Dict[Tuple, int]:
    """
    For each (x, y, period), count unique taxis in the n×n neighborhood.
    
    Args:
        presence_index: Output from build_taxi_presence_index
        neighborhood_size: The k value (neighborhood is (2k+1) × (2k+1))
        grid_dims: (x_max, y_max) grid dimensions
        progress_callback: Optional callback(stage, progress)
        
    Returns:
        Dict mapping (x, y, *period_key) -> active_taxi_count
    """
    x_max, y_max = grid_dims
    active_counts: Dict[Tuple, int] = {}
    
    periods = list(presence_index.keys())
    total_periods = len(periods)
    
    for p_idx, period_key in enumerate(periods):
        if progress_callback:
            progress_callback(
                f"Counting taxis for period {p_idx + 1}/{total_periods}",
                0.6 + 0.3 * (p_idx / total_periods)
            )
        
        cell_taxis = presence_index[period_key]
        
        # For each cell in the grid
        for x in range(1, x_max + 1):
            for y in range(1, y_max + 1):
                # Get all cells in the neighborhood
                neighborhood = get_neighborhood_cells(x, y, neighborhood_size, grid_dims)
                
                # Collect all unique taxis in the neighborhood
                neighborhood_taxis: Set[str] = set()
                for nx, ny in neighborhood:
                    if (nx, ny) in cell_taxis:
                        neighborhood_taxis.update(cell_taxis[(nx, ny)])
                
                # Store the count
                # Key format: (x, y, *period_key)
                full_key = (x, y) + period_key
                active_counts[full_key] = len(neighborhood_taxis)
    
    return active_counts


def generate_active_taxi_counts(
    config: ActiveTaxisConfig,
    progress_callback: Optional[Callable] = None,
) -> Tuple[Dict[Tuple, int], ProcessingStats]:
    """
    Main function to generate active taxi count dataset.
    
    Args:
        config: ActiveTaxisConfig with processing parameters
        progress_callback: Optional callback(stage, progress) for progress updates
        
    Returns:
        Tuple of (active_taxi_counts dict, ProcessingStats)
        
    The output dictionary has keys of the form:
        - For time_bucket: (x, y, time_bin, day) -> count
        - For hourly: (x, y, hour, day) -> count
        - For daily: (x, y, day) -> count
        - For all: (x, y, 'all') -> count
    """
    start_time = time.time()
    config.validate()
    
    stats = ProcessingStats()
    
    def update_progress(stage: str, progress: float):
        if progress_callback:
            progress_callback(stage, progress)
    
    # Stage 1: Load raw data files
    update_progress("Loading raw data files...", 0.0)
    dfs = []
    
    for i, filename in enumerate(config.input_files):
        filepath = config.raw_data_dir / filename
        if filepath.exists():
            try:
                if config.test_mode:
                    df = load_raw_data(
                        filepath,
                        sample_drivers=config.test_sample_size,
                        sample_days=config.test_days
                    )
                else:
                    df = load_raw_data(filepath)
                
                if len(df) > 0:
                    dfs.append(df)
                    update_progress(
                        f"Loaded {filename}",
                        0.1 * (i + 1) / len(config.input_files)
                    )
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
    
    if not dfs:
        raise ValueError("No valid data files found")
    
    # Concatenate all data
    combined_df = pd.concat(dfs, ignore_index=True)
    stats.total_records = len(combined_df)
    stats.unique_taxis = combined_df['plate_id'].nunique()
    
    # Stage 2: Compute global bounds
    update_progress("Computing global bounds...", 0.1)
    bounds = compute_global_bounds([combined_df])
    stats.global_bounds = bounds
    
    # Stage 3: Parse timestamps
    update_progress("Parsing timestamps...", 0.15)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    
    # Stage 4: Filter out Sundays
    update_progress("Filtering Sundays...", 0.18)
    combined_df['day'] = timestamp_to_day(combined_df['timestamp'], config.exclude_sunday)
    if config.exclude_sunday:
        combined_df = combined_df.dropna(subset=['day'])
    combined_df['day'] = combined_df['day'].astype(int)
    stats.records_after_sunday_filter = len(combined_df)
    
    # Stage 5: Apply spatial quantization
    update_progress("Applying spatial quantization...", 0.2)
    x_grid, y_grid = gps_to_grid(
        combined_df['latitude'].values,
        combined_df['longitude'].values,
        bounds,
        config.grid_size
    )
    # Apply offsets for alignment with pickup_dropoff_counts
    combined_df['x_grid'] = x_grid + config.x_grid_offset
    combined_df['y_grid'] = y_grid + config.y_grid_offset
    
    # Stage 6: Apply temporal quantization
    update_progress("Applying temporal quantization...", 0.25)
    combined_df['time_bin'] = timestamp_to_time_bin(combined_df['timestamp']) + config.time_offset
    combined_df['hour'] = timestamp_to_hour(combined_df['timestamp'])
    
    # Calculate unique cells
    unique_cells = combined_df.groupby(['x_grid', 'y_grid']).ngroups
    stats.unique_cells = unique_cells
    
    # Stage 7: Build taxi presence index
    update_progress("Building taxi presence index...", 0.3)
    presence_index = build_taxi_presence_index(
        combined_df,
        config.period_type,
        config.grid_dims,
        progress_callback
    )
    stats.unique_periods = len(presence_index)
    
    # Stage 8: Count active taxis in neighborhoods
    update_progress("Counting active taxis in neighborhoods...", 0.6)
    active_counts = count_active_taxis_in_neighborhoods(
        presence_index,
        config.neighborhood_size,
        config.grid_dims,
        progress_callback
    )
    
    # Compute statistics
    if active_counts:
        counts_array = np.array(list(active_counts.values()))
        stats.max_active_taxis_in_cell = int(counts_array.max())
        stats.avg_active_taxis_per_cell = float(counts_array.mean())
    
    stats.total_output_keys = len(active_counts)
    stats.processing_time_seconds = time.time() - start_time
    
    update_progress("Complete!", 1.0)
    
    return active_counts, stats


def generate_test_dataset(
    config: ActiveTaxisConfig,
    progress_callback: Optional[Callable] = None,
) -> Tuple[Dict[Tuple, int], ProcessingStats]:
    """
    Generate a test dataset using a subset of the data.
    
    This is a convenience wrapper that sets test_mode=True.
    
    Args:
        config: ActiveTaxisConfig (test_mode will be overridden to True)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (active_taxi_counts dict, ProcessingStats)
    """
    # Create a test config based on the provided config
    from dataclasses import replace
    test_config = replace(
        config,
        test_mode=True,
        test_sample_size=config.test_sample_size or 5,
        test_days=config.test_days or 3,
    )
    
    return generate_active_taxi_counts(test_config, progress_callback)


def save_output(
    data: Dict[Tuple, int],
    stats: ProcessingStats,
    config: ActiveTaxisConfig,
    output_path: Path,
) -> None:
    """
    Save the active taxi counts dataset to a pickle file.
    
    The output includes the data, statistics, and configuration for reproducibility.
    
    Args:
        data: Dictionary with active taxi counts
        stats: ProcessingStats from generation
        config: ActiveTaxisConfig used for generation
        output_path: Path to save the pickle file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_bundle = {
        'data': data,
        'stats': stats.to_dict(),
        'config': config.to_dict(),
        'version': '1.0.0',
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_bundle, f)


def load_output(filepath: Path) -> Tuple[Dict[Tuple, int], dict, dict]:
    """
    Load a previously saved active taxi counts dataset.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Tuple of (active_taxi_counts dict, stats dict, config dict)
    """
    with open(filepath, 'rb') as f:
        bundle = pickle.load(f)
    
    # Handle both old format (just data) and new format (bundle)
    if isinstance(bundle, dict) and 'data' in bundle:
        return bundle['data'], bundle.get('stats', {}), bundle.get('config', {})
    else:
        # Old format: just the data dict
        return bundle, {}, {}


def get_active_taxi_count(
    data: Dict[Tuple, int],
    x: int,
    y: int,
    period_key: Tuple,
) -> int:
    """
    Look up the active taxi count for a specific cell and period.
    
    Args:
        data: Active taxi counts dictionary
        x: Grid x coordinate (1-indexed)
        y: Grid y coordinate (1-indexed)
        period_key: Period identifier tuple (format depends on period_type)
        
    Returns:
        Number of active taxis in the neighborhood, or 0 if not found
    """
    full_key = (x, y) + period_key
    return data.get(full_key, 0)


def validate_key_format(
    data: Dict[Tuple, int],
    period_type: str,
) -> Dict[str, any]:
    """
    Validate and describe the key format in the dataset.
    
    Args:
        data: Active taxi counts dictionary
        period_type: Expected period type
        
    Returns:
        Dictionary with validation results and key format info
    """
    if not data:
        return {'valid': False, 'error': 'Empty dataset'}
    
    sample_keys = list(data.keys())[:5]
    key_lengths = set(len(k) for k in data.keys())
    
    expected_length = {
        'time_bucket': 4,  # (x, y, time_bin, day)
        'hourly': 4,       # (x, y, hour, day)
        'daily': 3,        # (x, y, day)
        'all': 3,          # (x, y, 'all')
    }
    
    expected = expected_length.get(period_type, 4)
    
    return {
        'valid': len(key_lengths) == 1 and expected in key_lengths,
        'expected_length': expected,
        'actual_lengths': list(key_lengths),
        'sample_keys': sample_keys,
        'period_type': period_type,
    }
