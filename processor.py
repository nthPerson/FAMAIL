"""
Raw Data Processor for Active Taxis Dataset Generation.

This module contains functions for loading and processing raw taxi GPS data.
The quantization functions are designed to be IDENTICAL to those used in
pickup_dropoff_counts/processor.py to ensure data model consistency.

CRITICAL: Any changes to quantization logic must be synchronized with
pickup_dropoff_counts/processor.py to maintain dataset alignment.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Set
from dataclasses import dataclass


@dataclass
class GlobalBounds:
    """
    Stores the global GPS coordinate bounds for quantization.
    
    These bounds must be computed from the ENTIRE dataset before any
    quantization occurs to ensure consistent grid cell assignments.
    """
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    
    def to_dict(self) -> dict:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }


@dataclass
class ProcessingStats:
    """Statistics from the processing run."""
    total_records: int = 0
    records_after_sunday_filter: int = 0
    unique_taxis: int = 0
    unique_cells: int = 0
    unique_periods: int = 0
    max_active_taxis_in_cell: int = 0
    avg_active_taxis_per_cell: float = 0.0
    total_output_keys: int = 0
    processing_time_seconds: float = 0.0
    global_bounds: Optional[GlobalBounds] = None
    
    def to_dict(self) -> dict:
        return {
            'total_records': self.total_records,
            'records_after_sunday_filter': self.records_after_sunday_filter,
            'unique_taxis': self.unique_taxis,
            'unique_cells': self.unique_cells,
            'unique_periods': self.unique_periods,
            'max_active_taxis_in_cell': self.max_active_taxis_in_cell,
            'avg_active_taxis_per_cell': self.avg_active_taxis_per_cell,
            'total_output_keys': self.total_output_keys,
            'processing_time_seconds': self.processing_time_seconds,
            'global_bounds': self.global_bounds.to_dict() if self.global_bounds else None,
        }


def load_raw_data(
    filepath: Path,
    sample_drivers: Optional[int] = None,
    sample_days: Optional[int] = None
) -> pd.DataFrame:
    """
    Load raw taxi GPS data from a pickle file.
    
    The raw data structure is:
    {
        plate_id: [              # Dict keyed by driver plate ID
            [                    # List of days
                [record],        # Day contains list of GPS records
                ...
            ],
            ...
        ],
        ...
    }
    
    Each record is a 6-element list:
    [plate_id, latitude, longitude, seconds_since_midnight, passenger_indicator, timestamp_str]
    
    Args:
        filepath: Path to the pickle file
        sample_drivers: If set, only load this many drivers (for testing)
        sample_days: If set, only load this many days per driver (for testing)
        
    Returns:
        DataFrame with columns: plate_id, latitude, longitude, seconds, passenger_indicator, timestamp
        
    Raises:
        EOFError: If the file is empty or corrupted
        ValueError: If the data structure is unexpected
    """
    # Check for empty file
    if filepath.stat().st_size == 0:
        raise EOFError(f"File is empty: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise EOFError(f"Failed to load pickle file {filepath}: {e}")
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    
    # Data is a dict keyed by plate_id
    # Each value is a list of days, each day is a list of GPS records
    all_records = []
    
    driver_keys = list(data.keys())
    if sample_drivers is not None:
        driver_keys = driver_keys[:sample_drivers]
    
    for plate_id in driver_keys:
        days_list = data[plate_id]
        
        # Optionally limit days
        if sample_days is not None:
            days_list = days_list[:sample_days]
        
        for day_records in days_list:
            # day_records is a list of GPS record lists
            all_records.extend(day_records)
    
    if not all_records:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[
            'plate_id', 'latitude', 'longitude', 'seconds', 'passenger_indicator', 'timestamp'
        ])
    
    # Convert to DataFrame with appropriate column names
    df = pd.DataFrame(all_records, columns=[
        'plate_id', 'latitude', 'longitude', 'seconds', 'passenger_indicator', 'timestamp'
    ])
    
    return df


def compute_global_bounds(dfs: List[pd.DataFrame]) -> GlobalBounds:
    """
    Compute global GPS bounds from a list of DataFrames.
    
    CRITICAL: Bounds must be computed from the entire combined dataset
    before any quantization occurs. This ensures grid cell consistency
    across all data.
    
    Args:
        dfs: List of DataFrames containing latitude and longitude columns
        
    Returns:
        GlobalBounds object with min/max lat/lon
    """
    all_lats = pd.concat([df['latitude'] for df in dfs])
    all_lons = pd.concat([df['longitude'] for df in dfs])
    
    return GlobalBounds(
        lat_min=all_lats.min(),
        lat_max=all_lats.max(),
        lon_min=all_lons.min(),
        lon_max=all_lons.max(),
    )


def gps_to_grid(
    lat: np.ndarray, 
    lon: np.ndarray, 
    bounds: GlobalBounds, 
    grid_size: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert GPS coordinates to grid cell indices using numpy.digitize.
    
    IDENTICAL to pickup_dropoff_counts/processor.py::gps_to_grid()
    
    Args:
        lat: Array of latitude values
        lon: Array of longitude values
        bounds: GlobalBounds object with coordinate bounds
        grid_size: Size of each grid cell in degrees
        
    Returns:
        Tuple of (x_grid, y_grid) arrays (0-indexed before offset)
    """
    lat_bins = np.arange(bounds.lat_min, bounds.lat_max + grid_size, grid_size)
    lon_bins = np.arange(bounds.lon_min, bounds.lon_max + grid_size, grid_size)
    
    x_grid = np.digitize(lat, lat_bins) - 1   # latitude → x_grid
    y_grid = np.digitize(lon, lon_bins) - 1   # longitude → y_grid
    
    return x_grid, y_grid


def timestamp_to_time_bin(timestamps: pd.Series) -> pd.Series:
    """
    Convert timestamps to time-of-day bins (5-minute intervals).
    
    IDENTICAL to pickup_dropoff_counts/processor.py::timestamp_to_time_bin()
    
    Args:
        timestamps: Series of datetime objects
        
    Returns:
        Series of time bin indices [0, 287] (before offset)
    """
    minutes_since_midnight = timestamps.dt.hour * 60 + timestamps.dt.minute
    time_bin = minutes_since_midnight // 5
    return time_bin


def timestamp_to_hour(timestamps: pd.Series) -> pd.Series:
    """
    Convert timestamps to hour of day.
    
    Args:
        timestamps: Series of datetime objects
        
    Returns:
        Series of hour indices [0, 23]
    """
    return timestamps.dt.hour


def timestamp_to_day(timestamps: pd.Series, exclude_sunday: bool = True) -> pd.Series:
    """
    Convert timestamps to day-of-week indices.
    
    Monday = 1, Tuesday = 2, ..., Saturday = 6
    Sunday is optionally excluded (returns NaN).
    
    IDENTICAL to pickup_dropoff_counts/processor.py::timestamp_to_day()
    
    Args:
        timestamps: Series of datetime objects
        exclude_sunday: If True, Sunday records get NaN
        
    Returns:
        Series of day indices [1, 6] with NaN for Sundays if excluded
    """
    dow = timestamps.dt.weekday  # Monday=0, Sunday=6
    
    if exclude_sunday:
        # Convert to 1-indexed, Sundays become NaN
        day = dow.where(dow != 6) + 1
    else:
        # Convert to 1-indexed including Sunday (7)
        day = dow + 1
    
    return day


def get_period_key(
    time_bin: int,
    hour: int,
    day: int,
    period_type: str
) -> Tuple:
    """
    Generate the period key based on period_type.
    
    Args:
        time_bin: 5-minute interval index [1, 288]
        hour: Hour of day [0, 23]
        day: Day of week [1, 6]
        period_type: 'time_bucket', 'hourly', 'daily', or 'all'
        
    Returns:
        Period key tuple appropriate for the period_type
    """
    if period_type == 'time_bucket':
        return (time_bin, day)
    elif period_type == 'hourly':
        return (hour, day)
    elif period_type == 'daily':
        return (day,)
    else:  # 'all'
        return ('all',)


def get_neighborhood_cells(
    x: int,
    y: int,
    k: int,
    grid_dims: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Get all cells in the (2k+1) × (2k+1) neighborhood around (x, y).
    
    Clips to valid grid boundaries.
    
    Args:
        x: Center cell x coordinate (1-indexed)
        y: Center cell y coordinate (1-indexed)
        k: Neighborhood size parameter (radius)
        grid_dims: (x_max, y_max) grid dimensions
        
    Returns:
        List of (x, y) tuples for all cells in the neighborhood
    """
    x_max, y_max = grid_dims
    cells = []
    
    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            nx, ny = x + dx, y + dy
            # Check bounds (1-indexed)
            if 1 <= nx <= x_max and 1 <= ny <= y_max:
                cells.append((nx, ny))
    
    return cells
