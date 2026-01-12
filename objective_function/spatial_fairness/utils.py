"""
Utility functions for the Spatial Fairness term.

This module contains:
- Gini coefficient computation
- Data loading and preprocessing
- Service rate calculations
- Validation functions
"""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict


# =============================================================================
# GINI COEFFICIENT COMPUTATION
# =============================================================================

def compute_gini(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient for a distribution of values.
    
    The Gini coefficient measures inequality in a distribution:
    - G = 0: Perfect equality (all values identical)
    - G = 1: Maximum inequality (one entity has everything)
    
    Formula:
        G = 1 + (1/n) - (2 / (n² * μ)) * Σ(n-i+1) * x_(i)
    
    where x_(i) is the i-th smallest value.
    
    Args:
        values: Array of non-negative values
        
    Returns:
        Gini coefficient in [0, 1]
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    
    # Handle edge cases
    if n == 0:
        return 0.0
    
    if n == 1:
        return 0.0
    
    total = np.sum(values)
    if total == 0:
        return 0.0  # No inequality among zeros
    
    mean_value = total / n
    
    # Sort values in ascending order
    sorted_values = np.sort(values)
    
    # Compute weighted sum: Σ(n-i+1) * x_(i)
    # Weights are [n, n-1, n-2, ..., 2, 1]
    weights = np.arange(n, 0, -1)
    weighted_sum = np.sum(weights * sorted_values)
    
    # Gini formula
    gini = 1.0 + (1.0 / n) - (2.0 / (n * n * mean_value)) * weighted_sum
    
    # Ensure result is in valid range
    return float(np.clip(gini, 0.0, 1.0))


def compute_gini_alternative(values: np.ndarray) -> float:
    """
    Alternative Gini computation using mean absolute difference.
    
    Formula:
        G = Σ|xi - xj| / (2 * n² * μ)
    
    This is mathematically equivalent but useful for verification.
    
    Args:
        values: Array of non-negative values
        
    Returns:
        Gini coefficient in [0, 1]
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    
    if n == 0 or n == 1:
        return 0.0
    
    total = np.sum(values)
    if total == 0:
        return 0.0
    
    mean_value = total / n
    
    # Compute mean absolute difference directly
    # Using pairwise differences
    sorted_values = np.sort(values)
    
    # For sorted values, sum of |xi - xj| can be computed efficiently
    # |xi - xj| for sorted = xj - xi for j > i
    # Sum = 2 * sum_i (2*i - n - 1) * x_i for i from 0 to n-1
    indices = np.arange(n)
    weights = 2 * indices - n + 1
    mad = np.sum(weights * sorted_values)
    
    gini = mad / (n * n * mean_value)
    
    return float(np.clip(gini, 0.0, 1.0))


def compute_lorenz_curve(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Lorenz curve for a distribution.
    
    The Lorenz curve plots cumulative share of values against
    cumulative share of population (sorted from lowest to highest).
    
    Args:
        values: Array of non-negative values
        
    Returns:
        Tuple of (x_values, y_values) for the Lorenz curve
        where x = cumulative population share [0, 1]
        and y = cumulative value share [0, 1]
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    
    total = np.sum(values)
    if total == 0:
        # Perfect equality line when all zeros
        return np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1)
    
    sorted_values = np.sort(values)
    cumsum = np.cumsum(sorted_values)
    
    # Prepend 0 for starting point
    x = np.concatenate([[0], np.arange(1, n + 1) / n])
    y = np.concatenate([[0], cumsum / total])
    
    return x, y


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_pickup_dropoff_counts(filepath: str) -> Dict[Tuple[int, int, int, int], Tuple[int, int]]:
    """
    Load pickup/dropoff counts from pickle file.
    
    Args:
        filepath: Path to pickup_dropoff_counts.pkl
        
    Returns:
        Dictionary mapping (x, y, time, day) -> (pickup_count, dropoff_count)
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def aggregate_counts_by_period(
    data: Dict[Tuple[int, int, int, int], Tuple[int, int]],
    period_type: str = "hourly",
    days_filter: Optional[List[int]] = None,
    time_filter: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[Tuple[Tuple[int, int], Any], int], Dict[Tuple[Tuple[int, int], Any], int]]:
    """
    Aggregate pickup/dropoff counts by spatial cell and time period.
    
    Args:
        data: Raw pickup_dropoff_counts data
        period_type: One of "time_bucket", "hourly", "daily", "all"
        days_filter: Optional list of days to include (1-6)
        time_filter: Optional tuple of (start_bucket, end_bucket) to include
        
    Returns:
        Tuple of (pickups, dropoffs) where each is a dict:
        {((x, y), period): count}
    """
    pickups: Dict[Tuple[Tuple[int, int], Any], int] = defaultdict(int)
    dropoffs: Dict[Tuple[Tuple[int, int], Any], int] = defaultdict(int)
    
    for key, counts in data.items():
        x, y, time_bucket, day = key
        pickup_count, dropoff_count = counts
        
        # Apply filters
        if days_filter is not None and day not in days_filter:
            continue
        
        if time_filter is not None:
            if time_bucket < time_filter[0] or time_bucket > time_filter[1]:
                continue
        
        # Determine period based on period_type
        if period_type == "time_bucket":
            period = (time_bucket, day)
        elif period_type == "hourly":
            hour = (time_bucket - 1) // 12  # Convert to 0-23
            period = (hour, day)
        elif period_type == "daily":
            period = day
        elif period_type == "all":
            period = "all"
        else:
            raise ValueError(f"Unknown period_type: {period_type}")
        
        cell = (x, y)
        pickups[(cell, period)] += pickup_count
        dropoffs[(cell, period)] += dropoff_count
    
    return dict(pickups), dict(dropoffs)


def get_unique_periods(
    pickups: Dict[Tuple[Tuple[int, int], Any], int],
    dropoffs: Dict[Tuple[Tuple[int, int], Any], int],
) -> List[Any]:
    """
    Get list of unique time periods from aggregated data.
    
    Args:
        pickups: Aggregated pickup counts
        dropoffs: Aggregated dropoff counts
        
    Returns:
        Sorted list of unique periods
    """
    periods = set()
    for key in pickups.keys():
        periods.add(key[1])  # key is ((x, y), period)
    for key in dropoffs.keys():
        periods.add(key[1])
    
    return sorted(periods, key=lambda p: (p,) if not isinstance(p, tuple) else p)


def compute_period_duration_days(
    period: Any,
    period_type: str,
    num_days: float = 21.0,
) -> float:
    """
    Compute the duration of a period in days.
    
    Args:
        period: The period identifier
        period_type: Type of period ("time_bucket", "hourly", "daily", "all")
        num_days: Total number of days in dataset
        
    Returns:
        Duration in days
    """
    if period_type == "time_bucket":
        # Each time bucket is 5 minutes = 1/288 of a day
        return 1.0 / 288.0
    elif period_type == "hourly":
        # Each hour is 1/24 of a day
        return 1.0 / 24.0
    elif period_type == "daily":
        # Each day is 1 day
        return 1.0
    elif period_type == "all":
        # All data = total days
        return num_days
    else:
        raise ValueError(f"Unknown period_type: {period_type}")


# =============================================================================
# SERVICE RATE COMPUTATION
# =============================================================================

def compute_service_rates_for_period(
    pickups: Dict[Tuple[Tuple[int, int], Any], int],
    dropoffs: Dict[Tuple[Tuple[int, int], Any], int],
    period: Any,
    grid_dims: Tuple[int, int],
    num_taxis: int,
    period_duration_days: float,
    include_zero_cells: bool = True,
    data_is_one_indexed: bool = True,
    min_activity_threshold: int = 0,
    active_taxis_data: Optional[Dict[Tuple, int]] = None,
    active_taxis_fallback: int = 1,
    period_type: str = "hourly",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Departure Service Rate (DSR) and Arrival Service Rate (ASR)
    for all cells in a given period.
    
    DSR = pickups / (N^p * period_duration)
    ASR = dropoffs / (N^p * period_duration)
    
    Where N^p is either:
        - num_taxis (constant) if active_taxis_data is None
        - active_taxi_count for each cell if active_taxis_data is provided
    
    Args:
        pickups: Aggregated pickup counts {((x, y), period): count}
        dropoffs: Aggregated dropoff counts
        period: The period to compute rates for
        grid_dims: Grid dimensions (x_cells, y_cells)
        num_taxis: Number of active taxis (used when active_taxis_data is None)
        period_duration_days: Duration of period in days
        include_zero_cells: Whether to include cells with zero activity
        data_is_one_indexed: Whether data uses 1-based indexing
        min_activity_threshold: Minimum total activity to include cell
        active_taxis_data: Pre-computed active taxi counts (optional)
        active_taxis_fallback: Value when active_taxi lookup returns 0
        period_type: Period type for active_taxis lookup
        
    Returns:
        Tuple of (dsr_values, asr_values) as numpy arrays
    """
    x_dim, y_dim = grid_dims
    
    dsr_list = []
    asr_list = []
    
    # Determine index range based on data indexing
    x_start = 1 if data_is_one_indexed else 0
    y_start = 1 if data_is_one_indexed else 0
    x_end = x_dim + x_start
    y_end = y_dim + y_start
    
    for x in range(x_start, x_end):
        for y in range(y_start, y_end):
            cell = (x, y)
            key = (cell, period)
            
            pickup_count = pickups.get(key, 0)
            dropoff_count = dropoffs.get(key, 0)
            
            total_activity = pickup_count + dropoff_count
            
            # Apply activity threshold
            if not include_zero_cells and total_activity == 0:
                continue
            
            if total_activity < min_activity_threshold:
                continue
            
            # Determine N^p (number of active taxis)
            if active_taxis_data is not None:
                # Use cell-specific active taxi count
                n_taxis = get_active_taxi_count(
                    active_taxis_data, x, y, period, period_type, active_taxis_fallback
                )
            else:
                # Use constant num_taxis
                n_taxis = num_taxis
            
            normalization = n_taxis * period_duration_days
            
            dsr = pickup_count / normalization if normalization > 0 else 0
            asr = dropoff_count / normalization if normalization > 0 else 0
            
            dsr_list.append(dsr)
            asr_list.append(asr)
    
    return np.array(dsr_list), np.array(asr_list)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_pickup_dropoff_data(
    data: Dict[Tuple[int, int, int, int], Tuple[int, int]]
) -> Tuple[bool, List[str]]:
    """
    Validate pickup/dropoff counts data.
    
    Args:
        data: Raw pickup_dropoff_counts data
        
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    
    if len(data) == 0:
        errors.append("Data is empty")
        return False, errors
    
    # Check structure
    sample_key = list(data.keys())[0]
    sample_value = list(data.values())[0]
    
    if not isinstance(sample_key, tuple) or len(sample_key) != 4:
        errors.append(f"Keys must be 4-tuples, got: {type(sample_key)}")
    
    if not isinstance(sample_value, (tuple, list)) or len(sample_value) != 2:
        errors.append(f"Values must be 2-element sequences, got: {type(sample_value)}")
    
    if errors:
        return False, errors
    
    # Check ranges
    x_vals = [k[0] for k in data.keys()]
    y_vals = [k[1] for k in data.keys()]
    time_vals = [k[2] for k in data.keys()]
    day_vals = [k[3] for k in data.keys()]
    
    # Check for negative counts
    neg_pickups = sum(1 for v in data.values() if v[0] < 0)
    neg_dropoffs = sum(1 for v in data.values() if v[1] < 0)
    
    if neg_pickups > 0:
        errors.append(f"{neg_pickups} entries have negative pickup counts")
    if neg_dropoffs > 0:
        errors.append(f"{neg_dropoffs} entries have negative dropoff counts")
    
    # Report statistics (not errors, just info)
    stats = {
        'x_range': (min(x_vals), max(x_vals)),
        'y_range': (min(y_vals), max(y_vals)),
        'time_range': (min(time_vals), max(time_vals)),
        'day_range': (min(day_vals), max(day_vals)),
        'total_keys': len(data),
    }
    
    return len(errors) == 0, errors


def get_data_statistics(
    data: Dict[Tuple[int, int, int, int], Tuple[int, int]]
) -> Dict[str, Any]:
    """
    Compute statistics about the pickup/dropoff data.
    
    Args:
        data: Raw pickup_dropoff_counts data
        
    Returns:
        Dictionary of statistics
    """
    if len(data) == 0:
        return {'error': 'Empty data'}
    
    pickups = [v[0] for v in data.values()]
    dropoffs = [v[1] for v in data.values()]
    
    x_vals = [k[0] for k in data.keys()]
    y_vals = [k[1] for k in data.keys()]
    time_vals = [k[2] for k in data.keys()]
    day_vals = [k[3] for k in data.keys()]
    
    return {
        'total_keys': len(data),
        'total_pickups': sum(pickups),
        'total_dropoffs': sum(dropoffs),
        'pickup_stats': {
            'mean': np.mean(pickups),
            'std': np.std(pickups),
            'min': min(pickups),
            'max': max(pickups),
            'nonzero_count': sum(1 for p in pickups if p > 0),
        },
        'dropoff_stats': {
            'mean': np.mean(dropoffs),
            'std': np.std(dropoffs),
            'min': min(dropoffs),
            'max': max(dropoffs),
            'nonzero_count': sum(1 for d in dropoffs if d > 0),
        },
        'spatial': {
            'x_range': (min(x_vals), max(x_vals)),
            'y_range': (min(y_vals), max(y_vals)),
            'unique_cells': len(set((k[0], k[1]) for k in data.keys())),
        },
        'temporal': {
            'time_range': (min(time_vals), max(time_vals)),
            'day_range': (min(day_vals), max(day_vals)),
            'unique_days': len(set(day_vals)),
        },
    }


# =============================================================================
# ACTIVE TAXIS DATA FUNCTIONS
# =============================================================================

def load_active_taxis_data(filepath: str) -> Dict[Tuple, int]:
    """
    Load pre-computed active taxi counts from pickle file.
    
    The active_taxis data contains counts of unique taxis present in each
    n×n neighborhood during each time period.
    
    Key formats depend on period_type used during generation:
        - hourly: (x, y, hour, day) -> count
        - daily: (x, y, day) -> count
        - time_bucket: (x, y, time_bin, day) -> count
        - all: (x, y, 'all') -> count
    
    Args:
        filepath: Path to active_taxis_*.pkl file
        
    Returns:
        Dictionary mapping (x, y, *period_key) -> active_taxi_count
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    import pickle
    from pathlib import Path
    
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Active taxis file not found: {filepath}")
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Invalid active_taxis data: expected non-empty dict, got {type(data)}")
    
    # Handle nested structure: extract 'data' if present
    if 'data' in data and isinstance(data['data'], dict):
        # This is the new format with 'data', 'stats', 'config' keys
        # Return just the data mapping
        return data['data']
    
    # Otherwise assume it's already in the correct format
    return data


def get_active_taxi_count(
    active_taxis_data: Dict[Tuple, int],
    x: int,
    y: int,
    period: Any,
    period_type: str,
    fallback: int = 1,
) -> int:
    """
    Look up the number of active taxis for a specific cell and period.
    
    Args:
        active_taxis_data: Loaded active_taxis data
        x: Grid cell x coordinate (1-indexed)
        y: Grid cell y coordinate (1-indexed)
        period: Period identifier (varies by period_type)
            - hourly: (hour, day) tuple
            - daily: day integer
            - time_bucket: (time_bin, day) tuple
            - all: "all" string
        period_type: Type of period aggregation
        fallback: Value to return if lookup fails or returns 0
        
    Returns:
        Number of active taxis, or fallback if not found or zero
    """
    # Build the lookup key based on period_type
    if period_type == "hourly":
        # period is (hour, day), key is (x, y, hour, day)
        hour, day = period
        key = (x, y, hour, day)
    elif period_type == "daily":
        # period is day, key is (x, y, day)
        key = (x, y, period)
    elif period_type == "time_bucket":
        # period is (time_bin, day), key is (x, y, time_bin, day)
        time_bin, day = period
        key = (x, y, time_bin, day)
    elif period_type == "all":
        # period is "all", key is (x, y, 'all')
        key = (x, y, 'all')
    else:
        return fallback
    
    # Look up the count
    count = active_taxis_data.get(key, 0)
    
    # Return fallback if count is 0 to avoid division by zero
    return count if count > 0 else fallback


def get_active_taxi_count_for_cell(
    active_taxis_data: Dict[Tuple, int],
    cell: Tuple[int, int],
    period: Any,
    period_type: str,
    fallback: int = 1,
) -> int:
    """
    Convenience function to look up active taxi count for a cell.
    
    Args:
        active_taxis_data: Loaded active_taxis data
        cell: (x, y) grid cell coordinates
        period: Period identifier
        period_type: Type of period aggregation
        fallback: Value to return if lookup fails
        
    Returns:
        Number of active taxis
    """
    x, y = cell
    return get_active_taxi_count(active_taxis_data, x, y, period, period_type, fallback)


def get_active_taxis_statistics(
    active_taxis_data: Dict[Tuple, int]
) -> Dict[str, Any]:
    """
    Compute statistics about the active_taxis data.
    
    Args:
        active_taxis_data: Loaded active_taxis data (can be nested dict with 'data', 'stats', 'config')
        
    Returns:
        Dictionary of statistics
    """
    # Handle nested structure from pickle files
    if isinstance(active_taxis_data, dict) and 'data' in active_taxis_data:
        # Extract the actual data mapping
        data = active_taxis_data['data']
        # Return pre-computed stats if available
        if 'stats' in active_taxis_data:
            stats = active_taxis_data['stats']
            return {
                'total_keys': stats.get('total_output_keys', 0),
                'period_type_guess': 'from_config',
                'count_stats': {
                    'mean': stats.get('avg_active_taxis_per_cell', 0.0),
                    'std': 0.0,
                    'min': 0,
                    'max': stats.get('max_active_taxis_in_cell', 0),
                    'median': 0.0,
                    'zero_count': 0,
                    'nonzero_count': stats.get('total_output_keys', 0),
                },
                'spatial': {
                    'x_range': (0, 0),
                    'y_range': (0, 0),
                    'unique_cells': stats.get('unique_cells', 0),
                },
            }
    else:
        data = active_taxis_data
    
    if len(data) == 0:
        return {'error': 'Empty data'}
    
    counts = list(data.values())
    keys = list(data.keys())
    
    # Determine period_type from key structure
    sample_key = keys[0]
    if len(sample_key) == 4:
        # (x, y, time_component, day) - hourly or time_bucket
        period_type_guess = "hourly_or_time_bucket"
    elif len(sample_key) == 3:
        if sample_key[2] == 'all':
            period_type_guess = "all"
        else:
            period_type_guess = "daily"
    else:
        period_type_guess = "unknown"
    
    return {
        'total_keys': len(active_taxis_data),
        'period_type_guess': period_type_guess,
        'count_stats': {
            'mean': np.mean(counts),
            'std': np.std(counts),
            'min': min(counts),
            'max': max(counts),
            'median': np.median(counts),
            'zero_count': sum(1 for c in counts if c == 0),
            'nonzero_count': sum(1 for c in counts if c > 0),
        },
        'spatial': {
            'x_range': (min(k[0] for k in keys), max(k[0] for k in keys)),
            'y_range': (min(k[1] for k in keys), max(k[1] for k in keys)),
            'unique_cells': len(set((k[0], k[1]) for k in keys)),
        },
    }


def validate_active_taxis_period_alignment(
    active_taxis_data: Dict[Tuple, int],
    config_period_type: str,
) -> Tuple[bool, str]:
    """
    Validate that active_taxis data period type matches config period type.
    
    Args:
        active_taxis_data: Loaded active_taxis data
        config_period_type: Period type from SpatialFairnessConfig
        
    Returns:
        Tuple of (is_aligned, message)
    """
    if len(active_taxis_data) == 0:
        return False, "Active taxis data is empty"
    
    sample_key = list(active_taxis_data.keys())[0]
    key_len = len(sample_key)
    
    # Determine expected key length for each period_type
    expected_lengths = {
        "hourly": 4,       # (x, y, hour, day)
        "time_bucket": 4,  # (x, y, time_bin, day)
        "daily": 3,        # (x, y, day)
        "all": 3,          # (x, y, 'all')
    }
    
    expected_len = expected_lengths.get(config_period_type)
    if expected_len is None:
        return False, f"Unknown period_type: {config_period_type}"
    
    if key_len != expected_len:
        return False, (
            f"Period type mismatch: config expects {config_period_type} "
            f"(key length {expected_len}), but data has key length {key_len}. "
            f"Sample key: {sample_key}"
        )
    
    # For 'all' period_type, verify the third element is 'all'
    if config_period_type == "all" and sample_key[2] != 'all':
        return False, (
            f"Period type mismatch: config expects 'all' but data key third element "
            f"is {sample_key[2]}, not 'all'"
        )
    
    return True, "Period types are aligned"
