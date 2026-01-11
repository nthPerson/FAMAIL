"""
Active Taxis Dataset Generation Tool.

This tool generates datasets that count the number of active taxis in an n×n grid
surrounding each cell for each time period. The generated datasets are used by the
Spatial Fairness and Causal Fairness objective function terms.

Usage:
    from active_taxis import ActiveTaxisConfig, generate_active_taxi_counts
    
    config = ActiveTaxisConfig(neighborhood_size=2, period_type='hourly')
    counts, stats = generate_active_taxi_counts(config)
"""

from .config import (
    ActiveTaxisConfig,
    DEFAULT_CONFIG,
    HOURLY_CONFIG,
    DAILY_CONFIG,
    TIME_BUCKET_CONFIG,
)

from .processor import (
    GlobalBounds,
    ProcessingStats,
    load_raw_data,
    compute_global_bounds,
    gps_to_grid,
    timestamp_to_time_bin,
    timestamp_to_day,
    timestamp_to_hour,
)

from .generation import (
    generate_active_taxi_counts,
    generate_test_dataset,
    save_output,
    load_output,
    get_active_taxi_count,
)

__all__ = [
    # Config
    'ActiveTaxisConfig',
    'DEFAULT_CONFIG',
    'HOURLY_CONFIG',
    'DAILY_CONFIG',
    'TIME_BUCKET_CONFIG',
    # Processor
    'GlobalBounds',
    'ProcessingStats',
    'load_raw_data',
    'compute_global_bounds',
    'gps_to_grid',
    'timestamp_to_time_bin',
    'timestamp_to_day',
    'timestamp_to_hour',
    # Generation
    'generate_active_taxi_counts',
    'generate_test_dataset',
    'save_output',
    'load_output',
    'get_active_taxi_count',
]
