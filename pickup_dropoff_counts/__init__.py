"""
Pickup/Dropoff Counts Processor

A data processing tool for counting taxi pickup and dropoff events
from raw GPS trajectory data.
"""

from .processor import (
    ProcessingConfig,
    GlobalBounds,
    ProcessingStats,
    ValidationResult,
    process_data,
    save_output,
    load_output,
    validate_against_existing,
    load_existing_volume_pickups,
    gps_to_grid,
    timestamp_to_time_bin,
    timestamp_to_day,
    detect_transitions,
)

__all__ = [
    'ProcessingConfig',
    'GlobalBounds', 
    'ProcessingStats',
    'ValidationResult',
    'process_data',
    'save_output',
    'load_output',
    'validate_against_existing',
    'load_existing_volume_pickups',
    'gps_to_grid',
    'timestamp_to_time_bin',
    'timestamp_to_day',
    'detect_transitions',
]
