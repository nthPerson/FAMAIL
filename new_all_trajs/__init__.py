"""
New All Trajs Dataset Generation Tool

This package provides tools for generating the all_trajs.pkl dataset in two steps:
1. Extract passenger-seeking trajectories from raw GPS data
2. Generate state features using the cGAIL feature generation logic
"""

from .config import (
    GlobalBounds, 
    ProcessingConfig
)

from .step1_processor import (
    load_raw_data, 
    compute_global_bounds, 
    gps_to_grid, 
    seconds_to_time_bin
)

__all__ = [
    GlobalBounds,
    ProcessingConfig,
    load_raw_data,
    compute_global_bounds,
    gps_to_grid,
    seconds_to_time_bin
]   



__version__ = "1.0.0"
