"""
Configuration for the New All Trajs Dataset Generator.

This module defines configuration dataclasses and constants for consistent
processing across all dataset generation operations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    
    # Grid quantization parameters (aligned with other datasets)
    grid_size: float = 0.01  # degrees per grid cell
    time_interval: int = 5   # minutes per time bucket (288 buckets per day)
    exclude_sunday: bool = True
    
    # Index offsets for alignment with existing datasets
    x_grid_offset: int = 1  # Applied to latitude grid indices
    y_grid_offset: int = 1  # Applied to longitude grid indices
    time_offset: int = 1    # Applied to time bucket indices (0-based -> 1-based)
    
    # Grid bounds (for alignment with existing datasets)
    # When set (not None), these override the empirical max from data
    x_grid_max: Optional[int] = None  # Set to match existing dataset
    y_grid_max: Optional[int] = None  # Set to match existing dataset
    
    # Paths (defaults will be set relative to module location)
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "raw_data")
    source_data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "source_data")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output")
    
    # Input files
    input_files: Tuple[str, ...] = (
        "taxi_record_07_50drivers.pkl",
        "taxi_record_08_50drivers.pkl",
        "taxi_record_09_50drivers.pkl",
    )
    
    # Feature data files
    traffic_file: str = "latest_traffic.pkl"
    volume_file: str = "latest_volume_pickups.pkl"
    train_airport_file: str = "train_airport.pkl"
    
    # Output file name
    output_filename: str = "new_all_trajs.pkl"


@dataclass
class GlobalBounds:
    """Stores the global GPS coordinate bounds for quantization."""
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
    total_drivers: int = 0
    total_trajectories: int = 0
    total_states: int = 0
    processing_time_seconds: float = 0.0
    global_bounds: Optional[GlobalBounds] = None


# Normalization constants for state features
# These are the same constants used in the original net-dis-cGAIL.ipynb
NORMALIZATION_CONSTANTS = {
    # Pickup counts normalization (from latest_volume_pickups)
    "pickup_mean": 1.7411234687501456,
    "pickup_std": 8.68915141275395,
    
    # Traffic volume normalization (from latest_volume_pickups)
    "volume_mean": 241.16497174549497,
    "volume_std": 864.8101217071693,
    
    # Traffic speed normalization (from latest_traffic)
    "speed_mean": 0.009096451857715626,
    "speed_std": 0.007786749371066213,
    
    # Traffic waiting time normalization (from latest_traffic)
    "wait_mean": 9.214922479890125,
    "wait_std": 20.839610665761285,
}


# Action code mapping
ACTION_CODES = {
    0: "Move north (y increases)",
    1: "Move northeast",
    2: "Move east (x increases)",
    3: "Move southeast",
    4: "Move south (y decreases)",
    5: "Move southwest",
    6: "Move west (x decreases)",
    7: "Move northwest",
    8: "Stay in place",
    9: "Stop (terminal action)",
}


# State vector field descriptions
STATE_VECTOR_FIELDS = {
    (0, 0): ("x_grid", "int", "Grid index for longitude position"),
    (1, 1): ("y_grid", "int", "Grid index for latitude position"),
    (2, 2): ("time_bucket", "int", "Discretized time-of-day slot ∈ [0, 287]"),
    (3, 3): ("day_index", "int", "Day-of-week indicator"),
    (4, 24): ("poi_manhattan_distance", "float", "Manhattan distances to 21 points of interest"),
    (25, 49): ("pickup_count_norm", "float", "Normalized pickup counts (5×5 window)"),
    (50, 74): ("traffic_volume_norm", "float", "Normalized traffic volumes (5×5 window)"),
    (75, 99): ("traffic_speed_norm", "float", "Normalized traffic speeds (5×5 window)"),
    (100, 124): ("traffic_wait_norm", "float", "Normalized traffic waiting times (5×5 window)"),
    (125, 125): ("action_code", "int", "Movement action label (0-9)"),
}
