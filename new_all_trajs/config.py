"""
Configuration settings for the New All Trajs dataset generation tool.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    
    # Grid quantization parameters
    grid_size: float = 0.01  # degrees
    time_interval: int = 5   # minutes (288 buckets per day)
    exclude_sunday: bool = True
    
    # Index offsets for alignment with existing datasets
    # These match the pickup_dropoff_counts and active_taxis processors
    x_grid_offset: int = 1  # Add 1 to x_grid indices
    y_grid_offset: int = 1  # Add 1 to y_grid indices
    time_offset: int = 0    # No offset for time buckets (0-287)
    
    # Minimum trajectory length (in states) to include
    min_trajectory_length: int = 2
    
    # Maximum trajectory length (in states) - for filtering outliers
    max_trajectory_length: Optional[int] = None
    
    # Paths
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
    
    # Default output filenames
    step1_output_filename: str = "passenger_seeking_trajs.pkl"
    step2_output_filename: str = "new_all_trajs.pkl"


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
    
    @classmethod
    def from_dict(cls, d: dict) -> 'GlobalBounds':
        return cls(
            lat_min=d["lat_min"],
            lat_max=d["lat_max"],
            lon_min=d["lon_min"],
            lon_max=d["lon_max"],
        )


@dataclass
class Step1Stats:
    """Statistics from Step 1 processing (passenger-seeking trajectory extraction)."""
    total_raw_records: int = 0
    records_after_sunday_filter: int = 0
    unique_drivers: int = 0
    total_trajectories: int = 0
    total_states: int = 0
    avg_trajectory_length: float = 0.0
    min_trajectory_length: int = 0
    max_trajectory_length: int = 0
    trajectories_per_driver_avg: float = 0.0
    processing_time_seconds: float = 0.0
    global_bounds: Optional[GlobalBounds] = None
    
    def to_dict(self) -> dict:
        return {
            'total_raw_records': self.total_raw_records,
            'records_after_sunday_filter': self.records_after_sunday_filter,
            'unique_drivers': self.unique_drivers,
            'total_trajectories': self.total_trajectories,
            'total_states': self.total_states,
            'avg_trajectory_length': self.avg_trajectory_length,
            'min_trajectory_length': self.min_trajectory_length,
            'max_trajectory_length': self.max_trajectory_length,
            'trajectories_per_driver_avg': self.trajectories_per_driver_avg,
            'processing_time_seconds': self.processing_time_seconds,
            'global_bounds': self.global_bounds.to_dict() if self.global_bounds else None,
        }


@dataclass 
class Step2Stats:
    """Statistics from Step 2 processing (state feature generation)."""
    input_trajectories: int = 0
    input_states: int = 0
    output_states: int = 0
    unique_drivers: int = 0
    feature_vector_length: int = 126
    poi_features_count: int = 21
    window_features_count: int = 100  # 4 x 25 features
    traffic_keys_found: int = 0
    volume_keys_found: int = 0
    traffic_keys_missing: int = 0
    volume_keys_missing: int = 0
    processing_time_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'input_trajectories': self.input_trajectories,
            'input_states': self.input_states,
            'output_states': self.output_states,
            'unique_drivers': self.unique_drivers,
            'feature_vector_length': self.feature_vector_length,
            'poi_features_count': self.poi_features_count,
            'window_features_count': self.window_features_count,
            'traffic_keys_found': self.traffic_keys_found,
            'volume_keys_found': self.volume_keys_found,
            'traffic_keys_missing': self.traffic_keys_missing,
            'volume_keys_missing': self.volume_keys_missing,
            'processing_time_seconds': self.processing_time_seconds,
        }


# Action code mapping for movement directions
ACTION_CODES = {
    'north': 0,      # y increases, x same
    'northeast': 1,  # y increases, x increases  
    'east': 2,       # y same, x increases
    'southeast': 3,  # y decreases, x increases
    'south': 4,      # y decreases, x same
    'southwest': 5,  # y decreases, x decreases
    'west': 6,       # y same, x decreases
    'northwest': 7,  # y increases, x decreases
    'stay': 8,       # no movement
    'stop': 9,       # terminal action
}

# Feature index ranges in state vector
FEATURE_INDICES = {
    'x_grid': 0,
    'y_grid': 1,
    'time_bucket': 2,
    'day_index': 3,
    'poi_manhattan_distance': (4, 24),  # 21 POI distances
    'pickup_count_norm': (25, 49),      # 25 values (5x5 window)
    'traffic_volume_norm': (50, 74),    # 25 values (5x5 window)
    'traffic_speed_norm': (75, 99),     # 25 values (5x5 window)
    'traffic_wait_norm': (100, 124),    # 25 values (5x5 window)
    'action_code': 125,
}

# Normalization constants (from the original cGAIL notebook)
NORMALIZATION_CONSTANTS = {
    'pickup_count': {'baseline': 1.7411234687501456, 'scale': 8.68915141275395},
    'traffic_volume': {'baseline': 241.16497174549497, 'scale': 864.8101217071693},
    'traffic_speed': {'baseline': 0.009096451857715626, 'scale': 0.007786749371066213},
    'traffic_wait': {'baseline': 9.214922479890125, 'scale': 20.839610665761285},
}
