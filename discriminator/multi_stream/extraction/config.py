"""
Configuration for the dual trajectory extraction and profile feature computation tool.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


def _project_root() -> Path:
    """Resolve project root from this file's location."""
    return Path(__file__).resolve().parents[3]


@dataclass
class ExtractionConfig:
    """Configuration for dual trajectory extraction."""

    # Grid quantization (must match project conventions)
    grid_size: float = 0.01
    time_interval: int = 5  # minutes → 288 buckets/day
    x_grid_offset: int = 1  # 1-indexed output
    y_grid_offset: int = 1
    exclude_weekends: bool = True

    # GPS quality filter (Shenzhen bounding box)
    lat_min_bound: float = 22.44
    lat_max_bound: float = 22.87
    lon_min_bound: float = 113.75
    lon_max_bound: float = 114.65

    # Segment filtering (addresses noisy micro-transitions from Phase 1)
    min_segment_states: int = 5  # minimum states AFTER deduplication
    min_segment_duration_sec: int = 300  # 5 minutes minimum
    max_segment_states: int = 1000

    # Paths (defaults relative to project root)
    raw_data_dir: Path = field(default_factory=lambda: _project_root() / "raw_data")
    feature_data_dir: Path = field(
        default_factory=lambda: _project_root() / "cGAIL_data_and_processing" / "Data" / "features_condition"
    )
    output_dir: Path = field(
        default_factory=lambda: _project_root() / "discriminator" / "multi_stream" / "extracted_data"
    )

    # Input files
    input_files: Tuple[str, ...] = (
        "taxi_record_07_50drivers.pkl",
        "taxi_record_08_50drivers.pkl",
        "taxi_record_09_50drivers.pkl",
    )

    # Profile feature pickle files
    home_loc_file: str = "home_loc_plates_dict_all.pkl"
    start_finish_file: str = "start_finishing_time.pkl"

    def to_dict(self) -> dict:
        return {
            "grid_size": self.grid_size,
            "time_interval": self.time_interval,
            "x_grid_offset": self.x_grid_offset,
            "y_grid_offset": self.y_grid_offset,
            "exclude_weekends": self.exclude_weekends,
            "lat_min_bound": self.lat_min_bound,
            "lat_max_bound": self.lat_max_bound,
            "lon_min_bound": self.lon_min_bound,
            "lon_max_bound": self.lon_max_bound,
            "min_segment_states": self.min_segment_states,
            "min_segment_duration_sec": self.min_segment_duration_sec,
            "max_segment_states": self.max_segment_states,
            "raw_data_dir": str(self.raw_data_dir),
            "feature_data_dir": str(self.feature_data_dir),
            "output_dir": str(self.output_dir),
            "input_files": list(self.input_files),
            "home_loc_file": self.home_loc_file,
            "start_finish_file": self.start_finish_file,
        }


@dataclass
class ExtractionStats:
    """Statistics from the dual trajectory extraction process."""

    # Record-level counts
    total_raw_records: int = 0
    records_outside_bbox: int = 0
    records_weekend_filtered: int = 0
    records_timestamp_invalid: int = 0

    # Driver-level counts
    drivers_processed: int = 0

    # Trajectory counts
    seeking_trajectories: int = 0
    driving_trajectories: int = 0
    seeking_states_total: int = 0
    driving_states_total: int = 0

    # Deduplication and filtering
    states_deduplicated: int = 0
    segments_filtered_too_short: int = 0
    segments_filtered_too_brief: int = 0

    # Length distributions (populated after extraction)
    seeking_length_stats: Optional[Dict[str, float]] = None
    driving_length_stats: Optional[Dict[str, float]] = None

    # Timing
    processing_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_raw_records": self.total_raw_records,
            "records_outside_bbox": self.records_outside_bbox,
            "records_weekend_filtered": self.records_weekend_filtered,
            "records_timestamp_invalid": self.records_timestamp_invalid,
            "drivers_processed": self.drivers_processed,
            "seeking_trajectories": self.seeking_trajectories,
            "driving_trajectories": self.driving_trajectories,
            "seeking_states_total": self.seeking_states_total,
            "driving_states_total": self.driving_states_total,
            "states_deduplicated": self.states_deduplicated,
            "segments_filtered_too_short": self.segments_filtered_too_short,
            "segments_filtered_too_brief": self.segments_filtered_too_brief,
            "seeking_length_stats": self.seeking_length_stats,
            "driving_length_stats": self.driving_length_stats,
            "processing_time_seconds": self.processing_time_seconds,
        }
