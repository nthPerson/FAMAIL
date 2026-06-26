"""Configuration constants for the unified GPS source-data generation tool.

All constants are opinionated defaults — there are no runtime config flags.
If a constant needs to change, edit this file rather than adding a CLI option.
"""
from __future__ import annotations
from pathlib import Path

# Spatial grid (matches famail_temporal.config)
GRID_SIZE_DEG: float = 0.01
X_GRID_MAX: int = 48
Y_GRID_MAX: int = 90
X_GRID_OFFSET: int = 1
Y_GRID_OFFSET: int = 1

# Time quantization
TIME_INTERVAL_MIN: int = 5
TIME_BUCKET_MAX: int = 288
HOUR_MAX: int = 23

# Day filter — weekdays only (permanent project decision)
WEEKDAY_DAYS: frozenset[int] = frozenset({1, 2, 3, 4, 5})

# Active-taxis neighborhood
NEIGHBORHOOD_SIZE: int = 5
NEIGHBORHOOD_K: int = NEIGHBORHOOD_SIZE // 2

# Profile features
PROFILE_FEATURE_NAMES: tuple[str, ...] = (
    "home_x", "home_y",
    "shift_start", "shift_end",
    "freq_grid_x", "freq_grid_y",
    "avg_seeking_dist", "avg_seeking_time",
    "avg_driving_dist", "avg_driving_time",
    "num_trips_per_day",
)
N_PROFILE_FEATURES: int = len(PROFILE_FEATURE_NAMES)
PROFILE_SHIFT_LOW_PCT: float = 5.0
PROFILE_SHIFT_HIGH_PCT: float = 95.0

# Per-trajectory removal warning threshold
REMOVAL_RATE_WARN_THRESHOLD: float = 0.05

# Maximum plausible duration for a single seeking or driving trajectory, expressed
# as a count of 5-minute time_buckets. A seeking or driving episode is typically
# minutes to at most a few hours; anything longer is treated as an artifact of
# segment extraction (e.g., a segment "stitched" across a weekend or off-duty
# period because no passenger-indicator transition happened). 96 buckets = 8
# hours — the length of a standard work day. Trajectories whose elapsed duration
# exceeds this are dropped as `implausibly_long`.
MAX_TRAJECTORY_DURATION_BUCKETS: int = 96

# Required driver count
EXPECTED_N_DRIVERS: int = 50

# I/O defaults
DEFAULT_RAW_INPUT_DIR: Path = Path("raw_data")
DEFAULT_OUTPUT_DIR: Path = Path("famail_temporal/source_data")

RAW_INPUT_FILENAMES: tuple[str, ...] = (
    "taxi_record_07_50drivers.pkl",
    "taxi_record_08_50drivers.pkl",
    "taxi_record_09_50drivers.pkl",
)

OUT_PICKUP_DROPOFF: str = "pickup_dropoff_counts.pkl"
OUT_ACTIVE_TAXIS: str = "active_taxis_5x5_hourly.pkl"
OUT_PASSENGER_SEEKING: str = "passenger_seeking_trajs.pkl"
OUT_MS_SEEKING: str = "ms_seeking_trajs.pkl"
OUT_MS_DRIVING: str = "ms_driving_trajs.pkl"
OUT_MS_PROFILE: str = "ms_profile_features.pkl"
OUT_MS_SEEKING_DAYS: str = "ms_seeking_calendar_days.pkl"
OUT_MS_DRIVING_DAYS: str = "ms_driving_calendar_days.pkl"
OUT_CALENDAR_DAY_MAP: str = "calendar_day_map.pkl"
OUT_DRIVER_MAPPING: str = "driver_index_mapping.pkl"
OUT_METADATA: str = "processing_metadata.json"

RANDOM_SEED: int = 0
OUTPUT_FORMAT_VERSION: str = "1.0.0"

# --- stuck-GPS sink filter (provisional; finalize from Stage-0 dry-run) ---
STUCK_GPS_MIN_PICKUPS: int = 1000        # absolute phantom-pickup floor
STUCK_GPS_MAX_DROPOFF_RATIO: float = 0.02  # near-zero dropoffs at the frozen coord
STUCK_GPS_COORD_PRECISION: int = 6       # lat/lon rounding (decimals)
STUCK_GPS_DROP: bool = True              # drop flagged pickup pings (vs suppress)
STUCK_GPS_EXPECTED_CELLS: frozenset = frozenset({(28, 52), (20, 28), (28, 28), (24, 5), (22, 46), (17, 38)})
