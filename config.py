"""
Configuration for Active Taxis Dataset Generation.

This module defines configuration options for generating active taxi count datasets.
The configuration ensures consistency with the pickup_dropoff_counts tool and
the Spatial Fairness / Causal Fairness objective function terms.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Literal


@dataclass
class ActiveTaxisConfig:
    """
    Configuration for active taxi count generation.
    
    The active taxi count represents the number of unique taxis that were present
    in an n×n grid neighborhood surrounding each cell during a given time period.
    
    Attributes:
        neighborhood_size: The "k" value where neighborhood is (2k+1) × (2k+1) cells.
                          k=0 means only the cell itself (1×1)
                          k=1 means 3×3 neighborhood
                          k=2 means 5×5 neighborhood
                          Default is k=2 (5×5 neighborhood)
        period_type: Time aggregation level for counting active taxis.
                    'time_bucket' = 5-minute intervals (288 per day)
                    'hourly' = 1-hour intervals (24 per day)
                    'daily' = entire day
                    'all' = aggregate across all time
        grid_size: Size of each grid cell in degrees (must match pickup_dropoff_counts)
        grid_dims: Tuple of (x_max, y_max) grid dimensions (must match pickup_dropoff_counts)
        exclude_sunday: If True, Sunday data is excluded (days 1-6 only)
        
        # Index offsets (must match pickup_dropoff_counts for consistency)
        x_grid_offset: Offset added to x_grid indices (default 2)
        y_grid_offset: Offset added to y_grid indices (default 1)
        time_offset: Offset added to time indices (default 1 for 1-based indexing)
        
        # Test mode
        test_mode: If True, process only a subset of data for validation
        test_sample_size: Number of drivers to sample in test mode
        test_days: Number of days to process in test mode (per month)
        
        # Paths
        raw_data_dir: Directory containing raw taxi GPS pickle files
        output_dir: Directory to save generated datasets
        
        # Input files to process
        input_files: List of raw data pickle files to process
    """
    
    # Neighborhood configuration
    neighborhood_size: int = 2  # k value, results in (2k+1)×(2k+1) neighborhood
    
    # Time period aggregation
    period_type: Literal['time_bucket', 'hourly', 'daily', 'all'] = 'hourly'
    
    # Grid configuration (must match pickup_dropoff_counts)
    grid_size: float = 0.01  # degrees
    grid_dims: Tuple[int, int] = (48, 90)  # (x_max, y_max)
    exclude_sunday: bool = True
    
    # Index offsets (must match pickup_dropoff_counts for consistency)
    x_grid_offset: int = 2
    y_grid_offset: int = 1
    time_offset: int = 1  # 0-based to 1-based
    
    # Test mode configuration
    test_mode: bool = False
    test_sample_size: int = 5  # number of drivers
    test_days: int = 3  # days per month
    
    # Paths
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "raw_data")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output")
    
    # Input files
    input_files: Tuple[str, ...] = (
        "taxi_record_07_50drivers.pkl",
    )
    
    @property
    def neighborhood_dims(self) -> int:
        """Return the actual neighborhood dimensions (2k+1)."""
        return 2 * self.neighborhood_size + 1
    
    @property
    def time_bins_per_day(self) -> int:
        """Return the number of time bins per day based on period_type."""
        if self.period_type == 'time_bucket':
            return 288  # 5-minute intervals
        elif self.period_type == 'hourly':
            return 24
        elif self.period_type == 'daily':
            return 1
        else:  # 'all'
            return 1
    
    def get_output_filename(self) -> str:
        """Generate a filename based on configuration."""
        parts = [
            "active_taxis",
            f"n{self.neighborhood_dims}x{self.neighborhood_dims}",
            self.period_type,
        ]
        if self.test_mode:
            parts.append("test")
        return "_".join(parts) + ".pkl"
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.neighborhood_size < 0:
            raise ValueError(f"neighborhood_size must be >= 0, got {self.neighborhood_size}")
        
        if self.period_type not in ('time_bucket', 'hourly', 'daily', 'all'):
            raise ValueError(f"Invalid period_type: {self.period_type}")
        
        if self.grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {self.grid_size}")
        
        if self.grid_dims[0] <= 0 or self.grid_dims[1] <= 0:
            raise ValueError(f"grid_dims must be positive, got {self.grid_dims}")
        
        if self.test_sample_size < 1:
            raise ValueError(f"test_sample_size must be >= 1, got {self.test_sample_size}")
    
    def to_dict(self) -> dict:
        """Convert configuration to a dictionary for serialization."""
        return {
            'neighborhood_size': self.neighborhood_size,
            'neighborhood_dims': self.neighborhood_dims,
            'period_type': self.period_type,
            'grid_size': self.grid_size,
            'grid_dims': self.grid_dims,
            'exclude_sunday': self.exclude_sunday,
            'x_grid_offset': self.x_grid_offset,
            'y_grid_offset': self.y_grid_offset,
            'time_offset': self.time_offset,
            'test_mode': self.test_mode,
            'test_sample_size': self.test_sample_size if self.test_mode else None,
            'test_days': self.test_days if self.test_mode else None,
            'input_files': list(self.input_files),
        }


# Predefined configurations
DEFAULT_CONFIG = ActiveTaxisConfig(
    neighborhood_size=2,  # 5×5 neighborhood
    period_type='hourly',
    test_mode=False,
)

HOURLY_CONFIG = ActiveTaxisConfig(
    neighborhood_size=2,
    period_type='hourly',
    test_mode=False,
)

DAILY_CONFIG = ActiveTaxisConfig(
    neighborhood_size=2,
    period_type='daily',
    test_mode=False,
)

TIME_BUCKET_CONFIG = ActiveTaxisConfig(
    neighborhood_size=2,
    period_type='time_bucket',
    test_mode=False,
)

# Small neighborhood configs (for testing different k values)
SMALL_NEIGHBORHOOD_CONFIG = ActiveTaxisConfig(
    neighborhood_size=1,  # 3×3 neighborhood
    period_type='hourly',
    test_mode=False,
)

LARGE_NEIGHBORHOOD_CONFIG = ActiveTaxisConfig(
    neighborhood_size=3,  # 7×7 neighborhood
    period_type='hourly',
    test_mode=False,
)

# Test mode configuration
TEST_CONFIG = ActiveTaxisConfig(
    neighborhood_size=2,
    period_type='hourly',
    test_mode=True,
    test_sample_size=5,
    test_days=3,
)
