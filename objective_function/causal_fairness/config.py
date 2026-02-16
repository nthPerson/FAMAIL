"""
Configuration for the Causal Fairness term.

This module defines all configurable parameters for the causal fairness
computation, including grid dimensions, temporal aggregation settings,
estimation methods, and data source paths.

Key Configuration Options:
    - estimation_method: How to estimate g(d) function (binning, linear, polynomial, etc.)
    - neighborhood_size: Size of spatial neighborhood for supply aggregation
    - period_type: Temporal aggregation granularity
    - min_demand: Minimum demand threshold to include a cell

Data Sources:
    - Demand: pickup_dropoff_counts.pkl (pickup counts)
    - Supply: active_taxis output files (taxi availability counts)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional, List, Literal
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import TermConfig


# Number of weekdays per month in the dataset
WEEKDAYS_JULY = 21
WEEKDAYS_AUGUST = 23
WEEKDAYS_SEPTEMBER = 22
WEEKDAYS_TOTAL = WEEKDAYS_JULY + WEEKDAYS_AUGUST + WEEKDAYS_SEPTEMBER  # 66


@dataclass
class CausalFairnessConfig(TermConfig):
    """
    Configuration for the Causal Fairness term.
    
    The causal fairness term measures how well taxi service supply aligns with
    passenger demand. It uses the coefficient of determination (R²) to measure
    what fraction of service ratio variance is explained by demand alone.
    
    Attributes:
        period_type: Temporal aggregation level
            - "time_bucket": Each 5-min bucket (finest, 288 periods/day)
            - "hourly": Each hour (24 periods/day)  
            - "daily": Each day (6 periods total)
            - "all": Aggregate all data into single period
        
        grid_dims: Spatial grid dimensions (x, y)
        
        estimation_method: Method for estimating g(d) = E[Y|D=d]
            - "binning": Group by demand bins, compute mean Y (default, robust)
            - "linear": Linear regression Y ~ D
            - "polynomial": Polynomial regression Y ~ poly(D)
            - "isotonic": Isotonic (monotonic) regression
            - "lowess": Local weighted regression (smoothing)
        
        n_bins: Number of bins for binning estimation method
        poly_degree: Polynomial degree for polynomial estimation
        lowess_frac: Fraction of data for lowess smoothing
        
        min_demand: Minimum demand (pickups) to include a cell in analysis
        max_ratio: Maximum service ratio to include (caps outliers)
        
        active_taxis_data_path: Path to active_taxis output file for supply data
        neighborhood_size: k for (2k+1)×(2k+1) neighborhood (must match active_taxis)
        
        num_days: Number of days in the dataset
        days_filter: Specific days to include (None = all)
        time_filter: Time bucket range to include (None = all)
    """
    
    # F_causal formulation
    formulation: Literal[
        "baseline", "option_b", "option_c"
    ] = "option_b"

    # Temporal aggregation
    period_type: str = "hourly"  # "time_bucket", "hourly", "daily", "all"

    # Spatial configuration
    grid_dims: Tuple[int, int] = (48, 90)  # (x_cells, y_cells)

    # g(d) estimation method (used by baseline formulation; option_b/c use power_basis)
    estimation_method: Literal[
        "binning", "linear", "polynomial", "isotonic", "lowess",
        "reciprocal", "log", "power_basis"
    ] = "binning"
    n_bins: int = 10                    # For binning method
    poly_degree: int = 2                # For polynomial method
    lowess_frac: float = 0.3            # For lowess method
    
    # Data filtering
    min_demand: int = 1                 # Minimum demand to include cell
    max_ratio: Optional[float] = None   # Maximum service ratio (None = no cap)
    include_zero_supply: bool = False   # Whether to include cells with zero supply
    
    # Supply data configuration
    active_taxis_data_path: Optional[str] = None  # Path to active_taxis_*.pkl
    neighborhood_size: int = 2          # k value (5×5 neighborhood by default)
    fallback_supply: int = 1            # Fallback when supply is zero
    
    # Temporal coverage
    num_days: float = 21.0              # Days in dataset (21 July, 23 Aug, 22 Sept, 66 total)
    
    # Filters
    days_filter: Optional[List[int]] = None  # e.g., [1, 2, 3] for Mon-Wed
    time_filter: Optional[Tuple[int, int]] = None  # e.g., (1, 144) for first half of day
    
    # Include all cells (for compatibility with spatial fairness dashboard)
    include_zero_cells: bool = False    # For causal, usually False (need demand > 0)
    data_is_one_indexed: bool = True    # Whether source data uses 1-based indexing

    # Demographics configuration (for option_b / option_c formulations)
    demographic_features: Optional[List[str]] = None  # Feature names for hat matrix (None = default set)
    demographics_data_path: Optional[str] = None      # Path to cell_demographics.pkl
    district_mapping_path: Optional[str] = None       # Path to grid_to_district_mapping.pkl
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        super().validate()
        
        valid_periods = ["time_bucket", "hourly", "daily", "all"]
        if self.period_type not in valid_periods:
            raise ValueError(
                f"Invalid period_type '{self.period_type}'. "
                f"Must be one of: {valid_periods}"
            )
        
        if self.grid_dims[0] <= 0 or self.grid_dims[1] <= 0:
            raise ValueError("Grid dimensions must be positive")
        
        valid_formulations = ["baseline", "option_b", "option_c"]
        if self.formulation not in valid_formulations:
            raise ValueError(
                f"Invalid formulation '{self.formulation}'. "
                f"Must be one of: {valid_formulations}"
            )

        valid_methods = [
            "binning", "linear", "polynomial", "isotonic", "lowess",
            "reciprocal", "log", "power_basis",
        ]
        if self.estimation_method not in valid_methods:
            raise ValueError(
                f"Invalid estimation_method '{self.estimation_method}'. "
                f"Must be one of: {valid_methods}"
            )
        
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        
        if self.poly_degree < 1:
            raise ValueError("poly_degree must be at least 1")
        
        if self.lowess_frac <= 0 or self.lowess_frac > 1:
            raise ValueError("lowess_frac must be in (0, 1]")
        
        if self.min_demand < 1:
            raise ValueError("min_demand must be at least 1 (cannot divide by zero demand)")
        
        if self.max_ratio is not None and self.max_ratio <= 0:
            raise ValueError("max_ratio must be positive if specified")
        
        if self.neighborhood_size < 0:
            raise ValueError("neighborhood_size must be non-negative")
        
        if self.num_days <= 0:
            raise ValueError("num_days must be positive")
        
        if self.fallback_supply <= 0:
            raise ValueError("fallback_supply must be positive (avoid division by zero)")


# =============================================================================
# PREDEFINED CONFIGURATIONS
# =============================================================================

# Default configuration - hourly aggregation with binning estimation
DEFAULT_CONFIG = CausalFairnessConfig(
    period_type="hourly",
    estimation_method="binning",
    n_bins=10,
    min_demand=1,
    num_days=21.0,  # July weekdays
    weight=0.33,
)

# Fine-grained temporal analysis
FINE_GRAINED_CONFIG = CausalFairnessConfig(
    period_type="time_bucket",  # 5-minute buckets
    estimation_method="binning",
    n_bins=5,  # Fewer bins due to less data per period
    min_demand=1,
    num_days=21.0,
)

# Daily aggregation for overview
DAILY_CONFIG = CausalFairnessConfig(
    period_type="daily",
    estimation_method="polynomial",
    poly_degree=2,
    min_demand=1,
    num_days=21.0,
)

# Full aggregation (single period)
AGGREGATE_CONFIG = CausalFairnessConfig(
    period_type="all",
    estimation_method="lowess",
    lowess_frac=0.3,
    min_demand=5,  # Higher threshold with more data
    num_days=21.0,
)

# July weekdays
JULY_CONFIG = CausalFairnessConfig(
    period_type="hourly",
    estimation_method="binning",
    num_days=WEEKDAYS_JULY,
    days_filter=None,
)

# August weekdays
AUGUST_CONFIG = CausalFairnessConfig(
    period_type="hourly",
    estimation_method="binning",
    num_days=WEEKDAYS_AUGUST,
    days_filter=None,
)

# September weekdays
SEPTEMBER_CONFIG = CausalFairnessConfig(
    period_type="hourly",
    estimation_method="binning",
    num_days=WEEKDAYS_SEPTEMBER,
    days_filter=None,
)

# All months combined
ALL_MONTHS_CONFIG = CausalFairnessConfig(
    period_type="hourly",
    estimation_method="binning",
    num_days=WEEKDAYS_TOTAL,
    days_filter=None,
)
