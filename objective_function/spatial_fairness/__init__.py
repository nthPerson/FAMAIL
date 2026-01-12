"""
Spatial Fairness Term for FAMAIL Objective Function.

This package implements the Spatial Fairness term ($F_{\text{spatial}}$) which
measures equality of taxi service distribution across geographic regions using
the Gini coefficient.

Example Usage:
    >>> from objective_function.spatial_fairness import SpatialFairnessTerm, SpatialFairnessConfig
    >>> import pickle
    >>> 
    >>> # Load data
    >>> with open('source_data/pickup_dropoff_counts.pkl', 'rb') as f:
    ...     data = pickle.load(f)
    >>> 
    >>> # Create term with default configuration
    >>> config = SpatialFairnessConfig(period_type="hourly")
    >>> term = SpatialFairnessTerm(config)
    >>> 
    >>> # Compute spatial fairness
    >>> result = term.compute({}, {'pickup_dropoff_counts': data})
    >>> print(f"Spatial Fairness: {result:.4f}")
    >>> 
    >>> # Get detailed breakdown
    >>> breakdown = term.compute_with_breakdown({}, {'pickup_dropoff_counts': data})
    >>> print(f"Avg Gini (Arrivals): {breakdown['components']['avg_gini_arrival']:.4f}")
    >>> print(f"Avg Gini (Departures): {breakdown['components']['avg_gini_departure']:.4f}")
"""

from .config import (
    SpatialFairnessConfig,
    FINE_GRAINED_CONFIG,
    HOURLY_CONFIG,
    DAILY_CONFIG,
    AGGREGATE_CONFIG,
    PEAK_HOURS_CONFIG,
    ACTIVE_CELLS_CONFIG,
)

from .term import SpatialFairnessTerm

from .utils import (
    compute_gini,
    compute_gini_alternative,
    compute_lorenz_curve,
    load_pickup_dropoff_counts,
    aggregate_counts_by_period,
    get_unique_periods,
    compute_period_duration_days,
    compute_service_rates_for_period,
    validate_pickup_dropoff_data,
    get_data_statistics,
)


__all__ = [
    # Main classes
    'SpatialFairnessTerm',
    'SpatialFairnessConfig',
    
    # Predefined configurations
    'FINE_GRAINED_CONFIG',
    'HOURLY_CONFIG',
    'DAILY_CONFIG',
    'AGGREGATE_CONFIG',
    'PEAK_HOURS_CONFIG',
    'ACTIVE_CELLS_CONFIG',
    
    # Utility functions
    'compute_gini',
    'compute_gini_alternative',
    'compute_lorenz_curve',
    'load_pickup_dropoff_counts',
    'aggregate_counts_by_period',
    'get_unique_periods',
    'compute_period_duration_days',
    'compute_service_rates_for_period',
    'validate_pickup_dropoff_data',
    'get_data_statistics',
]
