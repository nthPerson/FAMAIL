"""
Spatial Fairness Term Implementation.

This module implements the Spatial Fairness term ($F_{\text{spatial}}$) for the
FAMAIL objective function. The term measures equality in taxi service distribution
across geographic regions using the Gini coefficient.

Mathematical Formulation:
    F_spatial = 1 - (1/2|P|) * Σ_p (G_a^p + G_d^p)
    
where:
    - P = set of time periods
    - G_a^p = Gini coefficient of Arrival Service Rates in period p
    - G_d^p = Gini coefficient of Departure Service Rates in period p

Service Rate Formula:
    DSR_s^p = pickups_s^p / (N_s^p × T)
    
where N_s^p can be:
    - A constant (num_taxis) for all cells
    - Dynamic per-cell values from active_taxis dataset

Reference:
    Su et al. (2018) "Uncovering Spatial Inequality in Taxi Services"
"""

import sys
import os
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import ObjectiveFunctionTerm, TermMetadata, TrajectoryData, AuxiliaryData
from spatial_fairness.config import SpatialFairnessConfig
from spatial_fairness.utils import (
    compute_gini,
    compute_lorenz_curve,
    aggregate_counts_by_period,
    get_unique_periods,
    compute_period_duration_days,
    compute_service_rates_for_period,
    validate_pickup_dropoff_data,
    get_data_statistics,
    load_active_taxis_data,
    get_active_taxis_statistics,
    validate_active_taxis_period_alignment,
)


class SpatialFairnessTerm(ObjectiveFunctionTerm):
    """
    Spatial Fairness term based on Gini coefficient of service rates.
    
    Measures equality of taxi service distribution across geographic regions.
    Higher values indicate more equal distribution (more fair).
    
    Value Range: [0, 1]
        - F_spatial = 1: Perfect equality (all cells have identical service rates)
        - F_spatial = 0: Maximum inequality (all service concentrated in one cell)
    
    Example:
        >>> config = SpatialFairnessConfig(period_type="hourly")
        >>> term = SpatialFairnessTerm(config)
        >>> result = term.compute({}, {'pickup_dropoff_counts': data})
        >>> print(f"Spatial Fairness: {result:.4f}")
    """
    
    def __init__(self, config: Optional[SpatialFairnessConfig] = None):
        """
        Initialize the Spatial Fairness term.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        if config is None:
            config = SpatialFairnessConfig()
        super().__init__(config)
        self.config: SpatialFairnessConfig = config
    
    def _build_metadata(self) -> TermMetadata:
        """Build and return the term's metadata."""
        return TermMetadata(
            name="spatial_fairness",
            display_name="Spatial Fairness",
            version="1.0.0",
            description=(
                "Gini-based measure of taxi service distribution equality. "
                "Computes complement of average Gini coefficient across arrival "
                "and departure service rates for all time periods."
            ),
            value_range=(0.0, 1.0),
            higher_is_better=True,
            is_differentiable=False,  # Gini involves sorting
            required_data=["pickup_dropoff_counts"],
            optional_data=["all_trajs"],
            author="FAMAIL Team",
            last_updated="2026-01-10"
        )
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        self.config.validate()
    
    def compute(
        self,
        trajectories: TrajectoryData,
        auxiliary_data: AuxiliaryData
    ) -> float:
        """
        Compute the spatial fairness value.
        
        Args:
            trajectories: Dictionary of trajectories (can be empty if using auxiliary_data)
            auxiliary_data: Must contain 'pickup_dropoff_counts' key
            
        Returns:
            Spatial fairness value in [0, 1], higher = more fair
            
        Raises:
            ValueError: If required data is missing
        """
        # Get pickup/dropoff counts
        if 'pickup_dropoff_counts' not in auxiliary_data:
            raise ValueError(
                "auxiliary_data must contain 'pickup_dropoff_counts'. "
                "Load data with: pickle.load(open('source_data/pickup_dropoff_counts.pkl', 'rb'))"
            )
        
        data = auxiliary_data['pickup_dropoff_counts']
        
        # Get active_taxis data if provided
        active_taxis_data = auxiliary_data.get('active_taxis_counts', None)
        
        # Compute and return
        result = self._compute_from_counts(data, active_taxis_data)
        return result['value']
    
    def compute_with_breakdown(
        self,
        trajectories: TrajectoryData,
        auxiliary_data: AuxiliaryData
    ) -> Dict[str, Any]:
        """
        Compute spatial fairness with detailed breakdown.
        
        Returns comprehensive information for analysis and debugging.
        
        Args:
            trajectories: Dictionary of trajectories
            auxiliary_data: Must contain 'pickup_dropoff_counts'.
                           Optionally can contain 'active_taxis_counts' for
                           per-cell taxi counts when taxi_count_source='active_taxis_lookup'
            
        Returns:
            Dictionary with:
                - value: float - final spatial fairness value
                - components: Dict - per-period Gini coefficients
                - statistics: Dict - summary statistics
                - diagnostics: Dict - debugging information
        """
        if 'pickup_dropoff_counts' not in auxiliary_data:
            raise ValueError("auxiliary_data must contain 'pickup_dropoff_counts'")
        
        data = auxiliary_data['pickup_dropoff_counts']
        active_taxis_data = auxiliary_data.get('active_taxis_counts', None)
        
        return self._compute_from_counts(data, active_taxis_data)
    
    def _compute_from_counts(
        self,
        data: Dict[Tuple[int, int, int, int], Tuple[int, int]],
        active_taxis_data: Optional[Dict[Tuple, int]] = None,
    ) -> Dict[str, Any]:
        """
        Core computation from pickup/dropoff counts.
        
        Args:
            data: Raw pickup_dropoff_counts data
            active_taxis_data: Pre-loaded active_taxis data (optional)
            
        Returns:
            Complete breakdown including value and all intermediate results
        """
        start_time = time.perf_counter()
        
        # Validate data
        is_valid, validation_errors = validate_pickup_dropoff_data(data)
        if not is_valid:
            self._log(f"Data validation errors: {validation_errors}")
        
        # Get data statistics
        data_stats = get_data_statistics(data)
        
        # Handle active_taxis data loading if needed
        active_taxis_stats = None
        if self.config.taxi_count_source == "active_taxis_lookup":
            if active_taxis_data is None and self.config.active_taxis_data_path:
                # Load active_taxis data from path
                active_taxis_data = load_active_taxis_data(self.config.active_taxis_data_path)
                self._log(f"Loaded active_taxis data from {self.config.active_taxis_data_path}")
            
            if active_taxis_data is not None:
                # Validate period alignment
                is_aligned, alignment_msg = validate_active_taxis_period_alignment(
                    active_taxis_data, self.config.period_type
                )
                if not is_aligned:
                    self._log(f"Warning: {alignment_msg}")
                
                active_taxis_stats = get_active_taxis_statistics(active_taxis_data)
        
        # Aggregate by period
        pickups, dropoffs = aggregate_counts_by_period(
            data,
            period_type=self.config.period_type,
            days_filter=self.config.days_filter,
            time_filter=self.config.time_filter,
        )
        
        # Get unique periods
        periods = get_unique_periods(pickups, dropoffs)
        self._log(f"Processing {len(periods)} periods")
        
        # Compute Gini coefficients per period
        gini_arrivals = []
        gini_departures = []
        per_period_data = []
        
        for period in periods:
            # Compute period duration
            period_duration = compute_period_duration_days(
                period, self.config.period_type, self.config.num_days
            )
            
            # Compute service rates (with optional active_taxis lookup)
            dsr_values, asr_values = compute_service_rates_for_period(
                pickups=pickups,
                dropoffs=dropoffs,
                period=period,
                grid_dims=self.config.grid_dims,
                num_taxis=self.config.num_taxis,
                period_duration_days=period_duration,
                include_zero_cells=self.config.include_zero_cells,
                data_is_one_indexed=self.config.data_is_one_indexed,
                min_activity_threshold=self.config.min_activity_threshold,
                active_taxis_data=active_taxis_data if self.config.taxi_count_source == "active_taxis_lookup" else None,
                active_taxis_fallback=self.config.active_taxis_fallback,
                period_type=self.config.period_type,
            )
            
            # Compute Gini coefficients
            g_d = compute_gini(dsr_values)  # Departure (pickup) Gini
            g_a = compute_gini(asr_values)  # Arrival (dropoff) Gini
            
            gini_arrivals.append(g_a)
            gini_departures.append(g_d)
            
            # Store per-period details
            per_period_data.append({
                'period': period,
                'gini_arrival': g_a,
                'gini_departure': g_d,
                'gini_average': 0.5 * (g_a + g_d),
                'fairness': 1.0 - 0.5 * (g_a + g_d),
                'n_cells': len(dsr_values),
                'total_pickups': np.sum(dsr_values) * self.config.num_taxis * period_duration,
                'total_dropoffs': np.sum(asr_values) * self.config.num_taxis * period_duration,
            })
        
        # Aggregate across periods
        if len(gini_arrivals) > 0:
            avg_gini_arrival = float(np.mean(gini_arrivals))
            avg_gini_departure = float(np.mean(gini_departures))
            avg_gini = 0.5 * (avg_gini_arrival + avg_gini_departure)
            f_spatial = 1.0 - avg_gini
        else:
            avg_gini_arrival = 0.0
            avg_gini_departure = 0.0
            avg_gini = 0.0
            f_spatial = 1.0
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            'value': float(f_spatial),
            'components': {
                'avg_gini_arrival': avg_gini_arrival,
                'avg_gini_departure': avg_gini_departure,
                'avg_gini_combined': avg_gini,
                'per_period_gini_arrival': gini_arrivals,
                'per_period_gini_departure': gini_departures,
                'per_period_fairness': [1.0 - 0.5 * (ga + gd) for ga, gd in zip(gini_arrivals, gini_departures)],
                'per_period_data': per_period_data,
            },
            'statistics': {
                'n_periods': len(periods),
                'gini_arrival_stats': {
                    'mean': avg_gini_arrival,
                    'std': float(np.std(gini_arrivals)) if gini_arrivals else 0.0,
                    'min': float(np.min(gini_arrivals)) if gini_arrivals else 0.0,
                    'max': float(np.max(gini_arrivals)) if gini_arrivals else 0.0,
                },
                'gini_departure_stats': {
                    'mean': avg_gini_departure,
                    'std': float(np.std(gini_departures)) if gini_departures else 0.0,
                    'min': float(np.min(gini_departures)) if gini_departures else 0.0,
                    'max': float(np.max(gini_departures)) if gini_departures else 0.0,
                },
                'data_stats': data_stats,
                'active_taxis_stats': active_taxis_stats,
            },
            'diagnostics': {
                'computation_time_ms': elapsed_ms,
                'config': {
                    'period_type': self.config.period_type,
                    'grid_dims': self.config.grid_dims,
                    'taxi_count_source': self.config.taxi_count_source,
                    'num_taxis': self.config.num_taxis,
                    'active_taxis_data_path': self.config.active_taxis_data_path,
                    'active_taxis_neighborhood': self.config.active_taxis_neighborhood,
                    'active_taxis_fallback': self.config.active_taxis_fallback,
                    'num_days': self.config.num_days,
                    'include_zero_cells': self.config.include_zero_cells,
                },
                'validation_errors': validation_errors if not is_valid else [],
            },
        }
    
    def compute_for_single_period(
        self,
        data: Dict[Tuple[int, int, int, int], Tuple[int, int]],
        period: Any,
    ) -> Dict[str, Any]:
        """
        Compute spatial fairness for a single time period.
        
        Useful for analyzing temporal patterns or debugging.
        
        Args:
            data: Raw pickup_dropoff_counts data
            period: The period to analyze
            
        Returns:
            Dictionary with period-specific results
        """
        # Aggregate by period
        pickups, dropoffs = aggregate_counts_by_period(
            data,
            period_type=self.config.period_type,
            days_filter=self.config.days_filter,
            time_filter=self.config.time_filter,
        )
        
        # Compute period duration
        period_duration = compute_period_duration_days(
            period, self.config.period_type, self.config.num_days
        )
        
        # Compute service rates
        dsr_values, asr_values = compute_service_rates_for_period(
            pickups=pickups,
            dropoffs=dropoffs,
            period=period,
            grid_dims=self.config.grid_dims,
            num_taxis=self.config.num_taxis,
            period_duration_days=period_duration,
            include_zero_cells=self.config.include_zero_cells,
            data_is_one_indexed=self.config.data_is_one_indexed,
            min_activity_threshold=self.config.min_activity_threshold,
        )
        
        # Compute Gini coefficients
        g_d = compute_gini(dsr_values)
        g_a = compute_gini(asr_values)
        f_spatial = 1.0 - 0.5 * (g_a + g_d)
        
        # Compute Lorenz curves
        lorenz_pickup = compute_lorenz_curve(dsr_values)
        lorenz_dropoff = compute_lorenz_curve(asr_values)
        
        return {
            'period': period,
            'fairness': f_spatial,
            'gini_departure': g_d,
            'gini_arrival': g_a,
            'dsr_values': dsr_values,
            'asr_values': asr_values,
            'lorenz_pickup': lorenz_pickup,
            'lorenz_dropoff': lorenz_dropoff,
            'n_cells': len(dsr_values),
        }
    
    def get_spatial_heatmap_data(
        self,
        data: Dict[Tuple[int, int, int, int], Tuple[int, int]],
        period: Optional[Any] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get spatial distribution data for heatmap visualization.
        
        Args:
            data: Raw pickup_dropoff_counts data
            period: Optional specific period. If None, aggregates all data.
            
        Returns:
            Dictionary with 2D arrays for pickups and dropoffs
        """
        # Aggregate by period
        target_period_type = "all" if period is None else self.config.period_type
        pickups, dropoffs = aggregate_counts_by_period(
            data,
            period_type=target_period_type,
            days_filter=self.config.days_filter,
            time_filter=self.config.time_filter,
        )
        
        # Determine the period to use
        if period is None:
            period = "all"
        
        # Create 2D arrays
        x_dim, y_dim = self.config.grid_dims
        x_offset = 1 if self.config.data_is_one_indexed else 0
        y_offset = 1 if self.config.data_is_one_indexed else 0
        
        pickup_grid = np.zeros((x_dim, y_dim))
        dropoff_grid = np.zeros((x_dim, y_dim))
        
        for x in range(x_dim):
            for y in range(y_dim):
                cell = (x + x_offset, y + y_offset)
                key = (cell, period)
                
                pickup_grid[x, y] = pickups.get(key, 0)
                dropoff_grid[x, y] = dropoffs.get(key, 0)
        
        return {
            'pickups': pickup_grid,
            'dropoffs': dropoff_grid,
            'total': pickup_grid + dropoff_grid,
        }
