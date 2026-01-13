"""
Causal Fairness Term Implementation.

This module implements the Causal Fairness term ($F_{causal}$) for the
FAMAIL objective function. The term measures the degree to which taxi
service supply is explained by passenger demand alone.

Mathematical Formulation:
    F_causal = (1/|P|) * Σ_p F_causal^p
    
    where F_causal^p = R² = 1 - Var(R) / Var(Y)
    
    - Y = Service ratio (Supply / Demand)
    - R = Y - g(D) = Residual (unexplained variation)
    - g(d) = Expected service ratio given demand d

Value Range: [0, 1]
    - F_causal = 1: Service perfectly explained by demand (no contextual bias)
    - F_causal = 0: Service independent of demand (maximum unfairness)

Differentiability:
    This term is fully differentiable for gradient-based trajectory optimization.
    The g(d) function is pre-computed and frozen during optimization, ensuring
    gradients flow only through the service ratio computation.

Reference:
    FAMAIL project documentation based on counterfactual fairness literature
"""

import sys
import os
from typing import Dict, List, Any, Tuple, Optional, Callable
import numpy as np
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import ObjectiveFunctionTerm, TermMetadata, TrajectoryData, AuxiliaryData
from causal_fairness.config import CausalFairnessConfig
from causal_fairness.utils import (
    estimate_g_function,
    compute_r_squared,
    compute_r_squared_torch,
    compute_service_ratios,
    extract_demand_ratio_arrays,
    load_pickup_dropoff_counts,
    load_active_taxis_data,
    extract_demand_from_counts,
    aggregate_to_period,
    get_unique_periods,
    filter_by_period,
    filter_by_days,
    filter_by_time,
    get_data_statistics,
    validate_data_alignment,
    verify_causal_fairness_gradient,
    DifferentiableCausalFairness,
)


class CausalFairnessTerm(ObjectiveFunctionTerm):
    """
    Causal Fairness term based on R² of service ratio vs demand.
    
    Measures how well taxi service supply aligns with passenger demand.
    Higher values indicate that service is primarily driven by demand
    rather than other (potentially discriminatory) factors.
    
    Value Range: [0, 1]
        - F_causal = 1: Perfect demand-service alignment
        - F_causal = 0: No relationship between demand and service
    
    Example:
        >>> config = CausalFairnessConfig(
        ...     period_type="hourly",
        ...     estimation_method="binning",
        ...     active_taxis_data_path="path/to/active_taxis.pkl"
        ... )
        >>> term = CausalFairnessTerm(config)
        >>> result = term.compute(
        ...     {},
        ...     {
        ...         'pickup_dropoff_counts': demand_data,
        ...         'active_taxis': supply_data,
        ...     }
        ... )
        >>> print(f"Causal Fairness: {result:.4f}")
    """
    
    def __init__(self, config: Optional[CausalFairnessConfig] = None):
        """
        Initialize the Causal Fairness term.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        if config is None:
            config = CausalFairnessConfig()
        super().__init__(config)
        self.config: CausalFairnessConfig = config
        
        # Cache for g(d) function
        self._g_function: Optional[Callable] = None
        self._g_diagnostics: Optional[Dict] = None
    
    def _build_metadata(self) -> TermMetadata:
        """Build and return the term's metadata."""
        return TermMetadata(
            name="causal_fairness",
            display_name="Causal Fairness",
            version="1.0.0",
            description=(
                "R²-based measure of demand-service alignment. "
                "Computes the coefficient of determination between "
                "service ratios and expected ratios given demand. "
                "Uses differentiable variance computation for gradient-based optimization."
            ),
            value_range=(0.0, 1.0),
            higher_is_better=True,
            is_differentiable=True,
            required_data=["pickup_dropoff_counts"],
            optional_data=["active_taxis"],
            author="FAMAIL Research Team",
            last_updated="2026-01-12",
        )
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        self.config.validate()
    
    def compute(
        self,
        trajectories: TrajectoryData,
        auxiliary_data: AuxiliaryData,
    ) -> float:
        """
        Compute the causal fairness score.
        
        Args:
            trajectories: Dictionary of trajectories (not used directly)
            auxiliary_data: Must contain 'pickup_dropoff_counts' and optionally 'active_taxis'
            
        Returns:
            Causal fairness score in [0, 1]
        """
        result = self.compute_with_breakdown(trajectories, auxiliary_data)
        return result['value']
    
    def compute_with_breakdown(
        self,
        trajectories: TrajectoryData,
        auxiliary_data: AuxiliaryData,
    ) -> Dict[str, Any]:
        """
        Compute causal fairness with detailed breakdown.
        
        Args:
            trajectories: Dictionary of trajectories
            auxiliary_data: Must contain 'pickup_dropoff_counts'
            
        Returns:
            Dictionary containing:
                - value: The causal fairness score
                - components: Per-period breakdown
                - diagnostics: Debug information
        """
        start_time = time.time()
        
        # Extract data
        if 'pickup_dropoff_counts' not in auxiliary_data:
            raise ValueError("auxiliary_data must contain 'pickup_dropoff_counts'")
        
        raw_data = auxiliary_data['pickup_dropoff_counts']
        
        # Get supply data
        if 'active_taxis' in auxiliary_data:
            supply_data = auxiliary_data['active_taxis']
        elif self.config.active_taxis_data_path:
            supply_data = load_active_taxis_data(self.config.active_taxis_data_path)
        else:
            # Fallback: use dropoff counts as proxy for supply
            supply_data = self._extract_supply_proxy(raw_data)
        
        # Extract demand
        demand_raw = extract_demand_from_counts(raw_data)
        
        # Apply filters
        if self.config.days_filter:
            demand_raw = filter_by_days(demand_raw, self.config.days_filter)
            supply_data = filter_by_days(supply_data, self.config.days_filter)
        
        if self.config.time_filter:
            demand_raw = filter_by_time(demand_raw, self.config.time_filter)
            supply_data = filter_by_time(supply_data, self.config.time_filter)
        
        # Aggregate by period type
        demand = aggregate_to_period(demand_raw, self.config.period_type)
        supply = aggregate_to_period(supply_data, self.config.period_type)
        
        # Compute service ratios
        ratios = compute_service_ratios(
            demand, supply,
            min_demand=self.config.min_demand,
            max_ratio=self.config.max_ratio,
            include_zero_supply=self.config.include_zero_supply,
        )
        
        # Get unique periods
        periods = get_unique_periods(ratios)
        
        if not periods:
            return {
                'value': 0.0,
                'components': {
                    'per_period_data': [],
                    'per_period_r2': [],
                },
                'diagnostics': {
                    'error': 'No valid periods found',
                    'computation_time': time.time() - start_time,
                },
            }
        
        # Estimate g(d) function (using all data)
        all_demands, all_ratios, all_keys = extract_demand_ratio_arrays(demand, ratios)
        
        g_func, g_diagnostics = estimate_g_function(
            all_demands, all_ratios,
            method=self.config.estimation_method,
            n_bins=self.config.n_bins,
            poly_degree=self.config.poly_degree,
            lowess_frac=self.config.lowess_frac,
        )
        
        # Cache g function for differentiable computation
        self._g_function = g_func
        self._g_diagnostics = g_diagnostics
        
        # Compute per-period R²
        per_period_data = []
        per_period_r2 = []
        
        for period in periods:
            period_ratios = filter_by_period(ratios, period)
            period_demand = filter_by_period(demand, period)
            
            demands_p, ratios_p, keys_p = extract_demand_ratio_arrays(
                period_demand, period_ratios
            )
            
            if len(demands_p) < 2:
                # Not enough data for variance computation
                continue
            
            # Get expected values
            expected_p = g_func(demands_p)
            
            # Compute R²
            r2 = compute_r_squared(ratios_p, expected_p)
            
            per_period_r2.append(r2)
            per_period_data.append({
                'period': period,
                'time': period[0],
                'day': period[1],
                'r_squared': r2,
                'n_cells': len(demands_p),
                'mean_demand': float(np.mean(demands_p)),
                'mean_ratio': float(np.mean(ratios_p)),
                'mean_expected': float(np.mean(expected_p)),
                'var_ratio': float(np.var(ratios_p)),
                'var_residual': float(np.var(ratios_p - expected_p)),
            })
        
        # Aggregate to final score
        if per_period_r2:
            f_causal = float(np.mean(per_period_r2))
        else:
            f_causal = 0.0
        
        # Compute overall statistics
        data_stats = get_data_statistics(demand, supply, ratios)
        alignment_issues = validate_data_alignment(demand, supply)
        
        return {
            'value': f_causal,
            'components': {
                'per_period_data': per_period_data,
                'per_period_r2': per_period_r2,
                'g_diagnostics': g_diagnostics,
                'overall_r2': compute_r_squared(all_ratios, g_func(all_demands)) if len(all_demands) > 0 else 0.0,
                'demands': all_demands.tolist() if len(all_demands) > 0 else [],
                'ratios': all_ratios.tolist() if len(all_ratios) > 0 else [],
                'expected': g_func(all_demands).tolist() if len(all_demands) > 0 else [],
                'residuals': (all_ratios - g_func(all_demands)).tolist() if len(all_demands) > 0 else [],
            },
            'diagnostics': {
                'n_periods': len(per_period_r2),
                'n_total_cells': len(all_demands),
                'data_stats': data_stats,
                'alignment_issues': alignment_issues,
                'config': {
                    'period_type': self.config.period_type,
                    'estimation_method': self.config.estimation_method,
                    'min_demand': self.config.min_demand,
                    'n_bins': self.config.n_bins,
                },
                'computation_time': time.time() - start_time,
            },
        }
    
    def _extract_supply_proxy(self, raw_data: Dict) -> Dict[Tuple, int]:
        """
        Extract supply proxy from pickup/dropoff data.
        
        When active_taxis data is not available, use dropoff counts
        as a proxy for taxi availability.
        
        Args:
            raw_data: pickup_dropoff_counts data
            
        Returns:
            Supply proxy dictionary
        """
        supply = {}
        for key, counts in raw_data.items():
            if isinstance(counts, (list, tuple)) and len(counts) >= 2:
                supply[key] = int(counts[1])  # Dropoff count as supply proxy
            elif isinstance(counts, (int, float)):
                supply[key] = int(counts)
        return supply
    
    def get_residuals_grid(
        self,
        auxiliary_data: AuxiliaryData,
        aggregator: str = 'mean',
    ) -> np.ndarray:
        """
        Get residuals aggregated to a 2D grid for visualization.
        
        Args:
            auxiliary_data: Data containing pickup_dropoff_counts
            aggregator: How to aggregate residuals ('mean', 'sum', 'max')
            
        Returns:
            2D numpy array of shape (y_cells, x_cells)
        """
        from causal_fairness.utils import aggregate_to_grid
        
        result = self.compute_with_breakdown({}, auxiliary_data)
        
        # Build residual dictionary
        residual_dict = {}
        demands = result['components']['demands']
        residuals = result['components']['residuals']
        
        # We need the original keys - this is a limitation
        # For now, return the overall residual statistics
        
        if not residuals:
            return np.zeros((self.config.grid_dims[1], self.config.grid_dims[0]))
        
        # Simple visualization: mean residual per cell would require key tracking
        # This is a placeholder - proper implementation would track keys
        return np.zeros((self.config.grid_dims[1], self.config.grid_dims[0]))
    
    def get_differentiable_module(
        self,
        demands: np.ndarray,
        ratios: np.ndarray,
    ) -> DifferentiableCausalFairness:
        """
        Return a differentiable module for gradient-based optimization.
        
        This module uses a frozen g(d) function computed from the provided
        demands and ratios. During optimization, only the service ratios
        will receive gradients, not the g(d) estimation.
        
        Args:
            demands: Original demand values for fitting g(d)
            ratios: Original service ratios for fitting g(d)
            
        Returns:
            DifferentiableCausalFairness instance
        """
        # Fit g(d) function
        g_func, _ = estimate_g_function(
            demands, ratios,
            method=self.config.estimation_method,
            n_bins=self.config.n_bins,
            poly_degree=self.config.poly_degree,
            lowess_frac=self.config.lowess_frac,
        )
        
        return DifferentiableCausalFairness(
            frozen_demands=demands,
            g_function=g_func,
        )
    
    def compute_gradient(
        self,
        trajectories: TrajectoryData,
        auxiliary_data: AuxiliaryData,
    ) -> np.ndarray:
        """
        Compute gradient of causal fairness with respect to supply.
        
        This method demonstrates gradient computation for a single forward
        pass. For actual optimization, use get_differentiable_module().
        
        Args:
            trajectories: Trajectory data (not used directly)
            auxiliary_data: Must contain demand and supply data
            
        Returns:
            Gradient array w.r.t. supply values
        """
        import torch
        
        # Compute breakdown to get data
        result = self.compute_with_breakdown(trajectories, auxiliary_data)
        
        demands = np.array(result['components']['demands'], dtype=np.float32)
        ratios = np.array(result['components']['ratios'], dtype=np.float32)
        
        if len(demands) == 0:
            return np.array([])
        
        # Get differentiable module
        diff_module = self.get_differentiable_module(demands, ratios)
        
        # Create tensors
        demand_torch = torch.tensor(demands, dtype=torch.float32)
        # For gradient, we need supply = ratio * demand
        supply_torch = torch.tensor(ratios * demands, dtype=torch.float32, requires_grad=True)
        
        # Compute causal fairness
        f_causal = diff_module.compute(supply_torch, demand_torch)
        
        # Backward pass
        f_causal.backward()
        
        return supply_torch.grad.numpy() if supply_torch.grad is not None else np.zeros_like(demands)
    
    def verify_differentiability(self, n_cells: int = 100) -> Dict[str, Any]:
        """
        Verify that the causal fairness term is differentiable.
        
        Args:
            n_cells: Number of cells to test with
            
        Returns:
            Dictionary with verification results
        """
        return {
            'causal_fairness_verification': verify_causal_fairness_gradient(n_cells),
            'module_verification': DifferentiableCausalFairness.verify_gradients(n_cells),
        }
    
    def get_g_function(self) -> Optional[Callable]:
        """
        Get the cached g(d) function.
        
        Returns:
            The g(d) function if computed, else None
        """
        return self._g_function
    
    def get_g_diagnostics(self) -> Optional[Dict]:
        """
        Get diagnostics from g(d) estimation.
        
        Returns:
            Diagnostics dictionary if computed, else None
        """
        return self._g_diagnostics
