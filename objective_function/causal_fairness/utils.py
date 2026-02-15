"""
Utility functions for the Causal Fairness term.

This module contains:
- g(d) estimation functions (binning, regression, etc.)
- R² computation
- Service ratio calculations
- Data loading and preprocessing
- Differentiable PyTorch implementations
- Validation functions
"""

from typing import Dict, Tuple, List, Optional, Any, Callable, Literal
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict


# =============================================================================
# G(D) ESTIMATION FUNCTIONS
# =============================================================================

def estimate_g_binning(
    demands: np.ndarray,
    ratios: np.ndarray,
    n_bins: int = 10,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]:
    """
    Estimate g(d) using binning method.
    
    Groups demand values into bins and computes the mean service ratio
    for each bin. New predictions use the bin mean.
    
    Args:
        demands: Array of demand values (pickup counts)
        ratios: Array of service ratios (Y = S/D)
        n_bins: Number of demand bins
        
    Returns:
        Tuple of (prediction function, diagnostics dict)
    """
    from scipy.stats import binned_statistic
    
    # Handle edge case
    if len(demands) == 0:
        def empty_predict(d):
            return np.zeros_like(d, dtype=float)
        return empty_predict, {'method': 'binning', 'n_bins': 0, 'error': 'no data'}
    
    # Compute bin edges (use quantiles for even sample sizes per bin)
    try:
        bin_edges = np.percentile(demands, np.linspace(0, 100, n_bins + 1))
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([demands.min(), demands.max()])
    except Exception:
        bin_edges = np.linspace(demands.min(), demands.max(), n_bins + 1)
    
    # Compute bin means
    bin_means, _, bin_numbers = binned_statistic(
        demands, ratios, statistic='mean', bins=bin_edges
    )
    
    # Handle NaN bins (no samples)
    global_mean = np.nanmean(ratios)
    bin_means = np.where(np.isnan(bin_means), global_mean, bin_means)
    
    # Create prediction function
    def predict(d: np.ndarray) -> np.ndarray:
        d = np.atleast_1d(d)
        # Find bin indices
        bin_idx = np.digitize(d, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(bin_means) - 1)
        return bin_means[bin_idx]
    
    diagnostics = {
        'method': 'binning',
        'n_bins': len(bin_means),
        'bin_edges': bin_edges.tolist(),
        'bin_means': bin_means.tolist(),
        'global_mean': global_mean,
    }
    
    return predict, diagnostics


def estimate_g_linear(
    demands: np.ndarray,
    ratios: np.ndarray,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]:
    """
    Estimate g(d) using linear regression.
    
    Fits Y = β₀ + β₁D.
    
    Args:
        demands: Array of demand values
        ratios: Array of service ratios
        
    Returns:
        Tuple of (prediction function, diagnostics dict)
    """
    from sklearn.linear_model import LinearRegression
    
    if len(demands) == 0:
        def empty_predict(d):
            return np.zeros_like(d, dtype=float)
        return empty_predict, {'method': 'linear', 'error': 'no data'}
    
    X = demands.reshape(-1, 1)
    y = ratios
    
    model = LinearRegression()
    model.fit(X, y)
    
    def predict(d: np.ndarray) -> np.ndarray:
        d = np.atleast_1d(d).reshape(-1, 1)
        return model.predict(d)
    
    diagnostics = {
        'method': 'linear',
        'intercept': float(model.intercept_),
        'coefficient': float(model.coef_[0]),
        'r2_train': float(model.score(X, y)),
    }
    
    return predict, diagnostics


def estimate_g_polynomial(
    demands: np.ndarray,
    ratios: np.ndarray,
    degree: int = 2,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]:
    """
    Estimate g(d) using polynomial regression.
    
    Fits Y = Σ βₖDᵏ for k = 0, ..., degree.
    
    Args:
        demands: Array of demand values
        ratios: Array of service ratios
        degree: Polynomial degree
        
    Returns:
        Tuple of (prediction function, diagnostics dict)
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    
    if len(demands) == 0:
        def empty_predict(d):
            return np.zeros_like(d, dtype=float)
        return empty_predict, {'method': 'polynomial', 'error': 'no data'}
    
    X = demands.reshape(-1, 1)
    y = ratios
    
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('reg', LinearRegression()),
    ])
    model.fit(X, y)
    
    def predict(d: np.ndarray) -> np.ndarray:
        d = np.atleast_1d(d).reshape(-1, 1)
        return model.predict(d)
    
    diagnostics = {
        'method': 'polynomial',
        'degree': degree,
        'coefficients': model.named_steps['reg'].coef_.tolist(),
        'intercept': float(model.named_steps['reg'].intercept_),
        'r2_train': float(model.score(X, y)),
    }
    
    return predict, diagnostics


def estimate_g_isotonic(
    demands: np.ndarray,
    ratios: np.ndarray,
    increasing: bool = False,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]:
    """
    Estimate g(d) using isotonic (monotonic) regression.
    
    Fits a monotonic function to the data. By default, assumes
    service ratio may decrease with higher demand.
    
    Args:
        demands: Array of demand values
        ratios: Array of service ratios
        increasing: If True, fit increasing function; else decreasing
        
    Returns:
        Tuple of (prediction function, diagnostics dict)
    """
    from sklearn.isotonic import IsotonicRegression
    
    if len(demands) == 0:
        def empty_predict(d):
            return np.zeros_like(d, dtype=float)
        return empty_predict, {'method': 'isotonic', 'error': 'no data'}
    
    model = IsotonicRegression(increasing=increasing, out_of_bounds='clip')
    model.fit(demands, ratios)
    
    def predict(d: np.ndarray) -> np.ndarray:
        return model.predict(np.atleast_1d(d))
    
    diagnostics = {
        'method': 'isotonic',
        'increasing': increasing,
        'n_points': len(demands),
    }
    
    return predict, diagnostics


def estimate_g_lowess(
    demands: np.ndarray,
    ratios: np.ndarray,
    frac: float = 0.3,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]:
    """
    Estimate g(d) using LOWESS (locally weighted scatterplot smoothing).
    
    Provides flexible nonparametric estimation of E[Y|D].
    
    Args:
        demands: Array of demand values
        ratios: Array of service ratios
        frac: Fraction of data to use for each local regression
        
    Returns:
        Tuple of (prediction function, diagnostics dict)
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from scipy.interpolate import interp1d
    
    if len(demands) == 0:
        def empty_predict(d):
            return np.zeros_like(d, dtype=float)
        return empty_predict, {'method': 'lowess', 'error': 'no data'}
    
    # Fit lowess
    smoothed = lowess(ratios, demands, frac=frac, return_sorted=True)
    
    # Create interpolation function for predictions
    interp_func = interp1d(
        smoothed[:, 0], smoothed[:, 1],
        kind='linear', bounds_error=False,
        fill_value=(smoothed[0, 1], smoothed[-1, 1])
    )
    
    def predict(d: np.ndarray) -> np.ndarray:
        return interp_func(np.atleast_1d(d))
    
    diagnostics = {
        'method': 'lowess',
        'frac': frac,
        'n_points': len(demands),
        'smoothed_x_range': [float(smoothed[0, 0]), float(smoothed[-1, 0])],
    }
    
    return predict, diagnostics


def estimate_g_function(
    demands: np.ndarray,
    ratios: np.ndarray,
    method: Literal["binning", "linear", "polynomial", "isotonic", "lowess"] = "binning",
    n_bins: int = 10,
    poly_degree: int = 2,
    lowess_frac: float = 0.3,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]:
    """
    Estimate the expected service ratio function g(d) = E[Y|D=d].
    
    This function dispatches to the appropriate estimation method
    based on the method parameter.
    
    Args:
        demands: Array of demand values (D)
        ratios: Array of service ratios (Y = S/D)
        method: Estimation method
        n_bins: Number of bins (for binning method)
        poly_degree: Polynomial degree (for polynomial method)
        lowess_frac: Smoothing fraction (for lowess method)
        
    Returns:
        Tuple of (prediction function, diagnostics dict)
    """
    if method == "binning":
        return estimate_g_binning(demands, ratios, n_bins)
    elif method == "linear":
        return estimate_g_linear(demands, ratios)
    elif method == "polynomial":
        return estimate_g_polynomial(demands, ratios, poly_degree)
    elif method == "isotonic":
        return estimate_g_isotonic(demands, ratios)
    elif method == "lowess":
        return estimate_g_lowess(demands, ratios, lowess_frac)
    else:
        raise ValueError(f"Unknown estimation method: {method}")


# =============================================================================
# R² COMPUTATION
# =============================================================================

def compute_r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    Compute R² (coefficient of determination).
    
    R² = 1 - Var(residuals) / Var(y_true)
       = Var(y_pred) / Var(y_true)  (when y_pred = E[Y|X])
    
    Args:
        y_true: True values
        y_pred: Predicted values (from g(d))
        eps: Small constant for numerical stability
        
    Returns:
        R² value, clipped to [0, 1]
    """
    if len(y_true) == 0:
        return 0.0
    
    residuals = y_true - y_pred
    var_residuals = np.var(residuals)
    var_total = np.var(y_true)
    
    if var_total < eps:
        # No variance in y_true - perfect prediction (or no variation)
        return 1.0 if var_residuals < eps else 0.0
    
    r_squared = 1.0 - (var_residuals / var_total)
    
    return float(np.clip(r_squared, 0.0, 1.0))


def compute_r_squared_torch(
    y_true: 'torch.Tensor',
    y_pred: 'torch.Tensor',
    eps: float = 1e-8,
) -> 'torch.Tensor':
    """
    Compute R² using PyTorch for differentiability.
    
    Args:
        y_true: True values tensor (requires_grad may be True)
        y_pred: Predicted values tensor (from frozen g lookup)
        eps: Small constant for numerical stability
        
    Returns:
        Scalar tensor containing R² value
    """
    import torch
    
    residuals = y_true - y_pred
    var_residuals = torch.var(residuals)
    var_total = torch.var(y_true)
    
    # Safe division
    r_squared = 1.0 - (var_residuals / (var_total + eps))
    
    return torch.clamp(r_squared, 0.0, 1.0)


# =============================================================================
# SERVICE RATIO COMPUTATION
# =============================================================================

def compute_service_ratios(
    demand: Dict[Tuple, int],
    supply: Dict[Tuple, int],
    min_demand: int = 1,
    max_ratio: Optional[float] = None,
    include_zero_supply: bool = False,
) -> Dict[Tuple, float]:
    """
    Compute service ratios Y = S/D for each cell-period.
    
    Args:
        demand: Demand (pickup counts) per cell-period key
        supply: Supply (active taxis) per cell-period key
        min_demand: Minimum demand to include cell
        max_ratio: Maximum ratio (caps outliers) - None for no cap
        include_zero_supply: Whether to include cells with zero supply
        
    Returns:
        Dictionary of service ratios per cell-period
    """
    ratios = {}
    
    for key, d in demand.items():
        if d < min_demand:
            continue
        
        s = supply.get(key, 0)
        
        if (not include_zero_supply) and (s == 0):
            continue
        
        ratio = s / d
        
        if max_ratio is not None:
            ratio = min(ratio, max_ratio)
        
        ratios[key] = ratio
    
    return ratios


def extract_demand_ratio_arrays(
    demand: Dict[Tuple, int],
    ratios: Dict[Tuple, float],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
    """
    Extract aligned arrays of demand and ratio values.
    
    Args:
        demand: Demand dictionary
        ratios: Ratio dictionary
        
    Returns:
        Tuple of (demand_array, ratio_array, keys)
    """
    common_keys = sorted(set(demand.keys()) & set(ratios.keys()))
    
    if not common_keys:
        return np.array([]), np.array([]), []
    
    demands_arr = np.array([demand[k] for k in common_keys], dtype=float)
    ratios_arr = np.array([ratios[k] for k in common_keys], dtype=float)
    
    return demands_arr, ratios_arr, common_keys


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_pickup_dropoff_counts(filepath: str) -> Dict[Tuple, List[int]]:
    """
    Load pickup/dropoff counts data.
    
    Expected format:
        {(x, y, time_bucket, day_of_week): [pickup_count, dropoff_count], ...}
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Dictionary of counts
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def load_active_taxis_data(filepath: str) -> Dict[Tuple, int]:
    """
    Load active taxis data.
    
    Handles both bundle format (with 'data', 'stats', 'config' keys)
    and raw dictionary format.
    
    Expected format:
        {(x, y, time_bucket, day_of_week): active_count, ...}
        or {(x, y, hour, day_of_week): active_count, ...}
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Dictionary of active taxi counts
    """
    with open(filepath, 'rb') as f:
        loaded = pickle.load(f)
    
    # Handle bundle format (active_taxis output)
    if isinstance(loaded, dict):
        if 'data' in loaded:
            return loaded['data']
        # Check if it looks like the expected format (tuple keys)
        first_key = next(iter(loaded.keys()), None)
        if first_key is not None and isinstance(first_key, tuple):
            return loaded
    
    return loaded


def extract_demand_from_counts(
    pickup_dropoff_data: Dict[Tuple, List[int]],
) -> Dict[Tuple, int]:
    """
    Extract demand (pickup counts) from pickup_dropoff_counts data.
    
    Args:
        pickup_dropoff_data: Full pickup/dropoff data
        
    Returns:
        Dictionary of demand values per key
    """
    demand = {}
    for key, counts in pickup_dropoff_data.items():
        if isinstance(counts, (list, tuple)) and len(counts) >= 1:
            demand[key] = int(counts[0])  # First element is pickup count
        elif isinstance(counts, (int, float)):
            demand[key] = int(counts)
    return demand


def aggregate_to_period(
    data: Dict[Tuple, Any],
    period_type: str,
    aggregator: Callable = sum,
) -> Dict[Tuple, Any]:
    """
    Aggregate data by time period.
    
    Args:
        data: Input data with keys (x, y, time_bucket, day)
        period_type: "time_bucket", "hourly", "daily", or "all"
        aggregator: Function to aggregate values (default: sum)
        
    Returns:
        Aggregated data with new period keys
    """
    aggregated = defaultdict(list)
    
    # Detect if data is already at hourly resolution
    # (time values 0-23 instead of 1-288)
    time_values = [k[2] for k in data.keys() if len(k) >= 3]
    if time_values:
        max_time = max(int(t) if isinstance(t, (int, float, str)) and str(t).replace('.','').replace('-','').isdigit() else 0 for t in time_values)
        is_already_hourly = max_time <= 23
    else:
        is_already_hourly = False
    
    for key, value in data.items():
        x, y, time_bucket, day = key
        
        # Convert to int in case of string values
        try:
            time_bucket = int(time_bucket)
            day = int(day)
            x = int(x)
            y = int(y)
        except (ValueError, TypeError):
            continue  # Skip invalid keys
        
        if period_type == "time_bucket":
            new_key = (x, y, time_bucket, day)  # No aggregation
        elif period_type == "hourly":
            if is_already_hourly:
                # Data is already at hourly resolution
                hour = time_bucket
            else:
                # Convert 5-min buckets to hours (1-288 -> 0-23)
                hour = (time_bucket - 1) // 12  # 12 5-min buckets per hour
            new_key = (x, y, hour, day)
        elif period_type == "daily":
            new_key = (x, y, 0, day)  # All times to period 0
        elif period_type == "all":
            new_key = (x, y, 0, 0)  # All to single period
        else:
            new_key = (x, y, time_bucket, day)
        
        aggregated[new_key].append(value)
    
    # Apply aggregator
    result = {}
    for key, values in aggregated.items():
        result[key] = aggregator(values)
    
    return result


def get_unique_periods(
    data: Dict[Tuple, Any],
) -> List[Tuple[int, int]]:
    """
    Get unique (time, day) period identifiers from data.
    
    Args:
        data: Data dictionary with keys containing time and day
        
    Returns:
        List of unique (time, day) tuples
    """
    periods = set()
    for key in data.keys():
        if len(key) >= 4:
            periods.add((key[2], key[3]))  # (time, day)
        elif len(key) >= 2:
            periods.add((key[-2], key[-1]))
    return sorted(periods)


def filter_by_period(
    data: Dict[Tuple, Any],
    period: Tuple[int, int],
) -> Dict[Tuple, Any]:
    """
    Filter data to a specific period.
    
    Args:
        data: Full data dictionary
        period: (time, day) tuple
        
    Returns:
        Filtered data for that period
    """
    time_val, day_val = period
    filtered = {}
    
    for key, value in data.items():
        if len(key) >= 4 and key[2] == time_val and key[3] == day_val:
            filtered[key] = value
        elif len(key) >= 2 and key[-2] == time_val and key[-1] == day_val:
            filtered[key] = value
    
    return filtered


def filter_by_days(
    data: Dict[Tuple, Any],
    days: List[int],
) -> Dict[Tuple, Any]:
    """
    Filter data to specific days of week.
    
    Args:
        data: Data dictionary
        days: List of day values to include
        
    Returns:
        Filtered data
    """
    filtered = {}
    for key, value in data.items():
        if len(key) >= 4:
            if key[3] in days:
                filtered[key] = value
    return filtered


def filter_by_time(
    data: Dict[Tuple, Any],
    time_range: Tuple[int, int],
) -> Dict[Tuple, Any]:
    """
    Filter data to specific time range.
    
    Args:
        data: Data dictionary
        time_range: (start, end) time bucket range (inclusive)
        
    Returns:
        Filtered data
    """
    start, end = time_range
    filtered = {}
    for key, value in data.items():
        if len(key) >= 4:
            if start <= key[2] <= end:
                filtered[key] = value
    return filtered


# =============================================================================
# DATA STATISTICS AND VALIDATION
# =============================================================================

def get_data_statistics(
    demand: Dict[Tuple, int],
    supply: Dict[Tuple, int],
    ratios: Dict[Tuple, float],
) -> Dict[str, Any]:
    """
    Compute statistics about the data.
    
    Args:
        demand: Demand dictionary
        supply: Supply dictionary
        ratios: Service ratio dictionary
        
    Returns:
        Dictionary of statistics
    """
    demand_values = list(demand.values())
    supply_values = list(supply.values())
    ratio_values = list(ratios.values())
    
    stats = {
        'n_demand_entries': len(demand),
        'n_supply_entries': len(supply),
        'n_ratio_entries': len(ratios),
        'key_overlap': len(set(demand.keys()) & set(supply.keys())),
    }
    
    if demand_values:
        stats['demand'] = {
            'mean': float(np.mean(demand_values)),
            'std': float(np.std(demand_values)),
            'min': int(np.min(demand_values)),
            'max': int(np.max(demand_values)),
            'sum': int(np.sum(demand_values)),
        }
    
    if supply_values:
        stats['supply'] = {
            'mean': float(np.mean(supply_values)),
            'std': float(np.std(supply_values)),
            'min': int(np.min(supply_values)),
            'max': int(np.max(supply_values)),
        }
    
    if ratio_values:
        stats['ratio'] = {
            'mean': float(np.mean(ratio_values)),
            'std': float(np.std(ratio_values)),
            'min': float(np.min(ratio_values)),
            'max': float(np.max(ratio_values)),
            'median': float(np.median(ratio_values)),
        }
    
    return stats


def validate_data_alignment(
    demand: Dict[Tuple, int],
    supply: Dict[Tuple, int],
) -> List[str]:
    """
    Validate that demand and supply data are properly aligned.
    
    Args:
        demand: Demand dictionary
        supply: Supply dictionary
        
    Returns:
        List of warning/error messages (empty if valid)
    """
    issues = []
    
    demand_keys = set(demand.keys())
    supply_keys = set(supply.keys())
    
    overlap = demand_keys & supply_keys
    only_demand = demand_keys - supply_keys
    only_supply = supply_keys - demand_keys
    
    if len(overlap) == 0:
        issues.append("ERROR: No overlapping keys between demand and supply data")
    
    overlap_pct = len(overlap) / max(len(demand_keys), 1) * 100
    if overlap_pct < 50:
        issues.append(
            f"WARNING: Only {overlap_pct:.1f}% key overlap between demand and supply"
        )
    
    if len(only_demand) > 0:
        issues.append(
            f"INFO: {len(only_demand)} demand entries have no matching supply"
        )
    
    if len(only_supply) > 0:
        issues.append(
            f"INFO: {len(only_supply)} supply entries have no matching demand"
        )
    
    return issues


# =============================================================================
# DIFFERENTIABLE IMPLEMENTATIONS (PYTORCH)
# =============================================================================

def compute_causal_fairness_torch(
    service_ratios: 'torch.Tensor',
    expected_ratios: 'torch.Tensor',
    eps: float = 1e-8,
) -> 'torch.Tensor':
    """
    Compute differentiable causal fairness score.
    
    The expected_ratios (from frozen g lookup) should be detached
    to prevent gradient flow through the estimation function.
    
    Args:
        service_ratios: Tensor of Y values (requires_grad=True)
        expected_ratios: Tensor of g(D) values (frozen, no grad)
        eps: Numerical stability constant
        
    Returns:
        Scalar tensor containing causal fairness (R²)
    """
    import torch
    
    # Compute residuals
    residuals = service_ratios - expected_ratios
    
    # Compute variances
    var_residuals = torch.var(residuals)
    var_total = torch.var(service_ratios)
    
    # R² computation
    r_squared = 1.0 - (var_residuals / (var_total + eps))
    
    return torch.clamp(r_squared, 0.0, 1.0)


def verify_causal_fairness_gradient(
    n_samples: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Verify gradient flow through causal fairness computation.
    
    Args:
        n_samples: Number of test samples
        seed: Random seed
        
    Returns:
        Dictionary with verification results
    """
    import torch
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create test data
    # Demand-like values (positive integers)
    demands = torch.randint(1, 50, (n_samples,), dtype=torch.float32)
    
    # Supply-like values (positive, somewhat related to demand)
    noise = torch.randn(n_samples) * 2
    supply = demands * 0.5 + noise.abs() + 1
    supply.requires_grad_(True)
    
    # Service ratios Y = S/D
    service_ratios = supply / demands
    
    # Expected ratios (frozen - simulating g(d))
    # In practice, this comes from pre-computed lookup
    expected_ratios = (demands * 0.5 + 1) / demands  # Perfect prediction
    expected_ratios = expected_ratios.detach()  # Freeze
    
    # Compute causal fairness
    f_causal = compute_causal_fairness_torch(service_ratios, expected_ratios)
    
    # Backward pass
    f_causal.backward()
    
    # Collect results
    results = {
        'n_samples': n_samples,
        'f_causal': f_causal.item(),
        'gradient_exists': supply.grad is not None,
        'gradient_has_nan': supply.grad is not None and torch.isnan(supply.grad).any().item(),
        'gradient_has_inf': supply.grad is not None and torch.isinf(supply.grad).any().item(),
    }
    
    if supply.grad is not None:
        results['gradient_stats'] = {
            'mean': supply.grad.mean().item(),
            'std': supply.grad.std().item(),
            'min': supply.grad.min().item(),
            'max': supply.grad.max().item(),
        }
        results['gradients'] = supply.grad.numpy().copy()
    
    results['passed'] = (
        results['gradient_exists'] and
        not results['gradient_has_nan'] and
        not results['gradient_has_inf']
    )
    
    return results


def verify_gradient_with_estimation_method(
    method: Literal["binning", "linear", "polynomial", "isotonic", "lowess"],
    n_samples: int = 100,
    seed: int = 42,
    n_bins: int = 10,
    poly_degree: int = 2,
    lowess_frac: float = 0.3,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Verify gradient flow when using a specific g(d) estimation method.
    
    This is the key validation function for ensuring that using Isotonic
    or Binning estimation methods does not break gradient computation.
    
    The test validates that:
    1. g(d) can be fitted using the specified method
    2. Gradients flow through the causal fairness computation
    3. Gradients are numerically valid (no NaN/Inf)
    4. Gradient magnitudes are reasonable (not vanishing/exploding)
    
    CRITICAL INSIGHT:
    The g(d) function is PRE-COMPUTED and FROZEN before optimization.
    We are NOT differentiating through the fitting process - we only
    need gradients to flow through R = Y - g(D), where Y depends on supply S.
    
    Args:
        method: g(d) estimation method to test
        n_samples: Number of test samples
        seed: Random seed for reproducibility
        n_bins: Number of bins (for binning method)
        poly_degree: Polynomial degree (for polynomial method)
        lowess_frac: Smoothing fraction (for lowess method)
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary containing:
        - method: The tested method name
        - r2_fit: R² of the g(d) fit (how well g(d) fits the data)
        - f_causal: Computed causal fairness value
        - gradient_exists: Whether gradients were computed
        - gradient_valid: Whether gradients are numerically valid
        - gradient_stats: Statistics about gradient values
        - gradients: Raw gradient values
        - passed: Overall pass/fail status
        - diagnostics: Additional method-specific diagnostics
    """
    import torch
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    results = {
        'method': method,
        'n_samples': n_samples,
        'seed': seed,
    }
    
    try:
        # =================================================================
        # STEP 1: Generate realistic test data
        # =================================================================
        # Create demand values (positive, mimicking real pickup counts)
        demands_np = np.random.exponential(scale=10, size=n_samples).astype(np.float32)
        demands_np = np.clip(demands_np, 1, 100)
        
        # Create supply values related to demand (with noise)
        # Base relationship: supply tends to follow demand with some scatter
        noise = np.random.normal(0, 0.3, size=n_samples)
        supply_np = demands_np * (0.3 + 0.05 * noise) + np.random.uniform(0.5, 2, size=n_samples)
        supply_np = np.clip(supply_np, 0.1, None).astype(np.float32)
        
        # Compute service ratios (Y = S/D)
        ratios_np = supply_np / demands_np
        
        if verbose:
            print(f"Data generated: demand range [{demands_np.min():.1f}, {demands_np.max():.1f}], "
                  f"ratio range [{ratios_np.min():.3f}, {ratios_np.max():.3f}]")
        
        # =================================================================
        # STEP 2: Fit g(d) using the specified method
        # =================================================================
        g_func, diagnostics = estimate_g_function(
            demands_np, ratios_np,
            method=method,
            n_bins=n_bins,
            poly_degree=poly_degree,
            lowess_frac=lowess_frac,
        )
        results['diagnostics'] = diagnostics
        
        # Compute R² of the fit
        predicted_ratios = g_func(demands_np)
        r2_fit = compute_r_squared(ratios_np, predicted_ratios)
        results['r2_fit'] = r2_fit
        
        if verbose:
            print(f"g(d) fitted with {method}: R² = {r2_fit:.4f}")
        
        # =================================================================
        # STEP 3: Convert to PyTorch and compute gradients
        # =================================================================
        # Create PyTorch tensors
        demands_t = torch.tensor(demands_np, dtype=torch.float32)
        supply_t = torch.tensor(supply_np, dtype=torch.float32, requires_grad=True)
        
        # Compute service ratios (Y = S/D) - this is differentiable
        service_ratios_t = supply_t / (demands_t + 1e-8)
        
        # Get expected ratios from FROZEN g(d) lookup
        # The key point: g_func is pre-computed, we just look up values
        expected_ratios_np = g_func(demands_np)
        expected_ratios_t = torch.tensor(expected_ratios_np, dtype=torch.float32)
        # No requires_grad - this is frozen!
        
        # Compute causal fairness (R²)
        f_causal = compute_causal_fairness_torch(service_ratios_t, expected_ratios_t)
        results['f_causal'] = f_causal.item()
        
        if verbose:
            print(f"F_causal = {f_causal.item():.6f}")
        
        # =================================================================
        # STEP 4: Backward pass - compute gradients
        # =================================================================
        f_causal.backward()
        
        results['gradient_exists'] = supply_t.grad is not None
        
        if supply_t.grad is not None:
            grad = supply_t.grad
            
            # Check for numerical issues
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()
            is_zero = (grad.abs() < 1e-12).all().item()
            
            results['gradient_has_nan'] = has_nan
            results['gradient_has_inf'] = has_inf
            results['gradient_is_zero'] = is_zero
            results['gradient_valid'] = not has_nan and not has_inf and not is_zero
            
            # Gradient statistics
            results['gradient_stats'] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'min': grad.min().item(),
                'max': grad.max().item(),
                'abs_mean': grad.abs().mean().item(),
                'nonzero_count': (grad.abs() > 1e-12).sum().item(),
                'nonzero_pct': (grad.abs() > 1e-12).float().mean().item() * 100,
            }
            
            results['gradients'] = grad.numpy().copy()
            
            if verbose:
                print(f"Gradient stats: mean={results['gradient_stats']['mean']:.6e}, "
                      f"std={results['gradient_stats']['std']:.6e}")
        else:
            results['gradient_valid'] = False
            results['gradient_stats'] = None
        
        # =================================================================
        # STEP 5: Numerical gradient verification (finite differences)
        # =================================================================
        # Verify gradients match finite difference approximation
        # IMPORTANT: Use float64 for numerical check due to precision requirements
        eps_fd = 1e-4  # Larger epsilon needed for float32 precision
        idx_to_check = min(5, n_samples)  # Check first few elements
        
        def compute_r2_for_numerical_check(supply_arr, demand_arr, expected_arr):
            """Compute R² using PyTorch float64 for numerical precision."""
            ratios = torch.tensor(supply_arr / demand_arr, dtype=torch.float64)
            expected = torch.tensor(expected_arr, dtype=torch.float64)
            residuals = ratios - expected
            var_r = torch.var(residuals)
            var_y = torch.var(ratios)
            r2 = 1.0 - (var_r / (var_y + 1e-8))
            return torch.clamp(r2, 0.0, 1.0).item()
        
        numerical_grads = []
        analytic_grads = []
        
        expected_np = g_func(demands_np)  # Frozen g(d) lookup
        
        # Convert to float64 for numerical precision
        supply_np_f64 = supply_np.astype(np.float64)
        demands_np_f64 = demands_np.astype(np.float64)
        expected_np_f64 = expected_np.astype(np.float64)
        
        for i in range(idx_to_check):
            # Positive perturbation
            supply_plus = supply_np_f64.copy()
            supply_plus[i] += eps_fd
            f_plus = compute_r2_for_numerical_check(supply_plus, demands_np_f64, expected_np_f64)
            
            # Negative perturbation
            supply_minus = supply_np_f64.copy()
            supply_minus[i] -= eps_fd
            f_minus = compute_r2_for_numerical_check(supply_minus, demands_np_f64, expected_np_f64)
            
            # Numerical gradient
            num_grad = (f_plus - f_minus) / (2 * eps_fd)
            numerical_grads.append(num_grad)
            
            if supply_t.grad is not None:
                analytic_grads.append(supply_t.grad[i].item())
        
        if numerical_grads and analytic_grads:
            numerical_grads = np.array(numerical_grads)
            analytic_grads = np.array(analytic_grads)
            
            # Relative error
            denom = np.maximum(np.abs(numerical_grads), np.abs(analytic_grads))
            denom = np.maximum(denom, 1e-8)
            rel_errors = np.abs(numerical_grads - analytic_grads) / denom
            
            results['numerical_check'] = {
                'numerical_grads': numerical_grads.tolist(),
                'analytic_grads': analytic_grads.tolist(),
                'relative_errors': rel_errors.tolist(),
                'max_rel_error': rel_errors.max(),
                'mean_rel_error': rel_errors.mean(),
                'gradients_match': rel_errors.max() < 0.1,  # Within 10%
            }
            
            if verbose:
                print(f"Numerical check: max rel error = {rel_errors.max():.4f}")
        
        # =================================================================
        # STEP 6: Determine overall pass/fail
        # =================================================================
        results['passed'] = (
            results['gradient_exists'] and
            results.get('gradient_valid', False) and
            results.get('numerical_check', {}).get('gradients_match', True)
        )
        
    except Exception as e:
        results['error'] = str(e)
        results['passed'] = False
        import traceback
        results['traceback'] = traceback.format_exc()
    
    return results


def verify_all_estimation_methods(
    n_samples: int = 100,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Run gradient verification for all g(d) estimation methods.
    
    This comprehensive test validates that each estimation method
    (binning, isotonic, polynomial, linear, lowess) allows gradients
    to flow properly through the causal fairness computation.
    
    Args:
        n_samples: Number of test samples
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping method names to verification results
    """
    methods = ["binning", "linear", "polynomial", "isotonic", "lowess"]
    results = {}
    
    for method in methods:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Testing method: {method}")
            print('='*50)
        
        results[method] = verify_gradient_with_estimation_method(
            method=method,
            n_samples=n_samples,
            seed=seed,
            verbose=verbose,
        )
    
    return results


def create_gradient_verification_report(
    results: Dict[str, Dict[str, Any]],
) -> str:
    """
    Create a formatted report from gradient verification results.
    
    Args:
        results: Results from verify_all_estimation_methods
        
    Returns:
        Formatted markdown report
    """
    lines = [
        "# Gradient Verification Report",
        "",
        "## Summary",
        "",
        "| Method | R² Fit | F_causal | Gradient Valid | Numerical Check | Status |",
        "|--------|--------|----------|----------------|-----------------|--------|",
    ]
    
    for method, res in results.items():
        if res.get('error'):
            lines.append(f"| {method} | ERROR | - | - | - | ❌ FAILED |")
            continue
            
        r2_fit = res.get('r2_fit', 0)
        f_causal = res.get('f_causal', 0)
        grad_valid = "✅" if res.get('gradient_valid', False) else "❌"
        
        num_check = res.get('numerical_check', {})
        num_match = "✅" if num_check.get('gradients_match', False) else "⚠️"
        
        status = "✅ PASS" if res.get('passed', False) else "❌ FAIL"
        
        lines.append(
            f"| {method} | {r2_fit:.4f} | {f_causal:.4f} | {grad_valid} | {num_match} | {status} |"
        )
    
    lines.extend([
        "",
        "## Details",
        "",
    ])
    
    for method, res in results.items():
        lines.append(f"### {method.title()}")
        lines.append("")
        
        if res.get('error'):
            lines.append(f"**Error**: {res['error']}")
            lines.append("")
            continue
        
        lines.append(f"- **R² Fit**: {res.get('r2_fit', 0):.6f}")
        lines.append(f"- **F_causal**: {res.get('f_causal', 0):.6f}")
        
        grad_stats = res.get('gradient_stats', {})
        if grad_stats:
            lines.append(f"- **Gradient Mean**: {grad_stats.get('mean', 0):.6e}")
            lines.append(f"- **Gradient Std**: {grad_stats.get('std', 0):.6e}")
            lines.append(f"- **Non-zero %**: {grad_stats.get('nonzero_pct', 0):.1f}%")
        
        num_check = res.get('numerical_check', {})
        if num_check:
            lines.append(f"- **Max Rel Error**: {num_check.get('max_rel_error', 0):.6f}")
            lines.append(f"- **Numerical Match**: {num_check.get('gradients_match', False)}")
        
        lines.append("")
    
    return "\n".join(lines)


class DifferentiableCausalFairness:
    """
    Differentiable causal fairness computation using PyTorch.
    
    This class provides end-to-end differentiable computation of the
    causal fairness term, enabling gradient-based trajectory modification.
    
    The causal fairness is computed as R²:
        F_causal = 1 - Var(Y - g(D)) / Var(Y)
    
    where:
        - Y = Service ratio (Supply / Demand)
        - g(D) = Expected service ratio given demand (pre-computed, frozen)
    
    CRITICAL: The g(d) lookup must be pre-computed and frozen before
    optimization to ensure gradients only flow through the service ratios.
    
    Example:
        >>> # Pre-compute g(d) from original data
        >>> g_func, _ = estimate_g_function(demands, ratios, method='binning')
        >>> 
        >>> # Create module with frozen lookup
        >>> module = DifferentiableCausalFairness(demands_original, g_func)
        >>> 
        >>> # During optimization
        >>> supply = torch.tensor(supply_values, requires_grad=True)
        >>> demand = torch.tensor(demand_values)  # No grad needed
        >>> f_causal = module.compute(supply, demand)
        >>> f_causal.backward()
    """
    
    def __init__(
        self,
        frozen_demands: np.ndarray,
        g_function: Callable[[np.ndarray], np.ndarray],
        eps: float = 1e-8,
    ):
        """
        Initialize the differentiable causal fairness module.
        
        Args:
            frozen_demands: Original demand values for computing g(d)
            g_function: Pre-computed g(d) function
            eps: Numerical stability constant
        """
        import torch
        
        self.eps = eps
        self.g_function = g_function
        
        # Pre-compute expected ratios for the original demands
        self.frozen_demands = frozen_demands
        self.frozen_expected = g_function(frozen_demands)
    
    def compute(
        self,
        supply: 'torch.Tensor',
        demand: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Compute differentiable causal fairness.
        
        Args:
            supply: Supply tensor (requires_grad=True for optimization)
            demand: Demand tensor (no grad needed)
            
        Returns:
            Scalar tensor containing causal fairness (R²)
        """
        import torch
        
        # Compute service ratios
        service_ratios = supply / (demand + self.eps)
        
        # Get expected ratios from frozen g lookup
        # Use NumPy for lookup, then convert back to tensor
        demand_np = demand.detach().cpu().numpy()
        expected_np = self.g_function(demand_np)
        expected_ratios = torch.tensor(
            expected_np, 
            dtype=supply.dtype, 
            device=supply.device
        )
        
        # Compute causal fairness (R²)
        return compute_causal_fairness_torch(service_ratios, expected_ratios, self.eps)
    
    def compute_from_ratios(
        self,
        service_ratios: 'torch.Tensor',
        demand: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Compute causal fairness from pre-computed service ratios.
        
        Args:
            service_ratios: Y = S/D tensor (requires_grad=True)
            demand: Demand tensor for g(d) lookup
            
        Returns:
            Scalar tensor containing causal fairness
        """
        import torch
        
        # Get expected ratios from frozen g lookup
        demand_np = demand.detach().cpu().numpy()
        expected_np = self.g_function(demand_np)
        expected_ratios = torch.tensor(
            expected_np,
            dtype=service_ratios.dtype,
            device=service_ratios.device
        )
        
        return compute_causal_fairness_torch(service_ratios, expected_ratios, self.eps)
    
    @staticmethod
    def verify_gradients(n_cells: int = 100, seed: int = 42) -> Dict[str, Any]:
        """
        Verify gradient flow through the module.
        
        Args:
            n_cells: Number of cells to test
            seed: Random seed
            
        Returns:
            Dictionary with verification results
        """
        return verify_causal_fairness_gradient(n_cells, seed)


class DifferentiableCausalFairnessWithSoftCounts:
    """
    Extended differentiable causal fairness with soft count support.
    
    This class integrates the soft cell assignment mechanism with the
    causal fairness computation, enabling gradient-based trajectory
    optimization.
    
    The key extension over DifferentiableCausalFairness is the ability
    to compute fairness from trajectory pickup locations rather than
    pre-computed demand counts, with gradients flowing back to the locations.
    
    Note: Causal fairness measures how well service (supply/demand ratio)
    aligns with what we expect given demand. Modifying pickups changes demand,
    which affects the service ratio and residuals.
    
    Example:
        >>> from soft_cell_assignment import SoftCellAssignment
        >>> 
        >>> # Pre-compute g(d) from original data
        >>> g_func, _ = estimate_g_function(demands, ratios, 'binning')
        >>> 
        >>> # Create modules
        >>> soft_assign = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5)
        >>> fairness_module = DifferentiableCausalFairnessWithSoftCounts(
        ...     grid_dims=(48, 90),
        ...     g_function=g_func,
        ...     soft_assignment=soft_assign,
        ... )
        >>> 
        >>> # Optimize trajectory locations
        >>> locations = torch.tensor([[24.5, 45.3]], requires_grad=True)
        >>> original_cells = torch.tensor([[24, 45]])
        >>> 
        >>> f_causal = fairness_module.compute_from_locations(
        ...     locations, original_cells, base_demand, supply
        ... )
        >>> f_causal.backward()
        >>> print(locations.grad)  # Gradient for optimization
    """
    
    def __init__(
        self,
        grid_dims: Tuple[int, int],
        g_function: Callable[[np.ndarray], np.ndarray],
        soft_assignment: Optional['SoftCellAssignment'] = None,
        neighborhood_size: int = 5,
        initial_temperature: float = 1.0,
        eps: float = 1e-8,
    ):
        """
        Initialize the extended causal fairness module.
        
        Args:
            grid_dims: Spatial grid dimensions (x_cells, y_cells)
            g_function: Pre-computed g(d) function (MUST BE FROZEN)
            soft_assignment: Pre-created SoftCellAssignment module, or None to create
            neighborhood_size: Neighborhood size (if creating soft_assignment)
            initial_temperature: Initial temperature (if creating soft_assignment)
            eps: Numerical stability constant
        """
        import torch
        
        self.grid_dims = grid_dims
        self.g_function = g_function
        self.eps = eps
        
        # Create or use provided soft assignment module
        if soft_assignment is not None:
            self.soft_assignment = soft_assignment
        else:
            from soft_cell_assignment import SoftCellAssignment
            self.soft_assignment = SoftCellAssignment(
                grid_dims=grid_dims,
                neighborhood_size=neighborhood_size,
                initial_temperature=initial_temperature,
            )
    
    def compute_from_locations(
        self,
        pickup_locations: 'torch.Tensor',
        pickup_original_cells: 'torch.Tensor',
        base_demand: 'torch.Tensor',
        supply: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Compute causal fairness from trajectory pickup locations.
        
        This method:
        1. Computes soft cell assignments for pickup locations
        2. Updates demand counts using soft assignments
        3. Computes service ratios (supply / demand)
        4. Computes R²-based causal fairness
        
        Gradients flow back through the chain to the pickup locations.
        
        IMPORTANT: The g_function is evaluated on the CURRENT demand values,
        but remains frozen (not re-fitted). This is correct behavior.
        
        Args:
            pickup_locations: Continuous pickup coordinates [n_traj, 2]
            pickup_original_cells: Original pickup cells [n_traj, 2]
            base_demand: Demand from non-modified trajectories [x, y]
            supply: Supply (active taxis) per cell [x, y]
            
        Returns:
            Scalar tensor containing causal fairness value [0, 1]
        """
        import torch
        from soft_cell_assignment import compute_soft_counts
        # from objective_function.soft_cell_assignment import compute_soft_counts
        
        # Compute soft pickup assignments
        pickup_probs = self.soft_assignment(pickup_locations, pickup_original_cells)
        
        # Compute soft demand counts
        soft_demand = compute_soft_counts(
            pickup_probs, pickup_original_cells, self.grid_dims, base_demand
        )
        
        # Compute service ratios Y = S / D
        service_ratios = supply / (soft_demand + self.eps)
        
        # Get expected ratios from frozen g(d)
        # NOTE: We evaluate g at the SOFT demand values
        demand_np = soft_demand.detach().cpu().numpy()
        expected_np = self.g_function(demand_np.flatten())
        expected_ratios = torch.tensor(
            expected_np.reshape(self.grid_dims),
            dtype=service_ratios.dtype,
            device=service_ratios.device,
        )
        
        # Compute causal fairness
        service_ratios_flat = service_ratios.view(-1)
        expected_ratios_flat = expected_ratios.view(-1)
        
        return compute_causal_fairness_torch(
            service_ratios_flat, expected_ratios_flat, self.eps
        )
    
    def compute_from_single_trajectory(
        self,
        pickup_location: 'torch.Tensor',
        pickup_original_cell: 'torch.Tensor',
        current_demand: 'torch.Tensor',
        supply: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Compute causal fairness impact of modifying a single trajectory.
        
        This is optimized for modifying one trajectory at a time.
        
        Args:
            pickup_location: Continuous pickup coordinate [2] or [1, 2]
            pickup_original_cell: Original pickup cell [2] or [1, 2]
            current_demand: Current demand counts [x, y]
            supply: Supply counts [x, y]
            
        Returns:
            Scalar tensor containing causal fairness value
        """
        import torch
        from objective_function.soft_cell_assignment import update_counts_with_soft_assignment
        
        # Handle dimensions
        if pickup_location.dim() == 1:
            pickup_location = pickup_location.unsqueeze(0)
            pickup_original_cell = pickup_original_cell.unsqueeze(0)
        
        # Compute soft assignment
        pickup_probs = self.soft_assignment(pickup_location, pickup_original_cell)
        
        # Update demand (remove hard assignment, add soft)
        updated_demand = update_counts_with_soft_assignment(
            current_demand, pickup_probs, pickup_original_cell, subtract_original=True
        )
        
        # Compute service ratios
        service_ratios = supply / (updated_demand + self.eps)
        
        # Get expected ratios
        demand_np = updated_demand.detach().cpu().numpy()
        expected_np = self.g_function(demand_np.flatten())
        expected_ratios = torch.tensor(
            expected_np.reshape(self.grid_dims),
            dtype=service_ratios.dtype,
            device=service_ratios.device,
        )
        
        # Compute causal fairness
        return compute_causal_fairness_torch(
            service_ratios.view(-1), expected_ratios.view(-1), self.eps
        )
    
    def set_temperature(self, temperature: float) -> None:
        """Update soft assignment temperature."""
        self.soft_assignment.set_temperature(temperature)
    
    @staticmethod
    def verify_end_to_end_gradients(
        grid_dims: Tuple[int, int] = (10, 10),
        n_trajectories: int = 5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Verify gradients flow from trajectory locations to causal fairness.
        
        Args:
            grid_dims: Grid dimensions for testing
            n_trajectories: Number of test trajectories
            seed: Random seed
            
        Returns:
            Verification results dictionary
        """
        import torch
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create a simple g function for testing
        def simple_g(d):
            d = np.atleast_1d(d)
            return 10.0 / (d + 1.0)  # Simple inverse relationship
        
        # Create module
        module = DifferentiableCausalFairnessWithSoftCounts(
            grid_dims=grid_dims,
            g_function=simple_g,
            neighborhood_size=5,
            initial_temperature=1.0,
        )
        
        # Generate test data
        k = 2  # neighborhood half-size
        
        pickup_locations = torch.tensor([
            [np.random.uniform(k + 1, grid_dims[0] - k - 1),
             np.random.uniform(k + 1, grid_dims[1] - k - 1)]
            for _ in range(n_trajectories)
        ], dtype=torch.float32, requires_grad=True)
        
        pickup_original_cells = pickup_locations.detach().round()
        
        # Base demand and supply
        base_demand = torch.rand(grid_dims) * 10 + 1  # Avoid zeros
        supply = torch.ones(grid_dims) * 5.0
        
        # Forward pass
        f_causal = module.compute_from_locations(
            pickup_locations, pickup_original_cells,
            base_demand, supply,
        )
        
        # Backward pass
        f_causal.backward()
        
        # Check gradients
        pickup_grad = pickup_locations.grad
        
        results = {
            'f_causal': f_causal.item(),
            'pickup_gradient_exists': pickup_grad is not None,
            'pickup_gradient_has_nan': pickup_grad is not None and torch.isnan(pickup_grad).any().item(),
        }
        
        if pickup_grad is not None and not torch.isnan(pickup_grad).any():
            results['pickup_gradient_stats'] = {
                'mean': pickup_grad.mean().item(),
                'std': pickup_grad.std().item(),
                'min': pickup_grad.min().item(),
                'max': pickup_grad.max().item(),
            }
        
        results['passed'] = (
            results['pickup_gradient_exists'] and
            not results['pickup_gradient_has_nan']
        )
        
        return results


def compute_demand_conditional_deviation(
    service_ratios: np.ndarray,
    g_function: Callable[[np.ndarray], np.ndarray],
    demands: np.ndarray,
) -> np.ndarray:
    """
    Compute Demand-Conditional Deviation (DCD) for all cells.
    
    DCD = |Y - g(D)| where Y is the service ratio and g(D) is expected.
    
    Args:
        service_ratios: Array of service ratios Y = S/D
        g_function: Pre-computed g(d) function
        demands: Array of demand values
        
    Returns:
        Array of DCD values (same shape as inputs)
    """
    expected = g_function(demands.flatten()).reshape(demands.shape)
    return np.abs(service_ratios - expected)


def compute_trajectory_causal_attribution(
    demand: np.ndarray,
    supply: np.ndarray,
    g_function: Callable[[np.ndarray], np.ndarray],
    trajectory_pickup_cell: Tuple[int, int],
) -> Dict[str, float]:
    """
    Compute causal fairness attribution for a trajectory.
    
    Returns the DCD score for the trajectory's pickup cell.
    
    Args:
        demand: Demand grid
        supply: Supply grid
        g_function: Pre-computed g(d) function
        trajectory_pickup_cell: (x, y) pickup cell
        
    Returns:
        Dictionary with attribution metrics
    """
    eps = 1e-8
    
    # Compute service ratios
    service_ratios = supply / (demand + eps)
    
    # Compute expected ratios
    expected = g_function(demand.flatten()).reshape(demand.shape)
    
    # Compute residuals
    residuals = service_ratios - expected
    
    # Get values at trajectory's cell
    x, y = trajectory_pickup_cell
    
    return {
        'dcd': abs(residuals[x, y]),
        'residual': float(residuals[x, y]),
        'service_ratio': float(service_ratios[x, y]),
        'expected_ratio': float(expected[x, y]),
        'demand_at_cell': float(demand[x, y]),
        'supply_at_cell': float(supply[x, y]),
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals)),
    }


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def create_grid_heatmap_data(
    data: Dict[Tuple, float],
    grid_dims: Tuple[int, int],
    default_value: float = 0.0,
) -> np.ndarray:
    """
    Convert dictionary data to 2D grid array for heatmap visualization.
    
    Args:
        data: Dictionary with (x, y, ...) keys
        grid_dims: (x_size, y_size) dimensions
        default_value: Value for cells with no data
        
    Returns:
        2D numpy array of shape (y_size, x_size)
    """
    grid = np.full((grid_dims[1], grid_dims[0]), default_value)
    
    for key, value in data.items():
        x, y = key[0], key[1]
        # Handle 1-indexed data
        if x >= 1 and y >= 1:
            x_idx = x - 1
            y_idx = y - 1
        else:
            x_idx = x
            y_idx = y
        
        if 0 <= x_idx < grid_dims[0] and 0 <= y_idx < grid_dims[1]:
            grid[y_idx, x_idx] = value
    
    return grid


def aggregate_to_grid(
    data: Dict[Tuple, float],
    grid_dims: Tuple[int, int],
    aggregator: str = 'mean',
) -> np.ndarray:
    """
    Aggregate time-varying data to a single grid.
    
    Args:
        data: Dictionary with (x, y, time, day) keys
        grid_dims: (x_size, y_size) dimensions
        aggregator: 'mean', 'sum', 'max', 'min'
        
    Returns:
        2D numpy array
    """
    # Group by (x, y)
    cell_values = defaultdict(list)
    for key, value in data.items():
        x, y = key[0], key[1]
        cell_values[(x, y)].append(value)
    
    # Aggregate
    aggregated = {}
    for (x, y), values in cell_values.items():
        if aggregator == 'mean':
            aggregated[(x, y)] = np.mean(values)
        elif aggregator == 'sum':
            aggregated[(x, y)] = np.sum(values)
        elif aggregator == 'max':
            aggregated[(x, y)] = np.max(values)
        elif aggregator == 'min':
            aggregated[(x, y)] = np.min(values)
    
    return create_grid_heatmap_data(aggregated, grid_dims)


# =============================================================================
# DEMOGRAPHIC FAIRNESS METRICS (Phase 2)
# =============================================================================

def prepare_demographic_analysis_data(
    demands: np.ndarray,
    ratios: np.ndarray,
    expected: np.ndarray,
    keys: List[Tuple],
    demo_grid: np.ndarray,
    feature_names: List[str],
    district_id_grid: np.ndarray,
    valid_mask: np.ndarray,
    district_names: List[str],
    data_is_one_indexed: bool = True,
) -> pd.DataFrame:
    """
    Build unified DataFrame joining demand/supply/residual data with demographics per cell.

    Handles 1→0 index conversion for raw pickup data keys. Filters out unmapped
    cells (district_id == -1 or NaN demographics).

    Args:
        demands: Array of demand values (from breakdown['components']['demands'])
        ratios: Array of service ratios (from breakdown['components']['ratios'])
        expected: Array of expected ratios (from breakdown['components']['expected'])
        keys: List of (x, y, time, day) tuples from breakdown['components']['keys']
        demo_grid: Demographics grid of shape (48, 90, n_features), 0-indexed
        feature_names: List of demographic feature names
        district_id_grid: Grid of district IDs, shape (48, 90), -1 for unmapped
        valid_mask: Boolean mask, shape (48, 90), True for valid cells
        district_names: List of district names indexed by district ID
        data_is_one_indexed: If True, subtract 1 from x,y in keys to align with 0-indexed grids

    Returns:
        DataFrame with columns: x, y, demand, ratio, expected, residual,
        district_id, district_name, and one column per demographic feature.
    """
    records = []
    residuals = np.array(ratios) - np.array(expected)

    for i, key in enumerate(keys):
        x, y = int(key[0]), int(key[1])

        # Convert 1-indexed raw data keys to 0-indexed grid coordinates
        if data_is_one_indexed:
            x_grid = x - 1
            y_grid = y - 1
        else:
            x_grid = x
            y_grid = y

        # Bounds check
        if x_grid < 0 or x_grid >= demo_grid.shape[0]:
            continue
        if y_grid < 0 or y_grid >= demo_grid.shape[1]:
            continue

        # Skip unmapped cells
        if not valid_mask[x_grid, y_grid]:
            continue

        dist_id = int(district_id_grid[x_grid, y_grid])
        if dist_id < 0:
            continue

        # Skip cells with NaN demographics
        demo_vals = demo_grid[x_grid, y_grid, :]
        if np.any(np.isnan(demo_vals)):
            continue

        dist_name = district_names[dist_id] if dist_id < len(district_names) else f"District {dist_id}"

        record = {
            'x': x_grid,
            'y': y_grid,
            'demand': demands[i],
            'ratio': ratios[i],
            'expected': expected[i],
            'residual': residuals[i],
            'district_id': dist_id,
            'district_name': dist_name,
        }
        for fi, fname in enumerate(feature_names):
            record[fname] = demo_vals[fi]

        records.append(record)

    return pd.DataFrame(records)


def compute_option_a1_demographic_attribution(
    demands: np.ndarray,
    ratios: np.ndarray,
    demographic_features: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Option A1: Demographic Attribution via conditional regression g(D, x).

    Trains a model g(D, x) = E[Y | D, x] using polynomial demand features
    plus standardized demographics. Then measures how much of the model's
    predictions depend on demographics by comparing g(D, x) vs g(D, x̄):

        F_causal = 1 - Var[g(D,x) - g(D,x̄)] / Var[Y]

    where x̄ is the mean demographic vector. This captures the fraction of
    predicted variation that is *attributable to demographics* rather than demand.

    Args:
        demands: Array of demand values, shape (n,)
        ratios: Array of service ratios Y = S/D, shape (n,)
        demographic_features: Array of demographic features, shape (n, p)
        feature_names: List of feature names

    Returns:
        Dict with f_causal, var_attribution, var_y, g_r2, coefficients,
        predicted_full, predicted_mean_demo
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    n = len(demands)
    p = demographic_features.shape[1] if demographic_features.ndim > 1 else 0

    if n < 5 or p == 0:
        return {
            'f_causal': 1.0,
            'var_attribution': 0.0,
            'var_y': 0.0,
            'g_r2': 0.0,
            'coefficients': {},
            'error': 'Insufficient data',
        }

    # Build feature matrix: polynomial demand + standardized demographics
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_demand = poly.fit_transform(demands.reshape(-1, 1))

    scaler = StandardScaler()
    X_demo_scaled = scaler.fit_transform(demographic_features)

    X_full = np.hstack([X_demand, X_demo_scaled])

    # Fit g(D, x)
    model = LinearRegression()
    model.fit(X_full, ratios)
    g_r2 = max(0.0, model.score(X_full, ratios))

    # Predictions with actual demographics: g(D_c, x_c)
    predicted_full = model.predict(X_full)

    # Predictions with mean demographics: g(D_c, x̄)
    x_bar_scaled = np.zeros((1, p))  # Mean of standardized features = 0
    X_mean_demo = np.hstack([X_demand, np.tile(x_bar_scaled, (n, 1))])
    predicted_mean_demo = model.predict(X_mean_demo)

    # Demographic attribution: how much predictions change due to demographics
    attribution = predicted_full - predicted_mean_demo

    var_attribution = float(np.var(attribution))
    var_y = float(np.var(ratios))

    if var_y < 1e-10:
        f_causal = 1.0
    else:
        f_causal = max(0.0, 1.0 - var_attribution / var_y)

    # Extract demographic coefficients from the full model
    n_demand_features = X_demand.shape[1]
    demo_coefficients = {}
    for i, name in enumerate(feature_names):
        demo_coefficients[name] = float(model.coef_[n_demand_features + i])

    return {
        'f_causal': f_causal,
        'var_attribution': var_attribution,
        'var_y': var_y,
        'g_r2': g_r2,
        'intercept': float(model.intercept_),
        'coefficients': demo_coefficients,
        'demand_coefficients': model.coef_[:n_demand_features].tolist(),
        'predicted_full': predicted_full,
        'predicted_mean_demo': predicted_mean_demo,
        'attribution': attribution,
        'n_samples': n,
    }


def compute_option_a2_conditional_r_squared(
    demands: np.ndarray,
    ratios: np.ndarray,
    demographic_features: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Option A2: Conditional R² using demographically-aware g(D, x).

    Fits g(D, x) = E[Y | D, x] and computes R² of the full model.
    This is the "simpler alternative" from the reformulation plan:

        R_c = Y_c - g(D_c, x_c)
        F_causal = R² = 1 - Var(R) / Var(Y)

    Higher R² means the model (which includes demographics) better explains
    the observed service ratio. Note: this measures model fit quality, which
    is fundamentally different from measuring whether demographics *should*
    influence service.

    Args:
        demands: Array of demand values, shape (n,)
        ratios: Array of service ratios Y = S/D, shape (n,)
        demographic_features: Array of demographic features, shape (n, p)
        feature_names: List of feature names

    Returns:
        Dict with f_causal (= R²), residuals, model coefficients
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    n = len(demands)
    p = demographic_features.shape[1] if demographic_features.ndim > 1 else 0

    if n < 5 or p == 0:
        return {
            'f_causal': 0.0,
            'r_squared': 0.0,
            'residuals': np.array([]),
            'coefficients': {},
            'error': 'Insufficient data',
        }

    # Build feature matrix: polynomial demand + standardized demographics
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_demand = poly.fit_transform(demands.reshape(-1, 1))

    scaler = StandardScaler()
    X_demo_scaled = scaler.fit_transform(demographic_features)

    X_full = np.hstack([X_demand, X_demo_scaled])

    # Fit g(D, x)
    model = LinearRegression()
    model.fit(X_full, ratios)

    r_squared = max(0.0, model.score(X_full, ratios))

    predicted = model.predict(X_full)
    residuals = ratios - predicted

    # Also compute demand-only R² for comparison
    model_demand_only = LinearRegression()
    model_demand_only.fit(X_demand, ratios)
    r2_demand_only = max(0.0, model_demand_only.score(X_demand, ratios))

    # Extract coefficients
    n_demand_features = X_demand.shape[1]
    demo_coefficients = {}
    for i, name in enumerate(feature_names):
        demo_coefficients[name] = float(model.coef_[n_demand_features + i])

    return {
        'f_causal': r_squared,
        'r_squared': r_squared,
        'r2_demand_only': r2_demand_only,
        'r2_improvement': r_squared - r2_demand_only,
        'residuals': residuals,
        'predicted': predicted,
        'intercept': float(model.intercept_),
        'coefficients': demo_coefficients,
        'demand_coefficients': model.coef_[:n_demand_features].tolist(),
        'var_residual': float(np.var(residuals)),
        'var_y': float(np.var(ratios)),
        'n_samples': n,
    }


def compute_option_b_demographic_disparity(
    residuals: np.ndarray,
    demographic_features: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Option B: Regress residuals on demographics. F_causal = 1 - R².

    After removing the effect of demand via g₀(D), if residuals still correlate
    with demographics, that indicates demographic-driven unfairness.

    Args:
        residuals: Array of residuals R = Y - g₀(D), shape (n_cells,)
        demographic_features: Array of demographic features, shape (n_cells, n_features)
        feature_names: List of feature names

    Returns:
        Dict with f_causal, r_squared, coefficients, feature_importances
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    if len(residuals) < 5 or demographic_features.shape[1] == 0:
        return {
            'f_causal': 1.0,
            'r_squared': 0.0,
            'coefficients': {},
            'feature_importances': {},
            'error': 'Insufficient data',
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(demographic_features)

    model = LinearRegression()
    model.fit(X_scaled, residuals)

    r_squared = model.score(X_scaled, residuals)
    r_squared = max(0.0, r_squared)  # Clamp negative R²

    # Coefficients on standardized features = feature importances
    coefficients = {}
    feature_importances = {}
    for i, name in enumerate(feature_names):
        coefficients[name] = float(model.coef_[i])
        feature_importances[name] = float(abs(model.coef_[i]))

    # Predicted residuals from demographics
    predicted_residuals = model.predict(X_scaled)

    return {
        'f_causal': 1.0 - r_squared,
        'r_squared': r_squared,
        'intercept': float(model.intercept_),
        'coefficients': coefficients,
        'feature_importances': feature_importances,
        'predicted_residuals': predicted_residuals,
        'n_samples': len(residuals),
    }


def compute_option_c_partial_r_squared(
    demands: np.ndarray,
    ratios: np.ndarray,
    demographic_features: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Option C: Compare Y~D (poly degree 2) vs Y~D+x models. F_causal = 1 - ΔR².

    Measures the incremental explanatory power demographics have beyond demand.

    Args:
        demands: Array of demand values, shape (n_cells,)
        ratios: Array of service ratios, shape (n_cells,)
        demographic_features: Array of demographic features, shape (n_cells, n_features)
        feature_names: List of feature names

    Returns:
        Dict with f_causal, r2_reduced, r2_full, delta_r2, coefficients
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    if len(demands) < 5 or demographic_features.shape[1] == 0:
        return {
            'f_causal': 1.0,
            'r2_reduced': 0.0,
            'r2_full': 0.0,
            'delta_r2': 0.0,
            'error': 'Insufficient data',
        }

    # Reduced model: Y ~ D (polynomial degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_demand = poly.fit_transform(demands.reshape(-1, 1))

    model_reduced = LinearRegression()
    model_reduced.fit(X_demand, ratios)
    r2_reduced = max(0.0, model_reduced.score(X_demand, ratios))

    # Full model: Y ~ D + x (polynomial demand + standardized demographics)
    scaler = StandardScaler()
    X_demo_scaled = scaler.fit_transform(demographic_features)
    X_full = np.hstack([X_demand, X_demo_scaled])

    model_full = LinearRegression()
    model_full.fit(X_full, ratios)
    r2_full = max(0.0, model_full.score(X_full, ratios))

    delta_r2 = max(0.0, r2_full - r2_reduced)

    # Extract demographic coefficients from full model
    n_demand_features = X_demand.shape[1]
    demo_coefficients = {}
    for i, name in enumerate(feature_names):
        demo_coefficients[name] = float(model_full.coef_[n_demand_features + i])

    return {
        'f_causal': 1.0 - delta_r2,
        'r2_reduced': r2_reduced,
        'r2_full': r2_full,
        'delta_r2': delta_r2,
        'demand_coefficients': model_reduced.coef_.tolist(),
        'demo_coefficients': demo_coefficients,
        'n_samples': len(demands),
    }


def compute_option_d_group_fairness(
    demands: np.ndarray,
    ratios: np.ndarray,
    income_values: np.ndarray,
    n_demand_bins: int = 10,
    n_income_groups: int = 3,
) -> Dict[str, Any]:
    """
    Option D: Demand-stratified comparison across income groups.

    Bins cells by demand level, splits into income groups, and measures
    the disparity in mean service ratio between groups within each bin.

    F_causal = 1 - mean(disparity across bins)

    Args:
        demands: Array of demand values, shape (n_cells,)
        ratios: Array of service ratios, shape (n_cells,)
        income_values: Array of income proxy values, shape (n_cells,)
        n_demand_bins: Number of demand bins
        n_income_groups: Number of income groups (quantile-based)

    Returns:
        Dict with f_causal, per_bin_data, group_labels
    """
    if len(demands) < 10:
        return {
            'f_causal': 1.0,
            'per_bin_data': [],
            'error': 'Insufficient data',
        }

    # Create income groups using quantiles
    try:
        income_group_edges = np.percentile(
            income_values, np.linspace(0, 100, n_income_groups + 1)
        )
        income_group_edges = np.unique(income_group_edges)
        actual_n_groups = len(income_group_edges) - 1
    except Exception:
        return {
            'f_causal': 1.0,
            'per_bin_data': [],
            'error': 'Could not create income groups',
        }

    if actual_n_groups < 2:
        return {
            'f_causal': 1.0,
            'per_bin_data': [],
            'error': 'Insufficient income variation for grouping',
        }

    income_groups = np.digitize(income_values, income_group_edges[1:-1])  # 0-indexed groups

    # Create demand bins using quantiles
    try:
        demand_bin_edges = np.percentile(
            demands, np.linspace(0, 100, n_demand_bins + 1)
        )
        demand_bin_edges = np.unique(demand_bin_edges)
    except Exception:
        demand_bin_edges = np.linspace(demands.min(), demands.max(), n_demand_bins + 1)

    demand_bins = np.digitize(demands, demand_bin_edges[1:-1])  # 0-indexed bins

    # Compute per-bin disparity
    per_bin_data = []
    disparities = []

    for b in range(len(demand_bin_edges) - 1):
        bin_mask = demand_bins == b
        if bin_mask.sum() < 2:
            continue

        bin_center = (demand_bin_edges[b] + demand_bin_edges[min(b + 1, len(demand_bin_edges) - 1)]) / 2

        group_means = {}
        for g in range(actual_n_groups):
            group_mask = bin_mask & (income_groups == g)
            if group_mask.sum() > 0:
                group_means[g] = float(np.mean(ratios[group_mask]))

        if len(group_means) >= 2:
            disparity = max(group_means.values()) - min(group_means.values())
        else:
            disparity = 0.0

        disparities.append(disparity)
        per_bin_data.append({
            'bin_index': b,
            'demand_center': float(bin_center),
            'demand_range': [float(demand_bin_edges[b]), float(demand_bin_edges[min(b + 1, len(demand_bin_edges) - 1)])],
            'n_cells': int(bin_mask.sum()),
            'group_means': {f"Group {g}": m for g, m in group_means.items()},
            'disparity': float(disparity),
        })

    # Create group labels
    group_labels = []
    for g in range(actual_n_groups):
        low = income_group_edges[g]
        high = income_group_edges[min(g + 1, len(income_group_edges) - 1)]
        group_labels.append(f"Group {g} ({low:.0f}-{high:.0f})")

    mean_disparity = float(np.mean(disparities)) if disparities else 0.0

    # Normalize disparity by the range of Y to keep F_causal in [0, 1]
    y_range = float(np.ptp(ratios)) if len(ratios) > 1 else 1.0
    y_range = max(y_range, 1e-8)
    normalized_disparity = mean_disparity / y_range

    return {
        'f_causal': max(0.0, 1.0 - normalized_disparity),
        'mean_disparity': mean_disparity,
        'normalized_disparity': normalized_disparity,
        'per_bin_data': per_bin_data,
        'group_labels': group_labels,
        'n_demand_bins': len(demand_bin_edges) - 1,
        'n_income_groups': actual_n_groups,
        'n_samples': len(demands),
    }


def compute_residual_demographic_correlation(
    df: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute correlation matrix between residuals and demographic features.

    Args:
        df: DataFrame from prepare_demographic_analysis_data with 'residual'
            column and demographic feature columns
        feature_names: List of demographic feature names to correlate

    Returns:
        DataFrame correlation matrix (features × ['residual'])
    """
    cols = [f for f in feature_names if f in df.columns]
    if not cols or 'residual' not in df.columns:
        return pd.DataFrame()

    corr_data = {}
    for fname in cols:
        if df[fname].std() > 1e-10 and df['residual'].std() > 1e-10:
            corr_data[fname] = float(df['residual'].corr(df[fname]))
        else:
            corr_data[fname] = 0.0

    return pd.DataFrame({'Correlation with Residual': corr_data})


# =============================================================================
# G(D, X) ESTIMATOR EXPLORATION
# =============================================================================

def enrich_demographic_features(
    demo_grid: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Pre-compute derived demographic features from the raw (48, 90, 13) grid.

    Adds 7 derived features:
      - GDPperCapita: GDPin10000Yuan / (YearEndPermanentPop10k * 10000)
      - CompPerCapita: EmployeeCompensation100MYuan * 1e8 / AvgEmployedPersons
      - MigrantRatio: NonRegisteredPermanentPop10k / YearEndPermanentPop10k
      - LogGDP: log1p(GDPin10000Yuan)
      - LogHousingPrice: log1p(AvgHousingPricePerSqM)
      - LogCompensation: log1p(EmployeeCompensation100MYuan)
      - LogPopDensity: log1p(PopDensityPerKm2)

    Args:
        demo_grid: (48, 90, 13) demographics grid, NaN for unmapped cells
        feature_names: List of 13 raw feature names

    Returns:
        (enriched_grid, enriched_feature_names) where enriched_grid is
        shape (48, 90, 20) and enriched_feature_names has 20 entries.
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    derived = []
    derived_names = []
    eps = 1e-10

    # GDP per capita
    if 'GDPin10000Yuan' in name_to_idx and 'YearEndPermanentPop10k' in name_to_idx:
        gdp = demo_grid[:, :, name_to_idx['GDPin10000Yuan']]
        pop = demo_grid[:, :, name_to_idx['YearEndPermanentPop10k']]
        gdp_pc = gdp / (pop * 10000 + eps)
        derived.append(gdp_pc)
        derived_names.append('GDPperCapita')

    # Compensation per capita
    if 'EmployeeCompensation100MYuan' in name_to_idx and 'AvgEmployedPersons' in name_to_idx:
        comp = demo_grid[:, :, name_to_idx['EmployeeCompensation100MYuan']]
        emp = demo_grid[:, :, name_to_idx['AvgEmployedPersons']]
        comp_pc = comp * 1e8 / (emp + eps)
        derived.append(comp_pc)
        derived_names.append('CompPerCapita')

    # Migrant ratio
    if 'NonRegisteredPermanentPop10k' in name_to_idx and 'YearEndPermanentPop10k' in name_to_idx:
        non_reg = demo_grid[:, :, name_to_idx['NonRegisteredPermanentPop10k']]
        total = demo_grid[:, :, name_to_idx['YearEndPermanentPop10k']]
        migrant = non_reg / (total + eps)
        derived.append(migrant)
        derived_names.append('MigrantRatio')

    # Log transforms
    log_features = {
        'GDPin10000Yuan': 'LogGDP',
        'AvgHousingPricePerSqM': 'LogHousingPrice',
        'EmployeeCompensation100MYuan': 'LogCompensation',
        'PopDensityPerKm2': 'LogPopDensity',
    }
    for raw_name, log_name in log_features.items():
        if raw_name in name_to_idx:
            raw_vals = demo_grid[:, :, name_to_idx[raw_name]]
            derived.append(np.log1p(np.maximum(raw_vals, 0)))
            derived_names.append(log_name)

    if not derived:
        return demo_grid.copy(), list(feature_names)

    derived_stack = np.stack(derived, axis=-1)  # (48, 90, n_derived)
    enriched_grid = np.concatenate([demo_grid, derived_stack], axis=-1)
    enriched_names = list(feature_names) + derived_names

    return enriched_grid, enriched_names


def build_feature_matrix(
    demands: np.ndarray,
    demographic_features: np.ndarray,
    poly_degree: int = 2,
    include_interactions: bool = False,
    poly_transformer=None,
    scaler=None,
    demo_feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], Any, Any]:
    """
    Build the full feature matrix for g(D, x) models.

    Constructs: [poly(D, degree) | StandardScaler(x) | (optional) D * x_i interactions]

    Args:
        demands: (n,) demand values
        demographic_features: (n, p) demographic features (raw, will be standardized)
        poly_degree: Polynomial degree for demand features
        include_interactions: Whether to include D * x_i interaction terms
        poly_transformer: Pre-fitted PolynomialFeatures (for transform-only mode in CV)
        scaler: Pre-fitted StandardScaler (for transform-only mode in CV)
        demo_feature_names: Readable names for demographic features (e.g., "GDPperCapita").
            If provided, used instead of generic "x_0", "x_1", etc.

    Returns:
        (X, feature_names, poly_transformer, scaler)
    """
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    # Demand polynomial features
    if poly_transformer is None:
        poly_transformer = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_demand = poly_transformer.fit_transform(demands.reshape(-1, 1))
    else:
        X_demand = poly_transformer.transform(demands.reshape(-1, 1))

    # Readable demand feature names
    _sup = {2: '\u00B2', 3: '\u00B3', 4: '\u2074', 5: '\u2075'}
    if X_demand.shape[1] == 1:
        demand_names = ["Demand"]
    else:
        demand_names = ["Demand"] + [f"Demand{_sup.get(i+1, f'^{i+1}')}" for i in range(1, X_demand.shape[1])]

    # Standardize demographics
    if scaler is None:
        scaler = StandardScaler()
        X_demo = scaler.fit_transform(demographic_features)
    else:
        X_demo = scaler.transform(demographic_features)

    # Use readable demographic names if provided, else fall back to x_0, x_1, ...
    if demo_feature_names is not None and len(demo_feature_names) == X_demo.shape[1]:
        demo_names = list(demo_feature_names)
    else:
        demo_names = [f"x_{i}" for i in range(X_demo.shape[1])]

    parts = [X_demand, X_demo]
    names = demand_names + demo_names

    # Interaction terms: D * x_i (raw demand times each standardized demographic)
    if include_interactions:
        D_col = demands.reshape(-1, 1)
        interactions = D_col * X_demo
        parts.append(interactions)
        names += [f"Demand \u00d7 {n}" for n in demo_names]

    X = np.hstack(parts)
    return X, names, poly_transformer, scaler


def fit_g_dx_model(
    demands: np.ndarray,
    ratios: np.ndarray,
    demographic_features: np.ndarray,
    demo_feature_names: List[str],
    model_type: str = "ols",
    poly_degree: int = 2,
    include_interactions: bool = False,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    n_estimators: int = 100,
    max_depth: int = 5,
    hidden_layer_sizes: tuple = (64, 32),
) -> Dict[str, Any]:
    """
    Fit a g(D, x) model and return results + diagnostics.

    Supported model_type values:
      "ols", "ridge", "lasso", "elasticnet",
      "ols_interactions", "random_forest", "gradient_boosting",
      "neural_network"

    Args:
        demands: (n,) demand array
        ratios: (n,) service ratio array Y = S/D
        demographic_features: (n, p) raw demographic features
        demo_feature_names: p feature names
        model_type: one of the 8 supported types
        poly_degree: degree of polynomial demand features
        include_interactions: force interaction terms
        alpha: regularization strength (ridge/lasso/elasticnet/neural_network)
        l1_ratio: L1 ratio for elasticnet (0=ridge, 1=lasso)
        n_estimators: number of trees (RF/GB)
        max_depth: max tree depth (RF/GB)
        hidden_layer_sizes: tuple of ints defining NN hidden layers (neural_network)

    Returns:
        Dict with model, predict_fn, r2_train, residuals, feature_names,
        coefficients (linear) or feature_importances (trees), n_params.
    """
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor

    use_interactions = include_interactions or model_type == "ols_interactions"

    X, feat_names, poly, scaler = build_feature_matrix(
        demands, demographic_features,
        poly_degree=poly_degree,
        include_interactions=use_interactions,
        demo_feature_names=demo_feature_names,
    )

    # Select model
    model_map = {
        "ols": LinearRegression(),
        "ridge": Ridge(alpha=alpha),
        "lasso": Lasso(alpha=alpha, max_iter=10000),
        "elasticnet": ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000),
        "ols_interactions": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42,
        ),
        "neural_network": MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
            max_iter=1000, random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20,
        ),
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model_map[model_type]
    model.fit(X, ratios)

    r2_train = max(0.0, float(model.score(X, ratios)))
    predicted = model.predict(X)
    residuals = ratios - predicted

    # Build prediction closure that accepts raw inputs
    _poly, _scaler, _model = poly, scaler, model
    _use_interactions = use_interactions
    _poly_degree = poly_degree

    def predict_fn(new_demands, new_demo_features):
        X_new, _, _, _ = build_feature_matrix(
            new_demands, new_demo_features,
            poly_degree=_poly_degree,
            include_interactions=_use_interactions,
            poly_transformer=_poly,
            scaler=_scaler,
        )
        return _model.predict(X_new)

    # Extract coefficients or feature importances
    coefficients = None
    feature_importances = None
    is_linear = model_type in ("ols", "ridge", "lasso", "elasticnet", "ols_interactions")

    if is_linear:
        coefficients = {name: float(coef) for name, coef in zip(feat_names, model.coef_)}
    elif hasattr(model, 'feature_importances_'):
        feature_importances = {
            name: float(imp) for name, imp in zip(feat_names, model.feature_importances_)
        }
    # Neural networks: no direct coefficients or importances

    if model_type == "neural_network":
        # Count actual NN parameters: weights + biases across all layers
        n_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    else:
        n_params = X.shape[1] + 1  # features + intercept

    return {
        'model': model,
        'predict_fn': predict_fn,
        'r2_train': r2_train,
        'predicted': predicted,
        'residuals': residuals,
        'feature_names': feat_names,
        'demo_feature_names': list(demo_feature_names),
        'coefficients': coefficients,
        'feature_importances': feature_importances,
        'poly_transformer': poly,
        'scaler': scaler,
        'model_type': model_type,
        'n_params': n_params,
        'n_samples': len(demands),
        'include_interactions': use_interactions,
        'poly_degree': poly_degree,
    }


def lodo_cross_validate(
    demands: np.ndarray,
    ratios: np.ndarray,
    demographic_features: np.ndarray,
    district_ids: np.ndarray,
    demo_feature_names: List[str],
    model_type: str = "ols",
    poly_degree: int = 2,
    include_interactions: bool = False,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    n_estimators: int = 100,
    max_depth: int = 5,
    hidden_layer_sizes: tuple = (64, 32),
) -> Dict[str, Any]:
    """
    Leave-One-District-Out cross-validation for g(D, x) models.

    For each district: train on other districts, predict on held-out,
    collect out-of-fold predictions. Compute overall LODO R² from all
    OOF predictions.

    Transformers (PolynomialFeatures, StandardScaler) are re-fitted per fold
    on training data only to prevent data leakage.

    Args:
        demands, ratios, demographic_features: aligned data arrays
        district_ids: (n,) integer district IDs for each observation
        demo_feature_names: feature names
        model_type + hyperparams: same as fit_g_dx_model

    Returns:
        Dict with lodo_r2, per_district_r2, per_district_n,
        oof_predictions, oof_residuals.
    """
    unique_districts = np.unique(district_ids)
    n = len(demands)

    oof_predictions = np.full(n, np.nan)
    per_district_r2 = {}
    per_district_n = {}

    use_interactions = include_interactions or model_type == "ols_interactions"

    for dist_id in unique_districts:
        test_mask = district_ids == dist_id
        train_mask = ~test_mask

        if train_mask.sum() < 5 or test_mask.sum() < 1:
            continue

        # Fit on training data (transformers are re-fitted per fold)
        result = fit_g_dx_model(
            demands=demands[train_mask],
            ratios=ratios[train_mask],
            demographic_features=demographic_features[train_mask],
            demo_feature_names=demo_feature_names,
            model_type=model_type,
            poly_degree=poly_degree,
            include_interactions=include_interactions,
            alpha=alpha,
            l1_ratio=l1_ratio,
            n_estimators=n_estimators,
            max_depth=max_depth,
            hidden_layer_sizes=hidden_layer_sizes,
        )

        # Predict on held-out district
        preds = result['predict_fn'](
            demands[test_mask],
            demographic_features[test_mask],
        )
        oof_predictions[test_mask] = preds

        # Per-district R²
        y_test = ratios[test_mask]
        var_y = np.var(y_test)
        if var_y > 1e-10:
            r2 = max(0.0, 1.0 - np.var(y_test - preds) / var_y)
        else:
            r2 = 0.0

        per_district_r2[int(dist_id)] = float(r2)
        per_district_n[int(dist_id)] = int(test_mask.sum())

    # Overall LODO R²
    valid = ~np.isnan(oof_predictions)
    if valid.sum() > 0:
        var_y_all = np.var(ratios[valid])
        if var_y_all > 1e-10:
            lodo_r2 = max(0.0, 1.0 - np.var(ratios[valid] - oof_predictions[valid]) / var_y_all)
        else:
            lodo_r2 = 0.0
    else:
        lodo_r2 = 0.0

    oof_residuals = np.where(valid, ratios - oof_predictions, np.nan)

    return {
        'lodo_r2': float(lodo_r2),
        'per_district_r2': per_district_r2,
        'per_district_n': per_district_n,
        'oof_predictions': oof_predictions,
        'oof_residuals': oof_residuals,
        'n_folds': len(per_district_r2),
        'n_valid': int(valid.sum()),
    }


def compute_model_diagnostics(
    demands: np.ndarray,
    ratios: np.ndarray,
    demographic_features: np.ndarray,
    demo_feature_names: List[str],
    poly_degree: int = 2,
    include_interactions: bool = False,
) -> Dict[str, Any]:
    """
    Compute academic-grade diagnostics using statsmodels OLS.

    Provides coefficient p-values, VIF for multicollinearity, AIC/BIC,
    Breusch-Pagan heteroscedasticity test, and Durbin-Watson autocorrelation.

    Args:
        demands, ratios, demographic_features: data arrays
        demo_feature_names: feature names
        poly_degree: polynomial degree for demand features
        include_interactions: include D * x interactions

    Returns:
        Dict with coefficients_table, vif, aic, bic, breusch_pagan_p,
        durbin_watson, condition_number.
    """
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson as dw_stat

    X, feat_names, _, _ = build_feature_matrix(
        demands, demographic_features,
        poly_degree=poly_degree,
        include_interactions=include_interactions,
        demo_feature_names=demo_feature_names,
    )

    # Add constant for statsmodels
    X_const = sm.add_constant(X)
    const_names = ['const'] + feat_names

    model = sm.OLS(ratios, X_const).fit()

    # Coefficients table
    coef_data = {
        'Feature': const_names,
        'Coefficient': model.params.tolist(),
        'StdErr': model.bse.tolist(),
        't_stat': model.tvalues.tolist(),
        'p_value': model.pvalues.tolist(),
        'significant_05': [p < 0.05 for p in model.pvalues],
    }
    coefficients_table = pd.DataFrame(coef_data)

    # VIF (on X without constant)
    vif_data = []
    for i in range(X.shape[1]):
        try:
            vif_val = variance_inflation_factor(X, i)
        except Exception:
            vif_val = float('inf')
        vif_data.append({'Feature': feat_names[i], 'VIF': float(vif_val)})
    vif_df = pd.DataFrame(vif_data)

    # Breusch-Pagan test for heteroscedasticity
    try:
        bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(model.resid, X_const)
        breusch_pagan_p = float(bp_p)
    except Exception:
        breusch_pagan_p = float('nan')

    # Durbin-Watson
    try:
        dw = float(dw_stat(model.resid))
    except Exception:
        dw = float('nan')

    return {
        'coefficients_table': coefficients_table,
        'vif': vif_df,
        'aic': float(model.aic),
        'bic': float(model.bic),
        'r_squared': float(model.rsquared),
        'r_squared_adj': float(model.rsquared_adj),
        'breusch_pagan_p': breusch_pagan_p,
        'durbin_watson': dw,
        'condition_number': float(model.condition_number),
        'n_obs': int(model.nobs),
        'n_params': int(model.df_model + 1),
    }


def compute_permutation_importance(
    demands: np.ndarray,
    ratios: np.ndarray,
    demographic_features: np.ndarray,
    demo_feature_names: List[str],
    model_result: Dict[str, Any],
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation-based feature importance for a fitted g(D,x) model.

    For each feature, shuffles that column and measures R² drop.

    Args:
        demands, ratios, demographic_features: data arrays
        demo_feature_names: feature names
        model_result: output of fit_g_dx_model (needs model, poly_transformer, scaler)
        n_repeats: number of permutation repeats
        seed: random seed

    Returns:
        DataFrame with columns [Feature, Importance_Mean, Importance_Std]
        sorted by Importance_Mean descending.
    """
    from sklearn.inspection import permutation_importance as perm_imp

    X, feat_names, _, _ = build_feature_matrix(
        demands, demographic_features,
        poly_degree=model_result['poly_degree'],
        include_interactions=model_result['include_interactions'],
        poly_transformer=model_result['poly_transformer'],
        scaler=model_result['scaler'],
        demo_feature_names=demo_feature_names,
    )

    result = perm_imp(
        model_result['model'], X, ratios,
        n_repeats=n_repeats,
        random_state=seed,
        scoring='r2',
    )

    imp_data = []
    for i, name in enumerate(feat_names):
        imp_data.append({
            'Feature': name,
            'Importance_Mean': float(result.importances_mean[i]),
            'Importance_Std': float(result.importances_std[i]),
        })

    df = pd.DataFrame(imp_data)
    return df.sort_values('Importance_Mean', ascending=False).reset_index(drop=True)


# =============================================================================
# GPU ACCELERATION & HYPERPARAMETER SEARCH
# =============================================================================

def check_gpu_availability() -> Dict[str, Any]:
    """
    Check for GPU availability across different frameworks.

    Returns:
        Dict with gpu_available, device, framework info
    """
    import torch

    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'

    result = {
        'gpu_available': cuda_available,
        'device': device,
        'torch_cuda': cuda_available,
    }

    if cuda_available:
        result['gpu_name'] = torch.cuda.get_device_name(0)
        result['gpu_count'] = torch.cuda.device_count()

    # Check for XGBoost GPU support
    try:
        import xgboost as xgb
        result['xgboost_available'] = True
        result['xgboost_version'] = xgb.__version__
    except ImportError:
        result['xgboost_available'] = False

    return result


class PyTorchNeuralNetwork:
    """
    GPU-accelerated neural network using PyTorch.

    Compatible interface with sklearn models for drop-in replacement.
    """

    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        learning_rate=0.001,
        max_iter=1000,
        batch_size=64,
        alpha=0.0001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        device=None,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.alpha = alpha  # L2 regularization
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

        import torch
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model_ = None
        self.n_features_in_ = None
        self.n_iter_ = 0
        self.coefs_ = []
        self.intercepts_ = []

    def _build_model(self, input_size):
        """Build PyTorch sequential model."""
        import torch
        import torch.nn as nn

        layers = []
        prev_size = input_size

        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        model = nn.Sequential(*layers)
        return model.to(self.device)

    def fit(self, X, y):
        """Fit the neural network.

        Uses full-batch training when data fits in GPU memory (typical for
        tabular data), falling back to mini-batch DataLoader only for very
        large datasets. Full-batch eliminates thousands of per-batch kernel
        launch overheads that dominate training time for small operations.
        """
        import torch
        import torch.nn as nn
        import numpy as np
        import copy

        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.n_features_in_ = X.shape[1]

        # Train/validation split
        if self.early_stopping:
            n_val = max(1, int(len(X) * self.validation_fraction))
            indices = np.random.permutation(len(X))
            train_idx, val_idx = indices[n_val:], indices[:n_val]

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, y_train = X, y

        # Convert to tensors and move to device in one shot
        X_train_t = torch.from_numpy(X_train).to(self.device)
        y_train_t = torch.from_numpy(y_train).to(self.device)

        if self.early_stopping:
            X_val_t = torch.from_numpy(X_val).to(self.device)
            y_val_t = torch.from_numpy(y_val).to(self.device)

        # Decide training strategy: full-batch vs mini-batch
        # Full-batch is faster for tabular data because it avoids thousands
        # of kernel launch overheads per epoch. Only fall back to mini-batch
        # for very large datasets that risk OOM.
        data_bytes = X_train_t.nelement() * X_train_t.element_size()
        if self.device.type == 'cuda':
            gpu_mem = torch.cuda.get_device_properties(self.device).total_memory
            # Use full-batch if data < 25% of GPU memory
            use_full_batch = data_bytes < gpu_mem * 0.25
        else:
            # CPU: use full-batch for datasets under ~100MB
            use_full_batch = data_bytes < 100 * 1024 * 1024

        # Build model
        self.model_ = self._build_model(self.n_features_in_)

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.alpha,
        )
        criterion = nn.MSELoss()

        # Pre-build mini-batch infrastructure only if needed
        if not use_full_batch:
            from torch.utils.data import TensorDataset, DataLoader
            effective_batch = min(self.batch_size, len(X_train))
            dataset = TensorDataset(X_train_t, y_train_t)
            dataloader = DataLoader(
                dataset,
                batch_size=effective_batch,
                shuffle=True,
            )

        # Training loop
        best_val_loss = float('inf')
        best_state_dict = None
        patience_counter = 0

        for epoch in range(self.max_iter):
            self.model_.train()

            if use_full_batch:
                # Single forward/backward pass per epoch — eliminates
                # kernel launch overhead that dominates small-batch GPU training
                optimizer.zero_grad()
                outputs = self.model_(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
            else:
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model_(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Validation + best-model checkpointing
            if self.early_stopping:
                self.model_.eval()
                with torch.no_grad():
                    val_outputs = self.model_(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = copy.deepcopy(self.model_.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.n_iter_no_change:
                    self.n_iter_ = epoch + 1
                    break

        if self.n_iter_ == 0:
            self.n_iter_ = self.max_iter

        # Restore best weights (not last weights after patience exhaustion)
        if best_state_dict is not None:
            self.model_.load_state_dict(best_state_dict)

        # Store weights for compatibility
        self.coefs_ = []
        self.intercepts_ = []
        for module in self.model_.modules():
            if isinstance(module, nn.Linear):
                self.coefs_.append(module.weight.detach().cpu().numpy())
                self.intercepts_.append(module.bias.detach().cpu().numpy())

        return self

    def predict(self, X):
        """Predict using the trained model."""
        import torch
        import numpy as np

        self.model_.eval()
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X).to(self.device)

        with torch.no_grad():
            outputs = self.model_(X_t)

        return outputs.cpu().numpy().reshape(-1)

    def score(self, X, y):
        """Compute R² score."""
        import numpy as np

        y_pred = self.predict(X)
        y = np.asarray(y)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot < 1e-10:
            return 0.0

        return 1.0 - (ss_res / ss_tot)


def get_hyperparameter_ranges(model_type: str) -> Dict[str, Any]:
    """
    Get reasonable hyperparameter search ranges for each model type.

    Args:
        model_type: Model type identifier

    Returns:
        Dict mapping hyperparameter names to their search ranges.
        Ranges can be lists (categorical) or tuples (min, max) for continuous.
    """
    ranges = {
        'ridge': {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        },
        'lasso': {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        },
        'elasticnet': {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        'random_forest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [2, 3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [2, 3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
        },
        'xgboost': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [2, 3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
        },
        'neural_network': {
            'hidden_layer_sizes': [(32,), (64,), (128,), (64, 32), (128, 64), (64, 32, 16)],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
        },
        'pytorch_nn': {
            'hidden_layer_sizes': [(128, 64), (256, 128), (512, 256), (64, 32, 16), (128, 64, 32), (256, 128, 64, 32) ],
            # 'hidden_layer_sizes': [(32,), (64,), (128,), (64, 32), (128, 64), (64, 32, 16)],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'alpha': [0.01, 0.025, 0.05, 0.075, 0.10],
            # 'alpha': [0.0001, 0.001, 0.01, 0.1],
        },
    }

    return ranges.get(model_type, {})


def hyperparameter_search(
    demands: np.ndarray,
    ratios: np.ndarray,
    demographic_features: np.ndarray,
    district_ids: np.ndarray,
    demo_feature_names: List[str],
    model_type: str,
    param_grid: Dict[str, List[Any]] = None,
    poly_degree: int = 2,
    include_interactions: bool = False,
    search_method: str = 'grid',
    n_random_samples: int = 20,
    cv_strategy: str = 'lodo',
    use_gpu: bool = True,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Perform hyperparameter search for g(D,x) models.

    Args:
        demands, ratios, demographic_features, district_ids: data arrays
        demo_feature_names: feature names
        model_type: model type to search
        param_grid: custom parameter grid (if None, uses default ranges)
        poly_degree: polynomial degree for demand features
        include_interactions: include D * x interactions
        search_method: 'grid' for exhaustive, 'random' for random search
        n_random_samples: number of random samples (for random search)
        cv_strategy: 'lodo' for leave-one-district-out
        use_gpu: whether to use GPU acceleration (for applicable models)
        progress_callback: optional callback(current, total, info)

    Returns:
        Dict with:
            - best_params: dict of best hyperparameters
            - best_score: best LODO R²
            - all_results: list of all tried configurations and scores
            - best_model_result: full result dict from fit_g_dx_model
    """
    from itertools import product
    import random

    # Get default parameter ranges if not provided
    if param_grid is None:
        param_grid = get_hyperparameter_ranges(model_type)

    if not param_grid:
        raise ValueError(f"No parameter grid available for model_type '{model_type}'")

    # Generate parameter combinations
    if search_method == 'grid':
        # Exhaustive grid search
        param_names = sorted(param_grid.keys())
        param_values = [param_grid[k] for k in param_names]
        combinations = list(product(*param_values))
        param_combinations = [
            dict(zip(param_names, vals)) for vals in combinations
        ]
    elif search_method == 'random':
        # Random search — deduplicate to avoid wasting time
        seen = set()
        param_combinations = []
        param_names = sorted(param_grid.keys())

        # Cap attempts to avoid infinite loop on tiny search spaces
        max_attempts = n_random_samples * 5
        attempts = 0
        while len(param_combinations) < n_random_samples and attempts < max_attempts:
            attempts += 1
            combo = {
                name: random.choice(param_grid[name])
                for name in param_names
            }
            # Hashable key for dedup (handles tuples in values)
            key = tuple(sorted((k, str(v)) for k, v in combo.items()))
            if key not in seen:
                seen.add(key)
                param_combinations.append(combo)
    else:
        raise ValueError(f"Unknown search_method: {search_method}")

    # Search loop
    all_results = []
    best_score = -float('inf')
    best_params = None
    best_lodo_result = None

    total_combinations = len(param_combinations)

    for i, params in enumerate(param_combinations):
        # Notify progress
        if progress_callback:
            progress_callback(i + 1, total_combinations, params)

        try:
            # Build kwargs for lodo_cross_validate_hp — pass all params through
            cv_kwargs = {
                'demands': demands,
                'ratios': ratios,
                'demographic_features': demographic_features,
                'district_ids': district_ids,
                'demo_feature_names': demo_feature_names,
                'model_type': model_type,
                'poly_degree': poly_degree,
                'include_interactions': include_interactions,
                'use_gpu': use_gpu,
            }
            cv_kwargs.update(params)

            # Run cross-validation
            lodo_result = lodo_cross_validate_hp(**cv_kwargs)
            score = lodo_result['lodo_r2']

            result_entry = {
                'params': params.copy(),
                'lodo_r2': score,
                'per_district_r2': lodo_result['per_district_r2'],
            }
            all_results.append(result_entry)

            # Track best — only store the LODO result, not a redundant full fit
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_lodo_result = lodo_result

        except Exception as e:
            # Log failed configuration
            result_entry = {
                'params': params.copy(),
                'lodo_r2': -1.0,
                'error': str(e),
            }
            all_results.append(result_entry)

    # Fit final model on full data only once, with the best params
    best_model_result = None
    if best_params is not None:
        best_model_result = fit_g_dx_model_hp(
            demands=demands,
            ratios=ratios,
            demographic_features=demographic_features,
            demo_feature_names=demo_feature_names,
            model_type=model_type,
            poly_degree=poly_degree,
            include_interactions=include_interactions,
            use_gpu=use_gpu,
            **best_params,
        )

    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_results,
        'best_model_result': best_model_result,
        'best_lodo_result': best_lodo_result,
        'n_combinations_tried': len(param_combinations),
    }


def fit_g_dx_model_hp(
    demands: np.ndarray,
    ratios: np.ndarray,
    demographic_features: np.ndarray,
    demo_feature_names: List[str],
    model_type: str = "ols",
    poly_degree: int = 2,
    include_interactions: bool = False,
    use_gpu: bool = True,
    **hyperparams,
) -> Dict[str, Any]:
    """
    Extended version of fit_g_dx_model with hyperparameter support and GPU.

    Additional model types: 'xgboost', 'pytorch_nn'

    Args:
        Same as fit_g_dx_model, plus:
        use_gpu: enable GPU acceleration where available
        **hyperparams: model-specific hyperparameters

    Returns:
        Same as fit_g_dx_model
    """
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor

    use_interactions = include_interactions or model_type == "ols_interactions"

    X, feat_names, poly, scaler = build_feature_matrix(
        demands, demographic_features,
        poly_degree=poly_degree,
        include_interactions=use_interactions,
        demo_feature_names=demo_feature_names,
    )

    # Build model with hyperparameters
    if model_type == 'ols' or model_type == 'ols_interactions':
        model = LinearRegression()

    elif model_type == 'ridge':
        alpha = hyperparams.get('alpha', 1.0)
        model = Ridge(alpha=alpha)

    elif model_type == 'lasso':
        alpha = hyperparams.get('alpha', 1.0)
        model = Lasso(alpha=alpha, max_iter=10000)

    elif model_type == 'elasticnet':
        alpha = hyperparams.get('alpha', 1.0)
        l1_ratio = hyperparams.get('l1_ratio', 0.5)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

    elif model_type == 'random_forest':
        n_estimators = hyperparams.get('n_estimators', 100)
        max_depth = hyperparams.get('max_depth', 5)
        min_samples_split = hyperparams.get('min_samples_split', 2)
        min_samples_leaf = hyperparams.get('min_samples_leaf', 1)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )

    elif model_type == 'gradient_boosting':
        n_estimators = hyperparams.get('n_estimators', 100)
        max_depth = hyperparams.get('max_depth', 3)
        learning_rate = hyperparams.get('learning_rate', 0.1)
        subsample = hyperparams.get('subsample', 1.0)
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42,
        )

    elif model_type == 'xgboost':
        try:
            import xgboost as xgb

            n_estimators = hyperparams.get('n_estimators', 100)
            max_depth = hyperparams.get('max_depth', 3)
            learning_rate = hyperparams.get('learning_rate', 0.1)
            subsample = hyperparams.get('subsample', 1.0)
            colsample_bytree = hyperparams.get('colsample_bytree', 1.0)

            device = 'cuda' if use_gpu else 'cpu'
            import torch
            if not torch.cuda.is_available():
                device = 'cpu'

            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                device=device,
                random_state=42,
                tree_method='hist',
            )
        except ImportError:
            raise ValueError("XGBoost not installed. Install with: pip install xgboost")

    elif model_type == 'neural_network' or model_type == 'pytorch_nn':
        hidden_layer_sizes = hyperparams.get('hidden_layer_sizes', (64, 32))
        alpha = hyperparams.get('alpha', 0.0001)
        learning_rate = hyperparams.get('learning_rate', 0.001)

        if model_type == 'pytorch_nn':
            # Always use PyTorch implementation for pytorch_nn;
            # use_gpu controls the device, not whether PyTorch is used
            device = 'cuda' if use_gpu else 'cpu'
            model = PyTorchNeuralNetwork(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate=learning_rate,
                alpha=alpha,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                device=device,
                random_state=42,
            )
        else:
            # Use sklearn MLPRegressor for 'neural_network' type
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Fit model
    model.fit(X, ratios)

    r2_train = max(0.0, float(model.score(X, ratios)))
    predicted = model.predict(X)
    residuals = ratios - predicted

    # Build prediction closure
    _poly, _scaler, _model = poly, scaler, model
    _use_interactions = use_interactions
    _poly_degree = poly_degree

    def predict_fn(new_demands, new_demo_features):
        X_new, _, _, _ = build_feature_matrix(
            new_demands, new_demo_features,
            poly_degree=_poly_degree,
            include_interactions=_use_interactions,
            poly_transformer=_poly,
            scaler=_scaler,
        )
        return _model.predict(X_new)

    # Extract coefficients or feature importances
    coefficients = None
    feature_importances = None
    is_linear = model_type in ("ols", "ridge", "lasso", "elasticnet", "ols_interactions")

    if is_linear:
        coefficients = {name: float(coef) for name, coef in zip(feat_names, model.coef_)}
    elif hasattr(model, 'feature_importances_'):
        feature_importances = {
            name: float(imp) for name, imp in zip(feat_names, model.feature_importances_)
        }

    # Count parameters
    if model_type in ('neural_network', 'pytorch_nn'):
        if hasattr(model, 'coefs_'):
            n_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
        else:
            # Count parameters for PyTorch model: sum of (in * out + out) per layer
            hidden_layer_sizes = hyperparams.get('hidden_layer_sizes', (64, 32))
            layer_sizes = [X.shape[1]] + list(hidden_layer_sizes) + [1]
            n_params = sum(
                layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
                for i in range(len(layer_sizes) - 1)
            )
    else:
        n_params = X.shape[1] + 1

    return {
        'model': model,
        'predict_fn': predict_fn,
        'r2_train': r2_train,
        'predicted': predicted,
        'residuals': residuals,
        'feature_names': feat_names,
        'demo_feature_names': list(demo_feature_names),
        'coefficients': coefficients,
        'feature_importances': feature_importances,
        'poly_transformer': poly,
        'scaler': scaler,
        'model_type': model_type,
        'n_params': n_params,
        'n_samples': len(demands),
        'include_interactions': use_interactions,
        'poly_degree': poly_degree,
        'hyperparams': hyperparams,
    }


def lodo_cross_validate_hp(
    demands: np.ndarray,
    ratios: np.ndarray,
    demographic_features: np.ndarray,
    district_ids: np.ndarray,
    demo_feature_names: List[str],
    model_type: str = "ols",
    poly_degree: int = 2,
    include_interactions: bool = False,
    use_gpu: bool = True,
    **hyperparams,
) -> Dict[str, Any]:
    """
    Extended LODO cross-validation with hyperparameter and GPU support.

    Args:
        Same as lodo_cross_validate, plus:
        use_gpu: enable GPU acceleration
        **hyperparams: model-specific hyperparameters

    Returns:
        Same as lodo_cross_validate
    """
    unique_districts = np.unique(district_ids)
    n = len(demands)

    oof_predictions = np.full(n, np.nan)
    per_district_r2 = {}
    per_district_n = {}

    use_interactions = include_interactions or model_type == "ols_interactions"

    for dist_id in unique_districts:
        test_mask = district_ids == dist_id
        train_mask = ~test_mask

        if train_mask.sum() < 5 or test_mask.sum() < 1:
            continue

        # Fit on training data
        result = fit_g_dx_model_hp(
            demands=demands[train_mask],
            ratios=ratios[train_mask],
            demographic_features=demographic_features[train_mask],
            demo_feature_names=demo_feature_names,
            model_type=model_type,
            poly_degree=poly_degree,
            include_interactions=include_interactions,
            use_gpu=use_gpu,
            **hyperparams,
        )

        # Predict on held-out district
        preds = result['predict_fn'](
            demands[test_mask],
            demographic_features[test_mask],
        )
        oof_predictions[test_mask] = preds

        # Per-district R²
        y_test = ratios[test_mask]
        var_y = np.var(y_test)
        if var_y > 1e-10:
            r2 = max(0.0, 1.0 - np.var(y_test - preds) / var_y)
        else:
            r2 = 0.0

        per_district_r2[int(dist_id)] = float(r2)
        per_district_n[int(dist_id)] = int(test_mask.sum())

    # Overall LODO R²
    valid = ~np.isnan(oof_predictions)
    if valid.sum() > 0:
        var_y_all = np.var(ratios[valid])
        if var_y_all > 1e-10:
            lodo_r2 = max(0.0, 1.0 - np.var(ratios[valid] - oof_predictions[valid]) / var_y_all)
        else:
            lodo_r2 = 0.0
    else:
        lodo_r2 = 0.0

    oof_residuals = np.where(valid, ratios - oof_predictions, np.nan)

    return {
        'lodo_r2': float(lodo_r2),
        'per_district_r2': per_district_r2,
        'per_district_n': per_district_n,
        'oof_predictions': oof_predictions,
        'oof_residuals': oof_residuals,
        'n_folds': len(per_district_r2),
        'n_valid': int(valid.sum()),
    }
