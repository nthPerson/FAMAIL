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
