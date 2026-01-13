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
        
        if not include_zero_supply and s == 0:
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
