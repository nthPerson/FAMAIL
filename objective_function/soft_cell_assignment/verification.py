"""
Gradient Verification for Soft Cell Assignment.

This module provides comprehensive gradient verification utilities
to validate that gradients flow correctly through the soft cell
assignment and fairness term computations.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def verify_soft_assignment_gradients(
    grid_dims: Tuple[int, int] = (48, 90),
    neighborhood_size: int = 5,
    temperature: float = 1.0,
    n_samples: int = 10,
    seed: int = 42,
    numerical_eps: float = 1e-4,
    rtol: float = 1e-2,
) -> Dict[str, Any]:
    """
    Verify gradients of soft cell assignment.
    
    Compares analytical gradients from autograd with numerical gradients
    computed via finite differences.
    
    Args:
        grid_dims: Grid dimensions for testing
        neighborhood_size: Size of neighborhood
        temperature: Temperature parameter
        n_samples: Number of test samples
        seed: Random seed
        numerical_eps: Step size for numerical gradient
        rtol: Relative tolerance for gradient comparison
        
    Returns:
        Dictionary with verification results
    """
    if not TORCH_AVAILABLE:
        return {'error': 'PyTorch not available', 'passed': False}
    
    from .module import SoftCellAssignment
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create module
    module = SoftCellAssignment(
        grid_dims=grid_dims,
        neighborhood_size=neighborhood_size,
        initial_temperature=temperature,
    )
    
    results = {
        'grid_dims': grid_dims,
        'neighborhood_size': neighborhood_size,
        'temperature': temperature,
        'n_samples': n_samples,
        'samples': [],
    }
    
    all_passed = True
    max_rel_errors = []
    
    for sample_idx in range(n_samples):
        # Generate random test case
        # Location near center to avoid boundary issues
        k = (neighborhood_size - 1) // 2
        location = torch.tensor([
            [np.random.uniform(k + 1, grid_dims[0] - k - 1),
             np.random.uniform(k + 1, grid_dims[1] - k - 1)]
        ], dtype=torch.float32, requires_grad=True)
        
        original_cell = torch.tensor([
            [int(location[0, 0].item()), int(location[0, 1].item())]
        ], dtype=torch.float32)
        
        # Forward pass
        probs = module(location, original_cell)
        
        # Compute scalar loss (sum for gradient computation)
        loss = probs.sum()
        
        # Analytical gradient
        loss.backward()
        analytical_grad = location.grad.clone()
        
        # Numerical gradient
        numerical_grad = torch.zeros_like(location)
        
        for i in range(2):
            # Perturb location
            loc_plus = location.detach().clone()
            loc_plus[0, i] += numerical_eps
            probs_plus = module(loc_plus, original_cell)
            loss_plus = probs_plus.sum()
            
            loc_minus = location.detach().clone()
            loc_minus[0, i] -= numerical_eps
            probs_minus = module(loc_minus, original_cell)
            loss_minus = probs_minus.sum()
            
            numerical_grad[0, i] = (loss_plus - loss_minus) / (2 * numerical_eps)
        
        # Compare gradients
        abs_diff = (analytical_grad - numerical_grad).abs()
        rel_error = abs_diff / (numerical_grad.abs() + 1e-8)
        max_rel_error = rel_error.max().item()
        max_rel_errors.append(max_rel_error)
        
        sample_passed = max_rel_error < rtol
        all_passed = all_passed and sample_passed
        
        results['samples'].append({
            'location': location.detach().numpy().tolist(),
            'original_cell': original_cell.numpy().tolist(),
            'analytical_grad': analytical_grad.numpy().tolist(),
            'numerical_grad': numerical_grad.numpy().tolist(),
            'max_rel_error': max_rel_error,
            'passed': sample_passed,
        })
        
        # Reset gradient
        location.grad.zero_()
    
    results['all_passed'] = all_passed
    results['max_rel_error_overall'] = max(max_rel_errors)
    results['mean_rel_error'] = np.mean(max_rel_errors)
    results['passed'] = all_passed
    
    return results


def verify_end_to_end_gradients(
    grid_dims: Tuple[int, int] = (48, 90),
    neighborhood_size: int = 5,
    n_cells: int = 100,
    n_trajectories: int = 10,
    temperature: float = 1.0,
    seed: int = 42,
    numerical_eps: float = 1e-4,
    rtol: float = 1e-2,
) -> Dict[str, Any]:
    """
    Verify end-to-end gradient flow from trajectory location to fairness metrics.
    
    This tests the complete gradient chain:
    location → soft_assignment → soft_counts → service_rates → gini/r² → loss
    
    Args:
        grid_dims: Grid dimensions
        neighborhood_size: Neighborhood size
        n_cells: Number of cells to use (for testing)
        n_trajectories: Number of trajectories to optimize
        temperature: Soft assignment temperature
        seed: Random seed
        numerical_eps: Numerical gradient step size
        rtol: Relative tolerance
        
    Returns:
        Dictionary with comprehensive verification results
    """
    if not TORCH_AVAILABLE:
        return {'error': 'PyTorch not available', 'passed': False}
    
    from .module import SoftCellAssignment, compute_soft_counts
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    results = {
        'grid_dims': grid_dims,
        'n_cells': n_cells,
        'n_trajectories': n_trajectories,
        'tests': {},
    }
    
    # Use smaller grid for faster testing
    test_grid = (10, 10)
    k = (neighborhood_size - 1) // 2
    
    # Create soft assignment module
    soft_assign = SoftCellAssignment(
        grid_dims=test_grid,
        neighborhood_size=neighborhood_size,
        initial_temperature=temperature,
    )
    
    # Generate test trajectories
    locations = torch.tensor([
        [np.random.uniform(k + 1, test_grid[0] - k - 1),
         np.random.uniform(k + 1, test_grid[1] - k - 1)]
        for _ in range(n_trajectories)
    ], dtype=torch.float32, requires_grad=True)
    
    original_cells = locations.detach().round()
    
    # =========================================================================
    # Test 1: Soft Assignment Gradients
    # =========================================================================
    probs = soft_assign(locations, original_cells)
    loss_probs = probs.sum()
    loss_probs.backward()
    
    grad_from_probs = locations.grad.clone()
    has_grad_probs = grad_from_probs is not None and not torch.isnan(grad_from_probs).any()
    
    results['tests']['soft_assignment'] = {
        'gradient_exists': grad_from_probs is not None,
        'gradient_has_nan': torch.isnan(grad_from_probs).any().item() if grad_from_probs is not None else True,
        'gradient_mean': grad_from_probs.mean().item() if has_grad_probs else None,
        'gradient_std': grad_from_probs.std().item() if has_grad_probs else None,
        'passed': has_grad_probs,
    }
    locations.grad.zero_()
    
    # =========================================================================
    # Test 2: Soft Counts Gradients
    # =========================================================================
    probs = soft_assign(locations, original_cells)
    soft_counts = compute_soft_counts(probs, original_cells, test_grid)
    loss_counts = soft_counts.sum()
    loss_counts.backward()
    
    grad_from_counts = locations.grad.clone()
    has_grad_counts = grad_from_counts is not None and not torch.isnan(grad_from_counts).any()
    
    results['tests']['soft_counts'] = {
        'gradient_exists': grad_from_counts is not None,
        'gradient_has_nan': torch.isnan(grad_from_counts).any().item() if grad_from_counts is not None else True,
        'gradient_mean': grad_from_counts.mean().item() if has_grad_counts else None,
        'gradient_std': grad_from_counts.std().item() if has_grad_counts else None,
        'passed': has_grad_counts,
    }
    locations.grad.zero_()
    
    # =========================================================================
    # Test 3: Gini Coefficient Gradients
    # =========================================================================
    from objective_function.spatial_fairness.utils import compute_gini_torch
    
    probs = soft_assign(locations, original_cells)
    soft_counts = compute_soft_counts(probs, original_cells, test_grid)
    
    # Create mock active taxis and compute service rates
    active_taxis = torch.ones(test_grid) * 10.0
    period_duration = 1.0
    service_rates = soft_counts / (active_taxis * period_duration + 1e-8)
    
    # Flatten for Gini computation
    service_rates_flat = service_rates.view(-1)
    gini = compute_gini_torch(service_rates_flat)
    gini.backward()
    
    grad_from_gini = locations.grad.clone()
    has_grad_gini = grad_from_gini is not None and not torch.isnan(grad_from_gini).any()
    
    results['tests']['gini_coefficient'] = {
        'gini_value': gini.item(),
        'gradient_exists': grad_from_gini is not None,
        'gradient_has_nan': torch.isnan(grad_from_gini).any().item() if grad_from_gini is not None else True,
        'gradient_mean': grad_from_gini.mean().item() if has_grad_gini else None,
        'gradient_std': grad_from_gini.std().item() if has_grad_gini else None,
        'passed': has_grad_gini,
    }
    locations.grad.zero_()
    
    # =========================================================================
    # Test 4: Spatial Fairness Gradients
    # =========================================================================
    from objective_function.spatial_fairness.utils import DifferentiableSpatialFairness
    
    spatial_module = DifferentiableSpatialFairness(grid_dims=test_grid)
    
    probs = soft_assign(locations, original_cells)
    soft_counts = compute_soft_counts(probs, original_cells, test_grid)
    
    service_rates = soft_counts / (active_taxis * period_duration + 1e-8)
    service_rates_flat = service_rates.view(-1)
    
    # Use same rates for DSR and ASR for simplicity
    f_spatial = spatial_module.compute(service_rates_flat, service_rates_flat)
    f_spatial.backward()
    
    grad_from_spatial = locations.grad.clone()
    has_grad_spatial = grad_from_spatial is not None and not torch.isnan(grad_from_spatial).any()
    
    results['tests']['spatial_fairness'] = {
        'f_spatial_value': f_spatial.item(),
        'gradient_exists': grad_from_spatial is not None,
        'gradient_has_nan': torch.isnan(grad_from_spatial).any().item() if grad_from_spatial is not None else True,
        'gradient_mean': grad_from_spatial.mean().item() if has_grad_spatial else None,
        'gradient_std': grad_from_spatial.std().item() if has_grad_spatial else None,
        'passed': has_grad_spatial,
    }
    
    # =========================================================================
    # Overall Results
    # =========================================================================
    all_passed = all(test['passed'] for test in results['tests'].values())
    results['all_passed'] = all_passed
    results['passed'] = all_passed
    
    return results


def verify_causal_fairness_end_to_end(
    grid_dims: Tuple[int, int] = (10, 10),
    neighborhood_size: int = 5,
    n_trajectories: int = 10,
    temperature: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Verify gradient flow through causal fairness computation.
    
    Tests: location → soft_assignment → soft_demand → service_ratio → R²
    
    Args:
        grid_dims: Grid dimensions
        neighborhood_size: Neighborhood size
        n_trajectories: Number of test trajectories
        temperature: Soft assignment temperature
        seed: Random seed
        
    Returns:
        Verification results
    """
    if not TORCH_AVAILABLE:
        return {'error': 'PyTorch not available', 'passed': False}
    
    from .module import SoftCellAssignment, compute_soft_counts
    from objective_function.causal_fairness.utils import compute_causal_fairness_torch
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    results = {
        'grid_dims': grid_dims,
        'n_trajectories': n_trajectories,
    }
    
    k = (neighborhood_size - 1) // 2
    
    # Create module
    soft_assign = SoftCellAssignment(
        grid_dims=grid_dims,
        neighborhood_size=neighborhood_size,
        initial_temperature=temperature,
    )
    
    # Generate test data
    locations = torch.tensor([
        [np.random.uniform(k + 1, grid_dims[0] - k - 1),
         np.random.uniform(k + 1, grid_dims[1] - k - 1)]
        for _ in range(n_trajectories)
    ], dtype=torch.float32, requires_grad=True)
    
    original_cells = locations.detach().round()
    
    # Compute soft demand
    probs = soft_assign(locations, original_cells)
    soft_demand = compute_soft_counts(probs, original_cells, grid_dims)
    
    # Add base demand to avoid zeros
    base_demand = torch.ones(grid_dims) * 5.0
    total_demand = soft_demand + base_demand
    
    # Supply (fixed)
    supply = torch.ones(grid_dims) * 10.0
    
    # Service ratios
    service_ratios = supply / (total_demand + 1e-8)
    service_ratios_flat = service_ratios.view(-1)
    
    # Expected ratios (frozen g(d))
    demand_flat = total_demand.view(-1)
    # Simple linear g(d) for testing
    expected_ratios = 10.0 / (demand_flat.detach() + 1e-8)
    
    # Compute causal fairness
    f_causal = compute_causal_fairness_torch(service_ratios_flat, expected_ratios)
    f_causal.backward()
    
    grad = locations.grad
    has_grad = grad is not None and not torch.isnan(grad).any()
    
    results['f_causal_value'] = f_causal.item()
    results['gradient_exists'] = grad is not None
    results['gradient_has_nan'] = torch.isnan(grad).any().item() if grad is not None else True
    results['gradient_mean'] = grad.mean().item() if has_grad else None
    results['gradient_std'] = grad.std().item() if has_grad else None
    results['passed'] = has_grad
    
    return results


def create_soft_assignment_verification_report(
    results: Dict[str, Any],
    include_samples: bool = False,
) -> str:
    """
    Create a formatted report from verification results.
    
    Args:
        results: Results from verify_* functions
        include_samples: Whether to include per-sample details
        
    Returns:
        Formatted markdown report string
    """
    lines = []
    lines.append("# Soft Cell Assignment Gradient Verification Report")
    lines.append("")
    
    if 'error' in results:
        lines.append(f"**Error**: {results['error']}")
        return "\n".join(lines)
    
    # Overall status
    passed = results.get('passed', results.get('all_passed', False))
    status = "✅ PASSED" if passed else "❌ FAILED"
    lines.append(f"**Overall Status**: {status}")
    lines.append("")
    
    # Configuration
    if 'grid_dims' in results:
        lines.append("## Configuration")
        lines.append(f"- Grid Dimensions: {results.get('grid_dims')}")
        lines.append(f"- Neighborhood Size: {results.get('neighborhood_size', 'N/A')}")
        lines.append(f"- Temperature: {results.get('temperature', 'N/A')}")
        lines.append(f"- Number of Samples/Trajectories: {results.get('n_samples', results.get('n_trajectories', 'N/A'))}")
        lines.append("")
    
    # Test results
    if 'tests' in results:
        lines.append("## Test Results")
        lines.append("")
        for test_name, test_result in results['tests'].items():
            test_passed = test_result.get('passed', False)
            test_status = "✅" if test_passed else "❌"
            lines.append(f"### {test_status} {test_name.replace('_', ' ').title()}")
            
            for key, value in test_result.items():
                if key != 'passed':
                    if isinstance(value, float):
                        lines.append(f"- {key}: {value:.6e}")
                    else:
                        lines.append(f"- {key}: {value}")
            lines.append("")
    
    # Error statistics
    if 'max_rel_error_overall' in results:
        lines.append("## Error Statistics")
        lines.append(f"- Max Relative Error (Overall): {results['max_rel_error_overall']:.6e}")
        lines.append(f"- Mean Relative Error: {results.get('mean_rel_error', 'N/A'):.6e}" if results.get('mean_rel_error') else "")
        lines.append("")
    
    # Sample details
    if include_samples and 'samples' in results:
        lines.append("## Per-Sample Results")
        lines.append("")
        for i, sample in enumerate(results['samples']):
            sample_status = "✅" if sample['passed'] else "❌"
            lines.append(f"### Sample {i+1} {sample_status}")
            lines.append(f"- Location: {sample['location']}")
            lines.append(f"- Max Rel Error: {sample['max_rel_error']:.6e}")
            lines.append("")
    
    return "\n".join(lines)


def run_all_verifications(
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all gradient verification tests.
    
    Args:
        verbose: Print progress messages
        
    Returns:
        Combined results dictionary
    """
    all_results = {}
    
    if verbose:
        print("Running soft assignment gradient verification...")
    all_results['soft_assignment'] = verify_soft_assignment_gradients()
    
    if verbose:
        print("Running end-to-end gradient verification...")
    all_results['end_to_end'] = verify_end_to_end_gradients()
    
    if verbose:
        print("Running causal fairness gradient verification...")
    all_results['causal_fairness'] = verify_causal_fairness_end_to_end()
    
    # Overall status
    all_passed = all(r.get('passed', False) for r in all_results.values())
    all_results['all_passed'] = all_passed
    
    if verbose:
        status = "✅ All tests passed!" if all_passed else "❌ Some tests failed"
        print(f"\n{status}")
    
    return all_results
