"""
Gradient Flow Verification and Visualization Module.

This module provides tools to verify and visualize how gradients flow
through the FAMAIL objective function components.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
import numpy as np

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARDS_DIR = SCRIPT_DIR.parent
OBJECTIVE_FUNCTION_DIR = DASHBOARDS_DIR.parent
PROJECT_ROOT = OBJECTIVE_FUNCTION_DIR.parent
sys.path.insert(0, str(OBJECTIVE_FUNCTION_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class GradientStats:
    """Statistics for a gradient tensor."""
    mean: float
    std: float
    min_val: float
    max_val: float
    nonzero_count: int
    total_count: int
    has_nan: bool = False
    has_inf: bool = False
    
    @property
    def nonzero_ratio(self) -> float:
        return self.nonzero_count / max(self.total_count, 1)
    
    @classmethod
    def from_tensor(cls, grad: 'torch.Tensor', threshold: float = 1e-10) -> 'GradientStats':
        """Create stats from a gradient tensor."""
        if grad is None:
            return cls(
                mean=0, std=0, min_val=0, max_val=0,
                nonzero_count=0, total_count=0,
                has_nan=False, has_inf=False
            )
        return cls(
            mean=grad.mean().item(),
            std=grad.std().item() if grad.numel() > 1 else 0,
            min_val=grad.min().item(),
            max_val=grad.max().item(),
            nonzero_count=int((grad.abs() > threshold).sum().item()),
            total_count=grad.numel(),
            has_nan=bool(torch.isnan(grad).any()),
            has_inf=bool(torch.isinf(grad).any()),
        )


@dataclass
class TermGradientReport:
    """Report for a single objective function term."""
    term_name: str
    term_value: float
    gradient_available: bool
    gradient_stats: Optional[GradientStats] = None
    error_message: Optional[str] = None


@dataclass
class GradientFlowReport:
    """
    Complete report on gradient flow through the objective function.
    """
    overall_passed: bool
    total_objective: float
    term_reports: Dict[str, TermGradientReport] = field(default_factory=dict)
    combined_gradient_stats: Optional[GradientStats] = None
    temperature: float = 1.0
    timestamp: Optional[str] = None
    
    def add_term(self, report: TermGradientReport) -> None:
        """Add a term report."""
        self.term_reports[report.term_name] = report
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_passed': self.overall_passed,
            'total_objective': self.total_objective,
            'temperature': self.temperature,
            'timestamp': self.timestamp,
            'terms': {
                name: {
                    'value': r.term_value,
                    'gradient_available': r.gradient_available,
                    'error': r.error_message,
                    'stats': {
                        'mean': r.gradient_stats.mean if r.gradient_stats else None,
                        'std': r.gradient_stats.std if r.gradient_stats else None,
                        'nonzero_ratio': r.gradient_stats.nonzero_ratio if r.gradient_stats else None,
                    } if r.gradient_stats else None,
                }
                for name, r in self.term_reports.items()
            }
        }


# =============================================================================
# GRADIENT VERIFICATION FUNCTIONS
# =============================================================================

def verify_term_gradients(
    term_name: str,
    grid_dims: Tuple[int, int] = (10, 10),
    n_points: int = 50,
    temperature: float = 1.0,
    verbose: bool = False,
) -> TermGradientReport:
    """
    Verify gradients for a single objective function term.
    
    Args:
        term_name: One of 'spatial', 'causal', 'fidelity'
        grid_dims: Grid dimensions for testing
        n_points: Number of test points
        temperature: Soft assignment temperature
        verbose: Print detailed output
        
    Returns:
        TermGradientReport with results
    """
    if not TORCH_AVAILABLE:
        return TermGradientReport(
            term_name=term_name,
            term_value=0.0,
            gradient_available=False,
            error_message="PyTorch not available",
        )
    
    import torch
    torch.manual_seed(42)
    
    try:
        if term_name == 'spatial':
            return _verify_spatial_gradient(grid_dims, n_points, temperature, verbose)
        elif term_name == 'causal':
            return _verify_causal_gradient(grid_dims, n_points, temperature, verbose)
        elif term_name == 'fidelity':
            return _verify_fidelity_gradient(n_points, verbose)
        else:
            return TermGradientReport(
                term_name=term_name,
                term_value=0.0,
                gradient_available=False,
                error_message=f"Unknown term: {term_name}",
            )
    except Exception as e:
        return TermGradientReport(
            term_name=term_name,
            term_value=0.0,
            gradient_available=False,
            error_message=str(e),
        )


def _verify_spatial_gradient(
    grid_dims: Tuple[int, int],
    n_points: int,
    temperature: float,
    verbose: bool,
) -> TermGradientReport:
    """Verify spatial fairness term gradients."""
    import torch
    
    from soft_cell_assignment import SoftCellAssignment
    
    soft_assign = SoftCellAssignment(
        grid_dims=grid_dims,
        initial_temperature=temperature,
    )
    
    # Create test coordinates
    coords = torch.rand(n_points, 2) * torch.tensor(grid_dims, dtype=torch.float32)
    coords.requires_grad_(True)
    
    # Compute soft counts manually
    k = soft_assign.k
    soft_counts = torch.zeros(grid_dims, dtype=torch.float32)
    
    for i in range(n_points):
        loc = coords[i:i+1]
        cell = loc.floor().long().clamp(
            min=torch.tensor([0, 0]),
            max=torch.tensor([grid_dims[0]-1, grid_dims[1]-1])
        )
        
        probs = soft_assign(loc.float(), cell.float())
        cx, cy = cell[0, 0].item(), cell[0, 1].item()
        
        for di in range(-k, k+1):
            for dj in range(-k, k+1):
                ni, nj = int(cx + di), int(cy + dj)
                if 0 <= ni < grid_dims[0] and 0 <= nj < grid_dims[1]:
                    soft_counts[ni, nj] = soft_counts[ni, nj] + probs[0, di + k, dj + k]
    
    # Compute pairwise Gini
    counts_flat = soft_counts.flatten()
    n = counts_flat.numel()
    mean_val = counts_flat.mean() + 1e-8
    
    diff_matrix = torch.abs(counts_flat.unsqueeze(0) - counts_flat.unsqueeze(1))
    gini = diff_matrix.sum() / (2 * n * n * mean_val)
    
    f_spatial = 1.0 - gini
    
    # Backward pass
    f_spatial.backward()
    
    # Check gradients
    if coords.grad is None:
        return TermGradientReport(
            term_name='spatial',
            term_value=f_spatial.item(),
            gradient_available=False,
            error_message="No gradient computed",
        )
    
    stats = GradientStats.from_tensor(coords.grad)
    
    if verbose:
        print(f"Spatial fairness: {f_spatial.item():.4f}")
        print(f"Gradient stats: mean={stats.mean:.6f}, std={stats.std:.6f}")
        print(f"Nonzero gradients: {stats.nonzero_count}/{stats.total_count}")
    
    return TermGradientReport(
        term_name='spatial',
        term_value=f_spatial.item(),
        gradient_available=True,
        gradient_stats=stats,
    )


def _verify_causal_gradient(
    grid_dims: Tuple[int, int],
    n_points: int,
    temperature: float,
    verbose: bool,
) -> TermGradientReport:
    """Verify causal fairness term gradients."""
    import torch
    
    from soft_cell_assignment import SoftCellAssignment
    
    soft_assign = SoftCellAssignment(
        grid_dims=grid_dims,
        initial_temperature=temperature,
    )
    
    # Create test coordinates (demand)
    coords = torch.rand(n_points, 2) * torch.tensor(grid_dims, dtype=torch.float32)
    coords.requires_grad_(True)
    
    # Create fixed supply
    supply = torch.ones(grid_dims, dtype=torch.float32) * 10.0
    
    # Compute soft demand counts
    k = soft_assign.k
    demand = torch.zeros(grid_dims, dtype=torch.float32)
    
    for i in range(n_points):
        loc = coords[i:i+1]
        cell = loc.floor().long().clamp(
            min=torch.tensor([0, 0]),
            max=torch.tensor([grid_dims[0]-1, grid_dims[1]-1])
        )
        
        probs = soft_assign(loc.float(), cell.float())
        cx, cy = cell[0, 0].item(), cell[0, 1].item()
        
        for di in range(-k, k+1):
            for dj in range(-k, k+1):
                ni, nj = int(cx + di), int(cy + dj)
                if 0 <= ni < grid_dims[0] and 0 <= nj < grid_dims[1]:
                    demand[ni, nj] = demand[ni, nj] + probs[0, di + k, dj + k]
    
    # Filter active cells
    mask = demand.flatten() > 0.1
    D = demand.flatten()[mask]
    S = supply.flatten()[mask]
    
    if len(D) < 2:
        return TermGradientReport(
            term_name='causal',
            term_value=0.5,
            gradient_available=False,
            error_message="Insufficient active cells",
        )
    
    # Compute Y = S / D
    Y = S / (D + 1e-8)
    
    # Simple g(d) = constant for test
    g_d = Y.mean().detach()
    
    # Residual
    R = Y - g_d
    
    # R²
    var_Y = Y.var() + 1e-8
    var_R = R.var()
    r_squared = 1.0 - var_R / var_Y
    
    f_causal = torch.clamp(r_squared, 0.0, 1.0)
    
    # Backward pass
    f_causal.backward()
    
    if coords.grad is None:
        return TermGradientReport(
            term_name='causal',
            term_value=f_causal.item(),
            gradient_available=False,
            error_message="No gradient computed",
        )
    
    stats = GradientStats.from_tensor(coords.grad)
    
    if verbose:
        print(f"Causal fairness (R²): {f_causal.item():.4f}")
        print(f"Gradient stats: mean={stats.mean:.6f}, std={stats.std:.6f}")
        print(f"Nonzero gradients: {stats.nonzero_count}/{stats.total_count}")
    
    return TermGradientReport(
        term_name='causal',
        term_value=f_causal.item(),
        gradient_available=True,
        gradient_stats=stats,
    )


def _verify_fidelity_gradient(
    n_points: int,
    verbose: bool,
) -> TermGradientReport:
    """Verify fidelity term gradients (mock discriminator)."""
    import torch
    import torch.nn as nn
    
    # Create mock discriminator
    class MockDiscriminator(nn.Module):
        def __init__(self, hidden_dim: int = 32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.output = nn.Linear(hidden_dim, 1)
        
        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            # Simplified: compute similarity as sigmoid of dot product
            e1 = self.encoder(x1.mean(dim=1))
            e2 = self.encoder(x2.mean(dim=1))
            dot = (e1 * e2).sum(dim=-1)
            return torch.sigmoid(dot)
    
    disc = MockDiscriminator()
    
    # Create test trajectory features
    seq_len = 20
    batch_size = 4
    features = torch.randn(batch_size, seq_len, 4, requires_grad=True)
    reference = torch.randn(batch_size, seq_len, 4)
    
    # Forward pass
    similarity = disc(features, reference)
    f_fidelity = similarity.mean()
    
    # Backward
    f_fidelity.backward()
    
    if features.grad is None:
        return TermGradientReport(
            term_name='fidelity',
            term_value=f_fidelity.item(),
            gradient_available=False,
            error_message="No gradient computed",
        )
    
    stats = GradientStats.from_tensor(features.grad)
    
    if verbose:
        print(f"Fidelity: {f_fidelity.item():.4f}")
        print(f"Gradient stats: mean={stats.mean:.6f}, std={stats.std:.6f}")
    
    return TermGradientReport(
        term_name='fidelity',
        term_value=f_fidelity.item(),
        gradient_available=True,
        gradient_stats=stats,
    )


def verify_combined_gradients(
    grid_dims: Tuple[int, int] = (10, 10),
    n_points: int = 50,
    temperature: float = 1.0,
    alpha_spatial: float = 0.33,
    alpha_causal: float = 0.33,
    alpha_fidelity: float = 0.34,
    verbose: bool = False,
) -> GradientFlowReport:
    """
    Verify gradients for the complete combined objective function.
    
    Args:
        grid_dims: Grid dimensions for testing
        n_points: Number of test points
        temperature: Soft assignment temperature
        alpha_*: Term weights
        verbose: Print detailed output
        
    Returns:
        GradientFlowReport with complete results
    """
    from datetime import datetime
    
    report = GradientFlowReport(
        overall_passed=True,
        total_objective=0.0,
        temperature=temperature,
        timestamp=datetime.now().isoformat(),
    )
    
    # Verify individual terms
    for term in ['spatial', 'causal', 'fidelity']:
        term_report = verify_term_gradients(
            term_name=term,
            grid_dims=grid_dims,
            n_points=n_points,
            temperature=temperature,
            verbose=verbose,
        )
        report.add_term(term_report)
        
        if not term_report.gradient_available:
            if term == 'fidelity':
                # Fidelity is optional
                pass
            else:
                report.overall_passed = False
        elif term_report.gradient_stats:
            if term_report.gradient_stats.has_nan or term_report.gradient_stats.has_inf:
                report.overall_passed = False
    
    # Compute combined objective value
    report.total_objective = sum(
        alpha * report.term_reports[term].term_value
        for term, alpha in [
            ('spatial', alpha_spatial),
            ('causal', alpha_causal),
            ('fidelity', alpha_fidelity),
        ]
        if term in report.term_reports
    )
    
    return report


# =============================================================================
# GRADIENT FLOW VISUALIZATION
# =============================================================================

def create_gradient_flow_diagram(
    report: Optional[GradientFlowReport] = None,
    format: str = 'mermaid',
) -> str:
    """
    Create a diagram showing gradient flow through the objective function.
    
    Args:
        report: Optional gradient report to annotate with values
        format: Output format ('mermaid', 'ascii', 'dict')
        
    Returns:
        Diagram as string (or dict if format='dict')
    """
    if format == 'mermaid':
        return _create_mermaid_diagram(report)
    elif format == 'ascii':
        return _create_ascii_diagram(report)
    elif format == 'dict':
        return _create_dict_diagram(report)
    else:
        raise ValueError(f"Unknown format: {format}")


def _create_mermaid_diagram(report: Optional[GradientFlowReport]) -> str:
    """Create Mermaid flowchart diagram."""
    diagram = '''
graph TD
    subgraph Inputs
        T[("τ: Trajectory Coordinates<br/>(x, y) pairs")]
    end
    
    subgraph SoftCellAssignment["Soft Cell Assignment Module"]
        SA["σ(p|τ) = softmax(-||loc - cell||²/τ)"]
        SC["Soft Counts: C̃_i = Σ_τ σ(i|τ)"]
    end
    
    subgraph ObjectiveTerms["Objective Function Terms"]
        SF["F_spatial: Pairwise Gini<br/>1 - G(C̃_pickup, C̃_dropoff)"]
        CF["F_causal: R² Score<br/>1 - Var(R)/Var(Y)"]
        FF["F_fidelity: Discriminator<br/>D(modified, original)"]
    end
    
    subgraph Combined["Combined Objective"]
        L["L = α₁·F_causal + α₂·F_spatial + α₃·F_fidelity"]
    end
    
    subgraph Gradients["Gradient Flow"]
        G["∇_τ L → ∇_τ F_* → ∇_τ C̃ → ∇_τ σ"]
    end
    
    T --> SA
    SA --> SC
    SC --> SF
    SC --> CF
    T --> FF
    SF --> L
    CF --> L
    FF --> L
    L --> G
    G -.->|backprop| T
'''
    
    # Add annotations from report if available
    if report:
        if 'spatial' in report.term_reports:
            val = report.term_reports['spatial'].term_value
            passed = "✓" if report.term_reports['spatial'].gradient_available else "✗"
            # Note: Mermaid doesn't support dynamic annotations easily
    
    return diagram


def _create_ascii_diagram(report: Optional[GradientFlowReport]) -> str:
    """Create ASCII art diagram."""
    spatial_val = "?"
    causal_val = "?"
    fidelity_val = "?"
    total_val = "?"
    
    if report:
        if 'spatial' in report.term_reports:
            spatial_val = f"{report.term_reports['spatial'].term_value:.3f}"
        if 'causal' in report.term_reports:
            causal_val = f"{report.term_reports['causal'].term_value:.3f}"
        if 'fidelity' in report.term_reports:
            fidelity_val = f"{report.term_reports['fidelity'].term_value:.3f}"
        total_val = f"{report.total_objective:.3f}"
    
    diagram = f'''
╔══════════════════════════════════════════════════════════════════════════╗
║                    FAMAIL OBJECTIVE FUNCTION GRADIENT FLOW               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   ┌─────────────────┐                                                    ║
║   │  Trajectories τ │                                                    ║
║   │  (x, y) coords  │                                                    ║
║   └────────┬────────┘                                                    ║
║            │                                                             ║
║            ▼                                                             ║
║   ┌─────────────────────────────────────┐                                ║
║   │     Soft Cell Assignment Module     │                                ║
║   │ σ(p|τ) = softmax(-||loc-cell||²/τ)  │                                ║
║   │         (temperature τ)             │                                ║
║   └──────────────────┬──────────────────┘                                ║
║                      │                                                   ║
║            ┌─────────┴─────────┐                                         ║
║            ▼                   ▼                                         ║
║   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐           ║
║   │  Soft Counts    │ │  Soft Counts    │ │   Trajectory    │           ║
║   │   (Pickups)     │ │  (Dropoffs)     │ │    Features     │           ║
║   └────────┬────────┘ └────────┬────────┘ └────────┬────────┘           ║
║            │                   │                   │                     ║
║            ▼                   ▼                   ▼                     ║
║   ┌─────────────────────────────────────────────────────────────────┐   ║
║   │                     OBJECTIVE TERMS                              │   ║
║   │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │   ║
║   │  │  F_spatial     │  │  F_causal      │  │  F_fidelity    │     │   ║
║   │  │  = {spatial_val:^7}    │  │  = {causal_val:^7}     │  │  = {fidelity_val:^7}    │     │   ║
║   │  │  Pairwise Gini │  │  R² Score      │  │  Discriminator │     │   ║
║   │  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘     │   ║
║   └──────────┼───────────────────┼───────────────────┼──────────────┘   ║
║              │                   │                   │                   ║
║              └───────────────────┼───────────────────┘                   ║
║                                  ▼                                       ║
║            ┌─────────────────────────────────────────┐                   ║
║            │   L = α₁·F_causal + α₂·F_spatial        │                   ║
║            │         + α₃·F_fidelity                 │                   ║
║            │                                         │                   ║
║            │        Total L = {total_val:^7}                │                   ║
║            └────────────────────┬────────────────────┘                   ║
║                                 │                                        ║
║                                 ▼                                        ║
║            ┌─────────────────────────────────────────┐                   ║
║            │            ∇_τ L (Gradients)            │                   ║
║            │   Flow back through soft assignment     │                   ║
║            └─────────────────────────────────────────┘                   ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
'''
    return diagram


def _create_dict_diagram(report: Optional[GradientFlowReport]) -> Dict[str, Any]:
    """Create structured dictionary representation."""
    structure = {
        'input': {
            'name': 'Trajectory Coordinates',
            'shape': '(n_trajectories, seq_len, 2)',
            'requires_grad': True,
        },
        'soft_cell_assignment': {
            'formula': 'σ(p|τ) = softmax(-||loc - cell||² / temperature)',
            'outputs': ['soft_counts_pickup', 'soft_counts_dropoff'],
            'parameters': ['temperature', 'neighborhood_size'],
        },
        'terms': {
            'spatial': {
                'formula': 'F_spatial = 1 - G(C̃_pickup, C̃_dropoff)',
                'inputs': ['soft_counts_pickup', 'soft_counts_dropoff'],
                'value': report.term_reports['spatial'].term_value if report and 'spatial' in report.term_reports else None,
            },
            'causal': {
                'formula': 'F_causal = R² = 1 - Var(Y - g(D)) / Var(Y)',
                'inputs': ['soft_counts_demand', 'supply', 'g_function'],
                'value': report.term_reports['causal'].term_value if report and 'causal' in report.term_reports else None,
            },
            'fidelity': {
                'formula': 'F_fidelity = D(τ_modified, τ_original)',
                'inputs': ['trajectory_features', 'reference_features'],
                'value': report.term_reports['fidelity'].term_value if report and 'fidelity' in report.term_reports else None,
            },
        },
        'combined': {
            'formula': 'L = α₁·F_causal + α₂·F_spatial + α₃·F_fidelity',
            'value': report.total_objective if report else None,
        },
        'gradients': {
            'chain': ['∇_τ L', '∇_τ F_*', '∇_τ C̃', '∇_τ σ'],
            'backprop_target': 'trajectory_coordinates',
        },
    }
    
    return structure


# =============================================================================
# TEMPERATURE ANNEALING ANALYSIS
# =============================================================================

def analyze_temperature_schedule(
    temperatures: List[float],
    grid_dims: Tuple[int, int] = (10, 10),
    n_points: int = 50,
) -> Dict[str, Any]:
    """
    Analyze how gradient flow changes with temperature annealing.
    
    Args:
        temperatures: List of temperatures to test
        grid_dims: Grid dimensions
        n_points: Number of test points
        
    Returns:
        Analysis results
    """
    results = {
        'temperatures': temperatures,
        'spatial_values': [],
        'spatial_grad_means': [],
        'spatial_grad_stds': [],
        'causal_values': [],
        'causal_grad_means': [],
    }
    
    for temp in temperatures:
        # Test spatial
        spatial_report = verify_term_gradients(
            term_name='spatial',
            grid_dims=grid_dims,
            n_points=n_points,
            temperature=temp,
        )
        results['spatial_values'].append(spatial_report.term_value)
        if spatial_report.gradient_stats:
            results['spatial_grad_means'].append(spatial_report.gradient_stats.mean)
            results['spatial_grad_stds'].append(spatial_report.gradient_stats.std)
        else:
            results['spatial_grad_means'].append(0)
            results['spatial_grad_stds'].append(0)
        
        # Test causal
        causal_report = verify_term_gradients(
            term_name='causal',
            grid_dims=grid_dims,
            n_points=n_points,
            temperature=temp,
        )
        results['causal_values'].append(causal_report.term_value)
        if causal_report.gradient_stats:
            results['causal_grad_means'].append(causal_report.gradient_stats.mean)
        else:
            results['causal_grad_means'].append(0)
    
    return results
