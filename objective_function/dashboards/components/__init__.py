"""
FAMAIL Integrated Dashboard Components.

This package provides components for the integrated objective function dashboard:
- Combined objective function module
- Gradient flow visualization utilities
- Attribution integration (LIS + DCD)
"""

# Import conditionally to handle missing dependencies
try:
    from .combined_objective import (
        compute_combined_objective,
        create_default_g_function,
        ObjectiveResult,
    )
    
    # Torch-dependent import
    try:
        from .combined_objective import DifferentiableFAMAILObjective
    except ImportError:
        DifferentiableFAMAILObjective = None
        
except ImportError as e:
    print(f"Warning: Could not import combined_objective: {e}")
    DifferentiableFAMAILObjective = None
    compute_combined_objective = None
    create_default_g_function = None
    ObjectiveResult = None

try:
    from .gradient_flow import (
        create_gradient_flow_diagram,
        verify_term_gradients,
        verify_combined_gradients,
        analyze_temperature_schedule,
        GradientFlowReport,
        GradientStats,
        TermGradientReport,
    )
except ImportError as e:
    print(f"Warning: Could not import gradient_flow: {e}")
    create_gradient_flow_diagram = None
    verify_term_gradients = None
    verify_combined_gradients = None
    GradientFlowReport = None

try:
    from .attribution_integration import (
        compute_combined_attribution,
        select_trajectories_for_modification,
        compute_all_lis_scores,
        compute_all_dcd_scores,
        compute_local_inequality_score,
        compute_demand_conditional_deviation,
        load_trajectories_from_all_trajs,
        extract_cells_from_trajectories,
        create_mock_supply_data,
        AttributionResult,
        AttributionScores,
    )
except ImportError as e:
    print(f"Warning: Could not import attribution_integration: {e}")
    compute_combined_attribution = None
    select_trajectories_for_modification = None
    AttributionResult = None


__all__ = [
    # Combined Objective
    'DifferentiableFAMAILObjective',
    'compute_combined_objective',
    'create_default_g_function',
    'ObjectiveResult',
    
    # Gradient Flow
    'create_gradient_flow_diagram',
    'verify_term_gradients',
    'verify_combined_gradients',
    'analyze_temperature_schedule',
    'GradientFlowReport',
    'GradientStats',
    'TermGradientReport',
    
    # Attribution
    'compute_combined_attribution',
    'select_trajectories_for_modification',
    'compute_all_lis_scores',
    'compute_all_dcd_scores',
    'compute_local_inequality_score',
    'compute_demand_conditional_deviation',
    'load_trajectories_from_all_trajs',
    'extract_cells_from_trajectories',
    'create_mock_supply_data',
    'AttributionResult',
    'AttributionScores',
]
