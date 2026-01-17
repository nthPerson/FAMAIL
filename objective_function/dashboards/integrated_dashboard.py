"""
FAMAIL Integrated Objective Function Dashboard

A comprehensive Streamlit dashboard for testing, visualizing, and understanding:
1. Gradient flow through the combined objective function
2. How soft cell assignment enables differentiability
3. How LIS and DCD attribution methods work together for trajectory selection

Run with: streamlit run integrated_dashboard.py
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import pickle

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
OBJECTIVE_FUNCTION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = OBJECTIVE_FUNCTION_DIR.parent
sys.path.insert(0, str(OBJECTIVE_FUNCTION_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Check for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import dashboard components
from components.gradient_flow import (
    verify_term_gradients,
    verify_combined_gradients,
    create_gradient_flow_diagram,
    analyze_temperature_schedule,
    GradientFlowReport,
)
from components.combined_objective import (
    compute_combined_objective,
    create_default_g_function,
    ObjectiveResult,
)
from components.attribution_integration import (
    compute_combined_attribution,
    select_trajectories_for_modification,
    compute_all_lis_scores,
    compute_all_dcd_scores,
    load_trajectories_from_all_trajs,
    extract_cells_from_trajectories,
    create_mock_supply_data,
    AttributionResult,
)

if TORCH_AVAILABLE:
    from components.combined_objective import DifferentiableFAMAILObjective

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="FAMAIL Objective Function Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .formula-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        font-family: monospace;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render sidebar with comprehensive configuration for realistic testing."""
    st.sidebar.title("🎯 FAMAIL Dashboard")
    st.sidebar.markdown("---")
    
    # =========================================================================
    # GRID CONFIGURATION
    # =========================================================================
    st.sidebar.header("📐 Grid Configuration")
    grid_x = st.sidebar.number_input("Grid X", min_value=5, max_value=100, value=48)
    grid_y = st.sidebar.number_input("Grid Y", min_value=5, max_value=100, value=90)
    
    # =========================================================================
    # OBJECTIVE WEIGHTS
    # =========================================================================
    st.sidebar.header("⚖️ Objective Weights")
    alpha_spatial = st.sidebar.slider("α_spatial", 0.0, 1.0, 0.33, 0.01)
    alpha_causal = st.sidebar.slider("α_causal", 0.0, 1.0, 0.33, 0.01)
    alpha_fidelity = st.sidebar.slider("α_fidelity", 0.0, 1.0, 0.34, 0.01)
    
    # Normalize weights
    total_alpha = alpha_spatial + alpha_causal + alpha_fidelity
    if total_alpha > 0:
        alpha_spatial /= total_alpha
        alpha_causal /= total_alpha
        alpha_fidelity /= total_alpha
    
    # =========================================================================
    # SOFT CELL ASSIGNMENT
    # =========================================================================
    st.sidebar.header("🌡️ Soft Cell Assignment")
    temperature = st.sidebar.slider(
        "Temperature τ", 0.01, 5.0, 1.0, 0.01,
        help="Controls softness of cell assignment. Higher τ = softer assignment."
    )
    neighborhood_size = st.sidebar.select_slider(
        "Neighborhood Size", 
        options=[3, 5, 7, 9, 11],
        value=5,
        help="Window size for soft assignment (must be odd). 5 means 5×5 neighborhood."
    )
    
    use_annealing = st.sidebar.checkbox(
        "Use Temperature Annealing",
        value=False,
        help="Gradually decrease temperature during optimization."
    )
    if use_annealing:
        annealing_schedule = st.sidebar.selectbox(
            "Annealing Schedule",
            ["exponential", "linear", "cosine"],
            help="How temperature decreases over time."
        )
        final_temperature = st.sidebar.slider(
            "Final Temperature", 0.01, 1.0, 0.1, 0.01,
            help="Temperature at end of annealing."
        )
    else:
        annealing_schedule = None
        final_temperature = temperature
    
    st.sidebar.markdown("---")
    
    # =========================================================================
    # CAUSAL FAIRNESS g(d) CONFIGURATION
    # =========================================================================
    st.sidebar.header("📈 g(d) Estimation")
    
    g_estimation_method = st.sidebar.selectbox(
        "g(d) Approximation Method",
        ["isotonic", "binning", "linear", "polynomial", "lowess"],
        index=0,
        help="""
        Method to estimate E[Y|D=d]:
        - **isotonic**: Monotonic regression (recommended for sparse data)
        - **binning**: Group by demand bins, compute mean ratio per bin
        - **linear**: Simple linear regression Y = β₀ + β₁D
        - **polynomial**: Polynomial regression (degree configurable)
        - **lowess**: Locally weighted scatterplot smoothing
        """
    )
    
    # Method-specific parameters
    if g_estimation_method == "binning":
        n_bins = st.sidebar.slider(
            "Number of Bins", 5, 50, 10,
            help="Number of bins for demand discretization."
        )
    else:
        n_bins = 10  # Default
    
    if g_estimation_method == "polynomial":
        poly_degree = st.sidebar.slider(
            "Polynomial Degree", 1, 5, 2,
            help="Degree of polynomial for g(d) approximation."
        )
    else:
        poly_degree = 2
    
    if g_estimation_method == "lowess":
        lowess_frac = st.sidebar.slider(
            "LOWESS Fraction", 0.1, 0.9, 0.3, 0.05,
            help="Fraction of data for each local regression."
        )
    else:
        lowess_frac = 0.3
    
    freeze_g_function = st.sidebar.checkbox(
        "🧊 Freeze g(d) Lookup",
        value=True,
        help="Use frozen g(d) lookup table (computed from historical data) rather than recomputing each step."
    )
    
    st.sidebar.markdown("---")
    
    # =========================================================================
    # TEMPORAL CONFIGURATION
    # =========================================================================
    st.sidebar.header("⏰ Temporal Settings")
    
    period_type = st.sidebar.selectbox(
        "Period Type",
        ["hourly", "time_bucket", "daily", "all"],
        index=0,
        help="""
        Temporal granularity for aggregation:
        - **hourly**: Aggregate to 24 hour-level periods (recommended)
        - **time_bucket**: Use raw 5-minute buckets (288 per day)
        - **daily**: Aggregate all times to single period per day
        - **all**: Aggregate everything (no temporal dimension)
        """
    )
    
    # Day filtering
    days_options = {
        "Weekdays Only (Mon-Fri)": [0, 1, 2, 3, 4],
        "Weekends Only (Sat-Sun)": [5, 6],
        "All Days": [0, 1, 2, 3, 4, 5, 6],
        "Custom": None,
    }
    
    days_selection = st.sidebar.selectbox(
        "Days of Week",
        list(days_options.keys()),
        index=0,
        help="Filter data to specific days of week."
    )
    
    if days_selection == "Custom":
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        selected_days = st.sidebar.multiselect(
            "Select Days",
            options=list(range(7)),
            default=[0, 1, 2, 3, 4],
            format_func=lambda x: day_names[x]
        )
    else:
        selected_days = days_options[days_selection]
    
    st.sidebar.markdown("---")
    
    # =========================================================================
    # DATA FILTERING OPTIONS
    # =========================================================================
    st.sidebar.header("🔍 Data Filtering")
    
    include_zero_supply = st.sidebar.checkbox(
        "Include Zero Supply Cells",
        value=False,
        help="Include cells with zero taxi supply in causal fairness calculation."
    )
    
    include_zero_demand = st.sidebar.checkbox(
        "Include Zero Demand Cells",
        value=False,
        help="Include cells with zero demand in calculations."
    )
    
    min_demand_threshold = st.sidebar.number_input(
        "Min Demand Threshold",
        min_value=0, max_value=100, value=0,
        help="Exclude cells with demand below this threshold."
    )
    
    st.sidebar.markdown("---")
    
    # =========================================================================
    # ATTRIBUTION CONFIGURATION
    # =========================================================================
    st.sidebar.header("📊 Attribution Weights")
    
    lis_weight = st.sidebar.slider(
        "LIS Weight (Spatial)", 0.0, 1.0, 0.5, 0.01,
        help="Weight for Location Impact Score in trajectory selection."
    )
    dcd_weight = st.sidebar.slider(
        "DCD Weight (Causal)", 0.0, 1.0, 0.5, 0.01,
        help="Weight for Demand-Conditioned Deviation in trajectory selection."
    )
    
    # Trajectory selection method
    selection_method = st.sidebar.selectbox(
        "Trajectory Selection Method",
        ["Combined Score", "LIS Only", "DCD Only", "Demand Hierarchy Filter"],
        index=0,
        help="""
        How to select trajectories for modification:
        - **Combined Score**: Weighted combination of LIS and DCD
        - **LIS Only**: Pure spatial impact
        - **DCD Only**: Pure causal deviation
        - **Demand Hierarchy Filter**: Hierarchical selection (high-demand first, then fair selection)
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Tip: Use 'Realistic Test Config' expander in Integration Testing for preset configurations.")
    
    return {
        # Grid
        'grid_dims': (grid_x, grid_y),
        
        # Weights
        'alpha_spatial': alpha_spatial,
        'alpha_causal': alpha_causal,
        'alpha_fidelity': alpha_fidelity,
        
        # Soft cell assignment
        'temperature': temperature,
        'neighborhood_size': neighborhood_size,
        'use_annealing': use_annealing,
        'annealing_schedule': annealing_schedule,
        'final_temperature': final_temperature,
        
        # g(d) estimation
        'g_estimation_method': g_estimation_method,
        'n_bins': n_bins,
        'poly_degree': poly_degree,
        'lowess_frac': lowess_frac,
        'freeze_g_function': freeze_g_function,
        
        # Temporal
        'period_type': period_type,
        'selected_days': selected_days,
        
        # Filtering
        'include_zero_supply': include_zero_supply,
        'include_zero_demand': include_zero_demand,
        'min_demand_threshold': min_demand_threshold,
        
        # Attribution
        'lis_weight': lis_weight,
        'dcd_weight': dcd_weight,
        'selection_method': selection_method,
    }


# =============================================================================
# TAB 1: GRADIENT FLOW
# =============================================================================

def render_gradient_flow_tab(config: Dict[str, Any]):
    """Render the Objective Function Gradient Flow tab."""
    st.header("🔄 Objective Function Gradient Flow")
    
    st.markdown("""
    This tab helps you understand and verify how gradients flow through the 
    entire FAMAIL objective function, from trajectory coordinates to the final loss value.
    """)
    
    # Mathematical Formulation Section
    with st.expander("📐 Mathematical Formulation", expanded=False):
        st.markdown("""
        ### Combined Objective Function
        
        The FAMAIL objective combines three differentiable terms. All terms are designed to output
        values in [0, 1] where **higher is better** (maximization objective):
        """)
        
        st.latex(r"""
        \mathcal{L} = \alpha_1 \cdot F_{\text{causal}} + \alpha_2 \cdot F_{\text{spatial}} + \alpha_3 \cdot F_{\text{fidelity}}
        """)
        
        st.markdown("---")
        
        # Spatial Fairness - Full Formulation
        st.markdown("### 1. Spatial Fairness Term")
        st.markdown("Measures equitable distribution of pickups and dropoffs across grid cells using the Gini coefficient:")
        
        st.latex(r"""
        F_{\text{spatial}} = 1 - \frac{G(\tilde{C}_{\text{pickup}}) + G(\tilde{C}_{\text{dropoff}})}{2}
        """)
        
        st.markdown("Where the **Pairwise Gini coefficient** is computed as:")
        st.latex(r"""
        G(\tilde{C}) = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |\tilde{C}_i - \tilde{C}_j|}{2n^2 \cdot \bar{C}}
        """)
        
        st.markdown("""
        **Variables:**
        - $\\tilde{C}_i$ = Soft count in cell $i$ (differentiable via soft assignment)
        - $\\bar{C}$ = Mean count across all cells
        - $n$ = Total number of grid cells
        - Range: $G \\in [0, 1]$ where 0 = perfect equality, 1 = maximum inequality
        """)
        
        st.markdown("---")
        
        # Causal Fairness - Full Formulation
        st.markdown("### 2. Causal Fairness Term")
        st.markdown("Measures how well supply matches expected demand using R² regression quality:")
        
        st.latex(r"""
        F_{\text{causal}} = R^2 = 1 - \frac{\text{Var}(R)}{\text{Var}(Y)}
        """)
        
        st.markdown("Where the **residual** measures deviation from expected service ratio:")
        st.latex(r"""
        R_i = Y_i - g(D_i), \quad \text{where } Y_i = \frac{S_i}{D_i}
        """)
        
        st.markdown("""
        **Variables:**
        - $Y_i$ = Actual service ratio (supply/demand) in cell $i$
        - $D_i$ = Demand (pickup count) in cell $i$ - **differentiable via soft counts**
        - $S_i$ = Supply in cell $i$ (fixed during optimization)
        - $g(D_i)$ = Expected service ratio given demand $D_i$ (**frozen** baseline function)
        - $R_i$ = Residual: difference between actual and expected service ratio
        - Range: $R^2 \\in [0, 1]$ where 1 = perfect fit, 0 = no explanatory power
        
        **Key insight:** The $g(d)$ function is pre-fitted and frozen during optimization. 
        Only the demand distribution $D_i$ is modified through trajectory changes.
        """)
        
        st.markdown("---")
        
        # Fidelity - Full Formulation  
        st.markdown("### 3. Fidelity Term")
        st.markdown("Measures how similar modified trajectories are to original expert behavior:")
        
        st.latex(r"""
        F_{\text{fidelity}} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} D(\tau', \tau)
        """)
        
        st.markdown("""
        **Variables:**
        - $D(\cdot, \cdot)$ = Pre-trained ST-SiameseNet discriminator (same-driver probability)
        - $\\tau$ = Original trajectory
        - $\\tau'$ = Modified trajectory
        - $\\mathcal{T}$ = Set of trajectories being modified
        - Range: $F_{\\text{fidelity}} \\in [0, 1]$ where 1 = indistinguishable from original
        
        **Key insight:** The discriminator is pre-trained and frozen. Gradients flow through
        the trajectory features but not the discriminator weights.
        """)
        
        st.markdown("---")
        st.markdown("### Current Weights")
        col1, col2, col3 = st.columns(3)
        col1.metric("α_spatial", f"{config['alpha_spatial']:.3f}")
        col2.metric("α_causal", f"{config['alpha_causal']:.3f}")
        col3.metric("α_fidelity", f"{config['alpha_fidelity']:.3f}")
    
    # Gradient Flow Diagram
    with st.expander("📊 Gradient Flow Diagram", expanded=False):
        diagram = create_gradient_flow_diagram(format='ascii')
        st.code(diagram, language=None)
    
    # Gradient Verification Tests
    st.subheader("🧪 Gradient Verification Tests")
    
    if not TORCH_AVAILABLE:
        st.error("⚠️ PyTorch is required for gradient verification. Please install PyTorch.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        n_points = st.slider("Test Points", 10, 200, 50, 10)
    
    with col2:
        run_tests = st.button("▶️ Run Gradient Tests", type="primary")
    
    if run_tests:
        with st.spinner("Running gradient verification tests..."):
            report = verify_combined_gradients(
                grid_dims=config['grid_dims'],
                n_points=n_points,
                temperature=config['temperature'],
                alpha_spatial=config['alpha_spatial'],
                alpha_causal=config['alpha_causal'],
                alpha_fidelity=config['alpha_fidelity'],
                verbose=False,
            )
        
        # Overall status
        if report.overall_passed:
            st.markdown("""
            <div class="success-box">
                ✅ <strong>All gradient tests passed!</strong> Gradients flow correctly through the objective function.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="error-box">
                ❌ <strong>Some gradient tests failed.</strong> See details below.
            </div>
            """, unsafe_allow_html=True)
        
        # Per-term results
        st.markdown("### Term-by-Term Results")
        
        cols = st.columns(3)
        grad_data = {'Term': [], 'Mean Gradient': [], 'Abs Mean': [], 'Nonzero %': []}
        
        for i, (term_name, term_report) in enumerate(report.term_reports.items()):
            with cols[i]:
                st.markdown(f"**{term_name.upper()}**")
                st.metric("Value", f"{term_report.term_value:.4f}")
                
                if term_report.gradient_available:
                    st.success("✓ Gradient OK")
                    if term_report.gradient_stats:
                        stats = term_report.gradient_stats
                        st.caption(f"Mean: {stats.mean:.2e}")
                        st.caption(f"Nonzero: {stats.nonzero_ratio*100:.1f}%")
                        
                        # Collect data for comparison chart
                        grad_data['Term'].append(term_name.upper())
                        grad_data['Mean Gradient'].append(stats.mean)
                        grad_data['Abs Mean'].append(abs(stats.mean))
                        grad_data['Nonzero %'].append(stats.nonzero_ratio * 100)
                else:
                    st.error(f"✗ {term_report.error_message}")
        
        # Combined objective
        st.markdown("---")
        st.metric("Total Objective L", f"{report.total_objective:.4f}")
        
        # Gradient Comparison Visualization
        if grad_data['Term'] and PLOTLY_AVAILABLE:
            st.markdown("### Gradient Magnitude Comparison")
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=(
                "Absolute Gradient Magnitude (log scale)", 
                "Nonzero Gradient Percentage"
            ))
            
            # Bar chart of absolute gradient magnitudes
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            fig.add_trace(
                go.Bar(
                    x=grad_data['Term'], 
                    y=grad_data['Abs Mean'],
                    marker_color=colors[:len(grad_data['Term'])],
                    text=[f"{v:.2e}" for v in grad_data['Abs Mean']],
                    textposition='outside',
                    name='|Mean Gradient|'
                ),
                row=1, col=1
            )
            
            # Bar chart of nonzero percentages
            fig.add_trace(
                go.Bar(
                    x=grad_data['Term'], 
                    y=grad_data['Nonzero %'],
                    marker_color=colors[:len(grad_data['Term'])],
                    text=[f"{v:.1f}%" for v in grad_data['Nonzero %']],
                    textposition='outside',
                    name='Nonzero %'
                ),
                row=1, col=2
            )
            
            fig.update_yaxes(type="log", title_text="|Mean Gradient|", row=1, col=1)
            fig.update_yaxes(title_text="Percentage", range=[0, 110], row=1, col=2)
            fig.update_layout(height=350, showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Gradient magnitude analysis
            if len(grad_data['Abs Mean']) >= 2:
                max_grad = max(grad_data['Abs Mean'])
                min_grad = min(g for g in grad_data['Abs Mean'] if g > 0)
                ratio = max_grad / min_grad if min_grad > 0 else float('inf')
                
                if ratio > 1000:
                    st.warning(f"""
                    ⚠️ **Gradient Magnitude Imbalance Detected**
                    
                    The ratio between the largest and smallest gradient magnitudes is **{ratio:.1e}x**.
                    This can cause optimization issues:
                    - Terms with larger gradients will dominate updates
                    - Terms with smaller gradients may be effectively ignored
                    
                    **Recommendations:**
                    1. Consider **gradient normalization** per-term before combining
                    2. Adjust **α weights** to compensate (increase weight for low-gradient terms)
                    3. Use **adaptive learning rates** per-term
                    4. The causal term may have small gradients because it depends on aggregate statistics
                    """)
                else:
                    st.info(f"Gradient magnitude ratio: {ratio:.1f}x (acceptable)")
    
    # Temperature Annealing Analysis
    st.subheader("🌡️ Temperature Annealing Analysis")
    
    with st.expander("How temperature affects gradient flow"):
        st.markdown("""
        Temperature τ controls the "softness" of cell assignments:
        - **High τ (τ → ∞)**: Uniform distribution, large gradients but less precise
        - **Low τ (τ → 0)**: Sharp assignment, approaches hard assignment
        
        During optimization, we anneal τ from high to low.
        """)
        
        temps_to_test = st.multiselect(
            "Temperatures to test",
            options=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            default=[0.1, 0.5, 1.0, 2.0]
        )
        
        if st.button("Analyze Temperature Schedule") and TORCH_AVAILABLE:
            with st.spinner("Analyzing temperature schedule..."):
                results = analyze_temperature_schedule(
                    temperatures=temps_to_test,
                    grid_dims=config['grid_dims'],
                    n_points=50,
                )
            
            if PLOTLY_AVAILABLE:
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Objective Values", "Gradient Magnitude"))
                
                # Values plot
                fig.add_trace(
                    go.Scatter(x=results['temperatures'], y=results['spatial_values'], 
                              name='F_spatial', mode='lines+markers'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=results['temperatures'], y=results['causal_values'], 
                              name='F_causal', mode='lines+markers'),
                    row=1, col=1
                )
                
                # Gradient plot - convert to scientific notation friendly values
                grad_values = np.abs(results['spatial_grad_means'])
                
                fig.add_trace(
                    go.Scatter(
                        x=results['temperatures'], 
                        y=grad_values,
                        name='|∇F_spatial|', 
                        mode='lines+markers',
                        hovertemplate='τ=%{x:.2f}<br>|∇F|=%{y:.2e}<extra></extra>'
                    ),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Temperature τ", row=1, col=1)
                fig.update_xaxes(title_text="Temperature τ", row=1, col=2)
                fig.update_yaxes(title_text="Objective Value [0-1]", row=1, col=1)
                fig.update_yaxes(
                    title_text="Mean |∂L/∂coord| (per coordinate)", 
                    row=1, col=2,
                    tickformat='.2e'  # Scientific notation
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation for the results
                st.markdown("""
                **Interpreting the Results:**
                
                📊 **Objective Values (Left Plot):**
                - **F_spatial** typically increases with temperature because higher τ spreads counts 
                  more evenly across cells, reducing Gini inequality
                - **F_causal** may appear flat or near zero in these tests because:
                  - The test uses a simplified g(d) = constant, making all cells equally "fair"
                  - With random test points, variance in residuals equals variance in Y
                  - Real data with actual demand patterns will show more variation
                
                📈 **Gradient Magnitude (Right Plot):**
                - Shows the average absolute gradient per coordinate: $|\\partial \\mathcal{L} / \\partial x|$
                - Values are typically very small (10⁻⁴ to 10⁻⁶) because:
                  - Soft assignment spreads influence across many cells
                  - Each coordinate only affects a small region
                - Higher temperature → larger gradients (smoother landscape)
                - Lower temperature → smaller gradients (sharper but more precise)
                """)
            else:
                st.dataframe({
                    'Temperature': results['temperatures'],
                    'F_spatial': results['spatial_values'],
                    'F_causal': results['causal_values'],
                    '|∇F_spatial|': results['spatial_grad_means'],
                })


# =============================================================================
# TAB 2: SOFT CELL ASSIGNMENT
# =============================================================================

def render_soft_cell_assignment_tab(config: Dict[str, Any]):
    """Render the Soft Cell Assignment tab."""
    st.header("🔲 Soft Cell Assignment")
    
    st.markdown("""
    Soft cell assignment is the key technique that makes the objective function 
    differentiable. Instead of discrete hard assignments, we use a Gaussian softmax.
    """)
    
    # Formula
    with st.expander("📐 Mathematical Formulation", expanded=True):
        st.markdown("### Soft Assignment Formula")
        st.latex(r"""
        \sigma(p|\tau) = \frac{\exp(-\|loc - cell_p\|^2 / \tau)}{\sum_{q \in \mathcal{N}} \exp(-\|loc - cell_q\|^2 / \tau)}
        """)
        
        st.markdown("""
        Where:
        - $loc$ is the continuous location coordinate
        - $cell_p$ is the center of cell $p$
        - $\\tau$ is the temperature parameter
        - $\\mathcal{N}$ is the neighborhood of valid cells
        
        ### Soft Counts
        """)
        
        st.latex(r"""
        \tilde{C}_i = \sum_{\tau \in T} \sigma(i|\tau)
        """)
        
        st.markdown("The soft count for cell $i$ is the sum of soft probabilities from all trajectories.")
    
    # Interactive Visualization
    st.subheader("🎮 Interactive Soft Assignment")
    
    st.markdown("""
    Explore how a single point's location affects its soft assignment probabilities across neighboring cells.
    The visualization shows how probability mass is distributed based on the point's position within its cell.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Point Location**")
        st.caption("Simulate a trajectory endpoint at these grid coordinates")
        
        loc_x = st.slider(
            "X coordinate", 
            0.0, float(config['grid_dims'][0]-1), 
            float(config['grid_dims'][0]//2) + 0.3, 0.1,
            help="Horizontal position in the grid. The decimal part (e.g., .3) represents position within a cell. Try values like 5.0 (cell center) vs 5.5 (cell edge) to see how position affects assignment."
        )
        loc_y = st.slider(
            "Y coordinate", 
            0.0, float(config['grid_dims'][1]-1), 
            float(config['grid_dims'][1]//2) + 0.7, 0.1,
            help="Vertical position in the grid. Combined with X, this determines which cell the point is in and where within that cell."
        )
        
        temp = st.slider(
            "Temperature τ", 
            0.01, 5.0, config['temperature'], 0.01,
            help="Controls softness of assignment. Low τ (0.1): nearly hard assignment to center cell. High τ (2+): probability spreads to many neighbors. This is the key parameter for differentiability."
        )
        k = config['neighborhood_size']
        
        st.markdown("---")
        st.markdown("**What to observe:**")
        st.markdown("""
        - 🎯 **Center cell** (red X): Where the point would be hard-assigned
        - 🔵 **Color intensity**: Probability mass at each neighbor cell
        - 📊 **Position within cell**: Points near edges spread more probability to neighbors
        """)
    
    with col2:
        if TORCH_AVAILABLE and PLOTLY_AVAILABLE:
            # Compute soft assignment
            from soft_cell_assignment import SoftCellAssignment
            
            soft_assign = SoftCellAssignment(
                grid_dims=config['grid_dims'],
                neighborhood_size=k,
                initial_temperature=temp,
            )
            
            import torch
            loc = torch.tensor([[loc_x, loc_y]], dtype=torch.float32)
            cell = loc.floor().long().clamp(
                min=torch.tensor([0, 0]),
                max=torch.tensor([config['grid_dims'][0]-1, config['grid_dims'][1]-1])
            )
            
            probs = soft_assign(loc, cell.float())
            probs_np = probs[0].detach().numpy()
            
            # Get the actual k (radius) from the module for correct indexing
            actual_k = soft_assign.k
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=probs_np,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Probability"),
            ))
            
            # Mark center
            fig.add_trace(go.Scatter(
                x=[actual_k], y=[actual_k],
                mode='markers',
                marker=dict(color='red', size=15, symbol='x'),
                name='Center cell'
            ))
            
            fig.update_layout(
                title=f"Soft Assignment Probabilities (τ={temp:.2f})",
                xaxis_title="Δx from center",
                yaxis_title="Δy from center",
                height=400,
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show key probabilities
            center_prob = probs_np[actual_k, actual_k]
            total_prob = probs_np.sum()
            
            cols = st.columns(3)
            cols[0].metric("Center Cell Probability", f"{center_prob:.3f}")
            cols[1].metric("Total Probability", f"{total_prob:.6f}")
            cols[2].metric("Neighborhood Size", f"{2*actual_k+1}×{2*actual_k+1}")
        else:
            st.warning("PyTorch and Plotly required for visualization")
    
    # Temperature Comparison
    st.subheader("🌡️ Temperature Effect Comparison")
    
    if TORCH_AVAILABLE and PLOTLY_AVAILABLE:
        temps = [0.1, 0.5, 1.0, 2.0]
        
        from soft_cell_assignment import SoftCellAssignment
        import torch
        
        # Use neighborhood_size from config
        ns = config['neighborhood_size']
        
        fig = make_subplots(rows=1, cols=4, subplot_titles=[f"τ = {t}" for t in temps])
        
        for i, t in enumerate(temps):
            soft_assign = SoftCellAssignment(
                grid_dims=config['grid_dims'],
                neighborhood_size=ns,
                initial_temperature=t,
            )
            
            loc = torch.tensor([[loc_x, loc_y]], dtype=torch.float32)
            cell = loc.floor().long().clamp(
                min=torch.tensor([0, 0]),
                max=torch.tensor([config['grid_dims'][0]-1, config['grid_dims'][1]-1])
            )
            
            probs = soft_assign(loc, cell.float())
            probs_np = probs[0].detach().numpy()
            
            fig.add_trace(
                go.Heatmap(z=probs_np, colorscale='Blues', showscale=(i==3)),
                row=1, col=i+1
            )
        
        fig.update_layout(height=300, title_text="Soft Assignment at Different Temperatures")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Observations:**
        - **Low τ (0.1)**: Assignment concentrated in center cell (nearly hard)
        - **Medium τ (1.0)**: Soft spread to neighboring cells
        - **High τ (2.0)**: Wider spread, more uncertainty
        """)


# =============================================================================
# TAB 3: ATTRIBUTION METHODS
# =============================================================================

def render_attribution_tab(config: Dict[str, Any]):
    """Render the Attribution Methods tab."""
    st.header("🎯 Attribution Methods & Trajectory Selection")
    
    st.markdown("""
    This tab demonstrates how LIS (Local Inequality Score) and DCD (Demand-Conditional Deviation)
    work together to identify which trajectories should be modified to improve fairness.
    """)
    
    # Formulas
    with st.expander("📐 Attribution Formulas", expanded=False):
        st.markdown("""
        Attribution methods identify **which trajectories contribute most to unfairness**.
        By modifying high-attribution trajectories, we can improve fairness most efficiently.
        """)
        
        st.markdown("---")
        
        # LIS Full Definition
        st.markdown("### LIS: Local Inequality Score (Spatial Attribution)")
        
        st.latex(r"LIS_i = \frac{|c_i - \mu|}{\mu}")
        
        st.markdown("""
        **Variable Definitions:**
        - $c_i$ = Count (pickups or dropoffs) in cell $i$
        - $\\mu = \\frac{1}{n}\\sum_{j=1}^{n} c_j$ = Global mean count across all cells
        - $n$ = Total number of grid cells
        
        **Interpretation:**
        - $LIS_i = 0$: Cell has exactly the mean count (perfectly fair)
        - $LIS_i = 1$: Cell deviates from mean by 100% (e.g., count is 0 or 2×mean)
        - $LIS_i > 1$: Cell has extreme over/under-representation
        
        **Intuition:** Cells with high LIS contribute disproportionately to the Gini coefficient.
        Trajectories starting/ending in high-LIS cells are prime candidates for modification.
        """)
        
        st.markdown("**For trajectories**, we aggregate pickup and dropoff LIS:")
        st.latex(r"LIS_\tau = \max(LIS_{\text{pickup}}, LIS_{\text{dropoff}})")
        
        st.markdown("---")
        
        # DCD Full Definition
        st.markdown("### DCD: Demand-Conditional Deviation (Causal Attribution)")
        
        st.latex(r"DCD_i = |Y_i - g(D_i)|")
        
        st.markdown("""
        **Variable Definitions:**
        - $Y_i = S_i / D_i$ = Actual service ratio in cell $i$
        - $S_i$ = Supply (taxis available) in cell $i$
        - $D_i$ = Demand (pickup requests) in cell $i$
        - $g(D_i)$ = Expected service ratio for demand level $D_i$ (from pre-fitted baseline)
        
        **Interpretation:**
        - $DCD_i = 0$: Cell's service matches what's expected for its demand level
        - $DCD_i > 0$: Cell is over-served or under-served relative to expectation
        
        **Intuition:** The $g(d)$ function captures the "natural" relationship between demand and service.
        High DCD indicates **discrimination** — the cell gets different service than other cells with similar demand.
        This is the causal fairness criterion: outcomes shouldn't depend on location after controlling for demand.
        """)
        
        st.markdown("**For trajectories**, DCD is based on the pickup cell (demand location):")
        st.latex(r"DCD_\tau = DCD_{\text{pickup cell}}")
        
        st.markdown("---")
        
        # Combined Attribution
        st.markdown("### Combined Attribution Score")
        
        st.latex(r"Score_\tau = w_1 \cdot \widehat{LIS}_\tau + w_2 \cdot \widehat{DCD}_\tau")
        
        st.markdown(f"""
        **Where:**
        - $\\widehat{{LIS}}_\\tau$ = Normalized LIS (divided by max LIS across trajectories)
        - $\\widehat{{DCD}}_\\tau$ = Normalized DCD (divided by max DCD across trajectories)
        - $w_1$ = {config['lis_weight']:.2f} (LIS weight for spatial fairness)
        - $w_2$ = {config['dcd_weight']:.2f} (DCD weight for causal fairness)
        
        **Interpretation:**
        - Scores are normalized to [0, 1] for comparability
        - Higher combined score → trajectory contributes more to overall unfairness
        - Weights let you prioritize spatial vs. causal fairness improvements
        
        **Selection Strategy:** Trajectories with the highest combined scores are selected for modification,
        as changing them will have the largest impact on improving fairness.
        """)
    
    # Data Source Selection
    st.subheader("📂 Data Source")
    
    data_source = st.radio(
        "Choose data source",
        ["Generate Synthetic Data", "Load from File"],
        horizontal=True
    )
    
    trajectories = []
    pickup_counts = None
    dropoff_counts = None
    supply_counts = None
    
    if data_source == "Generate Synthetic Data":
        col1, col2 = st.columns(2)
        with col1:
            n_trajectories = st.number_input("Number of trajectories", 100, 10000, 1000, 100)
        with col2:
            cluster_factor = st.slider("Clustering factor", 0.0, 1.0, 0.5, 0.1)
        
        if st.button("Generate Data"):
            with st.spinner("Generating synthetic trajectories..."):
                trajectories, pickup_counts, dropoff_counts, supply_counts = generate_synthetic_data(
                    n_trajectories=n_trajectories,
                    grid_dims=config['grid_dims'],
                    cluster_factor=cluster_factor,
                )
                st.session_state['attribution_data'] = {
                    'trajectories': trajectories,
                    'pickup_counts': pickup_counts,
                    'dropoff_counts': dropoff_counts,
                    'supply_counts': supply_counts,
                }
                st.success(f"Generated {len(trajectories)} trajectories")
    
    else:  # Load from file
        data_paths = list(PROJECT_ROOT.glob("**/*.pkl"))
        all_trajs_paths = [p for p in data_paths if 'all_trajs' in p.name.lower()]
        
        if all_trajs_paths:
            selected_path = st.selectbox("Select data file", all_trajs_paths)
            n_samples = st.number_input("Sample size (0 = all)", 0, 100000, 1000, 100)
            
            if st.button("Load Data"):
                with st.spinner(f"Loading from {selected_path.name}..."):
                    try:
                        trajectories = load_trajectories_from_all_trajs(
                            selected_path,
                            n_samples=n_samples if n_samples > 0 else None,
                        )
                        pickup_counts, dropoff_counts, pickup_cells, dropoff_cells = \
                            extract_cells_from_trajectories(trajectories, config['grid_dims'])
                        supply_counts = create_mock_supply_data(pickup_counts)
                        
                        st.session_state['attribution_data'] = {
                            'trajectories': trajectories,
                            'pickup_counts': pickup_counts,
                            'dropoff_counts': dropoff_counts,
                            'supply_counts': supply_counts,
                        }
                        st.success(f"Loaded {len(trajectories)} trajectories")
                        
                        # Show diagnostic info
                        with st.expander("📊 Data Statistics", expanded=False):
                            n_pickup_cells = (pickup_counts > 0).sum()
                            n_dropoff_cells = (dropoff_counts > 0).sum()
                            st.write(f"Unique pickup cells: {n_pickup_cells}")
                            st.write(f"Unique dropoff cells: {n_dropoff_cells}")
                            st.write(f"Total pickups: {pickup_counts.sum():.0f}")
                            st.write(f"Total dropoffs: {dropoff_counts.sum():.0f}")
                            if trajectories:
                                st.write(f"Sample trajectory: {trajectories[0]}")
                                
                    except Exception as e:
                        st.error(f"Error loading data: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.warning("No .pkl files found in workspace")
    
    # Load cached data
    if 'attribution_data' in st.session_state:
        data = st.session_state['attribution_data']
        trajectories = data['trajectories']
        pickup_counts = data['pickup_counts']
        dropoff_counts = data['dropoff_counts']
        supply_counts = data['supply_counts']
    
    if trajectories and pickup_counts is not None:
        st.markdown("---")
        
        # Compute Attribution
        st.subheader("🔍 Attribution Analysis")
        
        # Create g(d) function
        demand_flat = pickup_counts.flatten()
        supply_flat = supply_counts.flatten()
        mask = demand_flat > 0.1
        
        if mask.sum() >= 2:
            g_function = create_default_g_function(
                demand_flat[mask],
                supply_flat[mask] / (demand_flat[mask] + 1e-8),
                method='isotonic'
            )
            
            # Compute attribution
            with st.spinner("Computing attribution scores..."):
                attribution_result = compute_combined_attribution(
                    trajectories=trajectories,
                    pickup_counts=pickup_counts,
                    dropoff_counts=dropoff_counts,
                    supply_counts=supply_counts,
                    g_function=g_function,
                    lis_weight=config['lis_weight'],
                    dcd_weight=config['dcd_weight'],
                )
            
            # Show results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Cell-Level Statistics")
                if attribution_result.cell_stats:
                    stats = attribution_result.cell_stats
                    st.metric("Mean LIS (Pickup)", f"{stats['pickup_lis_mean']:.4f}")
                    st.metric("Max LIS (Pickup)", f"{stats['pickup_lis_max']:.4f}")
                    st.metric("Mean DCD", f"{stats['dcd_mean']:.4f}")
            
            with col2:
                st.markdown("### Trajectory Scores")
                st.metric("Total Trajectories", len(attribution_result.trajectory_scores))
                
                # Score distribution
                scores = [s.combined_score for s in attribution_result.trajectory_scores]
                st.metric("Mean Score", f"{np.mean(scores):.4f}")
                st.metric("Max Score", f"{np.max(scores):.4f}")
        
            # Count and Attribution Heatmaps
            if PLOTLY_AVAILABLE:
                # First show the raw counts for context
                st.subheader("📊 Pickup and Dropoff Distributions")
                st.markdown("These count distributions are the input to the attribution calculations.")
                
                fig_counts = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Pickup Counts", "Dropoff Counts"),
                )
                
                fig_counts.add_trace(
                    go.Heatmap(
                        z=pickup_counts.T, 
                        colorscale='Greens',
                        hovertemplate='Cell (%{x}, %{y})<br>Pickup Count: %{z}<extra></extra>',
                        colorbar=dict(title="Count", x=0.46)
                    ), 
                    row=1, col=1
                )
                fig_counts.add_trace(
                    go.Heatmap(
                        z=dropoff_counts.T, 
                        colorscale='Purples',
                        hovertemplate='Cell (%{x}, %{y})<br>Dropoff Count: %{z}<extra></extra>',
                        colorbar=dict(title="Count", x=1.0)
                    ), 
                    row=1, col=2
                )
                
                fig_counts.update_layout(height=350)
                fig_counts.update_xaxes(title_text="Grid X")
                fig_counts.update_yaxes(title_text="Grid Y")
                st.plotly_chart(fig_counts, use_container_width=True)
                
                # Now show attribution maps
                st.subheader("🗺️ Cell-Level Attribution Maps")
                st.markdown("""
                These maps show which cells have the highest attribution scores. 
                Higher scores (brighter colors) indicate cells that contribute more to unfairness.
                """)
                
                pickup_lis = compute_all_lis_scores(pickup_counts)
                dropoff_lis = compute_all_lis_scores(dropoff_counts)
                dcd_scores = compute_all_dcd_scores(pickup_counts, supply_counts, g_function)
                
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=(
                        "LIS (Pickup) - Spatial", 
                        "LIS (Dropoff) - Spatial", 
                        "DCD - Causal"
                    ),
                )
                
                fig.add_trace(
                    go.Heatmap(
                        z=pickup_lis.T, 
                        colorscale='Reds',
                        hovertemplate='Cell (%{x}, %{y})<br>LIS Score: %{z:.3f}<extra></extra>',
                        colorbar=dict(title="LIS", x=0.3)
                    ), 
                    row=1, col=1
                )
                fig.add_trace(
                    go.Heatmap(
                        z=dropoff_lis.T, 
                        colorscale='Reds',
                        hovertemplate='Cell (%{x}, %{y})<br>LIS Score: %{z:.3f}<extra></extra>',
                        colorbar=dict(title="LIS", x=0.64)
                    ), 
                    row=1, col=2
                )
                fig.add_trace(
                    go.Heatmap(
                        z=dcd_scores.T, 
                        colorscale='Blues',
                        hovertemplate='Cell (%{x}, %{y})<br>DCD Score: %{z:.3f}<extra></extra>',
                        colorbar=dict(title="DCD", x=1.0)
                    ), 
                    row=1, col=3
                )
                
                fig.update_layout(height=400)
                fig.update_xaxes(title_text="Grid X")
                fig.update_yaxes(title_text="Grid Y")
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                st.markdown("""
                **Reading the maps:**
                - **LIS (Red):** Bright cells have counts far from the mean → modify trajectories here to reduce Gini
                - **DCD (Blue):** Bright cells have service ratios different from expected → modify to improve causal fairness
                - Trajectories with pickup/dropoff in bright cells get high attribution scores
                """)
            
            # Trajectory Selection
            st.subheader("📋 Trajectory Selection for Modification")
            
            col1, col2 = st.columns(2)
            with col1:
                n_select = st.slider("Number of trajectories to select", 1, 100, 10)
            with col2:
                selection_method = st.selectbox(
                    "Selection method",
                    ["top_n", "threshold", "diverse"]
                )
            
            selected = select_trajectories_for_modification(
                attribution_result,
                n_trajectories=n_select,
                selection_method=selection_method,
            )
            
            st.markdown(f"**Selected {len(selected)} trajectories for modification:**")
            
            # Create dataframe
            import pandas as pd
            df = pd.DataFrame([
                {
                    'Rank': i+1,
                    'Trajectory ID': s.trajectory_id,
                    'LIS Score': f"{s.lis_score:.4f}",
                    'DCD Score': f"{s.dcd_score:.4f}",
                    'Combined Score': f"{s.combined_score:.4f}",
                    'Pickup Cell': str(s.pickup_cell),
                    'Dropoff Cell': str(s.dropoff_cell),
                }
                for i, s in enumerate(selected)
            ])
            
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Selection",
                csv,
                "selected_trajectories.csv",
                "text/csv",
            )
        else:
            st.warning("Insufficient data to compute g(d) function")


# =============================================================================
# TAB 4: INTEGRATION TESTING
# =============================================================================

def _load_real_data_for_testing(
    config: Dict[str, Any],
    n_samples: int = 100,
) -> Optional[Dict[str, Any]]:
    """
    Load real trajectory data from all_trajs.pkl for integration testing.
    
    Returns:
        Dict with 'trajectories', 'pickup_coords', 'dropoff_coords', 'metadata' or None if failed
    """
    import torch
    
    # Find the all_trajs.pkl file
    possible_paths = [
        PROJECT_ROOT / 'data' / 'Processed_Data' / 'all_trajs.pkl',
        PROJECT_ROOT / 'source_data' / 'all_trajs.pkl',
        PROJECT_ROOT / 'cGAIL_data_and_processing' / 'Data' / 'trajectory_info' / 'all_trajs.pkl',
    ]
    
    filepath = None
    for path in possible_paths:
        if path.exists():
            filepath = path
            break
    
    if filepath is None:
        return None
    
    try:
        trajectories = load_trajectories_from_all_trajs(filepath, n_samples=n_samples)
        
        if len(trajectories) == 0:
            return None
        
        # Convert to coordinate tensors
        pickup_coords_list = []
        dropoff_coords_list = []
        
        for traj in trajectories:
            px, py = traj['pickup_cell']
            dx, dy = traj['dropoff_cell']
            # Add small random offset within cell for continuous coordinates
            pickup_coords_list.append([
                min(px + np.random.rand() * 0.9, config['grid_dims'][0] - 0.01),
                min(py + np.random.rand() * 0.9, config['grid_dims'][1] - 0.01)
            ])
            dropoff_coords_list.append([
                min(dx + np.random.rand() * 0.9, config['grid_dims'][0] - 0.01),
                min(dy + np.random.rand() * 0.9, config['grid_dims'][1] - 0.01)
            ])
        
        pickup_coords = torch.tensor(pickup_coords_list, dtype=torch.float32)
        dropoff_coords = torch.tensor(dropoff_coords_list, dtype=torch.float32)
        
        # Compute metadata
        unique_drivers = len(set(traj.get('driver_id', i) for i, traj in enumerate(trajectories)))
        avg_states = np.mean([traj.get('n_states', 0) for traj in trajectories if 'n_states' in traj]) if trajectories else 0
        
        # Cell distribution
        pickup_cells = [traj['pickup_cell'] for traj in trajectories]
        dropoff_cells = [traj['dropoff_cell'] for traj in trajectories]
        unique_pickup_cells = len(set(pickup_cells))
        unique_dropoff_cells = len(set(dropoff_cells))
        
        metadata = {
            'filepath': str(filepath),
            'n_trajectories': len(trajectories),
            'n_drivers': unique_drivers,
            'avg_states_per_traj': avg_states,
            'unique_pickup_cells': unique_pickup_cells,
            'unique_dropoff_cells': unique_dropoff_cells,
            'grid_dims': config['grid_dims'],
        }
        
        return {
            'trajectories': trajectories,
            'pickup_coords': pickup_coords,
            'dropoff_coords': dropoff_coords,
            'metadata': metadata,
        }
        
    except Exception as e:
        st.warning(f"Could not load real data: {e}")
        return None


def _load_supply_demand_data(
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Load real supply (active_taxis) and demand (pickup_dropoff_counts) data.
    
    Uses the full data processing pipeline with proper temporal filtering
    and g(d) estimation.
    
    Args:
        config: Dashboard configuration with period_type, selected_days, etc.
        
    Returns:
        Dict with 'demand', 'supply', 'g_function', 'diagnostics' or None
    """
    from causal_fairness.utils import (
        load_pickup_dropoff_counts,
        load_active_taxis_data,
        extract_demand_from_counts,
        aggregate_to_period,
        filter_by_days,
        estimate_g_function,
        compute_service_ratios,
        extract_demand_ratio_arrays,
    )
    
    # Paths to data files
    pickup_dropoff_path = PROJECT_ROOT / 'pickup_dropoff_counts' / 'output' / 'pickup_dropoff_counts.pkl'
    active_taxis_path = PROJECT_ROOT / 'active_taxis' / 'output' / 'active_taxis_5x5_hourly.pkl'
    
    # Select appropriate active_taxis file based on period_type
    period_type = config.get('period_type', 'hourly')
    if period_type == 'time_bucket':
        active_taxis_path = PROJECT_ROOT / 'active_taxis' / 'output' / 'active_taxis_5x5_time_bucket.pkl'
    elif period_type == 'daily':
        active_taxis_path = PROJECT_ROOT / 'active_taxis' / 'output' / 'active_taxis_5x5_daily.pkl'
    elif period_type == 'all':
        active_taxis_path = PROJECT_ROOT / 'active_taxis' / 'output' / 'active_taxis_5x5_all.pkl'
    
    diagnostics = {
        'pickup_dropoff_path': str(pickup_dropoff_path),
        'active_taxis_path': str(active_taxis_path),
        'period_type': period_type,
    }
    
    # Check if files exist
    if not pickup_dropoff_path.exists():
        diagnostics['error'] = f"pickup_dropoff_counts.pkl not found at {pickup_dropoff_path}"
        return {'diagnostics': diagnostics, 'error': True}
    
    if not active_taxis_path.exists():
        diagnostics['error'] = f"active_taxis file not found at {active_taxis_path}"
        return {'diagnostics': diagnostics, 'error': True}
    
    try:
        # Load raw data
        pickup_dropoff_data = load_pickup_dropoff_counts(str(pickup_dropoff_path))
        active_taxis_data = load_active_taxis_data(str(active_taxis_path))
        
        diagnostics['n_pickup_dropoff_entries'] = len(pickup_dropoff_data)
        diagnostics['n_active_taxis_entries'] = len(active_taxis_data)
        
        # Extract demand from pickup counts
        demand = extract_demand_from_counts(pickup_dropoff_data)
        
        # Aggregate to desired period
        if period_type != 'time_bucket':
            demand = aggregate_to_period(demand, period_type)
            # Note: active_taxis data should already be at the right granularity
        
        # Filter by days if specified
        selected_days = config.get('selected_days', [0, 1, 2, 3, 4, 5, 6])
        if selected_days and len(selected_days) < 7:
            demand = filter_by_days(demand, selected_days)
            active_taxis_data = filter_by_days(active_taxis_data, selected_days)
        
        diagnostics['n_demand_after_filter'] = len(demand)
        diagnostics['n_supply_after_filter'] = len(active_taxis_data)
        diagnostics['selected_days'] = selected_days
        
        # Handle zero filtering
        include_zero_supply = config.get('include_zero_supply', False)
        include_zero_demand = config.get('include_zero_demand', False)
        min_demand_threshold = config.get('min_demand_threshold', 0)
        
        # Compute service ratios
        # The min_demand parameter controls zero demand filtering
        ratios = compute_service_ratios(
            demand,
            active_taxis_data,
            min_demand=0 if include_zero_demand else max(1, min_demand_threshold),
            include_zero_supply=include_zero_supply,
        )
        
        diagnostics['n_ratio_entries'] = len(ratios)
        
        # Apply minimum demand threshold
        if min_demand_threshold > 0:
            ratios = {k: v for k, v in ratios.items() if demand.get(k, 0) >= min_demand_threshold}
            diagnostics['n_ratios_after_threshold'] = len(ratios)
        
        # Prepare data for g(d) estimation
        demands_arr, ratios_arr, common_keys = extract_demand_ratio_arrays(demand, ratios)
        
        diagnostics['n_common_entries'] = len(common_keys)
        diagnostics['demand_range'] = [float(demands_arr.min()), float(demands_arr.max())] if len(demands_arr) > 0 else [0, 0]
        diagnostics['ratio_range'] = [float(ratios_arr.min()), float(ratios_arr.max())] if len(ratios_arr) > 0 else [0, 0]
        
        # Estimate g(d) function
        g_method = config.get('g_estimation_method', 'isotonic')
        n_bins = config.get('n_bins', 10)
        poly_degree = config.get('poly_degree', 2)
        lowess_frac = config.get('lowess_frac', 0.3)
        
        g_function, g_diagnostics = estimate_g_function(
            demands_arr,
            ratios_arr,
            method=g_method,
            n_bins=n_bins,
            poly_degree=poly_degree,
            lowess_frac=lowess_frac,
        )
        
        diagnostics['g_estimation'] = g_diagnostics
        
        # Compute R² for diagnostics
        if len(demands_arr) > 0:
            from causal_fairness.utils import compute_r_squared
            g_predictions = g_function(demands_arr)
            r_squared = compute_r_squared(ratios_arr, g_predictions)
            diagnostics['g_r_squared'] = r_squared
        
        return {
            'demand': demand,
            'supply': active_taxis_data,
            'ratios': ratios,
            'g_function': g_function,
            'demands_array': demands_arr,
            'ratios_array': ratios_arr,
            'diagnostics': diagnostics,
            'error': False,
        }
        
    except Exception as e:
        import traceback
        diagnostics['error'] = str(e)
        diagnostics['traceback'] = traceback.format_exc()
        return {'diagnostics': diagnostics, 'error': True}


def _create_supply_tensor_from_data(
    supply_data: Dict[Tuple, int],
    grid_dims: Tuple[int, int],
    period: Optional[Tuple[int, int]] = None,
) -> 'torch.Tensor':
    """
    Convert supply dictionary to grid tensor.
    
    If period is specified, uses only that (time, day) period.
    Otherwise, sums across all periods.
    
    Args:
        supply_data: Dictionary {(x, y, time, day): count}
        grid_dims: Grid dimensions (x, y)
        period: Optional (time, day) tuple to filter
        
    Returns:
        Tensor of shape grid_dims with supply counts
    """
    import torch
    
    supply_tensor = torch.zeros(grid_dims, dtype=torch.float32)
    
    for key, value in supply_data.items():
        if len(key) >= 4:
            x, y, time_val, day_val = key[0], key[1], key[2], key[3]
            
            if period is not None:
                if (time_val, day_val) != period:
                    continue
            
            if 0 <= x < grid_dims[0] and 0 <= y < grid_dims[1]:
                supply_tensor[x, y] += value
    
    return supply_tensor


class FrozenGFunctionLookup:
    """
    Frozen g(d) lookup table for efficient use during optimization.
    
    Pre-computes g(d) values for a range of demand values and stores
    them for fast lookup during forward passes. This avoids recomputing
    g(d) estimation at each step.
    """
    
    def __init__(
        self,
        g_function: callable,
        demand_range: Tuple[float, float],
        n_lookup_points: int = 1000,
    ):
        """
        Initialize frozen lookup table.
        
        Args:
            g_function: The fitted g(d) function
            demand_range: (min_demand, max_demand) range
            n_lookup_points: Number of points in lookup table
        """
        import numpy as np
        
        self.min_demand, self.max_demand = demand_range
        self.n_points = n_lookup_points
        
        # Create lookup table
        self.demand_values = np.linspace(
            self.min_demand,
            self.max_demand,
            n_lookup_points
        )
        self.g_values = g_function(self.demand_values)
        
        # Pre-compute step size for fast indexing
        self.step = (self.max_demand - self.min_demand) / (n_lookup_points - 1) if n_lookup_points > 1 else 1.0
    
    def __call__(self, demands):
        """
        Look up g(d) values using linear interpolation.
        
        Args:
            demands: Array of demand values
            
        Returns:
            Array of g(d) values
        """
        import numpy as np
        
        demands = np.atleast_1d(demands)
        
        # Clamp to valid range
        clamped = np.clip(demands, self.min_demand, self.max_demand)
        
        # Compute indices
        indices = (clamped - self.min_demand) / self.step
        lower_idx = np.floor(indices).astype(int)
        lower_idx = np.clip(lower_idx, 0, self.n_points - 2)
        upper_idx = lower_idx + 1
        
        # Linear interpolation weights
        alpha = indices - lower_idx
        
        # Interpolate
        result = (1 - alpha) * self.g_values[lower_idx] + alpha * self.g_values[upper_idx]
        
        return result
    
    def get_lookup_table(self) -> Dict[str, np.ndarray]:
        """Return the lookup table for visualization."""
        return {
            'demands': self.demand_values.copy(),
            'g_values': self.g_values.copy(),
        }


def _create_frozen_g_function(
    supply_demand_data: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[callable, Dict[str, Any]]:
    """
    Create a frozen g(d) lookup function from loaded data.
    
    This is the recommended approach for optimization: estimate g(d)
    once from historical data, then freeze it for use in the objective.
    
    Args:
        supply_demand_data: Output from _load_supply_demand_data
        config: Dashboard configuration
        
    Returns:
        Tuple of (frozen_g_function, diagnostics)
    """
    if supply_demand_data.get('error'):
        # Return identity function if data loading failed
        def identity_g(d):
            return np.ones_like(np.atleast_1d(d))
        return identity_g, {'error': 'data_loading_failed'}
    
    g_function = supply_demand_data['g_function']
    diagnostics = supply_demand_data['diagnostics'].copy()
    
    # Get demand range
    demands_arr = supply_demand_data.get('demands_array', np.array([0, 100]))
    if len(demands_arr) > 0:
        demand_range = (float(demands_arr.min()), float(demands_arr.max()))
    else:
        demand_range = (0.0, 100.0)
    
    # Create frozen lookup
    if config.get('freeze_g_function', True):
        frozen = FrozenGFunctionLookup(
            g_function=g_function,
            demand_range=demand_range,
            n_lookup_points=1000,
        )
        diagnostics['frozen'] = True
        diagnostics['n_lookup_points'] = 1000
        return frozen, diagnostics
    else:
        diagnostics['frozen'] = False
        return g_function, diagnostics


def _run_comprehensive_gradient_test(
    config: Dict[str, Any],
    pickup_coords: 'torch.Tensor',
    dropoff_coords: 'torch.Tensor',
    use_real_data: bool = False,
    metadata: Optional[Dict] = None,
    use_real_supply_demand: bool = False,
    use_hard_counts: bool = True,
) -> Dict[str, Any]:
    """
    Run comprehensive gradient test with all objective terms.
    
    When use_real_supply_demand=True, loads actual supply/demand data
    and uses the full data processing pipeline with configured g(d) estimation.
    
    Args:
        config: Dashboard configuration (including g(d) settings, period_type, etc.)
        pickup_coords: Tensor of pickup coordinates
        dropoff_coords: Tensor of dropoff coordinates
        use_real_data: Whether trajectory data came from real file
        metadata: Optional trajectory metadata
        use_real_supply_demand: Whether to load real active_taxis/pickup_dropoff data
        use_hard_counts: Use fast hard cell assignment instead of soft (recommended for large datasets)
    
    Returns:
        Dict with test results
    """
    import torch
    
    n_trajectories = pickup_coords.shape[0]
    
    results = {
        'passed': True,
        'use_real_data': use_real_data,
        'use_real_supply_demand': use_real_supply_demand,
        'use_hard_counts': use_hard_counts,
        'n_trajectories': n_trajectories,
        'grid_dims': config['grid_dims'],
        'temperature': config['temperature'],
        'config_summary': {
            'period_type': config.get('period_type', 'hourly'),
            'g_estimation_method': config.get('g_estimation_method', 'isotonic'),
            'selected_days': config.get('selected_days', [0,1,2,3,4,5,6]),
            'include_zero_supply': config.get('include_zero_supply', False),
            'freeze_g_function': config.get('freeze_g_function', True),
            'neighborhood_size': config.get('neighborhood_size', 5),
        }
    }
    
    # =========================================================================
    # COMPUTE DEMAND COUNTS
    # =========================================================================
    
    if use_hard_counts:
        # FAST: Use hard (discrete) cell assignment - O(n) time, O(grid) memory
        demand_counts = torch.zeros(config['grid_dims'], dtype=torch.float32)
        
        # Compute cell indices
        cells = pickup_coords.floor().long()
        cells = cells.clamp(
            min=torch.tensor([0, 0]),
            max=torch.tensor([config['grid_dims'][0]-1, config['grid_dims'][1]-1])
        )
        
        # Count occurrences per cell (vectorized)
        for i in range(n_trajectories):
            cx, cy = cells[i, 0].item(), cells[i, 1].item()
            demand_counts[cx, cy] += 1.0
        
        results['count_method'] = 'hard'
    else:
        # SLOW: Use soft cell assignment - O(n * k²) time, more memory
        from soft_cell_assignment import SoftCellAssignment
        
        soft_assign = SoftCellAssignment(
            grid_dims=config['grid_dims'],
            initial_temperature=config['temperature'],
        )
        
        demand_counts = torch.zeros(config['grid_dims'], dtype=torch.float32)
        k = soft_assign.k
        
        # Process in batches to reduce memory pressure
        batch_size = 1000
        for batch_start in range(0, n_trajectories, batch_size):
            batch_end = min(batch_start + batch_size, n_trajectories)
            
            for i in range(batch_start, batch_end):
                loc = pickup_coords[i:i+1].float()
                cell = loc.floor().long().clamp(
                    min=torch.tensor([0, 0]),
                    max=torch.tensor([config['grid_dims'][0]-1, config['grid_dims'][1]-1])
                )
                
                probs = soft_assign(loc, cell.float())
                cx, cy = cell[0, 0].item(), cell[0, 1].item()
                
                for di in range(-k, k+1):
                    for dj in range(-k, k+1):
                        ni, nj = int(cx + di), int(cy + dj)
                        if 0 <= ni < config['grid_dims'][0] and 0 <= nj < config['grid_dims'][1]:
                            demand_counts[ni, nj] = demand_counts[ni, nj] + probs[0, di + k, dj + k].item()
        
        results['count_method'] = 'soft'
    
    # =========================================================================
    # SUPPLY AND g(d) CREATION
    # =========================================================================
    
    if use_real_supply_demand:
        # Load real supply/demand data with full pipeline
        supply_demand_result = _load_supply_demand_data(config)
        
        if supply_demand_result.get('error'):
            results['supply_demand_error'] = supply_demand_result.get('diagnostics', {}).get('error', 'Unknown error')
            results['passed'] = False
            # Fall back to synthetic supply
            use_real_supply_demand = False
        else:
            results['supply_demand_diagnostics'] = supply_demand_result['diagnostics']
            
            # Create frozen g(d) function
            g_function, g_diagnostics = _create_frozen_g_function(supply_demand_result, config)
            results['g_function_diagnostics'] = g_diagnostics
            
            # Create supply tensor from real data
            # Sum across all periods to get total supply per cell
            supply_tensor = _create_supply_tensor_from_data(
                supply_demand_result['supply'],
                config['grid_dims'],
                period=None,  # Sum all periods
            )
            # No scaling - use actual supply values as-is
    
    if not use_real_supply_demand:
        # Create synthetic supply with known relationship to demand
        baseline_supply = 3.0
        supply_factor = 1.5
        supply_tensor = demand_counts * supply_factor + baseline_supply
        
        # Add small noise
        torch.manual_seed(42)
        noise = torch.randn_like(supply_tensor) * 0.1 * supply_tensor.mean()
        supply_tensor = supply_tensor + noise
        supply_tensor = torch.clamp(supply_tensor, min=0.1)
        
        # Create corresponding g(d) function
        def g_function(d):
            return supply_factor + baseline_supply / (np.array(d) + 1e-8)
        
        results['supply_type'] = 'synthetic'
        results['g_function_type'] = 'synthetic_inverse'
    else:
        results['supply_type'] = 'real_active_taxis'
        results['g_function_type'] = f"real_{config.get('g_estimation_method', 'isotonic')}"
    
    # =========================================================================
    # CREATE AND RUN MODULE
    # =========================================================================
    
    # Create module with all terms enabled
    module = DifferentiableFAMAILObjective(
        alpha_spatial=config['alpha_spatial'],
        alpha_causal=config['alpha_causal'],
        alpha_fidelity=config['alpha_fidelity'],
        grid_dims=config['grid_dims'],
        temperature=config['temperature'],
        g_function=g_function,
    )
    
    # Clone and require gradients
    pickup_test = pickup_coords.clone().detach().requires_grad_(True)
    dropoff_test = dropoff_coords.clone().detach().requires_grad_(True)
    
    # Forward pass with supply
    total, terms = module(pickup_test, dropoff_test, supply_tensor=supply_tensor)
    
    # Backward pass
    total.backward()
    
    # =========================================================================
    # GATHER RESULTS
    # =========================================================================
    
    results.update({
        # Objective values
        'total_objective': total.item(),
        'f_spatial': terms['f_spatial'].item(),
        'f_causal': terms['f_causal'].item(),
        'f_fidelity': terms['f_fidelity'].item(),
        
        # Weights used
        'alpha_spatial': config['alpha_spatial'],
        'alpha_causal': config['alpha_causal'],
        'alpha_fidelity': config['alpha_fidelity'],
        
        # Supply/demand statistics
        'supply_stats': {
            'mean': float(supply_tensor.mean()),
            'std': float(supply_tensor.std()),
            'min': float(supply_tensor.min()),
            'max': float(supply_tensor.max()),
            'nonzero_cells': int((supply_tensor > 0).sum()),
        },
        'demand_stats': {
            'mean': float(demand_counts.mean()),
            'std': float(demand_counts.std()),
            'min': float(demand_counts.min()),
            'max': float(demand_counts.max()),
            'nonzero_cells': int((demand_counts > 0).sum()),
        },
    })
    
    # Add causal debug info if available
    if hasattr(module, '_last_causal_debug'):
        results['causal_debug'] = module._last_causal_debug
    
    # Check and record pickup gradients
    if pickup_test.grad is None:
        results['passed'] = False
        results['pickup_grad_error'] = 'No gradient computed'
    elif torch.isnan(pickup_test.grad).any():
        results['passed'] = False
        results['pickup_grad_error'] = 'NaN in gradients'
    elif torch.isinf(pickup_test.grad).any():
        results['passed'] = False
        results['pickup_grad_error'] = 'Inf in gradients'
    else:
        results['pickup_grad_stats'] = {
            'mean': pickup_test.grad.mean().item(),
            'std': pickup_test.grad.std().item(),
            'min': pickup_test.grad.min().item(),
            'max': pickup_test.grad.max().item(),
            'nonzero_count': int((pickup_test.grad.abs() > 1e-10).sum().item()),
            'total_count': pickup_test.grad.numel(),
        }
    
    # Check and record dropoff gradients
    if dropoff_test.grad is None:
        results['passed'] = False
        results['dropoff_grad_error'] = 'No gradient computed'
    elif torch.isnan(dropoff_test.grad).any():
        results['passed'] = False
        results['dropoff_grad_error'] = 'NaN in gradients'
    elif torch.isinf(dropoff_test.grad).any():
        results['passed'] = False
        results['dropoff_grad_error'] = 'Inf in gradients'
    else:
        results['dropoff_grad_stats'] = {
            'mean': dropoff_test.grad.mean().item(),
            'std': dropoff_test.grad.std().item(),
            'min': dropoff_test.grad.min().item(),
            'max': dropoff_test.grad.max().item(),
            'nonzero_count': int((dropoff_test.grad.abs() > 1e-10).sum().item()),
            'total_count': dropoff_test.grad.numel(),
        }
    
    # Add metadata if using real data
    if metadata:
        results['data_metadata'] = metadata
    
    return results


def render_integration_tab(config: Dict[str, Any]):
    """Render the Integration Testing tab."""
    st.header("🧪 Integration Testing")
    
    st.markdown("""
    Test the complete pipeline from trajectory coordinates through objective computation
    to trajectory selection. Tests can use **synthetic** or **real** data sources.
    """)
    
    if not TORCH_AVAILABLE:
        st.error("PyTorch is required for integration testing")
        return
    
    import torch
    
    # =========================================================================
    # DATA SOURCE SELECTION
    # =========================================================================
    st.subheader("📊 Data Source Selection")
    
    col_ds1, col_ds2 = st.columns([2, 1])
    with col_ds1:
        trajectory_source = st.radio(
            "Trajectory Data Source",
            ["Generated (Synthetic)", "Real Data (all_trajs.pkl)"],
            horizontal=True,
            help="Generated creates random trajectories. Real loads from all_trajs.pkl."
        )
    
    with col_ds2:
        # Trajectory count options
        trajectory_count_mode = st.selectbox(
            "Trajectory Count",
            ["Sample (5,000)", "Sample (10,000)", "Sample (20,000)", "All (~44K, slow)"],
            index=0,
            help="More trajectories = more accurate but slower. 5-10K recommended for quick testing."
        )
        
        # Map selection to actual count
        count_map = {
            "Sample (5,000)": 5000,
            "Sample (10,000)": 10000,
            "Sample (20,000)": 20000,
            "All (~44K, slow)": None,
        }
        n_test_samples = count_map[trajectory_count_mode]
    
    use_real_trajectories = trajectory_source == "Real Data (all_trajs.pkl)"
    
    # Memory efficiency options
    col_mem1, col_mem2 = st.columns(2)
    with col_mem1:
        use_hard_counts = st.checkbox(
            "Use hard cell counts during initialization (faster)",
            value=True,
            help="Use discrete cell assignment instead of soft during tensor initialization. Much faster for large datasets while maintaining statistical validity."
        )
    
    # Supply/Demand data source
    col_sd1, col_sd2 = st.columns([2, 1])
    with col_sd1:
        supply_source = st.radio(
            "Supply/Demand Data Source",
            ["Synthetic (Generated)", "Real Data (active_taxis + pickup_dropoff_counts)"],
            horizontal=True,
            help="""
            **Synthetic**: Creates supply with known relationship to demand for testing.
            **Real Data**: Loads actual active_taxis and pickup_dropoff_counts files
            with full processing pipeline (period aggregation, day filtering, g(d) estimation).
            """
        )
    
    use_real_supply_demand = supply_source == "Real Data (active_taxis + pickup_dropoff_counts)"
    
    # =========================================================================
    # REALISTIC TEST CONFIGURATION (EXPANDER)
    # =========================================================================
    with st.expander("⚙️ Realistic Test Configuration", expanded=use_real_supply_demand):
        st.markdown("""
        **Configure data processing options for realistic tests.**
        These settings match how the optimization will process data.
        """)
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        
        with col_cfg1:
            st.markdown("**Temporal Settings**")
            st.info(f"Period Type: **{config.get('period_type', 'hourly')}**")
            st.info(f"Days: **{config.get('selected_days', [0,1,2,3,4,5,6])}**")
            
        with col_cfg2:
            st.markdown("**g(d) Estimation**")
            st.info(f"Method: **{config.get('g_estimation_method', 'isotonic')}**")
            st.info(f"Frozen: **{'Yes' if config.get('freeze_g_function', True) else 'No'}**")
            if config.get('g_estimation_method') == 'binning':
                st.info(f"Bins: **{config.get('n_bins', 10)}**")
        
        with col_cfg3:
            st.markdown("**Data Filtering**")
            st.info(f"Zero Supply: **{'Include' if config.get('include_zero_supply', False) else 'Exclude'}**")
            st.info(f"Zero Demand: **{'Include' if config.get('include_zero_demand', False) else 'Exclude'}**")
            st.info(f"Min Demand: **{config.get('min_demand_threshold', 0)}**")
        
        st.markdown("---")
        st.markdown("**Soft Cell Assignment**")
        col_sca1, col_sca2, col_sca3 = st.columns(3)
        with col_sca1:
            st.info(f"Temperature τ: **{config.get('temperature', 1.0):.2f}**")
        with col_sca2:
            ns = config.get('neighborhood_size', 5)
            st.info(f"Neighborhood: **{ns}×{ns}** window")
        with col_sca3:
            if config.get('use_annealing', False):
                st.info(f"Annealing: **{config.get('annealing_schedule', 'exponential')}** to τ={config.get('final_temperature', 0.1):.2f}")
            else:
                st.info("Annealing: **Disabled**")
        
        st.caption("💡 Modify these settings in the sidebar.")
    
    # Pre-load trajectory data based on selection
    @st.cache_data
    def get_test_data(use_real: bool, n_samples: Optional[int], grid_dims: Tuple[int, int], seed: int = 42):
        if use_real:
            return _load_real_data_for_testing({'grid_dims': grid_dims}, n_samples=n_samples)
        else:
            # Generate synthetic data - need a concrete number
            actual_n = n_samples if n_samples is not None else 5000
            torch.manual_seed(seed)
            pickup_coords = torch.rand(actual_n, 2) * torch.tensor(grid_dims, dtype=torch.float32)
            dropoff_coords = torch.rand(actual_n, 2) * torch.tensor(grid_dims, dtype=torch.float32)
            return {
                'trajectories': None,
                'pickup_coords': pickup_coords,
                'dropoff_coords': dropoff_coords,
                'metadata': {
                    'n_trajectories': actual_n,
                    'data_type': 'synthetic',
                    'grid_dims': grid_dims,
                    'seed': seed,
                }
            }
    
    # Test scenarios
    st.markdown("---")
    st.subheader("📝 Test Scenarios")
    
    scenario = st.selectbox(
        "Select test scenario",
        [
            "Basic Gradient Flow",
            "Temperature Annealing",
            "Attribution Consistency",
            "Full Pipeline",
        ]
    )
    
    # =========================================================================
    # BASIC GRADIENT FLOW TEST
    # =========================================================================
    if scenario == "Basic Gradient Flow":
        st.markdown("""
        ### 🔬 Basic Gradient Flow Test
        
        **Purpose**: Verify that gradients flow correctly from the combined objective function 
        back to the trajectory coordinates. This is the fundamental requirement for gradient-based optimization.
        
        **What this test does**:
        1. Creates trajectory pickup/dropoff coordinates (from selected data source)
        2. Computes the **combined objective function** $L = α_{causal} · F_{causal} + α_{spatial} · F_{spatial} + α_{fidelity} · F_{fidelity}$
        3. Performs backpropagation via `L.backward()`
        4. Verifies gradients exist for all coordinates and contain no NaN/Inf values
        
        **Expected Results**:
        - All gradients should be non-null and contain valid floating-point values
        - Nonzero gradient count indicates how many coordinate components received gradient signal
        - Gradient magnitude (mean, std) indicates learning signal strength
        """)
        
        # Show data source summary
        st.info(f"📂 Trajectory source: **{trajectory_source}** | Supply/Demand source: **{supply_source}**")
        
        if st.button("🚀 Run Basic Gradient Test", key="run_basic_grad"):
            with st.spinner("Running comprehensive gradient test..."):
                # Load data
                test_data = get_test_data(use_real_trajectories, n_test_samples, config['grid_dims'])
                
                if test_data is None:
                    st.error("❌ Could not load test data. Ensure all_trajs.pkl exists.")
                    return
                
                # Run comprehensive test
                result = _run_comprehensive_gradient_test(
                    config=config,
                    pickup_coords=test_data['pickup_coords'],
                    dropoff_coords=test_data['dropoff_coords'],
                    use_real_data=use_real_trajectories,
                    metadata=test_data.get('metadata'),
                    use_real_supply_demand=use_real_supply_demand,
                    use_hard_counts=use_hard_counts,
                )
            
            # Display pass/fail status
            if result['passed']:
                st.success("✅ Basic gradient test PASSED! All gradients computed successfully.")
            else:
                st.error("❌ Basic gradient test FAILED. See details below.")
            
            # Data source info
            st.markdown("#### 📂 Data Sources Used")
            col_src1, col_src2 = st.columns(2)
            with col_src1:
                if use_real_trajectories and 'data_metadata' in result:
                    st.info(f"**Trajectories**: Real data - {result['data_metadata'].get('n_trajectories', 'N/A')} trajectories from {result['data_metadata'].get('n_drivers', 'N/A')} drivers")
                else:
                    st.info(f"**Trajectories**: Synthetic - {result['n_trajectories']} randomly generated")
            
            with col_src2:
                if result.get('use_real_supply_demand'):
                    diag = result.get('supply_demand_diagnostics', {})
                    st.info(f"**Supply/Demand**: Real active_taxis data\n- Period: {diag.get('period_type', 'hourly')}\n- g(d) R²: {diag.get('g_r_squared', 0):.3f}")
                else:
                    st.info(f"**Supply/Demand**: Synthetic\n- Type: {result.get('g_function_type', 'synthetic')}")
            
            # Show supply/demand diagnostics if using real data
            if result.get('use_real_supply_demand') and 'supply_demand_diagnostics' in result:
                with st.expander("📈 Supply/Demand Data Diagnostics", expanded=False):
                    diag = result['supply_demand_diagnostics']
                    col_d1, col_d2, col_d3 = st.columns(3)
                    with col_d1:
                        st.markdown("**Data Loading**")
                        st.write(f"Pickup/Dropoff entries: {diag.get('n_pickup_dropoff_entries', 'N/A'):,}")
                        st.write(f"Active taxis entries: {diag.get('n_active_taxis_entries', 'N/A'):,}")
                        st.write(f"After day filter: {diag.get('n_demand_after_filter', 'N/A'):,}")
                    with col_d2:
                        st.markdown("**Data Range**")
                        d_range = diag.get('demand_range', [0, 0])
                        r_range = diag.get('ratio_range', [0, 0])
                        st.write(f"Demand range: [{d_range[0]:.0f}, {d_range[1]:.0f}]")
                        st.write(f"Ratio range: [{r_range[0]:.2f}, {r_range[1]:.2f}]")
                        st.write(f"Common entries: {diag.get('n_common_entries', 'N/A'):,}")
                    with col_d3:
                        st.markdown("**g(d) Estimation**")
                        g_diag = diag.get('g_estimation', {})
                        st.write(f"Method: {g_diag.get('method', 'N/A')}")
                        st.write(f"Frozen: {diag.get('frozen', False)}")
                        st.write(f"R²: {diag.get('g_r_squared', 0):.4f}")
            
            # Objective Function Values
            st.markdown("#### 📊 Objective Function Values")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Total Objective (L)", 
                f"{result['total_objective']:.4f}",
                help="Combined weighted sum of all fairness terms"
            )
            col2.metric(
                "F_spatial", 
                f"{result['f_spatial']:.4f}",
                help="Spatial fairness term (1 - Gini)"
            )
            col3.metric(
                "F_causal", 
                f"{result['f_causal']:.4f}",
                help="Causal fairness term (R² score)"
            )
            col4.metric(
                "F_fidelity", 
                f"{result['f_fidelity']:.4f}",
                help="Trajectory fidelity term"
            )
            
            # Causal Debug Info (if f_causal is 0 or unexpected)
            if 'causal_debug' in result:
                debug = result['causal_debug']
                
                # Show warning if f_causal is 0
                if result['f_causal'] == 0:
                    st.warning("⚠️ **F_causal = 0**: This indicates R² ≤ 0 (the g(d) function doesn't explain variance)")
                
                with st.expander("🔍 Causal Term Debug Info (click to expand)"):
                    st.markdown(f"""
                    **Causal Fairness Computation Details:**
                    
                    The causal fairness term is computed as: $F_{{causal}} = \\max(0, R^2)$ where $R^2 = 1 - \\frac{{Var(R)}}{{Var(Y)}}$
                    
                    - **Active cells** (demand > 0.1): {debug.get('n_active_cells', 'N/A')}
                    - **Demand range**: [{debug.get('demand_range', ('?', '?'))[0]:.4f}, {debug.get('demand_range', ('?', '?'))[1]:.4f}]
                    - **Supply range**: [{debug.get('supply_range', ('?', '?'))[0]:.4f}, {debug.get('supply_range', ('?', '?'))[1]:.4f}]
                    - **Y = S/D range**: [{debug.get('Y_range', ('?', '?'))[0]:.4f}, {debug.get('Y_range', ('?', '?'))[1]:.4f}]
                    - **g(D) range**: [{debug.get('g_d_range', ('?', '?'))[0]:.4f}, {debug.get('g_d_range', ('?', '?'))[1]:.4f}]
                    - **Var(Y)**: {debug.get('var_Y', 'N/A'):.6f}
                    - **Var(R) = Var(Y - g(D))**: {debug.get('var_R', 'N/A'):.6f}
                    - **R² (raw, before clamp)**: {debug.get('r_squared_raw', 'N/A'):.6f}
                    
                    **Interpretation:**
                    - If Var(R) ≈ Var(Y), then g(d) explains ~0% of variance → R² ≈ 0
                    - If Var(R) > Var(Y), then g(d) adds noise → R² < 0 (clamped to 0)
                    - The g(d) function used is: g(d) = 0.8 × d^(-0.2)
                    """)
            
            # Gradient Statistics
            st.markdown("#### 📈 Gradient Statistics")
            
            col_p, col_d = st.columns(2)
            
            with col_p:
                st.markdown("**Pickup Coordinate Gradients**")
                if 'pickup_grad_error' in result:
                    st.error(f"Error: {result['pickup_grad_error']}")
                else:
                    stats = result['pickup_grad_stats']
                    st.markdown(f"""
                    | Metric | Value |
                    |--------|-------|
                    | Mean | {stats['mean']:.6e} |
                    | Std | {stats['std']:.6e} |
                    | Min | {stats['min']:.6e} |
                    | Max | {stats['max']:.6e} |
                    | Nonzero | {stats['nonzero_count']}/{stats['total_count']} ({100*stats['nonzero_count']/stats['total_count']:.1f}%) |
                    """)
            
            with col_d:
                st.markdown("**Dropoff Coordinate Gradients**")
                if 'dropoff_grad_error' in result:
                    st.error(f"Error: {result['dropoff_grad_error']}")
                else:
                    stats = result['dropoff_grad_stats']
                    st.markdown(f"""
                    | Metric | Value |
                    |--------|-------|
                    | Mean | {stats['mean']:.6e} |
                    | Std | {stats['std']:.6e} |
                    | Min | {stats['min']:.6e} |
                    | Max | {stats['max']:.6e} |
                    | Nonzero | {stats['nonzero_count']}/{stats['total_count']} ({100*stats['nonzero_count']/stats['total_count']:.1f}%) |
                    """)
            
            # Visualization
            st.markdown("#### 📊 Gradient Comparison Visualization")
            
            if PLOTLY_AVAILABLE and 'pickup_grad_stats' in result and 'dropoff_grad_stats' in result:
                # Create two separate figures since pie charts need domain type, not xy
                col_viz1, col_viz2 = st.columns(2)
                
                # Bar chart comparing gradient stats
                p_stats = result['pickup_grad_stats']
                d_stats = result['dropoff_grad_stats']
                
                with col_viz1:
                    metrics = ['|mean|', 'std']
                    p_values = [abs(p_stats['mean']), p_stats['std']]
                    d_values = [abs(d_stats['mean']), d_stats['std']]
                    
                    fig_bar = go.Figure()
                    fig_bar.add_trace(
                        go.Bar(name='Pickup', x=metrics, y=p_values, marker_color='#1f77b4')
                    )
                    fig_bar.add_trace(
                        go.Bar(name='Dropoff', x=metrics, y=d_values, marker_color='#ff7f0e')
                    )
                    
                    fig_bar.update_layout(
                        title="Gradient Magnitude Comparison",
                        height=300,
                        barmode='group',
                        showlegend=True,
                        yaxis_type="log",
                        yaxis_title="Magnitude (log scale)",
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col_viz2:
                    # Pie chart for nonzero coverage
                    total_nonzero = p_stats['nonzero_count'] + d_stats['nonzero_count']
                    total_zero = (p_stats['total_count'] - p_stats['nonzero_count']) + (d_stats['total_count'] - d_stats['nonzero_count'])
                    
                    fig_pie = go.Figure()
                    fig_pie.add_trace(
                        go.Pie(
                            labels=['Nonzero Gradients', 'Zero Gradients'],
                            values=[total_nonzero, total_zero],
                            marker_colors=['#2ca02c', '#d62728'],
                            hole=0.4
                        )
                    )
                    
                    fig_pie.update_layout(
                        title="Gradient Coverage",
                        height=300,
                        showlegend=True,
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Interpretation
            st.markdown("#### 📖 Interpretation")
            
            interpretation_lines = []
            
            if result['passed']:
                interpretation_lines.append("✅ **Gradient flow is healthy**: All coordinate gradients computed successfully without NaN/Inf values.")
            
            if 'pickup_grad_stats' in result:
                p_nonzero_pct = 100 * result['pickup_grad_stats']['nonzero_count'] / result['pickup_grad_stats']['total_count']
                if p_nonzero_pct > 90:
                    interpretation_lines.append(f"✅ **Excellent gradient coverage**: {p_nonzero_pct:.1f}% of pickup coordinates received gradient signal.")
                elif p_nonzero_pct > 50:
                    interpretation_lines.append(f"⚠️ **Partial gradient coverage**: Only {p_nonzero_pct:.1f}% of pickup coordinates received gradient signal. Consider lowering temperature.")
                else:
                    interpretation_lines.append(f"❌ **Poor gradient coverage**: Only {p_nonzero_pct:.1f}% of pickup coordinates received gradient signal. Temperature may be too low.")
            
            # Check gradient balance
            if 'pickup_grad_stats' in result and 'dropoff_grad_stats' in result:
                p_mag = abs(result['pickup_grad_stats']['mean'])
                d_mag = abs(result['dropoff_grad_stats']['mean'])
                
                if max(p_mag, d_mag) > 0:
                    ratio = max(p_mag, d_mag) / (min(p_mag, d_mag) + 1e-10)
                    if ratio > 10:
                        interpretation_lines.append(f"⚠️ **Gradient imbalance**: Pickup/dropoff gradient magnitudes differ by {ratio:.1f}x. This may cause uneven optimization.")
                    else:
                        interpretation_lines.append(f"✅ **Balanced gradients**: Pickup/dropoff gradient magnitudes are within {ratio:.1f}x of each other.")
            
            for line in interpretation_lines:
                st.markdown(line)
            
            # Raw JSON output (collapsible)
            with st.expander("📄 View Raw JSON Results"):
                st.json(result)
    
    # =========================================================================
    # TEMPERATURE ANNEALING TEST
    # =========================================================================
    elif scenario == "Temperature Annealing":
        st.markdown("""
        ### 🌡️ Temperature Annealing Test
        
        **Purpose**: Verify that gradient flow remains valid across a temperature annealing schedule.
        Temperature controls the "softness" of cell assignments:
        - **High τ (e.g., 1.0)**: Gradients spread across many neighboring cells (smoother but less precise)
        - **Low τ (e.g., 0.1)**: Gradients concentrate on fewer cells (sharper but may cause instability)
        
        **What this test does**:
        1. Tests gradient flow at multiple temperature values
        2. Records objective values and gradient statistics at each temperature
        3. Checks for gradient validity (no NaN/Inf) at all temperatures
        
        **Expected Results**:
        - Gradients should remain valid at all temperatures
        - Gradient magnitude typically increases as temperature decreases
        - Objective values may shift slightly with temperature
        """)
        
        # Temperature schedule configuration
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            temp_schedule = st.multiselect(
                "Temperature schedule",
                [2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.1, 0.05],
                default=[1.0, 0.5, 0.25, 0.1],
                help="Select temperature values to test"
            )
        with col_t2:
            st.info(f"Testing {len(temp_schedule)} temperature levels")
        
        if st.button("🚀 Run Annealing Test", key="run_annealing"):
            if not temp_schedule:
                st.error("Please select at least one temperature value")
                return
            
            temps = sorted(temp_schedule, reverse=True)
            results = []
            
            # Load data once
            test_data = get_test_data(use_real_trajectories, n_test_samples, config['grid_dims'])
            
            if test_data is None:
                st.error("❌ Could not load test data.")
                return
            
            progress = st.progress(0)
            status = st.empty()
            
            for i, t in enumerate(temps):
                status.text(f"Testing τ = {t}...")
                
                # Create config copy with this temperature
                temp_config = config.copy()
                temp_config['temperature'] = t
                
                result = _run_comprehensive_gradient_test(
                    config=temp_config,
                    pickup_coords=test_data['pickup_coords'].clone(),
                    dropoff_coords=test_data['dropoff_coords'].clone(),
                    use_real_data=use_real_trajectories,
                    use_real_supply_demand=use_real_supply_demand,
                    use_hard_counts=use_hard_counts,
                )
                
                results.append({
                    'temperature': t,
                    'passed': result['passed'],
                    'total_objective': result['total_objective'],
                    'f_spatial': result['f_spatial'],
                    'f_causal': result['f_causal'],
                    'pickup_grad_mean': result.get('pickup_grad_stats', {}).get('mean', None),
                    'pickup_grad_std': result.get('pickup_grad_stats', {}).get('std', None),
                    'pickup_nonzero_pct': 100 * result.get('pickup_grad_stats', {}).get('nonzero_count', 0) / result.get('pickup_grad_stats', {}).get('total_count', 1) if 'pickup_grad_stats' in result else 0,
                })
                
                progress.progress((i + 1) / len(temps))
            
            status.text("Complete!")
            
            # Overall status
            all_passed = all(r['passed'] for r in results)
            
            if all_passed:
                st.success(f"✅ All {len(temps)} temperature levels passed!")
            else:
                failed_temps = [r['temperature'] for r in results if not r['passed']]
                st.error(f"❌ Failed at temperatures: {failed_temps}")
            
            # Results Table
            st.markdown("#### 📋 Results Table")
            
            table_data = []
            for r in results:
                table_data.append({
                    "τ (Temperature)": f"{r['temperature']:.2f}",
                    "Passed": "✓" if r['passed'] else "✗",
                    "Total Objective": f"{r['total_objective']:.4f}",
                    "F_spatial": f"{r['f_spatial']:.4f}",
                    "F_causal": f"{r['f_causal']:.4f}",
                    "∇ Mean": f"{r['pickup_grad_mean']:.2e}" if r['pickup_grad_mean'] else "N/A",
                    "∇ Coverage": f"{r['pickup_nonzero_pct']:.1f}%",
                })
            
            st.table(table_data)
            
            # Visualization
            st.markdown("#### 📊 Temperature Annealing Visualization")
            
            if PLOTLY_AVAILABLE and len(results) > 1:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        "Objective Values vs Temperature",
                        "Gradient Mean Magnitude vs Temperature",
                        "Gradient Std vs Temperature",
                        "Gradient Coverage vs Temperature"
                    )
                )
                
                temps_plot = [r['temperature'] for r in results]
                
                # Objective values
                fig.add_trace(
                    go.Scatter(x=temps_plot, y=[r['total_objective'] for r in results],
                              mode='lines+markers', name='Total Objective', line=dict(color='#1f77b4')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=temps_plot, y=[r['f_spatial'] for r in results],
                              mode='lines+markers', name='F_spatial', line=dict(color='#2ca02c', dash='dash')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=temps_plot, y=[r['f_causal'] for r in results],
                              mode='lines+markers', name='F_causal', line=dict(color='#ff7f0e', dash='dot')),
                    row=1, col=1
                )
                
                # Gradient mean magnitude
                grad_means = [abs(r['pickup_grad_mean']) if r['pickup_grad_mean'] else 0 for r in results]
                fig.add_trace(
                    go.Scatter(x=temps_plot, y=grad_means,
                              mode='lines+markers', name='|∇ Mean|', line=dict(color='#d62728')),
                    row=1, col=2
                )
                
                # Gradient std
                grad_stds = [r['pickup_grad_std'] if r['pickup_grad_std'] else 0 for r in results]
                fig.add_trace(
                    go.Scatter(x=temps_plot, y=grad_stds,
                              mode='lines+markers', name='∇ Std', line=dict(color='#9467bd')),
                    row=2, col=1
                )
                
                # Coverage
                fig.add_trace(
                    go.Scatter(x=temps_plot, y=[r['pickup_nonzero_pct'] for r in results],
                              mode='lines+markers', name='Coverage %', line=dict(color='#17becf'),
                              fill='tozeroy', fillcolor='rgba(23, 190, 207, 0.2)'),
                    row=2, col=2
                )
                
                fig.update_xaxes(title_text="Temperature (τ)", row=2, col=1)
                fig.update_xaxes(title_text="Temperature (τ)", row=2, col=2)
                fig.update_yaxes(title_text="Value", row=1, col=1)
                fig.update_yaxes(title_text="Magnitude", type="log", row=1, col=2)
                fig.update_yaxes(title_text="Std Dev", type="log", row=2, col=1)
                fig.update_yaxes(title_text="Coverage (%)", row=2, col=2)
                
                fig.update_layout(height=600, showlegend=True)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("#### 📖 Interpretation")
            
            if all_passed:
                st.markdown("✅ **All temperature levels produced valid gradients.** The soft cell assignment remains differentiable across the annealing schedule.")
            
            # Analyze trends
            if len(results) > 1:
                first_coverage = results[0]['pickup_nonzero_pct']
                last_coverage = results[-1]['pickup_nonzero_pct']
                
                if last_coverage < first_coverage * 0.5:
                    st.markdown(f"⚠️ **Coverage drops significantly**: From {first_coverage:.1f}% at τ={results[0]['temperature']} to {last_coverage:.1f}% at τ={results[-1]['temperature']}. Low temperatures may cause sparse gradients.")
                
                # Check for gradient explosion
                first_std = results[0].get('pickup_grad_std', 0) or 0
                last_std = results[-1].get('pickup_grad_std', 0) or 0
                
                if last_std > first_std * 100:
                    st.markdown(f"⚠️ **Gradient explosion risk**: Gradient std increased by {last_std/first_std:.0f}x at low temperature. Consider stopping annealing earlier.")
                else:
                    st.markdown("✅ **Gradient magnitude stable**: No signs of gradient explosion across the temperature schedule.")
    
    # =========================================================================
    # ATTRIBUTION CONSISTENCY TEST
    # =========================================================================
    elif scenario == "Attribution Consistency":
        st.markdown("""
        ### 🎯 Attribution Consistency Test
        
        **Purpose**: Verify that LIS (Local Inequality Score) and DCD (Demand-Conditional Deviation)
        attribution methods correctly identify cells that contribute most to unfairness.
        
        **What this test does**:
        1. Creates cell count data with known extreme values
        2. Computes LIS scores for all cells (spatial attribution)
        3. Computes DCD scores for all cells (causal attribution)
        4. Verifies that extreme cells have correspondingly high attribution scores
        
        **Expected Results**:
        - Cells with very high or very low counts should have high LIS scores
        - Cells with unusual supply/demand ratios should have high DCD scores
        - The attribution scores should align with the cell count distributions
        """)
        
        if st.button("🚀 Run Attribution Consistency Test", key="run_attr"):
            # Load data
            test_data = get_test_data(use_real_trajectories, n_test_samples, config['grid_dims'])
            
            if test_data is None:
                st.error("❌ Could not load test data.")
                return
            
            with st.spinner("Computing attribution scores..."):
                # Compute cell counts from coordinates
                if use_real_trajectories and test_data['trajectories']:
                    trajs = test_data['trajectories']
                    pickup_counts = np.zeros(config['grid_dims'])
                    dropoff_counts = np.zeros(config['grid_dims'])
                    
                    for traj in trajs:
                        px, py = traj['pickup_cell']
                        dx, dy = traj['dropoff_cell']
                        px, py = min(px, config['grid_dims'][0]-1), min(py, config['grid_dims'][1]-1)
                        dx, dy = min(dx, config['grid_dims'][0]-1), min(dy, config['grid_dims'][1]-1)
                        pickup_counts[px, py] += 1
                        dropoff_counts[dx, dy] += 1
                else:
                    # Generate counts from synthetic coordinates
                    pickup_counts = np.zeros(config['grid_dims'])
                    dropoff_counts = np.zeros(config['grid_dims'])
                    
                    p_coords = test_data['pickup_coords'].numpy()
                    d_coords = test_data['dropoff_coords'].numpy()
                    
                    for i in range(len(p_coords)):
                        px, py = int(p_coords[i, 0]), int(p_coords[i, 1])
                        dx, dy = int(d_coords[i, 0]), int(d_coords[i, 1])
                        px, py = min(px, config['grid_dims'][0]-1), min(py, config['grid_dims'][1]-1)
                        dx, dy = min(dx, config['grid_dims'][0]-1), min(dy, config['grid_dims'][1]-1)
                        pickup_counts[px, py] += 1
                        dropoff_counts[dx, dy] += 1
                
                # Compute LIS scores
                pickup_lis = compute_all_lis_scores(pickup_counts)
                dropoff_lis = compute_all_lis_scores(dropoff_counts)
                
                # Create supply data and compute DCD
                supply_counts = create_mock_supply_data(pickup_counts)
                
                def simple_g(d):
                    return np.ones_like(d) * np.mean(supply_counts / (pickup_counts + 1))
                
            # Display results
            st.markdown("#### 📊 Cell Count Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Pickup Counts**")
                st.markdown(f"""
                - Total pickups: {int(pickup_counts.sum())}
                - Active cells: {int((pickup_counts > 0).sum())}
                - Mean: {pickup_counts.mean():.2f}
                - Max: {pickup_counts.max():.0f}
                - Std: {pickup_counts.std():.2f}
                """)
            
            with col2:
                st.markdown("**Dropoff Counts**")
                st.markdown(f"""
                - Total dropoffs: {int(dropoff_counts.sum())}
                - Active cells: {int((dropoff_counts > 0).sum())}
                - Mean: {dropoff_counts.mean():.2f}
                - Max: {dropoff_counts.max():.0f}
                - Std: {dropoff_counts.std():.2f}
                """)
            
            # LIS validation
            st.markdown("#### 🎯 LIS Attribution Validation")
            
            # Find extreme cells
            max_pickup_cell = np.unravel_index(np.argmax(pickup_counts), pickup_counts.shape)
            max_pickup_lis = pickup_lis[max_pickup_cell]
            mean_lis = pickup_lis.mean()
            
            col_v1, col_v2, col_v3 = st.columns(3)
            col_v1.metric("Max Pickup Cell", f"({max_pickup_cell[0]}, {max_pickup_cell[1]})")
            col_v2.metric("LIS at Max Cell", f"{max_pickup_lis:.4f}")
            col_v3.metric("Mean LIS", f"{mean_lis:.4f}")
            
            if max_pickup_lis > mean_lis:
                st.success("✅ LIS correctly identifies the highest-count cell as having above-average inequality contribution")
            else:
                st.warning("⚠️ LIS attribution may not be working correctly")
            
            # Visualization: LIS Heatmap
            st.markdown("#### 🗺️ Attribution Heatmaps")
            
            if PLOTLY_AVAILABLE:
                # Create subplots for counts and LIS
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        "Pickup Counts", "Pickup LIS Scores",
                        "Dropoff Counts", "Dropoff LIS Scores"
                    ),
                    horizontal_spacing=0.1,
                    vertical_spacing=0.15,
                )
                
                # Use a subset of the grid for better visualization
                display_dims = (min(config['grid_dims'][0], 30), min(config['grid_dims'][1], 30))
                
                # Pickup counts heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=pickup_counts[:display_dims[0], :display_dims[1]],
                        colorscale='Blues',
                        name='Pickup Counts',
                        hovertemplate='Cell (%{x}, %{y})<br>Count: %{z}<extra></extra>',
                        showscale=True,
                        colorbar=dict(x=0.45, len=0.4, y=0.8),
                    ),
                    row=1, col=1
                )
                
                # Pickup LIS heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=pickup_lis[:display_dims[0], :display_dims[1]],
                        colorscale='Reds',
                        name='Pickup LIS',
                        hovertemplate='Cell (%{x}, %{y})<br>LIS: %{z:.4f}<extra></extra>',
                        showscale=True,
                        colorbar=dict(x=1.0, len=0.4, y=0.8),
                    ),
                    row=1, col=2
                )
                
                # Dropoff counts heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=dropoff_counts[:display_dims[0], :display_dims[1]],
                        colorscale='Greens',
                        name='Dropoff Counts',
                        hovertemplate='Cell (%{x}, %{y})<br>Count: %{z}<extra></extra>',
                        showscale=True,
                        colorbar=dict(x=0.45, len=0.4, y=0.25),
                    ),
                    row=2, col=1
                )
                
                # Dropoff LIS heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=dropoff_lis[:display_dims[0], :display_dims[1]],
                        colorscale='Oranges',
                        name='Dropoff LIS',
                        hovertemplate='Cell (%{x}, %{y})<br>LIS: %{z:.4f}<extra></extra>',
                        showscale=True,
                        colorbar=dict(x=1.0, len=0.4, y=0.25),
                    ),
                    row=2, col=2
                )
                
                fig.update_layout(
                    height=700,
                    title_text=f"Cell Attribution Analysis (showing {display_dims[0]}x{display_dims[1]} subset)",
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("#### 📖 Interpretation")
            
            st.markdown("""
            The heatmaps above show the relationship between cell counts (left) and LIS attribution scores (right):
            
            - **LIS (Local Inequality Score)** measures how much each cell deviates from the mean count
            - Cells that are **very high** or **very low** compared to the mean have **high LIS scores**
            - High LIS cells are candidates for trajectory modification to improve spatial fairness
            
            **Consistency Check**: The LIS heatmap should show high values (bright colors) at cells that have 
            extreme values (either very high or very low) in the counts heatmap.
            """)
            
            # Correlation analysis
            pickup_corr = np.corrcoef(np.abs(pickup_counts.flatten() - pickup_counts.mean()), pickup_lis.flatten())[0, 1]
            st.markdown(f"**Correlation between |count - mean| and LIS**: {pickup_corr:.4f}")
            
            if pickup_corr > 0.9:
                st.success("✅ Strong correlation confirms LIS is working correctly")
            elif pickup_corr > 0.7:
                st.info("ℹ️ Good correlation between count deviation and LIS")
            else:
                st.warning("⚠️ Weak correlation may indicate issues with LIS computation")
    
    # =========================================================================
    # FULL PIPELINE TEST
    # =========================================================================
    elif scenario == "Full Pipeline":
        st.markdown("""
        ### 🔧 Full Pipeline Test
        
        **Purpose**: Run the complete FAMAIL optimization pipeline end-to-end, testing all components 
        working together: data loading → soft cell assignment → objective computation → gradient flow → 
        attribution → trajectory selection.
        
        **Pipeline Steps**:
        1. **Data Loading**: Load/generate trajectory coordinates
        2. **Objective Module**: Create the differentiable FAMAIL objective
        3. **Forward Pass**: Compute all objective terms (spatial, causal, fidelity)
        4. **Backward Pass**: Compute gradients via backpropagation
        5. **Attribution**: Compute LIS and DCD scores for all cells
        6. **Selection**: Identify trajectories for modification
        
        **Expected Results**:
        - All steps complete without errors
        - All objective terms return valid values in [0, 1]
        - Gradients exist and contain no NaN/Inf values
        - Attribution scores identify high-priority trajectories
        """)
        
        if st.button("🚀 Run Full Pipeline Test", key="run_pipeline"):
            progress = st.progress(0)
            status = st.empty()
            step_results = {}
            
            try:
                # Step 1: Load/generate data
                status.text("Step 1/6: Loading trajectory data...")
                
                test_data = get_test_data(use_real_trajectories, n_test_samples, config['grid_dims'])
                
                if test_data is None:
                    st.error("❌ Could not load test data.")
                    return
                
                pickup_coords = test_data['pickup_coords'].clone()
                dropoff_coords = test_data['dropoff_coords'].clone()
                n_trajs = pickup_coords.shape[0]
                
                step_results['data'] = {
                    'status': 'success',
                    'n_trajectories': n_trajs,
                    'data_source': 'real' if use_real_trajectories else 'synthetic',
                    'supply_source': 'real' if use_real_supply_demand else 'synthetic',
                }
                
                if use_real_trajectories and test_data.get('metadata'):
                    step_results['data']['metadata'] = test_data['metadata']
                
                progress.progress(1/6)
                
                # Step 2: Create objective module
                status.text("Step 2/6: Creating objective module...")
                
                # Compute demand counts (hard or soft based on use_hard_counts setting)
                if use_hard_counts:
                    # Fast vectorized hard cell assignment
                    cell_indices = pickup_coords.floor().long()
                    cell_indices[:, 0] = cell_indices[:, 0].clamp(0, config['grid_dims'][0] - 1)
                    cell_indices[:, 1] = cell_indices[:, 1].clamp(0, config['grid_dims'][1] - 1)
                    
                    demand_counts = torch.zeros(config['grid_dims'], dtype=torch.float32)
                    for i in range(n_trajs):
                        x, y = cell_indices[i, 0].item(), cell_indices[i, 1].item()
                        demand_counts[x, y] += 1.0
                else:
                    # Slower soft cell assignment (memory intensive)
                    from soft_cell_assignment import SoftCellAssignment
                    
                    soft_assign = SoftCellAssignment(
                        grid_dims=config['grid_dims'],
                        initial_temperature=config['temperature'],
                    )
                    
                    demand_counts = torch.zeros(config['grid_dims'], dtype=torch.float32)
                    k = soft_assign.k
                    
                    # Process in batches to reduce memory pressure
                    batch_size = 1000
                    for batch_start in range(0, n_trajs, batch_size):
                        batch_end = min(batch_start + batch_size, n_trajs)
                        for i in range(batch_start, batch_end):
                            loc = pickup_coords[i:i+1].float()
                            cell = loc.floor().long().clamp(
                                min=torch.tensor([0, 0]),
                                max=torch.tensor([config['grid_dims'][0]-1, config['grid_dims'][1]-1])
                            )
                            
                            probs = soft_assign(loc, cell.float())
                            cx, cy = cell[0, 0].item(), cell[0, 1].item()
                            
                            for di in range(-k, k+1):
                                for dj in range(-k, k+1):
                                    ni, nj = int(cx + di), int(cy + dj)
                                    if 0 <= ni < config['grid_dims'][0] and 0 <= nj < config['grid_dims'][1]:
                                        demand_counts[ni, nj] = demand_counts[ni, nj] + probs[0, di + k, dj + k].item()
                
                # Create supply and g(d) based on data source selection
                if use_real_supply_demand:
                    # Load real supply/demand data
                    supply_demand_result = _load_supply_demand_data(config)
                    
                    if supply_demand_result.get('error'):
                        st.warning(f"⚠️ Could not load real supply data: {supply_demand_result.get('diagnostics', {}).get('error', 'Unknown')}. Falling back to synthetic.")
                        use_real_supply = False
                    else:
                        use_real_supply = True
                        step_results['supply_demand_diagnostics'] = supply_demand_result['diagnostics']
                        
                        # Create frozen g(d)
                        g_function, g_diag = _create_frozen_g_function(supply_demand_result, config)
                        step_results['g_function_diagnostics'] = g_diag
                        
                        # Create supply tensor from real data
                        supply_tensor = _create_supply_tensor_from_data(
                            supply_demand_result['supply'],
                            config['grid_dims'],
                        )
                        # No scaling - use actual supply values as-is
                else:
                    use_real_supply = False
                
                if not use_real_supply:
                    # Create synthetic supply following: S = factor * D + baseline
                    # This gives Y = S/D = factor + baseline/D
                    baseline_supply = 3.0
                    supply_factor = 1.5
                    supply_tensor = demand_counts * supply_factor + baseline_supply
                    
                    # Add small noise for realism
                    torch.manual_seed(42)
                    noise = torch.randn_like(supply_tensor) * 0.1 * supply_tensor.mean()
                    supply_tensor = supply_tensor + noise
                    supply_tensor = torch.clamp(supply_tensor, min=0.1)
                    
                    # g(d) function matching the supply relationship
                    def g_function(d):
                        return supply_factor + baseline_supply / (np.array(d) + 1e-8)
                
                module = DifferentiableFAMAILObjective(
                    alpha_spatial=config['alpha_spatial'],
                    alpha_causal=config['alpha_causal'],
                    alpha_fidelity=config['alpha_fidelity'],
                    grid_dims=config['grid_dims'],
                    temperature=config['temperature'],
                    g_function=g_function,
                )
                
                step_results['module'] = {
                    'status': 'success',
                    'alpha_spatial': config['alpha_spatial'],
                    'alpha_causal': config['alpha_causal'],
                    'alpha_fidelity': config['alpha_fidelity'],
                    'temperature': config['temperature'],
                }
                
                progress.progress(2/6)
                
                # Step 3: Forward pass
                status.text("Step 3/6: Computing objective (forward pass)...")
                
                pickup_coords.requires_grad_(True)
                dropoff_coords.requires_grad_(True)
                
                total, terms = module(pickup_coords, dropoff_coords, supply_tensor=supply_tensor)
                
                step_results['forward'] = {
                    'status': 'success',
                    'total_objective': total.item(),
                    'f_spatial': terms['f_spatial'].item(),
                    'f_causal': terms['f_causal'].item(),
                    'f_fidelity': terms['f_fidelity'].item(),
                }
                
                # Add causal debug info
                if hasattr(module, '_last_causal_debug'):
                    step_results['forward']['causal_debug'] = module._last_causal_debug
                
                progress.progress(3/6)
                
                # Step 4: Backward pass
                status.text("Step 4/6: Computing gradients (backward pass)...")
                
                total.backward()
                
                has_pickup_grad = pickup_coords.grad is not None
                has_dropoff_grad = dropoff_coords.grad is not None
                pickup_grad_valid = has_pickup_grad and not torch.isnan(pickup_coords.grad).any() and not torch.isinf(pickup_coords.grad).any()
                dropoff_grad_valid = has_dropoff_grad and not torch.isnan(dropoff_coords.grad).any() and not torch.isinf(dropoff_coords.grad).any()
                
                step_results['backward'] = {
                    'status': 'success' if (pickup_grad_valid and dropoff_grad_valid) else 'warning',
                    'pickup_grad_exists': has_pickup_grad,
                    'pickup_grad_valid': pickup_grad_valid,
                    'dropoff_grad_exists': has_dropoff_grad,
                    'dropoff_grad_valid': dropoff_grad_valid,
                }
                
                if pickup_grad_valid:
                    step_results['backward']['pickup_grad_stats'] = {
                        'mean': pickup_coords.grad.mean().item(),
                        'std': pickup_coords.grad.std().item(),
                        'min': pickup_coords.grad.min().item(),
                        'max': pickup_coords.grad.max().item(),
                        'nonzero_pct': 100 * (pickup_coords.grad.abs() > 1e-10).sum().item() / pickup_coords.grad.numel(),
                    }
                
                if dropoff_grad_valid:
                    step_results['backward']['dropoff_grad_stats'] = {
                        'mean': dropoff_coords.grad.mean().item(),
                        'std': dropoff_coords.grad.std().item(),
                        'min': dropoff_coords.grad.min().item(),
                        'max': dropoff_coords.grad.max().item(),
                        'nonzero_pct': 100 * (dropoff_coords.grad.abs() > 1e-10).sum().item() / dropoff_coords.grad.numel(),
                    }
                
                progress.progress(4/6)
                
                # Step 5: Attribution
                status.text("Step 5/6: Computing attribution scores...")
                
                pickup_np = pickup_coords.detach().numpy()
                dropoff_np = dropoff_coords.detach().numpy()
                
                pickup_counts = np.zeros(config['grid_dims'])
                dropoff_counts = np.zeros(config['grid_dims'])
                
                trajs = []
                for i in range(n_trajs):
                    pi = min(int(pickup_np[i, 0]), config['grid_dims'][0] - 1)
                    pj = min(int(pickup_np[i, 1]), config['grid_dims'][1] - 1)
                    di = min(int(dropoff_np[i, 0]), config['grid_dims'][0] - 1)
                    dj = min(int(dropoff_np[i, 1]), config['grid_dims'][1] - 1)
                    
                    pickup_counts[pi, pj] += 1
                    dropoff_counts[di, dj] += 1
                    
                    trajs.append({
                        'trajectory_id': i,
                        'pickup_cell': (pi, pj),
                        'dropoff_cell': (di, dj),
                    })
                
                supply_counts = create_mock_supply_data(pickup_counts)
                
                def simple_g(d):
                    return np.ones_like(d) * 0.8
                
                attr_result = compute_combined_attribution(
                    trajectories=trajs,
                    pickup_counts=pickup_counts,
                    dropoff_counts=dropoff_counts,
                    supply_counts=supply_counts,
                    g_function=simple_g,
                    lis_weight=config['lis_weight'],
                    dcd_weight=config['dcd_weight'],
                )
                
                step_results['attribution'] = {
                    'status': 'success',
                    'n_trajectories_scored': len(attr_result.trajectory_scores),
                    'lis_weight': config['lis_weight'],
                    'dcd_weight': config['dcd_weight'],
                    'mean_combined_score': np.mean([s.combined_score for s in attr_result.trajectory_scores]),
                    'max_combined_score': max(s.combined_score for s in attr_result.trajectory_scores),
                }
                
                progress.progress(5/6)
                
                # Step 6: Selection
                status.text("Step 6/6: Selecting trajectories for modification...")
                
                n_select = min(10, n_trajs // 10)
                selected = select_trajectories_for_modification(attr_result, n_trajectories=n_select)
                
                step_results['selection'] = {
                    'status': 'success',
                    'n_requested': n_select,
                    'n_selected': len(selected),
                    'selected_ids': [s.trajectory_id for s in selected[:5]],  # First 5
                    'top_scores': [s.combined_score for s in selected[:5]],
                }
                
                progress.progress(6/6)
                status.text("✅ Pipeline complete!")
                
                # Overall status
                all_success = all(r.get('status') == 'success' for r in step_results.values())
                
                if all_success:
                    st.success("✅ Full pipeline completed successfully!")
                else:
                    st.warning("⚠️ Pipeline completed with warnings")
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Pipeline Results")
                
                # Data source info
                st.markdown("#### Step 1: Data Loading")
                if use_real_trajectories and 'metadata' in step_results['data']:
                    meta = step_results['data']['metadata']
                    st.info(f"""
                    📁 **Real Trajectory Data Loaded**
                    - Source: `{meta.get('filepath', 'all_trajs.pkl')}`
                    - Trajectories: {meta.get('n_trajectories', 'N/A')}
                    - Unique Drivers: {meta.get('n_drivers', 'N/A')}
                    - Avg States/Trajectory: {meta.get('avg_states_per_traj', 0):.1f}
                    - Unique Pickup Cells: {meta.get('unique_pickup_cells', 'N/A')}
                    - Unique Dropoff Cells: {meta.get('unique_dropoff_cells', 'N/A')}
                    """)
                else:
                    st.info(f"🎲 **Synthetic Data Generated**: {step_results['data']['n_trajectories']} random trajectories")
                
                # Supply/Demand source info
                if use_real_supply_demand:
                    st.info(f"""
                    📈 **Real Supply/Demand Data**
                    - Supply Source: active_taxis
                    - Period Type: {config.get('period_type', 'hourly')}
                    - g(d) Method: {config.get('g_estimation_method', 'isotonic')}
                    """)
                else:
                    st.info("🎲 **Synthetic Supply/Demand**: Generated with known relationship")
                
                # Objective values
                st.markdown("#### Step 3: Objective Function Values")
                st.markdown("""
                The objective function combines three fairness terms into a single value to optimize.
                Each term ranges from 0 (worst) to 1 (best), and the total is their weighted sum.
                """)
                
                fwd = step_results['forward']
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    "Total Objective (L)",
                    f"{fwd['total_objective']:.4f}",
                    help="L = α_s·F_spatial + α_c·F_causal + α_f·F_fidelity. Higher is better (more fair)."
                )
                col2.metric(
                    "F_spatial",
                    f"{fwd['f_spatial']:.4f}",
                    help="Spatial fairness = 1 - Gini(pickup, dropoff). Measures geographic equity of service."
                )
                col3.metric(
                    "F_causal",
                    f"{fwd['f_causal']:.4f}",
                    help="Causal fairness = R² of Y = g(D) + ε. Measures how well service ratio follows demand."
                )
                col4.metric(
                    "F_fidelity",
                    f"{fwd['f_fidelity']:.4f}",
                    help="Trajectory fidelity. Measures similarity to original trajectories (default 0.5 if no discriminator)."
                )
                
                # Causal debug info if f_causal is 0
                if fwd['f_causal'] == 0 and 'causal_debug' in fwd:
                    debug = fwd['causal_debug']
                    st.warning("⚠️ **F_causal = 0**: The g(d) function doesn't explain variance in the service ratio Y = S/D")
                    
                    with st.expander("🔍 View Causal Term Debug Details"):
                        st.markdown(f"""
                        **Why is F_causal = 0?**
                        
                        F_causal = max(0, R²) where R² = 1 - Var(R)/Var(Y) and R = Y - g(D)
                        
                        | Metric | Value |
                        |--------|-------|
                        | Active cells | {debug.get('n_active_cells', 'N/A')} |
                        | Demand range | [{debug.get('demand_range', ('?', '?'))[0]:.4f}, {debug.get('demand_range', ('?', '?'))[1]:.4f}] |
                        | Supply range | [{debug.get('supply_range', ('?', '?'))[0]:.4f}, {debug.get('supply_range', ('?', '?'))[1]:.4f}] |
                        | Y = S/D range | [{debug.get('Y_range', ('?', '?'))[0]:.4f}, {debug.get('Y_range', ('?', '?'))[1]:.4f}] |
                        | g(D) range | [{debug.get('g_d_range', ('?', '?'))[0]:.4f}, {debug.get('g_d_range', ('?', '?'))[1]:.4f}] |
                        | Var(Y) | {debug.get('var_Y', 'N/A'):.6f} |
                        | Var(R) | {debug.get('var_R', 'N/A'):.6f} |
                        | R² (raw) | {debug.get('r_squared_raw', 'N/A'):.6f} |
                        
                        **Issue**: When Var(R) ≥ Var(Y), R² ≤ 0, which gets clamped to 0.
                        This means g(d) doesn't capture the relationship between demand and service ratio.
                        """)
                
                # Gradient results
                st.markdown("#### Step 4: Gradient Verification")
                st.markdown("""
                Gradients are the learning signal for optimization. We verify:
                - **Exists**: Gradient tensor was computed
                - **Valid**: No NaN or Inf values
                - **Coverage**: Percentage of coordinates receiving non-zero gradient
                """)
                
                bwd = step_results['backward']
                
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.markdown("**Pickup Gradients**")
                    status_icon = "✅" if bwd['pickup_grad_valid'] else "❌"
                    st.markdown(f"{status_icon} Valid: {bwd['pickup_grad_valid']}")
                    
                    if 'pickup_grad_stats' in bwd:
                        gs = bwd['pickup_grad_stats']
                        st.markdown(f"- Mean: {gs['mean']:.2e}")
                        st.markdown(f"- Std: {gs['std']:.2e}")
                        st.markdown(f"- Coverage: {gs['nonzero_pct']:.1f}%")
                
                with col_g2:
                    st.markdown("**Dropoff Gradients**")
                    status_icon = "✅" if bwd['dropoff_grad_valid'] else "❌"
                    st.markdown(f"{status_icon} Valid: {bwd['dropoff_grad_valid']}")
                    
                    if 'dropoff_grad_stats' in bwd:
                        gs = bwd['dropoff_grad_stats']
                        st.markdown(f"- Mean: {gs['mean']:.2e}")
                        st.markdown(f"- Std: {gs['std']:.2e}")
                        st.markdown(f"- Coverage: {gs['nonzero_pct']:.1f}%")
                
                # Attribution results
                st.markdown("#### Step 5: Attribution Scores")
                st.markdown("""
                Attribution identifies which trajectories contribute most to unfairness:
                - **LIS Score**: Spatial contribution (high count deviation)
                - **DCD Score**: Causal contribution (unusual service ratio)
                - **Combined**: Weighted sum used for selection
                """)
                
                attr = step_results['attribution']
                
                col_a1, col_a2, col_a3 = st.columns(3)
                col_a1.metric("Trajectories Scored", attr['n_trajectories_scored'])
                col_a2.metric("Mean Combined Score", f"{attr['mean_combined_score']:.4f}")
                col_a3.metric("Max Combined Score", f"{attr['max_combined_score']:.4f}")
                
                # Selection results
                st.markdown("#### Step 6: Trajectory Selection")
                st.markdown("""
                The top-scoring trajectories are selected for modification during optimization.
                Modifying these trajectories has the highest potential to improve fairness.
                """)
                
                sel = step_results['selection']
                
                st.markdown(f"**Selected {sel['n_selected']}/{sel['n_requested']} trajectories for modification**")
                
                if sel['selected_ids']:
                    st.markdown("Top 5 selected trajectories:")
                    sel_table = []
                    for tid, score in zip(sel['selected_ids'], sel['top_scores']):
                        sel_table.append({"Trajectory ID": tid, "Combined Score": f"{score:.4f}"})
                    st.table(sel_table)
                
                # Raw results (collapsible)
                with st.expander("📄 View Full Pipeline Results JSON"):
                    st.json(step_results)
                
            except Exception as e:
                st.error(f"❌ Pipeline failed at: {status.text}")
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_synthetic_data(
    n_trajectories: int,
    grid_dims: Tuple[int, int],
    cluster_factor: float = 0.5,
) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic trajectory data."""
    np.random.seed(42)
    
    # Create cluster centers
    n_clusters = 5
    cluster_centers_pickup = np.random.rand(n_clusters, 2) * np.array(grid_dims)
    cluster_centers_dropoff = np.random.rand(n_clusters, 2) * np.array(grid_dims)
    
    trajectories = []
    pickup_counts = np.zeros(grid_dims)
    dropoff_counts = np.zeros(grid_dims)
    
    for i in range(n_trajectories):
        if np.random.rand() < cluster_factor:
            # Cluster assignment
            cluster = np.random.randint(n_clusters)
            pickup = cluster_centers_pickup[cluster] + np.random.randn(2) * 3
            dropoff = cluster_centers_dropoff[cluster] + np.random.randn(2) * 3
        else:
            # Random assignment
            pickup = np.random.rand(2) * np.array(grid_dims)
            dropoff = np.random.rand(2) * np.array(grid_dims)
        
        # Clamp to grid
        pi, pj = int(pickup[0]) % grid_dims[0], int(pickup[1]) % grid_dims[1]
        di, dj = int(dropoff[0]) % grid_dims[0], int(dropoff[1]) % grid_dims[1]
        
        pickup_counts[pi, pj] += 1
        dropoff_counts[di, dj] += 1
        
        trajectories.append({
            'trajectory_id': i,
            'pickup_cell': (pi, pj),
            'dropoff_cell': (di, dj),
            'pickup_coords': pickup,
            'dropoff_coords': dropoff,
        })
    
    # Create supply based on demand
    supply_counts = create_mock_supply_data(pickup_counts)
    
    return trajectories, pickup_counts, dropoff_counts, supply_counts


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    # Render sidebar and get config
    config = render_sidebar()
    
    # Main content
    st.title("🎯 FAMAIL Integrated Objective Function Dashboard")
    
    st.markdown("""
    This dashboard helps you test, visualize, and understand the FAMAIL objective function 
    for trajectory optimization. Use the tabs below to explore different aspects of the system.
    """)
    
    # Create tabs
    tabs = st.tabs([
        "🔄 Gradient Flow",
        "🔲 Soft Cell Assignment", 
        "🎯 Attribution Methods",
        "🧪 Integration Testing",
    ])
    
    with tabs[0]:
        render_gradient_flow_tab(config)
    
    with tabs[1]:
        render_soft_cell_assignment_tab(config)
    
    with tabs[2]:
        render_attribution_tab(config)
    
    with tabs[3]:
        render_integration_tab(config)
    
    # Footer
    st.markdown("---")
    st.caption("FAMAIL Objective Function Dashboard v1.0")


if __name__ == "__main__":
    main()
