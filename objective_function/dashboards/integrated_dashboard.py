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
    """Render sidebar with global configuration."""
    st.sidebar.title("🎯 FAMAIL Dashboard")
    st.sidebar.markdown("---")
    
    st.sidebar.header("📐 Grid Configuration")
    grid_x = st.sidebar.number_input("Grid X", min_value=5, max_value=100, value=48)
    grid_y = st.sidebar.number_input("Grid Y", min_value=5, max_value=100, value=90)
    
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
    
    st.sidebar.header("🌡️ Temperature")
    temperature = st.sidebar.slider("Temperature τ", 0.01, 5.0, 1.0, 0.01)
    neighborhood_size = st.sidebar.slider("Neighborhood Size k", 1, 10, 5)
    
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Attribution Weights")
    lis_weight = st.sidebar.slider("LIS Weight (Spatial)", 0.0, 1.0, 0.5, 0.01)
    dcd_weight = st.sidebar.slider("DCD Weight (Causal)", 0.0, 1.0, 0.5, 0.01)
    
    return {
        'grid_dims': (grid_x, grid_y),
        'alpha_spatial': alpha_spatial,
        'alpha_causal': alpha_causal,
        'alpha_fidelity': alpha_fidelity,
        'temperature': temperature,
        'neighborhood_size': neighborhood_size,
        'lis_weight': lis_weight,
        'dcd_weight': dcd_weight,
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
    with st.expander("📐 Mathematical Formulation", expanded=True):
        st.markdown("""
        ### Combined Objective Function
        
        The FAMAIL objective combines three differentiable terms:
        """)
        
        st.latex(r"""
        \mathcal{L} = \alpha_1 \cdot F_{\text{causal}} + \alpha_2 \cdot F_{\text{spatial}} + \alpha_3 \cdot F_{\text{fidelity}}
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Spatial Fairness**")
            st.latex(r"F_{\text{spatial}} = 1 - G(\tilde{C})")
            st.markdown("Pairwise Gini coefficient")
        
        with col2:
            st.markdown("**Causal Fairness**")
            st.latex(r"F_{\text{causal}} = R^2 = 1 - \frac{\text{Var}(R)}{\text{Var}(Y)}")
            st.markdown("R² with frozen g(d)")
        
        with col3:
            st.markdown("**Fidelity**")
            st.latex(r"F_{\text{fidelity}} = D(\tau', \tau)")
            st.markdown("Discriminator similarity")
        
        st.markdown("---")
        st.markdown("### Current Weights")
        st.markdown(f"""
        - α_spatial = {config['alpha_spatial']:.3f}
        - α_causal = {config['alpha_causal']:.3f}
        - α_fidelity = {config['alpha_fidelity']:.3f}
        """)
    
    # Gradient Flow Diagram
    with st.expander("📊 Gradient Flow Diagram", expanded=True):
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
                else:
                    st.error(f"✗ {term_report.error_message}")
        
        # Combined objective
        st.markdown("---")
        st.metric("Total Objective L", f"{report.total_objective:.4f}")
    
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
                
                # Gradient plot
                fig.add_trace(
                    go.Scatter(x=results['temperatures'], y=np.abs(results['spatial_grad_means']), 
                              name='|∇F_spatial|', mode='lines+markers'),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Temperature τ", row=1, col=1)
                fig.update_xaxes(title_text="Temperature τ", row=1, col=2)
                fig.update_yaxes(title_text="Value", row=1, col=1)
                fig.update_yaxes(title_text="Gradient Magnitude", row=1, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
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
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Point Location**")
        loc_x = st.slider("X coordinate", 0.0, float(config['grid_dims'][0]-1), 
                         float(config['grid_dims'][0]//2) + 0.3, 0.1)
        loc_y = st.slider("Y coordinate", 0.0, float(config['grid_dims'][1]-1), 
                         float(config['grid_dims'][1]//2) + 0.7, 0.1)
        
        temp = st.slider("Temperature τ", 0.01, 5.0, config['temperature'], 0.01)
        k = config['neighborhood_size']
    
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
    with st.expander("📐 Attribution Formulas", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### LIS: Local Inequality Score")
            st.latex(r"LIS_i = \frac{|c_i - \mu|}{\mu}")
            st.markdown("""
            Measures how much cell $i$ deviates from the mean count.
            - High LIS → Cell contributes more to Gini inequality
            - Used for **Spatial Fairness** attribution
            """)
        
        with col2:
            st.markdown("### DCD: Demand-Conditional Deviation")
            st.latex(r"DCD_i = |Y_i - g(D_i)|")
            st.markdown("""
            Measures how much the service ratio deviates from expected.
            - $Y_i = S_i / D_i$ (actual service ratio)
            - $g(D_i)$ is the frozen baseline function
            - Used for **Causal Fairness** attribution
            """)
        
        st.markdown("---")
        st.markdown("### Combined Attribution")
        st.latex(r"Score_\tau = w_1 \cdot LIS_\tau + w_2 \cdot DCD_\tau")
        st.markdown(f"Current weights: w₁ (LIS) = {config['lis_weight']:.2f}, w₂ (DCD) = {config['dcd_weight']:.2f}")
    
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
        
            # LIS and DCD Heatmaps
            if PLOTLY_AVAILABLE:
                st.subheader("🗺️ Cell-Level Attribution Maps")
                
                pickup_lis = compute_all_lis_scores(pickup_counts)
                dropoff_lis = compute_all_lis_scores(dropoff_counts)
                dcd_scores = compute_all_dcd_scores(pickup_counts, supply_counts, g_function)
                
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=("LIS (Pickup)", "LIS (Dropoff)", "DCD"),
                )
                
                fig.add_trace(go.Heatmap(z=pickup_lis.T, colorscale='Reds'), row=1, col=1)
                fig.add_trace(go.Heatmap(z=dropoff_lis.T, colorscale='Reds'), row=1, col=2)
                fig.add_trace(go.Heatmap(z=dcd_scores.T, colorscale='Blues'), row=1, col=3)
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
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

def render_integration_tab(config: Dict[str, Any]):
    """Render the Integration Testing tab."""
    st.header("🧪 Integration Testing")
    
    st.markdown("""
    Test the complete pipeline from trajectory coordinates through objective computation
    to trajectory selection.
    """)
    
    if not TORCH_AVAILABLE:
        st.error("PyTorch is required for integration testing")
        return
    
    # Test scenarios
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
    
    if scenario == "Basic Gradient Flow":
        st.markdown("""
        **Test**: Verify gradients flow from objective to coordinates.
        
        This test creates random trajectories, computes the combined objective,
        performs backpropagation, and verifies all gradients are valid.
        """)
        
        if st.button("Run Basic Gradient Test"):
            with st.spinner("Running test..."):
                result = DifferentiableFAMAILObjective.verify_gradients(
                    grid_dims=config['grid_dims'],
                    n_pickups=50,
                    n_dropoffs=50,
                    temperature=config['temperature'],
                )
            
            if result['passed']:
                st.success("✅ Basic gradient test passed!")
            else:
                st.error("❌ Basic gradient test failed")
            
            st.json(result)
    
    elif scenario == "Temperature Annealing":
        st.markdown("""
        **Test**: Verify gradients remain valid across temperature schedule.
        
        Tests gradient flow at τ = 1.0, 0.5, 0.25, 0.1.
        """)
        
        if st.button("Run Annealing Test"):
            temps = [1.0, 0.5, 0.25, 0.1]
            results = []
            
            progress = st.progress(0)
            for i, t in enumerate(temps):
                result = DifferentiableFAMAILObjective.verify_gradients(
                    grid_dims=config['grid_dims'],
                    n_pickups=50,
                    n_dropoffs=50,
                    temperature=t,
                )
                results.append((t, result['passed']))
                progress.progress((i+1) / len(temps))
            
            all_passed = all(r[1] for r in results)
            
            if all_passed:
                st.success("✅ All temperature levels passed!")
            else:
                st.error("❌ Some temperature levels failed")
            
            st.table([{"Temperature": t, "Passed": "✓" if p else "✗"} for t, p in results])
    
    elif scenario == "Attribution Consistency":
        st.markdown("""
        **Test**: Verify LIS and DCD attributions are consistent with cell counts.
        
        Cells with extreme counts should have high LIS.
        Cells with unusual service ratios should have high DCD.
        """)
        
        if st.button("Run Attribution Consistency Test"):
            # Generate test data
            np.random.seed(42)
            
            # Create counts with known extremes
            counts = np.random.poisson(10, config['grid_dims'])
            counts[0, 0] = 100  # Extreme high
            counts[5, 5] = 0    # Extreme low
            
            lis = compute_all_lis_scores(counts)
            
            # Check that extremes have high LIS
            high_cell_lis = lis[0, 0]
            low_cell_lis = lis[5, 5]
            mean_lis = lis.mean()
            
            st.write(f"LIS at extreme high cell (0,0): {high_cell_lis:.4f}")
            st.write(f"LIS at extreme low cell (5,5): {low_cell_lis:.4f}")
            st.write(f"Mean LIS: {mean_lis:.4f}")
            
            if high_cell_lis > mean_lis and low_cell_lis > mean_lis:
                st.success("✅ LIS correctly identifies extreme cells")
            else:
                st.warning("⚠️ LIS may not correctly identify extremes")
    
    elif scenario == "Full Pipeline":
        st.markdown("""
        **Test**: Run the complete pipeline:
        1. Create trajectories with coordinates
        2. Compute soft cell assignments
        3. Compute combined objective
        4. Verify gradients
        5. Compute attributions
        6. Select trajectories
        """)
        
        if st.button("Run Full Pipeline Test"):
            progress = st.progress(0)
            status = st.empty()
            
            try:
                # Step 1: Generate data
                status.text("Step 1/6: Generating trajectories...")
                import torch
                
                n_trajs = 100
                torch.manual_seed(42)
                pickup_coords = torch.rand(n_trajs, 2) * torch.tensor(config['grid_dims'], dtype=torch.float32)
                dropoff_coords = torch.rand(n_trajs, 2) * torch.tensor(config['grid_dims'], dtype=torch.float32)
                progress.progress(1/6)
                
                # Step 2: Create module
                status.text("Step 2/6: Creating objective module...")
                module = DifferentiableFAMAILObjective(
                    alpha_spatial=config['alpha_spatial'],
                    alpha_causal=config['alpha_causal'],
                    alpha_fidelity=config['alpha_fidelity'],
                    grid_dims=config['grid_dims'],
                    temperature=config['temperature'],
                )
                progress.progress(2/6)
                
                # Step 3: Forward pass
                status.text("Step 3/6: Computing objective...")
                pickup_coords.requires_grad_(True)
                dropoff_coords.requires_grad_(True)
                
                total, terms = module(pickup_coords, dropoff_coords)
                progress.progress(3/6)
                
                # Step 4: Backward pass
                status.text("Step 4/6: Verifying gradients...")
                total.backward()
                
                has_pickup_grad = pickup_coords.grad is not None and not torch.isnan(pickup_coords.grad).any()
                has_dropoff_grad = dropoff_coords.grad is not None and not torch.isnan(dropoff_coords.grad).any()
                progress.progress(4/6)
                
                # Step 5: Attribution
                status.text("Step 5/6: Computing attribution...")
                # Convert to attribution format
                pickup_np = pickup_coords.detach().numpy()
                dropoff_np = dropoff_coords.detach().numpy()
                
                pickup_counts = np.zeros(config['grid_dims'])
                dropoff_counts = np.zeros(config['grid_dims'])
                
                trajs = []
                for i in range(n_trajs):
                    pi, pj = int(pickup_np[i, 0]) % config['grid_dims'][0], int(pickup_np[i, 1]) % config['grid_dims'][1]
                    di, dj = int(dropoff_np[i, 0]) % config['grid_dims'][0], int(dropoff_np[i, 1]) % config['grid_dims'][1]
                    
                    pickup_counts[pi, pj] += 1
                    dropoff_counts[di, dj] += 1
                    
                    trajs.append({
                        'trajectory_id': i,
                        'pickup_cell': (pi, pj),
                        'dropoff_cell': (di, dj),
                    })
                
                supply_counts = create_mock_supply_data(pickup_counts)
                
                # Create simple g function
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
                progress.progress(5/6)
                
                # Step 6: Selection
                status.text("Step 6/6: Selecting trajectories...")
                selected = select_trajectories_for_modification(attr_result, n_trajectories=10)
                progress.progress(6/6)
                
                status.text("Complete!")
                
                # Results
                st.success("✅ Full pipeline completed successfully!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Objective", f"{total.item():.4f}")
                col2.metric("F_spatial", f"{terms['f_spatial'].item():.4f}")
                col3.metric("Pickup Grad OK", "✓" if has_pickup_grad else "✗")
                
                st.markdown(f"**Selected {len(selected)} trajectories for modification**")
                
            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
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
