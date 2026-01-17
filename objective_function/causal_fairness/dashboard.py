"""
Streamlit Dashboard for Causal Fairness Term Validation.

This dashboard provides an interactive interface for:
- Configuring causal fairness computation parameters
- Visualizing R² coefficients and demand-service relationships
- Analyzing temporal patterns in causal fairness
- Exploring residual distributions across the grid
- Comparing different g(d) estimation methods
- Verifying differentiability for gradient-based optimization

Usage:
    streamlit run dashboard.py

Requirements:
    pip install streamlit pandas plotly matplotlib seaborn scipy scikit-learn statsmodels
"""

import sys
import os
from pathlib import Path
import pickle
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directories to path
SCRIPT_DIR = Path(__file__).resolve().parent
OBJECTIVE_FUNCTION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = OBJECTIVE_FUNCTION_DIR.parent
sys.path.insert(0, str(OBJECTIVE_FUNCTION_DIR))

from config import (
    CausalFairnessConfig,
    WEEKDAYS_JULY, WEEKDAYS_AUGUST, WEEKDAYS_SEPTEMBER, WEEKDAYS_TOTAL
)
from term import CausalFairnessTerm
from utils import (
    estimate_g_function,
    compute_r_squared,
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
    verify_gradient_with_estimation_method,
    verify_all_estimation_methods,
    create_gradient_verification_report,
    DifferentiableCausalFairness,
    DifferentiableCausalFairnessWithSoftCounts,
    compute_demand_conditional_deviation,
    compute_trajectory_causal_attribution,
    create_grid_heatmap_data,
    aggregate_to_grid,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def load_data(filepath: str) -> Dict:
    """Load and cache data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def format_period(period: Tuple[int, int], period_type: str) -> str:
    """Format period tuple for display."""
    time_val, day_val = period
    
    day_names = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}
    day_str = day_names.get(day_val, f'Day {day_val}')
    
    if period_type == "hourly":
        return f"{day_str} {time_val:02d}:00"
    elif period_type == "daily":
        return day_str
    elif period_type == "all":
        return "All"
    else:
        return f"{day_str} Bucket {time_val}"


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_r2_over_time(per_period_data: List[Dict], period_type: str) -> go.Figure:
    """Plot R² values over time."""
    df = pd.DataFrame(per_period_data)
    
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create period labels
    df['period_label'] = df.apply(
        lambda row: format_period((row['time'], row['day']), period_type), axis=1
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df['r_squared'],
        mode='lines+markers',
        name='R² (Causal Fairness)',
        line=dict(color='purple', width=2),
        marker=dict(size=6),
        hovertemplate='%{text}<br>R²: %{y:.4f}<extra></extra>',
        text=df['period_label'],
    ))
    
    # Add trend line
    if len(df) > 2:
        z = np.polyfit(range(len(df)), df['r_squared'], 1)
        trend = np.polyval(z, range(len(df)))
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=trend,
            mode='lines',
            name='Trend',
            line=dict(color='gray', dash='dash', width=1),
        ))
    
    # Add average line
    avg_r2 = df['r_squared'].mean()
    fig.add_hline(
        y=avg_r2, line_dash="dot", line_color="green",
        annotation_text=f"Avg: {avg_r2:.4f}",
    )
    
    fig.update_layout(
        title=f"Causal Fairness (R²) Over Time ({period_type.title()} Aggregation)",
        xaxis_title="Period Index",
        yaxis_title="R² (Causal Fairness)",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=True,
    )
    
    return fig


def plot_demand_vs_ratio_scatter(
    demands: np.ndarray,
    ratios: np.ndarray,
    g_func,
    method: str,
) -> go.Figure:
    """Plot demand vs service ratio with fitted g(d) curve."""
    fig = go.Figure()
    
    # Scatter plot of actual data
    fig.add_trace(go.Scatter(
        x=demands,
        y=ratios,
        mode='markers',
        name='Observed (D, Y)',
        marker=dict(
            size=4,
            color='rgba(100, 100, 200, 0.3)',
        ),
        hovertemplate='Demand: %{x}<br>Ratio: %{y:.3f}<extra></extra>',
    ))
    
    # Fitted g(d) curve
    if g_func is not None and len(demands) > 0:
        d_sorted = np.sort(np.unique(demands))
        if len(d_sorted) > 100:
            d_plot = np.linspace(d_sorted.min(), d_sorted.max(), 100)
        else:
            d_plot = d_sorted
        
        g_vals = g_func(d_plot)
        
        fig.add_trace(go.Scatter(
            x=d_plot,
            y=g_vals,
            mode='lines',
            name=f'g(d) = E[Y|D] ({method})',
            line=dict(color='red', width=3),
        ))
    
    fig.update_layout(
        title="Demand vs Service Ratio with Expected Value Function g(d)",
        xaxis_title="Demand (D = Pickup Count)",
        yaxis_title="Service Ratio (Y = Supply / Demand)",
        height=450,
        showlegend=True,
    )
    
    return fig


def plot_residual_distribution(residuals: np.ndarray) -> go.Figure:
    """Plot distribution of residuals."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        name="Residuals",
        marker_color='steelblue',
        opacity=0.7,
    ))
    
    # Add mean line
    mean_r = np.mean(residuals)
    fig.add_vline(
        x=mean_r, line_dash="dash", line_color="red",
        annotation_text=f"Mean: {mean_r:.4f}",
    )
    
    # Add zero line
    fig.add_vline(x=0, line_color="black", line_width=2)
    
    fig.update_layout(
        title="Distribution of Residuals R = Y - g(D)",
        xaxis_title="Residual Value",
        yaxis_title="Count",
        height=350,
    )
    
    return fig


def plot_residual_vs_demand(
    demands: np.ndarray,
    residuals: np.ndarray,
) -> go.Figure:
    """Plot residuals against demand to check for patterns."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=demands,
        y=residuals,
        mode='markers',
        marker=dict(
            size=5,
            color=residuals,
            colorscale='RdBu',
            cmin=-max(abs(residuals.min()), abs(residuals.max())),
            cmax=max(abs(residuals.min()), abs(residuals.max())),
            colorbar=dict(title="Residual"),
            showscale=True,
        ),
        hovertemplate='Demand: %{x}<br>Residual: %{y:.3f}<extra></extra>',
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_color="black", line_dash="dash")
    
    fig.update_layout(
        title="Residuals vs Demand (Should Show No Pattern if g(d) is Correct)",
        xaxis_title="Demand (D)",
        yaxis_title="Residual (Y - g(D))",
        height=400,
    )
    
    return fig


def plot_r2_distribution(r2_values: List[float]) -> go.Figure:
    """Plot distribution of per-period R² values."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=r2_values,
        nbinsx=20,
        name="R² Distribution",
        marker_color='purple',
        opacity=0.7,
    ))
    
    # Add statistics
    mean_r2 = np.mean(r2_values)
    median_r2 = np.median(r2_values)
    
    fig.add_vline(
        x=mean_r2, line_dash="dash", line_color="green",
        annotation_text=f"Mean: {mean_r2:.3f}",
    )
    fig.add_vline(
        x=median_r2, line_dash="dot", line_color="orange",
        annotation_text=f"Median: {median_r2:.3f}",
    )
    
    fig.update_layout(
        title="Distribution of Per-Period R² Values",
        xaxis_title="R² (Causal Fairness)",
        yaxis_title="Count",
        xaxis=dict(range=[0, 1]),
        height=300,
    )
    
    return fig


def plot_hourly_pattern(per_period_data: List[Dict]) -> Optional[go.Figure]:
    """Plot hourly pattern of causal fairness."""
    df = pd.DataFrame(per_period_data)
    
    if 'time' not in df.columns or len(df) == 0:
        return None
    
    # Group by hour
    hourly_avg = df.groupby('time').agg({
        'r_squared': ['mean', 'std'],
        'n_cells': 'mean',
    }).reset_index()
    hourly_avg.columns = ['hour', 'r2_mean', 'r2_std', 'n_cells_avg']
    
    fig = go.Figure()
    
    # R² line with error band
    fig.add_trace(go.Scatter(
        x=hourly_avg['hour'],
        y=hourly_avg['r2_mean'],
        mode='lines+markers',
        name='Mean R²',
        line=dict(color='purple', width=3),
        marker=dict(size=8),
    ))
    
    # Error band
    if 'r2_std' in hourly_avg.columns:
        fig.add_trace(go.Scatter(
            x=list(hourly_avg['hour']) + list(hourly_avg['hour'][::-1]),
            y=list(hourly_avg['r2_mean'] + hourly_avg['r2_std']) + 
              list((hourly_avg['r2_mean'] - hourly_avg['r2_std'])[::-1]),
            fill='toself',
            fillcolor='rgba(128, 0, 128, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 Std Dev',
        ))
    
    fig.update_layout(
        title="Average Causal Fairness by Hour of Day",
        xaxis_title="Hour",
        yaxis_title="R² (Causal Fairness)",
        xaxis=dict(tickmode='linear', dtick=2),
        yaxis=dict(range=[0, 1]),
        height=400,
    )
    
    return fig


def plot_heatmap_grid(
    data_grid: np.ndarray,
    title: str,
    colorscale: str = 'RdBu',
    zmid: Optional[float] = None,
) -> go.Figure:
    """Plot a 2D heatmap of grid data."""
    fig = go.Figure(data=go.Heatmap(
        z=data_grid,
        colorscale=colorscale,
        zmid=zmid,
        colorbar=dict(title="Value"),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Grid X",
        yaxis_title="Grid Y",
        height=500,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    
    return fig


def plot_method_comparison(
    demands: np.ndarray,
    ratios: np.ndarray,
) -> go.Figure:
    """Compare different g(d) estimation methods."""
    methods = ['binning', 'linear', 'polynomial', 'isotonic', 'lowess']
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    fig = go.Figure()
    
    # Scatter plot of data
    fig.add_trace(go.Scatter(
        x=demands,
        y=ratios,
        mode='markers',
        name='Data',
        marker=dict(size=3, color='gray', opacity=0.3),
    ))
    
    # Plot each method
    d_sorted = np.linspace(demands.min(), demands.max(), 100)
    
    r2_values = {}
    
    for method, color in zip(methods, colors):
        try:
            g_func, diag = estimate_g_function(
                demands, ratios, method=method,
                n_bins=10, poly_degree=2, lowess_frac=0.3
            )
            g_vals = g_func(d_sorted)
            expected = g_func(demands)
            r2 = compute_r_squared(ratios, expected)
            r2_values[method] = r2
            
            fig.add_trace(go.Scatter(
                x=d_sorted,
                y=g_vals,
                mode='lines',
                name=f'{method} (R²={r2:.3f})',
                line=dict(color=color, width=2),
            ))
        except Exception as e:
            r2_values[method] = None
    
    fig.update_layout(
        title="Comparison of g(d) Estimation Methods",
        xaxis_title="Demand (D)",
        yaxis_title="Service Ratio (Y)",
        height=450,
        showlegend=True,
    )
    
    return fig, r2_values


def plot_gradient_distribution(gradients: np.ndarray) -> go.Figure:
    """Plot distribution of gradients."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=gradients,
        nbinsx=50,
        name="Gradient Values",
        marker_color='steelblue',
        opacity=0.7,
    ))
    
    mean_grad = np.mean(gradients)
    fig.add_vline(
        x=mean_grad, line_dash="dash", line_color="red",
        annotation_text=f"Mean: {mean_grad:.2e}",
    )
    
    fig.update_layout(
        title="Distribution of ∂F_causal/∂S (Gradients w.r.t. Supply)",
        xaxis_title="Gradient Value",
        yaxis_title="Count",
        height=350,
    )
    
    return fig


# =============================================================================
# GRADIENT VERIFICATION TAB
# =============================================================================

def render_gradient_verification_tab(term: CausalFairnessTerm):
    """Render the gradient verification tab."""
    st.subheader("🔬 Differentiability Verification")
    
    st.markdown("""
    This tab verifies that the Causal Fairness term is properly differentiable
    for gradient-based trajectory optimization. The term uses the R² coefficient
    of determination:
    
    $$F_{\\text{causal}} = R^2 = 1 - \\frac{\\text{Var}(Y - g(D))}{\\text{Var}(Y)}$$
    
    The g(d) function is **pre-computed and frozen** during optimization, so
    gradients flow only through the service ratio computation.
    """)
    
    # Check PyTorch availability
    try:
        import torch
        pytorch_available = True
        pytorch_version = torch.__version__
    except ImportError:
        pytorch_available = False
        pytorch_version = None
    
    # Status indicators
    st.markdown("### 📋 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if pytorch_available:
            st.success(f"✅ PyTorch {pytorch_version}")
        else:
            st.error("❌ PyTorch Not Installed")
    
    with col2:
        metadata = term._build_metadata()
        if metadata.is_differentiable:
            st.success("✅ Term is Differentiable")
        else:
            st.warning("⚠️ Term Not Differentiable")
    
    with col3:
        st.info(f"📦 Version {metadata.version}")
    
    st.divider()
    
    # Gradient Verification
    st.markdown("### 🧪 Gradient Verification Tests")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_cells = st.slider(
            "Number of cells",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
        )
    
    with col2:
        test_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=9999,
            value=42,
        )
    
    if st.button("🚀 Run Gradient Verification", type="primary"):
        if not pytorch_available:
            st.error("PyTorch is required for gradient verification.")
            return
        
        with st.spinner("Running verification tests..."):
            try:
                result = verify_causal_fairness_gradient(n_cells, test_seed)
                
                st.markdown("#### Results")
                
                if result['passed']:
                    st.success("✅ **Gradients Computed Successfully**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("F_causal", f"{result['f_causal']:.6f}")
                    
                    with col2:
                        st.metric("Gradient Mean", f"{result['gradient_stats']['mean']:.6e}")
                    
                    with col3:
                        st.metric("Gradient Std", f"{result['gradient_stats']['std']:.6e}")
                    
                    with col4:
                        grad_range = result['gradient_stats']['max'] - result['gradient_stats']['min']
                        st.metric("Gradient Range", f"{grad_range:.6e}")
                    
                    # Gradient distribution
                    if 'gradients' in result:
                        fig = plot_gradient_distribution(result['gradients'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("❌ **Gradient Computation Failed**")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)
    
    st.divider()
    
    # Integration explanation
    with st.expander("📝 Integration with Trajectory Modification"):
        st.markdown("""
        ### Gradient-Based Trajectory Optimization
        
        The differentiable causal fairness term enables optimization of taxi trajectories:
        
        1. **Pre-compute g(d)**: Before optimization, estimate g(d) = E[Y|D] from original data
        2. **Freeze g(d)**: Register as non-trainable (no gradients)
        3. **Forward pass**: Modified trajectories → Supply counts → Service ratios → R²
        4. **Backward pass**: Gradients flow through R² → ratios → supply → trajectory adjustments
        
        ```python
        # Example usage
        from causal_fairness import CausalFairnessTerm, DifferentiableCausalFairness
        
        # Pre-compute g(d) from original data
        term = CausalFairnessTerm(config)
        diff_module = term.get_differentiable_module(demands, ratios)
        
        # During optimization
        supply = torch.tensor(supply_values, requires_grad=True)
        demand = torch.tensor(demand_values)  # No grad
        f_causal = diff_module.compute(supply, demand)
        f_causal.backward()
        
        # supply.grad contains ∂F_causal/∂S
        ```
        
        ### Gradient Interpretation
        
        - **Positive gradient**: Increasing supply in this cell increases R² (improves causal fairness)
        - **Negative gradient**: Increasing supply decreases R² (hurts causal fairness)
        
        Cells with high demand but low service (large negative residuals) will typically
        have positive gradients, guiding trajectories toward underserved areas.
        """)


# =============================================================================
# ESTIMATION METHOD GRADIENT TESTS TAB
# =============================================================================

def render_method_gradient_tests_tab():
    """
    Render the estimation method gradient verification tab.
    
    This tab validates that each g(d) estimation method (especially Isotonic
    and Binning) allows gradients to flow properly during optimization.
    """
    st.subheader("🧪 Estimation Method Gradient Validation")
    
    st.markdown("""
    This tab validates that **each g(d) estimation method** allows gradients to flow
    properly through the Causal Fairness computation. This is critical for ensuring
    the trajectory modification algorithm can optimize using gradient descent.
    
    ### Why This Matters
    
    The Causal Fairness term uses R² as its score:
    
    $$F_{\\text{causal}} = R^2 = 1 - \\frac{\\text{Var}(Y - g(D))}{\\text{Var}(Y)}$$
    
    The g(d) function is **pre-computed and frozen** before optimization. We need
    to verify that gradients can flow through the residual computation $R = Y - g(D)$
    regardless of how g(d) was estimated.
    
    ### Key Insight
    
    > **Isotonic regression and Binning methods are NOT differentiable during fitting.**
    > However, we don't differentiate through the fitting process — we only use the
    > **frozen lookup values** during optimization. The gradients flow through:
    > $$\\frac{\\partial F_{causal}}{\\partial S} = \\frac{\\partial F_{causal}}{\\partial Y} \\cdot \\frac{\\partial Y}{\\partial S}$$
    > where $Y = S/D$ (service ratio).
    """)
    
    # Check PyTorch availability
    try:
        import torch
        pytorch_available = True
        pytorch_version = torch.__version__
    except ImportError:
        pytorch_available = False
        pytorch_version = None
    
    if not pytorch_available:
        st.error("❌ PyTorch is required for gradient verification tests.")
        return
    
    st.success(f"✅ PyTorch {pytorch_version} available")
    
    st.divider()
    
    # ==========================================================================
    # TEST CONFIGURATION
    # ==========================================================================
    
    st.markdown("### ⚙️ Test Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples = st.slider(
            "Number of samples",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="More samples = more robust test, but slower"
        )
    
    with col2:
        test_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=9999,
            value=42,
            help="For reproducibility"
        )
    
    with col3:
        methods_to_test = st.multiselect(
            "Methods to test",
            options=["binning", "isotonic", "polynomial", "linear", "lowess"],
            default=["binning", "isotonic", "polynomial", "linear"],
            help="Select estimation methods to validate"
        )
    
    st.divider()
    
    # ==========================================================================
    # RUN TESTS
    # ==========================================================================
    
    st.markdown("### 🚀 Run Gradient Tests")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        run_all = st.button("Run All Methods", type="primary")
    
    with col2:
        run_single = st.selectbox(
            "Or test single method:",
            options=["(Select method)"] + methods_to_test,
            help="Run test for a specific method"
        )
    
    # Store results in session state
    if 'method_gradient_results' not in st.session_state:
        st.session_state.method_gradient_results = {}
    
    # Run all methods
    if run_all:
        st.markdown("---")
        st.markdown("### 📊 Running All Tests...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        for i, method in enumerate(methods_to_test):
            status_text.text(f"Testing {method}...")
            progress_bar.progress((i + 1) / len(methods_to_test))
            
            result = verify_gradient_with_estimation_method(
                method=method,
                n_samples=n_samples,
                seed=test_seed,
                verbose=False,
            )
            results[method] = result
        
        st.session_state.method_gradient_results = results
        status_text.text("✅ All tests complete!")
        progress_bar.progress(1.0)
    
    # Run single method
    elif run_single and run_single != "(Select method)":
        st.markdown("---")
        st.markdown(f"### 📊 Testing {run_single}...")
        
        with st.spinner(f"Running gradient test for {run_single}..."):
            result = verify_gradient_with_estimation_method(
                method=run_single,
                n_samples=n_samples,
                seed=test_seed,
                verbose=False,
            )
            st.session_state.method_gradient_results[run_single] = result
    
    # ==========================================================================
    # DISPLAY RESULTS
    # ==========================================================================
    
    results = st.session_state.method_gradient_results
    
    if results:
        st.markdown("---")
        st.markdown("## 📋 Test Results")
        
        # Summary table
        st.markdown("### Summary")
        
        summary_data = []
        for method, res in results.items():
            if res.get('error'):
                summary_data.append({
                    'Method': method,
                    'R² Fit': 'ERROR',
                    'F_causal': 'ERROR',
                    'Gradient Valid': '❌',
                    'Numerical Match': '❌',
                    'Status': '❌ FAILED',
                })
            else:
                grad_valid = "✅" if res.get('gradient_valid', False) else "❌"
                num_check = res.get('numerical_check', {})
                num_match = "✅" if num_check.get('gradients_match', False) else "⚠️"
                status = "✅ PASS" if res.get('passed', False) else "❌ FAIL"
                
                summary_data.append({
                    'Method': method,
                    'R² Fit': f"{res.get('r2_fit', 0):.4f}",
                    'F_causal': f"{res.get('f_causal', 0):.4f}",
                    'Gradient Valid': grad_valid,
                    'Numerical Match': num_match,
                    'Status': status,
                })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
        
        # Overall status
        all_passed = all(res.get('passed', False) for res in results.values())
        if all_passed:
            st.success("✅ **All methods passed gradient verification!** "
                      "Isotonic and Binning methods are safe to use for trajectory optimization.")
        else:
            failed_methods = [m for m, r in results.items() if not r.get('passed', False)]
            st.warning(f"⚠️ **Some methods failed:** {', '.join(failed_methods)}")
        
        st.divider()
        
        # Detailed results per method
        st.markdown("### 🔍 Detailed Results")
        
        for method, res in results.items():
            with st.expander(f"📊 {method.title()} Method", expanded=(not res.get('passed', False))):
                if res.get('error'):
                    st.error(f"**Error:** {res['error']}")
                    if res.get('traceback'):
                        st.code(res['traceback'], language='python')
                    continue
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² Fit", f"{res.get('r2_fit', 0):.4f}",
                             help="How well g(d) fits the data")
                
                with col2:
                    st.metric("F_causal", f"{res.get('f_causal', 0):.4f}",
                             help="Computed causal fairness score")
                
                with col3:
                    grad_exists = "✅" if res.get('gradient_exists', False) else "❌"
                    st.metric("Gradient Exists", grad_exists)
                
                with col4:
                    passed = "✅ PASS" if res.get('passed', False) else "❌ FAIL"
                    st.metric("Status", passed)
                
                # Gradient statistics
                grad_stats = res.get('gradient_stats')
                if grad_stats:
                    st.markdown("**Gradient Statistics:**")
                    
                    cols = st.columns(5)
                    stats_items = [
                        ("Mean", f"{grad_stats.get('mean', 0):.6e}"),
                        ("Std", f"{grad_stats.get('std', 0):.6e}"),
                        ("Min", f"{grad_stats.get('min', 0):.6e}"),
                        ("Max", f"{grad_stats.get('max', 0):.6e}"),
                        ("Non-zero %", f"{grad_stats.get('nonzero_pct', 0):.1f}%"),
                    ]
                    
                    for col, (label, value) in zip(cols, stats_items):
                        col.metric(label, value)
                
                # Numerical check results
                num_check = res.get('numerical_check', {})
                if num_check:
                    st.markdown("**Numerical Gradient Verification:**")
                    
                    match_status = "✅ Matches" if num_check.get('gradients_match', False) else "⚠️ Mismatch"
                    st.write(f"- Status: {match_status}")
                    st.write(f"- Max Relative Error: {num_check.get('max_rel_error', 0):.6f}")
                    st.write(f"- Mean Relative Error: {num_check.get('mean_rel_error', 0):.6f}")
                    
                    # Show comparison table
                    if num_check.get('numerical_grads') and num_check.get('analytic_grads'):
                        compare_df = pd.DataFrame({
                            'Numerical': num_check['numerical_grads'],
                            'Analytic': num_check['analytic_grads'],
                            'Rel Error': num_check['relative_errors'],
                        })
                        st.dataframe(compare_df.style.format('{:.6e}'), use_container_width=True)
                
                # Gradient distribution plot
                if 'gradients' in res:
                    fig = plot_gradient_distribution(res['gradients'])
                    fig.update_layout(title=f"Gradient Distribution ({method})")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Method diagnostics
                if res.get('diagnostics'):
                    with st.expander("Method Diagnostics"):
                        st.json(res['diagnostics'])
        
        st.divider()
        
        # Download report
        st.markdown("### 📥 Export Results")
        
        report = create_gradient_verification_report(results)
        st.download_button(
            label="📥 Download Markdown Report",
            data=report,
            file_name="gradient_verification_report.md",
            mime="text/markdown",
        )
        
        # Show report preview
        with st.expander("📄 Report Preview"):
            st.markdown(report)
    
    st.divider()
    
    # ==========================================================================
    # EXPLANATION
    # ==========================================================================
    
    with st.expander("📝 Technical Explanation"):
        st.markdown("""
        ### How Gradient Verification Works
        
        For each estimation method, we:
        
        1. **Generate test data**: Create realistic demand and supply values
        2. **Fit g(d)**: Use the specified method to estimate E[Y|D]
        3. **Freeze g(d)**: Convert to tensor lookup (no gradients)
        4. **Forward pass**: Compute F_causal with gradient tracking on supply
        5. **Backward pass**: Call `.backward()` to compute gradients
        6. **Validate**: Check gradients are non-zero and numerically correct
        
        ### Numerical Gradient Check
        
        We verify analytic gradients using finite differences:
        
        $$\\frac{\\partial F}{\\partial S_i} \\approx \\frac{F(S_i + \\epsilon) - F(S_i - \\epsilon)}{2\\epsilon}$$
        
        A test **passes** if:
        - Gradients exist and are non-zero
        - No NaN or Inf values
        - Numerical and analytic gradients match within 10%
        
        ### Why Isotonic/Binning Methods Work
        
        Even though these methods use non-differentiable operations during **fitting**
        (sorting, bin assignment), the **lookup** during optimization is differentiable:
        
        ```python
        # FITTING (non-differentiable) - done ONCE before optimization
        g_func, _ = estimate_g_isotonic(demands, ratios)
        
        # LOOKUP (differentiable) - done during optimization
        expected = torch.tensor(g_func(demands_np))  # Just tensor values
        Y = supply / demand  # Differentiable
        R = Y - expected     # Differentiable (expected is constant)
        ```
        
        The key is that `expected` values are **constants** during optimization.
        Gradients flow through `Y = S/D`, not through `g(d)` fitting.
        """)


def render_soft_cell_assignment_tab(term: CausalFairnessTerm, breakdown: Dict):
    """
    Render the Soft Cell Assignment and Trajectory Attribution tab for Causal Fairness.
    
    This tab provides:
    1. Soft cell assignment configuration for pickup locations
    2. End-to-end gradient verification with soft counts
    3. DCD (Demand-Conditional Deviation) attribution visualization
    4. Temperature annealing demonstration
    
    Args:
        term: The configured CausalFairnessTerm instance
        breakdown: The computed breakdown with demand/supply data
    """
    st.subheader("🔄 Soft Cell Assignment & Trajectory Attribution")
    
    st.markdown("""
    This section demonstrates the **Soft Cell Assignment** module for causal fairness,
    enabling differentiable trajectory optimization by converting discrete pickup locations
    to probabilistic soft assignments.
    
    **Key Components:**
    
    1. **Soft Assignment** for pickup locations:
       $$\\sigma_c(x, y) = \\frac{\\exp(-d_c^2 / 2\\tau^2)}{Z}$$
    
    2. **Differentiable R²** with frozen g(d):
       $$R^2 = 1 - \\frac{\\text{Var}(Y - g(D))}{\\text{Var}(Y)}$$
    
    Where g(d) is **frozen** after initial fitting - gradients only flow through Y.
    """)
    
    # Check PyTorch availability
    try:
        import torch
        pytorch_available = True
    except ImportError:
        pytorch_available = False
    
    if not pytorch_available:
        st.error("❌ PyTorch is required for soft cell assignment. Install with: `pip install torch`")
        return
    
    import torch
    
    st.divider()
    
    # ==========================================================================
    # Section 1: Soft Assignment Configuration
    # ==========================================================================
    st.markdown("### ⚙️ Soft Assignment Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        neighborhood_size = st.slider(
            "Neighborhood Size",
            min_value=3,
            max_value=11,
            value=5,
            step=2,
            key="causal_neighborhood",
            help="Size of the neighborhood grid (must be odd)"
        )
    
    with col2:
        temperature = st.slider(
            "Temperature (τ)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            key="causal_temperature",
            help="Controls softness: lower = sharper, higher = softer"
        )
    
    with col3:
        grid_dims = term.config.grid_dims if hasattr(term, 'config') else (48, 90)
        st.info(f"Grid: {grid_dims[0]} × {grid_dims[1]}")
    
    # Visualize soft assignment kernel
    st.markdown("#### Soft Assignment Kernel Visualization")
    
    half = neighborhood_size // 2
    kernel = np.zeros((neighborhood_size, neighborhood_size))
    for i in range(neighborhood_size):
        for j in range(neighborhood_size):
            d_sq = (i - half)**2 + (j - half)**2
            kernel[i, j] = np.exp(-d_sq / (2 * temperature**2))
    kernel = kernel / kernel.sum()
    
    fig_kernel = px.imshow(
        kernel,
        title=f"Soft Assignment Kernel (τ={temperature})",
        color_continuous_scale="Purples",
        aspect="equal",
    )
    fig_kernel.update_layout(height=350)
    st.plotly_chart(fig_kernel, use_container_width=True)
    
    st.divider()
    
    # ==========================================================================
    # Section 2: End-to-End Gradient Verification
    # ==========================================================================
    st.markdown("### 🧪 End-to-End Gradient Verification")
    
    st.markdown("""
    Test the complete gradient chain: 
    **Pickup Location → Soft Assignment → Soft Demand → R² → Loss**
    
    Note: g(d) is frozen during this test - gradients only flow through the service ratio Y.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_grid_size = st.number_input(
            "Test Grid Size",
            min_value=5,
            max_value=20,
            value=10,
            key="causal_test_grid",
            help="Grid dimensions for testing"
        )
    
    with col2:
        n_test_trajectories = st.number_input(
            "Number of Test Trajectories",
            min_value=1,
            max_value=20,
            value=5,
            key="causal_n_trajectories",
            help="Number of trajectories to test"
        )
    
    if st.button("🚀 Run End-to-End Gradient Test", key="causal_soft_grad_test"):
        with st.spinner("Running gradient verification..."):
            try:
                # Run verification using the static method (temperature is set internally)
                result = DifferentiableCausalFairnessWithSoftCounts.verify_end_to_end_gradients(
                    grid_dims=(int(test_grid_size), int(test_grid_size)),
                    n_trajectories=int(n_test_trajectories),
                )
                
                # Display results
                if result.get('passed', False):
                    st.success("✅ **End-to-End Gradients Flow Correctly!**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Causal Fairness", f"{result.get('f_causal', 0):.6f}")
                    
                    with col2:
                        grad_stats = result.get('pickup_gradient_stats', {})
                        st.metric("Grad Mean", f"{grad_stats.get('mean', 0):.2e}")
                    
                    with col3:
                        st.metric("Grad Std", f"{grad_stats.get('std', 0):.2e}")
                    
                    with col4:
                        st.metric("Grad Range", f"{grad_stats.get('max', 0) - grad_stats.get('min', 0):.2e}")
                    
                    # Show gradient statistics
                    st.markdown("#### Pickup Location Gradient Statistics")
                    if 'pickup_gradient_stats' in result:
                        st.json(result['pickup_gradient_stats'])
                    
                    st.info("""
                    **Note:** These gradients show how changing pickup locations affects the 
                    causal fairness score (R²). The g(d) function remains frozen - only the 
                    service ratio Y = S/D changes with pickup location modifications.
                    """)
                    
                else:
                    st.error("❌ **Gradient computation failed**")
                    st.markdown("**Debug Info:**")
                    st.json(result)
                        
            except Exception as e:
                st.error(f"Error during verification: {str(e)}")
                st.exception(e)
    
    st.divider()
    
    # ==========================================================================
    # Section 3: Frozen g(d) Explanation
    # ==========================================================================
    st.markdown("### 🧊 Why g(d) Must Be Frozen")
    
    st.markdown("""
    The expected service ratio function g(d) = E[Y|D] captures the **baseline relationship** 
    between demand and service. It must be frozen during trajectory optimization for two reasons:
    
    1. **Conceptual**: g(d) represents the *structural relationship* we want to preserve - we 
       want trajectories that maintain this relationship, not change it.
    
    2. **Technical**: g(d) estimation involves non-differentiable operations (sorting, 
       binning). Freezing allows us to use any estimation method.
    
    **Gradient Flow:**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ✅ **During Optimization (Differentiable):**
        ```
        pickup_loc → soft_assign → demand_counts → Y = S/D
                                                    ↓
        R² = 1 - Var(Y - g(D))/Var(Y) ← gradients flow
        ```
        """)
    
    with col2:
        st.markdown("""
        ❄️ **Before Optimization (Frozen):**
        ```
        Historical data → demands, ratios
                       → fit g(d) with isotonic/binning/etc.
                       → freeze g(d) as lookup table
        ```
        """)
    
    st.divider()
    
    # ==========================================================================
    # Section 4: DCD Attribution
    # ==========================================================================
    st.markdown("### 📊 Demand-Conditional Deviation (DCD) Attribution")
    
    st.markdown("""
    The **DCD Score** measures how much each trajectory's pickup cell deviates from 
    the expected service ratio given its demand:
    
    $$\\text{DCD}_t = |Y_{\\text{cell}} - g(D_{\\text{cell}})|$$
    
    High DCD indicates the cell receives unexpectedly high or low service relative to demand.
    """)
    
    # Create sample visualization
    if st.button("🎲 Generate Sample DCD Visualization", key="dcd_vis"):
        with st.spinner("Computing DCD scores..."):
            # Create synthetic data
            np.random.seed(42)
            n_cells = 100
            
            # Generate demand (exponential - some high demand areas)
            demands = np.random.exponential(20, n_cells) + 5
            
            # Generate supply (correlated with demand but with noise)
            supply = demands * (0.8 + 0.4 * np.random.rand(n_cells))
            
            # Compute service ratios
            ratios = supply / demands
            
            # Fit g(d) - simple linear for visualization
            from scipy.stats import linregress
            slope, intercept, _, _, _ = linregress(demands, ratios)
            expected = slope * demands + intercept
            
            # Compute residuals and DCD
            residuals = ratios - expected
            dcd_scores = np.abs(residuals)
            
            # Create grid visualization (10x10)
            demand_grid = demands.reshape(10, 10)
            ratio_grid = ratios.reshape(10, 10)
            dcd_grid = dcd_scores.reshape(10, 10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_demand = px.imshow(
                    demand_grid,
                    title="Demand Distribution (Synthetic)",
                    color_continuous_scale="YlOrRd",
                )
                st.plotly_chart(fig_demand, use_container_width=True)
            
            with col2:
                fig_dcd = px.imshow(
                    dcd_grid,
                    title="DCD Scores (|Y - g(D)|)",
                    color_continuous_scale="RdBu_r",
                )
                st.plotly_chart(fig_dcd, use_container_width=True)
            
            # Scatter plot of demand vs ratio with g(d)
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=demands, y=ratios,
                mode='markers',
                marker=dict(
                    size=8,
                    color=dcd_scores,
                    colorscale='RdBu_r',
                    colorbar=dict(title="DCD"),
                ),
                name='Cells',
            ))
            
            # Add g(d) line
            d_sorted = np.sort(demands)
            fig_scatter.add_trace(go.Scatter(
                x=d_sorted,
                y=slope * d_sorted + intercept,
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='g(d) = E[Y|D]',
            ))
            
            fig_scatter.update_layout(
                title="Service Ratio vs Demand with g(d)",
                xaxis_title="Demand (D)",
                yaxis_title="Service Ratio (Y = S/D)",
                height=400,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Show high-DCD cells
            st.markdown("**Top 5 Cells by DCD Score (Candidates for Modification)**")
            
            top_indices = np.argsort(dcd_scores)[-5:][::-1]
            
            top_data = []
            for idx in top_indices:
                row, col = divmod(idx, 10)
                top_data.append({
                    'Cell': f"({row}, {col})",
                    'Demand': f"{demands[idx]:.1f}",
                    'Supply': f"{supply[idx]:.1f}",
                    'Ratio Y': f"{ratios[idx]:.3f}",
                    'Expected g(D)': f"{expected[idx]:.3f}",
                    'DCD': f"{dcd_scores[idx]:.4f}",
                    'Direction': "↓ Over-served" if residuals[idx] > 0 else "↑ Under-served",
                })
            
            st.dataframe(pd.DataFrame(top_data), use_container_width=True)
            
            st.info("""
            **Interpretation:** Cells with high DCD deviate most from the expected 
            demand-service relationship. The optimization should modify trajectories 
            to move supply from over-served cells (positive residual) to under-served 
            cells (negative residual).
            """)
    
    st.divider()
    
    # Integration code example
    with st.expander("📝 Soft Cell Assignment Integration Code"):
        st.code("""
# Example: Using soft cell assignment for causal fairness optimization

import torch
from causal_fairness.term import CausalFairnessTerm
from soft_cell_assignment import SoftCellAssignment

# Initialize term
term = CausalFairnessTerm(config)

# Get historical data for g(d) fitting
demands, ratios = get_historical_data()

# Get soft count module (g(d) is fitted internally and frozen)
soft_module = term.get_soft_count_module(
    demands=demands,
    ratios=ratios,
    neighborhood_size=5,
    initial_temperature=1.0,
)

# Create trajectory with differentiable pickup locations
pickup_locs = torch.tensor([[24.5, 45.3], [12.1, 67.8]], requires_grad=True)
pickup_cells = torch.tensor([[24, 45], [12, 67]])

# Compute causal fairness from locations
f_causal = soft_module.compute_from_locations(
    pickup_locs, pickup_cells,
    base_demand, supply,
)

# Backpropagate to get location gradients
(-f_causal).backward()  # Negative because we maximize R²

# Update trajectory (gradient ascent on R²)
pickup_locs.data -= learning_rate * pickup_locs.grad

# Note: g(d) remains frozen throughout - only Y = S/D changes
        """, language="python")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Causal Fairness Dashboard",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("📊 Causal Fairness Term Dashboard")
    st.markdown("""
    This dashboard helps validate and analyze the **Causal Fairness Term** ($F_{\\text{causal}}$)
    of the FAMAIL objective function. The term measures how well taxi service aligns with
    passenger demand using the coefficient of determination (R²).
    
    $$F_{\\text{causal}} = R^2 = 1 - \\frac{\\text{Var}(Y - g(D))}{\\text{Var}(Y)}$$
    
    where:
    - $Y = S/D$ = Service ratio (supply / demand)
    - $D$ = Demand (pickup counts)
    - $g(d) = E[Y|D=d]$ = Expected service ratio given demand
    """)
    
    # ==========================================================================
    # SIDEBAR CONFIGURATION
    # ==========================================================================
    
    st.sidebar.header("⚙️ Configuration")
    
    # Data file paths
    st.sidebar.subheader("📁 Data Files")
    
    default_demand_path = str(PROJECT_ROOT / "source_data" / "pickup_dropoff_counts.pkl")
    demand_path = st.sidebar.text_input(
        "Demand Data (pickup_dropoff_counts)",
        value=default_demand_path,
        help="Path to pickup_dropoff_counts.pkl file"
    )
    
    default_supply_path = str(PROJECT_ROOT / "active_taxis" / "output" / "active_taxis_5x5_hourly.pkl")
    supply_path = st.sidebar.text_input(
        "Supply Data (active_taxis)",
        value=default_supply_path,
        help="Path to active_taxis output file (optional)"
    )
    
    # Period type
    st.sidebar.subheader("⏰ Temporal Settings")
    
    period_type = st.sidebar.selectbox(
        "Period Type",
        options=["hourly", "daily", "time_bucket", "all"],
        index=0,
        help="Temporal aggregation level"
    )
    
    # Dataset time period
    dataset_period = st.sidebar.selectbox(
        "Dataset Time Period",
        options=["July (21 days)", "August (23 days)", "September (22 days)", "All (66 days)"],
        index=3,
    )
    
    num_days_map = {
        "July (21 days)": WEEKDAYS_JULY,
        "August (23 days)": WEEKDAYS_AUGUST,
        "September (22 days)": WEEKDAYS_SEPTEMBER,
        "All (66 days)": WEEKDAYS_TOTAL,
    }
    num_days = num_days_map[dataset_period]
    
    # Days filter
    st.sidebar.subheader("📅 Day Filters")
    
    days_options = {
        "All Weekdays": None,
        "Mon-Wed": [1, 2, 3],
        "Thu-Fri": [4, 5],
        "Monday Only": [1],
        "Friday Only": [5],
    }
    days_selection = st.sidebar.selectbox(
        "Days to Include",
        options=list(days_options.keys()),
        index=0,
    )
    days_filter = days_options[days_selection]
    
    # Estimation method
    st.sidebar.subheader("📈 Estimation Settings")
    
    estimation_method = st.sidebar.selectbox(
        "g(d) Estimation Method",
        options=["binning", "linear", "polynomial", "isotonic", "lowess"],
        index=0,
        help="Method to estimate E[Y|D]"
    )
    
    n_bins = st.sidebar.slider(
        "Number of Bins (for binning)",
        min_value=2,
        max_value=30,
        value=10,
    )
    
    poly_degree = st.sidebar.slider(
        "Polynomial Degree",
        min_value=1,
        max_value=5,
        value=2,
    )
    
    # Data filtering
    st.sidebar.subheader("🔧 Data Filtering")
    
    min_demand = st.sidebar.number_input(
        "Minimum Demand",
        min_value=1,
        max_value=50,
        value=1,
        help="Minimum pickup count to include cell"
    )
    
    include_zero_supply = st.sidebar.checkbox(
        "Include Zero Supply Cells",
        value=False,
    )
    
    max_ratio = st.sidebar.number_input(
        "Maximum Ratio (0 = no cap)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        help="Cap extreme ratios (0 = disabled)"
    )
    max_ratio = None if max_ratio == 0 else max_ratio
    
    # ==========================================================================
    # LOAD AND PROCESS DATA
    # ==========================================================================
    
    # Validate paths
    if not Path(demand_path).exists():
        st.error(f"❌ Demand data file not found: {demand_path}")
        st.stop()
    
    # Load demand data
    try:
        raw_data = load_data(demand_path)
        st.sidebar.success(f"✅ Loaded {len(raw_data):,} demand entries")
    except Exception as e:
        st.error(f"Failed to load demand data: {e}")
        st.stop()
    
    # Load supply data (optional) - use specialized loader for bundle format
    supply_data = None
    if Path(supply_path).exists():
        try:
            supply_data = load_active_taxis_data(supply_path)
            st.sidebar.success(f"✅ Loaded {len(supply_data):,} supply entries")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load supply data: {e}")
    else:
        st.sidebar.info("ℹ️ Using dropoff counts as supply proxy")
    
    # Build configuration
    config = CausalFairnessConfig(
        period_type=period_type,
        estimation_method=estimation_method,
        n_bins=n_bins,
        poly_degree=poly_degree,
        min_demand=min_demand,
        max_ratio=max_ratio,
        include_zero_supply=include_zero_supply,
        num_days=num_days,
        days_filter=days_filter,
        active_taxis_data_path=supply_path if Path(supply_path).exists() else None,
    )
    
    # Initialize term
    term = CausalFairnessTerm(config)
    
    # Prepare auxiliary data
    auxiliary_data = {'pickup_dropoff_counts': raw_data}
    if supply_data is not None:
        auxiliary_data['active_taxis'] = supply_data
    
    # Compute results
    with st.spinner("Computing causal fairness..."):
        try:
            breakdown = term.compute_with_breakdown({}, auxiliary_data)
        except Exception as e:
            st.error(f"Computation failed: {e}")
            st.exception(e)
            st.stop()
    
    # ==========================================================================
    # MAIN DISPLAY
    # ==========================================================================
    
    # Key metrics
    st.markdown("### 📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Causal Fairness (F_causal)",
            f"{breakdown['value']:.4f}",
            help="R² - coefficient of determination"
        )
    
    with col2:
        st.metric(
            "Overall R²",
            f"{breakdown['components'].get('overall_r2', 0):.4f}",
            help="R² computed on all data pooled"
        )
    
    with col3:
        st.metric(
            "Periods Analyzed",
            f"{breakdown['diagnostics']['n_periods']}",
        )
    
    with col4:
        st.metric(
            "Total Cells",
            f"{breakdown['diagnostics']['n_total_cells']:,}",
        )
    
    # Interpretation
    f_causal = breakdown['value']
    if f_causal >= 0.7:
        st.success(f"""
        ✅ **High Causal Fairness ({f_causal:.2%})**
        
        Service allocation is strongly aligned with demand. Most variation in service
        ratios is explained by demand differences.
        """)
    elif f_causal >= 0.4:
        st.warning(f"""
        ⚠️ **Moderate Causal Fairness ({f_causal:.2%})**
        
        Some service variation is explained by demand, but significant unexplained
        variation exists. This may indicate contextual biases.
        """)
    else:
        st.error(f"""
        ❌ **Low Causal Fairness ({f_causal:.2%})**
        
        Service allocation is weakly related to demand. Significant contextual
        factors (location, time) appear to influence service beyond demand.
        """)
    
    st.divider()
    
    # ==========================================================================
    # TABS
    # ==========================================================================
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Temporal Analysis",
        "🗺️ Spatial Distribution",
        "📊 Demand-Service Plot",
        "📉 Residual Analysis",
        "🔄 Method Comparison",
        "📉 Statistics",
        "🔬 Gradient Verification",
        "🧪 Method Gradient Tests",
        "🔄 Soft Cell Assignment",
    ])
    
    # --------------------------------------------------------------------------
    # TAB 1: Temporal Analysis
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader("Temporal Patterns")
        
        per_period_data = breakdown['components']['per_period_data']
        
        if len(per_period_data) > 1:
            # R² over time
            fig_time = plot_r2_over_time(per_period_data, period_type)
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Hourly pattern
            if period_type == "hourly":
                fig_hourly = plot_hourly_pattern(per_period_data)
                if fig_hourly:
                    st.plotly_chart(fig_hourly, use_container_width=True)
            
            # R² distribution
            r2_values = breakdown['components']['per_period_r2']
            if r2_values:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_r2_dist = plot_r2_distribution(r2_values)
                    st.plotly_chart(fig_r2_dist, use_container_width=True)
                
                with col2:
                    st.markdown("**R² Statistics**")
                    st.write(f"- Mean: {np.mean(r2_values):.4f}")
                    st.write(f"- Median: {np.median(r2_values):.4f}")
                    st.write(f"- Std: {np.std(r2_values):.4f}")
                    st.write(f"- Min: {np.min(r2_values):.4f}")
                    st.write(f"- Max: {np.max(r2_values):.4f}")
        else:
            st.info("Single period - no temporal variation to display.")
    
    # --------------------------------------------------------------------------
    # TAB 2: Spatial Distribution
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("Spatial Distribution")
        
        # Get data for grid visualization
        demands_arr = np.array(breakdown['components']['demands'])
        ratios_arr = np.array(breakdown['components']['ratios'])
        expected_arr = np.array(breakdown['components']['expected'])
        residuals_arr = np.array(breakdown['components']['residuals'])
        
        if len(demands_arr) == 0:
            st.info("No data available for spatial visualization.")
        else:
            st.markdown("""
            These heatmaps show the spatial distribution of key metrics across the 48×90 grid.
            Note: Values are aggregated across all time periods.
            """)
            
            # Build demand grid
            demand_dict = extract_demand_from_counts(raw_data)
            if days_filter:
                demand_dict = filter_by_days(demand_dict, days_filter)
            
            demand_grid = aggregate_to_grid(demand_dict, config.grid_dims, 'sum')
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_demand = plot_heatmap_grid(
                    demand_grid,
                    "Total Demand (Pickups)",
                    colorscale='Blues',
                )
                st.plotly_chart(fig_demand, use_container_width=True)
            
            with col2:
                # Supply grid (if available)
                if supply_data:
                    supply_dict = supply_data
                    if days_filter:
                        supply_dict = filter_by_days(supply_dict, days_filter)
                    supply_grid = aggregate_to_grid(supply_dict, config.grid_dims, 'sum')
                    
                    fig_supply = plot_heatmap_grid(
                        supply_grid,
                        "Total Supply (Active Taxis)",
                        colorscale='Greens',
                    )
                    st.plotly_chart(fig_supply, use_container_width=True)
                else:
                    st.info("Supply heatmap requires active_taxis data.")
            
            # Residual interpretation
            st.markdown("""
            **Interpretation:**
            - **Blue areas (high demand)**: More pickup requests
            - **Green areas (high supply)**: More taxi availability
            - Areas with demand but low supply may have negative residuals (underserved)
            """)
    
    # --------------------------------------------------------------------------
    # TAB 3: Demand-Service Plot
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("Demand vs Service Ratio")
        
        demands_arr = np.array(breakdown['components']['demands'])
        ratios_arr = np.array(breakdown['components']['ratios'])
        
        if len(demands_arr) == 0:
            st.info("No data available for scatter plot.")
        else:
            # Get g function
            g_func = term.get_g_function()
            
            fig_scatter = plot_demand_vs_ratio_scatter(
                demands_arr, ratios_arr, g_func, estimation_method
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # g(d) diagnostics
            g_diag = term.get_g_diagnostics()
            if g_diag:
                with st.expander("g(d) Estimation Details"):
                    st.json(g_diag)
            
            st.markdown("""
            **How to Read This Plot:**
            - Each point is a (cell, period) combination
            - X-axis: Demand (pickup count)
            - Y-axis: Service ratio (supply / demand)
            - Red line: Expected service ratio g(d) = E[Y|D=d]
            
            **Causal Fairness Interpretation:**
            - If points cluster tightly around g(d), R² is high → good causal fairness
            - If points scatter widely around g(d), R² is low → contextual bias may exist
            """)
    
    # --------------------------------------------------------------------------
    # TAB 4: Residual Analysis
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("Residual Analysis")
        
        residuals_arr = np.array(breakdown['components']['residuals'])
        demands_arr = np.array(breakdown['components']['demands'])
        
        if len(residuals_arr) == 0:
            st.info("No residual data available.")
        else:
            st.markdown("""
            Residuals $R = Y - g(D)$ represent the unexplained portion of service ratio.
            In a causally fair system, residuals should:
            - Be centered around zero
            - Show no pattern with demand
            - Be randomly distributed across space
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_res_dist = plot_residual_distribution(residuals_arr)
                st.plotly_chart(fig_res_dist, use_container_width=True)
            
            with col2:
                # Residual statistics
                st.markdown("**Residual Statistics**")
                st.write(f"- Mean: {np.mean(residuals_arr):.4f}")
                st.write(f"- Std: {np.std(residuals_arr):.4f}")
                st.write(f"- Min: {np.min(residuals_arr):.4f}")
                st.write(f"- Max: {np.max(residuals_arr):.4f}")
                st.write(f"- % Positive: {100 * np.mean(residuals_arr > 0):.1f}%")
                st.write(f"- % Negative: {100 * np.mean(residuals_arr < 0):.1f}%")
            
            # Residual vs demand
            fig_res_demand = plot_residual_vs_demand(demands_arr, residuals_arr)
            st.plotly_chart(fig_res_demand, use_container_width=True)
            
            st.markdown("""
            **Interpretation of Residual vs Demand Plot:**
            - If residuals show no pattern with demand, the g(d) estimation is appropriate
            - If residuals curve or trend, consider a different estimation method
            - Blue points: Over-served relative to expectation
            - Red points: Under-served relative to expectation
            """)
    
    # --------------------------------------------------------------------------
    # TAB 5: Method Comparison
    # --------------------------------------------------------------------------
    with tab5:
        st.subheader("g(d) Estimation Method Comparison")
        
        demands_arr = np.array(breakdown['components']['demands'])
        ratios_arr = np.array(breakdown['components']['ratios'])
        
        if len(demands_arr) == 0:
            st.info("No data available for method comparison.")
        else:
            st.markdown("""
            Compare different methods for estimating g(d) = E[Y|D=d]:
            
            | Method | Description | Best For |
            |--------|-------------|----------|
            | **Binning** | Group by demand bins, compute mean | Robust, non-parametric |
            | **Linear** | Fit Y = β₀ + β₁D | Simple trends |
            | **Polynomial** | Fit Y = Σβₖdᵏ | Curved relationships |
            | **Isotonic** | Monotonic fit | Ordered relationships |
            | **LOWESS** | Local smoothing | Complex patterns |
            """)
            
            fig_compare, r2_values = plot_method_comparison(demands_arr, ratios_arr)
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # R² comparison table
            st.markdown("**R² by Method:**")
            
            r2_df = pd.DataFrame([
                {'Method': method, 'R²': f"{r2:.4f}" if r2 is not None else "Error"}
                for method, r2 in r2_values.items()
            ])
            st.dataframe(r2_df, use_container_width=True)
            
            best_method = max(
                [(m, r) for m, r in r2_values.items() if r is not None],
                key=lambda x: x[1],
                default=(None, None)
            )
            if best_method[0]:
                st.success(f"📊 Best method: **{best_method[0]}** with R² = {best_method[1]:.4f}")
    
    # --------------------------------------------------------------------------
    # TAB 6: Statistics
    # --------------------------------------------------------------------------
    with tab6:
        st.subheader("Detailed Statistics")
        
        # Data statistics
        data_stats = breakdown['diagnostics']['data_stats']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Demand Statistics**")
            if 'demand' in data_stats:
                st.json(data_stats['demand'])
        
        with col2:
            st.markdown("**Supply Statistics**")
            if 'supply' in data_stats:
                st.json(data_stats['supply'])
        
        with col3:
            st.markdown("**Ratio Statistics**")
            if 'ratio' in data_stats:
                st.json(data_stats['ratio'])
        
        # Alignment issues
        alignment_issues = breakdown['diagnostics'].get('alignment_issues', [])
        if alignment_issues:
            st.markdown("**Data Alignment Issues:**")
            for issue in alignment_issues:
                if issue.startswith('ERROR'):
                    st.error(issue)
                elif issue.startswith('WARNING'):
                    st.warning(issue)
                else:
                    st.info(issue)
        
        # Per-period data table
        st.markdown("**Per-Period Data**")
        per_period_data = breakdown['components']['per_period_data']
        
        if per_period_data:
            df = pd.DataFrame(per_period_data)
            
            st.dataframe(
                df.style.format({
                    'r_squared': '{:.4f}',
                    'mean_demand': '{:.2f}',
                    'mean_ratio': '{:.4f}',
                    'mean_expected': '{:.4f}',
                    'var_ratio': '{:.4f}',
                    'var_residual': '{:.4f}',
                }),
                use_container_width=True,
                height=400,
            )
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="causal_fairness_results.csv",
                mime="text/csv",
            )
        
        # Configuration used
        st.markdown("**Configuration Used**")
        st.json(breakdown['diagnostics']['config'])
    
    # --------------------------------------------------------------------------
    # TAB 7: Gradient Verification
    # --------------------------------------------------------------------------
    with tab7:
        render_gradient_verification_tab(term)
    
    # --------------------------------------------------------------------------
    # TAB 8: Estimation Method Gradient Tests
    # --------------------------------------------------------------------------
    with tab8:
        render_method_gradient_tests_tab()
    
    # --------------------------------------------------------------------------
    # TAB 9: Soft Cell Assignment
    # --------------------------------------------------------------------------
    with tab9:
        render_soft_cell_assignment_tab(term, breakdown)
    
    # ==========================================================================
    # FOOTER
    # ==========================================================================
    
    st.divider()
    st.markdown("""
    ---
    **FAMAIL Causal Fairness Dashboard** | Version 1.1.0 (Soft Cell Assignment)  
    Based on counterfactual fairness principles and R² coefficient of determination.  
    *With soft cell assignment and frozen g(d) for gradient-based trajectory optimization.*
    """)


if __name__ == "__main__":
    main()
