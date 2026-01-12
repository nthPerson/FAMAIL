"""
Streamlit Dashboard for Spatial Fairness Term Validation.

This dashboard provides an interactive interface for:
- Configuring spatial fairness computation parameters
- Visualizing Gini coefficients and Lorenz curves
- Analyzing temporal patterns in spatial fairness
- Exploring service rate distributions across the grid

Usage:
    streamlit run dashboard.py

Requirements:
    pip install streamlit pandas plotly matplotlib seaborn
"""

import sys
import os
from pathlib import Path
import pickle
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent
OBJECTIVE_FUNCTION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = OBJECTIVE_FUNCTION_DIR.parent
sys.path.insert(0, str(OBJECTIVE_FUNCTION_DIR))

from config import SpatialFairnessConfig, WEEKDAYS_JULY, WEEKDAYS_AUGUST, WEEKDAYS_SEPTEMBER, WEEKDAYS_TOTAL
from term import SpatialFairnessTerm
from utils import (
    compute_gini,
    compute_lorenz_curve,
    aggregate_counts_by_period,
    get_unique_periods,
    get_data_statistics,
    validate_pickup_dropoff_data,
    load_active_taxis_data,
    get_active_taxis_statistics,
)
# from spatial_fairness.config import SpatialFairnessConfig
# from spatial_fairness.term import SpatialFairnessTerm
# from spatial_fairness.utils import (
#     compute_gini,
#     compute_lorenz_curve,
#     aggregate_counts_by_period,
#     get_unique_periods,
#     get_data_statistics,
#     validate_pickup_dropoff_data,
# )


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Spatial Fairness Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data(filepath: str) -> Dict:
    """Load and cache pickup/dropoff counts data."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_default_data_path() -> Optional[str]:
    """Find the default data file path."""
    possible_paths = [
        PROJECT_ROOT / "source_data" / "pickup_dropoff_counts.pkl",
        Path("source_data/pickup_dropoff_counts.pkl"),
        Path("../source_data/pickup_dropoff_counts.pkl"),
        Path("../../source_data/pickup_dropoff_counts.pkl"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def plot_lorenz_curve(dsr_values: np.ndarray, asr_values: np.ndarray, title: str = "Lorenz Curves") -> go.Figure:
    """Create a Lorenz curve plot for DSR and ASR."""
    from spatial_fairness.utils import compute_lorenz_curve
    
    x_dsr, y_dsr = compute_lorenz_curve(dsr_values)
    x_asr, y_asr = compute_lorenz_curve(asr_values)
    
    fig = go.Figure()
    
    # Perfect equality line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Equality',
        line=dict(dash='dash', color='gray'),
    ))
    
    # Departure (Pickup) Lorenz curve
    fig.add_trace(go.Scatter(
        x=x_dsr, y=y_dsr,
        mode='lines',
        name=f'Pickups (Gini={compute_gini(dsr_values):.3f})',
        line=dict(color='blue'),
    ))
    
    # Arrival (Dropoff) Lorenz curve
    fig.add_trace(go.Scatter(
        x=x_asr, y=y_asr,
        mode='lines',
        name=f'Dropoffs (Gini={compute_gini(asr_values):.3f})',
        line=dict(color='red'),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Cumulative Share of Grid Cells",
        yaxis_title="Cumulative Share of Service",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400,
    )
    
    return fig


def plot_gini_over_time(per_period_data: list, period_type: str) -> go.Figure:
    """Plot Gini coefficients over time periods."""
    df = pd.DataFrame(per_period_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df['gini_arrival'],
        mode='lines+markers',
        name='Gini (Dropoffs)',
        line=dict(color='red'),
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df['gini_departure'],
        mode='lines+markers',
        name='Gini (Pickups)',
        line=dict(color='blue'),
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df['fairness'],
        mode='lines+markers',
        name='Spatial Fairness',
        line=dict(color='green', width=3),
    ))
    
    fig.update_layout(
        title=f"Gini Coefficients and Fairness Over Time ({period_type})",
        xaxis_title="Period Index",
        yaxis_title="Value",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400,
    )
    
    return fig


def plot_service_heatmap(grid_data: np.ndarray, title: str) -> go.Figure:
    """Create a heatmap of service distribution.
    
    Grid orientation:
    - X axis (latitude-based, 48 cells): displayed vertically, X=0 at bottom
    - Y axis (longitude-based, 90 cells): displayed horizontally, Y=0 at left
    """
    # grid_data shape is (x_dim, y_dim) = (48, 90)
    # For px.imshow: rows = vertical axis, cols = horizontal axis
    # We want X vertical and Y horizontal, so NO transpose needed
    # Flip vertically so X=0 is at bottom (geographic convention: latitude increases upward)
    fig = px.imshow(
        np.flipud(grid_data),  # Flip so X=0 at bottom, Y stays left-to-right
        labels=dict(x="Y Grid (Longitude)", y="X Grid (Latitude)", color="Count"),
        title=title,
        color_continuous_scale="YlOrRd",
        aspect="auto",  # Allow non-square display since 48x90 grid
    )
    
    # Update Y axis (now showing X grid) to show correct labels after flip
    x_dim = grid_data.shape[0]
    fig.update_yaxes(
        tickmode='array',
        tickvals=list(range(0, x_dim, max(1, x_dim // 8))),
        ticktext=[str(x_dim - 1 - i) for i in range(0, x_dim, max(1, x_dim // 8))]
    )
    
    fig.update_layout(height=600)
    return fig


def plot_gini_heatmap(grid_data: np.ndarray, title: str, color_scale: str = "RdYlGn") -> go.Figure:
    """Create a heatmap of Gini coefficients or fairness values.
    
    Grid orientation:
    - X axis (latitude-based, 48 cells): displayed vertically, X=0 at bottom
    - Y axis (longitude-based, 90 cells): displayed horizontally, Y=0 at left
    """
    # grid_data shape is (x_dim, y_dim) = (48, 90)
    # Flip vertically so X=0 is at bottom
    fig = px.imshow(
        np.flipud(grid_data),  # Flip so X=0 at bottom, Y stays left-to-right
        labels=dict(x="Y Grid (Longitude)", y="X Grid (Latitude)", color="Value"),
        title=title,
        color_continuous_scale=color_scale,
        aspect="auto",  # Allow non-square display since 48x90 grid
        zmin=0,
        zmax=1,
    )
    
    # Update Y axis (now showing X grid) to show correct labels after flip
    x_dim = grid_data.shape[0]
    fig.update_yaxes(
        tickmode='array',
        tickvals=list(range(0, x_dim, max(1, x_dim // 8))),
        ticktext=[str(x_dim - 1 - i) for i in range(0, x_dim, max(1, x_dim // 8))]
    )
    
    fig.update_layout(height=600)
    return fig


def plot_gini_distribution(gini_values: list, title: str) -> go.Figure:
    """Plot histogram of Gini coefficients across periods."""
    fig = px.histogram(
        x=gini_values,
        nbins=30,
        labels={"x": "Gini Coefficient", "y": "Count"},
        title=title,
    )
    
    fig.add_vline(
        x=np.mean(gini_values),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {np.mean(gini_values):.3f}",
    )
    
    fig.update_layout(height=300)
    return fig


def plot_hourly_pattern(per_period_data: list) -> go.Figure:
    """Plot hourly pattern of spatial fairness (for hourly aggregation)."""
    df = pd.DataFrame(per_period_data)
    
    if 'period' not in df.columns:
        return None
    
    # Extract hour from period if it's hourly
    try:
        hours = [p[0] if isinstance(p, tuple) else p for p in df['period']]
        df['hour'] = hours
        
        # Group by hour across days
        hourly_avg = df.groupby('hour').agg({
            'fairness': 'mean',
            'gini_arrival': 'mean',
            'gini_departure': 'mean',
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour'],
            y=hourly_avg['fairness'],
            mode='lines+markers',
            name='Spatial Fairness',
            line=dict(color='green', width=3),
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour'],
            y=hourly_avg['gini_arrival'],
            mode='lines',
            name='Gini (Dropoffs)',
            line=dict(color='red', dash='dot'),
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour'],
            y=hourly_avg['gini_departure'],
            mode='lines',
            name='Gini (Pickups)',
            line=dict(color='blue', dash='dot'),
        ))
        
        fig.update_layout(
            title="Average Spatial Fairness by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Value",
            xaxis=dict(tickmode='linear', dtick=2),
            height=400,
        )
        
        return fig
    except Exception:
        return None


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.title("📊 Spatial Fairness Term Dashboard")
    st.markdown("""
    This dashboard helps validate and analyze the **Spatial Fairness Term** ($F_{\\text{spatial}}$)
    of the FAMAIL objective function. The term measures equality of taxi service distribution
    using the Gini coefficient.
    
    $$F_{\\text{spatial}} = 1 - \\frac{1}{2|P|} \\sum_{p \\in P} (G_a^p + G_d^p)$$
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data file selection
    default_path = get_default_data_path()
    data_path = st.sidebar.text_input(
        "Data File Path",
        value=default_path or "source_data/pickup_dropoff_counts.pkl",
        help="Path to pickup_dropoff_counts.pkl file"
    )
    
    if not os.path.exists(data_path):
        st.error(f"Data file not found: {data_path}")
        st.info("Please provide a valid path to pickup_dropoff_counts.pkl")
        return
    
    # Load data
    try:
        data = load_data(data_path)
        st.sidebar.success(f"✅ Loaded {len(data):,} records")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Configuration options
    st.sidebar.subheader("Term Configuration")
    
    period_type = st.sidebar.selectbox(
        "Period Type",
        options=["hourly", "daily", "time_bucket", "all"],
        index=0,
        help=(
            "Temporal aggregation granularity for computing Gini coefficients.\n\n"
            "• **hourly**: Compute fairness for each hour (24 periods/day)\n"
            "• **daily**: Compute fairness for each day\n"
            "• **time_bucket**: 5-minute intervals (288 periods/day, finest)\n"
            "• **all**: Single aggregate across all time"
        )
    )
    
    include_zero_cells = st.sidebar.checkbox(
        "Include Zero Cells",
        value=False,
        help=(
            "Whether to include grid cells with zero activity in Gini calculation.\n\n"
            "• **Checked**: All 4,320 cells included (more conservative fairness estimate)\n"
            "• **Unchecked**: Only active cells included (focuses on areas with service)"
        )
    )
    
    # Taxi count configuration
    st.sidebar.subheader("🚕 Taxi Count Configuration ($$N^p$$)")
    
    st.sidebar.markdown(
        """
        The departure service rate formula is: $$DSR = pickups / (N^p × T)$$
        
        $$N^p$$ can be either:
        - **Constant**: Same value for all cells (original approach)
        - **Active Taxis Lookup**: Dynamic per-cell values from pre-computed dataset
        """,
        help="This affects how service rates are normalized across different neighborhoods"
    )
    
    taxi_count_source = st.sidebar.radio(
        "Taxi Count Source",
        options=["constant", "active_taxis_lookup"],
        index=0,
        format_func=lambda x: "Constant" if x == "constant" else "Active Taxis Lookup",
        help=(
            "How to determine N^p (number of taxis) in service rate calculation.\n\n"
            "• **Constant**: Use a fixed number for all cells (e.g., 50 taxis)\n"
            "• **Active Taxis Lookup**: Use pre-computed counts of taxis in each neighborhood"
        )
    )
    
    # Conditional inputs based on taxi_count_source
    num_taxis = 50  # default
    active_taxis_data = None
    active_taxis_path = None
    fallback = 1  # default fallback value
    
    if taxi_count_source == "constant":
        num_taxis = st.sidebar.number_input(
            "Number of Taxis (N^p)",
            min_value=1,
            max_value=1000,
            value=50,
            help=(
                "Fixed number of taxis used to normalize service rates.\n\n"
                "This value is used as N^p in DSR = pickups / (N^p × T) for all cells."
            )
        )
    else:
        # Active taxis lookup mode
        default_active_path = PROJECT_ROOT / "source_data"
        
        # Try to find available active_taxis files
        active_files = []
        if default_active_path.exists():
            active_files = list(default_active_path.glob("active_taxis_*.pkl"))
        
        if active_files:
            # Map period_type to expected file suffix
            period_file_map = {
                "hourly": "hourly",
                "daily": "daily", 
                "time_bucket": "time_bucket",
                "all": "all"
            }
            
            # Find matching file for selected period_type
            expected_suffix = period_file_map.get(period_type, "hourly")
            matching_files = [f for f in active_files if expected_suffix in f.name]
            
            if matching_files:
                default_idx = 0
            else:
                matching_files = active_files
                default_idx = 0
            
            active_taxis_path = st.sidebar.selectbox(
                "Active Taxis Dataset",
                options=[str(f) for f in matching_files],
                index=default_idx,
                help=(
                    f"Select the active_taxis dataset file.\n\n"
                    f"⚠️ Ensure the period type matches: currently selected '{period_type}'"
                )
            )
        else:
            active_taxis_path = st.sidebar.text_input(
                "Active Taxis Dataset Path",
                value=str(PROJECT_ROOT / "../../source_data" / f"active_taxis_5x5_{period_type}.pkl"),
                help="Path to the active_taxis_*.pkl file"
            )
        
        # Load active_taxis data
        if active_taxis_path and os.path.exists(active_taxis_path):
            try:
                active_taxis_data = load_active_taxis_data(active_taxis_path)
                active_stats = get_active_taxis_statistics(active_taxis_data)
                st.sidebar.success(f"✅ Loaded {len(active_taxis_data):,} active_taxis records")
                st.sidebar.caption(
                    f"Count range: {active_stats['count_stats']['min']}-{active_stats['count_stats']['max']}, "
                    f"Mean: {active_stats['count_stats']['mean']:.1f}"
                )
            except Exception as e:
                st.sidebar.error(f"Error loading active_taxis: {e}")
                active_taxis_data = None
        elif active_taxis_path:
            st.sidebar.warning(f"⚠️ File not found: {active_taxis_path}")
        
        # Fallback value
        fallback = st.sidebar.number_input(
            "Fallback Value",
            min_value=1,
            max_value=100,
            value=1,
            help=(
                "Value to use when active_taxis lookup returns 0.\n\n"
                "This prevents division by zero in service rate calculation."
            )
        )
    
    # Temporal coverage
    st.sidebar.subheader("📅 Temporal Coverage")
    
    num_days_option = st.sidebar.selectbox(
        "Dataset Time Period",
        options=["july", "august", "september", "all"],
        index=0,
        format_func=lambda x: {
            "july": f"July 2016 ({WEEKDAYS_JULY} weekdays)",
            "august": f"August 2016 ({WEEKDAYS_AUGUST} weekdays)",
            "september": f"September 2016 ({WEEKDAYS_SEPTEMBER} weekdays)",
            "all": f"All Months ({WEEKDAYS_TOTAL} weekdays)"
        }[x],
        help=(
            "Select which month(s) of data you're analyzing.\n\n"
            "This affects the num_days parameter used in temporal normalization."
        )
    )
    
    num_days = {
        "july": float(WEEKDAYS_JULY),
        "august": float(WEEKDAYS_AUGUST),
        "september": float(WEEKDAYS_SEPTEMBER),
        "all": float(WEEKDAYS_TOTAL)
    }[num_days_option]
    
    st.sidebar.caption(f"Using num_days = {num_days}")
    
    # Day filter
    st.sidebar.subheader("Filters")
    day_options = [1, 2, 3, 4, 5, 6]
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    selected_days = st.sidebar.multiselect(
        "Days to Include",
        options=day_options,
        default=day_options,
        format_func=lambda x: day_labels[x-1],
    )
    
    days_filter = selected_days if len(selected_days) < 6 else None
    
    # Create configuration
    config = SpatialFairnessConfig(
        period_type=period_type,
        grid_dims=(48, 90),
        taxi_count_source=taxi_count_source,
        num_taxis=num_taxis,
        active_taxis_data_path=active_taxis_path if taxi_count_source == "active_taxis_lookup" else None,
        active_taxis_fallback=fallback if taxi_count_source == "active_taxis_lookup" else 1,
        num_days=num_days,
        include_zero_cells=include_zero_cells,
        days_filter=days_filter,
        verbose=False,
    )
    
    # Create term
    term = SpatialFairnessTerm(config)
    
    # Compute results
    auxiliary_data = {'pickup_dropoff_counts': data}
    if active_taxis_data is not None:
        auxiliary_data['active_taxis_counts'] = active_taxis_data
    
    with st.spinner("Computing spatial fairness..."):
        breakdown = term.compute_with_breakdown({}, auxiliary_data)
    
    # ==========================================================================
    # RESULTS DISPLAY
    # ==========================================================================
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Spatial Fairness",
            f"{breakdown['value']:.4f}",
            help="Higher is better (1.0 = perfect equality)"
        )
    
    with col2:
        st.metric(
            "Avg Gini (Pickups)",
            f"{breakdown['components']['avg_gini_departure']:.4f}",
            help="Inequality in pickup distribution"
        )
    
    with col3:
        st.metric(
            "Avg Gini (Dropoffs)",
            f"{breakdown['components']['avg_gini_arrival']:.4f}",
            help="Inequality in dropoff distribution"
        )
    
    with col4:
        st.metric(
            "Periods Analyzed",
            f"{breakdown['statistics']['n_periods']}",
            help="Number of time periods"
        )
    
    # Computation time
    st.caption(f"⏱️ Computation time: {breakdown['diagnostics']['computation_time_ms']:.1f} ms")
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Temporal Analysis",
        "🗺️ Spatial Distribution",
        "📊 Lorenz Curves",
        "📉 Statistics",
        "🔍 Raw Data",
    ])
    
    # Tab 1: Temporal Analysis
    with tab1:
        st.subheader("Temporal Patterns")
        
        per_period_data = breakdown['components']['per_period_data']
        
        if len(per_period_data) > 1:
            # Gini over time
            fig_time = plot_gini_over_time(per_period_data, period_type)
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Hourly pattern (if applicable)
            if period_type == "hourly":
                fig_hourly = plot_hourly_pattern(per_period_data)
                if fig_hourly:
                    st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Gini distributions
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist_pickup = plot_gini_distribution(
                    breakdown['components']['per_period_gini_departure'],
                    "Distribution of Pickup Gini Coefficients"
                )
                st.plotly_chart(fig_hist_pickup, use_container_width=True)
            
            with col2:
                fig_hist_dropoff = plot_gini_distribution(
                    breakdown['components']['per_period_gini_arrival'],
                    "Distribution of Dropoff Gini Coefficients"
                )
                st.plotly_chart(fig_hist_dropoff, use_container_width=True)
        else:
            st.info("Only one period available. Select finer temporal granularity for temporal analysis.")
    
    # Tab 2: Spatial Distribution
    with tab2:
        st.subheader("Spatial Distribution of Service")
        
        # Get heatmap data
        heatmap_data = term.get_spatial_heatmap_data(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pickup = plot_service_heatmap(heatmap_data['pickups'], "Pickup Distribution")
            st.plotly_chart(fig_pickup, use_container_width=True)
        
        with col2:
            fig_dropoff = plot_service_heatmap(heatmap_data['dropoffs'], "Dropoff Distribution")
            st.plotly_chart(fig_dropoff, use_container_width=True)
        
        # Total service heatmap
        fig_total = plot_service_heatmap(heatmap_data['total'], "Total Service (Pickups + Dropoffs)")
        st.plotly_chart(fig_total, use_container_width=True)
        
        # Grid statistics
        st.subheader("Grid Statistics")
        active_cells = np.sum(heatmap_data['total'] > 0)
        total_cells = heatmap_data['total'].size
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Cells", f"{active_cells:,}")
        with col2:
            st.metric("Total Cells", f"{total_cells:,}")
        with col3:
            st.metric("Activity Rate", f"{100*active_cells/total_cells:.1f}%")
        
        # Per-cell Gini coefficients and fairness
        st.divider()
        st.subheader("Spatial Distribution of Fairness Metrics")
        
        st.markdown("""
        These heatmaps show Gini coefficients and spatial fairness computed **per grid cell** across all time periods.
        - **Lower Gini** (green) = more equal service distribution over time for that cell
        - **Higher Gini** (red) = more unequal service distribution over time for that cell
        - **Higher Fairness** (green) = better overall fairness (1 - average Gini)
        """)
        
        # Compute per-cell Gini coefficients
        per_period_data = breakdown['components']['per_period_data']
        
        if len(per_period_data) > 1:
            # Initialize grids for per-cell Gini computation
            x_dim, y_dim = config.grid_dims
            gini_pickup_grid = np.zeros((x_dim, y_dim))
            gini_dropoff_grid = np.zeros((x_dim, y_dim))
            fairness_grid = np.zeros((x_dim, y_dim))
            
            # For each cell, compute Gini over all periods
            pickups_agg, dropoffs_agg = aggregate_counts_by_period(
                data,
                period_type=period_type,
                days_filter=config.days_filter,
                time_filter=config.time_filter,
            )
            
            x_offset = 1 if config.data_is_one_indexed else 0
            y_offset = 1 if config.data_is_one_indexed else 0
            
            for x in range(x_dim):
                for y in range(y_dim):
                    cell = (x + x_offset, y + y_offset)
                    
                    # Collect values for this cell across all periods
                    pickup_values = []
                    dropoff_values = []
                    
                    for period_info in per_period_data:
                        period = period_info['period']
                        pickup_key = (cell, period)
                        dropoff_key = (cell, period)
                        
                        pickup_values.append(pickups_agg.get(pickup_key, 0))
                        dropoff_values.append(dropoffs_agg.get(dropoff_key, 0))
                    
                    # Compute Gini for this cell
                    if len(pickup_values) > 0:
                        gini_pickup_grid[x, y] = compute_gini(np.array(pickup_values))
                        gini_dropoff_grid[x, y] = compute_gini(np.array(dropoff_values))
                        fairness_grid[x, y] = 1.0 - 0.5 * (gini_pickup_grid[x, y] + gini_dropoff_grid[x, y])
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig_gini_pickup = plot_gini_heatmap(
                    gini_pickup_grid,
                    "Per-Cell Gini Coefficient: Departures (Over Time)",
                    color_scale="RdYlGn_r"  # Reverse so low Gini is green
                )
                st.plotly_chart(fig_gini_pickup, use_container_width=True)
            
            with col2:
                fig_gini_dropoff = plot_gini_heatmap(
                    gini_dropoff_grid,
                    "Per-Cell Gini Coefficient: Arrivals (Over Time)",
                    color_scale="RdYlGn_r"  # Reverse so low Gini is green
                )
                st.plotly_chart(fig_gini_dropoff, use_container_width=True)
            
            # Fairness heatmap
            fig_fairness = plot_gini_heatmap(
                fairness_grid,
                "Per-Cell Spatial Fairness (Over Time)",
                color_scale="RdYlGn"  # Higher fairness is green
            )
            st.plotly_chart(fig_fairness, use_container_width=True)
            
            # Statistics for per-cell metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Cell Gini (Pickups)", f"{np.mean(gini_pickup_grid):.4f}")
            with col2:
                st.metric("Mean Cell Gini (Dropoffs)", f"{np.mean(gini_dropoff_grid):.4f}")
            with col3:
                st.metric("Mean Cell Fairness", f"{np.mean(fairness_grid):.4f}")
        else:
            st.info("Per-cell Gini computation requires multiple time periods. Select finer temporal granularity.")
          
    # Tab 3: Lorenz Curves
    with tab3:
        st.subheader("Lorenz Curves")
        
        st.markdown("""
        The **Lorenz curve** shows the cumulative distribution of service across grid cells.
        The further the curve bows from the diagonal (perfect equality), the higher the inequality.
        The **Gini coefficient** equals twice the area between the curve and the diagonal.
        """)
        
        # Select period for Lorenz curve
        per_period_data = breakdown['components']['per_period_data']
        
        if len(per_period_data) > 1:
            period_idx = st.slider(
                "Select Period",
                0, len(per_period_data) - 1, 0,
                help="Choose a specific time period to visualize"
            )
            
            selected_period = per_period_data[period_idx]
        else:
            selected_period = per_period_data[0] if per_period_data else None
        
        if selected_period:
            # Get data for this period
            period_result = term.compute_for_single_period(data, selected_period['period'])
            
            # Show metrics for this period
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Period Fairness", f"{period_result['fairness']:.4f}")
            with col2:
                st.metric("Pickup Gini", f"{period_result['gini_departure']:.4f}")
            with col3:
                st.metric("Dropoff Gini", f"{period_result['gini_arrival']:.4f}")
            
            # Lorenz curves
            fig_lorenz = plot_lorenz_curve(
                period_result['dsr_values'],
                period_result['asr_values'],
                f"Lorenz Curves for Period {selected_period['period']}"
            )
            st.plotly_chart(fig_lorenz, use_container_width=True)
        
        # Overall Lorenz curve (aggregated)
        st.subheader("Overall Lorenz Curves (Aggregated)")
        
        # Aggregate all data
        pickups_agg, dropoffs_agg = aggregate_counts_by_period(data, period_type="all")
        periods = get_unique_periods(pickups_agg, dropoffs_agg)
        
        if periods:
            from spatial_fairness.utils import compute_service_rates_for_period, compute_period_duration_days
            
            period = periods[0]
            period_duration = compute_period_duration_days(period, "all", num_days)
            
            dsr_all, asr_all = compute_service_rates_for_period(
                pickups_agg, dropoffs_agg, period,
                config.grid_dims, num_taxis, period_duration,
                include_zero_cells, True, 0
            )
            
            fig_lorenz_all = plot_lorenz_curve(dsr_all, asr_all, "Overall Lorenz Curves (All Data)")
            st.plotly_chart(fig_lorenz_all, use_container_width=True)
    
    # Tab 4: Statistics
    with tab4:
        st.subheader("Detailed Statistics")
        
        # Taxi count source info
        if taxi_count_source == "active_taxis_lookup":
            st.info(
                f"🚕 **Using Active Taxis Lookup** - N^p varies by cell and period.\n\n"
                f"Dataset: `{os.path.basename(active_taxis_path) if active_taxis_path else 'N/A'}`"
            )
            
            if breakdown['statistics'].get('active_taxis_stats'):
                at_stats = breakdown['statistics']['active_taxis_stats']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", f"{at_stats.get('total_keys', 0):,}")
                with col2:
                    st.metric("Mean Active Taxis", f"{at_stats['count_stats']['mean']:.1f}")
                with col3:
                    st.metric("Max Active Taxis", f"{at_stats['count_stats']['max']}")
                with col4:
                    st.metric("Zero Count Cells", f"{at_stats['count_stats']['zero_count']:,}")
        else:
            st.info(f"🚕 **Using Constant Taxi Count** - N^p = {num_taxis} for all cells")
        
        st.divider()
        
        # Gini statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Pickup (Departure) Gini Statistics**")
            pickup_stats = breakdown['statistics']['gini_departure_stats']
            st.json(pickup_stats)
        
        with col2:
            st.markdown("**Dropoff (Arrival) Gini Statistics**")
            dropoff_stats = breakdown['statistics']['gini_arrival_stats']
            st.json(dropoff_stats)
        
        # Data statistics
        st.markdown("**Data Statistics**")
        data_stats = breakdown['statistics']['data_stats']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Pickup Statistics**")
            st.json(data_stats.get('pickup_stats', {}))
        
        with col2:
            st.markdown("**Dropoff Statistics**")
            st.json(data_stats.get('dropoff_stats', {}))
        
        with col3:
            st.markdown("**Spatial Coverage**")
            st.json(data_stats.get('spatial', {}))
        
        # Configuration used
        st.markdown("**Configuration Used**")
        st.json(breakdown['diagnostics']['config'])
    
    # Tab 5: Raw Data
    with tab5:
        st.subheader("Per-Period Data")
        
        per_period_data = breakdown['components']['per_period_data']
        
        # Convert to DataFrame
        df = pd.DataFrame(per_period_data)
        
        # Format period column for display
        if 'period' in df.columns:
            df['period_str'] = df['period'].astype(str)
        
        # Select columns to display
        display_cols = ['period', 'fairness', 'gini_arrival', 'gini_departure', 'n_cells']
        available_cols = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            df[available_cols].style.format({
                'fairness': '{:.4f}',
                'gini_arrival': '{:.4f}',
                'gini_departure': '{:.4f}',
            }),
            use_container_width=True,
            height=400,
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="spatial_fairness_results.csv",
            mime="text/csv",
        )
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **FAMAIL Spatial Fairness Dashboard** | Version 1.0.0  
    Based on Su et al. (2018) "Uncovering Spatial Inequality in Taxi Services"
    """)


if __name__ == "__main__":
    main()
