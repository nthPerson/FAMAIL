"""
Streamlit Dashboard for Trajectory Fidelity Term Validation.

This dashboard provides an interactive interface for:
- Loading and exploring trajectory datasets
- Configuring discriminator model and fidelity parameters
- Computing fidelity scores with visualizations
- Analyzing per-driver and per-trajectory statistics
- Comparing original vs edited trajectories
- Verifying differentiability for gradient-based optimization

Usage:
    streamlit run dashboard.py

Requirements:
    pip install streamlit pandas plotly matplotlib seaborn torch
"""

import sys
import os
from pathlib import Path
import pickle
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import json

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directories to path
SCRIPT_DIR = Path(__file__).resolve().parent
OBJECTIVE_FUNCTION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = OBJECTIVE_FUNCTION_DIR.parent
sys.path.insert(0, str(OBJECTIVE_FUNCTION_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "discriminator" / "model"))

from config import (
    FidelityConfig,
    DEFAULT_CONFIG,
    HIGH_THRESHOLD_CONFIG,
    CONSERVATIVE_CONFIG,
)
from term import FidelityTerm
from utils import (
    load_discriminator,
    load_trajectory_data,
    get_trajectory_statistics,
    create_trajectory_pairs,
    compute_fidelity_scores,
    aggregate_fidelity_scores,
    compute_per_driver_statistics,
    compute_score_histogram,
    compute_length_correlation,
    get_model_info,
    DifferentiableFidelity,
    verify_fidelity_gradient,
    extract_trajectory_features,
    prepare_trajectory_batch,
)

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Fidelity Term Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CACHED DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_trajectories(filepath: str) -> Dict[Any, List[List[List[float]]]]:
    """Load and cache trajectory data."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


@st.cache_resource(show_spinner=False)
def load_model_cached(checkpoint_path: str, use_gpu: bool = True):
    """Load and cache discriminator model."""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    return load_discriminator(checkpoint_path, device=device)


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    st.title("🎯 Trajectory Fidelity Term Dashboard")
    st.markdown("""
    Interactive validation and exploration of the **Trajectory Fidelity Term** ($F_{\\text{fidelity}}$).
    
    The fidelity term measures how authentic edited trajectories appear compared to original expert trajectories
    using a pre-trained ST-SiameseNet discriminator.
    """)
    
    # ==========================================================================
    # SIDEBAR CONFIGURATION
    # ==========================================================================
    
    st.sidebar.header("⚙️ Configuration")
    
    # Data paths
    st.sidebar.subheader("📂 Data Sources")
    
    default_traj_path = str(PROJECT_ROOT / "source_data" / "all_trajs.pkl")
    trajectory_path = st.sidebar.text_input(
        "Trajectory Data Path",
        value=default_traj_path,
        help="Path to all_trajs.pkl or similar trajectory file"
    )
    
    # For demo purposes, we'll use the same file for both edited and original
    use_same_for_both = st.sidebar.checkbox(
        "Use same trajectories as edited & original (demo mode)",
        value=True,
        help="In demo mode, compares trajectories with themselves (should give high fidelity)"
    )
    
    if not use_same_for_both:
        edited_path = st.sidebar.text_input(
            "Edited Trajectories Path",
            value=trajectory_path,
            help="Path to edited trajectory file"
        )
    else:
        edited_path = trajectory_path
    
    # Model configuration
    st.sidebar.subheader("🤖 Discriminator Model")
    
    # Find available checkpoints
    checkpoint_dir = PROJECT_ROOT / "discriminator" / "model" / "checkpoints"
    available_checkpoints = []
    if checkpoint_dir.exists():
        for exp_dir in sorted(checkpoint_dir.iterdir()):
            if exp_dir.is_dir():
                best_pt = exp_dir / "best.pt"
                if best_pt.exists():
                    available_checkpoints.append(str(best_pt))
    
    if available_checkpoints:
        checkpoint_path = st.sidebar.selectbox(
            "Checkpoint",
            options=available_checkpoints,
            index=len(available_checkpoints) - 1,  # Select latest
            format_func=lambda x: Path(x).parent.name
        )
    else:
        checkpoint_path = st.sidebar.text_input(
            "Checkpoint Path",
            value=str(DEFAULT_CONFIG.checkpoint_path),
        )
    
    use_gpu = st.sidebar.checkbox(
        "Use GPU",
        value=True,
        disabled=not TORCH_AVAILABLE or not torch.cuda.is_available()
    )
    
    # Computation settings
    st.sidebar.subheader("🔧 Computation Settings")
    
    mode = st.sidebar.selectbox(
        "Pairing Mode",
        options=["same_agent", "paired", "batch"],
        index=0,
        help="""
        - same_agent: Compare each edited trajectory with original from same driver
        - paired: Compare by trajectory index
        - batch: Compare against random sample of originals
        """
    )
    
    aggregation = st.sidebar.selectbox(
        "Aggregation Method",
        options=["mean", "min", "threshold", "weighted"],
        index=0,
        help="""
        - mean: Average of all scores
        - min: Minimum score (worst-case)
        - threshold: Fraction above threshold
        - weighted: Length-weighted average
        """
    )
    
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Threshold for classifying as authentic"
    )
    
    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=1,
        max_value=256,
        value=32,
        help="Number of pairs per batch"
    )
    
    max_trajectory_length = st.sidebar.number_input(
        "Max Trajectory Length",
        min_value=100,
        max_value=2000,
        value=1000,
        help="Maximum states per trajectory"
    )
    
    # Driver/trajectory filtering
    st.sidebar.subheader("🔍 Filtering")
    
    max_drivers = st.sidebar.number_input(
        "Max Drivers to Process",
        min_value=1,
        max_value=50,
        value=50,
        help="Limit number of drivers (for faster testing)"
    )
    
    max_trajectories_per_driver = st.sidebar.number_input(
        "Max Trajectories per Driver",
        min_value=1,
        max_value=500,
        value=100,
        help="Limit trajectories per driver"
    )
    
    min_trajectory_length = st.sidebar.number_input(
        "Min Trajectory Length",
        min_value=2,
        max_value=50,
        value=2,
        help="Minimum states for valid trajectory"
    )
    
    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    
    # Validate paths
    if not Path(trajectory_path).exists():
        st.error(f"❌ Trajectory file not found: {trajectory_path}")
        st.stop()
    
    if not Path(checkpoint_path).exists():
        # Try relative path
        abs_checkpoint = PROJECT_ROOT / checkpoint_path
        if abs_checkpoint.exists():
            checkpoint_path = str(abs_checkpoint)
        else:
            st.error(f"❌ Checkpoint not found: {checkpoint_path}")
            st.stop()
    
    # Load trajectory data
    try:
        with st.spinner("Loading trajectories..."):
            original_trajectories = load_trajectories(trajectory_path)
            
            # Apply filtering
            filtered_original = {}
            for i, (driver_id, trajs) in enumerate(original_trajectories.items()):
                if i >= max_drivers:
                    break
                filtered_trajs = [
                    t for t in trajs[:max_trajectories_per_driver]
                    if len(t) >= min_trajectory_length
                ]
                if filtered_trajs:
                    filtered_original[driver_id] = filtered_trajs
            
            original_trajectories = filtered_original
            
            if use_same_for_both:
                edited_trajectories = original_trajectories
            else:
                edited_trajectories = load_trajectories(edited_path)
                # Apply same filtering
                filtered_edited = {}
                for i, (driver_id, trajs) in enumerate(edited_trajectories.items()):
                    if i >= max_drivers:
                        break
                    filtered_trajs = [
                        t for t in trajs[:max_trajectories_per_driver]
                        if len(t) >= min_trajectory_length
                    ]
                    if filtered_trajs:
                        filtered_edited[driver_id] = filtered_trajs
                edited_trajectories = filtered_edited
        
        st.sidebar.success(f"✅ Loaded {len(original_trajectories)} drivers")
    except Exception as e:
        st.error(f"Failed to load trajectories: {e}")
        st.stop()
    
    # Load model
    if not TORCH_AVAILABLE:
        st.error("❌ PyTorch is required for fidelity computation")
        st.stop()
    
    try:
        with st.spinner("Loading discriminator model..."):
            model, model_config, device = load_model_cached(checkpoint_path, use_gpu)
        st.sidebar.success(f"✅ Model on {device}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Make sure the discriminator model is trained and checkpoint exists.")
        st.stop()
    
    # ==========================================================================
    # MAIN CONTENT - TABS
    # ==========================================================================
    
    tabs = st.tabs([
        "📊 Overview",
        "📈 Score Distribution",
        "👤 Per-Driver Analysis",
        "🔬 Trajectory Comparison",
        "📉 Length Analysis",
        "🧪 Gradient Verification",
        "ℹ️ Model Info",
    ])
    
    # --------------------------------------------------------------------------
    # TAB 1: Overview
    # --------------------------------------------------------------------------
    with tabs[0]:
        st.header("📊 Fidelity Overview")
        
        # Build config
        config = FidelityConfig(
            checkpoint_path=checkpoint_path,
            mode=mode,
            aggregation=aggregation,
            threshold=threshold,
            batch_size=batch_size,
            max_trajectory_length=max_trajectory_length,
            use_gpu=use_gpu,
            min_trajectory_length=min_trajectory_length,
        )
        
        # Compute fidelity
        compute_btn = st.button("🚀 Compute Fidelity", type="primary")
        
        if compute_btn or 'fidelity_result' in st.session_state:
            if compute_btn:
                with st.spinner("Computing fidelity scores..."):
                    term = FidelityTerm(config)
                    result = term.compute_with_breakdown(
                        edited_trajectories,
                        {'original_trajectories': original_trajectories}
                    )
                    st.session_state['fidelity_result'] = result
                    st.session_state['config'] = config
            
            result = st.session_state['fidelity_result']
            
            # Display main result
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Fidelity Score",
                    f"{result['value']:.4f}",
                    help="Higher = more authentic"
                )
            
            with col2:
                st.metric(
                    "Trajectory Pairs",
                    f"{result['components'].get('n_pairs', 0):,}"
                )
            
            with col3:
                above_thresh = result['statistics'].get('above_threshold', 0)
                st.metric(
                    f"Above {threshold:.0%}",
                    f"{above_thresh:.1%}"
                )
            
            with col4:
                comp_time = result['diagnostics'].get('computation_time_seconds', 0)
                st.metric(
                    "Computation Time",
                    f"{comp_time:.2f}s"
                )
            
            # Score statistics
            st.subheader("Score Statistics")
            
            stats = result['statistics']
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '>50%', '>70%', '>90%'],
                'Value': [
                    f"{stats.get('score_mean', 0):.4f}",
                    f"{stats.get('score_std', 0):.4f}",
                    f"{stats.get('score_min', 0):.4f}",
                    f"{stats.get('score_max', 0):.4f}",
                    f"{stats.get('score_median', 0):.4f}",
                    f"{stats.get('above_threshold', 0):.1%}",
                    f"{stats.get('above_0.7', 0):.1%}",
                    f"{stats.get('above_0.9', 0):.1%}",
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
            # Interpretation
            st.subheader("Interpretation")
            
            fidelity = result['value']
            if fidelity >= 0.9:
                st.success("🌟 **Excellent Fidelity**: Edited trajectories are virtually indistinguishable from originals.")
            elif fidelity >= 0.7:
                st.success("✅ **Good Fidelity**: Edited trajectories maintain strong authenticity.")
            elif fidelity >= 0.5:
                st.warning("⚠️ **Moderate Fidelity**: Some trajectories show signs of editing.")
            else:
                st.error("❌ **Low Fidelity**: Edited trajectories are significantly different from originals.")
    
    # --------------------------------------------------------------------------
    # TAB 2: Score Distribution
    # --------------------------------------------------------------------------
    with tabs[1]:
        st.header("📈 Score Distribution")
        
        if 'fidelity_result' not in st.session_state:
            st.info("Click 'Compute Fidelity' in the Overview tab first.")
        else:
            result = st.session_state['fidelity_result']
            scores = np.array(result['components'].get('scores', []))
            
            if len(scores) == 0:
                st.warning("No scores available.")
            else:
                # Histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=scores,
                    nbinsx=30,
                    name="Score Distribution",
                    marker_color='steelblue',
                    opacity=0.7
                ))
                
                # Add threshold line
                fig.add_vline(
                    x=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold ({threshold:.2f})"
                )
                
                # Add mean line
                mean_score = np.mean(scores)
                fig.add_vline(
                    x=mean_score,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"Mean ({mean_score:.3f})"
                )
                
                fig.update_layout(
                    title="Distribution of Fidelity Scores",
                    xaxis_title="Fidelity Score",
                    yaxis_title="Count",
                    showlegend=True,
                    height=500,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # CDF plot
                st.subheader("Cumulative Distribution")
                
                sorted_scores = np.sort(scores)
                cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
                
                fig_cdf = go.Figure()
                fig_cdf.add_trace(go.Scatter(
                    x=sorted_scores,
                    y=cdf,
                    mode='lines',
                    name='CDF',
                    line=dict(color='steelblue', width=2)
                ))
                
                # Add percentile markers
                for p, label in [(0.25, '25th'), (0.5, '50th'), (0.75, '75th')]:
                    idx = int(p * len(sorted_scores))
                    val = sorted_scores[idx]
                    fig_cdf.add_trace(go.Scatter(
                        x=[val],
                        y=[p],
                        mode='markers+text',
                        name=f'{label} percentile',
                        text=[f'{val:.3f}'],
                        textposition='top right',
                        marker=dict(size=10)
                    ))
                
                fig_cdf.update_layout(
                    title="Cumulative Distribution Function",
                    xaxis_title="Fidelity Score",
                    yaxis_title="Cumulative Probability",
                    height=400,
                )
                
                st.plotly_chart(fig_cdf, use_container_width=True)
                
                # Box plot
                st.subheader("Box Plot")
                
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=scores,
                    name="Fidelity Scores",
                    boxpoints='outliers',
                    marker_color='steelblue'
                ))
                
                fig_box.update_layout(
                    title="Fidelity Score Box Plot",
                    yaxis_title="Fidelity Score",
                    height=400,
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
    
    # --------------------------------------------------------------------------
    # TAB 3: Per-Driver Analysis
    # --------------------------------------------------------------------------
    with tabs[2]:
        st.header("👤 Per-Driver Analysis")
        
        if 'fidelity_result' not in st.session_state:
            st.info("Click 'Compute Fidelity' in the Overview tab first.")
        else:
            result = st.session_state['fidelity_result']
            per_driver = result['statistics'].get('per_driver', {})
            
            if not per_driver:
                st.warning("No per-driver statistics available.")
            else:
                # Create DataFrame
                driver_data = []
                for driver_id, stats in per_driver.items():
                    driver_data.append({
                        'Driver ID': str(driver_id),
                        'N Pairs': stats['n_pairs'],
                        'Mean': stats['mean'],
                        'Std': stats['std'],
                        'Min': stats['min'],
                        'Max': stats['max'],
                    })
                
                df = pd.DataFrame(driver_data)
                df = df.sort_values('Mean', ascending=False)
                
                # Bar chart of mean scores
                fig = px.bar(
                    df,
                    x='Driver ID',
                    y='Mean',
                    error_y='Std',
                    title="Mean Fidelity Score by Driver",
                    color='Mean',
                    color_continuous_scale='RdYlGn',
                )
                fig.add_hline(y=threshold, line_dash="dash", line_color="red")
                fig.update_layout(height=500)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.subheader("Driver Statistics Table")
                
                st.dataframe(
                    df.style.format({
                        'Mean': '{:.4f}',
                        'Std': '{:.4f}',
                        'Min': '{:.4f}',
                        'Max': '{:.4f}',
                    }).background_gradient(subset=['Mean'], cmap='RdYlGn'),
                    hide_index=True,
                    use_container_width=True,
                )
                
                # Driver comparison
                st.subheader("Driver Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    best_drivers = df.nlargest(5, 'Mean')
                    st.markdown("**Top 5 Drivers (Highest Fidelity)**")
                    st.dataframe(
                        best_drivers[['Driver ID', 'Mean', 'N Pairs']],
                        hide_index=True,
                    )
                
                with col2:
                    worst_drivers = df.nsmallest(5, 'Mean')
                    st.markdown("**Bottom 5 Drivers (Lowest Fidelity)**")
                    st.dataframe(
                        worst_drivers[['Driver ID', 'Mean', 'N Pairs']],
                        hide_index=True,
                    )
    
    # --------------------------------------------------------------------------
    # TAB 4: Trajectory Comparison
    # --------------------------------------------------------------------------
    with tabs[3]:
        st.header("🔬 Trajectory Comparison")
        
        st.markdown("""
        Visualize individual trajectory pairs and their fidelity scores.
        """)
        
        if 'fidelity_result' not in st.session_state:
            st.info("Click 'Compute Fidelity' in the Overview tab first.")
        else:
            result = st.session_state['fidelity_result']
            scores = result['components'].get('scores', [])
            metadata = result['components'].get('metadata', [])
            
            if len(scores) == 0:
                st.warning("No trajectory pairs available.")
            else:
                # Select trajectory pair
                pair_options = []
                for i, (score, meta) in enumerate(zip(scores, metadata)):
                    driver_id = meta.get('driver_id', 'unknown')
                    traj_idx = meta.get('trajectory_idx', i)
                    pair_options.append(f"Pair {i}: Driver {driver_id}, Traj {traj_idx} (Score: {score:.3f})")
                
                selected_pair = st.selectbox(
                    "Select Trajectory Pair",
                    options=range(len(pair_options)),
                    format_func=lambda i: pair_options[i]
                )
                
                # Get trajectory pair
                meta = metadata[selected_pair]
                driver_id = meta.get('driver_id')
                traj_idx = meta.get('trajectory_idx', 0)
                
                if driver_id in edited_trajectories and traj_idx < len(edited_trajectories[driver_id]):
                    edited_traj = edited_trajectories[driver_id][traj_idx]
                    
                    if driver_id in original_trajectories and traj_idx < len(original_trajectories[driver_id]):
                        original_traj = original_trajectories[driver_id][traj_idx]
                    else:
                        original_traj = edited_traj
                    
                    # Display score
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fidelity Score", f"{scores[selected_pair]:.4f}")
                    with col2:
                        st.metric("Edited Length", len(edited_traj))
                    with col3:
                        st.metric("Original Length", len(original_traj))
                    
                    # Extract features for visualization
                    edited_features = np.array([[s[0], s[1]] for s in edited_traj])
                    original_features = np.array([[s[0], s[1]] for s in original_traj])
                    
                    # Plot trajectories
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=edited_features[:, 0],
                        y=edited_features[:, 1],
                        mode='lines+markers',
                        name='Edited',
                        line=dict(color='red', width=2),
                        marker=dict(size=4),
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=original_features[:, 0],
                        y=original_features[:, 1],
                        mode='lines+markers',
                        name='Original',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4),
                        opacity=0.7,
                    ))
                    
                    fig.update_layout(
                        title=f"Trajectory Comparison (Score: {scores[selected_pair]:.4f})",
                        xaxis_title="X Grid",
                        yaxis_title="Y Grid",
                        height=600,
                        showlegend=True,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature comparison
                    st.subheader("Feature Comparison (First 10 States)")
                    
                    n_show = min(10, len(edited_traj), len(original_traj))
                    
                    comparison_data = []
                    for i in range(n_show):
                        comparison_data.append({
                            'State': i,
                            'Edit X': edited_traj[i][0],
                            'Orig X': original_traj[i][0],
                            'Edit Y': edited_traj[i][1],
                            'Orig Y': original_traj[i][1],
                            'Edit Time': edited_traj[i][2],
                            'Orig Time': original_traj[i][2],
                        })
                    
                    st.dataframe(pd.DataFrame(comparison_data), hide_index=True)
                else:
                    st.warning("Could not find trajectory pair.")
    
    # --------------------------------------------------------------------------
    # TAB 5: Length Analysis
    # --------------------------------------------------------------------------
    with tabs[4]:
        st.header("📉 Trajectory Length Analysis")
        
        if 'fidelity_result' not in st.session_state:
            st.info("Click 'Compute Fidelity' in the Overview tab first.")
        else:
            result = st.session_state['fidelity_result']
            length_corr = result['statistics'].get('length_correlation', {})
            metadata = result['components'].get('metadata', [])
            scores = np.array(result['components'].get('scores', []))
            
            # Extract lengths
            edited_lengths = []
            original_lengths = []
            
            for meta in metadata:
                if 'edited_length' in meta:
                    edited_lengths.append(meta['edited_length'])
                if 'original_length' in meta:
                    original_lengths.append(meta['original_length'])
            
            if len(edited_lengths) > 0:
                # Correlation
                correlation = length_corr.get('correlation')
                if correlation is not None:
                    st.metric(
                        "Length-Fidelity Correlation",
                        f"{correlation:.4f}",
                        help="Correlation between trajectory length and fidelity score"
                    )
                
                # Scatter plot
                if len(edited_lengths) == len(scores):
                    fig = px.scatter(
                        x=edited_lengths,
                        y=scores,
                        title="Fidelity vs Trajectory Length",
                        labels={'x': 'Trajectory Length', 'y': 'Fidelity Score'},
                        trendline='ols',
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Length histogram
                st.subheader("Trajectory Length Distribution")
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=edited_lengths,
                    name='Edited',
                    opacity=0.7,
                ))
                if original_lengths:
                    fig.add_trace(go.Histogram(
                        x=original_lengths,
                        name='Original',
                        opacity=0.7,
                    ))
                
                fig.update_layout(
                    title="Trajectory Length Distribution",
                    xaxis_title="Length (states)",
                    yaxis_title="Count",
                    barmode='overlay',
                    height=400,
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No length information available.")
    
    # --------------------------------------------------------------------------
    # TAB 6: Gradient Verification
    # --------------------------------------------------------------------------
    with tabs[5]:
        st.header("🧪 Gradient Verification")
        
        st.markdown("""
        Verify that the fidelity term is differentiable and gradients are computed correctly.
        This is essential for gradient-based trajectory optimization.
        """)
        
        if not TORCH_AVAILABLE:
            st.error("PyTorch is required for gradient verification.")
        else:
            verify_btn = st.button("🔬 Verify Gradients", type="primary")
            
            if verify_btn:
                with st.spinner("Verifying gradients..."):
                    config = FidelityConfig(
                        checkpoint_path=checkpoint_path,
                        use_gpu=use_gpu,
                    )
                    term = FidelityTerm(config)
                    
                    # Use real data if available
                    if edited_trajectories:
                        verification = term.verify_differentiability(
                            edited_trajectories,
                            {'original_trajectories': original_trajectories}
                        )
                    else:
                        verification = term.verify_differentiability()
                    
                    st.session_state['gradient_verification'] = verification
            
            if 'gradient_verification' in st.session_state:
                verification = st.session_state['gradient_verification']
                
                # Display results
                if verification.get('success', False):
                    st.success("✅ Gradient verification passed!")
                else:
                    if 'error' in verification:
                        st.error(f"❌ Verification failed: {verification['error']}")
                    else:
                        st.warning("⚠️ Gradient verification completed with warnings.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Fidelity Value",
                        f"{verification.get('fidelity_value', 0):.4f}"
                    )
                
                with col2:
                    st.metric(
                        "Gradient Norm",
                        f"{verification.get('gradient_norm', 0):.6f}"
                    )
                
                # Gradient samples table
                samples = verification.get('gradient_samples', [])
                if samples:
                    st.subheader("Gradient Samples")
                    
                    sample_data = []
                    for s in samples:
                        sample_data.append({
                            'Position': str(s['position']),
                            'Numerical': f"{s['numerical']:.6f}",
                            'Analytic': f"{s['analytic']:.6f}",
                            'Abs Diff': f"{s['abs_diff']:.2e}",
                            'Rel Diff': f"{s['rel_diff']:.2e}",
                        })
                    
                    st.dataframe(pd.DataFrame(sample_data), hide_index=True)
                
                # Summary statistics
                st.subheader("Verification Summary")
                
                summary_data = {
                    'Metric': ['Max Abs Diff', 'Mean Abs Diff', 'Max Rel Diff', 'Mean Rel Diff'],
                    'Value': [
                        f"{verification.get('max_abs_diff', 0):.2e}",
                        f"{verification.get('mean_abs_diff', 0):.2e}",
                        f"{verification.get('max_rel_diff', 0):.2e}",
                        f"{verification.get('mean_rel_diff', 0):.2e}",
                    ]
                }
                st.dataframe(pd.DataFrame(summary_data), hide_index=True)
    
    # --------------------------------------------------------------------------
    # TAB 7: Model Info
    # --------------------------------------------------------------------------
    with tabs[6]:
        st.header("ℹ️ Model Information")
        
        # Model config
        st.subheader("Model Configuration")
        
        if model_config:
            config_df = pd.DataFrame([
                {'Parameter': k, 'Value': str(v)}
                for k, v in model_config.items()
            ])
            st.dataframe(config_df, hide_index=True)
        else:
            st.info("Model configuration not available.")
        
        # Training info
        model_info = get_model_info(checkpoint_path)
        
        if 'results' in model_info:
            st.subheader("Training Results")
            
            results = model_info['results']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best Val Loss", f"{results.get('best_val_loss', 0):.4f}")
            with col2:
                st.metric("Best Val Acc", f"{results.get('best_val_accuracy', 0):.2%}")
            with col3:
                st.metric("Best Val F1", f"{results.get('best_val_f1', 0):.4f}")
            with col4:
                st.metric("Epochs", results.get('epochs_trained', 0))
        
        if 'history' in model_info:
            st.subheader("Training History")
            
            history = model_info['history']
            
            # Plot training curves
            if 'train_loss' in history and 'val_loss' in history:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Loss', 'Accuracy', 'F1 Score', 'Learning Rate')
                )
                
                epochs = list(range(1, len(history['train_loss']) + 1))
                
                # Loss
                fig.add_trace(go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss'), row=1, col=1)
                fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss'), row=1, col=1)
                
                # Accuracy
                if 'train_accuracy' in history:
                    fig.add_trace(go.Scatter(x=epochs, y=history['train_accuracy'], name='Train Acc'), row=1, col=2)
                    fig.add_trace(go.Scatter(x=epochs, y=history['val_accuracy'], name='Val Acc'), row=1, col=2)
                
                # F1
                if 'train_f1' in history:
                    fig.add_trace(go.Scatter(x=epochs, y=history['train_f1'], name='Train F1'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=epochs, y=history['val_f1'], name='Val F1'), row=2, col=1)
                
                # LR
                if 'learning_rate' in history:
                    fig.add_trace(go.Scatter(x=epochs, y=history['learning_rate'], name='LR'), row=2, col=2)
                
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
        
        # Checkpoint info
        st.subheader("Checkpoint Information")
        
        st.markdown(f"""
        - **Path**: `{checkpoint_path}`
        - **Device**: {device}
        - **PyTorch Version**: {torch.__version__ if TORCH_AVAILABLE else 'N/A'}
        - **CUDA Available**: {torch.cuda.is_available() if TORCH_AVAILABLE else False}
        """)


if __name__ == "__main__":
    main()
