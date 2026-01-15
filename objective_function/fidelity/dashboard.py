"""
Streamlit Dashboard for Trajectory Fidelity Term Validation.

This dashboard provides an interactive interface for:
- Loading and exploring trajectory datasets
- Configuring discriminator model and fidelity parameters
- Computing fidelity scores with visualizations
- Analyzing per-driver and per-trajectory statistics
- Comparing original vs edited trajectories
- Verifying differentiability for gradient-based optimization
- **Testing discriminator behavior** with diagnostic modes

ST-iFGSM Paper Reference:
    The ST-SiameseNet discriminator should output:
    - > 0.5 for trajectories from the SAME agent (positive pairs)
    - < 0.5 for trajectories from DIFFERENT agents (negative pairs)
    
    In the trajectory editing context:
    - HIGH scores (> 0.5): Edited trajectory maintains driver identity
    - LOW scores (< 0.5): Edited trajectory has lost driver identity

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
import random

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
# CONSTANTS
# =============================================================================

# Test mode definitions
TEST_MODES = {
    "identical_trajectories": {
        "name": "🔬 Identical Trajectories",
        "description": "Compare each trajectory with ITSELF. Expected: scores ≈ 1.0 (identical inputs)",
        "expected_behavior": "Should return ~1.0 for all pairs since inputs are identical",
        "failure_indicates": "Model architecture issue or feature preprocessing problem"
    },
    "same_driver_same_day": {
        "name": "👤 Same Driver, Same Day", 
        "description": "Compare different trajectories from same driver on same day. Expected: scores > 0.5",
        "expected_behavior": "Should return > 0.5 (same agent identity)",
        "failure_indicates": "Model may not be learning driver identity correctly"
    },
    "same_driver_different_days": {
        "name": "📅 Same Driver, Different Days",
        "description": "Compare trajectories from same driver on different days. Expected: scores > 0.5",
        "expected_behavior": "Should return > 0.5 (same agent identity persists across days)",
        "failure_indicates": "Model may be overfitting to temporal patterns"
    },
    "different_drivers": {
        "name": "🚗 Different Drivers",
        "description": "Compare trajectories from DIFFERENT drivers. Expected: scores < 0.5",
        "expected_behavior": "Should return < 0.5 (different agent identities)",
        "failure_indicates": "Model not distinguishing between drivers"
    },
    "edited_vs_original": {
        "name": "✏️ Edited vs Original",
        "description": "Standard fidelity mode: compare edited trajectories with their originals",
        "expected_behavior": "Should return > 0.5 if edits preserve identity, < 0.5 if identity lost",
        "failure_indicates": "N/A - this is the actual use case"
    }
}

AGGREGATION_METHODS = {
    "mean": "Average of all scores. Best for overall assessment.",
    "min": "Minimum score (worst-case). Use when any failure is unacceptable.",
    "threshold": "Fraction of pairs above threshold. Good for pass/fail analysis.",
    "weighted": "Length-weighted average. Accounts for trajectory importance."
}


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
# TEST MODE HELPER FUNCTIONS
# =============================================================================

def create_identical_pairs(
    trajectories: Dict[Any, List[List[List[float]]]],
    max_pairs: int = 100,
    feature_range: Tuple[int, int] = (0, 4),
    min_length: int = 2
) -> Tuple[List[np.ndarray], List[np.ndarray], List[dict]]:
    """Create pairs where both trajectories are identical (traj compared with itself)."""
    traj1_list = []
    traj2_list = []
    metadata = []
    
    count = 0
    for driver_id, trajs in trajectories.items():
        for idx, traj in enumerate(trajs):
            if len(traj) < min_length:
                continue
            
            features = extract_trajectory_features(traj, feature_range)
            traj1_list.append(features)
            traj2_list.append(features.copy())  # Exact copy
            metadata.append({
                'driver_id': driver_id,
                'trajectory_idx': idx,
                'length': len(traj),
                'pair_type': 'identical'
            })
            
            count += 1
            if count >= max_pairs:
                return traj1_list, traj2_list, metadata
    
    return traj1_list, traj2_list, metadata


def create_same_driver_pairs(
    trajectories: Dict[Any, List[List[List[float]]]],
    max_pairs: int = 100,
    feature_range: Tuple[int, int] = (0, 4),
    min_length: int = 2,
    same_day: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray], List[dict]]:
    """Create pairs from same driver (different trajectories)."""
    traj1_list = []
    traj2_list = []
    metadata = []
    
    rng = random.Random(42)
    count = 0
    
    for driver_id, trajs in trajectories.items():
        valid_trajs = [(i, t) for i, t in enumerate(trajs) if len(t) >= min_length]
        
        if len(valid_trajs) < 2:
            continue
        
        # Get day indices for trajectories (assumes day_index is at position 3)
        traj_by_day = {}
        for idx, traj in valid_trajs:
            # Use first state's day_index as trajectory day
            day = int(traj[0][3]) if len(traj[0]) > 3 else 0
            if day not in traj_by_day:
                traj_by_day[day] = []
            traj_by_day[day].append((idx, traj))
        
        if same_day:
            # Pair trajectories from same day
            for day, day_trajs in traj_by_day.items():
                if len(day_trajs) < 2:
                    continue
                for i in range(len(day_trajs) - 1):
                    idx1, traj1 = day_trajs[i]
                    idx2, traj2 = day_trajs[i + 1]
                    
                    traj1_list.append(extract_trajectory_features(traj1, feature_range))
                    traj2_list.append(extract_trajectory_features(traj2, feature_range))
                    metadata.append({
                        'driver_id': driver_id,
                        'traj1_idx': idx1,
                        'traj2_idx': idx2,
                        'day': day,
                        'pair_type': 'same_driver_same_day'
                    })
                    
                    count += 1
                    if count >= max_pairs:
                        return traj1_list, traj2_list, metadata
        else:
            # Pair trajectories from different days
            days = list(traj_by_day.keys())
            if len(days) < 2:
                continue
            
            for i, day1 in enumerate(days):
                for day2 in days[i+1:]:
                    trajs1 = traj_by_day[day1]
                    trajs2 = traj_by_day[day2]
                    
                    # Sample one pair from each day combination
                    idx1, traj1 = rng.choice(trajs1)
                    idx2, traj2 = rng.choice(trajs2)
                    
                    traj1_list.append(extract_trajectory_features(traj1, feature_range))
                    traj2_list.append(extract_trajectory_features(traj2, feature_range))
                    metadata.append({
                        'driver_id': driver_id,
                        'traj1_idx': idx1,
                        'traj2_idx': idx2,
                        'day1': day1,
                        'day2': day2,
                        'pair_type': 'same_driver_different_days'
                    })
                    
                    count += 1
                    if count >= max_pairs:
                        return traj1_list, traj2_list, metadata
    
    return traj1_list, traj2_list, metadata


def create_different_driver_pairs(
    trajectories: Dict[Any, List[List[List[float]]]],
    max_pairs: int = 100,
    feature_range: Tuple[int, int] = (0, 4),
    min_length: int = 2
) -> Tuple[List[np.ndarray], List[np.ndarray], List[dict]]:
    """Create pairs from different drivers (negative pairs)."""
    traj1_list = []
    traj2_list = []
    metadata = []
    
    rng = random.Random(42)
    driver_ids = list(trajectories.keys())
    
    if len(driver_ids) < 2:
        return traj1_list, traj2_list, metadata
    
    count = 0
    for i, driver1 in enumerate(driver_ids):
        for driver2 in driver_ids[i+1:]:
            trajs1 = [t for t in trajectories[driver1] if len(t) >= min_length]
            trajs2 = [t for t in trajectories[driver2] if len(t) >= min_length]
            
            if not trajs1 or not trajs2:
                continue
            
            traj1 = rng.choice(trajs1)
            traj2 = rng.choice(trajs2)
            
            traj1_list.append(extract_trajectory_features(traj1, feature_range))
            traj2_list.append(extract_trajectory_features(traj2, feature_range))
            metadata.append({
                'driver1_id': driver1,
                'driver2_id': driver2,
                'pair_type': 'different_drivers'
            })
            
            count += 1
            if count >= max_pairs:
                return traj1_list, traj2_list, metadata
    
    return traj1_list, traj2_list, metadata


def run_discriminator_test(
    model: Any,
    traj1_list: List[np.ndarray],
    traj2_list: List[np.ndarray],
    batch_size: int = 32,
    device: str = "cpu"
) -> np.ndarray:
    """Run discriminator on trajectory pairs and return scores."""
    if not TORCH_AVAILABLE or len(traj1_list) == 0:
        return np.array([])
    
    return compute_fidelity_scores(
        model, traj1_list, traj2_list,
        batch_size=batch_size, device=device
    )


def analyze_test_results(
    scores: np.ndarray,
    test_mode: str,
    threshold: float = 0.5
) -> dict:
    """Analyze test results and determine if behavior is correct."""
    if len(scores) == 0:
        return {'error': 'No scores computed'}
    
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    above_threshold = float(np.mean(scores > threshold))
    below_threshold = float(np.mean(scores < threshold))
    
    # Determine expected behavior based on test mode
    if test_mode == "identical_trajectories":
        expected_range = (0.95, 1.0)  # Should be very close to 1.0
        passed = mean_score >= 0.9
        diagnosis = "PASS: Identical inputs produce high scores" if passed else \
                   "FAIL: Model not recognizing identical inputs - check architecture"
    elif test_mode in ["same_driver_same_day", "same_driver_different_days"]:
        expected_range = (0.5, 1.0)  # Should be above 0.5
        passed = mean_score > 0.5
        diagnosis = "PASS: Same-driver pairs recognized" if passed else \
                   "FAIL: Model not learning driver identity"
    elif test_mode == "different_drivers":
        expected_range = (0.0, 0.5)  # Should be below 0.5
        passed = mean_score < 0.5
        diagnosis = "PASS: Different drivers distinguished" if passed else \
                   "FAIL: Model not distinguishing between drivers"
    else:
        expected_range = (0.0, 1.0)
        passed = True
        diagnosis = "Standard fidelity computation"
    
    return {
        'mean': mean_score,
        'std': std_score,
        'min': min_score,
        'max': max_score,
        'above_threshold': above_threshold,
        'below_threshold': below_threshold,
        'expected_range': expected_range,
        'passed': passed,
        'diagnosis': diagnosis,
        'n_pairs': len(scores)
    }


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    st.title("🎯 Trajectory Fidelity Term Dashboard")
    st.markdown("""
    Interactive validation and exploration of the **Trajectory Fidelity Term** ($F_{\\text{fidelity}}$).
    
    The fidelity term measures how authentic edited trajectories appear compared to original expert trajectories
    using a pre-trained ST-SiameseNet discriminator.
    
    ---
    
    **ST-iFGSM Reference**: The discriminator outputs probability that two trajectories belong to the **same agent**:
    - **Score > 0.5**: Trajectories likely from same agent (maintains identity)
    - **Score < 0.5**: Trajectories likely from different agents (identity lost)
    """)
    
    # ==========================================================================
    # SIDEBAR CONFIGURATION
    # ==========================================================================
    
    st.sidebar.header("⚙️ Configuration")
    
    # Data paths
    st.sidebar.subheader("📂 Data Source")
    
    default_traj_path = str(PROJECT_ROOT / "source_data" / "all_trajs.pkl")
    trajectory_path = st.sidebar.text_input(
        "Trajectory Data Path",
        value=default_traj_path,
        help="""Path to the trajectory data file (all_trajs.pkl format).
        
This file contains expert trajectories organized by driver ID.
Each trajectory is a list of states with features:
[x_grid, y_grid, time_bucket, day_index, ...]"""
    )
    
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
            format_func=lambda x: Path(x).parent.name,
            help="""Select a trained discriminator model checkpoint.

The ST-SiameseNet discriminator learns to identify whether two 
trajectories belong to the same driver. Training on balanced 
positive/negative pairs is recommended (e.g., 5000 pos + 5000 neg)."""
        )
    else:
        checkpoint_path = st.sidebar.text_input(
            "Checkpoint Path",
            value=str(DEFAULT_CONFIG.checkpoint_path),
            help="Full path to the model checkpoint (.pt file)"
        )
    
    use_gpu = st.sidebar.checkbox(
        "Use GPU",
        value=True,
        disabled=not TORCH_AVAILABLE or not torch.cuda.is_available(),
        help="Use GPU acceleration for faster inference. Recommended for large datasets."
    )
    
    # Simplified computation settings
    st.sidebar.subheader("🔧 Settings")
    
    with st.sidebar.expander("Advanced Options", expanded=False):
        aggregation = st.selectbox(
            "Score Aggregation",
            options=list(AGGREGATION_METHODS.keys()),
            index=0,
            format_func=lambda x: f"{x.capitalize()}",
            help="\n".join([f"• **{k}**: {v}" for k, v in AGGREGATION_METHODS.items()])
        )
        
        threshold = st.slider(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="""Classification threshold from ST-iFGSM paper.

• **> 0.5**: Same agent (positive)
• **< 0.5**: Different agent (negative)

Default 0.5 aligns with the paper's methodology."""
        )
        
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=256,
            value=32,
            help="Number of trajectory pairs processed per batch. Higher = faster but more memory."
        )
        
        max_trajectory_length = st.number_input(
            "Max Length",
            min_value=100,
            max_value=2000,
            value=1000,
            help="Maximum trajectory length (states). Longer trajectories are truncated."
        )
    
    # Simplified filtering  
    st.sidebar.subheader("🔍 Data Filtering")
    
    max_drivers = st.sidebar.slider(
        "Drivers",
        min_value=1,
        max_value=50,
        value=10,
        help="Number of drivers to include in analysis. Start small for quick tests."
    )
    
    max_trajectories_per_driver = st.sidebar.slider(
        "Trajectories/Driver",
        min_value=1,
        max_value=100,
        value=20,
        help="Maximum trajectories per driver. Reduces computation for testing."
    )
    
    min_trajectory_length = st.sidebar.number_input(
        "Min Length",
        min_value=2,
        max_value=50,
        value=2,
        help="Minimum trajectory length required. Short trajectories may be unreliable."
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
            trajectories = load_trajectories(trajectory_path)
            
            # Apply filtering
            filtered_trajectories = {}
            for i, (driver_id, trajs) in enumerate(trajectories.items()):
                if i >= max_drivers:
                    break
                filtered_trajs = [
                    t for t in trajs[:max_trajectories_per_driver]
                    if len(t) >= min_trajectory_length
                ]
                if filtered_trajs:
                    filtered_trajectories[driver_id] = filtered_trajs
            
            trajectories = filtered_trajectories
            
            # Count trajectories
            n_trajectories = sum(len(trajs) for trajs in trajectories.values())
        
        st.sidebar.success(f"✅ {len(trajectories)} drivers, {n_trajectories} trajectories")
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
        "🔬 Discriminator Tests",
        "📊 Fidelity Analysis",
        "📈 Score Distribution",
        "👤 Per-Driver Analysis",
        "🔬 Trajectory Comparison",
        "🧪 Gradient Verification",
        "ℹ️ Model Info",
    ])
    
    # --------------------------------------------------------------------------
    # TAB 1: DISCRIMINATOR TESTS (NEW)
    # --------------------------------------------------------------------------
    with tabs[0]:
        st.header("🔬 Discriminator Model Tests")
        
        st.markdown("""
        **Validate that the discriminator model is working correctly** before using it for fidelity computation.
        
        These tests verify the model's behavior matches the ST-iFGSM paper specification:
        - **Identical trajectories** → Score ≈ 1.0 (same input = same output)
        - **Same driver pairs** → Score > 0.5 (recognizes driver identity)
        - **Different driver pairs** → Score < 0.5 (distinguishes drivers)
        
        ---
        """)
        
        # Test mode selection
        test_mode = st.selectbox(
            "Select Test Mode",
            options=list(TEST_MODES.keys()),
            format_func=lambda x: TEST_MODES[x]["name"],
            help="Choose a test to validate discriminator behavior"
        )
        
        # Display test info
        test_info = TEST_MODES[test_mode]
        st.info(f"""
        **{test_info['name']}**
        
        {test_info['description']}
        
        • **Expected Behavior**: {test_info['expected_behavior']}
        • **Failure Indicates**: {test_info['failure_indicates']}
        """)
        
        # Test configuration
        col1, col2 = st.columns(2)
        with col1:
            max_test_pairs = st.number_input(
                "Max Test Pairs",
                min_value=10,
                max_value=500,
                value=100,
                help="Number of trajectory pairs to test"
            )
        with col2:
            test_batch_size = st.number_input(
                "Test Batch Size", 
                min_value=1,
                max_value=128,
                value=32,
                help="Batch size for test inference"
            )
        
        # Run test button
        run_test = st.button("🚀 Run Test", type="primary", key="run_discriminator_test")
        
        if run_test:
            with st.spinner(f"Running {test_info['name']}..."):
                # Create appropriate pairs based on test mode
                if test_mode == "identical_trajectories":
                    traj1_list, traj2_list, metadata = create_identical_pairs(
                        trajectories, max_pairs=max_test_pairs,
                        min_length=min_trajectory_length
                    )
                elif test_mode == "same_driver_same_day":
                    traj1_list, traj2_list, metadata = create_same_driver_pairs(
                        trajectories, max_pairs=max_test_pairs,
                        min_length=min_trajectory_length, same_day=True
                    )
                elif test_mode == "same_driver_different_days":
                    traj1_list, traj2_list, metadata = create_same_driver_pairs(
                        trajectories, max_pairs=max_test_pairs,
                        min_length=min_trajectory_length, same_day=False
                    )
                elif test_mode == "different_drivers":
                    traj1_list, traj2_list, metadata = create_different_driver_pairs(
                        trajectories, max_pairs=max_test_pairs,
                        min_length=min_trajectory_length
                    )
                else:  # edited_vs_original - use same trajectory as both
                    traj1_list, traj2_list, metadata = create_identical_pairs(
                        trajectories, max_pairs=max_test_pairs,
                        min_length=min_trajectory_length
                    )
                
                if len(traj1_list) == 0:
                    st.error("❌ No valid trajectory pairs found for this test mode.")
                else:
                    # Run discriminator
                    scores = run_discriminator_test(
                        model, traj1_list, traj2_list,
                        batch_size=test_batch_size, device=device
                    )
                    
                    # Analyze results
                    results = analyze_test_results(scores, test_mode, threshold)
                    
                    # Store in session state
                    st.session_state['test_results'] = {
                        'mode': test_mode,
                        'scores': scores,
                        'metadata': metadata,
                        'analysis': results
                    }
        
        # Display results
        if 'test_results' in st.session_state:
            results = st.session_state['test_results']
            analysis = results['analysis']
            scores = results['scores']
            
            st.subheader("Test Results")
            
            # Pass/Fail indicator
            if analysis['passed']:
                st.success(f"✅ **{analysis['diagnosis']}**")
            else:
                st.error(f"❌ **{analysis['diagnosis']}**")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Mean Score",
                    f"{analysis['mean']:.4f}",
                    help="Average discriminator output across all test pairs"
                )
            
            with col2:
                expected = analysis['expected_range']
                st.metric(
                    "Expected Range",
                    f"{expected[0]:.1f} - {expected[1]:.1f}",
                    help="Score range expected for this test mode"
                )
            
            with col3:
                st.metric(
                    "Std Dev",
                    f"{analysis['std']:.4f}",
                    help="Score variability"
                )
            
            with col4:
                st.metric(
                    "Test Pairs",
                    f"{analysis['n_pairs']:,}",
                    help="Number of trajectory pairs tested"
                )
            
            # Score range
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min", f"{analysis['min']:.4f}")
            with col2:
                st.metric("Median", f"{float(np.median(scores)):.4f}")
            with col3:
                st.metric("Max", f"{analysis['max']:.4f}")
            
            # Histogram
            st.subheader("Score Distribution")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=scores,
                nbinsx=30,
                marker_color='steelblue',
                opacity=0.7
            ))
            
            # Add expected range shading
            expected = analysis['expected_range']
            fig.add_vrect(
                x0=expected[0], x1=expected[1],
                fillcolor="green", opacity=0.1,
                annotation_text="Expected", annotation_position="top"
            )
            
            # Add threshold line
            fig.add_vline(
                x=0.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold (0.5)"
            )
            
            # Add mean line
            fig.add_vline(
                x=analysis['mean'],
                line_dash="dot",
                line_color="blue",
                annotation_text=f"Mean ({analysis['mean']:.3f})"
            )
            
            fig.update_layout(
                title=f"Score Distribution - {TEST_MODES[results['mode']]['name']}",
                xaxis_title="Discriminator Score",
                yaxis_title="Count",
                xaxis=dict(range=[0, 1]),
                height=400,
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed breakdown
            with st.expander("📊 Detailed Statistics"):
                st.markdown(f"""
                | Statistic | Value |
                |-----------|-------|
                | Mean | {analysis['mean']:.6f} |
                | Std Dev | {analysis['std']:.6f} |
                | Min | {analysis['min']:.6f} |
                | Max | {analysis['max']:.6f} |
                | > 0.5 | {analysis['above_threshold']:.1%} |
                | < 0.5 | {analysis['below_threshold']:.1%} |
                | Pairs Tested | {analysis['n_pairs']} |
                """)
    
    # --------------------------------------------------------------------------
    # TAB 2: Fidelity Analysis (was Overview)
    # --------------------------------------------------------------------------
    with tabs[1]:
        st.header("📊 Fidelity Analysis")
        
        st.markdown("""
        Compute fidelity scores comparing **edited trajectories** with their **originals**.
        
        For testing purposes, you can use the same trajectories as both edited and original
        (should give high scores ~1.0 if the model is working correctly).
        """)
        
        # Pairing mode selection for this tab
        fidelity_mode = st.selectbox(
            "Comparison Mode",
            options=["identical", "same_agent"],
            format_func=lambda x: {
                "identical": "🔬 Identical (test mode - each trajectory vs itself)",
                "same_agent": "👤 Same Agent (different trajectories from same driver)"
            }[x],
            help="""Select how to pair trajectories for comparison:
            
• **Identical**: Compare each trajectory with itself (test mode, should give ~1.0)
• **Same Agent**: Compare different trajectories from the same driver (realistic scenario)"""
        )
        
        # Build config
        config = FidelityConfig(
            checkpoint_path=checkpoint_path,
            mode="same_agent",  # Always use same_agent for underlying logic
            aggregation=aggregation,
            threshold=threshold,
            batch_size=batch_size,
            max_trajectory_length=max_trajectory_length,
            use_gpu=use_gpu,
            min_trajectory_length=min_trajectory_length,
        )
        
        # Compute fidelity
        compute_btn = st.button("🚀 Compute Fidelity", type="primary", key="compute_fidelity")
        
        if compute_btn or 'fidelity_result' in st.session_state:
            if compute_btn:
                with st.spinner("Computing fidelity scores..."):
                    # Create pairs based on selected mode
                    if fidelity_mode == "identical":
                        traj1_list, traj2_list, metadata = create_identical_pairs(
                            trajectories, max_pairs=1000,
                            min_length=min_trajectory_length
                        )
                    else:
                        traj1_list, traj2_list, metadata = create_same_driver_pairs(
                            trajectories, max_pairs=1000,
                            min_length=min_trajectory_length, same_day=True
                        )
                    
                    if len(traj1_list) == 0:
                        st.error("No valid trajectory pairs found.")
                        st.stop()
                    
                    # Compute scores
                    scores = compute_fidelity_scores(
                        model, traj1_list, traj2_list,
                        batch_size=batch_size, device=device
                    )
                    
                    # Aggregate
                    fidelity_value = aggregate_fidelity_scores(scores, aggregation, threshold)
                    
                    # Compute per-driver statistics
                    per_driver_stats = compute_per_driver_statistics(scores, metadata)
                    
                    # Build result structure
                    result = {
                        'value': fidelity_value,
                        'components': {
                            'n_pairs': len(scores),
                            'scores': scores.tolist(),
                            'metadata': metadata,
                        },
                        'statistics': {
                            'score_mean': float(np.mean(scores)),
                            'score_std': float(np.std(scores)),
                            'score_min': float(np.min(scores)),
                            'score_max': float(np.max(scores)),
                            'score_median': float(np.median(scores)),
                            'above_threshold': float(np.mean(scores >= threshold)),
                            'above_0.7': float(np.mean(scores >= 0.7)),
                            'above_0.9': float(np.mean(scores >= 0.9)),
                            'per_driver': per_driver_stats,
                        },
                        'diagnostics': {
                            'computation_time_seconds': 0,
                        }
                    }
                    st.session_state['fidelity_result'] = result
                    st.session_state['config'] = config
            
            result = st.session_state['fidelity_result']
            
            # Display main result
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Fidelity Score",
                    f"{result['value']:.4f}",
                    help="Higher = more authentic (should be > 0.5 for same-agent pairs)"
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
                st.markdown("""
                **Troubleshooting Low Scores:**
                1. First run the **Discriminator Tests** tab to verify model behavior
                2. If "Identical Trajectories" test fails, the model may have issues
                3. Check model training results in the **Model Info** tab
                """)
    
    # --------------------------------------------------------------------------
    # TAB 3: Score Distribution
    # --------------------------------------------------------------------------
    with tabs[2]:
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
    # TAB 4: Per-Driver Analysis
    # --------------------------------------------------------------------------
    with tabs[3]:
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
    # TAB 5: Trajectory Comparison
    # --------------------------------------------------------------------------
    with tabs[4]:
        st.header("🔬 Trajectory Comparison")
        
        st.markdown("""
        Visualize individual trajectory pairs and their fidelity scores.
        """)
        
        if 'fidelity_result' not in st.session_state:
            st.info("Click 'Compute Fidelity' in the Fidelity Analysis tab first.")
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
                
                if driver_id in trajectories and traj_idx < len(trajectories[driver_id]):
                    traj = trajectories[driver_id][traj_idx]
                    
                    # Display score
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Fidelity Score", f"{scores[selected_pair]:.4f}")
                    with col2:
                        st.metric("Trajectory Length", len(traj))
                    
                    # Extract features for visualization
                    features = np.array([[s[0], s[1]] for s in traj])
                    
                    # Plot trajectory
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=features[:, 0],
                        y=features[:, 1],
                        mode='lines+markers',
                        name='Trajectory',
                        line=dict(color='steelblue', width=2),
                        marker=dict(size=4),
                    ))
                    
                    fig.update_layout(
                        title=f"Trajectory Visualization (Driver {driver_id}, Score: {scores[selected_pair]:.4f})",
                        xaxis_title="X Grid",
                        yaxis_title="Y Grid",
                        height=600,
                        showlegend=True,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature comparison
                    st.subheader("Trajectory Features (First 10 States)")
                    
                    n_show = min(10, len(traj))
                    
                    feature_data = []
                    for i in range(n_show):
                        feature_data.append({
                            'State': i,
                            'X Grid': traj[i][0],
                            'Y Grid': traj[i][1],
                            'Time Bucket': traj[i][2] if len(traj[i]) > 2 else 'N/A',
                            'Day Index': traj[i][3] if len(traj[i]) > 3 else 'N/A',
                        })
                    
                    st.dataframe(pd.DataFrame(feature_data), hide_index=True)
                else:
                    st.warning("Could not find trajectory pair.")
    
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
            verify_btn = st.button("🔬 Verify Gradients", type="primary", key="verify_gradients")
            
            if verify_btn:
                with st.spinner("Verifying gradients..."):
                    config = FidelityConfig(
                        checkpoint_path=checkpoint_path,
                        use_gpu=use_gpu,
                    )
                    term = FidelityTerm(config)
                    
                    # Use real data if available
                    if trajectories:
                        verification = term.verify_differentiability(
                            trajectories,
                            {'original_trajectories': trajectories}
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
        
        st.markdown("""
        Information about the trained discriminator model, including architecture,
        training history, and performance metrics.
        """)
        
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
