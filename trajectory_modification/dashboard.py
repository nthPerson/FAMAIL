"""
FAMAIL Trajectory Modification Dashboard

Streamlit dashboard for testing trajectory modification with the ST-iFGSM algorithm.
Allows visualization of trajectory modifications and fairness metric changes.
"""

import streamlit as st
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="FAMAIL Trajectory Modification",
    page_icon="🚕",
    layout="wide",
)

st.title("🚕 FAMAIL Trajectory Modification Testing Tool")
st.markdown("""
This dashboard tests the **Fairness-Aware Trajectory Editing Algorithm** (Modified ST-iFGSM).

**Algorithm**: `δ = clip(α · sign[∇L], -ε, ε)` where `L = α₁F_spatial + α₂F_causal + α₃F_fidelity`
""")

# ============================================================================
# Session State Initialization
# ============================================================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'trajectories' not in st.session_state:
    st.session_state.trajectories = []
if 'modification_history' not in st.session_state:
    st.session_state.modification_history = []
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []


# ============================================================================
# Data Loading Section
# ============================================================================

def load_data_section():
    """Data loading controls and status."""
    st.sidebar.header("📁 Data Loading")
    
    workspace_root = Path(__file__).resolve().parent.parent
    
    # Check for required files
    data_files = {
        'trajectories': workspace_root / 'source_data' / 'passenger_seeking_trajs_45-800.pkl',
        'all_trajs': workspace_root / 'source_data' / 'all_trajs.pkl',
        'traffic': workspace_root / 'source_data' / 'latest_traffic.pkl',
        'discriminator': workspace_root / 'discriminator' / 'model' / 'checkpoints',
    }
    
    st.sidebar.subheader("File Status")
    for name, path in data_files.items():
        exists = path.exists()
        icon = "✅" if exists else "❌"
        st.sidebar.write(f"{icon} {name}")
    
    # Load button
    max_trajs = st.sidebar.number_input("Max Trajectories", 10, 1000, 100, step=10)
    
    if st.sidebar.button("Load Data", type="primary"):
        with st.spinner("Loading data..."):
            try:
                from trajectory_modification import DataBundle
                bundle = DataBundle.load_default(max_trajectories=max_trajs)
                st.session_state.trajectories = bundle.trajectories
                st.session_state.pickup_counts = bundle.pickup_counts
                st.session_state.supply_counts = bundle.supply_counts
                st.session_state.active_taxis = bundle.active_taxis
                st.session_state.g_function = bundle.g_function
                st.session_state.data_loaded = True
                st.sidebar.success(f"Loaded {len(bundle.trajectories)} trajectories")
            except FileNotFoundError as e:
                st.sidebar.error(f"File not found: {e}")
            except Exception as e:
                st.sidebar.error(f"Error loading data: {e.with_traceback()}")
    
    return st.session_state.data_loaded


# ============================================================================
# Algorithm Parameters Section
# ============================================================================

def algorithm_params_section() -> Dict:
    """Algorithm parameter controls."""
    st.sidebar.header("⚙️ Algorithm Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        alpha = st.number_input("α (step size)", 0.01, 1.0, 0.1, format="%.2f",
                                help="ST-iFGSM step size")
        epsilon = st.number_input("ε (max perturb)", 1.0, 10.0, 3.0, format="%.1f",
                                  help="Maximum perturbation per dimension")
    
    with col2:
        max_iter = st.number_input("Max iterations", 10, 200, 50,
                                   help="Maximum ST-iFGSM iterations")
        conv_thresh = st.number_input("Convergence", 1e-6, 1e-2, 1e-4, format="%.1e",
                                      help="Convergence threshold")
    
    st.sidebar.subheader("Objective Weights")
    w1 = st.sidebar.slider("α₁ (Spatial)", 0.0, 1.0, 0.33, 0.01)
    w2 = st.sidebar.slider("α₂ (Causal)", 0.0, 1.0, 0.33, 0.01)
    w3 = st.sidebar.slider("α₃ (Fidelity)", 0.0, 1.0, 0.34, 0.01)
    
    # Normalize weights
    total = w1 + w2 + w3
    if total > 0:
        w1, w2, w3 = w1/total, w2/total, w3/total
    
    st.sidebar.caption(f"Normalized: ({w1:.2f}, {w2:.2f}, {w3:.2f})")
    
    return {
        'alpha': alpha,
        'epsilon': epsilon,
        'max_iterations': max_iter,
        'convergence_threshold': conv_thresh,
        'alpha_spatial': w1,
        'alpha_causal': w2,
        'alpha_fidelity': w3,
    }


# ============================================================================
# Trajectory Selection
# ============================================================================

def trajectory_selection_section():
    """Trajectory selection and preview."""
    st.header("📍 Trajectory Selection")
    
    if not st.session_state.data_loaded:
        st.info("Load data first using the sidebar controls.")
        return None
    
    trajectories = st.session_state.trajectories
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select Trajectory")
        
        # Selection mode
        selection_mode = st.radio(
            "Selection Mode",
            ["Single", "Batch", "Random Sample"],
            horizontal=True
        )
        
        if selection_mode == "Single":
            traj_idx = st.number_input(
                "Trajectory Index",
                0, len(trajectories) - 1, 0
            )
            selected_indices = [traj_idx]
            
        elif selection_mode == "Batch":
            start_idx = st.number_input("Start Index", 0, len(trajectories) - 1, 0)
            batch_size = st.number_input("Batch Size", 1, min(50, len(trajectories)), 10)
            selected_indices = list(range(start_idx, min(start_idx + batch_size, len(trajectories))))
            
        else:  # Random Sample
            sample_size = st.number_input("Sample Size", 1, min(50, len(trajectories)), 10)
            if st.button("🎲 Resample"):
                selected_indices = np.random.choice(len(trajectories), sample_size, replace=False).tolist()
                st.session_state.selected_indices = selected_indices
            selected_indices = st.session_state.get('selected_indices', list(range(sample_size)))
        
        st.write(f"**Selected:** {len(selected_indices)} trajectories")
        
    with col2:
        st.subheader("Trajectory Preview")
        
        if selected_indices and PLOTLY_AVAILABLE:
            # Show first selected trajectory
            traj = trajectories[selected_indices[0]]
            
            x_coords = [s.x_grid for s in traj.states]
            y_coords = [s.y_grid for s in traj.states]
            
            fig = go.Figure()
            
            # Trajectory path
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines+markers',
                name='Trajectory',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # Highlight pickup location (estimated)
            pickup_idx = int(len(traj.states) * 0.7)
            fig.add_trace(go.Scatter(
                x=[x_coords[pickup_idx]],
                y=[y_coords[pickup_idx]],
                mode='markers',
                name='Estimated Pickup',
                marker=dict(color='red', size=15, symbol='star')
            ))
            
            fig.update_layout(
                title=f"Trajectory {selected_indices[0]} ({len(traj.states)} states)",
                xaxis_title="Grid X",
                yaxis_title="Grid Y",
                height=400,
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback text display
            if selected_indices:
                traj = trajectories[selected_indices[0]]
                st.write(f"Length: {len(traj.states)} states")
                st.write(f"Driver ID: {traj.driver_id}")
    
    return selected_indices


# ============================================================================
# Modification Execution
# ============================================================================

def run_modification_section(selected_indices: List[int], params: Dict):
    """Execute trajectory modification."""
    st.header("🔧 Trajectory Modification")
    
    if not st.session_state.data_loaded or not selected_indices:
        st.info("Load data and select trajectories first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_discriminator = st.checkbox(
            "Use Discriminator for Fidelity",
            value=False,
            help="Load trained discriminator for F_fidelity computation"
        )
        
        if use_discriminator:
            checkpoint_path = st.text_input(
                "Checkpoint Path",
                "discriminator/model/checkpoints/best_model.pt"
            )
    
    with col2:
        show_iterations = st.checkbox("Show Iteration Details", value=True)
        save_results = st.checkbox("Save Results to File", value=False)
    
    if st.button("▶️ Run Modification", type="primary"):
        _execute_modification(selected_indices, params, use_discriminator, show_iterations)


def _execute_modification(
    selected_indices: List[int], 
    params: Dict, 
    use_discriminator: bool,
    show_iterations: bool
):
    """Execute the ST-iFGSM modification algorithm."""
    from trajectory_modification import (
        TrajectoryModifier, 
        FAMAILObjective, 
        GlobalMetrics,
        DiscriminatorAdapter,
    )
    
    device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    st.write(f"Running on device: **{device}**")
    
    # Initialize components
    discriminator = None
    if use_discriminator:
        try:
            adapter = DiscriminatorAdapter()
            checkpoint_dir = Path(__file__).parent.parent / 'discriminator' / 'model' / 'checkpoints'
            if adapter.load_checkpoint(checkpoint_dir):
                discriminator = adapter.model
                st.success("✅ Discriminator loaded")
            else:
                st.warning("⚠️ Could not load discriminator, using default fidelity")
        except Exception as e:
            st.warning(f"⚠️ Discriminator error: {e}")
    
    # Create objective function
    objective = FAMAILObjective(
        alpha_spatial=params['alpha_spatial'],
        alpha_causal=params['alpha_causal'],
        alpha_fidelity=params['alpha_fidelity'],
        g_function=st.session_state.g_function,
        discriminator=discriminator,
    )
    
    # Create modifier
    modifier = TrajectoryModifier(
        objective_fn=objective,
        alpha=params['alpha'],
        epsilon=params['epsilon'],
        max_iterations=params['max_iterations'],
        convergence_threshold=params['convergence_threshold'],
    )
    
    # Set global state
    modifier.set_global_state(
        pickup_counts=torch.tensor(st.session_state.pickup_counts, device=device, dtype=torch.float32),
        supply_counts=torch.tensor(st.session_state.supply_counts, device=device, dtype=torch.float32),
        active_taxis=torch.tensor(st.session_state.active_taxis, device=device, dtype=torch.float32),
    )
    
    # Initialize metrics tracker
    metrics = GlobalMetrics(
        g_function=st.session_state.g_function,
        alpha_weights=(params['alpha_spatial'], params['alpha_causal'], params['alpha_fidelity'])
    )
    metrics.initialize_from_data(
        st.session_state.pickup_counts,
        st.session_state.supply_counts,
        st.session_state.active_taxis,
    )
    
    # Record initial metrics
    initial_snapshot = metrics.compute_snapshot()
    
    # Process trajectories
    trajectories = st.session_state.trajectories
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    histories = []
    for i, idx in enumerate(selected_indices):
        status_text.text(f"Processing trajectory {idx} ({i+1}/{len(selected_indices)})")
        
        try:
            traj = trajectories[idx]
            history = modifier.modify_single(traj)
            histories.append(history)
            
            # Update metrics
            if history.iterations:
                orig_state = traj.states[int(len(traj.states) * 0.7)]
                mod_state = history.modified.states[int(len(history.modified.states) * 0.7)]
                metrics.update_pickup(
                    old_cell=(int(orig_state.x_grid), int(orig_state.y_grid)),
                    new_cell=(int(mod_state.x_grid), int(mod_state.y_grid)),
                    fidelity_score=history.iterations[-1].f_fidelity,
                )
                
        except Exception as e:
            st.error(f"Error processing trajectory {idx}: {e}")
        
        progress_bar.progress((i + 1) / len(selected_indices))
    
    status_text.text("Processing complete!")
    
    # Final metrics
    final_snapshot = metrics.compute_snapshot()
    
    # Store results
    st.session_state.modification_history = histories
    st.session_state.metrics_history.append({
        'initial': initial_snapshot,
        'final': final_snapshot,
        'num_modified': len(histories),
    })
    
    # Display results
    _display_modification_results(histories, initial_snapshot, final_snapshot, show_iterations)


def _display_modification_results(
    histories, 
    initial_snapshot, 
    final_snapshot,
    show_iterations: bool
):
    """Display modification results."""
    st.header("📊 Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_gini = final_snapshot.gini_coefficient - initial_snapshot.gini_coefficient
        color = "inverse" if delta_gini < 0 else "normal"
        st.metric(
            "Gini Coefficient",
            f"{final_snapshot.gini_coefficient:.4f}",
            f"{delta_gini:+.4f}",
            delta_color=color
        )
    
    with col2:
        delta_spatial = final_snapshot.f_spatial - initial_snapshot.f_spatial
        st.metric(
            "F_spatial",
            f"{final_snapshot.f_spatial:.4f}",
            f"{delta_spatial:+.4f}"
        )
    
    with col3:
        delta_causal = final_snapshot.f_causal - initial_snapshot.f_causal
        st.metric(
            "F_causal",
            f"{final_snapshot.f_causal:.4f}",
            f"{delta_causal:+.4f}"
        )
    
    with col4:
        delta_combined = final_snapshot.combined_objective - initial_snapshot.combined_objective
        st.metric(
            "Combined L",
            f"{final_snapshot.combined_objective:.4f}",
            f"{delta_combined:+.4f}"
        )
    
    # Convergence statistics
    st.subheader("Convergence Statistics")
    
    converged = sum(1 for h in histories if h.converged)
    avg_iterations = np.mean([h.total_iterations for h in histories])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Converged", f"{converged}/{len(histories)}")
    col2.metric("Avg Iterations", f"{avg_iterations:.1f}")
    col3.metric("Mean Fidelity", f"{final_snapshot.mean_fidelity:.4f}")
    
    # Iteration details
    if show_iterations and histories and PLOTLY_AVAILABLE:
        st.subheader("Objective Function Over Iterations")
        
        # Plot first trajectory's iterations
        history = histories[0]
        
        iterations = list(range(1, len(history.iterations) + 1))
        objectives = [r.objective_value for r in history.iterations]
        spatials = [r.f_spatial for r in history.iterations]
        causals = [r.f_causal for r in history.iterations]
        fidelities = [r.f_fidelity for r in history.iterations]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iterations, y=objectives, name='L (Combined)', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=iterations, y=spatials, name='F_spatial', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=iterations, y=causals, name='F_causal', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=iterations, y=fidelities, name='F_fidelity', line=dict(dash='dash')))
        
        fig.update_layout(
            title="Objective Function Evolution (First Trajectory)",
            xaxis_title="Iteration",
            yaxis_title="Value",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Perturbation visualization
    if histories and PLOTLY_AVAILABLE:
        st.subheader("Perturbation Magnitudes")
        
        final_perturbations = [
            np.linalg.norm(h.iterations[-1].perturbation) if h.iterations else 0 
            for h in histories
        ]
        
        fig = px.histogram(
            x=final_perturbations,
            nbins=20,
            labels={'x': 'Perturbation Magnitude (cells)'},
            title="Distribution of Final Perturbations"
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Heatmap Visualization
# ============================================================================

def heatmap_section():
    """Visualization of pickup distribution changes."""
    st.header("🗺️ Pickup Distribution")
    
    if not st.session_state.data_loaded:
        st.info("Load data first.")
        return
    
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available for visualization.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Before Modification")
        fig = px.imshow(
            st.session_state.pickup_counts.T,
            labels={'x': 'Grid X', 'y': 'Grid Y', 'color': 'Pickups'},
            title="Original Pickup Distribution",
            aspect='auto',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("After Modification")
        
        # Get modified counts if available
        if hasattr(st.session_state, 'modified_pickup_counts'):
            modified_counts = st.session_state.modified_pickup_counts
        else:
            # Use original if no modifications yet
            modified_counts = st.session_state.pickup_counts
        
        fig = px.imshow(
            modified_counts.T,
            labels={'x': 'Grid X', 'y': 'Grid Y', 'color': 'Pickups'},
            title="Modified Pickup Distribution",
            aspect='auto',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    # Sidebar: Data loading
    data_loaded = load_data_section()
    
    # Sidebar: Algorithm parameters
    params = algorithm_params_section()
    
    # Main content
    selected_indices = trajectory_selection_section()
    
    st.divider()
    
    run_modification_section(selected_indices or [], params)
    
    st.divider()
    
    heatmap_section()
    
    # Footer
    st.divider()
    st.caption("""
    **FAMAIL Trajectory Modification Tool** | 
    Algorithm: Modified ST-iFGSM | 
    δ = clip(α·sign[∇L], -ε, ε)
    """)


if __name__ == "__main__":
    main()
