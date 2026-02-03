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

st.title("🚕 FAMAIL Trajectory Modification Testing")
st.markdown("""
This dashboard tests the **Fairness-Aware Trajectory Editing Algorithm** (Modified ST-iFGSM).

**Algorithm Steps:**
1. **For each trajectory** chosen for modification:
2. **While** the discriminator still recognizes the modified trajectory as "same agent" **and** max iterations not exceeded:
   - Calculate the perturbation: `δ = clip(α · sign[∇L], -ε, ε)` where `L = α₁F_spatial + α₂F_causal + α₃F_fidelity`
   - Apply the perturbation to the pickup location of the trajectory
   - Interpolate trajectory points between the unmodified path and the new pickup location
3. **Return** the modified trajectory
4. **Recalculate** global fairness metrics with the new pickup distribution
5. **Repeat** from step 2 with the next trajectory to be modified
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
        'pickup_dropoff': workspace_root / 'source_data' / 'pickup_dropoff_counts.pkl',
        'active_taxis': workspace_root / 'source_data' / 'active_taxis_5x5_hourly.pkl',
        'discriminator': workspace_root / 'discriminator' / 'model' / 'checkpoints',
    }
    
    st.sidebar.subheader("File Status")
    for name, path in data_files.items():
        exists = path.exists()
        icon = "✅" if exists else "❌"
        st.sidebar.write(f"{icon} {name}")
    
    # Load button
    max_trajs = st.sidebar.number_input("Max Trajectories", 10, 1000, 100, step=10)

    # with st.spinner("Loading data..."):
    #     try:
    #         from trajectory_modification import DataBundle
    #         bundle = DataBundle.load_default(max_trajectories=max_trajs)
    #         # bundle = DataBundle.load_default(max_trajectories=max_trajs, workspace_root=workspace_root)
    #         st.session_state.trajectories = bundle.trajectories
    #         st.session_state.pickup_counts = bundle.pickup_counts
    #         st.session_state.supply_counts = bundle.supply_counts
    #         st.session_state.active_taxis = bundle.active_taxis
    #         st.session_state.g_function = bundle.g_function
    #         st.session_state.data_loaded = True
    #         st.sidebar.success(f"Loaded {len(bundle.trajectories)} trajectories")
    #     except FileNotFoundError as e:
    #         st.sidebar.error(f"File not found: {e}")
    #     except Exception as e:
    #         st.sidebar.error(f"Error loading data: {e.with_traceback()}")    
    if st.sidebar.button("Load Data", type="primary"):
        with st.spinner("Loading data..."):
            try:
                from trajectory_modification import DataBundle
                # Load with isotonic g(d) estimation and mean aggregation (RECOMMENDED)
                bundle = DataBundle.load_default(
                    max_trajectories=max_trajs,
                    estimate_g_from_data=True,  # Fit g(d) using isotonic regression
                    aggregation='mean',  # Use mean aggregation for causal fairness scale
                )
                st.session_state.trajectories = bundle.trajectories
                st.session_state.pickup_dropoff_data = bundle.pickup_dropoff_data
                st.session_state.pickup_grid = bundle.pickup_grid
                st.session_state.dropoff_grid = bundle.dropoff_grid
                st.session_state.active_taxis_data = bundle.active_taxis_data
                st.session_state.active_taxis_grid = bundle.active_taxis_grid
                st.session_state.g_function = bundle.g_function
                
                # NEW: Store causal fairness tensors (mean-aggregated for proper Y = S/D scale)
                st.session_state.causal_demand_grid = bundle.causal_demand_grid
                st.session_state.causal_supply_grid = bundle.causal_supply_grid
                st.session_state.g_function_diagnostics = bundle.g_function_diagnostics
                
                st.session_state.data_loaded = True
                st.sidebar.success(f"Loaded {len(bundle.trajectories)} trajectories")
                
                # Show g(d) fitting diagnostics
                if bundle.g_function_diagnostics:
                    diag = bundle.g_function_diagnostics
                    st.sidebar.info(f"g(d) fitted using {diag.get('method', 'unknown')} method")
                    if 'r_squared' in diag:
                        st.sidebar.write(f"g(d) R²: {diag['r_squared']:.4f}")
                    if 'demand_range' in diag:
                        st.sidebar.write(f"Demand range: {diag['demand_range']}")
                    if 'ratio_range' in diag:
                        st.sidebar.write(f"Y=S/D range: {diag['ratio_range']}")
                        
            except FileNotFoundError as e:
                st.sidebar.error(f"File not found: {e}")
            except Exception as e:
                import traceback
                st.sidebar.error(f"Error loading data: {e}")
                st.sidebar.code(traceback.format_exc())
    
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
                                      help="Convergence threshold: if the change in the objective function value between consecutive iterations is smaller than the threshold, the algorithm considers it 'converged' and stops.")
    
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
    
    # Guard against empty trajectory list
    if not trajectories:
        st.warning("No trajectories loaded. Please check your data files.")
        return None
    
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
            
            # Highlight pickup location (always the last state in trajectory)
            pickup_idx = len(traj.states) - 1
            fig.add_trace(go.Scatter(
                x=[x_coords[pickup_idx]],
                y=[y_coords[pickup_idx]],
                mode='markers',
                name='Pickup Location',
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
            value=True,
            help="Load trained discriminator for F_fidelity computation"
        )
        
        if use_discriminator:
            checkpoint_path = st.text_input(
                "Checkpoint Path",
                "discriminator/model/checkpoints/pass-seek_5000-20000_(84ident_72same_44diff)/best.pt"  # this is the model with the best performance
                # "discriminator/model/checkpoints/best_model.pt"
            )
    
    with col2:
        show_iterations = st.checkbox("Show Iteration Details", value=True)
        save_results = st.checkbox("Save Results to File", value=False)
    
    # Gradient mode configuration
    st.subheader("🎯 Gradient Configuration")
    
    grad_col1, grad_col2 = st.columns(2)
    
    with grad_col1:
        gradient_mode = st.radio(
            "Gradient Mode",
            ["soft_cell", "heuristic"],
            index=0,
            help="""
            **soft_cell**: Use soft cell assignment for differentiable gradient flow.
            Gradients flow from objective through soft counts to pickup location.
            More accurate but computationally heavier.
            
            **heuristic**: Use heuristic gradient based on local DSR differences.
            Points toward underserved neighboring cells. Faster but less precise.
            """
        )
    
    with grad_col2:
        if gradient_mode == "soft_cell":
            temperature = st.number_input(
                "Temperature (τ)",
                min_value=0.01,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Soft assignment temperature. Lower = sharper (more like hard), Higher = softer (more spread)"
            )
            
            use_annealing = st.checkbox(
                "Temperature Annealing",
                value=False,
                help="Gradually reduce temperature from τ_max to τ_min over iterations"
            )
            
            if use_annealing:
                tau_max = st.number_input("τ_max", 0.1, 5.0, 1.0, 0.1)
                tau_min = st.number_input("τ_min", 0.01, 1.0, 0.1, 0.01)
            else:
                tau_max, tau_min = 1.0, 0.1
        else:
            temperature, use_annealing, tau_max, tau_min = 1.0, False, 1.0, 0.1
    
    if st.button("▶️ Run Modification", type="primary"):
        _execute_modification(
            selected_indices, params, use_discriminator, show_iterations, 
            checkpoint_path if use_discriminator else None,
            gradient_mode=gradient_mode,
            temperature=temperature,
            use_annealing=use_annealing,
            tau_max=tau_max,
            tau_min=tau_min,
        )


def _execute_modification(
    selected_indices: List[int], 
    params: Dict, 
    use_discriminator: bool,
    show_iterations: bool,
    checkpoint_path: Optional[str] = None,
    gradient_mode: str = 'soft_cell',
    temperature: float = 1.0,
    use_annealing: bool = False,
    tau_max: float = 1.0,
    tau_min: float = 0.1,
):
    """Execute the ST-iFGSM modification algorithm."""
    from trajectory_modification import (
        TrajectoryModifier, 
        FAMAILObjective, 
        GlobalMetrics,
        DiscriminatorAdapter,
        MissingComponentError,
    )
    
    device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    st.write(f"Running on device: **{device}**")
    st.write(f"Gradient mode: **{gradient_mode}**")
    if gradient_mode == 'soft_cell':
        st.write(f"Temperature: **{temperature}** {'(with annealing)' if use_annealing else ''}")
    
    # Initialize components
    discriminator = None
    if use_discriminator:
        try:
            # Use the checkpoint path provided by the user
            workspace_root = Path(__file__).parent.parent
            if checkpoint_path:
                full_checkpoint_path = workspace_root / checkpoint_path
            else:
                # Default fallback path
                full_checkpoint_path = workspace_root / 'discriminator' / 'model' / 'checkpoints' / 'pass-seek_5000-20000_(84ident_72same_44diff)' / 'best.pt'
            
            st.info(f"Loading discriminator from: `{full_checkpoint_path}`")
            
            if full_checkpoint_path.exists():
                adapter = DiscriminatorAdapter(checkpoint_path=full_checkpoint_path, device=device)
                discriminator = adapter.model
                st.success("✅ Discriminator loaded successfully")
            else:
                st.error(f"❌ Discriminator checkpoint not found: {full_checkpoint_path}")
                st.warning("Fidelity term will raise an error. Please provide a valid checkpoint path.")
        except Exception as e:
            st.error(f"❌ Error loading discriminator: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Create objective function
    try:
        objective = FAMAILObjective(
            alpha_spatial=params['alpha_spatial'],
            alpha_causal=params['alpha_causal'],
            alpha_fidelity=params['alpha_fidelity'],
            g_function=st.session_state.g_function,
            discriminator=discriminator,
        )
        st.success("✅ Objective function initialized")
    except ImportError as e:
        st.error(f"❌ Failed to initialize objective function: {e}")
        return
    
    # Create modifier with gradient configuration
    modifier = TrajectoryModifier(
        objective_fn=objective,
        alpha=params['alpha'],
        epsilon=params['epsilon'],
        max_iterations=params['max_iterations'],
        convergence_threshold=params['convergence_threshold'],
        gradient_mode=gradient_mode,
        temperature=temperature,
        temperature_annealing=use_annealing,
        tau_max=tau_max,
        tau_min=tau_min,
    )
    st.success(f"✅ Modifier initialized with {gradient_mode} gradients")
    
    # Set global state with separate causal tensors for proper F_causal scaling
    # Spatial fairness uses sum-aggregated pickup/dropoff (total counts)
    # Causal fairness uses mean-aggregated demand/supply (per-period averages)
    causal_demand_grid = st.session_state.get('causal_demand_grid')
    causal_supply_grid = st.session_state.get('causal_supply_grid')
    
    modifier.set_global_state(
        pickup_counts=torch.tensor(st.session_state.pickup_grid, device=device, dtype=torch.float32),
        dropoff_counts=torch.tensor(st.session_state.dropoff_grid, device=device, dtype=torch.float32),
        active_taxis=torch.tensor(st.session_state.active_taxis_grid, device=device, dtype=torch.float32),
        causal_demand=torch.tensor(causal_demand_grid, device=device, dtype=torch.float32) if causal_demand_grid is not None else None,
        causal_supply=torch.tensor(causal_supply_grid, device=device, dtype=torch.float32) if causal_supply_grid is not None else None,
    )
    
    # Display causal fairness scale info
    if causal_demand_grid is not None and causal_supply_grid is not None:
        st.info(f"📊 Causal fairness using mean-aggregated tensors:\n"
                f"- Demand range: [{causal_demand_grid.min():.2f}, {causal_demand_grid.max():.2f}]\n"
                f"- Supply range: [{causal_supply_grid.min():.2f}, {causal_supply_grid.max():.2f}]")
    else:
        st.warning("⚠️ Causal demand/supply grids not available. F_causal may have scale issues.")
    
    # Initialize metrics tracker
    metrics = GlobalMetrics(
        g_function=st.session_state.g_function,
        alpha_weights=(params['alpha_spatial'], params['alpha_causal'], params['alpha_fidelity'])
    )
    metrics.initialize_from_data(
        st.session_state.pickup_grid,
        st.session_state.dropoff_grid,
        st.session_state.active_taxis_grid,
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
            
            # Update metrics (pickup is always the last state)
            if history.iterations:
                orig_state = traj.states[-1]  # Pickup is last state
                mod_state = history.modified.states[-1]  # Modified pickup is also last state
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
    
    # Store objective debug info for display
    st.session_state.last_objective_debug = {
        'spatial': objective._last_spatial_debug,
        'causal': objective._last_causal_debug,
        'general': objective.last_debug,
    }
    
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
    
    # Guard against empty histories
    if not histories:
        st.warning("No trajectories were successfully modified.")
        return
    
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
    
    # Iteration details - always show if checkbox is checked
    if show_iterations:
        st.subheader("📈 Iteration Details")
        
        # Select which trajectory to view
        traj_options = [f"Trajectory {i}: {h.total_iterations} iterations" for i, h in enumerate(histories)]
        selected_traj = st.selectbox("Select Trajectory", range(len(histories)), format_func=lambda x: traj_options[x])
        
        history = histories[selected_traj]
        
        # Show trajectory summary
        st.markdown(f"""
        **Trajectory Summary:**
        - Original pickup: ({history.original.states[-1].x_grid:.2f}, {history.original.states[-1].y_grid:.2f})
        - Modified pickup: ({history.modified.states[-1].x_grid:.2f}, {history.modified.states[-1].y_grid:.2f})
        - Total iterations: {history.total_iterations}
        - Converged: {history.converged}
        - Final objective: {history.final_objective:.4f}
        """)
        
        if history.iterations:
            # Show iteration data as a table
            import pandas as pd
            iter_data = {
                'Iteration': list(range(1, len(history.iterations) + 1)),
                'Objective L': [r.objective_value for r in history.iterations],
                'F_spatial': [r.f_spatial for r in history.iterations],
                'F_causal': [r.f_causal for r in history.iterations],
                'F_fidelity': [r.f_fidelity for r in history.iterations],
                'Grad Norm': [r.gradient_norm for r in history.iterations],
                'δx': [r.perturbation[0] for r in history.iterations],
                'δy': [r.perturbation[1] for r in history.iterations],
            }
            df = pd.DataFrame(iter_data)
            st.dataframe(df, use_container_width=True)
            
            # Plot if we have plotly and more than 1 iteration
            if PLOTLY_AVAILABLE and len(history.iterations) > 1:
                st.markdown("**Objective Evolution:**")
                
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
                    title=f"Objective Function Evolution (Trajectory {selected_traj})",
                    xaxis_title="Iteration",
                    yaxis_title="Value",
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            elif len(history.iterations) == 1:
                st.info("Only 1 iteration - trajectory converged immediately.")
        else:
            st.warning("No iterations recorded for this trajectory.")
    
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
    
    # Debug info expander
    with st.expander("🔍 Debug Info (Objective Function Details)"):
        if 'last_objective_debug' in st.session_state:
            debug_info = st.session_state.last_objective_debug
            
            st.markdown("### Spatial Fairness Debug")
            if debug_info.get('spatial'):
                spatial_debug = debug_info['spatial']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Gini DSR", f"{spatial_debug.get('gini_dsr', 'N/A'):.4f}" if isinstance(spatial_debug.get('gini_dsr'), (int, float)) else "N/A")
                    st.metric("Active Cells", spatial_debug.get('n_active_cells', 'N/A'))
                with col2:
                    st.metric("Gini ASR", f"{spatial_debug.get('gini_asr', 'N/A'):.4f}" if isinstance(spatial_debug.get('gini_asr'), (int, float)) else "N/A")
                    if 'dsr_range' in spatial_debug:
                        st.write(f"DSR range: {spatial_debug['dsr_range']}")
                    if 'asr_range' in spatial_debug:
                        st.write(f"ASR range: {spatial_debug['asr_range']}")
            else:
                st.write("No spatial debug info available")
            
            st.markdown("### Causal Fairness Debug")
            if debug_info.get('causal'):
                causal_debug = debug_info['causal']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Active Cells", causal_debug.get('n_active_cells', 'N/A'))
                    r_sq = causal_debug.get('r_squared_raw', 'N/A')
                    st.metric("R² (raw)", f"{r_sq:.4f}" if isinstance(r_sq, (int, float)) else "N/A")
                    st.metric("Var(Y)", f"{causal_debug.get('var_Y', 'N/A'):.4f}" if isinstance(causal_debug.get('var_Y'), (int, float)) else "N/A")
                with col2:
                    st.metric("Var(R)", f"{causal_debug.get('var_R', 'N/A'):.4f}" if isinstance(causal_debug.get('var_R'), (int, float)) else "N/A")
                    if 'demand_range' in causal_debug:
                        st.write(f"Demand range: {causal_debug['demand_range']}")
                    if 'Y_range' in causal_debug:
                        st.write(f"Y (S/D) range: {causal_debug['Y_range']}")
                    if 'g_d_range' in causal_debug:
                        st.write(f"g(D) range: {causal_debug['g_d_range']}")
                
                # Explanation if R² is negative
                if isinstance(r_sq, (int, float)) and r_sq < 0:
                    st.warning(f"""
                    **⚠️ F_causal = 0 because R² is negative ({r_sq:.4f})**
                    
                    This means `Var(residuals) > Var(Y)`, indicating a scale mismatch between:
                    - The actual Y = Supply/Demand values
                    - The g(D) function predictions
                    
                    **Possible causes:**
                    1. The g(D) function was fitted on different data scales (e.g., hourly vs. total)
                    2. The supply/demand definition doesn't match what g(D) expects
                    3. Missing or incorrect g_function_params.json file
                    """)
            else:
                st.write("No causal debug info available")
            
            st.markdown("### General Debug")
            st.json(debug_info.get('general', {}))
        else:
            st.info("Run a modification first to see debug info.")


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
            st.session_state.pickup_grid.T,
            labels={'x': 'Grid X', 'y': 'Grid Y', 'color': 'Pickups'},
            title="Original Pickup Distribution",
            aspect='auto',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("After Modification")
        
        # Get modified counts if available
        if hasattr(st.session_state, 'modified_pickup_grid'):
            modified_counts = st.session_state.modified_pickup_grid
        else:
            # Use original if no modifications yet
            modified_counts = st.session_state.pickup_grid
        
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
