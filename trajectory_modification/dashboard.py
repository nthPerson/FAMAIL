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
if 'last_initial_snapshot' not in st.session_state:
    st.session_state.last_initial_snapshot = None
if 'last_final_snapshot' not in st.session_state:
    st.session_state.last_final_snapshot = None
if 'selected_traj_idx' not in st.session_state:
    st.session_state.selected_traj_idx = 0


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
# Attribution-Based Trajectory Selection
# ============================================================================

def compute_cell_lis_scores(cell_counts: np.ndarray) -> np.ndarray:
    """
    Compute Local Inequality Score (LIS) for all cells.
    
    LIS measures how much each cell deviates from the global mean,
    normalized by the mean: LIS_i = |c_i - μ| / μ
    
    Higher LIS indicates the cell contributes more to spatial inequality.
    
    Args:
        cell_counts: Count array [grid_x, grid_y] (e.g., pickup counts)
        
    Returns:
        LIS array [grid_x, grid_y] with values >= 0
    """
    mean_val = cell_counts.mean()
    
    if mean_val < 1e-8:
        return np.zeros_like(cell_counts)
    
    lis = np.abs(cell_counts - mean_val) / mean_val
    return lis


def compute_cell_dcd_scores(
    demand_counts: np.ndarray,
    supply_counts: np.ndarray,
    g_function,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute Demand-Conditional Deviation (DCD) for all cells.
    
    DCD measures how much the actual service ratio Y = S/D deviates
    from the expected fair ratio g(D): DCD_i = |Y_i - g(D_i)|
    
    Higher DCD indicates the cell deviates more from causal fairness.
    
    Args:
        demand_counts: Demand array [grid_x, grid_y] (pickups)
        supply_counts: Supply array [grid_x, grid_y] (active taxis or dropoffs)
        g_function: Pre-fitted g(d) function mapping demand -> expected ratio
        eps: Numerical stability constant
        
    Returns:
        DCD array [grid_x, grid_y] with values >= 0
    """
    dcd = np.zeros_like(demand_counts, dtype=float)
    
    # If no g_function available, use simple deviation from mean
    if g_function is None:
        # Fallback: use deviation from mean service ratio
        mask = demand_counts > eps
        if not mask.any():
            return dcd
        Y = np.zeros_like(demand_counts, dtype=float)
        Y[mask] = supply_counts[mask] / (demand_counts[mask] + eps)
        mean_Y = Y[mask].mean()
        dcd[mask] = np.abs(Y[mask] - mean_Y)
        return dcd
    
    # Only compute for cells with sufficient demand
    mask = demand_counts > eps
    
    if not mask.any():
        return dcd
    
    # Compute actual service ratio Y = S / D for active cells
    Y = np.zeros_like(demand_counts, dtype=float)
    Y[mask] = supply_counts[mask] / (demand_counts[mask] + eps)
    
    # Compute expected ratio g(D) for all cells
    D_flat = demand_counts.flatten()
    g_d_flat = g_function(D_flat)
    g_d = g_d_flat.reshape(demand_counts.shape)
    
    # DCD = |Y - g(D)| for active cells
    dcd[mask] = np.abs(Y[mask] - g_d[mask])
    
    return dcd


def compute_trajectory_attribution_scores(
    trajectories: List,
    pickup_counts: np.ndarray,
    dropoff_counts: np.ndarray,
    supply_counts: np.ndarray,
    g_function,
    lis_weight: float = 0.5,
    dcd_weight: float = 0.5,
    normalize: bool = True,
) -> List[Dict]:
    """
    Compute combined attribution scores for all trajectories.
    
    Combined Score = w_lis * LIS_τ + w_dcd * DCD_τ
    
    Where:
    - LIS_τ = max(LIS_pickup, LIS_dropoff) for trajectory τ
    - DCD_τ = DCD at the pickup cell (since we modify pickups)
    
    Trajectories with higher combined scores have greater impact on
    fairness and should be prioritized for modification.
    
    Args:
        trajectories: List of Trajectory objects
        pickup_counts: Pickup count grid [grid_x, grid_y]
        dropoff_counts: Dropoff count grid [grid_x, grid_y]
        supply_counts: Supply grid [grid_x, grid_y] for DCD computation
        g_function: Pre-fitted g(d) function
        lis_weight: Weight for LIS (spatial fairness impact)
        dcd_weight: Weight for DCD (causal fairness impact)
        normalize: If True, normalize scores to [0, 1] range
        
    Returns:
        List of dicts with trajectory index, scores, and cell info
    """
    # Compute cell-level LIS scores
    pickup_lis = compute_cell_lis_scores(pickup_counts)
    dropoff_lis = compute_cell_lis_scores(dropoff_counts)
    
    # Compute cell-level DCD scores
    dcd_scores = compute_cell_dcd_scores(pickup_counts, supply_counts, g_function)
    
    # Compute per-trajectory scores
    trajectory_scores = []
    
    for idx, traj in enumerate(trajectories):
        # Get pickup cell (last state) and dropoff cell (first state)
        pickup_state = traj.states[-1]
        dropoff_state = traj.states[0]
        
        pickup_cell = (int(pickup_state.x_grid), int(pickup_state.y_grid))
        dropoff_cell = (int(dropoff_state.x_grid), int(dropoff_state.y_grid))
        
        # Get LIS values for this trajectory's cells
        px, py = pickup_cell
        dx, dy = dropoff_cell
        
        lis_pickup = 0.0
        if 0 <= px < pickup_lis.shape[0] and 0 <= py < pickup_lis.shape[1]:
            lis_pickup = float(pickup_lis[px, py])
        
        lis_dropoff = 0.0
        if 0 <= dx < dropoff_lis.shape[0] and 0 <= dy < dropoff_lis.shape[1]:
            lis_dropoff = float(dropoff_lis[dx, dy])
        
        # Trajectory LIS = max of pickup and dropoff LIS
        traj_lis = max(lis_pickup, lis_dropoff)
        
        # Get DCD value at pickup cell (we modify pickups, so focus on demand impact there)
        traj_dcd = 0.0
        if 0 <= px < dcd_scores.shape[0] and 0 <= py < dcd_scores.shape[1]:
            traj_dcd = float(dcd_scores[px, py])
        
        trajectory_scores.append({
            'index': idx,
            'trajectory_id': getattr(traj, 'trajectory_id', idx),
            'driver_id': getattr(traj, 'driver_id', 'N/A'),
            'lis_score': traj_lis,
            'dcd_score': traj_dcd,
            'pickup_cell': pickup_cell,
            'dropoff_cell': dropoff_cell,
            'lis_pickup': lis_pickup,
            'lis_dropoff': lis_dropoff,
        })
    
    if not trajectory_scores:
        return []
    
    # Normalize scores to [0, 1] if requested
    if normalize:
        lis_values = np.array([s['lis_score'] for s in trajectory_scores])
        dcd_values = np.array([s['dcd_score'] for s in trajectory_scores])
        
        lis_max = lis_values.max() if lis_values.max() > 0 else 1.0
        dcd_max = dcd_values.max() if dcd_values.max() > 0 else 1.0
        
        for s in trajectory_scores:
            s['lis_score_normalized'] = s['lis_score'] / lis_max
            s['dcd_score_normalized'] = s['dcd_score'] / dcd_max
    else:
        for s in trajectory_scores:
            s['lis_score_normalized'] = s['lis_score']
            s['dcd_score_normalized'] = s['dcd_score']
    
    # Compute combined scores using normalized values
    for s in trajectory_scores:
        s['combined_score'] = (
            lis_weight * s['lis_score_normalized'] + 
            dcd_weight * s['dcd_score_normalized']
        )
    
    return trajectory_scores


def select_top_k_by_attribution(
    trajectory_scores: List[Dict],
    k: int,
    selection_method: str = 'top_k',
) -> List[int]:
    """
    Select top-k trajectory indices by attribution score.
    
    Args:
        trajectory_scores: List of score dicts from compute_trajectory_attribution_scores
        k: Number of trajectories to select
        selection_method: 'top_k' (highest scores) or 'diverse' (spread across cells)
        
    Returns:
        List of trajectory indices sorted by combined score (descending)
    """
    if not trajectory_scores:
        return []
    
    if selection_method == 'top_k':
        # Sort by combined score descending and take top k
        sorted_scores = sorted(trajectory_scores, key=lambda x: x['combined_score'], reverse=True)
        return [s['index'] for s in sorted_scores[:k]]
    
    elif selection_method == 'diverse':
        # Greedy selection with penalty for already-selected cells
        # This spreads modifications across different areas
        sorted_scores = sorted(trajectory_scores, key=lambda x: x['combined_score'], reverse=True)
        
        selected = []
        cell_penalty = {}  # cell -> penalty count
        penalty_factor = 0.5
        
        for s in sorted_scores:
            if len(selected) >= k:
                break
            
            # Apply penalty for cells already selected
            adjusted_score = s['combined_score']
            pickup_cell = s['pickup_cell']
            
            if pickup_cell in cell_penalty:
                adjusted_score *= (1 - penalty_factor * cell_penalty[pickup_cell])
            
            # Always add if above threshold or if we don't have enough yet
            if adjusted_score > 0 or len(selected) < k:
                selected.append(s['index'])
                cell_penalty[pickup_cell] = cell_penalty.get(pickup_cell, 0) + 1
        
        return selected
    
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")


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
            ["Single", "Batch", "Random Sample", "Top-k by Fairness Impact"],
            horizontal=False  # Vertical layout to accommodate longer option name
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
            
        elif selection_mode == "Random Sample":
            sample_size = st.number_input("Sample Size", 1, min(50, len(trajectories)), 10)
            if st.button("🎲 Resample"):
                selected_indices = np.random.choice(len(trajectories), sample_size, replace=False).tolist()
                st.session_state.selected_indices = selected_indices
            selected_indices = st.session_state.get('selected_indices', list(range(sample_size)))
            
        else:  # Top-k by Fairness Impact
            st.markdown("**Attribution-based Selection**")
            st.caption("Selects trajectories that have the highest impact on fairness metrics (LIS + DCD)")
            
            # Check if required data is loaded
            pickup_grid = st.session_state.get('pickup_grid')
            dropoff_grid = st.session_state.get('dropoff_grid')
            active_taxis_grid = st.session_state.get('active_taxis_grid')
            g_function = st.session_state.get('g_function')
            
            if pickup_grid is None or dropoff_grid is None or active_taxis_grid is None:
                st.warning("⚠️ Required grid data not loaded. Please ensure pickup_dropoff_counts.pkl and active_taxis are available.")
                selected_indices = []
            else:
                # k value
                k_value = st.number_input(
                    "Top-k Trajectories", 
                    min_value=1, 
                    max_value=min(100, len(trajectories)), 
                    value=10,
                    help="Number of highest-impact trajectories to select"
                )
                
                # Weight sliders for LIS and DCD
                st.markdown("**Attribution Weights:**")
                lis_weight = st.slider(
                    "LIS Weight (Spatial Fairness)", 
                    min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                    help="Weight for Local Inequality Score (|c_i - μ| / μ)"
                )
                dcd_weight = st.slider(
                    "DCD Weight (Causal Fairness)", 
                    min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                    help="Weight for Demand-Conditional Deviation (|Y_i - g(D_i)|)"
                )
                
                # Selection method
                selection_method = st.radio(
                    "Selection Strategy",
                    ["Top-k", "Diverse (spread across cells)"],
                    horizontal=True,
                    help="Top-k: highest scores. Diverse: spreads selection across different areas."
                )
                method_key = "top_k" if selection_method == "Top-k" else "diverse"
                
                # Compute button
                if st.button("🎯 Compute Attribution Scores", type="primary"):
                    with st.spinner("Computing attribution scores for all trajectories..."):
                        try:
                            # Compute attribution scores for all trajectories
                            trajectory_scores = compute_trajectory_attribution_scores(
                                trajectories=trajectories,
                                pickup_counts=pickup_grid,
                                dropoff_counts=dropoff_grid,
                                supply_counts=active_taxis_grid,
                                g_function=g_function,
                                lis_weight=lis_weight,
                                dcd_weight=dcd_weight,
                                normalize=True
                            )
                            
                            # Select top-k
                            selected_indices = select_top_k_by_attribution(
                                trajectory_scores=trajectory_scores,
                                k=k_value,
                                selection_method=method_key
                            )
                            
                            # Store in session state
                            st.session_state.attribution_scores = trajectory_scores
                            st.session_state.attribution_selected_indices = selected_indices
                            
                            # Show statistics
                            selected_scores = [trajectory_scores[i] for i in selected_indices]
                            avg_combined = np.mean([s['combined_score'] for s in selected_scores])
                            avg_lis = np.mean([s['lis_score'] for s in selected_scores])
                            avg_dcd = np.mean([s['dcd_score'] for s in selected_scores])
                            
                            st.success(f"✅ Selected {len(selected_indices)} high-impact trajectories")
                            st.markdown(f"""
                            **Selection Statistics:**
                            - Avg Combined Score: `{avg_combined:.4f}`
                            - Avg LIS Score: `{avg_lis:.4f}`
                            - Avg DCD Score: `{avg_dcd:.4f}`
                            """)
                            
                        except Exception as e:
                            st.error(f"Error computing attribution: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                            selected_indices = []
                
                # Use cached selection if available
                selected_indices = st.session_state.get('attribution_selected_indices', [])
                
                # Show score details for selected trajectories
                if selected_indices and 'attribution_scores' in st.session_state:
                    with st.expander("📊 Attribution Score Details", expanded=False):
                        scores = st.session_state.attribution_scores
                        for idx in selected_indices[:10]:  # Show top 10 details
                            s = scores[idx]
                            st.markdown(f"**Traj {idx}** (Driver {trajectories[idx].driver_id}): "
                                       f"Combined={s['combined_score']:.4f}, "
                                       f"LIS={s['lis_score']:.4f}, "
                                       f"DCD={s['dcd_score']:.4f}, "
                                       f"Pickup=({s['pickup_cell'][0]}, {s['pickup_cell'][1]})")
        
        st.write(f"**Selected:** {len(selected_indices)} trajectories")
        
        # Show trajectory statistics for selected trajectories
        if selected_indices:
            selected_trajs = [trajectories[i] for i in selected_indices]
            state_counts = [len(t.states) for t in selected_trajs]
            driver_ids = set(t.driver_id for t in selected_trajs)
            st.caption(f"States per traj: min={min(state_counts)}, max={max(state_counts)}, avg={np.mean(state_counts):.1f}")
            st.caption(f"Unique drivers: {len(driver_ids)} ({sorted(driver_ids)})")
        
    with col2:
        st.subheader("Trajectory Preview")
        
        if selected_indices and PLOTLY_AVAILABLE:
            # Determine which trajectories to display based on selection mode
            if selection_mode == "Single":
                display_indices = [selected_indices[0]]
            else:
                # For Batch or Random Sample, show all selected trajectories
                display_indices = selected_indices
            
            # Color palette for multiple trajectories
            colors = px.colors.qualitative.Plotly
            
            fig = go.Figure()
            
            # Grid dimensions (original data is 48 x 90)
            # After transformation, display will be 90 wide x 48 tall
            orig_grid_x, orig_grid_y = 48, 90
            display_width, display_height = orig_grid_y, orig_grid_x  # 90 x 48 after transform
            
            def transform_coords(x, y):
                """
                Transform coordinates from data grid to display grid.
                Original data: (x, y) in 48x90 grid (x: 0-47, y: 0-89)
                Display: 90 wide × 48 tall (swap x and y for horizontal layout)
                """
                new_x = y
                new_y = x
                return new_x, new_y
            
            for i, traj_idx in enumerate(display_indices):
                traj = trajectories[traj_idx]
                color = colors[i % len(colors)]
                
                # Transform coordinates for display
                transformed_coords = [transform_coords(s.x_grid, s.y_grid) for s in traj.states]
                x_coords = [c[0] for c in transformed_coords]
                y_coords = [c[1] for c in transformed_coords]
                
                # Get trajectory and driver IDs for label
                traj_id = getattr(traj, 'trajectory_id', 'N/A')
                driver_id = getattr(traj, 'driver_id', 'N/A')
                n_states = len(traj.states)
                label = f"Traj {traj_idx} (D:{driver_id}, {n_states}pts)"
                
                # Draw trajectory path
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines+markers',
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    legendgroup=f"traj_{traj_idx}",
                    hovertemplate=f"{label}<br>State %{{customdata}}<br>Display: (%{{x:.1f}}, %{{y:.1f}})<extra></extra>",
                    customdata=list(range(len(x_coords))),
                ))
                
                # Add direction arrows between states
                for j in range(len(x_coords) - 1):
                    x_start, y_start = x_coords[j], y_coords[j]
                    x_end, y_end = x_coords[j + 1], y_coords[j + 1]
                    
                    # Calculate midpoint for arrow placement
                    mid_x = (x_start + x_end) / 2
                    mid_y = (y_start + y_end) / 2
                    
                    # Add arrow annotation
                    fig.add_annotation(
                        x=mid_x + (x_end - x_start) * 0.2,
                        y=mid_y + (y_end - y_start) * 0.2,
                        ax=mid_x - (x_end - x_start) * 0.2,
                        ay=mid_y - (y_end - y_start) * 0.2,
                        xref="x", yref="y",
                        axref="x", ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.2,
                        arrowwidth=1.5,
                        arrowcolor=color,
                        opacity=0.7,
                    )
                
                # Mark dropoff location (first state) with a circle
                fig.add_trace(go.Scatter(
                    x=[x_coords[0]],
                    y=[y_coords[0]],
                    mode='markers',
                    name=f'Dropoff (Traj {traj_idx})',
                    marker=dict(color=color, size=14, symbol='circle', 
                               line=dict(color='white', width=2)),
                    legendgroup=f"traj_{traj_idx}",
                    showlegend=False,
                    hovertemplate=f"Dropoff<br>{label}<br>Display: (%{{x:.1f}}, %{{y:.1f}})<extra></extra>",
                ))
                
                # Mark pickup location (last state) with a star
                fig.add_trace(go.Scatter(
                    x=[x_coords[-1]],
                    y=[y_coords[-1]],
                    mode='markers',
                    name=f'Pickup (Traj {traj_idx})',
                    marker=dict(color=color, size=16, symbol='star',
                               line=dict(color='white', width=1)),
                    legendgroup=f"traj_{traj_idx}",
                    showlegend=False,
                    hovertemplate=f"Pickup<br>{label}<br>Display: (%{{x:.1f}}, %{{y:.1f}})<extra></extra>",
                ))
            
            # Configure layout for square grid cells (now 90 wide x 48 tall)
            # dtick=1 shows every cell boundary
            fig.update_layout(
                title=f"Trajectory Preview ({len(display_indices)} trajectory{'s' if len(display_indices) > 1 else ''})  |  ⭐ = Pickup  ● = Dropoff",
                xaxis=dict(
                    title="Grid (transformed)",
                    range=[-0.5, display_width - 0.5],
                    constrain="domain",
                    dtick=1,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)',
                    minor=dict(dtick=1, showgrid=True, gridcolor='rgba(128, 128, 128, 0.15)'),
                ),
                yaxis=dict(
                    title="",
                    range=[-0.5, display_height - 0.5],
                    scaleanchor="x",
                    scaleratio=1,
                    dtick=1,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)',
                    minor=dict(dtick=1, showgrid=True, gridcolor='rgba(128, 128, 128, 0.15)'),
                ),
                height=450,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.25,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255, 255, 255, 0.9)",
                ),
                margin=dict(b=100),  # Make room for legend below
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
    st.session_state.last_initial_snapshot = initial_snapshot
    st.session_state.last_final_snapshot = final_snapshot
    st.session_state.selected_traj_idx = 0  # Reset selection when new results are generated
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
    
    st.success(f"✅ Modification complete! Modified {len(histories)} trajectories.")


def display_results_section():
    """
    Display modification results from session state.
    
    This function is called from main() and reads results from session state,
    allowing the results to persist across reruns triggered by widget interactions.
    """
    st.header("📊 Results")
    
    # Check if we have results to display
    histories = st.session_state.get('modification_history', [])
    initial_snapshot = st.session_state.get('last_initial_snapshot')
    final_snapshot = st.session_state.get('last_final_snapshot')
    
    if not histories or initial_snapshot is None or final_snapshot is None:
        st.info("Run a trajectory modification to see results here.")
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
    
    # Iteration details section with fragment to prevent full reruns
    _display_iteration_details_fragment(histories)
    
    # Perturbation visualization
    _display_perturbation_histogram(histories)
    
    # Debug info
    _display_debug_info()


@st.fragment
def _display_iteration_details_fragment(histories):
    """
    Display iteration details for individual trajectories.
    
    Using @st.fragment prevents full app reruns when the user cycles
    through trajectories with the number input widget.
    """
    st.subheader("📈 Iteration Details")
    
    # Ensure selected index is valid
    max_idx = len(histories) - 1
    current_idx = st.session_state.get('selected_traj_idx', 0)
    if current_idx > max_idx:
        current_idx = 0
        st.session_state.selected_traj_idx = 0
    
    # Trajectory selector with info display
    col_select, col_info = st.columns([1, 2])
    
    with col_select:
        selected_traj = st.number_input(
            "Select Trajectory",
            min_value=0,
            max_value=max_idx,
            value=current_idx,
            step=1,
            key="traj_selector",
            help="Use +/- to cycle through modified trajectories",
            on_change=_on_traj_selection_change,
        )
    
    history = histories[selected_traj]
    
    # Display trajectory and agent identification info
    with col_info:
        traj_id = getattr(history.original, 'trajectory_id', 'N/A')
        driver_id = getattr(history.original, 'driver_id', 'N/A')
        st.markdown(f"""
        **Trajectory {selected_traj + 1} of {len(histories)}**  
        📋 Trajectory ID: `{traj_id}`  
        🚕 Agent/Driver ID: `{driver_id}`
        """)
    
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


def _on_traj_selection_change():
    """Callback to update session state when trajectory selection changes."""
    st.session_state.selected_traj_idx = st.session_state.traj_selector


def _display_perturbation_histogram(histories):
    """Display histogram of perturbation magnitudes."""
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


def _display_debug_info():
    """Display debug information in an expander."""
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
    
    # Display results from session state (persists across widget interactions)
    display_results_section()
    
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
