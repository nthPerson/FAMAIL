"""
FAMAIL Trajectory Modification Dashboard

Streamlit dashboard for testing trajectory modification with the ST-iFGSM algorithm.
Allows visualization of trajectory modifications and fairness metric changes.
"""

import streamlit as st
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple
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

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for Streamlit
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


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

                # Store hat matrices and demographics for Option B/C formulations
                st.session_state.hat_matrices = bundle.hat_matrices
                st.session_state.g0_power_basis_func = bundle.g0_power_basis_func
                st.session_state.active_cell_indices = bundle.active_cell_indices

                # Store multi-stream data for V3 discriminator
                st.session_state.ms_driving_trajs = bundle.ms_driving_trajs
                st.session_state.ms_seeking_trajs = bundle.ms_seeking_trajs
                st.session_state.ms_profile_features = bundle.ms_profile_features
                st.session_state.ms_seeking_days = bundle.ms_seeking_days
                st.session_state.ms_driving_days = bundle.ms_driving_days

                st.session_state.data_loaded = True
                st.sidebar.success(f"Loaded {len(bundle.trajectories)} trajectories")

                # Report multi-stream data status
                if bundle.ms_driving_trajs is not None:
                    n_drivers = len(bundle.ms_driving_trajs)
                    n_driving = sum(len(v) for v in bundle.ms_driving_trajs.values())
                    st.sidebar.info(f"V3 multi-stream: {n_drivers} drivers, {n_driving} driving trajs")
                
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
        conv_thresh = st.number_input("Convergence", 1e-8, 1e-2, 1e-6, format="%.1e",
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

    st.sidebar.subheader("F_causal Formulation")
    causal_formulation = st.sidebar.selectbox(
        "Formulation",
        ["baseline", "option_b", "option_c"],
        index=1,
        key="causal_formulation",
        help=(
            "**baseline**: Historical R² with isotonic g₀(D). "
            "**option_b** (default): Demographic residual independence via hat matrix. "
            "**option_c**: Partial ΔR² via dual hat matrices. "
            "See FCAUSAL_FORMULATIONS.md for details."
        ),
    )

    return {
        'alpha': alpha,
        'epsilon': epsilon,
        'max_iterations': max_iter,
        'convergence_threshold': conv_thresh,
        'alpha_spatial': w1,
        'alpha_causal': w2,
        'alpha_fidelity': w3,
        'causal_formulation': causal_formulation,
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
    causal_formulation: str = "baseline",
    hat_matrices: Optional[Dict] = None,
    g0_power_basis_func: Optional[Callable] = None,
    active_cell_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Demand-Conditional Deviation (DCD) for all cells.

    Supports two formulations:
    - **Baseline**: DCD_c = |Y_c - g(D_c)| — deviation from demand-predicted service ratio.
    - **Option B**: DCD_c = |R̂_c| where R̂ = H_demo @ R and R = Y - g₀(D).
      Measures how much of the demand-controlled residual is explained by demographics.
      Cells where demographics strongly predict the residual are prioritized for modification.

    Args:
        demand_counts: Demand array [grid_x, grid_y] (pickups)
        supply_counts: Supply array [grid_x, grid_y] (active taxis or dropoffs)
        g_function: Pre-fitted g(d) function mapping demand -> expected ratio
        eps: Numerical stability constant
        causal_formulation: 'baseline' or 'option_b'
        hat_matrices: Pre-computed hat matrices (required for option_b).
                     Must contain 'H_demo' key with (n, n) demographic hat matrix.
        g0_power_basis_func: Power basis g₀(D) function (required for option_b)
        active_cell_indices: Boolean mask identifying active cells for hat matrix (required for option_b)

    Returns:
        DCD array [grid_x, grid_y] with values >= 0
    """
    dcd = np.zeros_like(demand_counts, dtype=float)

    # ── Option B: Demographic residual projection ──
    if (causal_formulation == "option_b"
            and hat_matrices is not None
            and g0_power_basis_func is not None
            and active_cell_indices is not None):

        H_demo = hat_matrices.get('H_demo')
        if H_demo is not None:
            # Extract active cells (aligned with hat matrix rows)
            D = demand_counts.flatten()[active_cell_indices]
            S = supply_counts.flatten()[active_cell_indices]
            Y = S / (D + eps)

            # Demand-controlled residuals: R = Y - g₀(D)
            g0_D = g0_power_basis_func(D)
            R = Y - g0_D

            # Project residuals onto demographics: R̂ = H_demo @ R
            R_hat = H_demo @ R

            # DCD = magnitude of demographically-explained residual per cell
            DCD_active = np.abs(R_hat)

            # Map back to full grid
            dcd_flat = dcd.flatten()
            dcd_flat[active_cell_indices] = DCD_active
            dcd = dcd_flat.reshape(demand_counts.shape)

            return dcd

    # ── Baseline: |Y - g(D)| ──

    # If no g_function available, use simple deviation from mean
    if g_function is None:
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
    causal_formulation: str = "baseline",
    hat_matrices: Optional[Dict] = None,
    g0_power_basis_func: Optional[Callable] = None,
    active_cell_indices: Optional[np.ndarray] = None,
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
        causal_formulation: 'baseline' or 'option_b' — controls DCD computation
        hat_matrices: Pre-computed hat matrices (for option_b DCD)
        g0_power_basis_func: Power basis g₀(D) function (for option_b DCD)
        active_cell_indices: Boolean mask for hat matrix cells (for option_b DCD)

    Returns:
        List of dicts with trajectory index, scores, and cell info
    """
    # Compute cell-level LIS scores
    pickup_lis = compute_cell_lis_scores(pickup_counts)
    dropoff_lis = compute_cell_lis_scores(dropoff_counts)

    # Compute cell-level DCD scores (dispatches to baseline or option_b)
    dcd_scores = compute_cell_dcd_scores(
        pickup_counts, supply_counts, g_function,
        causal_formulation=causal_formulation,
        hat_matrices=hat_matrices,
        g0_power_basis_func=g0_power_basis_func,
        active_cell_indices=active_cell_indices,
    )
    
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
            horizontal=False,  # Vertical layout to accommodate longer option name
            index=3
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
                                normalize=True,
                                causal_formulation=st.session_state.get('causal_formulation', 'baseline'),
                                hat_matrices=st.session_state.get('hat_matrices'),
                                g0_power_basis_func=st.session_state.get('g0_power_basis_func'),
                                active_cell_indices=st.session_state.get('active_cell_indices'),
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

        if selected_indices and MATPLOTLIB_AVAILABLE:
            # Determine which trajectories to display based on selection mode
            if selection_mode == "Single":
                display_indices = [selected_indices[0]]
            else:
                display_indices = selected_indices

            colors = _get_trajectory_color_palette(len(display_indices))

            fig, ax = plt.subplots(figsize=(10, 5.5))

            all_display_x = []
            all_display_y = []

            for i, traj_idx in enumerate(display_indices):
                traj = trajectories[traj_idx]
                color = colors[i]

                driver_id = getattr(traj, 'driver_id', 'N/A')
                n_states = len(traj.states)
                label = f"Traj {traj_idx} (D:{driver_id}, {n_states}pts)"

                _plot_trajectory_matplotlib(ax, traj.states, color=color, label=label)

                all_display_x.extend(s.y_grid for s in traj.states)
                all_display_y.extend(s.x_grid for s in traj.states)

            _set_trajectory_axes_bounds(ax, all_display_x, all_display_y)

            ax.set_title(
                f"Trajectory Preview ({len(display_indices)} traj)  |  "
                f"x = Start (Dropoff)   * = End (Pickup)",
                fontsize=10,
            )
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.legend(
                loc='upper center', bbox_to_anchor=(0.5, -0.08),
                ncol=min(3, len(display_indices)),
                fontsize=8, framealpha=0.9,
            )

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        elif selected_indices and not MATPLOTLIB_AVAILABLE:
            st.warning("Matplotlib not available for trajectory visualization.")
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
                "checkpoints/20260316_223817/best.pt"  # V3 multi-stream (Ren-aligned)
                # "discriminator/model/checkpoints/pass-seek_5000-20000_(84ident_72same_44diff)/best.pt"  # V2 single-stream
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
                value=True,
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
    use_annealing: bool = True,
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
                model_version = getattr(adapter, 'model_version', 'v1')
                st.success(f"✅ Discriminator loaded successfully (version: {model_version})")
            else:
                st.error(f"❌ Discriminator checkpoint not found: {full_checkpoint_path}")
                st.warning("Fidelity term will raise an error. Please provide a valid checkpoint path.")
        except Exception as e:
            st.error(f"❌ Error loading discriminator: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Create objective function
    causal_formulation = params.get('causal_formulation', 'baseline')
    try:
        objective = FAMAILObjective(
            alpha_spatial=params['alpha_spatial'],
            alpha_causal=params['alpha_causal'],
            alpha_fidelity=params['alpha_fidelity'],
            g_function=st.session_state.g_function,
            discriminator=discriminator,
            causal_formulation=causal_formulation,
            hat_matrices=st.session_state.get('hat_matrices'),
            g0_power_basis_func=st.session_state.get('g0_power_basis_func'),
            active_cell_indices=st.session_state.get('active_cell_indices'),
        )
        st.success(f"✅ Objective function initialized (F_causal: {causal_formulation})")
    except ImportError as e:
        st.error(f"❌ Failed to initialize objective function: {e}")
        return
    
    # Build multi-stream context for V3 discriminator if available
    multi_stream_context = None
    ms_driving = st.session_state.get('ms_driving_trajs')
    if ms_driving is not None and getattr(adapter, 'model_version', 'v1') == 'v3':
        try:
            from trajectory_modification.multi_stream_context import MultiStreamContextBuilder
            multi_stream_context = MultiStreamContextBuilder(
                driving_trajs=ms_driving,
                seeking_trajs=st.session_state.get('ms_seeking_trajs', {}),
                profile_features=st.session_state.get('ms_profile_features', {}),
                seeking_days=st.session_state.get('ms_seeking_days'),
                driving_days=st.session_state.get('ms_driving_days'),
                device=device,
            )
            st.success("✅ V3 multi-stream context loaded (3 streams: seeking + driving + profile)")
        except Exception as e:
            st.warning(f"⚠️ Failed to build multi-stream context: {e}. Using single-stream mode.")

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
        multi_stream_context=multi_stream_context,
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
        alpha_weights=(params['alpha_spatial'], params['alpha_causal'], params['alpha_fidelity']),
        hat_matrices=st.session_state.get('hat_matrices'),
        active_cell_indices=st.session_state.get('active_cell_indices'),
        g0_power_basis_func=st.session_state.get('g0_power_basis_func'),
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
    
    # Summary metrics: Before/After comparison
    st.subheader("Summary Metrics")

    # "Before" row
    st.caption("**Before** (initial state)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric(
        "F_spatial",
        f"{initial_snapshot.f_spatial:.4f}",
        help="Initial F_spatial score (1 − Gini). Higher = more equitable spatial distribution.",
    )
    b2.metric(
        "F_causal",
        f"{initial_snapshot.f_causal:.4f}",
        help="Initial F_causal score (Option B: demographic residual independence). Higher = less demographic bias.",
    )
    b3.metric(
        "F_fidelity",
        f"{initial_snapshot.f_fidelity:.4f}",
        help="Initial fidelity score (discriminator authenticity). Higher = more realistic trajectories.",
    )
    b4.metric(
        "Combined L",
        f"{initial_snapshot.combined_objective:.4f}",
        help="Initial weighted objective L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity.",
    )

    # "After" row with deltas
    st.caption("**After** (post-modification)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric(
        "F_spatial",
        f"{final_snapshot.f_spatial:.4f}",
        f"{final_snapshot.f_spatial - initial_snapshot.f_spatial:+.4f}",
        help="Final F_spatial score after trajectory modification.",
    )
    a2.metric(
        "F_causal",
        f"{final_snapshot.f_causal:.4f}",
        f"{final_snapshot.f_causal - initial_snapshot.f_causal:+.4f}",
        help="Final F_causal score after trajectory modification.",
    )
    a3.metric(
        "F_fidelity",
        f"{final_snapshot.f_fidelity:.4f}",
        f"{final_snapshot.f_fidelity - initial_snapshot.f_fidelity:+.4f}",
        help="Final fidelity score after trajectory modification.",
    )
    a4.metric(
        "Combined L",
        f"{final_snapshot.combined_objective:.4f}",
        f"{final_snapshot.combined_objective - initial_snapshot.combined_objective:+.4f}",
        help="Final weighted objective L after trajectory modification.",
    )
    
    # Modification statistics
    st.subheader("Modification Statistics")

    converged = sum(1 for h in histories if h.converged)
    avg_iterations = np.mean([h.total_iterations for h in histories])

    # Compute perturbation magnitudes
    perturbation_mags = [
        np.linalg.norm(h.iterations[-1].perturbation) if h.iterations else 0.0
        for h in histories
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Converged", f"{converged}/{len(histories)}")
    col2.metric("Avg Iterations", f"{avg_iterations:.1f}")
    col3.metric("Mean Fidelity", f"{final_snapshot.mean_fidelity:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Min Perturbation", f"{min(perturbation_mags):.4f}", help="Minimum perturbation magnitude (grid cells)")
    col5.metric("Avg Perturbation", f"{np.mean(perturbation_mags):.4f}", help="Average perturbation magnitude (grid cells)")
    col6.metric("Max Perturbation", f"{max(perturbation_mags):.4f}", help="Maximum perturbation magnitude (grid cells)")
    
    # NEW: Before/After Fairness Visualization (with trajectory selection built-in)
    st.divider()
    _display_before_after_fairness_viz(histories)
    
    # Iteration details section (now uses the same trajectory selected in the viz)
    st.divider()
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
    
    Note: This section now syncs with the trajectory selector in the 
    Before/After visualization above.
    """
    st.subheader("📈 Iteration Details")
    
    # Sync with visualization trajectory selector if available
    max_idx = len(histories) - 1
    
    # Use viz_selected_traj_idx if set, otherwise use selected_traj_idx
    viz_idx = st.session_state.get('viz_selected_traj_idx')
    current_idx = viz_idx if viz_idx is not None else st.session_state.get('selected_traj_idx', 0)
    
    if current_idx > max_idx:
        current_idx = 0
    
    # Trajectory selector with info display
    col_select, col_info = st.columns([1, 2])
    
    with col_select:
        selected_traj = st.number_input(
            "Select Trajectory",
            min_value=0,
            max_value=max_idx,
            value=current_idx,
            step=1,
            key="traj_selector_details",
            help="Use +/- to cycle through modified trajectories (synced with visualization above)",
        )
        # Sync back to viz selector
        st.session_state.viz_selected_traj_idx = selected_traj
        st.session_state.selected_traj_idx = selected_traj
    
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
        styled_df = df.style.format({
            'Objective L': '{:.8f}',
            'F_spatial': '{:.8f}',
            'F_causal': '{:.8f}',
            'F_fidelity': '{:.8f}',
        })
        st.dataframe(styled_df, use_container_width=True)
        
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


# ============================================================================
# Before/After Fairness Visualization
# ============================================================================

def _transform_coords_for_viz(x, y):
    """
    Transform coordinates from data grid to display grid.
    Original data: (x, y) in 48x90 grid (x: 0-47, y: 0-89)
    Display: 90 wide × 48 tall (swap x and y for horizontal layout)
    """
    return y, x


def _plot_trajectory_matplotlib(
    ax,
    trajectory_states: list,
    color: str = '#636EFA',
    label: str = '',
    show_start_end: bool = True,
    line_markersize: int = 3,
    linewidth: float = 1.0,
    alpha: float = 0.9,
):
    """
    Plot a single trajectory on a matplotlib Axes using the clean 'o-' style.

    Matches the ST-Siamese-Attack reference visualization:
    - Sequential line with small circle markers
    - Start (dropoff, states[0]) = black 'x'
    - End (pickup, states[-1]) = black '*'

    Coordinate transform is inlined: display_x = y_grid, display_y = x_grid.
    """
    display_x = [s.y_grid for s in trajectory_states]
    display_y = [s.x_grid for s in trajectory_states]

    ax.plot(
        display_x, display_y, 'o-',
        color=color,
        markersize=line_markersize,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=2,
    )

    if show_start_end and len(display_x) >= 1:
        ax.scatter(
            [display_x[0]], [display_y[0]],
            c='k', marker='x', s=80, zorder=3, linewidths=1.5,
        )
        ax.scatter(
            [display_x[-1]], [display_y[-1]],
            c='k', marker='*', s=120, zorder=3,
        )


def _get_trajectory_color_palette(n_colors: int) -> list:
    """Return n distinct colors from matplotlib's tab10 colormap."""
    cmap = plt.cm.get_cmap('tab10')
    return [cmap(i % 10) for i in range(n_colors)]


def _set_trajectory_axes_bounds(
    ax,
    all_display_x: list,
    all_display_y: list,
    padding: float = 2.0,
    grid_limits: tuple = (90, 48),
):
    """
    Auto-zoom axes to trajectory bounding box with padding.
    Sets equal aspect ratio for square grid cells.
    """
    if not all_display_x or not all_display_y:
        ax.set_xlim(-0.5, grid_limits[0] - 0.5)
        ax.set_ylim(-0.5, grid_limits[1] - 0.5)
    else:
        x_min = max(-0.5, min(all_display_x) - padding)
        x_max = min(grid_limits[0] - 0.5, max(all_display_x) + padding)
        y_min = max(-0.5, min(all_display_y) - padding)
        y_max = min(grid_limits[1] - 0.5, max(all_display_y) + padding)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')


def _compute_cell_fairness_scores(
    pickup_counts: np.ndarray,
    dropoff_counts: np.ndarray,
    supply_counts: np.ndarray,
    g_function,
    fairness_type: str = 'spatial',
    causal_formulation: str = 'baseline',
    hat_matrices=None,
    g0_power_basis_func=None,
    active_cell_indices=None,
) -> np.ndarray:
    """
    Compute cell-level fairness scores.

    Args:
        pickup_counts: [48, 90] pickup counts
        dropoff_counts: [48, 90] dropoff counts
        supply_counts: [48, 90] active taxis / supply
        g_function: Expected service ratio function
        fairness_type: 'spatial' for LIS, 'causal' for DCD
        causal_formulation: 'baseline' or 'option_b'
        hat_matrices: Dict with 'H_demo' key for Option B
        g0_power_basis_func: Power basis g₀(D) function for Option B
        active_cell_indices: Indices of cells with valid demographic data

    Returns:
        [48, 90] array of fairness scores (lower is better/more fair)
    """
    if fairness_type == 'spatial':
        # LIS = |c_i - μ| / μ
        return compute_cell_lis_scores(pickup_counts)
    else:
        # DCD: baseline |Y_i - g(D_i)| or Option B |R̂_i| = |H_demo @ R|_i
        return compute_cell_dcd_scores(
            pickup_counts, supply_counts, g_function,
            causal_formulation=causal_formulation,
            hat_matrices=hat_matrices,
            g0_power_basis_func=g0_power_basis_func,
            active_cell_indices=active_cell_indices,
        )


def _apply_trajectory_modification_to_grid(
    base_pickup_grid: np.ndarray,
    histories,
    selected_indices: List[int] = None,
) -> np.ndarray:
    """
    Apply trajectory modifications to create a modified pickup grid.
    
    Args:
        base_pickup_grid: Original [48, 90] pickup counts
        histories: List of ModificationHistory objects
        selected_indices: If provided, only apply modifications for these history indices
        
    Returns:
        Modified [48, 90] pickup counts
    """
    modified_grid = base_pickup_grid.copy().astype(float)
    
    indices_to_use = selected_indices if selected_indices is not None else range(len(histories))
    
    for idx in indices_to_use:
        if idx >= len(histories):
            continue
            
        history = histories[idx]
        if not history.iterations:
            continue
            
        # Get original and modified pickup locations (last state)
        orig_pickup = history.original.states[-1]
        mod_pickup = history.modified.states[-1]
        
        orig_x, orig_y = int(orig_pickup.x_grid), int(orig_pickup.y_grid)
        mod_x, mod_y = int(mod_pickup.x_grid), int(mod_pickup.y_grid)
        
        # Bounds check
        if 0 <= orig_x < 48 and 0 <= orig_y < 90:
            modified_grid[orig_x, orig_y] = max(0, modified_grid[orig_x, orig_y] - 1)
        if 0 <= mod_x < 48 and 0 <= mod_y < 90:
            modified_grid[mod_x, mod_y] += 1
    
    return modified_grid


@st.fragment
def _display_before_after_fairness_viz(histories):
    """
    Display side-by-side before/after visualization with fairness-colored grids.
    
    Uses @st.fragment to prevent full reruns when user interacts with controls.
    """
    st.subheader("🔄 Before/After Modification Comparison")
    
    if not histories:
        st.info("No modification results to visualize.")
        return
    
    if not MATPLOTLIB_AVAILABLE:
        st.warning("Matplotlib not available for visualization.")
        return

    # Get required data from session state
    pickup_grid = st.session_state.get('pickup_grid')
    dropoff_grid = st.session_state.get('dropoff_grid')
    active_taxis_grid = st.session_state.get('active_taxis_grid')
    g_function = st.session_state.get('g_function')
    trajectories = st.session_state.get('trajectories', [])
    
    if pickup_grid is None or active_taxis_grid is None:
        st.warning("Required grid data not loaded.")
        return
    
    # Control panel
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
    
    with ctrl_col1:
        fairness_type = st.radio(
            "Fairness Metric",
            ["Spatial (LIS)", "Causal (DCD)"],
            horizontal=True,
            help="Spatial: Local Inequality Score |cᵢ − μ| / μ\nCausal (Option B): Demographic Residual Deviation |R̂ᵢ| where R = Y − g₀(D) and R̂ = H_demo·R"
        )
        fairness_key = 'spatial' if 'Spatial' in fairness_type else 'causal'
    
    with ctrl_col2:
        view_mode = st.radio(
            "Trajectory View",
            ["Individual", "All Trajectories"],
            horizontal=True,
            help="Individual: Cycle through trajectories one by one\nAll: Show all modified trajectories at once"
        )
    
    with ctrl_col3:
        if view_mode == "Individual":
            max_idx = len(histories) - 1
            current_idx = st.session_state.get('viz_selected_traj_idx', 0)
            if current_idx > max_idx:
                current_idx = 0
            
            selected_viz_traj = st.number_input(
                "Select Trajectory",
                min_value=0,
                max_value=max_idx,
                value=current_idx,
                step=1,
                key="viz_traj_selector",
                help="Use +/- to cycle through modified trajectories",
            )
            st.session_state.viz_selected_traj_idx = selected_viz_traj
            traj_indices_to_show = [selected_viz_traj]
            
            # Show trajectory info
            history = histories[selected_viz_traj]
            driver_id = getattr(history.original, 'driver_id', 'N/A')
            st.caption(f"Driver: {driver_id} | Traj {selected_viz_traj + 1}/{len(histories)}")
        else:
            traj_indices_to_show = list(range(len(histories)))
            st.caption(f"Showing all {len(histories)} trajectories")
    
    # Compute fairness scores for BEFORE state
    before_fairness = _compute_cell_fairness_scores(
        pickup_grid, dropoff_grid, active_taxis_grid, g_function, fairness_key,
        causal_formulation=st.session_state.get('causal_formulation', 'baseline'),
        hat_matrices=st.session_state.get('hat_matrices'),
        g0_power_basis_func=st.session_state.get('g0_power_basis_func'),
        active_cell_indices=st.session_state.get('active_cell_indices'),
    )
    
    # Compute modified pickup grid for selected trajectories
    modified_pickup_grid = _apply_trajectory_modification_to_grid(
        pickup_grid, histories, traj_indices_to_show
    )
    
    # Compute fairness scores for AFTER state
    after_fairness = _compute_cell_fairness_scores(
        modified_pickup_grid, dropoff_grid, active_taxis_grid, g_function, fairness_key,
        causal_formulation=st.session_state.get('causal_formulation', 'baseline'),
        hat_matrices=st.session_state.get('hat_matrices'),
        g0_power_basis_func=st.session_state.get('g0_power_basis_func'),
        active_cell_indices=st.session_state.get('active_cell_indices'),
    )
    
    # Determine common color scale range for both grids
    vmin = min(before_fairness.min(), after_fairness.min())
    vmax = max(before_fairness.max(), after_fairness.max())
    
    # Create side-by-side plots
    before_col, after_col = st.columns(2)

    # Color palette for trajectories
    colors = _get_trajectory_color_palette(len(traj_indices_to_show))

    # === BEFORE VISUALIZATION ===
    with before_col:
        st.markdown("**BEFORE Modification**")

        fig_before, ax_before = plt.subplots(figsize=(8, 5))

        # Fairness heatmap: (48, 90) array with origin='lower' puts row 0 (south) at bottom
        # extent maps cols 0-89 to horizontal x and rows 0-47 to vertical y,
        # consistent with trajectory transform (display_x = y_grid, display_y = x_grid)
        im_before = ax_before.imshow(
            before_fairness,
            cmap='RdYlGn_r',
            vmin=vmin, vmax=vmax,
            origin='lower',
            aspect='equal',
            extent=[-0.5, 89.5, -0.5, 47.5],
            zorder=1,
            interpolation='nearest',
        )

        # Draw original trajectories
        all_display_x = []
        all_display_y = []

        for i, hist_idx in enumerate(traj_indices_to_show):
            history = histories[hist_idx]
            traj = history.original
            color = colors[i]
            driver_id = getattr(traj, 'driver_id', 'N/A')
            label = f"Traj {hist_idx} (D:{driver_id})"

            _plot_trajectory_matplotlib(ax_before, traj.states, color=color, label=label)

            all_display_x.extend(s.y_grid for s in traj.states)
            all_display_y.extend(s.x_grid for s in traj.states)

        _set_trajectory_axes_bounds(ax_before, all_display_x, all_display_y, padding=3.0)

        ax_before.set_title(
            f"Original | {fairness_type}\nx = Dropoff   * = Pickup",
            fontsize=9,
        )

        cbar_before = fig_before.colorbar(
            im_before, ax=ax_before, orientation='horizontal',
            fraction=0.06, pad=0.12, aspect=30,
        )
        cbar_before.set_label(fairness_key.upper(), fontsize=8)
        cbar_before.ax.tick_params(labelsize=7)

        ax_before.legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.18),
            ncol=min(3, len(traj_indices_to_show)),
            fontsize=7, framealpha=0.9,
        )

        fig_before.tight_layout()
        st.pyplot(fig_before)
        plt.close(fig_before)

        # Show stats
        active_cells = (before_fairness > 0).sum()
        avg_fairness = before_fairness[before_fairness > 0].mean() if active_cells > 0 else 0
        st.caption(f"Avg {fairness_key.upper()}: {avg_fairness:.4f} | Active cells: {active_cells}")

    # === AFTER VISUALIZATION ===
    with after_col:
        st.markdown("**AFTER Modification**")

        fig_after, ax_after = plt.subplots(figsize=(8, 5))

        im_after = ax_after.imshow(
            after_fairness,
            cmap='RdYlGn_r',
            vmin=vmin, vmax=vmax,
            origin='lower',
            aspect='equal',
            extent=[-0.5, 89.5, -0.5, 47.5],
            zorder=1,
            interpolation='nearest',
        )

        # Draw modified trajectories
        all_display_x = []
        all_display_y = []

        for i, hist_idx in enumerate(traj_indices_to_show):
            history = histories[hist_idx]
            traj = history.modified
            color = colors[i]
            driver_id = getattr(history.original, 'driver_id', 'N/A')
            label = f"Traj {hist_idx} (D:{driver_id})"

            _plot_trajectory_matplotlib(ax_after, traj.states, color=color, label=label)

            all_display_x.extend(s.y_grid for s in traj.states)
            all_display_y.extend(s.x_grid for s in traj.states)

            # Draw red arrow from original pickup to modified pickup
            orig_pickup = history.original.states[-1]
            mod_pickup = history.modified.states[-1]
            orig_dx, orig_dy = orig_pickup.y_grid, orig_pickup.x_grid
            mod_dx, mod_dy = mod_pickup.y_grid, mod_pickup.x_grid

            if abs(orig_dx - mod_dx) > 0.1 or abs(orig_dy - mod_dy) > 0.1:
                ax_after.annotate(
                    '',
                    xy=(mod_dx, mod_dy),
                    xytext=(orig_dx, orig_dy),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color='red',
                        lw=2.0,
                        alpha=0.8,
                        mutation_scale=15,
                    ),
                    zorder=4,
                )

        _set_trajectory_axes_bounds(ax_after, all_display_x, all_display_y, padding=3.0)

        ax_after.set_title(
            f"Modified | {fairness_type}\nx = Dropoff   * = Pickup",
            fontsize=9,
        )

        cbar_after = fig_after.colorbar(
            im_after, ax=ax_after, orientation='horizontal',
            fraction=0.06, pad=0.12, aspect=30,
        )
        cbar_after.set_label(fairness_key.upper(), fontsize=8)
        cbar_after.ax.tick_params(labelsize=7)

        ax_after.legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.18),
            ncol=min(3, len(traj_indices_to_show)),
            fontsize=7, framealpha=0.9,
        )

        fig_after.tight_layout()
        st.pyplot(fig_after)
        plt.close(fig_after)

        # Show stats
        active_cells_after = (after_fairness > 0).sum()
        avg_fairness_after = after_fairness[after_fairness > 0].mean() if active_cells_after > 0 else 0
        delta = avg_fairness_after - avg_fairness
        delta_str = f"({'↓' if delta < 0 else '↑'}{abs(delta):.4f})" if delta != 0 else ""
        st.caption(f"Avg {fairness_key.upper()}: {avg_fairness_after:.4f} {delta_str} | Active cells: {active_cells_after}")
    
    # Summary of modification impact
    if traj_indices_to_show:
        with st.expander("📊 Modification Details", expanded=False):
            for hist_idx in traj_indices_to_show[:10]:  # Limit to first 10
                history = histories[hist_idx]
                orig_pickup = history.original.states[-1]
                mod_pickup = history.modified.states[-1]
                
                orig_cell = (int(orig_pickup.x_grid), int(orig_pickup.y_grid))
                mod_cell = (int(mod_pickup.x_grid), int(mod_pickup.y_grid))
                
                # Get fairness scores at original and modified cells
                orig_fairness_score = before_fairness[orig_cell[0], orig_cell[1]]
                mod_fairness_score = after_fairness[mod_cell[0], mod_cell[1]]
                
                driver_id = getattr(history.original, 'driver_id', 'N/A')
                label = "F_spatial" if fairness_key == 'spatial' else "F_causal"
                st.markdown(
                    f"**Traj {hist_idx}** (Driver {driver_id}): "
                    f"({orig_cell[0]}, {orig_cell[1]}) → ({mod_cell[0]}, {mod_cell[1]}) | "
                    f"{label}: {orig_fairness_score:.4f} → {mod_fairness_score:.4f}"
                )


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
