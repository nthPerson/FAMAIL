"""
F_causal Formulation Dashboard — Demographic Residual Independence

Interactive Streamlit dashboard for exploring and understanding the
Demographic Residual Independence formulation of the causal fairness
term in the FAMAIL objective function.

Run from project root:
    streamlit run objective_function/causal_fairness/f_causal_term_reformulation/chosen_formulation/option_b_dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Callable
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as tnn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
DASHBOARD_DIR = Path(__file__).resolve().parent
REFORMULATION_DIR = DASHBOARD_DIR.parent
CAUSAL_DIR = REFORMULATION_DIR.parent
OBJ_FUNC_DIR = CAUSAL_DIR.parent
PROJECT_ROOT = OBJ_FUNC_DIR.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from objective_function.causal_fairness.utils import (
    load_pickup_dropoff_counts,
    load_active_taxis_data,
    extract_demand_from_counts,
    extract_demand_ratio_arrays,
    aggregate_to_period,
    compute_service_ratios,
    estimate_g_power_basis,
    estimate_g_isotonic,
    build_power_basis_features,
    build_hat_matrix,
    build_centering_matrix,
    precompute_hat_matrices,
    compute_fcausal_option_b_numpy,
    enrich_demographic_features,
)

D3_COLORS = px.colors.qualitative.D3


# ---------------------------------------------------------------------------
# Neural network g₀(D) helpers
# ---------------------------------------------------------------------------

def _parse_hidden_layers(s: str) -> Optional[List[int]]:
    """Parse comma-separated hidden layer sizes, e.g. '64, 32' → [64, 32]."""
    try:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            return None
        layers = [int(p) for p in parts]
        if any(l <= 0 for l in layers):
            return None
        return layers
    except (ValueError, AttributeError):
        return None


def _train_g0_nn(
    demands: np.ndarray,
    Y: np.ndarray,
    hidden_layers: List[int],
    epochs: int = 1000,
    lr: float = 0.005,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Train a feedforward neural network g₀(D) and return predictions + diagnostics.

    Args:
        demands: 1-D array of demand values (input).
        Y: 1-D array of service ratios (target).
        hidden_layers: list of hidden layer widths, e.g. [64, 32].
        epochs: number of full-batch training iterations.
        lr: Adam learning rate.
        seed: random seed for reproducibility.

    Returns:
        predictions: numpy array of g₀(D) predictions on the input demands.
        diagnostics: dict with architecture info, loss history, etc.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Normalize inputs for stable training
    D_mean, D_std = float(demands.mean()), float(demands.std()) + 1e-8
    Y_mean, Y_std = float(Y.mean()), float(Y.std()) + 1e-8

    D_norm = (demands - D_mean) / D_std
    Y_norm = (Y - Y_mean) / Y_std

    D_t = torch.tensor(D_norm, dtype=torch.float32).unsqueeze(1)
    Y_t = torch.tensor(Y_norm, dtype=torch.float32)

    # Build feedforward network: Input(1) → [Linear → ReLU] × L → Linear(1)
    layers_list: List[tnn.Module] = []
    in_dim = 1
    for h in hidden_layers:
        layers_list.append(tnn.Linear(in_dim, h))
        layers_list.append(tnn.ReLU())
        in_dim = h
    layers_list.append(tnn.Linear(in_dim, 1))
    model = tnn.Sequential(*layers_list)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = tnn.MSELoss()

    loss_history: List[float] = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(D_t).squeeze(-1)
        loss = loss_fn(pred, Y_t)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    model.eval()

    # Final predictions on training data (de-normalized)
    with torch.no_grad():
        pred_norm = model(D_t).squeeze(-1).numpy()
    predictions = pred_norm * Y_std + Y_mean

    n_params = sum(p.numel() for p in model.parameters())

    # Build architecture description string
    arch_parts = ["Input(1)"]
    for h in hidden_layers:
        arch_parts.append(f"Linear({h})")
        arch_parts.append("ReLU")
    arch_parts.append("Linear(1)")
    arch_str = " → ".join(arch_parts)

    diagnostics = {
        "method": "neural_network",
        "hidden_layers": list(hidden_layers),
        "architecture": arch_str,
        "epochs": epochs,
        "lr": lr,
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "n_params": n_params,
    }

    return predictions, diagnostics


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="F_causal Explorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(filepath: str):
    with open(filepath, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_and_prepare_data(
    pickup_path: str,
    active_path: str,
    demo_path: str,
    district_path: str,
    period_type: str = "hourly",
    min_demand: int = 1,
    g0_model_type: str = "Power Basis",
    nn_hidden_layers: str = "",
):
    """Load all data sources and compute service ratios, residuals, hat matrices.

    Two levels of data are produced:
      - **Period-level** (one observation per cell-period pair): used to fit g₀(D)
        and report R², matching the approach in CausalFairnessTerm.compute().
      - **Cell-level** (one observation per cell): aggregated as ratio-of-totals
        (Y_c = ΣS / ΣD), used for hat matrix, F_causal, and demographic analysis.
    """
    # --- Load raw counts ---
    counts = load_pickup_dropoff_counts(pickup_path)
    active = load_active_taxis_data(active_path)

    # --- Extract demand (pickup counts) from [pickup, dropoff] list values ---
    demand_raw = extract_demand_from_counts(counts)

    # --- Aggregate to period ---
    demand_dict = aggregate_to_period(demand_raw, period_type)
    supply_dict = aggregate_to_period(active, period_type)

    # --- Compute service ratios at period level ---
    ratios = compute_service_ratios(demand_dict, supply_dict, min_demand=min_demand)

    # =====================================================================
    # PERIOD-LEVEL: fit g₀(D) on all (cell, period) observations
    # This matches the approach in CausalFairnessTerm.compute() and gives
    # the expected R² ≈ 0.445 for isotonic / ≈ 0.40 for power basis.
    # =====================================================================
    demands_period, ratios_period, keys_period = extract_demand_ratio_arrays(
        demand_dict, ratios
    )

    # Always fit power basis at period level (for reference R²)
    g0_pb_func, g0_pb_diag = estimate_g_power_basis(demands_period, ratios_period)
    g0_pb_pred_period = g0_pb_func(demands_period)

    # Always fit isotonic at period level (for reference)
    g0_iso_func, g0_iso_diag = estimate_g_isotonic(demands_period, ratios_period)
    g0_iso_pred_period = g0_iso_func(demands_period)

    # Selected model: either power basis or neural network
    use_nn = g0_model_type == "Neural Network"
    nn_diag_period = None
    nn_diag_cell = None

    if use_nn:
        hidden = _parse_hidden_layers(nn_hidden_layers)
        if hidden is None:
            hidden = [64, 32]  # fallback default
        g0_pred_period, nn_diag_period = _train_g0_nn(
            demands_period, ratios_period, hidden
        )
        g0_diag = nn_diag_period
    else:
        g0_pred_period = g0_pb_pred_period
        g0_diag = g0_pb_diag

    # Compute R² at period level for all models
    ss_tot_period = np.sum((ratios_period - ratios_period.mean()) ** 2)
    if ss_tot_period > 1e-10:
        g0_r2 = 1.0 - np.sum((ratios_period - g0_pred_period) ** 2) / ss_tot_period
        g0_iso_r2 = 1.0 - np.sum((ratios_period - g0_iso_pred_period) ** 2) / ss_tot_period
        g0_pb_r2 = 1.0 - np.sum((ratios_period - g0_pb_pred_period) ** 2) / ss_tot_period
    else:
        g0_r2 = 0.0
        g0_iso_r2 = 0.0
        g0_pb_r2 = 0.0

    # =====================================================================
    # CELL-LEVEL: aggregate properly using ratio-of-totals (not avg-of-ratios)
    #   Y_c = total_supply_c / total_demand_c   (avoids Jensen's inequality)
    #   D_c = mean period demand                 (same scale as g₀ training data)
    # =====================================================================
    cell_total_demand = {}
    cell_total_supply = {}
    cell_n_periods = {}

    for key, ratio in ratios.items():
        x, y = key[0], key[1]
        cell_key = (x, y)
        if cell_key not in cell_total_demand:
            cell_total_demand[cell_key] = 0
            cell_total_supply[cell_key] = 0
            cell_n_periods[cell_key] = 0
        cell_total_demand[cell_key] += demand_dict.get(key, 0)
        cell_total_supply[cell_key] += supply_dict.get(key, 0)
        cell_n_periods[cell_key] += 1

    # --- Load demographics ---
    demo_data = load_data(demo_path)
    district_data = load_data(district_path)

    demographics_grid = demo_data["demographics_grid"]  # (48, 90, n_raw)
    raw_feature_names = list(demo_data["feature_names"])

    demographics_grid, feature_names = enrich_demographic_features(
        demographics_grid, raw_feature_names
    )

    district_ids = district_data.get("district_id_grid", district_data.get("district_ids", None))
    valid_mask = district_data.get("valid_mask", None)
    district_names_list = district_data.get("district_names", None)  # ordered by district_id

    # --- Build cell-level arrays for active cells ---
    default_features = ["AvgHousingPricePerSqM", "GDPperCapita", "CompPerCapita"]
    feat_indices = [feature_names.index(f) for f in default_features if f in feature_names]
    used_features = [feature_names[i] for i in feat_indices]

    cell_list = []
    demands_list = []     # D_c = mean period demand (same scale as g₀)
    ratios_list = []      # Y_c = total_S / total_D  (ratio of totals)
    supply_list = []
    demo_list = []
    district_list = []

    for (x, y), total_d in cell_total_demand.items():
        if x < 0 or x >= 48 or y < 0 or y >= 90:
            continue
        if valid_mask is not None and not valid_mask[x, y]:
            continue
        if total_d < min_demand:
            continue

        demo_vals = demographics_grid[x, y, feat_indices]
        if np.any(np.isnan(demo_vals)) or np.any(demo_vals == 0):
            continue

        total_s = cell_total_supply[(x, y)]
        n_per = cell_n_periods[(x, y)]

        cell_list.append((x, y))
        demands_list.append(total_d / n_per)       # mean period demand
        ratios_list.append(total_s / total_d)       # ratio of totals
        supply_list.append(total_s / n_per)         # mean period supply
        demo_list.append(demo_vals)
        if district_ids is not None and district_names_list is not None:
            did = int(district_ids[x, y])
            district_list.append(district_names_list[did] if 0 <= did < len(district_names_list) else "Unknown")
        elif district_ids is not None:
            district_list.append(str(int(district_ids[x, y])))
        else:
            district_list.append("Unknown")

    demands = np.array(demands_list, dtype=np.float64)
    Y = np.array(ratios_list, dtype=np.float64)
    supply_arr = np.array(supply_list, dtype=np.float64)
    demo_features = np.array(demo_list, dtype=np.float64)
    districts = np.array(district_list)

    # =====================================================================
    # CELL-LEVEL g₀: fit g₀ on cell-level (D, Y) for centered residuals
    #
    # The period-level g₀ cannot be directly applied to cell-level data
    # because cell Y = ΣS/ΣD (ratio-of-totals) follows a different
    # distribution than period-level Y = S/D. Applying the period g₀ to
    # cell-level mean demands produces systematically biased residuals
    # (mean ≈ -5 instead of 0). Fitting g₀ at the cell level ensures
    # the OLS residuals are centered, which is essential for the hat
    # matrix decomposition and F_causal computation.
    # =====================================================================

    # Always: power basis at cell level (for reference)
    g0_pb_cell_func, g0_pb_cell_diag = estimate_g_power_basis(demands, Y)
    g0_pb_pred_cell = g0_pb_cell_func(demands)

    # Always: isotonic at cell level (for reference)
    g0_iso_cell_func, g0_iso_cell_diag = estimate_g_isotonic(demands, Y)
    g0_iso_pred = g0_iso_cell_func(demands)
    R_iso = Y - g0_iso_pred

    # Selected model at cell level
    if use_nn:
        hidden = _parse_hidden_layers(nn_hidden_layers)
        if hidden is None:
            hidden = [64, 32]
        g0_pred, nn_diag_cell = _train_g0_nn(demands, Y, hidden)
        g0_cell_diag = nn_diag_cell
    else:
        g0_pred = g0_pb_pred_cell
        g0_cell_diag = g0_pb_cell_diag

    R = Y - g0_pred  # residuals (centered by OLS construction for power basis)

    # Cell-level R² (for reference — period-level R² is the primary metric)
    ss_tot_cell = np.sum((Y - Y.mean()) ** 2)
    if ss_tot_cell > 1e-10:
        g0_cell_r2 = 1.0 - np.sum((Y - g0_pred) ** 2) / ss_tot_cell
        g0_iso_cell_r2 = 1.0 - np.sum((Y - g0_iso_pred) ** 2) / ss_tot_cell
        g0_pb_cell_r2 = 1.0 - np.sum((Y - g0_pb_pred_cell) ** 2) / ss_tot_cell
    else:
        g0_cell_r2 = 0.0
        g0_iso_cell_r2 = 0.0
        g0_pb_cell_r2 = 0.0

    # --- Pre-compute hat matrices (cell-level) ---
    hat_result = precompute_hat_matrices(demands, demo_features, feature_names=used_features)

    # --- Compute F_causal ---
    fcausal_result = compute_fcausal_option_b_numpy(
        R, hat_result["I_minus_H_demo"], hat_result["M"]
    )

    # --- Compute regression coefficients: β = (X'X)⁻¹X'R ---
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(demo_features)
    X_design = np.column_stack([np.ones(len(R)), X_scaled])
    beta = np.linalg.lstsq(X_design, R, rcond=None)[0]
    R_hat = X_design @ beta  # demographic-predicted residuals

    return {
        # Cell-level data (for hat matrix, F_causal, spatial maps)
        "cells": cell_list,
        "demands": demands,
        "Y": Y,
        "supply": supply_arr,
        "R": R,
        "R_iso": R_iso,
        "g0_pred": g0_pred,
        "g0_iso_pred": g0_iso_pred,
        "demo_features": demo_features,
        "feature_names": used_features,
        "districts": districts,
        "hat_result": hat_result,
        "fcausal_result": fcausal_result,
        "beta": beta,
        "R_hat": R_hat,
        "X_design": X_design,
        "n_cells": len(cell_list),
        # Period-level data (for g₀ model quality and scatter plots)
        "demands_period": demands_period,
        "ratios_period": ratios_period,
        "g0_pred_period": g0_pred_period,
        "g0_iso_pred_period": g0_iso_pred_period,
        "n_period_obs": len(demands_period),
        # g₀ diagnostics
        "g0_diag": g0_diag,            # period-level fit diagnostics (active model)
        "g0_iso_diag": g0_iso_diag,
        "g0_cell_diag": g0_cell_diag,  # cell-level fit diagnostics (active model)
        "g0_pb_diag": g0_pb_diag,      # power basis diagnostics (always available)
        # R² at both levels
        "g0_r2": g0_r2,                # period-level R² of active model
        "g0_iso_r2": g0_iso_r2,        # period-level isotonic (reference)
        "g0_pb_r2": g0_pb_r2,          # period-level power basis (reference)
        "g0_cell_r2": g0_cell_r2,      # cell-level R² of active model
        "g0_iso_cell_r2": g0_iso_cell_r2,
        "g0_pb_cell_r2": g0_pb_cell_r2,
        # Power basis predictions (always available for scatter plot comparison)
        "g0_pb_pred_period": g0_pb_pred_period,
        # Model selection metadata
        "g0_model_type": g0_model_type,
        "nn_diag_period": nn_diag_period,
        "nn_diag_cell": nn_diag_cell,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🎯 F_causal Explorer")
st.sidebar.caption("Demographic Residual Independence")

st.sidebar.divider()
st.sidebar.subheader("Data Sources")

pickup_path = st.sidebar.text_input(
    "Pickup/Dropoff Counts",
    str(PROJECT_ROOT / "source_data" / "pickup_dropoff_counts.pkl"),
)
active_path = st.sidebar.text_input(
    "Active Taxis",
    str(PROJECT_ROOT / "source_data" / "active_taxis_5x5_hourly.pkl"),
)
demo_path = st.sidebar.text_input(
    "Cell Demographics",
    str(PROJECT_ROOT / "source_data" / "cell_demographics.pkl"),
)
district_path = st.sidebar.text_input(
    "District Mapping",
    str(PROJECT_ROOT / "source_data" / "grid_to_district_mapping.pkl"),
)

st.sidebar.divider()
st.sidebar.subheader("Parameters")
period_type = st.sidebar.selectbox("Period Type", ["hourly", "daily", "all"], index=0)
min_demand = st.sidebar.slider("Min Demand per Cell", 1, 20, 1)

st.sidebar.divider()
st.sidebar.subheader("g₀(D) Model")
g0_model_type = st.sidebar.selectbox(
    "Model Type",
    ["Power Basis", "Neural Network"],
    index=0,
    help="Power Basis: OLS with hand-crafted features (FWL-compatible). "
         "Neural Network: learned nonlinear mapping (more flexible).",
)
nn_hidden_layers = ""
if g0_model_type == "Neural Network":
    nn_hidden_layers = st.sidebar.text_input(
        "Hidden Layers (comma-separated neuron counts)",
        "32, 16",
        help="e.g. '32, 16' → two hidden layers with 32 and 316 neurons. "
             "Input (demand) and output (predicted Y) layers are added automatically.",
    )
    parsed = _parse_hidden_layers(nn_hidden_layers)
    if parsed is None:
        st.sidebar.error("Invalid architecture. Use comma-separated positive integers, e.g. '64, 32'.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data_loaded = False
data = None

for p in [pickup_path, active_path, demo_path, district_path]:
    if not Path(p).exists():
        st.error(f"File not found: `{p}`")
        st.stop()

try:
    data = load_and_prepare_data(
        pickup_path, active_path, demo_path, district_path,
        period_type=period_type, min_demand=min_demand,
        g0_model_type=g0_model_type, nn_hidden_layers=nn_hidden_layers,
    )
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_formulation, tab_matrix, tab_g0, tab_causal, tab_real, tab_validate = st.tabs([
    "🏠 Overview",
    "📐 The Formulation",
    "🔢 Matrix Algebra",
    "📈 g₀(D) Demand Model",
    "⚖️ Causal Foundations",
    "🗺️ Real Data Analysis",
    "✅ Validation",
])


# =========================================================================
# TAB 1: Overview
# =========================================================================
with tab_overview:
    st.header("F_causal: Demographic Residual Independence")
    st.markdown("*The reformulated causal fairness term for the FAMAIL objective function*")

    # --- Key metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("F_causal", f"{data['fcausal_result']['f_causal']:.4f}")
    with col2:
        st.metric("R²_demo", f"{data['fcausal_result']['r2_demo']:.4f}")
    with col3:
        st.metric("Active Cells", data["n_cells"])
    with col4:
        _g0_label = "g₀(D) R² [NN]" if data["g0_model_type"] == "Neural Network" else "g₀(D) R²"
        st.metric(_g0_label, f"{data['g0_r2']:.4f}")

    st.divider()

    # --- The one-sentence pitch ---
    st.info(
        "**Core question**: After accounting for demand, do demographics still predict service? "
        "The reformulated F_causal measures demographic independence "
        "of residuals — and the hat matrix makes this analytically exact and fully differentiable."
    )

    # --- Formula ---
    st.latex(
        r"F_{\text{causal}} = \frac{R'(I - H)R}{R'MR} = 1 - R^2_{\text{demo}}"
    )

    # --- Expanded comparison table (full width) ---
    st.subheader("What Changed: Baseline vs. Reformulated F_causal")

    st.markdown(r"""
| Aspect | Old (Baseline) | Reformulated |
|--------|----------------|--------------|
| **Formula** | $F = R^2 = 1 - \dfrac{\text{Var}(Y - g(D))}{\text{Var}(Y)}$ | $F = \dfrac{R'(I - H)R}{R'MR} = 1 - R^2_{\text{demo}}$ |
| **Core question** | *"How well does the demand model g(D) explain the observed service ratio Y?"* — measures how tightly supply tracks demand across the city. | *"After removing the demand effect, do demographic features still predict the remaining service variation?"* — measures whether demographics have a **direct** influence on service beyond what demand explains. |
| **What it measures** | **Demand alignment.** A high score means that the demand model g(D) accounts for most of the variance in Y = S/D. This conflates demand alignment with fairness: a city could score perfectly even if wealthy districts are systematically over-served, as long as g(D) captures that pattern. | **Demographic independence of residuals.** A high score means that after removing demand's effect via $R = Y - g_0(D)$, the remaining variation in service is **not** predictable from demographic features (housing price, GDP, compensation). Only the direct demographic path $X \to Y$ is penalized. |
| **Residuals** | $R = Y - g(D)$ where $g(D)$ may be fit using demand **and** demographics jointly (e.g., $g(D, x)$), making it unclear what the residuals actually isolate. | $R = Y - g_0(D)$ where $g_0$ is fit using **demand only** (power basis). Residuals cleanly represent the service variation that demand alone cannot explain — the part that *could* be due to demographics, infrastructure, or randomness. |
| **Demographic model** | None — demographics are embedded implicitly in $g(D, x)$ during model fitting. No explicit test of demographic influence. | **Explicit via hat matrix** $H = X(X'X)^{-1}X'$. This constant projection matrix regresses the residuals onto demographics analytically, yielding $\hat{R} = HR$ (the demographic-predicted residual) without ever fitting a separate model during optimization. |
| **Gradient quality** | Partially degenerate. Because $g(D)$ absorbs both demand and demographic effects, the gradient $\partial F / \partial Y$ cannot distinguish between correcting demand misalignment and correcting demographic unfairness. In practice, it may push trajectories in uninformative directions. | **Clean and directionally correct.** The gradient $\frac{\partial F}{\partial R_c} = \frac{2}{R'MR}\left[((I-H)R)_c - F \cdot (R_c - \bar{R})\right]$ is negative for over-served wealthy cells and positive for under-served poor cells. It targets demographic-driven disparity specifically. |
| **Causal basis** | Weak. Measures model fit quality ($R^2$ of $g(D)$ vs. $Y$), which is a statistical property of the demand model, not a statement about causal fairness. Does not test whether demographics *should* influence service. | **Strong.** Directly operationalizes **counterfactual fairness** (Kusner et al., 2017) at the aggregate level, computes the **Controlled Direct Effect** via regression coefficients $\hat{\beta}$, and isolates the **direct causal path** $X \to Y$ via **mediation analysis** (conditioning on the mediator $D$). Equivalent to the standard **partial $R^2$** via the Frisch-Waugh-Lovell theorem. |
| **Optimization target** | Push $R^2$ of g(D) toward 1 — make g(D) fit Y as tightly as possible. This rewards better demand modeling, not fairer outcomes. | Push $R^2_{\text{demo}}$ toward 0 — eliminate the direct demographic influence on service residuals. This directly targets the unfair causal pathway while preserving the acceptable demand-mediated pathway. |
    """)

    st.divider()

    # --- How it works ---
    st.subheader("How It Works")
    st.markdown("""
1. **Remove demand effect**: Fit g₀(D) to get residuals R = Y − g₀(D)
2. **Test demographic influence**: Use hat matrix H to regress R on demographics
3. **Compute fairness**: F = fraction of residual variance NOT explained by demographics
4. **Optimize**: Gradient pushes service from over-served wealthy cells to under-served poor cells
    """)

    st.divider()

    # --- Pipeline diagram ---
    st.subheader("Data Flow")
    st.code("""
    PRE-COMPUTED (once)                    DURING OPTIMIZATION (each step)
    ═══════════════════                    ════════════════════════════════

    g₀(D): Power basis fit                Trajectory → Soft Cell → S_c, D_c
    H: X(X'X)⁻¹X' (demographics)                          ↓
    M: I − 11'/N (centering)              Y_c = S_c / D_c    (differentiable)
                                                    ↓
                                          R_c = Y_c − g₀(D_c) (g₀ frozen)
                                                    ↓
                                    ┌───────────────┴──────────────┐
                                    │                              │
                              SS_res = R'(I−H)R           SS_tot = R'MR
                                    │                              │
                                    └───────────────┬──────────────┘
                                                    ↓
                                          F_causal = SS_res / SS_tot
                                                    ↓
                                          ∂F/∂R → ∂R/∂Y → ∂Y/∂positions
    """, language=None)


# =========================================================================
# TAB 2: The Formulation
# =========================================================================
with tab_formulation:
    st.header("The Formulation in Detail")

    st.latex(
        r"\boxed{F_{\text{causal}} = \frac{R'(I - H)R}{R'MR} = 1 - R^2_{\text{demo}}}"
    )

    st.markdown("Each component has a specific role. Expand the sections below for details.")

    # --- Component: R (residuals) ---
    with st.expander("**R = Y − g₀(D)** — Demand-Adjusted Residuals", expanded=True):
        st.markdown(r"""
**Definition**: $R_c = Y_c - g_0(D_c)$ where $Y_c = S_c / D_c$ is the service ratio and $g_0(D_c)$ is the expected service ratio given demand.

**Interpretation**: How much more or less service does cell $c$ receive than what demand alone would predict?
- $R_c > 0$: Over-served relative to demand
- $R_c < 0$: Under-served relative to demand
- $R_c \approx 0$: Service matches demand expectation

**During optimization**: R changes because Y changes (pickups move between cells). g₀(D) is frozen — it's a pre-computed lookup table.
        """)

        col1, col2 = st.columns(2)
        with col1:
            fig_r = px.histogram(
                x=data["R"], nbins=50, labels={"x": "Residual R", "y": "Count"},
                title="Distribution of Residuals R = Y − g₀(D)",
                color_discrete_sequence=[D3_COLORS[0]],
            )
            fig_r.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_r.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_r, use_container_width=True)

        with col2:
            fig_yr = px.scatter(
                x=data["demands"], y=data["R"],
                labels={"x": "Demand (D)", "y": "Residual (R)"},
                title="Residuals vs Demand (should show no trend)",
                color_discrete_sequence=[D3_COLORS[1]],
                opacity=0.5,
            )
            fig_yr.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_yr.update_layout(height=350)
            st.plotly_chart(fig_yr, use_container_width=True)

    # --- Component: H (hat matrix) ---
    with st.expander("**H = X(X'X)⁻¹X'** — The Hat Matrix (Demographic Projection)"):
        st.markdown(r"""
**Definition**: $H$ is the hat matrix that projects any vector onto the column space of $X$ (demographics).

**What it does**: Given residuals $R$, the product $HR = \hat{R}$ gives the OLS-best-fit of $R$ using only demographic features. This is what demographics *predict* the residual should be.

**Key properties**:
- **Symmetric**: $H = H'$
- **Idempotent**: $HH = H$ (projecting twice = projecting once)
- **Constant**: $H$ depends only on demographics $X$, which never change during optimization
- **Implicit re-fitting**: $HR$ always gives the current OLS fit — coefficients update automatically

**Design matrix** $X$: [intercept | standardized AvgHousingPrice | standardized GDPperCapita | standardized CompPerCapita]
        """)

        diag = data["hat_result"]["diagnostics"]
        st.markdown(f"""
| Property | Value |
|----------|-------|
| Matrix dimension | {diag['n_cells']} × {diag['n_cells']} |
| Demographic features | {diag['n_demo_features']} ({', '.join(diag.get('feature_names', []))}) |
| H rank | {diag.get('H_demo_rank', 'N/A')} |
        """)

    # --- Component: (I-H) ---
    with st.expander("**(I − H)** — The Residual-Maker Matrix"):
        st.markdown(r"""
**Definition**: $(I - H)$ projects any vector onto the space **orthogonal** to demographics.

**What it does**: $(I-H)R = e$ is the part of the residual that demographics *cannot explain*. This is the "fair" component of the service variation.

**Decomposition**: Every residual vector splits into two perpendicular parts:

$$R = \underbrace{HR}_{\text{demographic component}} + \underbrace{(I-H)R}_{\text{fair component}}$$

The fairness score $F_{\text{causal}}$ is the ratio of the fair component's variance to the total variance.
        """)

    # --- Component: M ---
    with st.expander("**M = I − 11'/N** — The Centering Matrix"):
        st.markdown(r"""
**Definition**: $M$ subtracts the mean from any vector: $Mv = v - \bar{v}$.

**In the denominator**: $R'MR = \sum_c (R_c - \bar{R})^2 = N \cdot \text{Var}(R)$. This is the total sum of squares ($SS_{\text{tot}}$) — the total centered variance of the residuals.

**Why centering matters**: Without centering, the intercept in the demographic regression would absorb the mean, and the decomposition wouldn't be exact.
        """)

    # --- Score and gradient ---
    with st.expander("**Score Interpretation and Gradient**"):
        st.markdown(r"""
**Score range**: $F_{\text{causal}} \in [0, 1]$, higher = fairer.

| Score | Meaning |
|-------|---------|
| F = 1.0 | Demographics explain **none** of the residual → perfectly fair |
| F = 0.0 | Demographics explain **all** of the residual → maximally unfair |
| F ≈ 0.95–0.98 | Weak demographic influence (typical for Shenzhen data) |

**Gradient** (how the optimizer adjusts each cell):

$$\frac{\partial F}{\partial R_c} = \frac{2}{R'MR}\left[((I-H)R)_c - F \cdot (R_c - \bar{R})\right]$$

| Cell situation | Gradient sign | Optimizer action |
|----------------|:------------:|-----------------|
| Over-served wealthy cell | **−** | Push service down |
| Under-served poor cell | **+** | Push service up |
| Anomalous (not demographic) | ~0 | Leave alone |
        """)


# =========================================================================
# TAB 3: Matrix Algebra
# =========================================================================
with tab_matrix:
    st.header("Matrix Algebra Walkthrough")
    st.markdown("Interactive exploration of the hat matrix projection and variance decomposition.")

    # --- Mini example ---
    st.subheader("Step-by-Step with Toy Data")
    st.markdown("A 4-cell example to build intuition before the real data.")

    n_toy = st.slider("Number of toy cells", 4, 20, 6)
    np.random.seed(42)
    toy_income = np.random.randn(n_toy)  # standardized income
    toy_income.sort()
    toy_R = 0.3 * toy_income + 0.05 * np.random.randn(n_toy)  # R correlated with income

    X_toy = np.column_stack([np.ones(n_toy), toy_income])
    H_toy = build_hat_matrix(X_toy)
    M_toy = build_centering_matrix(n_toy)

    R_hat_toy = H_toy @ toy_R
    e_toy = (np.eye(n_toy) - H_toy) @ toy_R

    ss_res_toy = toy_R @ (np.eye(n_toy) - H_toy) @ toy_R
    ss_tot_toy = toy_R @ M_toy @ toy_R
    r2_toy = 1 - ss_res_toy / ss_tot_toy if ss_tot_toy > 1e-10 else 0
    f_toy = 1 - r2_toy

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²_demo (toy)", f"{r2_toy:.4f}")
    with col2:
        st.metric("F_causal (toy)", f"{f_toy:.4f}")
    with col3:
        st.metric("SS_res / SS_tot", f"{ss_res_toy:.4f} / {ss_tot_toy:.4f}")

    # Decomposition visualization
    fig_decomp = go.Figure()
    x_labels = [f"Cell {i+1}" for i in range(n_toy)]

    fig_decomp.add_trace(go.Bar(
        name="R (total residual)", x=x_labels, y=toy_R,
        marker_color=D3_COLORS[0], opacity=0.4,
    ))
    fig_decomp.add_trace(go.Bar(
        name="HR (demographic component)", x=x_labels, y=R_hat_toy,
        marker_color=D3_COLORS[1],
    ))
    fig_decomp.add_trace(go.Bar(
        name="(I−H)R (fair component)", x=x_labels, y=e_toy,
        marker_color=D3_COLORS[2],
    ))
    fig_decomp.update_layout(
        title="Residual Decomposition: R = HR + (I−H)R",
        barmode="group", height=400,
        yaxis_title="Value",
    )
    st.plotly_chart(fig_decomp, use_container_width=True)

    with st.expander("Verify orthogonality: R̂ · e = 0"):
        dot_product = R_hat_toy @ e_toy
        st.markdown(f"**R̂ · e = {dot_product:.2e}** (should be ≈ 0)")
        st.markdown(f"**‖R‖² = {np.sum(toy_R**2):.4f}**")
        st.markdown(f"**‖R̂‖² + ‖e‖² = {np.sum(R_hat_toy**2) + np.sum(e_toy**2):.4f}** (Pythagorean theorem)")

    with st.expander("View the Hat Matrix H"):
        st.markdown(f"H is a {n_toy}×{n_toy} matrix. Each row sums to 1 (because of the intercept).")
        df_H = pd.DataFrame(
            H_toy, columns=[f"c{i+1}" for i in range(n_toy)],
            index=[f"c{i+1}" for i in range(n_toy)],
        )
        st.dataframe(df_H.style.format("{:.3f}").background_gradient(cmap="RdBu_r", vmin=-0.5, vmax=0.5), height=250)

    st.divider()

    # --- Real data decomposition ---
    st.subheader("Real Data: Variance Decomposition")

    R_hat_real = data["R_hat"]
    e_real = data["R"] - R_hat_real

    ss_explained = np.sum((R_hat_real - R_hat_real.mean()) ** 2)
    ss_residual = data["fcausal_result"]["ss_res"]
    ss_total = data["fcausal_result"]["ss_tot"]

    fig_pie = go.Figure(data=[go.Pie(
        labels=["Fair (not demographic)", "Demographic influence"],
        values=[data["fcausal_result"]["f_causal"], data["fcausal_result"]["r2_demo"]],
        marker_colors=[D3_COLORS[2], D3_COLORS[3]],
        textinfo="label+percent",
        hole=0.4,
    )])
    fig_pie.update_layout(
        title=f"Variance Decomposition of Residuals (N={data['n_cells']} cells)",
        height=400,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
| Quantity | Value |
|----------|-------|
| SS_tot (R'MR) | {ss_total:.4f} |
| SS_res (R'(I−H)R) | {ss_residual:.4f} |
| SS_explained | {ss_total - ss_residual:.4f} |
| R²_demo | {data['fcausal_result']['r2_demo']:.6f} |
| **F_causal** | **{data['fcausal_result']['f_causal']:.6f}** |
        """)

    with col2:
        fig_scatter_rhat = px.scatter(
            x=data["R"], y=R_hat_real,
            labels={"x": "Actual Residual (R)", "y": "Demographic-Predicted (R̂ = HR)"},
            title="R̂ vs R: How well do demographics predict residuals?",
            color_discrete_sequence=[D3_COLORS[1]],
            opacity=0.6,
        )
        lim = max(abs(data["R"].min()), abs(data["R"].max()))
        fig_scatter_rhat.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim,
                                   line=dict(dash="dash", color="gray"))
        fig_scatter_rhat.update_layout(height=400)
        st.plotly_chart(fig_scatter_rhat, use_container_width=True)


# =========================================================================
# TAB 4: g₀(D) Demand Model
# =========================================================================
with tab_g0:
    _is_nn = data["g0_model_type"] == "Neural Network"
    _model_label = "neural net" if _is_nn else "power basis"

    st.header("g₀(D): The Demand Model")
    st.markdown(
        f"g₀(D) removes the demand effect from service ratios to produce residuals. "
        f"**Active model: {data['g0_model_type']}.**"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"Period-Level R² ({_model_label})", f"{data['g0_r2']:.4f}")
    with col2:
        st.metric("Period-Level R² (isotonic)", f"{data['g0_iso_r2']:.4f}")
    with col3:
        st.metric(f"Cell-Level R² ({_model_label})", f"{data['g0_cell_r2']:.4f}")
    with col4:
        st.metric("Cell-Level R² (isotonic)", f"{data['g0_iso_cell_r2']:.4f}")

    # Show power basis R² for comparison when NN is active
    if _is_nn:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Power Basis R² (period, reference)", f"{data['g0_pb_r2']:.4f}")
        with col2:
            st.metric("Power Basis R² (cell, reference)", f"{data['g0_pb_cell_r2']:.4f}")

    with st.expander("**Why two models?** Role of g₀(D) vs Hat Matrix"):
        st.markdown("""
| Component | Role | Depends on |
|-----------|------|:----------:|
| **g₀(D)** | Remove demand effect → produce residuals R | Demand (changes during optimization) |
| **H** (hat matrix) | Project R onto demographics → measure influence | Demographics (fixed, constant) |

The hat matrix replaces the need for a **demographic model**.
But g₀(D) — the **demand-only model** — is still essential for producing the residuals.
Any differentiable function g₀: D → Y works — the F_causal formulation is agnostic to
how g₀ is implemented. Use the sidebar to switch between **Power Basis** (OLS, FWL-compatible)
and **Neural Network** (learned nonlinear mapping).
        """)

    with st.expander("**Why two R² values?** Period-level vs cell-level g₀"):
        st.markdown(f"""
g₀(D) is fit at **two levels of aggregation**, each serving a distinct purpose:

| Level | N observations | R² ({_model_label}) | Purpose |
|-------|:-----------:|:----------:|---------|
| **Period-level** | {data['n_period_obs']:,} | {data['g0_r2']:.4f} | Model quality metric — how well g₀ captures the D→Y relationship in raw (cell, period) data |
| **Cell-level** | {data['n_cells']:,} | {data['g0_cell_r2']:.4f} | Residual analysis — fitted on cell-aggregated Y=ΣS/ΣD for the hat matrix decomposition |

**Why the cell-level R² is lower**: Cell Y = ΣS/ΣD (ratio of totals) follows a different
distribution than period-level Y = S/D. At the cell level, demand-driven variance is smoothed
out by averaging over many periods, leaving less variance for g₀ to explain. Increasing the
**Min Demand** filter in the sidebar produces cleaner cells and higher cell-level R².

**Why cell-level g₀ is used for F_causal**: The hat matrix operates on cell-level residuals.
Cell-level g₀ ensures residuals are **centered** (mean ≈ 0), which is essential for the hat
matrix's orthogonal decomposition R = R̂ + e to be exact.
        """)

    # --- Demand vs service ratio scatter (period-level data, what g₀ was fit on) ---
    sort_idx = np.argsort(data["demands_period"])
    demands_sorted = data["demands_period"][sort_idx]
    g0_sorted = data["g0_pred_period"][sort_idx]
    g0_iso_sorted = data["g0_iso_pred_period"][sort_idx]
    g0_pb_sorted = data["g0_pb_pred_period"][sort_idx]

    fig_g0 = go.Figure()
    fig_g0.add_trace(go.Scatter(
        x=data["demands_period"], y=data["ratios_period"], mode="markers",
        name="Observed Y = S/D (period-level)", marker=dict(color=D3_COLORS[0], opacity=0.15, size=3),
    ))

    if _is_nn:
        # Show NN curve (active), power basis (reference), and isotonic (reference)
        fig_g0.add_trace(go.Scatter(
            x=demands_sorted, y=g0_sorted, mode="lines",
            name=f"g₀(D) Neural Net (R²={data['g0_r2']:.4f})",
            line=dict(color=D3_COLORS[3], width=3),
        ))
        fig_g0.add_trace(go.Scatter(
            x=demands_sorted, y=g0_pb_sorted, mode="lines",
            name=f"g₀(D) Power Basis (R²={data['g0_pb_r2']:.4f})",
            line=dict(color=D3_COLORS[1], width=2, dash="dot"),
        ))
    else:
        # Show power basis (active) and isotonic (reference)
        fig_g0.add_trace(go.Scatter(
            x=demands_sorted, y=g0_sorted, mode="lines",
            name=f"g₀(D) Power Basis (R²={data['g0_r2']:.4f})",
            line=dict(color=D3_COLORS[1], width=3),
        ))

    fig_g0.add_trace(go.Scatter(
        x=demands_sorted, y=g0_iso_sorted, mode="lines",
        name=f"g₀(D) Isotonic (R²={data['g0_iso_r2']:.4f})",
        line=dict(color=D3_COLORS[2], width=2, dash="dash"),
    ))

    fig_g0.update_layout(
        title=f"Period-Level: Demand → Service Ratio (N={data['n_period_obs']:,}, "
              f"Active model R²={data['g0_r2']:.4f})",
        xaxis_title="Demand (D)", yaxis_title="Service Ratio (Y = S/D)",
        height=500,
    )
    st.plotly_chart(fig_g0, use_container_width=True)

    # --- Model-specific details ---
    if _is_nn:
        with st.expander("**Neural Network Model Details**", expanded=True):
            nn_diag = data["nn_diag_period"]
            st.markdown(f"**Architecture**: `{nn_diag['architecture']}`")
            st.markdown(f"**Parameters**: {nn_diag['n_params']:,} trainable weights")
            st.markdown(f"**Training**: {nn_diag['epochs']:,} epochs, lr={nn_diag['lr']}")
            st.markdown(f"**Final MSE loss** (normalized): {nn_diag['final_loss']:.6f}")

            # Training loss curve
            loss_hist = nn_diag["loss_history"]
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=list(range(len(loss_hist))),
                y=loss_hist,
                mode="lines",
                name="Training Loss (MSE)",
                line=dict(color=D3_COLORS[1]),
            ))
            fig_loss.update_layout(
                title="Training Loss Curve (Period-Level g₀)",
                xaxis_title="Epoch",
                yaxis_title="MSE Loss (normalized scale)",
                height=350,
            )
            st.plotly_chart(fig_loss, use_container_width=True)

            st.markdown(f"""
**R² comparison** (period-level):

| Model | R² | Parameters |
|-------|:--:|:----------:|
| **Neural Network** (active) | **{data['g0_r2']:.4f}** | {nn_diag['n_params']:,} |
| Power Basis (reference) | {data['g0_pb_r2']:.4f} | 4 |
| Isotonic (reference) | {data['g0_iso_r2']:.4f} | non-parametric |
            """)

            st.info(
                "**FWL compatibility note**: The Frisch-Waugh-Lovell theorem requires g₀ "
                "to be linear in parameters. With a neural network g₀, the FWL equivalence "
                "does **not** hold. However, the F_causal formulation remains valid — it still "
                "measures demographic independence of residuals via the hat matrix. "
                "The mediation analysis interpretation (isolating the direct X→Y path by "
                "conditioning on D) is preserved regardless of g₀'s functional form."
            )

        # Cell-level NN details
        if data["nn_diag_cell"] is not None:
            with st.expander("**Cell-Level Neural Network Details**"):
                nn_cell = data["nn_diag_cell"]
                st.markdown(f"**Architecture**: `{nn_cell['architecture']}`")
                st.markdown(f"**Final MSE loss** (normalized): {nn_cell['final_loss']:.6f}")

                loss_hist_cell = nn_cell["loss_history"]
                fig_loss_cell = go.Figure()
                fig_loss_cell.add_trace(go.Scatter(
                    x=list(range(len(loss_hist_cell))),
                    y=loss_hist_cell,
                    mode="lines",
                    name="Training Loss (MSE)",
                    line=dict(color=D3_COLORS[3]),
                ))
                fig_loss_cell.update_layout(
                    title="Training Loss Curve (Cell-Level g₀)",
                    xaxis_title="Epoch",
                    yaxis_title="MSE Loss (normalized scale)",
                    height=300,
                )
                st.plotly_chart(fig_loss_cell, use_container_width=True)
    else:
        with st.expander("**Power Basis Model Details**"):
            st.latex(r"g_0(D) = \beta_0 + \frac{\beta_1}{D+1} + \frac{\beta_2}{\sqrt{D+1}} + \beta_3 \cdot \sqrt{D+1}")

            coeffs = data["g0_pb_diag"].get("coefficients", {})
            intercept = data["g0_pb_diag"].get("intercept", "N/A")
            if coeffs:
                st.markdown(f"""
| Term | Coefficient | Interpretation |
|------|:-----------:|---------------|
| Intercept (β₀) | {intercept} | Asymptotic base level |
| 1/(D+1) (β₁) | {coeffs.get('D^(-1)', 'N/A')} | Rapid hyperbolic decay |
| 1/√(D+1) (β₂) | {coeffs.get('D^(-0.5)', 'N/A')} | Moderate decay |
| √(D+1) (β₃) | {coeffs.get('D^(0.5)', 'N/A')} | Slow sub-linear growth |
                """)

            st.markdown("""
**Why power basis over isotonic?**
- Matches isotonic R² within 0.0003
- Linear in parameters → hat matrix compatible
- Enables Frisch-Waugh-Lovell validation
- Baseline formulation retains isotonic for backward compatibility
            """)


# =========================================================================
# TAB 5: Causal Foundations
# =========================================================================
with tab_causal:
    st.header("Causal Analysis Foundations")
    st.markdown(
        "The reformulated F_causal is grounded in established causal inference and algorithmic fairness theory. "
        "This section maps the formulation to each framework."
    )

    # --- Causal DAG ---
    st.subheader("1. The Causal Graph")

    # Build DAG with Plotly
    fig_dag = go.Figure()

    # Node positions
    nodes = {
        "Demographics\n(X)": (0.5, 1.0),
        "Demand\n(D)": (0.15, 0.5),
        "Service Ratio\n(Y = S/D)": (0.5, 0.0),
        "Infrastructure,\nPOI, etc.": (0.85, 0.5),
    }

    # Edges
    edges = [
        ("Demographics\n(X)", "Demand\n(D)", "mediated\n(acceptable)", D3_COLORS[2]),
        ("Demographics\n(X)", "Service Ratio\n(Y = S/D)", "DIRECT\n(unfair)", D3_COLORS[3]),
        ("Demographics\n(X)", "Infrastructure,\nPOI, etc.", "", "#999999"),
        ("Demand\n(D)", "Service Ratio\n(Y = S/D)", "", D3_COLORS[2]),
        ("Infrastructure,\nPOI, etc.", "Service Ratio\n(Y = S/D)", "", "#999999"),
    ]

    for src, dst, label, color in edges:
        x0, y0 = nodes[src]
        x1, y1 = nodes[dst]
        width = 4 if "DIRECT" in label else 2
        dash = None if "DIRECT" in label else "dot"
        fig_dag.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=width,
            arrowcolor=color,
        )
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            fig_dag.add_annotation(
                x=mx, y=my, text=f"<b>{label}</b>", showarrow=False,
                font=dict(size=11, color=color),
                bgcolor="white", borderpad=2,
            )

    for name, (x, y) in nodes.items():
        color = D3_COLORS[3] if "Demographic" in name else D3_COLORS[0] if "Service" in name else D3_COLORS[2]
        fig_dag.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text", text=[name],
            textposition="middle center",
            marker=dict(size=60, color=color, opacity=0.15),
            textfont=dict(size=12, color="black"),
            showlegend=False, hoverinfo="skip",
        ))

    fig_dag.update_layout(
        title="Causal DAG: Demographics → Service",
        xaxis=dict(range=[-0.15, 1.15], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.3, 1.3], showgrid=False, zeroline=False, showticklabels=False),
        height=420, plot_bgcolor="white",
    )
    st.plotly_chart(fig_dag, use_container_width=True)

    st.markdown("""
    **The formulation isolates the DIRECT path** (X → Y) by removing the mediated path (X → D → Y)
    via g₀(D), then testing whether residuals R = Y − g₀(D) correlate with X.
    """)

    st.divider()

    # --- Counterfactual Fairness ---
    st.subheader("2. Counterfactual Fairness (Kusner et al., 2017)")

    with st.expander("Definition and Connection to F_causal", expanded=True):
        st.markdown(r"""
**Definition**: A decision is counterfactually fair if, in a counterfactual world where the
individual belonged to a different demographic group, the decision would remain the same.

**The counterfactual question for FAMAIL**: *"If a grid cell were located in a
different-income district (everything else equal), would its service ratio change?"*

**How F_causal answers this**:
- If $R^2_{\text{demo}} = 0$ → Changing a cell's district would NOT change its residual → **Counterfactually fair**
- If $R^2_{\text{demo}} > 0$ → Changing district WOULD change residual → **Counterfactually unfair**

$F_{\text{causal}} = 1 - R^2_{\text{demo}}$ directly operationalizes counterfactual fairness at the
**aggregate level**: it measures the degree to which the service distribution would change
under counterfactual demographic reassignment.
        """)

    # Visualization: what-if by district
    districts = data["districts"]
    unique_districts = sorted(set(districts))
    if len(unique_districts) > 1:
        df_district = pd.DataFrame({
            "District": districts,
            "Residual (R)": data["R"],
            "Predicted (R̂)": data["R_hat"],
            "Demand": data["demands"],
        })
        fig_cf = px.box(
            df_district, x="District", y="Residual (R)",
            title="Residual Distribution by District — Counterfactual Fairness Test",
            color="District", color_discrete_sequence=D3_COLORS,
        )
        fig_cf.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_cf.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_cf, use_container_width=True)
        st.caption(
            "If F_causal = 1 (perfect fairness), all districts would have the same residual distribution. "
            "Differences across districts indicate demographic influence."
        )

    st.divider()

    # --- Controlled Direct Effect ---
    st.subheader("3. Controlled Direct Effect — CDE (Pearl, 2001)")

    with st.expander("Definition and Connection to F_causal", expanded=True):
        st.markdown(r"""
**Definition**: The CDE measures the direct effect of treatment $X$ on outcome $Y$,
while **holding a mediator $Z$ constant**:

$$\text{CDE}(z) = E[Y \mid do(X = x_1), Z = z] - E[Y \mid do(X = x_0), Z = z]$$

**In FAMAIL**:
- $X$ = demographic features (the "treatment")
- $Y$ = service ratio (the "outcome")
- $Z = D$ = demand (the "mediator", controlled via g₀(D))

**How F_causal computes CDE**: The regression coefficients $\hat{\beta}$ from regressing R on X
give the CDE per unit change in each demographic feature:

$$\text{CDE}_j = \hat{\beta}_j$$

**Y terms for CDE evaluation** (Meeting 23 action item):
- **Y1** = $\bar{Y}$: overall average service ratio
- **Y2** = $g_0(D)$: demand-conditional expected service
- **Y3** = $g_0(D) + \hat{\beta}'x$: demand + demographic conditional expected service
        """)

    # Show CDE coefficients
    beta = data["beta"]
    feature_names = data["feature_names"]

    cde_data = []
    for i, fname in enumerate(feature_names):
        cde_data.append({"Feature": fname, "CDE (β)": beta[i + 1]})  # skip intercept

    df_cde = pd.DataFrame(cde_data)

    fig_cde = px.bar(
        df_cde, x="Feature", y="CDE (β)",
        title="Controlled Direct Effect: Regression Coefficients (β)",
        color="Feature", color_discrete_sequence=D3_COLORS,
    )
    fig_cde.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_cde.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_cde, use_container_width=True)

    st.caption(
        "Each bar shows how much a 1-standard-deviation increase in that demographic feature "
        "changes the expected service ratio, **holding demand constant**. "
        "Positive = over-served; Negative = under-served."
    )

    st.divider()

    # --- Mediation Analysis ---
    st.subheader("4. Mediation Analysis (Baron & Kenny, 1986; Pearl, 2001)")

    with st.expander("Direct vs. Indirect Effects", expanded=False):
        st.markdown(r"""
**Total effect** of demographics on service = **Direct effect** + **Indirect effect**

- **Indirect effect** (X → D → Y): Demographics influence demand, which influences service.
  This is a market mechanism — **acceptable**.
- **Direct effect** (X → Y, controlling for D): Demographics influence service *beyond* demand.
  Driver preference for affluent areas, infrastructure advantages, etc. — **potentially unfair**.

**The formulation isolates the direct effect** by conditioning on demand through g₀(D). The residual
R = Y − g₀(D) removes the indirect (mediated) path, leaving only the direct demographic influence.

$F_{\text{causal}} = 1 - R^2_{\text{demo}}$ measures the strength of this direct effect.
The optimization target is to **eliminate the direct path** while preserving the indirect path.
        """)

    st.divider()

    # --- FWL Theorem ---
    st.subheader("5. Frisch-Waugh-Lovell Theorem")

    with st.expander("Formal Equivalence of the Two-Stage Procedure", expanded=False):
        st.markdown(r"""
**FWL Theorem**: Regressing $Y$ on $[D, X]$ and extracting the partial R² of $X$ is
**equivalent** to:
1. Regressing $Y$ on $D$ → get residuals $R$
2. Regressing $R$ on $X$ → get $R^2_{\text{demo}}$

This is **exactly** the two-stage procedure used by F_causal. The FWL theorem guarantees that our
approach gives the **same result** as the gold-standard multiple regression — as long as
g₀(D) is linear in parameters (true for power basis).

**Implication**: The formulation is not an approximation. It is mathematically equivalent to the
standard partial R² computation from a single regression of Y on both demand and demographics.
        """)

    st.divider()

    # --- Conditional Statistical Parity ---
    st.subheader("6. Conditional Statistical Parity")

    with st.expander("Fairness Through Awareness vs. Unawareness", expanded=False):
        st.markdown(r"""
**Fairness through unawareness** (baseline): Ignore demographics entirely.
Problem: ignoring demographics doesn't mean they aren't influencing outcomes.

**Fairness through awareness** (F_causal): Explicitly measure demographic influence
to identify and correct disparities. Demographics appear only in the hat matrix
(the auditing tool), never in trajectory modification directly.

**Conditional statistical parity**: $E[Y \mid X, D] = E[Y \mid D]$ — service
shouldn't depend on demographics after controlling for demand.
$F_{\text{causal}} = 1$ corresponds to exact conditional statistical parity at the mean.
        """)

    # --- Summary scoring ---
    st.divider()
    st.subheader("Framework Alignment Summary")

    framework_data = pd.DataFrame({
        "Framework": [
            "Counterfactual Fairness",
            "Controlled Direct Effect",
            "Mediation Analysis",
            "Partial R² / FWL",
            "Conditional Statistical Parity",
            "Fairness Through Awareness",
        ],
        "Alignment": [
            "Aggregate-level counterfactual fairness via R²_demo = 0 test",
            "β coefficients = CDE per demographic feature",
            "Isolates direct X→Y path by conditioning on D",
            "Exact equivalence via FWL theorem",
            "F_causal = 1 ↔ E[Y|X,D] = E[Y|D]",
            "Demographics used to audit, not predict",
        ],
        "Strength": ["Strong", "Strong", "Strong", "Exact", "At-the-mean", "Strong"],
    })

    st.dataframe(
        framework_data.style.apply(
            lambda x: ["background-color: #e8f5e9" if v in ["Strong", "Exact"]
                       else "background-color: #fff3e0" for v in x],
            subset=["Strength"],
        ),
        use_container_width=True, hide_index=True,
    )


# =========================================================================
# TAB 6: Real Data Analysis
# =========================================================================
with tab_real:
    st.header("Real Data Analysis: Shenzhen Taxi Service")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("F_causal", f"{data['fcausal_result']['f_causal']:.6f}")
    with col2:
        st.metric("R²_demo", f"{data['fcausal_result']['r2_demo']:.6f}")
    with col3:
        st.metric("N cells", data["n_cells"])
    with col4:
        mean_Y = data["Y"].mean()
        st.metric("Mean Y", f"{mean_Y:.3f}")

    st.divider()

    # --- Spatial maps ---
    st.subheader("Spatial Distribution")

    grid_R = np.full((48, 90), np.nan)
    grid_Rhat = np.full((48, 90), np.nan)
    grid_Y = np.full((48, 90), np.nan)
    grid_D = np.full((48, 90), np.nan)

    for i, (x, y) in enumerate(data["cells"]):
        grid_R[x, y] = data["R"][i]
        grid_Rhat[x, y] = data["R_hat"][i]
        grid_Y[x, y] = data["Y"][i]
        grid_D[x, y] = data["demands"][i]

    map_choice = st.selectbox(
        "Select map variable",
        ["Residual (R)", "Demographic-Predicted (R̂)", "Fair Component (R − R̂)",
         "Service Ratio (Y)", "Demand (D)"],
    )

    grid_map = {
        "Residual (R)": grid_R,
        "Demographic-Predicted (R̂)": grid_Rhat,
        "Fair Component (R − R̂)": grid_R - grid_Rhat,
        "Service Ratio (Y)": grid_Y,
        "Demand (D)": grid_D,
    }[map_choice]

    use_diverging = map_choice in ["Residual (R)", "Demographic-Predicted (R̂)", "Fair Component (R − R̂)"]
    colorscale = "RdBu_r" if use_diverging else "Viridis"
    zmid = 0 if use_diverging else None

    fig_map = px.imshow(
        np.flipud(grid_map),
        color_continuous_scale=colorscale,
        color_continuous_midpoint=zmid,
        labels={"color": map_choice},
        title=f"Spatial Map: {map_choice}",
        aspect="auto",
    )
    fig_map.update_layout(height=500)
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()

    # --- Demographics vs residuals ---
    st.subheader("Demographics vs. Residuals")

    feature_select = st.selectbox("Demographic Feature", data["feature_names"])
    feat_idx = data["feature_names"].index(feature_select)
    feat_vals = data["demo_features"][:, feat_idx]

    fig_demo_r = px.scatter(
        x=feat_vals, y=data["R"],
        color=data["districts"],
        labels={"x": feature_select, "y": "Residual (R)", "color": "District"},
        title=f"Residual vs {feature_select} (colored by district)",
        color_discrete_sequence=D3_COLORS,
        opacity=0.7,
    )
    # Add regression line
    z = np.polyfit(feat_vals, data["R"], 1)
    x_line = np.linspace(feat_vals.min(), feat_vals.max(), 100)
    fig_demo_r.add_trace(go.Scatter(
        x=x_line, y=np.polyval(z, x_line),
        mode="lines", name="Linear trend",
        line=dict(color="black", dash="dash", width=2),
    ))
    fig_demo_r.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_demo_r.update_layout(height=450)
    st.plotly_chart(fig_demo_r, use_container_width=True)

    st.divider()

    # --- District summary ---
    st.subheader("District-Level Summary")

    district_stats = []
    for d in sorted(set(data["districts"])):
        mask = data["districts"] == d
        district_stats.append({
            "District": d,
            "N Cells": int(mask.sum()),
            "Mean Demand": float(data["demands"][mask].mean()),
            "Mean Y": float(data["Y"][mask].mean()),
            "Mean R": float(data["R"][mask].mean()),
            "Mean R̂": float(data["R_hat"][mask].mean()),
            "Std R": float(data["R"][mask].std()),
            f"Mean {data['feature_names'][0]}": float(data["demo_features"][mask, 0].mean()),
        })

    df_districts = pd.DataFrame(district_stats)
    st.dataframe(
        df_districts.style.format({
            "Mean Demand": "{:.1f}", "Mean Y": "{:.3f}",
            "Mean R": "{:.4f}", "Mean R̂": "{:.4f}", "Std R": "{:.4f}",
            f"Mean {data['feature_names'][0]}": "{:.0f}",
        }).background_gradient(subset=["Mean R̂"], cmap="RdBu_r"),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "R̂ (demographic-predicted residual) shows the hat matrix's assessment of each district's "
        "demographic advantage/disadvantage. Districts with large |R̂| contribute most to R²_demo."
    )


# =========================================================================
# TAB 7: Validation
# =========================================================================
with tab_validate:
    st.header("Validation")
    st.markdown("Mathematical and empirical checks confirming the formulation's correctness.")

    # --- Check 1: FWL Equivalence ---
    st.subheader("1. Frisch-Waugh-Lovell Equivalence Check")

    if data["g0_model_type"] == "Neural Network":
        st.warning(
            "**FWL theorem does not apply** when g₀ is a neural network. "
            "The FWL theorem requires g₀ to be linear in parameters (true for the power basis, "
            "but not for neural networks). The check below uses a power-basis joint regression "
            "as reference — a mismatch with the NN-based R²_demo is **expected** and does not "
            "indicate an error in F_causal."
        )

    with st.expander("Details", expanded=True):
        st.markdown("""
        The FWL theorem says: regressing R on demographics gives the same R² as the
        partial R² from a single regression of Y on [demand features, demographics].
        """)

        # Compute single-regression R² for comparison (always uses power basis features)
        X_demand = build_power_basis_features(data["demands"], include_intercept=True)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_demo_scaled = scaler.fit_transform(data["demo_features"])

        # Reduced model: Y ~ demand (power basis)
        from sklearn.linear_model import LinearRegression
        model_red = LinearRegression().fit(X_demand, data["Y"])
        r2_demand = model_red.score(X_demand, data["Y"])

        # Full model: Y ~ demand + demographics
        X_full = np.column_stack([X_demand, X_demo_scaled])
        model_full = LinearRegression().fit(X_full, data["Y"])
        r2_full = model_full.score(X_full, data["Y"])

        delta_r2 = r2_full - r2_demand

        # F_causal's R²_demo
        r2_demo_option_b = data["fcausal_result"]["r2_demo"]

        # FWL relationship: ΔR² = R²_demo × (1 - R²_demand)
        fwl_predicted = r2_demo_option_b * (1 - r2_demand)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R²_demand (reduced, power basis)", f"{r2_demand:.6f}")
        with col2:
            st.metric("R²_full (demand + demo)", f"{r2_full:.6f}")
        with col3:
            st.metric("ΔR² (full − reduced)", f"{delta_r2:.6f}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R²_demo (F_causal)", f"{r2_demo_option_b:.6f}")
        with col2:
            st.metric("FWL prediction: R²_demo × (1−R²_demand)", f"{fwl_predicted:.6f}")
        with col3:
            match = abs(delta_r2 - fwl_predicted) < 0.001
            if data["g0_model_type"] == "Neural Network":
                st.metric("ΔR² ≈ FWL prediction?",
                          "N/A (nonlinear g₀)" if not match else "✅ Yes (coincidental)")
            else:
                st.metric("ΔR² ≈ FWL prediction?", "✅ Yes" if match else "❌ No")

    st.divider()

    # --- Check 2: Gradient Direction ---
    st.subheader("2. Gradient Direction Verification")

    with st.expander("Details", expanded=True):
        st.markdown("""
        The gradient should push over-served wealthy cells DOWN and under-served poor cells UP.
        """)

        # Compute gradient: ∂F/∂R = (2/SS_tot) * [(I-H)R - F * MR]
        I_minus_H = data["hat_result"]["I_minus_H_demo"]
        M_mat = data["hat_result"]["M"]
        R = data["R"]
        F = data["fcausal_result"]["f_causal"]
        ss_tot = data["fcausal_result"]["ss_tot"]

        IH_R = I_minus_H @ R
        MR = M_mat @ R
        gradient = (2.0 / (ss_tot + 1e-10)) * (IH_R - F * MR)

        # Classify cells
        R_hat = data["R_hat"]
        over_served_wealthy = (R > np.median(R)) & (R_hat > 0)
        under_served_poor = (R < np.median(R)) & (R_hat < 0)

        grad_over = gradient[over_served_wealthy].mean() if over_served_wealthy.any() else 0
        grad_under = gradient[under_served_poor].mean() if under_served_poor.any() else 0

        col1, col2 = st.columns(2)
        with col1:
            sign_correct = grad_over < 0
            st.metric(
                "Avg gradient (over-served wealthy)",
                f"{grad_over:.6f}",
                delta="Negative ✅" if sign_correct else "Positive ❌",
                delta_color="normal" if sign_correct else "inverse",
            )
        with col2:
            sign_correct = grad_under > 0
            st.metric(
                "Avg gradient (under-served poor)",
                f"{grad_under:.6f}",
                delta="Positive ✅" if sign_correct else "Negative ❌",
                delta_color="normal" if sign_correct else "inverse",
            )

        fig_grad = px.scatter(
            x=R_hat, y=gradient,
            labels={"x": "R̂ (demographic prediction)", "y": "∂F/∂R (gradient)"},
            title="Gradient vs Demographic Prediction: Should be negatively correlated",
            color_discrete_sequence=[D3_COLORS[4]],
            opacity=0.5,
        )
        fig_grad.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_grad.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_grad.update_layout(height=400)
        st.plotly_chart(fig_grad, use_container_width=True)

        st.caption(
            "Cells where demographics predict over-service (R̂ > 0) should get negative gradient "
            "(push down). Cells where demographics predict under-service (R̂ < 0) should get "
            "positive gradient (push up). The negative correlation confirms correct gradient direction."
        )

    st.divider()

    # --- Check 3: Orthogonality ---
    st.subheader("3. Orthogonality Check: R̂ ⊥ e")

    with st.expander("Details", expanded=True):
        e_real = data["R"] - data["R_hat"]
        dot_prod = data["R_hat"] @ e_real
        st.metric("R̂ · e (should be ≈ 0)", f"{dot_prod:.2e}")

        norm_R = np.linalg.norm(data["R"]) ** 2
        norm_Rhat = np.linalg.norm(data["R_hat"]) ** 2
        norm_e = np.linalg.norm(e_real) ** 2
        st.markdown(f"""
| Check | Value |
|-------|-------|
| ‖R‖² | {norm_R:.4f} |
| ‖R̂‖² + ‖e‖² | {norm_Rhat + norm_e:.4f} |
| Difference | {abs(norm_R - norm_Rhat - norm_e):.2e} |
        """)
        st.markdown("**Pythagorean theorem**: ‖R‖² = ‖R̂‖² + ‖e‖² confirms orthogonal decomposition.")

    st.divider()

    # --- Check 4: CDE Y terms ---
    st.subheader("4. CDE Y-Term Decomposition")

    with st.expander("Details", expanded=True):
        Y1 = data["Y"].mean()
        Y2 = data["g0_pred"]
        Y3 = data["g0_pred"] + data["R_hat"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Y1 = E[Y]", f"{Y1:.4f}")
        with col2:
            st.metric("Y2 range = g₀(D)", f"[{Y2.min():.3f}, {Y2.max():.3f}]")
        with col3:
            st.metric("Y3 range = g₀(D) + β'x", f"[{Y3.min():.3f}, {Y3.max():.3f}]")

        st.markdown(f"""
**CDE per feature** (how much demographics shift service, holding demand constant):
        """)

        for i, fname in enumerate(data["feature_names"]):
            b = data["beta"][i + 1]
            st.markdown(f"- **{fname}**: CDE = {b:+.6f} per std. dev.")

        # Show Y2 vs Y3 to visualize demographic shift
        fig_y23 = px.scatter(
            x=Y2, y=Y3,
            labels={"x": "Y2 = g₀(D) (demand only)", "y": "Y3 = g₀(D) + β'x (demand + demographics)"},
            title="Y2 vs Y3: Demographic Shift of Expected Service",
            color=data["districts"],
            color_discrete_sequence=D3_COLORS,
            opacity=0.7,
        )
        lim_y = [min(Y2.min(), Y3.min()), max(Y2.max(), Y3.max())]
        fig_y23.add_shape(type="line", x0=lim_y[0], y0=lim_y[0], x1=lim_y[1], y1=lim_y[1],
                          line=dict(dash="dash", color="gray"))
        fig_y23.update_layout(height=400)
        st.plotly_chart(fig_y23, use_container_width=True)

        st.caption(
            "Points above the diagonal: demographics predict higher service than demand alone. "
            "Points below: demographics predict lower service. The spread around the diagonal "
            "is the CDE — the direct demographic effect on service."
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.caption(
    f"Dashboard: F_causal Explorer\n"
    f"Data: {data['n_cells']} active cells\n"
    f"F_causal = {data['fcausal_result']['f_causal']:.6f}\n"
    f"R²_demo = {data['fcausal_result']['r2_demo']:.6f}"
)
