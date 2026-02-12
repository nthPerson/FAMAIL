"""
Demographic Explorer: g(D, x) Estimator Tuning Dashboard.

This standalone dashboard helps explore the relationship between demographic
data and taxi service patterns in Shenzhen, and find the best g(D, x) estimator
for the causal fairness term.

Features:
- Side-by-side spatial maps of demographic variables
- District-level service summaries
- Multiple model architecture comparison (OLS, Ridge, Lasso, ElasticNet, RF, GBT)
- Leave-One-District-Out cross-validation
- Statistical diagnostics (VIF, p-values, AIC/BIC, residual tests)
- Feature engineering and correlation analysis
- Cross-validation detail and fairness impact assessment

Usage:
    streamlit run objective_function/causal_fairness/demographic_explorer.py

See DEMOGRAPHIC_EXPLORER_GUIDE.md for detailed usage instructions.
"""

import sys
from pathlib import Path
import pickle
from typing import Dict, Any, List
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
    WEEKDAYS_JULY, WEEKDAYS_AUGUST, WEEKDAYS_SEPTEMBER, WEEKDAYS_TOTAL,
)
from term import CausalFairnessTerm
from utils import (
    estimate_g_function,
    compute_r_squared,
    load_active_taxis_data,
    prepare_demographic_analysis_data,
    compute_residual_demographic_correlation,
    enrich_demographic_features,
    build_feature_matrix,
    fit_g_dx_model,
    lodo_cross_validate,
    compute_model_diagnostics,
    compute_permutation_importance,
    compute_option_a1_demographic_attribution,
    compute_option_b_demographic_disparity,
    compute_option_c_partial_r_squared,
)

# District colors (D3 qualitative palette)
D3_COLORS = px.colors.qualitative.D3


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def load_data(filepath: str) -> Dict:
    """Load and cache data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def cached_enrich(demo_grid, feature_names_tuple):
    """Cached wrapper for enrich_demographic_features."""
    return enrich_demographic_features(demo_grid, list(feature_names_tuple))


def build_grid_from_df(df: pd.DataFrame, value_col: str, agg: str = 'mean') -> np.ndarray:
    """Aggregate DataFrame column to a (48, 90) grid."""
    grid = np.full((48, 90), np.nan)
    grouped = df.groupby(['x', 'y'])[value_col].agg(agg)
    for (x, y), val in grouped.items():
        if 0 <= x < 48 and 0 <= y < 90:
            grid[x, y] = val
    return grid


def build_district_colorscale(n_districts: int):
    """Build discrete colorscale for district map."""
    qual_colors = D3_COLORS[:n_districts]
    colorscale = []
    for i, color in enumerate(qual_colors):
        low = i / n_districts
        high = (i + 1) / n_districts
        colorscale.append([low, color])
        colorscale.append([high, color])
    return colorscale


def _style_threshold(val, thresholds, colors):
    """Apply background color based on threshold ranges.

    Args:
        val: Value to evaluate
        thresholds: List of (lower, upper) bounds (checked in order)
        colors: List of CSS background-color strings matching thresholds
    Returns:
        CSS string for cell styling
    """
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ''
    for (lo, hi), color in zip(thresholds, colors):
        if lo <= v < hi:
            return f'background-color: {color}'
    return f'background-color: {colors[-1]}' if colors else ''


# Reusable color constants for table styling
_GREEN = '#c8e6c9'
_YELLOW = '#fff9c4'
_RED = '#ffcdd2'
_BLUE_LIGHT = '#bbdefb'


@st.cache_data(show_spinner="Computing causal fairness baseline...")
def cached_compute_baseline(
    demand_path: str,
    supply_path: str,
    demo_path: str,
    district_path: str,
    period_type: str,
    num_days: int,
    days_filter_tuple,
    estimation_method: str,
    n_bins: int,
    gd_poly_degree: int,
    min_demand: int,
    include_zero_supply: bool,
    max_ratio_val: float,
):
    """Cache-friendly computation of baseline g(D) and master DataFrame.

    Returns all serializable data needed by the dashboard.
    g_func (a closure) must be reconstructed outside the cache.
    """
    raw_data = load_data(demand_path)

    supply_data = None
    if Path(supply_path).exists():
        try:
            supply_data = load_active_taxis_data(supply_path)
        except Exception:
            pass

    demo_data = load_data(demo_path)
    district_data = load_data(district_path)

    # Enrich demographics
    enriched_grid, enriched_names = enrich_demographic_features(
        demo_data['demographics_grid'], list(demo_data['feature_names']),
    )
    raw_feature_names = list(demo_data['feature_names'])

    # Build config and compute
    max_ratio = max_ratio_val if max_ratio_val > 0 else None
    days_filter = list(days_filter_tuple) if days_filter_tuple else None
    config = CausalFairnessConfig(
        period_type=period_type,
        estimation_method=estimation_method,
        n_bins=n_bins,
        poly_degree=gd_poly_degree,
        min_demand=min_demand,
        max_ratio=max_ratio,
        include_zero_supply=include_zero_supply,
        num_days=num_days,
        days_filter=days_filter,
        active_taxis_data_path=supply_path if Path(supply_path).exists() else None,
    )

    term = CausalFairnessTerm(config)
    auxiliary_data = {'pickup_dropoff_counts': raw_data}
    if supply_data is not None:
        auxiliary_data['active_taxis'] = supply_data

    breakdown = term.compute_with_breakdown({}, auxiliary_data)

    components = breakdown['components']
    demands = np.array(components['demands'])
    ratios = np.array(components['ratios'])
    expected = np.array(components['expected'])
    keys = components['keys']

    df = prepare_demographic_analysis_data(
        demands=demands, ratios=ratios, expected=expected, keys=keys,
        demo_grid=enriched_grid, feature_names=enriched_names,
        district_id_grid=district_data['district_id_grid'],
        valid_mask=district_data['valid_mask'],
        district_names=district_data['district_names'],
        data_is_one_indexed=True,
    )

    # Convert district_data sets to lists for serializability
    district_data_serializable = {
        k: (list(v) if isinstance(v, set) else v)
        for k, v in district_data.items()
    }

    return (
        df, breakdown, enriched_grid, enriched_names,
        raw_feature_names, district_data_serializable,
    )


# =============================================================================
# TAB 1: SPATIAL MAPS
# =============================================================================

def render_spatial_maps_tab(
    df: pd.DataFrame,
    enriched_grid: np.ndarray,
    enriched_names: List[str],
    valid_mask: np.ndarray,
    district_id_grid: np.ndarray,
    district_names: List[str],
):
    """Render side-by-side demographic heatmaps."""
    st.header("Spatial Maps")
    st.markdown("Compare any two variables spatially across the 48×90 grid.")

    # Build variable options: demographics + computed service fields
    variable_options = list(enriched_names) + ['MeanDemand', 'MeanServiceRatio', 'MeanResidual', 'DistrictMap']

    col1, col2 = st.columns(2)

    with col1:
        var_left = st.selectbox("Left Map Variable", variable_options, index=0, key="map_left")
    with col2:
        var_right = st.selectbox("Right Map Variable", variable_options,
                                 index=min(1, len(variable_options) - 1), key="map_right")

    def get_grid(var_name):
        if var_name == 'DistrictMap':
            grid = district_id_grid.astype(float)
            grid[~valid_mask] = np.nan
            return grid, 'District ID', None
        elif var_name == 'MeanDemand':
            return build_grid_from_df(df, 'demand'), 'Mean Demand', 'Viridis'
        elif var_name == 'MeanServiceRatio':
            return build_grid_from_df(df, 'ratio'), 'Mean Service Ratio', 'Viridis'
        elif var_name == 'MeanResidual':
            return build_grid_from_df(df, 'residual'), 'Mean Residual', 'RdBu_r'
        elif var_name in enriched_names:
            idx = enriched_names.index(var_name)
            grid = enriched_grid[:, :, idx].copy()
            grid[~valid_mask] = np.nan
            return grid, var_name, 'Viridis'
        return np.full((48, 90), np.nan), var_name, 'Viridis'

    col1, col2 = st.columns(2)

    for col, var_name in [(col1, var_left), (col2, var_right)]:
        with col:
            grid, label, cscale = get_grid(var_name)
            if var_name == 'DistrictMap':
                n_d = len(district_names)
                fig = px.imshow(
                    np.flipud(grid), aspect="auto",
                    color_continuous_scale=build_district_colorscale(n_d),
                    title=f"District Map",
                    labels=dict(x="Y Grid (Longitude)", y="X Grid (Latitude)", color="District"),
                )
                # Add district name annotations
                tickvals = list(range(n_d))
                ticktext = district_names[:n_d]
                fig.update_coloraxes(
                    colorbar=dict(tickvals=[(i + 0.5) / n_d * (n_d - 1) for i in range(n_d)],
                                  ticktext=ticktext),
                )
            else:
                fig = px.imshow(
                    np.flipud(grid), aspect="auto",
                    color_continuous_scale=cscale or 'Viridis',
                    title=var_name,
                    labels=dict(x="Y Grid (Longitude)", y="X Grid (Latitude)", color=label),
                )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Correlation between the two selected variables
    if var_left != var_right and var_left != 'DistrictMap' and var_right != 'DistrictMap':
        grid_l, _, _ = get_grid(var_left)
        grid_r, _, _ = get_grid(var_right)
        mask = ~np.isnan(grid_l) & ~np.isnan(grid_r)
        if mask.sum() > 2:
            corr = float(np.corrcoef(grid_l[mask], grid_r[mask])[0, 1])
            st.metric(f"Pearson Correlation: {var_left} vs {var_right}", f"{corr:.4f}")


# =============================================================================
# TAB 2: DISTRICT SUMMARIES
# =============================================================================

def render_district_summaries_tab(
    df: pd.DataFrame,
    g_func,
    district_names: List[str],
):
    """Render district-level service summaries."""
    st.header("District Summaries")

    # 1. Service ratio vs demand scatter by district
    st.subheader("1. Service Ratio vs Demand by District")
    fig = go.Figure()

    districts = sorted(df['district_name'].unique())
    for i, dist in enumerate(districts):
        mask = df['district_name'] == dist
        fig.add_trace(go.Scatter(
            x=df.loc[mask, 'demand'], y=df.loc[mask, 'ratio'],
            mode='markers', name=dist, marker=dict(size=3, color=D3_COLORS[i % len(D3_COLORS)]),
            opacity=0.4,
        ))

    # g(D) overlay
    d_range = np.linspace(max(df['demand'].min(), 0.1), df['demand'].max(), 200)
    try:
        g_vals = g_func(d_range)
        fig.add_trace(go.Scatter(
            x=d_range, y=g_vals, mode='lines', name='g(D)',
            line=dict(color='black', width=2, dash='dash'),
        ))
    except Exception:
        pass

    fig.update_layout(
        xaxis_title="Demand (D)", yaxis_title="Service Ratio (Y = S/D)",
        height=500, legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2 & 3. Mean ratio and mean residual by district
    col1, col2 = st.columns(2)

    dist_stats = df.groupby('district_name').agg(
        mean_ratio=('ratio', 'mean'),
        mean_residual=('residual', 'mean'),
        count=('ratio', 'count'),
    ).sort_values('mean_ratio')

    with col1:
        st.subheader("2. Mean Service Ratio by District")
        fig = px.bar(
            dist_stats.reset_index(), x='district_name', y='mean_ratio',
            color='mean_ratio', color_continuous_scale='Viridis',
            title="Mean Service Ratio by District",
        )
        fig.update_layout(xaxis_title="District", yaxis_title="Mean Y = S/D", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("3. Mean Residual by District")
        colors = ['#d32f2f' if r < 0 else '#1976d2' for r in dist_stats['mean_residual']]
        fig = go.Figure(go.Bar(
            x=dist_stats.index, y=dist_stats['mean_residual'],
            marker_color=colors,
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title="District", yaxis_title="Mean Residual (Y - g(D))",
            title="Mean Residual by District (red=under-served, blue=over-served)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # 4. Residual box plots
    st.subheader("4. Residual Distribution by District")
    fig = px.box(
        df, x='district_name', y='residual',
        color='district_name', color_discrete_sequence=D3_COLORS,
        title="Residual Distribution by District",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="District", yaxis_title="Residual (Y - g(D))",
        showlegend=False, height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TAB 3: MODEL COMPARISON
# =============================================================================

MODEL_TYPES = {
    'OLS': 'ols',
    'Ridge': 'ridge',
    'Lasso': 'lasso',
    'ElasticNet': 'elasticnet',
    'OLS + Interactions': 'ols_interactions',
    'Random Forest': 'random_forest',
    'Gradient Boosting': 'gradient_boosting',
}


def render_model_comparison_tab(
    df: pd.DataFrame,
    selected_features: List[str],
    poly_degree: int,
    alpha: float,
    l1_ratio: float,
    n_estimators: int,
    max_depth: int,
):
    """Render model comparison tab — fit and compare g(D, x) models."""
    st.header("Model Comparison")
    st.markdown(
        "Compare different g(D, x) model architectures. "
        "**LODO R²** (Leave-One-District-Out) is the primary evaluation metric."
    )

    avail_features = [f for f in selected_features if f in df.columns]
    if len(avail_features) == 0:
        st.warning("No demographic features selected. Check the sidebar.")
        return

    demo_matrix = df[avail_features].values
    demands = df['demand'].values
    ratios = df['ratio'].values
    district_ids = df['district_id'].values

    # Model selector (checkboxes)
    st.subheader("Select Models to Fit")
    cols = st.columns(4)
    selected_models = {}
    for i, (label, mtype) in enumerate(MODEL_TYPES.items()):
        with cols[i % 4]:
            default = mtype in ('ols', 'ridge', 'lasso')
            selected_models[label] = st.checkbox(label, value=default, key=f"model_{mtype}")

    models_to_fit = {label: mtype for label, mtype in MODEL_TYPES.items() if selected_models.get(label)}

    if not models_to_fit:
        st.info("Select at least one model above.")
        return

    # Fit models button
    if st.button("Fit Models", type="primary"):
        results = {}
        progress = st.progress(0)
        status = st.status("Fitting models...", expanded=True)

        total = len(models_to_fit)
        for i, (label, mtype) in enumerate(models_to_fit.items()):
            status.update(label=f"Fitting {label}...")

            # Fit on full data
            model_result = fit_g_dx_model(
                demands, ratios, demo_matrix, avail_features,
                model_type=mtype, poly_degree=poly_degree,
                alpha=alpha, l1_ratio=l1_ratio,
                n_estimators=n_estimators, max_depth=max_depth,
            )

            # LODO cross-validation
            lodo_result = lodo_cross_validate(
                demands, ratios, demo_matrix, district_ids, avail_features,
                model_type=mtype, poly_degree=poly_degree,
                alpha=alpha, l1_ratio=l1_ratio,
                n_estimators=n_estimators, max_depth=max_depth,
            )

            # Diagnostics (OLS-based, only for linear models)
            diag = None
            is_linear = mtype in ('ols', 'ridge', 'lasso', 'elasticnet', 'ols_interactions')
            if is_linear:
                try:
                    diag = compute_model_diagnostics(
                        demands, ratios, demo_matrix, avail_features,
                        poly_degree=poly_degree,
                        include_interactions=(mtype == 'ols_interactions'),
                    )
                except Exception:
                    pass

            results[label] = {
                'model_result': model_result,
                'lodo_result': lodo_result,
                'diagnostics': diag,
                'model_type': mtype,
            }

            progress.progress((i + 1) / total)

        status.update(label="All models fitted!", state="complete")
        st.session_state['model_results'] = results
        st.session_state['model_features'] = avail_features
        st.session_state['model_poly_degree'] = poly_degree

    # Display results
    if 'model_results' not in st.session_state:
        st.info("Click 'Fit Models' to run the comparison.")
        return

    results = st.session_state['model_results']
    _display_model_results(results, df)


def _display_model_results(results: Dict, df: pd.DataFrame):
    """Display model comparison results with color coding and auto-identification."""
    # Build summary table
    summary_rows = []
    for label, r in results.items():
        mr = r['model_result']
        lr = r['lodo_result']
        diag = r.get('diagnostics')
        row = {
            'Model': label,
            'Train R²': mr['r2_train'],
            'LODO R²': lr['lodo_r2'],
            'Overfit Gap': mr['r2_train'] - lr['lodo_r2'],
            'N_params': mr['n_params'],
            'AIC': diag['aic'] if diag else None,
            'BIC': diag['bic'] if diag else None,
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('LODO R²', ascending=False)

    st.subheader("Comparison Table")

    # Color-coded styling
    best_label = summary_df.iloc[0]['Model']

    def _color_gap(val):
        try:
            v = float(val)
            if v < 0.05:
                return f'background-color: {_GREEN}'
            elif v < 0.10:
                return f'background-color: {_YELLOW}'
            return f'background-color: {_RED}'
        except (ValueError, TypeError):
            return ''

    def _highlight_best(row):
        return [f'background-color: {_BLUE_LIGHT}' if row['Model'] == best_label else '' for _ in row]

    styled = summary_df.style.apply(_highlight_best, axis=1)
    styled = styled.applymap(_color_gap, subset=['Overfit Gap'])
    styled = styled.format({
        'Train R²': '{:.4f}', 'LODO R²': '{:.4f}', 'Overfit Gap': '{:.4f}',
        'AIC': lambda x: f'{x:.0f}' if pd.notna(x) else '—',
        'BIC': lambda x: f'{x:.0f}' if pd.notna(x) else '—',
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ How to interpret the comparison table"):
        st.markdown("""
        - **LODO R²** is the primary metric — how well the model generalizes to unseen districts.
        - **Overfit Gap** = Train R² − LODO R². Green (< 0.05) is healthy, red (> 0.10) is concerning.
        - **AIC/BIC**: Lower = better. BIC penalizes complexity more than AIC.
        - **N_params**: Keep low relative to 10 districts to avoid overfitting.
        - The **blue row** highlights the best model by LODO R².
        """)

    # Best model + auto-recommendation (Phase 5)
    best_r = results[best_label]
    best_lodo = best_r['lodo_result']['lodo_r2']
    best_gap = best_r['model_result']['r2_train'] - best_lodo

    if best_lodo < 0.01:
        st.warning(
            f"**No model achieves meaningful LODO R²** (best: {best_label} = {best_lodo:.4f}). "
            "Demographics may not explain service residuals in this configuration. "
            "Consider: (1) changing g(D) method, (2) adding more features, "
            "(3) using 'all' temporal aggregation for more data per district."
        )
    elif best_gap > 0.10:
        st.warning(
            f"**Recommendation:** {best_label} has the best LODO R² ({best_lodo:.4f}) "
            f"but shows overfitting (gap = {best_gap:.4f}). Consider using a regularized model "
            "(Ridge/Lasso) or reducing the number of features."
        )
    else:
        st.success(
            f"**Recommendation:** Based on LODO R² ({best_lodo:.4f}) and overfit gap "
            f"({best_gap:.4f}), **{best_label}** is recommended for the causal fairness term."
        )

    # Individual overfitting / underfitting flags
    for label, r in results.items():
        gap = r['model_result']['r2_train'] - r['lodo_result']['lodo_r2']
        lodo = r['lodo_result']['lodo_r2']
        if gap > 0.1 and label != best_label:
            st.caption(f"⚠️ {label}: overfit gap = {gap:.4f}")
        if lodo < 0.01 and label != best_label:
            st.caption(f"ℹ️ {label}: LODO R² = {lodo:.4f} (near zero — minimal predictive power)")

    # Per-model expanders
    st.subheader("Model Details")
    for label, r in results.items():
        with st.expander(f"{label} — LODO R² = {r['lodo_result']['lodo_r2']:.4f}"):
            mr = r['model_result']
            lr = r['lodo_result']

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Train R²", f"{mr['r2_train']:.4f}",
                      help="R² on training data (all districts). Can overfit.")
            c2.metric("LODO R²", f"{lr['lodo_r2']:.4f}",
                      help="Leave-One-District-Out cross-validated R². Primary generalization metric.")
            c3.metric("N_params", mr['n_params'],
                      help="Number of model parameters. Keep low relative to 10 districts.")
            if r.get('diagnostics'):
                c4.metric("AIC", f"{r['diagnostics']['aic']:.0f}",
                          help="Akaike Information Criterion. Lower = better fit-complexity trade-off.")

            # Coefficient or importance chart
            if mr['coefficients'] is not None:
                coef_df = pd.DataFrame([
                    {'Feature': k, 'Coefficient': v}
                    for k, v in mr['coefficients'].items()
                ]).sort_values('Coefficient', key=abs, ascending=False)
                fig = px.bar(
                    coef_df, x='Feature', y='Coefficient',
                    title=f"{label}: Coefficients",
                    color='Coefficient', color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0,
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            elif mr['feature_importances'] is not None:
                imp_df = pd.DataFrame([
                    {'Feature': k, 'Importance': v}
                    for k, v in mr['feature_importances'].items()
                ]).sort_values('Importance', ascending=False)
                fig = px.bar(
                    imp_df, x='Feature', y='Importance',
                    title=f"{label}: Feature Importances",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            # Actual vs predicted + LODO residual histogram
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.scatter(
                    x=mr['predicted'], y=df['ratio'].values[:len(mr['predicted'])],
                    labels=dict(x="Predicted Y", y="Actual Y"),
                    title="Actual vs Predicted (Train)",
                    opacity=0.3,
                )
                fit_min = float(min(mr['predicted']))
                fit_max = float(max(mr['predicted']))
                fig.add_trace(go.Scatter(
                    x=[fit_min, fit_max], y=[fit_min, fit_max],
                    mode='lines', name='Perfect fit',
                    line=dict(color='red', dash='dash'),
                ))
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                oof_resid = lr['oof_residuals']
                valid_resid = oof_resid[~np.isnan(oof_resid)]
                if len(valid_resid) > 0:
                    fig = px.histogram(
                        x=valid_resid, nbins=50,
                        title="LODO Residual Distribution",
                        labels=dict(x="OOF Residual (Y - ŷ)"),
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

            # Per-district LODO R²
            if lr['per_district_r2']:
                dist_r2_df = pd.DataFrame([
                    {'District': f"D{k}", 'R²': v, 'N': lr['per_district_n'].get(k, 0)}
                    for k, v in sorted(lr['per_district_r2'].items())
                ])
                fig = px.bar(
                    dist_r2_df, x='District', y='R²',
                    title=f"{label}: Per-District LODO R²",
                    hover_data=['N'],
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TAB 4: STATISTICAL DIAGNOSTICS
# =============================================================================

def render_diagnostics_tab(df: pd.DataFrame):
    """Render statistical diagnostics for a selected model."""
    st.header("Statistical Diagnostics")

    if 'model_results' not in st.session_state:
        st.info("Fit models in the 'Model Comparison' tab first.")
        return

    results = st.session_state['model_results']
    model_labels = list(results.keys())
    selected_label = st.selectbox("Select Model", model_labels)

    r = results[selected_label]
    mr = r['model_result']
    diag = r.get('diagnostics')

    if diag is None:
        # Compute diagnostics for this model
        avail_features = st.session_state.get('model_features', [])
        demo_matrix = df[avail_features].values if avail_features else np.array([])
        if len(avail_features) > 0:
            try:
                diag = compute_model_diagnostics(
                    df['demand'].values, df['ratio'].values,
                    demo_matrix, avail_features,
                    poly_degree=mr.get('poly_degree', 2),
                    include_interactions=mr.get('include_interactions', False),
                )
            except Exception as e:
                st.error(f"Could not compute diagnostics: {e}")
                return
        else:
            st.warning("No demographic features available for diagnostics.")
            return

    # 1. Coefficients table (color-coded p-values)
    st.subheader("1. Coefficient Significance")
    coef_display = diag['coefficients_table'].copy()
    coef_display['Sig'] = coef_display['significant_05'].map({True: '*', False: ''})
    show_cols = ['Feature', 'Coefficient', 'StdErr', 't_stat', 'p_value', 'Sig']
    coef_show = coef_display[show_cols].copy()

    def _color_pvalue(val):
        try:
            p = float(val)
            if p < 0.05:
                return f'background-color: {_GREEN}'
            elif p < 0.10:
                return f'background-color: {_YELLOW}'
            return f'background-color: {_RED}'
        except (ValueError, TypeError):
            return ''

    styled_coef = coef_show.style.applymap(_color_pvalue, subset=['p_value'])
    styled_coef = styled_coef.format({
        'Coefficient': '{:.6f}', 'StdErr': '{:.6f}',
        't_stat': '{:.3f}', 'p_value': '{:.4e}',
    })
    st.dataframe(styled_coef, use_container_width=True, hide_index=True)

    # 2. VIF table (color-coded)
    st.subheader("2. Variance Inflation Factors (VIF)")
    vif_df = diag['vif'].copy()
    vif_df['Status'] = vif_df['VIF'].apply(
        lambda v: 'High collinearity' if v > 10 else ('Moderate' if v > 5 else 'OK')
    )

    def _color_vif(val):
        try:
            v = float(val)
            if v > 10:
                return f'background-color: {_RED}'
            elif v > 5:
                return f'background-color: {_YELLOW}'
            return f'background-color: {_GREEN}'
        except (ValueError, TypeError):
            return ''

    styled_vif = vif_df.style.applymap(_color_vif, subset=['VIF'])
    styled_vif = styled_vif.format({'VIF': '{:.2f}'})
    st.dataframe(styled_vif, use_container_width=True, hide_index=True)

    high_vif = vif_df[vif_df['VIF'] > 10]
    if len(high_vif) > 0:
        st.warning(f"Features with VIF > 10: {', '.join(high_vif['Feature'].tolist())}. Consider removing.")

    with st.expander("ℹ️ How to interpret VIF"):
        st.markdown("""
        - **VIF < 5** (green): No concerning multicollinearity.
        - **VIF 5-10** (yellow): Moderate. Coefficients may be unstable.
        - **VIF > 10** (red): High. Feature is nearly a linear combination of others. Remove or combine.
        - With only 10 districts, even moderate VIF can destabilize estimates.
        """)

    # 3. Information criteria
    st.subheader("3. Model Fit Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AIC", f"{diag['aic']:.0f}",
              help="Akaike Information Criterion. Lower = better fit-complexity trade-off.")
    c2.metric("BIC", f"{diag['bic']:.0f}",
              help="Bayesian IC. Penalizes complexity more than AIC. Lower = better.")
    c3.metric("R²", f"{diag['r_squared']:.4f}",
              help="Proportion of Y variance explained by the model.")
    c4.metric("Adj R²", f"{diag['r_squared_adj']:.4f}",
              help="R² adjusted for number of predictors. Penalizes adding useless features.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Condition Number", f"{diag['condition_number']:.0f}",
              help="Measures multicollinearity severity. > 1000 = concerning, > 10000 = severe.")
    c2.metric("Durbin-Watson", f"{diag['durbin_watson']:.4f}",
              help="Tests for autocorrelation. ~2.0 = none, < 1.5 or > 2.5 = concern.")
    bp_str = f"{diag['breusch_pagan_p']:.4e}"
    c3.metric("Breusch-Pagan p", bp_str,
              help="Tests for heteroscedasticity (unequal error variance). p < 0.05 = detected.")

    # 4. Residual diagnostics
    st.subheader("4. Residual Diagnostics")
    col_a, col_b = st.columns(2)

    residuals = mr['residuals']
    predicted = mr['predicted']

    with col_a:
        # Q-Q plot
        from scipy.stats import probplot
        osm, osr = probplot(residuals, dist='norm', fit=False)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Residuals',
                                 marker=dict(size=2, opacity=0.5)))
        qq_min, qq_max = min(osm), max(osm)
        fig.add_trace(go.Scatter(x=[qq_min, qq_max], y=[qq_min, qq_max],
                                 mode='lines', name='Normal', line=dict(color='red', dash='dash')))
        fig.update_layout(title="Q-Q Plot", xaxis_title="Theoretical Quantiles",
                          yaxis_title="Sample Quantiles", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Residuals vs fitted
        fig = px.scatter(
            x=predicted, y=residuals,
            labels=dict(x="Fitted Values", y="Residuals"),
            title="Residuals vs Fitted Values",
            opacity=0.3,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ How to interpret residual diagnostic plots"):
        st.markdown("""
        **Q-Q Plot** (left):
        - Points should follow the red diagonal line if residuals are normally distributed.
        - S-shape at tails suggests heavy tails or outliers. Curve suggests skewness.
        - Minor deviations are acceptable with large samples (>1000 observations).

        **Residuals vs Fitted** (right):
        - **Good**: Random scatter around y=0 with constant spread (homoscedasticity).
        - **Bad**: Fan/funnel shape = heteroscedasticity (variance depends on fitted value).
        - **Bad**: Curved pattern = model is missing a non-linear relationship.
        """)

    # 5. Permutation importance
    st.subheader("5. Permutation Feature Importance")
    avail_features = st.session_state.get('model_features', [])
    if avail_features:
        with st.spinner("Computing permutation importance..."):
            demo_matrix = df[avail_features].values
            perm_df = compute_permutation_importance(
                df['demand'].values, df['ratio'].values,
                demo_matrix, avail_features, mr, n_repeats=10,
            )
        fig = px.bar(
            perm_df, x='Feature', y='Importance_Mean',
            error_y='Importance_Std',
            title="Permutation Importance (R² drop when feature shuffled)",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ How to interpret permutation importance"):
            st.markdown("""
            - Shows the **R² drop** when each feature's values are randomly shuffled.
            - **Higher bar** = feature contributes more to predictive accuracy.
            - **Negative importance** = shuffling *improves* the model (feature may add noise — consider removing).
            - Error bars show variability across shuffles (wider = less stable importance).
            """)


# =============================================================================
# TAB 1: FEATURE ANALYSIS & SELECTION
# =============================================================================

def _compute_recommended_features(
    all_features: List[str],
    corr_matrix: pd.DataFrame,
    df: pd.DataFrame,
    raw_feature_names: List[str],
) -> tuple:
    """Auto-select a minimal, well-conditioned feature set.

    VIF is recomputed on surviving candidates after correlation-pair removal,
    because VIF on all 20 features with only 10 district profiles is singular (all inf).

    Returns:
        (recommended_list, removal_reasons_dict)
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    candidates = set(all_features)
    reasons = {}

    # Always remove AreaKm2 (geographic constant, not demographic)
    if 'AreaKm2' in candidates:
        candidates.discard('AreaKm2')
        reasons['AreaKm2'] = 'Geographic constant (not demographic)'

    # Remove log+raw duplicates (perfect collinearity)
    log_raw_pairs = [
        ('AvgHousingPricePerSqM', 'LogHousingPrice'),
        ('GDPin10000Yuan', 'LogGDP'),
        ('EmployeeCompensation100MYuan', 'LogCompensation'),
        ('PopDensityPerKm2', 'LogPopDensity'),
    ]
    for raw_name, log_name in log_raw_pairs:
        if raw_name in candidates and log_name in candidates:
            # Keep derived (log) over raw for these near-perfect correlations
            candidates.discard(log_name)
            reasons[log_name] = f'Near-perfect correlation with {raw_name} (log transform)'

    # Remove one from each remaining high-correlation pair
    for i in range(len(all_features)):
        for j in range(i + 1, len(all_features)):
            f1, f2 = all_features[i], all_features[j]
            if f1 not in candidates or f2 not in candidates:
                continue
            r = abs(corr_matrix.loc[f1, f2]) if f1 in corr_matrix.index and f2 in corr_matrix.index else 0
            if r > 0.85:
                # Prefer derived over raw
                f1_raw = f1 in raw_feature_names
                f2_raw = f2 in raw_feature_names
                if f1_raw and not f2_raw:
                    to_remove = f1
                elif f2_raw and not f1_raw:
                    to_remove = f2
                else:
                    # Both same type: remove arbitrarily (second in pair)
                    to_remove = f2
                kept = f2 if to_remove == f1 else f1
                candidates.discard(to_remove)
                reasons[to_remove] = f'Correlated with {kept} (r={r:.2f})'

    # Recompute VIF on surviving candidates only (not the original 20-feature set)
    surviving = sorted(candidates)
    if len(surviving) >= 2:
        X_surv = df[surviving].dropna()
        if len(X_surv) > 0:
            X_arr = X_surv.values
            vif_post = {}
            for i, fname in enumerate(surviving):
                try:
                    vif_post[fname] = float(variance_inflation_factor(X_arr, i))
                except Exception:
                    vif_post[fname] = float('inf')

            # Iteratively remove highest-VIF feature until all VIF < 10
            while True:
                worst = max(vif_post, key=vif_post.get)
                if vif_post[worst] <= 10 or len(vif_post) <= 2:
                    break
                candidates.discard(worst)
                reasons[worst] = f'High VIF ({vif_post[worst]:.1f}) after pair removal'
                # Recompute VIF on remaining
                remaining = sorted(candidates)
                if len(remaining) < 2:
                    break
                X_rem = df[remaining].dropna().values
                vif_post = {}
                for i, fname in enumerate(remaining):
                    try:
                        vif_post[fname] = float(variance_inflation_factor(X_rem, i))
                    except Exception:
                        vif_post[fname] = float('inf')

    return sorted(candidates), reasons


def render_feature_analysis_tab(
    df: pd.DataFrame,
    enriched_names: List[str],
    raw_feature_names: List[str],
    selected_features: List[str],
):
    """Render feature analysis, selection, and engineering tab."""
    st.header("Feature Analysis & Selection")

    avail_features = [f for f in enriched_names if f in df.columns]
    sel_avail = [f for f in selected_features if f in df.columns]

    # =================================================================
    # 0. Auto-Recommend Features
    # =================================================================
    st.subheader("Auto-Recommend Features",
                 help="Analyzes all features for redundancy and recommends a minimal set.")

    if len(avail_features) >= 2:
        # Correlation analysis (all features)
        corr_all = df[avail_features].corr()

        # Identify correlated pairs
        high_corr_pairs = []
        for i in range(len(avail_features)):
            for j in range(i + 1, len(avail_features)):
                r = abs(corr_all.iloc[i, j])
                if r > 0.85:
                    high_corr_pairs.append((avail_features[i], avail_features[j], r))

        # Compute recommendation (VIF is computed internally on surviving candidates)
        recommended, removal_reasons = _compute_recommended_features(
            avail_features, corr_all, df, raw_feature_names,
        )

        # Compute VIF on recommended features (meaningful, unlike all-features VIF)
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        rec_vif = {}
        if len(recommended) >= 2:
            X_rec = df[recommended].dropna().values
            for i, fname in enumerate(recommended):
                try:
                    rec_vif[fname] = float(variance_inflation_factor(X_rec, i))
                except Exception:
                    rec_vif[fname] = float('inf')

        # Display findings
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Correlated Pairs** (|r| > 0.85)")
            if high_corr_pairs:
                for f1, f2, r in high_corr_pairs:
                    st.markdown(f"- {f1} & {f2}: r={r:.3f}")
            else:
                st.success("No highly correlated pairs found.")
        with col_b:
            st.markdown("**Recommended Feature VIF**")
            if rec_vif:
                high_rec_vif = {k: v for k, v in rec_vif.items() if v > 10}
                if high_rec_vif:
                    for fname, vif in sorted(high_rec_vif.items(), key=lambda x: -x[1]):
                        st.markdown(f"- {fname}: VIF={vif:.1f} ⚠️")
                else:
                    st.success(f"All {len(rec_vif)} recommended features have VIF < 10.")
                    for fname, vif in sorted(rec_vif.items()):
                        st.markdown(f"- {fname}: VIF={vif:.1f}")
            elif len(recommended) < 2:
                st.info("Need at least 2 features to compute VIF.")

        # Recommendation box
        if removal_reasons:
            with st.expander("Removal reasons", expanded=False):
                for fname, reason in sorted(removal_reasons.items()):
                    st.markdown(f"- **{fname}**: {reason}")

        st.info(f"**Recommended features ({len(recommended)}):** {', '.join(recommended) if recommended else 'None'}")

        if st.button("✅ Apply Recommended Features", type="primary",
                      help="Updates sidebar checkboxes to match the recommendation."):
            # Defer update to next rerun (can't modify widget keys after instantiation)
            st.session_state['_apply_recommended'] = list(recommended)
            if 'model_results' in st.session_state:
                del st.session_state['model_results']
            st.rerun()

        with st.expander("ℹ️ How auto-recommendation works"):
            st.markdown("""
            1. **Remove `AreaKm2`** — geographic constant, not a demographic predictor.
            2. **Correlated pairs** (|r| > 0.85): Remove one from each pair.
               Prefer derived/per-capita features over raw totals.
            3. **High VIF** (> 10): Remove features that are linear combinations of others.
            4. **Target**: 3-5 features for 10 districts (avoids overfitting).
            """)

    st.divider()

    # =================================================================
    # 1. Feature Summary
    # =================================================================
    st.subheader("Feature Summary")

    derived_formulas = {
        'GDPperCapita': 'GDPin10000Yuan / (Pop × 10000)',
        'CompPerCapita': 'Compensation × 1e8 / AvgEmployed',
        'MigrantRatio': 'NonRegistered / TotalPop',
        'LogGDP': 'log1p(GDPin10000Yuan)',
        'LogHousingPrice': 'log1p(AvgHousingPricePerSqM)',
        'LogCompensation': 'log1p(Compensation)',
        'LogPopDensity': 'log1p(PopDensityPerKm2)',
    }

    summary_rows = []
    for fname in avail_features:
        vals = df[fname].dropna()
        is_derived = fname not in raw_feature_names
        summary_rows.append({
            'Feature': fname,
            'Type': 'Derived' if is_derived else 'Raw',
            'Selected': '✓' if fname in selected_features else '',
            'Formula': derived_formulas.get(fname, '—'),
            'Mean': f"{vals.mean():.4g}" if len(vals) > 0 else '—',
            'Std': f"{vals.std():.4g}" if len(vals) > 0 else '—',
            'Min': f"{vals.min():.4g}" if len(vals) > 0 else '—',
            'Max': f"{vals.max():.4g}" if len(vals) > 0 else '—',
            'Unique': len(vals.unique()),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # =================================================================
    # 2. Correlation Matrix
    # =================================================================
    if len(sel_avail) >= 2:
        st.subheader("Feature Correlation Matrix")
        corr_matrix = df[sel_avail].corr()

        fig = px.imshow(
            corr_matrix, text_auto='.2f',
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
            title="Pairwise Correlations (selected features)",
            aspect='auto',
        )
        fig.update_layout(height=max(300, 50 * len(sel_avail)))
        st.plotly_chart(fig, use_container_width=True)

        # Flag high correlations
        high_corr = []
        for i in range(len(sel_avail)):
            for j in range(i + 1, len(sel_avail)):
                c = abs(corr_matrix.iloc[i, j])
                if c > 0.85:
                    high_corr.append((sel_avail[i], sel_avail[j], corr_matrix.iloc[i, j]))
        if high_corr:
            st.warning("High correlations detected (|r| > 0.85) among selected features:")
            for f1, f2, c in high_corr:
                st.markdown(f"- **{f1}** & **{f2}**: r = {c:.3f}")

        with st.expander("ℹ️ How to interpret the correlation matrix"):
            st.markdown("""
            - **Red** = positive correlation, **Blue** = negative correlation.
            - |r| > 0.85 (flagged): features share redundant information.
            - Including highly correlated features inflates VIF and destabilizes coefficients.
            - Prefer keeping one from each correlated pair (derived/per-capita over raw).
            """)

    # =================================================================
    # 3. Feature vs Residual Scatter Grid
    # =================================================================
    if sel_avail and 'residual' in df.columns:
        st.subheader("Feature vs Residual (by District)")
        n_cols = min(3, len(sel_avail))
        cols = st.columns(n_cols)
        for i, fname in enumerate(sel_avail[:9]):
            with cols[i % n_cols]:
                fig = px.scatter(
                    df, x=fname, y='residual', color='district_name',
                    color_discrete_sequence=D3_COLORS,
                    title=fname, opacity=0.3,
                    labels=dict(y="Residual"),
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ How to interpret feature-residual scatters"):
            st.markdown("""
            - A **clear trend** (upward/downward slope) suggests the feature explains residual variance.
            - **No trend** (random scatter) means the feature doesn't predict service beyond demand.
            - Color by district reveals if the relationship is consistent or district-specific.
            """)

    # =================================================================
    # 4. Distribution Comparison (raw vs log)
    # =================================================================
    log_pairs = [
        ('GDPin10000Yuan', 'LogGDP'),
        ('AvgHousingPricePerSqM', 'LogHousingPrice'),
        ('EmployeeCompensation100MYuan', 'LogCompensation'),
        ('PopDensityPerKm2', 'LogPopDensity'),
    ]
    available_pairs = [(r, l) for r, l in log_pairs if r in df.columns and l in df.columns]
    if available_pairs:
        st.subheader("Raw vs Log-Transformed Distributions")
        for raw_name, log_name in available_pairs[:4]:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=raw_name, nbins=30, title=f"Raw: {raw_name}")
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.histogram(df, x=log_name, nbins=30, title=f"Log: {log_name}")
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

def render_export_config(
    df: pd.DataFrame,
    selected_label,
    *,
    district_names: List[str],
    estimation_method: str,
    n_bins: int,
    gd_poly_degree: int,
    model_poly_degree: int,
    alpha: float,
    l1_ratio: float,
    n_estimators: int,
    max_depth: int,
    min_demand: int,
    include_zero_supply: bool,
    max_ratio_val: float,
    period_type: str,
):
    """Render the export configuration section at the bottom of the CV Detail tab."""
    import json
    from datetime import datetime

    st.subheader("6. Export Model Configuration")

    if 'model_results' not in st.session_state or selected_label is None:
        st.info("Fit models in the Model Training tab first, then select a model above to export its configuration.")
        return

    results = st.session_state['model_results']
    if selected_label not in results:
        st.warning(f"Model '{selected_label}' not found in results.")
        return

    r = results[selected_label]
    mr = r['model_result']
    lr = r['lodo_result']
    features = st.session_state.get('model_features', [])

    # Parse model type from label (e.g. "Ridge (α=1.00)" → "ridge")
    model_type = selected_label.split("(")[0].strip().lower().replace(" ", "_")

    # Build per-district R² dict with names
    per_district_r2 = {}
    for d_id, r2_val in sorted(lr.get('per_district_r2', {}).items()):
        d_name = district_names[d_id] if d_id < len(district_names) else f"D{d_id}"
        per_district_r2[d_name] = round(r2_val, 6)

    # Build coefficients dict (mr['coefficients'] is a dict {name: float})
    coefficients = {}
    if mr.get('coefficients') and isinstance(mr['coefficients'], dict):
        for name, coef in mr['coefficients'].items():
            coefficients[name] = round(float(coef), 8)
    elif mr.get('feature_importances') and isinstance(mr['feature_importances'], dict):
        # Tree models store importances instead of coefficients
        for name, imp in mr['feature_importances'].items():
            coefficients[name] = round(float(imp), 8)
    if mr.get('intercept') is not None:
        coefficients["(intercept)"] = round(float(mr['intercept']), 8)

    # Assemble config
    config = {
        "model_config": {
            "model_type": model_type,
            "label": selected_label,
            "features": features,
            "poly_degree": model_poly_degree,
            "include_interactions": mr.get('include_interactions', False),
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "n_estimators": n_estimators if "forest" in model_type or "gradient" in model_type or "rf" in model_type or "gb" in model_type else None,
            "max_depth": max_depth if "forest" in model_type or "gradient" in model_type or "rf" in model_type or "gb" in model_type else None,
        },
        "baseline_config": {
            "estimation_method": estimation_method,
            "n_bins": n_bins,
            "gd_poly_degree": gd_poly_degree,
            "period_type": period_type,
            "min_demand": min_demand,
            "include_zero_supply": include_zero_supply,
            "max_ratio": max_ratio_val,
        },
        "performance": {
            "train_r2": round(mr.get('r2_train', 0), 6),
            "lodo_r2": round(lr.get('lodo_r2', 0), 6),
            "overfit_gap": round(mr.get('r2_train', 0) - lr.get('lodo_r2', 0), 6),
            "aic": round(mr.get('aic', 0), 2) if mr.get('aic') else None,
            "bic": round(mr.get('bic', 0), 2) if mr.get('bic') else None,
            "per_district_r2": per_district_r2,
        },
        "data_summary": {
            "n_districts": len(district_names),
            "n_observations": len(df),
            "n_features": len(features),
        },
        "coefficients": coefficients,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "dashboard_version": "demographic_explorer v2",
    }

    # Remove None values from model_config for cleanliness
    config["model_config"] = {k: v for k, v in config["model_config"].items() if v is not None}

    # --- Display ---
    col_summary, col_json = st.columns([1, 1])

    with col_summary:
        st.markdown("##### Summary")
        gap = config["performance"]["overfit_gap"]
        gap_label = "healthy" if gap < 0.05 else ("moderate" if gap < 0.10 else "concerning")
        st.markdown(f"""
| Setting | Value |
|---------|-------|
| **Model** | {selected_label} |
| **Features** | {', '.join(features) if features else 'None'} |
| **Poly degree** | {model_poly_degree} |
| **Baseline** | {estimation_method}, {n_bins} bins, degree {gd_poly_degree} |
| **Period** | {period_type} |
| **Train R²** | {config['performance']['train_r2']:.4f} |
| **LODO R²** | {config['performance']['lodo_r2']:.4f} |
| **Overfit gap** | {gap:.4f} ({gap_label}) |
| **Observations** | {config['data_summary']['n_observations']:,} |
""")

    with col_json:
        st.markdown("##### Configuration JSON")
        json_str = json.dumps(config, indent=2, ensure_ascii=False)
        st.code(json_str, language="json")

    # Download button
    st.download_button(
        label="📥 Download Configuration (JSON)",
        data=json.dumps(config, indent=2, ensure_ascii=False),
        file_name=f"gdx_config_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        help="Download the full model configuration as a JSON file for later implementation.",
    )

    with st.expander("ℹ️ How to use this configuration"):
        st.markdown("""
This JSON captures everything needed to reproduce the g(D, x) model:

1. **`model_config`**: The model type, features, and hyperparameters. Use this to reconstruct the scikit-learn estimator.
2. **`baseline_config`**: How g(D) was estimated. Needed to compute residuals for the causal fairness term.
3. **`performance`**: Train/LODO R² and per-district breakdown. Use LODO R² as the primary quality indicator.
4. **`coefficients`**: The fitted model weights. For linear models, these can be applied directly without retraining.
5. **`data_summary`**: Context about the dataset used for fitting.

**Next steps**: Pass this configuration to implement g(D, x) in `objective_function/causal_fairness/term.py`, replacing the current demand-only g(D) model where appropriate.
""")


# =============================================================================
# TAB 4: CROSS-VALIDATION DETAIL
# =============================================================================

def render_cv_detail_tab(
    df: pd.DataFrame,
    enriched_grid: np.ndarray,
    enriched_names: List[str],
    valid_mask: np.ndarray,
    district_names: List[str],
    g_func,
    *,
    estimation_method: str = "binning",
    n_bins: int = 10,
    gd_poly_degree: int = 2,
    model_poly_degree: int = 2,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    n_estimators: int = 100,
    max_depth: int = 3,
    min_demand: int = 1,
    include_zero_supply: bool = False,
    max_ratio_val: float = 0.0,
    period_type: str = "hourly",
):
    """Render detailed cross-validation analysis and export configuration."""
    st.header("Cross-Validation Detail")

    if 'model_results' not in st.session_state:
        st.info("Fit models in the 'Model Comparison' tab first.")
        return

    results = st.session_state['model_results']
    model_labels = list(results.keys())
    selected_label = st.selectbox("Select Model for CV Detail", model_labels, key="cv_model_select")

    r = results[selected_label]
    lr = r['lodo_result']
    mr = r['model_result']

    # 1. LODO R² overview
    st.subheader("1. LODO R² Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall LODO R²", f"{lr['lodo_r2']:.4f}",
              help="Aggregated R² from all out-of-fold predictions. Primary generalization metric.")
    c2.metric("Train R²", f"{mr['r2_train']:.4f}",
              help="R² on full training data. Compare with LODO R² to assess overfitting.")
    c3.metric("Overfit Gap", f"{mr['r2_train'] - lr['lodo_r2']:.4f}",
              help="Train R² minus LODO R². < 0.05 healthy, > 0.10 concerning.")

    # Per-district bar chart
    if lr['per_district_r2']:
        dist_r2_data = []
        for d_id, r2 in sorted(lr['per_district_r2'].items()):
            d_name = district_names[d_id] if d_id < len(district_names) else f"D{d_id}"
            dist_r2_data.append({
                'District': d_name,
                'LODO R²': r2,
                'N': lr['per_district_n'].get(d_id, 0),
            })
        dist_r2_df = pd.DataFrame(dist_r2_data)
        fig = px.bar(
            dist_r2_df, x='District', y='LODO R²',
            title="Per-District LODO R²", hover_data=['N'],
            color='LODO R²', color_continuous_scale='Viridis',
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ How to interpret per-district LODO R²"):
            st.markdown("""
- **Positive R²**: The model explains some variance in the held-out district. Higher is better.
- **Near-zero R²**: The model has no predictive power for that district — its service patterns are unique.
- **Negative R²**: The model predicts *worse* than the district mean. This district actively contradicts the model trained on the other 9 districts.
- **Uniform bars**: All districts similar → model generalizes consistently.
- **One outlier**: A single low/negative bar → that district has unique demographics or service patterns. Consider investigating it in the District Summaries tab.
""")

    # 2. Per-district prediction scatters
    st.subheader("2. Per-District Predictions (OOF)")
    oof_preds = lr['oof_predictions']
    valid_oof = ~np.isnan(oof_preds)

    if valid_oof.sum() > 0:
        unique_dists = sorted(df['district_id'].unique())
        n_dists = len(unique_dists)
        n_plot_cols = min(4, n_dists)
        n_plot_rows = (n_dists + n_plot_cols - 1) // n_plot_cols

        fig = make_subplots(
            rows=n_plot_rows, cols=n_plot_cols,
            subplot_titles=[district_names[d] if d < len(district_names) else f"D{d}"
                           for d in unique_dists],
        )
        for idx, d_id in enumerate(unique_dists):
            row = idx // n_plot_cols + 1
            col = idx % n_plot_cols + 1
            mask = (df['district_id'].values == d_id) & valid_oof
            if mask.sum() > 0:
                fig.add_trace(
                    go.Scatter(
                        x=oof_preds[mask], y=df['ratio'].values[mask],
                        mode='markers', marker=dict(size=2, opacity=0.4),
                        name=district_names[d_id] if d_id < len(district_names) else f"D{d_id}",
                        showlegend=False,
                    ),
                    row=row, col=col,
                )
        fig.update_layout(height=250 * n_plot_rows, title_text="Actual vs OOF Predicted per District")
        st.plotly_chart(fig, use_container_width=True)

    # 3. OOF residual spatial map
    st.subheader("3. OOF Residual Spatial Map")
    oof_resid = lr['oof_residuals']
    if valid_oof.sum() > 0:
        # Build a temporary df for grid aggregation
        resid_df = df.copy()
        resid_df['oof_residual'] = oof_resid
        resid_df = resid_df[~np.isnan(resid_df['oof_residual'])]
        grid = build_grid_from_df(resid_df, 'oof_residual')
        abs_max = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 0.01)

        fig = px.imshow(
            np.flipud(grid), aspect="auto",
            color_continuous_scale='RdBu_r', zmin=-abs_max, zmax=abs_max,
            title="Mean OOF Residual per Cell (red=under-predicted, blue=over-predicted)",
            labels=dict(x="Y Grid (Longitude)", y="X Grid (Latitude)", color="OOF Residual"),
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ How to interpret the OOF residual map"):
            st.markdown("""
- **Red cells**: Under-predicted — actual service ratio was higher than the model expected (under-served areas the model missed).
- **Blue cells**: Over-predicted — actual service ratio was lower than predicted (over-served areas the model missed).
- **White/near-zero**: Model predicts well for these cells.
- **Spatial clusters**: If red/blue cells cluster geographically, the model has systematic spatial bias — it may be missing a spatially-varying factor.
- **Scattered noise**: Random red/blue patches suggest the model captures the main patterns and residuals are unsystematic.
- This map uses **out-of-fold** predictions, so clusters here represent genuine model failures, not overfitting artifacts.
""")

    # 4. g(D) vs g(D,x) comparison
    st.subheader("4. g(D) vs g(D, x) per District")
    if g_func is not None and lr['per_district_r2']:
        # Compute per-district R² for demand-only g(D)
        gd_r2_per_dist = {}
        for d_id in sorted(lr['per_district_r2'].keys()):
            mask = df['district_id'].values == d_id
            if mask.sum() > 1:
                y_d = df['ratio'].values[mask]
                try:
                    gd_pred = g_func(df['demand'].values[mask])
                    var_y = np.var(y_d)
                    if var_y > 1e-10:
                        gd_r2_per_dist[d_id] = max(0.0, 1.0 - np.var(y_d - gd_pred) / var_y)
                    else:
                        gd_r2_per_dist[d_id] = 0.0
                except Exception:
                    gd_r2_per_dist[d_id] = 0.0

        compare_data = []
        for d_id in sorted(lr['per_district_r2'].keys()):
            d_name = district_names[d_id] if d_id < len(district_names) else f"D{d_id}"
            compare_data.append({'District': d_name, 'Model': 'g(D)', 'R²': gd_r2_per_dist.get(d_id, 0)})
            compare_data.append({'District': d_name, 'Model': f'g(D,x) [{selected_label}]',
                               'R²': lr['per_district_r2'][d_id]})

        fig = px.bar(
            pd.DataFrame(compare_data), x='District', y='R²', color='Model',
            barmode='group', title="Per-District R²: g(D) vs g(D, x)",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ How to interpret g(D) vs g(D,x)"):
            st.markdown("""
- **g(D,x) bar taller**: Demographics improve prediction for that district. The district has a service pattern correlated with its demographic profile.
- **Both bars similar**: Demographics add little beyond demand alone. The current g(D) model is sufficient for that district.
- **g(D) bar taller**: Adding demographics actually *hurts* for this district (possible overfitting to other districts' patterns).
- **If most districts show improvement**: Strong evidence to adopt g(D,x) in the causal fairness term.
- **If only 1-2 districts improve**: The signal may be idiosyncratic — the demographic effect isn't systematic across the city.
""")

    # 5. Fairness implications
    st.subheader("5. Fairness Metric Implications")
    avail_features = st.session_state.get('model_features', [])
    if avail_features and valid_oof.sum() > 0:
        demo_matrix = df[avail_features].values
        demands = df['demand'].values
        ratios = df['ratio'].values

        # Option B: residual regression using OOF residuals
        valid_mask_oof = ~np.isnan(oof_resid)
        if valid_mask_oof.sum() > 10:
            result_b = compute_option_b_demographic_disparity(
                oof_resid[valid_mask_oof], demo_matrix[valid_mask_oof], avail_features,
            )
            result_c = compute_option_c_partial_r_squared(
                demands, ratios, demo_matrix, avail_features,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Option B: F_causal (1 - R²(OOF_resid ~ x))",
                          f"{result_b['f_causal']:.4f}",
                          help="Using OOF residuals instead of g(D) residuals")
            with col2:
                st.metric("Option C: F_causal (1 - ΔR²)",
                          f"{result_c['f_causal']:.4f}",
                          help="Incremental R² from adding demographics")

            st.info(
                "These fairness scores show how the g(D, x) model's residuals relate to demographics. "
                "Higher F_causal = less demographic bias remaining after accounting for demand."
            )

            with st.expander("ℹ️ Understanding fairness metric options"):
                st.markdown("""
- **Option B** (1 − R²(OOF_resid ~ x)): Regresses the model's out-of-fold residuals on demographics. If demographics *still* predict residuals after g(D,x) accounts for them, there's remaining bias. Score near 1.0 = low remaining bias.
- **Option C** (1 − ΔR²): Measures the *incremental* R² from adding demographics to the demand-only model. Score near 1.0 = demographics add little beyond demand.
- **Comparing scores**: If Option B >> Option C, the model successfully absorbed demographic effects into its predictions. If Option B ≈ Option C, the demographic signal wasn't fully captured.
- **Both near 1.0**: Demand alone explains service well — demographics don't add much. This is actually a *good* outcome for fairness (no demographic bias to correct).
""")

    # ── Export Configuration ─────────────────────────────────────────────
    st.divider()
    render_export_config(
        df, selected_label if 'model_results' in st.session_state else None,
        district_names=district_names,
        estimation_method=estimation_method, n_bins=n_bins,
        gd_poly_degree=gd_poly_degree, model_poly_degree=model_poly_degree,
        alpha=alpha, l1_ratio=l1_ratio,
        n_estimators=n_estimators, max_depth=max_depth,
        min_demand=min_demand, include_zero_supply=include_zero_supply,
        max_ratio_val=max_ratio_val, period_type=period_type,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.set_page_config(
        page_title="Demographic Explorer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔬 Demographic Explorer: g(D, x) Estimator")
    st.markdown(
        "Explore demographic data, compare model architectures, "
        "and find the best g(D, x) estimator for the causal fairness term."
    )

    # =========================================================================
    # SIDEBAR
    # =========================================================================

    st.sidebar.header("⚙️ Configuration")

    # --- Data Files ---
    st.sidebar.subheader("📁 Data Files")
    demand_path = st.sidebar.text_input(
        "Demand Data", value=str(PROJECT_ROOT / "source_data" / "pickup_dropoff_counts.pkl"),
    )
    supply_path = st.sidebar.text_input(
        "Supply Data", value=str(PROJECT_ROOT / "source_data" / "active_taxis_5x5_hourly.pkl"),
    )
    demo_path = st.sidebar.text_input(
        "Demographics Data", value=str(PROJECT_ROOT / "source_data" / "cell_demographics.pkl"),
    )
    district_path = st.sidebar.text_input(
        "District Mapping", value=str(PROJECT_ROOT / "source_data" / "grid_to_district_mapping.pkl"),
    )

    # --- Temporal ---
    st.sidebar.subheader("🕐 Temporal Settings")
    period_type = st.sidebar.selectbox(
        "Aggregation Period", ["hourly", "daily", "all"], index=0,
        help="How to aggregate 5-min time buckets. 'hourly' = 24 periods/day, 'daily' = 1/day, 'all' = single aggregate.",
    )
    dataset_option = st.sidebar.selectbox(
        "Dataset Period",
        ["July (21 days)", "August (23 days)", "September (22 days)", "All (66 days)"],
        index=3,
        help="Which months of weekday data to include. More days = more data but slower.",
    )
    num_days_map = {
        "July (21 days)": WEEKDAYS_JULY,
        "August (23 days)": WEEKDAYS_AUGUST,
        "September (22 days)": WEEKDAYS_SEPTEMBER,
        "All (66 days)": WEEKDAYS_TOTAL,
    }
    num_days = num_days_map[dataset_option]

    # Day-of-week filter
    days_option = st.sidebar.selectbox(
        "Days to Include",
        ["All Weekdays", "Mon-Wed", "Thu-Fri", "Monday Only", "Friday Only"],
        index=0,
        help="Filter by day-of-week. Useful to check if patterns differ Mon-Wed vs Thu-Fri.",
    )
    days_options = {
        "All Weekdays": None,
        "Mon-Wed": [1, 2, 3],
        "Thu-Fri": [4, 5],
        "Monday Only": [1],
        "Friday Only": [5],
    }
    days_filter = days_options[days_option]

    # --- Baseline g(D) ---
    st.sidebar.subheader("📈 Baseline g(D)")
    estimation_method = st.sidebar.selectbox(
        "g(D) Method", ["binning", "polynomial", "isotonic", "lowess", "linear"],
        index=0,
        help="Method for estimating g(D) = E[Y|D]. Binning is most robust. Polynomial can overfit with high degree.",
    )
    n_bins = st.sidebar.slider("N Bins (binning)", 3, 30, 10,
        help="Number of demand bins for binning method. More bins = finer resolution but noisier.",
    )
    gd_poly_degree = st.sidebar.slider("Poly Degree (g(D))", 1, 5, 2,
        help="Polynomial degree for baseline. Degree 2 is usually sufficient.",
    )

    # --- Data Filtering ---
    st.sidebar.subheader("🔍 Data Filtering")
    min_demand = st.sidebar.number_input("Min Demand", 0, 100, 1,
        help="Minimum pickup count to include a cell. Low-demand cells have noisy ratios.",
    )
    include_zero_supply = st.sidebar.checkbox("Include Zero Supply", value=True,
        help="Include cells with no taxis available? These have ratio=0, which can skew distributions.",
    )
    max_ratio_val = st.sidebar.number_input(
        "Max Ratio (0=no cap)", 0.0, 1000.0, 0.0, step=10.0,
        help="Cap service ratio at this value. Set to 0 for no cap. Removes extreme outliers.",
    )
    max_ratio = max_ratio_val if max_ratio_val > 0 else None

    # --- Feature Selection (checkboxes) ---
    st.sidebar.subheader("🧬 Feature Selection")

    # We'll populate these after loading data
    # Placeholder — actual feature checkboxes are built after enrichment

    # --- Model Settings ---
    st.sidebar.subheader("🔧 Model Settings")
    model_poly_degree = st.sidebar.slider("Demand Poly Degree (g(D,x))", 1, 5, 2,
        help="Polynomial demand features in g(D,x). Higher = more flexible demand modeling.",
    )
    alpha = st.sidebar.slider(
        "Regularization Alpha", 0.001, 100.0, 1.0,
        help="Regularization strength for Ridge/Lasso/ElasticNet. Higher = simpler model. Start ~1.0.",
    )
    l1_ratio = st.sidebar.slider("L1 Ratio (ElasticNet)", 0.0, 1.0, 0.5, step=0.1,
        help="ElasticNet L1/L2 mix. 0.0 = pure Ridge, 1.0 = pure Lasso, 0.5 = balanced.",
    )
    n_estimators = st.sidebar.slider("N Estimators (Trees)", 10, 500, 100, step=10,
        help="Number of trees for RF/GB. More = better but slower. 100 is a good start.",
    )
    max_depth = st.sidebar.slider("Max Depth (Trees)", 2, 20, 5,
        help="Max tree depth. With only 10 districts, keep low (3-5) to avoid overfitting.",
    )

    # =========================================================================
    # DATA LOADING (cached — only reruns when data/temporal/filter settings change)
    # =========================================================================

    # Validate files exist
    for label, path in [("Demand", demand_path), ("Demographics", demo_path), ("District Mapping", district_path)]:
        if not Path(path).exists():
            st.error(f"{label} file not found: {path}")
            st.stop()

    days_filter_tuple = tuple(days_filter) if days_filter else None
    try:
        (
            df, breakdown, enriched_grid, enriched_names,
            raw_feature_names, district_data,
        ) = cached_compute_baseline(
            demand_path, supply_path, demo_path, district_path,
            period_type, num_days, days_filter_tuple,
            estimation_method, n_bins, gd_poly_degree,
            min_demand, include_zero_supply, max_ratio_val,
        )
    except Exception as e:
        st.error(f"Computation failed: {e}")
        st.exception(e)
        st.stop()

    if len(df) == 0:
        st.error("No data matched between demand observations and demographic grid.")
        st.stop()

    # Reconstruct g_func from cached data (fast, ~ms)
    cached_demands = np.array(breakdown['components']['demands'])
    cached_ratios = np.array(breakdown['components']['ratios'])
    g_func, _ = estimate_g_function(
        cached_demands, cached_ratios,
        method=estimation_method, n_bins=n_bins, poly_degree=gd_poly_degree,
    )

    # Feature selection checkboxes (session-state backed)
    derived_names = [n for n in enriched_names if n not in raw_feature_names]
    all_feature_names = list(raw_feature_names) + derived_names

    # Process deferred "Apply Recommended" (must run BEFORE widget instantiation)
    if '_apply_recommended' in st.session_state:
        rec = st.session_state.pop('_apply_recommended')
        for fname in all_feature_names:
            st.session_state[f"feat_{fname}"] = (fname in rec)
        st.session_state['feature_states_initialized'] = True

    # Initialize defaults once
    if 'feature_states_initialized' not in st.session_state:
        raw_defaults = {'AvgHousingPricePerSqM'}
        derived_defaults = {'GDPperCapita', 'CompPerCapita'}
        for fname in raw_feature_names:
            st.session_state[f"feat_{fname}"] = (fname in raw_defaults)
        for fname in derived_names:
            st.session_state[f"feat_{fname}"] = (fname in derived_defaults)
        st.session_state['feature_states_initialized'] = True

    with st.sidebar.container():
        st.sidebar.markdown("**Raw Features**")
        selected_features = []
        for fname in raw_feature_names:
            if st.sidebar.checkbox(fname, key=f"feat_{fname}"):
                selected_features.append(fname)

        if derived_names:
            st.sidebar.markdown("**Derived Features**")
            for fname in derived_names:
                if st.sidebar.checkbox(fname, key=f"feat_{fname}"):
                    selected_features.append(fname)

    # Stale results warning
    if 'model_features' in st.session_state:
        if set(selected_features) != set(st.session_state['model_features']):
            st.warning(
                "⚠️ Features changed since last model fit. "
                "Click **Fit Models** in the Model Training tab to update results."
            )

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(df):,}",
              help="Total (cell, time-period) observations after filtering.")
    c2.metric("Districts", df['district_name'].nunique(),
              help="Unique Shenzhen districts with demographic data.")
    c3.metric("g(D) R²", f"{breakdown['value']:.4f}",
              help="R² of demand-only model g(D). Higher = more variance explained by demand alone.")
    c4.metric("Selected Features", len(selected_features),
              help="Demographic features checked in the sidebar for model training.")

    # =========================================================================
    # TABS
    # =========================================================================

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧪 1. Feature Analysis",
        "🔬 2. Model Training",
        "📋 3. Diagnostics",
        "✅ 4. Cross-Validation",
        "🗺️ 5. Spatial Maps",
        "📊 6. District Summaries",
    ])

    with tab1:
        render_feature_analysis_tab(
            df, enriched_names, raw_feature_names, selected_features,
        )

    with tab2:
        render_model_comparison_tab(
            df, selected_features,
            poly_degree=model_poly_degree,
            alpha=alpha, l1_ratio=l1_ratio,
            n_estimators=n_estimators, max_depth=max_depth,
        )

    with tab3:
        render_diagnostics_tab(df)

    with tab4:
        render_cv_detail_tab(
            df, enriched_grid, enriched_names,
            district_data['valid_mask'],
            district_data['district_names'],
            g_func,
            estimation_method=estimation_method,
            n_bins=n_bins,
            gd_poly_degree=gd_poly_degree,
            model_poly_degree=model_poly_degree,
            alpha=alpha, l1_ratio=l1_ratio,
            n_estimators=n_estimators, max_depth=max_depth,
            min_demand=min_demand,
            include_zero_supply=include_zero_supply,
            max_ratio_val=max_ratio_val,
            period_type=period_type,
        )

    with tab5:
        render_spatial_maps_tab(
            df, enriched_grid, enriched_names,
            district_data['valid_mask'],
            district_data['district_id_grid'],
            district_data['district_names'],
        )

    with tab6:
        render_district_summaries_tab(df, g_func, district_data['district_names'])

    # Sidebar export hint
    st.sidebar.divider()
    st.sidebar.subheader("💾 Export")
    if 'model_results' in st.session_state:
        st.sidebar.success("Models fitted — export available in **Tab 4: Cross-Validation** (section 6).")
    else:
        st.sidebar.info("Fit models first (Tab 2) to enable export.")

    # Footer
    st.divider()
    st.caption(
        "Demographic Explorer v2.0.0 | "
        "See DEMOGRAPHIC_EXPLORER_GUIDE.md for usage instructions"
    )


if __name__ == "__main__":
    main()
