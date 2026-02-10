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
    """Display model comparison results."""
    # Build summary table
    summary_rows = []
    for label, r in results.items():
        mr = r['model_result']
        lr = r['lodo_result']
        diag = r.get('diagnostics')
        row = {
            'Model': label,
            'Train R²': f"{mr['r2_train']:.4f}",
            'LODO R²': f"{lr['lodo_r2']:.4f}",
            'Overfit Gap': f"{mr['r2_train'] - lr['lodo_r2']:.4f}",
            'N_params': mr['n_params'],
            'AIC': f"{diag['aic']:.0f}" if diag else '—',
            'BIC': f"{diag['bic']:.0f}" if diag else '—',
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Sort by LODO R² descending
    summary_df['_lodo_sort'] = [float(r['lodo_result']['lodo_r2']) for r in results.values()]
    summary_df = summary_df.sort_values('_lodo_sort', ascending=False).drop(columns='_lodo_sort')

    st.subheader("Comparison Table")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Best model banner
    best_label = max(results, key=lambda k: results[k]['lodo_result']['lodo_r2'])
    best_lodo = results[best_label]['lodo_result']['lodo_r2']
    st.success(f"Best LODO R²: **{best_label}** with R² = {best_lodo:.4f}")

    # Overfitting warnings
    for label, r in results.items():
        gap = r['model_result']['r2_train'] - r['lodo_result']['lodo_r2']
        if gap > 0.1:
            st.warning(
                f"**{label}**: Train R² ({r['model_result']['r2_train']:.4f}) >> "
                f"LODO R² ({r['lodo_result']['lodo_r2']:.4f}). Gap = {gap:.4f}. "
                f"Possible overfitting."
            )

    # Per-model expanders
    st.subheader("Model Details")
    for label, r in results.items():
        with st.expander(f"{label} — LODO R² = {r['lodo_result']['lodo_r2']:.4f}"):
            mr = r['model_result']
            lr = r['lodo_result']

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Train R²", f"{mr['r2_train']:.4f}")
            c2.metric("LODO R²", f"{lr['lodo_r2']:.4f}")
            c3.metric("N_params", mr['n_params'])
            if r.get('diagnostics'):
                c4.metric("AIC", f"{r['diagnostics']['aic']:.0f}")

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

    # 1. Coefficients table
    st.subheader("1. Coefficient Significance")
    coef_display = diag['coefficients_table'].copy()
    coef_display['p_value'] = coef_display['p_value'].apply(lambda p: f"{p:.4e}")
    coef_display['Sig'] = coef_display['significant_05'].map({True: '*', False: ''})
    st.dataframe(
        coef_display[['Feature', 'Coefficient', 'StdErr', 't_stat', 'p_value', 'Sig']],
        use_container_width=True, hide_index=True,
    )

    # 2. VIF table
    st.subheader("2. Variance Inflation Factors (VIF)")
    vif_df = diag['vif'].copy()
    vif_df['Status'] = vif_df['VIF'].apply(
        lambda v: 'High collinearity' if v > 10 else ('Moderate' if v > 5 else 'OK')
    )
    st.dataframe(vif_df, use_container_width=True, hide_index=True)
    high_vif = vif_df[vif_df['VIF'] > 10]
    if len(high_vif) > 0:
        st.warning(f"Features with VIF > 10: {', '.join(high_vif['Feature'].tolist())}. Consider removing.")

    # 3. Information criteria
    st.subheader("3. Model Fit Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AIC", f"{diag['aic']:.0f}")
    c2.metric("BIC", f"{diag['bic']:.0f}")
    c3.metric("R²", f"{diag['r_squared']:.4f}")
    c4.metric("Adj R²", f"{diag['r_squared_adj']:.4f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Condition Number", f"{diag['condition_number']:.0f}",
              help="Values > 1000 suggest multicollinearity")
    c2.metric("Durbin-Watson", f"{diag['durbin_watson']:.4f}",
              help="~2.0 = no autocorrelation; <1.5 or >2.5 = concern")
    bp_str = f"{diag['breusch_pagan_p']:.4e}"
    c3.metric("Breusch-Pagan p", bp_str,
              help="p < 0.05 = heteroscedasticity detected")

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


# =============================================================================
# TAB 5: FEATURE ENGINEERING
# =============================================================================

def render_feature_engineering_tab(
    df: pd.DataFrame,
    enriched_names: List[str],
    raw_feature_names: List[str],
    selected_features: List[str],
):
    """Render feature analysis and engineering tab."""
    st.header("Feature Engineering")

    avail_features = [f for f in enriched_names if f in df.columns]

    # 1. Feature summary table
    st.subheader("1. Feature Summary")

    derived_formulas = {
        'GDPperCapita': 'GDPin10000Yuan / (Pop * 10000)',
        'CompPerCapita': 'Compensation * 1e8 / AvgEmployed',
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
            'Formula': derived_formulas.get(fname, '—'),
            'Mean': f"{vals.mean():.4g}" if len(vals) > 0 else '—',
            'Std': f"{vals.std():.4g}" if len(vals) > 0 else '—',
            'Min': f"{vals.min():.4g}" if len(vals) > 0 else '—',
            'Max': f"{vals.max():.4g}" if len(vals) > 0 else '—',
            'Unique (district-level)': len(vals.unique()),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # 2. Correlation matrix
    sel_avail = [f for f in selected_features if f in df.columns]
    if len(sel_avail) >= 2:
        st.subheader("2. Feature Correlation Matrix")
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
            st.warning("High correlations detected (|r| > 0.85):")
            for f1, f2, c in high_corr:
                st.markdown(f"- **{f1}** & **{f2}**: r = {c:.3f}")

    # 3. Feature-residual scatter grid
    if sel_avail and 'residual' in df.columns:
        st.subheader("3. Feature vs Residual (by District)")
        n_cols = min(3, len(sel_avail))
        cols = st.columns(n_cols)
        for i, fname in enumerate(sel_avail[:9]):  # Max 9 features displayed
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

    # 4. Distribution comparison (raw vs log)
    log_pairs = [
        ('GDPin10000Yuan', 'LogGDP'),
        ('AvgHousingPricePerSqM', 'LogHousingPrice'),
        ('EmployeeCompensation100MYuan', 'LogCompensation'),
        ('PopDensityPerKm2', 'LogPopDensity'),
    ]
    available_pairs = [(r, l) for r, l in log_pairs if r in df.columns and l in df.columns]
    if available_pairs:
        st.subheader("4. Raw vs Log-Transformed Distributions")
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
# TAB 6: CROSS-VALIDATION DETAIL
# =============================================================================

def render_cv_detail_tab(
    df: pd.DataFrame,
    enriched_grid: np.ndarray,
    enriched_names: List[str],
    valid_mask: np.ndarray,
    district_names: List[str],
    g_func,
):
    """Render detailed cross-validation analysis."""
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
    c1.metric("Overall LODO R²", f"{lr['lodo_r2']:.4f}")
    c2.metric("Train R²", f"{mr['r2_train']:.4f}")
    c3.metric("Overfit Gap", f"{mr['r2_train'] - lr['lodo_r2']:.4f}")

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
    )
    dataset_option = st.sidebar.selectbox(
        "Dataset Period",
        ["July (21 days)", "August (23 days)", "September (22 days)", "All (66 days)"],
        index=3,
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
    )
    n_bins = st.sidebar.slider("N Bins (binning)", 3, 30, 10)
    gd_poly_degree = st.sidebar.slider("Poly Degree (g(D))", 1, 5, 2)

    # --- Data Filtering ---
    st.sidebar.subheader("🔍 Data Filtering")
    min_demand = st.sidebar.number_input("Min Demand", 0, 100, 1)
    include_zero_supply = st.sidebar.checkbox("Include Zero Supply", value=True)
    max_ratio_val = st.sidebar.number_input(
        "Max Ratio (0=no cap)", 0.0, 1000.0, 0.0, step=10.0,
    )
    max_ratio = max_ratio_val if max_ratio_val > 0 else None

    # --- Feature Selection (checkboxes) ---
    st.sidebar.subheader("🧬 Feature Selection")

    # We'll populate these after loading data
    # Placeholder — actual feature checkboxes are built after enrichment

    # --- Model Settings ---
    st.sidebar.subheader("🔧 Model Settings")
    model_poly_degree = st.sidebar.slider("Demand Poly Degree (g(D,x))", 1, 5, 2)
    alpha = st.sidebar.slider(
        "Regularization Alpha", 0.001, 100.0, 1.0,
        help="For Ridge/Lasso/ElasticNet",
    )
    l1_ratio = st.sidebar.slider("L1 Ratio (ElasticNet)", 0.0, 1.0, 0.5, step=0.1)
    n_estimators = st.sidebar.slider("N Estimators (Trees)", 10, 500, 100, step=10)
    max_depth = st.sidebar.slider("Max Depth (Trees)", 2, 20, 5)

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    # Validate files exist
    for label, path in [("Demand", demand_path), ("Demographics", demo_path), ("District Mapping", district_path)]:
        if not Path(path).exists():
            st.error(f"{label} file not found: {path}")
            st.stop()

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
    enriched_grid, enriched_names = cached_enrich(
        demo_data['demographics_grid'],
        tuple(demo_data['feature_names']),
    )
    raw_feature_names = list(demo_data['feature_names'])

    # Feature selection checkboxes (now that we know enriched_names)
    # We use a container in the sidebar since we deferred this
    with st.sidebar.container():
        st.sidebar.markdown("**Raw Features**")
        raw_defaults = {'AvgHousingPricePerSqM'}
        selected_features = []
        for fname in raw_feature_names:
            if st.sidebar.checkbox(fname, value=(fname in raw_defaults), key=f"feat_{fname}"):
                selected_features.append(fname)

        derived_names = [n for n in enriched_names if n not in raw_feature_names]
        if derived_names:
            st.sidebar.markdown("**Derived Features**")
            derived_defaults = {'GDPperCapita', 'CompPerCapita'}
            for fname in derived_names:
                if st.sidebar.checkbox(fname, value=(fname in derived_defaults), key=f"feat_{fname}"):
                    selected_features.append(fname)

    # Configure and compute causal fairness term
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

    with st.spinner("Computing causal fairness baseline..."):
        try:
            breakdown = term.compute_with_breakdown({}, auxiliary_data)
        except Exception as e:
            st.error(f"Computation failed: {e}")
            st.exception(e)
            st.stop()

    components = breakdown['components']
    demands = np.array(components['demands'])
    ratios = np.array(components['ratios'])
    expected = np.array(components['expected'])
    keys = components['keys']

    # Build master DataFrame with enriched features
    df = prepare_demographic_analysis_data(
        demands=demands, ratios=ratios, expected=expected, keys=keys,
        demo_grid=enriched_grid, feature_names=enriched_names,
        district_id_grid=district_data['district_id_grid'],
        valid_mask=district_data['valid_mask'],
        district_names=district_data['district_names'],
        data_is_one_indexed=True,
    )

    if len(df) == 0:
        st.error("No data matched between demand observations and demographic grid.")
        st.stop()

    # Get g(D) function for overlays
    g_func = term.get_g_function()

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(df):,}")
    c2.metric("Districts", df['district_name'].nunique())
    c3.metric("g(D) R²", f"{breakdown['value']:.4f}")
    c4.metric("Selected Features", len(selected_features))

    # =========================================================================
    # TABS
    # =========================================================================

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ Spatial Maps",
        "📊 District Summaries",
        "🔬 Model Comparison",
        "📋 Statistical Diagnostics",
        "🧪 Feature Engineering",
        "✅ Cross-Validation Detail",
    ])

    with tab1:
        render_spatial_maps_tab(
            df, enriched_grid, enriched_names,
            district_data['valid_mask'],
            district_data['district_id_grid'],
            district_data['district_names'],
        )

    with tab2:
        render_district_summaries_tab(df, g_func, district_data['district_names'])

    with tab3:
        render_model_comparison_tab(
            df, selected_features,
            poly_degree=model_poly_degree,
            alpha=alpha, l1_ratio=l1_ratio,
            n_estimators=n_estimators, max_depth=max_depth,
        )

    with tab4:
        render_diagnostics_tab(df)

    with tab5:
        render_feature_engineering_tab(df, enriched_names, raw_feature_names, selected_features)

    with tab6:
        render_cv_detail_tab(
            df, enriched_grid, enriched_names,
            district_data['valid_mask'],
            district_data['district_names'],
            g_func,
        )

    # Footer
    st.divider()
    st.caption(
        "Demographic Explorer v1.0.0 | "
        "See DEMOGRAPHIC_EXPLORER_GUIDE.md for usage instructions"
    )


if __name__ == "__main__":
    main()
