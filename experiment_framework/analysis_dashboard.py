"""
FAMAIL Experiment Analysis Dashboard.

Interactive Streamlit dashboard for loading, visualizing, and comparing
experiment results from the FAMAIL Experiment & Analysis Framework.

Run:
    streamlit run experiment_framework/analysis_dashboard.py
    python -m experiment_framework dashboard --results-dir experiment_results/
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_framework.experiment_result import ExperimentResult

st.set_page_config(
    page_title="FAMAIL Experiment Analysis",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────
# Sidebar: Load experiment results
# ─────────────────────────────────────────────────────────────

def _discover_runs(results_dir: str) -> list:
    """Find all run directories with results.json."""
    rd = Path(results_dir)
    if not rd.exists():
        return []
    return sorted(
        [d for d in rd.iterdir() if d.is_dir() and (d / 'results.json').exists()],
        reverse=True,
    )


st.sidebar.title("Experiment Analysis")

# Results directory
default_dir = sys.argv[-1] if len(sys.argv) > 2 and sys.argv[-2] == '--results-dir' else 'experiment_results'
results_dir = st.sidebar.text_input("Results directory", value=default_dir)
runs = _discover_runs(results_dir)

if not runs:
    st.warning(f"No experiment results found in `{results_dir}`. Run an experiment first.")
    st.info("```\npython -m experiment_framework run --top-k 10 --max-iterations 50\n```")
    st.stop()

# Select runs to display
run_names = [r.name for r in runs]
selected_run = st.sidebar.selectbox("Primary run", run_names)

# Optional comparison run
compare_run = st.sidebar.selectbox("Compare with (optional)", ["None"] + run_names)

# Load results
result = ExperimentResult.load(str(Path(results_dir) / selected_run))
compare_result = None
if compare_run != "None":
    compare_result = ExperimentResult.load(str(Path(results_dir) / compare_run))


# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────

tab_overview, tab_global, tab_trajectory, tab_gradient, tab_compare, tab_attribution, tab_heatmap = (
    st.tabs([
        "Overview",
        "Global Evolution",
        "Per-Trajectory",
        "Gradient Decomposition",
        "Cross-Run Comparison",
        "Attribution Effectiveness",
        "Spatial Heatmap",
    ])
)


# ─────────────────────────────────────────────────────────────
# Tab 1: Experiment Overview
# ─────────────────────────────────────────────────────────────

with tab_overview:
    st.header(f"Experiment: {result.config.get('experiment_name', 'unknown')}")
    if result.config.get('description'):
        st.markdown(f"> {result.config['description']}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Duration", f"{result.duration_seconds:.1f}s")
    with col2:
        st.metric("Trajectories Modified", len(result.trajectory_results))
    with col3:
        n_conv = sum(1 for tr in result.trajectory_results if tr.converged)
        st.metric("Converged", f"{n_conv}/{len(result.trajectory_results)}")

    # Configuration table
    st.subheader("Configuration")
    cfg = result.config
    config_items = {
        'Top-k': cfg.get('top_k'),
        'Step size (α)': cfg.get('alpha'),
        'Max perturbation (ε)': cfg.get('epsilon'),
        'Max iterations': cfg.get('max_iterations'),
        'Weight: spatial': cfg.get('alpha_spatial'),
        'Weight: causal': cfg.get('alpha_causal'),
        'Weight: fidelity': cfg.get('alpha_fidelity'),
        'Causal formulation': cfg.get('causal_formulation'),
        'Gradient decomp.': cfg.get('record_gradient_decomposition'),
        'Discriminator': cfg.get('discriminator_checkpoint', 'N/A'),
    }
    st.table(pd.DataFrame(list(config_items.items()), columns=['Parameter', 'Value']))

    # Before/After metrics
    st.subheader("Global Fairness Metrics")
    metrics_data = []
    for key in ('gini', 'f_spatial', 'f_causal', 'f_fidelity', 'combined'):
        before = result.initial_snapshot.get(key, 0)
        after = result.final_snapshot.get(key, 0)
        delta = after - before
        metrics_data.append({
            'Metric': key,
            'Before': f'{before:.4f}',
            'After': f'{after:.4f}',
            'Delta': f'{delta:+.4f}',
        })
    st.table(pd.DataFrame(metrics_data))


# ─────────────────────────────────────────────────────────────
# Tab 2: Global Fairness Evolution
# ─────────────────────────────────────────────────────────────

with tab_global:
    st.header("Global Fairness Evolution")

    if not result.global_snapshots:
        st.info("No global snapshots recorded. Set snapshot_every_n >= 1.")
    else:
        df_snap = pd.DataFrame(result.global_snapshots)

        # Line chart
        metrics_to_plot = st.multiselect(
            "Metrics to plot",
            ['gini', 'f_spatial', 'f_causal', 'f_fidelity', 'combined'],
            default=['f_spatial', 'f_causal', 'combined'],
        )

        fig = go.Figure()
        for metric in metrics_to_plot:
            if metric in df_snap.columns:
                fig.add_trace(go.Scatter(
                    x=df_snap['after_n_trajectories'],
                    y=df_snap[metric],
                    mode='lines+markers',
                    name=metric,
                ))

        fig.update_layout(
            xaxis_title='Trajectories Modified',
            yaxis_title='Score',
            height=500,
            hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("Raw snapshot data"):
            st.dataframe(df_snap)


# ─────────────────────────────────────────────────────────────
# Tab 3: Per-Trajectory Deep Dive
# ─────────────────────────────────────────────────────────────

with tab_trajectory:
    st.header("Per-Trajectory Analysis")

    if not result.trajectory_results:
        st.info("No trajectories modified.")
    else:
        # Summary table
        traj_data = []
        for tr in result.trajectory_results:
            traj_data.append({
                'Index': tr.trajectory_index,
                'Driver': tr.driver_id,
                'Orig': f"({tr.original_pickup[0]}, {tr.original_pickup[1]})",
                'Modified': f"({tr.modified_pickup[0]}, {tr.modified_pickup[1]})",
                'Iterations': tr.total_iterations,
                'Converged': tr.converged,
                'Objective': f'{tr.final_objective:.4f}',
                'Perturbation': f'{tr.perturbation_magnitude:.2f}',
            })
        st.dataframe(pd.DataFrame(traj_data), use_container_width=True)

        # Drill-down
        traj_options = [f"Traj {tr.trajectory_index} (driver {tr.driver_id})"
                        for tr in result.trajectory_results]
        selected = st.selectbox("Select trajectory for detail view", traj_options)
        selected_idx = traj_options.index(selected)
        tr = result.trajectory_results[selected_idx]

        if tr.iterations:
            st.subheader(f"Iteration Trace: Trajectory {tr.trajectory_index}")

            iter_df = pd.DataFrame([{
                'iteration': it.iteration,
                'objective': it.objective,
                'f_spatial': it.f_spatial,
                'f_causal': it.f_causal,
                'f_fidelity': it.f_fidelity,
                'gradient_norm': it.gradient_norm,
            } for it in tr.iterations])

            # Plot term values
            fig = go.Figure()
            for col in ['objective', 'f_spatial', 'f_causal', 'f_fidelity']:
                fig.add_trace(go.Scatter(
                    x=iter_df['iteration'], y=iter_df[col],
                    mode='lines+markers', name=col,
                ))
            fig.update_layout(
                xaxis_title='Iteration', yaxis_title='Value', height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Gradient norm
            fig_grad = go.Figure()
            fig_grad.add_trace(go.Scatter(
                x=iter_df['iteration'], y=iter_df['gradient_norm'],
                mode='lines+markers', name='Gradient Norm',
            ))
            fig_grad.update_layout(
                xaxis_title='Iteration', yaxis_title='Gradient Norm', height=300,
            )
            st.plotly_chart(fig_grad, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Tab 4: Gradient Decomposition
# ─────────────────────────────────────────────────────────────

with tab_gradient:
    st.header("Gradient Decomposition Analysis")

    has_decomp = any(
        it.gradient_decomposition is not None
        for tr in result.trajectory_results
        for it in tr.iterations
    )

    if not has_decomp:
        st.info("Gradient decomposition was not recorded for this experiment. "
                "Re-run with `--gradient-decomposition` to enable.")
    else:
        # Per-trajectory gradient fraction chart
        traj_options_g = [f"Traj {tr.trajectory_index}" for tr in result.trajectory_results]
        selected_g = st.selectbox("Select trajectory", traj_options_g, key='grad_traj')
        sel_idx = traj_options_g.index(selected_g)
        tr = result.trajectory_results[sel_idx]

        decomp_data = []
        for it in tr.iterations:
            gd = it.gradient_decomposition
            if gd:
                decomp_data.append({
                    'iteration': it.iteration,
                    'spatial': gd.spatial_fraction,
                    'causal': gd.causal_fraction,
                    'fidelity': gd.fidelity_fraction,
                    'alignment': gd.alignment_spatial_causal,
                })

        if decomp_data:
            df_decomp = pd.DataFrame(decomp_data)

            # Stacked area chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_decomp['iteration'], y=df_decomp['spatial'],
                mode='lines', name='Spatial', stackgroup='one',
                fillcolor='rgba(166, 25, 46, 0.6)', line=dict(color='#A6192E'),
            ))
            fig.add_trace(go.Scatter(
                x=df_decomp['iteration'], y=df_decomp['causal'],
                mode='lines', name='Causal', stackgroup='one',
                fillcolor='rgba(0, 128, 128, 0.6)', line=dict(color='#008080'),
            ))
            fig.add_trace(go.Scatter(
                x=df_decomp['iteration'], y=df_decomp['fidelity'],
                mode='lines', name='Fidelity', stackgroup='one',
                fillcolor='rgba(100, 100, 100, 0.6)', line=dict(color='#666'),
            ))
            fig.update_layout(
                title='Gradient Contribution by Term',
                xaxis_title='Iteration', yaxis_title='Fraction',
                yaxis_range=[0, 1], height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Alignment chart
            fig_align = go.Figure()
            fig_align.add_trace(go.Scatter(
                x=df_decomp['iteration'], y=df_decomp['alignment'],
                mode='lines+markers', name='Spatial-Causal Alignment',
                line=dict(color='#008080'),
            ))
            fig_align.add_hline(y=0, line_dash='dash', line_color='gray')
            fig_align.update_layout(
                title='Spatial-Causal Gradient Alignment (cosine similarity)',
                xaxis_title='Iteration', yaxis_title='Cosine Similarity',
                yaxis_range=[-1, 1], height=300,
            )
            st.plotly_chart(fig_align, use_container_width=True)

        # Aggregate statistics across all trajectories
        st.subheader("Aggregate Gradient Statistics")
        all_spatial, all_causal, all_fidelity, all_align = [], [], [], []
        for tr in result.trajectory_results:
            for it in tr.iterations:
                gd = it.gradient_decomposition
                if gd:
                    all_spatial.append(gd.spatial_fraction)
                    all_causal.append(gd.causal_fraction)
                    all_fidelity.append(gd.fidelity_fraction)
                    all_align.append(gd.alignment_spatial_causal)

        if all_spatial:
            stats_df = pd.DataFrame({
                'Metric': ['Spatial fraction', 'Causal fraction', 'Fidelity fraction', 'Alignment'],
                'Mean': [np.mean(all_spatial), np.mean(all_causal), np.mean(all_fidelity), np.mean(all_align)],
                'Std': [np.std(all_spatial), np.std(all_causal), np.std(all_fidelity), np.std(all_align)],
            })
            st.table(stats_df.style.format({'Mean': '{:.3f}', 'Std': '{:.3f}'}))


# ─────────────────────────────────────────────────────────────
# Tab 5: Cross-Run Comparison
# ─────────────────────────────────────────────────────────────

with tab_compare:
    st.header("Cross-Run Comparison")

    if compare_result is None:
        st.info("Select a comparison run in the sidebar to enable this view.")
    else:
        st.subheader("Side-by-Side Metrics")
        comp_data = []
        for key in ('gini', 'f_spatial', 'f_causal', 'f_fidelity', 'combined'):
            r1_before = result.initial_snapshot.get(key, 0)
            r1_after = result.final_snapshot.get(key, 0)
            r2_before = compare_result.initial_snapshot.get(key, 0)
            r2_after = compare_result.final_snapshot.get(key, 0)
            comp_data.append({
                'Metric': key,
                f'{selected_run} (before)': f'{r1_before:.4f}',
                f'{selected_run} (after)': f'{r1_after:.4f}',
                f'{selected_run} (Δ)': f'{r1_after - r1_before:+.4f}',
                f'{compare_run} (before)': f'{r2_before:.4f}',
                f'{compare_run} (after)': f'{r2_after:.4f}',
                f'{compare_run} (Δ)': f'{r2_after - r2_before:+.4f}',
            })
        st.table(pd.DataFrame(comp_data))

        # Overlay global evolution curves
        if result.global_snapshots and compare_result.global_snapshots:
            st.subheader("Global Evolution Overlay")
            metric_to_compare = st.selectbox("Metric", ['f_spatial', 'f_causal', 'combined', 'gini'])

            df1 = pd.DataFrame(result.global_snapshots)
            df2 = pd.DataFrame(compare_result.global_snapshots)

            fig = go.Figure()
            if metric_to_compare in df1.columns:
                fig.add_trace(go.Scatter(
                    x=df1['after_n_trajectories'], y=df1[metric_to_compare],
                    mode='lines+markers', name=selected_run,
                ))
            if metric_to_compare in df2.columns:
                fig.add_trace(go.Scatter(
                    x=df2['after_n_trajectories'], y=df2[metric_to_compare],
                    mode='lines+markers', name=compare_run,
                ))
            fig.update_layout(
                xaxis_title='Trajectories Modified',
                yaxis_title=metric_to_compare,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Tab 6: Attribution Effectiveness
# ─────────────────────────────────────────────────────────────

with tab_attribution:
    st.header("Attribution Effectiveness")

    if not result.attribution_scores or not result.trajectory_results:
        st.info("No attribution or trajectory data available.")
    else:
        # Build comparison data: attribution score vs actual improvement
        attr_map = {a['index']: a.get('combined_score', 0) for a in result.attribution_scores if a.get('index') is not None}
        scatter_data = []
        for tr in result.trajectory_results:
            attr_score = attr_map.get(tr.trajectory_index, 0)
            scatter_data.append({
                'trajectory_index': tr.trajectory_index,
                'attribution_score': attr_score,
                'final_objective': tr.final_objective,
                'perturbation': tr.perturbation_magnitude,
            })

        df_attr = pd.DataFrame(scatter_data)

        if len(df_attr) >= 3:
            from scipy.stats import spearmanr
            rho, p_value = spearmanr(df_attr['attribution_score'], df_attr['final_objective'])
            st.metric("Spearman Correlation (attribution vs objective)", f"{rho:.3f}", f"p={p_value:.3f}")

        fig = px.scatter(
            df_attr,
            x='attribution_score',
            y='final_objective',
            hover_data=['trajectory_index'],
            title='Attribution Score vs Final Objective',
            labels={'attribution_score': 'Attribution Combined Score', 'final_objective': 'Final Objective L'},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Tab 7: Spatial Heatmap
# ─────────────────────────────────────────────────────────────

with tab_heatmap:
    st.header("Modification Spatial Overview")

    if not result.trajectory_results:
        st.info("No trajectory results.")
    else:
        # Plot all modifications as arrows on a grid
        grid_x, grid_y = result.config.get('grid_dims', (48, 90))

        fig = go.Figure()

        # Add arrow annotations for each modification
        for tr in result.trajectory_results:
            ox, oy = tr.original_pickup
            mx, my = tr.modified_pickup
            if ox != mx or oy != my:
                fig.add_annotation(
                    x=my, y=mx, ax=oy, ay=ox,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.5,
                    arrowwidth=2, arrowcolor='#A6192E',
                )

            # Original position marker
            fig.add_trace(go.Scatter(
                x=[oy], y=[ox], mode='markers',
                marker=dict(size=8, color='#008080', symbol='circle'),
                name=f'Traj {tr.trajectory_index} (orig)',
                showlegend=False,
                hovertext=f'Traj {tr.trajectory_index}: ({ox},{oy}) -> ({mx},{my})',
            ))

        fig.update_layout(
            title='Pickup Modifications (arrows show movement)',
            xaxis_title='y_grid (longitude)',
            yaxis_title='x_grid (latitude)',
            xaxis_range=[0, grid_y],
            yaxis_range=[0, grid_x],
            height=600,
            width=900,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        moved = sum(1 for tr in result.trajectory_results
                    if tr.original_pickup != tr.modified_pickup)
        st.write(f"**{moved}/{len(result.trajectory_results)}** trajectories changed cells")
