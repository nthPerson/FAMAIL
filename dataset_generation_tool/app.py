"""Streamlit UI for the Trajectory Pair Dataset Generator.

Launch with:
    streamlit run discriminator/dataset_generation_tool/app.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# from dataset_generation_tool import (
#     GenerationConfig,
#     assemble_dataset,
#     dataset_to_npz_bytes,
#     dataset_to_pt_bytes,
#     sample_json,
# )
from generation import (
    GenerationConfig,
    assemble_dataset,
    dataset_to_npz_bytes,
    dataset_to_pt_bytes,
    sample_json,
)


st.set_page_config(page_title="Trajectory Pair Dataset Generator", layout="wide")

DEFAULT_DATA_PATH = Path("/home/robert/FAMAIL/discriminator/create_traj_pair_dataset/source_data/all_trajs.pkl").resolve()


def _build_config() -> Tuple[GenerationConfig, Dict, int]:
    with st.sidebar:
        st.header("Configuration")
        data_path_str = st.text_input("Path to all_trajs.pkl", value=str(DEFAULT_DATA_PATH))
        per_agent_mode = st.checkbox(
            "Per-agent counts mode",
            value=False,
            help=(
                "When checked:\n"
                "• Positive pairs = N matching pairs **for each** of the 50 agents\n"
                "• Negative pairs = N pairs **for each** agent-to-other-agent combination\n\n"
                "Example with 50 agents:\n"
                "• 100 positive → 100 × 50 = 5,000 total matching pairs\n"
                "• 10 negative → 10 × 50 × 49 = 24,500 total non-matching pairs\n\n"
                "This ensures comprehensive coverage for discriminator training."
            ),
        )
        if per_agent_mode:
            pos_pairs = st.number_input(
                "# Positive Pairs per Agent",
                min_value=1,
                max_value=1000,
                value=100,
                help="Pairs per agent. Total = this × num_agents (e.g., 100 × 50 = 5,000).",
            )
            neg_pairs = st.number_input(
                "# Negative Pairs per Agent Combination",
                min_value=1,
                max_value=100,
                value=10,
                help="Pairs per (agent_i, agent_j) combo. Total = this × N × (N-1) (e.g., 10 × 50 × 49 = 24,500).",
            )
        else:
            pos_pairs = st.number_input(
                "# Positive Pairs (Total)",
                min_value=1,
                max_value=100000,
                value=500,
                help="Total matching pairs across all agents.",
            )
            neg_pairs = st.number_input(
                "# Negative Pairs (Total)",
                min_value=1,
                max_value=100000,
                value=500,
                help="Total non-matching pairs across all agents.",
            )
        days = st.slider("Concatenated trajectory length (days)", min_value=1, max_value=7, value=2)
        feature_start = int(
            st.number_input(
                "Feature start index (inclusive, >=4)",
                min_value=4,
                max_value=126,
                value=4,
                help="Optional: start of extra feature slice; indices 0-3 are always included.",
            )
        )
        feature_end = int(
            st.number_input(
                "Feature end index (exclusive, <=126)",
                min_value=feature_start,
                max_value=126,
                value=feature_start,
                help="Leave equal to start to include no additional features beyond 0-3.",
            )
        )
        padding = st.selectbox(
            "Padding / truncation strategy",
            options=["pad_to_longer", "truncate_to_shorter", "fixed_length"],
            index=1,
        )
        fixed_length = None
        if padding == "fixed_length":
            fixed_length = st.number_input("Fixed sequence length", min_value=1, max_value=20000, value=2000)
        pos_strategy = st.selectbox("Positive pair strategy", options=["random", "sequential"], index=0)
        neg_strategy = st.selectbox("Negative pair strategy", options=["random", "round_robin"], index=0)
        distribution = st.selectbox("Agent sampling distribution", options=["proportional", "uniform"], index=0)
        seed_text = st.text_input("Random seed (leave blank for random)", value="42")
        ensure_coverage = st.checkbox("Ensure every agent appears", value=True, disabled=per_agent_mode)
        preview_cap = st.slider("Preview pair cap", min_value=4, max_value=40, value=12)
        
        # Show estimated totals when in per-agent mode
        if per_agent_mode:
            st.divider()
            st.markdown("**Estimated Totals** (assuming 50 agents):")
            est_pos = int(pos_pairs) * 50
            est_neg = int(neg_pairs) * 50 * 49
            st.markdown(f"- Positive pairs: {int(pos_pairs)} × 50 = **{est_pos:,}**")
            st.markdown(f"- Negative pairs: {int(neg_pairs)} × 50 × 49 = **{est_neg:,}**")
            st.markdown(f"- Total pairs: **{est_pos + est_neg:,}**")
    seed = int(seed_text) if seed_text.strip() else None
    cfg = GenerationConfig(
        data_path=Path(data_path_str),
        positive_pairs=int(pos_pairs),
        negative_pairs=int(neg_pairs),
        days=int(days),
        feature_start=int(feature_start),
        feature_end=int(feature_end),
        padding=padding,
        fixed_length=int(fixed_length) if fixed_length else None,
        positive_strategy=pos_strategy,
        negative_strategy=neg_strategy,
        agent_distribution=distribution,
        seed=seed,
        ensure_agent_coverage=ensure_coverage if not per_agent_mode else True,  # Always ensure coverage in per-agent mode
        per_agent_counts=per_agent_mode,
    )
    cache_key = {
        "data_path": str(cfg.data_path),
        "positive_pairs": cfg.positive_pairs,
        "negative_pairs": cfg.negative_pairs,
        "days": cfg.days,
        "feature_start": cfg.feature_start,
        "feature_end": cfg.feature_end,
        "padding": cfg.padding,
        "fixed_length": cfg.fixed_length,
        "positive_strategy": cfg.positive_strategy,
        "negative_strategy": cfg.negative_strategy,
        "agent_distribution": cfg.agent_distribution,
        "seed": cfg.seed,
        "ensure_agent_coverage": cfg.ensure_agent_coverage,
        "per_agent_counts": cfg.per_agent_counts,
    }
    return cfg, cache_key, int(preview_cap)


@st.cache_data(show_spinner=False)
def _generate_cached(config_dict: Dict, preview_only: bool, preview_cap: int):
    cfg = GenerationConfig(**{**config_dict, "data_path": Path(config_dict["data_path"])})
    return assemble_dataset(cfg, preview_only=preview_only, preview_cap=preview_cap)


def _render_preview(dataset: Dict[str, np.ndarray], metadata: Dict):
    st.subheader("Preview")
    st.write(
        {
            "x1": dataset["x1"].shape,
            "x2": dataset["x2"].shape,
            "label": dataset["label"].shape,
            "mask1": dataset["mask1"].shape,
        }
    )
    labels = dataset["label"]
    st.metric("Positives", int((labels == 0).sum()))
    st.metric("Negatives", int((labels == 1).sum()))
    lengths_table = []
    for i in range(min(8, labels.shape[0])):
        lengths_table.append(
            {
                "idx": i,
                "label": int(labels[i]),
                "len_x1": int(dataset["mask1"][i].sum()),
                "len_x2": int(dataset["mask2"][i].sum()),
            }
        )
    st.dataframe(lengths_table, hide_index=True)
    _render_pca(dataset)
    st.subheader("Metadata")
    st.json(metadata)


def _render_pca(dataset: Dict[str, np.ndarray]):
    try:
        from sklearn.decomposition import PCA
    except ImportError:  # pragma: no cover
        st.warning("Install scikit-learn to view PCA projection.")
        return
    labels = dataset["label"]
    max_points = min(120, labels.shape[0])
    x1 = dataset["x1"][:max_points]
    mask1 = dataset["mask1"][:max_points]
    flattened = []
    for arr, mask in zip(x1, mask1):
        effective = arr.copy()
        effective[mask == 0] = 0.0
        flattened.append(effective.flatten())
    flat = np.stack(flattened, axis=0)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(flat)
    import pandas as pd

    df = pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "label": labels[:max_points].astype(str)})
    st.subheader("PCA Projection (sampled trajectories)")
    chart = (
        alt.Chart(df)
        .mark_circle(size=80, opacity=0.8)
        .encode(
            x="pc1:Q",
            y="pc2:Q",
            color=alt.Color("label:N", scale=alt.Scale(domain=["0", "1"], range=["#1f77b4", "#d62728"]), legend=alt.Legend(title="label")),
            tooltip=["label", "pc1", "pc2"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def _pair_pca(x1: np.ndarray, x2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> pd.DataFrame:
    try:
        from sklearn.decomposition import PCA
    except ImportError:  # pragma: no cover
        return pd.DataFrame()
    samples = []
    for arr, mask, name in [(x1, mask1, "x1"), (x2, mask2, "x2")]:
        effective = arr.copy()
        effective[mask == 0] = 0.0
        samples.append(effective.flatten())
    if not samples:
        return pd.DataFrame()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(np.stack(samples))
    return pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "which": ["x1", "x2"]})


def _segment_dataframe(component_lengths: List[int], traj_indices: List[int], raw_len: int, align_len: int, global_len: int, label_prefix: str) -> pd.DataFrame:
    rows = []
    pos = 0
    for idx, length in zip(traj_indices, component_lengths):
        start = pos
        end = pos + length
        rows.append({"start": start, "end": end, "kind": f"segment_{label_prefix}", "traj_idx": idx, "length": length})
        pos = end
    if raw_len > align_len:
        rows.append({"start": align_len, "end": raw_len, "kind": f"truncated_{label_prefix}", "traj_idx": None, "length": raw_len - align_len})
    if raw_len < align_len:
        rows.append({"start": raw_len, "end": align_len, "kind": f"padded_{label_prefix}", "traj_idx": None, "length": align_len - raw_len})
    if align_len < global_len:
        rows.append({"start": align_len, "end": global_len, "kind": f"global_pad_{label_prefix}", "traj_idx": None, "length": global_len - align_len})
    return pd.DataFrame(rows)


def _render_pair_explorer(dataset: Dict[str, np.ndarray], pair_info: List[Dict[str, Any]], metadata: Dict[str, Any]):
    if not pair_info:
        return
    st.subheader("Sample Pair Explorer")
    max_idx = len(pair_info) - 1
    pair_idx = st.number_input("Pair index", min_value=0, max_value=max_idx, value=0, step=1)
    info = pair_info[int(pair_idx)]
    st.write({
        "label": info.get("label"),
        "agent_a": info.get("agent_a"),
        "agent_b": info.get("agent_b"),
        "len_raw_a": info.get("len_raw_a"),
        "len_raw_b": info.get("len_raw_b"),
        "align_len": info.get("align_len"),
        "padding_mode": metadata.get("config", {}).get("padding"),
    })

    x1 = dataset["x1"][pair_idx]
    x2 = dataset["x2"][pair_idx]
    m1 = dataset["mask1"][pair_idx]
    m2 = dataset["mask2"][pair_idx]

    pair_df = _pair_pca(x1, x2, m1, m2)
    if not pair_df.empty:
        st.markdown("**Per-pair PCA (x1 vs x2)**")
        # Fixed zoom per request: widen horizontal, tighten vertical
        x_domain = [-2200.0, 2200.0]
        # x_domain = [-1200.0, 1200.0]
        y_domain = [-0.005, 0.005]
        chart = (
            alt.Chart(pair_df)
            .mark_circle(size=160, opacity=0.85)
            .encode(
                x=alt.X("pc1:Q", scale=alt.Scale(domain=x_domain)),
                y=alt.Y("pc2:Q", scale=alt.Scale(domain=y_domain)),
                color=alt.Color("which:N", scale=alt.Scale(range=["#1f77b4", "#d62728"])),
                tooltip=["which", "pc1", "pc2"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Install scikit-learn to view per-pair PCA.")

    st.markdown("**Concatenation breakdown**")
    global_len = int(dataset["x1"].shape[1])
    df_a = _segment_dataframe(
        info.get("component_lengths_a", []),
        info.get("traj_indices_a", []),
        info.get("len_raw_a", 0),
        info.get("align_len", 0),
        global_len,
        "a",
    )
    df_b = _segment_dataframe(
        info.get("component_lengths_b", []),
        info.get("traj_indices_b", []),
        info.get("len_raw_b", 0),
        info.get("align_len", 0),
        global_len,
        "b",
    )
    df_a["which"] = "x1"
    df_b["which"] = "x2"
    df = pd.concat([df_a, df_b], ignore_index=True)
    tooltip = ["which", "kind", "traj_idx", "length", "start", "end"]
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="start:Q",
            x2="end:Q",
            y="which:N",
            color="kind:N",
            tooltip=tooltip,
        )
        .properties(height=160)
    )
    st.altair_chart(chart, use_container_width=True)


def main():
    st.title("Trajectory Pair Dataset Generator")
    cfg, cache_key, preview_cap = _build_config()
    
    # Dataset size summary
    if cfg.per_agent_counts:
        num_agents = 50  # Expected number of agents
        pos_per_agent = cfg.positive_pairs
        neg_per_combo = cfg.negative_pairs
        total_pos = pos_per_agent * num_agents
        total_neg = neg_per_combo * num_agents * (num_agents - 1)
        total_samples = total_pos + total_neg
        
        st.subheader("📊 Dataset Size Summary")
        col_pos, col_neg, col_total = st.columns(3)
        with col_pos:
            st.metric("Total Positive Pairs", f"{total_pos:,}")
            st.caption(f"{pos_per_agent} per agent × {num_agents} agents")
        with col_neg:
            st.metric("Total Negative Pairs", f"{total_neg:,}")
            st.caption(f"{neg_per_combo} per combo × {num_agents} × {num_agents - 1}")
        with col_total:
            st.metric("Total Pairs", f"{total_samples:,}")
            pos_pct = (total_pos / total_samples * 100) if total_samples > 0 else 0
            st.caption(f"{pos_pct:.1f}% positive / {100 - pos_pct:.1f}% negative")
        
        with st.expander("ℹ️ Coverage Details", expanded=False):
            st.markdown(f"""
**Per-Agent Counts Mode** ensures comprehensive coverage for discriminator training:

| Metric | Formula | Result |
|--------|---------|--------|
| Positive pairs | {pos_per_agent} × {num_agents} agents | **{total_pos:,}** |
| Negative pairs | {neg_per_combo} × {num_agents} × {num_agents - 1} combos | **{total_neg:,}** |
| **Total** | {total_pos:,} + {total_neg:,} | **{total_samples:,}** |

✅ Every agent will have **{pos_per_agent}** same-agent (matching) pairs  
✅ Every agent pair (A→B) will have **{neg_per_combo}** different-agent (non-matching) pairs  
✅ All **{num_agents}** agents will be fully represented in both positive and negative samples
            """)
        st.divider()
    else:
        # Standard mode - show simple totals
        total_pos = cfg.positive_pairs
        total_neg = cfg.negative_pairs
        total_samples = total_pos + total_neg
        
        st.subheader("📊 Dataset Size Summary")
        col_pos, col_neg, col_total = st.columns(3)
        with col_pos:
            st.metric("Positive Pairs", f"{total_pos:,}")
        with col_neg:
            st.metric("Negative Pairs", f"{total_neg:,}")
        with col_total:
            st.metric("Total Pairs", f"{total_samples:,}")
            pos_pct = (total_pos / total_samples * 100) if total_samples > 0 else 0
            st.caption(f"{pos_pct:.1f}% positive / {100 - pos_pct:.1f}% negative")
        st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        preview_clicked = st.button("Generate Preview", type="primary")
    with col2:
        full_clicked = st.button("Generate Full Dataset for Download")

    if preview_clicked:
        try:
            with st.spinner("Generating preview..."):
                dataset, metadata, pair_info = _generate_cached(cache_key, preview_only=True, preview_cap=preview_cap)
            st.session_state["preview_dataset"] = dataset
            st.session_state["preview_metadata"] = metadata
            st.session_state["preview_pair_info"] = pair_info
        except Exception as exc:  # pragma: no cover
            st.error(f"Preview failed: {exc}")

    if full_clicked:
        try:
            with st.spinner("Generating full dataset..."):
                dataset, metadata, pair_info = _generate_cached(cache_key, preview_only=False, preview_cap=preview_cap)
            st.session_state["full_dataset"] = dataset
            st.session_state["full_metadata"] = metadata
            st.session_state["full_pair_info"] = pair_info
            st.success("Dataset ready. Use the download buttons below.")
        except Exception as exc:  # pragma: no cover
            st.error(f"Full generation failed: {exc}")

    # Persistent preview display
    if "preview_dataset" in st.session_state and "preview_metadata" in st.session_state:
        _render_preview(st.session_state["preview_dataset"], st.session_state["preview_metadata"])
        _render_pair_explorer(
            st.session_state["preview_dataset"],
            st.session_state.get("preview_pair_info", []),
            st.session_state["preview_metadata"],
        )

    # Persistent download buttons for full dataset
    if "full_dataset" in st.session_state and "full_metadata" in st.session_state:
        dataset = st.session_state["full_dataset"]
        metadata = st.session_state["full_metadata"]
        st.subheader("Downloads")
        npz_bytes = dataset_to_npz_bytes(dataset)
        st.download_button(
            label="Download .npz",
            data=npz_bytes,
            file_name="pairs_dataset.npz",
            mime="application/octet-stream",
            key="dl_npz",
        )
        try:
            pt_bytes = dataset_to_pt_bytes(dataset)
            st.download_button(
                label="Download .pt",
                data=pt_bytes,
                file_name="pairs_dataset.pt",
                mime="application/octet-stream",
                key="dl_pt",
            )
        except ImportError:
            st.info("PyTorch not installed; .pt export unavailable.")
        st.download_button(
            label="Download metadata (.json)",
            data=json.dumps(metadata, indent=2),
            file_name="pairs_metadata.json",
            mime="application/json",
            key="dl_meta",
        )
        st.download_button(
            label="Download sample (JSON)",
            data=sample_json(dataset, metadata, k=5),
            file_name="pairs_sample.json",
            mime="application/json",
            key="dl_sample",
        )


if __name__ == "__main__":
    main()
