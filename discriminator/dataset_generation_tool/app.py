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

from generation import (
    GenerationConfig,
    assemble_dataset,
    dataset_to_npz_bytes,
    dataset_to_pt_bytes,
    sample_json,
)


st.set_page_config(page_title="Trajectory Pair Dataset Generator", layout="wide")

DEFAULT_DATA_PATH = Path("/home/robert/FAMAIL/discriminator/dataset_generation_tool/source_data/all_trajs.pkl").resolve()


def _build_config() -> Tuple[GenerationConfig, Dict, int]:
    with st.sidebar:
        st.header("Configuration")
        data_path_str = st.text_input(
            "Path to all_trajs.pkl", 
            value=str(DEFAULT_DATA_PATH),
            help=(
                "Path to the pickle file containing trajectory data.\n\n"
                "**Expected format:** Dict mapping agent_id → List[np.ndarray]\n"
                "Each array should be shape (timesteps, features) with at least 4 features:\n"
                "- [0] x_grid: Grid x-coordinate (0-49)\n"
                "- [1] y_grid: Grid y-coordinate (0-89)\n"
                "- [2] time_bucket: Time of day (0-287, 5-min intervals)\n"
                "- [3] day_index: Day of week (1-5, Mon-Fri)"
            ),
        )
        per_agent_mode = st.checkbox(
            "Per-agent counts mode",
            value=False,
            help=(
                "**Per-Agent Mode** ensures comprehensive coverage for discriminator training.\n\n"
                "When checked:\n"
                "• Positive pairs = N matching pairs **for each** of the 50 agents\n"
                "• Negative pairs = N pairs **for each** agent-to-other-agent combination\n\n"
                "**Example with 50 agents:**\n"
                "• 100 positive → 100 × 50 = 5,000 total matching pairs\n"
                "• 10 negative → 10 × 50 × 49 = 24,500 total non-matching pairs\n\n"
                "**When unchecked:** The counts specify the total number of pairs across all agents."
            ),
        )
        if per_agent_mode:
            pos_pairs = st.number_input(
                "# Positive Pairs per Agent",
                min_value=1,
                max_value=1000,
                value=100,
                help=(
                    "Number of same-agent (positive) pairs to generate **per agent**.\n\n"
                    "**Formula:** Total positive pairs = this × num_agents\n"
                    "**Example:** 100 × 50 agents = 5,000 total positive pairs\n\n"
                    "Higher values improve per-agent representation but increase dataset size."
                ),
            )
            neg_pairs = st.number_input(
                "# Negative Pairs per Agent Combination",
                min_value=1,
                max_value=100,
                value=10,
                help=(
                    "Number of different-agent (negative) pairs to generate **per agent pair**.\n\n"
                    "**Formula:** Total negative pairs = this × N × (N-1)\n"
                    "**Example:** 10 × 50 × 49 = 24,500 total negative pairs\n\n"
                    "This ensures every agent-pair combination is well-represented."
                ),
            )
        else:
            pos_pairs = st.number_input(
                "# Positive Pairs (Total)",
                min_value=1,
                max_value=100000,
                value=500,
                help=(
                    "Total number of same-agent (positive) pairs to generate.\n\n"
                    "**Label = 1:** Pairs where both trajectories come from the same driver.\n"
                    "The discriminator should output high similarity (~1.0) for these.\n\n"
                    "**Tip:** Use 'Ensure every agent appears' to guarantee coverage."
                ),
            )
            neg_pairs = st.number_input(
                "# Negative Pairs (Total)",
                min_value=1,
                max_value=100000,
                value=500,
                help=(
                    "Total number of different-agent (negative) pairs to generate.\n\n"
                    "**Label = 0:** Pairs where trajectories come from different drivers.\n"
                    "The discriminator should output low similarity (~0.0) for these.\n\n"
                    "**Recommended:** Use 2-4× more negatives than positives for class balance."
                ),
            )
        days = st.slider(
            "Concatenated trajectory length (days)", 
            min_value=1, 
            max_value=5, 
            value=2,
            help=(
                "Number of consecutive days to concatenate into each trajectory segment.\n\n"
                "**Dataset contains Monday-Friday data only** (5 days max).\n\n"
                "• **1 day:** Single-day patterns (shorter sequences)\n"
                "• **2-3 days:** Multi-day patterns (recommended for learning habits)\n"
                "• **5 days:** Full workweek patterns (longest sequences)\n\n"
                "Longer trajectories capture more behavioral patterns but require more memory."
            ),
        )
        feature_start = int(
            st.number_input(
                "Feature start index (inclusive, >=4)",
                min_value=4,
                max_value=126,
                value=4,
                help=(
                    "Start index for additional features beyond the core 4 (x, y, time, day).\n\n"
                    "**Core features [0-3] are always included:**\n"
                    "• [0] x_grid — Grid x-coordinate\n"
                    "• [1] y_grid — Grid y-coordinate\n"
                    "• [2] time_bucket — Time of day (0-287)\n"
                    "• [3] day_index — Day of week (1-5)\n\n"
                    "Set equal to 'Feature end index' to include only core features."
                ),
            )
        )
        feature_end = int(
            st.number_input(
                "Feature end index (exclusive, <=126)",
                min_value=feature_start,
                max_value=126,
                value=feature_start,
                help=(
                    "End index (exclusive) for additional features.\n\n"
                    "**Feature slice:** Will include indices [feature_start : feature_end]\n"
                    "**Final features:** Indices 0-3 + selected slice\n\n"
                    "Leave equal to start to include **only** the core 4 features."
                ),
            )
        )
        padding = st.selectbox(
            "Padding / truncation strategy",
            options=["pad_to_longer", "truncate_to_shorter", "fixed_length"],
            index=0,
            help=(
                "How to handle trajectory pairs with different lengths:\n\n"
                "• **pad_to_longer:** Pad shorter trajectory with zeros to match longer\n"
                "  ✓ Preserves all data, uses masks for valid timesteps\n"
                "  ✗ May waste memory on very long trajectories\n\n"
                "• **truncate_to_shorter:** Truncate longer trajectory to match shorter\n"
                "  ✓ Uniform lengths, no padding artifacts\n"
                "  ✗ Loses data from longer trajectories\n\n"
                "• **fixed_length:** Pad or truncate to a specific length\n"
                "  ✓ Consistent memory usage, predictable batch sizes\n"
                "  ✗ Requires choosing appropriate fixed length"
            ),
        )
        fixed_length = None
        if padding == "fixed_length":
            fixed_length = st.number_input(
                "Fixed sequence length", 
                min_value=1, 
                max_value=20000, 
                value=2000,
                help=(
                    "Target sequence length for all pairs.\n\n"
                    "• Trajectories **shorter** than this will be zero-padded\n"
                    "• Trajectories **longer** than this will be truncated\n\n"
                    "**Tip:** Check 'Length Stats' in metadata to choose appropriate value."
                ),
            )
        pos_strategy = st.selectbox(
            "Positive pair strategy", 
            options=["random", "sequential"], 
            index=0,
            help=(
                "Strategy for selecting same-agent trajectory pairs:\n\n"
                "• **random:** Randomly select two non-overlapping segments\n"
                "  ✓ Maximum diversity in paired trajectories\n"
                "  ✓ Good for general pattern learning\n\n"
                "• **sequential:** Prefer temporally distant segments\n"
                "  ✓ Tests if model recognizes same driver across time\n"
                "  ✓ Better for learning persistent driving habits"
            ),
        )
        
        st.divider()
        st.markdown("**🎯 Negative Pair Settings**")
        st.caption("Critical for discriminator performance")
        
        neg_strategy = st.selectbox(
            "Negative pair base strategy", 
            options=["random", "round_robin", "temporal_hard", "spatial_hard", "mixed_hard"], 
            index=0,
            help=(
                "Base strategy for selecting different-agent trajectory pairs:\n\n"
                "**Standard Strategies:**\n"
                "• **random:** Randomly select two trajectories from different agents\n"
                "  ✓ Fast, simple baseline\n"
                "  ✗ May create 'easy' negatives (different times/locations)\n\n"
                "• **round_robin:** Cycle through agent pairs systematically\n"
                "  ✓ Ensures all agent combinations are covered\n\n"
                "**Hard Negative Strategies** (recommended for better discrimination):\n"
                "• **temporal_hard:** Pair trajectories that overlap in time\n"
                "  ✓ Forces model to learn driver-specific patterns, not time patterns\n\n"
                "• **spatial_hard:** Pair trajectories visiting similar locations\n"
                "  ✓ Forces model to learn movement patterns, not location patterns\n\n"
                "• **mixed_hard:** Require both temporal AND spatial similarity\n"
                "  ✓ Hardest negatives: different drivers, same place, same time\n"
                "  ✓ Best for robust discriminator training"
            ),
        )
        
        # Hard negative ratio control (only show if not using hard strategy directly)
        hard_negative_ratio = 0.0
        temporal_overlap_threshold = 0.5
        spatial_similarity_threshold = 0.3
        
        if neg_strategy in ["random", "round_robin"]:
            hard_negative_ratio = st.slider(
                "Hard negative ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help=(
                    "Fraction of negative pairs that should be 'hard negatives'.\n\n"
                    "**Hard negatives** are pairs from different drivers where the trajectories\n"
                    "are similar in time and/or space, making them harder to distinguish.\n\n"
                    "• **0.0:** All negatives are random (baseline)\n"
                    "• **0.3-0.5:** Mix of easy and hard negatives (recommended)\n"
                    "• **1.0:** All negatives are hard (may slow generation)\n\n"
                    "**Why hard negatives matter:**\n"
                    "Your model showed 8.9% accuracy on different drivers because it learned\n"
                    "to distinguish by time/location rather than driving patterns. Hard negatives\n"
                    "force it to learn the actual driver-specific characteristics."
                ),
            )
            if hard_negative_ratio > 0:
                st.info(f"~{int(neg_pairs * hard_negative_ratio)} hard negatives will be generated")
        
        # Advanced hard negative settings
        with st.expander("⚙️ Hard Negative Thresholds", expanded=False):
            st.caption("Fine-tune what counts as a 'hard' negative")
            
            temporal_overlap_threshold = st.slider(
                "Temporal overlap threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help=(
                    "Minimum temporal overlap required for temporal hard negatives.\n\n"
                    "**Overlap = (intersection of time ranges) / (union of time ranges)**\n\n"
                    "• **0.0:** Any overlap counts\n"
                    "• **0.5:** At least 50% time overlap required\n"
                    "• **1.0:** Complete time overlap (very restrictive)\n\n"
                    "Lower values = more candidates, less strict matching."
                ),
            )
            
            spatial_similarity_threshold = st.slider(
                "Spatial similarity threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help=(
                    "Minimum spatial similarity (Jaccard index) for spatial hard negatives.\n\n"
                    "**Jaccard = |cells_A ∩ cells_B| / |cells_A ∪ cells_B|**\n\n"
                    "• **0.0:** Any shared cells count\n"
                    "• **0.3:** At least 30% cell overlap (recommended)\n"
                    "• **0.8:** High overlap required (very restrictive)\n\n"
                    "Lower values = more candidates, covers nearby routes."
                ),
            )
        
        distribution = st.selectbox(
            "Agent sampling distribution", 
            options=["proportional", "uniform"], 
            index=0,
            help=(
                "How to weight agents during random selection:\n\n"
                "• **proportional:** Weight by trajectory data volume\n"
                "  ✓ Agents with more data appear more often\n"
                "  ✓ Reflects natural data distribution\n"
                "  ✗ May under-represent agents with less data\n\n"
                "• **uniform:** Equal probability for all agents\n"
                "  ✓ Every agent equally represented\n"
                "  ✗ May over-sample sparse agents"
            ),
        )
        
        st.divider()
        st.markdown("**✨ Identical Pair Settings**")
        st.caption("Recommended for better model generalization")
        identical_ratio = st.slider(
            "Identical pair ratio",
            min_value=0.0,
            max_value=0.5,
            value=0.15,
            step=0.05,
            help=(
                "Fraction of positive pairs that should be identical (same trajectory vs itself).\n\n"
                "**Why this matters:** The discriminator should output ~1.0 when comparing\n"
                "a trajectory to itself. Without identical pairs in training, the model\n"
                "learns only to distinguish different trajectories, not to recognize identical ones.\n\n"
                "• **0.0:** No identical pairs (not recommended)\n"
                "• **0.10-0.20:** Good balance (recommended)\n"
                "• **0.50:** Half of positives are identical\n\n"
                "Identical pairs help calibrate the model's similarity scores."
            ),
        )
        if identical_ratio > 0:
            estimated_identical = int(pos_pairs * identical_ratio)
            st.info(f"~{estimated_identical} identical pairs will be included in training")
        
        st.divider()
        seed_text = st.text_input(
            "Random seed (leave blank for random)", 
            value="42",
            help=(
                "Seed for reproducible dataset generation.\n\n"
                "• **Set a value (e.g., 42):** Same configuration = same dataset\n"
                "• **Leave blank:** Different dataset each time\n\n"
                "Use a fixed seed for reproducible experiments."
            ),
        )
        ensure_coverage = st.checkbox(
            "Ensure every agent appears", 
            value=True, 
            disabled=per_agent_mode,
            help=(
                "Guarantee that every agent appears in at least one positive and one negative pair.\n\n"
                "**Enabled:** First samples one pair per agent, then fills remaining randomly.\n"
                "**Disabled:** Purely random sampling (some agents may be missing).\n\n"
                "⚠️ Automatically enabled in per-agent mode."
            ),
        )
        preview_cap = st.slider(
            "Preview pair cap", 
            min_value=4, 
            max_value=40, 
            value=12,
            help=(
                "Maximum pairs to show in preview mode.\n\n"
                "Keeps preview fast while showing representative sample.\n"
                "Full dataset generation is unaffected by this setting."
            ),
        )
        
        # Show estimated totals when in per-agent mode
        if per_agent_mode:
            st.divider()
            st.markdown("**📊 Estimated Totals** (assuming 50 agents):")
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
        identical_pair_ratio=identical_ratio,
        hard_negative_ratio=hard_negative_ratio,
        temporal_overlap_threshold=temporal_overlap_threshold,
        spatial_similarity_threshold=spatial_similarity_threshold,
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
        "identical_pair_ratio": cfg.identical_pair_ratio,
        "hard_negative_ratio": cfg.hard_negative_ratio,
        "temporal_overlap_threshold": cfg.temporal_overlap_threshold,
        "spatial_similarity_threshold": cfg.spatial_similarity_threshold,
    }
    return cfg, cache_key, int(preview_cap)


@st.cache_data(show_spinner=False)
def _generate_cached(config_dict: Dict, preview_only: bool, preview_cap: int):
    cfg = GenerationConfig(**{**config_dict, "data_path": Path(config_dict["data_path"])})
    return assemble_dataset(cfg, preview_only=preview_only, preview_cap=preview_cap)


def _render_preview(dataset: Dict[str, np.ndarray], metadata: Dict):
    st.subheader("Preview")
    
    # Check if per-agent mode was configured (but preview uses total counts)
    cfg = metadata.get("config", {})
    if cfg.get("per_agent_counts", False):
        st.warning(
            "⚠️ **Preview Mode Limitation**: Preview uses capped total counts for speed. "
            "The full dataset will use per-agent counts as configured. "
            "Click 'Generate Full Dataset' to see actual coverage."
        )
    
    # Array shapes with explanation
    x1_shape = dataset["x1"].shape
    st.markdown("**Output Array Shapes:**")
    
    with st.expander("📐 Shape Explanation", expanded=True):
        n_pairs, seq_len, n_features = x1_shape
        st.markdown(f"""
| Array | Shape | Description |
|-------|-------|-------------|
| `x1` | `{list(x1_shape)}` | First trajectory in each pair: **{n_pairs}** pairs × **{seq_len}** timesteps × **{n_features}** features |
| `x2` | `{list(dataset['x2'].shape)}` | Second trajectory in each pair (same dimensions as x1) |
| `label` | `{list(dataset['label'].shape)}` | Labels for each pair: **1** = same agent (positive), **0** = different agents (negative) |
| `mask1` | `{list(dataset['mask1'].shape)}` | Validity mask for x1: **1** = real data, **0** = padding |
| `mask2` | `{list(dataset['mask2'].shape)}` | Validity mask for x2 |

**Features** (indices 0-3, always included):
- `[0]` x_grid — grid x-coordinate
- `[1]` y_grid — grid y-coordinate  
- `[2]` time_bucket — time of day (1-288, each bucket = 5 minutes)
- `[3]` day_index — day of week (1-5, Mon-Fri only)
        """)
    
    labels = dataset["label"]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Positive Pairs (label=1)", int((labels == 1).sum()))
    with col2:
        st.metric("Negative Pairs (label=0)", int((labels == 0).sum()))
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
            # Label 1 = same agent (blue), Label 0 = different agents (red)
            color=alt.Color("label:N", scale=alt.Scale(domain=["0", "1"], range=["#d62728", "#1f77b4"]), legend=alt.Legend(title="label")),
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
    st.subheader("🔍 Sample Pair Explorer")
    
    # Filter options
    col1, col2 = st.columns([2, 1])
    with col1:
        max_idx = len(pair_info) - 1
        pair_idx = st.number_input("Pair index", min_value=0, max_value=max_idx, value=0, step=1)
    with col2:
        # Quick filter by pair type
        pair_types = list(set(p.get("pair_type", "unknown") for p in pair_info))
        filter_type = st.selectbox("Filter by type", ["all"] + sorted(pair_types))
    
    # Apply filter and get pair info
    if filter_type != "all":
        filtered_indices = [i for i, p in enumerate(pair_info) if p.get("pair_type") == filter_type]
        if filtered_indices:
            if pair_idx not in filtered_indices:
                pair_idx = filtered_indices[0]
            st.caption(f"Showing {len(filtered_indices)} '{filter_type}' pairs")
    
    info = pair_info[int(pair_idx)]
    
    # Enhanced pair information display
    pair_type = info.get("pair_type", "unknown")
    label = info.get("label", -1)
    
    # Color-coded pair type display
    type_colors = {
        "identical": "🔵 Identical (same trajectory vs itself)",
        "same_agent": "🟢 Same Agent (positive pair)",
        "different_agent": "🔴 Different Agent (negative pair)"
    }
    type_display = type_colors.get(pair_type, pair_type)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Pair Type:** {type_display}")
        st.markdown(f"**Label:** {'1 (Same Agent)' if label == 1 else '0 (Different Agent)'}")
    with col2:
        st.markdown(f"**Agent A:** `{info.get('agent_a')}`")
        st.markdown(f"**Agent B:** `{info.get('agent_b')}`")
    with col3:
        st.markdown(f"**Raw Length A:** {info.get('len_raw_a', 0):,} timesteps")
        st.markdown(f"**Raw Length B:** {info.get('len_raw_b', 0):,} timesteps")
    
    # Time overlap information for negative pairs
    if pair_type == "different_agent":
        start_a = info.get("start_time_a", 0)
        end_a = info.get("end_time_a", 0)
        start_b = info.get("start_time_b", 0)
        end_b = info.get("end_time_b", 0)
        
        overlap_start = max(start_a, start_b)
        overlap_end = min(end_a, end_b)
        
        if overlap_start < overlap_end:
            total_span = max(end_a, end_b) - min(start_a, start_b)
            overlap_ratio = (overlap_end - overlap_start) / total_span if total_span > 0 else 0
            st.info(f"⏱️ **Temporal Overlap:** {overlap_ratio:.1%} (time range {overlap_start:.0f} - {overlap_end:.0f})")
        else:
            st.caption("⏱️ No temporal overlap between these trajectories")

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


def _render_dataset_validation(dataset: Dict[str, np.ndarray], metadata: Dict[str, Any]):
    """Render comprehensive dataset validation with agent distribution histograms."""
    st.subheader("🔍 Dataset Validation Report")
    
    cfg = metadata.get("config", {})
    counts = metadata.get("counts", {})
    length_stats = metadata.get("length_stats", {})
    agent_usage = metadata.get("agent_usage", {})
    
    # === Configuration Summary ===
    with st.expander("⚙️ Configuration Summary", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sampling Settings**")
            st.markdown(f"- **Mode**: {'Per-Agent Counts' if cfg.get('per_agent_counts') else 'Total Counts'}")
            st.markdown(f"- **Positive pairs config**: {cfg.get('positive_pairs', 'N/A')}")
            st.markdown(f"- **Negative pairs config**: {cfg.get('negative_pairs', 'N/A')}")
            identical_ratio = cfg.get('identical_pair_ratio', 0)
            if identical_ratio > 0:
                st.markdown(f"- **Identical pair ratio**: {identical_ratio:.0%}")
            st.markdown(f"- **Positive strategy**: {cfg.get('positive_strategy', 'N/A')}")
            st.markdown(f"- **Negative strategy**: {cfg.get('negative_strategy', 'N/A')}")
            hard_ratio = cfg.get('hard_negative_ratio', 0)
            if hard_ratio > 0:
                st.markdown(f"- **Hard negative ratio**: {hard_ratio:.0%}")
            st.markdown(f"- **Agent distribution**: {cfg.get('agent_distribution', 'N/A')}")
        with col2:
            st.markdown("**Data Settings**")
            st.markdown(f"- **Days concatenated**: {cfg.get('days', 'N/A')}")
            st.markdown(f"- **Feature slice**: indices 0-3 + [{cfg.get('feature_start', 4)}:{cfg.get('feature_end', 4)}]")
            st.markdown(f"- **Padding mode**: {cfg.get('padding', 'N/A')}")
            if cfg.get('fixed_length'):
                st.markdown(f"- **Fixed length**: {cfg.get('fixed_length')}")
            st.markdown(f"- **Random seed**: {cfg.get('seed', 'None (random)')}")
            st.markdown(f"- **Ensure coverage**: {cfg.get('ensure_agent_coverage', False)}")
            # Hard negative thresholds
            if hard_ratio > 0 or cfg.get('negative_strategy') in ['temporal_hard', 'spatial_hard', 'mixed_hard']:
                st.markdown(f"- **Temporal overlap threshold**: {cfg.get('temporal_overlap_threshold', 0.5):.0%}")
                st.markdown(f"- **Spatial similarity threshold**: {cfg.get('spatial_similarity_threshold', 0.3):.0%}")
    
    # === Validation Checks ===
    st.markdown("### ✅ Validation Checks")
    
    validation_results = []
    all_passed = True
    
    # Check 1: Pair counts match
    actual_pos = counts.get("positive_pairs", 0)
    actual_neg = counts.get("negative_pairs", 0)
    actual_identical = counts.get("identical_pairs", 0)
    actual_total = counts.get("total_pairs", 0)
    
    labels = dataset["label"]
    # Labels: 1 = same agent (positive + identical), 0 = different agents (negative)
    computed_pos = int((labels == 1).sum())
    computed_neg = int((labels == 0).sum())
    
    # Expected label=1 count includes both positive pairs AND identical pairs
    expected_label_1 = actual_pos + actual_identical
    
    check_pos = expected_label_1 == computed_pos
    check_neg = actual_neg == computed_neg
    check_total = actual_total == (computed_pos + computed_neg)
    
    validation_results.append({
        "Check": "Same-agent pairs (label=1: positive + identical)",
        "Expected": expected_label_1,
        "Actual": computed_pos,
        "Status": "✅ Pass" if check_pos else "❌ Fail"
    })
    if actual_identical > 0:
        validation_results.append({
            "Check": "  └─ Including identical pairs",
            "Expected": actual_identical,
            "Actual": actual_identical,
            "Status": "ℹ️ Info"
        })
    validation_results.append({
        "Check": "Negative pair count (label=0)",
        "Expected": actual_neg,
        "Actual": computed_neg,
        "Status": "✅ Pass" if check_neg else "❌ Fail"
    })
    validation_results.append({
        "Check": "Total pair count",
        "Expected": actual_total,
        "Actual": computed_pos + computed_neg,
        "Status": "✅ Pass" if check_total else "❌ Fail"
    })
    all_passed = all_passed and check_pos and check_neg and check_total
    
    # Check 2: Array shapes consistency
    n_pairs, seq_len, n_features = dataset["x1"].shape
    check_x2_shape = dataset["x2"].shape == (n_pairs, seq_len, n_features)
    check_mask1_shape = dataset["mask1"].shape == (n_pairs, seq_len)
    check_mask2_shape = dataset["mask2"].shape == (n_pairs, seq_len)
    check_label_shape = dataset["label"].shape == (n_pairs,)
    
    validation_results.append({
        "Check": "x2 shape matches x1",
        "Expected": f"({n_pairs}, {seq_len}, {n_features})",
        "Actual": str(dataset["x2"].shape),
        "Status": "✅ Pass" if check_x2_shape else "❌ Fail"
    })
    validation_results.append({
        "Check": "mask1 shape",
        "Expected": f"({n_pairs}, {seq_len})",
        "Actual": str(dataset["mask1"].shape),
        "Status": "✅ Pass" if check_mask1_shape else "❌ Fail"
    })
    validation_results.append({
        "Check": "mask2 shape",
        "Expected": f"({n_pairs}, {seq_len})",
        "Actual": str(dataset["mask2"].shape),
        "Status": "✅ Pass" if check_mask2_shape else "❌ Fail"
    })
    validation_results.append({
        "Check": "label shape",
        "Expected": f"({n_pairs},)",
        "Actual": str(dataset["label"].shape),
        "Status": "✅ Pass" if check_label_shape else "❌ Fail"
    })
    all_passed = all_passed and check_x2_shape and check_mask1_shape and check_mask2_shape and check_label_shape
    
    # Check 3: Feature count
    expected_features = 4 + max(0, cfg.get("feature_end", 4) - cfg.get("feature_start", 4))
    check_features = n_features == expected_features
    validation_results.append({
        "Check": "Feature count",
        "Expected": expected_features,
        "Actual": n_features,
        "Status": "✅ Pass" if check_features else "❌ Fail"
    })
    all_passed = all_passed and check_features
    
    # Check 4: Sequence length matches padded_length
    padded_len = length_stats.get("padded_length", seq_len)
    check_seq_len = seq_len == padded_len
    validation_results.append({
        "Check": "Sequence length = padded_length",
        "Expected": padded_len,
        "Actual": seq_len,
        "Status": "✅ Pass" if check_seq_len else "❌ Fail"
    })
    all_passed = all_passed and check_seq_len
    
    # Check 5: Labels are valid (0 or 1)
    unique_labels = set(np.unique(labels).tolist())
    valid_labels = unique_labels.issubset({0, 1})
    validation_results.append({
        "Check": "Labels are valid (0 or 1)",
        "Expected": "{0, 1}",
        "Actual": str(unique_labels),
        "Status": "✅ Pass" if valid_labels else "❌ Fail"
    })
    all_passed = all_passed and valid_labels
    
    # Check 6: Per-agent coverage (if enabled)
    num_agents_in_data = len(agent_usage)
    if cfg.get("per_agent_counts"):
        # In per-agent mode, all agents should be represented
        expected_agents = 50  # Assumed
        check_agent_count = num_agents_in_data >= expected_agents * 0.9  # Allow 10% tolerance
        validation_results.append({
            "Check": "Agent coverage (per-agent mode)",
            "Expected": f"≥{int(expected_agents * 0.9)} agents",
            "Actual": f"{num_agents_in_data} agents",
            "Status": "✅ Pass" if check_agent_count else "⚠️ Warning"
        })
        
        # Check positive pairs per agent
        pos_counts = [v.get("pos", 0) for v in agent_usage.values()]
        min_pos = min(pos_counts) if pos_counts else 0
        expected_pos_per_agent = cfg.get("positive_pairs", 0)
        check_pos_per_agent = min_pos >= expected_pos_per_agent * 0.9
        validation_results.append({
            "Check": "Min positive pairs per agent",
            "Expected": f"≥{int(expected_pos_per_agent * 0.9)}",
            "Actual": min_pos,
            "Status": "✅ Pass" if check_pos_per_agent else "⚠️ Warning"
        })
    
    # Display validation table
    st.dataframe(pd.DataFrame(validation_results), hide_index=True, use_container_width=True)
    
    if all_passed:
        st.success("✅ All validation checks passed!")
    else:
        st.warning("⚠️ Some validation checks failed or have warnings. Review the table above.")
    
    # === Dataset Characteristics Summary ===
    st.markdown("### 📈 Dataset Characteristics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pairs", f"{actual_total:,}")
    with col2:
        st.metric("Positive Pairs", f"{actual_pos:,}", delta=f"{actual_pos/actual_total*100:.1f}%" if actual_total > 0 else "0%")
    with col3:
        st.metric("Negative Pairs", f"{actual_neg:,}", delta=f"{actual_neg/actual_total*100:.1f}%" if actual_total > 0 else "0%")
    with col4:
        st.metric("Unique Agents", f"{num_agents_in_data}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sequence Length", f"{seq_len:,}")
    with col2:
        st.metric("Features", n_features)
    with col3:
        combined = length_stats.get("combined", {})
        st.metric("Avg Raw Length", f"{combined.get('mean', 0):.0f}")
    with col4:
        st.metric("Dataset Hash", metadata.get("dataset_hash", "N/A")[:8] + "...")
    
    # === Hard Negative Breakdown (if available) ===
    neg_breakdown = counts.get("negative_breakdown")
    if neg_breakdown:
        st.markdown("### 🎯 Negative Pair Difficulty Breakdown")
        st.markdown("""
        Hard negatives are crucial for training a discriminator that can distinguish 
        different drivers based on driving patterns rather than just time/location differences.
        """)
        
        total_neg = actual_neg
        random_count = neg_breakdown.get("random", 0)
        temporal_hard = neg_breakdown.get("temporal_hard", 0)
        spatial_hard = neg_breakdown.get("spatial_hard", 0)
        mixed_hard = neg_breakdown.get("mixed_hard", 0)
        hard_total = temporal_hard + spatial_hard + mixed_hard
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pct = random_count / total_neg * 100 if total_neg > 0 else 0
            st.metric("Random/Easy", f"{random_count:,}", delta=f"{pct:.1f}%", delta_color="off")
        with col2:
            pct = temporal_hard / total_neg * 100 if total_neg > 0 else 0
            st.metric("Temporal Hard", f"{temporal_hard:,}", delta=f"{pct:.1f}%", delta_color="off")
        with col3:
            pct = spatial_hard / total_neg * 100 if total_neg > 0 else 0
            st.metric("Spatial Hard", f"{spatial_hard:,}", delta=f"{pct:.1f}%", delta_color="off")
        with col4:
            pct = mixed_hard / total_neg * 100 if total_neg > 0 else 0
            st.metric("Mixed Hard", f"{mixed_hard:,}", delta=f"{pct:.1f}%", delta_color="off")
        
        # Pie chart of negative breakdown
        neg_breakdown_df = pd.DataFrame({
            "type": ["Random/Easy", "Temporal Hard", "Spatial Hard", "Mixed Hard"],
            "count": [random_count, temporal_hard, spatial_hard, mixed_hard],
            "description": [
                "Random pairs from different drivers",
                "Similar time, different drivers",
                "Similar location, different drivers", 
                "Same time AND location, different drivers"
            ]
        })
        neg_breakdown_df = neg_breakdown_df[neg_breakdown_df["count"] > 0]
        
        if not neg_breakdown_df.empty:
            pie_chart = (
                alt.Chart(neg_breakdown_df)
                .mark_arc(innerRadius=50)
                .encode(
                    theta=alt.Theta("count:Q", stack=True),
                    color=alt.Color(
                        "type:N",
                        scale=alt.Scale(
                            domain=["Random/Easy", "Temporal Hard", "Spatial Hard", "Mixed Hard"],
                            range=["#aec7e8", "#ff7f0e", "#2ca02c", "#d62728"]
                        )
                    ),
                    tooltip=["type", "count", "description"]
                )
                .properties(height=250, title="Negative Pair Composition")
            )
            st.altair_chart(pie_chart, use_container_width=True)
        
        # Recommendation based on hard negative ratio
        actual_hard_ratio = counts.get("hard_negative_actual_ratio", 0)
        if hard_total > 0:
            if actual_hard_ratio < 0.3:
                st.warning(f"""
                ⚠️ **Low hard negative ratio ({actual_hard_ratio:.1%})**
                
                Your discriminator may learn to distinguish drivers by time/location rather than 
                driving patterns. Consider increasing the hard negative ratio to 40-60% for better 
                discrimination performance on the "Different Drivers" test scenario.
                """)
            elif actual_hard_ratio >= 0.3 and actual_hard_ratio < 0.6:
                st.success(f"""
                ✅ **Good hard negative ratio ({actual_hard_ratio:.1%})**
                
                This mix of easy and hard negatives should help your discriminator learn robust 
                driver-specific patterns while maintaining stable training.
                """)
            else:
                st.info(f"""
                ℹ️ **High hard negative ratio ({actual_hard_ratio:.1%})**
                
                This dataset emphasizes challenging negatives. If training becomes unstable or 
                converges too slowly, consider reducing the hard negative ratio slightly.
                """)
    
    # === Agent Distribution Histograms ===
    st.markdown("### 📊 Agent Representation Distribution")
    
    if agent_usage:
        # Prepare data for histograms
        agent_ids = list(agent_usage.keys())
        pos_counts = [agent_usage[a].get("pos", 0) for a in agent_ids]
        neg_counts = [agent_usage[a].get("neg", 0) for a in agent_ids]
        total_counts = [pos_counts[i] + neg_counts[i] for i in range(len(agent_ids))]
        
        agent_df = pd.DataFrame({
            "agent_id": agent_ids,
            "positive": pos_counts,
            "negative": neg_counts,
            "total": total_counts
        })
        
        # Sort by agent_id for consistent display
        try:
            agent_df["agent_id_int"] = agent_df["agent_id"].astype(int)
            agent_df = agent_df.sort_values("agent_id_int")
        except ValueError:
            agent_df = agent_df.sort_values("agent_id")
        
        tab1, tab2, tab3 = st.tabs(["📊 Positive Distribution", "📊 Negative Distribution", "📊 Total Distribution"])
        
        with tab1:
            st.markdown("**Positive pairs (same-agent) per agent:**")
            pos_chart = (
                alt.Chart(agent_df)
                .mark_bar(color="#1f77b4")
                .encode(
                    x=alt.X("agent_id:N", sort=None, title="Agent ID"),
                    y=alt.Y("positive:Q", title="Positive Pair Count"),
                    tooltip=["agent_id", "positive"]
                )
                .properties(height=300)
            )
            st.altair_chart(pos_chart, use_container_width=True)
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min", min(pos_counts))
            with col2:
                st.metric("Max", max(pos_counts))
            with col3:
                st.metric("Mean", f"{np.mean(pos_counts):.1f}")
            with col4:
                st.metric("Std Dev", f"{np.std(pos_counts):.1f}")
        
        with tab2:
            st.markdown("**Negative pairs (different-agent) per agent:**")
            neg_chart = (
                alt.Chart(agent_df)
                .mark_bar(color="#d62728")
                .encode(
                    x=alt.X("agent_id:N", sort=None, title="Agent ID"),
                    y=alt.Y("negative:Q", title="Negative Pair Count"),
                    tooltip=["agent_id", "negative"]
                )
                .properties(height=300)
            )
            st.altair_chart(neg_chart, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min", min(neg_counts))
            with col2:
                st.metric("Max", max(neg_counts))
            with col3:
                st.metric("Mean", f"{np.mean(neg_counts):.1f}")
            with col4:
                st.metric("Std Dev", f"{np.std(neg_counts):.1f}")
        
        with tab3:
            st.markdown("**Total appearances per agent (positive + negative):**")
            total_chart = (
                alt.Chart(agent_df)
                .mark_bar(color="#2ca02c")
                .encode(
                    x=alt.X("agent_id:N", sort=None, title="Agent ID"),
                    y=alt.Y("total:Q", title="Total Pair Count"),
                    tooltip=["agent_id", "positive", "negative", "total"]
                )
                .properties(height=300)
            )
            st.altair_chart(total_chart, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min", min(total_counts))
            with col2:
                st.metric("Max", max(total_counts))
            with col3:
                st.metric("Mean", f"{np.mean(total_counts):.1f}")
            with col4:
                st.metric("Std Dev", f"{np.std(total_counts):.1f}")
        
        # Stacked bar chart for comparison
        st.markdown("**Combined View: Positive vs Negative per Agent**")
        melted_df = agent_df.melt(
            id_vars=["agent_id"],
            value_vars=["positive", "negative"],
            var_name="type",
            value_name="count"
        )
        stacked_chart = (
            alt.Chart(melted_df)
            .mark_bar()
            .encode(
                x=alt.X("agent_id:N", sort=None, title="Agent ID"),
                y=alt.Y("count:Q", title="Pair Count"),
                color=alt.Color("type:N", scale=alt.Scale(domain=["positive", "negative"], range=["#1f77b4", "#d62728"])),
                tooltip=["agent_id", "type", "count"]
            )
            .properties(height=300)
        )
        st.altair_chart(stacked_chart, use_container_width=True)
        
        # Coverage analysis
        st.markdown("### 🎯 Coverage Analysis")
        
        agents_with_pos = sum(1 for c in pos_counts if c > 0)
        agents_with_neg = sum(1 for c in neg_counts if c > 0)
        agents_with_both = sum(1 for i in range(len(pos_counts)) if pos_counts[i] > 0 and neg_counts[i] > 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Agents with Positives", f"{agents_with_pos}/{num_agents_in_data}")
        with col2:
            st.metric("Agents with Negatives", f"{agents_with_neg}/{num_agents_in_data}")
        with col3:
            st.metric("Agents with Both", f"{agents_with_both}/{num_agents_in_data}")
        
        # Identify under-represented agents
        if cfg.get("per_agent_counts"):
            expected_pos = cfg.get("positive_pairs", 0)
            expected_neg = cfg.get("negative_pairs", 0) * (num_agents_in_data - 1)
            
            underrep_pos = [agent_ids[i] for i in range(len(pos_counts)) if pos_counts[i] < expected_pos * 0.8]
            underrep_neg = [agent_ids[i] for i in range(len(neg_counts)) if neg_counts[i] < expected_neg * 0.8]
            
            if underrep_pos or underrep_neg:
                with st.expander("⚠️ Under-represented Agents", expanded=False):
                    if underrep_pos:
                        st.warning(f"Agents with <80% expected positive pairs: {', '.join(map(str, underrep_pos[:10]))}" + 
                                  (f" (+{len(underrep_pos)-10} more)" if len(underrep_pos) > 10 else ""))
                    if underrep_neg:
                        st.warning(f"Agents with <80% expected negative pairs: {', '.join(map(str, underrep_neg[:10]))}" +
                                  (f" (+{len(underrep_neg)-10} more)" if len(underrep_neg) > 10 else ""))
    else:
        st.warning("No agent usage data available for distribution analysis.")
    
    # === Length Distribution ===
    st.markdown("### 📏 Sequence Length Analysis")
    
    # Compute actual lengths from masks
    lengths_x1 = dataset["mask1"].sum(axis=1)
    lengths_x2 = dataset["mask2"].sum(axis=1)
    
    length_df = pd.DataFrame({
        "x1_length": lengths_x1,
        "x2_length": lengths_x2,
        "label": dataset["label"]
    })
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**x1 Length Distribution**")
        x1_hist = (
            alt.Chart(length_df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("x1_length:Q", bin=alt.Bin(maxbins=30), title="Length"),
                y=alt.Y("count()", title="Frequency"),
                # Label 1 = same agent (blue), Label 0 = different agents (red)
                color=alt.Color("label:N", scale=alt.Scale(domain=[0, 1], range=["#d62728", "#1f77b4"]))
            )
            .properties(height=200)
        )
        st.altair_chart(x1_hist, use_container_width=True)
    
    with col2:
        st.markdown("**x2 Length Distribution**")
        x2_hist = (
            alt.Chart(length_df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("x2_length:Q", bin=alt.Bin(maxbins=30), title="Length"),
                y=alt.Y("count()", title="Frequency"),
                # Label 1 = same agent (blue), Label 0 = different agents (red)
                color=alt.Color("label:N", scale=alt.Scale(domain=[0, 1], range=["#d62728", "#1f77b4"]))
            )
            .properties(height=200)
        )
        st.altair_chart(x2_hist, use_container_width=True)


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
        
        # Validation section
        st.divider()
        _render_dataset_validation(dataset, metadata)
        
        st.divider()
        st.subheader("📥 Downloads & Save")
        
        # Save to directory option
        with st.expander("💾 Save to Directory (for training)", expanded=True):
            st.markdown("""
            Save the dataset to a directory for use with the training scripts.
            This creates files compatible with the discriminator model's data loaders.
            """)
            
            default_save_dir = str(Path("/home/robert/FAMAIL/discriminator/datasets").resolve())
            save_dir = st.text_input(
                "Save directory",
                value=default_save_dir,
                help="Directory where dataset files will be saved"
            )
            
            col_name, col_split = st.columns(2)
            with col_name:
                dataset_name = st.text_input(
                    "Dataset name",
                    value="trajectory_pairs",
                    help="Name for this dataset (used as subdirectory name)"
                )
            with col_split:
                val_split = st.slider(
                    "Validation split",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    help="Fraction of data to use for validation"
                )
            
            if st.button("💾 Save Dataset", type="primary", key="save_to_dir"):
                try:
                    save_path = Path(save_dir) / dataset_name
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    # Split dataset
                    n_samples = len(dataset["label"])
                    n_val = int(n_samples * val_split)
                    n_train = n_samples - n_val
                    
                    # Shuffle indices
                    rng = np.random.default_rng(metadata.get("config", {}).get("seed", 42))
                    indices = rng.permutation(n_samples)
                    train_idx = indices[:n_train]
                    val_idx = indices[n_train:]
                    
                    # Save train split
                    train_path = save_path / "train.npz"
                    np.savez_compressed(
                        train_path,
                        x1=dataset["x1"][train_idx],
                        x2=dataset["x2"][train_idx],
                        label=dataset["label"][train_idx],
                        mask1=dataset["mask1"][train_idx],
                        mask2=dataset["mask2"][train_idx]
                    )
                    
                    # Save val split
                    val_path = save_path / "val.npz"
                    np.savez_compressed(
                        val_path,
                        x1=dataset["x1"][val_idx],
                        x2=dataset["x2"][val_idx],
                        label=dataset["label"][val_idx],
                        mask1=dataset["mask1"][val_idx],
                        mask2=dataset["mask2"][val_idx]
                    )
                    
                    # Compute positive/negative counts per split
                    train_labels = dataset["label"][train_idx]
                    val_labels = dataset["label"][val_idx]
                    n_train_pos = int((train_labels == 1).sum())
                    n_train_neg = int((train_labels == 0).sum())
                    n_val_pos = int((val_labels == 1).sum())
                    n_val_neg = int((val_labels == 0).sum())
                    
                    # Save metadata
                    save_metadata = {
                        **metadata,
                        "split_info": {
                            "val_split": val_split,
                            "n_train": n_train,
                            "n_val": n_val,
                            "n_train_pos": n_train_pos,
                            "n_train_neg": n_train_neg,
                            "n_val_pos": n_val_pos,
                            "n_val_neg": n_val_neg,
                            "train_file": "train.npz",
                            "val_file": "val.npz"
                        }
                    }
                    with open(save_path / "metadata.json", "w") as f:
                        json.dump(save_metadata, f, indent=2)
                    
                    st.success(f"✅ Dataset saved to: `{save_path}`")
                    st.markdown(f"""
                    **Saved files:**
                    - `train.npz`: {n_train:,} samples ({100 - val_split*100:.0f}%) — {n_train_pos:,} pos, {n_train_neg:,} neg
                    - `val.npz`: {n_val:,} samples ({val_split*100:.0f}%) — {n_val_pos:,} pos, {n_val_neg:,} neg
                    - `metadata.json`: Configuration and statistics
                    
                    **Use with training:**
                    ```bash
                    cd /home/robert/FAMAIL/discriminator/model
                    python train.py --data-dir "{save_path}"
                    ```
                    """)
                except Exception as e:
                    st.error(f"❌ Failed to save: {e}")
        
        st.markdown("---")
        st.markdown("**Or download files directly:**")
        
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
