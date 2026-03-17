"""Streamlit Training Dashboard for the Discriminator Model.

Launch with:
    streamlit run discriminator/model/training_dashboard.py

Features:
- Dataset selection and exploration
- Hyperparameter configuration
- Training progress visualization
- Model evaluation and metrics
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
import queue
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Add model directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

try:
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Page config
st.set_page_config(
    page_title="Discriminator Training Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Default paths
DEFAULT_DATASET_DIR = Path("/home/robert/FAMAIL/discriminator/datasets").resolve()
MULTI_STREAM_DATASET_DIR = Path("/home/robert/FAMAIL/discriminator/multi_stream/datasets").resolve()
DEFAULT_CHECKPOINT_DIR = Path("/home/robert/FAMAIL/discriminator/model/checkpoints").resolve()


def check_dependencies():
    """Check if required dependencies are available."""
    if not TORCH_AVAILABLE:
        st.error("❌ PyTorch is required. Install with: `pip install torch`")
        st.stop()


def load_dataset_info(dataset_dir: Path) -> Dict[str, Any]:
    """Load dataset information from a directory."""
    info = {
        "path": str(dataset_dir),
        "has_train": (dataset_dir / "train.npz").exists(),
        "has_val": (dataset_dir / "val.npz").exists(),
        "has_test": (dataset_dir / "test.npz").exists(),
        "has_metadata": (dataset_dir / "metadata.json").exists()
    }
    
    # Load metadata if available
    if info["has_metadata"]:
        with open(dataset_dir / "metadata.json") as f:
            info["metadata"] = json.load(f)
    
    # Get file sizes and sample counts
    for split in ["train", "val", "test"]:
        npz_path = dataset_dir / f"{split}.npz"
        if npz_path.exists():
            with np.load(npz_path) as data:
                info[f"{split}_samples"] = len(data["label"])
                info[f"{split}_pos"] = int((data["label"] == 1).sum())
                info[f"{split}_neg"] = int((data["label"] == 0).sum())

                x1_shape = data["x1"].shape
                if len(x1_shape) == 4:
                    # Multi-stream V3: [N_pairs, N_trajs, L, 4]
                    info["is_multi_stream"] = True
                    info[f"{split}_n_trajs"] = x1_shape[1]
                    info[f"{split}_seq_len"] = x1_shape[2]
                    info[f"{split}_features"] = x1_shape[3]
                    if "driving_1" in data:
                        info[f"{split}_driving_seq_len"] = data["driving_1"].shape[2]
                    if "profile_1" in data:
                        info[f"{split}_profile_dim"] = data["profile_1"].shape[1]
                else:
                    # V1/V2: [N_pairs, L, 4]
                    info["is_multi_stream"] = False
                    info[f"{split}_seq_len"] = x1_shape[1]
                    info[f"{split}_features"] = x1_shape[2]
            info[f"{split}_size_mb"] = npz_path.stat().st_size / (1024 * 1024)

    return info


def list_available_datasets(*base_dirs: Path) -> List[Path]:
    """List available dataset directories from one or more base directories."""
    datasets = []
    for base_dir in base_dirs:
        if base_dir.exists():
            for item in base_dir.iterdir():
                if item.is_dir() and (item / "train.npz").exists():
                    datasets.append(item)
    return sorted(datasets)


def list_checkpoints(checkpoint_dir: Path) -> List[Path]:
    """List available checkpoint directories."""
    checkpoints = []
    if checkpoint_dir.exists():
        for item in checkpoint_dir.iterdir():
            if item.is_dir() and (item / "best.pt").exists():
                checkpoints.append(item)
    return sorted(checkpoints, reverse=True)


def load_training_history(checkpoint_dir: Path) -> Optional[Dict]:
    """Load training history from checkpoint directory."""
    history_path = checkpoint_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return None


def render_dataset_selector():
    """Render dataset selection UI."""
    st.subheader("📁 Dataset Selection")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        dataset_dir_str = st.text_input(
            "Dataset directory",
            value=str(DEFAULT_DATASET_DIR),
            help="Base directory containing dataset subdirectories. "
                 "Multi-stream datasets are also searched automatically."
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("🔄 Refresh")

    dataset_dir = Path(dataset_dir_str)

    # List available datasets from both single-stream and multi-stream dirs
    datasets = list_available_datasets(dataset_dir, MULTI_STREAM_DATASET_DIR)
    
    if not datasets:
        st.warning(f"No datasets found in {dataset_dir}. Generate a dataset using the Dataset Generator tool first.")
        
        # Option to use custom path
        custom_path = st.text_input(
            "Or enter a custom dataset path:",
            help="Path to a directory containing train.npz and val.npz"
        )
        if custom_path and Path(custom_path).exists():
            datasets = [Path(custom_path)]
    
    if datasets:
        # Label datasets with their parent dir for disambiguation
        def _dataset_label(d: Path) -> str:
            # Show parent context if from multi-stream dir
            if MULTI_STREAM_DATASET_DIR in d.parents or d.parent == MULTI_STREAM_DATASET_DIR:
                return f"multi_stream/{d.name}"
            return d.name

        dataset_options = {_dataset_label(d): d for d in datasets}
        selected_name = st.selectbox(
            "Select dataset",
            options=list(dataset_options.keys()),
            index=0
        )
        selected_dataset = dataset_options[selected_name]
        
        # Show dataset info
        info = load_dataset_info(selected_dataset)
        
        with st.expander("📊 Dataset Details", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if info.get("has_train"):
                    st.metric("Training Samples", f"{info.get('train_samples', 0):,}")
                    st.caption(f"Pos: {info.get('train_pos', 0):,} | Neg: {info.get('train_neg', 0):,}")
                    
            with col2:
                if info.get("has_val"):
                    st.metric("Validation Samples", f"{info.get('val_samples', 0):,}")
                    st.caption(f"Pos: {info.get('val_pos', 0):,} | Neg: {info.get('val_neg', 0):,}")
                    
            with col3:
                if info.get("is_multi_stream"):
                    st.metric("Dataset Type", "Multi-Stream (V3)")
                    n_trajs = info.get('train_n_trajs', '?')
                    seek_l = info.get('train_seq_len', '?')
                    drive_l = info.get('train_driving_seq_len', '?')
                    prof_d = info.get('train_profile_dim', '?')
                    st.caption(
                        f"{n_trajs} trajs/stream | "
                        f"Seek L={seek_l}, Drive L={drive_l} | "
                        f"Profile dim={prof_d}"
                    )
                elif info.get("train_seq_len"):
                    st.metric("Sequence Length", info.get('train_seq_len', 0))
                    st.caption(f"Features: {info.get('train_features', 0)}")
            
            # Show config from metadata if available
            if "metadata" in info:
                cfg = info["metadata"].get("config", {})
                st.markdown("**Configuration:**")
                st.json(cfg)
        
        return selected_dataset, info
    
    return None, None


def render_hyperparameters(is_multi_stream: bool = False):
    """Render hyperparameter configuration UI."""
    st.subheader("⚙️ Hyperparameters")

    with st.expander("Model Architecture", expanded=True):
        # Model version selector
        version_help = {
            "v1": "Original Siamese LSTM (concatenation-based)",
            "v2": "Improved with distance-based similarity",
            "v3": "Multi-stream Ren-aligned (seeking + driving + profile)",
        }
        default_version = "v3" if is_multi_stream else "v2"
        available_versions = ["v1", "v2", "v3"]
        model_version = st.selectbox(
            "Model Version",
            available_versions,
            index=available_versions.index(default_version),
            format_func=lambda v: f"{v.upper()} — {version_help[v]}",
            help="V3 requires multi-stream datasets with seeking, driving, and profile data."
        )

        if is_multi_stream and model_version != "v3":
            st.warning("Dataset is multi-stream but selected model is not V3. "
                       "V1/V2 will only use the seeking stream.")
        elif not is_multi_stream and model_version == "v3":
            st.error("V3 requires a multi-stream dataset with driving and profile data.")

        col1, col2 = st.columns(2)
        with col1:
            lstm_dims_str = st.text_input(
                "LSTM Hidden Dims",
                "200, 100",
                help="Hidden dimensions per LSTM layer (comma-separated). Default: 200, 100 follows ST-SiameseNet."
            )
            lstm_hidden_dims = tuple(int(x.strip()) for x in lstm_dims_str.split(",") if x.strip())
            classifier_dims_str = st.text_input(
                "Classifier Hidden Dims",
                "64, 32, 8",
                help="Hidden dimensions for classifier MLP (comma-separated). Default: 64, 32, 8 follows ST-SiameseNet."
            )
            classifier_dims = tuple(int(x.strip()) for x in classifier_dims_str.split(",") if x.strip())
        with col2:
            dropout = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
            bidirectional = st.checkbox("Bidirectional", value=False)

        # Combination mode (V2 and V3)
        if model_version in ("v2", "v3"):
            combo_options = ["difference", "concatenation", "distance", "hybrid"]
            default_combo = "concatenation" if model_version == "v3" else "difference"
            combination_mode = st.selectbox(
                "Combination Mode",
                combo_options,
                index=combo_options.index(default_combo),
                help="How branch embeddings are combined. 'concatenation' matches Ren et al. for V3."
            )
        else:
            combination_mode = None

        # V3-specific parameters
        if model_version == "v3":
            st.markdown("**V3 Multi-Stream Options:**")
            v3_col1, v3_col2, v3_col3 = st.columns(3)
            with v3_col1:
                streams_str = st.text_input(
                    "Active Streams", "seeking, driving, profile",
                    help="Comma-separated list of active streams"
                )
                streams = tuple(s.strip() for s in streams_str.split(",") if s.strip())
                n_trajs_per_stream = st.number_input(
                    "Trajs per Stream", 1, 20, 5,
                    help="Number of independent trajectories per stream per branch (Ren: 5)"
                )
            with v3_col2:
                traj_projection_dim = st.number_input(
                    "Traj Projection Dim", 8, 128, 48,
                    help="Per-trajectory projection dimension after LSTM (Ren: 48)"
                )
                n_profile_features = st.number_input(
                    "Profile Features", 1, 50, 11,
                    help="Number of profile input features (default: 11)"
                )
            with v3_col3:
                profile_hidden_str = st.text_input(
                    "Profile Hidden Dims", "64, 32",
                    help="Profile encoder hidden dims (Ren: 64, 32)"
                )
                profile_hidden_dims = tuple(int(x.strip()) for x in profile_hidden_str.split(",") if x.strip())
                profile_output_dim = st.number_input(
                    "Profile Output Dim", 2, 64, 8,
                    help="Profile embedding dimension (Ren: 8)"
                )
    
    with st.expander("Training", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.number_input("Epochs", 1, 500, 100)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256], index=2)
        with col2:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
                value=1e-3
            )
            weight_decay = st.select_slider(
                "Weight Decay",
                options=[0, 1e-5, 1e-4, 1e-3],
                value=1e-4
            )
        with col3:
            early_stopping = st.number_input("Early Stopping Patience", 0, 50, 10)
            scheduler = st.selectbox("LR Scheduler", ["plateau", "cosine", "none"])
    
    with st.expander("Other (hardware, workers, etc.)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            device = st.selectbox("Device", ["auto", "cuda", "cpu"])
            num_workers = st.number_input("DataLoader Workers", 0, 8, 0)
        with col2:
            seed = st.number_input("Random Seed", 0, 10000, 42)
            save_best_only = st.checkbox("Save Best Only", value=True)
    
    params = {
        "model_version": model_version,
        "lstm_hidden_dims": lstm_hidden_dims,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "classifier_hidden_dims": classifier_dims,
        "combination_mode": combination_mode,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "early_stopping_patience": early_stopping,
        "scheduler": scheduler,
        "device": device,
        "num_workers": num_workers,
        "seed": seed,
        "save_best_only": save_best_only,
    }

    if model_version == "v3":
        params.update({
            "streams": streams,
            "n_trajs_per_stream": n_trajs_per_stream,
            "traj_projection_dim": traj_projection_dim,
            "n_profile_features": n_profile_features,
            "profile_hidden_dims": profile_hidden_dims,
            "profile_output_dim": profile_output_dim,
        })

    return params


def render_training_progress(history: Dict):
    """Render training progress charts."""
    st.subheader("📈 Training Progress")
    
    if not history or not history.get("train_loss"):
        st.info("No training history available yet.")
        return
    
    n_epochs = len(history["train_loss"])
    epochs = list(range(1, n_epochs + 1))
    
    # Helper function to pad/truncate arrays to match epoch count
    def normalize_array(arr, target_len):
        """Ensure array has exactly target_len elements, padding with None if needed."""
        if arr is None:
            return [None] * target_len
        arr = list(arr)
        if len(arr) < target_len:
            return arr + [None] * (target_len - len(arr))
        elif len(arr) > target_len:
            return arr[:target_len]
        return arr
    
    # Create dataframe for plotting with normalized arrays
    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": normalize_array(history.get("train_loss"), n_epochs),
        "val_loss": normalize_array(history.get("val_loss"), n_epochs),
        "val_accuracy": normalize_array(history.get("val_accuracy"), n_epochs),
        "val_positive_accuracy": normalize_array(history.get("val_positive_accuracy"), n_epochs),
        "val_negative_accuracy": normalize_array(history.get("val_negative_accuracy"), n_epochs),
        "val_identical_score": normalize_array(history.get("val_identical_score"), n_epochs),
        "val_f1": normalize_array(history.get("val_f1"), n_epochs),
        "val_auc": normalize_array(history.get("val_auc"), n_epochs),
        "learning_rate": normalize_array(history.get("learning_rates"), n_epochs)
    })
    
    if ALTAIR_AVAILABLE:
        tab1, tab2, tab3, tab4 = st.tabs(["📉 Loss", "📊 Metrics", "🎯 Split Accuracy", "📈 Learning Rate"])
        
        with tab1:
            # Loss curves
            loss_df = df.melt(
                id_vars=["epoch"],
                value_vars=["train_loss", "val_loss"],
                var_name="type",
                value_name="loss"
            )
            loss_chart = alt.Chart(loss_df).mark_line(point=True).encode(
                x=alt.X("epoch:Q", title="Epoch"),
                y=alt.Y("loss:Q", title="Loss"),
                color=alt.Color("type:N", scale=alt.Scale(
                    domain=["train_loss", "val_loss"],
                    range=["#1f77b4", "#d62728"]
                ))
            ).properties(height=300)
            st.altair_chart(loss_chart, use_container_width=True)
            
            # Best epoch marker
            best_epoch = history.get("best_epoch", 0)
            best_loss = history.get("best_val_loss", 0)
            st.markdown(f"**Best:** Epoch {best_epoch} with val_loss = {best_loss:.4f}")
        
        with tab2:
            # Metrics curves
            metrics_cols = ["val_accuracy", "val_f1", "val_auc"]
            if any(df[col].notna().any() for col in metrics_cols):
                metrics_df = df.melt(
                    id_vars=["epoch"],
                    value_vars=metrics_cols,
                    var_name="metric",
                    value_name="value"
                )
                metrics_chart = alt.Chart(metrics_df).mark_line(point=True).encode(
                    x=alt.X("epoch:Q", title="Epoch"),
                    y=alt.Y("value:Q", title="Value", scale=alt.Scale(domain=[0, 1])),
                    color="metric:N"
                ).properties(height=300)
                st.altair_chart(metrics_chart, use_container_width=True)
                
                # Show final metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Accuracy", f"{df['val_accuracy'].iloc[-1]:.4f}")
                with col2:
                    st.metric("Final F1", f"{df['val_f1'].iloc[-1]:.4f}")
                with col3:
                    st.metric("Final AUC", f"{df['val_auc'].iloc[-1]:.4f}")
        
        with tab3:
            # Split accuracy charts - CRITICAL for discriminator monitoring
            st.markdown("**Split Accuracy:** Shows how well the model performs on positive (same-agent) vs negative (different-agent) pairs separately.")
            # st.caption("⚠️ If these diverge significantly, the model may be biased toward one class!")
            
            # Check which split metrics are available
            has_pos_acc = "val_positive_accuracy" in df.columns and df["val_positive_accuracy"].notna().any()
            has_neg_acc = "val_negative_accuracy" in df.columns and df["val_negative_accuracy"].notna().any()
            has_identical = "val_identical_score" in df.columns and df["val_identical_score"].notna().any()
            
            # Build list of available metrics dynamically
            available_cols = []
            col_names = []
            color_domain = []
            color_range = []
            
            if has_pos_acc:
                available_cols.append("val_positive_accuracy")
                col_names.append("Positive Acc (Same Agent)")
                color_domain.append("Positive Acc (Same Agent)")
                color_range.append("#2ca02c")
            if has_neg_acc:
                available_cols.append("val_negative_accuracy")
                col_names.append("Negative Acc (Diff Agent)")
                color_domain.append("Negative Acc (Diff Agent)")
                color_range.append("#d62728")
            if has_identical:
                available_cols.append("val_identical_score")
                col_names.append("Identical Score")
                color_domain.append("Identical Score")
                color_range.append("#9467bd")
            
            if available_cols:
                # Build the split dataframe with available columns
                split_df = df[["epoch"] + available_cols].copy()
                split_df.columns = ["epoch"] + col_names
                
                split_df_melted = split_df.melt(
                    id_vars=["epoch"],
                    var_name="metric",
                    value_name="value"
                )
                # Remove NaN values for cleaner plotting
                split_df_melted = split_df_melted.dropna(subset=["value"])
                
                split_chart = alt.Chart(split_df_melted).mark_line(point=True).encode(
                    x=alt.X("epoch:Q", title="Epoch"),
                    y=alt.Y("value:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("metric:N", scale=alt.Scale(
                        domain=color_domain,
                        range=color_range
                    ))
                ).properties(height=300, title="Split Accuracy Over Training")
                
                st.altair_chart(split_chart, use_container_width=True)
                
                # Show available metrics
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    if has_pos_acc:
                        pos_acc = df['val_positive_accuracy'].dropna().iloc[-1] if df['val_positive_accuracy'].notna().any() else None
                        if pos_acc is not None:
                            st.metric("➕ Positive Accuracy", f"{pos_acc:.4f}", 
                                      help="Correct predictions on same-agent pairs")
                    else:
                        st.metric("➕ Positive Accuracy", "N/A", 
                                  help="Not recorded in this training run")
                with metric_cols[1]:
                    if has_neg_acc:
                        neg_acc = df['val_negative_accuracy'].dropna().iloc[-1] if df['val_negative_accuracy'].notna().any() else None
                        if neg_acc is not None:
                            st.metric("➖ Negative Accuracy", f"{neg_acc:.4f}",
                                      help="Correct predictions on different-agent pairs")
                    else:
                        st.metric("➖ Negative Accuracy", "N/A",
                                  help="Not recorded in this training run")
                with metric_cols[2]:
                    if has_identical:
                        id_score = df['val_identical_score'].dropna().iloc[-1] if df['val_identical_score'].notna().any() else None
                        if id_score is not None:
                            st.metric("🔄 Identical Score", f"{id_score:.4f}",
                                      help="Score when comparing trajectory to itself (should be ~1.0)")
                    else:
                        st.metric("🔄 Identical Score", "N/A",
                                  help="Not recorded in this training run")
                
                # Imbalance warning (only if both pos and neg accuracy available)
                if has_pos_acc and has_neg_acc and len(df) > 0:
                    pos_acc = df['val_positive_accuracy'].dropna().iloc[-1]
                    neg_acc = df['val_negative_accuracy'].dropna().iloc[-1]
                    imbalance = abs(pos_acc - neg_acc)
                    if imbalance > 0.3:
                        st.error(f"⚠️ **Severe Imbalance Detected:** {imbalance:.2f} gap between positive and negative accuracy. "
                                f"The model may be biased toward predicting {'same-agent' if pos_acc > neg_acc else 'different-agent'}.")
                    elif imbalance > 0.15:
                        st.warning(f"⚡ **Moderate Imbalance:** {imbalance:.2f} gap. Monitor this as training continues.")
                
                # Note about missing metrics
                if not (has_pos_acc and has_neg_acc):
                    st.info("ℹ️ Positive/Negative accuracy breakdown not available for this training run. "
                           "Showing available metrics only.")
            else:
                st.info("Split accuracy data not available. This may be from an older training run.")
        
        with tab4:
            if df["learning_rate"].notna().any():
                lr_chart = alt.Chart(df).mark_line(point=True, color="#2ca02c").encode(
                    x=alt.X("epoch:Q", title="Epoch"),
                    y=alt.Y("learning_rate:Q", title="Learning Rate", scale=alt.Scale(type="log"))
                ).properties(height=300)
                st.altair_chart(lr_chart, use_container_width=True)
    else:
        # Fallback without altair
        st.line_chart(df[["train_loss", "val_loss"]])


def render_evaluation_results(results: Dict):
    """Render evaluation results."""
    st.subheader("🎯 Evaluation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{results.get('accuracy', 0):.4f}")
    with col2:
        st.metric("F1 Score", f"{results.get('f1', 0):.4f}")
    with col3:
        st.metric("Precision", f"{results.get('precision', 0):.4f}")
    with col4:
        st.metric("Recall", f"{results.get('recall', 0):.4f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ROC AUC", f"{results.get('auc', 0):.4f}")
    with col2:
        st.metric("Samples", f"{results.get('n_samples', 0):,}")
    
    # Confusion matrix
    if "confusion_matrix" in results:
        st.markdown("**Confusion Matrix:**")
        cm = np.array(results["confusion_matrix"])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0 (diff)", "Actual 1 (same)"],
            columns=["Pred 0 (diff)", "Pred 1 (same)"]
        )
        st.dataframe(cm_df)
        
        # Per-class metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Negative Accuracy", f"{results.get('negative_accuracy', 0):.4f}")
        with col2:
            st.metric("Positive Accuracy", f"{results.get('positive_accuracy', 0):.4f}")


def render_checkpoint_explorer():
    """Render checkpoint exploration UI."""
    st.subheader("📂 Checkpoint Explorer")
    
    checkpoint_base = st.text_input(
        "Checkpoint directory",
        value=str(DEFAULT_CHECKPOINT_DIR)
    )
    
    checkpoints = list_checkpoints(Path(checkpoint_base))
    
    if not checkpoints:
        st.info("No trained models found. Train a model first.")
        return None
    
    checkpoint_options = {cp.name: cp for cp in checkpoints}
    selected_name = st.selectbox(
        "Select checkpoint",
        options=list(checkpoint_options.keys())
    )
    selected_checkpoint = checkpoint_options[selected_name]
    
    # Show training history
    history = load_training_history(selected_checkpoint)
    if history:
        render_training_progress(history)
    
    # Load config
    config_path = selected_checkpoint / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        with st.expander("📋 Training Configuration"):
            st.json(config)
    
    return selected_checkpoint


def start_training(dataset_path: Path, params: Dict, experiment_name: str, progress_container=None) -> None:
    """Start training with optional progress tracking for Streamlit.
    
    Args:
        dataset_path: Path to dataset directory
        params: Hyperparameter dictionary
        experiment_name: Name for this experiment
        progress_container: Optional Streamlit container for progress updates
    """
    # Import from local modules (when running from model/ directory)
    from model import (
        SiameseLSTMDiscriminator,
        SiameseLSTMDiscriminatorV2,
        MultiStreamSiameseDiscriminator,
    )
    from dataset import TrajectoryPairDataset, create_data_loaders, load_dataset_from_directory
    from trainer import Trainer, TrainingConfig
    
    # Load data
    datasets = load_dataset_from_directory(dataset_path)
    train_loader, val_loader = create_data_loaders(
        datasets["train"],
        datasets["val"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"]
    )
    
    # Load dataset metadata if available
    dataset_info = {
        'path': str(dataset_path),
        'name': dataset_path.name,
    }
    metadata_path = dataset_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            dataset_info['metadata'] = metadata
            # Extract key counts from metadata for easy access
            if 'counts' in metadata:
                dataset_info['total_pairs'] = metadata['counts'].get('total_pairs')
                dataset_info['positive_pairs'] = metadata['counts'].get('positive_pairs')
                dataset_info['negative_pairs'] = metadata['counts'].get('negative_pairs')
            if 'config' in metadata:
                dataset_info['source_data_path'] = metadata['config'].get('data_path')
    
    # Create model based on version
    model_version = params.get("model_version", "v1")
    if model_version == "v3":
        model = MultiStreamSiameseDiscriminator(
            lstm_hidden_dims=params["lstm_hidden_dims"],
            dropout=params["dropout"],
            bidirectional=params["bidirectional"],
            classifier_hidden_dims=params["classifier_hidden_dims"],
            combination_mode=params.get("combination_mode", "concatenation"),
            n_profile_features=params.get("n_profile_features", 11),
            profile_hidden_dims=params.get("profile_hidden_dims", (64, 32)),
            profile_output_dim=params.get("profile_output_dim", 8),
            streams=params.get("streams", ("seeking", "driving", "profile")),
            n_trajs_per_stream=params.get("n_trajs_per_stream", 5),
            traj_projection_dim=params.get("traj_projection_dim", 48),
        )
    elif model_version == "v2":
        model = SiameseLSTMDiscriminatorV2(
            lstm_hidden_dims=params["lstm_hidden_dims"],
            dropout=params["dropout"],
            bidirectional=params["bidirectional"],
            classifier_hidden_dims=params["classifier_hidden_dims"],
            combination_mode=params.get("combination_mode", "difference"),
        )
    else:
        model = SiameseLSTMDiscriminator(
            lstm_hidden_dims=params["lstm_hidden_dims"],
            dropout=params["dropout"],
            bidirectional=params["bidirectional"],
            classifier_hidden_dims=params["classifier_hidden_dims"],
        )

    # Create config
    from dataclasses import fields
    valid_fields = {f.name for f in fields(TrainingConfig)}
    config = TrainingConfig(
        **{k: v for k, v in params.items() if k in valid_fields},
        checkpoint_dir=str(DEFAULT_CHECKPOINT_DIR)
    )
    
    # Generate experiment name if not provided (None or empty)
    if not experiment_name:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        # Clean up the experiment name (strip whitespace, ensure it's valid)
        experiment_name = experiment_name.strip()
        if not experiment_name:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=experiment_name,
        dataset_info=dataset_info
    )
    
    # Setup progress callback if container provided
    if progress_container is not None:
        epoch_times = []
        start_time = time.time()
        
        # Create progress elements in the container
        with progress_container:
            st.info(f"🚀 Starting training with experiment name: **{experiment_name}**")
            st.caption(f"📁 Checkpoint directory: `{trainer.checkpoint_dir}`")
            progress_bar = st.progress(0, text="Initializing training...")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                epoch_display = st.empty()
            with col2:
                time_elapsed_display = st.empty()
            with col3:
                time_remaining_display = st.empty()
            with col4:
                eta_display = st.empty()
            
            st.markdown("---")
            
            # STOP TRAINING button - uses session state to signal stop
            stop_col1, stop_col2 = st.columns([1, 4])
            with stop_col1:
                if st.button("🛑 Stop Training", type="secondary", key="stop_training_btn"):
                    st.session_state['stop_training_requested'] = True
                    st.warning("⏳ Stop requested. Training will stop after current epoch...")
            with stop_col2:
                stop_status = st.empty()
            
            st.markdown("---")
            
            # Primary metrics row
            metric_cols = st.columns(5)
            with metric_cols[0]:
                train_loss_display = st.empty()
            with metric_cols[1]:
                val_loss_display = st.empty()
            with metric_cols[2]:
                accuracy_display = st.empty()
            with metric_cols[3]:
                f1_display = st.empty()
            with metric_cols[4]:
                auc_display = st.empty()
            
            # Split accuracy and identical score row (NEW)
            st.markdown("**Detailed Accuracy Breakdown:**")
            detail_cols = st.columns(4)
            with detail_cols[0]:
                pos_acc_display = st.empty()
            with detail_cols[1]:
                neg_acc_display = st.empty()
            with detail_cols[2]:
                identical_display = st.empty()
            with detail_cols[3]:
                split_balance_display = st.empty()
            
            # Status and best epoch
            status_display = st.empty()
            
            # Live chart placeholder
            chart_placeholder = st.empty()
        
        def format_time(seconds: float) -> str:
            """Format seconds into human-readable string."""
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                mins = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{mins}m {secs}s"
            else:
                hours = int(seconds // 3600)
                mins = int((seconds % 3600) // 60)
                return f"{hours}h {mins}m"
        
        def progress_callback(epoch, total_epochs, epoch_time, train_loss, val_metrics, identical_metrics, is_best, should_stop):
            """Callback to update Streamlit progress elements."""
            epoch_times.append(epoch_time)
            
            # Check for stop request from UI button
            stop_requested = st.session_state.get('stop_training_requested', False)
            
            # Calculate timing
            elapsed = time.time() - start_time
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = total_epochs - epoch
            estimated_remaining = avg_epoch_time * remaining_epochs
            eta = datetime.now() + timedelta(seconds=estimated_remaining)
            
            # Update progress bar
            progress_pct = epoch / total_epochs
            progress_bar.progress(
                progress_pct,
                text=f"Epoch {epoch}/{total_epochs} ({progress_pct*100:.0f}%)"
            )
            
            # Update timing displays
            epoch_display.metric("📊 Epoch", f"{epoch} / {total_epochs}")
            time_elapsed_display.metric("⏱️ Elapsed", format_time(elapsed))
            time_remaining_display.metric("⏳ Remaining", format_time(estimated_remaining) if remaining_epochs > 0 else "Done!")
            eta_display.metric("🏁 ETA", eta.strftime("%H:%M:%S") if remaining_epochs > 0 else "Complete")
            
            # Update primary metrics
            train_loss_display.metric("Train Loss", f"{train_loss:.4f}")
            val_loss_display.metric("Val Loss", f"{val_metrics['loss']:.4f}")
            accuracy_display.metric("Accuracy", f"{val_metrics.get('accuracy', 0):.4f}")
            f1_display.metric("F1 Score", f"{val_metrics.get('f1', 0):.4f}")
            auc_display.metric("ROC AUC", f"{val_metrics.get('auc', 0):.4f}")
            
            # Update detailed accuracy breakdown (NEW)
            pos_acc = val_metrics.get('positive_accuracy', 0)
            neg_acc = val_metrics.get('negative_accuracy', 0)
            identical_score = identical_metrics.get('identical_mean', 0)
            
            # Color-code based on performance
            pos_delta = None if epoch == 1 else pos_acc - (trainer.history.val_positive_accuracy[-2] if len(trainer.history.val_positive_accuracy) > 1 else pos_acc)
            neg_delta = None if epoch == 1 else neg_acc - (trainer.history.val_negative_accuracy[-2] if len(trainer.history.val_negative_accuracy) > 1 else neg_acc)
            
            pos_acc_display.metric(
                "➕ Positive Acc",
                f"{pos_acc:.4f}",
                delta=f"{pos_delta:+.4f}" if pos_delta is not None else None,
                help="Accuracy on same-agent pairs (label=1). Should be high."
            )
            neg_acc_display.metric(
                "➖ Negative Acc",
                f"{neg_acc:.4f}",
                delta=f"{neg_delta:+.4f}" if neg_delta is not None else None,
                help="Accuracy on different-agent pairs (label=0). Critical for discriminator - watch this!"
            )
            identical_display.metric(
                "🔄 Identical Score",
                f"{identical_score:.4f}",
                delta="⚠️ LOW" if identical_score < 0.5 else ("✓" if identical_score > 0.9 else None),
                help="Mean score when comparing trajectory to itself. Should be ~1.0"
            )
            
            # Show split balance warning
            split_diff = abs(pos_acc - neg_acc)
            if split_diff > 0.3:
                split_balance_display.error(f"⚠️ Imbalance: {split_diff:.2f}")
            elif split_diff > 0.15:
                split_balance_display.warning(f"⚡ Gap: {split_diff:.2f}")
            else:
                split_balance_display.success(f"✓ Balanced: {split_diff:.2f}")
            
            # Status message
            if stop_requested:
                status_display.error(f"🛑 Stop requested by user. Saving checkpoints...")
                stop_status.info("Checkpoints saved. Training stopped.")
            elif should_stop:
                status_display.warning(f"🛑 Early stopping triggered! Best epoch: {trainer.history.best_epoch}")
            elif is_best:
                status_display.success(f"⭐ New best model! Val Loss: {val_metrics['loss']:.4f}")
            else:
                patience_left = config.early_stopping_patience - trainer.early_stopping.counter
                status_display.info(f"Training... (Early stopping patience: {patience_left} epochs remaining)")
            
            # Update live chart if we have enough data
            if ALTAIR_AVAILABLE and len(epoch_times) > 1:
                history = trainer.history
                chart_df = pd.DataFrame({
                    "epoch": list(range(1, len(history.train_loss) + 1)),
                    "train_loss": history.train_loss,
                    "val_loss": history.val_loss,
                })
                
                # Melt for Altair
                chart_df_melted = chart_df.melt(
                    id_vars=["epoch"],
                    value_vars=["train_loss", "val_loss"],
                    var_name="type",
                    value_name="loss"
                )
                
                chart = alt.Chart(chart_df_melted).mark_line(point=True).encode(
                    x=alt.X("epoch:Q", title="Epoch", scale=alt.Scale(domain=[1, total_epochs])),
                    y=alt.Y("loss:Q", title="Loss"),
                    color=alt.Color("type:N", scale=alt.Scale(
                        domain=["train_loss", "val_loss"],
                        range=["#1f77b4", "#d62728"]
                    ), legend=alt.Legend(title="Loss Type"))
                ).properties(height=200, title="Training Progress")
                
                chart_placeholder.altair_chart(chart, use_container_width=True)
            
            # Return False to stop training if user requested stop
            if stop_requested:
                # Clear the stop flag for next training run
                st.session_state['stop_training_requested'] = False
                return False  # Signal trainer to stop
            
            return True  # Continue training
        
        # Initialize stop flag
        st.session_state['stop_training_requested'] = False
        
        # Train with progress callback
        trainer.train(verbose=True, progress_callback=progress_callback)
        
        # Final update (progress_pct was from last callback, just use the bar state)
        if st.session_state.get('stop_training_requested', False):
            # Don't update bar - let it stay at last position
            st.session_state['stop_training_requested'] = False
        else:
            progress_bar.progress(1.0, text="✅ Training Complete!")
        
    else:
        # Train without progress tracking
        trainer.train(verbose=True)
    
    # Build training summary
    history = trainer.history
    total_time = sum(history.epoch_times)
    epochs_completed = len(history.train_loss)
    
    summary = {
        "checkpoint_dir": str(trainer.checkpoint_dir),
        "experiment_name": experiment_name,
        "epochs_completed": epochs_completed,
        "epochs_configured": params["epochs"],
        "early_stopped": epochs_completed < params["epochs"],
        "total_time_seconds": total_time,
        "avg_epoch_time": total_time / epochs_completed if epochs_completed > 0 else 0,
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "final_train_loss": history.train_loss[-1] if history.train_loss else None,
        "final_val_loss": history.val_loss[-1] if history.val_loss else None,
        "final_accuracy": history.val_accuracy[-1] if history.val_accuracy else None,
        "final_f1": history.val_f1[-1] if history.val_f1 else None,
        "final_auc": history.val_auc[-1] if history.val_auc else None,
        "initial_lr": history.learning_rates[0] if history.learning_rates else None,
        "final_lr": history.learning_rates[-1] if history.learning_rates else None,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "device": str(trainer.device),
        "model_params": sum(p.numel() for p in model.parameters()),
    }
    
    return trainer.checkpoint_dir, summary


def render_training_summary(summary: Dict):
    """Render a training summary after completion."""
    st.subheader("📋 Training Summary")
    
    # Format time helper
    def fmt_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {mins}m {secs}s"
    
    # Training completion status
    if summary.get("early_stopped"):
        st.info(f"🛑 Training stopped early at epoch {summary['epochs_completed']} / {summary['epochs_configured']} (early stopping triggered)")
    else:
        st.success(f"✅ Training completed all {summary['epochs_completed']} epochs")
    
    # Time metrics
    st.markdown("#### ⏱️ Time")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Training Time", fmt_time(summary.get("total_time_seconds", 0)))
    with col2:
        st.metric("Avg Time per Epoch", fmt_time(summary.get("avg_epoch_time", 0)))
    with col3:
        st.metric("Device", summary.get("device", "Unknown").upper())
    
    # Best model info
    st.markdown("#### 🏆 Best Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Epoch", summary.get("best_epoch", "N/A"))
    with col2:
        best_loss = summary.get("best_val_loss")
        st.metric("Best Val Loss", f"{best_loss:.4f}" if best_loss else "N/A")
    with col3:
        params = summary.get("model_params", 0)
        if params > 1_000_000:
            param_str = f"{params / 1_000_000:.2f}M"
        elif params > 1_000:
            param_str = f"{params / 1_000:.1f}K"
        else:
            param_str = str(params)
        st.metric("Model Parameters", param_str)
    
    # Final metrics
    st.markdown("#### 📊 Final Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        final_train = summary.get("final_train_loss")
        st.metric("Train Loss", f"{final_train:.4f}" if final_train else "N/A")
    with col2:
        final_val = summary.get("final_val_loss")
        st.metric("Val Loss", f"{final_val:.4f}" if final_val else "N/A")
    with col3:
        final_acc = summary.get("final_accuracy")
        st.metric("Accuracy", f"{final_acc:.4f}" if final_acc else "N/A")
    with col4:
        final_f1 = summary.get("final_f1")
        st.metric("F1 Score", f"{final_f1:.4f}" if final_f1 else "N/A")
    with col5:
        final_auc = summary.get("final_auc")
        st.metric("ROC AUC", f"{final_auc:.4f}" if final_auc else "N/A")
    
    # Dataset info
    st.markdown("#### 📁 Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Train Samples", f"{summary.get('train_samples', 0):,}")
    with col2:
        st.metric("Val Samples", f"{summary.get('val_samples', 0):,}")
    with col3:
        total = summary.get('train_samples', 0) + summary.get('val_samples', 0)
        st.metric("Total Samples", f"{total:,}")
    
    # Learning rate schedule
    if summary.get("initial_lr") and summary.get("final_lr"):
        st.markdown("#### 📈 Learning Rate")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Initial LR", f"{summary['initial_lr']:.2e}")
        with col2:
            st.metric("Final LR", f"{summary['final_lr']:.2e}")
        with col3:
            reduction = summary['initial_lr'] / summary['final_lr'] if summary['final_lr'] > 0 else 1
            st.metric("LR Reduction", f"{reduction:.1f}x" if reduction > 1 else "None")
    
    # Checkpoint location
    st.markdown("#### 💾 Checkpoint")
    st.code(summary.get("checkpoint_dir", "Unknown"), language="bash")
    
    # Show raw summary in expander
    with st.expander("🔍 Raw Summary Data"):
        st.json(summary)


def main():
    st.title("🧠 Discriminator Training Dashboard")
    
    check_dependencies()
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏋️ Train New Model", "📊 View Results", "🔍 Evaluate Model"]
    )
    
    if page == "🏋️ Train New Model":
        st.markdown("""
        Train a new Siamese LSTM discriminator model for trajectory pair classification.
        
        **Workflow:**
        1. Select a dataset (generated with the Dataset Generator tool)
        2. Configure hyperparameters
        3. Start training
        4. Monitor progress
        """)
        
        st.divider()
        
        # Dataset selection
        dataset_path, dataset_info = render_dataset_selector()
        
        if dataset_path is None:
            st.stop()
        
        st.divider()
        
        # Hyperparameters — auto-detect if dataset is multi-stream
        is_multi_stream = dataset_info.get("is_multi_stream", False) if dataset_info else False
        params = render_hyperparameters(is_multi_stream=is_multi_stream)
        
        st.divider()
        
        # Experiment name and time estimation
        st.subheader("🏷️ Experiment")
        
        # Initialize session state for experiment name if not exists
        if 'experiment_name_input' not in st.session_state:
            st.session_state.experiment_name_input = ""
        
        # Use session state with a unique key to persist user input across reruns
        experiment_name = st.text_input(
            "Experiment name",
            key="experiment_name_input",
            placeholder=f"e.g., my_model_v1 (or leave empty for timestamp)",
            help="Custom name for this experiment. Leave empty to auto-generate a timestamp."
        )
        
        # If empty, will generate timestamp at training time (handled in start_training)
        if not experiment_name.strip():
            experiment_name = None
            st.caption(f"💡 Will use default: `{datetime.now().strftime('%Y%m%d_%H%M%S')}`")
        else:
            st.caption(f"✓ Checkpoint directory: `checkpoints/{experiment_name.strip()}/`")
        
        # Training time estimation
        if dataset_info:
            n_train = dataset_info.get("train_samples", 0)
            n_val = dataset_info.get("val_samples", 0)
            batch_size = params["batch_size"]
            epochs = params["epochs"]
            
            # Estimate batches per epoch
            train_batches = (n_train + batch_size - 1) // batch_size
            val_batches = (n_val + batch_size - 1) // batch_size
            
            # Rough time estimate (will be refined during training)
            # These are rough estimates - actual time depends on hardware
            # Assume ~0.05s per batch for LSTM on GPU, ~0.2s on CPU
            device = params.get("device", "auto")
            if device == "auto":
                import torch
                has_cuda = torch.cuda.is_available()
            else:
                has_cuda = device == "cuda"
            
            time_per_batch = 0.05 if has_cuda else 0.2
            est_epoch_time = (train_batches + val_batches) * time_per_batch
            est_total_time = est_epoch_time * epochs
            
            with st.expander("⏱️ Training Time Estimate", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train Batches/Epoch", f"{train_batches:,}")
                with col2:
                    st.metric("Val Batches/Epoch", f"{val_batches:,}")
                with col3:
                    def _fmt_time(secs):
                        if secs < 60:
                            return f"{secs:.0f}s"
                        elif secs < 3600:
                            return f"{secs/60:.1f}m"
                        else:
                            return f"{secs/3600:.1f}h"
                    st.metric("Est. per Epoch", _fmt_time(est_epoch_time))
                with col4:
                    st.metric("Est. Total", _fmt_time(est_total_time))
                
                st.caption(f"💡 Estimates assume {'GPU' if has_cuda else 'CPU'} training. Actual times may vary. "
                          f"Early stopping (patience={params['early_stopping_patience']}) may end training sooner.")
        
        st.divider()
        
        # Training buttons
        col1, col2 = st.columns(2)
        with col1:
            start_clicked = st.button("🚀 Start Training", type="primary")
        
        with col2:
            # Show CLI command
            with st.expander("🖥️ CLI Command"):
                exp_name_arg = f'--experiment-name "{experiment_name}"' if experiment_name else ''
                model_version = params.get("model_version", "v1")

                cmd_parts = [
                    f'python -m discriminator.model.train \\',
                    f'    --model-version {model_version} \\',
                    f'    --data-dir "{dataset_path}" \\',
                    f'    --lstm-hidden-dims "{",".join(map(str, params["lstm_hidden_dims"]))}" \\',
                    f'    --dropout {params["dropout"]} \\',
                ]
                if not params['bidirectional']:
                    cmd_parts.append('    --no-bidirectional \\')
                cmd_parts.append(
                    f'    --classifier-dims "{",".join(map(str, params["classifier_hidden_dims"]))}" \\'
                )
                if params.get("combination_mode"):
                    cmd_parts.append(f'    --combination-mode {params["combination_mode"]} \\')
                if model_version == "v3":
                    cmd_parts.extend([
                        f'    --streams "{",".join(params.get("streams", ("seeking","driving","profile")))}" \\',
                        f'    --n-trajs-per-stream {params.get("n_trajs_per_stream", 5)} \\',
                        f'    --traj-projection-dim {params.get("traj_projection_dim", 48)} \\',
                        f'    --profile-hidden-dims "{",".join(map(str, params.get("profile_hidden_dims", (64,32))))}" \\',
                        f'    --profile-output-dim {params.get("profile_output_dim", 8)} \\',
                        f'    --n-profile-features {params.get("n_profile_features", 11)} \\',
                    ])
                cmd_parts.extend([
                    f'    --epochs {params["epochs"]} \\',
                    f'    --batch-size {params["batch_size"]} \\',
                    f'    --lr {params["learning_rate"]} \\',
                    f'    --early-stopping {params["early_stopping_patience"]} \\',
                    f'    --scheduler {params["scheduler"]} \\',
                ])
                if exp_name_arg:
                    cmd_parts.append(f'    {exp_name_arg} \\')
                cmd_parts.append(f'    --output "{DEFAULT_CHECKPOINT_DIR}"')

                st.code('\n'.join(cmd_parts), language="bash")
        
        # Progress container (placed below buttons, will be populated during training)
        progress_container = st.container()
        
        # Summary container (placed after progress, will be populated after training)
        summary_container = st.container()
        
        if start_clicked:
            try:
                checkpoint_dir, training_summary = start_training(
                    dataset_path,
                    params,
                    experiment_name,
                    progress_container=progress_container
                )
                st.session_state["last_checkpoint"] = str(checkpoint_dir)
                st.session_state["last_training_summary"] = training_summary
                st.balloons()  # Celebration animation!
                
                # Render training summary
                with summary_container:
                    st.divider()
                    render_training_summary(training_summary)
                    
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    elif page == "📊 View Results":
        st.markdown("View training results and metrics from previous experiments.")
        
        st.divider()
        
        checkpoint = render_checkpoint_explorer()
        
    elif page == "🔍 Evaluate Model":
        st.markdown("Evaluate a trained model on a test dataset.")
        
        st.divider()
        
        # Select checkpoint
        checkpoint = render_checkpoint_explorer()
        
        if checkpoint is None:
            st.stop()
        
        st.divider()
        
        # Select test data
        st.subheader("📁 Test Data")
        dataset_path, dataset_info = render_dataset_selector()
        
        if dataset_path is None:
            st.stop()
        
        # Check for test split
        test_path = dataset_path / "test.npz"
        use_val_as_test = not test_path.exists()
        
        if use_val_as_test:
            st.warning("No test.npz found. Using val.npz for evaluation.")
            eval_path = dataset_path / "val.npz"
        else:
            eval_path = test_path
        
        if st.button("🔍 Evaluate Model", type="primary"):
            with st.spinner("Evaluating..."):
                try:
                    from dataset import (
                        TrajectoryPairDataset,
                        MultiStreamPairDataset,
                        load_dataset_from_directory,
                    )
                    from trainer import load_model_from_checkpoint
                    from torch.utils.data import DataLoader

                    # Load model
                    model, ckpt = load_model_from_checkpoint(checkpoint / "best.pt")
                    model_version = ckpt.get('model_config', {}).get('model_version', 'v1')

                    # Auto-detect dataset type
                    with np.load(eval_path) as probe:
                        is_ms = "driving_1" in probe
                    if is_ms:
                        test_dataset = MultiStreamPairDataset(eval_path)
                    else:
                        test_dataset = TrajectoryPairDataset(eval_path)
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model.to(device)
                    model.eval()

                    # Manual evaluation
                    all_preds = []
                    all_probs = []
                    all_labels = []

                    with torch.no_grad():
                        for batch in test_loader:
                            x1 = batch['x1'].to(device)
                            x2 = batch['x2'].to(device)
                            mask1 = batch['mask1'].to(device)
                            mask2 = batch['mask2'].to(device)
                            labels = batch['label']

                            # Build multi-stream kwargs if present
                            kwargs = {}
                            if 'driving_1' in batch:
                                kwargs['driving_1'] = batch['driving_1'].to(device)
                                kwargs['driving_2'] = batch['driving_2'].to(device)
                                kwargs['mask_d1'] = batch['mask_d1'].to(device)
                                kwargs['mask_d2'] = batch['mask_d2'].to(device)
                            if 'profile_1' in batch:
                                kwargs['profile_1'] = batch['profile_1'].to(device)
                                kwargs['profile_2'] = batch['profile_2'].to(device)

                            outputs = model(x1, x2, mask1, mask2, **kwargs).squeeze(-1)
                            probs = outputs.cpu().numpy()
                            preds = (probs >= 0.5).astype(float)

                            all_probs.extend(probs.tolist())
                            all_preds.extend(preds.tolist())
                            all_labels.extend(labels.numpy().tolist())
                    
                    # Compute metrics
                    if SKLEARN_AVAILABLE:
                        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
                        
                        results = {
                            'accuracy': accuracy_score(all_labels, all_preds),
                            'f1': f1_score(all_labels, all_preds, zero_division=0),
                            'precision': precision_score(all_labels, all_preds, zero_division=0),
                            'recall': recall_score(all_labels, all_preds, zero_division=0),
                            'auc': roc_auc_score(all_labels, all_probs),
                            'n_samples': len(all_labels),
                            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
                        }
                        
                        cm = np.array(results['confusion_matrix'])
                        tn, fp, fn, tp = cm.ravel()
                        results['negative_accuracy'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                        results['positive_accuracy'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                        
                        render_evaluation_results(results)
                    else:
                        correct = sum(p == l for p, l in zip(all_preds, all_labels))
                        st.metric("Accuracy", f"{correct / len(all_labels):.4f}")
                    
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
