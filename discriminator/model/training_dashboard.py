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
                info[f"{split}_seq_len"] = data["x1"].shape[1]
                info[f"{split}_features"] = data["x1"].shape[2]
            info[f"{split}_size_mb"] = npz_path.stat().st_size / (1024 * 1024)
    
    return info


def list_available_datasets(base_dir: Path) -> List[Path]:
    """List available dataset directories."""
    datasets = []
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
            help="Base directory containing dataset subdirectories"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("🔄 Refresh")
    
    dataset_dir = Path(dataset_dir_str)
    
    # List available datasets
    datasets = list_available_datasets(dataset_dir)
    
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
        dataset_options = {d.name: d for d in datasets}
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
                if info.get("train_seq_len"):
                    st.metric("Sequence Length", info.get('train_seq_len', 0))
                    st.caption(f"Features: {info.get('train_features', 0)}")
            
            # Show config from metadata if available
            if "metadata" in info:
                cfg = info["metadata"].get("config", {})
                st.markdown("**Configuration:**")
                st.json(cfg)
        
        return selected_dataset, info
    
    return None, None


def render_hyperparameters():
    """Render hyperparameter configuration UI."""
    st.subheader("⚙️ Hyperparameters")
    
    with st.expander("Model Architecture", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            # LSTM hidden dims - text input for advanced users, default follows ST-SiameseNet
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
    
    return {
        "lstm_hidden_dims": lstm_hidden_dims,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "classifier_hidden_dims": classifier_dims,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "early_stopping_patience": early_stopping,
        "scheduler": scheduler,
        "device": device,
        "num_workers": num_workers,
        "seed": seed,
        "save_best_only": save_best_only
    }


def render_training_progress(history: Dict):
    """Render training progress charts."""
    st.subheader("📈 Training Progress")
    
    if not history or not history.get("train_loss"):
        st.info("No training history available yet.")
        return
    
    epochs = list(range(1, len(history["train_loss"]) + 1))
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_accuracy": history.get("val_accuracy", []),
        "val_f1": history.get("val_f1", []),
        "val_auc": history.get("val_auc", []),
        "learning_rate": history.get("learning_rates", [])
    })
    
    if ALTAIR_AVAILABLE:
        tab1, tab2, tab3 = st.tabs(["📉 Loss", "📊 Metrics", "📈 Learning Rate"])
        
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
    from model import SiameseLSTMDiscriminator
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
    
    # Create model
    model = SiameseLSTMDiscriminator(
        lstm_hidden_dims=params["lstm_hidden_dims"],
        dropout=params["dropout"],
        bidirectional=params["bidirectional"],
        classifier_hidden_dims=params["classifier_hidden_dims"]
    )
    
    # Create config
    from dataclasses import fields
    valid_fields = {f.name for f in fields(TrainingConfig)}
    config = TrainingConfig(
        **{k: v for k, v in params.items() if k in valid_fields},
        checkpoint_dir=str(DEFAULT_CHECKPOINT_DIR)
    )
    
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
            
            # Metrics row
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
            
            # Update metrics
            train_loss_display.metric("Train Loss", f"{train_loss:.4f}")
            val_loss_display.metric("Val Loss", f"{val_metrics['loss']:.4f}")
            accuracy_display.metric("Accuracy", f"{val_metrics.get('accuracy', 0):.4f}")
            f1_display.metric("F1 Score", f"{val_metrics.get('f1', 0):.4f}")
            auc_display.metric("ROC AUC", f"{val_metrics.get('auc', 0):.4f}")
            
            # Status message
            if should_stop:
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
            
            return True  # Continue training
        
        # Train with progress callback
        trainer.train(verbose=True, progress_callback=progress_callback)
        
        # Final update
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
        
        # Hyperparameters
        params = render_hyperparameters()
        
        st.divider()
        
        # Experiment name and time estimation
        st.subheader("🏷️ Experiment")
        default_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = st.text_input("Experiment name", value=default_name)
        
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
                cmd = f"""python train.py \\
    --data-dir "{dataset_path}" \\
    --lstm_hidden_dims {params['lstm_hidden_dims']} \\
    --dropout {params['dropout']} \\
    {"--no-bidirectional" if not params['bidirectional'] else ""} \\
    --classifier-dims "{','.join(map(str, params['classifier_hidden_dims']))}" \\
    --epochs {params['epochs']} \\
    --batch-size {params['batch_size']} \\
    --lr {params['learning_rate']} \\
    --early-stopping {params['early_stopping_patience']} \\
    --scheduler {params['scheduler']} \\
    --experiment-name "{experiment_name}" \\
    --output "{DEFAULT_CHECKPOINT_DIR}"
"""
                st.code(cmd, language="bash")
        
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
                    from dataset import TrajectoryPairDataset
                    from trainer import load_model_from_checkpoint
                    from torch.utils.data import DataLoader
                    
                    # Load model
                    model, _ = load_model_from_checkpoint(checkpoint / "best.pt")
                    
                    # Load test data
                    test_dataset = TrajectoryPairDataset(eval_path)
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                    
                    # Create dummy trainer for evaluation
                    # (This is a workaround - ideally we'd have a standalone evaluate function)
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
                            
                            outputs = model(x1, x2, mask1, mask2).squeeze(-1)
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
