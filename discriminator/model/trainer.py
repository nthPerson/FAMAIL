"""Training utilities for the discriminator model.

Provides a Trainer class with:
- Training loop with BCE loss
- Validation with metrics (accuracy, ROC AUC, F1)
- Early stopping
- Checkpointing
- Learning rate scheduling
- Logging
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader

# Metrics
try:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Plot generation — optional, degrades gracefully if matplotlib is missing.
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; safe in training jobs
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model architecture
    lstm_hidden_dims: Tuple[int, ...] = (200, 100)
    dropout: float = 0.2
    bidirectional: bool = True
    classifier_hidden_dims: Tuple[int, ...] = (64, 32, 8)
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # LR scheduling
    scheduler: str = "plateau"  # "plateau", "cosine", or "none"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    
    # Misc
    device: str = "auto"  # "auto", "cuda", "cpu"
    # num_workers=4 + pin_memory=True (in dataset.py) + non_blocking=True on
    # .to() calls (in this file) together keep the GPU saturated on the
    # Siamese workload. The prior default of 0 produced ~0% GPU utilization
    # even with cuda device because the data loader ran synchronously in
    # the main thread between batches.
    num_workers: int = 4
    # Mixed-precision training — wraps forward pass in autocast() and uses
    # GradScaler for the backward pass. Gives ~1.5-2x speedup on Ampere+
    # GPUs via Tensor Cores. Automatically disabled on CPU. The BCELoss
    # final step is forced to FP32 to avoid the sigmoid→log(0) failure
    # mode that FP16 precision limits can trigger.
    amp: bool = True
    seed: int = 42
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "TrainingConfig":
        # Handle tuple conversion
        if "classifier_hidden_dims" in d and isinstance(d["classifier_hidden_dims"], list):
            d["classifier_hidden_dims"] = tuple(d["classifier_hidden_dims"])
        if "lstm_hidden_dims" in d and isinstance(d["lstm_hidden_dims"], list):
            d["lstm_hidden_dims"] = tuple(d["lstm_hidden_dims"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingHistory:
    """Records training history."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    val_positive_accuracy: List[float] = field(default_factory=list)  # Accuracy on positive (same-agent) pairs
    val_negative_accuracy: List[float] = field(default_factory=list)  # Accuracy on negative (diff-agent) pairs
    val_identical_score: List[float] = field(default_factory=list)  # Mean score for identical trajectories
    val_f1: List[float] = field(default_factory=list)
    val_auc: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "TrainingHistory":
        return cls(**d)


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        
    def __call__(self, value: float) -> bool:
        """Check if training should stop.
        
        Args:
            value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
            
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
            
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
        return self.should_stop


class Trainer:
    """Trainer for the discriminator model."""
    
    def __init__(self,
                 model: nn.Module,
                 config: TrainingConfig,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 experiment_name: Optional[str] = None,
                 dataset_info: Optional[Dict[str, Any]] = None):
        """Initialize trainer.
        
        Args:
            model: The discriminator model
            config: Training configuration
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            experiment_name: Optional name for this experiment
            dataset_info: Optional dict with dataset metadata (path, pos/neg counts, etc.)
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset_info = dataset_info or {}
        
        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
            
        self.model = model.to(self.device, non_blocking=True)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed-precision scaler. Active only on CUDA + config.amp=True.
        # On CPU, GradScaler is a no-op (its .scale/.step methods pass
        # through). We pre-check the device type here so the training loop
        # branches cleanly without per-batch cost.
        self.amp_enabled = bool(getattr(config, "amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode='min'
        )
        
        # Experiment tracking
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(config.checkpoint_dir) / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # History
        self.history = TrainingHistory()
        
        # Set random seed
        self._set_seed(config.seed)
        
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience
            )
        elif self.config.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        else:
            return None
            
    def _extract_multi_stream_kwargs(self, batch: Dict) -> Dict:
        """Extract optional multi-stream inputs from a batch dict.

        Returns keyword arguments for driving and profile streams,
        ready to pass to model.forward(**kwargs). Returns empty dict
        for single-stream (V1/V2) datasets.
        """
        kwargs = {}
        if 'driving_1' in batch:
            kwargs['driving_1'] = batch['driving_1'].to(self.device, non_blocking=True)
            kwargs['driving_2'] = batch['driving_2'].to(self.device, non_blocking=True)
            kwargs['mask_d1'] = batch['mask_d1'].to(self.device, non_blocking=True)
            kwargs['mask_d2'] = batch['mask_d2'].to(self.device, non_blocking=True)
        if 'profile_1' in batch:
            kwargs['profile_1'] = batch['profile_1'].to(self.device, non_blocking=True)
            kwargs['profile_2'] = batch['profile_2'].to(self.device, non_blocking=True)
        return kwargs

    def _train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            x1 = batch['x1'].to(self.device, non_blocking=True)
            x2 = batch['x2'].to(self.device, non_blocking=True)
            mask1 = batch['mask1'].to(self.device, non_blocking=True)
            mask2 = batch['mask2'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            kwargs = self._extract_multi_stream_kwargs(batch)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass under autocast when AMP is enabled. The output
            # is cast back to FP32 before BCELoss because FP16 sigmoid can
            # produce exactly 0 or 1, which BCELoss evaluates as log(0).
            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                outputs = self.model(x1, x2, mask1, mask2, **kwargs).squeeze(-1)
            outputs = outputs.float()
            loss = self.criterion(outputs, labels)

            # Backward pass via GradScaler. On CPU (scaler disabled) this
            # reduces to plain loss.backward() — no behavioural difference.
            self.scaler.scale(loss).backward()

            # Gradient clipping — must unscale first so max_norm is applied
            # in the unscaled gradient space that downstream code expects.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        for batch in self.val_loader:
            x1 = batch['x1'].to(self.device, non_blocking=True)
            x2 = batch['x2'].to(self.device, non_blocking=True)
            mask1 = batch['mask1'].to(self.device, non_blocking=True)
            mask2 = batch['mask2'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            kwargs = self._extract_multi_stream_kwargs(batch)

            # Forward pass — autocast for speed, FP32 cast before BCELoss
            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                outputs = self.model(x1, x2, mask1, mask2, **kwargs).squeeze(-1)
            outputs = outputs.float()

            # Compute loss
            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            
            # Collect predictions
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(float)
            
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
        # Compute metrics
        avg_loss = total_loss / len(all_labels)
        
        metrics = {'loss': avg_loss}
        
        if SKLEARN_AVAILABLE:
            metrics['accuracy'] = accuracy_score(all_labels, all_preds)
            metrics['f1'] = f1_score(all_labels, all_preds, zero_division=0)
            metrics['precision'] = precision_score(all_labels, all_preds, zero_division=0)
            metrics['recall'] = recall_score(all_labels, all_preds, zero_division=0)
            
            try:
                metrics['auc'] = roc_auc_score(all_labels, all_probs)
            except ValueError:
                metrics['auc'] = 0.5  # Default if only one class present
                
            # Confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Per-class (split) accuracy - critical for monitoring discriminator performance
            # TN=correct negatives, FP=false positives, FN=false negatives, TP=correct positives
            tn, fp, fn, tp = cm.ravel()
            metrics['negative_accuracy'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['positive_accuracy'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['true_negative'] = int(tn)
            metrics['false_positive'] = int(fp)
            metrics['false_negative'] = int(fn)
            metrics['true_positive'] = int(tp)
        else:
            # Basic accuracy without sklearn
            correct = sum(p == l for p, l in zip(all_preds, all_labels))
            metrics['accuracy'] = correct / len(all_labels)
            # Manual split accuracy
            pos_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == 1 and p == l)
            pos_total = sum(1 for l in all_labels if l == 1)
            neg_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == 0 and p == l)
            neg_total = sum(1 for l in all_labels if l == 0)
            metrics['positive_accuracy'] = pos_correct / pos_total if pos_total > 0 else 0.0
            metrics['negative_accuracy'] = neg_correct / neg_total if neg_total > 0 else 0.0
            
        return metrics
    
    @torch.no_grad()
    def _validate_identical_trajectories(self, n_samples: int = 100) -> Dict[str, float]:
        """Test model behavior on identical trajectory pairs.
        
        This is a crucial sanity check: the model should output high similarity
        (close to 1.0) when comparing a trajectory to itself.
        
        Args:
            n_samples: Number of trajectories to test
            
        Returns:
            Dict with identical trajectory metrics
        """
        self.model.eval()
        all_scores = []
        
        samples_tested = 0
        for batch in self.val_loader:
            x1 = batch['x1'].to(self.device, non_blocking=True)
            mask1 = batch['mask1'].to(self.device, non_blocking=True)

            # Build identical-pair kwargs for multi-stream
            id_kwargs = {}
            if 'driving_1' in batch:
                d1 = batch['driving_1'].to(self.device, non_blocking=True)
                md1 = batch['mask_d1'].to(self.device, non_blocking=True)
            if 'profile_1' in batch:
                p1 = batch['profile_1'].to(self.device, non_blocking=True)

            # Test each trajectory against itself
            for i in range(min(len(x1), n_samples - samples_tested)):
                traj = x1[i:i+1]
                mask = mask1[i:i+1]

                kwargs = {}
                if 'driving_1' in batch:
                    kwargs['driving_1'] = d1[i:i+1]
                    kwargs['driving_2'] = d1[i:i+1]
                    kwargs['mask_d1'] = md1[i:i+1]
                    kwargs['mask_d2'] = md1[i:i+1]
                if 'profile_1' in batch:
                    kwargs['profile_1'] = p1[i:i+1]
                    kwargs['profile_2'] = p1[i:i+1]

                # Compare trajectory to itself
                output = self.model(traj, traj, mask, mask, **kwargs).squeeze().item()
                all_scores.append(output)
                samples_tested += 1

                if samples_tested >= n_samples:
                    break

            if samples_tested >= n_samples:
                break
        
        if not all_scores:
            return {'identical_mean': 0.0, 'identical_std': 0.0, 'identical_min': 0.0}
            
        scores_arr = np.array(all_scores)
        return {
            'identical_mean': float(scores_arr.mean()),
            'identical_std': float(scores_arr.std()),
            'identical_min': float(scores_arr.min()),
            'identical_max': float(scores_arr.max()),
            'identical_above_0.9': float((scores_arr >= 0.9).mean()),
            'identical_above_0.5': float((scores_arr >= 0.5).mean()),
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history.to_dict(),
            'model_config': self.model.config if hasattr(self.model, 'config') else {}
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            
        # Save numbered checkpoint if not save_best_only
        if not self.config.save_best_only:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
            torch.save(checkpoint, epoch_path)
    
    def save_results_json(self, training_start_time: Optional[str] = None):
        """Save comprehensive training results to results.json.
        
        This creates a single JSON file with all information needed to understand
        the trained model: dataset info, model architecture, hyperparameters,
        and performance metrics.
        
        Args:
            training_start_time: ISO format timestamp when training started
        """
        # Calculate model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Get model architecture details
        model_config = self.model.config if hasattr(self.model, 'config') else {}
        
        # Calculate layer information
        layer_info = {}
        if hasattr(self.model, 'encoder'):
            encoder = self.model.encoder
            layer_info['encoder'] = {
                'type': 'LSTM',
                'input_dim': 6,  # After normalization
                'lstm_hidden_dims': encoder.lstm_hidden_dims,
                'num_layers': encoder.num_layers,
                'bidirectional': encoder.bidirectional,
                'output_dim': encoder.output_dim
            }
        if hasattr(self.model, 'classifier'):
            classifier_layers = []
            for name, module in self.model.classifier.named_modules():
                if isinstance(module, nn.Linear):
                    classifier_layers.append({
                        'type': 'Linear',
                        'in_features': module.in_features,
                        'out_features': module.out_features
                    })
            layer_info['classifier'] = classifier_layers
        
        # Training statistics
        epochs_completed = len(self.history.train_loss)
        total_time = sum(self.history.epoch_times) if self.history.epoch_times else 0
        
        # Best epoch metrics (from validation during training)
        best_epoch = self.history.best_epoch
        best_metrics = {}
        if best_epoch > 0 and best_epoch <= len(self.history.val_loss):
            idx = best_epoch - 1
            best_metrics = {
                'epoch': best_epoch,
                'val_loss': self.history.val_loss[idx],
                'val_accuracy': self.history.val_accuracy[idx] if self.history.val_accuracy else None,
                'val_f1': self.history.val_f1[idx] if self.history.val_f1 else None,
                'val_auc': self.history.val_auc[idx] if self.history.val_auc else None,
                'train_loss': self.history.train_loss[idx],
            }
        
        # Final epoch metrics
        final_metrics = {}
        if epochs_completed > 0:
            final_metrics = {
                'epoch': epochs_completed,
                'train_loss': self.history.train_loss[-1],
                'val_loss': self.history.val_loss[-1] if self.history.val_loss else None,
                'val_accuracy': self.history.val_accuracy[-1] if self.history.val_accuracy else None,
                'val_f1': self.history.val_f1[-1] if self.history.val_f1 else None,
                'val_auc': self.history.val_auc[-1] if self.history.val_auc else None,
            }
        
        # Dataset sample counts
        train_samples = len(self.train_loader.dataset)
        val_samples = len(self.val_loader.dataset)
        
        # Try to get pos/neg counts from dataset
        train_pos, train_neg = None, None
        val_pos, val_neg = None, None
        
        if hasattr(self.train_loader.dataset, 'labels'):
            labels = self.train_loader.dataset.labels
            train_pos = int((labels == 1).sum())
            train_neg = int((labels == 0).sum())
        elif hasattr(self.train_loader.dataset, 'data') and 'label' in self.train_loader.dataset.data:
            labels = self.train_loader.dataset.data['label']
            train_pos = int((labels == 1).sum())
            train_neg = int((labels == 0).sum())
            
        if hasattr(self.val_loader.dataset, 'labels'):
            labels = self.val_loader.dataset.labels
            val_pos = int((labels == 1).sum())
            val_neg = int((labels == 0).sum())
        elif hasattr(self.val_loader.dataset, 'data') and 'label' in self.val_loader.dataset.data:
            labels = self.val_loader.dataset.data['label']
            val_pos = int((labels == 1).sum())
            val_neg = int((labels == 0).sum())
        
        # Build comprehensive results dictionary
        results = {
            'experiment': {
                'name': self.experiment_name,
                'checkpoint_dir': str(self.checkpoint_dir),
                'training_started': training_start_time or datetime.now().isoformat(),
                'training_completed': datetime.now().isoformat(),
            },
            'dataset': {
                **self.dataset_info,  # Include any passed dataset metadata
                'train_samples': train_samples,
                'train_positive': train_pos,
                'train_negative': train_neg,
                'val_samples': val_samples,
                'val_positive': val_pos,
                'val_negative': val_neg,
                'total_samples': train_samples + val_samples,
            },
            'model': {
                'architecture': 'SiameseLSTMDiscriminator',
                'config': model_config,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'layers': layer_info,
            },
            'hyperparameters': {
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'epochs_configured': self.config.epochs,
                'early_stopping_patience': self.config.early_stopping_patience,
                'early_stopping_min_delta': self.config.early_stopping_min_delta,
                'scheduler': self.config.scheduler,
                'scheduler_patience': self.config.scheduler_patience,
                'scheduler_factor': self.config.scheduler_factor,
                'seed': self.config.seed,
            },
            'training': {
                'device': str(self.device),
                'epochs_completed': epochs_completed,
                'early_stopped': epochs_completed < self.config.epochs,
                'total_time_seconds': total_time,
                'avg_epoch_time_seconds': total_time / epochs_completed if epochs_completed > 0 else 0,
                'initial_learning_rate': self.history.learning_rates[0] if self.history.learning_rates else self.config.learning_rate,
                'final_learning_rate': self.history.learning_rates[-1] if self.history.learning_rates else self.config.learning_rate,
            },
            'performance': {
                'best': best_metrics,
                'final': final_metrics,
            }
        }
        
        # Save results.json
        results_path = self.checkpoint_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
            
    def load_checkpoint(self, path: Union[str, Path]):
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if 'history' in checkpoint:
            self.history = TrainingHistory.from_dict(checkpoint['history'])
            
        return checkpoint.get('epoch', 0)
    
    def train(self, verbose: bool = True, progress_callback: callable = None) -> TrainingHistory:
        """Run full training loop.
        
        Args:
            verbose: Whether to print progress
            progress_callback: Optional callback function called after each epoch with signature:
                callback(epoch, total_epochs, epoch_time, train_loss, val_metrics, is_best, should_stop)
                Returns True to continue training, False to abort.
            
        Returns:
            TrainingHistory with metrics from all epochs
        """
        if verbose:
            print(f"Training on {self.device}")
            print(f"Train samples: {len(self.train_loader.dataset)}")
            print(f"Val samples: {len(self.val_loader.dataset)}")
            print(f"Checkpoint directory: {self.checkpoint_dir}")
            print("-" * 60)
        
        # Record training start time
        training_start_time = datetime.now().isoformat()
            
        # Save config
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
            
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self._train_epoch()
            
            # Validate
            val_metrics = self._validate()
            
            # Validate identical trajectory behavior (crucial sanity check)
            identical_metrics = self._validate_identical_trajectories(n_samples=100)
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_metrics['loss'])
            self.history.val_accuracy.append(val_metrics.get('accuracy', 0))
            self.history.val_positive_accuracy.append(val_metrics.get('positive_accuracy', 0))
            self.history.val_negative_accuracy.append(val_metrics.get('negative_accuracy', 0))
            self.history.val_f1.append(val_metrics.get('f1', 0))
            self.history.val_auc.append(val_metrics.get('auc', 0))
            self.history.val_identical_score.append(identical_metrics.get('identical_mean', 0))
            self.history.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            self.history.epoch_times.append(epoch_time)
            
            # Check if best
            is_best = val_metrics['loss'] < self.history.best_val_loss
            if is_best:
                self.history.best_val_loss = val_metrics['loss']
                self.history.best_epoch = epoch
                
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                    
            # Print progress
            if verbose:
                best_marker = " *" if is_best else ""
                identical_score = identical_metrics.get('identical_mean', 0)
                identical_warn = " ⚠️ LOW" if identical_score < 0.5 else ""
                pos_acc = val_metrics.get('positive_accuracy', 0)
                neg_acc = val_metrics.get('negative_accuracy', 0)
                # Warn if split accuracies are very imbalanced (suggests model bias)
                split_warn = ""
                if abs(pos_acc - neg_acc) > 0.3:
                    split_warn = " ⚠️"
                print(f"Epoch {epoch:3d}/{self.config.epochs} | "
                      f"Train: {train_loss:.4f} | "
                      f"Val: {val_metrics['loss']:.4f} | "
                      f"Acc: {val_metrics.get('accuracy', 0):.3f} "
                      f"[+:{pos_acc:.3f} -:{neg_acc:.3f}]{split_warn} | "
                      f"Id: {identical_score:.3f}{identical_warn} | "
                      f"{epoch_time:.1f}s{best_marker}")
                
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}!")
                    print(f"Best epoch: {self.history.best_epoch}")
                # Notify callback about early stopping
                if progress_callback:
                    progress_callback(
                        epoch=epoch,
                        total_epochs=self.config.epochs,
                        epoch_time=epoch_time,
                        train_loss=train_loss,
                        val_metrics=val_metrics,
                        identical_metrics=identical_metrics,
                        is_best=False,
                        should_stop=True
                    )
                break
            
            # Call progress callback if provided
            if progress_callback:
                continue_training = progress_callback(
                    epoch=epoch,
                    total_epochs=self.config.epochs,
                    epoch_time=epoch_time,
                    train_loss=train_loss,
                    val_metrics=val_metrics,
                    identical_metrics=identical_metrics,
                    is_best=is_best,
                    should_stop=False
                )
                # Allow callback to abort training
                if continue_training is False:
                    if verbose:
                        print(f"\nTraining aborted by callback at epoch {epoch}")
                    break
                
        # Save final history
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history.to_dict(), f, indent=2)
        
        # Save comprehensive results summary
        self.save_results_json(training_start_time=training_start_time)

        # Generate paper-ready training plots. Errors are caught non-fatally
        # because plot failures shouldn't lose the trained model.
        try:
            self.generate_training_plots(verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[warn] plot generation failed: {e}")

        if verbose:
            print("-" * 60)
            print(f"Training complete!")
            print(f"Best validation loss: {self.history.best_val_loss:.4f} at epoch {self.history.best_epoch}")
            if MATPLOTLIB_AVAILABLE:
                print(f"Training plots: {self.checkpoint_dir / 'plots'}")

        return self.history

    def generate_training_plots(self, verbose: bool = True):
        """Render PNG plots of training metrics to ``<checkpoint_dir>/plots/``.

        Generates loss curves, accuracy curves, AUC/F1 curves, identical-
        pair sanity curve, learning-rate schedule, ROC curve, precision-
        recall curve, and a four-panel training summary. ROC and PR
        curves use the final-epoch model predictions over the val set.
        """
        if not MATPLOTLIB_AVAILABLE:
            if verbose:
                print("[info] matplotlib not available; skipping training plots")
            return None

        plots_dir = self.checkpoint_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        h = self.history
        epochs = list(range(1, len(h.train_loss) + 1))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, h.train_loss, label='train', color='tab:blue', alpha=0.85)
        ax.plot(epochs, h.val_loss, label='val', color='tab:orange', alpha=0.85)
        if h.best_epoch:
            ax.axvline(h.best_epoch, linestyle=':', color='gray',
                       label=f'best epoch ({h.best_epoch})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BCE loss')
        ax.set_title('Training & validation loss')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "loss_curves.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, h.val_accuracy, label='overall', color='tab:blue')
        ax.plot(epochs, h.val_positive_accuracy, label='positive (same driver)',
                color='tab:green', alpha=0.7)
        ax.plot(epochs, h.val_negative_accuracy, label='negative (diff driver)',
                color='tab:red', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation accuracy per epoch')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0.4, 1.01)
        fig.tight_layout()
        fig.savefig(plots_dir / "accuracy_curves.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, h.val_auc, label='AUC', color='tab:purple')
        ax.plot(epochs, h.val_f1, label='F1', color='tab:brown')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Validation AUC and F1 per epoch')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0.4, 1.01)
        fig.tight_layout()
        fig.savefig(plots_dir / "auc_f1_curves.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, h.val_identical_score, color='teal')
        ax.axhline(0.5, linestyle=':', color='red',
                   label='warning threshold (0.5)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean identical-pair score')
        ax.set_title('Identical-trajectory probability (Siamese sanity)')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0.0, 1.01)
        fig.tight_layout()
        fig.savefig(plots_dir / "identical_curve.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, h.learning_rates, color='tab:olive')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning rate')
        ax.set_title('Learning-rate schedule')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "learning_rate_curve.png", dpi=150)
        plt.close(fig)

        if SKLEARN_AVAILABLE:
            try:
                self._generate_roc_and_pr_plots(plots_dir)
            except Exception as e:
                if verbose:
                    print(f"[warn] ROC/PR curve generation failed: {e}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(epochs, h.train_loss, label='train')
        axes[0, 0].plot(epochs, h.val_loss, label='val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        axes[0, 1].plot(epochs, h.val_accuracy, label='acc', color='tab:blue')
        axes[0, 1].plot(epochs, h.val_auc, label='AUC', color='tab:purple')
        axes[0, 1].plot(epochs, h.val_f1, label='F1', color='tab:brown')
        axes[0, 1].set_title('Validation metrics')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        axes[1, 0].plot(epochs, h.val_identical_score, color='teal')
        axes[1, 0].axhline(0.5, linestyle=':', color='red')
        axes[1, 0].set_title('Identical-pair score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylim(0, 1.01)
        axes[1, 0].grid(alpha=0.3)
        axes[1, 1].plot(epochs, h.learning_rates, color='tab:olive')
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_title('Learning rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].grid(alpha=0.3)
        fig.suptitle(
            f'Training summary — best val_loss {h.best_val_loss:.4f} at '
            f'epoch {h.best_epoch}/{len(h.train_loss)}'
        )
        fig.tight_layout()
        fig.savefig(plots_dir / "training_summary.png", dpi=150)
        plt.close(fig)

        if verbose:
            print(f"[info] training plots written to {plots_dir}")
        return plots_dir

    @torch.no_grad()
    def _generate_roc_and_pr_plots(self, plots_dir: Path) -> None:
        """Collect val-set predictions and render ROC + PR plots."""
        self.model.train(mode=False)  # switch to inference mode
        all_probs: List[float] = []
        all_labels: List[float] = []
        for batch in self.val_loader:
            x1 = batch['x1'].to(self.device, non_blocking=True)
            x2 = batch['x2'].to(self.device, non_blocking=True)
            mask1 = batch['mask1'].to(self.device, non_blocking=True)
            mask2 = batch['mask2'].to(self.device, non_blocking=True)
            labels = batch['label']
            if 'driving_1' in batch:
                outputs = self.model(
                    x1=x1, x2=x2, mask1=mask1, mask2=mask2,
                    driving_1=batch['driving_1'].to(self.device, non_blocking=True),
                    driving_2=batch['driving_2'].to(self.device, non_blocking=True),
                    mask_d1=batch['mask_d1'].to(self.device, non_blocking=True),
                    mask_d2=batch['mask_d2'].to(self.device, non_blocking=True),
                    profile_1=batch['profile_1'].to(self.device, non_blocking=True),
                    profile_2=batch['profile_2'].to(self.device, non_blocking=True),
                )
            else:
                outputs = self.model(x1, x2, mask1, mask2)
            all_probs.extend(outputs.squeeze(-1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

        labels_np = np.asarray(all_labels)
        probs_np = np.asarray(all_probs)

        fpr, tpr, _ = roc_curve(labels_np, probs_np)
        auc = roc_auc_score(labels_np, probs_np)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color='tab:purple', label=f'ROC (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], linestyle=':', color='gray', label='chance')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curve — validation set')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "roc_curve.png", dpi=150)
        plt.close(fig)

        prec, rec, _ = precision_recall_curve(labels_np, probs_np)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(rec, prec, color='tab:green')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision–Recall curve — validation set')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "precision_recall_curve.png", dpi=150)
        plt.close(fig)
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader, verbose: bool = True) -> Dict[str, Any]:
        """Evaluate model on test data.
        
        Args:
            test_loader: DataLoader for test data
            verbose: Whether to print results
            
        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        for batch in test_loader:
            x1 = batch['x1'].to(self.device, non_blocking=True)
            x2 = batch['x2'].to(self.device, non_blocking=True)
            mask1 = batch['mask1'].to(self.device, non_blocking=True)
            mask2 = batch['mask2'].to(self.device, non_blocking=True)
            labels = batch['label']
            
            outputs = self.model(x1, x2, mask1, mask2).squeeze(-1)
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(float)
            
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            
        results = {}
        
        if SKLEARN_AVAILABLE:
            results['accuracy'] = accuracy_score(all_labels, all_preds)
            results['f1'] = f1_score(all_labels, all_preds, zero_division=0)
            results['precision'] = precision_score(all_labels, all_preds, zero_division=0)
            results['recall'] = recall_score(all_labels, all_preds, zero_division=0)
            
            try:
                results['auc'] = roc_auc_score(all_labels, all_probs)
            except ValueError:
                results['auc'] = 0.5
                
            cm = confusion_matrix(all_labels, all_preds)
            results['confusion_matrix'] = cm.tolist()
            
            # Per-class accuracy
            tn, fp, fn, tp = cm.ravel()
            results['true_negative'] = int(tn)
            results['false_positive'] = int(fp)
            results['false_negative'] = int(fn)
            results['true_positive'] = int(tp)
            results['negative_accuracy'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            results['positive_accuracy'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            correct = sum(p == l for p, l in zip(all_preds, all_labels))
            results['accuracy'] = correct / len(all_labels)
            
        results['n_samples'] = len(all_labels)
        results['predictions'] = all_preds
        results['probabilities'] = all_probs
        results['labels'] = all_labels
        
        if verbose:
            print("\nEvaluation Results:")
            print("-" * 40)
            print(f"  Samples: {results['n_samples']}")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            if SKLEARN_AVAILABLE:
                print(f"  F1 Score: {results['f1']:.4f}")
                print(f"  Precision: {results['precision']:.4f}")
                print(f"  Recall: {results['recall']:.4f}")
                print(f"  ROC AUC: {results['auc']:.4f}")
                print(f"\nConfusion Matrix:")
                print(f"  TN: {results['true_negative']} | FP: {results['false_positive']}")
                print(f"  FN: {results['false_negative']} | TP: {results['true_positive']}")
                
        return results


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = "auto"
) -> Tuple[nn.Module, Dict]:
    """Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    # Support both package and direct imports
    try:
        from .model import (
            SiameseLSTMDiscriminator,
            SiameseLSTMDiscriminatorV2,
            MultiStreamSiameseDiscriminator,
        )
    except ImportError:
        from model import (
            SiameseLSTMDiscriminator,
            SiameseLSTMDiscriminatorV2,
            MultiStreamSiameseDiscriminator,
        )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reject ancient single-LSTM architecture checkpoints
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if 'encoder.lstm.weight_ih_l0' in state_dict:
        raise ValueError(
            "This checkpoint uses the old single nn.LSTM architecture. "
            "Please train a new model with the current stacked LSTM architecture."
        )

    # Reconstruct model from saved config, dispatching on model_version
    model_config = checkpoint.get('model_config', {})
    version = model_config.get('model_version', 'v1')

    if version == 'v3':
        model = MultiStreamSiameseDiscriminator(**model_config)
    elif version == 'v2':
        model = SiameseLSTMDiscriminatorV2(**model_config)
    else:
        # Legacy V1 checkpoints may use old key names (hidden_dim, num_layers)
        # instead of current (lstm_hidden_dims). Normalize before constructing.
        if 'hidden_dim' in model_config and 'lstm_hidden_dims' not in model_config:
            hd = model_config.pop('hidden_dim')
            nl = model_config.pop('num_layers', 2)
            model_config['lstm_hidden_dims'] = tuple([hd] * nl)
        # Remove keys not accepted by the V1 constructor
        v1_keys = {'lstm_hidden_dims', 'dropout', 'bidirectional', 'classifier_hidden_dims'}
        filtered = {k: v for k, v in model_config.items() if k in v1_keys}
        model = SiameseLSTMDiscriminator(**filtered)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint
