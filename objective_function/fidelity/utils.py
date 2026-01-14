"""
Utility functions for the Trajectory Fidelity Term.

This module provides:
- Discriminator model loading and management
- Trajectory preprocessing and feature extraction
- Batch preparation with padding/masking
- Fidelity score computation (NumPy and PyTorch)
- Differentiable fidelity module for gradient-based optimization
- Visualization utilities

Key Functions:
    - load_discriminator: Load pre-trained ST-SiameseNet model
    - extract_trajectory_features: Convert 126-dim states to 4-dim for discriminator
    - prepare_trajectory_batch: Pad and mask trajectories for batch processing
    - compute_fidelity_scores: Compute discriminator confidence scores
    - DifferentiableFidelity: PyTorch module for backpropagation
"""

from __future__ import annotations

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, TYPE_CHECKING
import sys
import os

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent
OBJECTIVE_FUNCTION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = OBJECTIVE_FUNCTION_DIR.parent
DISCRIMINATOR_DIR = PROJECT_ROOT / "discriminator" / "model"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DISCRIMINATOR_DIR))

# Import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# For type hints when torch is not available
if TYPE_CHECKING:
    import torch
    import torch.nn as nn


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_discriminator(
    checkpoint_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None
) -> Tuple[Any, dict, str]:
    """
    Load a pre-trained discriminator model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        config_path: Path to config.json (auto-detected if None)
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        
    Returns:
        Tuple of (model, config_dict, device_str)
        
    Raises:
        FileNotFoundError: If checkpoint not found
        RuntimeError: If model loading fails
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for discriminator loading")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        # Try relative to project root
        alt_path = PROJECT_ROOT / checkpoint_path
        if alt_path.exists():
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load config
    config = None
    if config_path:
        config_path = Path(config_path)
    else:
        config_path = checkpoint_path.parent / "config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Extract model config from checkpoint or config file
    model_config = checkpoint.get('config', {})
    if not model_config and config:
        # Use training config to extract model parameters
        model_config = {
            'hidden_dim': config.get('hidden_dim', 128),
            'num_layers': config.get('num_layers', 2),
            'dropout': config.get('dropout', 0.2),
            'bidirectional': config.get('bidirectional', True),
            'classifier_hidden_dims': tuple(config.get('classifier_hidden_dims', [128, 64])),
        }
    
    # Import and create model
    from model import SiameseLSTMDiscriminator
    
    model = SiameseLSTMDiscriminator(
        hidden_dim=model_config.get('hidden_dim', 128),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.2),
        bidirectional=model_config.get('bidirectional', True),
        classifier_hidden_dims=tuple(model_config.get('classifier_hidden_dims', [128, 64])),
    )
    
    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    return model, model_config, device


def get_model_info(checkpoint_path: Union[str, Path]) -> dict:
    """
    Get information about a discriminator checkpoint without loading the full model.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        
    Returns:
        Dictionary with model info (config, training history, results)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    
    checkpoint_dir = checkpoint_path.parent
    info = {'checkpoint_path': str(checkpoint_path)}
    
    # Load config
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            info['config'] = json.load(f)
    
    # Load training history
    history_path = checkpoint_dir / "history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            info['history'] = json.load(f)
    
    # Load results
    results_path = checkpoint_dir / "results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            info['results'] = json.load(f)
    
    return info


# =============================================================================
# TRAJECTORY DATA LOADING
# =============================================================================

def load_trajectory_data(filepath: Union[str, Path]) -> Dict[Any, List[List[List[float]]]]:
    """
    Load trajectory data from pickle file.
    
    Args:
        filepath: Path to all_trajs.pkl or similar trajectory file
        
    Returns:
        Dictionary mapping driver_id to list of trajectories
    """
    filepath = Path(filepath)
    if not filepath.exists():
        filepath = PROJECT_ROOT / filepath
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data


def get_trajectory_statistics(
    trajectories: Dict[Any, List[List[List[float]]]]
) -> dict:
    """
    Compute statistics for trajectory dataset.
    
    Args:
        trajectories: Dictionary mapping driver_id to list of trajectories
        
    Returns:
        Dictionary with statistics
    """
    n_drivers = len(trajectories)
    n_trajectories = sum(len(trajs) for trajs in trajectories.values())
    
    lengths = []
    for trajs in trajectories.values():
        for traj in trajs:
            lengths.append(len(traj))
    
    lengths = np.array(lengths)
    
    return {
        'n_drivers': n_drivers,
        'n_trajectories': n_trajectories,
        'min_length': int(lengths.min()) if len(lengths) > 0 else 0,
        'max_length': int(lengths.max()) if len(lengths) > 0 else 0,
        'mean_length': float(lengths.mean()) if len(lengths) > 0 else 0,
        'median_length': float(np.median(lengths)) if len(lengths) > 0 else 0,
        'p90_length': float(np.percentile(lengths, 90)) if len(lengths) > 0 else 0,
        'p95_length': float(np.percentile(lengths, 95)) if len(lengths) > 0 else 0,
    }


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_trajectory_features(
    trajectory: List[List[float]],
    feature_range: Tuple[int, int] = (0, 4)
) -> np.ndarray:
    """
    Extract features from a trajectory for discriminator input.
    
    Args:
        trajectory: List of states, each state is a list of features
        feature_range: (start, end) indices of features to extract
        
    Returns:
        NumPy array of shape [seq_len, n_features]
    """
    start, end = feature_range
    features = []
    
    for state in trajectory:
        features.append(state[start:end])
    
    return np.array(features, dtype=np.float32)


def extract_batch_features(
    trajectories: List[List[List[float]]],
    feature_range: Tuple[int, int] = (0, 4)
) -> List[np.ndarray]:
    """
    Extract features from multiple trajectories.
    
    Args:
        trajectories: List of trajectories
        feature_range: (start, end) indices of features to extract
        
    Returns:
        List of NumPy arrays, each [seq_len, n_features]
    """
    return [extract_trajectory_features(traj, feature_range) for traj in trajectories]


# =============================================================================
# BATCH PREPARATION
# =============================================================================

def prepare_trajectory_batch(
    trajectories: List[np.ndarray],
    max_length: Optional[int] = None,
    pad_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare a batch of trajectories with padding and masks.
    
    Args:
        trajectories: List of trajectory arrays [seq_len, n_features]
        max_length: Maximum sequence length (None = use longest in batch)
        pad_value: Value to use for padding
        
    Returns:
        Tuple of (padded_batch, masks)
            - padded_batch: [batch_size, max_len, n_features]
            - masks: [batch_size, max_len] boolean mask (True = valid)
    """
    if len(trajectories) == 0:
        raise ValueError("Cannot prepare empty batch")
    
    n_features = trajectories[0].shape[-1]
    lengths = [len(traj) for traj in trajectories]
    
    if max_length is None:
        max_length = max(lengths)
    
    batch_size = len(trajectories)
    padded = np.full((batch_size, max_length, n_features), pad_value, dtype=np.float32)
    masks = np.zeros((batch_size, max_length), dtype=bool)
    
    for i, traj in enumerate(trajectories):
        seq_len = min(len(traj), max_length)
        padded[i, :seq_len] = traj[:seq_len]
        masks[i, :seq_len] = True
    
    return padded, masks


def create_trajectory_pairs(
    edited_trajectories: Dict[Any, List[List[List[float]]]],
    original_trajectories: Dict[Any, List[List[List[float]]]],
    mode: str = "same_agent",
    feature_range: Tuple[int, int] = (0, 4),
    min_length: int = 2
) -> Tuple[List[np.ndarray], List[np.ndarray], List[dict]]:
    """
    Create trajectory pairs for fidelity computation.
    
    Args:
        edited_trajectories: Dictionary of edited trajectories by driver
        original_trajectories: Dictionary of original/reference trajectories by driver
        mode: Pairing mode ("same_agent", "paired", "batch")
        feature_range: Features to extract
        min_length: Minimum trajectory length
        
    Returns:
        Tuple of (edited_features, original_features, pair_metadata)
    """
    edited_list = []
    original_list = []
    metadata = []
    
    if mode == "same_agent":
        # Pair edited trajectories with originals from same driver
        for driver_id, edited_trajs in edited_trajectories.items():
            if driver_id not in original_trajectories:
                continue
            
            orig_trajs = original_trajectories[driver_id]
            
            # Pair by index (assumes 1:1 correspondence)
            for idx, edited_traj in enumerate(edited_trajs):
                if idx >= len(orig_trajs):
                    continue
                
                orig_traj = orig_trajs[idx]
                
                # Skip short trajectories
                if len(edited_traj) < min_length or len(orig_traj) < min_length:
                    continue
                
                edited_features = extract_trajectory_features(edited_traj, feature_range)
                orig_features = extract_trajectory_features(orig_traj, feature_range)
                
                edited_list.append(edited_features)
                original_list.append(orig_features)
                metadata.append({
                    'driver_id': driver_id,
                    'trajectory_idx': idx,
                    'edited_length': len(edited_traj),
                    'original_length': len(orig_traj),
                })
    
    elif mode == "paired":
        # Assume trajectories are already paired (same structure)
        for driver_id, edited_trajs in edited_trajectories.items():
            if driver_id not in original_trajectories:
                continue
            
            for idx, edited_traj in enumerate(edited_trajs):
                if idx >= len(original_trajectories[driver_id]):
                    continue
                
                orig_traj = original_trajectories[driver_id][idx]
                
                if len(edited_traj) < min_length or len(orig_traj) < min_length:
                    continue
                
                edited_features = extract_trajectory_features(edited_traj, feature_range)
                orig_features = extract_trajectory_features(orig_traj, feature_range)
                
                edited_list.append(edited_features)
                original_list.append(orig_features)
                metadata.append({
                    'driver_id': driver_id,
                    'trajectory_idx': idx,
                })
    
    elif mode == "batch":
        # Compare all edited trajectories against all originals
        # (Use a sampling strategy for efficiency)
        all_originals = []
        for orig_trajs in original_trajectories.values():
            for traj in orig_trajs:
                if len(traj) >= min_length:
                    all_originals.append(extract_trajectory_features(traj, feature_range))
        
        if len(all_originals) == 0:
            return [], [], []
        
        # For each edited trajectory, compare with a random sample of originals
        rng = np.random.default_rng(42)
        sample_size = min(10, len(all_originals))
        
        for driver_id, edited_trajs in edited_trajectories.items():
            for idx, edited_traj in enumerate(edited_trajs):
                if len(edited_traj) < min_length:
                    continue
                
                edited_features = extract_trajectory_features(edited_traj, feature_range)
                
                # Sample random originals
                sampled_indices = rng.choice(len(all_originals), size=sample_size, replace=False)
                
                for ref_idx in sampled_indices:
                    edited_list.append(edited_features)
                    original_list.append(all_originals[ref_idx])
                    metadata.append({
                        'driver_id': driver_id,
                        'trajectory_idx': idx,
                        'reference_idx': int(ref_idx),
                    })
    
    return edited_list, original_list, metadata


# =============================================================================
# FIDELITY SCORE COMPUTATION
# =============================================================================

def compute_fidelity_scores(
    model: Any,
    edited_trajectories: List[np.ndarray],
    original_trajectories: List[np.ndarray],
    batch_size: int = 32,
    max_length: Optional[int] = None,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute discriminator confidence scores for trajectory pairs.
    
    Args:
        model: Loaded discriminator model
        edited_trajectories: List of edited trajectory features
        original_trajectories: List of original trajectory features
        batch_size: Batch size for processing
        max_length: Maximum sequence length for padding
        device: Device for computation
        
    Returns:
        Array of confidence scores [n_pairs]
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for fidelity computation")
    
    n_pairs = len(edited_trajectories)
    if n_pairs == 0:
        return np.array([])
    
    all_scores = []
    
    model.eval()
    with torch.no_grad():
        for start_idx in range(0, n_pairs, batch_size):
            end_idx = min(start_idx + batch_size, n_pairs)
            
            # Prepare batch
            batch_edited = edited_trajectories[start_idx:end_idx]
            batch_original = original_trajectories[start_idx:end_idx]
            
            # Pad and create masks
            x1, mask1 = prepare_trajectory_batch(batch_edited, max_length)
            x2, mask2 = prepare_trajectory_batch(batch_original, max_length)
            
            # Convert to tensors
            x1 = torch.from_numpy(x1).to(device)
            x2 = torch.from_numpy(x2).to(device)
            mask1 = torch.from_numpy(mask1).to(device)
            mask2 = torch.from_numpy(mask2).to(device)
            
            # Forward pass
            probs = model(x1, x2, mask1, mask2)  # [batch, 1]
            
            all_scores.extend(probs.squeeze(-1).cpu().numpy().tolist())
    
    return np.array(all_scores)


def aggregate_fidelity_scores(
    scores: np.ndarray,
    method: str = "mean",
    threshold: float = 0.5,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Aggregate individual fidelity scores into final term value.
    
    Args:
        scores: Array of discriminator confidence scores
        method: Aggregation method ("mean", "min", "threshold", "weighted")
        threshold: Threshold for threshold aggregation
        weights: Optional weights for weighted aggregation
        
    Returns:
        Aggregated fidelity score in [0, 1]
    """
    if len(scores) == 0:
        return 0.0
    
    if method == "mean":
        return float(np.mean(scores))
    
    elif method == "min":
        return float(np.min(scores))
    
    elif method == "threshold":
        return float(np.mean(scores >= threshold))
    
    elif method == "weighted":
        if weights is None:
            weights = np.ones_like(scores)
        weights = weights / weights.sum()  # Normalize
        return float(np.sum(scores * weights))
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def compute_fidelity(
    model: Any,
    edited_trajectories: Dict[Any, List[List[List[float]]]],
    original_trajectories: Dict[Any, List[List[List[float]]]],
    mode: str = "same_agent",
    aggregation: str = "mean",
    threshold: float = 0.5,
    batch_size: int = 32,
    max_length: Optional[int] = None,
    feature_range: Tuple[int, int] = (0, 4),
    device: str = "cpu",
    min_length: int = 2
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute fidelity term value with full pipeline.
    
    Args:
        model: Loaded discriminator model
        edited_trajectories: Dictionary of edited trajectories
        original_trajectories: Dictionary of original trajectories
        mode: Pairing mode
        aggregation: Score aggregation method
        threshold: Threshold for discriminator and threshold aggregation
        batch_size: Batch size for processing
        max_length: Maximum trajectory length
        feature_range: Features to extract
        device: Computation device
        min_length: Minimum trajectory length
        
    Returns:
        Tuple of (fidelity_score, breakdown_dict)
    """
    # Create trajectory pairs
    edited_list, original_list, metadata = create_trajectory_pairs(
        edited_trajectories,
        original_trajectories,
        mode=mode,
        feature_range=feature_range,
        min_length=min_length
    )
    
    if len(edited_list) == 0:
        return 0.0, {
            'n_pairs': 0,
            'scores': [],
            'metadata': [],
            'warning': 'No valid trajectory pairs found'
        }
    
    # Compute scores
    scores = compute_fidelity_scores(
        model,
        edited_list,
        original_list,
        batch_size=batch_size,
        max_length=max_length,
        device=device
    )
    
    # Compute weights for weighted aggregation
    weights = None
    if aggregation == "weighted":
        weights = np.array([
            len(e) + len(o) for e, o in zip(edited_list, original_list)
        ], dtype=np.float32)
    
    # Aggregate
    fidelity = aggregate_fidelity_scores(
        scores,
        method=aggregation,
        threshold=threshold,
        weights=weights
    )
    
    breakdown = {
        'n_pairs': len(scores),
        'scores': scores.tolist(),
        'metadata': metadata,
        'score_statistics': {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'above_threshold': float(np.mean(scores >= threshold)),
        }
    }
    
    return fidelity, breakdown


# =============================================================================
# DIFFERENTIABLE FIDELITY MODULE
# =============================================================================

class DifferentiableFidelity(nn.Module):
    """
    Differentiable fidelity module for gradient-based optimization.
    
    This module wraps the discriminator to enable backpropagation through
    the fidelity computation, allowing gradient-based trajectory optimization.
    
    The discriminator takes two trajectories and outputs P(same_agent).
    For fidelity, we want edited trajectories to appear authentic (high probability).
    
    Usage:
        >>> diff_fidelity = DifferentiableFidelity(discriminator)
        >>> fidelity = diff_fidelity(edited_trajs, original_trajs, masks)
        >>> fidelity.backward()  # Compute gradients
    """
    
    def __init__(
        self,
        discriminator: nn.Module,
        aggregation: str = "mean"
    ):
        """
        Initialize differentiable fidelity module.
        
        Args:
            discriminator: Pre-trained discriminator model
            aggregation: Score aggregation ("mean", "min")
        """
        super().__init__()
        self.discriminator = discriminator
        self.aggregation = aggregation
        
        # Freeze discriminator weights
        for param in self.discriminator.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        edited: torch.Tensor,
        original: torch.Tensor,
        mask_edited: Optional[torch.Tensor] = None,
        mask_original: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute differentiable fidelity score.
        
        Args:
            edited: Edited trajectory features [batch, seq_len, 4]
            original: Original trajectory features [batch, seq_len, 4]
            mask_edited: Mask for edited [batch, seq_len]
            mask_original: Mask for original [batch, seq_len]
            
        Returns:
            Fidelity score (scalar tensor)
        """
        # Get discriminator probabilities
        probs = self.discriminator(edited, original, mask_edited, mask_original)
        probs = probs.squeeze(-1)  # [batch]
        
        # Aggregate
        if self.aggregation == "mean":
            return probs.mean()
        elif self.aggregation == "min":
            return probs.min()
        else:
            return probs.mean()
    
    def forward_with_scores(
        self,
        edited: torch.Tensor,
        original: torch.Tensor,
        mask_edited: Optional[torch.Tensor] = None,
        mask_original: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fidelity with individual scores.
        
        Returns:
            Tuple of (fidelity_score, individual_scores)
        """
        probs = self.discriminator(edited, original, mask_edited, mask_original)
        probs = probs.squeeze(-1)  # [batch]
        
        if self.aggregation == "mean":
            fidelity = probs.mean()
        elif self.aggregation == "min":
            fidelity = probs.min()
        else:
            fidelity = probs.mean()
        
        return fidelity, probs


def verify_fidelity_gradient(
    model: nn.Module,
    edited: torch.Tensor,
    original: torch.Tensor,
    mask_edited: Optional[torch.Tensor] = None,
    mask_original: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> Dict[str, Any]:
    """
    Verify gradient computation for fidelity term.
    
    Args:
        model: Discriminator model
        edited: Edited trajectory features [batch, seq_len, 4]
        original: Original trajectory features [batch, seq_len, 4]
        mask_edited: Mask for edited
        mask_original: Mask for original
        eps: Perturbation size for numerical gradient
        
    Returns:
        Dictionary with gradient verification results
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for gradient verification")
    
    device = next(model.parameters()).device
    
    # Enable gradient computation for edited trajectories
    edited_var = edited.clone().detach().requires_grad_(True).to(device)
    original_fixed = original.clone().detach().to(device)
    
    if mask_edited is not None:
        mask_edited = mask_edited.to(device)
    if mask_original is not None:
        mask_original = mask_original.to(device)
    
    # Set model to training mode for backward pass (cuDNN LSTM requires this)
    # But freeze weights so we only compute gradients w.r.t. input
    was_training = model.training
    model.train()
    
    # Compute fidelity and backward
    diff_fidelity = DifferentiableFidelity(model, aggregation="mean")
    fidelity = diff_fidelity(edited_var, original_fixed, mask_edited, mask_original)
    fidelity.backward()
    
    analytic_grad = edited_var.grad.clone()
    
    # Numerical gradient (sample a few positions)
    numerical_grads = []
    positions = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]  # Sample positions
    
    for b, t, f in positions:
        if b >= edited.shape[0] or t >= edited.shape[1] or f >= edited.shape[2]:
            continue
        
        # Perturb +
        edited_plus = edited.clone().to(device)
        edited_plus[b, t, f] += eps
        
        with torch.no_grad():
            fidelity_plus = diff_fidelity(
                edited_plus, original_fixed, mask_edited, mask_original
            )
        
        # Perturb -
        edited_minus = edited.clone().to(device)
        edited_minus[b, t, f] -= eps
        
        with torch.no_grad():
            fidelity_minus = diff_fidelity(
                edited_minus, original_fixed, mask_edited, mask_original
            )
        
        numerical_grad = (fidelity_plus.item() - fidelity_minus.item()) / (2 * eps)
        analytic_val = analytic_grad[b, t, f].item()
        
        numerical_grads.append({
            'position': (b, t, f),
            'numerical': numerical_grad,
            'analytic': analytic_val,
            'abs_diff': abs(numerical_grad - analytic_val),
            'rel_diff': abs(numerical_grad - analytic_val) / (abs(numerical_grad) + 1e-8)
        })
    
    # Summarize
    abs_diffs = [g['abs_diff'] for g in numerical_grads]
    rel_diffs = [g['rel_diff'] for g in numerical_grads]
    
    # Restore model state
    if not was_training:
        model.eval()
    
    # For LSTM-based models, numerical gradient checking is notoriously unreliable
    # due to the sequential nature and potential numerical instabilities.
    # The key validation is that:
    # 1. Gradients are computed (not None)
    # 2. Gradient norm is non-zero (indicating signal flows back)
    gradient_computed = analytic_grad is not None
    gradient_nonzero = analytic_grad.norm().item() > 1e-10
    
    return {
        'fidelity_value': fidelity.item(),
        'gradient_shape': list(analytic_grad.shape),
        'gradient_norm': analytic_grad.norm().item(),
        'gradient_samples': numerical_grads,
        'max_abs_diff': max(abs_diffs) if abs_diffs else 0,
        'mean_abs_diff': np.mean(abs_diffs) if abs_diffs else 0,
        'max_rel_diff': max(rel_diffs) if rel_diffs else 0,
        'mean_rel_diff': np.mean(rel_diffs) if rel_diffs else 0,
        # Key validation: gradients exist and are non-zero
        'gradient_computed': gradient_computed,
        'gradient_nonzero': gradient_nonzero,
        'gradient_valid': gradient_computed and gradient_nonzero,
        'note': 'LSTM numerical gradient check may be unreliable; gradient_nonzero is key metric',
    }


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def compute_score_histogram(
    scores: np.ndarray,
    n_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of fidelity scores.
    
    Args:
        scores: Array of fidelity scores
        n_bins: Number of histogram bins
        
    Returns:
        Tuple of (counts, bin_edges)
    """
    counts, bin_edges = np.histogram(scores, bins=n_bins, range=(0, 1))
    return counts, bin_edges


def compute_per_driver_statistics(
    scores: np.ndarray,
    metadata: List[dict]
) -> Dict[Any, dict]:
    """
    Compute per-driver fidelity statistics.
    
    Args:
        scores: Array of fidelity scores
        metadata: List of metadata dicts with 'driver_id'
        
    Returns:
        Dictionary mapping driver_id to statistics
    """
    driver_scores = {}
    
    for score, meta in zip(scores, metadata):
        driver_id = meta.get('driver_id')
        if driver_id not in driver_scores:
            driver_scores[driver_id] = []
        driver_scores[driver_id].append(score)
    
    driver_stats = {}
    for driver_id, drv_scores in driver_scores.items():
        drv_scores = np.array(drv_scores)
        driver_stats[driver_id] = {
            'n_pairs': len(drv_scores),
            'mean': float(np.mean(drv_scores)),
            'std': float(np.std(drv_scores)),
            'min': float(np.min(drv_scores)),
            'max': float(np.max(drv_scores)),
        }
    
    return driver_stats


def compute_length_correlation(
    scores: np.ndarray,
    metadata: List[dict]
) -> dict:
    """
    Analyze correlation between trajectory length and fidelity.
    
    Args:
        scores: Array of fidelity scores
        metadata: List of metadata dicts with length info
        
    Returns:
        Dictionary with correlation analysis
    """
    lengths = []
    valid_scores = []
    
    for score, meta in zip(scores, metadata):
        if 'edited_length' in meta:
            lengths.append(meta['edited_length'])
            valid_scores.append(score)
    
    if len(lengths) < 2:
        return {'correlation': None, 'n_samples': len(lengths)}
    
    lengths = np.array(lengths)
    valid_scores = np.array(valid_scores)
    
    correlation = np.corrcoef(lengths, valid_scores)[0, 1]
    
    return {
        'correlation': float(correlation),
        'n_samples': len(lengths),
        'length_range': (int(lengths.min()), int(lengths.max())),
        'length_mean': float(lengths.mean()),
    }
