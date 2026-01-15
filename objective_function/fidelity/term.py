"""
Trajectory Fidelity Term Implementation.

This module implements the Trajectory Fidelity term ($F_{\text{fidelity}}$) for the
FAMAIL objective function. The term measures how authentic edited trajectories
remain compared to genuine expert driver trajectories.

Mathematical Formulation:
    F_fidelity = (1/|T'|) * Σ_{τ' ∈ T'} Discriminator(τ', τ_ref)
    
where:
    - T' = set of edited trajectories
    - τ_ref = reference trajectory (original from same driver)
    - Discriminator(τ', τ_ref) = P(same agent | τ', τ_ref)

The discriminator is a pre-trained ST-SiameseNet model that outputs the
probability that two trajectories belong to the same agent. High fidelity
means edited trajectories are indistinguishable from originals.

Value Range: [0, 1]
    - F_fidelity = 1: Perfectly authentic (edited = original style)
    - F_fidelity = 0: Completely artificial (clearly distinguishable)

Reference:
    ST-iFGSM paper for discriminator-based trajectory authenticity
"""

import sys
import os
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import ObjectiveFunctionTerm, TermMetadata
from fidelity.config import FidelityConfig
from fidelity.utils import (
    load_discriminator,
    load_trajectory_data,
    get_trajectory_statistics,
    create_trajectory_pairs,
    compute_fidelity_scores,
    aggregate_fidelity_scores,
    compute_fidelity,
    DifferentiableFidelity,
    verify_fidelity_gradient,
    compute_per_driver_statistics,
    compute_score_histogram,
    compute_length_correlation,
)

# Optional PyTorch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# Global model cache
_MODEL_CACHE: Dict[str, Tuple[Any, dict, str]] = {}


class FidelityTerm(ObjectiveFunctionTerm):
    """
    Trajectory Fidelity term based on discriminator confidence.
    
    Measures how authentic edited trajectories appear compared to originals.
    Uses a pre-trained ST-SiameseNet discriminator to classify whether
    trajectory pairs belong to the same agent.
    
    Value Range: [0, 1]
        - F_fidelity = 1: Perfect authenticity (indistinguishable from original)
        - F_fidelity = 0: Completely artificial (easily detected as edited)
    
    Example:
        >>> config = FidelityConfig(checkpoint_path="path/to/best.pt")
        >>> term = FidelityTerm(config)
        >>> result = term.compute(edited_trajectories, {'original': original_trajectories})
        >>> print(f"Fidelity: {result:.4f}")
    """
    
    def __init__(self, config: Optional[FidelityConfig] = None):
        """
        Initialize the Fidelity term.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        if config is None:
            config = FidelityConfig()
        super().__init__(config)
        self.config: FidelityConfig = config
        
        # Model will be loaded lazily
        self._model = None
        self._model_config = None
        self._device = None
    
    def _build_metadata(self) -> TermMetadata:
        """Build and return the term's metadata."""
        return TermMetadata(
            name="trajectory_fidelity",
            display_name="Trajectory Fidelity",
            version="1.0.0",
            description=(
                "Discriminator-based measure of trajectory authenticity. "
                "Uses ST-SiameseNet to compute probability that edited trajectories "
                "match the style of original expert trajectories. "
                "Supports differentiable computation for gradient-based optimization."
            ),
            value_range=(0.0, 1.0),
            higher_is_better=True,
            is_differentiable=True,
            required_data=["original_trajectories"],
            optional_data=["trajectory_metadata"],
            author="FAMAIL Research Team",
            last_updated="2025-01-12",
        )
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        self.config.validate()
    
    def _ensure_model_loaded(self) -> None:
        """Ensure discriminator model is loaded."""
        if self._model is not None:
            return
        
        checkpoint_path = str(self.config.get_checkpoint_abs_path())
        
        # Check cache
        if self.config.cache_model and checkpoint_path in _MODEL_CACHE:
            self._model, self._model_config, self._device = _MODEL_CACHE[checkpoint_path]
            return
        
        # Load model
        device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        
        self._model, self._model_config, self._device = load_discriminator(
            checkpoint_path,
            config_path=self.config.get_config_path(),
            device=device
        )
        
        # Cache
        if self.config.cache_model:
            _MODEL_CACHE[checkpoint_path] = (self._model, self._model_config, self._device)
    
    def compute(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> float:
        """
        Compute the fidelity term value.
        
        Args:
            trajectories: Dictionary of edited trajectories
                {driver_id: [[state1, state2, ...], [state1, ...], ...]}
            auxiliary_data: Must contain 'original_trajectories' with same structure
            
        Returns:
            Fidelity value in [0, 1], higher = more authentic
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for fidelity computation")
        
        # Get original trajectories
        original_trajectories = auxiliary_data.get('original_trajectories')
        if original_trajectories is None:
            raise ValueError(
                "auxiliary_data must contain 'original_trajectories'. "
                "These are the reference trajectories to compare edited ones against."
            )
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Compute fidelity
        fidelity, _ = compute_fidelity(
            model=self._model,
            edited_trajectories=trajectories,
            original_trajectories=original_trajectories,
            mode=self.config.mode,
            aggregation=self.config.aggregation,
            threshold=self.config.threshold,
            batch_size=self.config.batch_size,
            max_length=self.config.max_trajectory_length,
            feature_range=self.config.get_feature_range(),
            device=self._device,
            min_length=self.config.min_trajectory_length,
        )
        
        return fidelity
    
    def compute_with_breakdown(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute fidelity with detailed breakdown.
        
        Args:
            trajectories: Dictionary of edited trajectories
            auxiliary_data: Must contain 'original_trajectories'
            
        Returns:
            Dictionary containing:
                - 'value': float - the final fidelity value
                - 'components': Dict - intermediate values
                - 'statistics': Dict - summary statistics
                - 'diagnostics': Dict - debugging information
        """
        start_time = time.time()
        
        if not TORCH_AVAILABLE:
            return {
                'value': 0.0,
                'components': {},
                'statistics': {},
                'diagnostics': {'error': 'PyTorch not available'},
            }
        
        # Get original trajectories
        original_trajectories = auxiliary_data.get('original_trajectories')
        if original_trajectories is None:
            return {
                'value': 0.0,
                'components': {},
                'statistics': {},
                'diagnostics': {'error': 'original_trajectories not provided'},
            }
        
        # Ensure model is loaded
        try:
            self._ensure_model_loaded()
        except Exception as e:
            return {
                'value': 0.0,
                'components': {},
                'statistics': {},
                'diagnostics': {'error': f'Failed to load model: {str(e)}'},
            }
        
        # Create trajectory pairs
        feature_range = self.config.get_feature_range()
        edited_list, original_list, metadata = create_trajectory_pairs(
            trajectories,
            original_trajectories,
            mode=self.config.mode,
            feature_range=feature_range,
            min_length=self.config.min_trajectory_length,
        )
        
        if len(edited_list) == 0:
            return {
                'value': 0.0,
                'components': {'n_pairs': 0},
                'statistics': {},
                'diagnostics': {'warning': 'No valid trajectory pairs found'},
            }
        
        # Compute scores
        scores = compute_fidelity_scores(
            self._model,
            edited_list,
            original_list,
            batch_size=self.config.batch_size,
            max_length=self.config.max_trajectory_length,
            device=self._device,
        )
        
        # Compute weights for weighted aggregation
        weights = None
        if self.config.aggregation == "weighted":
            weights = np.array([
                len(e) + len(o) for e, o in zip(edited_list, original_list)
            ], dtype=np.float32)
        
        # Aggregate
        fidelity = aggregate_fidelity_scores(
            scores,
            method=self.config.aggregation,
            threshold=self.config.threshold_aggregation_cutoff,
            weights=weights,
        )
        
        computation_time = time.time() - start_time
        
        # Compute additional statistics
        per_driver_stats = compute_per_driver_statistics(scores, metadata)
        hist_counts, hist_edges = compute_score_histogram(scores)
        length_corr = compute_length_correlation(scores, metadata)
        
        # Get trajectory statistics
        edited_stats = get_trajectory_statistics(trajectories)
        original_stats = get_trajectory_statistics(original_trajectories)
        
        return {
            'value': fidelity,
            'components': {
                'n_pairs': len(scores),
                'n_edited_drivers': len(trajectories),
                'n_original_drivers': len(original_trajectories),
                'scores': scores.tolist(),
                'metadata': metadata,
                'aggregation_method': self.config.aggregation,
                'mode': self.config.mode,
            },
            'statistics': {
                'score_mean': float(np.mean(scores)),
                'score_std': float(np.std(scores)),
                'score_min': float(np.min(scores)),
                'score_max': float(np.max(scores)),
                'score_median': float(np.median(scores)),
                'above_threshold': float(np.mean(scores >= self.config.threshold)),
                'above_0.7': float(np.mean(scores >= 0.7)),
                'above_0.9': float(np.mean(scores >= 0.9)),
                'per_driver': per_driver_stats,
                'histogram': {
                    'counts': hist_counts.tolist(),
                    'edges': hist_edges.tolist(),
                },
                'length_correlation': length_corr,
            },
            'diagnostics': {
                'computation_time_seconds': computation_time,
                'device': self._device,
                'edited_trajectory_stats': edited_stats,
                'original_trajectory_stats': original_stats,
                'model_config': self._model_config,
                'feature_range': feature_range,
            },
        }
    
    def get_differentiable_module(self) -> Optional['DifferentiableFidelity']:
        """
        Get a differentiable PyTorch module for gradient-based optimization.
        
        Returns:
            DifferentiableFidelity module, or None if PyTorch unavailable
        """
        if not TORCH_AVAILABLE:
            return None
        
        self._ensure_model_loaded()
        
        return DifferentiableFidelity(
            discriminator=self._model,
            aggregation=self.config.aggregation if self.config.aggregation in ["mean", "min"] else "mean"
        )
    
    def compute_gradient(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Compute gradient of fidelity with respect to edited trajectories.
        
        Args:
            trajectories: Dictionary of edited trajectories
            auxiliary_data: Must contain 'original_trajectories'
            
        Returns:
            Dictionary mapping driver_id to gradient arrays, or None if not differentiable
        """
        if not TORCH_AVAILABLE:
            return None
        
        original_trajectories = auxiliary_data.get('original_trajectories')
        if original_trajectories is None:
            return None
        
        self._ensure_model_loaded()
        
        # Get differentiable module
        diff_module = self.get_differentiable_module()
        feature_range = self.config.get_feature_range()
        
        gradients = {}
        
        # Process each driver separately
        for driver_id, edited_trajs in trajectories.items():
            if driver_id not in original_trajectories:
                continue
            
            orig_trajs = original_trajectories[driver_id]
            driver_grads = []
            
            for idx, edited_traj in enumerate(edited_trajs):
                if idx >= len(orig_trajs):
                    continue
                
                orig_traj = orig_trajs[idx]
                
                if len(edited_traj) < self.config.min_trajectory_length:
                    continue
                
                # Extract features
                from fidelity.utils import extract_trajectory_features, prepare_trajectory_batch
                
                edited_features = extract_trajectory_features(edited_traj, feature_range)
                orig_features = extract_trajectory_features(orig_traj, feature_range)
                
                # Prepare batch
                edited_batch, mask_edited = prepare_trajectory_batch([edited_features])
                orig_batch, mask_orig = prepare_trajectory_batch([orig_features])
                
                # Convert to tensors with gradient
                edited_tensor = torch.from_numpy(edited_batch).to(self._device).requires_grad_(True)
                orig_tensor = torch.from_numpy(orig_batch).to(self._device)
                mask_edited_t = torch.from_numpy(mask_edited).to(self._device)
                mask_orig_t = torch.from_numpy(mask_orig).to(self._device)
                
                # Forward and backward
                fidelity = diff_module(edited_tensor, orig_tensor, mask_edited_t, mask_orig_t)
                fidelity.backward()
                
                # Extract gradient
                grad = edited_tensor.grad.cpu().numpy()[0]  # Remove batch dimension
                driver_grads.append(grad)
            
            if driver_grads:
                gradients[driver_id] = driver_grads
        
        return gradients if gradients else None
    
    def verify_differentiability(
        self,
        trajectories: Optional[Dict[str, List[List[List[float]]]]] = None,
        auxiliary_data: Optional[Dict[str, Any]] = None,
        n_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Verify that gradients are computed correctly.
        
        Args:
            trajectories: Optional test trajectories (uses synthetic if None)
            auxiliary_data: Optional auxiliary data
            n_samples: Number of gradient samples to verify
            
        Returns:
            Dictionary with verification results
        """
        if not TORCH_AVAILABLE:
            return {'success': False, 'error': 'PyTorch not available'}
        
        self._ensure_model_loaded()
        
        # Create synthetic test data if not provided
        if trajectories is None:
            # Create simple synthetic trajectories
            np.random.seed(42)
            n_steps = 50
            
            edited_features = np.random.rand(1, n_steps, 4).astype(np.float32)
            # Scale to reasonable ranges
            edited_features[..., 0] *= 49  # x_grid
            edited_features[..., 1] *= 89  # y_grid
            edited_features[..., 2] *= 287  # time_bucket
            edited_features[..., 3] = (edited_features[..., 3] * 4 + 1).astype(int)  # day_index (1-5, Mon-Fri)
            
            original_features = edited_features + np.random.randn(1, n_steps, 4).astype(np.float32) * 0.1
            original_features = np.clip(original_features, 0, None)
        else:
            # Use provided trajectories
            feature_range = self.config.get_feature_range()
            
            # Get first trajectory pair
            driver_id = list(trajectories.keys())[0]
            edited_traj = trajectories[driver_id][0]
            
            orig_trajs = auxiliary_data.get('original_trajectories', {})
            if driver_id in orig_trajs and len(orig_trajs[driver_id]) > 0:
                orig_traj = orig_trajs[driver_id][0]
            else:
                orig_traj = edited_traj  # Self-comparison
            
            from fidelity.utils import extract_trajectory_features, prepare_trajectory_batch
            
            edited_features, _ = prepare_trajectory_batch(
                [extract_trajectory_features(edited_traj, feature_range)]
            )
            original_features, _ = prepare_trajectory_batch(
                [extract_trajectory_features(orig_traj, feature_range)]
            )
        
        # Convert to tensors
        edited_tensor = torch.from_numpy(edited_features).to(self._device)
        original_tensor = torch.from_numpy(original_features).to(self._device)
        
        # Verify gradients
        result = verify_fidelity_gradient(
            self._model,
            edited_tensor,
            original_tensor,
        )
        
        result['success'] = result.get('gradient_valid', False)
        
        return result
    
    @property
    def metadata(self) -> TermMetadata:
        """Get the term's metadata."""
        return self._metadata
    
    def __repr__(self) -> str:
        return (
            f"FidelityTerm(\n"
            f"  config={self.config},\n"
            f"  model_loaded={self._model is not None},\n"
            f"  device={self._device}\n"
            f")"
        )
