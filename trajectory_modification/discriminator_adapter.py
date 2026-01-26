"""
Discriminator adapter for FAMAIL trajectory modification.

Wraps the trained SiameseLSTM discriminator for evaluating trajectory
similarity during the modification loop.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .trajectory import Trajectory


class DiscriminatorAdapter:
    """
    Adapter for the trained trajectory discriminator.
    
    The discriminator evaluates f(τ, τ') ∈ [0, 1]:
    - f ≈ 1: Trajectories appear to be from the same agent
    - f ≈ 0: Trajectories appear to be from different agents
    
    During modification, we perturb τ' until f(τ, τ') drops below threshold,
    indicating the modification is significant enough to affect driver identity.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = 'cpu',
        threshold: float = 0.5,
    ):
        """
        Initialize discriminator adapter.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device for inference ('cpu' or 'cuda')
            threshold: Similarity threshold for "same agent" classification
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for discriminator")
        
        self.device = torch.device(device)
        self.threshold = threshold
        self.model = None
        
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load trained model from checkpoint."""
        import sys
        checkpoint_path = Path(checkpoint_path)
        
        # Add discriminator model to path
        model_dir = checkpoint_path.parent.parent
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        
        from discriminator.model import SiameseLSTMDiscriminator, SiameseLSTMDiscriminatorV2
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Determine model version from checkpoint
        config = checkpoint.get('config', checkpoint.get('model_config', {}))
        model_version = config.get('model_version', 'v1')
        
        if model_version == 'v2':
            self.model = SiameseLSTMDiscriminatorV2(
                lstm_hidden_dims=config.get('lstm_hidden_dims', (200, 100)),
                dropout=config.get('dropout', 0.2),
                bidirectional=config.get('bidirectional', True),
                combination_mode=config.get('combination_mode', 'difference'),
            )
        else:
            self.model = SiameseLSTMDiscriminator(
                lstm_hidden_dims=config.get('lstm_hidden_dims', (200, 100)),
                dropout=config.get('dropout', 0.2),
                bidirectional=config.get('bidirectional', True),
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def evaluate(
        self,
        tau_original: Trajectory,
        tau_modified: Trajectory,
    ) -> float:
        """
        Evaluate similarity between original and modified trajectory.
        
        Args:
            tau_original: Original trajectory τ
            tau_modified: Modified trajectory τ'
            
        Returns:
            Similarity score f(τ, τ') ∈ [0, 1]
        """
        if self.model is None:
            # Return high similarity if no model loaded (for testing)
            return 0.9
        
        # Convert to tensors
        x1 = tau_original.to_tensor().unsqueeze(0).to(self.device)
        x2 = tau_modified.to_tensor().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            similarity = self.model(x1, x2)
        
        return float(similarity.item())
    
    def is_same_agent(
        self,
        tau_original: Trajectory,
        tau_modified: Trajectory,
    ) -> bool:
        """Check if discriminator identifies trajectories as same agent."""
        return self.evaluate(tau_original, tau_modified) >= self.threshold
    
    def get_similarity_with_grad(
        self,
        tau_features: 'torch.Tensor',
        tau_prime_features: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Get similarity with gradient tracking (for optimization).
        
        Args:
            tau_features: Original trajectory tensor [1, seq_len, 4]
            tau_prime_features: Modified trajectory tensor [1, seq_len, 4]
            
        Returns:
            Similarity tensor with gradients
        """
        if self.model is None:
            return torch.tensor(0.9, requires_grad=True)
        
        # Enable grad for this forward pass
        with torch.enable_grad():
            similarity = self.model(
                tau_features.to(self.device),
                tau_prime_features.to(self.device),
            )
        
        return similarity
