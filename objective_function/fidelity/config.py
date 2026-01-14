"""
Configuration for the Trajectory Fidelity Term.

This module defines all configurable parameters for the fidelity computation,
including discriminator model settings, batch processing options, and 
aggregation strategies.

Fidelity Computation Modes:
    1. "same_agent" - Compare edited trajectories with their originals from same agent
    2. "paired" - Compare edited trajectories with specific paired reference trajectories
    3. "batch" - Evaluate single trajectories against a reference set
    
Aggregation Strategies:
    - "mean": Average discriminator scores across all trajectory pairs
    - "min": Minimum score (most conservative - worst-case authenticity)
    - "threshold": Fraction of trajectories above threshold
    - "weighted": Weighted average based on trajectory lengths
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional, List, Literal, Union
import sys
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import TermConfig


# Default checkpoint path (relative to project root)
DEFAULT_CHECKPOINT = "discriminator/model/checkpoints/20251229_131300/best.pt"


@dataclass
class FidelityConfig(TermConfig):
    """
    Configuration for the Trajectory Fidelity term.
    
    Attributes:
        checkpoint_path: Path to trained discriminator checkpoint (.pt file)
        config_path: Path to discriminator config JSON (auto-detected if None)
        
        mode: How to pair trajectories for comparison
            - "same_agent": Compare edited trajectory with original from same agent
            - "paired": Compare with explicitly paired reference trajectories
            - "batch": Evaluate against a reference trajectory set
            
        aggregation: How to aggregate per-pair scores into final fidelity
            - "mean": Average of all discriminator scores
            - "min": Minimum score (worst-case fidelity)
            - "threshold": Fraction of pairs exceeding threshold
            - "weighted": Length-weighted average
            
        threshold: Classification threshold for discriminator (default 0.5)
        threshold_aggregation_cutoff: Cutoff for threshold aggregation mode
        
        batch_size: Number of trajectory pairs per batch for GPU processing
        max_trajectory_length: Maximum states per trajectory (for padding)
        use_gpu: Whether to use GPU acceleration if available
        
        extract_features: Which features to extract from 126-element state vectors
            - "base_4": [x_grid, y_grid, time_bucket, day_index] (indices 0-3)
            - "all_126": Full state vector (requires compatible discriminator)
            - (start, end): Custom range, e.g., (0, 4) for base features
            
        require_valid_original: If True, skip pairs where original is missing
        min_trajectory_length: Minimum states for valid trajectory
        
        cache_model: Whether to cache loaded model in memory
    """
    
    # Model paths
    checkpoint_path: str = DEFAULT_CHECKPOINT
    config_path: Optional[str] = None  # Auto-detected from checkpoint directory
    
    # Computation mode
    mode: Literal["same_agent", "paired", "batch"] = "same_agent"
    
    # Aggregation strategy
    aggregation: Literal["mean", "min", "threshold", "weighted"] = "mean"
    threshold: float = 0.5  # Discriminator classification threshold
    threshold_aggregation_cutoff: float = 0.5  # For threshold aggregation
    
    # Batch processing
    batch_size: int = 32
    max_trajectory_length: int = 1000
    use_gpu: bool = True
    
    # Feature extraction
    extract_features: Union[Literal["base_4", "all_126"], Tuple[int, int]] = "base_4"
    
    # Validation
    require_valid_original: bool = True
    min_trajectory_length: int = 2
    
    # Caching
    cache_model: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        super().validate()
        
        # Validate checkpoint path
        if not Path(self.checkpoint_path).exists():
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            abs_path = project_root / self.checkpoint_path
            if not abs_path.exists():
                raise ValueError(
                    f"Discriminator checkpoint not found: {self.checkpoint_path}\n"
                    f"Also tried: {abs_path}"
                )
        
        # Validate mode
        if self.mode not in ["same_agent", "paired", "batch"]:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # Validate aggregation
        if self.aggregation not in ["mean", "min", "threshold", "weighted"]:
            raise ValueError(f"Invalid aggregation: {self.aggregation}")
        
        # Validate threshold
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {self.threshold}")
        
        if not 0.0 <= self.threshold_aggregation_cutoff <= 1.0:
            raise ValueError(
                f"Threshold aggregation cutoff must be in [0, 1], "
                f"got {self.threshold_aggregation_cutoff}"
            )
        
        # Validate batch size
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be >= 1, got {self.batch_size}")
        
        # Validate max trajectory length
        if self.max_trajectory_length < self.min_trajectory_length:
            raise ValueError(
                f"max_trajectory_length ({self.max_trajectory_length}) must be >= "
                f"min_trajectory_length ({self.min_trajectory_length})"
            )
        
        # Validate feature extraction
        if isinstance(self.extract_features, tuple):
            start, end = self.extract_features
            if start < 0 or end > 126 or start >= end:
                raise ValueError(
                    f"Invalid feature range: ({start}, {end}). "
                    f"Must be 0 <= start < end <= 126"
                )
    
    def get_checkpoint_abs_path(self) -> Path:
        """Get absolute path to checkpoint file."""
        path = Path(self.checkpoint_path)
        if path.exists():
            return path.resolve()
        
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        return (project_root / self.checkpoint_path).resolve()
    
    def get_config_path(self) -> Optional[Path]:
        """Get path to config.json for the checkpoint."""
        if self.config_path:
            return Path(self.config_path)
        
        # Auto-detect from checkpoint directory
        checkpoint_dir = self.get_checkpoint_abs_path().parent
        config_file = checkpoint_dir / "config.json"
        if config_file.exists():
            return config_file
        return None
    
    def load_discriminator_config(self) -> Optional[dict]:
        """Load discriminator configuration from config.json."""
        config_path = self.get_config_path()
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_feature_range(self) -> Tuple[int, int]:
        """Get feature extraction range as (start, end) indices."""
        if self.extract_features == "base_4":
            return (0, 4)
        elif self.extract_features == "all_126":
            return (0, 126)
        elif isinstance(self.extract_features, tuple):
            return self.extract_features
        else:
            raise ValueError(f"Invalid extract_features: {self.extract_features}")
    
    def __repr__(self) -> str:
        return (
            f"FidelityConfig(\n"
            f"  checkpoint={Path(self.checkpoint_path).name},\n"
            f"  mode='{self.mode}',\n"
            f"  aggregation='{self.aggregation}',\n"
            f"  threshold={self.threshold},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )


# =============================================================================
# PREDEFINED CONFIGURATIONS
# =============================================================================

DEFAULT_CONFIG = FidelityConfig(
    checkpoint_path=DEFAULT_CHECKPOINT,
    mode="same_agent",
    aggregation="mean",
    threshold=0.5,
    batch_size=32,
    use_gpu=True,
    extract_features="base_4",
)

HIGH_THRESHOLD_CONFIG = FidelityConfig(
    checkpoint_path=DEFAULT_CHECKPOINT,
    mode="same_agent",
    aggregation="threshold",
    threshold=0.7,
    threshold_aggregation_cutoff=0.8,
    batch_size=32,
    use_gpu=True,
)

CONSERVATIVE_CONFIG = FidelityConfig(
    checkpoint_path=DEFAULT_CHECKPOINT,
    mode="same_agent",
    aggregation="min",
    threshold=0.5,
    batch_size=32,
    use_gpu=True,
)

BATCH_EVAL_CONFIG = FidelityConfig(
    checkpoint_path=DEFAULT_CHECKPOINT,
    mode="batch",
    aggregation="mean",
    threshold=0.5,
    batch_size=64,
    use_gpu=True,
)
