"""Modified ST-SiameseNet Discriminator for FaMAIL.

This module implements a Siamese LSTM-based discriminator model that determines
whether two input trajectories belong to the same agent (driver).

Key components:
- SiameseLSTMDiscriminator: The original model architecture (concatenation-based)
- SiameseLSTMDiscriminatorV2: Improved model with distance-based similarity
- FeatureNormalizer: Handles spatial/temporal feature normalization
- TrajectoryPairDataset: PyTorch Dataset for loading trajectory pairs
- Trainer: Training loop with validation and metrics

V2 Improvements:
- Distance-based embedding combination (|emb1 - emb2|) instead of concatenation
- Naturally handles identical trajectories (zero difference = high similarity)
- Optional additional metrics: cosine similarity, euclidean distance
"""

from .model import SiameseLSTMDiscriminator, SiameseLSTMDiscriminatorV2, FeatureNormalizer
from .dataset import TrajectoryPairDataset, load_dataset_from_directory
from .trainer import Trainer, TrainingConfig

__all__ = [
    "SiameseLSTMDiscriminator",
    "SiameseLSTMDiscriminatorV2",
    "FeatureNormalizer",
    "TrajectoryPairDataset",
    "load_dataset_from_directory",
    "Trainer",
    "TrainingConfig",
]
