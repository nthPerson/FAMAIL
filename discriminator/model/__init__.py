"""Modified ST-SiameseNet Discriminator for FaMAIL.

This module implements a Siamese LSTM-based discriminator model that determines
whether two input trajectories belong to the same agent (driver).

Key components:
- SiameseLSTMDiscriminator: The main model architecture
- FeatureNormalizer: Handles spatial/temporal feature normalization
- TrajectoryPairDataset: PyTorch Dataset for loading trajectory pairs
- Trainer: Training loop with validation and metrics
"""

from .model import SiameseLSTMDiscriminator, FeatureNormalizer
from .dataset import TrajectoryPairDataset, load_dataset_from_directory
from .trainer import Trainer, TrainingConfig

__all__ = [
    "SiameseLSTMDiscriminator",
    "FeatureNormalizer",
    "TrajectoryPairDataset",
    "load_dataset_from_directory",
    "Trainer",
    "TrainingConfig",
]
