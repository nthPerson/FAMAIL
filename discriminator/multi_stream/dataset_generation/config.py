"""Configuration for multi-stream dataset generation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass
class MultiStreamGenerationConfig:
    """Configuration for Ren-aligned day-based pair sampling.

    Each pair samples N trajectories per stream (seeking/driving) per branch,
    where a branch represents one driver on one calendar day.
    """

    # Input/output paths
    extracted_data_dir: Path = field(
        default_factory=lambda: _project_root() / "discriminator" / "multi_stream" / "extracted_data"
    )
    output_dir: Path = field(
        default_factory=lambda: _project_root() / "discriminator" / "multi_stream" / "datasets" / "default"
    )

    # Pair counts
    positive_pairs: int = 5000
    negative_pairs: int = 5000
    identical_pair_ratio: float = 0.1  # fraction of positive pairs that are identical

    # Ren-aligned parameters
    n_trajs_per_stream: int = 5        # trajectories per stream per branch (Ren: 5)
    min_trajs_per_day: int = 5         # minimum trajs in BOTH streams for day to be usable

    # Alignment (independent per stream)
    seeking_padding: str = "pad_to_longer"   # pad_to_longer | fixed_length
    driving_padding: str = "pad_to_longer"
    seeking_fixed_length: Optional[int] = None
    driving_fixed_length: Optional[int] = None

    # Sampling
    positive_strategy: str = "random"       # random
    negative_strategy: str = "random"       # random
    agent_distribution: str = "uniform"     # uniform | proportional
    ensure_agent_coverage: bool = True

    # Profile
    profile_noise_std: float = 0.0     # Gaussian noise on profile features (0 = off)

    # Split
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "extracted_data_dir": str(self.extracted_data_dir),
            "output_dir": str(self.output_dir),
            "positive_pairs": self.positive_pairs,
            "negative_pairs": self.negative_pairs,
            "identical_pair_ratio": self.identical_pair_ratio,
            "n_trajs_per_stream": self.n_trajs_per_stream,
            "min_trajs_per_day": self.min_trajs_per_day,
            "seeking_padding": self.seeking_padding,
            "driving_padding": self.driving_padding,
            "seeking_fixed_length": self.seeking_fixed_length,
            "driving_fixed_length": self.driving_fixed_length,
            "positive_strategy": self.positive_strategy,
            "negative_strategy": self.negative_strategy,
            "agent_distribution": self.agent_distribution,
            "ensure_agent_coverage": self.ensure_agent_coverage,
            "profile_noise_std": self.profile_noise_std,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
        }
