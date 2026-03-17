"""Multi-stream dataset generation for Ren-aligned ST-SiameseNet training.

Day-based pair sampling: each branch represents one driver on one calendar day,
with N independently-encoded trajectories per stream (seeking + driving) plus
a profile feature vector.
"""

from .config import MultiStreamGenerationConfig
from .generation import generate_multi_stream_dataset, load_multi_stream_data

__all__ = [
    "MultiStreamGenerationConfig",
    "generate_multi_stream_dataset",
    "load_multi_stream_data",
]
