"""
Multi-stream Siamese discriminator — inference-only port.

Only four classes are ported from discriminator/model/model.py:
  - FeatureNormalizer
  - SiameseLSTMEncoder
  - ProfileEncoder
  - MultiStreamSiameseDiscriminator

Training loops, dataset classes, and five deprecated alternate architectures
(SiameseLSTMDiscriminator, TransformerEncoder, SiameseTransformerDiscriminator,
SiameseLSTMDiscriminatorV2) are intentionally excluded.

Classes are lifted verbatim from the original model.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureNormalizer(nn.Module):
    """Normalizes raw trajectory features for model input.

    Input features (4): [x_grid, y_grid, time_bucket, day_index]
    Output features (6): [x_norm, y_norm, sin_time, cos_time, sin_day, cos_day]

    Spatial normalization:
        - x_norm = x_grid / 49 (grid is 50 wide, 0-49)
        - y_norm = y_grid / 89 (grid is 90 tall, 0-89)

    Temporal cyclic encoding:
        - time_bucket ∈ [0, 287] → angle = 2π * time_bucket / 288
        - day_index ∈ [1, 5] → angle = 2π * (day_index - 1) / 5 (Monday=1 to Friday=5)
        - Output: (sin(angle), cos(angle)) for each

    Note: Our dataset only contains weekday data (Monday-Friday), so we use
    5-day cyclic encoding instead of 7-day.
    """

    def __init__(self,
                 x_max: float = 49.0,
                 y_max: float = 89.0,
                 time_buckets: int = 288,
                 days_in_week: int = 5):
        """Initialize the normalizer.

        Args:
            x_max: Maximum x_grid value (default 49 for 50-wide grid)
            y_max: Maximum y_grid value (default 89 for 90-tall grid)
            time_buckets: Number of time buckets per day (default 288 = 5-min intervals)
            days_in_week: Number of days in cycle (default 5 for Mon-Fri data)
        """
        super().__init__()
        self.x_max = x_max
        self.y_max = y_max
        self.time_buckets = time_buckets
        self.days_in_week = days_in_week

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize raw features.

        Args:
            x: Raw features tensor [batch, seq_len, 4] or [seq_len, 4]
                Features: [x_grid, y_grid, time_bucket, day_index]

        Returns:
            Normalized features [batch, seq_len, 6] or [seq_len, 6]
                Features: [x_norm, y_norm, sin_time, cos_time, sin_day, cos_day]
        """
        # Handle both batched and unbatched input
        original_shape = x.shape
        if len(original_shape) == 2:
            x = x.unsqueeze(0)

        # Extract individual features
        x_grid = x[..., 0]       # [batch, seq_len]
        y_grid = x[..., 1]       # [batch, seq_len]
        time_bucket = x[..., 2]  # [batch, seq_len]
        day_index = x[..., 3]    # [batch, seq_len]

        # Spatial normalization (min-max to [0, 1])
        x_norm = x_grid / self.x_max
        y_norm = y_grid / self.y_max

        # Temporal cyclic encoding
        time_angle = 2 * math.pi * time_bucket / self.time_buckets
        # day_index is 1-indexed (1=Mon, 5=Fri), convert to 0-indexed for cyclic encoding
        day_angle = 2 * math.pi * (day_index - 1) / self.days_in_week

        sin_time = torch.sin(time_angle)
        cos_time = torch.cos(time_angle)
        sin_day = torch.sin(day_angle)
        cos_day = torch.cos(day_angle)

        # Stack normalized features
        normalized = torch.stack([
            x_norm, y_norm, sin_time, cos_time, sin_day, cos_day
        ], dim=-1)

        # Restore original batch dimension if needed
        if len(original_shape) == 2:
            normalized = normalized.squeeze(0)

        return normalized
