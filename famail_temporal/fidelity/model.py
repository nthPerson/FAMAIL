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
from typing import Optional, Tuple


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


class SiameseLSTMEncoder(nn.Module):
    """LSTM encoder for trajectory sequences.

    Processes a sequence of trajectory features and produces a fixed-size embedding.
    Supports masking for variable-length sequences.

    Architecture follows ST-SiameseNet from ST-iFGSM paper, supporting variable
    hidden dimensions per layer (e.g., [200, 100] for a 2-layer LSTM where the
    first layer has 200 hidden units and the second has 100).
    """

    def __init__(self,
                 input_dim: int = 6,
                 lstm_hidden_dims: Tuple[int, ...] = (200, 100),
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        """Initialize the LSTM encoder.

        Args:
            input_dim: Number of input features (6 after normalization)
            lstm_hidden_dims: Tuple of hidden dimensions for each LSTM layer.
                              Default (200, 100) follows ST-SiameseNet architecture.
                              Length determines the number of layers.
            dropout: Dropout probability between layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        self.lstm_hidden_dims = lstm_hidden_dims
        self.num_layers = len(lstm_hidden_dims)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = dropout

        # Build stacked LSTM layers with variable hidden dimensions
        # Each layer is a separate nn.LSTM to allow different hidden sizes
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        current_input_dim = input_dim
        for i, hidden_dim in enumerate(lstm_hidden_dims):
            lstm = nn.LSTM(
                input_size=current_input_dim,
                hidden_size=hidden_dim,
                num_layers=1,  # Single layer each
                batch_first=True,
                bidirectional=bidirectional
            )
            self.lstm_layers.append(lstm)

            # Add dropout between layers (not after the last layer)
            if i < len(lstm_hidden_dims) - 1 and dropout > 0:
                self.dropout_layers.append(nn.Dropout(dropout))

            # Next layer's input is current layer's output
            current_input_dim = hidden_dim * self.num_directions

        # Output dimension is the final layer's hidden dim * directions
        self.output_dim = lstm_hidden_dims[-1] * self.num_directions

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a trajectory sequence.

        Args:
            x: Normalized features [batch, seq_len, input_dim]
            mask: Boolean mask [batch, seq_len] where True = valid timestep

        Returns:
            Embedding [batch, output_dim]
        """
        batch_size = x.size(0)

        # Calculate actual sequence lengths from mask (if provided)
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            lengths = lengths.clamp(min=1)
        else:
            lengths = None

        # Process through each LSTM layer
        current_output = x
        final_h_n = None

        for i, lstm in enumerate(self.lstm_layers):
            if lengths is not None:
                # Pack padded sequence for efficient LSTM processing
                packed = nn.utils.rnn.pack_padded_sequence(
                    current_output, lengths, batch_first=True, enforce_sorted=False
                )
                packed_output, (h_n, _) = lstm(packed)
                # Unpack for next layer
                current_output, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_output, batch_first=True
                )
            else:
                current_output, (h_n, _) = lstm(current_output)

            final_h_n = h_n

            # Apply dropout between layers
            if i < len(self.dropout_layers):
                current_output = self.dropout_layers[i](current_output)

        # h_n shape: [num_directions, batch, hidden_dim] (since each lstm has 1 layer)
        # We want the final layer's hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            # Forward: final_h_n[0], Backward: final_h_n[1]
            embedding = torch.cat([final_h_n[0], final_h_n[1]], dim=-1)
        else:
            # Just use the hidden state
            embedding = final_h_n[0]

        return embedding  # [batch, output_dim]


class ProfileEncoder(nn.Module):
    """FCN encoder for driver profile features (FCN_P stream).

    Processes a fixed-length profile feature vector through FC layers
    to produce a fixed-size embedding. Used as the profile stream in
    the multi-stream ST-SiameseNet (Ren et al., KDD 2020).

    Architecture follows Ren et al. Appendix A.3:
        FC layers [64, 32] → output_dim=8, with ReLU activation and dropout.
        No activation on the final layer (raw embedding representation).
    """

    def __init__(self,
                 input_dim: int = 11,
                 hidden_dims: Tuple[int, ...] = (64, 32),
                 output_dim: int = 8,
                 dropout: float = 0.2):
        """Initialize the profile encoder.

        Args:
            input_dim: Number of profile features (default 11)
            hidden_dims: Hidden layer sizes before the output layer
            output_dim: Embedding dimension (default 8)
            dropout: Dropout probability
        """
        super().__init__()

        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hdim

        # Final projection to output_dim (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode profile features.

        Args:
            x: Profile feature vector [batch, input_dim]

        Returns:
            Embedding [batch, output_dim]
        """
        return self.network(x)
