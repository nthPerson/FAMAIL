"""Modified ST-SiameseNet Discriminator Model Architecture.

Implements a Siamese LSTM network with shared weights for determining
whether two trajectory sequences belong to the same agent.

Architecture:
1. Feature Normalization: 
   - Spatial (x, y): min-max normalization to [0,1]
   - Temporal (time_bucket, day_index): cyclic sin/cos encoding (2 features each)
   - Final input: 6 features per timestep
   
2. Siamese LSTM Encoder:
   - Shared-weight LSTM processes both trajectories
   - Final hidden state as embedding
   
3. Embedding Combination:
   - Concatenation of both embeddings
   
4. Classifier:
   - FC layers with dropout
   - Sigmoid output: 1 = same agent, 0 = different agent
"""

import torch
import torch.nn as nn
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
        - day_index ∈ [0, 6] → angle = 2π * day_index / 7
        - Output: (sin(angle), cos(angle)) for each
    """
    
    def __init__(self, 
                 x_max: float = 49.0,
                 y_max: float = 89.0,
                 time_buckets: int = 288,
                 days_in_week: int = 7):
        """Initialize the normalizer.
        
        Args:
            x_max: Maximum x_grid value (default 49 for 50-wide grid)
            y_max: Maximum y_grid value (default 89 for 90-tall grid)
            time_buckets: Number of time buckets per day (default 288 = 5-min intervals)
            days_in_week: Number of days in cycle (default 7)
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
        day_angle = 2 * math.pi * day_index / self.days_in_week
        
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
    """
    
    def __init__(self,
                 input_dim: int = 6,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        """Initialize the LSTM encoder.
        
        Args:
            input_dim: Number of input features (6 after normalization)
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability between layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension after bidirectional concatenation
        self.output_dim = hidden_dim * self.num_directions
        
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
        seq_len = x.size(1)
        
        if mask is not None:
            # Calculate actual sequence lengths from mask
            lengths = mask.sum(dim=1).cpu()
            
            # Clamp to at least 1 to avoid errors with empty sequences
            lengths = lengths.clamp(min=1)
            
            # Pack padded sequence for efficient LSTM processing
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            
            # Process through LSTM
            _, (h_n, _) = self.lstm(packed)
        else:
            # No masking - process full sequences
            _, (h_n, _) = self.lstm(x)
        
        # h_n shape: [num_layers * num_directions, batch, hidden_dim]
        # We want the final layer's hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            # Last layer forward: h_n[-2]
            # Last layer backward: h_n[-1]
            embedding = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            # Just use the last layer's hidden state
            embedding = h_n[-1]
            
        return embedding  # [batch, output_dim]


class SiameseLSTMDiscriminator(nn.Module):
    """Siamese LSTM Discriminator for trajectory pair classification.
    
    Determines whether two trajectories belong to the same agent.
    
    Architecture:
        1. Feature normalization (4 → 6 features)
        2. Shared LSTM encoder for both trajectories
        3. Concatenate embeddings
        4. FC classifier with sigmoid output
        
    Output:
        - 1 = same agent (positive pair)
        - 0 = different agent (negative pair)
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 classifier_hidden_dims: Tuple[int, ...] = (128, 64)):
        """Initialize the discriminator.
        
        Args:
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            classifier_hidden_dims: Hidden layer sizes for the classifier MLP
        """
        super().__init__()
        
        # Feature normalizer (4 raw features → 6 normalized features)
        self.normalizer = FeatureNormalizer()
        
        # Shared LSTM encoder
        self.encoder = SiameseLSTMEncoder(
            input_dim=6,  # After normalization
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Classifier MLP
        # Input: concatenation of two embeddings
        encoder_output_dim = self.encoder.output_dim
        classifier_input_dim = encoder_output_dim * 2  # Concatenated embeddings
        
        layers = []
        prev_dim = classifier_input_dim
        for hdim in classifier_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hdim
            
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
        
        # Store config for checkpointing
        self.config = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "classifier_hidden_dims": classifier_hidden_dims
        }
        
    def encode(self, 
               x: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a single trajectory.
        
        Args:
            x: Raw features [batch, seq_len, 4]
            mask: Boolean mask [batch, seq_len]
            
        Returns:
            Embedding [batch, encoder_output_dim]
        """
        x_norm = self.normalizer(x)
        return self.encoder(x_norm, mask)
    
    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor,
                mask1: Optional[torch.Tensor] = None,
                mask2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute same-agent probability for trajectory pairs.
        
        Args:
            x1: First trajectory raw features [batch, seq_len, 4]
            x2: Second trajectory raw features [batch, seq_len, 4]
            mask1: Boolean mask for x1 [batch, seq_len]
            mask2: Boolean mask for x2 [batch, seq_len]
            
        Returns:
            Same-agent probability [batch, 1] in range [0, 1]
        """
        # Encode both trajectories with shared encoder
        emb1 = self.encode(x1, mask1)  # [batch, emb_dim]
        emb2 = self.encode(x2, mask2)  # [batch, emb_dim]
        
        # Concatenate embeddings
        combined = torch.cat([emb1, emb2], dim=-1)  # [batch, 2*emb_dim]
        
        # Classify
        logits = self.classifier(combined)  # [batch, 1]
        probs = torch.sigmoid(logits)
        
        return probs
    
    def predict(self,
                x1: torch.Tensor,
                x2: torch.Tensor,
                mask1: Optional[torch.Tensor] = None,
                mask2: Optional[torch.Tensor] = None,
                threshold: float = 0.5) -> torch.Tensor:
        """Predict binary labels for trajectory pairs.
        
        Args:
            x1, x2, mask1, mask2: Same as forward()
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Binary predictions [batch] where 1 = same agent
        """
        probs = self.forward(x1, x2, mask1, mask2)
        return (probs.squeeze(-1) >= threshold).long()
    
    def get_embeddings(self,
                       x1: torch.Tensor,
                       x2: torch.Tensor,
                       mask1: Optional[torch.Tensor] = None,
                       mask2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get trajectory embeddings (useful for visualization).
        
        Args:
            x1, x2, mask1, mask2: Same as forward()
            
        Returns:
            Tuple of (emb1, emb2) each [batch, emb_dim]
        """
        emb1 = self.encode(x1, mask1)
        emb2 = self.encode(x2, mask2)
        return emb1, emb2


# Alternative: Transformer-based architecture (for future experimentation)
class TransformerEncoder(nn.Module):
    """Transformer encoder for trajectory sequences (experimental).
    
    Alternative to LSTM encoder using self-attention mechanism.
    """
    
    def __init__(self,
                 input_dim: int = 6,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 max_seq_len: int = 1000):
        """Initialize transformer encoder.
        
        Args:
            input_dim: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_dim = d_model
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode trajectory sequence.
        
        Args:
            x: Input features [batch, seq_len, input_dim]
            mask: Boolean mask [batch, seq_len] where True = valid
            
        Returns:
            Embedding [batch, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc
        
        # Create attention mask if provided
        # TransformerEncoder expects src_key_padding_mask where True = IGNORE
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # Invert: True becomes False (attend), False becomes True (ignore)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # [batch, seq_len, d_model]
        
        # Pool over sequence (mean pooling over valid positions)
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            # Simple mean pooling
            x = x.mean(dim=1)
            
        return x  # [batch, d_model]


class SiameseTransformerDiscriminator(nn.Module):
    """Siamese Transformer Discriminator (experimental alternative).
    
    Uses transformer encoder instead of LSTM. May be better for longer sequences.
    """
    
    def __init__(self,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 classifier_hidden_dims: Tuple[int, ...] = (128, 64),
                 max_seq_len: int = 1000):
        """Initialize transformer discriminator."""
        super().__init__()
        
        self.normalizer = FeatureNormalizer()
        
        self.encoder = TransformerEncoder(
            input_dim=6,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # Classifier
        classifier_input_dim = self.encoder.output_dim * 2
        
        layers = []
        prev_dim = classifier_input_dim
        for hdim in classifier_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
        
        self.config = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
            "classifier_hidden_dims": classifier_hidden_dims,
            "max_seq_len": max_seq_len
        }
        
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.normalizer(x)
        return self.encoder(x_norm, mask)
    
    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor,
                mask1: Optional[torch.Tensor] = None,
                mask2: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb1 = self.encode(x1, mask1)
        emb2 = self.encode(x2, mask2)
        combined = torch.cat([emb1, emb2], dim=-1)
        logits = self.classifier(combined)
        return torch.sigmoid(logits)
