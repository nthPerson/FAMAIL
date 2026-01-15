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


class SiameseLSTMDiscriminator(nn.Module):
    """Siamese LSTM Discriminator for trajectory pair classification.
    
    Determines whether two trajectories belong to the same agent.
    
    Architecture follows ST-SiameseNet from ST-iFGSM paper:
        1. Feature normalization (4 → 6 features)
        2. Shared LSTM encoder with variable hidden dims per layer
        3. Concatenate embeddings
        4. FC classifier with sigmoid output
        
    Default architecture:
        - LSTM: [200, 100] (200 hidden units in layer 1, 100 in layer 2)
        - Classifier: [64, 32, 8] (three hidden layers with these sizes)
        - Output: 1 (binary classification)
        
    Output:
        - 1 = same agent (positive pair)
        - 0 = different agent (negative pair)
    """
    
    def __init__(self,
                 lstm_hidden_dims: Tuple[int, ...] = (200, 100),
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 classifier_hidden_dims: Tuple[int, ...] = (64, 32, 8)):
        """Initialize the discriminator.
        
        Args:
            lstm_hidden_dims: Hidden dimensions for each LSTM layer.
                              Default (200, 100) follows ST-SiameseNet.
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            classifier_hidden_dims: Hidden layer sizes for the classifier MLP.
                                    Default (64, 32, 8) follows ST-SiameseNet.
        """
        super().__init__()
        
        # Feature normalizer (4 raw features → 6 normalized features)
        self.normalizer = FeatureNormalizer()
        
        # Shared LSTM encoder
        self.encoder = SiameseLSTMEncoder(
            input_dim=6,  # After normalization
            lstm_hidden_dims=lstm_hidden_dims,
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
            "lstm_hidden_dims": lstm_hidden_dims,
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


class SiameseLSTMDiscriminatorV2(nn.Module):
    """Improved Siamese LSTM Discriminator with distance-based similarity.
    
    This version addresses the issue where the original model fails on identical
    trajectories because it was trained only on different trajectories.
    
    Key improvements:
    1. Distance-based combination: Uses |emb1 - emb2| instead of concatenation
       - Identical trajectories → zero difference → high similarity
       - Different trajectories → non-zero difference → classified separately
    
    2. Optional similarity metrics: Can combine multiple similarity measures
       - Euclidean distance: ||emb1 - emb2||
       - Cosine similarity: emb1 · emb2 / (||emb1|| * ||emb2||)
       - Element-wise difference: |emb1 - emb2|
    
    Architecture follows ST-SiameseNet from ST-iFGSM paper:
        1. Feature normalization (4 → 6 features)
        2. Shared LSTM encoder with variable hidden dims per layer
        3. Similarity-based combination (not concatenation)
        4. FC classifier with sigmoid output
        
    Default architecture:
        - LSTM: [200, 100] (200 hidden units in layer 1, 100 in layer 2)
        - Classifier: [64, 32, 8] (three hidden layers with these sizes)
        - Output: 1 (binary classification)
        
    Output:
        - 1 = same agent (positive pair)
        - 0 = different agent (negative pair)
    """
    
    def __init__(self,
                 lstm_hidden_dims: Tuple[int, ...] = (200, 100),
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 classifier_hidden_dims: Tuple[int, ...] = (64, 32, 8),
                 combination_mode: str = "difference"):
        """Initialize the improved discriminator.
        
        Args:
            lstm_hidden_dims: Hidden dimensions for each LSTM layer.
                              Default (200, 100) follows ST-SiameseNet.
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            classifier_hidden_dims: Hidden layer sizes for the classifier MLP.
                                    Default (64, 32, 8) follows ST-SiameseNet.
            combination_mode: How to combine embeddings
                - "difference": |emb1 - emb2| (absolute element-wise difference)
                - "distance": Additional distance features (cosine, euclidean)
                - "hybrid": Both difference and concatenation
        """
        super().__init__()
        
        self.combination_mode = combination_mode
        
        # Feature normalizer (4 raw features → 6 normalized features)
        self.normalizer = FeatureNormalizer()
        
        # Shared LSTM encoder
        self.encoder = SiameseLSTMEncoder(
            input_dim=6,  # After normalization
            lstm_hidden_dims=lstm_hidden_dims,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Calculate classifier input dimension based on combination mode
        encoder_output_dim = self.encoder.output_dim
        
        if combination_mode == "difference":
            # Just the element-wise difference
            classifier_input_dim = encoder_output_dim
        elif combination_mode == "distance":
            # Difference + cosine similarity + euclidean distance
            classifier_input_dim = encoder_output_dim + 2
        elif combination_mode == "hybrid":
            # Concatenation + difference + distance metrics
            classifier_input_dim = encoder_output_dim * 3 + 2
        else:
            raise ValueError(f"Unknown combination_mode: {combination_mode}")
        
        # Classifier MLP
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
            "lstm_hidden_dims": lstm_hidden_dims,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "classifier_hidden_dims": classifier_hidden_dims,
            "combination_mode": combination_mode,
            "model_version": "v2"
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
    
    def _combine_embeddings(self, 
                            emb1: torch.Tensor, 
                            emb2: torch.Tensor) -> torch.Tensor:
        """Combine embeddings based on combination mode.
        
        Args:
            emb1: First embedding [batch, emb_dim]
            emb2: Second embedding [batch, emb_dim]
            
        Returns:
            Combined features [batch, combined_dim]
        """
        # Element-wise absolute difference
        diff = torch.abs(emb1 - emb2)  # [batch, emb_dim]
        
        if self.combination_mode == "difference":
            return diff
        
        # Additional distance metrics
        # Cosine similarity: normalized dot product
        emb1_norm = emb1 / (emb1.norm(dim=-1, keepdim=True) + 1e-8)
        emb2_norm = emb2 / (emb2.norm(dim=-1, keepdim=True) + 1e-8)
        cosine_sim = (emb1_norm * emb2_norm).sum(dim=-1, keepdim=True)  # [batch, 1]
        
        # Euclidean distance (normalized by embedding dimension)
        euclidean_dist = diff.norm(dim=-1, keepdim=True) / math.sqrt(emb1.size(-1))  # [batch, 1]
        
        if self.combination_mode == "distance":
            return torch.cat([diff, cosine_sim, euclidean_dist], dim=-1)
        
        if self.combination_mode == "hybrid":
            # Also include original concatenation for backward compatibility
            concat = torch.cat([emb1, emb2], dim=-1)
            return torch.cat([concat, diff, cosine_sim, euclidean_dist], dim=-1)
        
        raise ValueError(f"Unknown combination_mode: {self.combination_mode}")
    
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
        
        # Combine using similarity-based method
        combined = self._combine_embeddings(emb1, emb2)
        
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
