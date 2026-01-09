# Modified ST-SiameseNet Discriminator

A Siamese LSTM-based discriminator model for determining whether two trajectory sequences belong to the same agent (driver).

## Architecture

The model uses a shared-weight Siamese architecture:

```
Trajectory A ─→ ┌──────────────┐      ┌────────────┐
                │   Feature    │ ─→   │   LSTM     │ ─→ Embedding A
                │ Normalizer   │      │  Encoder   │         │
Trajectory B ─→ │  (shared)    │ ─→   │ (shared)   │ ─→ Embedding B ─→ Concatenate ─→ Classifier ─→ P(same agent)
                └──────────────┘      └────────────┘
```

### Feature Normalization

Raw input features (4):
- `x_grid`: Grid x-coordinate (0-49)
- `y_grid`: Grid y-coordinate (0-89)
- `time_bucket`: Time of day (0-287, 5-minute intervals)
- `day_index`: Day of week (0-6)

Normalized features (6):
- `x_norm = x_grid / 49` (min-max to [0, 1])
- `y_norm = y_grid / 89` (min-max to [0, 1])
- `sin_time = sin(2π × time_bucket / 288)` (cyclic encoding)
- `cos_time = cos(2π × time_bucket / 288)`
- `sin_day = sin(2π × day_index / 7)` (cyclic encoding)
- `cos_day = cos(2π × day_index / 7)`

### LSTM Encoder

- Bidirectional LSTM (default)
- 2 layers with 128 hidden units
- Processes variable-length sequences with masking
- Final hidden state as embedding

### Classifier

- Concatenates both trajectory embeddings
- MLP: FC(256 → 128) → ReLU → Dropout → FC(128 → 64) → ReLU → Dropout → FC(64 → 1)
- Sigmoid activation for binary output

### Output

- **1** = same agent (positive pair)
- **0** = different agents (negative pair)

## Installation

```bash
cd /home/robert/FAMAIL/discriminator/model
pip install -r requirements.txt
```

## Usage

### 1. Generate Dataset

First, use the Dataset Generation Tool to create training data:

```bash
# Launch the Streamlit UI
streamlit run ../dataset_generation_tool/app.py
```

Configure your dataset and click "Save Dataset" to save to a directory.

### 2. Train via Command Line

```bash
# Basic training
python train.py --data-dir /path/to/dataset

# With custom hyperparameters
python train.py \
    --data-dir ./datasets/my_dataset \
    --hidden-dim 256 \
    --num-layers 3 \
    --epochs 150 \
    --lr 0.0005 \
    --batch-size 64 \
    --early-stopping 15

# From a single file (auto-split)
python train.py --data dataset.npz --val-split 0.2
```

### 3. Train via Streamlit Dashboard

```bash
streamlit run training_dashboard.py
```

The dashboard provides:
- Dataset selection and exploration
- Interactive hyperparameter configuration
- Training progress visualization
- Model evaluation

### 4. Load a Trained Model

```python
from model import load_model_from_checkpoint

model, checkpoint = load_model_from_checkpoint("checkpoints/experiment_name/best.pt")

# Make predictions
x1 = torch.tensor(...)  # [batch, seq_len, 4]
x2 = torch.tensor(...)  # [batch, seq_len, 4]
mask1 = torch.tensor(...)  # [batch, seq_len]
mask2 = torch.tensor(...)  # [batch, seq_len]

probs = model(x1, x2, mask1, mask2)  # [batch, 1] in [0, 1]
preds = model.predict(x1, x2, mask1, mask2)  # Binary: 1=same, 0=different
```

## File Structure

```
discriminator/model/
├── __init__.py              # Package exports
├── model.py                 # Model architectures
├── dataset.py               # Dataset loading utilities
├── trainer.py               # Training loop and utilities
├── train.py                 # CLI training script
├── training_dashboard.py    # Streamlit dashboard
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 128 | LSTM hidden state dimension |
| `num_layers` | 2 | Number of LSTM layers |
| `dropout` | 0.2 | Dropout probability |
| `bidirectional` | True | Use bidirectional LSTM |
| `classifier_hidden_dims` | (128, 64) | Classifier MLP hidden sizes |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 1e-3 | Initial learning rate |
| `weight_decay` | 1e-4 | L2 regularization |
| `epochs` | 100 | Maximum training epochs |
| `early_stopping_patience` | 10 | Epochs without improvement before stopping |
| `scheduler` | "plateau" | LR scheduler: "plateau", "cosine", or "none" |

## Dataset Format

Expected `.npz` file structure:
- `x1`: First trajectories `[N, L, 4]`
- `x2`: Second trajectories `[N, L, 4]`
- `label`: Binary labels `[N]` where 1=same agent, 0=different
- `mask1`: Validity mask for x1 `[N, L]`
- `mask2`: Validity mask for x2 `[N, L]`

## Metrics

During training, the following metrics are tracked:
- **Loss**: Binary Cross-Entropy
- **Accuracy**: Classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve

## Checkpoints

Checkpoints are saved to `checkpoints/<experiment_name>/`:
- `best.pt`: Model with best validation loss
- `latest.pt`: Most recent checkpoint
- `config.json`: Training configuration
- `history.json`: Training metrics history

## Example Workflow

```bash
# 1. Generate dataset using Streamlit UI
streamlit run ../dataset_generation_tool/app.py
# Configure: per-agent mode, 100 pos/agent, 10 neg/combo
# Save to: ./datasets/full_coverage

# 2. Train model
python train.py \
    --data-dir ./datasets/full_coverage \
    --hidden-dim 128 \
    --num-layers 2 \
    --epochs 100 \
    --early-stopping 15 \
    --experiment-name full_coverage_v1

# 3. View results in dashboard
streamlit run training_dashboard.py
# Navigate to "View Results" and select your experiment
```

## Experimental: Transformer Architecture

An alternative transformer-based encoder is also implemented (`SiameseTransformerDiscriminator`). This may perform better on longer sequences but requires more computation.

```python
from model import SiameseTransformerDiscriminator

model = SiameseTransformerDiscriminator(
    d_model=128,
    nhead=4,
    num_layers=2,
    dropout=0.2
)
```

## License

FaMAIL Project - San Diego State University - 2026.
