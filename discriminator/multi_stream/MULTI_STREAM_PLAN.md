# Discriminator Multi-Stream Enhancement Plan

## Overview

This document describes the plan for enhancing the ST-SiameseNet discriminator by implementing two additional data streams from the Ren et al. (KDD 2020) architecture:

1. **Driving Trajectories (LSTM_D)** — passenger-onboard sequences (currently missing)
2. **Profile Features (FCN)** — driver-specific statistical features (currently missing)

The current FAMAIL discriminator only uses **passenger-seeking trajectories** (LSTM_S stream). Adding these streams will bring the architecture into closer alignment with the original ST-SiameseNet and is expected to significantly improve discriminatory power.

### Current Architecture (Single-Stream)

```
Trajectory_A (seeking) ──┐
                         ├─→ Shared LSTM_S Encoder → Embedding_A
Trajectory_B (seeking) ──┤
                         ├─→ Shared LSTM_S Encoder → Embedding_B
                         │
                         └─→ |Emb_A - Emb_B| → FC Classifier → P(same_agent)
```

### Target Architecture (Three-Stream, per Ren et al.)

```
Seeking_A ──→ LSTM_S ──→ Emb_S_A ─┐
Driving_A ──→ LSTM_D ──→ Emb_D_A ─┤
Profile_A ──→ FCN_P  ──→ Emb_P_A ─┤
                                  ├─→ Combine → FC Classifier → P(same_agent)
Seeking_B ──→ LSTM_S ──→ Emb_S_B ─┤
Driving_B ──→ LSTM_D ──→ Emb_D_B ─┤
Profile_B ──→ FCN_P  ──→ Emb_P_B ─┘
```

Six sub-networks total (three stream types × two Siamese branches), where each pair shares weights: LSTM_S shared for seeking, LSTM_D shared for driving, FCN_P shared for profile features. The classifier learns dissimilarity from the combined embeddings of all three streams.

---

## Phase 1: Profile Feature Investigation

### 1.1 Goal

Examine the six candidate pickle files in `cGAIL_data_and_processing/Data/features_condition/` to determine their structure, coverage, and suitability as driver profile features. The output is a Jupyter notebook with findings and a final feature vector specification.

### 1.2 Candidate Files and Hypothesized Mapping to Ren's Features

Ren et al. defined 11 profile features per driver per time period:

| Ren Feature | Description | Candidate File | Notes |
|---|---|---|---|
| f_p,1 & f_p,2 | Coords of longest-staying grid (rest location) | `home_loc_plates_dict_all.pkl` | May contain home/rest GPS coords |
| f_p,3 & f_p,4 | Break start & end time | `start_finishing_time.pkl` | Likely shift start/end times |
| f_p,5 & f_p,6 | Coords of most frequently visited grid | *None — engineer from trajectories* | Computable from trajectory data |
| f_p,7 & f_p,8 | Avg seeking trip distance & time | `ave_monthly_working_time_distance.pkl` | Monthly aggregates |
| f_p,9 & f_p,10 | Avg driving trip distance & time | `trip_info_dict_789.pkl` | Trip-level statistics |
| f_p,11 | Number of trips served | `trip_info_dict_789.pkl` | Count of driving trajectories |
| *(extra)* | Average waiting time between trips | `waiting_dict.pkl` | Seeking efficiency proxy |
| *(reference)* | Train/airport POI locations | `train_airport.pkl` | Already used in state features; not a profile feature but useful context |

### 1.3 Investigation Notebook

**Location**: `discriminator/notebooks/profile_feature_investigation.ipynb`

The notebook should contain the following sections:

#### Section 1: File Structure Inventory
For each of the 6 pickle files:
- Load with `pickle.load()`, print `type()`, `len()`, sample keys
- Determine the key type (plate ID strings vs. numeric indices)
- Display the value structure: `type(value)`, shape/len, sample values
- Check driver coverage: how many of our 50 target drivers appear?
- Build a mapping table: `{plate_id → driver_index}` using `new_all_trajs/step1_processor.py`'s `create_driver_index_mapping()` logic or the known plate IDs from raw data

#### Section 2: Per-File Deep Dive

**`home_loc_plates_dict_all.pkl`**
- Expected: `{plate_id: [lat, lon]}` or `{plate_id: [x_grid, y_grid]}`
- Check coordinate system: raw GPS vs. grid coords
- If GPS: convert to grid coords using project's `gps_to_grid()` function
- Visualize home locations on the 48×90 grid heatmap
- Assess: Is this a proxy for f_p,1/f_p,2 (longest-staying grid)?

**`start_finishing_time.pkl`**
- Expected: `{plate_id: [start_time, finish_time, ...]}` per day or per period
- Determine units: seconds since midnight? time bucket index? datetime?
- Compute per-driver statistics: mean start time, mean end time, variance
- Assess: Maps to f_p,3/f_p,4 (break/shift timing)

**`ave_monthly_working_time_distance.pkl`**
- Expected: `{plate_id: (total_time, total_distance)}` or per-month tuple
- Determine units: hours? minutes? km?
- Check per-month vs. aggregate structure
- Assess: Can derive f_p,7/f_p,8 (avg seeking metrics) or f_p,9/f_p,10 (avg driving metrics)

**`trip_info_dict_789.pkl`**
- Expected: `{plate_id: {month: {...trip stats...}}}` for months 7, 8, 9
- Explore nested structure thoroughly; determine available fields
- Look for: trip count, avg duration, avg distance, distance distribution
- Assess: Maps to f_p,9/f_p,10/f_p,11

**`waiting_dict.pkl`**
- Expected: `{plate_id: {some_key: waiting_stats}}`
- Determine: average inter-trip waiting time per driver
- Assess: This is an additional feature beyond Ren's 11 (seeking efficiency)

**`train_airport.pkl`**
- Already documented: 5 POI locations as `{name: [(x, y), ...]}`
- Verify against existing `cGAIL_data_and_processing/Data/features_condition/train_airport.pkl` usage
- Not a profile feature per se — but could compute "proximity to major transport hubs" as a profile feature (how often a driver operates near airports/stations)

#### Section 3: Coverage and Alignment Analysis
- Cross-reference all file keys against the 50 target driver plate IDs
- Identify missing drivers per file
- Determine whether features are per-day, per-month, or per-study-period
- Decide on temporal granularity for profile features (Ren used T = 1 day)

#### Section 4: Engineerable Features
Features that can be computed from the raw trajectory data (`raw_data/taxi_record_0*_50drivers.pkl`) if the pickle files are insufficient:

| Feature | Derivation | Ren Equivalent |
|---|---|---|
| Most frequently visited grid (x, y) | Mode of grid cell visits per driver-day | f_p,5 & f_p,6 |
| Number of trips served per day | Count of 0→1 transitions per driver-day | f_p,11 |
| Average seeking trip distance | Sum of Euclidean cell-to-cell distances in seeking trajectories | f_p,7 |
| Average seeking trip time | (last_timestamp - first_timestamp) per seeking trajectory | f_p,8 |
| Average driving trip distance | Same logic for driving trajectories | f_p,9 |
| Average driving trip time | Same logic for driving trajectories | f_p,10 |
| Spatial entropy | Entropy of grid cell visit distribution per driver | *New: measures spatial spread* |
| Action distribution | Histogram of movement directions (8 compass + stay) | *New: captures direction preferences* |
| Speed percentiles | p25, p50, p75 of speed computed from consecutive GPS points | *New: driving style indicator* |

These features are straightforward to compute from the raw GPS records during the trajectory extraction phase (Phase 2). They should be considered as supplements or replacements if the existing pickle files prove insufficient.

#### Section 5: Final Feature Vector Specification
- Define the final profile feature vector: dimensions, ordering, normalization strategy
- Document which features come from existing pickle files vs. engineered
- Specify normalization: min-max to [0,1] per feature across all drivers, or z-score
- Target: a fixed-length vector per driver per time period (e.g., per day or per month)

**Ren's profile-learner architecture**: FCN with hidden units [64, 32, 8] and ReLU activation. The input dimension equals the number of profile features (Ren used 11). We should target a similar dimensionality (11-15 features) to keep the profile-learner architecture unchanged.

### 1.4 Decision Point

After completing the notebook, we will decide:
1. Which features to include in the profile vector
2. What temporal granularity to use (per-day vs. per-month)
3. Whether to engineer additional features from raw data
4. How to handle missing data (imputation vs. exclusion)

---

## Phase 2: Trajectory Extraction Tool (Seeking + Driving)

### 2.1 Goal

Build a new trajectory extraction tool that produces **both** passenger-seeking and driving (passenger-onboard) trajectories from the raw GPS data. This replaces the existing `new_all_trajs/` tool, which:
- Only extracts seeking trajectories
- Has known issues with duplicated state connections in visualizations
- Also generates 126-element state features (not needed at this stage)

### 2.2 Design: `discriminator/trajectory_extraction/`

A new standalone module (not modifying `new_all_trajs/`, which remains for backward compatibility). The module should be clean, tested, and produce both trajectory types.

#### Directory Structure

```
discriminator/trajectory_extraction/
├── __init__.py
├── extractor.py          # Core extraction logic
├── config.py             # Configuration dataclass
├── app.py                # Streamlit dashboard
├── profile_features.py   # Profile feature computation (Phase 2.5)
└── requirements.txt
```

#### 2.3 Extraction Logic (`extractor.py`)

**Input**: Raw GPS pickle files (`raw_data/taxi_record_0*_50drivers.pkl`)

**Raw record format**: `[plate_id, latitude, longitude, seconds_since_midnight, passenger_indicator, timestamp_str]`

**Output**: Two trajectory dictionaries and (optionally) profile features:

```python
{
    "seeking": {
        driver_index: [
            [[x, y, t, day], [x, y, t, day], ...],  # trajectory 1
            ...
        ],
        ...
    },
    "driving": {
        driver_index: [
            [[x, y, t, day], [x, y, t, day], ...],  # trajectory 1
            ...
        ],
        ...
    }
}
```

#### Seeking Trajectory Extraction (same as current, with fixes)
- **Start**: `passenger_indicator` transitions 1→0 (dropoff event)
- **End**: `passenger_indicator` transitions 0→1 (pickup event)
- **Contains**: Only states where `passenger_indicator == 0`
- **Fix**: Deduplicate consecutive identical states `(x, y, t)` to eliminate the duplicated-connection visualization issue. The current tool appends every GPS record as a state even when the quantized grid cell hasn't changed, producing repeated edges in trajectory visualizations. The new tool should collapse consecutive records that map to the same `(x_grid, y_grid, time_bucket)` into a single state.

#### Driving Trajectory Extraction (new)
- **Start**: `passenger_indicator` transitions 0→1 (pickup event)
- **End**: `passenger_indicator` transitions 1→0 (dropoff event)
- **Contains**: Only states where `passenger_indicator == 1`
- **Same quantization**: `gps_to_grid()` with 0.01° cells, 5-min time buckets
- **Same deduplication**: Collapse consecutive identical quantized states

#### Quantization Parameters (must match project conventions)
- Grid size: 0.01° (consistent with `pickup_dropoff_counts/processor.py`)
- Time interval: 5 minutes (288 buckets/day)
- Grid bounds: computed globally from all three months' data
- Coordinate offsets: +1 for both x_grid and y_grid (1-indexed output for compatibility with `data_loader.py` which subtracts 1)
- Time offset: 0 (0-indexed time buckets, 0-287)
- Weekend exclusion: True (raw data is weekday-only, but safety filter)

#### Length Filtering
- Minimum trajectory length: configurable (default: 10 states)
- Maximum trajectory length: configurable (default: 1000 states)
- Note: Driving trajectories are typically shorter than seeking trajectories (Ren reports ~1:1 ratio of driving:seeking trips per day, with seeking averaging ~14 km). We should analyze and report the length distributions for both types.

#### Implementation Considerations
- **Sort records by timestamp** before processing each driver. The current tool assumes records are chronologically ordered, but this should be enforced explicitly to prevent transition detection errors.
- **Handle day boundaries**: A trajectory that starts before midnight and continues after midnight should be split into two trajectories (different days) to maintain the per-day temporal granularity.
- **GPS quality filter**: Skip records where lat/lon are clearly outside the Shenzhen bounding box (lat 22.44-22.87, lon 113.75-114.65).

### 2.4 Streamlit Dashboard (`app.py`)

A minimal dashboard for:
1. Selecting raw data files (multi-select for July, August, September)
2. Configuring extraction parameters (min/max length, grid size)
3. Running extraction with progress bar
4. Displaying summary statistics:
   - Per-driver trajectory counts (seeking and driving)
   - Length distribution histograms for both trajectory types
   - Spatial coverage heatmaps
   - Sample trajectory visualizations
5. Saving output to a configurable path

### 2.5 Profile Feature Computation (`profile_features.py`)

After trajectory extraction, compute the profile features per driver per time period. This module runs as a second pass over the extracted trajectories and/or raw GPS data.

**Computed features** (targeting 11-15 dimensions, matching Ren's approach):

```python
def compute_profile_features(
    seeking_trajs: Dict[int, List[List[List[int]]]],
    driving_trajs: Dict[int, List[List[List[int]]]],
    raw_data: Dict[str, List],  # Optional, for GPS-level features
    time_period: str = "monthly",  # "daily" | "monthly" | "full"
) -> Dict[int, np.ndarray]:
    """
    Returns {driver_index: feature_vector} where feature_vector is
    a fixed-length numpy array of profile features.
    """
```

Features to compute (superset — final selection based on Phase 1 findings):

| Index | Feature | Source | Normalization |
|---|---|---|---|
| 0, 1 | Home/rest location (x_grid, y_grid) | `home_loc_plates_dict_all.pkl` or longest-stay analysis | Min-max [0,1] |
| 2, 3 | Shift start & end time (time_bucket) | `start_finishing_time.pkl` or raw GPS timestamps | Min-max [0,1] |
| 4, 5 | Most frequently visited grid (x_grid, y_grid) | Mode of grid cell visits from trajectories | Min-max [0,1] |
| 6 | Avg seeking trip duration (time_buckets) | Computed from seeking trajectory lengths | Min-max [0,1] |
| 7 | Avg seeking trip distance (grid cells) | Sum of Euclidean hops in seeking trajectories | Min-max [0,1] |
| 8 | Avg driving trip duration (time_buckets) | Computed from driving trajectory lengths | Min-max [0,1] |
| 9 | Avg driving trip distance (grid cells) | Sum of Euclidean hops in driving trajectories | Min-max [0,1] |
| 10 | Number of trips served (count) | Count of driving trajectories per period | Min-max [0,1] |

Optional additional features (if they improve discrimination — to be determined experimentally):
- Average inter-trip waiting time
- Spatial entropy of grid cell visits
- Speed percentiles (requires raw GPS data with precise timestamps)

**Normalization**: All features normalized to [0, 1] using min-max normalization computed across all 50 drivers. Normalization parameters saved alongside the features for inference-time use.

**Output format**:
```python
# Saved as pickle:
{
    "features": {driver_index: np.ndarray([f0, f1, ..., f10])},  # per-driver
    "feature_names": ["home_x", "home_y", "shift_start", ...],
    "normalization": {"min": np.ndarray, "max": np.ndarray},
    "time_period": "monthly",
    "n_features": 11
}
```

---

## Phase 3: Model Architecture Enhancement

### 3.1 Goal

Extend `SiameseLSTMDiscriminatorV2` to support three input streams. The new model class should be backward-compatible — it should work with seeking-only data (single stream) by default, and optionally accept driving trajectories and profile features.

### 3.2 New Model Class: `MultiStreamSiameseDiscriminator`

**Location**: `discriminator/model/model.py` (add as a new class, preserve existing V1/V2)

```
class MultiStreamSiameseDiscriminator(nn.Module):
    """Three-stream ST-SiameseNet following Ren et al. (KDD 2020).

    Streams:
        1. LSTM_S: Shared-weight LSTM for seeking trajectories
        2. LSTM_D: Shared-weight LSTM for driving trajectories
        3. FCN_P: Shared-weight FCN for profile features

    The three stream embeddings are combined and passed through
    a dissimilarity-learner (FC classifier) to produce P(same_agent).
    """
```

#### Architecture Details

**Stream 1 — LSTM_S (Seeking Encoder)**
- Input: `[batch, seq_len, 4]` → FeatureNormalizer → `[batch, seq_len, 6]`
- Architecture: Same as current `SiameseLSTMEncoder` (200, 100 hidden dims, bidirectional)
- Output: `[batch, emb_dim_s]` where `emb_dim_s = 200` (100 × 2 for bidirectional)

**Stream 2 — LSTM_D (Driving Encoder)**
- Input: `[batch, seq_len, 4]` → FeatureNormalizer (shared with LSTM_S) → `[batch, seq_len, 6]`
- Architecture: Separate LSTM encoder, same architecture as LSTM_S but **independent weights** (per Ren: "each two identical sub-networks share the set of weights" within a stream, but LSTM_S and LSTM_D have different weights)
- Output: `[batch, emb_dim_d]` where `emb_dim_d = 200`
- Note: Ren states LSTM_D has "the same components of neurons as LSTM_S" — same architecture, different weights

**Stream 3 — FCN_P (Profile Feature Learner)**
- Input: `[batch, n_profile_features]` (e.g., `[batch, 11]`)
- Architecture: FC layers [64, 32, 8] with ReLU activation (per Ren Appendix A.3)
- Output: `[batch, emb_dim_p]` where `emb_dim_p = 8`
- Shared weights between both branches of the Siamese pair

**Dissimilarity Learner (Classifier)**
Following the V2 distance-based approach (not V1 concatenation), compute per-stream differences and combine:

```python
# Per-stream differences
diff_s = |emb_s_A - emb_s_B|    # [batch, 200]
diff_d = |emb_d_A - emb_d_B|    # [batch, 200]
diff_p = |emb_p_A - emb_p_B|    # [batch, 8]

# Concatenate all differences
combined = cat([diff_s, diff_d, diff_p])  # [batch, 408]

# FC classifier
classifier: 408 → 64 → 32 → 8 → 1 (sigmoid)
```

This preserves the V2 distance-based insight (identical inputs → zero difference → high similarity) while extending it to three streams.

#### Graceful Degradation

The model should handle missing streams gracefully for backward compatibility and for cases where a driver has seeking but no driving trajectories for a given period:

```python
def forward(self,
            seeking_1, seeking_2,           # Required
            mask_s1, mask_s2,               # Required
            driving_1=None, driving_2=None, # Optional
            mask_d1=None, mask_d2=None,     # Optional
            profile_1=None, profile_2=None  # Optional
           ) -> torch.Tensor:
```

**When driving trajectories are missing**: Use a learned `driving_default_embedding` parameter (zero-initialized) as the embedding. The difference of two default embeddings is zero, so this stream contributes no signal — effectively reducing to a two-stream model.

**When profile features are missing**: Same approach — use a learned `profile_default_embedding`.

This design means:
- Training with all three streams: full discriminatory power
- Inference with seeking-only: degrades gracefully to V2-like behavior
- Inference with seeking + profile: two-stream model

### 3.3 Model Configuration

Extend the existing config dict to include multi-stream parameters:

```python
config = {
    "model_version": "v3",           # New version identifier
    "lstm_hidden_dims": (200, 100),  # Shared architecture for both LSTM_S and LSTM_D
    "dropout": 0.2,
    "bidirectional": True,
    "classifier_hidden_dims": (64, 32, 8),
    "combination_mode": "difference",  # For all streams
    "n_profile_features": 11,         # Profile feature vector dimension
    "profile_hidden_dims": (64, 32, 8),  # FCN_P architecture
    "streams": ["seeking", "driving", "profile"],  # Active streams
}
```

### 3.4 Loss Function

Unchanged: Binary cross-entropy loss, same as V2. The only difference is that the classifier receives richer input from three streams instead of one.

```
min_θ −(y·log(D_θ(X1,X2)) + (1−y)·log(1−D_θ(X1,X2)))

where X1 = (seeking_1, driving_1, profile_1)
      X2 = (seeking_2, driving_2, profile_2)
```

This matches Ren et al. Equation 3 exactly.

---

## Phase 4: Dataset Generation Enhancement

### 4.1 Goal

Extend the `discriminator/dataset_generation_tool/` to produce multi-stream training datasets that include seeking trajectories, driving trajectories, and profile features.

### 4.2 New Dataset Format

**Output**: `.npz` files with the following arrays:

```python
{
    # Seeking trajectories (existing)
    'seeking_1': np.ndarray,     # [N, L_s, 4]
    'seeking_2': np.ndarray,     # [N, L_s, 4]
    'mask_s1': np.ndarray,       # [N, L_s]
    'mask_s2': np.ndarray,       # [N, L_s]

    # Driving trajectories (new)
    'driving_1': np.ndarray,     # [N, L_d, 4]
    'driving_2': np.ndarray,     # [N, L_d, 4]
    'mask_d1': np.ndarray,       # [N, L_d]
    'mask_d2': np.ndarray,       # [N, L_d]

    # Profile features (new)
    'profile_1': np.ndarray,     # [N, F_p]
    'profile_2': np.ndarray,     # [N, F_p]

    # Labels
    'label': np.ndarray,         # [N]

    # Backward compatibility aliases
    'x1': np.ndarray,            # Same as seeking_1 (for V1/V2 model loading)
    'x2': np.ndarray,            # Same as seeking_2
    'mask1': np.ndarray,         # Same as mask_s1
    'mask2': np.ndarray,         # Same as mask_s2
}
```

Note: `L_s` and `L_d` may differ because seeking and driving trajectories have different length distributions. Each is padded independently.

### 4.3 Pair Sampling Strategy

The key constraint is that all three streams for a given pair must come from the **same time period** for both branches:

**For a positive pair (same agent, label=1)**:
1. Select agent A, time periods T1 and T2
2. Sample a seeking trajectory from A in T1, and one from A in T2
3. Sample a driving trajectory from A in T1, and one from A in T2
4. Extract profile features for A in T1 and T2

**For a negative pair (different agents, label=0)**:
1. Select agents A and B, time periods T1 and T2
2. Sample seeking from A in T1, seeking from B in T2
3. Sample driving from A in T1, driving from B in T2
4. Extract profile features for A in T1 and B in T2

**Time period definition**: Following Ren, T = 1 day for profile features. This means profile features are computed per driver per day. If per-day granularity proves too noisy (only ~20 trips per driver per day), we can fall back to per-month or per-study-period.

### 4.4 Handling Missing Data

Some drivers may have seeking trajectories on a given day but no driving trajectories (rare but possible). Options:
1. **Skip the day**: Only include days where both trajectory types exist
2. **Fill with zeros**: Use zero-padded dummy trajectories with all-zero masks
3. **Random substitute**: Use a driving trajectory from a nearby day

Recommendation: Option 1 (skip) for clean training data. Option 2 for inference when the model must produce a score regardless.

### 4.5 Input Data Sources

The generation tool needs access to:
1. **Seeking trajectories**: Output from Phase 2 extraction tool
2. **Driving trajectories**: Output from Phase 2 extraction tool
3. **Profile features**: Output from Phase 2.5 profile computation

All three should be saved as a single bundle or in a known directory structure:

```
discriminator/extracted_data/
├── seeking_trajs.pkl           # {driver_index: [trajectories]}
├── driving_trajs.pkl           # {driver_index: [trajectories]}
├── profile_features.pkl        # {driver_index: feature_vector} per period
└── extraction_metadata.json    # Bounds, config, driver mapping
```

### 4.6 GenerationConfig Extension

```python
@dataclass
class MultiStreamGenerationConfig(GenerationConfig):
    # New fields
    seeking_data_path: Path         # Path to seeking_trajs.pkl
    driving_data_path: Path         # Path to driving_trajs.pkl
    profile_data_path: Path         # Path to profile_features.pkl
    include_driving: bool = True    # Include driving stream
    include_profile: bool = True    # Include profile stream
    driving_padding: str = "pad_to_longer"   # Independent padding for driving
    profile_time_period: str = "monthly"     # "daily" | "monthly" | "full"
```

---

## Phase 5: Training Pipeline Updates

### 5.1 Dataset Class

**New file or extension**: `discriminator/model/dataset.py`

Add a `MultiStreamTrajectoryPairDataset` that loads the enhanced `.npz` files:

```python
class MultiStreamTrajectoryPairDataset(Dataset):
    def __getitem__(self, idx):
        return {
            'seeking_1': self.seeking_1[idx],
            'seeking_2': self.seeking_2[idx],
            'mask_s1': self.mask_s1[idx],
            'mask_s2': self.mask_s2[idx],
            'driving_1': self.driving_1[idx],
            'driving_2': self.driving_2[idx],
            'mask_d1': self.mask_d1[idx],
            'mask_d2': self.mask_d2[idx],
            'profile_1': self.profile_1[idx],
            'profile_2': self.profile_2[idx],
            'label': self.label[idx],
        }
```

Backward-compatible: if `driving_1` key is absent in the `.npz` file, return None for those fields.

### 5.2 Trainer Updates

**File**: `discriminator/model/trainer.py`

Changes needed:
1. **Model instantiation**: Support `model_version="v3"` to create `MultiStreamSiameseDiscriminator`
2. **Training loop**: Unpack multi-stream batch and pass all six inputs to `model.forward()`
3. **Validation metrics**: Same metrics (accuracy, F1, AUC, split accuracy, identical trajectory validation), but also report per-stream ablation scores (seeking-only, driving-only, combined) to understand each stream's contribution
4. **Checkpoint format**: Store `n_profile_features` and `profile_hidden_dims` in config

### 5.3 CLI Updates

**File**: `discriminator/model/train.py`

Add new command-line arguments:

```bash
python train.py \
    --model-version v3 \
    --n-profile-features 11 \
    --profile-hidden-dims 64,32,8 \
    --streams seeking,driving,profile \
    --data-dir discriminator/datasets/multi_stream/
```

### 5.4 Training Strategy

**Hyperparameters** (starting point, based on Ren's Appendix A.3):

| Parameter | Value | Source |
|---|---|---|
| Optimizer | Adam (β1=0.9, β2=0.999) | Ren A.3 |
| Learning rate | 0.00006 | Ren A.3 (note: much lower than our current 0.001) |
| Batch size | 1 (Ren) or 32 (ours) | Ren used 1 due to variable-length; we can use 32 with padding |
| LSTM_S hidden dims | (200, 100) | Ren A.3 |
| LSTM_D hidden dims | (200, 100) | Ren A.3 ("same components as LSTM_S") |
| Profile FCN dims | (64, 32, 8) | Ren A.3 |
| Classifier dims | (64, 32, 8, 1) | Ren A.3 |
| Iterations | 1,000,000 | Ren A.3 (we use epochs instead) |

**Training schedule recommendation**:
1. First, train seeking-only (V2) as a baseline to compare against
2. Then train with all three streams (V3) using the same dataset split
3. Compare F1 scores, per-stream ablation, and identical trajectory validation
4. Tune learning rate (Ren's 6e-5 vs our 1e-3) — the additional model capacity may benefit from a lower rate

---

## Phase 6: Fidelity Term Integration

### 6.1 Impact on Objective Function

The fidelity term (`objective_function/fidelity/`) uses the discriminator to score trajectory modifications. It currently compares edited seeking trajectories against originals. With the multi-stream model:

**What changes**:
- The discriminator now expects three inputs per branch (seeking, driving, profile)
- During trajectory modification, only **seeking trajectories are modified** (the pickup relocation targets seeking behavior)
- Driving trajectories and profile features remain **unchanged** (they serve as additional context about the driver's identity)

**Implication**: When computing fidelity for an edited trajectory:
- `seeking_1` = edited seeking trajectory
- `seeking_2` = original seeking trajectory (from same driver)
- `driving_1` = unmodified driving trajectory (from same driver, same period)
- `driving_2` = same or different driving trajectory (from same driver)
- `profile_1` = profile features (from same driver, same period)
- `profile_2` = same profile features (identical for same driver/period)

Since driving and profile are identical between the two branches, their difference contributions are zero. The discriminator score is therefore driven primarily by the seeking trajectory difference, but the LSTM_S encoder has been trained in the context of all three streams, so its learned representations may be richer.

### 6.2 Files to Update

1. **`objective_function/fidelity/config.py`**: Add fields for driving trajectory and profile feature sources
2. **`objective_function/fidelity/utils.py`**: Update `load_discriminator()` to handle V3 models; update `prepare_trajectory_batch()` to prepare multi-stream inputs
3. **`objective_function/fidelity/term.py`**: Update `compute()` and `compute_with_breakdown()` to pass multi-stream inputs
4. **`objective_function/fidelity/term.py`**: Update `DifferentiableFidelity` for gradient computation through multi-stream model

### 6.3 Gradient Computation Consideration

The ST-iFGSM algorithm computes gradients of the fidelity loss w.r.t. the edited trajectory positions. With the multi-stream model, the gradient computation only flows through the seeking stream (since driving and profile are held constant). This means:

```
∂F_fidelity/∂seeking_positions = ∂D_θ(seeking_edited, seeking_orig, ...)/∂seeking_edited
```

The gradient path is: seeking trajectory → FeatureNormalizer → LSTM_S → embedding → difference → classifier → loss. This is the same computational graph as V2, just with a richer classifier input. No fundamental changes to the gradient computation are needed — the driving and profile streams are simply detached from the gradient.

---

## Phase 7: Validation and Testing

### 7.1 Ablation Study

Train and evaluate the following configurations to measure each stream's contribution:

| Config | Seeking | Driving | Profile | Expected Improvement |
|---|---|---|---|---|
| Baseline (V2) | ✓ | ✗ | ✗ | Current performance |
| +Driving | ✓ | ✓ | ✗ | Moderate (Ren showed seeking > driving for discrimination) |
| +Profile | ✓ | ✗ | ✓ | Moderate (profile features alone got ~0.79 F1 in Ren's SVM) |
| Full (V3) | ✓ | ✓ | ✓ | Best (Ren's full model: 0.85 F1) |

### 7.2 Metrics

Same as current training metrics, plus:
- Per-stream contribution analysis (freeze streams and measure degradation)
- Profile feature importance ranking (permutation importance)
- Comparison with Ren's reported results (F1 = 0.8508)

### 7.3 Visualization

Update the training dashboard (`discriminator/model/training_dashboard.py`) to display:
- Per-stream embedding t-SNE visualizations
- Profile feature distribution comparisons between same/different agent pairs
- Multi-stream vs. single-stream performance curves

---

## Implementation Order and Dependencies

```
Phase 1: Profile Feature Investigation (Notebook)
    │   No code dependencies. Can start immediately.
    │   Output: Feature vector specification document.
    │
Phase 2: Trajectory Extraction Tool
    │   Depends on: Raw data files (already available)
    │   Output: seeking_trajs.pkl, driving_trajs.pkl
    │
    ├── Phase 2.5: Profile Feature Computation
    │   Depends on: Phase 1 (feature spec), Phase 2 (extracted trajectories)
    │   Output: profile_features.pkl
    │
Phase 3: Model Architecture (MultiStreamSiameseDiscriminator)
    │   Depends on: Phase 1 (n_profile_features dimension)
    │   Can be developed in parallel with Phase 2.
    │
Phase 4: Dataset Generation Enhancement
    │   Depends on: Phase 2 + 2.5 (data), Phase 3 (model spec)
    │
Phase 5: Training Pipeline Updates
    │   Depends on: Phase 3 (model), Phase 4 (datasets)
    │
Phase 6: Fidelity Term Integration
    │   Depends on: Phase 5 (trained model)
    │
Phase 7: Validation and Testing
        Depends on: Phase 6 (integrated system)
```

**Parallelizable work**:
- Phase 1 and Phase 2 can proceed concurrently
- Phase 3 can begin once Phase 1 produces a feature dimension count
- Phase 4 depends on both Phase 2/2.5 and Phase 3

**Estimated scope**:
- Phase 1: Investigation notebook (1 session)
- Phase 2: Extraction tool (~500 lines of code)
- Phase 2.5: Profile computation (~200 lines)
- Phase 3: Model architecture (~300 lines, additive to existing model.py)
- Phase 4: Dataset generation updates (~200 lines of modifications)
- Phase 5: Training pipeline updates (~150 lines of modifications)
- Phase 6: Fidelity term updates (~100 lines of modifications)
- Phase 7: Validation runs and analysis

---

## Risk Assessment

### Technical Risks

1. **Profile feature pickle files may be unusable**
   - *Mitigation*: All 11 Ren features can be engineered from raw GPS data if needed
   - *Impact*: Adds engineering effort to Phase 2.5

2. **Driving trajectories may be too short for LSTM encoding**
   - *Mitigation*: Ren reported ~1:1 seeking:driving ratio with adequate lengths. If too short, increase grid resolution or reduce minimum length threshold
   - *Impact*: May need architecture adjustment (shorter LSTM or attention mechanism)

3. **Three-stream model may overfit with only 50 drivers**
   - *Mitigation*: Ren trained with 500 agents; with 50 we have much less diversity. Use strong regularization (dropout, weight decay), data augmentation (temporal shifts), and careful cross-validation
   - *Impact*: May not reach Ren's 0.85 F1, but should still improve over single-stream

4. **Backward compatibility breakage**
   - *Mitigation*: V3 model is a new class; V1/V2 remain untouched. Dataset format includes backward-compatible aliases. Fidelity term auto-detects model version.
   - *Impact*: Low — existing checkpoints and datasets continue to work

### Data Risks

5. **Plate ID mismatch between feature pickles and raw data**
   - *Mitigation*: Phase 1 notebook explicitly checks coverage
   - *Impact*: May need to build a plate ID mapping table

6. **Temporal alignment between trajectory types**
   - *Mitigation*: Extract both trajectory types from the same raw GPS records in a single pass
   - *Impact*: Low if extraction is done correctly

---

## Open Questions

1. **Profile feature temporal granularity**: Should we compute per-day (more data points, noisier) or per-month (fewer points, more stable)? Ren used T = 1 day. Our 50-driver dataset may benefit from per-month to reduce noise.

2. **Learning rate**: Ren used 6e-5, our current pipeline uses 1e-3. Should we start with Ren's rate for the multi-stream model, or tune independently?

3. **Driving trajectory usage in fidelity**: Since the modification algorithm only edits seeking trajectories, driving trajectories provide identity context but don't receive gradient updates. Should we consider also modifying driving trajectories in future work?

4. **Online features**: Ren also extracts speed as an "online feature" appended to each trajectory timestep (making each step `(grid, time, speed)` instead of `(grid, time)`). Should we add speed computation to the trajectory extraction? This would require computing speed from consecutive GPS positions, which is straightforward from the raw data.
