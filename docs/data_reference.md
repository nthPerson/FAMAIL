# FAMAIL: Data Reference

**Datasets and data structures used by the FAMAIL trajectory modification framework.**

---

## 1. Overview

The trajectory modification framework operates on three primary datasets, all stored as Python pickle files in the `source_data/` directory:

| Dataset | File | Description |
|---------|------|-------------|
| Passenger-seeking trajectories | `passenger_seeking_trajs_45-800.pkl` | Taxi trajectories from passenger-seeking to pickup |
| Pickup/dropoff counts | `pickup_dropoff_counts.pkl` | Historical pickup and dropoff frequencies per spatiotemporal state |
| Active taxis | `active_taxis_5x5_hourly.pkl` | Number of unique taxis present in each cell neighborhood per hour |

All datasets cover the same study area: a $48 \times 90$ spatial grid over Shenzhen, China, derived from GPS data collected from 50 expert taxi drivers.

---

## 2. Spatial Grid

### 2.1 Dimensions

- **X-axis (latitude)**: 48 cells, indexed $[0, 47]$ (0-indexed in code) or $[1, 48]$ (1-indexed in raw data files)
- **Y-axis (longitude)**: 90 cells, indexed $[0, 89]$ (0-indexed in code) or $[1, 90]$ (1-indexed in raw data files)
- **Total cells**: $48 \times 90 = 4{,}320$

### 2.2 Coordinate Indexing

> **Important**: Different datasets use different indexing conventions. The trajectory modification code normalizes all coordinates to **0-indexed** internally.

| Dataset | X range | Y range | Indexing |
|---------|---------|---------|----------|
| Passenger-seeking trajectories | $[0, 47]$ | $[0, 89]$ | 0-indexed |
| Pickup/dropoff counts | $[0, 47]$ | $[0, 89]$ | 0-indexed |
| Active taxis | $[1, 48]$ | $[1, 90]$ | 1-indexed |

The data loading code converts 1-indexed active taxi coordinates to 0-indexed by subtracting 1 from each spatial coordinate.

---

## 3. Dataset 1: Passenger-Seeking Trajectories

### 3.1 File

`source_data/passenger_seeking_trajs_45-800.pkl`

### 3.2 Structure

```python
{
    driver_id: [                    # Dict keyed by driver ID (int)
        [                           # List of trajectories per driver
            [x, y, time_bucket, day_index],  # State 0 (start of seeking)
            [x, y, time_bucket, day_index],  # State 1
            ...
            [x, y, time_bucket, day_index],  # State N (pickup location)
        ],
        ...                         # More trajectories for this driver
    ],
    ...                             # More drivers
}
```

### 3.3 State Vector

Each state is a 4-element list:

| Index | Field | Type | Description | Range |
|-------|-------|------|-------------|-------|
| 0 | `x_grid` | `int` | Grid x-coordinate (latitude) | $[0, 47]$ |
| 1 | `y_grid` | `int` | Grid y-coordinate (longitude) | $[0, 89]$ |
| 2 | `time_bucket` | `int` | Discretized time-of-day (5-min intervals) | $[0, 287]$ (0-indexed) |
| 3 | `day_index` | `int` | Day-of-week indicator | Day identifier |

### 3.4 Trajectory Semantics

Each trajectory represents a taxi's path from the beginning of **passenger-seeking** behavior to the **pickup** event:

- `states[0]`: Where the driver started seeking a passenger
- `states[1:-1]`: Intermediate movement during seeking
- `states[-1]`: **Pickup location** — this is the state that the modification algorithm edits

### 3.5 Relationship to all_trajs.pkl

The `passenger_seeking_trajs_45-800.pkl` dataset has the same hierarchical structure as the `all_trajs.pkl` dataset, with one key difference:

- **`all_trajs.pkl`**: 126-element state vectors (spatial, temporal, POI distances, traffic features, action codes)
- **`passenger_seeking_trajs_45-800.pkl`**: 4-element state vectors (only the first 4 elements: spatial and temporal)

Only the first 4 state elements are used by the trajectory modification framework.

### 3.6 Loading

```python
from trajectory_modification import TrajectoryLoader

loader = TrajectoryLoader()
trajectories = loader.load_passenger_seeking(max_trajectories=100)

# Each trajectory is a Trajectory object with:
traj = trajectories[0]
print(traj.pickup_cell)    # (x, y) of the last state
print(traj.n_states)       # Number of states in the trajectory
print(traj.driver_id)      # Driver who generated this trajectory
```

---

## 4. Dataset 2: Pickup/Dropoff Counts

### 4.1 File

`source_data/pickup_dropoff_counts.pkl`

### 4.2 Structure

```python
{
    (x_grid, y_grid, time_bucket, day_of_week): [pickup_count, dropoff_count],
    ...
}
```

A dictionary mapping spatiotemporal state keys to pickup and dropoff counts.

### 4.3 Key Schema

| Index | Field | Type | Description | Range |
|-------|-------|------|-------------|-------|
| 0 | `x_grid` | `int` | Grid x-coordinate | $[0, 47]$ |
| 1 | `y_grid` | `int` | Grid y-coordinate | $[0, 89]$ |
| 2 | `time_bucket` | `int` | 5-minute time interval (1-indexed) | $[1, 288]$ |
| 3 | `day_of_week` | `int` | Day of week (1-indexed) | $[1, 6]$ (Mon–Sat) |

### 4.4 Value Schema

Each value is a 2-element list:

| Index | Field | Type | Description |
|-------|-------|------|-------------|
| 0 | `pickup_count` | `int` | Number of passenger pickups at this state |
| 1 | `dropoff_count` | `int` | Number of passenger dropoffs at this state |

### 4.5 Coverage

- **Total keys**: ~234,000 non-zero state keys
- **Theoretical state space**: $48 \times 90 \times 288 \times 6 = 7{,}464{,}960$
- **Coverage**: ~3.1% (sparse — most cells have zero events in most time periods)
- **Source**: 3 months of expert driver data (July, August, September)

### 4.6 Time Bucket Conversion

```python
# Convert 1-indexed time_bucket to HH:MM
minutes = (time_bucket - 1) * 5
hours = minutes // 60
mins = minutes % 60

# Convert time_bucket to hour (for matching with active_taxis)
hour = (time_bucket - 1) // 12  # Maps [1, 288] → [0, 23]
```

### 4.7 Usage in the Framework

The pickup/dropoff counts serve two purposes:

1. **Spatial fairness**: Aggregated as **sums** across all time periods to produce a $48 \times 90$ grid of total pickup and dropoff counts. These are used to compute DSR and ASR.

2. **Causal fairness**: Aggregated as **means** across time periods (after conversion to hourly granularity) to produce mean demand per cell. This is used to compute $Y = S/D$ at the same scale as the $g(d)$ function was fitted.

---

## 5. Dataset 3: Active Taxis

### 5.1 File

`source_data/active_taxis_5x5_hourly.pkl`

### 5.2 Structure

The file contains a bundle with metadata:

```python
{
    'data': {
        (x_grid, y_grid, hour, day_of_week): active_count,
        ...
    },
    'stats': {...},     # Processing statistics
    'config': {...},    # Generation configuration
    'version': '1.0.0'
}
```

### 5.3 Key Schema (Hourly Variant)

| Index | Field | Type | Description | Range |
|-------|-------|------|-------------|-------|
| 0 | `x_grid` | `int` | Grid x-coordinate (1-indexed) | $[1, 48]$ |
| 1 | `y_grid` | `int` | Grid y-coordinate (1-indexed) | $[1, 90]$ |
| 2 | `hour` | `int` | Hour of day | $[0, 23]$ |
| 3 | `day_of_week` | `int` | Day of week (1-indexed) | $[1, 6]$ (Mon–Sat) |

### 5.4 Value

A single integer: the number of unique taxis that had at least one GPS reading in the $5 \times 5$ neighborhood of cell $(x, y)$ during the specified hour and day.

- **Range**: $[0, 50]$ (50 drivers in the dataset)
- **Typical range**: $0$–$15$ for most cells; up to $30$–$40$ in high-activity areas

### 5.5 Neighborhood Definition

An "active taxi" count for cell $(x, y)$ includes any taxi that appeared in any cell within a $5 \times 5$ window centered on $(x, y)$. This provides a smoothed measure of taxi availability that accounts for nearby supply.

### 5.6 Usage in the Framework

Active taxis data serves two purposes:

1. **Spatial fairness normalization**: Aggregated as **means** across time periods to produce a $48 \times 90$ grid. Used as the denominator in DSR ($\text{pickups}/A_c$) and ASR ($\text{dropoffs}/A_c$) to normalize service counts by taxi availability.

2. **Causal fairness supply**: Aggregated as **means** across time periods to produce the supply grid $S_c$. Used in the service ratio $Y_c = S_c / D_c$ for the causal fairness $R^2$ computation.

A floor of $0.1$ is applied to the active taxis grid to prevent division by zero in DSR/ASR calculations.

---

## 6. Aggregation Details

### 6.1 Temporal Aggregation to 2D Grid

All three datasets are temporally indexed (by time bucket, hour, and/or day). The framework aggregates each dataset to a 2D spatial grid ($48 \times 90$) by combining all time periods.

| Grid | Source | Keys | Aggregation | Output Shape |
|------|--------|------|-------------|--------------|
| `pickup_grid` | pickup_dropoff_counts | $(x, y, t, d)$ | **Sum** of pickups | $(48, 90)$ |
| `dropoff_grid` | pickup_dropoff_counts | $(x, y, t, d)$ | **Sum** of dropoffs | $(48, 90)$ |
| `active_taxis_grid` | active_taxis | $(x, y, h, d)$ | **Mean** of counts | $(48, 90)$ |
| `causal_demand_grid` | pickup_dropoff_counts | $(x, y, h, d)$ | **Mean** of hourly pickups | $(48, 90)$ |
| `causal_supply_grid` | active_taxis | $(x, y, h, d)$ | **Mean** of counts | $(48, 90)$ |

### 6.2 Why Different Aggregations?

**Sum for spatial fairness**: Total pickup and dropoff counts capture the overall volume of service. Spatial fairness asks "is the total service distribution equitable?" which requires summed counts.

**Mean for causal fairness**: The $g(d)$ function is fitted on hourly-averaged data. For $Y = S/D$ to be in the same scale as $g(d)$ predictions, both $S$ and $D$ must use the same temporal averaging. Using sums would produce values orders of magnitude larger than those $g(d)$ was trained on.

### 6.3 Coordinate Conversion During Aggregation

The pickup/dropoff data uses 0-indexed coordinates in some variants and 1-indexed in others. The `PickupDropoffLoader` converts 1-indexed to 0-indexed via `x - 1, y - 1`. The `ActiveTaxisLoader` similarly converts 1-indexed coordinates.

---

## 7. G-Function Data Pipeline

The $g(d)$ function is estimated from the intersection of the pickup/dropoff and active taxis datasets:

```
pickup_dropoff_counts.pkl
    │
    ├─ Extract demand per state: D_{x,y,t,d} = pickup_count
    │
    └─ Convert to hourly: D_{x,y,h,d} = sum of pickups in hour h
                                          │
active_taxis_5x5_hourly.pkl ──────────────┤
    │                                      │
    └─ Match keys: S_{x,y,h,d}            │
                                           │
              ┌────────────────────────────┘
              │
    Compute Y = S / D for matched keys (D ≥ 1)
              │
    Aggregate per cell (mean across time periods)
              │
    Fit isotonic regression: g(D) ≈ E[Y | D]
              │
    Return: g_function, diagnostics
```

---

## 8. DataBundle

The `DataBundle` class consolidates all loaded data into a single object:

```python
@dataclass
class DataBundle:
    trajectories: List[Trajectory]           # Loaded trajectories
    pickup_dropoff_data: Dict                 # Raw pickup/dropoff counts
    pickup_grid: np.ndarray                   # Sum-aggregated pickups (48, 90)
    dropoff_grid: np.ndarray                  # Sum-aggregated dropoffs (48, 90)
    active_taxis_data: Dict                   # Raw active taxis counts
    active_taxis_grid: np.ndarray             # Mean-aggregated active taxis (48, 90)
    g_function: Callable                      # Fitted g(d) function
    causal_demand_grid: Optional[np.ndarray]  # Mean demand per cell (48, 90)
    causal_supply_grid: Optional[np.ndarray]  # Mean supply per cell (48, 90)
    g_function_diagnostics: Optional[Dict]    # g(d) fitting information
```

### 8.1 Loading With Recommended Settings

```python
from trajectory_modification import DataBundle

bundle = DataBundle.load_default(
    max_trajectories=100,
    estimate_g_from_data=True,   # Fit g(d) via isotonic regression (recommended)
    aggregation='mean',          # Mean aggregation for causal fairness (recommended)
)
```

### 8.2 Converting to Tensors

```python
tensors = bundle.to_tensors(device='cpu')
# Returns: {'pickup_grid': Tensor, 'dropoff_grid': Tensor, 'active_taxis_grid': Tensor}
```

The causal fairness grids (`causal_demand_grid`, `causal_supply_grid`) remain as numpy arrays on the `DataBundle` and are converted to tensors separately when passed to the modifier.

---

## 9. Trajectory Representation in Code

### 9.1 TrajectoryState

```python
@dataclass
class TrajectoryState:
    x_grid: float       # Grid x-coordinate [0, 47]
    y_grid: float       # Grid y-coordinate [0, 89]
    time_bucket: int    # Time period [0, 287] (0-indexed)
    day_index: int      # Day of week
```

### 9.2 Trajectory

```python
@dataclass
class Trajectory:
    trajectory_id: Any              # Unique identifier
    driver_id: Any                  # Driver who generated this trajectory
    states: List[TrajectoryState]   # Sequence from seeking start to pickup
    metadata: dict                  # Optional metadata
```

Key properties:
- `trajectory.pickup_state` → the final `TrajectoryState` (pickup location)
- `trajectory.pickup_cell` → `(int(x), int(y))` tuple of the pickup cell
- `trajectory.to_tensor()` → `torch.Tensor` of shape `[seq_len, 4]`
- `trajectory.apply_perturbation(delta)` → new `Trajectory` with modified pickup

### 9.3 Discriminator Input Format

The discriminator expects trajectory tensors of shape `[batch, seq_len, 4]` where each row is `[x_grid, y_grid, time_bucket, day_index]`. The `Trajectory.to_tensor()` method produces the `[seq_len, 4]` tensor, and a batch dimension is prepended before passing to the discriminator.
