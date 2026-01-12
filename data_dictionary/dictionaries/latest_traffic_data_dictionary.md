# latest_traffic.pkl Data Dictionary

## Overview

The `latest_traffic.pkl` dataset contains aggregated traffic condition data indexed by spatiotemporal state keys. It provides traffic speed and waiting time metrics for specific grid locations at specific times, derived from historical traffic data. This dataset serves as a lookup table for enriching trajectory data with traffic-related features.

**File Format:** Python pickle file (`.pkl`)  
**Data Type:** Dictionary  
**Approximate Size:** ~6.4 KB (JSON equivalent)

---

## Data Structure

### Hierarchy

```
latest_traffic.pkl
├── (x_grid, y_grid, time_bucket, day_of_week) → [traffic_speed, waiting_time, unverified_value]
├── (x_grid, y_grid, time_bucket, day_of_week) → [traffic_speed, waiting_time, unverified_value]
└── ... (multiple state keys)
```

### Loading the Data

```python
import pickle

# Load the dataset
latest_traffic = pickle.load(open('path/to/latest_traffic.pkl', 'rb'))

# Get list of state keys
traffic_keys = list(latest_traffic.keys())
print(f'Number of keys: {len(traffic_keys)}')

# Access traffic data for a specific state
state_key = traffic_keys[0]
traffic_data = latest_traffic[state_key]
print(f'State key: {state_key}')
print(f'Traffic data: {traffic_data}')
```

---

## Key Structure

Each key in the dictionary is a **4-element tuple** representing a unique spatiotemporal state.

### Key Schema

| Index | Field | Data Type | Description | Value Range |
|-------|-------|-----------|-------------|-------------|
| 0 | `x_grid` | `int` | Grid index for longitude position | 0-49 (typical) |
| 1 | `y_grid` | `int` | Grid index for latitude position | 0-89 (typical) |
| 2 | `time_bucket` | `int` | Discretized time-of-day slot | [0, 287] |
| 3 | `day_of_week` | `int` | Day-of-week indicator | [0, 6] or [1, 7] |

### Key Format

```python
# Key format: (x_grid, y_grid, time_bucket, day_of_week)
example_keys = [
    (36, 75, 161, 4),
    (44, 34, 176, 6),
    (17, 16, 202, 1),
]
```

### Time Bucket Calculation

The `time_bucket` field represents a 5-minute interval within a 24-hour day:

```python
# Convert time_bucket to human-readable time
def time_bucket_to_time(bucket):
    """Convert time bucket to HH:MM format."""
    minutes = bucket * 5
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

# Example: bucket 161 = 13:25 (1:25 PM)
# Calculation: 161 * 5 = 805 minutes = 13 hours 25 minutes
```

---

## Value Structure

Each value in the dictionary is a **3-element list** containing traffic metrics.

### Value Schema

| Index | Field | Data Type | Description | Typical Range |
|-------|-------|-----------|-------------|---------------|
| 0 | `traffic_speed` | `float` | Normalized traffic speed metric | 0.0 - 0.025 |
| 1 | `waiting_time` | `float` | Average waiting time (seconds or normalized) | 0.0 - 50+ |
| 2 | `unverified_value` | `float` | Additional metric (purpose unverified) | Varies widely |

### Sample Values

```python
# Example entries from the dataset
{
    (36, 75, 161, 4): [0.00016876003676169666, 0.0, 15.0],
    (44, 34, 176, 6): [0.008716961972847947, 4.033333333333333, 19.025],
    (17, 16, 202, 1): [0.015280338351589621, 11.847619047619048, 255.42499999999998],
    (35, 30, 272, 4): [0.0037694678983152185, 26.533333333333335, 34.9],
}
```

---

## Field Descriptions

### Index 0: `traffic_speed`

- **Type:** Float
- **Description:** Normalized traffic speed for the given spatiotemporal state
- **Typical Range:** 0.0 to ~0.025
- **Notes:**
  - Higher values indicate faster traffic flow
  - Value appears to be normalized (not raw speed in km/h or mph)
  - Used as source for `traffic_speed_norm` features in `all_trajs.pkl`

### Index 1: `waiting_time`

- **Type:** Float
- **Description:** Average traffic waiting time for the given state
- **Typical Range:** 0.0 to 50+ (can be higher in congested areas)
- **Units:** Likely seconds or a normalized time unit
- **Notes:**
  - Higher values indicate more congestion/delays
  - Value of 0.0 indicates free-flowing traffic
  - Used as source for `traffic_wait_norm` features in `all_trajs.pkl`

### Index 2: `unverified_value`

- **Type:** Float
- **Description:** Additional traffic-related metric (purpose not definitively verified)
- **Typical Range:** Highly variable (0.25 to 1000+)
- **Possible Interpretations:**
  - Sample count or observation frequency
  - Aggregated traffic volume
  - Confidence score
- **Note:** This field's meaning should be verified against source documentation

---

## Common Access Patterns

### Direct State Lookup

```python
def get_traffic_data(latest_traffic, x, y, time_bucket, day):
    """
    Retrieve traffic data for a specific state.
    
    Returns:
        tuple: (traffic_speed, waiting_time, unverified) or None if not found
    """
    key = (x, y, time_bucket, day)
    if key in latest_traffic:
        return tuple(latest_traffic[key])
    return None

# Usage
traffic = get_traffic_data(latest_traffic, 17, 16, 202, 1)
if traffic:
    speed, wait, extra = traffic
    print(f"Speed: {speed}, Wait: {wait}")
```

### Safe Lookup with Defaults

```python
def get_traffic_with_defaults(latest_traffic, x, y, time_bucket, day,
                               default_speed=0.0, default_wait=0.0):
    """
    Retrieve traffic data with fallback defaults for missing states.
    """
    key = (x, y, time_bucket, day)
    if key in latest_traffic:
        data = latest_traffic[key]
        return data[0], data[1]
    return default_speed, default_wait

# Usage (with normalization defaults from all_trajs processing)
SPEED_DEFAULT = -0.009096 / 0.0077867  # Normalized baseline/scale
WAIT_DEFAULT = -9.2149 / 20.8396       # Normalized baseline/scale
```

### Iterate Over All States

```python
for state_key, values in latest_traffic.items():
    x, y, t, day = state_key
    speed, wait, extra = values
    # Process each state
```

### Extract All Keys for a Specific Day

```python
def get_states_by_day(latest_traffic, target_day):
    """Get all state keys for a specific day of week."""
    return [key for key in latest_traffic.keys() if key[3] == target_day]

# Get all Monday states (assuming day_index 1 = Monday)
monday_states = get_states_by_day(latest_traffic, 1)
```

### Extract Traffic for a 5×5 Window

```python
def get_traffic_window(latest_traffic, center_x, center_y, time_bucket, day):
    """
    Extract traffic data for a 5×5 window centered on (center_x, center_y).
    Returns 25-element lists for speed and wait time.
    """
    speeds = []
    waits = []
    
    for dx in range(-2, 3):  # -2, -1, 0, 1, 2
        for dy in range(-2, 3):
            x, y = center_x + dx, center_y + dy
            key = (x, y, time_bucket, day)
            
            if key in latest_traffic:
                speeds.append(latest_traffic[key][0])
                waits.append(latest_traffic[key][1])
            else:
                # Apply default values for missing states
                speeds.append(0.0)  # or use normalization defaults
                waits.append(0.0)
    
    return speeds, waits  # Each list has 25 elements
```

---

## Data Statistics

### Coverage

- **Total States:** Variable (check `len(latest_traffic)`)
- **Grid Coverage:** Not all (x, y, t, day) combinations have entries
- **Sparse Data:** Missing keys indicate no traffic data available for that state

### Value Distributions

```python
import numpy as np

# Analyze traffic speed distribution
speeds = [v[0] for v in latest_traffic.values()]
print(f"Speed - Min: {min(speeds):.6f}, Max: {max(speeds):.6f}, Mean: {np.mean(speeds):.6f}")

# Analyze waiting time distribution
waits = [v[1] for v in latest_traffic.values()]
print(f"Wait - Min: {min(waits):.4f}, Max: {max(waits):.4f}, Mean: {np.mean(waits):.4f}")
```

---

## Usage in FAMAIL Pipeline

### Integration with all_trajs.pkl

This dataset is used to populate indices **75-99** (`traffic_speed_norm`) and **100-124** (`traffic_wait_norm`) in the `all_trajs.pkl` state vectors.

```python
# In the feature construction pipeline (net-dis-cGAIL.ipynb):
# For each state in a trajectory:
#   1. Extract (x, y, time_bucket, day) from base state
#   2. Generate 5×5 window coordinates
#   3. Look up traffic data for each window cell
#   4. Apply normalization
#   5. Append to feature vector
```

### Normalization Constants

When traffic data is missing for a state, the following normalization defaults are applied:

| Metric | Baseline | Scale |
|--------|----------|-------|
| Traffic Speed | -0.009096 | 0.0077867 |
| Traffic Wait | -9.2149 | 20.8396 |

---

## Data Provenance

- **Source:** Historical traffic monitoring data for Shenzhen
- **Aggregation:** Values represent aggregated/averaged metrics from historical observations
- **Temporal Resolution:** 5-minute intervals (288 buckets per day)
- **Spatial Resolution:** Matches the grid system used in trajectory data

---

## Important Notes

1. **Sparse Coverage:** Not all spatiotemporal states have corresponding traffic data
2. **Value Index 2:** The third value in each entry (`unverified_value`) should be used with caution until its meaning is verified
3. **Normalization:** Raw values from this dataset are normalized before use in `all_trajs.pkl`
4. **Key Format:** Keys are tuples, not strings; convert appropriately if loading from JSON
5. **Day Indexing:** Verify the day-of-week encoding (0-indexed or 1-indexed) against other data sources

---

## Related Datasets

- `all_trajs.pkl` - Trajectory data that uses this dataset for feature enrichment
- `latest_volume_pickups.pkl` - Companion dataset for pickup and volume features
- `train_airport.pkl` - POI location data
