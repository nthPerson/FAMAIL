# latest_volume_pickups.pkl Data Dictionary

## Overview

The `latest_volume_pickups.pkl` dataset contains aggregated taxi pickup and traffic volume data indexed by spatiotemporal state keys. It provides counts of taxi pickups and total traffic volume for specific grid locations at specific times, derived from historical transportation data. This dataset serves as a lookup table for enriching trajectory data with demand-related features.

**File Format:** Python pickle file (`.pkl`)  
**Data Type:** Dictionary  
**Approximate Size:** ~2.9 KB (JSON equivalent)

---

## Data Structure

### Hierarchy

```
latest_volume_pickups.pkl
├── (x_grid, y_grid, time_bucket, day_of_week) → [pickup_count, traffic_volume]
├── (x_grid, y_grid, time_bucket, day_of_week) → [pickup_count, traffic_volume]
└── ... (multiple state keys)
```

### Loading the Data

```python
import pickle

# Load the dataset
latest_volume_pickups = pickle.load(open('path/to/latest_volume_pickups.pkl', 'rb'))

# Get list of state keys
volume_keys = list(latest_volume_pickups.keys())
print(f'Number of keys: {len(volume_keys)}')

# Access data for a specific state
state_key = volume_keys[0]
volume_data = latest_volume_pickups[state_key]
print(f'State key: {state_key}')
print(f'Volume data: {volume_data}')
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
    (17, 16, 202, 1),
    (46, 40, 226, 1),
    (18, 13, 34, 2),
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

# Examples:
# bucket 161 = 13:25 (1:25 PM)
# bucket 287 = 23:55 (11:55 PM)
# bucket 0   = 00:00 (midnight)
```

---

## Value Structure

Each value in the dictionary is a **2-element list** containing demand and volume metrics.

### Value Schema

| Index | Field | Data Type | Description | Typical Range |
|-------|-------|-----------|-------------|---------------|
| 0 | `pickup_count` | `int` | Number of taxi pickups in the cell | 0 - 20+ |
| 1 | `traffic_volume` | `int` | Total traffic/trip volume in the cell | 1 - 2500+ |

### Sample Values

```python
# Example entries from the dataset
{
    (36, 75, 161, 4): [0, 15],
    (17, 16, 202, 1): [3, 547],
    (46, 40, 226, 1): [13, 581],
    (18, 13, 34, 2): [4, 2302],
    (7, 19, 110, 5): [6, 354],
    (32, 50, 287, 4): [16, 1018],
}
```

---

## Field Descriptions

### Index 0: `pickup_count`

- **Type:** Integer
- **Description:** Number of taxi pickups that occurred in the specified grid cell during the given time bucket and day
- **Typical Range:** 0 to 20+ (can be higher in high-demand areas)
- **Notes:**
  - Value of 0 indicates no pickups recorded for that state
  - Higher values indicate taxi demand hotspots
  - Used as source for `pickup_count_norm` features in `all_trajs.pkl`

### Index 1: `traffic_volume`

- **Type:** Integer
- **Description:** Total traffic/trip volume observed in the grid cell during the specified time and day
- **Typical Range:** 1 to 2500+ (varies significantly by location)
- **Notes:**
  - Represents aggregate traffic activity, not just taxi-related
  - Higher values indicate busier areas
  - Used as source for `traffic_volume_norm` features in `all_trajs.pkl`
  - Generally much larger than pickup_count as it includes all traffic

---

## Common Access Patterns

### Direct State Lookup

```python
def get_volume_data(latest_volume_pickups, x, y, time_bucket, day):
    """
    Retrieve pickup and volume data for a specific state.
    
    Returns:
        tuple: (pickup_count, traffic_volume) or None if not found
    """
    key = (x, y, time_bucket, day)
    if key in latest_volume_pickups:
        return tuple(latest_volume_pickups[key])
    return None

# Usage
data = get_volume_data(latest_volume_pickups, 17, 16, 202, 1)
if data:
    pickups, volume = data
    print(f"Pickups: {pickups}, Volume: {volume}")
```

### Safe Lookup with Defaults

```python
def get_volume_with_defaults(latest_volume_pickups, x, y, time_bucket, day,
                              default_pickups=0, default_volume=0):
    """
    Retrieve volume data with fallback defaults for missing states.
    """
    key = (x, y, time_bucket, day)
    if key in latest_volume_pickups:
        data = latest_volume_pickups[key]
        return data[0], data[1]
    return default_pickups, default_volume

# Usage (with normalization defaults from all_trajs processing)
PICKUP_DEFAULT = -1.7411 / 8.6891    # Normalized baseline/scale
VOLUME_DEFAULT = -241.1649 / 864.8101  # Normalized baseline/scale
```

### Iterate Over All States

```python
for state_key, values in latest_volume_pickups.items():
    x, y, t, day = state_key
    pickups, volume = values
    # Process each state
```

### Find High-Demand Locations

```python
def find_hotspots(latest_volume_pickups, min_pickups=10):
    """Find state keys with high pickup counts."""
    hotspots = []
    for key, values in latest_volume_pickups.items():
        if values[0] >= min_pickups:
            hotspots.append((key, values[0]))
    return sorted(hotspots, key=lambda x: x[1], reverse=True)

# Get top pickup locations
top_hotspots = find_hotspots(latest_volume_pickups, min_pickups=10)
for key, count in top_hotspots[:10]:
    print(f"Location {key}: {count} pickups")
```

### Extract Data for a 5×5 Window

```python
def get_volume_window(latest_volume_pickups, center_x, center_y, time_bucket, day):
    """
    Extract pickup and volume data for a 5×5 window centered on (center_x, center_y).
    Returns 25-element lists for pickups and volumes.
    """
    pickups = []
    volumes = []
    
    for dx in range(-2, 3):  # -2, -1, 0, 1, 2
        for dy in range(-2, 3):
            x, y = center_x + dx, center_y + dy
            key = (x, y, time_bucket, day)
            
            if key in latest_volume_pickups:
                pickups.append(latest_volume_pickups[key][0])
                volumes.append(latest_volume_pickups[key][1])
            else:
                # Apply default values for missing states
                pickups.append(0)
                volumes.append(0)
    
    return pickups, volumes  # Each list has 25 elements
```

### Aggregate by Day of Week

```python
from collections import defaultdict

def aggregate_by_day(latest_volume_pickups):
    """Aggregate total pickups and volume by day of week."""
    day_totals = defaultdict(lambda: {'pickups': 0, 'volume': 0})
    
    for key, values in latest_volume_pickups.items():
        day = key[3]
        day_totals[day]['pickups'] += values[0]
        day_totals[day]['volume'] += values[1]
    
    return dict(day_totals)

# Usage
by_day = aggregate_by_day(latest_volume_pickups)
for day, totals in sorted(by_day.items()):
    print(f"Day {day}: {totals['pickups']} pickups, {totals['volume']} volume")
```

---

## Data Statistics

### Coverage

- **Total States:** Variable (check `len(latest_volume_pickups)`)
- **Grid Coverage:** Not all (x, y, t, day) combinations have entries
- **Sparse Data:** Missing keys indicate no recorded activity for that state

### Value Distributions

```python
import numpy as np

# Analyze pickup count distribution
pickups = [v[0] for v in latest_volume_pickups.values()]
print(f"Pickups - Min: {min(pickups)}, Max: {max(pickups)}, Mean: {np.mean(pickups):.2f}")

# Analyze traffic volume distribution
volumes = [v[1] for v in latest_volume_pickups.values()]
print(f"Volume - Min: {min(volumes)}, Max: {max(volumes)}, Mean: {np.mean(volumes):.2f}")

# Count states with zero pickups
zero_pickup_count = sum(1 for p in pickups if p == 0)
print(f"States with 0 pickups: {zero_pickup_count} ({100*zero_pickup_count/len(pickups):.1f}%)")
```

---

## Usage in FAMAIL Pipeline

### Integration with all_trajs.pkl

This dataset is used to populate indices **25-49** (`pickup_count_norm`) and **50-74** (`traffic_volume_norm`) in the `all_trajs.pkl` state vectors.

```python
# In the feature construction pipeline (net-dis-cGAIL.ipynb):
# For each state in a trajectory:
#   1. Extract (x, y, time_bucket, day) from base state
#   2. Generate 5×5 window coordinates
#   3. Look up volume/pickup data for each window cell
#   4. Apply normalization
#   5. Append to feature vector
```

### Normalization Constants

When volume/pickup data is missing for a state, the following normalization defaults are applied:

| Metric | Baseline | Scale |
|--------|----------|-------|
| Pickup Count | -1.7411 | 8.6891 |
| Traffic Volume | -241.1649 | 864.8101 |

### Normalization Formula

```python
def normalize(value, baseline, scale):
    """Standard normalization formula used in feature construction."""
    return (value - baseline) / scale
```

---

## Data Provenance

- **Source:** Historical taxi GPS and transaction records from Shenzhen
- **Aggregation:** Values represent aggregated counts from historical observations
- **Temporal Resolution:** 5-minute intervals (288 buckets per day)
- **Spatial Resolution:** Matches the grid system used in trajectory data (approximately 50×90 cells)

---

## Important Notes

1. **Sparse Coverage:** Not all spatiotemporal states have corresponding data entries
2. **Integer Values:** Unlike `latest_traffic.pkl`, values here are integers (counts)
3. **Zero Pickups Common:** Many states have `pickup_count = 0` but non-zero `traffic_volume`
4. **Key Format:** Keys are tuples, not strings; convert appropriately if loading from JSON
5. **Day Indexing:** Verify the day-of-week encoding (0-indexed or 1-indexed) against other data sources
6. **Demand Indicator:** `pickup_count` is a key indicator of taxi demand and can be used for hotspot analysis

---

## Relationship Between Fields

The ratio of `pickup_count` to `traffic_volume` can indicate the "taxi density" of an area:

```python
def compute_taxi_density(pickups, volume):
    """
    Compute taxi density ratio.
    Higher values indicate areas where taxis are more prevalent relative to total traffic.
    """
    if volume == 0:
        return 0.0
    return pickups / volume

# Example analysis
for key, values in latest_volume_pickups.items():
    density = compute_taxi_density(values[0], values[1])
    if density > 0.02:  # 2% or higher taxi presence
        print(f"High taxi density at {key}: {density:.4f}")
```

---

## Related Datasets

- `all_trajs.pkl` - Trajectory data that uses this dataset for feature enrichment
- `latest_traffic.pkl` - Companion dataset for traffic speed and waiting time features
- `train_airport.pkl` - POI location data for train stations and airports
