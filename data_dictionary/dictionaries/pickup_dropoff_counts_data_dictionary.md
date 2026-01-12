# pickup_dropoff_counts.pkl Data Dictionary

## Overview

The `pickup_dropoff_counts.pkl` dataset contains aggregated pickup and dropoff event counts indexed by spatiotemporal state keys. It provides historical pickup and dropoff frequencies for specific grid locations at specific times, aggregated from three months of expert driver data (July, August, and September from `taxi_record_07_50drivers.pkl`, `taxi_record_08_50drivers.pkl`, and `taxi_record_09_50drivers.pkl`). This dataset serves as a lookup table for understanding demand patterns and contextualizing agent decisions in the FAMAIL reinforcement learning framework.

**File Format:** Python pickle file (`.pkl`)  
**Data Type:** Dictionary  
**Source Data:** Expert driver trajectories from 50 drivers over 3 months  
**Coverage:** ~3.1% of theoretical state space (sparse representation)

---

## Data Structure

### Hierarchy

```
pickup_dropoff_counts.pkl
├── (x_grid, y_grid, time_bucket, day_of_week) → [pickup_count, dropoff_count]
├── (x_grid, y_grid, time_bucket, day_of_week) → [pickup_count, dropoff_count]
└── ... (~234,000 non-zero state keys)
```

### Loading the Data

```python
import pickle

# Load the dataset
pickup_dropoff_counts = pickle.load(open('path/to/pickup_dropoff_counts.pkl', 'rb'))

# Get list of state keys
state_keys = list(pickup_dropoff_counts.keys())
print(f'Number of keys: {len(state_keys)}')  # Output: ~234,000

# Access counts for a specific state
example_key = (4, 22, 201, 2)
counts = pickup_dropoff_counts[example_key]
print(f'Pickups: {counts[0]}, Dropoffs: {counts[1]}')  # Output: (20, 14)

# Iterate through the first 10 states
for key in list(pickup_dropoff_counts.keys())[:10]:
    pickups, dropoffs = pickup_dropoff_counts[key]
    print(f'State {key}: {pickups} pickups, {dropoffs} dropoffs')
```

---

## Key Structure

Each key in the dictionary is a **4-element tuple** representing a unique spatiotemporal state from the complete FAMAIL state space.

### Key Schema

| Index | Field | Data Type | Description | Value Range |
|-------|-------|-----------|-------------|-------------|
| 0 | `x_grid` | `int` | Grid index for latitude position | [0, 47] (48 values) |
| 1 | `y_grid` | `int` | Grid index for longitude position | [0, 89] (90 values) |
| 2 | `time_bucket` | `int` | Discretized time-of-day slot | [1, 288] (1-indexed) |
| 3 | `day_of_week` | `int` | Day-of-week indicator | [1, 6] (Mon-Sat, 1-indexed) |

### Key Format

```python
# Key format: (x_grid, y_grid, time_bucket, day_of_week)
example_keys = [
    (4, 22, 201, 2),   # Tuesday, ~16:45 (4:45 PM)
    (15, 45, 144, 1),  # Monday, 12:00 PM
    (30, 67, 72, 5),   # Friday, 6:00 AM
]
```

### Time Bucket Calculation

The `time_bucket` field represents a 5-minute interval within a 24-hour day. **Note:** This dataset uses **1-indexed** time buckets (1-288), unlike some other datasets that use 0-indexed (0-287).

```python
# Convert time_bucket to human-readable time (1-indexed)
def time_bucket_to_time(bucket):
    """Convert 1-indexed time bucket to HH:MM format."""
    minutes = (bucket - 1) * 5
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

# Example: bucket 201 = 16:40 (4:40 PM)
# Calculation: (201 - 1) * 5 = 1000 minutes = 16 hours 40 minutes
```

### Day-of-Week Mapping

```python
# Day-of-week encoding (1-indexed)
day_mapping = {
    1: 'Monday',
    2: 'Tuesday',
    3: 'Wednesday',
    4: 'Thursday',
    5: 'Friday',
    6: 'Saturday'
}
# Note: Sunday (day 0 or 7) is not present in the dataset
```

---

## Value Structure

Each value in the dictionary is a **2-element list** containing aggregated event counts.

### Value Schema

| Index | Field | Data Type | Description | Typical Range |
|-------|-------|-----------|-------------|---------------|
| 0 | `pickup_count` | `int` | Total pickups at this state | 0 - several hundred |
| 1 | `dropoff_count` | `int` | Total dropoffs at this state | 0 - several hundred |

### Sample Values

```python
# Format: (x_grid, y_grid, time_bucket, day_of_week) → [pickup_count, dropoff_count]
{
    (4, 22, 201, 2): [20, 14],    # High activity location
    (15, 45, 144, 1): [1, 2],     # Low activity location
    (30, 67, 72, 5): [5, 3],      # Moderate activity location
}
```

---

## State Space Properties

### Dimensions

The pickup_dropoff_counts dataset is defined over a 4-dimensional state space:

| Dimension | Size | Description |
|-----------|------|-------------|
| `x_grid` | 48 | Latitude dimension of discretized grid |
| `y_grid` | 90 | Longitude dimension of discretized grid |
| `time_bucket` | 288 | 5-minute intervals (24 hours × 12 intervals/hour) |
| `day_of_week` | 6 | Monday through Saturday |

**Theoretical Maximum Keys:** 48 × 90 × 288 × 6 = **7,464,960 possible states**

### Sparsity

The dataset exhibits significant sparsity, which is typical for real-world spatiotemporal event data:

- **Observed Non-Zero Keys:** ~234,000 (~3.1% of theoretical maximum)
- **Interpretation:** Most grid cells at most times have no pickup or dropoff events
- **Implication:** The dataset efficiently represents only states with observed activity

### Coverage Characteristics

```python
# Example sparsity analysis
total_possible_states = 48 * 90 * 288 * 6  # 7,464,960
observed_states = len(pickup_dropoff_counts)  # ~234,000
coverage_percentage = (observed_states / total_possible_states) * 100  # ~3.1%

print(f"Observed states: {observed_states:,}")
print(f"Coverage: {coverage_percentage:.2f}%")
```

---

## Count Distribution Statistics

### Pickup Counts

Based on analysis of non-zero cells:

| Statistic | Value | Description |
|-----------|-------|-------------|
| **Mean** | ~1.39 | Average pickups per non-zero state |
| **Median** | 1 | Typical state has 1 pickup |
| **Mode** | 1 | Most common count value |
| **Max** | Several hundred | High-traffic locations/times |
| **Distribution** | Heavy right tail | Few states have very high counts |

### Dropoff Counts

Dropoff statistics follow a similar distribution to pickups:

- **Pattern:** Comparable to pickup distribution
- **Correlation:** Often correlated with pickup counts at the same state
- **Variance:** May differ at boundary locations (e.g., airport, city limits)

### Distribution Characteristics

```python
# Typical distribution analysis
import matplotlib.pyplot as plt

pickup_counts = [counts[0] for counts in pickup_dropoff_counts.values()]
dropoff_counts = [counts[1] for counts in pickup_dropoff_counts.values()]

# Heavy right-tailed distribution (typical for count data)
# - Most states: 1-5 events
# - Few states: 100+ events (major hubs, peak hours)
```

---

## Usage Examples

### Basic Lookup

```python
import pickle

# Load data
data = pickle.load(open('pickup_dropoff_counts.pkl', 'rb'))

# Query specific state
state_key = (x, y, time, day)
if state_key in data:
    pickups, dropoffs = data[state_key]
    print(f"State {state_key}: {pickups} pickups, {dropoffs} dropoffs")
else:
    print(f"No recorded activity for state {state_key}")
```

### Temporal Analysis

```python
# Find peak pickup times for a specific location
location = (15, 45)  # (x_grid, y_grid)
day = 3  # Wednesday

pickup_by_time = {}
for time_bucket in range(1, 289):
    key = (*location, time_bucket, day)
    if key in data:
        pickup_by_time[time_bucket] = data[key][0]

# Find busiest time slot
busiest_time = max(pickup_by_time.items(), key=lambda x: x[1])
print(f"Peak time: bucket {busiest_time[0]} with {busiest_time[1]} pickups")
```

### Spatial Heatmap

```python
# Aggregate pickups across all times for a specific day
import numpy as np

day = 1  # Monday
heatmap = np.zeros((48, 90))

for (x, y, time, d), (pickups, dropoffs) in data.items():
    if d == day:
        heatmap[x, y] += pickups

# Visualize high-demand areas
import matplotlib.pyplot as plt
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.colorbar(label='Total Pickups')
plt.title(f'Pickup Heatmap - Day {day}')
plt.show()
```

### Demand Ratio Analysis

```python
# Calculate pickup-to-dropoff ratio for each state
for key, (pickups, dropoffs) in data.items():
    if dropoffs > 0:
        ratio = pickups / dropoffs
        if ratio > 2.0:  # More pickups than dropoffs
            print(f"High pickup area: {key}, ratio={ratio:.2f}")
        elif ratio < 0.5:  # More dropoffs than pickups
            print(f"High dropoff area: {key}, ratio={ratio:.2f}")
```

---

## Relationship to Other Datasets

### Integration with `all_trajs.pkl`

The pickup_dropoff_counts dataset complements trajectory data by providing demand context:

```python
# Enrich trajectory state with demand information
state_key = (x_grid, y_grid, time_bucket, day_of_week)
if state_key in pickup_dropoff_counts:
    pickups, dropoffs = pickup_dropoff_counts[state_key]
    # Use as features for decision-making
```

### Comparison with `latest_volume_pickups.pkl`

- **pickup_dropoff_counts:** Aggregated counts from expert drivers
- **latest_volume_pickups:** May contain different temporal/spatial granularity
- **Key Difference:** pickup_dropoff_counts includes both pickups AND dropoffs

---

## Data Quality Notes

### Completeness

- **Source Period:** 3 months (July-September)
- **Driver Count:** 50 expert drivers
- **Missing Data:** Zero counts are implicitly represented by absent keys

### Known Limitations

1. **Temporal Coverage:** Limited to weekdays (Monday-Saturday); no Sunday data
2. **Spatial Sparsity:** Only ~3.1% of possible states have recorded events
3. **Seasonal Bias:** Summer months only (July-September)
4. **Sample Size:** Limited to 50 drivers; may not represent full population

### Data Validation

```python
# Check for anomalies
for key, (pickups, dropoffs) in pickup_dropoff_counts.items():
    # Validate grid bounds
    assert 0 <= key[0] < 48, f"Invalid x_grid: {key[0]}"
    assert 0 <= key[1] < 90, f"Invalid y_grid: {key[1]}"
    
    # Validate time bounds (1-indexed)
    assert 1 <= key[2] <= 288, f"Invalid time_bucket: {key[2]}"
    
    # Validate day bounds (1-indexed, Mon-Sat)
    assert 1 <= key[3] <= 6, f"Invalid day_of_week: {key[3]}"
    
    # Validate counts are non-negative
    assert pickups >= 0, f"Negative pickup count: {pickups}"
    assert dropoffs >= 0, f"Negative dropoff count: {dropoffs}"
```

---

## Version History

- **Current Version:** As described in this document
- **Source Files:** 
  - `taxi_record_07_50drivers.pkl`
  - `taxi_record_08_50drivers.pkl`
  - `taxi_record_09_50drivers.pkl`
- **Processing:** Aggregated pickup and dropoff events by spatiotemporal state

---

## Related Resources

- **Explorer Notebook:** `data_dictionary/explorers/pickup_dropoff_counts_explorer.ipynb`
- **Processing Module:** `pickup_dropoff_counts/` (processor and tools)
- **Related Datasets:**
  - `all_trajs.pkl` - Expert driver trajectories
  - `latest_volume_pickups.pkl` - Alternative pickup volume data
  - `latest_traffic.pkl` - Traffic condition data

---

## Contact & Support

For questions about this dataset or to report data quality issues, please refer to the main FAMAIL project documentation or contact the data processing team.
