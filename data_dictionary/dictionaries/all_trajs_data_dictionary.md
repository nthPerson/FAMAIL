# all_trajs.pkl Data Dictionary

## Overview

The `all_trajs.pkl` dataset contains feature-augmented taxi driver trajectory data for 50 expert drivers. This is the primary dataset used for training and evaluation in the cGAIL (conditional Generative Adversarial Imitation Learning) framework and for future trajectory modification in the FAMAIL project. Each trajectory represents a sequence of states capturing a driver's movement through a discretized grid map, along with rich contextual features derived from traffic conditions, pickup patterns, and proximity to points of interest.

**File Format:** Python pickle file (`.pkl`)  
**Data Type:** Nested dictionary  
**Approximate Size:** ~380 MB (JSON equivalent)

---

## Data Structure

### Hierarchy

```
all_trajs.pkl
├── driver_key_0 (int or str)
│   ├── trajectory_0 (list of states)
│   │   ├── state_0 (list of 126 elements)
│   │   ├── state_1 (list of 126 elements)
│   │   └── ...
│   ├── trajectory_1
│   └── ...
├── driver_key_1
│   └── ...
└── ... (50 drivers total)
```

### Loading the Data

```python
import pickle

# Load the dataset
all_trajs = pickle.load(open('path/to/all_trajs.pkl', 'rb'))

# Get list of driver keys
driver_keys = list(all_trajs.keys())
print(f'Number of drivers: {len(driver_keys)}')  # Output: 50

# Access trajectories for a specific driver
driver_0_trajectories = all_trajs[driver_keys[0]]

# Access a specific trajectory
trajectory = all_trajs[driver_keys[0]][0]

# Access a specific state within a trajectory
state = all_trajs[driver_keys[0]][0][0]
print(f'State vector length: {len(state)}')  # Output: 126
```

---

## Top-Level Structure

| Property | Type | Description |
|----------|------|-------------|
| Keys | `int` or `str` | Driver identifiers (50 unique keys) |
| Values | `list[list[list]]` | List of trajectories for each driver |

### Trajectory Properties

- **Ordering:** Trajectories within each driver's list are sorted chronologically by both time-of-day (index 2) and day-of-week (index 3)
- **Length:** Variable number of trajectories per driver
- **Structure:** Each trajectory is a list of consecutive states

---

## State Vector Schema

Each state is a **126-element list** (indices 0-125) representing the complete feature set for a single timestep in a trajectory.

### Index Reference Table

| Index Range | Field Name | Data Type | Description |
|-------------|------------|-----------|-------------|
| 0 | `x_grid` | `int` | Grid index for longitude position |
| 1 | `y_grid` | `int` | Grid index for latitude position |
| 2 | `time_bucket` | `int` | Discretized time-of-day slot ∈ [0, 287] |
| 3 | `day_index` | `int` | Day-of-week indicator |
| 4–24 | `poi_manhattan_distance` | `float` | Manhattan distances to 21 points of interest |
| 25–49 | `pickup_count_norm` | `float` | Normalized pickup counts (5×5 window) |
| 50–74 | `traffic_volume_norm` | `float` | Normalized traffic volumes (5×5 window) |
| 75–99 | `traffic_speed_norm` | `float` | Normalized traffic speeds (5×5 window) |
| 100–124 | `traffic_wait_norm` | `float` | Normalized traffic waiting times (5×5 window) |
| 125 | `action_code` | `int` | Movement action label (0-9) |

---

## Detailed Field Descriptions

### Base Location Fields (Indices 0-3)

#### Index 0: `x_grid`
- **Type:** Integer
- **Description:** Horizontal grid cell index representing the longitude position on a discretized map
- **Range:** Varies based on grid resolution (typically 0-49)

#### Index 1: `y_grid`
- **Type:** Integer
- **Description:** Vertical grid cell index representing the latitude position on a discretized map
- **Range:** Varies based on grid resolution (typically 0-89)

#### Index 2: `time_bucket`
- **Type:** Integer
- **Description:** Discretized time-of-day slot. With 288 buckets, each represents a 5-minute interval (24 hours × 60 minutes / 5 = 288)
- **Range:** [0, 287]
- **Formula:** `time_bucket = (hour * 60 + minute) // 5`

#### Index 3: `day_index`
- **Type:** Integer
- **Description:** Day-of-week indicator
- **Range:** [0, 6] or [1, 7] depending on encoding
- **Note:** Consult source data for specific day mapping

---

### Points of Interest Distance Features (Indices 4-24)

**21 features total** representing Manhattan distances to key locations.

#### Source Data
- Derived from: `train_airport.pkl`
- Contains distances to train stations, airports, and major commercial areas in Shenzhen

#### Known Points of Interest (21 total)
1. 深圳北站 (Shenzhen North Railway Station)
2. 深圳东站 (Shenzhen East Railway Station)
3. 深圳站 (Shenzhen Railway Station)
4. 福田站 (Futian Station)
5. 宝安机场 (Bao'an Airport)
6. 深圳西站 (Shenzhen West Railway Station)
7. coco park
8. Mixc
9. coastal city
10. kk mall
11. *...and 11 additional POIs*

#### Usage
```python
state = all_trajs[driver_key][traj_idx][state_idx]
poi_distances = state[4:25]  # 21 distances
```

---

### 5×5 Window Feature Blocks (Indices 25-124)

The following four feature blocks each contain **25 values** representing a 5×5 spatial window centered on the current `(x_grid, y_grid)` position.

#### Window Layout
```
x_range = [x-2, x-1, x, x+1, x+2]
y_range = [y-2, y-1, y, y+1, y+2]

Order: Row-major over x_range × y_range
Index mapping: i = (x_offset + 2) * 5 + (y_offset + 2)
```

#### Indices 25-49: `pickup_count_norm`
- **Type:** Float (normalized)
- **Description:** Normalized taxi pickup counts in each cell of the 5×5 window
- **Source:** `latest_volume_pickups.pkl`, value index [0]
- **Default normalization:** baseline=-1.7411, scale=8.6891 (when missing)

#### Indices 50-74: `traffic_volume_norm`
- **Type:** Float (normalized)
- **Description:** Normalized traffic/trip volumes in each cell of the 5×5 window
- **Source:** `latest_volume_pickups.pkl`, value index [1]
- **Default normalization:** baseline=-241.1649, scale=864.8101 (when missing)

#### Indices 75-99: `traffic_speed_norm`
- **Type:** Float (normalized)
- **Description:** Normalized traffic speeds in each cell of the 5×5 window
- **Source:** `latest_traffic.pkl`, value index [0]
- **Default normalization:** baseline=-0.009096, scale=0.0077867 (when missing)

#### Indices 100-124: `traffic_wait_norm`
- **Type:** Float (normalized)
- **Description:** Normalized traffic waiting times in each cell of the 5×5 window
- **Source:** `latest_traffic.pkl`, value index [1]
- **Default normalization:** baseline=-9.2149, scale=20.8396 (when missing)

---

### Action Code (Index 125)

#### Type
Integer ∈ [0, 9]

#### Action Code Mapping

| Code | Movement Description | Δx | Δy |
|------|---------------------|----|----|
| 0 | Move north (y increases) | 0 | +1 |
| 1 | Move northeast | +1 | +1 |
| 2 | Move east (x increases) | +1 | 0 |
| 3 | Move southeast | +1 | -1 |
| 4 | Move south (y decreases) | 0 | -1 |
| 5 | Move southwest | -1 | -1 |
| 6 | Move west (x decreases) | -1 | 0 |
| 7 | Move northwest | -1 | +1 |
| 8 | Stay in place | 0 | 0 |
| 9 | Stop (terminal action) | N/A | N/A |

#### Notes
- Action code 9 (stop) is triggered when current or next position is `(0, 0)`
- Actions 0-7 represent 8-directional movement on the grid
- Action 8 represents the driver staying stationary

---

## Common Access Patterns

### Iterate Over All Trajectories
```python
for driver_key in all_trajs:
    for traj_idx, trajectory in enumerate(all_trajs[driver_key]):
        for state_idx, state in enumerate(trajectory):
            x, y = state[0], state[1]
            time_bucket, day = state[2], state[3]
            action = state[125]
```

### Extract Feature Subsets
```python
state = all_trajs[driver_key][traj_idx][state_idx]

# Base features
location = (state[0], state[1])           # (x_grid, y_grid)
temporal = (state[2], state[3])           # (time_bucket, day_index)

# POI distances
poi_distances = state[4:25]               # 21 values

# 5x5 window features
pickup_counts = state[25:50]              # 25 values
traffic_volumes = state[50:75]            # 25 values
traffic_speeds = state[75:100]            # 25 values
traffic_waits = state[100:125]            # 25 values

# Action
action = state[125]                       # 1 value

# All features (excluding action)
features = state[:125]                    # 125 values
```

### Count Total Trajectories
```python
total_trajectories = sum(len(all_trajs[key]) for key in all_trajs)
```

### Build State Key for Lookup
```python
def state_to_key(state):
    """Convert state to lookup key for traffic/volume dictionaries."""
    return (state[0], state[1], state[2], state[3])
```

---

## Data Provenance

### Source Files
| Feature | Source File | Access Pattern |
|---------|-------------|----------------|
| POI distances | `train_airport.pkl` | Manhattan distance calculation |
| Pickup counts | `latest_volume_pickups.pkl` | `volume[(x,y,t,day)][0]` |
| Traffic volumes | `latest_volume_pickups.pkl` | `volume[(x,y,t,day)][1]` |
| Traffic speeds | `latest_traffic.pkl` | `traffic[(x,y,t,day)][0]` |
| Traffic waits | `latest_traffic.pkl` | `traffic[(x,y,t,day)][1]` |
| Action codes | Computed on-the-fly | `judging_action(current_xy, next_xy)` |

### Processing Pipeline
1. Raw trajectory data with `[driver_id, x_grid, y_grid, time_bucket, day_index]`
2. Driver ID stripped, base state becomes `[x, y, t, day]`
3. POI distances computed and appended
4. 5×5 window features computed and normalized
5. Action codes derived from consecutive position changes

---

## Important Notes

1. **Feature Indices 0-124** are deterministic features; **index 125** is the supervised action label
2. **Normalization constants** are embedded in the processing pipeline and derived from dataset statistics
3. **Missing values** in source dictionaries trigger default normalization values
4. **Grid coordinates** `(0, 0)` are treated as a special "stop" indicator
5. **Trajectory chronology** is preserved within each driver's trajectory list

---

## Related Datasets

- `latest_traffic.pkl` - Source for traffic speed and waiting time features
- `latest_volume_pickups.pkl` - Source for pickup count and traffic volume features
- `train_airport.pkl` - Source for POI location data
- `cgail_trajs2.pkl` - Raw trajectory data before feature augmentation (not recommended for FAMAIL analysis)
