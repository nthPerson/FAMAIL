## Objective

Count the number of taxi pickup and dropoff events for each spatiotemporal key `(x_grid, y_grid, time, day)` using the same quantization method applied to the existing processed datasets.

---

## Input Data

### Source Files

- `taxi_record_07_50drivers.pkl`: Raw GPS data from 50 drivers, July 2016
- `taxi_record_08_50drivers.pkl`: Raw GPS data from 50 drivers, August 2016
- `taxi_record_09_50drivers.pkl`: Raw GPS data from 50 drivers, September 2016

### Record Attributes (using an example record)

| Index | Example Value | Interpretation |
| --- | --- | --- |
| 0 | `'Ã§Â²Â¤SW794X'` | `plate_id` (taxi identifier, encoding issue on Chinese characters) |
| 1 | `22.831083` | `latitude` (matches Shenzhen's lat range 22.44°–22.87°) |
| 2 | `114.158012` | `longitude` (matches Shenzhen's lon range 113.75°–114.65°) |
| 3 | `48` | Seconds since 00:00:00 |
| 4 | `0` | `passenger_indicator` (0 = empty, 1 = passenger onboard) |
| 5 | `'2016-07-01 00:00:48'` | `timestamp` |
- `plate_id`: Unique taxi identifier
- `latitude`: GPS latitude (degrees)
- `longitude`: GPS longitude (degrees)
- `passenger_indicator`: Binary (1 = passenger onboard, 0 = empty)
- `timestamp`: Date and time of record

---

## Output Specification

### Output File

`pickup_dropoff_counts.pkl`

### Output Format

Python dictionary serialized via pickle:

- **Keys:** `(x_grid, y_grid, time, day)` — tuple of four integers
- **Values:** `(pickup_count, dropoff_count)` — tuple of two integers
    - `value[0]`: pickup count for this key
    - `value[1]`: dropoff count for this key

**Example structure:**

```python
{
    (12, 45, 96, 1): (3, 2),   # 3 pickups, 2 dropoffs at grid (12,45), time bin 96, Monday
    (12, 45, 97, 1): (1, 0),
    (8, 22, 150, 3): (5, 4),
    ...
}

```

---

## Quantization Specification

### Spatial Quantization (GPS → Grid)

**Method:** `numpy.digitize` with data-driven bounds

**Parameters:**

- `grid_size = 0.01` (degrees)
- Bounds computed from the **combined dataset** (all three input files):
    - `lat_min = combined_df['latitude'].min()`
    - `lat_max = combined_df['latitude'].max()`
    - `lon_min = combined_df['longitude'].min()`
    - `lon_max = combined_df['longitude'].max()`

**Implementation:**

```python
import numpy as np

def gps_to_grid(lat, lon, lat_min, lat_max, lon_min, lon_max, grid_size=0.01):
    lat_bins = np.arange(lat_min, lat_max, grid_size)
    lon_bins = np.arange(lon_min, lon_max, grid_size)

    x_grid = np.digitize(lat, lat_bins) - 1   # latitude → x_grid
    y_grid = np.digitize(lon, lon_bins) - 1   # longitude → y_grid

    return x_grid, y_grid

```

**Critical:** Bounds must be computed from the **entire combined dataset** before any quantization occurs. Load all three files, concatenate, compute min/max, then apply quantization.

### Temporal Quantization

**Time-of-day binning:**

- 5-minute intervals → 288 slots per day
- `time` ∈ [0, 287]

```python
def timestamp_to_time_bin(timestamp):
    minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
    time_bin = minutes_since_midnight // 5
    return time_bin  # range: [0, 287]
```

**Day-of-week:**

- Monday = 1, Tuesday = 2, ..., Saturday = 6
- Sunday is **excluded** from analysis
- `day` ∈ [1, 6]

```python
def timestamp_to_day(timestamp):
    dow = timestamp.weekday()  # Monday=0, Sunday=6
    if dow == 6:  # Sunday
        return None  # exclude this record
    return dow + 1  # shift to 1-indexed
```

---

## Pickup and Dropoff Detection Logic

A **pickup** occurs when `passenger_indicator` transitions from `0 → 1`.

A **dropoff** occurs when `passenger_indicator` transitions from `1 → 0`.

**Detection method:**

1. Sort records by `(plate_id, timestamp)` (dataset is already sorted chronologically, but this is included for clarity)
2. For each taxi, compute transitions: `transition = passenger_indicator.diff()`
3. `transition == 1` → pickup event at that record's location/time
4. `transition == -1` → dropoff event at that record's location/time

**Edge case:** The first record for each taxi driver/expert and for each day has no prior state; it should be ignored for transition detection (transition will be NaN).

---

## Sample Data (from `taxi_record_0X_50drivers.pkl`)

**Each record contains:**

- Driver ID
- latitude
- longitude
- Seconds since 00:00:00
- Passenger indicator
- Timestamp

**Example Data**

```json
[['Ã§Â²Â¤SW794X', 22.82855, 114.161819, 3, 0, '2016-07-01 00:00:03'],
 ['Ã§Â²Â¤SW794X', 22.82935, 114.160568, 18, 0, '2016-07-01 00:00:18'],
 ['Ã§Â²Â¤SW794X', 22.8302, 114.159264, 33, 0, '2016-07-01 00:00:33'],
 ['Ã§Â²Â¤SW794X', 22.831083, 114.158012, 48, 0, '2016-07-01 00:00:48'],
 ['Ã§Â²Â¤SW794X', 22.831717, 114.157898, 63, 0, '2016-07-01 00:01:03'],
 ['Ã§Â²Â¤SW794X', 22.831734, 114.157898, 78, 0, '2016-07-01 00:01:18'],
 ['Ã§Â²Â¤SW794X', 22.831734, 114.157898, 93, 0, '2016-07-01 00:01:33'],
 ['Ã§Â²Â¤SW794X', 22.831751, 114.157867, 108, 0, '2016-07-01 00:01:48'],
 ['Ã§Â²Â¤SW794X', 22.845383, 114.156349, 498, 0, '2016-07-01 00:08:18'],
 ['Ã§Â²Â¤SW794X', 22.845383, 114.156364, 528, 0, '2016-07-01 00:08:48'],
 ...]
```

## Implementation Steps

1. **Load all three input files** and concatenate into a single DataFrame
2. **Compute global bounds** (`lat_min`, `lat_max`, `lon_min`, `lon_max`) from the combined DataFrame
3. **Parse timestamps** to datetime objects if not already
4. **Filter out Sundays** (`timestamp.weekday() == 6`)
5. **Apply spatial quantization** using global bounds → add `x_grid`, `y_grid` columns
6. **Apply temporal quantization** → add `time`, `day` columns
7. **Sort by** `(plate_id, timestamp)`
8. **Detect transitions:** Group by `plate_id`, compute `passenger_indicator.diff()`
9. **Identify events:**
    - Pickup: rows where `transition == 1`
    - Dropoff: rows where `transition == -1`
10. **Aggregate:**
    - Group pickup events by `(x_grid, y_grid, time, day)`, count
    - Group dropoff events by `(x_grid, y_grid, time, day)`, count
11. **Merge** pickup and dropoff counts into a single dictionary with keys `(x_grid, y_grid, time, day)` and values `(pickup_count, dropoff_count)`
    - If a key has pickups but no dropoffs (or vice versa), use 0 for the missing count
12. **Save** dictionary to `pickup_dropoff_counts.pkl` using pickle

---

## Reference: Original Quantization Code

The following code is from `TrajectoryPreprocessing.ipynb` and represents the quantization approach used for existing processed datasets:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def gps_to_grid(lat, lon, lat_min, lat_max, lon_min, lon_max, grid_size):
    """Convert GPS coordinates to grid cell indices."""
    lat_bins = np.arange(lat_min, lat_max, grid_size)
    lon_bins = np.arange(lon_min, lon_max, grid_size)

    lat_idx = np.digitize(lat, lat_bins) - 1
    lon_idx = np.digitize(lon, lon_bins) - 1

    return lat_idx, lon_idx

def bin_time(timestamp, start_time, interval):
    """Bin timestamps into fixed intervals."""
    delta = timestamp - start_time
    return delta // interval

def split_trajectories(df, time_threshold=timedelta(minutes=5)):
    """Split a trajectory into separate segments if time difference exceeds threshold."""
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['trajectory_id'] = (df['timestamp'].diff() > time_threshold).cumsum()
    return df

def process_gps_data(df, grid_size=0.01, time_interval=5):
    """Process GPS data into grid cells and time bins."""
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_time = df['timestamp'].min()
    interval = timedelta(minutes=time_interval)

    # Split trajectories based on time threshold
    df = split_trajectories(df)

    # Convert GPS to grid and bin times
    df[['grid_x', 'grid_y']] = df.apply(lambda row: gps_to_grid(
        row['latitude'], row['longitude'], lat_min, lat_max, lon_min, lon_max, grid_size
    ), axis=1, result_type='expand')

    df['time_bin'] = df['timestamp'].apply(lambda x: bin_time(x, start_time, interval))

    return df

```

**Note on time binning:** The reference code uses `bin_time()` which bins relative to `start_time` (earliest timestamp in dataset). However, for this task, use **time-of-day binning** (minutes since midnight // 5) to produce `time` ∈ [0, 287], which aligns with the existing processed datasets.

---

## Validation Strategy

Compare generated pickup counts against the existing dataset that contains pickup data keyed by `(x_grid, y_grid, time, day)`.

**Validation steps:**

1. Load existing pickup dataset (`latest_volume_pickups.pkl`)
2. For each key in the existing dataset, compare `existing_pickup_count` vs `generated_pickup_count` using matching keys (see `data_dictionary/latest_volume_pickups_data_dictionary.md` for details)
3. Compute match rate and investigate discrepancies
4. Potential causes of mismatch:
    - Different global bounds (data-driven scope mismatch)
    - Off-by-one errors in bin indexing
    - Edge case handling differences

**Success criteria:** Exact or near-exact match between generated and existing pickup counts.

---

## Additional Notes

- The `split_trajectories` function in the reference code is for trajectory segmentation and is **not relevant** to pickup/dropoff detection
- Ensure consistent column naming when loading pickle files—verify actual column names in the raw data
- If memory is constrained, compute global bounds in a first pass, then process files individually in a second pass