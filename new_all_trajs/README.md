# New All Trajs Dataset Generator

A tool to regenerate the `all_trajs.pkl` dataset from raw GPS trajectory data using consistent quantization and feature generation logic.

## Overview

This tool processes raw taxi GPS data and creates state vectors that include:
- Spatial position (grid coordinates)
- Temporal features (time bucket, day of week)
- POI distances (Manhattan distance to 21 points of interest)
- Traffic features (speeds, waiting times)
- Volume features (pickup counts, traffic volumes)
- Movement actions (direction of travel)

## Installation

```bash
cd new_all_trajs
pip install -r requirements.txt
```

## Usage

### Streamlit Dashboard

Run the interactive dashboard:

```bash
cd /path/to/FAMAIL
streamlit run new_all_trajs/app.py
```

The dashboard provides:
- **Generate Tab**: Configure and generate new datasets
- **Analyze Tab**: Load and analyze existing datasets

### Programmatic Usage

```python
from pathlib import Path
from new_all_trajs.config import ProcessingConfig
from new_all_trajs.processor import process_data, save_output

# Configure processing
config = ProcessingConfig(
    raw_data_dir=Path("raw_data"),
    source_data_dir=Path("source_data"),
    output_dir=Path("new_all_trajs/output"),
    output_filename="new_all_trajs.pkl"
)

# Process data
all_trajs, stats = process_data(config)

# Save output
save_output(all_trajs, config.output_dir / config.output_filename)

print(f"Generated {stats.total_trajectories} trajectories from {stats.total_drivers} drivers")
```

## Output Format

The generated dataset is a Python dictionary with structure:

```
Dict[int, List[List[126-element vectors]]]
  │      │    │    └── State vector (126 floats)
  │      │    └── Trajectory (list of states)
  │      └── List of trajectories for this driver
  └── Driver ID (integer key)
```

### State Vector Schema (126 elements)

| Index Range | Field | Description |
|-------------|-------|-------------|
| 0 | x_grid | Longitude grid index |
| 1 | y_grid | Latitude grid index |
| 2 | time_bucket | Time-of-day slot [1-288] |
| 3 | day_index | Day-of-week [1-6] |
| 4-24 | poi_distances | Manhattan distances to 21 POIs |
| 25-49 | pickup_counts | Normalized pickup counts (5×5 window) |
| 50-74 | traffic_volumes | Normalized traffic volumes (5×5 window) |
| 75-99 | traffic_speeds | Normalized traffic speeds (5×5 window) |
| 100-124 | traffic_waits | Normalized waiting times (5×5 window) |
| 125 | action_code | Movement action (0-9) |

### Action Codes

| Code | Direction |
|------|-----------|
| 0 | North |
| 1 | Northeast |
| 2 | East |
| 3 | Southeast |
| 4 | South |
| 5 | Southwest |
| 6 | West |
| 7 | Northwest |
| 8 | Stay |
| 9 | Stop |

## Data Sources

### Required Files

**Raw Data** (`raw_data/`):
- `taxi_record_07_50drivers.pkl` - July GPS records
- `taxi_record_08_50drivers.pkl` - August GPS records
- `taxi_record_09_50drivers.pkl` - September GPS records

**Source Data** (`source_data/`):
- `latest_traffic.pkl` - Traffic speeds and waiting times by (x, y, t, day)
- `latest_volume_pickups.pkl` - Pickup counts and traffic volumes by (x, y, t, day)
- `train_airport.pkl` - POI locations (21 points of interest)

## Quantization Logic

The tool uses consistent quantization with other FAMAIL datasets:

- **Grid Size**: 0.01 degrees (~1.1 km at Shenzhen's latitude)
- **Time Interval**: 5 minutes (288 buckets per day)
- **Coordinate Offsets**: x_offset=1, y_offset=1, time_offset=1

### Grid Conversion
```python
x_grid = int((longitude - 113.7) / 0.01) + 1
y_grid = int((latitude - 22.45) / 0.01) + 1
```

### Time Conversion
```python
time_bucket = (seconds % 86400) // 300 + 1  # 1-288
```

## Feature Generation

Features are generated following the logic from `net-dis-cGAIL.ipynb`:

1. **POI Distances**: Manhattan distance to each of 21 POIs
2. **Window Features**: For each feature type (pickup, volume, speed, wait), extract values from a 5×5 grid window centered on the current position
3. **Normalization**: Features are z-score normalized using pre-computed statistics from the original dataset
4. **Actions**: Computed from consecutive grid positions

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| grid_size | 0.01 | Grid cell size in degrees |
| time_interval | 5 | Time bucket duration in minutes |
| exclude_sunday | True | Exclude Sunday data (day_index > 6) |
| x_offset | 1 | X coordinate offset for grid indices |
| y_offset | 1 | Y coordinate offset for grid indices |
| time_offset | 1 | Time offset for time bucket indices |

## Normalization Constants

```python
NORMALIZATION_CONSTANTS = {
    "pickup_mean": 1.7413786476868327,
    "pickup_std": 8.689488814143064,
    "volume_mean": 241.16197612732095,
    "volume_std": 864.8135920522847,
    "speed_mean": 0.009106327032150646,
    "speed_std": 0.007797315461703698,
    "wait_mean": 9.21530779817006,
    "wait_std": 20.84113700700525,
}
```

## Project Structure

```
new_all_trajs/
├── __init__.py         # Package initialization
├── app.py              # Streamlit dashboard
├── config.py           # Configuration dataclasses and constants
├── processor.py        # Core processing logic
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── output/             # Generated datasets
```

## Related Documentation

- [All Trajs Data Dictionary](../data_dictionary/dictionaries/all_trajs_data_dictionary.md)
- [net-dis-cGAIL Notebook](../cGAIL_data_and_processing/net-dis-cGAIL.ipynb)
- [Pickup Dropoff Counts Tool](../pickup_dropoff_counts/)
