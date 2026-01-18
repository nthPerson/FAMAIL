# New All Trajs Generator

A tool for recreating the `all_trajs.pkl` dataset in two distinct steps:

1. **Step 1**: Extract passenger-seeking trajectories from raw taxi GPS data
2. **Step 2**: Generate state features using cGAIL feature generation logic

## Overview

The original `all_trajs.pkl` dataset contains 126-element state vectors for taxi passenger-seeking trajectories. This tool recreates that dataset from raw GPS data using the same quantization and feature generation logic as other project components.

### State Vector Structure (126 elements)

| Index Range | Feature | Description |
|-------------|---------|-------------|
| 0 | x_grid | Grid cell x-coordinate |
| 1 | y_grid | Grid cell y-coordinate |
| 2 | time_bucket | 5-minute time bin (0-287) |
| 3 | day_index | Day of week (1-6, excludes Sunday) |
| 4-24 | POI distances | Manhattan distances to 21 POIs |
| 25-49 | pickup_count | Normalized pickup counts (5×5 window) |
| 50-74 | traffic_volume | Normalized traffic volumes (5×5 window) |
| 75-99 | traffic_speed | Normalized traffic speeds (5×5 window) |
| 100-124 | traffic_wait | Normalized traffic wait times (5×5 window) |
| 125 | action_code | Movement action (0-9) |

### Action Codes

| Code | Direction |
|------|-----------|
| 0 | North (y+) |
| 1 | Northeast |
| 2 | East (x+) |
| 3 | Southeast |
| 4 | South (y-) |
| 5 | Southwest |
| 6 | West (x-) |
| 7 | Northwest |
| 8 | Stay |
| 9 | Stop (terminal) |

## Installation

```bash
cd new_all_trajs
pip install -r requirements.txt
```

## Usage

### Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard provides three tabs:
- **Generate Passenger Seeking**: Run Step 1 extraction
- **Generate State Features**: Run Step 2 feature generation
- **Analyze**: Examine output datasets

### Command Line

**Step 1: Extract Passenger-Seeking Trajectories**

```bash
python step1_processor.py --output output/passenger_seeking_trajs.pkl --min-length 2
```

**Step 2: Generate State Features**

```bash
python step2_processor.py --input output/passenger_seeking_trajs.pkl --output output/new_all_trajs.pkl
```

## Data Requirements

### Step 1 Input Files
Located in `raw_data/`:
- `taxi_record_07_50drivers.pkl`
- `taxi_record_08_50drivers.pkl`
- `taxi_record_09_50drivers.pkl`

### Step 2 Feature Data Files
Located in `source_data/`:
- `latest_traffic.pkl` - Traffic speed and wait times
- `latest_volume_pickups.pkl` - Pickup counts and traffic volumes
- `train_airport.pkl` - POI locations (21 places)

## Output Format

### Step 1 Output
```python
{
    driver_index: [  # int key (0, 1, 2, ...)
        [  # trajectory list
            [x, y, time, day],  # state
            [x, y, time, day],
            ...
        ],
        [...],  # another trajectory
    ],
    ...
}
```

### Step 2 Output (all_trajs.pkl format)
```python
{
    driver_index: [  # int key (0, 1, 2, ...)
        [  # trajectory list
            [x, y, time, day, poi_1, ..., poi_21, pickup_1, ..., action],  # 126 elements
            [...],
            ...
        ],
        [...],
    ],
    ...
}
```

## Quantization Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| grid_size | 0.01° | Grid cell size in degrees |
| time_interval | 5 min | Time bin size |
| x_grid_offset | 1 | Added to x indices |
| y_grid_offset | 1 | Added to y indices |
| exclude_sunday | True | Filter Sunday records |

## Normalization Constants

From the original cGAIL notebook:

| Feature | Baseline | Scale |
|---------|----------|-------|
| pickup_count | 1.7411 | 8.6892 |
| traffic_volume | 241.165 | 864.81 |
| traffic_speed | 0.00910 | 0.00779 |
| traffic_wait | 9.2149 | 20.84 |

Formula: `normalized = (value - baseline) / scale`

## File Structure

```
new_all_trajs/
├── __init__.py           # Package initialization
├── app.py                # Streamlit dashboard
├── config.py             # Configuration classes and constants
├── step1_processor.py    # Step 1: Trajectory extraction
├── step2_processor.py    # Step 2: Feature generation
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── output/               # Generated datasets
```

## Alignment with Other Tools

This tool uses the **same quantization logic** as:
- `active_taxis/processor.py`
- `pickup_dropoff_counts/processor.py`

This ensures grid cell consistency across all derived datasets.
