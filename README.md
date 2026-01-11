# Active Taxis Dataset Generation Tool

This tool generates datasets that count the number of active taxis in an n×n grid neighborhood for each cell and time period. These datasets support the **Spatial Fairness** and **Causal Fairness** objective function terms.

## Overview

An "active taxi" is defined as any taxi that was present (had at least one GPS reading) in the n×n neighborhood surrounding a cell during a given time period.

### Why This Tool?

The Spatial Fairness term requires knowing the number of active taxis in a neighborhood to compute service rates:

$$\text{DSR}_s^p = \frac{\text{pickups}_s^p}{N_s^p \cdot T}$$

Where $N_s^p$ is the number of active taxis in cell $s$ during period $p$. Computing this online during objective function evaluation would be very expensive, so we pre-compute these counts.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Using the Dashboard

```bash
cd active_taxis
streamlit run app.py
```

### Programmatic Usage

```python
from active_taxis import (
    ActiveTaxisConfig,
    generate_active_taxi_counts,
    save_output,
    load_output,
    get_active_taxi_count,
)

# Configure
config = ActiveTaxisConfig(
    neighborhood_size=2,      # 5×5 neighborhood
    period_type='hourly',     # Hourly aggregation
    test_mode=False,          # Full dataset
)

# Generate
counts, stats = generate_active_taxi_counts(config)

# Save
save_output(counts, stats, config, Path('output/active_taxis_5x5_hourly.pkl'))

# Load and use
data, stats, config = load_output(Path('output/active_taxis_5x5_hourly.pkl'))

# Look up count for a specific cell and period
count = get_active_taxi_count(data, x=25, y=45, period_key=(12, 3))  # hour 12, day 3
```

## Configuration Options

### Neighborhood Size

The `neighborhood_size` parameter (k) determines the size of the neighborhood:
- `k=0`: 1×1 (just the cell itself)
- `k=1`: 3×3 neighborhood
- `k=2`: 5×5 neighborhood (default, recommended)
- `k=3`: 7×7 neighborhood

### Period Type

| Type | Description | Key Format | Periods/Day |
|------|-------------|------------|-------------|
| `time_bucket` | 5-minute intervals | `(x, y, bucket, day)` | 288 |
| `hourly` | 1-hour intervals | `(x, y, hour, day)` | 24 |
| `daily` | Full day | `(x, y, day)` | 1 |
| `all` | All time aggregated | `(x, y, 'all')` | 1 |

### Test Mode

For validation and development, use test mode to process a small subset:

```python
config = ActiveTaxisConfig(
    test_mode=True,
    test_sample_size=5,   # Only 5 drivers
    test_days=3,          # Only 3 days per month
)
```

## Output Format

### Key Structure

The output is a dictionary mapping spatiotemporal keys to active taxi counts:

```python
# For hourly period_type:
{
    (x, y, hour, day): count,
    (1, 1, 0, 1): 15,      # Cell (1,1), hour 0, Monday: 15 active taxis
    (1, 1, 0, 2): 14,      # Cell (1,1), hour 0, Tuesday: 14 active taxis
    ...
}

# For daily period_type:
{
    (x, y, day): count,
    (1, 1, 1): 35,         # Cell (1,1), Monday: 35 active taxis
    ...
}
```

### Coordinate System

Coordinates are 1-indexed to match `pickup_dropoff_counts`:
- `x`: 1 to 48 (latitude-based grid)
- `y`: 1 to 90 (longitude-based grid)
- `day`: 1 (Monday) to 6 (Saturday)
- `hour`: 0 to 23
- `time_bucket`: 1 to 288

## Consistency with pickup_dropoff_counts

This tool uses **identical** quantization functions to `pickup_dropoff_counts`:

| Function | Description |
|----------|-------------|
| `gps_to_grid()` | GPS → grid cell conversion |
| `timestamp_to_time_bin()` | Timestamp → 5-min bucket |
| `timestamp_to_day()` | Timestamp → day of week |

This ensures that the active taxi counts align perfectly with pickup/dropoff counts for the same cells and periods.

## Directory Structure

```
active_taxis/
├── __init__.py           # Package exports
├── config.py             # ActiveTaxisConfig dataclass
├── processor.py          # Data loading and quantization
├── generation.py         # Active taxi counting algorithm
├── app.py                # Streamlit dashboard
├── README.md             # This file
├── requirements.txt      # Dependencies
├── docs/                 # Additional documentation
│   └── algorithm.md      # Detailed algorithm description
└── output/               # Generated datasets
    └── .gitkeep
```

## Algorithm Overview

1. **Load raw GPS data** from `taxi_record_XX_50drivers.pkl`
2. **Apply quantization** (same as pickup_dropoff_counts)
3. **Build presence index**: For each period, track which taxis were present in each cell
4. **Count neighborhoods**: For each (cell, period), count unique taxis in the n×n neighborhood

### Performance

| Mode | Records | Time (approx) |
|------|---------|---------------|
| Test (5 drivers, 3 days) | ~50K | ~5s |
| Full (50 drivers, July) | ~2.4M | ~3-5 min |
| Full (3 months) | ~7M | ~10-15 min |

## Integration with Spatial Fairness

The Spatial Fairness term uses this data as follows:

```python
from spatial_fairness import SpatialFairnessTerm, SpatialFairnessConfig
from active_taxis import load_output

# Load precomputed active taxi counts
active_taxi_data, _, _ = load_output('output/active_taxis_5x5_hourly.pkl')

# Configure spatial fairness to use active taxi data
config = SpatialFairnessConfig(
    period_type='hourly',
    use_active_taxi_data=True,
)

# The term will use active_taxi_data instead of a fixed num_taxis
term = SpatialFairnessTerm(config)
result = term.compute(
    {},
    {
        'pickup_dropoff_counts': pickup_data,
        'active_taxis': active_taxi_data,
    }
)
```

## References

- FAMAIL Objective Function Development Plan
- Spatial Fairness Term Specification
- pickup_dropoff_counts Tool Documentation
