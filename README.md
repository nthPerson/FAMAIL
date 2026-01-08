# Pickup/Dropoff Counts Processor

A data processing tool that counts pickup and dropoff events from raw taxi GPS trajectory data, organized by spatiotemporal keys `(x_grid, y_grid, time_bucket, day_of_week)`.

## Overview

This tool processes raw taxi GPS data from Shenzhen to:
1. **Count pickup events** - when a taxi picks up a passenger (passenger indicator 0 → 1)
2. **Count dropoff events** - when a taxi drops off a passenger (passenger indicator 1 → 0)
3. **Aggregate by spatiotemporal key** - grid cell location + time bucket + day of week

## Quick Start

### 1. Install Dependencies

```bash
cd pickup_dropoff_counts
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### 3. Alternative: Run via Command Line

```bash
python processor.py --output output/pickup_dropoff_counts.pkl
```

## Input Data

### Required Raw Data Files

Place these files in the `raw_data/` directory:
- `taxi_record_07_50drivers.pkl` - July 2016 GPS data
- `taxi_record_08_50drivers.pkl` - August 2016 GPS data
- `taxi_record_09_50drivers.pkl` - September 2016 GPS data

### Record Format

Each record contains:
| Index | Field | Description |
|-------|-------|-------------|
| 0 | `plate_id` | Taxi identifier |
| 1 | `latitude` | GPS latitude (degrees) |
| 2 | `longitude` | GPS longitude (degrees) |
| 3 | `seconds` | Seconds since midnight |
| 4 | `passenger_indicator` | 0 = empty, 1 = passenger onboard |
| 5 | `timestamp` | Date and time string |

## Output Format

### File: `pickup_dropoff_counts.pkl`

Python dictionary serialized via pickle:

```python
{
    (x_grid, y_grid, time_bucket, day): (pickup_count, dropoff_count),
    (12, 45, 96, 1): (3, 2),   # 3 pickups, 2 dropoffs
    (8, 22, 150, 3): (5, 4),
    ...
}
```

### Key Structure
- `x_grid`: Latitude grid index (0-based)
- `y_grid`: Longitude grid index (0-based)
- `time_bucket`: 5-minute time slot [0, 287]
- `day`: Day of week (Monday=1 ... Saturday=6)

### Value Structure
- `pickup_count`: Number of pickups in this cell/time/day
- `dropoff_count`: Number of dropoffs in this cell/time/day

## Quantization Specification

### Spatial Quantization
- **Grid size**: 0.01 degrees (~1.1 km)
- **Method**: `numpy.digitize` with data-driven bounds
- **Bounds**: Computed from combined dataset before quantization

### Temporal Quantization
- **Time buckets**: 5-minute intervals → 288 slots per day
- **Day indexing**: Monday=1, Tuesday=2, ..., Saturday=6
- **Sunday**: Excluded from analysis

### Event Detection
- **Pickup**: `passenger_indicator` transitions 0 → 1
- **Dropoff**: `passenger_indicator` transitions 1 → 0

## Dashboard Features

### 📊 Process Data Tab
- Verify input file status
- Configure processing parameters
- Run data processing with progress tracking
- View processing statistics

### ✅ Validation Tab
- Compare generated counts against existing `latest_volume_pickups.pkl`
- View match rates and correlation
- Analyze top discrepancies
- Identify potential causes of mismatch

### 📈 Visualizations Tab
- **Spatial heatmaps**: Geographic distribution of pickups/dropoffs
- **Temporal patterns**: Time-of-day trends
- **Daily patterns**: Day-of-week variations
- **Distributions**: Count histograms and statistics

## Validation Strategy

The tool compares generated pickup counts against the existing `latest_volume_pickups.pkl` dataset:

1. **Key matching**: Find common spatiotemporal keys
2. **Count comparison**: Compare pickup values for matching keys
3. **Metrics computed**:
   - Exact match rate
   - Close match rate (within 20%)
   - Pearson correlation
4. **Discrepancy analysis**: Identify and explain mismatches

### Success Criteria
- Correlation > 0.9
- Exact match rate > 80%

### Common Causes of Discrepancies
- Different global bounds (data scope)
- Off-by-one errors in bin indexing
- Edge case handling differences
- Different Sunday filtering logic

## File Structure

```
pickup_dropoff_counts/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── processor.py              # Core processing logic
├── app.py                    # Streamlit dashboard
├── docs/
│   └── data_quantization_plan.md  # Detailed specification
└── output/
    └── pickup_dropoff_counts.pkl  # Generated output
```

## API Reference

### `ProcessingConfig`
Configuration dataclass for processing parameters.

```python
from processor import ProcessingConfig

config = ProcessingConfig(
    grid_size=0.01,           # Grid cell size in degrees
    time_interval=5,          # Time bin size in minutes
    exclude_sunday=True,      # Whether to exclude Sunday
    raw_data_dir=Path("..."), # Path to raw data
    output_dir=Path("...")    # Path for output
)
```

### `process_data(config, progress_callback)`
Main processing function.

```python
from processor import process_data, ProcessingConfig

config = ProcessingConfig()
counts, stats = process_data(config)

# counts: Dict[(x, y, t, day)] → (pickups, dropoffs)
# stats: ProcessingStats with metadata
```

### `validate_against_existing(generated, existing_path)`
Compare against existing dataset.

```python
from processor import validate_against_existing

result = validate_against_existing(counts, Path("latest_volume_pickups.pkl"))
print(f"Correlation: {result.correlation}")
print(f"Match rate: {result.exact_pickup_matches / result.matching_keys}")
```

## See Also

- [Data Quantization Plan](docs/data_quantization_plan.md) - Detailed specification
- [Latest Volume Pickups Data Dictionary](../data_dictionary/dictionaries/latest_volume_pickups_data_dictionary.md) - Reference dataset documentation
