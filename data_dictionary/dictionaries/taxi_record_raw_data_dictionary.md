# Raw GPS Taxi Data Dictionary

## Overview

The raw GPS taxi datasets contain unprocessed GPS trajectory data for 50 expert taxi drivers in Shenzhen, China, collected across three consecutive months in 2016. These datasets serve as the foundational source data for the FAMAIL project and are used to generate processed datasets including `pickup_dropoff_counts`, `latest_volume_pickups`, and feature-augmented trajectories in `all_trajs.pkl`.

**Dataset Files:**
- `taxi_record_07_50drivers.pkl` - July 2016 GPS data
- `taxi_record_08_50drivers.pkl` - August 2016 GPS data  
- `taxi_record_09_50drivers.pkl` - September 2016 GPS data

**File Format:** Python pickle file (`.pkl`)  
**Data Type:** Nested dictionary  
**Approximate Size:** Varies by month (~100-200 MB each)

---

## Data Structure

### Hierarchy

```
taxi_record_{month}_50drivers.pkl
├── driver_key_0 (str or bytes)
│   ├── record_0 (list of 6 elements)
│   ├── record_1 (list of 6 elements)
│   ├── record_2 (list of 6 elements)
│   └── ... (variable number of records)
├── driver_key_1
│   └── ...
└── ... (50 drivers total)
```

### Loading the Data

```python
import pickle

# Load a single month's dataset
with open('taxi_record_07_50drivers.pkl', 'rb') as f:
    data = pickle.load(f)

# Get list of driver keys
driver_keys = list(data.keys())
print(f'Number of drivers: {len(driver_keys)}')  # Output: 50

# Access records for a specific driver
driver_records = data[driver_keys[0]]
print(f'Number of records for driver 0: {len(driver_records)}')

# Access a specific GPS record
record = data[driver_keys[0]][0]
print(f'Record structure: {record}')
# Example output: ['粤SW794X', 22.82855, 114.161819, 3, 0, '2016-07-01 00:00:03']
```

---

## Top-Level Structure

| Property | Type | Description |
|----------|------|-------------|
| Keys | `str` or `bytes` | Taxi plate identifiers (50 unique drivers) |
| Values | `list[list]` | List of GPS trajectory records for each driver |

### Driver-Level Properties

- **Number of drivers:** 50 (consistent across all three datasets)
- **Ordering:** Records within each driver's list are chronologically ordered by timestamp
- **Length:** Variable number of records per driver (typically 10,000 - 50,000+ records per month)
- **Structure:** Each driver's data is a flat list of individual GPS records

---

## GPS Record Schema

Each GPS record is a **6-element list** representing a single GPS reading at a specific moment in time.

### Record Structure

| Index | Field Name | Data Type | Description |
|-------|------------|-----------|-------------|
| 0 | `plate_id` | `str` or `bytes` | Taxi vehicle identifier (e.g., '粤SW794X') |
| 1 | `latitude` | `float` | GPS latitude coordinate in degrees |
| 2 | `longitude` | `float` | GPS longitude coordinate in degrees |
| 3 | `seconds` | `int` | Seconds elapsed since midnight (0-86399) |
| 4 | `passenger_indicator` | `int` | Occupancy status: 0 = empty, 1 = passenger onboard |
| 5 | `timestamp` | `str` | Full timestamp in format 'YYYY-MM-DD HH:MM:SS' |

---

## Detailed Field Descriptions

### Index 0: `plate_id`
- **Type:** String or bytes (encoded)
- **Description:** Unique identifier for the taxi vehicle
- **Format:** Chinese characters followed by alphanumeric (e.g., '粤SW794X')
- **Encoding Note:** May appear as byte strings (e.g., 'Ã§Â²Â¤SW794X') depending on encoding
- **Consistency:** Same plate ID appears consistently within a single driver's records

### Index 1: `latitude`
- **Type:** Float
- **Description:** GPS latitude coordinate in decimal degrees
- **Range:** Approximately 22.45 - 22.86 (Shenzhen area)
- **Precision:** Typically 6 decimal places (~0.1 meter accuracy)
- **Coordinate System:** WGS84 (standard GPS coordinate system)

### Index 2: `longitude`
- **Type:** Float
- **Description:** GPS longitude coordinate in decimal degrees
- **Range:** Approximately 113.75 - 114.62 (Shenzhen area)
- **Precision:** Typically 6 decimal places (~0.1 meter accuracy)
- **Coordinate System:** WGS84 (standard GPS coordinate system)

### Index 3: `seconds`
- **Type:** Integer
- **Description:** Time of day represented as seconds elapsed since midnight
- **Range:** 0 - 86399 (0 = 00:00:00, 86399 = 23:59:59)
- **Usage:** Can be converted to time buckets for temporal analysis
  - 5-minute buckets: `time_bucket = seconds // 300` (288 buckets per day)
  - Hourly buckets: `hour = seconds // 3600` (24 buckets per day)
- **Conversion to time:**
  ```python
  hours = seconds // 3600
  minutes = (seconds % 3600) // 60
  secs = seconds % 60
  time_str = f"{hours:02d}:{minutes:02d}:{secs:02d}"
  ```

### Index 4: `passenger_indicator`
- **Type:** Integer (binary)
- **Description:** Indicates whether taxi is occupied by a passenger
- **Values:**
  - `0` = Empty taxi (no passenger)
  - `1` = Occupied taxi (passenger onboard)
- **Usage:** Used to detect pickup and dropoff events
  - **Pickup event:** Transition from 0 → 1 (consecutive records)
  - **Dropoff event:** Transition from 1 → 0 (consecutive records)
- **Distribution:** Varies by time of day and driver behavior

### Index 5: `timestamp`
- **Type:** String
- **Description:** Full date and time of the GPS reading
- **Format:** `'YYYY-MM-DD HH:MM:SS'`
- **Examples:**
  - `'2016-07-01 00:00:03'`
  - `'2016-08-15 14:23:47'`
  - `'2016-09-30 23:59:58'`
- **Parsing:**
  ```python
  from datetime import datetime
  dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
  ```
- **Timezone:** Local time (Shenzhen, China = UTC+8)

---

## Temporal Coverage

### July 2016 Dataset (`taxi_record_07_50drivers.pkl`)

| Property | Value |
|----------|-------|
| **Month** | July 2016 |
| **Date Range** | 2016-07-01 to 2016-07-29 |
| **Days of Week** | Monday - Friday (weekdays only, no weekends) |
| **Unique Dates** | 21 dates with data |
| **ISO Week Numbers** | Weeks 26, 27, 28, 29, 30 (5 weeks) |
| **Total GPS Records** | 2,695,153 records (across all 50 drivers) |
| **Time Coverage** | 24-hour data (00:00:00 - 23:59:59) |
| **Records per Driver** | Mean: 53,903 / Std Dev: 26,301 / Min: 4,757 / Max: 128,570 |
| **Pickup Events** | 128,133 |
| **Dropoff Events** | 128,085 |

**Week-by-Week Breakdown:**
- Week 26: Jul 01 (Fri)
- Week 27: Jul 04-08 (Mon-Fri)
- Week 28: Jul 11-15 (Mon-Fri)
- Week 29: Jul 18-22 (Mon-Fri)
- Week 30: Jul 25-29 (Mon-Fri)

**Day of Week Distribution:**
- Monday: 485,264 records (18.0%)
- Tuesday: 461,643 records (17.1%)
- Wednesday: 521,816 records (19.4%)
- Thursday: 542,906 records (20.1%)
- Friday: 683,524 records (25.4%)

### August 2016 Dataset (`taxi_record_08_50drivers.pkl`)

| Property | Value |
|----------|-------|
| **Month** | August 2016 |
| **Date Range** | 2016-08-01 to 2016-08-31 |
| **Days of Week** | Monday - Friday (weekdays only, no weekends) |
| **Unique Dates** | 23 dates with data |
| **ISO Week Numbers** | Weeks 31, 32, 33, 34, 35 (5 weeks) |
| **Total GPS Records** | 2,952,403 records (across all 50 drivers) |
| **Time Coverage** | 24-hour data (00:00:00 - 23:59:59) |
| **Records per Driver** | Mean: 59,048 / Std Dev: 27,349 / Min: 18,288 / Max: 119,227 |
| **Pickup Events** | 116,829 |
| **Dropoff Events** | 116,808 |

**Week-by-Week Breakdown:**
- Week 31: Aug 01-05 (Mon-Fri)
- Week 32: Aug 08-12 (Mon-Fri)
- Week 33: Aug 15-19 (Mon-Fri)
- Week 34: Aug 22-26 (Mon-Fri)
- Week 35: Aug 29-31 (Mon-Wed)

**Day of Week Distribution:**
- Monday: 615,703 records (20.9%)
- Tuesday: 637,187 records (21.6%)
- Wednesday: 648,214 records (22.0%)
- Thursday: 538,960 records (18.3%)
- Friday: 512,339 records (17.4%)

### September 2016 Dataset (`taxi_record_09_50drivers.pkl`)

| Property | Value |
|----------|-------|
| **Month** | September 2016 |
| **Date Range** | 2016-09-01 to 2016-09-30 |
| **Days of Week** | Monday - Friday (weekdays only, no weekends) |
| **Unique Dates** | 22 dates with data |
| **ISO Week Numbers** | Weeks 35, 36, 37, 38, 39 (5 weeks) |
| **Total GPS Records** | 2,476,205 records (across all 50 drivers) |
| **Time Coverage** | 24-hour data (00:00:00 - 23:59:59) |
| **Records per Driver** | Mean: 49,524 / Std Dev: 27,539 / Min: 11,198 / Max: 151,625 |
| **Pickup Events** | 80,734 |
| **Dropoff Events** | 80,709 |

**Week-by-Week Breakdown:**
- Week 35: Sep 01-02 (Thu-Fri)
- Week 36: Sep 05-09 (Mon-Fri)
- Week 37: Sep 12-16 (Mon-Fri)
- Week 38: Sep 19-23 (Mon-Fri)
- Week 39: Sep 26-30 (Mon-Fri)

**Day of Week Distribution:**
- Monday: 501,885 records (20.3%)
- Tuesday: 526,773 records (21.3%)
- Wednesday: 412,732 records (16.7%)
- Thursday: 495,067 records (20.0%)
- Friday: 539,748 records (21.8%)

### Common Temporal Characteristics

- **Sampling Rate:** Variable, typically 10-30 seconds between consecutive records
- **Weekend Data:** **No Saturday or Sunday data** - all three datasets contain weekdays only (Monday-Friday)
- **Coverage:** Full 24-hour daily coverage when data is present (00:00:00 - 23:59:59)
- **Continuity:** Weekday-only pattern with consistent weekly gaps (no weekend data)
- **Week Definition:** ISO week numbering (Monday = start of week)
- **Event Balance:** Pickup and dropoff events are nearly balanced (difference < 100 events per month), indicating complete trip data

---

## Spatial Coverage

### Geographic Bounds (Across All Three Datasets)

| Dimension | Minimum | Maximum | Span |
|-----------|---------|---------|------|
| **Latitude** | 22.442450° | 22.869999° | 0.427549° (~47.5 km) |
| **Longitude** | 113.750099° | 114.558197° | 0.808098° (~78.5 km) |

**Dataset-Specific Spatial Bounds:**

**July 2016:**
- Latitude: [22.444304, 22.869999]
- Longitude: [113.750099, 114.535934]
- Passenger indicator distribution: On (1): 1,074,402 (39.9%), Off (0): 1,620,751 (60.1%)

**August 2016:**
- Latitude: [22.454800, 22.869949]  
- Longitude: [113.750221, 114.558197]
- Passenger indicator distribution: On (1): 1,105,304 (37.4%), Off (0): 1,847,099 (62.6%)

**September 2016:**
- Latitude: [22.442450, 22.869999]
- Longitude: [113.750191, 114.552597]
- Passenger indicator distribution: On (1): 986,207 (39.8%), Off (0): 1,489,998 (60.2%)

### Geographic Context

- **City:** Shenzhen, Guangdong Province, China
- **Coordinate System:** WGS84 (World Geodetic System 1984)
- **Coverage Area:** Urban and suburban Shenzhen
- **Key Regions:** Downtown districts, airport, train stations, major business areas
- **Spatial Consistency:** All three datasets cover highly similar geographic regions with slight variations in the exact boundaries

### Spatial Quantization

For processing tasks, these coordinates are typically quantized into grid cells:

```python
# Standard grid quantization (used in pickup_dropoff_counts)
grid_size = 0.01  # degrees (~1.1 km)

# Based on actual bounds:
lat_min, lat_max = 22.442450, 22.869999
lon_min, lon_max = 113.750099, 114.558197

x_grid = int((longitude - lon_min) / grid_size)
y_grid = int((latitude - lat_min) / grid_size)

# Results in approximately:
# - 81 grid cells in longitude (x) direction (114.558197 - 113.750099) / 0.01
# - 43 grid cells in latitude (y) direction (22.869999 - 22.442450) / 0.01
# - Total: ~3,483 possible grid cells
```

---

## Data Quality and Characteristics

### Record Distribution

**Actual Statistics (from analysis):**

**July 2016:**
- Total Records: 2,695,153
- Records per Driver: Mean 53,903, Std Dev 26,301, Min 4,757, Max 128,570
- Passenger indicator: On (1): 1,074,402 (39.9%), Off (0): 1,620,751 (60.1%)

**August 2016:**
- Total Records: 2,952,403
- Records per Driver: Mean 59,048, Std Dev 27,349, Min 18,288, Max 119,227
- Passenger indicator: On (1): 1,105,304 (37.4%), Off (0): 1,847,099 (62.6%)

**September 2016:**
- Total Records: 2,476,205
- Records per Driver: Mean 49,524, Std Dev 27,539, Min 11,198, Max 151,625
- Passenger indicator: On (1): 986,207 (39.8%), Off (0): 1,489,998 (60.2%)

### Key Characteristics

- **Temporal Density:** Data collected throughout 24-hour periods on weekdays
- **Spatial Density:** Higher concentration in downtown and high-traffic areas
- **Driver Variability:** Significant variation in records per driver (coefficient of variation ~50-55%)

### Known Data Patterns

1. **Weekday-Only Data:** All three datasets contain Monday-Friday data exclusively, with no weekend coverage
2. **Full Day Coverage:** Complete 24-hour coverage (00:00:00 to 23:59:59) when data is present
3. **Driver Variation:** Significant variability in total records per driver (3-5x difference between min and max)
4. **Weekday Gaps:** No data for weekends (Saturdays and Sundays completely absent)
5. **Coordinate Precision:** Standard GPS precision (~5-10 meters accuracy)
6. **Event Balance:** Pickup and dropoff events are well-balanced (ratio ~1.000), indicating complete trip coverage

---

## Event Detection

### Pickup Events

A pickup occurs when the `passenger_indicator` transitions from `0` → `1` between consecutive records:

```python
def detect_pickups(records):
    pickups = []
    for i in range(1, len(records)):
        if records[i-1][4] == 0 and records[i][4] == 1:
            # Pickup occurred at record i
            pickups.append({
                'location': (records[i][1], records[i][2]),  # (lat, lon)
                'timestamp': records[i][5],
                'seconds': records[i][3]
            })
    return pickups
```

### Dropoff Events

A dropoff occurs when the `passenger_indicator` transitions from `1` → `0` between consecutive records:

```python
def detect_dropoffs(records):
    dropoffs = []
    for i in range(1, len(records)):
        if records[i-1][4] == 1 and records[i][4] == 0:
            # Dropoff occurred at record i
            dropoffs.append({
                'location': (records[i][1], records[i][2]),  # (lat, lon)
                'timestamp': records[i][5],
                'seconds': records[i][3]
            })
    return dropoffs
```

### Typical Event Counts

**Actual Event Counts (from analysis):**

| Dataset | Pickup Events (0→1) | Dropoff Events (1→0) | Difference | Balance Ratio |
|---------|-------------------|---------------------|------------|---------------|
| **July 2016** | 128,133 | 128,085 | 48 | 1.000 |
| **August 2016** | 116,829 | 116,808 | 21 | 1.000 |
| **September 2016** | 80,734 | 80,709 | 25 | 1.000 |

**Per-Driver Averages:**
- **July 2016:** ~2,563 pickups and ~2,562 dropoffs per driver
- **August 2016:** ~2,337 pickups and ~2,336 dropoffs per driver
- **September 2016:** ~1,615 pickups and ~1,614 dropoffs per driver

**Key Observations:**
- Pickup and dropoff counts are extremely well-balanced (difference < 100 events per month across all 50 drivers)
- The balance ratio of ~1.000 indicates high-quality, complete trip data
- September has notably fewer events, likely due to fewer recorded dates (22 vs 21-23 in other months)

---

## Data Processing Examples

### Example 1: Load and Explore a Dataset

```python
import pickle
from datetime import datetime

# Load dataset
with open('taxi_record_07_50drivers.pkl', 'rb') as f:
    data = pickle.load(f)

# Get basic statistics
driver_keys = list(data.keys())
total_records = sum(len(data[driver]) for driver in driver_keys)

print(f"Number of drivers: {len(driver_keys)}")
print(f"Total GPS records: {total_records:,}")

# Examine first driver's data
first_driver = driver_keys[0]
records = data[first_driver]
print(f"\nFirst driver ({first_driver}): {len(records)} records")
print(f"First record: {records[0]}")
print(f"Last record: {records[-1]}")
```

### Example 2: Extract Date Range

```python
from datetime import datetime

def get_date_range(data):
    """Extract the date range covered by the dataset"""
    timestamps = []
    
    for driver_key, records in data.items():
        for record in records:
            timestamp_str = record[5]
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            timestamps.append(dt)
    
    return min(timestamps), max(timestamps)

min_date, max_date = get_date_range(data)
print(f"Date range: {min_date} to {max_date}")
print(f"Total days: {(max_date - min_date).days + 1}")
```

### Example 3: Count Pickups and Dropoffs by Day

```python
from datetime import datetime
from collections import defaultdict

def count_events_by_day(data):
    """Count pickup and dropoff events by day"""
    pickups_by_day = defaultdict(int)
    dropoffs_by_day = defaultdict(int)
    
    for driver_key, records in data.items():
        for i in range(1, len(records)):
            prev_passenger = records[i-1][4]
            curr_passenger = records[i][4]
            timestamp_str = records[i][5]
            
            date = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').date()
            
            if prev_passenger == 0 and curr_passenger == 1:
                pickups_by_day[date] += 1
            elif prev_passenger == 1 and curr_passenger == 0:
                dropoffs_by_day[date] += 1
    
    return pickups_by_day, dropoffs_by_day

pickups, dropoffs = count_events_by_day(data)
for date in sorted(pickups.keys()):
    print(f"{date}: {pickups[date]} pickups, {dropoffs[date]} dropoffs")
```

### Example 4: Create Spatiotemporal Counts

```python
import numpy as np

def create_spatiotemporal_counts(data, grid_size=0.01):
    """Create pickup/dropoff counts by (x_grid, y_grid, time_bucket, day_of_week)"""
    from datetime import datetime
    from collections import defaultdict
    
    # Get global bounds
    all_lats = []
    all_lons = []
    for driver_key, records in data.items():
        for record in records:
            all_lats.append(record[1])
            all_lons.append(record[2])
    
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)
    
    counts = defaultdict(lambda: [0, 0])  # [pickups, dropoffs]
    
    for driver_key, records in data.items():
        for i in range(1, len(records)):
            prev_passenger = records[i-1][4]
            curr_passenger = records[i][4]
            
            lat = records[i][1]
            lon = records[i][2]
            seconds = records[i][3]
            timestamp_str = records[i][5]
            
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            day_of_week = dt.weekday() + 1  # Monday=1, Sunday=7
            
            # Skip Sunday
            if day_of_week == 7:
                continue
            
            # Quantize
            x_grid = int((lon - lon_min) / grid_size)
            y_grid = int((lat - lat_min) / grid_size)
            time_bucket = seconds // 300  # 5-minute buckets
            
            key = (x_grid, y_grid, time_bucket, day_of_week)
            
            # Count events
            if prev_passenger == 0 and curr_passenger == 1:
                counts[key][0] += 1  # Pickup
            elif prev_passenger == 1 and curr_passenger == 0:
                counts[key][1] += 1  # Dropoff
    
    return dict(counts)

counts = create_spatiotemporal_counts(data)
print(f"Generated {len(counts)} unique spatiotemporal keys")
```

---

## Relationship to Other FAMAIL Datasets

### Downstream Datasets

These raw GPS datasets are the source data for several processed datasets:

1. **`pickup_dropoff_counts.pkl`**
   - Generated by detecting pickup/dropoff events (passenger_indicator transitions)
   - Aggregated by spatiotemporal key: (x_grid, y_grid, time_bucket, day_of_week)
   - Tool: `pickup_dropoff_counts/processor.py`

2. **`latest_volume_pickups.pkl`**
   - Similar to pickup_dropoff_counts but with different processing methodology
   - Contains pickup volume data used in `all_trajs.pkl` feature generation

3. **`latest_traffic.pkl`**
   - Derived from raw GPS data to calculate traffic speeds, volumes, and wait times
   - Aggregated by spatiotemporal cells

4. **`all_trajs.pkl`**
   - Feature-augmented trajectories for cGAIL training
   - Uses raw GPS data + processed datasets for comprehensive feature vectors

### Data Flow

```
taxi_record_{07,08,09}_50drivers.pkl (RAW GPS DATA)
            |
            ├──> pickup_dropoff_counts.pkl
            ├──> latest_volume_pickups.pkl
            ├──> latest_traffic.pkl
            |
            └──> all_trajs.pkl (feature-augmented)
```

---

## Usage Guidelines

### When to Use These Datasets

- **✓ Generating new spatiotemporal aggregations** (counts, traffic metrics)
- **✓ Analyzing temporal patterns** (hourly, daily, weekly trends)
- **✓ Event detection** (pickups, dropoffs, trip reconstruction)
- **✓ Spatial analysis** (hot spots, coverage, route analysis)
- **✓ Custom time period processing** (specific weeks, days, or hours)

### When to Use Processed Datasets Instead

- **✗ Training machine learning models** → Use `all_trajs.pkl`
- **✗ Quick pickup/dropoff lookups** → Use `pickup_dropoff_counts.pkl` or `latest_volume_pickups.pkl`
- **✗ Traffic condition queries** → Use `latest_traffic.pkl`

### Processing Considerations

1. **Memory Usage:** Each dataset is 100-200 MB; loading all three requires ~600 MB RAM
2. **Processing Time:** Iterating through all records can take several minutes
3. **Weekday-Only Data:** All datasets contain Monday-Friday data exclusively - no weekend coverage
4. **Global Bounds:** Use these consistent bounds across all datasets for spatial quantization:
   - Latitude: [22.442450, 22.869999]
   - Longitude: [113.750099, 114.558197]
5. **Event Detection:** Always check consecutive records for passenger_indicator transitions
6. **Data Structure:** Remember the three-level hierarchy: Dictionary → Driver Keys → GPS Records (list of 6-element lists)

---

## Cross-Dataset Summary

### Consistency Verification

All three datasets share the following consistent characteristics:

✓ **Structural Consistency:**
- 50 drivers in each dataset
- 6-field GPS record structure: [plate_id, lat, lon, seconds, passenger, timestamp]
- Chronologically ordered records within each driver

✓ **Spatial Consistency:**
- All datasets cover the same geographic region (Shenzhen, China)
- Consistent coordinate system (WGS84)
- Similar latitude range: 22.44° - 22.87°
- Similar longitude range: 113.75° - 114.56°

✓ **Temporal Consistency:**
- All datasets contain weekday-only data (Monday-Friday)
- No Saturday or Sunday data in any dataset
- Full 24-hour daily coverage (00:00:00 - 23:59:59)
- 5 ISO weeks of coverage per dataset
- 21-23 unique dates per dataset

✓ **Event Balance:**
- Pickup and dropoff events are well-balanced (ratio ~1.000)
- Indicates complete trip data with minimal missing dropoffs/pickups

### Summary Statistics Table

| Metric | July 2016 | August 2016 | September 2016 |
|--------|-----------|-------------|----------------|
| **Total Records** | 2,695,153 | 2,952,403 | 2,476,205 |
| **Unique Dates** | 21 | 23 | 22 |
| **Date Range** | Jul 01-29 | Aug 01-31 | Sep 01-30 |
| **ISO Weeks** | 26-30 | 31-35 | 35-39 |
| **Pickup Events** | 128,133 | 116,829 | 80,734 |
| **Dropoff Events** | 128,085 | 116,808 | 80,709 |
| **Lat Range** | [22.4443, 22.8700] | [22.4548, 22.8699] | [22.4425, 22.8700] |
| **Lon Range** | [113.7501, 114.5359] | [113.7502, 114.5582] | [113.7502, 114.5526] |
| **Mean Records/Driver** | 53,903 | 59,048 | 49,524 |
| **Passenger On %** | 39.9% | 37.4% | 39.8% |
| **Passenger Off %** | 60.1% | 62.6% | 60.2% |

---

## Data Explorer

For interactive exploration of these datasets and reproduction of all statistics above, see:
- **Notebook:** `data_dictionary/explorers/taxi_record_raw_explorer.ipynb`
- **Purpose:** Analyzes temporal coverage, date ranges, event counts, and structural consistency
- **Output:** Detailed statistics, visualizations, temporal analysis, and cross-dataset verification
- **HTML Export:** `data_dictionary/explorers/taxi_record_raw_explorer_outputs/taxi_record_raw_explorer.html`

---

## File Locations

### Production Data
```
raw_data/
├── taxi_record_07_50drivers.pkl
├── taxi_record_08_50drivers.pkl
└── taxi_record_09_50drivers.pkl
```

### Documentation Data (for exploration only)
```
data_dictionary/datasets/raw_gps_data/
├── taxi_record_07_50drivers.pkl
├── taxi_record_08_50drivers.pkl
└── taxi_record_09_50drivers.pkl
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-10 | Initial data dictionary creation with comprehensive analysis |
| | | - Added exact temporal coverage for all three datasets |
| | | - Included actual statistics from data explorer analysis |
| | | - Documented weekday-only pattern (no weekend data) |
| | | - Added precise spatial bounds and passenger distribution |
| | | - Included pickup/dropoff event counts and balance verification |
| | | - Added cross-dataset summary comparison table |

---

## See Also

- **[Pickup/Dropoff Counts Processor README](../../pickup_dropoff_counts/README.md)** - Tool for processing raw GPS into spatiotemporal counts
- **[Pickup/Dropoff Counts Data Dictionary](pickup_dropoff_counts_data_dictionary.md)** - Processed output format
- **[Latest Volume Pickups Data Dictionary](latest_volume_pickups_data_dictionary.md)** - Alternative pickup aggregation
- **[All Trajs Data Dictionary](all_trajs_data_dictionary.md)** - Feature-augmented trajectories
- **[Data Explorer Notebook](../explorers/taxi_record_raw_explorer.ipynb)** - Interactive data exploration
