# Active Taxis Counting Algorithm

## Overview

This document describes the algorithm for counting active taxis in an n×n grid neighborhood for each cell and time period.

## Definitions

### Active Taxi

A taxi is considered "active" in a neighborhood during a time period if it had at least one GPS reading in any cell within that neighborhood during that period.

### Neighborhood

Given a cell at position (x, y) and a neighborhood size parameter k, the neighborhood consists of all cells (x', y') where:
- x - k ≤ x' ≤ x + k
- y - k ≤ y' ≤ y + k
- 1 ≤ x' ≤ x_max (within grid bounds)
- 1 ≤ y' ≤ y_max (within grid bounds)

This results in a (2k+1) × (2k+1) neighborhood, clipped at grid boundaries.

### Time Period

The time period is determined by the `period_type` configuration:

| Period Type | Description | Key Components |
|-------------|-------------|----------------|
| `time_bucket` | 5-minute intervals | (time_bucket, day) |
| `hourly` | 1-hour intervals | (hour, day) |
| `daily` | Full day | (day,) |
| `all` | All time aggregated | ('all',) |

## Algorithm Steps

### Step 1: Load and Parse Raw Data

```python
# Load raw GPS data
df = load_raw_data(filepath)

# Parse timestamps
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter Sundays if configured
df['day'] = timestamp_to_day(df['timestamp'], exclude_sunday=True)
df = df.dropna(subset=['day'])
```

### Step 2: Apply Quantization

```python
# Spatial quantization (same as pickup_dropoff_counts)
x_grid, y_grid = gps_to_grid(df['latitude'], df['longitude'], bounds)
df['x_grid'] = x_grid + x_grid_offset  # 1-indexed
df['y_grid'] = y_grid + y_grid_offset

# Temporal quantization
df['time_bin'] = timestamp_to_time_bin(df['timestamp']) + time_offset
df['hour'] = timestamp_to_hour(df['timestamp'])
```

### Step 3: Build Taxi Presence Index

Create an index that maps each period to the set of taxis present in each cell:

```python
# Index structure: {period_key: {(x, y): {taxi_id1, taxi_id2, ...}}}
presence_index = defaultdict(lambda: defaultdict(set))

for row in df.itertuples():
    period_key = get_period_key(row.time_bin, row.hour, row.day, period_type)
    presence_index[period_key][(row.x_grid, row.y_grid)].add(row.plate_id)
```

**Complexity:** O(n) where n is the number of GPS records.

### Step 4: Count Active Taxis in Neighborhoods

For each (cell, period), count unique taxis in the neighborhood:

```python
for period_key in presence_index:
    cell_taxis = presence_index[period_key]
    
    for x in range(1, x_max + 1):
        for y in range(1, y_max + 1):
            # Get neighborhood cells
            neighborhood = get_neighborhood_cells(x, y, k, grid_dims)
            
            # Collect unique taxis in neighborhood
            active_taxis = set()
            for (nx, ny) in neighborhood:
                if (nx, ny) in cell_taxis:
                    active_taxis.update(cell_taxis[(nx, ny)])
            
            # Store count
            output[(x, y) + period_key] = len(active_taxis)
```

**Complexity:** O(P × G × N) where:
- P = number of periods
- G = number of grid cells (48 × 90 = 4,320)
- N = neighborhood size ((2k+1)²)

For k=2 (5×5 neighborhood) and hourly aggregation:
- P ≈ 144 (24 hours × 6 days)
- G = 4,320
- N = 25

Total operations ≈ 15.5 million, which completes in a few minutes.

## Output Format

The output is a dictionary with keys of the form `(x, y, *period_key)`:

```python
# For hourly:
{
    (1, 1, 0, 1): 12,    # Cell (1,1), hour 0, Monday: 12 active taxis
    (1, 1, 1, 1): 15,    # Cell (1,1), hour 1, Monday: 15 active taxis
    ...
}

# For daily:
{
    (1, 1, 1): 35,       # Cell (1,1), Monday: 35 unique taxis
    (1, 1, 2): 38,       # Cell (1,1), Tuesday: 38 unique taxis
    ...
}
```

## Memory Considerations

### Presence Index Size

The presence index stores sets of taxi IDs for each (cell, period). 

Worst case: Every taxi visits every cell in every period
- 50 taxis × 4,320 cells × 144 periods = 31M entries

Actual case: Much smaller due to sparse taxi presence
- Typical: < 1M entries

### Output Size

The output dictionary has a fixed size based on configuration:
- For hourly: 48 × 90 × 24 × 6 = 622,080 entries
- For daily: 48 × 90 × 6 = 25,920 entries

Each entry is a small integer, so total memory is manageable (< 50 MB for hourly).

## Validation

### Expected Properties

1. **Bounds**: `0 ≤ count ≤ 50` (total number of taxis in dataset)
2. **Spatial correlation**: Adjacent cells should have similar counts
3. **Temporal patterns**: Higher counts during peak hours
4. **Neighborhood effect**: Larger neighborhoods → higher average counts

### Validation Tests

1. **Boundary cells**: Cells at grid edges have smaller neighborhoods (fewer cells)
2. **Empty cells**: Cells with no taxi presence should have count based only on neighbors
3. **Total consistency**: Sum of all counts should scale with (2k+1)² for different k

## Performance Optimization

### Current Optimizations

1. **Set operations**: Using Python sets for efficient union operations
2. **Batch processing**: Processing all records in one pass for presence index
3. **Memory efficiency**: Sparse representation of presence index

### Potential Future Optimizations

1. **NumPy vectorization**: Convert to array operations for counting
2. **Parallel processing**: Process different periods in parallel
3. **Caching**: Cache neighborhood definitions for each cell
4. **Incremental updates**: Support adding new data without full recomputation
