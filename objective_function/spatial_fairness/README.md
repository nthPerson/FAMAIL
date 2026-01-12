# Spatial Fairness Term

This module implements the **Spatial Fairness Term** ($F_{spatial}$) for the FAMAIL objective function.

## Overview

The Spatial Fairness Term measures equality of taxi service distribution across geographic regions using the Gini coefficient. A higher value indicates more equitable service distribution.

### Mathematical Formulation

$$F_{spatial} = 1 - \frac{1}{2|P|} \sum_{p \in P} (G_a^p + G_d^p)$$

Where:
- $P$ = set of time periods
- $G_a^p$ = Gini coefficient for arrival (dropoff) service rates in period $p$
- $G_d^p$ = Gini coefficient for departure (pickup) service rates in period $p$

### Gini Coefficient Formula

$$G = 1 + \frac{1}{n} - \frac{2}{n^2 \mu} \sum_{i=1}^{n} (n - i + 1) \cdot x_{(i)}$$

Where $x_{(i)}$ are service rate values sorted in ascending order.

## Installation

```bash
pip install numpy pandas
```

For the dashboard:
```bash
pip install streamlit plotly
```

## Usage

### Basic Usage

```python
import pickle
from spatial_fairness import SpatialFairnessTerm, SpatialFairnessConfig

# Load data
with open('source_data/pickup_dropoff_counts.pkl', 'rb') as f:
    data = pickle.load(f)

# Create configuration
config = SpatialFairnessConfig(
    period_type='hourly',     # 'time_bucket', 'hourly', 'daily', or 'all'
    grid_dims=(48, 90),       # x, y grid dimensions
    num_taxis=50,             # number of active taxis
    num_days=21.0,            # days in dataset
    include_zero_cells=True,  # include cells with no activity
)

# Compute spatial fairness
term = SpatialFairnessTerm(config)
result = term.compute({}, {'pickup_dropoff_counts': data})
print(f"Spatial Fairness: {result:.4f}")
```

### Detailed Breakdown

```python
breakdown = term.compute_with_breakdown({}, {'pickup_dropoff_counts': data})

print(f"Overall F_spatial: {breakdown['value']:.4f}")
print(f"Avg Gini (Pickups):  {breakdown['components']['avg_gini_departure']:.4f}")
print(f"Avg Gini (Dropoffs): {breakdown['components']['avg_gini_arrival']:.4f}")
print(f"Periods analyzed: {breakdown['statistics']['n_periods']}")
```

### Using Predefined Configurations

```python
from spatial_fairness.config import (
    HOURLY_CONFIG,
    DAILY_CONFIG,
    PEAK_HOURS_CONFIG,
    ACTIVE_CELLS_CONFIG,
)

# Use hourly aggregation
term = SpatialFairnessTerm(HOURLY_CONFIG)

# Only analyze peak hours (7-9am, 5-8pm)
term = SpatialFairnessTerm(PEAK_HOURS_CONFIG)

# Exclude cells with zero activity
term = SpatialFairnessTerm(ACTIVE_CELLS_CONFIG)
```

## Dashboard

Run the interactive dashboard for visualization and validation:

```bash
cd objective_function/spatial_fairness
streamlit run dashboard.py
```

The dashboard provides:
- **Configuration panel**: Select period type, day filters, and computation options
- **Temporal analysis**: Gini coefficients over time, hourly patterns
- **Spatial distribution**: Heatmaps of pickups and dropoffs
- **Lorenz curves**: Visualize inequality distribution
- **Statistics**: Detailed breakdown of results
- **Raw data**: Export results to CSV

## Module Structure

```
spatial_fairness/
├── __init__.py           # Package exports
├── config.py             # SpatialFairnessConfig dataclass
├── term.py               # SpatialFairnessTerm implementation
├── utils.py              # Gini computation, data aggregation
├── dashboard.py          # Streamlit visualization dashboard
├── README.md             # This file
└── tests/
    ├── __init__.py
    └── test_spatial_fairness.py  # Unit tests
```

## Data Format

The input data (`pickup_dropoff_counts`) should be a dictionary with:
- **Key**: `(x, y, time_bucket, day)` tuple (1-indexed coordinates)
- **Value**: `(pickup_count, dropoff_count)` tuple

Where:
- `x`: Grid x-coordinate [1, 48]
- `y`: Grid y-coordinate [1, 90]
- `time_bucket`: 5-minute interval index [1, 288]
- `day`: Day of week [1, 6] (Mon-Sat)

## Running Tests

```bash
cd objective_function
python -m pytest spatial_fairness/tests/ -v
```

## Results on Real Data

With the July 2016 taxi dataset (50 drivers, 21 weekdays):

| Aggregation | F_spatial | Gini (Pickups) | Gini (Dropoffs) | Periods |
|-------------|-----------|----------------|-----------------|---------|
| Hourly      | 0.1911    | 0.8118         | 0.8061          | 144     |
| Daily       | 0.2013    | 0.8044         | 0.7930          | 6       |
| All         | 0.0431    | 0.9642         | 0.9495          | 1       |

**Interpretation**: The low F_spatial values (~0.2) indicate significant spatial inequality in taxi service distribution, with Gini coefficients around 0.80 showing that taxi services are highly concentrated in certain areas.

## References

- Su, R., McBride, E.C., & Goulias, K.G. (2018). "Uncovering Spatial Inequality in Taxi Services in the Context of a Subsidy Cut"
- FAMAIL Project - Fairness-Aware Mobility AI Learning

## License

MIT License - See project root for details.
