"""Test SpatialFairnessTerm with real data."""
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
from spatial_fairness import SpatialFairnessTerm, SpatialFairnessConfig

# Load real data
data_path = Path(__file__).parent.parent.parent / "source_data" / "pickup_dropoff_counts.pkl"
with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(f'Loaded {len(data):,} records')

# Test with hourly aggregation
config = SpatialFairnessConfig(
    period_type='hourly',
    grid_dims=(48, 90),
    num_taxis=50,
    num_days=21.0,
    include_zero_cells=True,
    verbose=True,
)

term = SpatialFairnessTerm(config)
breakdown = term.compute_with_breakdown({}, {'pickup_dropoff_counts': data})

print()
print('=' * 50)
print('SPATIAL FAIRNESS RESULTS (Hourly Aggregation)')
print('=' * 50)
print(f"Overall F_spatial: {breakdown['value']:.4f}")
print(f"Avg Gini (Pickups):  {breakdown['components']['avg_gini_departure']:.4f}")
print(f"Avg Gini (Dropoffs): {breakdown['components']['avg_gini_arrival']:.4f}")
print(f"Number of periods: {breakdown['statistics']['n_periods']}")
print(f"Computation time: {breakdown['diagnostics']['computation_time_ms']:.1f} ms")
print()

# Show Gini range
gini_dep = breakdown['components']['per_period_gini_departure']
gini_arr = breakdown['components']['per_period_gini_arrival']
print(f"Pickup Gini range: [{min(gini_dep):.4f}, {max(gini_dep):.4f}]")
print(f"Dropoff Gini range: [{min(gini_arr):.4f}, {max(gini_arr):.4f}]")
print()

# Also test daily aggregation
print('=' * 50)
print('SPATIAL FAIRNESS RESULTS (Daily Aggregation)')
print('=' * 50)

config_daily = SpatialFairnessConfig(
    period_type='daily',
    grid_dims=(48, 90),
    num_taxis=50,
    num_days=21.0,
    include_zero_cells=True,
)

term_daily = SpatialFairnessTerm(config_daily)
breakdown_daily = term_daily.compute_with_breakdown({}, {'pickup_dropoff_counts': data})

print(f"Overall F_spatial: {breakdown_daily['value']:.4f}")
print(f"Avg Gini (Pickups):  {breakdown_daily['components']['avg_gini_departure']:.4f}")
print(f"Avg Gini (Dropoffs): {breakdown_daily['components']['avg_gini_arrival']:.4f}")
print(f"Number of periods: {breakdown_daily['statistics']['n_periods']}")
print()

# Test aggregate (all data)
print('=' * 50)
print('SPATIAL FAIRNESS RESULTS (All Data Aggregated)')
print('=' * 50)

config_all = SpatialFairnessConfig(
    period_type='all',
    grid_dims=(48, 90),
    num_taxis=50,
    num_days=21.0,
    include_zero_cells=True,
)

term_all = SpatialFairnessTerm(config_all)
breakdown_all = term_all.compute_with_breakdown({}, {'pickup_dropoff_counts': data})

print(f"Overall F_spatial: {breakdown_all['value']:.4f}")
print(f"Gini (Pickups):  {breakdown_all['components']['avg_gini_departure']:.4f}")
print(f"Gini (Dropoffs): {breakdown_all['components']['avg_gini_arrival']:.4f}")
print()
print("✅ All tests completed successfully!")
