#!/usr/bin/env python3
"""Empirical sweep answering the SF supply/demand regime questions:
  - Does subsampling to 50 drivers (Shenzhen fleet size) restore F_causal signal?
  - Does per-cell supply (vs 5x5) help?
  - Does DEMAND_FLOOR help?
F_causal proxy = 1 - R^2(residual ~ demographics), which equals the real
F_causal = R'(I-H_demo)R / R'MR. g0(D) via isotonic regression (matches the
pipeline's isotonic diagnostic). CPU only.  Run with FAMAIL_CITY=sf.
"""
import os, numpy as np
from sklearn.isotonic import IsotonicRegression
from famail_temporal import config
from famail_temporal.second_dataset.data.source_generation.sf_raw_loader import load_sf_raw
from famail_temporal.second_dataset.data.source_generation.sf_config import grid_from_points
from famail_temporal.second_dataset.data.source_generation.sf_segmentation import segment_driver
from famail_temporal.second_dataset.data.source_generation.sf_grid_counts import count_pickup_dropoff, count_active_taxis_5x5
from famail_temporal.second_dataset.data.source_generation.sf_demographics import build_cell_demographics
from famail_temporal.data.aggregation import aggregate_pickup_dropoff, aggregate_active_taxis, dataset_n_days

SD = os.path.join(config.PACKAGE_ROOT, "source_data", "second_dataset")
df = load_sf_raw(os.path.join(SD, "cabspottingdata"))
grid = grid_from_points(df["lat"].to_numpy(), df["lon"].to_numpy())
print(f"grid {grid.x_grid_max}x{grid.y_grid_max}, {df['driver_id'].nunique()} drivers")

# segment once; collect terminal-cell pickups/dropoffs per driver
seg_by = {}
for did, g in df.groupby("driver_id"):
    seg_by[int(did)] = segment_driver(g, grid)

rng = np.random.RandomState(42)
drivers = sorted(seg_by)
sub = set(rng.choice(drivers, size=50, replace=False))

demo = build_cell_demographics(grid, os.path.join(SD, "demographics", "acs_2006_2010_tracts.csv"),
                               os.path.join(SD, "demographics", "tiger_2010_tracts_06_CA.zip"))[0]
demo_finite = np.isfinite(demo).all(axis=2)

def picks_drops(driver_set):
    p, d = [], []
    for did in driver_set:
        s = seg_by[did]
        p.extend(tr[-1] for tr in s.seeking)
        d.extend(tr[-1] for tr in s.driving)
    return p, d

def multiple_r2(R, X):
    Xs = (X - X.mean(0)) / X.std(0)
    A = np.column_stack([np.ones(len(R)), Xs])
    beta, *_ = np.linalg.lstsq(A, R, rcond=None)
    ss_res = ((R - A @ beta) ** 2).sum(); ss_tot = ((R - R.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

print(f"\n{'fleet':6} {'supply':9} {'floor':6} {'n_act':7} {'%clmp':6} {'meanY':8} {'r(S,house)':10} {'F_causal':9}")
for fleet_name, dset, fdf in [("536", drivers, df), ("50", sub, df[df.driver_id.isin(sub)])]:
    p, d = picks_drops(dset)
    nd = max(dataset_n_days(count_pickup_dropoff(p, d)), 1)
    demand_3d, _ = aggregate_pickup_dropoff(count_pickup_dropoff(p, d), nd)
    for sup_name, k in [("5x5", 2), ("per-cell", 0)]:
        supply_3d = aggregate_active_taxis(count_active_taxis_5x5(fdf, grid, k=k), nd)
        for floor in [0.1, 0.5, 1.0]:
            mask = (supply_3d > config.ACTIVE_SUPPLY_THRESHOLD) & demo_finite[:, :, None]
            xs, ys, ts = np.where(mask)
            if len(xs) < 50:
                print(f"{fleet_name:6} {sup_name:9} {floor:<6} n_active<50 skip"); continue
            D = demand_3d[mask]; S = supply_3d[mask]
            demoU = np.stack([demo[xs, ys, j] for j in range(3)], axis=1)
            Dc = np.maximum(D, floor); Y = S / Dc
            iso = IsotonicRegression(out_of_bounds="clip").fit(Dc, Y)
            R = Y - iso.predict(Dc)
            fc = 1 - multiple_r2(R, demoU)
            rS = np.corrcoef(S, demoU[:, 0])[0, 1]
            pct = 100 * (D < floor).mean()
            print(f"{fleet_name:6} {sup_name:9} {floor:<6} {len(xs):<7} {pct:<6.0f} {Y.mean():<8.1f} {rS:<+10.3f} {fc:<9.4f}")
