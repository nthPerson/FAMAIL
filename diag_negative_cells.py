"""Diagnose the 'negative values in pickup_N' bug in the trajectory modifier.

Hypothesis: pickup_mass (= 1 / (n_hours_per_block * n_days)) is the mean-hourly
contribution of ONE raw pickup event. For many cells in the real pickup_3d the
value is less than a single pickup_mass, so the modifier's pre-loop subtraction
  base_3d[orig_cx, orig_cy, t_block] -= pickup_mass
drives the cell negative, which then propagates through the objective.

We confirm this by:
  1. Loading the real bundle.
  2. Computing pickup_mass per time block.
  3. Counting the number of ACTIVE cells whose pickup_3d value is less than
     the corresponding pickup_mass — those are trajectories that would fail.
  4. For the top-k used in the failing run (same selection as the runner),
     count how many of their pickup cells are under-funded.
"""
from __future__ import annotations
import numpy as np

from famail_temporal.data.loader import DataBundle
from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution, rank_trajectories, select_top_k,
)
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour


def main():
    print("Loading bundle with max_trajectories=5000 (matching failing run)...")
    bundle = DataBundle.load(max_trajectories=5000)

    T = bundle.pickup_3d.shape[-1]
    n_days = bundle.n_days
    print(f"\nn_days={n_days}, T={T}")
    print(f"pickup_3d.min={bundle.pickup_3d.min():.6f}, "
          f"pickup_3d.max={bundle.pickup_3d.max():.6f}")

    # pickup_mass per time block
    print("\n-- pickup_mass per time block --")
    for t in range(T):
        n_hours = int(bundle.n_hours_per_block[t])
        pmass = 1.0 / (n_hours * n_days)
        slice_t = bundle.pickup_3d[..., t]
        n_below = int(((slice_t > 0) & (slice_t < pmass)).sum())
        n_zero = int((slice_t == 0).sum())
        n_positive = int((slice_t > 0).sum())
        print(f"  t={t}: n_hours={n_hours}, pickup_mass={pmass:.8f}, "
              f"zero_cells={n_zero}, positive_cells={n_positive}, "
              f"positive_but_below_pmass={n_below}, "
              f"min_positive={slice_t[slice_t > 0].min() if n_positive else float('nan'):.8f}")

    # Active-unit view
    print("\n-- Active-unit view --")
    mask = bundle.mask_3d
    for t in range(T):
        n_hours = int(bundle.n_hours_per_block[t])
        pmass = 1.0 / (n_hours * n_days)
        active_t = mask[..., t] & (bundle.pickup_3d[..., t] >= 0)
        vals = bundle.pickup_3d[..., t][mask[..., t]]
        n_under = int((vals < pmass).sum())
        n_total = int(mask[..., t].sum())
        print(f"  t={t}: n_active={n_total}, "
              f"active_with_pickup_below_pmass={n_under} "
              f"({100.0*n_under/max(n_total,1):.1f}%)")

    # Top-k trajectory analysis (mimics runner)
    print("\n-- Top-200 trajectories (as in failing run) --")
    attr_unsigned, _ = compute_per_unit_attribution(bundle)
    scored = rank_trajectories(bundle.trajectories, attr_unsigned, bundle.unit_map)
    top_k_indices = select_top_k(scored, k=200)
    print(f"len(top_k_indices)={len(top_k_indices)}")

    under_count = 0
    zero_count = 0
    cell_tally: dict[tuple[int, int, int], int] = {}
    for idx in top_k_indices:
        traj = bundle.trajectories[idx]
        ps = traj.states[-1]
        cx, cy = int(ps.x_grid), int(ps.y_grid)
        hour = time_bucket_to_hour(ps.time_bucket)
        t_block = hour_to_block_index(hour)
        n_hours = int(bundle.n_hours_per_block[t_block])
        pmass = 1.0 / (n_hours * n_days)
        val = float(bundle.pickup_3d[cx, cy, t_block])
        key = (cx, cy, t_block)
        cell_tally[key] = cell_tally.get(key, 0) + 1
        if val == 0.0:
            zero_count += 1
        elif val < pmass:
            under_count += 1

    print(f"  pickup cells with value 0     : {zero_count}")
    print(f"  pickup cells with 0 < val < pmass: {under_count}")
    print(f"  unique (cell, t_block) cells   : {len(cell_tally)}")
    # Count cells claimed by >1 trajectory
    multi = [(k, v) for k, v in cell_tally.items() if v > 1]
    print(f"  cells claimed by > 1 traj      : {len(multi)}")
    # Worst offender: cell with most trajectories
    if multi:
        multi.sort(key=lambda kv: -kv[1])
        k0, n0 = multi[0]
        val0 = float(bundle.pickup_3d[k0])
        pmass0 = 1.0 / (int(bundle.n_hours_per_block[k0[2]]) * n_days)
        budget = val0 / pmass0 if pmass0 > 0 else float('inf')
        print(f"  worst cell: {k0} claimed by {n0} trajs, "
              f"pickup_3d={val0:.6f}, pmass={pmass0:.6f}, "
              f"mass_budget≈{budget:.2f} trajectories")

    # Simulate the batch: sequentially decrement cells; flag when a cell would go negative
    print("\n-- Simulating sequential modify_batch (budget check) --")
    running = bundle.pickup_3d.copy()
    first_fail_at = None
    fail_examples = []
    for i, idx in enumerate(top_k_indices):
        traj = bundle.trajectories[idx]
        ps = traj.states[-1]
        cx, cy = int(ps.x_grid), int(ps.y_grid)
        hour = time_bucket_to_hour(ps.time_bucket)
        t_block = hour_to_block_index(hour)
        n_hours = int(bundle.n_hours_per_block[t_block])
        pmass = 1.0 / (n_hours * n_days)

        # Within-iteration subtract (the -pickup_mass at start of modify_single)
        # would take running[cell] to running[cell] - pmass. If this is negative,
        # the modifier will fail this iteration.
        if running[cx, cy, t_block] < pmass - 1e-12:
            if first_fail_at is None:
                first_fail_at = i
            if len(fail_examples) < 5:
                fail_examples.append((i, traj.trajectory_id, (cx, cy, t_block),
                                      float(running[cx, cy, t_block]), float(pmass)))

        # Don't mutate running for ordering-free estimate; we want an optimistic
        # count (if a trajectory moves, it might rebalance). For a pessimistic
        # estimate we DO subtract at the originating cell whenever it moved.
        # Here we assume it always moves (worst case for negative bookkeeping):
        running[cx, cy, t_block] = running[cx, cy, t_block] - pmass

    total_below = int((running < 0).sum())
    print(f"  after worst-case sequential subtraction: "
          f"{total_below} cells went negative")
    print(f"  first trajectory that would underflow during its own subtract: "
          f"{first_fail_at}")
    for ex in fail_examples:
        batch_pos, tid, cell, val, pmass = ex
        print(f"    batch_pos={batch_pos}, traj_id={tid}, cell={cell}, "
              f"running_val={val:.6f}, pmass={pmass:.6f}")


if __name__ == "__main__":
    main()
