"""Verify the three inconsistencies between passenger_seeking_trajs_45-800.pkl
and pickup_dropoff_counts.pkl by inspecting the raw file contents.

SECURITY NOTE: This script loads project-local .pkl files that were produced
by this same project's trusted tooling (pickup_dropoff_counts/processor.py
and new_all_trajs/step1_processor.py). It does not load pickles from any
external or untrusted source.

Hypotheses:
  H1: Trajectory time_buckets are 0-indexed (0..287), while
      pickup_dropoff_counts time_buckets are 1-indexed (1..288).
  H2: Trajectory days are in {1..5} (Mon-Fri only, weekends excluded),
      while pickup_dropoff_counts days are in {1..6} (Mon-Sat, only
      Sunday excluded).
  H3: The "pickup state" in each trajectory is the LAST SEEKING state
      (passenger=0), not the pickup-transition state (passenger=1).
      So the trajectory's pickup cell may differ from the cell recorded
      in pickup_dropoff_counts for the same event.
"""
from __future__ import annotations
import pickle  # Loading project-local, trusted .pkl files only. See note above.
from pathlib import Path
from collections import Counter

RAW = Path("famail_temporal/raw_data")


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    print("Loading passenger_seeking_trajs_45-800.pkl ...")
    trajs_by_driver = load(RAW / "passenger_seeking_trajs_45-800.pkl")
    print("Loading pickup_dropoff_counts.pkl ...")
    counts = load(RAW / "pickup_dropoff_counts.pkl")

    traj_tbs: Counter[int] = Counter()
    traj_days: Counter[int] = Counter()
    traj_cells = 0
    for did, trajs in trajs_by_driver.items():
        for traj in trajs:
            for state in traj:
                traj_tbs[int(state[2])] += 1
                traj_days[int(state[3])] += 1
                traj_cells += 1

    count_tbs: Counter[int] = Counter()
    count_days: Counter[int] = Counter()
    pickup_map = {}  # (x,y,t,d) -> pickup_count, only non-zero
    for (x, y, t, d), (p, dr) in counts.items():
        count_tbs[int(t)] += 1
        count_days[int(d)] += 1
        if p > 0:
            pickup_map[(int(x), int(y), int(t), int(d))] = int(p)

    print("\n== H1: time_bucket distribution ==")
    print(f"  trajectories:    min={min(traj_tbs)}, max={max(traj_tbs)}, "
          f"unique_values={len(traj_tbs)}, total_states={traj_cells}")
    print(f"  pickup_3d keys:  min={min(count_tbs)}, max={max(count_tbs)}, "
          f"unique_values={len(count_tbs)}")

    print("\n== H2: day distribution ==")
    print(f"  trajectories:    {sorted(traj_days.items())}")
    print(f"  pickup_3d keys:  days_present={sorted(count_days.keys())}")

    # ---------- H3: pickup cell alignment ----------
    def key_hit(x, y, t, d):
        return pickup_map.get((x, y, t, d), 0) > 0

    same_cell_hit_raw = 0
    same_cell_hit_plus1 = 0
    spatial_neighbor_hit = 0  # 3x3 around (x,y) at same (t+1, d)
    temporal_neighbor_hit = 0  # same (x,y), t+1 ± 1
    any_nearby_hit = 0  # union of spatial ± temporal ± diagonal
    no_hit_anywhere = 0
    n_trajs = 0

    for did, trajs in trajs_by_driver.items():
        for traj in trajs:
            if not traj:
                continue
            last = traj[-1]
            x, y, t, d = int(last[0]), int(last[1]), int(last[2]), int(last[3])
            n_trajs += 1

            if key_hit(x, y, t, d):
                same_cell_hit_raw += 1
            if key_hit(x, y, t + 1, d):
                same_cell_hit_plus1 += 1

            # 3x3 xy neighbor at t+1, d
            spatial = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if key_hit(x + dx, y + dy, t + 1, d):
                        spatial = True
                        break
                if spatial:
                    break
            if spatial:
                spatial_neighbor_hit += 1

            # t+1 ± 1 bucket at same (x, y)
            temporal = key_hit(x, y, t, d) or key_hit(x, y, t + 2, d)
            if temporal:
                temporal_neighbor_hit += 1

            # Any hit in the 3x3x3 neighborhood around (x, y, t+1)
            any_nearby = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dt in (-1, 0, 1):
                        if key_hit(x + dx, y + dy, t + 1 + dt, d):
                            any_nearby = True
                            break
                    if any_nearby:
                        break
                if any_nearby:
                    break
            if any_nearby:
                any_nearby_hit += 1
            else:
                no_hit_anywhere += 1

    print("\n== H3: pickup-cell alignment across trajectories ==")
    print(f"  total trajectories checked: {n_trajs}")
    print(f"  same-cell exact match (traj t, no offset) : "
          f"{same_cell_hit_raw:>6} ({100.0*same_cell_hit_raw/n_trajs:.2f}%)")
    print(f"  same-cell exact match (traj t+1, offset)  : "
          f"{same_cell_hit_plus1:>6} ({100.0*same_cell_hit_plus1/n_trajs:.2f}%)")
    print(f"  hit in 3x3 xy neighborhood at (t+1, d)    : "
          f"{spatial_neighbor_hit:>6} ({100.0*spatial_neighbor_hit/n_trajs:.2f}%)")
    print(f"  hit in t+1 ± 1 at same (x, y, d)          : "
          f"{temporal_neighbor_hit:>6} ({100.0*temporal_neighbor_hit/n_trajs:.2f}%)")
    print(f"  ANY hit in 3x3x3 around (x, y, t+1, d)    : "
          f"{any_nearby_hit:>6} ({100.0*any_nearby_hit/n_trajs:.2f}%)")
    print(f"  no hit anywhere nearby                    : "
          f"{no_hit_anywhere:>6} ({100.0*no_hit_anywhere/n_trajs:.2f}%)")


if __name__ == "__main__":
    main()
