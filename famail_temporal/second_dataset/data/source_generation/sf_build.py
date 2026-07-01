"""SF source_data assembler (Task 3.6).

Orchestrates the tested SF components into the full set of `source_data`
artifacts that `preprocess.py` + `loader.py` consume (see docs/SF_SOURCE_SCHEMA.md),
written into the SF source dir. Run:

    python -m famail_temporal.second_dataset.data.source_generation.sf_build
"""
from __future__ import annotations

import pickle
from pathlib import Path

from famail_temporal.second_dataset.data.source_generation.sf_raw_loader import load_sf_raw
from famail_temporal.second_dataset.data.source_generation.sf_config import grid_from_points
from famail_temporal.second_dataset.data.source_generation.sf_segmentation import segment_driver
from famail_temporal.second_dataset.data.source_generation.sf_grid_counts import (
    count_pickup_dropoff, count_active_taxis_5x5, build_valid_mask,
)
from famail_temporal.second_dataset.data.source_generation.sf_demographics import build_cell_demographics
from famail_temporal.second_dataset.data.source_generation.sf_multistream import assemble_multistream
from famail_temporal.data.source_generation import config as sg_config


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def build(cab_dir: str, acs_csv: str, tiger_zip: str, out_dir: str,
          driver_ids=None, grid=None):
    """Assemble SF source_data. Optionally restrict to `driver_ids` (a fleet
    subsample); pass a fixed `grid` (computed from the FULL fleet) so cell
    indices align across subsample variants."""
    out = Path(out_dir)
    df = load_sf_raw(cab_dir)
    if grid is None:                       # default: grid from the full footprint
        grid = grid_from_points(df["lat"].to_numpy(), df["lon"].to_numpy())
    if driver_ids is not None:
        df = df[df["driver_id"].isin(set(driver_ids))].reset_index(drop=True)
    print(f"[sf_build] grid {grid.x_grid_max}x{grid.y_grid_max}, "
          f"{len(df):,} pings, {df['driver_id'].nunique()} drivers", flush=True)

    print("[sf_build] segmenting per driver ...", flush=True)
    per_driver, all_pick, all_drop = {}, [], []
    for did, g in df.groupby("driver_id"):
        seg = segment_driver(g, grid)
        per_driver[int(did)] = (g, seg)
        # The editor reads a trajectory's pickup as its FINAL state
        # (Trajectory.pickup = states[-1]; a passenger-seeking trajectory ends
        # where the passenger boards). Count one pickup at the TERMINAL cell of
        # each seeking trajectory (and one dropoff at each driving trajectory's
        # terminal cell) so pickup_dropoff_counts aligns cell-for-cell with
        # passenger_seeking_trajs — otherwise the editor subtracts a trajectory's
        # mass from a cell whose count was recorded one ping later (occ=1), driving
        # it negative (compute_fspatial guard) and misaligning the fairness signal.
        all_pick.extend(tr[-1] for tr in seg.seeking)
        all_drop.extend(tr[-1] for tr in seg.driving)

    print("[sf_build] gridding counts ...", flush=True)
    pd_counts = count_pickup_dropoff(all_pick, all_drop)
    active = count_active_taxis_5x5(df, grid)

    print("[sf_build] mapping demographics (majority-overlap) ...", flush=True)
    demo_grid, demo_names = build_cell_demographics(grid, acs_csv, tiger_zip)
    valid = build_valid_mask(grid)

    print("[sf_build] assembling multi-stream + profiles ...", flush=True)
    ms = assemble_multistream(per_driver, grid)

    _write(out / "pickup_dropoff_counts.pkl", pd_counts)
    _write(out / "active_taxis_5x5_hourly.pkl", {
        "data": active,
        "stats": {"n_entries": len(active)},
        "config": {"neighborhood_dims": sg_config.NEIGHBORHOOD_SIZE, "period_type": "hourly"},
        "version": sg_config.OUTPUT_FORMAT_VERSION,
    })
    _write(out / "cell_demographics.pkl",
           {"demographics_grid": demo_grid, "feature_names": demo_names})
    _write(out / "grid_to_district_mapping.pkl", {"valid_mask": valid})
    _write(out / "passenger_seeking_trajs.pkl", ms["passenger_seeking"])
    _write(out / "driver_index_mapping.pkl", ms["driver_mapping"])
    _write(out / "ms_seeking_trajs.pkl", ms["ms_seeking"])
    _write(out / "ms_driving_trajs.pkl", ms["ms_driving"])
    _write(out / "ms_seeking_calendar_days.pkl", ms["ms_seeking_days"])
    _write(out / "ms_driving_calendar_days.pkl", ms["ms_driving_days"])
    _write(out / "ms_profile_features.pkl", {
        "features": ms["profiles_raw"],
        "features_normalized": ms["profiles_normalized"],
        "feature_names": ms["profile_feature_names"],
        "normalization": {"mean": ms["profile_mean"], "std": ms["profile_std"]},
        "n_features": len(ms["profile_feature_names"]),
    })
    n = len(list(out.glob("*.pkl")))
    print(f"[sf_build] wrote {n} source_data files to {out}", flush=True)
    return grid


def main():
    from famail_temporal import config as fc
    sd = fc.PACKAGE_ROOT / "source_data" / "second_dataset"
    build(
        cab_dir=str(sd / "cabspottingdata"),
        acs_csv=str(sd / "demographics" / "acs_2006_2010_tracts.csv"),
        tiger_zip=str(sd / "demographics" / "tiger_2010_tracts_06_CA.zip"),
        out_dir=str(sd / "sf_source"),
    )


if __name__ == "__main__":
    main()
