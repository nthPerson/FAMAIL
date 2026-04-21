"""Orchestrate the full source-data generation pipeline."""
from __future__ import annotations
import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from famail_temporal.data.source_generation import config
from famail_temporal.data.source_generation.event_stream import build_event_stream
from famail_temporal.data.source_generation.invariants import (
    apply_per_trajectory_invariants, check_systemic_invariants,
)
from famail_temporal.data.source_generation.removal import RemovalSummary
from famail_temporal.data.source_generation.views.active_taxis import (
    build_active_taxis_counts,
)
from famail_temporal.data.source_generation.views.calendars import (
    build_calendar_days_per_driver,
)
from famail_temporal.data.source_generation.views.pickup_dropoff import (
    build_pickup_dropoff_counts,
)
from famail_temporal.data.source_generation.views.profile import (
    compute_profile_features, zscore_normalize,
)
from famail_temporal.data.source_generation.views.trajectories import (
    build_driver_index_mapping, build_trajectories,
)
from famail_temporal.data.source_generation.writer import (
    OutputPaths, write_all_outputs,
)


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    paths: OutputPaths
    n_seeking_kept: int
    n_driving_kept: int
    n_removals: int


def _git_sha_or_none() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


def run_generation(
    input_dir: Path,
    output_dir: Path,
    expect_n_drivers: int | None = None,
) -> RunResult:
    """Run the full pipeline end-to-end.

    expect_n_drivers: override to relax the driver-count invariant
    (useful for testing with fewer than 50 synthetic drivers).
    """
    expect_n_drivers = expect_n_drivers or config.EXPECTED_N_DRIVERS
    log.info("Building event stream from %s", input_dir)
    es = build_event_stream(Path(input_dir))

    log.info("Building views…")
    pickup_dropoff_raw = build_pickup_dropoff_counts(es.df)
    active_taxis = build_active_taxis_counts(es.df)
    trajs = build_trajectories(es.df)
    mapping = build_driver_index_mapping(es.df)

    n_seeking_extracted = sum(len(v) for v in trajs.seeking_by_plate.values())
    n_driving_extracted = sum(len(v) for v in trajs.driving_by_plate.values())
    log.info("Extracted %d seeking + %d driving trajectories",
             n_seeking_extracted, n_driving_extracted)

    pickup_only = {k: (v[0], 0) for k, v in pickup_dropoff_raw.items()}
    dropoff_only = {k: (0, v[1]) for k, v in pickup_dropoff_raw.items()}

    log.info("Applying per-trajectory invariants…")
    kept_trajs, removals = apply_per_trajectory_invariants(
        trajs, pickup_only, dropoff_only,
        plate_to_idx=mapping["plate_to_idx"],
    )

    # Rebuild pickup/dropoff counts from surviving trajectory endpoints so
    # systemic invariant #5 (sum(counts) == n_trajectories) holds after
    # per-trajectory removals.
    pickup_dropoff_final: dict = {}
    for traj_list in kept_trajs.seeking_by_plate.values():
        for t in traj_list:
            key = tuple(t[-1])
            p, d = pickup_dropoff_final.get(key, (0, 0))
            pickup_dropoff_final[key] = (p + 1, d)
    for traj_list in kept_trajs.driving_by_plate.values():
        for t in traj_list:
            key = tuple(t[-1])
            p, d = pickup_dropoff_final.get(key, (0, 0))
            pickup_dropoff_final[key] = (p, d + 1)

    removal_summary = RemovalSummary(
        total_seeking_extracted=n_seeking_extracted,
        total_driving_extracted=n_driving_extracted,
        removals=removals,
    )
    if removal_summary.removal_rate() > config.REMOVAL_RATE_WARN_THRESHOLD:
        log.warning(
            "Per-trajectory removal rate %.2f%% exceeds threshold %.2f%%",
            100 * removal_summary.removal_rate(),
            100 * config.REMOVAL_RATE_WARN_THRESHOLD,
        )

    log.info("Computing profile features…")
    raw_features = compute_profile_features(es.df, kept_trajs)
    n_drivers_actual = len(mapping["plate_to_idx"])
    ordered_plates = [mapping["idx_to_plate"][i] for i in range(n_drivers_actual)]
    raw_matrix = np.array([
        [raw_features[p][f] for f in config.PROFILE_FEATURE_NAMES]
        for p in ordered_plates
    ], dtype=float)
    normalized, mean, std = zscore_normalize(raw_matrix)

    log.info("Checking systemic invariants…")
    pickup_for_check = {k: (v[0], 0) for k, v in pickup_dropoff_final.items()}
    dropoff_for_check = {k: (0, v[1]) for k, v in pickup_dropoff_final.items()}
    check_systemic_invariants(
        kept_trajs, pickup_for_check, dropoff_for_check,
        profile_matrix=normalized if n_drivers_actual == expect_n_drivers else None,
        n_drivers=n_drivers_actual,
        expect_n_drivers=expect_n_drivers,
    )

    log.info("Writing outputs to %s", output_dir)
    ms_seeking = {
        mapping["plate_to_idx"][p]: kept_trajs.seeking_by_plate.get(p, [])
        for p in mapping["plate_to_idx"].keys()
    }
    ms_driving = {
        mapping["plate_to_idx"][p]: kept_trajs.driving_by_plate.get(p, [])
        for p in mapping["plate_to_idx"].keys()
    }
    ms_calendars = build_calendar_days_per_driver(kept_trajs, mapping)
    ms_profile_payload = {
        "normalized": normalized,
        "mean": mean,
        "std": std,
        "feature_names": list(config.PROFILE_FEATURE_NAMES),
    }
    paths = write_all_outputs(
        out_dir=Path(output_dir),
        pickup_dropoff=pickup_dropoff_final,
        active_taxis=active_taxis,
        passenger_seeking_trajs=kept_trajs.seeking_by_plate,
        ms_seeking=ms_seeking,
        ms_driving=ms_driving,
        ms_profile=ms_profile_payload,
        ms_calendars=ms_calendars,
        driver_mapping=mapping,
        removal_summary=removal_summary,
        metadata_extras={
            "n_days": es.n_days,
            "bounds": {
                "lat_min": es.bounds.lat_min, "lat_max": es.bounds.lat_max,
                "lon_min": es.bounds.lon_min, "lon_max": es.bounds.lon_max,
            },
            "git_sha": _git_sha_or_none(),
            "config_snapshot": {
                "GRID_SIZE_DEG": config.GRID_SIZE_DEG,
                "NEIGHBORHOOD_SIZE": config.NEIGHBORHOOD_SIZE,
                "TIME_INTERVAL_MIN": config.TIME_INTERVAL_MIN,
                "WEEKDAY_DAYS": sorted(config.WEEKDAY_DAYS),
            },
        },
    )
    return RunResult(
        paths=paths,
        n_seeking_kept=sum(len(v) for v in kept_trajs.seeking_by_plate.values()),
        n_driving_kept=sum(len(v) for v in kept_trajs.driving_by_plate.values()),
        n_removals=len(removals),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="famail_temporal.data.source_generation",
        description="Unified GPS source-data generation for famail_temporal.",
    )
    p.add_argument("--input-dir", type=Path, default=config.DEFAULT_RAW_INPUT_DIR,
                   help="Directory containing the 3 taxi_record_*.pkl files.")
    p.add_argument("--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR,
                   help="Directory to write the 10 output files.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    result = run_generation(args.input_dir, args.output_dir)
    log.info(
        "Done: %d seeking + %d driving kept; %d removals; outputs at %s",
        result.n_seeking_kept, result.n_driving_kept,
        result.n_removals, args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
