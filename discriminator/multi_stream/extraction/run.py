"""
CLI entry point for dual trajectory extraction and profile feature computation.

Usage:
    python -m discriminator.multi_stream.extraction.run
    python -m discriminator.multi_stream.extraction.run --min-states 10 --min-duration 600
    python -m discriminator.multi_stream.extraction.run --skip-profiles
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from .config import ExtractionConfig
from .extractor import run_extraction
from .profile_features import compute_profile_features


def main():
    parser = argparse.ArgumentParser(
        description="Extract seeking/driving trajectories and compute profile features"
    )
    parser.add_argument("--min-states", type=int, default=5,
                        help="Minimum states after deduplication (default: 5)")
    parser.add_argument("--min-duration", type=int, default=300,
                        help="Minimum segment duration in seconds (default: 300)")
    parser.add_argument("--max-states", type=int, default=1000,
                        help="Maximum states per trajectory (default: 1000)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: discriminator/multi_stream/extracted_data)")
    parser.add_argument("--skip-profiles", action="store_true",
                        help="Skip profile feature computation")
    args = parser.parse_args()

    # Build config
    config = ExtractionConfig(
        min_segment_states=args.min_states,
        min_segment_duration_sec=args.min_duration,
        max_segment_states=args.max_states,
    )
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    print("=" * 65)
    print("Phase 2A: Dual Trajectory Extraction")
    print("=" * 65)
    print(f"  Config: min_states={config.min_segment_states}, "
          f"min_duration={config.min_segment_duration_sec}s, "
          f"max_states={config.max_segment_states}")
    print()
    print("Loading raw data...")

    # Progress callback
    last_pct = [-1]
    def progress(stage, pct):
        pct_int = int(pct * 100)
        if pct_int > last_pct[0]:
            last_pct[0] = pct_int
            print(f"\r  [{pct_int:3d}%] {stage}", end="", flush=True)
            if pct_int == 100:
                print()

    # Run extraction
    (seeking, driving, index_to_plate, calendar_days,
     seeking_cal_days, driving_cal_days, cal_day_map,
     bounds, stats) = run_extraction(config, progress)

    # Print extraction summary
    print(f"\n{'─' * 65}")
    print("Extraction Summary")
    print(f"{'─' * 65}")
    print(f"  Drivers:              {stats.drivers_processed}")
    print(f"  Total raw records:    {stats.total_raw_records:,}")
    print(f"  Records out-of-bbox:  {stats.records_outside_bbox:,}")
    print(f"  Records weekend:      {stats.records_weekend_filtered:,}")
    print(f"  Records invalid:      {stats.records_timestamp_invalid:,}")
    print(f"  States deduplicated:  {stats.states_deduplicated:,}")
    print(f"  Segments too short:   {stats.segments_filtered_too_short:,}")
    print(f"  Segments too brief:   {stats.segments_filtered_too_brief:,}")
    print(f"\n  Seeking trajectories: {stats.seeking_trajectories:,} "
          f"({stats.seeking_states_total:,} states)")
    print(f"  Driving trajectories: {stats.driving_trajectories:,} "
          f"({stats.driving_states_total:,} states)")

    if stats.seeking_length_stats:
        s = stats.seeking_length_stats
        print(f"  Seeking lengths:      min={s['min']}, p25={s['p25']:.0f}, "
              f"median={s['median']:.0f}, p75={s['p75']:.0f}, max={s['max']}")
    if stats.driving_length_stats:
        d = stats.driving_length_stats
        print(f"  Driving lengths:      min={d['min']}, p25={d['p25']:.0f}, "
              f"median={d['median']:.0f}, p75={d['p75']:.0f}, max={d['max']}")

    # Per-driver summary
    print(f"\n  Per-driver trajectory counts:")
    print(f"  {'Idx':<5} {'Plate':<12} {'Seeking':>8} {'Driving':>8} {'Ratio':>7}")
    print(f"  {'─'*42}")
    for idx in sorted(index_to_plate.keys()):
        n_seek = len(seeking.get(idx, []))
        n_drive = len(driving.get(idx, []))
        ratio = f"{n_drive/n_seek:.2f}" if n_seek > 0 else "N/A"
        plate = index_to_plate[idx]
        print(f"  {idx:<5} {plate:<12} {n_seek:>8} {n_drive:>8} {ratio:>7}")

    print(f"\n  Processing time: {stats.processing_time_seconds:.1f}s")

    # Save trajectories
    config.output_dir.mkdir(parents=True, exist_ok=True)

    seeking_path = config.output_dir / "seeking_trajs.pkl"
    with open(seeking_path, "wb") as f:
        pickle.dump(seeking, f)
    print(f"\n  Saved: {seeking_path}")

    driving_path = config.output_dir / "driving_trajs.pkl"
    with open(driving_path, "wb") as f:
        pickle.dump(driving, f)
    print(f"  Saved: {driving_path}")

    # Save calendar day mappings (for day-based pair sampling)
    seek_cal_path = config.output_dir / "seeking_calendar_days.pkl"
    with open(seek_cal_path, "wb") as f:
        pickle.dump(seeking_cal_days, f)
    print(f"  Saved: {seek_cal_path}")

    drive_cal_path = config.output_dir / "driving_calendar_days.pkl"
    with open(drive_cal_path, "wb") as f:
        pickle.dump(driving_cal_days, f)
    print(f"  Saved: {drive_cal_path}")

    cal_map_path = config.output_dir / "calendar_day_map.pkl"
    with open(cal_map_path, "wb") as f:
        pickle.dump(cal_day_map, f)
    print(f"  Saved: {cal_map_path}")
    print(f"  Calendar days tracked: {len(cal_day_map)} unique dates")

    # Profile features
    if not args.skip_profiles:
        print(f"\n{'=' * 65}")
        print("Phase 2B: Profile Feature Computation")
        print(f"{'=' * 65}")

        profile_data = compute_profile_features(
            seeking, driving, index_to_plate, calendar_days, config, bounds, progress
        )

        profile_path = config.output_dir / "profile_features.pkl"
        with open(profile_path, "wb") as f:
            pickle.dump(profile_data, f)
        print(f"\n  Saved: {profile_path}")

        # Print feature summary
        print(f"\n  Profile features for {len(profile_data['features'])} drivers:")
        print(f"  {'Idx':<4} {'Feature':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'─'*56}")
        features_arr = np.array([profile_data['features'][k]
                                  for k in sorted(profile_data['features'].keys())])
        for i, name in enumerate(profile_data['feature_names']):
            col = features_arr[:, i]
            print(f"  {i:<4} {name:<20} {col.mean():>10.2f} {col.std():>10.2f} "
                  f"{col.min():>10.2f} {col.max():>10.2f}")

        print(f"\n  Z-score normalization parameters:")
        mean = profile_data['normalization']['mean']
        std = profile_data['normalization']['std']
        for i, name in enumerate(profile_data['feature_names']):
            print(f"    {name:<20} mean={mean[i]:>10.4f}  std={std[i]:>10.4f}")

        # Home location coverage
        hlc = profile_data['home_loc_coverage']
        print(f"\n  Home location: {hlc['from_pickle']} pickle + "
              f"{hlc['from_fallback']} fallback = {hlc['total']} total")

    # Save metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config.to_dict(),
        "bounds": bounds.to_dict(),
        "driver_mapping": {str(k): v for k, v in index_to_plate.items()},
        "stats": stats.to_dict(),
        "calendar_day_map": {str(k): v for k, v in cal_day_map.items()},
        "n_calendar_days": len(cal_day_map),
    }
    metadata_path = config.output_dir / "extraction_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Saved: {metadata_path}")

    print(f"\n{'=' * 65}")
    print(f"All outputs saved to {config.output_dir}/")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
