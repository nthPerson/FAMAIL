"""CLI entry point for multi-stream dataset generation.

Usage:
    python -m discriminator.multi_stream.dataset_generation
    python -m discriminator.multi_stream.dataset_generation --positive-pairs 10000 --negative-pairs 10000
    python -m discriminator.multi_stream.dataset_generation --n-trajs 5 --min-trajs-per-day 3
"""

import argparse
from pathlib import Path

from .config import MultiStreamGenerationConfig
from .generation import (
    generate_multi_stream_dataset,
    load_multi_stream_data,
    find_usable_days,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-stream training dataset (Ren-aligned day-based sampling)"
    )
    parser.add_argument("--extracted-data-dir", type=str, default=None,
                        help="Path to extracted data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for dataset files")
    parser.add_argument("--positive-pairs", type=int, default=5000)
    parser.add_argument("--negative-pairs", type=int, default=5000)
    parser.add_argument("--identical-ratio", type=float, default=0.1,
                        help="Fraction of positive pairs that are identical (default: 0.1)")
    parser.add_argument("--n-trajs", type=int, default=5,
                        help="Trajectories per stream per branch (default: 5)")
    parser.add_argument("--min-trajs-per-day", type=int, default=5,
                        help="Minimum trajs in both streams for day to be usable (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seeking-fixed-length", type=int, default=None,
                        help="Fixed sequence length for seeking trajectories (truncate/pad)")
    parser.add_argument("--driving-fixed-length", type=int, default=None,
                        help="Fixed sequence length for driving trajectories (truncate/pad)")
    parser.add_argument("--profile-noise", type=float, default=0.0,
                        help="Gaussian noise std on profile features (default: 0)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze usable days, don't generate dataset")
    args = parser.parse_args()

    config = MultiStreamGenerationConfig(
        positive_pairs=args.positive_pairs,
        negative_pairs=args.negative_pairs,
        identical_pair_ratio=args.identical_ratio,
        n_trajs_per_stream=args.n_trajs,
        min_trajs_per_day=args.min_trajs_per_day,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        profile_noise_std=args.profile_noise,
    )
    if args.seeking_fixed_length:
        config.seeking_padding = "fixed_length"
        config.seeking_fixed_length = args.seeking_fixed_length
    if args.driving_fixed_length:
        config.driving_padding = "fixed_length"
        config.driving_fixed_length = args.driving_fixed_length
    if args.extracted_data_dir:
        config.extracted_data_dir = Path(args.extracted_data_dir)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    print("=" * 65)
    print("Phase 4: Multi-Stream Dataset Generation")
    print("=" * 65)
    print(f"  Config: n_trajs={config.n_trajs_per_stream}, "
          f"min_trajs_per_day={config.min_trajs_per_day}")
    print(f"  Pairs: {config.positive_pairs} positive "
          f"({config.identical_pair_ratio:.0%} identical), "
          f"{config.negative_pairs} negative")
    print()

    # Load and analyze
    print("Loading extracted data...")
    seeking_by_day, driving_by_day, profile_features, cal_day_map, index_to_plate = \
        load_multi_stream_data(config.extracted_data_dir)

    print(f"  Drivers: {len(profile_features)}")
    print(f"  Calendar days: {len(cal_day_map)}")
    print(f"  Profile features dim: {next(iter(profile_features.values())).shape[0]}")
    print()

    # Usable days analysis
    usable = find_usable_days(seeking_by_day, driving_by_day, config)
    total_usable = sum(len(d) for d in usable.values())
    multi_day = sum(1 for d in usable.values() if len(d) >= 2)

    print(f"Usable days (min {config.min_trajs_per_day} trajs/stream/day):")
    print(f"  Drivers with usable days: {len(usable)} / {len(profile_features)}")
    print(f"  Drivers with 2+ usable days: {multi_day}")
    print(f"  Total usable driver-days: {total_usable}")
    print()
    print(f"  {'Driver':<8} {'Plate':<12} {'Usable Days':>12} {'Seek Trajs':>11} {'Drive Trajs':>12}")
    print(f"  {'─' * 57}")
    for driver in sorted(usable.keys()):
        n_days = len(usable[driver])
        n_seek = sum(len(seeking_by_day[driver].get(d, [])) for d in usable[driver])
        n_drive = sum(len(driving_by_day[driver].get(d, [])) for d in usable[driver])
        plate = index_to_plate.get(driver, "?")
        print(f"  {driver:<8} {plate:<12} {n_days:>12} {n_seek:>11} {n_drive:>12}")

    if args.analyze_only:
        print("\n  --analyze-only: skipping generation")
        return

    if not usable:
        print("\n  ERROR: No usable driver-days found. Try lowering --min-trajs-per-day")
        return

    if multi_day == 0:
        print("\n  WARNING: No drivers with 2+ usable days. "
              "All positive pairs will be identical.")

    # Generate
    print(f"\n{'=' * 65}")
    print("Generating pairs...")

    last_pct = [-1]
    def progress(stage, pct):
        pct_int = int(pct * 100)
        if pct_int > last_pct[0]:
            last_pct[0] = pct_int
            print(f"\r  [{pct_int:3d}%] {stage}", end="", flush=True)
            if pct_int == 100:
                print()

    metadata = generate_multi_stream_dataset(config, progress)

    # Print results
    print(f"\n{'─' * 65}")
    print("Generation Summary")
    print(f"{'─' * 65}")
    pc = metadata["pair_counts"]
    sc = metadata["split_counts"]
    si = metadata.get("shape_info", {})
    print(f"  Positive pairs:  {pc['positive']} ({pc['identical']} identical)")
    print(f"  Negative pairs:  {pc['negative']}")
    print(f"  Total pairs:     {pc['total']}")
    print(f"\n  Train: {sc.get('train', 0)}, Val: {sc.get('val', 0)}, Test: {sc.get('test', 0)}")
    if si:
        print(f"\n  Shapes per pair:")
        print(f"    Seeking:  [{si['n_trajs_per_stream']}, {si['seeking_length']}, 4]")
        print(f"    Driving:  [{si['n_trajs_per_stream']}, {si['driving_length']}, 4]")
        print(f"    Profile:  [{si['profile_dim']}]")

    ac = metadata["agent_coverage"]
    print(f"\n  Agent coverage: {ac['positive_agents']} in positives, "
          f"{ac['negative_agents']} in negatives "
          f"(of {ac['total_available']} available)")
    print(f"\n  Generation time: {metadata['generation_time_seconds']:.1f}s")
    print(f"  Output: {config.output_dir}/")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
