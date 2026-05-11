"""Hand-built synthetic raw-GPS fixture + expected outputs.

Two drivers × a handful of weekday records, all expected outputs hand-
computed in this file. Referenced from test_golden.py. When future changes
appear to alter output numerics, diff against this fixture's answers.
"""
from __future__ import annotations
import pickle
from pathlib import Path


def build_raw_fixture(output_dir: Path) -> None:
    data_07: dict = {
        "AAA": [
            ["AAA", 22.500, 113.800, 0,     1, "2016-07-04 00:00:00"],
            ["AAA", 22.500, 113.800, 60,    1, "2016-07-04 00:01:00"],
            ["AAA", 22.500, 113.800, 120,   0, "2016-07-04 00:02:00"],
            ["AAA", 22.500, 113.800, 180,   0, "2016-07-04 00:03:00"],
            ["AAA", 22.500, 113.800, 240,   0, "2016-07-04 00:04:00"],
            ["AAA", 22.500, 113.810, 300,   0, "2016-07-04 00:05:00"],
            ["AAA", 22.500, 113.810, 360,   1, "2016-07-04 00:06:00"],
            ["AAA", 22.500, 113.810, 420,   1, "2016-07-04 00:07:00"],
            ["AAA", 22.500, 113.810, 480,   0, "2016-07-04 00:08:00"],
        ],
        "BBB": [
            ["BBB", 22.600, 114.000, 0,     0, "2016-07-04 00:00:00"],
            ["BBB", 22.600, 114.000, 60,    1, "2016-07-04 00:01:00"],
            ["BBB", 22.600, 114.000, 120,   0, "2016-07-04 00:02:00"],
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "taxi_record_07_50drivers.pkl", "wb") as f:
        pickle.dump(data_07, f)
    with open(output_dir / "taxi_record_08_50drivers.pkl", "wb") as f:
        pickle.dump({}, f)
    with open(output_dir / "taxi_record_09_50drivers.pkl", "wb") as f:
        pickle.dump({}, f)


def expected_seeking_trajectories() -> dict[str, list]:
    return {
        "AAA": [
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 2, 2, 1],
                [1, 2, 2, 1],
            ],
        ],
        "BBB": [
            [
                [10, 20, 1, 1],
                [10, 20, 1, 1],
            ],
        ],
    }


def expected_pickup_count_at_AAA_endpoint() -> dict:
    return {(1, 2, 2, 1): (1, 0)}
