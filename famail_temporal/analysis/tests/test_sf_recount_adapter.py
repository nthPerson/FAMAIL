"""TDD test for the SF ping-loader adapter (D1 Task 1,
``famail_temporal/analysis/sf_recount_adapter.py``).

Uses a REAL SF Cabspotting source file (gitignored, present on this dev
machine) truncated to its first ``_N_ROWS`` lines, mirroring the
skip-if-absent real-data pattern in
``famail_temporal/data/source_generation/tests/test_golden.py``
(``test_smoke_on_real_raw_if_present``). Kept intentionally tiny/CPU-only per
the task brief -- never loads the full 536-driver SF raw dataset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from famail_temporal.analysis.sf_recount_adapter import load_sf_pings

_REAL_CAB_DIR = Path("famail_temporal/source_data/second_dataset/cabspottingdata")
_N_ROWS = 300

_EXPECTED_COLUMNS = [
    "plate_id", "x_grid", "y_grid", "hour", "day_index", "time_bucket",
    "passenger_indicator", "is_pickup", "is_dropoff", "is_transition",
    "segment_id",
]


def _tiny_raw_dir(tmp_path: Path) -> Path:
    """Copy the first _N_ROWS raw lines of ONE real taxi file into an
    isolated tmp raw_dir -- one small file, never the full fleet."""
    files = sorted(_REAL_CAB_DIR.glob("new_*.txt"))
    if not files:
        pytest.skip(f"No real SF raw data under {_REAL_CAB_DIR}")
    src = files[0]
    lines = src.read_text().splitlines()[:_N_ROWS]
    d = tmp_path / "cabspottingdata"
    d.mkdir()
    (d / src.name).write_text("\n".join(lines) + "\n")
    return d


def test_load_sf_pings_schema_and_dtypes(tmp_path):
    raw_dir = _tiny_raw_dir(tmp_path)
    df = load_sf_pings(raw_dir)

    assert list(df.columns) == _EXPECTED_COLUMNS
    assert len(df) > 0

    # SZ's own es.df["plate_id"] is also pandas' string dtype (verified via
    # build_event_stream(Path("raw_data")).df.dtypes), not literal `object`.
    assert pd.api.types.is_string_dtype(df["plate_id"])
    for col in ["x_grid", "y_grid", "hour", "day_index", "time_bucket",
                "passenger_indicator", "segment_id"]:
        assert pd.api.types.is_integer_dtype(df[col]), col
    for col in ["is_pickup", "is_dropoff", "is_transition"]:
        assert pd.api.types.is_bool_dtype(df[col]), col


def test_load_sf_pings_grid_coords_within_sf12_bounds(tmp_path):
    raw_dir = _tiny_raw_dir(tmp_path)
    df = load_sf_pings(raw_dir)

    # sf12 production grid dims (famail_temporal/config.py:35, GRID_DIMS for
    # FAMAIL_CITY=sf*): X_GRID_MAX=32, Y_GRID_MAX=30.
    assert df["x_grid"].between(1, 32).all()
    assert df["y_grid"].between(1, 30).all()
    assert df["hour"].between(0, 23).all()
    assert df["time_bucket"].between(1, 288).all()


def test_load_sf_pings_occupancy_flag_is_binary(tmp_path):
    raw_dir = _tiny_raw_dir(tmp_path)
    df = load_sf_pings(raw_dir)
    assert set(df["passenger_indicator"].unique()) <= {0, 1}


def test_load_sf_pings_plate_id_matches_production_naming(tmp_path):
    raw_dir = _tiny_raw_dir(tmp_path)
    df = load_sf_pings(raw_dir)
    # sf_multistream.py:142 `plate = f"cab_{idx:04d}"`; a single-file slice
    # is exactly one driver -> integer-encoded index 0 -> "cab_0000".
    assert set(df["plate_id"].unique()) == {"cab_0000"}


def test_load_sf_pings_sorted_ascending_within_driver(tmp_path):
    """Segment/transition detection (transitions.py diff-based) requires
    each driver's rows to be chronologically ordered."""
    raw_dir = _tiny_raw_dir(tmp_path)
    df = load_sf_pings(raw_dir)
    for _, g in df.groupby("plate_id"):
        # segment_id is a per-driver running count of transitions seen so
        # far -- non-decreasing iff rows are in the order transitions.py
        # processed them (chronological).
        assert g["segment_id"].is_monotonic_increasing
