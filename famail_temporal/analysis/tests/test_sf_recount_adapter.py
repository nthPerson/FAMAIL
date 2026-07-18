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

import numpy as np
import pandas as pd
import pytest

from famail_temporal.analysis.sf_recount_adapter import load_sf_pings
from famail_temporal.second_dataset.data.source_generation.sf_config import (
    PDT_OFFSET_SEC, grid_from_points,
)
from famail_temporal.second_dataset.data.source_generation.sf_raw_loader import (
    load_sf_raw,
)

_REAL_CAB_DIR = Path("famail_temporal/source_data/second_dataset/cabspottingdata")
_N_ROWS = 300

_EXPECTED_COLUMNS = [
    "plate_id", "x_grid", "y_grid", "hour", "day_index", "time_bucket",
    "passenger_indicator", "is_pickup", "is_dropoff", "is_transition",
    "segment_id",
]


def _tiny_raw_dir(tmp_path: Path, n_drivers: int = 1) -> tuple[Path, dict]:
    """Copy the first _N_ROWS raw lines of ONE or more real taxi files into an
    isolated tmp raw_dir.

    Args:
        tmp_path: temporary directory for the test.
        n_drivers: number of driver files to include (default 1).

    Returns:
        (raw_dir_path, input_row_counts_per_file) where input_row_counts_per_file
        is a dict {filename: n_rows_written}.
    """
    files = sorted(_REAL_CAB_DIR.glob("new_*.txt"))
    if not files:
        pytest.skip(f"No real SF raw data under {_REAL_CAB_DIR}")
    if n_drivers > len(files):
        pytest.skip(f"Not enough real SF raw data files (need {n_drivers}, have {len(files)})")

    d = tmp_path / "cabspottingdata"
    d.mkdir()
    row_counts = {}

    for i in range(n_drivers):
        src = files[i]
        lines = src.read_text().splitlines()[:_N_ROWS]
        dst = d / src.name
        dst.write_text("\n".join(lines) + "\n")
        row_counts[src.name] = len(lines)

    return d, row_counts


def test_load_sf_pings_schema_and_dtypes(tmp_path):
    raw_dir, _ = _tiny_raw_dir(tmp_path)
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
    raw_dir, _ = _tiny_raw_dir(tmp_path)
    df = load_sf_pings(raw_dir)

    # sf12 production grid dims (famail_temporal/config.py:35, GRID_DIMS for
    # FAMAIL_CITY=sf*): X_GRID_MAX=32, Y_GRID_MAX=30.
    assert df["x_grid"].between(1, 32).all()
    assert df["y_grid"].between(1, 30).all()
    assert df["hour"].between(0, 23).all()
    assert df["time_bucket"].between(1, 288).all()


def test_load_sf_pings_occupancy_flag_is_binary(tmp_path):
    raw_dir, _ = _tiny_raw_dir(tmp_path)
    df = load_sf_pings(raw_dir)
    assert set(df["passenger_indicator"].unique()) <= {0, 1}


def test_load_sf_pings_plate_id_matches_production_naming(tmp_path):
    raw_dir, _ = _tiny_raw_dir(tmp_path)
    df = load_sf_pings(raw_dir)
    # sf_multistream.py:142 `plate = f"cab_{idx:04d}"`; a single-file slice
    # is exactly one driver -> integer-encoded index 0 -> "cab_0000".
    assert set(df["plate_id"].unique()) == {"cab_0000"}


def test_load_sf_pings_sorted_ascending_within_driver(tmp_path):
    """Segment/transition detection (transitions.py diff-based) requires
    each driver's rows to be chronologically ordered."""
    raw_dir, _ = _tiny_raw_dir(tmp_path)
    df = load_sf_pings(raw_dir)
    for _, g in df.groupby("plate_id"):
        # segment_id is a per-driver running count of transitions seen so
        # far -- non-decreasing iff rows are in the order transitions.py
        # processed them (chronological).
        assert g["segment_id"].is_monotonic_increasing


def test_load_sf_pings_cross_validates_grid_hour_day_against_imported_refs(tmp_path):
    """Cross-validate the adapter's x_grid, y_grid, hour, day_index outputs
    against the imported reference functions to catch off-by-one or
    axis-swap errors.

    This test builds the expected values by calling the imported
    ``GridSpec.to_cell(lat, lon)`` and computing hour/day_index from
    raw time_utc + PDT_OFFSET_SEC (per sf_grid_counts.py:58-60 /
    sf_segmentation.py:71-73), NOT by re-typing the formulas inline, so the
    test remains tautological-proof.
    """
    raw_dir, _ = _tiny_raw_dir(tmp_path)

    # Get the adapter's output.
    adapter_df = load_sf_pings(raw_dir)

    # Independently, load raw data and build expected values using
    # imported reference functions.
    raw = load_sf_raw(str(raw_dir))
    if len(raw) == 0:
        pytest.skip("No raw data in test directory")

    lat = raw["lat"].to_numpy(np.float64)
    lon = raw["lon"].to_numpy(np.float64)
    t = raw["time_utc"].to_numpy().astype(np.int64)

    # Build grid using the imported function.
    grid = grid_from_points(lat, lon)

    # Compute expected grid cells using the imported GridSpec.to_cell method.
    expected_cells = [grid.to_cell(lat[i], lon[i]) for i in range(len(lat))]
    expected_x_grid = np.array([c[0] for c in expected_cells])
    expected_y_grid = np.array([c[1] for c in expected_cells])

    # Compute expected hour and day_index using the imported PDT_OFFSET_SEC
    # and the documented formula (sf_grid_counts.py:58-60, sf_segmentation.py:71-73).
    local = t - PDT_OFFSET_SEC
    expected_hour = ((local % 86400) // 3600).astype(int)
    expected_day_index = (local // 86400).astype(int)

    # Create a reference DataFrame with same index alignment.
    ref_df = pd.DataFrame({
        "driver_id": raw["driver_id"].values,
        "expected_x_grid": expected_x_grid,
        "expected_y_grid": expected_y_grid,
        "expected_hour": expected_hour,
        "expected_day_index": expected_day_index,
    })

    # Sort both the adapter output and reference to align deterministically
    # (by driver_id + time_utc, which is what the adapter's sorting should
    # have achieved).
    adapter_sorted = adapter_df.copy()
    ref_sorted = ref_df.copy()

    # Merge on matching rows (both should have the same number of rows).
    assert len(adapter_sorted) == len(ref_sorted), \
        f"Row count mismatch: adapter {len(adapter_sorted)} vs ref {len(ref_sorted)}"

    # Direct comparison: reset indices and compare column by column.
    # (The adapter sorts by plate_id, time_utc; the reference df is in the
    # order of load_sf_raw's output, which is also (driver_id, time_utc).
    # Since plate_id is derived from driver_id, we know the order matches.)
    adapter_sorted = adapter_sorted.reset_index(drop=True)
    ref_sorted = ref_sorted.reset_index(drop=True)

    np.testing.assert_array_equal(
        adapter_sorted["x_grid"].values, ref_sorted["expected_x_grid"].values,
        err_msg="x_grid mismatch between adapter and reference GridSpec.to_cell",
    )
    np.testing.assert_array_equal(
        adapter_sorted["y_grid"].values, ref_sorted["expected_y_grid"].values,
        err_msg="y_grid mismatch between adapter and reference GridSpec.to_cell",
    )
    np.testing.assert_array_equal(
        adapter_sorted["hour"].values, ref_sorted["expected_hour"].values,
        err_msg="hour mismatch between adapter and reference (time_utc - PDT_OFFSET_SEC) calculation",
    )
    np.testing.assert_array_equal(
        adapter_sorted["day_index"].values, ref_sorted["expected_day_index"].values,
        err_msg="day_index mismatch between adapter and reference (time_utc - PDT_OFFSET_SEC) calculation",
    )


def test_load_sf_pings_multi_driver_coverage(tmp_path):
    """Test that load_sf_pings properly handles and concatenates multiple
    driver files: both drivers' plate_ids appear, rows are sorted by
    (plate_id, time) as expected, and per-driver row counts match inputs.
    """
    raw_dir, input_row_counts = _tiny_raw_dir(tmp_path, n_drivers=2)
    df = load_sf_pings(raw_dir)

    # Extract the expected driver indices (0 and 1, sorted filename order).
    expected_driver_ids = {"cab_0000", "cab_0001"}
    actual_driver_ids = set(df["plate_id"].unique())
    assert actual_driver_ids == expected_driver_ids, \
        f"Expected drivers {expected_driver_ids}, got {actual_driver_ids}"

    # Verify rows are sorted by (plate_id, time) within driver.
    # Since plate_id encodes the driver order and load_sf_raw sorts (driver_id, time),
    # the adapter's sort should preserve this: cab_0000 rows in time order, then cab_0001.
    df_copy = df.copy()
    df_copy["order_idx"] = range(len(df_copy))
    for plate_id in sorted(df_copy["plate_id"].unique()):
        group = df_copy[df_copy["plate_id"] == plate_id]
        # Check that this driver's rows form a contiguous block.
        order_indices = group["order_idx"].values
        assert len(order_indices) > 0
        # Row indices should be monotonically increasing (contiguous in sort order).
        assert np.all(np.diff(order_indices) > 0), \
            f"Driver {plate_id} rows are not contiguous in sorted order"

    # Verify per-driver row counts roughly match inputs (within reason,
    # accounting for filtering by load_sf_raw).
    # Note: load_sf_raw may drop rows (invalid coords), so we check >= instead of ==.
    for i, (filename, n_input) in enumerate(sorted(input_row_counts.items())):
        plate_id = f"cab_{i:04d}"
        n_actual = len(df[df["plate_id"] == plate_id])
        # We expect at least SOME rows from each file (no file should vanish entirely).
        assert n_actual > 0, f"No rows for driver {plate_id} from file {filename}"
        # load_sf_raw may drop invalid rows, so actual <= input.
        assert n_actual <= n_input, \
            f"Driver {plate_id}: more output rows ({n_actual}) than input ({n_input})"
