"""Tests for the SF Cabspotting raw loader (Task 3.1)."""
import numpy as np

from famail_temporal.second_dataset.data.source_generation.sf_raw_loader import load_sf_raw


def _write_cab(d, name, rows):
    (d / f"new_{name}.txt").write_text(
        "\n".join(f"{lat} {lon} {occ} {t}" for lat, lon, occ, t in rows) + "\n"
    )


def test_load_sf_raw_schema_and_sorting(tmp_path):
    d = tmp_path / "cabspottingdata"
    d.mkdir()
    # Cabspotting files are newest-first; loader must sort ascending per driver.
    _write_cab(d, "alpha", [
        (37.78, -122.41, 0, 1213084700),
        (37.79, -122.42, 1, 1213084600),
    ])
    _write_cab(d, "bravo", [
        (37.75, -122.39, 1, 1213084650),
    ])

    df = load_sf_raw(str(d))

    assert list(df.columns) == ["driver_id", "lat", "lon", "occupancy", "time_utc"]
    assert set(df["occupancy"].unique()) <= {0, 1}
    assert df["driver_id"].nunique() == 2            # alpha, bravo -> 0, 1
    assert df["driver_id"].dtype.kind in "iu"
    # within each driver, time ascending
    for _, g in df.groupby("driver_id"):
        assert g["time_utc"].is_monotonic_increasing


def test_load_sf_raw_drops_invalid_coords(tmp_path):
    d = tmp_path / "cabspottingdata"
    d.mkdir()
    _write_cab(d, "x", [
        (37.78, -122.41, 0, 100),   # valid SF
        (0.0, 0.0, 0, 200),         # invalid (null island) -> dropped
        (40.0, -120.0, 1, 300),     # outside Bay Area window -> dropped
    ])

    df = load_sf_raw(str(d))

    assert len(df) == 1
    assert df.iloc[0]["lat"] == 37.78
