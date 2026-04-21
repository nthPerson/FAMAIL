"""Tests for views/trajectories.py."""
from __future__ import annotations
import pandas as pd

from famail_temporal.data.source_generation.views.trajectories import (
    build_trajectories, build_driver_index_mapping,
)


def _event_df():
    records = [
        {"plate_id": "A", "x_grid": 5, "y_grid": 10, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 1, "is_pickup": False, "is_dropoff": False, "segment_id": 0,
         "timestamp": "2016-07-04 00:00:00"},
        {"plate_id": "A", "x_grid": 5, "y_grid": 11, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 1, "is_pickup": False, "is_dropoff": False, "segment_id": 0,
         "timestamp": "2016-07-04 00:01:00"},
        {"plate_id": "A", "x_grid": 6, "y_grid": 11, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": True, "segment_id": 0,
         "timestamp": "2016-07-04 00:02:00"},
        {"plate_id": "A", "x_grid": 6, "y_grid": 12, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": False, "segment_id": 1,
         "timestamp": "2016-07-04 00:03:00"},
        {"plate_id": "A", "x_grid": 7, "y_grid": 12, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": False, "segment_id": 1,
         "timestamp": "2016-07-04 00:04:00"},
        {"plate_id": "A", "x_grid": 7, "y_grid": 13, "time_bucket": 2, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": False, "segment_id": 1,
         "timestamp": "2016-07-04 00:05:00"},
        {"plate_id": "A", "x_grid": 8, "y_grid": 13, "time_bucket": 2, "day_index": 1,
         "passenger_indicator": 1, "is_pickup": True, "is_dropoff": False, "segment_id": 1,
         "timestamp": "2016-07-04 00:06:00"},
        {"plate_id": "A", "x_grid": 8, "y_grid": 14, "time_bucket": 2, "day_index": 1,
         "passenger_indicator": 1, "is_pickup": False, "is_dropoff": False, "segment_id": 2,
         "timestamp": "2016-07-04 00:07:00"},
        {"plate_id": "A", "x_grid": 9, "y_grid": 14, "time_bucket": 2, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": True, "segment_id": 2,
         "timestamp": "2016-07-04 00:08:00"},
    ]
    return pd.DataFrame(records)


def test_driver_mapping_is_lexicographic():
    df = pd.DataFrame({"plate_id": ["Z", "A", "M"]})
    mapping = build_driver_index_mapping(df)
    assert mapping["plate_to_idx"] == {"A": 0, "M": 1, "Z": 2}
    assert mapping["idx_to_plate"] == {0: "A", 1: "M", 2: "Z"}


def test_build_trajectories_extracts_seeking_and_driving():
    df = _event_df()
    result = build_trajectories(df)
    A_seeking = result.seeking_by_plate.get("A", [])
    A_driving = result.driving_by_plate.get("A", [])
    assert len(A_seeking) == 1
    assert len(A_driving) == 2


def test_seeking_state_minus_one_is_pickup_cell():
    df = _event_df()
    result = build_trajectories(df)
    A_seek0 = result.seeking_by_plate["A"][0]
    assert A_seek0[-1] == [8, 13, 2, 1]


def test_driving_state_minus_one_is_dropoff_cell():
    df = _event_df()
    result = build_trajectories(df)
    A_drv0 = result.driving_by_plate["A"][0]
    assert A_drv0[-1] == [6, 11, 1, 1]


def test_seeking_dates_sidecar_is_parallel_to_trajectories():
    """Per-trajectory calendar dates must be emitted alongside trajectories,
    parallel in order and length — discriminator pair-sampling depends on this."""
    df = _event_df()
    df = df.copy()
    # event_stream would normally populate calendar_date; build_trajectories
    # has to honor whatever's already on the DataFrame.
    df["calendar_date"] = df["timestamp"].str[:10]
    result = build_trajectories(df)
    assert len(result.seeking_dates_by_plate["A"]) == len(
        result.seeking_by_plate["A"]
    )
    assert result.seeking_dates_by_plate["A"][0] == "2016-07-04"


def test_driving_dates_sidecar_is_parallel_to_trajectories():
    df = _event_df()
    df = df.copy()
    df["calendar_date"] = df["timestamp"].str[:10]
    result = build_trajectories(df)
    assert len(result.driving_dates_by_plate["A"]) == len(
        result.driving_by_plate["A"]
    )
    # Both driving trajectories are from 2016-07-04.
    assert result.driving_dates_by_plate["A"] == ["2016-07-04", "2016-07-04"]


def test_min_length_filter_drops_length_1_segments():
    df = pd.DataFrame([{
        "plate_id": "A", "x_grid": 5, "y_grid": 10, "time_bucket": 1, "day_index": 1,
        "passenger_indicator": 1, "is_pickup": True, "is_dropoff": False,
        "segment_id": 0, "timestamp": "2016-07-04 00:00:00",
    }])
    result = build_trajectories(df)
    assert result.seeking_by_plate.get("A", []) == []


def test_incomplete_trailing_segment_dropped():
    df = pd.DataFrame([
        {"plate_id": "A", "x_grid": 5, "y_grid": 10, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": False,
         "segment_id": 0, "timestamp": "2016-07-04 00:00:00"},
        {"plate_id": "A", "x_grid": 5, "y_grid": 11, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": False,
         "segment_id": 0, "timestamp": "2016-07-04 00:01:00"},
    ])
    result = build_trajectories(df)
    assert result.seeking_by_plate.get("A", []) == []
    assert result.driving_by_plate.get("A", []) == []
