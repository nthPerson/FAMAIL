"""Tests for event_stream.py — the enriched DataFrame used as the single
source of truth across all views."""
from __future__ import annotations
import pickle
from pathlib import Path

import pandas as pd
import pytest

from famail_temporal.data.source_generation.event_stream import build_event_stream


def _write_pkl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


@pytest.fixture
def tiny_raw(tmp_path):
    path = tmp_path / "taxi_record_07_50drivers.pkl"
    _write_pkl(path, {
        "A": [
            ["A", 22.5, 113.8, 0,    1, "2016-07-04 00:00:00"],
            ["A", 22.5, 113.8, 60,   1, "2016-07-04 00:01:00"],
            ["A", 22.5, 113.8, 120,  0, "2016-07-04 00:02:00"],
            ["A", 22.5, 113.8, 180,  0, "2016-07-04 00:03:00"],
            ["A", 22.5, 113.8, 240,  0, "2016-07-04 00:04:00"],
            ["A", 22.5, 113.8, 300,  0, "2016-07-04 00:05:00"],
            ["A", 22.5, 113.8, 360,  1, "2016-07-04 00:06:00"],
            ["A", 22.5, 113.8, 420,  1, "2016-07-04 00:07:00"],
            ["A", 22.5, 113.8, 480,  0, "2016-07-04 00:08:00"],
        ],
        "B": [
            ["B", 22.6, 114.0, 0,    0, "2016-07-04 00:00:00"],
            ["B", 22.6, 114.0, 60,   1, "2016-07-04 00:01:00"],
            ["B", 22.6, 114.0, 120,  0, "2016-07-04 00:02:00"],
        ],
    })
    _write_pkl(tmp_path / "taxi_record_08_50drivers.pkl", {})
    _write_pkl(tmp_path / "taxi_record_09_50drivers.pkl", {})
    return tmp_path


def test_build_event_stream_returns_dataframe(tiny_raw):
    es = build_event_stream(tiny_raw)
    assert isinstance(es.df, pd.DataFrame)
    for col in ("plate_id", "x_grid", "y_grid", "time_bucket",
                "hour", "day_index", "is_pickup", "is_dropoff",
                "segment_id", "passenger_indicator"):
        assert col in es.df.columns


def test_build_event_stream_drops_weekends(tmp_path):
    _write_pkl(tmp_path / "taxi_record_07_50drivers.pkl", {
        "A": [
            ["A", 22.5, 113.8, 0, 0, "2016-07-02 12:00:00"],
            ["A", 22.5, 113.8, 0, 0, "2016-07-04 12:00:00"],
        ],
    })
    _write_pkl(tmp_path / "taxi_record_08_50drivers.pkl", {})
    _write_pkl(tmp_path / "taxi_record_09_50drivers.pkl", {})
    es = build_event_stream(tmp_path)
    assert len(es.df) == 1
    assert es.df.iloc[0]["day_index"] == 1


def test_build_event_stream_is_sorted_per_driver(tiny_raw):
    es = build_event_stream(tiny_raw)
    for plate, group in es.df.groupby("plate_id"):
        ts = list(group["timestamp"])
        assert ts == sorted(ts)


def test_build_event_stream_has_correct_transitions(tiny_raw):
    es = build_event_stream(tiny_raw)
    A = es.df[es.df["plate_id"] == "A"].reset_index(drop=True)
    assert A.loc[2, "is_dropoff"] == True
    assert A.loc[6, "is_pickup"] == True
    assert A.loc[8, "is_dropoff"] == True
    assert A["is_pickup"].sum() == 1
    assert A["is_dropoff"].sum() == 2


def test_build_event_stream_computes_n_days(tiny_raw):
    es = build_event_stream(tiny_raw)
    assert es.n_days >= 1


def test_build_event_stream_computes_global_bounds(tiny_raw):
    es = build_event_stream(tiny_raw)
    assert es.bounds.lat_min == pytest.approx(22.5)
    assert es.bounds.lat_max == pytest.approx(22.6)
