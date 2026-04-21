"""Tests for raw_loader.py."""
from __future__ import annotations
import pickle
from pathlib import Path

import pandas as pd
import pytest

from famail_temporal.data.source_generation.raw_loader import (
    load_raw_file, concat_raw_records,
)


def _write_pkl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def test_load_raw_file_flat_structure(tmp_path):
    path = tmp_path / "flat.pkl"
    _write_pkl(path, {
        "PLATE_A": [
            ["PLATE_A", 22.5, 114.0, 0, 0, "2016-07-01 00:00:00"],
            ["PLATE_A", 22.5, 114.0, 60, 1, "2016-07-01 00:01:00"],
        ],
    })
    df = load_raw_file(path)
    assert len(df) == 2
    assert list(df.columns) == [
        "plate_id", "latitude", "longitude", "seconds",
        "passenger_indicator", "timestamp",
    ]
    assert df.iloc[0]["plate_id"] == "PLATE_A"
    assert df.iloc[0]["passenger_indicator"] == 0
    assert df.iloc[1]["passenger_indicator"] == 1


def test_load_raw_file_nested_day_lists(tmp_path):
    path = tmp_path / "nested.pkl"
    _write_pkl(path, {
        "PLATE_B": [
            [["PLATE_B", 22.5, 114.0, 0, 0, "2016-07-01 00:00:00"]],
            [["PLATE_B", 22.5, 114.0, 60, 1, "2016-07-02 00:01:00"]],
        ],
    })
    df = load_raw_file(path)
    assert len(df) == 2


def test_load_raw_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_file(tmp_path / "does_not_exist.pkl")


def test_load_raw_file_rejects_bad_structure(tmp_path):
    path = tmp_path / "bad.pkl"
    _write_pkl(path, ["not a dict"])
    with pytest.raises(ValueError, match="expected dict"):
        load_raw_file(path)


def test_concat_raw_records(tmp_path):
    p1 = tmp_path / "a.pkl"
    p2 = tmp_path / "b.pkl"
    _write_pkl(p1, {"A": [["A", 22.5, 114.0, 0, 0, "2016-07-01 00:00:00"]]})
    _write_pkl(p2, {"B": [["B", 22.5, 114.0, 0, 0, "2016-08-01 00:00:00"]]})

    df = concat_raw_records([p1, p2])
    assert len(df) == 2
    assert set(df["plate_id"]) == {"A", "B"}
