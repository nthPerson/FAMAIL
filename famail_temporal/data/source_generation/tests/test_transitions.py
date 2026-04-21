"""Tests for transition detection."""
from __future__ import annotations
import pandas as pd

from famail_temporal.data.source_generation.transitions import (
    add_transition_columns, assign_segment_ids,
)


def _make_driver_df(plate: str, passenger_seq: list[int]) -> pd.DataFrame:
    n = len(passenger_seq)
    return pd.DataFrame({
        "plate_id": [plate] * n,
        "timestamp": [f"2016-07-04 00:00:{i:02d}" for i in range(n)],
        "passenger_indicator": passenger_seq,
    })


def test_add_transition_columns_detects_pickups_and_dropoffs():
    df = _make_driver_df("A", [1, 1, 0, 0, 0, 0, 1, 1, 0])
    out = add_transition_columns(df)
    assert out["is_pickup"].tolist()  == [False, False, False, False, False, False, True,  False, False]
    assert out["is_dropoff"].tolist() == [False, False, True,  False, False, False, False, False, True]


def test_add_transition_columns_per_driver():
    dfA = _make_driver_df("A", [1, 0])
    dfB = _make_driver_df("B", [0, 1])
    df = pd.concat([dfA, dfB], ignore_index=True)
    out = add_transition_columns(df)
    assert out.loc[0, "is_dropoff"] == False
    assert out.loc[1, "is_dropoff"] == True
    assert out.loc[2, "is_pickup"] == False
    assert out.loc[3, "is_pickup"] == True


def test_assign_segment_ids_increments_after_each_transition():
    df = _make_driver_df("A", [1, 1, 0, 0, 0, 0, 1, 1, 0])
    df = add_transition_columns(df)
    df = assign_segment_ids(df)
    assert df["segment_id"].tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2]


def test_assign_segment_ids_per_driver_independent():
    dfA = _make_driver_df("A", [1, 1, 0, 0, 1])
    dfB = _make_driver_df("B", [0, 1, 1, 0])
    df = pd.concat([dfA, dfB], ignore_index=True)
    df = add_transition_columns(df)
    df = assign_segment_ids(df)
    a_rows = df[df["plate_id"] == "A"]["segment_id"].tolist()
    b_rows = df[df["plate_id"] == "B"]["segment_id"].tolist()
    assert a_rows == [0, 0, 0, 1, 1]
    assert b_rows == [0, 0, 1, 1]
