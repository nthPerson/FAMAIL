"""Tests for views/pickup_dropoff.py."""
from __future__ import annotations
import pandas as pd

from famail_temporal.data.source_generation.views.pickup_dropoff import (
    build_pickup_dropoff_counts,
)


def _make_event_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_empty_df_returns_empty_dict():
    df = _make_event_df([])
    out = build_pickup_dropoff_counts(df)
    assert out == {}


def test_single_pickup_contributes_one_pickup_zero_dropoff():
    df = _make_event_df([
        {"x_grid": 5, "y_grid": 10, "time_bucket": 20, "day_index": 1,
         "is_pickup": True, "is_dropoff": False},
    ])
    out = build_pickup_dropoff_counts(df)
    assert out == {(5, 10, 20, 1): (1, 0)}


def test_multiple_events_aggregate_per_key():
    df = _make_event_df([
        {"x_grid": 5, "y_grid": 10, "time_bucket": 20, "day_index": 1,
         "is_pickup": True, "is_dropoff": False},
        {"x_grid": 5, "y_grid": 10, "time_bucket": 20, "day_index": 1,
         "is_pickup": True, "is_dropoff": False},
        {"x_grid": 5, "y_grid": 10, "time_bucket": 20, "day_index": 1,
         "is_pickup": False, "is_dropoff": True},
    ])
    out = build_pickup_dropoff_counts(df)
    assert out == {(5, 10, 20, 1): (2, 1)}
