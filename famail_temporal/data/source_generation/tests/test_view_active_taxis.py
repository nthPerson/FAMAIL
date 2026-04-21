"""Tests for views/active_taxis.py."""
from __future__ import annotations
import pandas as pd

from famail_temporal.data.source_generation.views.active_taxis import (
    build_active_taxis_counts,
)


def _row(plate, x, y, hour, day, passenger=0):
    return {
        "plate_id": plate, "x_grid": x, "y_grid": y,
        "hour": hour, "day_index": day,
        "passenger_indicator": passenger,
    }


def test_empty_returns_empty_dict():
    df = pd.DataFrame(columns=[
        "plate_id", "x_grid", "y_grid", "hour",
        "day_index", "passenger_indicator",
    ])
    assert build_active_taxis_counts(df) == {}


def test_single_empty_driver_counts_in_5x5_neighborhood():
    df = pd.DataFrame([_row("A", 10, 10, 5, 1, passenger=0)])
    out = build_active_taxis_counts(df)
    assert out[(10, 10, 5, 1)] == 1
    assert out[(8, 8, 5, 1)] == 1
    assert out[(12, 12, 5, 1)] == 1


def test_occupied_only_driver_not_counted():
    df = pd.DataFrame([_row("B", 10, 10, 5, 1, passenger=1)])
    assert build_active_taxis_counts(df) == {}


def test_driver_with_any_empty_ping_counts_once():
    df = pd.DataFrame([
        _row("A", 10, 10, 5, 1, passenger=1),
        _row("A", 10, 10, 5, 1, passenger=0),
        _row("A", 10, 10, 5, 1, passenger=1),
    ])
    out = build_active_taxis_counts(df)
    assert out[(10, 10, 5, 1)] == 1


def test_two_distinct_drivers_count_as_two():
    df = pd.DataFrame([
        _row("A", 10, 10, 5, 1, passenger=0),
        _row("B", 10, 10, 5, 1, passenger=0),
    ])
    assert build_active_taxis_counts(df)[(10, 10, 5, 1)] == 2


def test_different_hours_independent():
    df = pd.DataFrame([
        _row("A", 10, 10, 5, 1, passenger=0),
        _row("A", 10, 10, 6, 1, passenger=0),
    ])
    out = build_active_taxis_counts(df)
    assert out[(10, 10, 5, 1)] == 1
    assert out[(10, 10, 6, 1)] == 1


def test_neighborhood_edge_of_grid_clamped():
    df = pd.DataFrame([_row("A", 1, 1, 5, 1, passenger=0)])
    out = build_active_taxis_counts(df)
    assert out[(1, 1, 5, 1)] == 1
    assert out[(3, 3, 5, 1)] == 1
    for (x, y, _, _) in out.keys():
        assert x >= 1 and y >= 1
