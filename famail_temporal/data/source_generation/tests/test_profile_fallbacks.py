"""Tests for the home_x/y fallback cascade."""
from __future__ import annotations
import pandas as pd
import pytest

from famail_temporal.data.source_generation.views.profile import (
    compute_home_xy_with_fallback,
)


def _event_df(tb_cells: list[tuple[int, int, int]]):
    return pd.DataFrame([
        {"plate_id": "A", "x_grid": x, "y_grid": y,
         "time_bucket": tb, "hour": 0, "day_index": 1,
         "calendar_date": "2016-07-04", "seconds": 0, "passenger_indicator": 0}
        for (tb, x, y) in tb_cells
    ])


def test_home_uses_tb_1_records_when_present():
    df = _event_df([(1, 5, 10), (1, 5, 10), (50, 99, 99)])
    result = compute_home_xy_with_fallback(df)
    assert result["home_x"] == 5 and result["home_y"] == 10
    assert result["fallback_used"] == "none"


def test_home_falls_back_to_first_hour_when_no_tb_1():
    df = _event_df([(5, 7, 20), (10, 7, 20), (50, 99, 99)])
    result = compute_home_xy_with_fallback(df)
    assert result["home_x"] == 7 and result["home_y"] == 20
    assert result["fallback_used"] == "first_hour"


def test_home_falls_back_to_all_records_when_no_first_hour():
    df = _event_df([(50, 3, 4), (50, 3, 4), (200, 99, 99)])
    result = compute_home_xy_with_fallback(df)
    assert result["home_x"] == 3 and result["home_y"] == 4
    assert result["fallback_used"] == "all_records"


def test_home_fallback_empty_driver_raises():
    df = _event_df([])
    with pytest.raises(ValueError, match="no records"):
        compute_home_xy_with_fallback(df)
