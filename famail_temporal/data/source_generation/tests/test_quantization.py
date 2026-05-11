"""Tests for quantization.py."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from famail_temporal.data.source_generation.quantization import (
    GlobalBounds, compute_global_bounds, gps_to_grid,
    seconds_to_time_bucket, seconds_to_hour, timestamp_to_day,
)


def test_compute_global_bounds():
    lat = pd.Series([22.5, 22.8, 22.6])
    lon = pd.Series([113.8, 114.5, 114.0])
    b = compute_global_bounds(lat, lon)
    assert b.lat_min == pytest.approx(22.5)
    assert b.lat_max == pytest.approx(22.8)
    assert b.lon_min == pytest.approx(113.8)
    assert b.lon_max == pytest.approx(114.5)


def test_gps_to_grid_returns_1_indexed():
    b = GlobalBounds(lat_min=22.5, lat_max=22.9, lon_min=113.8, lon_max=114.5)
    x, y = gps_to_grid(22.5, 113.8, b)
    assert (int(x), int(y)) == (1, 1)


def test_gps_to_grid_upper_corner_within_max():
    b = GlobalBounds(lat_min=22.5, lat_max=22.9, lon_min=113.8, lon_max=114.5)
    x, y = gps_to_grid(22.89, 114.49, b)
    assert 1 <= int(x) <= 48
    assert 1 <= int(y) <= 90


def test_gps_to_grid_vectorized():
    b = GlobalBounds(lat_min=22.5, lat_max=22.9, lon_min=113.8, lon_max=114.5)
    lats = np.array([22.5, 22.6, 22.8])
    lons = np.array([113.8, 114.0, 114.4])
    xs, ys = gps_to_grid(lats, lons, b)
    assert xs.shape == (3,)
    assert ys.shape == (3,)
    assert (xs >= 1).all() and (ys >= 1).all()


def test_seconds_to_time_bucket_midnight_is_1():
    assert int(seconds_to_time_bucket(0)) == 1
    assert int(seconds_to_time_bucket(60)) == 1
    assert int(seconds_to_time_bucket(4 * 60 + 59)) == 1


def test_seconds_to_time_bucket_first_hour_boundary():
    assert int(seconds_to_time_bucket(5 * 60)) == 2
    assert int(seconds_to_time_bucket(60 * 60 - 1)) == 12
    assert int(seconds_to_time_bucket(60 * 60)) == 13


def test_seconds_to_time_bucket_last_is_288():
    last_second = 24 * 60 * 60 - 1
    assert int(seconds_to_time_bucket(last_second)) == 288


def test_seconds_to_time_bucket_vectorized():
    arr = np.array([0, 60, 60 * 60, 23 * 60 * 60], dtype=int)
    out = seconds_to_time_bucket(arr)
    assert list(out) == [1, 1, 13, 277]


def test_seconds_to_hour():
    assert int(seconds_to_hour(0)) == 0
    assert int(seconds_to_hour(60 * 60)) == 1
    assert int(seconds_to_hour(23 * 60 * 60)) == 23


def test_timestamp_to_day_weekdays():
    assert timestamp_to_day("2016-07-04 08:00:00") == 1
    assert timestamp_to_day("2016-07-08 08:00:00") == 5


def test_timestamp_to_day_weekends_return_none():
    assert timestamp_to_day("2016-07-02 12:00:00") is None
    assert timestamp_to_day("2016-07-03 12:00:00") is None


def test_timestamp_to_day_bad_format():
    assert timestamp_to_day("not a date") is None
