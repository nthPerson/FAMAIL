"""Tests for views/profile.py."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from famail_temporal.data.source_generation.views.profile import (
    compute_profile_features, zscore_normalize,
)
from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult,
)


def _df_with_midnight_records(plate="A", midnight_cells=None):
    midnight_cells = midnight_cells or [(5, 10), (5, 10), (6, 11)]
    rows = []
    for i, (x, y) in enumerate(midnight_cells):
        rows.append({
            "plate_id": plate, "x_grid": x, "y_grid": y,
            "time_bucket": 1, "hour": 0, "day_index": 1,
            "calendar_date": f"2016-07-0{i+4}",
            "seconds": 60 * i,
            "passenger_indicator": 0,
        })
    return pd.DataFrame(rows)


def test_home_x_y_mode_of_time_bucket_1_cells():
    df = _df_with_midnight_records("A", [(5, 10), (5, 10), (6, 11)])
    trajs = TrajectoriesResult()
    features = compute_profile_features(df, trajs)
    assert features["A"]["home_x"] == 5
    assert features["A"]["home_y"] == 10


def test_shift_start_end_5th_95th_percentile():
    df = pd.DataFrame([{
        "plate_id": "A", "x_grid": 5, "y_grid": 10,
        "time_bucket": tb, "hour": tb // 12, "day_index": 1,
        "calendar_date": "2016-07-04", "seconds": 0, "passenger_indicator": 0,
    } for tb in [10, 50, 100, 150, 200, 250, 288]])
    trajs = TrajectoriesResult()
    features = compute_profile_features(df, trajs)
    assert features["A"]["shift_start"] == pytest.approx(22, abs=15)
    assert features["A"]["shift_end"] == pytest.approx(276, abs=15)


def test_zscore_normalize_50_drivers():
    raw = np.arange(50 * 11, dtype=float).reshape(50, 11)
    normalized, mean, std = zscore_normalize(raw)
    assert normalized.shape == (50, 11)
    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-6)
