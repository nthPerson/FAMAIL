"""Tests for data.loader DataBundle dataclass."""
import dataclasses
import numpy as np
import pytest
import torch.nn as nn

from famail_temporal.data.active_mask import UnitIndexMap
from famail_temporal.data.loader import DataBundle
from famail_temporal.fairness.g0_power_basis import G0Function
from famail_temporal.fidelity.context import MultiStreamData


def test_databundle_is_frozen_dataclass():
    assert dataclasses.is_dataclass(DataBundle)
    mask = np.zeros((2, 2, 4), dtype=bool)
    umap = UnitIndexMap.from_mask(mask, grid_shape=(2, 2))
    bundle = DataBundle(
        pickup_3d=np.zeros((2, 2, 4), dtype=np.float32),
        dropoff_3d=np.zeros((2, 2, 4), dtype=np.float32),
        active_taxis_3d=np.zeros((2, 2, 4), dtype=np.float32),
        mask_3d=mask,
        unit_map=umap,
        n_hours_per_block=np.array([3, 6, 4, 11], dtype=np.int32),
        n_days=65,
        g0_func=G0Function(coefficients=np.zeros(4), d_min=0.01, d_max=10.0),
        hat_matrices={'I_minus_H_demo': np.eye(1), 'M': np.eye(1)},
        trajectories=[],
        multi_stream=MultiStreamData(
            driving_trajs={}, seeking_trajs={},
            profile_features={}, seeking_days={}, driving_days={},
        ),
        discriminator=nn.Identity(),
    )
    assert bundle.n_days == 65
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.n_days = 100


def test_databundle_is_kw_only():
    """kw_only=True prevents positional construction, guarding against
    12-field positional-argument footguns."""
    with pytest.raises(TypeError):
        DataBundle(
            np.zeros((2, 2, 4), dtype=np.float32),  # positional pickup_3d
            np.zeros((2, 2, 4), dtype=np.float32),
            np.zeros((2, 2, 4), dtype=np.float32),
            np.zeros((2, 2, 4), dtype=bool),
        )


@pytest.mark.slow
def test_databundle_load_real_data():
    """End-to-end DataBundle.load() — skip if raw data missing."""
    from famail_temporal import config

    required = [
        config.RAW_DATA_DIR / "pickup_dropoff_counts.pkl",
        config.RAW_DATA_DIR / "active_taxis_5x5_hourly.pkl",
        config.RAW_DATA_DIR / "cell_demographics.pkl",
        config.RAW_DATA_DIR / "grid_to_district_mapping.pkl",
    ]
    for path in required:
        if not path.exists():
            pytest.skip(f"Raw data missing: {path}")

    cache_files = list(config.CACHE_DIR.glob("*.pkl"))
    if not cache_files:
        pytest.skip("Cache empty — run preprocess first")

    bundle = DataBundle.load(max_trajectories=10, max_drivers=2)
    assert bundle.pickup_3d.shape == (*config.GRID_DIMS, config.T)
    assert bundle.unit_map.n_units >= config.MIN_TOTAL_ACTIVE_UNITS
    assert bundle.unit_map.n_units == bundle.hat_matrices['I_minus_H_demo'].shape[0]
