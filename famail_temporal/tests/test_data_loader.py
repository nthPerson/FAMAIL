"""Tests for data.loader DataBundle dataclass."""
import dataclasses
import pickle
import numpy as np
import pytest
import torch.nn as nn

from famail_temporal.data.active_mask import UnitIndexMap
from famail_temporal.data.loader import (
    DataBundle, _resolve_driver_id, _load_driver_index_mapping,
)
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


def test_resolve_driver_id_passes_int_through():
    """If the raw key is already an int, return it unchanged regardless of mapping."""
    assert _resolve_driver_id(7, plate_to_idx={}) == 7
    assert _resolve_driver_id(7, plate_to_idx={"粤B123": 0}) == 7


def test_resolve_driver_id_returns_raw_key_when_mapping_empty():
    """No mapping available → return the raw key unchanged (tests/legacy path)."""
    assert _resolve_driver_id("粤B123", plate_to_idx={}) == "粤B123"


def test_resolve_driver_id_converts_plate_to_int_via_mapping():
    """Primary path: plate string → int driver_idx via the sidecar mapping."""
    mapping = {"粤B123": 0, "粤B456": 1}
    assert _resolve_driver_id("粤B123", plate_to_idx=mapping) == 0
    assert _resolve_driver_id("粤B456", plate_to_idx=mapping) == 1


def test_resolve_driver_id_raises_on_missing_mapping_entry():
    """Plate in data but not in mapping = drift between the two files.
    Surface the drift immediately rather than produce silently wrong downstream
    keys."""
    mapping = {"粤B123": 0}
    with pytest.raises(KeyError, match="missing from driver_index_mapping"):
        _resolve_driver_id("粤B999", plate_to_idx=mapping)


def test_load_driver_index_mapping_absent_returns_empty(tmp_path, monkeypatch):
    """If driver_index_mapping.pkl isn't present, return {} (don't crash)."""
    from famail_temporal import config
    monkeypatch.setattr(config, "SOURCE_DATA_DIR", tmp_path)
    assert _load_driver_index_mapping() == {}


def test_load_driver_index_mapping_round_trips(tmp_path, monkeypatch):
    """If the mapping file is present, load it and return its contents."""
    from famail_temporal import config
    monkeypatch.setattr(config, "SOURCE_DATA_DIR", tmp_path)
    payload = {
        "plate_to_idx": {"粤B123": 0, "粤B456": 1},
        "idx_to_plate": {0: "粤B123", 1: "粤B456"},
    }
    path = tmp_path / "driver_index_mapping.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    loaded = _load_driver_index_mapping()
    assert loaded["plate_to_idx"] == payload["plate_to_idx"]


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
        config.SOURCE_DATA_DIR / "pickup_dropoff_counts.pkl",
        config.SOURCE_DATA_DIR / "active_taxis_5x5_hourly.pkl",
        config.SOURCE_DATA_DIR / "cell_demographics.pkl",
        config.SOURCE_DATA_DIR / "grid_to_district_mapping.pkl",
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
    assert bundle.unit_map.n_units == bundle.hat_matrices['X_demo'].shape[0]

    # Print bundle statistics for researcher inspection
    print(f"\n  Grid dims: {bundle.pickup_3d.shape}")
    print(f"  N active units: {bundle.unit_map.n_units}")
    print(f"  Units per block: {bundle.unit_map.units_per_block.tolist()}")
    print(f"  N trajectories: {len(bundle.trajectories)}")
    print(f"  N days: {bundle.n_days}")
    print(f"  g0 range: [{bundle.g0_func.d_min:.2f}, {bundle.g0_func.d_max:.2f}]")
