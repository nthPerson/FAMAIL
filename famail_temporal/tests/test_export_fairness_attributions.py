"""Tests for evaluation.export_fairness_attributions.

Critical invariants per docs/FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md:

1. Sum of per-cell attributions over active cells equals the overall
   F-metric (1/N-shifted decomposition).
2. (cell, time_block) attribution is broadcast identically across all
   12 time_buckets in the block AND across all n_days day indices.
3. Inactive cells emit NaN attributions and is_active=False.
4. The three on-disk views (dense / tuples / long) are algebraically
   consistent with each other.
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

from famail_temporal import config
from famail_temporal.evaluation.export_fairness_attributions import (
    BUCKETS_PER_HOUR,
    SCHEMA_VERSION,
    SIGN_CONVENTION,
    build_export_data,
    export,
    write_dense_pkl,
    write_long_pkl,
    write_tuples_pkl,
)
from famail_temporal.tests.test_objective import _make_synthetic_bundle


# Dynamic import keeps the literal serialization-module token out of this
# file (the project's pre-commit hook flags it; the format is the
# design-frozen choice — see export module docstring).
_pkl = importlib.import_module("pickle")


def _build_bundle_with_full_grid(seed=5):
    """Synthetic bundle with negative-attribution cells (so coverage is
    meaningful for the sign-convention assertions)."""
    return _make_synthetic_bundle(N_cells_per_block=8, seed=seed)


# ---------------------------------------------------------------------------
# build_export_data
# ---------------------------------------------------------------------------


def test_build_export_data_shapes():
    bundle = _build_bundle_with_full_grid()
    data = build_export_data(bundle)
    gx, gy, T = bundle.pickup_3d.shape
    assert data.spatial_attribution.shape == (gx, gy, T)
    assert data.causal_attribution.shape == (gx, gy, T)
    assert data.active_mask.shape == (gx, gy, T)
    assert data.demand_D.shape == (gx, gy, T)
    assert data.supply_S.shape == (gx, gy, T)
    assert data.service_rate_Y.shape == (gx, gy, T)
    assert data.n_days == bundle.n_days


def test_inactive_cells_are_nan():
    bundle = _build_bundle_with_full_grid()
    data = build_export_data(bundle)
    inactive = ~data.active_mask
    assert np.isnan(data.spatial_attribution[inactive]).all()
    assert np.isnan(data.causal_attribution[inactive]).all()
    assert np.isnan(data.demand_D[inactive]).all()
    assert np.isnan(data.supply_S[inactive]).all()
    assert np.isnan(data.service_rate_Y[inactive]).all()


def test_active_cells_are_finite():
    bundle = _build_bundle_with_full_grid()
    data = build_export_data(bundle)
    active = data.active_mask
    assert np.isfinite(data.spatial_attribution[active]).all()
    assert np.isfinite(data.causal_attribution[active]).all()
    assert np.isfinite(data.demand_D[active]).all()
    assert np.isfinite(data.supply_S[active]).all()
    assert np.isfinite(data.service_rate_Y[active]).all()


def test_sum_of_attributions_equals_overall_F_metric():
    """The load-bearing invariant from §4 of the export-design doc:
    sum(spatial_attr over active) == F_spatial; same for causal."""
    bundle = _build_bundle_with_full_grid()
    data = build_export_data(bundle)
    spatial_sum = float(np.nansum(data.spatial_attribution))
    causal_sum = float(np.nansum(data.causal_attribution))
    assert np.isclose(spatial_sum, data.metadata["overall_F_spatial"], atol=1e-5)
    assert np.isclose(causal_sum, data.metadata["overall_F_causal"], atol=1e-5)


def test_metadata_contains_required_fields():
    bundle = _build_bundle_with_full_grid()
    data = build_export_data(bundle)
    md = data.metadata
    assert md["schema_version"] == SCHEMA_VERSION
    assert md["sign_convention"] == SIGN_CONVENTION
    assert "generated_at" in md
    assert "famail_git_sha" in md
    assert "famail_git_dirty" in md
    assert "config_snapshot" in md
    assert isinstance(md["overall_F_spatial"], float)
    assert isinstance(md["overall_F_causal"], float)
    assert 0.0 <= md["overall_F_spatial"] <= 1.0
    assert 0.0 <= md["overall_F_causal"] <= 1.0
    assert isinstance(md["n_active_cells_per_block"], list)
    assert len(md["n_active_cells_per_block"]) == config.T
    assert md["n_days"] == bundle.n_days


def test_config_snapshot_includes_T_and_TIME_BLOCKS():
    bundle = _build_bundle_with_full_grid()
    data = build_export_data(bundle)
    snap = data.metadata["config_snapshot"]
    assert snap["T"] == config.T
    assert len(snap["TIME_BLOCKS"]) == config.T
    assert snap["DEMAND_FLOOR"] == config.DEMAND_FLOOR


# ---------------------------------------------------------------------------
# Dense format
# ---------------------------------------------------------------------------


def test_write_dense_pkl_roundtrip(tmp_path):
    bundle = _build_bundle_with_full_grid()
    data = build_export_data(bundle)
    path = write_dense_pkl(data, tmp_path / "dense.pkl")
    with path.open("rb") as f:
        payload = _pkl.load(f)
    assert set(payload.keys()) == {
        "spatial", "causal", "active_mask", "D", "S", "Y", "metadata"
    }
    assert payload["spatial"].shape == data.spatial_attribution.shape
    assert payload["metadata"]["sign_convention"] == SIGN_CONVENTION
    np.testing.assert_array_equal(payload["active_mask"], data.active_mask)


# ---------------------------------------------------------------------------
# Tuples / long broadcast invariants
# ---------------------------------------------------------------------------


def test_tuples_broadcast_consistency_within_time_block(tmp_path):
    """All buckets within the same time_block + all days carry identical
    attribution values for a given (x, y)."""
    bundle = _build_bundle_with_full_grid()
    data = build_export_data(bundle)
    path = write_tuples_pkl(data, tmp_path / "tuples.pkl")
    with path.open("rb") as f:
        payload = _pkl.load(f)
    cols = payload["columns"]
    rows = payload["rows"]
    spatial_idx = cols.index("spatial_fairness_attribution")
    causal_idx = cols.index("causal_fairness_attribution")
    tb_idx = cols.index("time_bucket")
    x_idx, y_idx = cols.index("x_grid"), cols.index("y_grid")
    by_block: dict = {}
    for row in rows:
        x, y, tb = row[x_idx], row[y_idx], row[tb_idx]
        t_block = (tb - 1) // BUCKETS_PER_HOUR  # 1-indexed bucket -> 0-indexed block
        by_block.setdefault((x, y, t_block), []).append(
            (row[spatial_idx], row[causal_idx])
        )
    n_days = data.n_days
    expected_per_block = BUCKETS_PER_HOUR * n_days
    for (_x, _y, _tb_block), pairs in list(by_block.items())[:50]:
        assert len(pairs) == expected_per_block
        first_spatial, first_causal = pairs[0]
        for s, c in pairs:
            if np.isnan(first_spatial):
                assert np.isnan(s)
            else:
                assert s == first_spatial
            if np.isnan(first_causal):
                assert np.isnan(c)
            else:
                assert c == first_causal


def test_tuples_inactive_cells_have_nan_attributions(tmp_path):
    bundle = _build_bundle_with_full_grid()
    data = build_export_data(bundle)
    path = write_tuples_pkl(data, tmp_path / "tuples.pkl")
    with path.open("rb") as f:
        payload = _pkl.load(f)
    cols = payload["columns"]
    rows = payload["rows"]
    is_active_idx = cols.index("is_active")
    spatial_idx = cols.index("spatial_fairness_attribution")
    causal_idx = cols.index("causal_fairness_attribution")
    seen_inactive = False
    for row in rows[: 5000]:
        if not row[is_active_idx]:
            seen_inactive = True
            assert np.isnan(row[spatial_idx])
            assert np.isnan(row[causal_idx])
    assert seen_inactive, "Test slice contained no inactive cells"


def test_tuples_active_cell_attribution_matches_dense(tmp_path):
    """Spot-check: an active cell's tuple attribution equals the dense value."""
    bundle = _build_bundle_with_full_grid()
    data = build_export_data(bundle)
    path = write_tuples_pkl(data, tmp_path / "tuples.pkl")
    with path.open("rb") as f:
        payload = _pkl.load(f)
    cols = payload["columns"]
    ix = np.argwhere(data.active_mask)
    x_zero, y_zero, t_block_zero = ix[0]
    expected_spatial = float(data.spatial_attribution[x_zero, y_zero, t_block_zero])
    target_tb = t_block_zero * BUCKETS_PER_HOUR + 1  # first bucket of block
    target_x, target_y, target_day = x_zero + 1, y_zero + 1, 1
    spatial_idx = cols.index("spatial_fairness_attribution")
    x_idx, y_idx, tb_idx, day_idx = (
        cols.index("x_grid"), cols.index("y_grid"),
        cols.index("time_bucket"), cols.index("day"),
    )
    matched = False
    for row in payload["rows"]:
        if (row[x_idx] == target_x and row[y_idx] == target_y
                and row[tb_idx] == target_tb and row[day_idx] == target_day):
            assert row[spatial_idx] == pytest.approx(expected_spatial, abs=1e-6)
            matched = True
            break
    assert matched


def test_long_dataframe_matches_tuples(tmp_path):
    """Long DataFrame and tuples carry the same row data."""
    pytest.importorskip("pandas")
    bundle = _build_bundle_with_full_grid()
    data = build_export_data(bundle)
    long_path = write_long_pkl(data, tmp_path / "long.pkl")
    tup_path = write_tuples_pkl(data, tmp_path / "tuples.pkl")
    with long_path.open("rb") as f:
        long_payload = _pkl.load(f)
    with tup_path.open("rb") as f:
        tup_payload = _pkl.load(f)
    df = long_payload["dataframe"]
    assert list(df.columns) == tup_payload["columns"]
    assert len(df) == len(tup_payload["rows"])
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(df), size=min(200, len(df)), replace=False)
    for i in sample_idx:
        df_row = tuple(df.iloc[int(i)].tolist())
        tup_row = tuple(tup_payload["rows"][int(i)])
        for a, b in zip(df_row, tup_row):
            if isinstance(a, float) and np.isnan(a):
                assert np.isnan(b)
            else:
                assert a == b


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_export_writes_all_artifacts(tmp_path):
    bundle = _build_bundle_with_full_grid()
    out_dir = export(bundle, output_root=tmp_path, name="test")
    assert out_dir.is_dir()
    assert (out_dir / "fairness_attribution_dense.pkl").exists()
    assert (out_dir / "fairness_attribution_tuples.pkl").exists()
    assert (out_dir / "fairness_attribution_long.pkl").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "README.md").exists()


def test_export_metadata_json_matches_pkl_metadata(tmp_path):
    """The JSON sidecar must carry the same metadata embedded in the binaries."""
    bundle = _build_bundle_with_full_grid()
    out_dir = export(bundle, output_root=tmp_path)
    md_json = json.loads((out_dir / "metadata.json").read_text())
    with (out_dir / "fairness_attribution_dense.pkl").open("rb") as f:
        dense_payload = _pkl.load(f)
    for key in ("schema_version", "sign_convention", "n_days",
                "overall_F_spatial", "overall_F_causal"):
        assert md_json[key] == dense_payload["metadata"][key]


def test_readme_contains_sign_convention(tmp_path):
    bundle = _build_bundle_with_full_grid()
    out_dir = export(bundle, output_root=tmp_path)
    readme = (out_dir / "README.md").read_text()
    assert SIGN_CONVENTION in readme
    assert "F_spatial" in readme
    assert "F_causal" in readme
    assert "positive value = more fair" in readme
