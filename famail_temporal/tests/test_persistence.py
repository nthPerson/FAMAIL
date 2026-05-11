"""Tests for evaluation.persistence.

Note: pickle is used here for structural compatibility with the existing
passenger_seeking_trajs_45-800.pkl dataset. This is documented in the
evaluation framework spec.
"""
import json
import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest

from famail_temporal.evaluation.persistence import (
    write, _conditional_gzip_pickle,
)
from famail_temporal.evaluation.runner import ExperimentResult


def _fake_result() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="2026-04-16T00-00-00_test",
        config_snapshot={"EPSILON_BALL": 2.0, "T": 4},
        config_overrides={"EPSILON_BALL": 2.0},
        diagnostics_enabled=True,
        effective_alpha_spatial=0.33,
        effective_alpha_causal=0.33,
        effective_alpha_fidelity=0.34,
        f_spatial_before=0.3, f_spatial_after=0.4,
        f_causal_before=0.5,  f_causal_after=0.55,
        gini_dsr_before=0.7,  gini_dsr_after=0.6,
        gini_asr_before=0.8,  gini_asr_after=0.8,
        grid_before=np.ones((4, 4, 2, 4), dtype=np.float32),
        grid_after=np.ones((4, 4, 2, 4), dtype=np.float32) * 2.0,
        per_cell_fairness_attribution=np.arange(10, dtype=np.float32),
        gradient_sensitivity_before=None,
        gradient_sensitivity_after=None,
        modified_trajectory_ids=[0, 1],
        histories=[],
        top_k_scores=[0.9, 0.5],
        augmented_trajs_before={0: [[[1, 2, 3, 0, 0.1, 0.2, 0.3, 0.4]]]},
        augmented_trajs_after={0:  [[[1, 2, 3, 0, 0.2, 0.3, 0.4, 0.5]]]},
    )


def test_write_creates_directory(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    assert out_dir.is_dir()
    assert out_dir.name == "2026-04-16T00-00-00_test"


def test_write_produces_metrics_json_with_provenance(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    data = json.loads((out_dir / "metrics.json").read_text())
    assert data["experiment_id"] == "2026-04-16T00-00-00_test"
    assert "git_sha" in data
    assert "git_dirty" in data
    assert "command_line" in data
    assert "timestamp_utc" in data
    assert data["diagnostics_enabled"] is True
    assert data["metrics_before"]["f_spatial"] == pytest.approx(0.3)
    assert data["metrics_after"]["f_spatial"] == pytest.approx(0.4)
    assert data["deltas"]["f_spatial"] == pytest.approx(0.1, abs=1e-6)


def test_write_produces_grid_pickles_with_dict_schema(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    for name in ("grid_before.pkl", "grid_after.pkl"):
        with open(out_dir / name, "rb") as f:
            obj = pickle.load(f)
        assert set(obj.keys()) == {"grid", "channel_names", "time_blocks", "active_mask"}
        assert obj["grid"].shape == (4, 4, 2, 4)


def test_write_produces_modified_trajectory_ids_json(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    data = json.loads((out_dir / "modified_trajectory_ids.json").read_text())
    assert data["modified_trajectory_ids"] == [0, 1]


def test_write_skips_sensitivity_pickles_when_diagnostics_disabled(tmp_path):
    result = _fake_result()
    from dataclasses import replace
    result = replace(result, diagnostics_enabled=False)
    out_dir = write(result, output_root=tmp_path)
    assert not (out_dir / "gradient_sensitivity_before.pkl").exists()
    assert not (out_dir / "gradient_sensitivity_after.pkl").exists()


def test_conditional_gzip_uncompressed_when_small(tmp_path):
    obj = {"a": list(range(100))}
    path = tmp_path / "small.pkl"
    written = _conditional_gzip_pickle(obj, path)
    assert written.suffix == ".pkl"
    assert path.exists()


def test_conditional_gzip_compressed_when_large(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "famail_temporal.evaluation.persistence._gzip_threshold_bytes",
        lambda: 10,
    )
    obj = {"a": list(range(100))}
    path = tmp_path / "big.pkl"
    written = _conditional_gzip_pickle(obj, path)
    assert written.suffix == ".gz"
    assert written.exists()
    with gzip.open(written, "rb") as f:
        roundtrip = pickle.load(f)
    assert roundtrip == obj


def test_write_csv_files_exist(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    assert (out_dir / "per_unit_attribution.csv").exists()
    assert (out_dir / "trajectories.csv").exists()
