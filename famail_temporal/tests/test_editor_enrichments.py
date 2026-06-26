"""Unit + integration tests for the editor enrichment captures (E6/E7/E8/E35)."""
import numpy as np
import pytest

from famail_temporal.evaluation import persistence as P


def test_attribution_distribution_payload_counts_and_arrays():
    all_scores = np.array([-3.0, -1.0, 0.0, 2.0], dtype=np.float32)
    edited = np.array([-3.0, -1.0], dtype=np.float32)
    p = P._attribution_distribution_payload(all_scores, edited)
    assert int(p["n_total"]) == 4
    assert int(p["n_negative"]) == 2      # strictly-negative αᵢ marks the editable pool
    assert int(p["n_edited"]) == 2
    np.testing.assert_array_equal(p["all_scores"], all_scores)
    np.testing.assert_array_equal(p["edited_scores"], edited)


def test_write_emits_attribution_distribution_npz(tmp_path):
    from famail_temporal.tests.test_persistence import _fake_result
    from dataclasses import replace
    result = replace(
        _fake_result(),
        all_trajectory_scores=np.array([-2.0, -1.0, 0.5, 3.0], dtype=np.float32),
        top_k_scores=[-2.0, -1.0],
    )
    out_dir = P.write(result, output_root=tmp_path)
    npz = np.load(out_dir / "attribution_distribution.npz")
    assert int(npz["n_total"]) == 4 and int(npz["n_negative"]) == 2
    assert int(npz["n_edited"]) == 2
    import json
    meta = json.loads((out_dir / "metrics.json").read_text())
    assert "attribution_distribution" in meta["artifact_paths"]


def test_write_skips_attribution_distribution_when_scores_absent(tmp_path):
    from famail_temporal.tests.test_persistence import _fake_result
    out_dir = P.write(_fake_result(), output_root=tmp_path)  # all_trajectory_scores defaults None
    assert not (out_dir / "attribution_distribution.npz").exists()
