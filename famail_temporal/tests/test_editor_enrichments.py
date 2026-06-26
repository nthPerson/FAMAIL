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


def test_origin_dest_fairness_reads_correct_channels():
    gb = np.zeros((4, 4, 2, 4), dtype=np.float32)
    ga = np.zeros((4, 4, 2, 4), dtype=np.float32)
    gb[1, 2, 0, 0] = 0.11   # origin spatial BEFORE
    gb[1, 2, 0, 1] = 0.13   # origin causal  BEFORE
    ga[1, 2, 0, 1] = 0.05   # origin causal  AFTER
    ga[3, 0, 0, 0] = 0.21   # dest spatial   AFTER
    ga[3, 0, 0, 1] = 0.23   # dest causal    AFTER
    vals = P._origin_dest_fairness(gb, ga, (1, 2), (3, 0), 0)
    # order: o_spatial_b, o_spatial_a, o_causal_b, o_causal_a,
    #        d_spatial_b, d_spatial_a, d_causal_b, d_causal_a
    assert vals[0] == pytest.approx(0.11)
    assert vals[2] == pytest.approx(0.13)
    assert vals[3] == pytest.approx(0.05)
    assert vals[5] == pytest.approx(0.21)
    assert vals[7] == pytest.approx(0.23)


def test_trajectories_csv_has_origin_dest_columns(tmp_path):
    import csv
    from famail_temporal.tests.test_persistence import _fake_result
    out_dir = P.write(_fake_result(), output_root=tmp_path)  # histories=[] -> header only
    with open(out_dir / "trajectories.csv") as f:
        header = next(csv.reader(f))
    for col in ("origin_causal_attr_before", "dest_causal_attr_after",
                "origin_spatial_attr_before", "dest_spatial_attr_after"):
        assert col in header
