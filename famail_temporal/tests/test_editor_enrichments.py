"""Unit + integration tests for the editor enrichment captures (E6/E7/E8/E35)."""
import numpy as np
import pytest
from types import SimpleNamespace

from famail_temporal.evaluation import persistence as P


def _mr(c, s, fi, o):
    return SimpleNamespace(f_causal=c, f_spatial=s, f_fidelity=fi, objective_value=o)


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


def test_convergence_curve_handles_ragged_iterations():
    h1 = SimpleNamespace(iterations=[_mr(0.5, 0.3, 0.1, 1.0), _mr(0.6, 0.35, 0.1, 0.9)])
    h2 = SimpleNamespace(iterations=[_mr(0.7, 0.4, 0.2, 1.2)])  # patience fired early
    c = P._convergence_curve([h1, h2])
    assert list(c["iteration"]) == [0, 1]
    assert list(c["n_contributing"]) == [2, 1]
    assert c["mean_f_causal"][0] == pytest.approx(0.6)    # (0.5 + 0.7) / 2
    assert c["mean_f_causal"][1] == pytest.approx(0.6)    # only h1 reached iter 1
    assert c["mean_f_spatial"][0] == pytest.approx(0.35)  # (0.3 + 0.4) / 2
    assert c["mean_f_fidelity"][1] == pytest.approx(0.1)


def test_convergence_curve_empty_histories():
    c = P._convergence_curve([])
    assert c["iteration"].size == 0 and c["mean_f_causal"].size == 0


def test_write_emits_convergence_curve_npz(tmp_path):
    from famail_temporal.tests.test_persistence import _fake_result
    out_dir = P.write(_fake_result(), output_root=tmp_path)  # histories=[] -> empty curve
    assert (out_dir / "convergence_curve.npz").exists()
    npz = np.load(out_dir / "convergence_curve.npz")
    assert npz["iteration"].size == 0          # empty histories -> empty curve
    import json
    meta = json.loads((out_dir / "metrics.json").read_text())
    assert "convergence_curve" in meta["artifact_paths"]


def test_end_to_end_real_history_csv_and_convergence(tmp_path):
    """A real ModificationHistory flows a DATA ROW through trajectories.csv
    (E7 cols) and into a non-empty convergence_curve.npz (E8/E35)."""
    import csv
    from dataclasses import replace
    from famail_temporal.tests.test_persistence import _fake_result
    from famail_temporal.tests.test_modifier import _make_test_trajectory
    from famail_temporal.algorithm.modifier import ModificationResult, ModificationHistory

    orig = _make_test_trajectory(driver_id=7, pickup_xy=(1, 1), time_bucket=10)
    modc = _make_test_trajectory(driver_id=7, pickup_xy=(2, 2), time_bucket=10)
    iters = [
        ModificationResult(iteration=0, objective_value=1.0, f_spatial=0.30,
                           f_causal=0.50, f_fidelity=0.10, gradient_norm=0.0,
                           cumulative_delta=np.zeros(2, dtype=np.float32)),
        ModificationResult(iteration=1, objective_value=0.9, f_spatial=0.35,
                           f_causal=0.55, f_fidelity=0.10, gradient_norm=0.0,
                           cumulative_delta=np.zeros(2, dtype=np.float32)),
    ]
    hist = ModificationHistory(original=orig, modified=modc, iterations=iters,
                               converged=True, total_iterations=2)
    # grid sized to contain the pickup cells at any t_block (T=24 covers hourly)
    gb = np.zeros((8, 8, 24, 4), dtype=np.float32)
    ga = np.zeros((8, 8, 24, 4), dtype=np.float32)
    result = replace(
        _fake_result(), histories=[hist], top_k_scores=[-1.0],
        grid_before=gb, grid_after=ga,
        all_trajectory_scores=np.array([-1.0, 0.25], dtype=np.float32),
    )
    out_dir = P.write(result, output_root=tmp_path)

    # E7: one real data row carrying the 8 origin/dest fairness columns
    with open(out_dir / "trajectories.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    for col in ("origin_causal_attr_before", "origin_causal_attr_after",
                "dest_causal_attr_before", "dest_causal_attr_after"):
        assert col in rows[0]

    # E8/E35: convergence curve aggregates the 2 iterations of the 1 history
    npz = np.load(out_dir / "convergence_curve.npz")
    assert list(npz["iteration"]) == [0, 1]
    assert int(npz["n_contributing"][0]) == 1
    assert npz["mean_f_causal"][0] == pytest.approx(0.50)
    assert npz["mean_f_causal"][1] == pytest.approx(0.55)
