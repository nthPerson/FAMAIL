"""Tests for the alpha-sweep post-processing summary (table + Pareto scatter)."""
import json

import pytest

from famail_temporal.analysis import alpha_sweep_summary as ass


def _mk_point(results_dir, ts, tag, a_sp, a_ca, d_sp, d_ca):
    d = results_dir / f"{ts}_alpha_sweep_{tag}_filtered"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "effective_alphas": {"alpha_spatial": a_sp, "alpha_causal": a_ca,
                             "alpha_fidelity": 0.1},
        "deltas": {"f_spatial": d_sp, "f_causal": d_ca,
                   "gini_dsr": -0.01, "gini_asr": -0.0002},
    }))
    return d


def _mk_anchor(results_dir, d_sp=0.006357, d_ca=0.022218):
    d = results_dir / "2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "effective_alphas": {"alpha_spatial": 0.2, "alpha_causal": 0.7,
                             "alpha_fidelity": 0.1},
        "deltas": {"f_spatial": d_sp, "f_causal": d_ca},
    }))
    return d


def test_load_points_reads_deltas_alphas_and_flags_pending(tmp_path):
    _mk_point(tmp_path, "2026-07-09T17-11-50", "s00_c90_f10", 0.0, 0.9,
              0.0057, 0.0221)
    _mk_point(tmp_path, "2026-07-10T02-06-37", "s10_c80_f10", 0.1, 0.8,
              0.0061, 0.0226)
    anchor = _mk_anchor(tmp_path)
    rows, pending = ass.load_points(tmp_path, anchor_dir=anchor)
    # anchor + 2 found points; 3 of the 5 sweep tags pending
    assert sorted(pending) == ["s35_c55_f10", "s55_c35_f10", "s80_c10_f10"]
    assert len(rows) == 3
    anchor_rows = [r for r in rows if r["is_anchor"]]
    assert len(anchor_rows) == 1
    a = anchor_rows[0]
    assert a["alphas"] == (0.2, 0.7, 0.1)
    assert a["d_f_causal"] == pytest.approx(0.022218)
    s00 = next(r for r in rows if r["tag"] == "s00_c90_f10")
    assert s00["alphas"] == (0.0, 0.9, 0.1)
    assert s00["d_f_spatial"] == pytest.approx(0.0057)
    assert not s00["is_anchor"]
    # rows sorted by alpha_spatial ascending
    assert [r["alphas"][0] for r in rows] == sorted(r["alphas"][0] for r in rows)


def test_load_points_newest_dir_wins_on_duplicate_tag(tmp_path):
    _mk_point(tmp_path, "2026-07-09T00-00-00", "s00_c90_f10", 0.0, 0.9,
              0.9990, 0.9990)   # stale rerun
    _mk_point(tmp_path, "2026-07-10T00-00-00", "s00_c90_f10", 0.0, 0.9,
              0.0057, 0.0221)   # newest
    rows, _ = ass.load_points(tmp_path, anchor_dir=None)
    s00 = next(r for r in rows if r["tag"] == "s00_c90_f10")
    assert s00["d_f_causal"] == pytest.approx(0.0221)


def _rows():
    def row(tag, a, dsp, dca, anchor=False):
        return {"tag": tag, "alphas": a, "d_f_spatial": dsp, "d_f_causal": dca,
                "is_anchor": anchor, "source": f"/x/{tag}"}
    return [
        row("s00", (0.0, 0.9, 0.1), 0.0057, 0.0221),
        row("anchor", (0.2, 0.7, 0.1), 0.0064, 0.0222, anchor=True),
        row("dominated", (0.1, 0.8, 0.1), 0.0050, 0.0210),   # < anchor on both
        row("spatialmax", (0.8, 0.1, 0.1), 0.0100, 0.0100),
        row("negspatial", (0.9, 0.0, 0.1), -0.0010, 0.0300), # best causal, spatial < 0
    ]


def test_pareto_flags_maximize_both_axes():
    flags = ass.pareto_flags(_rows())
    by_tag = dict(zip([r["tag"] for r in _rows()], flags))
    assert by_tag["dominated"] is False          # anchor beats it on both
    assert by_tag["anchor"] is True
    assert by_tag["spatialmax"] is True          # best spatial
    assert by_tag["negspatial"] is True          # best causal
    assert by_tag["s00"] is False                # anchor >= on both, > on one


def test_shipped_criterion_max_causal_subject_to_nonneg_spatial():
    best = ass.shipped_criterion(_rows())
    # negspatial has the max ΔF_causal but fails ΔF_spatial >= 0
    assert best["tag"] == "anchor"


def test_render_table_marks_pareto_anchor_and_pending():
    md = ass.render_table(_rows(), pending=["s55_c35_f10"])
    assert "PENDING" in md and "s55_c35_f10" in md
    assert "★" in md                              # anchor marker
    assert "(0.2, 0.7, 0.1)" in md
    assert "+0.0222" in md
    assert "criterion" in md.lower()              # shipped-criterion line present


def test_cli_writes_outputs(tmp_path):
    _mk_point(tmp_path, "2026-07-09T17-11-50", "s00_c90_f10", 0.0, 0.9,
              0.0057, 0.0221)
    anchor = _mk_anchor(tmp_path)
    out = tmp_path / "out"
    rc = ass.main(["--results-dir", str(tmp_path), "--anchor-dir", str(anchor),
                   "--out", str(out)])
    assert rc == 0
    assert (out / "alpha_sweep_summary.md").exists()
    assert (out / "alpha_pareto.png").exists()
    payload = json.loads((out / "alpha_sweep_summary.json").read_text())
    assert payload["pending"] == ["s10_c80_f10", "s35_c55_f10",
                                  "s55_c35_f10", "s80_c10_f10"]
    assert len(payload["rows"]) == 2
