"""Task 5: tests for the 5-row baseline comparison-table assembler.

Pure JSON-in -> md/json-out; no torch. Fixtures are tiny synthetic arm dirs
(metrics.json only, no histories.pkl needed since the assembler never reads
trajectories) + hand-authored famail/raw JSON stubs.
"""
import json
import subprocess
import sys

import pytest

from famail_temporal.baselines.assemble_baseline_table import (
    build_rows,
    render_json,
    render_markdown,
)


def _write_arm(tmp_path, name, mode, *, with_fidelity=True, causal_delta=0.03,
               spatial_delta=0.05, adjacency_rate=0.05, n_edited=10,
               mean_final_p=0.62):
    arm_dir = tmp_path / name
    arm_dir.mkdir()
    meta = {
        "arm": {
            "mode": mode,
            "epsilon": 0.1,
            "seed": 0,
            "n_edited": n_edited,
            "adjacency_violation_rate": adjacency_rate,
            "mean_final_p": mean_final_p,
            "mean_iterations": 12.0,
        },
        "fairness": {
            "f_spatial_before": 0.50,
            "f_spatial_after": 0.50 + spatial_delta,
            "f_causal_before": 0.60,
            "f_causal_after": 0.60 + causal_delta,
            "deltas": {"f_spatial": spatial_delta, "f_causal": causal_delta},
        },
    }
    if with_fidelity:
        meta["fidelity"] = {
            "fidelity_a": {"mean": 0.83, "std": 0.05, "n": 10,
                           "separation": 0.1, "trusted": True},
            "gate": {"high_matched": 0.83, "low_mismatched": 0.2, "margin": 0.3,
                     "passed": True, "n_matched": 10, "n_mismatched": 10},
            "fidelity_b": {"per_stat": {"a": 0.01}, "terminal_cell_js": 0.03,
                           "aggregate": 0.02},
        }
    (arm_dir / "metrics.json").write_text(json.dumps(meta))
    return arm_dir


def _write_famail_stub(tmp_path):
    path = tmp_path / "famail.json"
    path.write_text(json.dumps({
        "label": "FAMAIL",
        "fidelity_a": 0.838,
        "gate_passed": True,
        "fidelity_b": 0.0000153,
        "f_causal_before": 0.7958,
        "f_causal_after": 0.8180,
        "f_spatial_before": 0.080,
        "f_spatial_after": 0.101,
        "adjacency_violation_rate": 0.0,
        "mean_final_p": None,
        "n": 5000,
    }))
    return path


def _write_raw_stub(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps({
        "label": "raw",
        "fidelity_a": 1.0,
        "gate_passed": None,
        "fidelity_b": 0.0,
        "f_causal_before": 0.7958,
        "f_causal_after": 0.7958,
        "f_spatial_before": 0.080,
        "f_spatial_after": 0.080,
        "adjacency_violation_rate": 0.0,
        "mean_final_p": None,
        "n": 5000,
    }))
    return path


@pytest.fixture
def three_arms(tmp_path):
    ifgsm = _write_arm(tmp_path, "arm_ifgsm", "ifgsm", with_fidelity=True,
                        causal_delta=-0.01, spatial_delta=-0.02, adjacency_rate=0.123)
    # fgsm intentionally has NO fidelity block -> must render "—", never KeyError.
    fgsm = _write_arm(tmp_path, "arm_fgsm", "fgsm", with_fidelity=False,
                       causal_delta=-0.02, spatial_delta=-0.03, adjacency_rate=0.2)
    random_ = _write_arm(tmp_path, "arm_random", "random", with_fidelity=True,
                          causal_delta=-0.05, spatial_delta=-0.04, adjacency_rate=0.5)
    return [ifgsm, fgsm, random_]


@pytest.fixture
def famail_raw_json(tmp_path):
    return _write_famail_stub(tmp_path), _write_raw_stub(tmp_path)


def test_build_rows_order_and_labels(three_arms, famail_raw_json):
    famail_json, raw_json = famail_raw_json
    rows = build_rows(arm_dirs=three_arms, famail_json=famail_json, raw_json=raw_json)
    assert [r["label"] for r in rows] == ["raw", "FAMAIL", "ifgsm", "fgsm", "random"]
    assert len(rows) == 5


def test_deltas_computed_as_after_minus_before(three_arms, famail_raw_json):
    famail_json, raw_json = famail_raw_json
    rows = build_rows(arm_dirs=three_arms, famail_json=famail_json, raw_json=raw_json)
    by_label = {r["label"]: r for r in rows}

    # raw: deltas are 0 by definition (equal before/after in the stub).
    assert by_label["raw"]["delta_f_causal"] == pytest.approx(0.0)
    assert by_label["raw"]["delta_f_spatial"] == pytest.approx(0.0)

    # FAMAIL: after - before from the stub's own values.
    assert by_label["FAMAIL"]["delta_f_causal"] == pytest.approx(0.8180 - 0.7958)
    assert by_label["FAMAIL"]["delta_f_spatial"] == pytest.approx(0.101 - 0.080)

    # arm rows: after - before from each arm's own fairness block.
    assert by_label["ifgsm"]["delta_f_causal"] == pytest.approx(-0.01)
    assert by_label["ifgsm"]["delta_f_spatial"] == pytest.approx(-0.02)
    assert by_label["fgsm"]["delta_f_causal"] == pytest.approx(-0.02)
    assert by_label["random"]["delta_f_causal"] == pytest.approx(-0.05)


def test_missing_fidelity_block_renders_em_dash_not_keyerror(three_arms, famail_raw_json):
    famail_json, raw_json = famail_raw_json
    rows = build_rows(arm_dirs=three_arms, famail_json=famail_json, raw_json=raw_json)
    by_label = {r["label"]: r for r in rows}
    fgsm_row = by_label["fgsm"]
    assert fgsm_row["fidelity_a"] is None
    assert fgsm_row["gate_passed"] is None
    assert fgsm_row["fidelity_b"] is None

    md = render_markdown(rows)
    fgsm_line = [l for l in md.splitlines() if l.startswith("| fgsm")][0]
    cells = [c.strip() for c in fgsm_line.strip("|").split("|")]
    # row, Fidelity-A, gate, Fidelity-B(JS) -> first three data cells are em-dash
    assert cells[1] == "—"
    assert cells[2] == "—"
    assert cells[3] == "—"


def test_adjacency_rendered_as_percent_one_decimal(three_arms, famail_raw_json):
    famail_json, raw_json = famail_raw_json
    rows = build_rows(arm_dirs=three_arms, famail_json=famail_json, raw_json=raw_json)
    by_label = {r["label"]: r for r in rows}
    assert by_label["ifgsm"]["adjacency_violation_pct"] == pytest.approx(12.3)
    assert by_label["random"]["adjacency_violation_pct"] == pytest.approx(50.0)

    md = render_markdown(rows)
    random_line = [l for l in md.splitlines() if l.startswith("| random")][0]
    assert "50.0%" in random_line
    ifgsm_line = [l for l in md.splitlines() if l.startswith("| ifgsm")][0]
    assert "12.3%" in ifgsm_line


def test_markdown_contains_all_five_row_labels(three_arms, famail_raw_json):
    famail_json, raw_json = famail_raw_json
    rows = build_rows(arm_dirs=three_arms, famail_json=famail_json, raw_json=raw_json)
    md = render_markdown(rows)
    for label in ("raw", "FAMAIL", "ifgsm", "fgsm", "random"):
        assert f"| {label} " in md or md.count(f"| {label} |") >= 0
        assert any(line.startswith(f"| {label} ") for line in md.splitlines())


def test_markdown_floats_rounded_to_4_decimals(three_arms, famail_raw_json):
    famail_json, raw_json = famail_raw_json
    rows = build_rows(arm_dirs=three_arms, famail_json=famail_json, raw_json=raw_json)
    md = render_markdown(rows)
    famail_line = [l for l in md.splitlines() if l.startswith("| FAMAIL")][0]
    # 0.8180 - 0.7958 = 0.022200000000000042 in float; rounded md shows 4 decimals.
    assert "0.0222" in famail_line


def test_json_output_has_five_rows_full_precision(three_arms, famail_raw_json):
    famail_json, raw_json = famail_raw_json
    rows = build_rows(arm_dirs=three_arms, famail_json=famail_json, raw_json=raw_json)
    payload = render_json(rows)
    assert len(payload["rows"]) == 5
    famail_row = [r for r in payload["rows"] if r["label"] == "FAMAIL"][0]
    # full (unrounded) precision preserved in the json, unlike the md table.
    assert famail_row["delta_f_causal"] == pytest.approx(0.8180 - 0.7958, abs=1e-12)


def test_cli_writes_both_files(tmp_path, three_arms, famail_raw_json):
    famail_json, raw_json = famail_raw_json
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable, "-m", "famail_temporal.baselines.assemble_baseline_table",
        "--arm-dirs", *[str(d) for d in three_arms],
        "--famail-json", str(famail_json),
        "--raw-json", str(raw_json),
        "--out", str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    md_path = out_dir / "baseline_table.md"
    json_path = out_dir / "baseline_table.json"
    assert md_path.exists()
    assert json_path.exists()
    for label in ("raw", "FAMAIL", "ifgsm", "fgsm", "random"):
        assert any(line.startswith(f"| {label} ") for line in md_path.read_text().splitlines())
    payload = json.loads(json_path.read_text())
    assert len(payload["rows"]) == 5


def test_no_torch_import():
    """Guard: this module must stay a pure JSON-in -> md/json-out assembler."""
    import famail_temporal.baselines.assemble_baseline_table as mod
    src_path = mod.__file__
    with open(src_path) as f:
        src = f.read()
    assert "import torch" not in src
    assert "from torch" not in src
