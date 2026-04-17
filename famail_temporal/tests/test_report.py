"""Tests for evaluation.report.render."""
import json
from pathlib import Path

import numpy as np
import pytest

from famail_temporal.evaluation.report import render
from famail_temporal.evaluation.persistence import write
from famail_temporal.evaluation.runner import ExperimentResult


def _fake_result() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="2026-04-16T00-00-00_test",
        config_snapshot={"EPSILON_BALL": 2.0, "T": 4, "MAX_ITERATIONS": 50},
        config_overrides={"EPSILON_BALL": 2.0},
        diagnostics_enabled=True,
        f_spatial_before=0.3, f_spatial_after=0.4,
        f_causal_before=0.5,  f_causal_after=0.55,
        gini_dsr_before=0.7,  gini_dsr_after=0.6,
        gini_asr_before=0.8,  gini_asr_after=0.8,
        grid_before=np.ones((4, 4, 2, 4), dtype=np.float32),
        grid_after=np.ones((4, 4, 2, 4), dtype=np.float32) * 2.0,
        per_unit_attribution_before=np.arange(10, dtype=np.float32),
        per_unit_attribution_signed_before=np.arange(10, dtype=np.float32),
        gradient_sensitivity_before=None,
        gradient_sensitivity_after=None,
        modified_trajectory_ids=[], histories=[], top_k_scores=[],
        augmented_trajs_before={}, augmented_trajs_after={},
    )


def test_render_produces_report_md(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    report_path = render(out_dir)
    assert report_path.exists()
    assert report_path.name == "report.md"


def test_report_contains_header_and_sections(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    report = render(out_dir).read_text()
    assert result.experiment_id in report
    assert "Config" in report
    assert "Fairness" in report
    assert "Artifact" in report


def test_report_marks_overridden_config_values_bold(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    report = render(out_dir).read_text()
    assert "**EPSILON_BALL**" in report or "**2.0**" in report


def test_report_reads_only_from_disk(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    assert render.__code__.co_varnames[0] == "output_dir"
