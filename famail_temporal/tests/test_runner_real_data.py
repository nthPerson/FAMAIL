"""Real-data end-to-end smoke test for evaluation.runner (slow)."""
import json
from pathlib import Path

import numpy as np
import pytest

from famail_temporal.evaluation.runner import run_experiment
from famail_temporal.evaluation.persistence import write
from famail_temporal.evaluation.report import render


@pytest.mark.slow
def test_real_data_end_to_end(tmp_path):
    result = run_experiment(
        max_trajectories=200, k=5,
        config_overrides={"MAX_ITERATIONS": 5},
        diagnostics_enabled=True,
    )
    out_dir = write(result, output_root=tmp_path)
    report_path = render(out_dir)

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "grid_before.pkl").exists()
    assert (out_dir / "grid_after.pkl").exists()
    assert (out_dir / "trajectories.csv").exists()
    assert (out_dir / "per_unit_attribution.csv").exists()
    assert (out_dir / "modified_trajectory_ids.json").exists()
    assert report_path.exists()

    m = json.loads((out_dir / "metrics.json").read_text())
    assert np.isclose(
        m["metrics_before"]["f_spatial"],
        1.0 - float(np.nansum(result.grid_before[..., 0])),
        atol=1e-5,
    )
    assert np.isclose(
        m["metrics_before"]["f_causal"],
        1.0 - float(np.nansum(result.grid_before[..., 1])),
        atol=1e-5,
    )
