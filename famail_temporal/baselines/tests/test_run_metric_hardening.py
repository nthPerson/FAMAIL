"""Unit tests for the metric-hardening CLI's pure helpers."""
import json

from famail_temporal.baselines import run_metric_hardening as r


def test_result_to_json_roundtrips():
    result = {
        "transmission": {
            "js_target": 0.02, "js_generated": 0.014,
            "transmission_ratio": 0.70,
            "js_b0_vs_raw": 0.001, "js_famail_vs_edited": 0.005,
        },
        "di_b0": {"di_primary": 1.02, "di_supplementary": 0.98},
        "di_famail": {"di_primary": 1.07, "di_supplementary": 0.93},
        "localized_b0": {"f_causal_localized": 0.42, "f_causal_global": 0.808, "n_edited_active_units": 3773},
        "localized_famail": {"f_causal_localized": 0.46, "f_causal_global": 0.812, "n_edited_active_units": 3773},
        "edit_dir": "famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup",
    }
    blob = r.result_to_json(result)
    loaded = json.loads(blob)
    assert loaded["transmission"]["transmission_ratio"] == 0.70
    assert loaded["di_famail"]["di_primary"] == 1.07
    assert loaded["localized_b0"]["n_edited_active_units"] == 3773
