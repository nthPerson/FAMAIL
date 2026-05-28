"""Unit test for run_b0 result serialization."""
import json

from famail_temporal.baselines.gan import run_b0 as r


def test_result_to_json_roundtrips():
    result = {
        "generated": {"f_spatial": 0.08, "f_causal": 0.80,
                      "gini_dsr": 0.9, "gini_asr": 0.9},
        "corpus": {"f_spatial": 0.082, "f_causal": 0.805,
                   "gini_dsr": 0.94, "gini_asr": 0.9},
        "n_generated": 105401,
    }
    blob = r.result_to_json(result)
    loaded = json.loads(blob)
    assert loaded["n_generated"] == 105401
    assert loaded["corpus"]["f_causal"] == 0.805
