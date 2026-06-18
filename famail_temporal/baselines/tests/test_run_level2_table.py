import json

from famail_temporal.baselines import run_level2_table as r2


def _result():
    return {
        "edit_dir": "x", "seeds": [0, 1],
        "gate": {"high_matched": 0.84, "low_mismatched": 0.17, "margin": 0.2,
                 "passed": True, "n_matched": 10, "n_mismatched": 10},
        "n_eval_drivers": 5, "trusted": True,
        "per_source": {
            s: {"f_causal": {"mean": 0.81, "std": 0.004, "values": [0.81, 0.81]},
                "f_spatial": {"mean": 0.08, "std": 0.001, "values": [0.08, 0.08]},
                "fidelity_a": {"mean": 0.84, "std": 0.003, "values": [0.84, 0.84]},
                "fidelity_b": {"mean": 0.05, "std": 0.002, "values": [0.05, 0.05]}}
            for s in ("raw", "edited", "bcgen", "gangen")
        },
        "paired": {"f_causal": {
            "raw": {"diffs": [0.01, 0.012], "mean": 0.011, "std": 0.0014, "n": 2, "wilcoxon_p": 0.5},
            "bcgen": {"diffs": [0.01, 0.01], "mean": 0.01, "std": 0.0, "n": 2, "wilcoxon_p": None},
            "gangen": {"diffs": [0.0, 0.0], "mean": 0.0, "std": 0.0, "n": 2, "wilcoxon_p": None},
        }},
    }


def test_render_level2_table_has_sources_gate_and_paired():
    md = r2.render_level2_table(_result())
    for s in ("raw", "edited", "bcgen", "gangen"):
        assert s in md
    assert "PASSED" in md
    assert "edited" in md and "raw" in md
    assert "0.011" in md or "+0.011" in md   # headline paired mean appears


def test_result_to_json_round_trips():
    assert json.loads(r2.result_to_json(_result()))["paired"]["f_causal"]["raw"]["mean"] == 0.011
