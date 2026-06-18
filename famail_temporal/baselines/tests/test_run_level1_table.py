"""run_level1_table pure helpers: JSON round-trip + table rendering + alignment."""
import json

import torch

from famail_temporal.baselines import run_level1_table as r
from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import active_units, make_traj_at
from famail_temporal.baselines.gan.sequences import (
    trajectory_context, trajectory_to_tokens,
)


def _fake_result():
    return {
        "gate": {"high_real_real": 0.82, "low_collapsed": 0.41,
                 "low_shuffled": 0.39, "margin": 0.2, "passed": True},
        "sources": {
            "raw":    {"f_causal": 0.8052, "f_spatial": 0.0822,
                       "fidelity_a": 1.0, "fidelity_a_std": 0.0, "fidelity_a_n": 500,
                       "fidelity_a_trusted": True, "fidelity_b": 0.0, "n_empty": 0},
            "edited": {"f_causal": 0.8180, "f_spatial": 0.0824,
                       "fidelity_a": 0.79, "fidelity_a_std": 0.05, "fidelity_a_n": 500,
                       "fidelity_a_trusted": True, "fidelity_b": 0.03, "n_empty": 0},
            "bc":     {"f_causal": 0.8062, "f_spatial": 0.0828,
                       "fidelity_a": 0.71, "fidelity_a_std": 0.06, "fidelity_a_n": 498,
                       "fidelity_a_trusted": True, "fidelity_b": 0.05, "n_empty": 2},
            "gan":    {"f_causal": 0.8198, "f_spatial": 0.0843,
                       "fidelity_a": 0.22, "fidelity_a_std": 0.04, "fidelity_a_n": 495,
                       "fidelity_a_trusted": True, "fidelity_b": 0.61, "n_empty": 5},
        },
        "gan_max_len": 52,
        "edit_dir": "famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup",
    }


def test_result_to_json_roundtrips():
    blob = r.result_to_json(_fake_result())
    loaded = json.loads(blob)
    assert loaded["sources"]["edited"]["f_causal"] == 0.8180
    assert loaded["gate"]["passed"] is True
    assert loaded["sources"]["gan"]["n_empty"] == 5          # diagnostic persisted
    assert loaded["sources"]["bc"]["fidelity_a_n"] == 498


def test_render_table_contains_sources_and_gate_verdict():
    md = r.render_table(_fake_result())
    assert "Fidelity-A" in md and "Fidelity-B" in md
    for label in ("raw", "edited", "bc", "gan"):
        assert label in md
    assert "0.8180" in md          # edited f_causal rendered
    assert "PASSED" in md or "passed" in md
    assert "single-seed" in md     # fairness columns annotated


def test_render_table_gate_failed_marks_untrusted():
    # The gate is EXPECTED to fail on the real discriminator (planning-measured
    # gap << margin), so the failed/untrusted render path is the one that ships.
    res = _fake_result()
    res["gate"]["passed"] = False
    for s in res["sources"].values():
        s["fidelity_a_trusted"] = False
    md = r.render_table(res)
    assert "FAILED" in md
    assert "(untrusted)" in md      # every Fidelity-A cell flagged


def test_train_and_generate_alignment_contexts_match_filtered_train():
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 12)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    out = r._train_and_generate(
        bundle.trajectories, adv_epochs=0, gan_loss="bce", n_critic=1,
        mle_epochs=1, max_len=8, max_tokens=256, device=torch.device("cpu"),
        seed=0,
    )
    ft, ctx = out["filtered_train"], out["contexts"]
    assert len(ft) == len(ctx)
    expected = [t for t in bundle.trajectories
                if len(trajectory_to_tokens(t)) <= 256]
    assert len(ft) == len(expected)
    for i in range(len(ft)):
        assert ctx[i] == trajectory_context(ft[i])   # index alignment
