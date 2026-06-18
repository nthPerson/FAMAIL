import json
import random

import torch

from famail_temporal.baselines import run_level1_table_v2 as r2
from famail_temporal.baselines import fidelity_eval as fe


def test_render_table_v2_contains_all_sources_and_gate():
    result = {
        "edit_dir": "x",
        "gate": {"high_matched": 0.9, "low_mismatched": 0.3, "margin": 0.2,
                 "passed": True, "n_matched": 10, "n_mismatched": 10},
        "n_eval_drivers": 5,
        "sources": {
            k: {
                "f_causal": 0.8, "f_spatial": 0.08,
                "fidelity_a": 0.7, "fidelity_a_separation": 0.4,
                "fidelity_a_trusted": True,
                "fidelity_b": 0.05,
                "fidelity_b_per_component": {"length": 0.01, "terminal_cell": 0.02},
                "n_empty": 0,
            } for k in ("raw", "edited", "bc", "gan")
        },
    }
    md = r2.render_table_v2(result)
    assert "PASSED" in md
    for k in ("raw", "edited", "bc", "gan"):
        assert k in md
    # round-trips as JSON
    assert json.loads(r2.result_to_json(result))["gate"]["passed"] is True


def test_select_eval_drivers_filters_and_caps():
    class _T:
        def __init__(self, d): self.driver_id = d
    groups = {0: [_T(0)] * 10, 1: [_T(1)] * 3, 2: [_T(2)] * 8, 3: [_T(3)] * 7}
    out = r2._select_eval_drivers(groups, min_trajs=6, max_drivers=2)
    assert out == [0, 2]          # driver 1 (only 3) excluded; sorted; capped to 2


def test_build_source_pairs_alignment_smoke():
    """matched/mismatched pair lists are equal length and well-formed."""
    rng = random.Random(0)
    def _tt(base, L=4):
        return torch.tensor(
            [[base + i + 1.0, base + i + 1.0, 10.0, 1.0] for i in range(L)],
            dtype=torch.float32,
        )
    import numpy as np
    real_ctx = [_tt(10), _tt(20), _tt(30), _tt(40)]
    prof_d = np.zeros(11, dtype=np.float32)
    prof_dp = np.ones(11, dtype=np.float32)
    matched, mismatched = r2._build_source_pairs(
        real_slot0=[_tt(0), _tt(1)],
        source_slot0=[_tt(5), _tt(6)],
        real_context=real_ctx,
        source_context_other=real_ctx,
        profile_d=prof_d, profile_dp=prof_dp, rng=rng,
    )
    assert len(matched) == 2 and len(mismatched) == 2
    # each pair is ((set,mask,prof),(set,mask,prof))
    (sl, ml, pl), (sr, mr, pr) = matched[0]
    assert sl.shape[0] == fe.N_TRAJS_PER_BRANCH
