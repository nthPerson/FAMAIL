"""Unit tests for the pure runner-enrichment helpers (Plan 4)."""
import math
import numpy as np
import pytest

from famail_temporal.baselines import _enrich as E


def test_t_ci_basic_and_degenerate():
    lo, hi = E.t_ci([1.0, 2.0, 3.0, 4.0, 5.0], confidence=0.95)
    assert lo < 3.0 < hi                      # CI brackets the mean (3.0)
    assert math.isnan(E.t_ci([1.0])[0])       # <2 values -> nan
    assert math.isnan(E.t_ci([])[0])


def test_shannon_entropy_bits():
    assert E.shannon_entropy_bits([1, 1, 1, 1]) == pytest.approx(2.0)  # uniform 4 -> 2 bits
    assert E.shannon_entropy_bits([1, 0, 0, 0]) == pytest.approx(0.0)  # one cell -> 0 bits
    assert E.shannon_entropy_bits([0, 0, 0]) == 0.0                    # empty -> 0
    assert E.shannon_entropy_bits(np.array([5.0, 5.0])) == pytest.approx(1.0)


def test_degeneracy_scalars():
    # 3 generated trajectories of lengths 2, 3, 4; terminals at 3 distinct cells
    gen_cells = [[(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 1), (2, 2), (3, 3)]]
    terminal_pickups = [(1, 1, 0), (2, 2, 0), (3, 3, 0)]
    d = E.degeneracy_scalars(terminal_pickups, gen_cells, n_cells=4320)
    assert d["mean_trip_length"] == pytest.approx(3.0)
    assert d["std_trip_length"] == pytest.approx(1.0)            # ddof=1 of [2,3,4]
    assert d["terminal_cell_entropy_bits"] == pytest.approx(math.log2(3), abs=1e-6)  # 3 equally-used cells


def test_effective_edited_fraction():
    assert E.effective_edited_fraction(2000, 95297, 1) == pytest.approx(2000 / 95297)
    # w=30: (2000*30)/(2000*30 + 93297)
    assert E.effective_edited_fraction(2000, 95297, 30) == pytest.approx(60000 / (60000 + 93297))


def test_dose_response_table():
    per_arm = {
        "edited_w10": {"fidelity_b": {"mean": 0.30}, "fidelity_a": {"mean": 0.84}},
        "edited_w30": {"fidelity_b": {"mean": 0.31}, "fidelity_a": {"mean": 0.83}},
    }
    paired = {"f_causal": {
        "edited_w10": {"mean": 0.018, "wilcoxon_p": 0.06},
        "edited_w30": {"mean": 0.027, "wilcoxon_p": 0.03},
    }}
    rows = E.dose_response_table(per_arm, paired, [10, 30])
    assert rows[0] == {"w": 10, "delta_f_causal": 0.018, "wilcoxon_p": 0.06,
                       "fidelity_b": 0.30, "fidelity_a": 0.84}
    assert rows[1]["w"] == 30 and rows[1]["delta_f_causal"] == 0.027


def test_chosen_placebo_ids_deterministic():
    raw_ids = list(range(10))
    edited = {0, 1, 2}
    a = E.chosen_placebo_ids(raw_ids, edited, placebo_seed=12345, k=3)
    b = E.chosen_placebo_ids(raw_ids, edited, placebo_seed=12345, k=3)
    assert a == b                                  # deterministic
    assert len(a) == 3
    assert all(i not in edited for i in a)         # never picks edited ids
    # default k = len(edited)
    assert len(E.chosen_placebo_ids(raw_ids, edited, placebo_seed=1)) == 3


def test_paired_stats_t_ci_augmentation_shape():
    # mirrors the in-runner augmentation: every leaf with 'diffs' gains a 't_ci' pair
    paired = {"f_causal": {"raw": {"diffs": [0.01, 0.02, 0.03, 0.015, 0.025], "mean": 0.02}}}
    leaf = paired["f_causal"]["raw"]
    leaf["t_ci"] = list(E.t_ci(leaf["diffs"]))
    assert len(leaf["t_ci"]) == 2 and leaf["t_ci"][0] < 0.02 < leaf["t_ci"][1]
