"""fidelity_eval.distributional_fidelity: histogram + JS over trajectory stats."""
import math

from famail_temporal.baselines import fidelity_eval as fe


def _stats(lengths, disps, covs):
    return [
        {"length": l, "mean_displacement": d, "coverage": c}
        for l, d, c in zip(lengths, disps, covs)
    ]


def test_identical_distributions_have_zero_divergence():
    s = _stats([10, 12, 14, 16], [1.0, 1.1, 0.9, 1.0], [8, 9, 10, 11])
    out = fe.distributional_fidelity(s, list(s), bins=10)
    assert math.isclose(out["per_stat"]["length"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["per_stat"]["mean_displacement"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["per_stat"]["coverage"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["aggregate"], 0.0, abs_tol=1e-9)


def test_disjoint_length_distributions_have_high_divergence():
    raw = _stats([10, 11, 12, 13], [1.0]*4, [8, 8, 8, 8])
    gen = _stats([50, 52, 54, 56], [1.0]*4, [8, 8, 8, 8])   # collapsed-like lengths
    out = fe.distributional_fidelity(gen, raw, bins=20)
    # length distributions are disjoint -> JS near 1 bit; coverage identical -> ~0
    assert out["per_stat"]["length"] > 0.9
    assert math.isclose(out["per_stat"]["coverage"], 0.0, abs_tol=1e-9)
    assert out["aggregate"] > 0.0


def test_aggregate_is_mean_of_three():
    raw = _stats([10, 20], [1.0, 2.0], [5, 6])
    gen = _stats([10, 20], [1.0, 2.0], [5, 6])
    out = fe.distributional_fidelity(gen, raw, bins=8)
    ps = out["per_stat"]
    assert math.isclose(
        out["aggregate"],
        (ps["length"] + ps["mean_displacement"] + ps["coverage"]) / 3.0,
        rel_tol=1e-9,
    )


def test_stat_ranges_pools_min_max_across_all_sources():
    raw = _stats([10, 13], [1.0, 1.0], [8, 8])
    gen = _stats([50, 56], [0.5, 2.0], [3, 20])
    ranges = fe.stat_ranges([raw, gen])
    assert ranges["length"] == (10.0, 56.0)         # pooled across both sources
    assert ranges["mean_displacement"] == (0.5, 2.0)
    assert ranges["coverage"] == (3.0, 20.0)


def test_shared_grid_path_identical_is_zero_and_values_bounded():
    # Exercises the PRODUCTION path: a precomputed shared grid passed via
    # ranges=, as the orchestrator does (the ranges=None fallback is test-only).
    raw = _stats([10, 11, 12, 13], [1.0]*4, [8, 8, 8, 8])
    gen = _stats([50, 52, 54, 56], [1.0]*4, [8, 8, 8, 8])
    ranges = fe.stat_ranges([raw, gen])             # one grid spanning both
    # Identical distributions on the shared grid -> 0 divergence.
    same = fe.distributional_fidelity(raw, raw, bins=20, ranges=ranges)
    assert math.isclose(same["aggregate"], 0.0, abs_tol=1e-9)
    # Disjoint lengths on the shared grid -> high length JS, all values in [0,1].
    out = fe.distributional_fidelity(gen, raw, bins=20, ranges=ranges)
    assert out["per_stat"]["length"] > 0.9
    for v in out["per_stat"].values():
        assert 0.0 <= v <= 1.0 + 1e-9


def test_distributional_fidelity_rejects_empty_inputs():
    raw = _stats([10, 20], [1.0, 2.0], [5, 6])
    import pytest
    with pytest.raises(ValueError):
        fe.distributional_fidelity([], raw)
    with pytest.raises(ValueError):
        fe.distributional_fidelity(raw, [])
