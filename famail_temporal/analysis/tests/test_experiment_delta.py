"""TDD tests for experiment_delta.py — RED phase.

These tests are written BEFORE the implementation exists.
Each test exercises a specific pure function with synthetic minimal dicts.
"""
from __future__ import annotations
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _approx(a, b, tol=1e-9):
    return abs(a - b) < tol


# ── l1_delta ─────────────────────────────────────────────────────────────────

class TestL1Delta:
    """l1_delta(dirty, clean) -> {source: {metric: {dirty, clean, delta}}}"""

    def test_basic_delta_arithmetic(self):
        from famail_temporal.analysis.experiment_delta import l1_delta
        dirty = {"sources": {
            "raw":    {"f_causal": 0.800, "f_spatial": 0.080, "fidelity_a": 0.840, "fidelity_b": 0.000},
            "edited": {"f_causal": 0.810, "f_spatial": 0.082, "fidelity_a": 0.838, "fidelity_b": 0.168},
            "bc":     {"f_causal": 0.807, "f_spatial": 0.083, "fidelity_a": 0.841, "fidelity_b": 0.010},
            "gan":    {"f_causal": 0.814, "f_spatial": 0.083, "fidelity_a": 0.842, "fidelity_b": 0.322},
        }}
        clean = {"sources": {
            "raw":    {"f_causal": 0.807, "f_spatial": 0.103, "fidelity_a": 0.847, "fidelity_b": 0.000},
            "edited": {"f_causal": 0.819, "f_spatial": 0.103, "fidelity_a": 0.842, "fidelity_b": 0.151},
            "bc":     {"f_causal": 0.804, "f_spatial": 0.105, "fidelity_a": 0.848, "fidelity_b": 0.011},
            "gan":    {"f_causal": 0.815, "f_spatial": 0.104, "fidelity_a": 0.849, "fidelity_b": 0.292},
        }}
        result = l1_delta(dirty, clean)
        # Check structure
        assert set(result.keys()) == {"raw", "edited", "bc", "gan"}
        for src in result:
            assert set(result[src].keys()) >= {"f_causal", "f_spatial", "fidelity_a", "fidelity_b"}
            for metric in ("f_causal",):
                row = result[src][metric]
                assert set(row.keys()) == {"dirty", "clean", "delta"}
        # Check arithmetic: delta = clean - dirty
        assert _approx(result["edited"]["f_causal"]["dirty"], 0.810)
        assert _approx(result["edited"]["f_causal"]["clean"], 0.819)
        assert _approx(result["edited"]["f_causal"]["delta"], 0.819 - 0.810)

    def test_edited_fairest_in_both_dirty_and_clean(self):
        """Edited should have highest f_causal in both dirty and clean."""
        from famail_temporal.analysis.experiment_delta import l1_delta
        dirty = {"sources": {
            "raw":    {"f_causal": 0.800, "f_spatial": 0.080, "fidelity_a": 0.840, "fidelity_b": 0.000},
            "edited": {"f_causal": 0.815, "f_spatial": 0.082, "fidelity_a": 0.838, "fidelity_b": 0.168},
            "bc":     {"f_causal": 0.807, "f_spatial": 0.083, "fidelity_a": 0.841, "fidelity_b": 0.010},
            "gan":    {"f_causal": 0.814, "f_spatial": 0.083, "fidelity_a": 0.842, "fidelity_b": 0.322},
        }}
        clean = {"sources": {
            "raw":    {"f_causal": 0.807, "f_spatial": 0.103, "fidelity_a": 0.847, "fidelity_b": 0.000},
            "edited": {"f_causal": 0.819, "f_spatial": 0.103, "fidelity_a": 0.842, "fidelity_b": 0.151},
            "bc":     {"f_causal": 0.804, "f_spatial": 0.105, "fidelity_a": 0.848, "fidelity_b": 0.011},
            "gan":    {"f_causal": 0.815, "f_spatial": 0.104, "fidelity_a": 0.849, "fidelity_b": 0.292},
        }}
        result = l1_delta(dirty, clean)
        dirty_edited = result["edited"]["f_causal"]["dirty"]
        clean_edited = result["edited"]["f_causal"]["clean"]
        dirty_others = [result[s]["f_causal"]["dirty"] for s in ("raw", "bc", "gan")]
        clean_others = [result[s]["f_causal"]["clean"] for s in ("raw", "bc", "gan")]
        assert dirty_edited > max(dirty_others)
        assert clean_edited > max(clean_others)

    def test_missing_source_graceful(self):
        """Missing sources should produce None values, not crash."""
        from famail_temporal.analysis.experiment_delta import l1_delta
        dirty = {"sources": {"raw": {"f_causal": 0.800, "f_spatial": 0.080, "fidelity_a": 0.840, "fidelity_b": 0.0}}}
        clean = {"sources": {"raw": {"f_causal": 0.807, "f_spatial": 0.103, "fidelity_a": 0.847, "fidelity_b": 0.0}}}
        result = l1_delta(dirty, clean)
        assert result["edited"]["f_causal"]["dirty"] is None
        assert result["edited"]["f_causal"]["clean"] is None
        assert result["edited"]["f_causal"]["delta"] is None

    def test_fidelity_a_delta_computed(self):
        """Fidelity_a delta = clean - dirty."""
        from famail_temporal.analysis.experiment_delta import l1_delta
        dirty = {"sources": {
            "raw":    {"f_causal": 0.800, "f_spatial": 0.080, "fidelity_a": 0.840, "fidelity_b": 0.0},
            "edited": {"f_causal": 0.810, "f_spatial": 0.082, "fidelity_a": 0.838, "fidelity_b": 0.168},
            "bc":     {"f_causal": 0.807, "f_spatial": 0.083, "fidelity_a": 0.841, "fidelity_b": 0.010},
            "gan":    {"f_causal": 0.814, "f_spatial": 0.083, "fidelity_a": 0.842, "fidelity_b": 0.322},
        }}
        clean = {"sources": {
            "raw":    {"f_causal": 0.807, "f_spatial": 0.103, "fidelity_a": 0.847, "fidelity_b": 0.0},
            "edited": {"f_causal": 0.819, "f_spatial": 0.103, "fidelity_a": 0.842, "fidelity_b": 0.151},
            "bc":     {"f_causal": 0.804, "f_spatial": 0.105, "fidelity_a": 0.848, "fidelity_b": 0.011},
            "gan":    {"f_causal": 0.815, "f_spatial": 0.104, "fidelity_a": 0.849, "fidelity_b": 0.292},
        }}
        result = l1_delta(dirty, clean)
        delta = result["edited"]["fidelity_a"]["delta"]
        assert _approx(delta, 0.842 - 0.838)


# ── l2_delta ─────────────────────────────────────────────────────────────────

class TestL2Delta:
    """l2_delta(dirty, clean) -> {paired_edited_raw: {dirty_mean, clean_mean, delta, dirty_p, clean_p}, per_source: {...}}"""

    def test_paired_fields_present(self):
        from famail_temporal.analysis.experiment_delta import l2_delta
        dirty = {
            "per_source": {
                "raw":    {"f_causal": {"mean": 0.808}, "f_spatial": {"mean": 0.083}},
                "edited": {"f_causal": {"mean": 0.806}, "f_spatial": {"mean": 0.084}},
                "bcgen":  {"f_causal": {"mean": 0.810}, "f_spatial": {"mean": 0.083}},
                "gangen": {"f_causal": {"mean": 0.814}, "f_spatial": {"mean": 0.084}},
            },
            "paired": {"f_causal": {"raw": {"mean": -0.0022, "wilcoxon_p": 0.0625}}},
        }
        clean = {
            "per_source": {
                "raw":    {"f_causal": {"mean": 0.808}, "f_spatial": {"mean": 0.105}},
                "edited": {"f_causal": {"mean": 0.807}, "f_spatial": {"mean": 0.105}},
                "bcgen":  {"f_causal": {"mean": 0.807}, "f_spatial": {"mean": 0.105}},
                "gangen": {"f_causal": {"mean": 0.816}, "f_spatial": {"mean": 0.104}},
            },
            "paired": {"f_causal": {"raw": {"mean": -0.0009, "wilcoxon_p": 0.4375}}},
        }
        result = l2_delta(dirty, clean)
        pr = result["paired_edited_raw"]
        assert set(pr.keys()) >= {"dirty_mean", "clean_mean", "delta", "dirty_p", "clean_p"}
        assert _approx(pr["dirty_mean"], -0.0022)
        assert _approx(pr["clean_mean"], -0.0009)
        assert _approx(pr["delta"], -0.0009 - (-0.0022))
        assert pr["dirty_p"] == 0.0625
        assert pr["clean_p"] == 0.4375

    def test_per_source_present(self):
        from famail_temporal.analysis.experiment_delta import l2_delta
        dirty = {
            "per_source": {"edited": {"f_causal": {"mean": 0.806}, "f_spatial": {"mean": 0.084}}},
            "paired": {"f_causal": {"raw": {"mean": -0.0022, "wilcoxon_p": 0.0625}}},
        }
        clean = {
            "per_source": {"edited": {"f_causal": {"mean": 0.807}, "f_spatial": {"mean": 0.105}}},
            "paired": {"f_causal": {"raw": {"mean": -0.0009, "wilcoxon_p": 0.4375}}},
        }
        result = l2_delta(dirty, clean)
        ps = result["per_source"]
        assert "edited" in ps
        assert _approx(ps["edited"]["f_causal"]["dirty"], 0.806)
        assert _approx(ps["edited"]["f_causal"]["clean"], 0.807)
        assert _approx(ps["edited"]["f_causal"]["delta"], 0.807 - 0.806)

    def test_missing_source_graceful(self):
        from famail_temporal.analysis.experiment_delta import l2_delta
        dirty = {"per_source": {}, "paired": {"f_causal": {"raw": {"mean": -0.002, "wilcoxon_p": 0.0625}}}}
        clean = {"per_source": {}, "paired": {"f_causal": {"raw": {"mean": -0.001, "wilcoxon_p": 0.44}}}}
        result = l2_delta(dirty, clean)
        # Should not crash; sources produce None
        ps = result["per_source"]
        for src in ("raw", "edited", "bcgen", "gangen"):
            assert ps[src]["f_causal"]["dirty"] is None


# ── wbc_delta ─────────────────────────────────────────────────────────────────

class TestWbcDelta:
    """wbc_delta(dirty, clean) -> {arm: {dirty_delta_vs_raw, clean_delta_vs_raw, p_dirty, p_clean, status}}"""

    def test_compared_arm_arithmetic(self):
        from famail_temporal.analysis.experiment_delta import wbc_delta
        dirty = {"paired_vs_raw": {"f_causal": {
            "edited_w10": {"mean": 0.0186, "wilcoxon_p": 0.03125},
            "edited_w30": {"mean": 0.0274, "wilcoxon_p": 0.03125},
        }}}
        clean = {"paired_vs_raw": {"f_causal": {
            "edited_w10": {"mean": 0.0175, "wilcoxon_p": 0.03125},
            "edited_w30": {"mean": 0.0260, "wilcoxon_p": 0.03125},
        }}}
        result = wbc_delta(dirty, clean)
        arm = result["edited_w30"]
        assert arm["status"] == "compared"
        assert _approx(arm["dirty_delta_vs_raw"], 0.0274)
        assert _approx(arm["clean_delta_vs_raw"], 0.0260)
        assert arm["p_dirty"] == 0.03125
        assert arm["p_clean"] == 0.03125

    def test_clean_only_arm_flagged(self):
        """Arms present in clean but absent in dirty -> status 'clean_only'."""
        from famail_temporal.analysis.experiment_delta import wbc_delta
        dirty = {"paired_vs_raw": {"f_causal": {
            "edited_w30": {"mean": 0.0274, "wilcoxon_p": 0.03125},
        }}}
        clean = {"paired_vs_raw": {"f_causal": {
            "edited_w30": {"mean": 0.0260, "wilcoxon_p": 0.03125},
            "most_fair_w10": {"mean": 0.0001, "wilcoxon_p": 1.0},  # clean-only
        }}}
        result = wbc_delta(dirty, clean)
        assert result["most_fair_w10"]["status"] == "clean_only"
        assert result["most_fair_w10"]["dirty_delta_vs_raw"] is None
        assert _approx(result["most_fair_w10"]["clean_delta_vs_raw"], 0.0001)

    def test_dirty_only_arm_flagged(self):
        """Arms in dirty but absent in clean (edge case)."""
        from famail_temporal.analysis.experiment_delta import wbc_delta
        dirty = {"paired_vs_raw": {"f_causal": {
            "edited_w10": {"mean": 0.0186, "wilcoxon_p": 0.03125},
            "legacy_arm": {"mean": 0.0050, "wilcoxon_p": 0.5},
        }}}
        clean = {"paired_vs_raw": {"f_causal": {
            "edited_w10": {"mean": 0.0175, "wilcoxon_p": 0.03125},
        }}}
        result = wbc_delta(dirty, clean)
        assert result["legacy_arm"]["status"] == "dirty_only"
        assert result["legacy_arm"]["clean_delta_vs_raw"] is None

    def test_union_of_arms(self):
        """All arms from both dirty and clean appear in the output."""
        from famail_temporal.analysis.experiment_delta import wbc_delta
        dirty = {"paired_vs_raw": {"f_causal": {"a": {"mean": 0.01, "wilcoxon_p": 0.03}}}}
        clean = {"paired_vs_raw": {"f_causal": {"a": {"mean": 0.02, "wilcoxon_p": 0.03},
                                                 "b": {"mean": 0.005, "wilcoxon_p": 0.5}}}}
        result = wbc_delta(dirty, clean)
        assert set(result.keys()) == {"a", "b"}


# ── variance_delta ────────────────────────────────────────────────────────────

class TestVarianceDelta:
    """variance_delta(dirty, clean) -> {f_causal: {dirty, clean, delta}, f_spatial: {...}}"""

    def test_delta_arithmetic(self):
        from famail_temporal.analysis.experiment_delta import variance_delta
        dirty = {"paired_delta": {
            "f_causal": {"mean": -0.00114, "std": 0.00190},
            "f_spatial": {"mean":  0.000885, "std": 0.000486},
        }}
        clean = {"paired_delta": {
            "f_causal": {"mean": -0.000370, "std": 0.00144},
            "f_spatial": {"mean": -0.000215, "std": 0.000409},
        }}
        result = variance_delta(dirty, clean)
        assert set(result.keys()) == {"f_causal", "f_spatial"}
        fc = result["f_causal"]
        assert set(fc.keys()) >= {"dirty", "clean", "delta"}
        assert _approx(fc["dirty"], -0.00114)
        assert _approx(fc["clean"], -0.000370)
        assert _approx(fc["delta"], -0.000370 - (-0.00114))

    def test_f_spatial_delta(self):
        from famail_temporal.analysis.experiment_delta import variance_delta
        dirty = {"paired_delta": {
            "f_causal": {"mean": -0.001, "std": 0.002},
            "f_spatial": {"mean": 0.001, "std": 0.0005},
        }}
        clean = {"paired_delta": {
            "f_causal": {"mean": -0.0003, "std": 0.0014},
            "f_spatial": {"mean": -0.0002, "std": 0.0004},
        }}
        result = variance_delta(dirty, clean)
        fs = result["f_spatial"]
        assert _approx(fs["dirty"], 0.001)
        assert _approx(fs["clean"], -0.0002)
        assert _approx(fs["delta"], -0.0002 - 0.001)

    def test_missing_metric_graceful(self):
        """If f_spatial missing from paired_delta, result should be None not crash."""
        from famail_temporal.analysis.experiment_delta import variance_delta
        dirty = {"paired_delta": {"f_causal": {"mean": -0.001, "std": 0.002}}}
        clean = {"paired_delta": {"f_causal": {"mean": -0.0003, "std": 0.0014}}}
        result = variance_delta(dirty, clean)
        assert result["f_causal"]["dirty"] is not None
        assert result["f_spatial"]["dirty"] is None
        assert result["f_spatial"]["clean"] is None
        assert result["f_spatial"]["delta"] is None
