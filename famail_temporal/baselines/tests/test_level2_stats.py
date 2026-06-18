from famail_temporal.baselines import run_level2_table as r2


def test_paired_diff_stats_basic():
    per_seed = {
        "edited": [0.82, 0.83, 0.81, 0.84, 0.82],
        "raw":    [0.80, 0.81, 0.80, 0.81, 0.80],
        "bcgen":  [0.80, 0.80, 0.81, 0.80, 0.81],
    }
    out = r2._paired_diff_stats(per_seed, baseline="edited")
    assert set(out) == {"raw", "bcgen"}
    raw = out["raw"]
    assert raw["n"] == 5
    assert abs(raw["mean"] - sum(e - r for e, r in
               zip(per_seed["edited"], per_seed["raw"])) / 5) < 1e-9
    assert len(raw["diffs"]) == 5
    # wilcoxon_p present (float) or None if scipy missing
    assert raw["wilcoxon_p"] is None or 0.0 <= raw["wilcoxon_p"] <= 1.0


def test_paired_diff_stats_handles_constant_and_missing_scipy():
    per_seed = {"edited": [0.5, 0.5], "raw": [0.5, 0.5]}
    out = r2._paired_diff_stats(per_seed, baseline="edited")
    assert out["raw"]["mean"] == 0.0
    assert out["raw"]["wilcoxon_p"] is None   # all-zero diffs -> no test
