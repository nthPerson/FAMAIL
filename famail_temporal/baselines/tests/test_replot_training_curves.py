"""Tests for replot_training_curves.py — TDD-first, headless, no GPU.

Run with:
    python -m pytest famail_temporal/baselines/tests/test_replot_training_curves.py -v
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Pure-helper tests (no matplotlib)
# ---------------------------------------------------------------------------
from famail_temporal.baselines.replot_training_curves import (
    clip_report,
    default_window,
    read_series_csv,
    robust_ylim,
    rolling_mean,
)


# ---------------------------------------------------------------------------
# 1. read_series_csv
# ---------------------------------------------------------------------------
def test_read_series_csv_basic(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("step,loss\n0,1.5\n1,2.5\n2,0.5\n")
    result = read_series_csv(p)
    assert result == [1.5, 2.5, 0.5]


def test_read_series_csv_header_skipped(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("step,loss\n10,99.0\n20,88.0\n30,77.0\n")
    result = read_series_csv(p)
    # Only 3 floats — header not included
    assert len(result) == 3
    assert result[0] == 99.0


def test_read_series_csv_order_preserved(tmp_path):
    p = tmp_path / "test.csv"
    rows = [5.0, 3.0, 7.0]
    lines = ["step,loss\n"] + [f"{i},{v}\n" for i, v in enumerate(rows)]
    p.write_text("".join(lines))
    result = read_series_csv(p)
    assert result == rows


# ---------------------------------------------------------------------------
# 2. rolling_mean
# ---------------------------------------------------------------------------
def test_rolling_mean_window_1_returns_copy():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = rolling_mean(vals, 1)
    assert result == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_rolling_mean_window_0_returns_copy():
    vals = [1.0, 2.0, 3.0]
    result = rolling_mean(vals, 0)
    assert result == [1.0, 2.0, 3.0]


def test_rolling_mean_same_length():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = rolling_mean(vals, 3)
    assert len(result) == 5


def test_rolling_mean_middle_value():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = rolling_mean(vals, 3)
    # Middle index 2: window covers [1,2]=max(0,2-1)=1 to 2+1+1=4 → values[1:4]=[2,3,4]
    assert abs(result[2] - 3.0) < 1e-9


def test_rolling_mean_no_nans():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = rolling_mean(vals, 3)
    for v in result:
        assert not math.isnan(v)


def test_rolling_mean_large_window():
    # Window larger than series — should still work without errors
    vals = [1.0, 2.0, 3.0]
    result = rolling_mean(vals, 100)
    assert len(result) == 3
    for v in result:
        assert not math.isnan(v)


# ---------------------------------------------------------------------------
# 3. robust_ylim
# ---------------------------------------------------------------------------
def test_robust_ylim_excludes_outlier():
    # 100 normal values 0-99, one extreme outlier — realistic series length
    vals = list(range(100)) + [1000]
    result = robust_ylim(vals, lo_pct=1.0, hi_pct=99.0)
    assert result is not None
    y_lo, y_hi = result
    # Outlier at 1000 should be well excluded from hi (99th pct of 100 normal vals ≈ 99)
    assert y_hi < 200


def test_robust_ylim_too_few_returns_none():
    # Only one finite value
    result = robust_ylim([5.0])
    assert result is None


def test_robust_ylim_all_equal_returns_none():
    result = robust_ylim([3.0, 3.0, 3.0, 3.0])
    assert result is None


def test_robust_ylim_margin_expansion():
    vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    result = robust_ylim(vals, lo_pct=0.0, hi_pct=100.0, margin=0.1)
    assert result is not None
    y_lo, y_hi = result
    # span=5, margin expands by 0.1*5=0.5 on each side
    assert y_lo < 0.0
    assert y_hi > 5.0


def test_robust_ylim_ignores_nan():
    vals = [0.0, float("nan"), 1.0, 2.0, float("inf"), 3.0]
    # Should not raise; finite values are [0, 1, 2, 3]
    result = robust_ylim(vals)
    assert result is not None


# ---------------------------------------------------------------------------
# 4. clip_report
# ---------------------------------------------------------------------------
def test_clip_report_no_clipped():
    vals = [1.0, 2.0, 3.0]
    r = clip_report(vals, 0.0, 5.0)
    assert r["n_clipped"] == 0
    assert r["max"] == 3.0
    assert r["argmax"] == 2


def test_clip_report_counts_clipped():
    # spike at index 3
    vals = [0.5, 0.6, 0.7, 999.0, 0.8]
    r = clip_report(vals, 0.0, 10.0)
    assert r["n_clipped"] == 1
    assert r["max"] == 999.0
    assert r["argmax"] == 3


def test_clip_report_below_lo_clipped():
    vals = [-50.0, 1.0, 2.0]
    r = clip_report(vals, 0.0, 10.0)
    assert r["n_clipped"] == 1


def test_clip_report_both_sides_clipped():
    vals = [-100.0, 1.0, 2.0, 3.0, 9999.0]
    r = clip_report(vals, 0.0, 100.0)
    assert r["n_clipped"] == 2


# ---------------------------------------------------------------------------
# 5. default_window
# ---------------------------------------------------------------------------
def test_default_window_short_series():
    # n=100 → max(1, 100//200) = max(1, 0) = 1
    assert default_window(100) == 1


def test_default_window_medium_series():
    # n=40000 → max(1, 40000//200) = 200, below cap 500
    assert default_window(40000) == 200


def test_default_window_very_long_capped():
    # n=1_000_000 → max(1, 5000) → capped at 500
    assert default_window(1_000_000) == 500


def test_default_window_minimum_one():
    # Never returns 0
    assert default_window(1) >= 1
    assert default_window(0) >= 1


# ---------------------------------------------------------------------------
# 6. Headless render tests
# ---------------------------------------------------------------------------
def test_render_smoothed_creates_nonempty_png(tmp_path):
    from famail_temporal.baselines.replot_training_curves import render_smoothed

    out = tmp_path / "s.png"
    returned = render_smoothed("t", {"a": [3.0, 2.0, 1.0, 0.5, 0.4]}, out)
    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_smoothed_multiple_series(tmp_path):
    from famail_temporal.baselines.replot_training_curves import render_smoothed

    out = tmp_path / "multi.png"
    render_smoothed(
        "Multi",
        {"series_a": [1.0, 0.9, 0.8], "series_b": [2.0, 1.5, 1.0]},
        out,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_adversarial_linear_with_spike(tmp_path):
    """Linear mode with outlier in d → exercises clip annotation."""
    from famail_temporal.baselines.replot_training_curves import render_adversarial

    out = tmp_path / "adv.png"
    returned = render_adversarial(
        [0.1, 0.2, 0.3],
        [1.0, 2.0, 80000.0],  # spike → clipped
        out,
    )
    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_adversarial_symlog(tmp_path):
    """Symlog mode — spike visible, no clip annotation."""
    from famail_temporal.baselines.replot_training_curves import render_adversarial

    out = tmp_path / "adv_symlog.png"
    returned = render_adversarial(
        [0.1, 0.2, 0.3],
        [1.0, 2.0, 80000.0],
        out,
        yscale="symlog",
    )
    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_adversarial_no_spike(tmp_path):
    """All points inside clip bounds — n_clipped==0, no annotation."""
    from famail_temporal.baselines.replot_training_curves import render_adversarial

    out = tmp_path / "adv_clean.png"
    render_adversarial(
        [0.5, 0.4, 0.3],
        [0.6, 0.5, 0.4],
        out,
    )
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# 7. main() integration
# ---------------------------------------------------------------------------
def _write_csv(path: Path, losses: list[float]) -> None:
    path.write_text("step,loss\n" + "".join(f"{i},{v}\n" for i, v in enumerate(losses)))


def _make_level1_curves(curves_dir: Path) -> None:
    """Populate a curves dir with GAN MLE + adversarial CSVs."""
    curves_dir.mkdir(parents=True, exist_ok=True)
    # GAN MLE
    _write_csv(curves_dir / "gan_mle_batch.csv", [2.0 - i * 0.01 for i in range(300)])
    # Adversarial — spike in d at step 150
    g_vals = [0.5 + 0.001 * i for i in range(300)]
    d_vals = [0.6 for _ in range(300)]
    d_vals[150] = 50000.0  # spike
    _write_csv(curves_dir / "gan_adv_g_batch.csv", g_vals)
    _write_csv(curves_dir / "gan_adv_d_batch.csv", d_vals)


def test_main_returns_0_and_writes_pngs(tmp_path):
    from famail_temporal.baselines.replot_training_curves import main

    curves_dir = tmp_path / "curves"
    _make_level1_curves(curves_dir)

    ret = main(["--curves-dir", str(curves_dir)])
    assert ret == 0

    assert (curves_dir / "gan_mle.png").exists()
    assert (curves_dir / "gan_adversarial.png").exists()


def test_main_custom_out_dir(tmp_path):
    from famail_temporal.baselines.replot_training_curves import main

    curves_dir = tmp_path / "curves"
    out_dir = tmp_path / "out"
    _make_level1_curves(curves_dir)

    ret = main(["--curves-dir", str(curves_dir), "--out-dir", str(out_dir)])
    assert ret == 0
    assert (out_dir / "gan_mle.png").exists()
    assert (out_dir / "gan_adversarial.png").exists()


def test_main_empty_dir_returns_1(tmp_path):
    from famail_temporal.baselines.replot_training_curves import main

    empty = tmp_path / "empty"
    empty.mkdir()
    ret = main(["--curves-dir", str(empty)])
    assert ret == 1


def test_main_bc_mle(tmp_path):
    from famail_temporal.baselines.replot_training_curves import main

    curves_dir = tmp_path / "curves"
    curves_dir.mkdir()
    _write_csv(curves_dir / "bc_mle_batch.csv", [1.0, 0.9, 0.8] * 10)

    ret = main(["--curves-dir", str(curves_dir)])
    assert ret == 0
    assert (curves_dir / "bc_mle.png").exists()


def test_main_variance_seeds(tmp_path):
    from famail_temporal.baselines.replot_training_curves import main

    curves_dir = tmp_path / "curves"
    curves_dir.mkdir()
    for seed in range(5):
        _write_csv(curves_dir / f"b0_seed{seed}_mle.csv", [1.0 - seed * 0.05] * 20)
        _write_csv(curves_dir / f"famail_seed{seed}_mle.csv", [0.8 - seed * 0.05] * 20)

    ret = main(["--curves-dir", str(curves_dir)])
    assert ret == 0
    assert (curves_dir / "b0_mle.png").exists()
    assert (curves_dir / "famail_mle.png").exists()


def test_main_symlog_flag(tmp_path):
    from famail_temporal.baselines.replot_training_curves import main

    curves_dir = tmp_path / "curves"
    _make_level1_curves(curves_dir)

    ret = main(["--curves-dir", str(curves_dir), "--yscale", "symlog"])
    assert ret == 0
    assert (curves_dir / "gan_adversarial.png").exists()
