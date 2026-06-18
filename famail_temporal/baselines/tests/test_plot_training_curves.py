"""Tests for plot_training_curves.py — TDD-first, headless, no GPU."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Pure-helper tests — no matplotlib needed
# ---------------------------------------------------------------------------
from famail_temporal.baselines.plot_training_curves import (
    flatten_level1_curves,
    series_csv,
    variance_model_series,
)


# ---------------------------------------------------------------------------
# 1. series_csv
# ---------------------------------------------------------------------------
def test_series_csv_basic():
    result = series_csv([2.0, 1.5])
    assert result == "step,loss\n0,2.0\n1,1.5\n"


def test_series_csv_empty():
    result = series_csv([])
    assert result == "step,loss\n"


# ---------------------------------------------------------------------------
# 2. flatten_level1_curves
# ---------------------------------------------------------------------------
FAKE_CURVES = {
    "bc": {
        "mle_epoch_losses": [1.0, 0.9],
        "mle_batch_losses": [1.1, 1.0, 0.95],
        "adv": None,
    },
    "gan": {
        "mle_epoch_losses": [1.2, 1.1],
        "mle_batch_losses": [1.3, 1.2, 1.1, 1.0],
        "adv": {
            "g_epoch_losses": [0.5, 0.4],
            "d_epoch_losses": [0.6, 0.5],
            "g_batch_losses": [0.55, 0.50, 0.45],
            "d_batch_losses": [0.65, 0.60, 0.55],
        },
    },
}


def test_flatten_level1_curves_expected_keys():
    result = flatten_level1_curves(FAKE_CURVES)
    expected = {
        "bc_mle_epoch",
        "bc_mle_batch",
        "gan_mle_epoch",
        "gan_mle_batch",
        "gan_adv_g_epoch",
        "gan_adv_d_epoch",
        "gan_adv_g_batch",
        "gan_adv_d_batch",
    }
    assert set(result.keys()) == expected


def test_flatten_level1_curves_no_bc_adv_keys():
    result = flatten_level1_curves(FAKE_CURVES)
    for key in result:
        assert not key.startswith("bc_adv_"), f"unexpected bc_adv_ key: {key}"


def test_flatten_level1_curves_values_spot_check():
    result = flatten_level1_curves(FAKE_CURVES)
    assert result["bc_mle_batch"] == [1.1, 1.0, 0.95]
    assert result["gan_adv_g_batch"] == [0.55, 0.50, 0.45]
    assert result["gan_adv_d_epoch"] == [0.6, 0.5]


def test_flatten_level1_curves_omits_empty_series():
    curves = {
        "bc": {
            "mle_epoch_losses": [],          # empty → omitted
            "mle_batch_losses": [1.0, 0.9],
            "adv": None,
        },
        "gan": {
            "mle_epoch_losses": [1.2],
            "mle_batch_losses": [],           # empty → omitted
            "adv": {
                "g_epoch_losses": [0.5],
                "d_epoch_losses": [],         # empty → omitted
                "g_batch_losses": [],
                "d_batch_losses": [0.6],
            },
        },
    }
    result = flatten_level1_curves(curves)
    assert "bc_mle_epoch" not in result    # empty source
    assert "bc_mle_batch" in result
    assert "gan_mle_batch" not in result   # empty source
    assert "gan_adv_d_epoch" not in result # empty source
    assert "gan_adv_d_batch" in result


# ---------------------------------------------------------------------------
# 3. variance_model_series
# ---------------------------------------------------------------------------
SEED_ENTRIES_MIXED = [
    # seed 0: new file — has mle_batch_losses (non-empty) → use it
    {
        "seed": 0,
        "b0": {
            "mle_losses": [1.0, 0.9, 0.8],
            "mle_batch_losses": [1.1, 1.0, 0.95, 0.9],
            "adv_curve": None,
        },
    },
    # seed 1: old file — no mle_batch_losses → fall back to mle_losses
    {
        "seed": 1,
        "b0": {
            "mle_losses": [1.2, 1.1, 1.0],
            # no mle_batch_losses key at all
        },
    },
]


def test_variance_model_series_keys():
    result = variance_model_series(SEED_ENTRIES_MIXED, "b0")
    assert set(result.keys()) == {"b0_seed0_mle", "b0_seed1_mle"}


def test_variance_model_series_uses_batch_when_present():
    result = variance_model_series(SEED_ENTRIES_MIXED, "b0")
    # seed 0 has mle_batch_losses → should be used
    assert result["b0_seed0_mle"] == [1.1, 1.0, 0.95, 0.9]


def test_variance_model_series_fallback_to_epoch():
    result = variance_model_series(SEED_ENTRIES_MIXED, "b0")
    # seed 1 has only mle_losses → per-epoch fallback
    assert result["b0_seed1_mle"] == [1.2, 1.1, 1.0]


def test_variance_model_series_skips_seed_with_neither():
    entries = [
        {"seed": 3, "b0": {}},           # neither key
        {"seed": 4, "b0": {"mle_batch_losses": [], "mle_losses": []}},  # both empty
        {"seed": 5, "b0": {"mle_losses": [0.8]}},  # only epoch
    ]
    result = variance_model_series(entries, "b0")
    assert "b0_seed3_mle" not in result
    assert "b0_seed4_mle" not in result
    assert "b0_seed5_mle" in result


# ---------------------------------------------------------------------------
# 4. Headless render test
# ---------------------------------------------------------------------------
def test_plot_series_group_creates_nonempty_png(tmp_path):
    # Import here so matplotlib Agg backend is already set by the module
    from famail_temporal.baselines.plot_training_curves import plot_series_group

    out_png = tmp_path / "x.png"
    returned = plot_series_group("Test title", {"a": [1.0, 0.5, 0.2]}, out_png)
    assert returned == out_png
    assert out_png.exists()
    assert out_png.stat().st_size > 0


# ---------------------------------------------------------------------------
# 5. main() integration test
# ---------------------------------------------------------------------------
def test_main_level1_writes_csv_and_png(tmp_path):
    from famail_temporal.baselines.plot_training_curves import main

    # Write a minimal training_curves.json
    curves = {
        "bc": {
            "mle_epoch_losses": [1.0, 0.9],
            "mle_batch_losses": [1.05, 0.95],
            "adv": None,
        },
        "gan": {
            "mle_epoch_losses": [1.2, 1.1],
            "mle_batch_losses": [1.25, 1.15],
            "adv": {
                "g_epoch_losses": [0.5, 0.4],
                "d_epoch_losses": [0.6, 0.5],
                "g_batch_losses": [0.55, 0.45],
                "d_batch_losses": [0.65, 0.55],
            },
        },
    }
    (tmp_path / "training_curves.json").write_text(json.dumps(curves))

    out_dir = tmp_path / "out"
    ret = main(["--level1-dir", str(tmp_path), "--out-dir", str(out_dir)])

    assert ret == 0
    pngs = list(out_dir.glob("*.png"))
    csvs = list(out_dir.glob("*.csv"))
    assert len(pngs) >= 1, "expected at least one PNG"
    assert len(csvs) >= 1, "expected at least one CSV"


def test_main_no_args_errors():
    from famail_temporal.baselines.plot_training_curves import main

    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0
