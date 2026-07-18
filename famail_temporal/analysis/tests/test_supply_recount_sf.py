"""Tests for the SF (sf12) counting path added to
``famail_temporal.analysis.supply_recount`` (D1 Task 2).

Two things are covered:

1. ``test_build_active_taxis_counts_sf_*`` -- fast, synthetic-data unit tests
   pinning the two documented divergences from SZ's
   ``active_taxis_view.build_active_taxis_counts`` (Task 1 adapter docstring
   "Known divergence" section): no occupancy filter, city-aware clip bounds.
2. ``test_sf_g_repro_gate`` -- the G-repro INTEGRATION gate (spec
   ``docs/superpowers/specs/2026-07-17-d1-sf-tier2-recount-design.md`` §2):
   recounting the UNEDITED sf12 corpus from raw Cabspotting pings, via the
   new SF-mirrored counting path, must reproduce the production
   ``bundle.active_taxis_3d`` grid EXACTLY (MAE 0.0). Runs in a SUBPROCESS
   with ``FAMAIL_CITY=sf12`` set before any ``famail_temporal`` import --
   ``famail_temporal.config`` resolves ``CITY`` at import time and caches it
   module-level, so flipping ``os.environ`` mid-process after some other test
   in the same pytest session has already imported ``famail_temporal.config``
   (with the default "shenzhen") would silently leave the wrong city active.
   A subprocess is the only reliable way to get a clean sf12 config, and it
   mirrors how ``supply_recount.main()`` itself is actually invoked (a fresh
   process per ``--city``, module docstring lines ~420-423).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from famail_temporal.analysis.supply_recount import _build_active_taxis_counts_sf

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CAB_DIR = _REPO_ROOT / "famail_temporal/source_data/second_dataset/cabspottingdata"
_SF12_CACHE = _REPO_ROOT / "famail_temporal/cache/sf_12"
_SF12_SOURCE = _REPO_ROOT / "famail_temporal/source_data/second_dataset/sf_source_12"

_SF_DATA_ABSENT = not (
    _CAB_DIR.is_dir() and any(_CAB_DIR.glob("new_*.txt"))
    and _SF12_CACHE.is_dir() and _SF12_SOURCE.is_dir()
)


# ---------------------------------------------------------------------------
# Fast synthetic-data unit tests for the SF-mirrored counter.
# ---------------------------------------------------------------------------

def _row(plate, x, y, hour, day, occ):
    return {
        "plate_id": plate, "x_grid": x, "y_grid": y, "hour": hour,
        "day_index": day, "passenger_indicator": occ,
    }


def test_build_active_taxis_counts_sf_no_occupancy_filter():
    """SF's own count_active_taxis_5x5 (sf_grid_counts.py:47-64) has no
    occupancy filter -- a driver present only while occupied (occ=1) must
    still be counted. SZ's build_active_taxis_counts would return {} here
    (it filters to passenger_indicator == 0 first)."""
    df = pd.DataFrame([_row("cab_0001", 5, 5, 10, 1, occ=1)])
    counts = _build_active_taxis_counts_sf(df, x_grid_max=32, y_grid_max=30, k=2)
    assert counts.get((5, 5, 10, 1)) == 1


def test_build_active_taxis_counts_sf_clip_bounds_are_parameterized():
    """A ping near the edge of a SMALL (e.g. sf12 32x30) grid must not spread
    into cells beyond x_grid_max/y_grid_max -- unlike SZ's hardcoded 48x90
    clip (data/source_generation/config.X_GRID_MAX/Y_GRID_MAX), which would
    permit cells up to 48/90 regardless of the grid actually in use."""
    df = pd.DataFrame([_row("cab_0001", 32, 30, 0, 1, occ=0)])
    counts = _build_active_taxis_counts_sf(df, x_grid_max=32, y_grid_max=30, k=2)
    assert all(x <= 32 and y <= 30 for (x, y, _, _) in counts.keys())
    # The full 5x5 neighborhood is clipped to 3x3 at this corner (x in
    # [30,32], y in [28,30]).
    assert set(counts.keys()) == {
        (x, y, 0, 1) for x in range(30, 33) for y in range(28, 31)
    }


def test_build_active_taxis_counts_sf_distinct_taxi_dedup():
    """Two pings from the SAME driver in the same source cell/hour still
    count as 1 (distinct-taxi, not distinct-ping)."""
    df = pd.DataFrame([
        _row("cab_0001", 5, 5, 10, 1, occ=0),
        _row("cab_0001", 5, 5, 10, 1, occ=1),  # same cell/hour/day, 2nd ping
    ])
    counts = _build_active_taxis_counts_sf(df, x_grid_max=32, y_grid_max=30, k=2)
    assert counts[(5, 5, 10, 1)] == 1


def test_build_active_taxis_counts_sf_empty_df():
    counts = _build_active_taxis_counts_sf(
        pd.DataFrame(columns=["plate_id", "x_grid", "y_grid", "hour", "day_index"]),
        x_grid_max=32, y_grid_max=30, k=2,
    )
    assert counts == {}


# ---------------------------------------------------------------------------
# G-repro integration gate.
# ---------------------------------------------------------------------------

_GATE_SCRIPT = r"""
import json, os
os.environ["FAMAIL_CITY"] = "sf12"

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.data.aggregation import aggregate_active_taxis
from famail_temporal.analysis.sf_recount_adapter import load_sf_pings
from famail_temporal.analysis.supply_recount import (
    recount_tier2_sf, _grid_compare, _load_driver_mapping,
)
from famail_temporal.second_dataset.data.source_generation import sf_grid_counts

bundle = DataBundle.load()
idx_to_plate = _load_driver_mapping(config)
target_plates = set(idx_to_plate.values())
assert target_plates, "empty driver_index_mapping.pkl for sf12"

# Mirrors sf_build.build(): the production grid is derived from the FULL
# cabspotting fleet's lat/lon (grid_from_points runs before the driver_ids
# filter, sf_build.py:37-41) -- so raw_dir must be the full fleet dir to
# reproduce the same grid quantization.
raw_dir = config.PACKAGE_ROOT / "source_data" / "second_dataset" / "cabspottingdata"
raw_sf_df = load_sf_pings(raw_dir)
es_df = raw_sf_df[raw_sf_df["plate_id"].isin(target_plates)].reset_index(drop=True)
assert len(es_df) > 0, "no rows left after filtering to the sf12 driver subset"

n_days = bundle.n_days
x_grid_max, y_grid_max = config.GRID_DIMS
S_before, _ = recount_tier2_sf(
    es_df, n_days, x_grid_max, y_grid_max, sf_grid_counts.NEIGHBORHOOD_K,
    aggregate_active_taxis,
)
cmp = _grid_compare(S_before, bundle.active_taxis_3d, bundle.mask_3d)
print("__RESULT__" + json.dumps(cmp))
"""


@pytest.mark.skipif(_SF_DATA_ABSENT, reason=(
    "SF Cabspotting raw data / preprocessed sf12 cache absent on this machine "
    f"(checked {_CAB_DIR}, {_SF12_CACHE}, {_SF12_SOURCE})"
))
def test_sf_g_repro_gate():
    """G-repro (design spec §2): recounting the UNEDITED sf12 corpus (no
    substitutions) via the new SF-mirrored counting path must reproduce the
    production active_taxis_3d grid EXACTLY -- MAE 0.0, over every active
    (mask_3d) cell. A nonzero MAE here is a STOP condition (diagnose against
    the source-generation anchors; do not tune toward agreement), not
    something this test should be relaxed to tolerate.
    """
    env = dict(os.environ)
    env["FAMAIL_CITY"] = "sf12"
    proc = subprocess.run(
        [sys.executable, "-c", _GATE_SCRIPT],
        cwd=str(_REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"G-repro subprocess failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    result_lines = [l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")]
    assert result_lines, f"no __RESULT__ line in subprocess stdout:\n{proc.stdout}"
    cmp = json.loads(result_lines[-1][len("__RESULT__"):])

    # Sanity: comparison must actually cover real active cells, not an
    # accidentally-empty mask (which would make MAE == 0.0 trivially true).
    assert cmp["n_active_cells"] > 0, cmp

    assert cmp["mae"] == 0.0, (
        f"G-repro gate FAILED: unedited sf12 recount does not exactly "
        f"reproduce production active_taxis_3d (MAE={cmp['mae']!r}, "
        f"max_abs_diff={cmp['max_abs_diff']!r}). STOP per the design spec -- "
        f"diagnose against the source-generation anchors, do not tune "
        f"toward agreement. Full comparison: {cmp}"
    )
    assert cmp["max_abs_diff"] == 0.0, cmp
