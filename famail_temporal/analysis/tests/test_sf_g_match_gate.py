"""G-match integration gate (D1 Task 3): SF substitution replay.

Replays the committed sf12 supply-lift corpus's ``histories.pkl`` against the
SF ping adapter's counting DataFrame via the EXISTING city-agnostic
``supply_recount.apply_substitutions`` machinery -- fed the SF-native match
lookup ``sf_recount_adapter.build_sf_seeking_lookup`` (300s-gap segmentation +
weekday day space, spec addendum 2026-07-17) -- and asserts 100% of the
histories match their raw source sequences: ``n_matched == n_histories`` AND
``n_unmatched == 0`` (design spec ``2026-07-17-d1-sf-tier2-recount-design.md``
§2 "G-match").

Also pins the count/match split's key invariant: substitution only moves cells
(x_grid/y_grid), so every row's ``day_index`` and ``hour`` in the counting df
are unchanged by the replay -- the absolute-day counting path (Task 2 G-repro,
MAE 0.0) is untouched.

Runs in a SUBPROCESS with ``FAMAIL_CITY=sf12`` set before any
``famail_temporal`` import (``famail_temporal.config`` resolves + caches CITY at
import time), mirroring Task 2's ``test_sf_g_repro_gate``. Reuses Task 2's
skip-when-SF-data-absent guards.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

# Reuse Task 2's gate fixtures (repo root, SF-data-present guard) verbatim.
from famail_temporal.analysis.tests.test_supply_recount_sf import (
    _REPO_ROOT, _CAB_DIR, _SF12_CACHE, _SF12_SOURCE, _SF_DATA_ABSENT,
)

_GATE_SCRIPT = r"""
import json, os
os.environ["FAMAIL_CITY"] = "sf12"

from famail_temporal import config
from famail_temporal.analysis.sf_recount_adapter import (
    load_sf_pings, build_sf_seeking_lookup,
)
from famail_temporal.analysis.supply_recount import (
    apply_substitutions, _load_driver_mapping, _load_histories,
)

edit_dir = config.PACKAGE_ROOT / "results" / "2026-07-11T11-31-55_supply_lift_a10_sf12_filtered"
idx_to_plate = _load_driver_mapping(config)
target_plates = set(idx_to_plate.values())
assert target_plates, "empty driver_index_mapping.pkl for sf12"

# Mirror supply_recount.main()'s SF branch exactly: quantize the FULL fleet
# (production grid derivation) then restrict to the 12 sf12 plates.
raw_dir = config.PACKAGE_ROOT / "source_data" / "second_dataset" / "cabspottingdata"
raw_sf_df = load_sf_pings(raw_dir)
es_df = raw_sf_df[raw_sf_df["plate_id"].isin(target_plates)].reset_index(drop=True)
assert len(es_df) > 0, "no rows left after filtering to the sf12 driver subset"

histories = _load_histories(edit_dir)

# SF-native match lookup (300s-gap + weekday), row indices into es_df.
sf_seg_lookup = build_sf_seeking_lookup(es_df, raw_dir, idx_to_plate)
df_after, stats = apply_substitutions(
    es_df, histories, idx_to_plate, seg_lookup=sf_seg_lookup,
)
stats.pop("unmatched_examples", None)

# Count/match split invariant: replay moves only x/y cells; day_index & hour
# (the absolute-day counting keys) must be bit-identical pre/post replay.
stats["day_index_invariant"] = bool(df_after["day_index"].equals(es_df["day_index"]))
stats["hour_invariant"] = bool(df_after["hour"].equals(es_df["hour"]))
stats["n_xy_rows_changed"] = int(
    ((df_after["x_grid"] != es_df["x_grid"]) | (df_after["y_grid"] != es_df["y_grid"])).sum()
)
print("__RESULT__" + json.dumps(stats))
"""


@pytest.mark.skipif(_SF_DATA_ABSENT, reason=(
    "SF Cabspotting raw data / preprocessed sf12 cache absent on this machine "
    f"(checked {_CAB_DIR}, {_SF12_CACHE}, {_SF12_SOURCE})"
))
def test_sf_g_match_gate():
    """G-match (design spec §2): every edited-corpus history must match its raw
    source seeking-segment sequence via the same replay identification used on
    SZ -- ``n_matched == n_histories`` and ``n_unmatched == 0``. A shortfall is
    a STOP condition (diagnose against the source-generation anchors, in
    particular the adapter's documented segmentation/day divergences; do not
    relax the criteria toward agreement), not something this gate tolerates.
    """
    env = dict(os.environ)
    env["FAMAIL_CITY"] = "sf12"
    proc = subprocess.run(
        [sys.executable, "-c", _GATE_SCRIPT],
        cwd=str(_REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"G-match subprocess failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    result_lines = [l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")]
    assert result_lines, f"no __RESULT__ line in subprocess stdout:\n{proc.stdout}"
    stats = json.loads(result_lines[-1][len("__RESULT__"):])

    n_hist = stats["n_histories"]
    assert n_hist > 0, stats  # guard against a trivially-empty corpus

    assert stats["n_matched"] == n_hist, (
        f"G-match gate FAILED: only {stats['n_matched']}/{n_hist} histories "
        f"matched their raw source sequences via apply_substitutions. STOP per "
        f"the design spec -- diagnose (which histories fail, and whether the "
        f"adapter's 300s-gap / day-encoding divergences explain them); do not "
        f"relax the criteria. Full stats: {stats}"
    )
    assert stats["n_unmatched"] == 0, (
        f"G-match gate FAILED: n_unmatched={stats['n_unmatched']} (expected 0). "
        f"Full stats: {stats}"
    )

    # Count/match split invariant: substitution moves only cells, never the
    # absolute-day counting keys.
    assert stats["day_index_invariant"], (
        f"substitution changed day_index -- it must only move x/y cells: {stats}"
    )
    assert stats["hour_invariant"], (
        f"substitution changed hour -- it must only move x/y cells: {stats}"
    )
    # Sanity: the replay actually did move some cells (not a trivial no-op).
    assert stats["n_moved_states"] > 0 and stats["n_xy_rows_changed"] > 0, stats
