"""SF (Cabspotting) ping-loader adapter for the tier-2 supply recount tool
(D1 Task 1 -- discovery + adapter only; wiring into ``supply_recount.py`` is a
later task and is NOT done here).

``load_sf_pings(raw_dir)`` loads raw SF Cabspotting ``new_*.txt`` traces and
returns a row-level DataFrame in EXACTLY the schema
``famail_temporal.analysis.supply_recount``'s ``recount_tier2()`` /
``apply_substitutions()`` consume for Shenzhen -- so a later task can plug SF
into that tool's raw-GPS re-segmentation path without changing its logic.
Every transform below is IMPORTED or REPLICATED VERBATIM from the SF
source-generation pipeline (never re-derived); ping presence is counted AS
PINGED (no interpolation, no gap "fixing" -- SF GPS gaps up to ~18.6 cells are
a known, accepted property of this raw data; see
`.superpowers/sdd/task-11e-sf-eval-report.md` §1 for empirical grounding).

=== Step 1: the consumed SZ schema (discovery) ===================

Source: ``famail_temporal/analysis/supply_recount.py``.

``recount_tier2()`` (supply_recount.py:213-219) calls
``active_taxis_view.build_active_taxis_counts(df)``
(``famail_temporal/data/source_generation/views/active_taxis.py:14-59``),
which reads:
  - ``plate_id``            -- driver key (str).
  - ``x_grid``, ``y_grid``  -- 1-indexed grid cell (int), clipped to the
                                city's [1, X_GRID_MAX] x [1, Y_GRID_MAX].
  - ``hour``                -- 0-indexed local hour of day, int in [0, 23].
  - ``day_index``           -- calendar-day grouping key (int).
  - ``passenger_indicator`` -- 0 = empty/seeking, 1 = occupied/driving (int);
                                ``build_active_taxis_counts`` filters to
                                ``passenger_indicator == 0`` before counting
                                distinct-taxi presence in each cell's 5x5
                                neighborhood.

``apply_substitutions()`` / ``_segment_rows_by_driver()``
(supply_recount.py:85-192) re-segment the SAME raw event-stream df by
grouping on:
  - ``plate_id``    (``df.groupby("plate_id")``)
  - ``segment_id``  (``driver_df.groupby("segment_id")``) -- each group is one
                     candidate seeking-trajectory segment; the group's LAST
                     row's ``is_pickup`` must be True for the segment to be
                     kept (supply_recount.py:96).
  - within a segment: ``x_grid``, ``y_grid``, ``time_bucket``, ``day_index``
    (1-indexed cell + 5-min bucket + day) are read per-row to build the
    state-value tuple used to match ``histories.pkl`` trajectories back to
    raw rows (supply_recount.py:98-101).

Both consumers together require the row-level df to carry (in addition to the
raw lat/lon-derived columns): ``plate_id, x_grid, y_grid, hour, day_index,
time_bucket, passenger_indicator, is_pickup, segment_id``. These are exactly
the columns ``famail_temporal/data/source_generation/event_stream.py``'s
``build_event_stream()`` produces for Shenzhen (verified empirically:
``es.df[[...]].dtypes`` -> plate_id=str(object), x_grid/y_grid/hour/
day_index/time_bucket/passenger_indicator/segment_id=int64, is_pickup/
is_dropoff/is_transition=bool; SZ x_grid in [1,43], y_grid in [1,81] (well
inside the [1,48]x[1,90] city max), day_index in {1..5} (Mon-Fri only, see
``quantization.timestamp_to_day``), hour in [0,23], time_bucket in [1,288]).
``is_dropoff`` / ``is_transition`` are not read by ``recount_tier2`` /
``apply_substitutions`` directly but ARE part of SZ's actual event-stream df
(``add_transition_columns`` produces all three together) and are produced
here for the same reason (free byproduct of the reused transition logic, see
Step 2 below) -- this adapter's output is a strict superset of nothing;
every column below is either consumed or a same-computation byproduct.

=== Step 2: SF source-generation pipeline anchors =================

Source dir: ``famail_temporal/second_dataset/data/source_generation/``.

- Raw load -- ``sf_raw_loader.load_sf_raw`` (sf_raw_loader.py:22-55): parses
  ``new_<name>.txt`` (one line = ``lat lon occupancy time_utc``, integer-
  encodes each cab by SORTED FILENAME order, sorts each cab's pings ascending
  by time_utc). Columns: ``driver_id, lat, lon, occupancy, time_utc``.
  Occupancy convention (sf_raw_loader.py:3-6 docstring): "occupancy 1 =
  with-fare/driving, 0 = free/seeking" -- the SAME 0/1 convention as SZ's
  ``passenger_indicator`` (renamed here, not reinterpreted).

- Production grid derivation -- THE thing this task's STOP condition
  requires locating; found at ``sf_config.grid_from_points``
  (sf_config.py:43-52): builds a ``GridSpec`` from the 0.5th/99.5th
  percentile-trimmed lat/lon bbox of whatever points are passed in.
  ``sf_build.build()`` (sf_build.py:31-43) calls it on the FULL cabspotting
  fleet's lat/lon when no fixed ``grid`` is supplied (sf_build.py:38-39:
  ``grid = grid_from_points(df["lat"].to_numpy(), df["lon"].to_numpy())``)
  -- this is how the production sf12 ``active_taxis_3d``-equivalent grid
  (``active_taxis_5x5_hourly.pkl``, built by
  ``sf_grid_counts.count_active_taxis_5x5``, sf_grid_counts.py:36-75) got its
  documented 32x30 dimensions (``famail_temporal/config.py:35``:
  ``GRID_DIMS = (32, 30) if CITY.startswith("sf") else (48, 90)``;
  ``second_dataset/docs/SF_PHASE2_DECISIONS.md:16``). ``load_sf_pings``
  below calls ``grid_from_points`` the SAME way (on whatever ``raw_dir``
  holds) -- feeding it the full 536-file cabspotting fleet reproduces the
  exact production 32x30 grid; feeding it a small slice (as this module's
  unit test does, for speed) yields a smaller, internally-consistent grid
  (empirically verified: a single real driver's FULL 23k-ping file already
  bboxes to only 21x25, well under 32x30; small N-row slices are far
  smaller still) -- coordinates are still valid, just not necessarily
  reproducing the exact production cell indices, which requires the full
  fleet (Task 2's concern when it wires this in for the real recount run).

- Grid-cell quantization formula -- ``GridSpec.to_cell`` (sf_config.py:35-40)
  / vectorized equivalent duplicated in ``sf_grid_counts.count_active_taxis_5x5``
  (sf_grid_counts.py:54-57) and ``sf_segmentation.segment_driver``
  (sf_segmentation.py:67-70): 1-indexed, clipped floor-division cell index.
  Replicated verbatim below (same formula, cited both anchors since the
  pipeline itself duplicates it).

- Local-time (PDT) hour/day -- ``sf_grid_counts.count_active_taxis_5x5``
  (sf_grid_counts.py:58-60) / ``sf_segmentation.segment_driver``
  (sf_segmentation.py:71-73): ``local = time_utc - PDT_OFFSET_SEC``
  (``sf_config.PDT_OFFSET_SEC = 7*3600``, sf_config.py:17); ``hour =
  (local % 86400) // 3600``; ``day_index = local // 86400`` (an ABSOLUTE
  local epoch-day serial counter, NOT a 1-5 Mon-Fri weekday enum like SZ's
  ``day_index`` -- SF is documented 7-day, ``sf_config.py:15``:
  ``DAYS_IN_WEEK = 7  # SF is 7-day (vs Shenzhen Mon-Fri)``, no weekend
  filter is applied anywhere in the SF pipeline). Replicated verbatim.

- 5-min time_bucket -- ``sf_segmentation.segment_driver`` (sf_segmentation.py:72):
  ``tb = (local % 86400) // 300 + 1`` (1-indexed, naturally in [1, 288] since
  ``local % 86400`` in [0, 86399] -- matches SZ's ``TIME_BUCKET_MAX = 288``
  (data/source_generation/config.py:18) / SF's own
  ``N_TIME_BUCKETS = 288`` (sf_config.py:16)). Replicated verbatim.

- Plate-id naming -- ``sf_multistream.py:142``: ``plate = f"cab_{idx:04d}"``,
  the ACTUAL production convention written into ``driver_index_mapping.pkl``
  ("idx_to_plate") that ``supply_recount._load_driver_mapping``
  (supply_recount.py:199-204) reads for ``--city sf12``
  (``SOURCE_DATA_DIR`` resolved in ``famail_temporal/config.py:25``).
  Reused verbatim here so ``plate_id`` values line up with that mapping.

=== Known divergence (flagged for Task 2's wiring, NOT resolved here) =====

``is_pickup`` / ``is_dropoff`` / ``segment_id`` below are produced by
REUSING Shenzhen's OWN diff-based transition/segment-id bookkeeping
(``famail_temporal.data.source_generation.transitions.add_transition_columns``
/ ``assign_segment_ids``, transitions.py:15-22 / 25-34) applied to the SF
``passenger_indicator`` column -- NOT SF's own ``sf_segmentation.segment_driver``
gap+occupancy segmentation rule (sf_segmentation.py:75-77, 102-112). This is
a deliberate choice, not an oversight:
  - SZ's ``_segment_rows_by_driver`` / ``apply_substitutions`` require the
    segment-boundary CONVENTION where a transition row is the LAST row of
    the segment it closes (transitions.py:1-9 docstring) -- exactly what
    ``add_transition_columns`` / ``assign_segment_ids`` produce, by
    construction, for ANY passenger_indicator-like column. There is no
    SF-native per-ping ``segment_id`` column to "import" instead (SF's own
    pipeline works in terms of whole trajectory LISTS, not a row-level
    segment_id).
  - SZ's transition rule has NO time-gap check: any 0->1 (or 1->0) change in
    ``passenger_indicator`` is a transition, however much wall-clock time
    separates the two pings. SF's OWN ``segment_driver`` instead only counts
    a transition as a real pickup/dropoff EVENT when the gap between the two
    pings is <= ``gap_sec`` (300s, sf_segmentation.py:102-108); a large gap
    is a mere segment break with NO recorded pickup/dropoff. This adapter's
    reused SZ-side logic will flag every occupancy flip as ``is_pickup`` /
    ``is_dropoff`` regardless of gap size -- a real semantic gap between the
    two pipelines that the next task must account for before treating this
    adapter's ``is_pickup`` as equivalent to SF's own pickup/dropoff counts
    (``pickup_dropoff_counts.pkl``).
  - Separately (not this adapter's output, but relevant to whoever wires
    ``recount_tier2`` for SF): SZ's ``build_active_taxis_counts``
    (active_taxis.py:20) filters to ``passenger_indicator == 0`` before
    counting distinct-taxi presence; SF's own
    ``sf_grid_counts.count_active_taxis_5x5`` (sf_grid_counts.py:47-64) does
    NOT filter by occupancy at all -- it counts a taxi present regardless of
    fare status. Reconciling (or deliberately not reconciling) that
    difference is Task 2's decision, not this module's.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from famail_temporal.data.source_generation.transitions import (
    add_transition_columns, assign_segment_ids,
)
from famail_temporal.second_dataset.data.source_generation.sf_config import (
    PDT_OFFSET_SEC, grid_from_points,
)
from famail_temporal.second_dataset.data.source_generation.sf_raw_loader import (
    load_sf_raw,
)

_COLUMNS = [
    "plate_id", "x_grid", "y_grid", "hour", "day_index", "time_bucket",
    "passenger_indicator", "is_pickup", "is_dropoff", "is_transition",
    "segment_id",
]


def load_sf_pings(raw_dir: Path) -> pd.DataFrame:
    """Load SF Cabspotting raw pings from ``raw_dir`` and return a row-level
    DataFrame in EXACTLY the schema ``supply_recount.recount_tier2()`` /
    ``apply_substitutions()`` consume for Shenzhen (see module docstring for
    the full column-by-column derivation + anchors).

    ``raw_dir`` is passed straight through to ``sf_raw_loader.load_sf_raw``
    (glob of ``new_*.txt``); pass the full cabspotting directory to
    reproduce the production 32x30 grid, or a directory with a subset of
    files/rows for a fast, small-scale run (grid dims will be smaller and
    internally consistent, but will not match the production cell indices --
    see the "production grid derivation" note in the module docstring).
    """
    # sf_raw_loader.py:22-55 -- IMPORTED verbatim. Columns:
    # [driver_id, lat, lon, occupancy, time_utc], invalid Bay-Area coords
    # already dropped, sorted (driver_id, time_utc) ascending per driver.
    raw = load_sf_raw(str(raw_dir))
    if len(raw) == 0:
        return pd.DataFrame(columns=_COLUMNS)

    lat = raw["lat"].to_numpy(np.float64)
    lon = raw["lon"].to_numpy(np.float64)
    t = raw["time_utc"].to_numpy().astype(np.int64)

    # sf_config.py:43-52 -- grid_from_points, IMPORTED verbatim (never
    # re-derived); see "Production grid derivation" in the module docstring
    # for what raw_dir must contain to reproduce the production 32x30 grid.
    grid = grid_from_points(lat, lon)

    # sf_segmentation.py:67-70 (== sf_grid_counts.py:54-57): vectorized,
    # 1-indexed, clipped cell quantization -- identical formula to
    # GridSpec.to_cell (sf_config.py:35-40). Replicated verbatim.
    x_grid = np.clip(
        np.floor((lat - grid.lat_min) / grid.cell_deg).astype(int),
        0, grid.x_grid_max - 1,
    ) + 1
    y_grid = np.clip(
        np.floor((lon - grid.lon_min) / grid.cell_deg).astype(int),
        0, grid.y_grid_max - 1,
    ) + 1

    # sf_grid_counts.py:58-60 (== sf_segmentation.py:71-73): local (PDT)
    # time -> 0-indexed hour + absolute epoch-day serial. Replicated
    # verbatim (day_index is NOT a 1-5 weekday enum here -- see "Step 2"
    # docstring note).
    local = t - PDT_OFFSET_SEC
    hour = ((local % 86400) // 3600).astype(int)
    day_index = (local // 86400).astype(int)

    # sf_segmentation.py:72 -- 1-indexed 5-min time_bucket. Replicated
    # verbatim.
    time_bucket = ((local % 86400) // 300 + 1).astype(int)

    # sf_raw_loader.py:3-6 docstring -- occupancy 1=with-fare/driving,
    # 0=free/seeking -- the SAME convention as SZ's passenger_indicator,
    # renamed (not reinterpreted).
    passenger_indicator = raw["occupancy"].to_numpy().astype(int)

    # sf_multistream.py:142 -- production plate-name convention, reused so
    # plate_id matches the real driver_index_mapping.pkl this city's
    # supply_recount._load_driver_mapping reads.
    plate_id = np.array(
        [f"cab_{int(d):04d}" for d in raw["driver_id"].to_numpy()], dtype=object,
    )

    out = pd.DataFrame({
        "plate_id": plate_id,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "hour": hour,
        "day_index": day_index,
        "time_bucket": time_bucket,
        "passenger_indicator": passenger_indicator,
        "_time_utc": t,  # sort key only; dropped before returning
    })
    # event_stream.py:53 -- SZ sorts (plate_id, timestamp) ascending,
    # stable, before transition/segment assignment. load_sf_raw already
    # sorts (driver_id, time_utc) per-driver; re-sort here for the same
    # guarantee across the (possibly multi-file) concatenated result.
    out = (
        out.sort_values(["plate_id", "_time_utc"], kind="stable")
        .drop(columns="_time_utc")
        .reset_index(drop=True)
    )

    # transitions.py:15-22, 25-34 -- REUSED VERBATIM from the SZ pipeline
    # (deliberately NOT SF's own sf_segmentation.segment_driver rule -- see
    # "Known divergence" in the module docstring) so the segment-boundary
    # convention matches what supply_recount._segment_rows_by_driver /
    # apply_substitutions expect.
    out = add_transition_columns(out)
    out = assign_segment_ids(out)

    return out[_COLUMNS]
