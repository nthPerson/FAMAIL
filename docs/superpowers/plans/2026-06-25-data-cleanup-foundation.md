# Data Cleanup Foundation (Stuck-GPS Filter + Cleaned Source Data) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect and remove the 6 stuck-GPS pickup-sink artifacts from the raw event stream, regenerate `source_data/` + the preprocess cache, and persist the sink-audit data — producing a validated cleaned dataset that every later plan builds on.

**Architecture:** A pure detection/filter module (`stuck_gps.py`) operating on the enriched event-stream DataFrame, injected into `build_event_stream` *after* quantization+sort but *before* transition/segment detection (the only place exact lat/lon + driver + cell coexist). A `--dry-run-sinks` CLI mode reports the sink characteristics (concentration distribution + threshold-sensitivity) without a full regen, so thresholds and the drop-vs-suppress behavior are chosen from data, not guessed. The hybrid rule asserts the flagged cells equal the 6 known sinks (loud regression guard).

**Tech Stack:** Python 3.12, pandas, numpy, pytest. Source-gen pipeline under `famail_temporal/data/source_generation/`.

**Spec:** `docs/superpowers/specs/2026-06-25-data-cleanup-rerun-design.md` (this plan implements Stage 0–2 + captures E1, E2, E3, E32, E34).

## Global Constraints

(Copied from the spec; every task implicitly includes these.)

- **TDD** the filter + every new persistence field (schema present + shapes). Project workflow: brainstorm→spec→plan→subagent.
- **Filter rule = Hybrid (data-driven + guard):** concentration detection PLUS `assert flagged_cells == {(28,52),(20,28),(28,28),(24,5),(22,46),(17,38)}` (famail 0-indexed).
- **Thresholds + drop-vs-suppress are finalized from the Stage-0 dry-run, NOT pre-chosen.** Default behavior = **drop** the flagged pickup pings, *pending* the dry-run confirming whether those coords also carry `passenger_indicator==0` empties feeding `active_taxis`.
- **Detect pickups via raw `passenger_indicator` 0→1 per driver** (`is_pickup` does not exist yet at the injection point).
- **`n_drivers` must remain 50** (systemic invariant #6); verify no sink driver is fully removed.
- **Preserve before/after:** do NOT overwrite the existing `source_data/`; the Stage-1 regen writes the cleaned data, but first **back up the dirty `source_data/` to `source_data_dirty/`** (decision §5 — needed for the now-or-never dirty-vs-clean artifacts in later plans).
- **Stage 0 is the algorithm-change-protocol gate** — Task 6 STOPS for user/PI sign-off on the validated thresholds + drop-vs-suppress before the full regen.

---

## File Structure

- **Create** `famail_temporal/data/source_generation/stuck_gps.py` — detection, filter, threshold-sensitivity, audit (pure functions on a DataFrame).
- **Create** `famail_temporal/data/source_generation/tests/test_stuck_gps.py` — unit tests.
- **Modify** `famail_temporal/data/source_generation/event_stream.py` — refactor the load→quantize→sort prefix into `_load_quantized_sorted`; call the filter; add `sink_audit` to `EventStream`.
- **Modify** `famail_temporal/data/source_generation/config.py` — add `STUCK_GPS_*` constants.
- **Modify** `famail_temporal/data/source_generation/cli.py` — thread `sink_audit` into `metadata_extras`; add `--dry-run-sinks`; save the pre-rebuild raw pickup dict (E3) + per-driver before/after pickup vectors (E34).

---

### Task 1: Sink detection (pure function)

**Files:**
- Create: `famail_temporal/data/source_generation/stuck_gps.py`
- Test: `famail_temporal/data/source_generation/tests/test_stuck_gps.py`

**Interfaces:**
- Produces: `pickup_mask(df) -> pd.Series[bool]`; `detect_stuck_gps_sinks(df, *, min_pickups: int, coord_dominance: float, coord_precision: int) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(flagged, distribution)`. `flagged` columns: `plate_id, lat_r, lon_r, x_grid, y_grid, n_pickups, cell_total, cell_share, driver_total`. `distribution` = all `(plate_id, lat_r, lon_r)` pickup-group sizes, sorted desc (for E2 + the threshold curve).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stuck_gps.py
import pandas as pd
from famail_temporal.data.source_generation.stuck_gps import (
    pickup_mask, detect_stuck_gps_sinks,
)

def _df(rows):
    # rows: list of (plate_id, passenger_indicator, latitude, longitude, x_grid, y_grid)
    return pd.DataFrame(rows, columns=[
        "plate_id", "passenger_indicator", "latitude", "longitude", "x_grid", "y_grid",
    ])

def test_pickup_mask_flags_0_to_1_transitions_per_driver():
    df = _df([
        ("A", 0, 1.0, 1.0, 5, 5),
        ("A", 1, 1.0, 1.0, 5, 5),   # pickup (0->1)
        ("A", 0, 2.0, 2.0, 6, 6),   # dropoff (1->0), not a pickup
        ("B", 1, 9.0, 9.0, 1, 1),   # first row of B, diff is NaN -> not a pickup
    ])
    m = pickup_mask(df)
    assert list(m) == [False, True, False, False]

def test_detect_flags_a_concentrated_single_coord_sink():
    rows = []
    # driver SINK: 50 pickups frozen at one exact coord in cell (28,52)
    for _ in range(50):
        rows.append(("SINK", 0, 0.0, 0.0, 28, 52))
        rows.append(("SINK", 1, 12.345678, 98.765432, 28, 52))  # frozen pickup coord
    # one normal pickup elsewhere in a different cell
    rows.append(("NORM", 0, 1.0, 1.0, 5, 5))
    rows.append(("NORM", 1, 1.111111, 2.222222, 5, 5))
    df = _df(rows)
    flagged, dist = detect_stuck_gps_sinks(
        df, min_pickups=10, coord_dominance=0.9, coord_precision=6,
    )
    assert len(flagged) == 1
    row = flagged.iloc[0]
    assert (int(row.x_grid), int(row.y_grid)) == (28, 52)
    assert int(row.n_pickups) == 50
    assert row.cell_share == 1.0
    # distribution is sorted desc and includes both groups
    assert int(dist.iloc[0].n_pickups) == 50
    assert len(dist) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/data/source_generation/tests/test_stuck_gps.py -q`
Expected: FAIL — `ModuleNotFoundError: ... stuck_gps`.

- [ ] **Step 3: Write minimal implementation**

```python
# stuck_gps.py
"""Detect & remove per-driver stuck-GPS pickup 'sinks' (frozen meter-on
coordinates emitting thousands of phantom pickups in a single cell).

Runs on the enriched event-stream df AFTER quantization+sort but BEFORE
transitions are computed, so pickups are detected from the raw
passenger_indicator 0->1 transition per driver (is_pickup does not exist yet).
"""
from __future__ import annotations
import pandas as pd


def pickup_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of pickup events (passenger_indicator 0->1 within driver)."""
    diff = df.groupby("plate_id")["passenger_indicator"].diff()
    return diff == 1


def detect_stuck_gps_sinks(
    df: pd.DataFrame, *, min_pickups: int, coord_dominance: float, coord_precision: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag (driver, exact-coord) pickup groups that are large AND dominate
    their grid cell's pickups. Returns (flagged, distribution)."""
    pk = df[pickup_mask(df)].copy()
    pk["lat_r"] = pk["latitude"].round(coord_precision)
    pk["lon_r"] = pk["longitude"].round(coord_precision)

    grp = pk.groupby(["plate_id", "lat_r", "lon_r"], sort=False)
    sizes = grp.size().rename("n_pickups").reset_index()
    cells = grp[["x_grid", "y_grid"]].first().reset_index()
    sizes = sizes.merge(cells, on=["plate_id", "lat_r", "lon_r"])

    cell_total = pk.groupby(["x_grid", "y_grid"]).size().rename("cell_total").reset_index()
    driver_total = pk.groupby("plate_id").size().rename("driver_total").reset_index()
    sizes = sizes.merge(cell_total, on=["x_grid", "y_grid"]).merge(driver_total, on="plate_id")
    sizes["cell_share"] = sizes["n_pickups"] / sizes["cell_total"]

    distribution = sizes.sort_values("n_pickups", ascending=False).reset_index(drop=True)
    flagged = distribution[
        (distribution["n_pickups"] >= min_pickups)
        & (distribution["cell_share"] >= coord_dominance)
    ].reset_index(drop=True)
    return flagged, distribution
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/data/source_generation/tests/test_stuck_gps.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/data/source_generation/stuck_gps.py famail_temporal/data/source_generation/tests/test_stuck_gps.py
git commit -m "feat(source-gen): stuck-GPS sink detection on the event stream"
```

---

### Task 2: Filter (drop + hybrid assertion guard)

**Files:**
- Modify: `famail_temporal/data/source_generation/stuck_gps.py`
- Test: `famail_temporal/data/source_generation/tests/test_stuck_gps.py`

**Interfaces:**
- Produces: `filter_stuck_gps_sinks(df, *, min_pickups, coord_dominance, coord_precision, expected_cells: set[tuple[int,int]] | None, drop: bool = True) -> tuple[pd.DataFrame, dict]` returning `(cleaned_df, audit)`. `audit` keys: `sinks` (list of per-sink dicts: `plate_id, lat, lon, x_grid, y_grid, n_pickups, cell_share, driver_total`), `n_pickups_removed`, `n_rows_removed`, `flagged_cells`. Raises `AssertionError` if `expected_cells` is given and the flagged cell set differs.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_stuck_gps.py
import pytest
from famail_temporal.data.source_generation.stuck_gps import filter_stuck_gps_sinks

def _sink_df():
    rows = []
    for _ in range(50):
        rows.append(("SINK", 0, 0.0, 0.0, 28, 52))
        rows.append(("SINK", 1, 12.345678, 98.765432, 28, 52))
    rows.append(("NORM", 0, 1.0, 1.0, 5, 5))
    rows.append(("NORM", 1, 1.111111, 2.222222, 5, 5))
    return _df(rows)

def test_filter_drops_flagged_pickups_and_keeps_drivers():
    df = _sink_df()
    cleaned, audit = filter_stuck_gps_sinks(
        df, min_pickups=10, coord_dominance=0.9, coord_precision=6, expected_cells=None,
    )
    # the 50 frozen pickup rows are gone; the SINK driver still exists (its 50 indicator-0 rows remain)
    assert cleaned["plate_id"].nunique() == 2
    assert audit["n_rows_removed"] == 50
    assert audit["flagged_cells"] == [(28, 52)]
    assert audit["sinks"][0]["n_pickups"] == 50
    # normal pickup survives
    assert ((cleaned["plate_id"] == "NORM") & (cleaned["passenger_indicator"] == 1)).sum() == 1

def test_hybrid_guard_asserts_expected_cells():
    df = _sink_df()
    # correct expectation passes
    filter_stuck_gps_sinks(df, min_pickups=10, coord_dominance=0.9, coord_precision=6,
                           expected_cells={(28, 52)})
    # wrong expectation raises
    with pytest.raises(AssertionError):
        filter_stuck_gps_sinks(df, min_pickups=10, coord_dominance=0.9, coord_precision=6,
                               expected_cells={(1, 1)})

def test_filter_is_noop_on_clean_data():
    df = _df([
        ("A", 0, 1.0, 1.0, 5, 5), ("A", 1, 1.1, 2.1, 5, 5),
        ("B", 0, 3.0, 3.0, 7, 7), ("B", 1, 3.1, 4.1, 7, 7),
    ])
    cleaned, audit = filter_stuck_gps_sinks(
        df, min_pickups=10, coord_dominance=0.9, coord_precision=6, expected_cells=set(),
    )
    assert len(cleaned) == len(df)
    assert audit["n_rows_removed"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/data/source_generation/tests/test_stuck_gps.py -q`
Expected: FAIL — `ImportError: cannot import name 'filter_stuck_gps_sinks'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to stuck_gps.py
def filter_stuck_gps_sinks(
    df: pd.DataFrame, *, min_pickups: int, coord_dominance: float,
    coord_precision: int, expected_cells: set | None, drop: bool = True,
) -> tuple[pd.DataFrame, dict]:
    flagged, _dist = detect_stuck_gps_sinks(
        df, min_pickups=min_pickups, coord_dominance=coord_dominance,
        coord_precision=coord_precision,
    )
    flagged_cells = sorted({(int(r.x_grid), int(r.y_grid)) for r in flagged.itertuples()})
    if expected_cells is not None:
        assert set(flagged_cells) == set(expected_cells), (
            f"stuck-GPS filter flagged {set(flagged_cells)} != expected {set(expected_cells)}"
        )

    audit = {
        "sinks": [
            {"plate_id": r.plate_id, "lat": float(r.lat_r), "lon": float(r.lon_r),
             "x_grid": int(r.x_grid), "y_grid": int(r.y_grid),
             "n_pickups": int(r.n_pickups), "cell_share": float(r.cell_share),
             "driver_total": int(r.driver_total)}
            for r in flagged.itertuples()
        ],
        "flagged_cells": flagged_cells,
        "n_pickups_removed": int(flagged["n_pickups"].sum()) if len(flagged) else 0,
    }

    if not drop or len(flagged) == 0:
        audit["n_rows_removed"] = 0
        return df.reset_index(drop=True), audit

    pk = pickup_mask(df)
    lat_r = df["latitude"].round(coord_precision)
    lon_r = df["longitude"].round(coord_precision)
    key = list(zip(df["plate_id"], lat_r, lon_r))
    flagged_keys = {(r.plate_id, float(r.lat_r), float(r.lon_r)) for r in flagged.itertuples()}
    is_flagged_pickup = pk.values & pd.Series(
        [k in flagged_keys for k in key], index=df.index
    ).values
    cleaned = df[~is_flagged_pickup].reset_index(drop=True)
    audit["n_rows_removed"] = int(is_flagged_pickup.sum())
    return cleaned, audit
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/data/source_generation/tests/test_stuck_gps.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/data/source_generation/stuck_gps.py famail_temporal/data/source_generation/tests/test_stuck_gps.py
git commit -m "feat(source-gen): stuck-GPS filter with drop + hybrid cell-set guard"
```

---

### Task 3: Threshold-sensitivity curve (E32)

**Files:**
- Modify: `famail_temporal/data/source_generation/stuck_gps.py`
- Test: `famail_temporal/data/source_generation/tests/test_stuck_gps.py`

**Interfaces:**
- Produces: `threshold_sensitivity(df, thresholds: list[int], *, coord_dominance, coord_precision) -> list[dict]` → `[{"min_pickups": t, "n_flagged_cells": k}, ...]`. Backs the "the 6-sink set is stable across a wide threshold band" figure.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_stuck_gps.py
from famail_temporal.data.source_generation.stuck_gps import threshold_sensitivity

def test_threshold_sensitivity_plateaus_then_drops():
    df = _sink_df()  # one 50-pickup sink + one 1-pickup normal cell
    curve = threshold_sensitivity(df, thresholds=[1, 10, 60], coord_dominance=0.9, coord_precision=6)
    by_t = {c["min_pickups"]: c["n_flagged_cells"] for c in curve}
    assert by_t[10] == 1     # only the sink
    assert by_t[60] == 0     # threshold above the sink size -> nothing
    assert by_t[1] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/data/source_generation/tests/test_stuck_gps.py::test_threshold_sensitivity_plateaus_then_drops -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to stuck_gps.py
def threshold_sensitivity(
    df: pd.DataFrame, thresholds: list[int], *, coord_dominance: float, coord_precision: int,
) -> list[dict]:
    _, dist = detect_stuck_gps_sinks(
        df, min_pickups=1, coord_dominance=coord_dominance, coord_precision=coord_precision,
    )
    out = []
    for t in thresholds:
        hit = dist[(dist["n_pickups"] >= t) & (dist["cell_share"] >= coord_dominance)]
        n_cells = hit.drop_duplicates(["x_grid", "y_grid"]).shape[0]
        out.append({"min_pickups": int(t), "n_flagged_cells": int(n_cells)})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/data/source_generation/tests/test_stuck_gps.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/data/source_generation/stuck_gps.py famail_temporal/data/source_generation/tests/test_stuck_gps.py
git commit -m "feat(source-gen): stuck-GPS threshold-sensitivity curve (E32)"
```

---

### Task 4: Config constants + wire the filter into `build_event_stream`

**Files:**
- Modify: `famail_temporal/data/source_generation/config.py`
- Modify: `famail_temporal/data/source_generation/event_stream.py`
- Test: `famail_temporal/data/source_generation/tests/test_stuck_gps.py`

**Interfaces:**
- Consumes: `filter_stuck_gps_sinks` (Task 2).
- Produces: `EventStream.sink_audit: dict`. `build_event_stream(raw_dir, *, apply_sink_filter: bool = True)`. New `config.STUCK_GPS_*` constants. New helper `event_stream._load_quantized_sorted(raw_dir) -> tuple[pd.DataFrame, GlobalBounds]` returning the sorted-but-pre-transition df + bounds — used by both the builder and the dry-run. (`n_days`/`driver_calendar_days` stay computed in `build_event_stream` after filtering.)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_stuck_gps.py
from famail_temporal.data.source_generation import config as sgconfig

def test_config_has_stuck_gps_constants():
    assert isinstance(sgconfig.STUCK_GPS_EXPECTED_CELLS, (set, frozenset))
    assert (28, 52) in sgconfig.STUCK_GPS_EXPECTED_CELLS
    assert sgconfig.STUCK_GPS_COORD_PRECISION == 6
    assert sgconfig.STUCK_GPS_DROP is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/data/source_generation/tests/test_stuck_gps.py::test_config_has_stuck_gps_constants -q`
Expected: FAIL — `AttributeError: ... STUCK_GPS_EXPECTED_CELLS`.

- [ ] **Step 3: Write minimal implementation**

Add to `config.py` (PROVISIONAL thresholds — finalized in Task 6 from the dry-run):

```python
# --- stuck-GPS sink filter (provisional; finalize from Stage-0 dry-run) ---
STUCK_GPS_MIN_PICKUPS = 1000        # absolute phantom-pickup floor
STUCK_GPS_COORD_DOMINANCE = 0.99    # one exact coord's share of its cell's pickups
STUCK_GPS_COORD_PRECISION = 6       # lat/lon rounding (decimals)
STUCK_GPS_DROP = True               # drop flagged pickup pings (vs suppress)
STUCK_GPS_EXPECTED_CELLS = {(28, 52), (20, 28), (28, 28), (24, 5), (22, 46), (17, 38)}
```

Refactor `event_stream.py`: extract lines 30–46 into `_load_quantized_sorted(raw_dir)` returning the sorted df + bounds; then in `build_event_stream`, after obtaining the sorted df and BEFORE `add_transition_columns`, insert:

```python
from famail_temporal.data.source_generation.stuck_gps import filter_stuck_gps_sinks

# (inside build_event_stream, after df = _load_quantized_sorted(...))
sink_audit: dict = {}
if apply_sink_filter:
    df, sink_audit = filter_stuck_gps_sinks(
        df,
        min_pickups=config.STUCK_GPS_MIN_PICKUPS,
        coord_dominance=config.STUCK_GPS_COORD_DOMINANCE,
        coord_precision=config.STUCK_GPS_COORD_PRECISION,
        expected_cells=config.STUCK_GPS_EXPECTED_CELLS,
        drop=config.STUCK_GPS_DROP,
    )
df = add_transition_columns(df)
df = assign_segment_ids(df)
...
return EventStream(df=df, bounds=bounds, n_days=n_days,
                   driver_calendar_days=driver_calendar_days, sink_audit=sink_audit)
```

Add `sink_audit: dict` (default `dict` via `field(default_factory=dict)`) to the `EventStream` dataclass, and `apply_sink_filter: bool = True` to `build_event_stream`'s signature.

- [ ] **Step 4: Run the test + the existing source-gen suite to verify pass + no regression**

Run: `python -m pytest famail_temporal/data/source_generation/tests/ -q`
Expected: PASS (new constant test + all existing source-gen tests green). If any existing test calls `build_event_stream` on synthetic data and now trips the assertion, pass `apply_sink_filter=False` in that test or `expected_cells` via a clean fixture.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/data/source_generation/config.py famail_temporal/data/source_generation/event_stream.py famail_temporal/data/source_generation/tests/test_stuck_gps.py
git commit -m "feat(source-gen): wire stuck-GPS filter into build_event_stream + config knobs"
```

---

### Task 5: Persist the audit + dry-run CLI + raw-pickup/per-driver captures (E1/E2/E3/E34) and the `--dry-run-sinks` mode

**Files:**
- Modify: `famail_temporal/data/source_generation/cli.py`
- Test: `famail_temporal/data/source_generation/tests/test_stuck_gps.py`

**Interfaces:**
- Consumes: `EventStream.sink_audit`, `detect_stuck_gps_sinks`, `threshold_sensitivity`.
- Produces: `run_generation` writes `metadata_extras["stuck_gps_sinks"] = es.sink_audit` and `metadata_extras["raw_pickup_counts_pre_rebuild"]` (E3) + `metadata_extras["per_driver_pickups"]` (E34). New `report_stuck_gps(input_dir) -> dict` (the dry-run) and a `--dry-run-sinks` flag that writes `source_data/stuck_gps_report.json` (audit + distribution top-K + threshold curve) and exits before views.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_stuck_gps.py
from famail_temporal.data.source_generation import cli as sgcli

def test_report_stuck_gps_returns_audit_and_curve(monkeypatch):
    df = _sink_df()
    # stub the heavy load step so the dry-run runs on synthetic data
    monkeypatch.setattr(sgcli, "_load_event_df_for_report", lambda _in: df)
    rep = sgcli.report_stuck_gps("ignored", expected_cells=None)
    assert rep["audit"]["flagged_cells"] == [(28, 52)]
    assert rep["distribution_top"][0]["n_pickups"] == 50
    assert any(c["min_pickups"] for c in rep["threshold_curve"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/data/source_generation/tests/test_stuck_gps.py::test_report_stuck_gps_returns_audit_and_curve -q`
Expected: FAIL — `AttributeError: ... report_stuck_gps`.

- [ ] **Step 3: Write minimal implementation**

In `cli.py`, add the dry-run helpers (reuse `_load_quantized_sorted` from Task 4 via a thin seam so it's monkeypatchable):

```python
from famail_temporal.data.source_generation.event_stream import _load_quantized_sorted
from famail_temporal.data.source_generation import stuck_gps, config

def _load_event_df_for_report(input_dir):
    df, _bounds = _load_quantized_sorted(Path(input_dir))
    return df

def report_stuck_gps(input_dir, *, expected_cells=config.STUCK_GPS_EXPECTED_CELLS,
                     top_k: int = 50) -> dict:
    df = _load_event_df_for_report(input_dir)
    _cleaned, audit = stuck_gps.filter_stuck_gps_sinks(
        df, min_pickups=config.STUCK_GPS_MIN_PICKUPS,
        coord_dominance=config.STUCK_GPS_COORD_DOMINANCE,
        coord_precision=config.STUCK_GPS_COORD_PRECISION,
        expected_cells=expected_cells, drop=False,   # report-only
    )
    _flagged, dist = stuck_gps.detect_stuck_gps_sinks(
        df, min_pickups=1, coord_dominance=config.STUCK_GPS_COORD_DOMINANCE,
        coord_precision=config.STUCK_GPS_COORD_PRECISION,
    )
    curve = stuck_gps.threshold_sensitivity(
        df, thresholds=[100, 250, 500, 1000, 2000, 5000, 10000],
        coord_dominance=config.STUCK_GPS_COORD_DOMINANCE,
        coord_precision=config.STUCK_GPS_COORD_PRECISION,
    )
    return {
        "audit": audit,
        "distribution_top": dist.head(top_k).to_dict(orient="records"),
        "threshold_curve": curve,
    }
```

Add to `_build_parser`: `p.add_argument("--dry-run-sinks", action="store_true", help="Report stuck-GPS sinks (audit + concentration distribution + threshold curve) and exit.")`. In `main`, before `run_generation`, branch:

```python
if args.dry_run_sinks:
    import json
    rep = report_stuck_gps(args.input_dir, expected_cells=None)  # report-only: don't assert
    out = Path(args.output_dir) / "stuck_gps_report.json"
    out.write_text(json.dumps(rep, indent=2, default=float))
    log.info("Wrote stuck-GPS dry-run report to %s (flagged cells: %s)",
             out, rep["audit"]["flagged_cells"])
    return 0
```

In `run_generation`, add to `metadata_extras`:

```python
"stuck_gps_sinks": es.sink_audit,
"raw_pickup_counts_pre_rebuild": {str(k): v[0] for k, v in pickup_dropoff_raw.items()},
"per_driver_pickups": (
    es.df[stuck_gps.pickup_mask(es.df)].groupby("plate_id").size().to_dict()
),
```

(Note `pickup_dropoff_raw` is the pre-rebuild dict already computed at `cli.py:73` — E3.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/data/source_generation/tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/data/source_generation/cli.py famail_temporal/data/source_generation/tests/test_stuck_gps.py
git commit -m "feat(source-gen): --dry-run-sinks report + persist sink audit / raw-pickup / per-driver (E1/E2/E3/E34)"
```

---

### Task 6: Stage-0 dry-run on REAL data → validate gate → finalize thresholds (ALGORITHM-CHANGE-PROTOCOL CHECKPOINT)

**Files:** none (operational). Possibly `config.py` if thresholds change.

- [ ] **Step 1: Run the dry-run on the real raw data**

Run: `python -m famail_temporal.data.source_generation --dry-run-sinks -v`
Expected: writes `famail_temporal/source_data/stuck_gps_report.json`; logs the flagged cells.

- [ ] **Step 2: Validate the gate (STOP if not satisfied)**

Inspect `stuck_gps_report.json`:
- `audit.flagged_cells` (with the provisional thresholds) **== the 6 known cells** `{(28,52),(20,28),(28,28),(24,5),(22,46),(17,38)}`. If not, adjust `STUCK_GPS_MIN_PICKUPS`/`STUCK_GPS_COORD_DOMINANCE` using `threshold_curve` (look for the flat plateau at 6) and re-run.
- Confirm from `distribution_top` the sink groups are ~10–12k each and far above the next real coord-group (justifies the cutoff band).
- Check whether the frozen coords also carry `passenger_indicator==0` empties (inspect the raw rows at those coords) → decides **drop vs suppress**. Default = drop; switch `STUCK_GPS_DROP`/behavior only if empties materially feed `active_taxis`.

- [ ] **Step 3: STOP for user/PI sign-off**

Per the algorithm-change protocol, present the validated thresholds + the flagged-cell confirmation + the drop-vs-suppress finding, and get explicit approval before the full regen. Do NOT proceed to Task 7 without it.

---

### Task 7: Regenerate cleaned `source_data/` + preprocess cache (Stage 1–2)

**Files:** none (operational; reads/writes data artifacts).

- [ ] **Step 1: Back up the dirty source data (preserve before/after)**

Run: `cp -r famail_temporal/source_data famail_temporal/source_data_dirty`
Expected: `source_data_dirty/` holds the pre-cleanup artifacts (needed for the now-or-never dirty-vs-clean captures in later plans).

- [ ] **Step 2: Regenerate cleaned source data**

Run: `python -m famail_temporal.data.source_generation -v`
Expected: rebuilds `source_data/*.pkl` + `processing_metadata.json` with the `stuck_gps_sinks` audit block; the hybrid assertion passes (build aborts loudly if the flagged set ever drifts from the 6 cells).

- [ ] **Step 3: Verify Stage-1 gate**

Run: `python -c "import json,pathlib; m=json.loads(pathlib.Path('famail_temporal/source_data/processing_metadata.json').read_text()); print('sinks', m.get('stuck_gps_sinks',{}).get('flagged_cells')); print('n_removed', m.get('stuck_gps_sinks',{}).get('n_pickups_removed'))"`
Expected: 6 flagged cells; a large `n_pickups_removed`. Then confirm `n_drivers == 50` (the systemic-invariant check inside the run did not raise) and eval drivers retain ≥6 trajectories.

- [ ] **Step 4: Preprocess to cache tensors**

Run: `python -m famail_temporal.preprocess --force`
Expected: regenerates `cache/*_T24_thr0.5*.pkl`; logs g0 fit + active-unit counts.

- [ ] **Step 5: Verify Stage-2 gate + commit the cleaned-data provenance**

Confirm cache files' mtimes updated. Commit any threshold changes + the recorded metadata note (the large `source_data/*.pkl` are gitignored; commit only code/config + a short note if added):

```bash
git add famail_temporal/data/source_generation/config.py
git commit -m "chore(source-gen): finalize stuck-GPS thresholds from Stage-0 dry-run + regenerate cleaned source data"
```

---

## Self-Review

**Spec coverage (this plan = Stage 0–2 + E1/E2/E3/E32/E34):**
- Stuck-GPS filter (hybrid rule + guard) → Tasks 1–4 ✓
- Drop-vs-suppress + thresholds finalized from dry-run → Tasks 5–6 ✓ (provisional config + `--dry-run-sinks` + checkpoint)
- Sink audit E1 / concentration distribution E2 / threshold curve E32 → Tasks 2,3,5 ✓
- Raw-pickup-dict E3 / per-driver vectors E34 → Task 5 ✓
- n_drivers==50 guard → Task 7 Step 3 ✓ (enforced by existing `check_systemic_invariants`)
- Preserve before/after (`source_data_dirty/`) → Task 7 Step 1 ✓
- Algorithm-change-protocol gate → Task 6 Step 3 ✓

**Deferred to later plans (out of scope here):** provenance bundle E19–E22/E29/E30/E37–E39 (Plan 2); editor enrichments E6–E8/E35 + editor run (Plan 3); experiment-runner enrichments E9–E18/E24–E28/E36 + runs (Plan 4); analysis/figure scripts E22/E23/E31/E33/E40/E16/E17 + execution runbook (Plan 5). Several downstream captures (E22/E23) depend on `source_data_dirty/` created in Task 7 Step 1.

**Placeholder scan:** none — all code/commands concrete. **Type consistency:** `filter_stuck_gps_sinks`/`detect_stuck_gps_sinks`/`threshold_sensitivity`/`pickup_mask`/`report_stuck_gps` signatures consistent across tasks; `EventStream.sink_audit` produced in Task 4, consumed in Task 5.

> **Note:** the provisional thresholds in Task 4 are deliberately not final — Task 6 validates them against the real concentration distribution before any irreversible regen, per the don't-guess principle.
