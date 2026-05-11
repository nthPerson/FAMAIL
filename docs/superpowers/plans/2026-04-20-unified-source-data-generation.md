# Unified GPS Source-Data Generation Tool — Implementation Plan

> **Serialization note:** Several inputs and outputs use Python `.pkl` files. This is required for drop-in compatibility with `famail_temporal/preprocess.py` and `famail_temporal/data/loader.py`, which consume these files today. The tool only reads and writes `.pkl` files produced by this project's own trusted tooling (raw taxi GPS data and the tool's own outputs); it never loads `.pkl` from external or untrusted sources. See §1 of the design spec for full rationale.
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Subagent model policy (hard requirement):** Every implementation subagent and every code-review subagent dispatched by this plan MUST be run on the Opus model. The orchestrator must pass `model: "opus"` when launching subagents via the Agent tool.
>
> **Subagent skill policy (hard requirement):** Every subagent must be instructed in its prompt to invoke the following superpowers skills:
>
> - **Implementation subagents:** `superpowers:test-driven-development` (write the failing test first; red → green → refactor), `superpowers:verification-before-completion` (run the tests and show passing output before claiming done), and `superpowers:systematic-debugging` (when a test fails, diagnose the root cause instead of patching the symptom).
> - **Review subagents:** `superpowers:requesting-code-review` → `superpowers:code-reviewer` at every phase checkpoint; `superpowers:receiving-code-review` for the follow-up pass that acts on feedback.
> - **Cleanup passes:** `code-simplifier:code-simplifier` and/or `superpowers:simplify` after each phase checkpoint, *before* the code review, to strip dead code and tighten obvious verbosity.
>
> Every subagent's prompt must explicitly name the skills it is required to use.

**Goal:** Build a unified GPS source-data generation tool under `famail_temporal/data/source_generation/` that produces all 8 GPS-derived files (plus 2 sidecars) consumed by `famail_temporal`, replacing the legacy `pickup_dropoff_counts/`, `active_taxis/`, and `new_all_trajs/` tools with a single pipeline whose cross-file consistency holds by construction.

**Architecture:** Six sequential phases. Phase 1 scaffolds the package, loads raw GPS, and implements quantization primitives. Phase 2 adds transition detection and the enriched event stream (the single source of truth). Phase 3 adds the five view modules that derive each output file. Phase 4 adds the invariant suite, output writer, and CLI. Phase 5 is golden-dataset testing plus the full end-to-end smoke test. Phase 6 integrates with `famail_temporal` (filename update, changelog). Every phase ends at a commit with all tests green and a code-review checkpoint.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest. No new third-party dependencies.

**Spec reference:** [`docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md`](../specs/2026-04-20-unified-source-data-generation-design.md)

---

## File Structure

### Files to create

| Path | Responsibility |
|---|---|
| `famail_temporal/data/source_generation/__init__.py` | Package init + re-exports |
| `famail_temporal/data/source_generation/config.py` | Constants: GPS grid size, time interval, neighborhood, weekday set, removal threshold |
| `famail_temporal/data/source_generation/raw_loader.py` | Load + concat the 3 raw taxi pickle files → pandas DataFrame |
| `famail_temporal/data/source_generation/quantization.py` | `gps_to_grid`, `seconds_to_time_bucket`, `seconds_to_hour`, `timestamp_to_day` |
| `famail_temporal/data/source_generation/transitions.py` | `add_transition_columns`, `assign_segment_ids` |
| `famail_temporal/data/source_generation/event_stream.py` | `build_event_stream` — enriched DataFrame + metadata |
| `famail_temporal/data/source_generation/removal.py` | `RemovalRecord`, `RemovalSummary` dataclasses |
| `famail_temporal/data/source_generation/views/__init__.py` | View-package init |
| `famail_temporal/data/source_generation/views/pickup_dropoff.py` | `build_pickup_dropoff_counts` |
| `famail_temporal/data/source_generation/views/active_taxis.py` | `build_active_taxis_counts` (5×5 neighborhood, empty-only) |
| `famail_temporal/data/source_generation/views/trajectories.py` | `build_trajectories`, `build_driver_index_mapping` |
| `famail_temporal/data/source_generation/views/profile.py` | `compute_profile_features`, `zscore_normalize`, `compute_home_xy_with_fallback` |
| `famail_temporal/data/source_generation/views/calendars.py` | `build_calendar_days_per_driver` |
| `famail_temporal/data/source_generation/invariants.py` | `apply_per_trajectory_invariants`, `check_systemic_invariants` |
| `famail_temporal/data/source_generation/writer.py` | `write_all_outputs`, `write_active_taxis_bundle`, `write_metadata_json` |
| `famail_temporal/data/source_generation/cli.py` | `run_generation`, `main` orchestration |
| `famail_temporal/data/source_generation/__main__.py` | Enables `python -m famail_temporal.data.source_generation` |
| `famail_temporal/data/source_generation/tests/__init__.py` | Test-package init |
| `famail_temporal/data/source_generation/tests/test_raw_loader.py` | Unit tests for raw loader |
| `famail_temporal/data/source_generation/tests/test_quantization.py` | Unit tests for quantization |
| `famail_temporal/data/source_generation/tests/test_transitions.py` | Unit tests for transitions |
| `famail_temporal/data/source_generation/tests/test_event_stream.py` | Unit tests for event stream |
| `famail_temporal/data/source_generation/tests/test_view_pickup_dropoff.py` | Unit tests for pickup_dropoff view |
| `famail_temporal/data/source_generation/tests/test_view_active_taxis.py` | Unit tests for active_taxis view |
| `famail_temporal/data/source_generation/tests/test_view_trajectories.py` | Unit tests for trajectories view |
| `famail_temporal/data/source_generation/tests/test_view_profile.py` | Unit tests for profile view |
| `famail_temporal/data/source_generation/tests/test_profile_fallbacks.py` | `home_x/y` fallback cascade tests |
| `famail_temporal/data/source_generation/tests/test_view_calendars.py` | Unit tests for calendars view |
| `famail_temporal/data/source_generation/tests/test_invariants.py` | Invariant suite tests |
| `famail_temporal/data/source_generation/tests/test_writer.py` | Writer + metadata tests |
| `famail_temporal/data/source_generation/tests/test_cli.py` | CLI end-to-end test |
| `famail_temporal/data/source_generation/tests/golden_fixtures.py` | Hand-built synthetic raw fixture + expected outputs |
| `famail_temporal/data/source_generation/tests/test_golden.py` | Golden test + slow real-data smoke test |

### Files to modify

| Path | Change |
|---|---|
| `famail_temporal/data/loader.py:95` | Rename: `passenger_seeking_trajs_45-800.pkl` → `passenger_seeking_trajs.pkl` |
| `CHANGELOG.md` | Append semantic-change entry + required operator action |

---

## Phase 1 — Scaffold + raw loading + quantization

### Task 1: Scaffold the `source_generation` sub-package

**Files:**
- Create: `famail_temporal/data/source_generation/__init__.py`
- Create: `famail_temporal/data/source_generation/config.py`
- Create: `famail_temporal/data/source_generation/tests/__init__.py`
- Create: `famail_temporal/data/source_generation/views/__init__.py`

- [ ] **Step 1.1: Create empty `__init__.py` files**

```bash
mkdir -p famail_temporal/data/source_generation/views
mkdir -p famail_temporal/data/source_generation/tests
touch famail_temporal/data/source_generation/__init__.py
touch famail_temporal/data/source_generation/views/__init__.py
touch famail_temporal/data/source_generation/tests/__init__.py
```

- [ ] **Step 1.2: Write `config.py`**

Create `famail_temporal/data/source_generation/config.py`:

```python
"""Configuration constants for the unified GPS source-data generation tool.

All constants are opinionated defaults — there are no runtime config flags.
If a constant needs to change, edit this file rather than adding a CLI option.
"""
from __future__ import annotations
from pathlib import Path

# Spatial grid (matches famail_temporal.config)
GRID_SIZE_DEG: float = 0.01
X_GRID_MAX: int = 48
Y_GRID_MAX: int = 90
X_GRID_OFFSET: int = 1
Y_GRID_OFFSET: int = 1

# Time quantization
TIME_INTERVAL_MIN: int = 5
TIME_BUCKET_MAX: int = 288
HOUR_MAX: int = 23

# Day filter — weekdays only (permanent project decision)
WEEKDAY_DAYS: frozenset[int] = frozenset({1, 2, 3, 4, 5})

# Active-taxis neighborhood
NEIGHBORHOOD_SIZE: int = 5
NEIGHBORHOOD_K: int = NEIGHBORHOOD_SIZE // 2

# Profile features
PROFILE_FEATURE_NAMES: tuple[str, ...] = (
    "home_x", "home_y",
    "shift_start", "shift_end",
    "freq_grid_x", "freq_grid_y",
    "avg_seeking_dist", "avg_seeking_time",
    "avg_driving_dist", "avg_driving_time",
    "num_trips_per_day",
)
N_PROFILE_FEATURES: int = len(PROFILE_FEATURE_NAMES)
PROFILE_SHIFT_LOW_PCT: float = 5.0
PROFILE_SHIFT_HIGH_PCT: float = 95.0

# Per-trajectory removal warning threshold
REMOVAL_RATE_WARN_THRESHOLD: float = 0.05

# Required driver count
EXPECTED_N_DRIVERS: int = 50

# I/O defaults
DEFAULT_RAW_INPUT_DIR: Path = Path("raw_data")
DEFAULT_OUTPUT_DIR: Path = Path("famail_temporal/raw_data")

RAW_INPUT_FILENAMES: tuple[str, ...] = (
    "taxi_record_07_50drivers.pkl",
    "taxi_record_08_50drivers.pkl",
    "taxi_record_09_50drivers.pkl",
)

OUT_PICKUP_DROPOFF: str = "pickup_dropoff_counts.pkl"
OUT_ACTIVE_TAXIS: str = "active_taxis_5x5_hourly.pkl"
OUT_PASSENGER_SEEKING: str = "passenger_seeking_trajs.pkl"
OUT_MS_SEEKING: str = "ms_seeking_trajs.pkl"
OUT_MS_DRIVING: str = "ms_driving_trajs.pkl"
OUT_MS_PROFILE: str = "ms_profile_features.pkl"
OUT_MS_SEEKING_DAYS: str = "ms_seeking_calendar_days.pkl"
OUT_MS_DRIVING_DAYS: str = "ms_driving_calendar_days.pkl"
OUT_DRIVER_MAPPING: str = "driver_index_mapping.pkl"
OUT_METADATA: str = "processing_metadata.json"

RANDOM_SEED: int = 0
OUTPUT_FORMAT_VERSION: str = "1.0.0"
```

- [ ] **Step 1.3: Populate `__init__.py` with a docstring**

Create `famail_temporal/data/source_generation/__init__.py`:

```python
"""Unified GPS source-data generation tool.

Single-entry-point tool that reads the 3 raw taxi GPS pickle files and
produces all 8 source datasets consumed by `famail_temporal`. Cross-file
consistency is enforced by construction: everything derives from one
enriched event stream produced in one pass.

See docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md
for the full design rationale.
"""
```

- [ ] **Step 1.4: Sanity check — import the package**

```bash
.venv/bin/python -c "from famail_temporal.data.source_generation import config; print(config.N_PROFILE_FEATURES, config.WEEKDAY_DAYS)"
```

Expected output: `11 frozenset({1, 2, 3, 4, 5})`

- [ ] **Step 1.5: Commit**

```bash
git add famail_temporal/data/source_generation/
git commit -m "feat(data): scaffold source_generation sub-package"
```

---

### Task 2: Raw loader

**Files:**
- Create: `famail_temporal/data/source_generation/raw_loader.py`
- Create: `famail_temporal/data/source_generation/tests/test_raw_loader.py`

- [ ] **Step 2.1: Write the failing test**

Create `famail_temporal/data/source_generation/tests/test_raw_loader.py`:

```python
"""Tests for raw_loader.py."""
from __future__ import annotations
import pickle
from pathlib import Path

import pandas as pd
import pytest

from famail_temporal.data.source_generation.raw_loader import (
    load_raw_file, concat_raw_records,
)


def _write_pkl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def test_load_raw_file_flat_structure(tmp_path):
    path = tmp_path / "flat.pkl"
    _write_pkl(path, {
        "PLATE_A": [
            ["PLATE_A", 22.5, 114.0, 0, 0, "2016-07-01 00:00:00"],
            ["PLATE_A", 22.5, 114.0, 60, 1, "2016-07-01 00:01:00"],
        ],
    })
    df = load_raw_file(path)
    assert len(df) == 2
    assert list(df.columns) == [
        "plate_id", "latitude", "longitude", "seconds",
        "passenger_indicator", "timestamp",
    ]
    assert df.iloc[0]["plate_id"] == "PLATE_A"
    assert df.iloc[0]["passenger_indicator"] == 0
    assert df.iloc[1]["passenger_indicator"] == 1


def test_load_raw_file_nested_day_lists(tmp_path):
    path = tmp_path / "nested.pkl"
    _write_pkl(path, {
        "PLATE_B": [
            [["PLATE_B", 22.5, 114.0, 0, 0, "2016-07-01 00:00:00"]],
            [["PLATE_B", 22.5, 114.0, 60, 1, "2016-07-02 00:01:00"]],
        ],
    })
    df = load_raw_file(path)
    assert len(df) == 2


def test_load_raw_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_file(tmp_path / "does_not_exist.pkl")


def test_load_raw_file_rejects_bad_structure(tmp_path):
    path = tmp_path / "bad.pkl"
    _write_pkl(path, ["not a dict"])
    with pytest.raises(ValueError, match="expected dict"):
        load_raw_file(path)


def test_concat_raw_records(tmp_path):
    p1 = tmp_path / "a.pkl"
    p2 = tmp_path / "b.pkl"
    _write_pkl(p1, {"A": [["A", 22.5, 114.0, 0, 0, "2016-07-01 00:00:00"]]})
    _write_pkl(p2, {"B": [["B", 22.5, 114.0, 0, 0, "2016-08-01 00:00:00"]]})

    df = concat_raw_records([p1, p2])
    assert len(df) == 2
    assert set(df["plate_id"]) == {"A", "B"}
```

- [ ] **Step 2.2: Run the test — expect failure**

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_raw_loader.py -v
```

Expected: ImportError on `raw_loader` (module doesn't exist yet).

- [ ] **Step 2.3: Implement `raw_loader.py`**

Create `famail_temporal/data/source_generation/raw_loader.py`:

```python
"""Load the raw taxi GPS files into a concatenated DataFrame.

Only loads project-internal, trusted files produced by the taxi-GPS data
pipeline. Never deserializes arbitrary external content.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import pandas as pd
import pickle

_COLUMNS = [
    "plate_id", "latitude", "longitude",
    "seconds", "passenger_indicator", "timestamp",
]


def _flatten_driver_records(records_obj) -> list[list]:
    """Handle both flat and nested day-list raw structures."""
    if not isinstance(records_obj, list) or not records_obj:
        return []
    first = records_obj[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        flat: list[list] = []
        for day_list in records_obj:
            if isinstance(day_list, list):
                for rec in day_list:
                    if isinstance(rec, (list, tuple)) and len(rec) >= 6:
                        flat.append(list(rec[:6]))
        return flat
    return [
        list(r[:6]) for r in records_obj
        if isinstance(r, (list, tuple)) and len(r) >= 6
    ]


def load_raw_file(path: Path) -> pd.DataFrame:
    """Load one raw taxi_record_*.pkl file into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"{path.name}: expected dict keyed by plate_id, got {type(data).__name__}"
        )
    all_records: list[list] = []
    for plate_id, records_obj in data.items():
        for rec in _flatten_driver_records(records_obj):
            rec[0] = str(plate_id) if rec[0] is None else str(rec[0])
            all_records.append(rec)
    if not all_records:
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.DataFrame(all_records, columns=_COLUMNS)
    df["plate_id"] = df["plate_id"].astype(str)
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["seconds"] = df["seconds"].astype(int)
    df["passenger_indicator"] = df["passenger_indicator"].astype(int)
    df["timestamp"] = df["timestamp"].astype(str)
    return df


def concat_raw_records(paths: Iterable[Path]) -> pd.DataFrame:
    """Concatenate multiple raw files into a single DataFrame."""
    dfs = [load_raw_file(p) for p in paths]
    dfs = [d for d in dfs if len(d) > 0]
    if not dfs:
        raise ValueError("concat_raw_records: no non-empty raw files found")
    return pd.concat(dfs, ignore_index=True)
```

- [ ] **Step 2.4: Run the tests — expect pass**

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_raw_loader.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add famail_temporal/data/source_generation/raw_loader.py \
        famail_temporal/data/source_generation/tests/test_raw_loader.py
git commit -m "feat(source_generation): add raw_loader for taxi_record pkl files"
```

---

### Task 3: Quantization primitives

**Files:**
- Create: `famail_temporal/data/source_generation/quantization.py`
- Create: `famail_temporal/data/source_generation/tests/test_quantization.py`

- [ ] **Step 3.1: Write the failing test**

Create `famail_temporal/data/source_generation/tests/test_quantization.py`:

```python
"""Tests for quantization.py."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from famail_temporal.data.source_generation.quantization import (
    GlobalBounds, compute_global_bounds, gps_to_grid,
    seconds_to_time_bucket, seconds_to_hour, timestamp_to_day,
)


def test_compute_global_bounds():
    lat = pd.Series([22.5, 22.8, 22.6])
    lon = pd.Series([113.8, 114.5, 114.0])
    b = compute_global_bounds(lat, lon)
    assert b.lat_min == pytest.approx(22.5)
    assert b.lat_max == pytest.approx(22.8)
    assert b.lon_min == pytest.approx(113.8)
    assert b.lon_max == pytest.approx(114.5)


def test_gps_to_grid_returns_1_indexed():
    b = GlobalBounds(lat_min=22.5, lat_max=22.9, lon_min=113.8, lon_max=114.5)
    x, y = gps_to_grid(22.5, 113.8, b)
    assert (int(x), int(y)) == (1, 1)


def test_gps_to_grid_upper_corner_within_max():
    b = GlobalBounds(lat_min=22.5, lat_max=22.9, lon_min=113.8, lon_max=114.5)
    x, y = gps_to_grid(22.89, 114.49, b)
    assert 1 <= int(x) <= 48
    assert 1 <= int(y) <= 90


def test_gps_to_grid_vectorized():
    b = GlobalBounds(lat_min=22.5, lat_max=22.9, lon_min=113.8, lon_max=114.5)
    lats = np.array([22.5, 22.6, 22.8])
    lons = np.array([113.8, 114.0, 114.4])
    xs, ys = gps_to_grid(lats, lons, b)
    assert xs.shape == (3,)
    assert ys.shape == (3,)
    assert (xs >= 1).all() and (ys >= 1).all()


def test_seconds_to_time_bucket_midnight_is_1():
    assert int(seconds_to_time_bucket(0)) == 1
    assert int(seconds_to_time_bucket(60)) == 1
    assert int(seconds_to_time_bucket(4 * 60 + 59)) == 1


def test_seconds_to_time_bucket_first_hour_boundary():
    assert int(seconds_to_time_bucket(5 * 60)) == 2
    assert int(seconds_to_time_bucket(60 * 60 - 1)) == 12
    assert int(seconds_to_time_bucket(60 * 60)) == 13


def test_seconds_to_time_bucket_last_is_288():
    last_second = 24 * 60 * 60 - 1
    assert int(seconds_to_time_bucket(last_second)) == 288


def test_seconds_to_time_bucket_vectorized():
    arr = np.array([0, 60, 60 * 60, 23 * 60 * 60], dtype=int)
    out = seconds_to_time_bucket(arr)
    assert list(out) == [1, 1, 13, 277]


def test_seconds_to_hour():
    assert int(seconds_to_hour(0)) == 0
    assert int(seconds_to_hour(60 * 60)) == 1
    assert int(seconds_to_hour(23 * 60 * 60)) == 23


def test_timestamp_to_day_weekdays():
    assert timestamp_to_day("2016-07-04 08:00:00") == 1
    assert timestamp_to_day("2016-07-08 08:00:00") == 5


def test_timestamp_to_day_weekends_return_none():
    assert timestamp_to_day("2016-07-02 12:00:00") is None
    assert timestamp_to_day("2016-07-03 12:00:00") is None


def test_timestamp_to_day_bad_format():
    assert timestamp_to_day("not a date") is None
```

- [ ] **Step 3.2: Run tests — expect failure**

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_quantization.py -v
```

Expected: ImportError.

- [ ] **Step 3.3: Implement `quantization.py`**

Create `famail_temporal/data/source_generation/quantization.py`:

```python
"""Authoritative spatial and temporal quantization primitives.

Every module in the source-generation pipeline calls these functions (and only
these) for lat/lon → (x, y), seconds → time_bucket, timestamp → day.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

from famail_temporal.data.source_generation import config


Scalar = Union[int, float]
Array = np.ndarray


@dataclass(frozen=True)
class GlobalBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def compute_global_bounds(
    latitudes: pd.Series, longitudes: pd.Series,
) -> GlobalBounds:
    if len(latitudes) == 0 or len(longitudes) == 0:
        raise ValueError("compute_global_bounds: empty input")
    return GlobalBounds(
        lat_min=float(latitudes.min()),
        lat_max=float(latitudes.max()),
        lon_min=float(longitudes.min()),
        lon_max=float(longitudes.max()),
    )


def _bins(bound_min: float, bound_max: float) -> np.ndarray:
    return np.arange(bound_min, bound_max + config.GRID_SIZE_DEG, config.GRID_SIZE_DEG)


def gps_to_grid(
    lat: Union[Scalar, Array, pd.Series],
    lon: Union[Scalar, Array, pd.Series],
    bounds: GlobalBounds,
) -> tuple[Array, Array]:
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    lat_bins = _bins(bounds.lat_min, bounds.lat_max)
    lon_bins = _bins(bounds.lon_min, bounds.lon_max)
    x0 = np.digitize(lat_arr, lat_bins) - 1
    y0 = np.digitize(lon_arr, lon_bins) - 1
    x0 = np.clip(x0, 0, config.X_GRID_MAX - 1)
    y0 = np.clip(y0, 0, config.Y_GRID_MAX - 1)
    x = x0 + config.X_GRID_OFFSET
    y = y0 + config.Y_GRID_OFFSET
    return x, y


def seconds_to_time_bucket(seconds: Union[Scalar, Array, pd.Series]) -> Array:
    """Convert seconds-since-midnight to 1-indexed 5-min time_bucket.

    00:00:00 → 1; 00:04:59 → 1; 00:05:00 → 2; 23:59:59 → 288.
    """
    s_arr = np.asarray(seconds, dtype=int)
    bucket_0idx = s_arr // (config.TIME_INTERVAL_MIN * 60)
    bucket_1idx = bucket_0idx + 1
    return np.clip(bucket_1idx, 1, config.TIME_BUCKET_MAX)


def seconds_to_hour(seconds: Union[Scalar, Array, pd.Series]) -> Array:
    """Convert seconds-since-midnight to 0-indexed hour [0, 23]."""
    s_arr = np.asarray(seconds, dtype=int)
    h = s_arr // 3600
    return np.clip(h, 0, config.HOUR_MAX)


def timestamp_to_day(ts: str) -> int | None:
    """Convert a 'YYYY-MM-DD HH:MM:SS' string to a 1-indexed weekday index.

    Mon=1 .. Fri=5. Returns None for Sat, Sun, or unparseable input.
    """
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None
    dow = dt.weekday()
    if dow >= 5:
        return None
    return dow + 1
```

- [ ] **Step 3.4: Run tests — expect pass**

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_quantization.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add famail_temporal/data/source_generation/quantization.py \
        famail_temporal/data/source_generation/tests/test_quantization.py
git commit -m "feat(source_generation): add quantization primitives"
```

### Phase 1 checkpoint

- [ ] **Run all source_generation tests**

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/ -v
```

Expected: all tests PASS, 0 failures.

- [ ] **Dispatch code-review subagent** (see policy at top of this plan):
  - Model: opus
  - Skills: `superpowers:requesting-code-review`, `superpowers:code-reviewer`
  - Scope: the Phase 1 files (`config.py`, `raw_loader.py`, `quantization.py`, tests)
  - Reviewer prompt must ask for: correctness, TDD discipline, YAGNI violations, unclear naming

- [ ] **Dispatch simplifier subagent** if reviewer notes verbosity:
  - Skills: `code-simplifier:code-simplifier` or `superpowers:simplify`

Address any blocking review items before moving on.

---

## Phase 2 — Event stream (single source of truth)

### Task 4: Transition detection

**Files:**
- Create: `famail_temporal/data/source_generation/transitions.py`
- Create: `famail_temporal/data/source_generation/tests/test_transitions.py`

- [ ] **Step 4.1: Write the failing test**

Create `famail_temporal/data/source_generation/tests/test_transitions.py`:

```python
"""Tests for transition detection."""
from __future__ import annotations
import pandas as pd

from famail_temporal.data.source_generation.transitions import (
    add_transition_columns, assign_segment_ids,
)


def _make_driver_df(plate: str, passenger_seq: list[int]) -> pd.DataFrame:
    n = len(passenger_seq)
    return pd.DataFrame({
        "plate_id": [plate] * n,
        "timestamp": [f"2016-07-04 00:00:{i:02d}" for i in range(n)],
        "passenger_indicator": passenger_seq,
    })


def test_add_transition_columns_detects_pickups_and_dropoffs():
    df = _make_driver_df("A", [1, 1, 0, 0, 0, 0, 1, 1, 0])
    out = add_transition_columns(df)
    assert out["is_pickup"].tolist()  == [False, False, False, False, False, False, True,  False, False]
    assert out["is_dropoff"].tolist() == [False, False, True,  False, False, False, False, False, True]


def test_add_transition_columns_per_driver():
    dfA = _make_driver_df("A", [1, 0])
    dfB = _make_driver_df("B", [0, 1])
    df = pd.concat([dfA, dfB], ignore_index=True)
    out = add_transition_columns(df)
    assert out.loc[0, "is_dropoff"] == False
    assert out.loc[1, "is_dropoff"] == True
    assert out.loc[2, "is_pickup"] == False
    assert out.loc[3, "is_pickup"] == True


def test_assign_segment_ids_increments_after_each_transition():
    df = _make_driver_df("A", [1, 1, 0, 0, 0, 0, 1, 1, 0])
    df = add_transition_columns(df)
    df = assign_segment_ids(df)
    assert df["segment_id"].tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2]


def test_assign_segment_ids_per_driver_independent():
    dfA = _make_driver_df("A", [1, 1, 0, 0, 1])
    dfB = _make_driver_df("B", [0, 1, 1, 0])
    df = pd.concat([dfA, dfB], ignore_index=True)
    df = add_transition_columns(df)
    df = assign_segment_ids(df)
    a_rows = df[df["plate_id"] == "A"]["segment_id"].tolist()
    b_rows = df[df["plate_id"] == "B"]["segment_id"].tolist()
    assert a_rows == [0, 0, 0, 1, 1]
    assert b_rows == [0, 0, 1, 1]
```

- [ ] **Step 4.2: Run tests — expect failure**

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_transitions.py -v
```

- [ ] **Step 4.3: Implement `transitions.py`**

Create `famail_temporal/data/source_generation/transitions.py`:

```python
"""Per-driver passenger-indicator transition detection.

A pickup is a 0→1 transition; a dropoff is a 1→0 transition. Each transition
row is the FINAL (post-transition) state of its trajectory:
- The 1→0 row is the last state of a driving trajectory.
- The 0→1 row is the last state of a seeking trajectory.

`assign_segment_ids` gives each row a segment_id such that the transition row
is the LAST row of its segment and the next row starts a new segment.
"""
from __future__ import annotations
import pandas as pd


def add_transition_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add `is_pickup`, `is_dropoff`, `is_transition` columns (per driver)."""
    out = df.copy()
    diff = out.groupby("plate_id")["passenger_indicator"].diff()
    out["is_pickup"] = diff == 1
    out["is_dropoff"] = diff == -1
    out["is_transition"] = out["is_pickup"] | out["is_dropoff"]
    return out


def assign_segment_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign per-driver `segment_id` such that each transition row is the
    LAST row of its segment (cumsum of is_transition, shifted by 1)."""
    out = df.copy()
    out["segment_id"] = (
        out.groupby("plate_id")["is_transition"]
        .apply(lambda s: s.cumsum().shift(1).fillna(0).astype(int))
        .reset_index(level=0, drop=True)
    )
    return out
```

- [ ] **Step 4.4: Run tests — expect pass**

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_transitions.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add famail_temporal/data/source_generation/transitions.py \
        famail_temporal/data/source_generation/tests/test_transitions.py
git commit -m "feat(source_generation): add per-driver transition detection"
```

---

### Task 5: Enriched event stream

**Files:**
- Create: `famail_temporal/data/source_generation/event_stream.py`
- Create: `famail_temporal/data/source_generation/tests/test_event_stream.py`

- [ ] **Step 5.1: Write the failing test**

Create `famail_temporal/data/source_generation/tests/test_event_stream.py`:

```python
"""Tests for event_stream.py — the enriched DataFrame used as the single
source of truth across all views."""
from __future__ import annotations
import pickle
from pathlib import Path

import pandas as pd
import pytest

from famail_temporal.data.source_generation.event_stream import build_event_stream


def _write_pkl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


@pytest.fixture
def tiny_raw(tmp_path):
    path = tmp_path / "taxi_record_07_50drivers.pkl"
    _write_pkl(path, {
        "A": [
            ["A", 22.5, 113.8, 0,    1, "2016-07-04 00:00:00"],
            ["A", 22.5, 113.8, 60,   1, "2016-07-04 00:01:00"],
            ["A", 22.5, 113.8, 120,  0, "2016-07-04 00:02:00"],
            ["A", 22.5, 113.8, 180,  0, "2016-07-04 00:03:00"],
            ["A", 22.5, 113.8, 240,  0, "2016-07-04 00:04:00"],
            ["A", 22.5, 113.8, 300,  0, "2016-07-04 00:05:00"],
            ["A", 22.5, 113.8, 360,  1, "2016-07-04 00:06:00"],
            ["A", 22.5, 113.8, 420,  1, "2016-07-04 00:07:00"],
            ["A", 22.5, 113.8, 480,  0, "2016-07-04 00:08:00"],
        ],
        "B": [
            ["B", 22.6, 114.0, 0,    0, "2016-07-04 00:00:00"],
            ["B", 22.6, 114.0, 60,   1, "2016-07-04 00:01:00"],
            ["B", 22.6, 114.0, 120,  0, "2016-07-04 00:02:00"],
        ],
    })
    _write_pkl(tmp_path / "taxi_record_08_50drivers.pkl", {})
    _write_pkl(tmp_path / "taxi_record_09_50drivers.pkl", {})
    return tmp_path


def test_build_event_stream_returns_dataframe(tiny_raw):
    es = build_event_stream(tiny_raw)
    assert isinstance(es.df, pd.DataFrame)
    for col in ("plate_id", "x_grid", "y_grid", "time_bucket",
                "hour", "day_index", "is_pickup", "is_dropoff",
                "segment_id", "passenger_indicator"):
        assert col in es.df.columns


def test_build_event_stream_drops_weekends(tmp_path):
    _write_pkl(tmp_path / "taxi_record_07_50drivers.pkl", {
        "A": [
            ["A", 22.5, 113.8, 0, 0, "2016-07-02 12:00:00"],  # Saturday
            ["A", 22.5, 113.8, 0, 0, "2016-07-04 12:00:00"],  # Monday
        ],
    })
    _write_pkl(tmp_path / "taxi_record_08_50drivers.pkl", {})
    _write_pkl(tmp_path / "taxi_record_09_50drivers.pkl", {})
    es = build_event_stream(tmp_path)
    assert len(es.df) == 1
    assert es.df.iloc[0]["day_index"] == 1


def test_build_event_stream_is_sorted_per_driver(tiny_raw):
    es = build_event_stream(tiny_raw)
    for plate, group in es.df.groupby("plate_id"):
        ts = list(group["timestamp"])
        assert ts == sorted(ts)


def test_build_event_stream_has_correct_transitions(tiny_raw):
    es = build_event_stream(tiny_raw)
    A = es.df[es.df["plate_id"] == "A"].reset_index(drop=True)
    assert A.loc[2, "is_dropoff"] == True
    assert A.loc[6, "is_pickup"] == True
    assert A.loc[8, "is_dropoff"] == True
    assert A["is_pickup"].sum() == 1
    assert A["is_dropoff"].sum() == 2


def test_build_event_stream_computes_n_days(tiny_raw):
    es = build_event_stream(tiny_raw)
    assert es.n_days >= 1


def test_build_event_stream_computes_global_bounds(tiny_raw):
    es = build_event_stream(tiny_raw)
    assert es.bounds.lat_min == pytest.approx(22.5)
    assert es.bounds.lat_max == pytest.approx(22.6)
```

- [ ] **Step 5.2: Run tests — expect failure**

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_event_stream.py -v
```

- [ ] **Step 5.3: Implement `event_stream.py`**

Create `famail_temporal/data/source_generation/event_stream.py`:

```python
"""Build the single enriched event-stream DataFrame — the load-bearing
intermediate representation every view derives from."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from famail_temporal.data.source_generation import config
from famail_temporal.data.source_generation.quantization import (
    GlobalBounds, compute_global_bounds, gps_to_grid,
    seconds_to_time_bucket, seconds_to_hour, timestamp_to_day,
)
from famail_temporal.data.source_generation.raw_loader import concat_raw_records
from famail_temporal.data.source_generation.transitions import (
    add_transition_columns, assign_segment_ids,
)


@dataclass(frozen=True)
class EventStream:
    df: pd.DataFrame
    bounds: GlobalBounds
    n_days: int
    driver_calendar_days: dict[str, set[int]]


def build_event_stream(raw_dir: Path) -> EventStream:
    """Load raw → quantize → weekday-filter → sort → detect transitions."""
    paths = [raw_dir / name for name in config.RAW_INPUT_FILENAMES]
    df = concat_raw_records(paths)

    bounds = compute_global_bounds(df["latitude"], df["longitude"])

    xs, ys = gps_to_grid(df["latitude"].values, df["longitude"].values, bounds)
    df["x_grid"] = xs
    df["y_grid"] = ys
    df["time_bucket"] = seconds_to_time_bucket(df["seconds"].values)
    df["hour"] = seconds_to_hour(df["seconds"].values)

    df["day_index"] = df["timestamp"].map(timestamp_to_day)
    df["calendar_date"] = df["timestamp"].str[:10]
    df = df.dropna(subset=["day_index"]).reset_index(drop=True)
    df["day_index"] = df["day_index"].astype(int)

    df = df.sort_values(["plate_id", "timestamp"], kind="stable").reset_index(drop=True)

    df = add_transition_columns(df)
    df = assign_segment_ids(df)

    n_days = df["calendar_date"].nunique()
    driver_calendar_days: dict[str, set[int]] = (
        df.groupby("plate_id")["day_index"].apply(lambda s: set(int(d) for d in s)).to_dict()
    )

    return EventStream(
        df=df, bounds=bounds, n_days=n_days,
        driver_calendar_days=driver_calendar_days,
    )
```

- [ ] **Step 5.4: Run tests — expect pass**

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_event_stream.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5.5: Commit**

```bash
git add famail_temporal/data/source_generation/event_stream.py \
        famail_temporal/data/source_generation/tests/test_event_stream.py
git commit -m "feat(source_generation): add enriched event-stream build"
```

### Phase 2 checkpoint

- [ ] Run all tests in `famail_temporal/data/source_generation/tests/`.
- [ ] Dispatch code-review subagent (Opus, `superpowers:code-reviewer`). Scope: Phase 2 files.
- [ ] Dispatch simplifier if reviewer flags verbosity.

---

## Phase 3 — Views

### Task 6: pickup_dropoff view

**Files:**
- Create: `famail_temporal/data/source_generation/views/pickup_dropoff.py`
- Create: `famail_temporal/data/source_generation/tests/test_view_pickup_dropoff.py`

- [ ] **Step 6.1: Write the failing test**

Create `famail_temporal/data/source_generation/tests/test_view_pickup_dropoff.py`:

```python
"""Tests for views/pickup_dropoff.py."""
from __future__ import annotations
import pandas as pd

from famail_temporal.data.source_generation.views.pickup_dropoff import (
    build_pickup_dropoff_counts,
)


def _make_event_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_empty_df_returns_empty_dict():
    df = _make_event_df([])
    out = build_pickup_dropoff_counts(df)
    assert out == {}


def test_single_pickup_contributes_one_pickup_zero_dropoff():
    df = _make_event_df([
        {"x_grid": 5, "y_grid": 10, "time_bucket": 20, "day_index": 1,
         "is_pickup": True, "is_dropoff": False},
    ])
    out = build_pickup_dropoff_counts(df)
    assert out == {(5, 10, 20, 1): (1, 0)}


def test_multiple_events_aggregate_per_key():
    df = _make_event_df([
        {"x_grid": 5, "y_grid": 10, "time_bucket": 20, "day_index": 1,
         "is_pickup": True, "is_dropoff": False},
        {"x_grid": 5, "y_grid": 10, "time_bucket": 20, "day_index": 1,
         "is_pickup": True, "is_dropoff": False},
        {"x_grid": 5, "y_grid": 10, "time_bucket": 20, "day_index": 1,
         "is_pickup": False, "is_dropoff": True},
    ])
    out = build_pickup_dropoff_counts(df)
    assert out == {(5, 10, 20, 1): (2, 1)}
```

- [ ] **Step 6.2: Run tests — expect failure**

- [ ] **Step 6.3: Implement `views/pickup_dropoff.py`**

Create `famail_temporal/data/source_generation/views/pickup_dropoff.py`:

```python
"""View: produce pickup_dropoff_counts dictionary from the event stream.

Output schema: dict[(x, y, time_bucket, day_index)] -> (pickup_count, dropoff_count)
Only cells with at least one event appear. All counts are non-negative integers.
"""
from __future__ import annotations

import pandas as pd


def build_pickup_dropoff_counts(
    df: pd.DataFrame,
) -> dict[tuple[int, int, int, int], tuple[int, int]]:
    events = df[df["is_pickup"] | df["is_dropoff"]]
    if len(events) == 0:
        return {}
    grouped = (
        events
        .assign(_p=events["is_pickup"].astype(int),
                _d=events["is_dropoff"].astype(int))
        .groupby(["x_grid", "y_grid", "time_bucket", "day_index"], sort=False)
        .agg(pickup=("_p", "sum"), dropoff=("_d", "sum"))
        .reset_index()
    )
    out: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for row in grouped.itertuples(index=False):
        key = (int(row.x_grid), int(row.y_grid), int(row.time_bucket), int(row.day_index))
        out[key] = (int(row.pickup), int(row.dropoff))
    return out
```

- [ ] **Step 6.4: Run tests — expect pass.** Commit.

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_view_pickup_dropoff.py -v
git add famail_temporal/data/source_generation/views/pickup_dropoff.py \
        famail_temporal/data/source_generation/tests/test_view_pickup_dropoff.py
git commit -m "feat(source_generation): add pickup_dropoff_counts view"
```

---

### Task 7: active_taxis view

**Files:**
- Create: `famail_temporal/data/source_generation/views/active_taxis.py`
- Create: `famail_temporal/data/source_generation/tests/test_view_active_taxis.py`

- [ ] **Step 7.1: Write the failing test**

Create `famail_temporal/data/source_generation/tests/test_view_active_taxis.py`:

```python
"""Tests for views/active_taxis.py."""
from __future__ import annotations
import pandas as pd

from famail_temporal.data.source_generation.views.active_taxis import (
    build_active_taxis_counts,
)


def _row(plate, x, y, hour, day, passenger=0):
    return {
        "plate_id": plate, "x_grid": x, "y_grid": y,
        "hour": hour, "day_index": day,
        "passenger_indicator": passenger,
    }


def test_empty_returns_empty_dict():
    df = pd.DataFrame(columns=[
        "plate_id", "x_grid", "y_grid", "hour",
        "day_index", "passenger_indicator",
    ])
    assert build_active_taxis_counts(df) == {}


def test_single_empty_driver_counts_in_5x5_neighborhood():
    df = pd.DataFrame([_row("A", 10, 10, 5, 1, passenger=0)])
    out = build_active_taxis_counts(df)
    assert out[(10, 10, 5, 1)] == 1
    assert out[(8, 8, 5, 1)] == 1
    assert out[(12, 12, 5, 1)] == 1


def test_occupied_only_driver_not_counted():
    df = pd.DataFrame([_row("B", 10, 10, 5, 1, passenger=1)])
    assert build_active_taxis_counts(df) == {}


def test_driver_with_any_empty_ping_counts_once():
    df = pd.DataFrame([
        _row("A", 10, 10, 5, 1, passenger=1),
        _row("A", 10, 10, 5, 1, passenger=0),
        _row("A", 10, 10, 5, 1, passenger=1),
    ])
    out = build_active_taxis_counts(df)
    assert out[(10, 10, 5, 1)] == 1


def test_two_distinct_drivers_count_as_two():
    df = pd.DataFrame([
        _row("A", 10, 10, 5, 1, passenger=0),
        _row("B", 10, 10, 5, 1, passenger=0),
    ])
    assert build_active_taxis_counts(df)[(10, 10, 5, 1)] == 2


def test_different_hours_independent():
    df = pd.DataFrame([
        _row("A", 10, 10, 5, 1, passenger=0),
        _row("A", 10, 10, 6, 1, passenger=0),
    ])
    out = build_active_taxis_counts(df)
    assert out[(10, 10, 5, 1)] == 1
    assert out[(10, 10, 6, 1)] == 1


def test_neighborhood_edge_of_grid_clamped():
    df = pd.DataFrame([_row("A", 1, 1, 5, 1, passenger=0)])
    out = build_active_taxis_counts(df)
    assert out[(1, 1, 5, 1)] == 1
    assert out[(3, 3, 5, 1)] == 1
    for (x, y, _, _) in out.keys():
        assert x >= 1 and y >= 1
```

- [ ] **Step 7.2: Run tests — expect failure.**

- [ ] **Step 7.3: Implement `views/active_taxis.py`**

Create `famail_temporal/data/source_generation/views/active_taxis.py`:

```python
"""View: produce active_taxis counts (5×5 neighborhood, hourly, empty-only).

A driver counts as active at target cell (cx, cy) during (hour, day) if they
had >=1 GPS ping in the 5×5 neighborhood around (cx, cy) during that hour,
AND at least one of those pings had passenger_indicator == 0.
"""
from __future__ import annotations

import pandas as pd

from famail_temporal.data.source_generation import config


def build_active_taxis_counts(
    df: pd.DataFrame,
) -> dict[tuple[int, int, int, int], int]:
    if len(df) == 0:
        return {}

    empties = df[df["passenger_indicator"] == 0][
        ["plate_id", "x_grid", "y_grid", "hour", "day_index"]
    ].copy()
    if len(empties) == 0:
        return {}

    empties = empties.drop_duplicates(
        subset=["plate_id", "x_grid", "y_grid", "hour", "day_index"]
    )

    k = config.NEIGHBORHOOD_K
    pieces: list[pd.DataFrame] = []
    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            exp = empties.copy()
            exp["x_grid"] = exp["x_grid"] + dx
            exp["y_grid"] = exp["y_grid"] + dy
            pieces.append(exp)
    expanded = pd.concat(pieces, ignore_index=True)

    expanded = expanded[
        (expanded["x_grid"] >= 1) & (expanded["x_grid"] <= config.X_GRID_MAX)
        & (expanded["y_grid"] >= 1) & (expanded["y_grid"] <= config.Y_GRID_MAX)
    ]

    expanded = expanded.drop_duplicates(
        subset=["plate_id", "x_grid", "y_grid", "hour", "day_index"]
    )

    counts = (
        expanded
        .groupby(["x_grid", "y_grid", "hour", "day_index"], sort=False)
        .size()
        .reset_index(name="count")
    )

    return {
        (int(r.x_grid), int(r.y_grid), int(r.hour), int(r.day_index)): int(r.count)
        for r in counts.itertuples(index=False)
    }
```

- [ ] **Step 7.4: Run tests — expect pass.** Commit.

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_view_active_taxis.py -v
git add famail_temporal/data/source_generation/views/active_taxis.py \
        famail_temporal/data/source_generation/tests/test_view_active_taxis.py
git commit -m "feat(source_generation): add active_taxis view (available-only, 5x5)"
```

---

### Task 8: trajectories view

**Files:**
- Create: `famail_temporal/data/source_generation/views/trajectories.py`
- Create: `famail_temporal/data/source_generation/tests/test_view_trajectories.py`

- [ ] **Step 8.1: Write the failing test**

Create `famail_temporal/data/source_generation/tests/test_view_trajectories.py`:

```python
"""Tests for views/trajectories.py."""
from __future__ import annotations
import pandas as pd

from famail_temporal.data.source_generation.views.trajectories import (
    build_trajectories, build_driver_index_mapping,
)


def _event_df():
    records = [
        {"plate_id": "A", "x_grid": 5, "y_grid": 10, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 1, "is_pickup": False, "is_dropoff": False, "segment_id": 0,
         "timestamp": "2016-07-04 00:00:00"},
        {"plate_id": "A", "x_grid": 5, "y_grid": 11, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 1, "is_pickup": False, "is_dropoff": False, "segment_id": 0,
         "timestamp": "2016-07-04 00:01:00"},
        {"plate_id": "A", "x_grid": 6, "y_grid": 11, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": True, "segment_id": 0,
         "timestamp": "2016-07-04 00:02:00"},
        {"plate_id": "A", "x_grid": 6, "y_grid": 12, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": False, "segment_id": 1,
         "timestamp": "2016-07-04 00:03:00"},
        {"plate_id": "A", "x_grid": 7, "y_grid": 12, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": False, "segment_id": 1,
         "timestamp": "2016-07-04 00:04:00"},
        {"plate_id": "A", "x_grid": 7, "y_grid": 13, "time_bucket": 2, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": False, "segment_id": 1,
         "timestamp": "2016-07-04 00:05:00"},
        {"plate_id": "A", "x_grid": 8, "y_grid": 13, "time_bucket": 2, "day_index": 1,
         "passenger_indicator": 1, "is_pickup": True, "is_dropoff": False, "segment_id": 1,
         "timestamp": "2016-07-04 00:06:00"},
        {"plate_id": "A", "x_grid": 8, "y_grid": 14, "time_bucket": 2, "day_index": 1,
         "passenger_indicator": 1, "is_pickup": False, "is_dropoff": False, "segment_id": 2,
         "timestamp": "2016-07-04 00:07:00"},
        {"plate_id": "A", "x_grid": 9, "y_grid": 14, "time_bucket": 2, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": True, "segment_id": 2,
         "timestamp": "2016-07-04 00:08:00"},
    ]
    return pd.DataFrame(records)


def test_driver_mapping_is_lexicographic():
    df = pd.DataFrame({"plate_id": ["Z", "A", "M"]})
    mapping = build_driver_index_mapping(df)
    assert mapping["plate_to_idx"] == {"A": 0, "M": 1, "Z": 2}
    assert mapping["idx_to_plate"] == {0: "A", 1: "M", 2: "Z"}


def test_build_trajectories_extracts_seeking_and_driving():
    df = _event_df()
    result = build_trajectories(df)
    A_seeking = result.seeking_by_plate.get("A", [])
    A_driving = result.driving_by_plate.get("A", [])
    assert len(A_seeking) == 1
    assert len(A_driving) == 2


def test_seeking_state_minus_one_is_pickup_cell():
    df = _event_df()
    result = build_trajectories(df)
    A_seek0 = result.seeking_by_plate["A"][0]
    assert A_seek0[-1] == [8, 13, 2, 1]


def test_driving_state_minus_one_is_dropoff_cell():
    df = _event_df()
    result = build_trajectories(df)
    A_drv0 = result.driving_by_plate["A"][0]
    assert A_drv0[-1] == [6, 11, 1, 1]


def test_min_length_filter_drops_length_1_segments():
    df = pd.DataFrame([{
        "plate_id": "A", "x_grid": 5, "y_grid": 10, "time_bucket": 1, "day_index": 1,
        "passenger_indicator": 1, "is_pickup": True, "is_dropoff": False,
        "segment_id": 0, "timestamp": "2016-07-04 00:00:00",
    }])
    result = build_trajectories(df)
    assert result.seeking_by_plate.get("A", []) == []


def test_incomplete_trailing_segment_dropped():
    df = pd.DataFrame([
        {"plate_id": "A", "x_grid": 5, "y_grid": 10, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": False,
         "segment_id": 0, "timestamp": "2016-07-04 00:00:00"},
        {"plate_id": "A", "x_grid": 5, "y_grid": 11, "time_bucket": 1, "day_index": 1,
         "passenger_indicator": 0, "is_pickup": False, "is_dropoff": False,
         "segment_id": 0, "timestamp": "2016-07-04 00:01:00"},
    ])
    result = build_trajectories(df)
    assert result.seeking_by_plate.get("A", []) == []
    assert result.driving_by_plate.get("A", []) == []
```

- [ ] **Step 8.2: Run tests — expect failure.**

- [ ] **Step 8.3: Implement `views/trajectories.py`**

Create `famail_temporal/data/source_generation/views/trajectories.py`:

```python
"""View: produce the three trajectory files + driver index mapping.

state[-1] convention:
  - Seeking trajectory: last state is the pickup-transition record (passenger=1).
  - Driving trajectory: last state is the dropoff-transition record (passenger=0).

Only complete segments (ending in a transition row) with length >= 2 are emitted.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import pandas as pd


Trajectory = list[list[int]]


@dataclass
class TrajectoriesResult:
    seeking_by_plate: dict[str, list[Trajectory]] = field(default_factory=dict)
    driving_by_plate: dict[str, list[Trajectory]] = field(default_factory=dict)


def build_driver_index_mapping(df: pd.DataFrame) -> dict:
    plates = sorted(df["plate_id"].unique())
    plate_to_idx: dict[str, int] = {p: i for i, p in enumerate(plates)}
    idx_to_plate: dict[int, str] = {i: p for p, i in plate_to_idx.items()}
    return {"plate_to_idx": plate_to_idx, "idx_to_plate": idx_to_plate}


def _segment_is_seeking(segment: pd.DataFrame) -> bool:
    return bool(segment.iloc[-1]["is_pickup"])


def _segment_is_driving(segment: pd.DataFrame) -> bool:
    return bool(segment.iloc[-1]["is_dropoff"])


def _segment_to_trajectory(segment: pd.DataFrame) -> Trajectory:
    return [
        [int(r.x_grid), int(r.y_grid), int(r.time_bucket), int(r.day_index)]
        for r in segment.itertuples(index=False)
    ]


def build_trajectories(df: pd.DataFrame) -> TrajectoriesResult:
    result = TrajectoriesResult()
    for plate, driver_df in df.groupby("plate_id", sort=False):
        seeking: list[Trajectory] = []
        driving: list[Trajectory] = []
        for _, seg in driver_df.groupby("segment_id", sort=True):
            if len(seg) < 2:
                continue
            if _segment_is_seeking(seg):
                seeking.append(_segment_to_trajectory(seg))
            elif _segment_is_driving(seg):
                driving.append(_segment_to_trajectory(seg))
        if seeking:
            result.seeking_by_plate[plate] = seeking
        if driving:
            result.driving_by_plate[plate] = driving
    return result
```

- [ ] **Step 8.4: Run tests — expect pass.** Commit.

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_view_trajectories.py -v
git add famail_temporal/data/source_generation/views/trajectories.py \
        famail_temporal/data/source_generation/tests/test_view_trajectories.py
git commit -m "feat(source_generation): add trajectories view (seeking + driving)"
```

---

### Task 9: profile view (with home_x/y fallback cascade)

**Files:**
- Create: `famail_temporal/data/source_generation/views/profile.py`
- Create: `famail_temporal/data/source_generation/tests/test_view_profile.py`
- Create: `famail_temporal/data/source_generation/tests/test_profile_fallbacks.py`

- [ ] **Step 9.1: Write the failing tests**

Create `famail_temporal/data/source_generation/tests/test_view_profile.py`:

```python
"""Tests for views/profile.py."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from famail_temporal.data.source_generation.views.profile import (
    compute_profile_features, zscore_normalize,
)
from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult,
)


def _df_with_midnight_records(plate="A", midnight_cells=None):
    midnight_cells = midnight_cells or [(5, 10), (5, 10), (6, 11)]
    rows = []
    for i, (x, y) in enumerate(midnight_cells):
        rows.append({
            "plate_id": plate, "x_grid": x, "y_grid": y,
            "time_bucket": 1, "hour": 0, "day_index": 1,
            "calendar_date": f"2016-07-0{i+4}",
            "seconds": 60 * i,
            "passenger_indicator": 0,
        })
    return pd.DataFrame(rows)


def test_home_x_y_mode_of_time_bucket_1_cells():
    df = _df_with_midnight_records("A", [(5, 10), (5, 10), (6, 11)])
    trajs = TrajectoriesResult()
    features = compute_profile_features(df, trajs)
    assert features["A"]["home_x"] == 5
    assert features["A"]["home_y"] == 10


def test_shift_start_end_5th_95th_percentile():
    df = pd.DataFrame([{
        "plate_id": "A", "x_grid": 5, "y_grid": 10,
        "time_bucket": tb, "hour": tb // 12, "day_index": 1,
        "calendar_date": "2016-07-04", "seconds": 0, "passenger_indicator": 0,
    } for tb in [10, 50, 100, 150, 200, 250, 288]])
    trajs = TrajectoriesResult()
    features = compute_profile_features(df, trajs)
    assert features["A"]["shift_start"] == pytest.approx(22, abs=15)
    assert features["A"]["shift_end"] == pytest.approx(276, abs=15)


def test_zscore_normalize_50_drivers():
    raw = np.arange(50 * 11, dtype=float).reshape(50, 11)
    normalized, mean, std = zscore_normalize(raw)
    assert normalized.shape == (50, 11)
    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-6)
```

Create `famail_temporal/data/source_generation/tests/test_profile_fallbacks.py`:

```python
"""Tests for the home_x/y fallback cascade."""
from __future__ import annotations
import pandas as pd
import pytest

from famail_temporal.data.source_generation.views.profile import (
    compute_home_xy_with_fallback,
)


def _event_df(tb_cells: list[tuple[int, int, int]]):
    return pd.DataFrame([
        {"plate_id": "A", "x_grid": x, "y_grid": y,
         "time_bucket": tb, "hour": 0, "day_index": 1,
         "calendar_date": "2016-07-04", "seconds": 0, "passenger_indicator": 0}
        for (tb, x, y) in tb_cells
    ])


def test_home_uses_tb_1_records_when_present():
    df = _event_df([(1, 5, 10), (1, 5, 10), (50, 99, 99)])
    result = compute_home_xy_with_fallback(df)
    assert result["home_x"] == 5 and result["home_y"] == 10
    assert result["fallback_used"] == "none"


def test_home_falls_back_to_first_hour_when_no_tb_1():
    df = _event_df([(5, 7, 20), (10, 7, 20), (50, 99, 99)])
    result = compute_home_xy_with_fallback(df)
    assert result["home_x"] == 7 and result["home_y"] == 20
    assert result["fallback_used"] == "first_hour"


def test_home_falls_back_to_all_records_when_no_first_hour():
    df = _event_df([(50, 3, 4), (50, 3, 4), (200, 99, 99)])
    result = compute_home_xy_with_fallback(df)
    assert result["home_x"] == 3 and result["home_y"] == 4
    assert result["fallback_used"] == "all_records"


def test_home_fallback_empty_driver_raises():
    df = _event_df([])
    with pytest.raises(ValueError, match="no records"):
        compute_home_xy_with_fallback(df)
```

- [ ] **Step 9.2: Run tests — expect failure.**

- [ ] **Step 9.3: Implement `views/profile.py`**

Create `famail_temporal/data/source_generation/views/profile.py`:

```python
"""View: compute the 11-dim driver profile features and z-score normalize."""
from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd

from famail_temporal.data.source_generation import config
from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult, Trajectory,
)


def _mode_xy(sub: pd.DataFrame) -> tuple[int, int] | None:
    if len(sub) == 0:
        return None
    grouped = sub.groupby(["x_grid", "y_grid"]).size().reset_index(name="n")
    top = grouped.sort_values(
        ["n", "x_grid", "y_grid"], ascending=[False, True, True]
    ).iloc[0]
    return int(top.x_grid), int(top.y_grid)


def compute_home_xy_with_fallback(driver_df: pd.DataFrame) -> dict:
    if len(driver_df) == 0:
        raise ValueError("compute_home_xy_with_fallback: driver has no records")
    primary = driver_df[driver_df["time_bucket"] == 1]
    mode = _mode_xy(primary)
    if mode is not None:
        return {"home_x": mode[0], "home_y": mode[1], "fallback_used": "none"}
    first_hour = driver_df[driver_df["time_bucket"].between(1, 12)]
    mode = _mode_xy(first_hour)
    if mode is not None:
        return {"home_x": mode[0], "home_y": mode[1], "fallback_used": "first_hour"}
    mode = _mode_xy(driver_df)
    if mode is None:
        raise ValueError("compute_home_xy_with_fallback: driver has no records")
    return {"home_x": mode[0], "home_y": mode[1], "fallback_used": "all_records"}


def _trajectory_manhattan_length(traj: Trajectory) -> int:
    total = 0
    for a, b in zip(traj, traj[1:]):
        total += abs(a[0] - b[0]) + abs(a[1] - b[1])
    return total


def _trajectory_duration_minutes(
    driver_df: pd.DataFrame, traj: Trajectory,
) -> float | None:
    if len(traj) < 2:
        return None
    def find_s(state):
        matched = driver_df[
            (driver_df["x_grid"] == state[0])
            & (driver_df["y_grid"] == state[1])
            & (driver_df["time_bucket"] == state[2])
            & (driver_df["day_index"] == state[3])
        ]
        if len(matched) == 0:
            return None
        return matched.iloc[0]["seconds"]
    s_first = find_s(traj[0])
    s_last = find_s(traj[-1])
    if s_first is None or s_last is None:
        return None
    return max(0.0, (s_last - s_first) / 60.0)


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def compute_profile_features(
    df: pd.DataFrame, trajs: TrajectoriesResult,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for plate, driver_df in df.groupby("plate_id", sort=False):
        home = compute_home_xy_with_fallback(driver_df)

        tbs = driver_df["time_bucket"].values
        shift_start = float(np.percentile(tbs, config.PROFILE_SHIFT_LOW_PCT))
        shift_end = float(np.percentile(tbs, config.PROFILE_SHIFT_HIGH_PCT))

        seeking = trajs.seeking_by_plate.get(plate, [])
        driving = trajs.driving_by_plate.get(plate, [])

        if seeking:
            pickup_cells = pd.DataFrame(
                [(t[-1][0], t[-1][1]) for t in seeking],
                columns=["x_grid", "y_grid"],
            )
            freq = _mode_xy(pickup_cells)
            freq_grid_x, freq_grid_y = freq if freq else (home["home_x"], home["home_y"])
        else:
            freq_grid_x, freq_grid_y = home["home_x"], home["home_y"]

        avg_seek_dist = _mean_or_zero([_trajectory_manhattan_length(t) for t in seeking])
        avg_drive_dist = _mean_or_zero([_trajectory_manhattan_length(t) for t in driving])
        avg_seek_time = _mean_or_zero([
            d for d in (_trajectory_duration_minutes(driver_df, t) for t in seeking)
            if d is not None
        ])
        avg_drive_time = _mean_or_zero([
            d for d in (_trajectory_duration_minutes(driver_df, t) for t in driving)
            if d is not None
        ])

        total_pickups = (
            int(driver_df["is_pickup"].sum())
            if "is_pickup" in driver_df.columns
            else len(seeking)
        )
        distinct_dates = driver_df["calendar_date"].nunique()
        num_trips_per_day = (
            total_pickups / distinct_dates if distinct_dates > 0 else 0.0
        )

        out[plate] = {
            "home_x": home["home_x"],
            "home_y": home["home_y"],
            "shift_start": shift_start,
            "shift_end": shift_end,
            "freq_grid_x": freq_grid_x,
            "freq_grid_y": freq_grid_y,
            "avg_seeking_dist": avg_seek_dist,
            "avg_seeking_time": avg_seek_time,
            "avg_driving_dist": avg_drive_dist,
            "avg_driving_time": avg_drive_time,
            "num_trips_per_day": float(num_trips_per_day),
            "fallback_used": home["fallback_used"],
        }
    return out


def zscore_normalize(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = raw.mean(axis=0)
    std = raw.std(axis=0, ddof=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    normalized = (raw - mean) / std_safe
    return normalized, mean, std
```

- [ ] **Step 9.4: Run tests — expect pass.** Commit.

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_view_profile.py \
                 famail_temporal/data/source_generation/tests/test_profile_fallbacks.py -v
git add famail_temporal/data/source_generation/views/profile.py \
        famail_temporal/data/source_generation/tests/test_view_profile.py \
        famail_temporal/data/source_generation/tests/test_profile_fallbacks.py
git commit -m "feat(source_generation): add 11-feature profile view with fallback"
```

---

### Task 10: calendars view

**Files:**
- Create: `famail_temporal/data/source_generation/views/calendars.py`
- Create: `famail_temporal/data/source_generation/tests/test_view_calendars.py`

- [ ] **Step 10.1: Write the failing test**

Create `famail_temporal/data/source_generation/tests/test_view_calendars.py`:

```python
"""Tests for views/calendars.py."""
from __future__ import annotations

from famail_temporal.data.source_generation.views.calendars import (
    build_calendar_days_per_driver,
)
from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult,
)


def test_calendar_days_lists_unique_day_indices():
    result = TrajectoriesResult(
        seeking_by_plate={
            "A": [
                [[5, 10, 1, 1], [5, 11, 1, 1], [6, 11, 1, 1]],
                [[5, 10, 1, 3], [5, 11, 1, 3], [6, 11, 1, 3]],
                [[5, 10, 1, 1], [5, 11, 1, 1], [6, 11, 1, 1]],
            ],
        },
        driving_by_plate={
            "A": [
                [[6, 11, 1, 2], [7, 12, 1, 2], [7, 13, 1, 2]],
            ],
        },
    )
    mapping = {"plate_to_idx": {"A": 0}, "idx_to_plate": {0: "A"}}
    out = build_calendar_days_per_driver(result, mapping)
    assert out["seeking"] == {0: [1, 3]}
    assert out["driving"] == {0: [2]}


def test_missing_driver_produces_empty_list():
    result = TrajectoriesResult()
    mapping = {"plate_to_idx": {"A": 0}, "idx_to_plate": {0: "A"}}
    out = build_calendar_days_per_driver(result, mapping)
    assert out["seeking"] == {0: []}
    assert out["driving"] == {0: []}
```

- [ ] **Step 10.2: Run tests — expect failure.**

- [ ] **Step 10.3: Implement `views/calendars.py`**

Create `famail_temporal/data/source_generation/views/calendars.py`:

```python
"""View: per-driver calendar-day lists for the ms_{seeking,driving}_calendar_days files.

Currently unused by famail_temporal (loaded but not consumed in the context
builder today); provided for forward-compatibility with same-day context sampling.
"""
from __future__ import annotations

from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult,
)


def build_calendar_days_per_driver(
    trajs: TrajectoriesResult, mapping: dict,
) -> dict[str, dict[int, list[int]]]:
    seeking: dict[int, list[int]] = {}
    driving: dict[int, list[int]] = {}
    for plate, idx in mapping["plate_to_idx"].items():
        seek_days = sorted({t[0][3] for t in trajs.seeking_by_plate.get(plate, [])})
        drive_days = sorted({t[0][3] for t in trajs.driving_by_plate.get(plate, [])})
        seeking[idx] = seek_days
        driving[idx] = drive_days
    return {"seeking": seeking, "driving": driving}
```

- [ ] **Step 10.4: Run tests — expect pass.** Commit.

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_view_calendars.py -v
git add famail_temporal/data/source_generation/views/calendars.py \
        famail_temporal/data/source_generation/tests/test_view_calendars.py
git commit -m "feat(source_generation): add calendar_days view"
```

### Phase 3 checkpoint

- [ ] Run all tests in `famail_temporal/data/source_generation/tests/`.
- [ ] Dispatch code-review subagent (Opus, `superpowers:code-reviewer`) on the five view modules.
- [ ] Dispatch simplifier subagent if reviewer flags verbosity.

---

## Phase 4 — Invariants, writer, CLI

### Task 11: Invariants (per-trajectory + systemic)

**Files:**
- Create: `famail_temporal/data/source_generation/removal.py`
- Create: `famail_temporal/data/source_generation/invariants.py`
- Create: `famail_temporal/data/source_generation/tests/test_invariants.py`

- [ ] **Step 11.1: Write the failing test**

Create `famail_temporal/data/source_generation/tests/test_invariants.py`:

```python
"""Tests for invariants.py."""
from __future__ import annotations
import pytest

from famail_temporal.data.source_generation.invariants import (
    apply_per_trajectory_invariants, check_systemic_invariants,
    SystemicInvariantError,
)
from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult,
)


def _valid_seeking_traj():
    return [[5, 10, 1, 1], [5, 11, 1, 1], [6, 11, 2, 1]]


def test_per_trajectory_drops_out_of_bounds():
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [
            _valid_seeking_traj(),
            [[5, 10, 1, 1], [999, 999, 1, 1], [6, 11, 2, 1]],
        ],
    })
    pickup_counts = {(6, 11, 2, 1): (1, 0)}
    dropoff_counts: dict = {}
    kept, removals = apply_per_trajectory_invariants(
        trajs, pickup_counts, dropoff_counts,
    )
    assert len(kept.seeking_by_plate["A"]) == 1
    assert len(removals) == 1
    assert removals[0].removal_reason_category == "out_of_bounds"


def test_per_trajectory_drops_no_matching_pickup_count():
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [[[5, 10, 1, 1], [5, 11, 1, 1], [99, 99, 2, 1]]],
    })
    kept, removals = apply_per_trajectory_invariants(trajs, {}, {})
    assert kept.seeking_by_plate.get("A", []) == []
    assert len(removals) == 1
    assert removals[0].removal_reason_category == "no_matching_count"


def test_per_trajectory_drops_degenerate_length():
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [[[5, 10, 1, 1]]],
    })
    kept, removals = apply_per_trajectory_invariants(trajs, {}, {})
    assert kept.seeking_by_plate.get("A", []) == []
    assert len(removals) == 1
    assert removals[0].removal_reason_category == "degenerate_length"


def test_systemic_count_mismatch_raises():
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [_valid_seeking_traj()],
    })
    pickup_counts = {(6, 11, 2, 1): (2, 0)}
    with pytest.raises(SystemicInvariantError):
        check_systemic_invariants(
            trajs, pickup_counts, {}, profile_matrix=None, n_drivers=1,
            expect_n_drivers=1,
        )


def test_systemic_wrong_driver_count_raises():
    trajs = TrajectoriesResult(seeking_by_plate={"A": [_valid_seeking_traj()]})
    pickup_counts = {(6, 11, 2, 1): (1, 0)}
    with pytest.raises(SystemicInvariantError, match="50"):
        check_systemic_invariants(
            trajs, pickup_counts, {}, profile_matrix=None, n_drivers=1,
            expect_n_drivers=50,
        )
```

- [ ] **Step 11.2: Run tests — expect failure.**

- [ ] **Step 11.3: Implement `removal.py` and `invariants.py`**

Create `famail_temporal/data/source_generation/removal.py`:

```python
"""Per-trajectory removal record + summary."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal


RemovalCategory = Literal[
    "out_of_bounds",
    "degenerate_length",
    "no_matching_count",
    "temporal_order",
]


@dataclass
class RemovalRecord:
    driver_id: str
    driver_idx: int | None
    trajectory_index_within_driver: int
    kind: Literal["seeking", "driving"]
    which_invariant: int
    failing_values: dict[str, Any]
    n_states_before_removal: int
    removal_reason_category: RemovalCategory

    def to_dict(self) -> dict[str, Any]:
        return {
            "driver_id": self.driver_id,
            "driver_idx": self.driver_idx,
            "trajectory_index_within_driver": self.trajectory_index_within_driver,
            "kind": self.kind,
            "which_invariant": self.which_invariant,
            "failing_values": self.failing_values,
            "n_states_before_removal": self.n_states_before_removal,
            "removal_reason_category": self.removal_reason_category,
        }


@dataclass
class RemovalSummary:
    total_seeking_extracted: int = 0
    total_driving_extracted: int = 0
    removals: list[RemovalRecord] = field(default_factory=list)

    def counts_by_category(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.removals:
            out[r.removal_reason_category] = out.get(r.removal_reason_category, 0) + 1
        return out

    def total_extracted(self) -> int:
        return self.total_seeking_extracted + self.total_driving_extracted

    def removal_rate(self) -> float:
        total = self.total_extracted()
        return len(self.removals) / total if total > 0 else 0.0
```

Create `famail_temporal/data/source_generation/invariants.py`:

```python
"""Per-trajectory + systemic invariant enforcement (see §6 of the design spec)."""
from __future__ import annotations

import numpy as np

from famail_temporal.data.source_generation import config
from famail_temporal.data.source_generation.removal import (
    RemovalRecord,
)
from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult, Trajectory,
)


class SystemicInvariantError(Exception):
    """Raised when an invariant failure cannot be attributed to one trajectory."""


def _validate_single_trajectory(
    traj: Trajectory, kind: str,
    pickup_counts: dict, dropoff_counts: dict,
) -> tuple[bool, int, str, dict]:
    if len(traj) < 2:
        return False, 3, "degenerate_length", {"n_states": len(traj)}
    tbs = [s[2] for s in traj]
    for a, b in zip(tbs, tbs[1:]):
        if b < a:
            return False, 4, "temporal_order", {"time_buckets": tbs}
    for s in traj:
        x, y, tb, day = s
        if not (1 <= x <= config.X_GRID_MAX):
            return False, 2, "out_of_bounds", {"state": s, "axis": "x"}
        if not (1 <= y <= config.Y_GRID_MAX):
            return False, 2, "out_of_bounds", {"state": s, "axis": "y"}
        if not (1 <= tb <= config.TIME_BUCKET_MAX):
            return False, 2, "out_of_bounds", {"state": s, "axis": "time_bucket"}
        if day not in config.WEEKDAY_DAYS:
            return False, 2, "out_of_bounds", {"state": s, "axis": "day"}
    x, y, tb, day = traj[-1]
    key = (x, y, tb, day)
    if kind == "seeking":
        p, _ = pickup_counts.get(key, (0, 0))
        if p < 1:
            return False, 1, "no_matching_count", {"endpoint": key, "pickup_count": p}
    else:
        _, d = dropoff_counts.get(key, (0, 0))
        if d < 1:
            return False, 1, "no_matching_count", {"endpoint": key, "dropoff_count": d}
    return True, -1, "", {}


def apply_per_trajectory_invariants(
    trajs: TrajectoriesResult,
    pickup_counts: dict,
    dropoff_counts: dict,
    plate_to_idx: dict[str, int] | None = None,
) -> tuple[TrajectoriesResult, list[RemovalRecord]]:
    """Validate each trajectory; drop violations and record them."""
    plate_to_idx = plate_to_idx or {}
    kept = TrajectoriesResult()
    removals: list[RemovalRecord] = []

    def process(by_plate: dict[str, list[Trajectory]], kind: str):
        for plate, traj_list in by_plate.items():
            keep_list: list[Trajectory] = []
            for idx, traj in enumerate(traj_list):
                ok, inv_num, category, fv = _validate_single_trajectory(
                    traj, kind, pickup_counts, dropoff_counts,
                )
                if ok:
                    keep_list.append(traj)
                else:
                    removals.append(RemovalRecord(
                        driver_id=plate,
                        driver_idx=plate_to_idx.get(plate),
                        trajectory_index_within_driver=idx,
                        kind=kind,
                        which_invariant=inv_num,
                        failing_values=fv,
                        n_states_before_removal=len(traj),
                        removal_reason_category=category,
                    ))
            if keep_list:
                if kind == "seeking":
                    kept.seeking_by_plate[plate] = keep_list
                else:
                    kept.driving_by_plate[plate] = keep_list

    process(trajs.seeking_by_plate, "seeking")
    process(trajs.driving_by_plate, "driving")
    return kept, removals


def check_systemic_invariants(
    trajs: TrajectoriesResult,
    pickup_counts: dict,
    dropoff_counts: dict,
    profile_matrix: np.ndarray | None,
    n_drivers: int,
    expect_n_drivers: int = config.EXPECTED_N_DRIVERS,
) -> None:
    """Raise SystemicInvariantError on any systemic invariant failure."""
    total_pickups = sum(v[0] for v in pickup_counts.values())
    total_dropoffs = sum(v[1] for v in dropoff_counts.values())
    n_seeking = sum(len(v) for v in trajs.seeking_by_plate.values())
    n_driving = sum(len(v) for v in trajs.driving_by_plate.values())
    if total_pickups != n_seeking:
        raise SystemicInvariantError(
            f"#5: sum(pickup_counts)={total_pickups} != n_seeking={n_seeking}"
        )
    if total_dropoffs != n_driving:
        raise SystemicInvariantError(
            f"#5: sum(dropoff_counts)={total_dropoffs} != n_driving={n_driving}"
        )
    if n_drivers != expect_n_drivers:
        raise SystemicInvariantError(
            f"#6: got {n_drivers} unique drivers; expected {expect_n_drivers}"
        )
    if profile_matrix is not None:
        if profile_matrix.shape != (expect_n_drivers, config.N_PROFILE_FEATURES):
            raise SystemicInvariantError(
                f"#7: profile shape {profile_matrix.shape} != "
                f"({expect_n_drivers}, {config.N_PROFILE_FEATURES})"
            )
        if np.isnan(profile_matrix).any():
            raise SystemicInvariantError("#7: profile contains NaN")
        col_mean = profile_matrix.mean(axis=0)
        col_std = profile_matrix.std(axis=0, ddof=0)
        if not np.allclose(col_mean, 0.0, atol=1e-5):
            raise SystemicInvariantError(
                f"#7: profile column means not ~0: {col_mean}"
            )
        if not np.allclose(col_std, 1.0, atol=1e-5):
            raise SystemicInvariantError(
                f"#7: profile column stds not ~1: {col_std}"
            )
```

- [ ] **Step 11.4: Run tests — expect pass.** Commit.

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_invariants.py -v
git add famail_temporal/data/source_generation/removal.py \
        famail_temporal/data/source_generation/invariants.py \
        famail_temporal/data/source_generation/tests/test_invariants.py
git commit -m "feat(source_generation): add per-trajectory + systemic invariants"
```

---

### Task 12: Writer + metadata sidecar

**Files:**
- Create: `famail_temporal/data/source_generation/writer.py`
- Create: `famail_temporal/data/source_generation/tests/test_writer.py`

- [ ] **Step 12.1: Write the failing test**

Create `famail_temporal/data/source_generation/tests/test_writer.py`:

```python
"""Tests for writer.py."""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np

from famail_temporal.data.source_generation.writer import (
    write_all_outputs, write_active_taxis_bundle, write_metadata_json,
)
from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult,
)
from famail_temporal.data.source_generation.removal import (
    RemovalSummary, RemovalRecord,
)
from famail_temporal.data.source_generation import config


def test_writer_creates_all_files(tmp_path):
    pickup_dropoff = {(1, 1, 1, 1): (1, 0), (1, 1, 2, 1): (0, 1)}
    active_counts = {(1, 1, 0, 1): 1}
    trajs = TrajectoriesResult(
        seeking_by_plate={"A": [[[1, 1, 1, 1], [1, 1, 1, 1]]]},
        driving_by_plate={"A": [[[1, 1, 2, 1], [1, 1, 2, 1]]]},
    )
    mapping = {"plate_to_idx": {"A": 0}, "idx_to_plate": {0: "A"}}
    ms_seeking = {0: trajs.seeking_by_plate["A"]}
    ms_driving = {0: trajs.driving_by_plate["A"]}
    profile = {
        "normalized": np.zeros((1, 11), dtype=float),
        "mean": np.zeros(11), "std": np.ones(11),
        "feature_names": list(config.PROFILE_FEATURE_NAMES),
    }
    calendars = {"seeking": {0: [1]}, "driving": {0: [1]}}

    paths = write_all_outputs(
        out_dir=tmp_path,
        pickup_dropoff=pickup_dropoff,
        active_taxis=active_counts,
        passenger_seeking_trajs=trajs.seeking_by_plate,
        ms_seeking=ms_seeking,
        ms_driving=ms_driving,
        ms_profile=profile,
        ms_calendars=calendars,
        driver_mapping=mapping,
        removal_summary=RemovalSummary(),
        metadata_extras={
            "n_days": 3,
            "bounds": {"lat_min": 22.5, "lat_max": 22.9, "lon_min": 113.8, "lon_max": 114.5},
            "git_sha": "abc123",
            "config_snapshot": {},
        },
    )
    for p in [paths.pickup_dropoff, paths.active_taxis, paths.passenger_seeking,
              paths.ms_seeking, paths.ms_driving, paths.ms_profile,
              paths.ms_seeking_days, paths.ms_driving_days,
              paths.driver_mapping, paths.metadata]:
        assert p.exists(), f"missing file {p}"


def test_active_taxis_bundle_format(tmp_path):
    counts = {(1, 1, 0, 1): 5}
    path = tmp_path / "active_taxis_5x5_hourly.pkl"
    write_active_taxis_bundle(
        path, counts, stats={"n_entries": 1}, config_snapshot={"neighborhood_dims": 5},
    )
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    assert bundle["data"] == counts
    assert "stats" in bundle and "config" in bundle and "version" in bundle


def test_metadata_json_records_removals(tmp_path):
    summary = RemovalSummary(
        total_seeking_extracted=100, total_driving_extracted=100,
        removals=[RemovalRecord(
            driver_id="A", driver_idx=0, trajectory_index_within_driver=3,
            kind="seeking", which_invariant=1,
            failing_values={"endpoint": (99, 99, 1, 1)},
            n_states_before_removal=5, removal_reason_category="no_matching_count",
        )],
    )
    extras = {"n_days": 65, "bounds": {"lat_min": 22, "lat_max": 23, "lon_min": 113, "lon_max": 115}}
    path = tmp_path / "processing_metadata.json"
    write_metadata_json(path, summary, extras)
    with open(path) as f:
        m = json.load(f)
    assert m["n_days"] == 65
    assert m["removal_summary"]["total_extracted"] == 200
    assert m["removal_summary"]["removals"][0]["removal_reason_category"] == "no_matching_count"
    assert m["removal_summary"]["counts_by_category"]["no_matching_count"] == 1
```

- [ ] **Step 12.2: Run tests — expect failure.**

- [ ] **Step 12.3: Implement `writer.py`**

Create `famail_temporal/data/source_generation/writer.py`:

```python
"""Write all tool outputs: 8 output files + driver mapping + metadata JSON."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import pickle

from famail_temporal.data.source_generation import config
from famail_temporal.data.source_generation.removal import RemovalSummary


@dataclass(frozen=True)
class OutputPaths:
    pickup_dropoff: Path
    active_taxis: Path
    passenger_seeking: Path
    ms_seeking: Path
    ms_driving: Path
    ms_profile: Path
    ms_seeking_days: Path
    ms_driving_days: Path
    driver_mapping: Path
    metadata: Path


def _pickle_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def write_active_taxis_bundle(
    path: Path, counts: dict, stats: dict, config_snapshot: dict,
) -> None:
    bundle = {
        "data": counts,
        "stats": stats,
        "config": config_snapshot,
        "version": config.OUTPUT_FORMAT_VERSION,
    }
    _pickle_write(path, bundle)


def write_profile_bundle(
    path: Path, normalized, mean, std, feature_names: list[str],
    n_features: int, drivers_mapping: dict,
) -> None:
    features = {
        int(idx): normalized[int(idx)].astype(float)
        for idx in drivers_mapping["idx_to_plate"].keys()
    }
    bundle = {
        "features": features,
        "features_normalized": features,
        "feature_names": feature_names,
        "normalization": {"mean": mean.astype(float), "std": std.astype(float)},
        "n_features": n_features,
    }
    _pickle_write(path, bundle)


def write_metadata_json(
    path: Path, removal_summary: RemovalSummary, extras: dict,
) -> None:
    removals_dict_list = [r.to_dict() for r in removal_summary.removals]
    metadata = dict(extras)
    metadata["removal_summary"] = {
        "total_seeking_extracted": removal_summary.total_seeking_extracted,
        "total_driving_extracted": removal_summary.total_driving_extracted,
        "total_extracted": removal_summary.total_extracted(),
        "n_removed": len(removal_summary.removals),
        "removal_rate": removal_summary.removal_rate(),
        "counts_by_category": removal_summary.counts_by_category(),
        "removals": removals_dict_list,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def write_all_outputs(
    out_dir: Path,
    pickup_dropoff: dict,
    active_taxis: dict,
    passenger_seeking_trajs: dict,
    ms_seeking: dict,
    ms_driving: dict,
    ms_profile: dict,
    ms_calendars: dict,
    driver_mapping: dict,
    removal_summary: RemovalSummary,
    metadata_extras: dict,
) -> OutputPaths:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = OutputPaths(
        pickup_dropoff=out_dir / config.OUT_PICKUP_DROPOFF,
        active_taxis=out_dir / config.OUT_ACTIVE_TAXIS,
        passenger_seeking=out_dir / config.OUT_PASSENGER_SEEKING,
        ms_seeking=out_dir / config.OUT_MS_SEEKING,
        ms_driving=out_dir / config.OUT_MS_DRIVING,
        ms_profile=out_dir / config.OUT_MS_PROFILE,
        ms_seeking_days=out_dir / config.OUT_MS_SEEKING_DAYS,
        ms_driving_days=out_dir / config.OUT_MS_DRIVING_DAYS,
        driver_mapping=out_dir / config.OUT_DRIVER_MAPPING,
        metadata=out_dir / config.OUT_METADATA,
    )
    _pickle_write(paths.pickup_dropoff, pickup_dropoff)
    write_active_taxis_bundle(
        paths.active_taxis, active_taxis,
        stats={"n_entries": len(active_taxis)},
        config_snapshot={
            "neighborhood_dims": config.NEIGHBORHOOD_SIZE,
            "period_type": "hourly",
        },
    )
    _pickle_write(paths.passenger_seeking, passenger_seeking_trajs)
    _pickle_write(paths.ms_seeking, ms_seeking)
    _pickle_write(paths.ms_driving, ms_driving)
    write_profile_bundle(
        paths.ms_profile,
        ms_profile["normalized"], ms_profile["mean"], ms_profile["std"],
        ms_profile["feature_names"], config.N_PROFILE_FEATURES, driver_mapping,
    )
    _pickle_write(paths.ms_seeking_days, ms_calendars["seeking"])
    _pickle_write(paths.ms_driving_days, ms_calendars["driving"])
    _pickle_write(paths.driver_mapping, driver_mapping)
    write_metadata_json(paths.metadata, removal_summary, metadata_extras)
    return paths
```

- [ ] **Step 12.4: Run tests — expect pass.** Commit.

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_writer.py -v
git add famail_temporal/data/source_generation/writer.py \
        famail_temporal/data/source_generation/tests/test_writer.py
git commit -m "feat(source_generation): add output writer + metadata sidecar"
```

---

### Task 13: CLI entry point

**Files:**
- Create: `famail_temporal/data/source_generation/cli.py`
- Create: `famail_temporal/data/source_generation/__main__.py`
- Create: `famail_temporal/data/source_generation/tests/test_cli.py`

- [ ] **Step 13.1: Write the failing test**

Create `famail_temporal/data/source_generation/tests/test_cli.py`:

```python
"""Tests for the CLI orchestration (run_generation)."""
from __future__ import annotations
import pickle
from pathlib import Path


from famail_temporal.data.source_generation.cli import run_generation


def _write_pkl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _minimal_raw_fixture(tmp_path):
    raw = tmp_path / "raw"
    for filename in ("taxi_record_07_50drivers.pkl",
                     "taxi_record_08_50drivers.pkl",
                     "taxi_record_09_50drivers.pkl"):
        _write_pkl(raw / filename, {})
    records: dict = {}
    for i in range(50):
        plate = f"PLATE_{i:02d}"
        records[plate] = [
            [plate, 22.5, 113.8, 0,    1, "2016-07-04 00:00:00"],
            [plate, 22.5, 113.8, 60,   0, "2016-07-04 00:01:00"],
            [plate, 22.5, 113.8, 120,  0, "2016-07-04 00:02:00"],
            [plate, 22.5, 113.8, 180,  0, "2016-07-04 00:03:00"],
            [plate, 22.5, 113.8, 240,  1, "2016-07-04 00:04:00"],
            [plate, 22.5, 113.8, 300,  1, "2016-07-04 00:05:00"],
            [plate, 22.5, 113.8, 360,  0, "2016-07-04 00:06:00"],
        ]
    _write_pkl(raw / "taxi_record_07_50drivers.pkl", records)
    return raw


def test_cli_runs_end_to_end(tmp_path):
    raw = _minimal_raw_fixture(tmp_path)
    out = tmp_path / "out"
    result = run_generation(input_dir=raw, output_dir=out)
    expected = [
        "pickup_dropoff_counts.pkl",
        "active_taxis_5x5_hourly.pkl",
        "passenger_seeking_trajs.pkl",
        "ms_seeking_trajs.pkl",
        "ms_driving_trajs.pkl",
        "ms_profile_features.pkl",
        "ms_seeking_calendar_days.pkl",
        "ms_driving_calendar_days.pkl",
        "driver_index_mapping.pkl",
        "processing_metadata.json",
    ]
    for name in expected:
        assert (out / name).exists(), f"missing output: {name}"
    assert result.n_seeking_kept >= 1
    assert result.n_driving_kept >= 1
```

- [ ] **Step 13.2: Run test — expect failure.**

- [ ] **Step 13.3: Implement `cli.py`**

Create `famail_temporal/data/source_generation/cli.py`:

```python
"""Orchestrate the full source-data generation pipeline."""
from __future__ import annotations
import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from famail_temporal.data.source_generation import config
from famail_temporal.data.source_generation.event_stream import build_event_stream
from famail_temporal.data.source_generation.invariants import (
    apply_per_trajectory_invariants, check_systemic_invariants,
)
from famail_temporal.data.source_generation.removal import RemovalSummary
from famail_temporal.data.source_generation.views.active_taxis import (
    build_active_taxis_counts,
)
from famail_temporal.data.source_generation.views.calendars import (
    build_calendar_days_per_driver,
)
from famail_temporal.data.source_generation.views.pickup_dropoff import (
    build_pickup_dropoff_counts,
)
from famail_temporal.data.source_generation.views.profile import (
    compute_profile_features, zscore_normalize,
)
from famail_temporal.data.source_generation.views.trajectories import (
    build_driver_index_mapping, build_trajectories,
)
from famail_temporal.data.source_generation.writer import (
    OutputPaths, write_all_outputs,
)


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    paths: OutputPaths
    n_seeking_kept: int
    n_driving_kept: int
    n_removals: int


def _git_sha_or_none() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


def run_generation(
    input_dir: Path,
    output_dir: Path,
    expect_n_drivers: int | None = None,
) -> RunResult:
    """Run the full pipeline end-to-end.

    expect_n_drivers: override to relax the driver-count invariant
    (useful for testing with fewer than 50 synthetic drivers).
    """
    expect_n_drivers = expect_n_drivers or config.EXPECTED_N_DRIVERS
    log.info("Building event stream from %s", input_dir)
    es = build_event_stream(Path(input_dir))

    log.info("Building views…")
    pickup_dropoff_raw = build_pickup_dropoff_counts(es.df)
    active_taxis = build_active_taxis_counts(es.df)
    trajs = build_trajectories(es.df)
    mapping = build_driver_index_mapping(es.df)

    n_seeking_extracted = sum(len(v) for v in trajs.seeking_by_plate.values())
    n_driving_extracted = sum(len(v) for v in trajs.driving_by_plate.values())
    log.info("Extracted %d seeking + %d driving trajectories",
             n_seeking_extracted, n_driving_extracted)

    pickup_only = {k: (v[0], 0) for k, v in pickup_dropoff_raw.items()}
    dropoff_only = {k: (0, v[1]) for k, v in pickup_dropoff_raw.items()}

    log.info("Applying per-trajectory invariants…")
    kept_trajs, removals = apply_per_trajectory_invariants(
        trajs, pickup_only, dropoff_only,
        plate_to_idx=mapping["plate_to_idx"],
    )

    # Rebuild pickup/dropoff counts from surviving trajectory endpoints so
    # systemic invariant #5 (sum(counts) == n_trajectories) holds after
    # per-trajectory removals.
    pickup_dropoff_final: dict = {}
    for traj_list in kept_trajs.seeking_by_plate.values():
        for t in traj_list:
            key = tuple(t[-1])
            p, d = pickup_dropoff_final.get(key, (0, 0))
            pickup_dropoff_final[key] = (p + 1, d)
    for traj_list in kept_trajs.driving_by_plate.values():
        for t in traj_list:
            key = tuple(t[-1])
            p, d = pickup_dropoff_final.get(key, (0, 0))
            pickup_dropoff_final[key] = (p, d + 1)

    removal_summary = RemovalSummary(
        total_seeking_extracted=n_seeking_extracted,
        total_driving_extracted=n_driving_extracted,
        removals=removals,
    )
    if removal_summary.removal_rate() > config.REMOVAL_RATE_WARN_THRESHOLD:
        log.warning(
            "Per-trajectory removal rate %.2f%% exceeds threshold %.2f%%",
            100 * removal_summary.removal_rate(),
            100 * config.REMOVAL_RATE_WARN_THRESHOLD,
        )

    log.info("Computing profile features…")
    raw_features = compute_profile_features(es.df, kept_trajs)
    n_drivers_actual = len(mapping["plate_to_idx"])
    ordered_plates = [mapping["idx_to_plate"][i] for i in range(n_drivers_actual)]
    raw_matrix = np.array([
        [raw_features[p][f] for f in config.PROFILE_FEATURE_NAMES]
        for p in ordered_plates
    ], dtype=float)
    normalized, mean, std = zscore_normalize(raw_matrix)

    log.info("Checking systemic invariants…")
    pickup_for_check = {k: (v[0], 0) for k, v in pickup_dropoff_final.items()}
    dropoff_for_check = {k: (0, v[1]) for k, v in pickup_dropoff_final.items()}
    check_systemic_invariants(
        kept_trajs, pickup_for_check, dropoff_for_check,
        profile_matrix=normalized if n_drivers_actual == expect_n_drivers else None,
        n_drivers=n_drivers_actual,
        expect_n_drivers=expect_n_drivers,
    )

    log.info("Writing outputs to %s", output_dir)
    ms_seeking = {
        mapping["plate_to_idx"][p]: kept_trajs.seeking_by_plate.get(p, [])
        for p in mapping["plate_to_idx"].keys()
    }
    ms_driving = {
        mapping["plate_to_idx"][p]: kept_trajs.driving_by_plate.get(p, [])
        for p in mapping["plate_to_idx"].keys()
    }
    ms_calendars = build_calendar_days_per_driver(kept_trajs, mapping)
    ms_profile_payload = {
        "normalized": normalized,
        "mean": mean,
        "std": std,
        "feature_names": list(config.PROFILE_FEATURE_NAMES),
    }
    paths = write_all_outputs(
        out_dir=Path(output_dir),
        pickup_dropoff=pickup_dropoff_final,
        active_taxis=active_taxis,
        passenger_seeking_trajs=kept_trajs.seeking_by_plate,
        ms_seeking=ms_seeking,
        ms_driving=ms_driving,
        ms_profile=ms_profile_payload,
        ms_calendars=ms_calendars,
        driver_mapping=mapping,
        removal_summary=removal_summary,
        metadata_extras={
            "n_days": es.n_days,
            "bounds": {
                "lat_min": es.bounds.lat_min, "lat_max": es.bounds.lat_max,
                "lon_min": es.bounds.lon_min, "lon_max": es.bounds.lon_max,
            },
            "git_sha": _git_sha_or_none(),
            "config_snapshot": {
                "GRID_SIZE_DEG": config.GRID_SIZE_DEG,
                "NEIGHBORHOOD_SIZE": config.NEIGHBORHOOD_SIZE,
                "TIME_INTERVAL_MIN": config.TIME_INTERVAL_MIN,
                "WEEKDAY_DAYS": sorted(config.WEEKDAY_DAYS),
            },
        },
    )
    return RunResult(
        paths=paths,
        n_seeking_kept=sum(len(v) for v in kept_trajs.seeking_by_plate.values()),
        n_driving_kept=sum(len(v) for v in kept_trajs.driving_by_plate.values()),
        n_removals=len(removals),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="famail_temporal.data.source_generation",
        description="Unified GPS source-data generation for famail_temporal.",
    )
    p.add_argument("--input-dir", type=Path, default=config.DEFAULT_RAW_INPUT_DIR,
                   help="Directory containing the 3 taxi_record_*.pkl files.")
    p.add_argument("--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR,
                   help="Directory to write the 10 output files.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    result = run_generation(args.input_dir, args.output_dir)
    log.info(
        "Done: %d seeking + %d driving kept; %d removals; outputs at %s",
        result.n_seeking_kept, result.n_driving_kept,
        result.n_removals, args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 13.4: Write `__main__.py`**

Create `famail_temporal/data/source_generation/__main__.py`:

```python
"""Allow `python -m famail_temporal.data.source_generation`."""
import sys
from famail_temporal.data.source_generation.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 13.5: Run tests — expect pass**

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_cli.py -v
```

- [ ] **Step 13.6: Sanity-check the CLI invocation**

```bash
.venv/bin/python -m famail_temporal.data.source_generation --help
```

Expected: argparse-generated help for the tool.

- [ ] **Step 13.7: Commit**

```bash
git add famail_temporal/data/source_generation/cli.py \
        famail_temporal/data/source_generation/__main__.py \
        famail_temporal/data/source_generation/tests/test_cli.py
git commit -m "feat(source_generation): add CLI orchestration + entry point"
```

### Phase 4 checkpoint

- [ ] Run all tests in `famail_temporal/data/source_generation/tests/`.
- [ ] Dispatch code-review subagent (Opus, `superpowers:code-reviewer`) on Phase 4 files.
- [ ] Dispatch simplifier if reviewer flags verbosity.

---

## Phase 5 — Golden test + smoke test

### Task 14: Golden end-to-end test

**Files:**
- Create: `famail_temporal/data/source_generation/tests/golden_fixtures.py`
- Create: `famail_temporal/data/source_generation/tests/test_golden.py`

- [ ] **Step 14.1: Write the golden fixture**

Create `famail_temporal/data/source_generation/tests/golden_fixtures.py`:

```python
"""Hand-built synthetic raw-GPS fixture + expected outputs.

Two drivers × a handful of weekday records, all expected outputs hand-
computed in this file. Referenced from test_golden.py. When future changes
appear to alter output numerics, diff against this fixture's answers.
"""
from __future__ import annotations
import pickle
from pathlib import Path


def build_raw_fixture(output_dir: Path) -> None:
    data_07: dict = {
        "AAA": [
            ["AAA", 22.500, 113.800, 0,     1, "2016-07-04 00:00:00"],
            ["AAA", 22.500, 113.800, 60,    1, "2016-07-04 00:01:00"],
            ["AAA", 22.500, 113.800, 120,   0, "2016-07-04 00:02:00"],
            ["AAA", 22.500, 113.800, 180,   0, "2016-07-04 00:03:00"],
            ["AAA", 22.500, 113.800, 240,   0, "2016-07-04 00:04:00"],
            ["AAA", 22.500, 113.810, 300,   0, "2016-07-04 00:05:00"],
            ["AAA", 22.500, 113.810, 360,   1, "2016-07-04 00:06:00"],
            ["AAA", 22.500, 113.810, 420,   1, "2016-07-04 00:07:00"],
            ["AAA", 22.500, 113.810, 480,   0, "2016-07-04 00:08:00"],
        ],
        "BBB": [
            ["BBB", 22.600, 114.000, 0,     0, "2016-07-04 00:00:00"],
            ["BBB", 22.600, 114.000, 60,    1, "2016-07-04 00:01:00"],
            ["BBB", 22.600, 114.000, 120,   0, "2016-07-04 00:02:00"],
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "taxi_record_07_50drivers.pkl", "wb") as f:
        pickle.dump(data_07, f)
    with open(output_dir / "taxi_record_08_50drivers.pkl", "wb") as f:
        pickle.dump({}, f)
    with open(output_dir / "taxi_record_09_50drivers.pkl", "wb") as f:
        pickle.dump({}, f)


def expected_seeking_trajectories() -> dict[str, list]:
    return {
        "AAA": [
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 2, 1, 1],
                [1, 2, 2, 1],
            ],
        ],
    }


def expected_pickup_count_at_AAA_endpoint() -> dict:
    return {(1, 2, 2, 1): (1, 0)}
```

- [ ] **Step 14.2: Write `test_golden.py`**

Create `famail_temporal/data/source_generation/tests/test_golden.py`:

```python
"""End-to-end golden test on a hand-built fixture + slow real-data smoke test."""
from __future__ import annotations
import pickle
from pathlib import Path

import pytest

from famail_temporal.data.source_generation.cli import run_generation
from famail_temporal.data.source_generation.tests.golden_fixtures import (
    build_raw_fixture, expected_seeking_trajectories,
    expected_pickup_count_at_AAA_endpoint,
)


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_golden_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    build_raw_fixture(raw)

    result = run_generation(raw, out, expect_n_drivers=2)

    seeking = _load_pkl(out / "passenger_seeking_trajs.pkl")
    expected = expected_seeking_trajectories()
    assert set(seeking.keys()) == set(expected.keys())
    for plate, trajs in expected.items():
        assert seeking[plate] == trajs

    pd_counts = _load_pkl(out / "pickup_dropoff_counts.pkl")
    expected_pick = expected_pickup_count_at_AAA_endpoint()
    for key, value in expected_pick.items():
        assert key in pd_counts
        assert pd_counts[key][0] == value[0]

    # Invariant check: every seeking trajectory's endpoint has count >= 1.
    for plate, trajs in seeking.items():
        for traj in trajs:
            key = tuple(traj[-1])
            p, _ = pd_counts.get(key, (0, 0))
            assert p >= 1, f"endpoint {key} missing from pickup_counts"

    # Systemic invariant #5: total pickups == total seeking trajectories.
    total_pickups = sum(v[0] for v in pd_counts.values())
    total_seeking = sum(len(v) for v in seeking.values())
    assert total_pickups == total_seeking


@pytest.mark.slow
def test_smoke_on_real_raw_if_present(tmp_path):
    """Run on real raw GPS data if present under raw_data/; skip otherwise."""
    real_raw = Path("raw_data")
    required = [
        "taxi_record_07_50drivers.pkl",
        "taxi_record_08_50drivers.pkl",
        "taxi_record_09_50drivers.pkl",
    ]
    for name in required:
        if not (real_raw / name).exists():
            pytest.skip(f"Missing real raw file: {real_raw / name}")

    out = tmp_path / "smoke_out"
    result = run_generation(real_raw, out, expect_n_drivers=50)
    assert result.n_seeking_kept >= 100
    assert result.n_driving_kept >= 100
    for name in [
        "pickup_dropoff_counts.pkl", "active_taxis_5x5_hourly.pkl",
        "passenger_seeking_trajs.pkl", "ms_seeking_trajs.pkl",
        "ms_driving_trajs.pkl", "ms_profile_features.pkl",
        "ms_seeking_calendar_days.pkl", "ms_driving_calendar_days.pkl",
        "driver_index_mapping.pkl", "processing_metadata.json",
    ]:
        assert (out / name).exists()
```

- [ ] **Step 14.3: Run golden test — expect pass**

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_golden.py::test_golden_end_to_end -v
```

- [ ] **Step 14.4: Run the slow smoke test (skips if raw data not present)**

```bash
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_golden.py::test_smoke_on_real_raw_if_present --run-slow -v
```

Expected (no raw data): SKIPPED.
Expected (with raw data): PASS.

- [ ] **Step 14.5: Commit**

```bash
git add famail_temporal/data/source_generation/tests/golden_fixtures.py \
        famail_temporal/data/source_generation/tests/test_golden.py
git commit -m "test(source_generation): add golden + slow smoke tests"
```

### Phase 5 checkpoint

- [ ] Run the full source_generation test suite:
  ```bash
  .venv/bin/pytest famail_temporal/data/source_generation/tests/ -v
  ```
- [ ] Dispatch code-review subagent (Opus, `superpowers:code-reviewer`) on the whole package.

---

## Phase 6 — Integration with famail_temporal

### Task 15: Update loader filename reference

**Files:**
- Modify: `famail_temporal/data/loader.py:95`

- [ ] **Step 15.1: Update the filename reference**

Edit [`famail_temporal/data/loader.py:95`](famail_temporal/data/loader.py#L95):

```python
# Before:
    path = config.RAW_DATA_DIR / "passenger_seeking_trajs_45-800.pkl"

# After:
    path = config.RAW_DATA_DIR / "passenger_seeking_trajs.pkl"
```

- [ ] **Step 15.2: Run existing loader tests**

```bash
.venv/bin/pytest famail_temporal/tests/test_data_loader.py -v
```

If any test relies on the old filename, update it. If the old raw file still exists and the new one does not, create a symlink to unblock:

```bash
# Only if old raw file exists and new one doesn't (local dev convenience):
[ -e famail_temporal/raw_data/passenger_seeking_trajs_45-800.pkl ] \
  && [ ! -e famail_temporal/raw_data/passenger_seeking_trajs.pkl ] \
  && ln -s passenger_seeking_trajs_45-800.pkl famail_temporal/raw_data/passenger_seeking_trajs.pkl
```

- [ ] **Step 15.3: Commit**

```bash
git add famail_temporal/data/loader.py
git commit -m "chore(data): rename passenger_seeking_trajs.pkl (drop _45-800 suffix)"
```

---

### Task 16: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 16.1: Append a dated CHANGELOG entry**

At the top of `CHANGELOG.md`, add:

```markdown
## 2026-04-20 — Unified source-data generation tool

Added `famail_temporal/data/source_generation/` — a unified tool that
generates all 8 GPS-derived source datasets from raw taxi_record_*.pkl.
Replaces the legacy `pickup_dropoff_counts/`, `active_taxis/`, and
`new_all_trajs/` tools with a single pipeline whose cross-file consistency
holds by construction.

**Semantic changes (results from before this commit are NOT directly
comparable to results from after):**

- `active_taxis` definition changed from "any driver present in 5×5 neighborhood"
  to "driver with at least one empty (passenger=0) ping in 5×5 neighborhood."
  F_spatial's DSR denominator now represents service-capacity rather than
  traffic-presence.
- Each seeking trajectory's `states[-1]` is now the pickup-transition record
  (first passenger=1 ping), not the last seeking ping. The modifier's
  mass-balance invariant (`pickup_3d[states[-1].cell] >= 1`) now holds by
  construction.
- Day filter unified to weekdays-only (Mon-Fri, day_index ∈ {1..5}) across
  all 8 output files. Saturday records are no longer included in
  `pickup_dropoff_counts`.
- Profile feature `home_x/y` redefined as mode-of-cell at `time_bucket == 1`
  (midnight), not mode-of-trajectory-start-cell.
- Profile features `shift_start`/`shift_end` redefined as 5th/95th percentile
  (previously min/max).

**Required operator action after pulling this change:**
1. Regenerate source data:
   `python -m famail_temporal.data.source_generation --input-dir raw_data/ --output-dir famail_temporal/raw_data/`
2. Regenerate preprocess cache:
   `python -m famail_temporal.preprocess --force`
3. Retrain the v3 discriminator on the new multi-stream files before running
   experiments with F_fidelity enabled.
```

- [ ] **Step 16.2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record unified source-data generation + semantic changes"
```

### Phase 6 checkpoint

- [ ] Dispatch final code-review subagent (Opus, `superpowers:code-reviewer`) on the complete changeset diff from the starting branch.
- [ ] Address blocking review items; re-review as needed.
- [ ] Confirm all tests still green:
  ```bash
  .venv/bin/pytest famail_temporal/data/source_generation/tests/ -v
  .venv/bin/pytest famail_temporal/tests/ -v
  ```

---

## Done criteria

- [ ] All 8 output files + 2 sidecars are produced under the output directory when the tool runs on the 3 real `taxi_record_*.pkl` files.
- [ ] All unit tests and the golden test pass; the `--run-slow` smoke test passes when real raw data is present.
- [ ] `famail_temporal/data/loader.py` loads `passenger_seeking_trajs.pkl` (new filename).
- [ ] `famail_temporal.preprocess --force` runs to completion with the new inputs.
- [ ] `python -m famail_temporal.evaluation.runner --max-trajectories 200 -k 5 --override MAX_ITERATIONS=5` runs to completion and produces a `report.md`.
- [ ] All code-review checkpoints cleared.
- [ ] CHANGELOG entry committed.

---

End of plan.
