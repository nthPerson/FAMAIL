# `views/` — Per-output-file view modules

## Purpose

Each view is a **pure function of the enriched event-stream DataFrame** (produced by
`event_stream.py::build_event_stream`). A view reads pre-quantized columns and produces exactly
one output artifact's in-memory representation; it never re-quantizes, never re-filters days,
never re-detects transitions. This is what makes cross-file consistency hold by construction:
since every view sees the same DataFrame, they can't disagree on coordinate, time, or day
conventions.

Views never talk to `writer.py` directly — that's the CLI orchestrator's job. A view returns a
dict (or dataclass) and the CLI threads that into `write_all_outputs`.

---

## Files

| File | Output file produced | Return shape | Purpose |
|---|---|---|---|
| `pickup_dropoff.py` | `pickup_dropoff_counts.pkl` | `dict[(x, y, tb, day)] -> (pickup, dropoff)` | Groups pickup and dropoff events per `(cell, time_bucket, day)` key. Sparse — only cells with at least one event appear. |
| `active_taxis.py` | `active_taxis_5x5_hourly.pkl` | `dict[(x, y, hour, day)] -> int` | Counts unique drivers who had ≥1 **empty** (`passenger_indicator == 0`) GPS ping in the 5×5 neighborhood centered at `(x, y)` during `(hour, day)`. The neighborhood expansion is explicit (25 shifted copies, dedup per-driver at both ends). |
| `trajectories.py` | `passenger_seeking_trajs.pkl`, `ms_seeking_trajs.pkl`, `ms_driving_trajs.pkl`, `driver_index_mapping.pkl` | `TrajectoriesResult` dataclass + `{plate_to_idx, idx_to_plate}` | Walks per-driver segments; classifies each as seeking (ends at pickup), driving (ends at dropoff), or incomplete (dropped). Also produces the lexicographic plate↔int driver-index mapping. |
| `profile.py` | `ms_profile_features.pkl` | `dict[plate_id] -> {11 feature name → value, fallback_used}` + `zscore_normalize` helper | Computes the 11-dim driver profile vector (home_x/y with fallback cascade, 5th/95th percentile shift_start/end, mode freq_grid_x/y, avg seek/drive dist/time, trips per day). Exports `zscore_normalize` for the CLI to apply across all 50 drivers after collection. |
| `calendars.py` | `ms_seeking_calendar_days.pkl`, `ms_driving_calendar_days.pkl` | `{"seeking": dict[int idx] -> list[int day], "driving": dict[int idx] -> list[int day]}` | Sorted unique day-indices on which each driver has at least one seeking/driving trajectory. Reserved for forward-compatibility (loaded by famail_temporal but not currently consumed). |

---

## Design principles shared across all views

1. **Pure functions.** Every view takes the event-stream DataFrame (and optionally the `TrajectoriesResult` + driver mapping) as argument and returns a new value. No side effects, no filesystem writes.

2. **Small and focused** — each view is 20-150 lines. Each can be read, tested, and reasoned about independently. When a view grows beyond that, it's usually a signal to factor out a helper into a sibling module (e.g., profile's fallback cascade lives in the same file but as a separate public function, `compute_home_xy_with_fallback`, so it can be unit-tested in isolation).

3. **1-indexed on disk** — all grid coordinates (`x`, `y`) and time buckets (`tb`) in view outputs are **1-indexed** (matching the schemas already consumed by `famail_temporal/data/loader.py` and `famail_temporal/data/aggregation.py`). Hour (in active_taxis) is **0-indexed**. This matches the conventions in [`source_generation/config.py`](../config.py).

4. **Python-native int in dict keys.** Every dict key in a view output uses `int(...)` on all numeric components so the serialized artifact holds Python ints, not `numpy.int64`. Consumers that compare against plain Python int tuples get the equality they expect.

5. **`sort=False` on `groupby`** — output is an unordered dict, so sorting the group keys during the groupby is wasted work.

---

## Consumer expectations (what the rest of `famail_temporal/` needs)

- [`famail_temporal/preprocess.py`](../../../preprocess.py) unwraps `active_taxis_raw['data']` from the active_taxis bundle → `write_active_taxis_bundle` emits that shape.
- [`famail_temporal/data/loader.py`](../../loader.py) reads `profile_raw["features_normalized"]` (already z-scored). The discriminator's dataset-generation pipeline (in the parent monorepo) reads `profile_raw["features"]` (raw vectors, then applies z-score itself using the stored `mean` / `std`) → `write_profile_bundle` emits `features` as RAW vectors and `features_normalized` as NORMALIZED vectors. The two are **not aliases**; conflating them causes double-normalization.
- [`famail_temporal/data/loader.py::_load_multi_stream`](../../loader.py) casts multi-stream dict keys to `int(k)` → `trajectories.py` produces `ms_seeking_trajs` and `ms_driving_trajs` with int driver_idx keys directly.
- [`famail_temporal/data/loader.py`](../../loader.py) reads `passenger_seeking_trajs.pkl` → `trajectories.py` produces this file with plate_id string keys.

Contract verified by the golden test in [`../tests/test_golden.py`](../tests/test_golden.py).

---

## Dependencies

- `famail_temporal.data.source_generation.config` (constants only)
- `pandas`, `numpy`
- `famail_temporal.data.source_generation.views.trajectories` is imported by `profile.py` and `calendars.py` for the `TrajectoriesResult` / `Trajectory` type aliases
- No imports from outside `famail_temporal/data/source_generation/`
