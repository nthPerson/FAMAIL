# `source_generation/` — Unified raw-GPS to source-dataset generation

## Purpose

One tool, one pass over the 3 raw taxi GPS pickle files
(`raw_data/taxi_record_{07,08,09}_50drivers.pkl`), produces all 8 source datasets +
a driver-index mapping + a processing-metadata JSON that the rest of `famail_temporal/` consumes.
Replaces three independent legacy tools (`pickup_dropoff_counts/`, `active_taxis/`,
`new_all_trajs/`) with a single pipeline whose **cross-file consistency holds by construction**:
every output derives from one enriched event-stream DataFrame produced in one pass.

The design problem this solves: the three legacy tools disagreed on the
`time_bucket` offset (0-indexed vs 1-indexed), on the weekend filter (drop Sat+Sun vs drop Sun only),
and on the pickup-cell semantic (last seeking GPS ping vs first passenger=1 ping). Those disagreements
manifested as a runtime error: `pickup_3d[states[-1].cell]` was zero for ~23% of trajectories, making
the modifier's mass-balance bookkeeping underflow. See [`docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md`](../../../docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md) for the full history.

> **Serialization note.** The tool both reads and writes `.pkl` files. Inputs are the project's
> own raw taxi GPS files; outputs are the 8 downstream files consumed by `famail_temporal.preprocess`
> and `famail_temporal.data.loader`. Pickle I/O is only performed against paths the project itself
> writes — never against files from external or untrusted sources.

---

## Files

| File | Role |
|---|---|
| `__main__.py` | Makes `python -m famail_temporal.data.source_generation` work |
| `cli.py` | `run_generation(input_dir, output_dir, expect_n_drivers=None) -> RunResult` + argparse entry. Orchestrates the whole pipeline. |
| `config.py` | Opinionated constants: grid (48×90, 0.01°), time (288 buckets of 5 min, 1-indexed), hour (0-indexed), weekday-only day filter (days 1..5), 5×5 neighborhood, 11 profile feature names, 5% removal-rate warn threshold. No runtime config flags. |
| `raw_loader.py` | `load_raw_file(path)`, `concat_raw_records(paths)` — reads one/many `taxi_record_*.pkl` files (handles both flat and nested day-list shapes) into a pandas DataFrame with typed columns. |
| `quantization.py` | The **single** authoritative definitions of `gps_to_grid`, `seconds_to_time_bucket`, `seconds_to_hour`, `timestamp_to_day`, plus the `GlobalBounds` dataclass. Every view reads pre-quantized columns; no view re-quantizes. |
| `transitions.py` | `add_transition_columns` detects per-driver 0→1 (pickup) and 1→0 (dropoff) transitions via `groupby.diff()`. `assign_segment_ids` gives each row a per-driver segment_id such that transition rows are the LAST row of their segment. |
| `event_stream.py` | `build_event_stream(raw_dir) -> EventStream`. Orchestrates load → quantize → weekday-filter → sort → transition-detect → segment-assign into the single enriched DataFrame. Returns the DataFrame plus metadata (`GlobalBounds`, `n_days`, per-driver calendar-day sets). |
| `views/` | Five deterministic view modules; see [`views/README.md`](views/README.md). Each is a pure function of the event-stream DataFrame. |
| `removal.py` | `RemovalRecord` + `RemovalSummary` dataclasses. One record per dropped trajectory with full diagnostic payload (driver, which invariant, failing values, state count). |
| `invariants.py` | `apply_per_trajectory_invariants` drops offenders and records them; `check_systemic_invariants` aborts with `SystemicInvariantError`. See "Invariants" below. |
| `writer.py` | `write_all_outputs(...)` serializes all 10 artifacts. `write_active_taxis_bundle` uses the legacy `{data, stats, config, version}` wrapper expected by `preprocess.py`. `write_profile_bundle` emits both `features` and `features_normalized` keys for loader compatibility. `write_metadata_json` emits the removal summary + audit extras. |
| `tests/` | TDD unit tests per module + golden end-to-end + slow smoke test. See [`tests/README.md`](tests/README.md). |

---

## Architecture

**Principle: one pass over raw GPS → enriched event-stream DataFrame (single source of truth) → N deterministic views, each producing one output file.**

```
raw_data/taxi_record_{07,08,09}_50drivers.pkl
                  │
                  ▼
       ┌────────────────────────────┐
       │  Ingestion + Enrichment    │   (single pass, event_stream.py)
       ├────────────────────────────┤
       │  1. Load + concat          │   (raw_loader.py)
       │  2. Global GPS bounds      │   (quantization.py)
       │  3. Quantize:              │
       │     - gps → (x, y) 1-idx   │
       │     - seconds → time_bucket│
       │     - timestamp → day_idx  │
       │     - drop weekends        │
       │  4. Sort (plate_id, ts)    │
       │  5. Detect transitions     │   (transitions.py)
       │  6. Assign segment_id      │
       └──────────────┬─────────────┘
                      │
                      ▼
         ENRICHED EVENT STREAM
         (single pandas DataFrame)
                      │
  ┌──────┬────────┬───┴────┬───────┬──────────┬──────────┐
  ▼      ▼        ▼        ▼       ▼          ▼          ▼
[pu_do][active] [trajs] [profile][calend.] [mapping] [metadata]
 view   view    view     view    view      view      view
  │      │       │         │       │         │         │
  ▼      ▼       ▼         ▼       ▼         ▼         ▼
pickup_ active_ passenger_ms_     ms_{se,   driver_   metadata.
dropoff taxis_  seeking_   profile dr}_cal_ index_    json
counts  5x5_    trajs +    featur  days     mapping
.pkl    hourly  ms_{se,dr} es.pkl  .pkl     .pkl
        .pkl    }_trajs.pkl
                    │
                    ▼
         (per-trajectory invariants applied in cli.py;
          counts re-derived from survivors so
          systemic #5 holds by construction)
```

---

## Key design choices

### 1. Single source of truth — the enriched event-stream DataFrame

`event_stream.py::build_event_stream` is the only place in the whole package where the raw GPS
data meets the quantization primitives. Every view reads pre-computed columns (`x_grid`,
`y_grid`, `time_bucket`, `hour`, `day_index`, `is_pickup`, `is_dropoff`, `segment_id`, ...);
no view re-quantizes, re-filters, or re-detects transitions. Cross-file coordinate drift,
time-bucket drift, and day-filter drift are all impossible by construction.

### 2. Invariants enforced by construction, verified by assertion

The design spec splits invariants into two classes:

**Enforced by construction** (architectural):
- Single quantization function → single cell-coordinate convention.
- Single time quantization → single time-bucket convention.
- Single weekday filter on the shared DataFrame → Saturday cannot leak into any view.
- Single transition-detection pass → pickups, dropoffs, seeking segments, driving segments all consistent.
- `n_days` is a scalar computed once and passed to every writer.

**Asserted before writing** (defense in depth, split by scope):
- **Per-trajectory** failures drop the offender and record a `RemovalRecord` (with driver, which invariant, failing values, state count, removal category) in `processing_metadata.json`. The run continues. If per-trajectory removals exceed a threshold (default 5%), a loud warning fires, but the run still completes.
- **Systemic** failures — count mismatches, wrong driver count, profile-matrix shape or normalization errors — raise `SystemicInvariantError` and abort.

The user-stated load-bearing invariant — *every seeking trajectory's `states[-1]` has `pickup_counts[cell] ≥ 1`* — holds **by construction**: after per-trajectory removal, the CLI re-derives pickup/dropoff counts from the surviving trajectory endpoints. `sum(pickup_counts) == len(seeking_trajectories)` is then automatically true.

### 3. Trajectory `states[-1]` is the post-transition cell (symmetric for seeking and driving)

- **Seeking trajectory:** `state[-1]` is the first GPS ping where `passenger_indicator == 1` (the pickup transition itself). Cell of `state[-1]` is the actual pickup location.
- **Driving trajectory:** `state[-1]` is the first `passenger_indicator == 0` ping after the dropoff transition (the dropoff itself).

This symmetry has two consequences: (a) the modifier's claim *"we relocated the pickup at cell X"* is literally true — cell X is the pickup, not a pre-pickup proxy; (b) the discriminator cannot learn an easy endpoint-type shortcut because both streams end on post-transition cells.

### 4. `active_taxis` = available-only (research-aligned definition)

A driver counts as "active" at target cell `(cx, cy)` during `(hour, day)` if they had ≥1 GPS
ping in the 5×5 neighborhood around `(cx, cy)` during that hour **and at least one of those pings
had `passenger_indicator == 0`**. Occupied taxis never contribute to supply.

The 5×5 neighborhood smoothing is preserved from the legacy tool so a taxi in an adjacent
cell can still respond to demand at the center. The available-only filter is new — it makes the
F_spatial DSR denominator represent *service capacity*, not traffic presence.

### 5. Day filter: weekdays only

`timestamp_to_day` returns `None` for Saturday and Sunday; `event_stream.py` drops those rows
via `dropna(subset=["day_index"])`. Applied once at the shared event stream means Saturday
cannot leak into any of the 8 outputs. This is a permanent project-wide decision per the design spec.

### 6. Profile features: 11 features with home-cell fallback cascade

`views/profile.py::compute_profile_features` computes the 11 driver-identity features (`home_x/y`,
`shift_start/end`, `freq_grid_x/y`, `avg_seeking/driving_dist/time`, `num_trips_per_day`) and
z-score-normalizes across the 50 drivers. `home_x/y` uses a fallback cascade:
1. Primary: mode of cells where `time_bucket == 1` (first 5-min bucket of the day).
2. Fallback 1: mode over `time_bucket ∈ [1..12]` (first hour).
3. Fallback 2: mode over all of the driver's records (with an entry in `processing_metadata.json`).

The primary definition captures the driver's physical home (midnight cell), not a session
artifact (where the last dropoff happened to be).

### 7. Deterministic driver indexing

Sort plate_ids lexicographically, assign 0..49. Emitted as `driver_index_mapping.pkl`
(bijective). Reruns on identical raw GPS produce identical outputs — reviewer can verify.

---

## Invariants — per-trajectory vs systemic

| # | Invariant | Scope | Action on failure |
|---|---|---|---|
| 1 | Every trajectory `state[-1]` has a matching count ≥ 1 | per-trajectory | drop + `RemovalRecord(category="no_matching_count")` |
| 2 | Every state has valid coords (`x∈[1,48]`, `y∈[1,90]`, `tb∈[1,288]`, `day∈{1..5}`) | per-trajectory | drop + `category="out_of_bounds"` |
| 3 | Every trajectory has ≥ 2 states | per-trajectory | drop + `category="degenerate_length"` |
| 4 | Temporal order within a trajectory is non-decreasing in `time_bucket` | per-trajectory | drop + `category="temporal_order"` |
| 5 | `sum(pickup_counts) == n_seeking_trajectories` and `sum(dropoff_counts) == n_driving_trajectories` | systemic | `SystemicInvariantError` |
| 6 | Exactly 50 unique drivers | systemic | `SystemicInvariantError` |
| 7 | Profile matrix is `50×11`, no NaN, column mean ≈ 0, column std ≈ 1 (constant columns excepted) | systemic | `SystemicInvariantError` |
| 8 | For every cell/hour/day with a pickup, at least one driver is counted as active | systemic | `SystemicInvariantError` |

If per-trajectory removal rate exceeds `config.REMOVAL_RATE_WARN_THRESHOLD` (default 5%), the
CLI emits a loud warning but does NOT abort — the removals are transparent in
`processing_metadata.json` for researcher audit.

---

## API surface

```bash
# CLI (what you'll use 95% of the time)
python -m famail_temporal.data.source_generation \
    --input-dir raw_data/ \
    --output-dir famail_temporal/source_data/ \
    --verbose

# --help:
python -m famail_temporal.data.source_generation --help
```

```python
# Programmatic (for tests and golden-dataset workflows)
from pathlib import Path
from famail_temporal.data.source_generation.cli import run_generation

result = run_generation(
    input_dir=Path("raw_data/"),
    output_dir=Path("famail_temporal/source_data/"),
    expect_n_drivers=50,     # override to 2 for synthetic fixtures; default 50
)

# result.paths      — OutputPaths dataclass (all 10 file paths)
# result.n_seeking_kept
# result.n_driving_kept
# result.n_removals
```

**Outputs produced by a full run** (under `--output-dir`):

| File | Schema | Consumer |
|---|---|---|
| `pickup_dropoff_counts.pkl` | `dict[(x, y, tb, day)] -> (pickup, dropoff)` | `preprocess.py` → `pickup_3d`, `dropoff_3d` |
| `active_taxis_5x5_hourly.pkl` | `{data: dict[(x, y, hour, day)] -> int, stats, config, version}` bundle | `preprocess.py` → `active_taxis_3d` |
| `passenger_seeking_trajs.pkl` | `dict[plate_id str] -> list[list[[x, y, tb, day]]]` | `loader.py` → `bundle.trajectories` |
| `ms_seeking_trajs.pkl` | `dict[int driver_idx] -> list[trajectories]` | discriminator seeking stream |
| `ms_driving_trajs.pkl` | `dict[int driver_idx] -> list[trajectories]` | discriminator driving stream |
| `ms_profile_features.pkl` | `{features, features_normalized, feature_names, normalization, n_features}` bundle | discriminator profile stream |
| `ms_seeking_calendar_days.pkl` | `dict[int driver_idx] -> list[int day_idx]` | reserved; loaded but currently unused |
| `ms_driving_calendar_days.pkl` | same | reserved |
| `driver_index_mapping.pkl` | `{plate_to_idx, idx_to_plate}` | sidecar for joining plate-keyed and int-keyed files |
| `processing_metadata.json` | run config + GPS bounds + git SHA + full removal summary | sidecar for audit and reproducibility |

---

## Dependencies

- Project: `famail_temporal.data.source_generation.config` (constants only)
- Standard library: `pickle`, `pathlib`, `json`, `subprocess`, `logging`, `argparse`, `dataclasses`, `typing`, `datetime`
- Third-party: `pandas`, `numpy`

No imports from outside `famail_temporal/data/source_generation/` into the tool's own source, except `config.py`. This keeps the producer side cleanly decoupled from the consumer side — the tool could be lifted out of `famail_temporal/` with only a single import rewrite.

---

## Paper-section hook

This package corresponds to the **"Source-Data Generation"** appendix subsection. Worth highlighting to reviewers:
1. The single-event-stream architecture eliminates a whole class of cross-file seam bugs present in the predecessor tooling.
2. The per-trajectory vs systemic invariant split makes the pipeline robust to real-world data noise without hiding it — removals are auditable per-driver in `processing_metadata.json`.
3. The "available-only" active-taxis definition is a deliberate semantic choice; the fairness metric's DSR denominator now represents service capacity, not traffic presence.
4. The pickup-transition endpoint semantic is the load-bearing invariant that makes the modifier's mass-balance bookkeeping correct for every trajectory.

For operator and researcher instructions, see [`SOURCE_DATASET_GENERATION_QUICKSTART.md`](SOURCE_DATASET_GENERATION_QUICKSTART.md).
