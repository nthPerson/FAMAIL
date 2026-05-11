# Unified GPS Source-Data Generation Tool — Design Spec

**Date:** 2026-04-20
**Status:** Approved, ready for implementation planning
**Location:** `famail_temporal/data/source_generation/` (new sub-package)
**Replaces:** the legacy `pickup_dropoff_counts/`, `active_taxis/`, and `new_all_trajs/` tools

> **File-format note.** Several inputs and outputs are Python `.pkl` files. This is the format already consumed by `famail_temporal` and its data dictionary. The tool only reads and writes `.pkl` files produced by this project's own trusted tooling — not files from external or untrusted sources.

---

## 1. Problem statement

Investigation of a runtime `ValueError: pickup_N, dropoff_N, and active_taxis_N must not contain negative values` during a `--max-trajectories 5000 -k 200` experiment revealed three independent inconsistencies at the seam between two existing data-generation tools that were supposed to agree:

1. **`time_bucket` offset mismatch.** `new_all_trajs/config.py` sets `time_offset=0` (buckets in `[0, 287]`); `pickup_dropoff_counts/processor.py` sets `time_offset=1` (buckets in `[1, 288]`). The trajectory file is 0-indexed, the counts file is 1-indexed. A workaround for this in `famail_temporal/data/aggregation.py::time_bucket_to_hour` silently shifts ~8% of trajectory pickups into the wrong time block.
2. **Weekend filter mismatch.** `new_all_trajs/step1_processor.py` uses `exclude_weekends=True` (drops Sat AND Sun, keeping days 1-5). `pickup_dropoff_counts/processor.py` uses `exclude_sunday=True` (drops only Sun, keeping days 1-6). The two aggregates cover different day sets.
3. **Pickup-cell semantic mismatch.** In `step1_processor.py`, a seeking trajectory is terminated when `passenger_indicator` transitions `0 → 1`, but the transitioning record (where `passenger_indicator == 1`) is not appended. So `state[-1]` is the last GPS ping with `passenger_indicator == 0` — one ping before the actual pickup. Meanwhile `pickup_dropoff_counts` counts the pickup event at the transitioning record itself. The taxi can cross a grid cell between these two consecutive GPS pings, so the two tools attribute the same pickup to different cells.

Empirical confirmation (diagnostic run over all 39,091 trajectories in the current pipeline):

| Match condition | Trajectories matched |
|---|---:|
| Exact cell match, no time-offset correction | 50.48% |
| Exact cell match, +1 time-offset correction | 76.85% |
| Match within 3×3 spatial neighborhood at corrected time | 93.14% |
| Match within 3×3×3 xy × time neighborhood | 96.39% |
| No match anywhere nearby | 3.61% |

The user-stated invariant — *every trajectory's pickup cell must have a corresponding non-zero count in `pickup_3d`* — is currently violated for ~23% of trajectories. In batched modifier runs, 10 of the top-200 trajectories had `pickup_3d == 0` at their pickup cell, causing the modifier's pre-loop subtract to produce negative cell values and trigger the `ValueError`.

Rather than patching each seam, we are rebuilding source-data generation as a single unified pipeline that guarantees cross-file consistency by construction.

---

## 2. Scope

### In scope

The unified tool produces **8 GPS-derived files** plus 2 sidecar artifacts:

| # | File | Purpose |
|---|---|---|
| 1 | `pickup_dropoff_counts.pkl` | Source for `pickup_3d` and `dropoff_3d` after aggregation by `famail_temporal.preprocess` |
| 2 | `active_taxis_5x5_hourly.pkl` | Source for `active_taxis_3d` |
| 3 | `passenger_seeking_trajs.pkl` | Primary trajectory file, consumed directly by the modifier |
| 4 | `ms_seeking_trajs.pkl` | Multi-stream discriminator context (seeking stream) |
| 5 | `ms_driving_trajs.pkl` | Multi-stream discriminator context (driving stream) |
| 6 | `ms_profile_features.pkl` | Per-driver 11-dim profile vectors, z-score normalized |
| 7 | `ms_seeking_calendar_days.pkl` | Per-driver list of day indices with seeking activity |
| 8 | `ms_driving_calendar_days.pkl` | Per-driver list of day indices with driving activity |
| — | `driver_index_mapping.pkl` | Bijective plate_id ↔ int driver_idx mapping (sidecar) |
| — | `processing_metadata.json` | Run config, stats, warnings, GPS bounds (sidecar) |

### Out of scope

- `cell_demographics.pkl` and `grid_to_district_mapping.pkl` — sourced from census/district data, not GPS.
- v3 discriminator retraining — separate project phase; see §9.
- 126-dim state feature vectors — legacy `new_all_trajs/step2` output; not consumed by the current v3 discriminator (which takes raw 4-dim states and normalizes internally).

---

## 3. Settled design decisions

### 3.1 Pickup-cell semantic: `state[-1]` = pickup-transition record

A seeking trajectory's final state is the **first GPS record where `passenger_indicator == 1`** — the pickup itself — not the last seeking ping.

**Rationale:**
- The consuming code at `famail_temporal/utils/trajectory.py` already documents this invariant. The `pickup_state` property returns `states[-1]`, and the docstring says *"The pickup is the final state (states[-1])."* The legacy data source violates a contract the consumer already claimed.
- The research claim is *"we relocate the pickup at cell X."* For that claim to be literally true, `state[-1]` must be the pickup cell.
- Makes the user-stated invariant `pickup_3d[state[-1].cell] >= 1` true by construction: the same transition-detection pass produces both the trajectory endpoint and the pickup count.

### 3.2 Day filter: weekdays only (day_index ∈ {1..5})

All outputs exclude Saturday and Sunday. This is a **permanent project-wide decision** stated by the FAMAIL research team.

**Rationale:**
- Saturday data was found to be unreliable for this dataset.
- Sunday records are essentially absent in the raw data (drivers don't operate Sundays).
- Simplifies `n_days` consistency across all 8 files (single day filter on the shared event stream).
- Paper narrative stays clean: "weekday-pattern fairness" with no weekend regime confound.

### 3.3 Active taxi definition: available-only

For cell `(x, y, hour, day)`, count a driver as "active" if they had ≥1 GPS ping in the **5×5 neighborhood** centered at `(x, y)` during `hour` on `day`, **AND** at least one of those pings had `passenger_indicator == 0`.

**Rationale:**
- Occupied taxis cannot accept a new passenger; counting them as "supply" conflates traffic-presence with service-capacity.
- The F_spatial DSR denominator thus represents *service opportunity*, making the fairness claim stronger and reviewer-defensible ("our metric measures inequality in how well *available* taxis are matched to demand").
- The 5×5 neighborhood smoothing is preserved from the legacy tool — a driver in an adjacent cell can still respond to demand at the center cell.

### 3.4 Scope: regenerate all 8 GPS-derived files

The multi-stream files (ms_*.pkl) are produced by the same tool in the same pass, using the same conventions as the primary files.

**Rationale:**
- Avoids a future seam where multi-stream conventions drift from primary-trajectory conventions.
- Enables clean v3 discriminator retraining on internally-consistent inputs.
- Zero additional algorithmic risk — multi-stream files are simple re-keyings or inversions of the primary extraction.
- The 5 multi-stream files were silently inheriting legacy indexing bugs; leaving them out of scope would leave those bugs latent.

### 3.5 Multi-stream trajectory file design

- **`ms_seeking_trajs.pkl`**: identical content to `passenger_seeking_trajs.pkl`, keyed by `int` driver_idx (0..49) instead of plate_id string.
- **`ms_driving_trajs.pkl`**: trajectories between each pickup (0→1) and dropoff (1→0). `state[-1]` = the first `passenger=0` record AFTER the dropoff transition — mirrors seeking's convention. Dict structure identical to `ms_seeking_trajs.pkl`.
- **Driver indexing**: plate_ids sorted lexicographically and assigned 0..49. Bijective mapping emitted as `driver_index_mapping.pkl`.
- **Trajectory filters**: min length ≥ 2 states. **No max length cap.** **No max-trajectories-per-driver cap.** Legacy `max_trajectories_per_driver=5000` and length bounds were pragmatic artifacts, not research-driven.
- **Filename rename**: `passenger_seeking_trajs_45-800.pkl` → `passenger_seeking_trajs.pkl`. The `_45-800` suffix encoded legacy length bounds that no longer apply.

### 3.6 Symmetric trajectory-endpoint convention

For both seeking AND driving trajectories, `state[-1]` is the post-transition record: pickup (seeking) or dropoff (driving). This makes the discriminator's two streams symmetric and prevents it from learning an easy endpoint-type shortcut rather than actual trajectory realism.

### 3.7 Profile features: same 11, with 2 operational refinements

Preserve the 11 feature semantics from the existing `ms_profile_features.pkl`:

| # | Feature | Definition (refined) |
|---|---|---|
| 1 | `home_x` | Mode of `x` across trajectory states where `time_bucket == 1` (first 5-minute bucket of the day). Fallback cascade if empty: mode over `time_bucket ∈ [1..12]`; if still empty, mode over all states with a warning in metadata. |
| 2 | `home_y` | As above, for `y`. |
| 3 | `shift_start` | **5th percentile** of active time_buckets across all the driver's states. |
| 4 | `shift_end` | **95th percentile** of active time_buckets. |
| 5 | `freq_grid_x` | Mode of pickup-cell `x` across the driver's seeking trajectories (`state[-1]`). |
| 6 | `freq_grid_y` | Mode of pickup-cell `y`. |
| 7 | `avg_seeking_dist` | Mean `sum(\|Δx\|+\|Δy\|)` across consecutive states within each seeking trajectory (Manhattan cells traversed). |
| 8 | `avg_seeking_time` | Mean `(state[-1].timestamp − state[0].timestamp)` per seeking trajectory, in minutes. |
| 9 | `avg_driving_dist` | Same as #7 but for driving trajectories. |
| 10 | `avg_driving_time` | Same as #8 but for driving trajectories. |
| 11 | `num_trips_per_day` | Total pickups ÷ number of active weekday-days for that driver. |

All 11 features z-score normalized across the 50 drivers. Normalization stats (`mean`, `std` per feature) saved alongside.

**Refinements over legacy:**
- `home_x/y` was previously the mode of trajectory-start cells — biased toward wherever the last dropoff happened. The new definition uses `time_bucket == 1` (midnight), which is a cleaner proxy for physical home. The research-team direction here was explicit: trajectory-start cells could be anywhere in the city, while midnight cells are a strong prior on where the driver sleeps.
- `shift_start/end` was previously min/max (brittle to outlier pings, e.g., driver 0 had `shift_start=1, shift_end=287`). The new definitions use 5th/95th percentiles for robustness.

**Why preserve the 11 feature schema instead of redesigning:**
- v3 architecture expects exactly 11 features. Keeping the same set means retraining validation is a clean before/after comparison ("does v3 retrained on unified data outperform v3 on legacy data?"). Changing features AND data AND retraining means you can't isolate which change drove the performance delta.
- Profile features only carry per-driver conditioning into the discriminator; the modifier's gradient doesn't flow through them. Redefining them affects discriminator calibration but not modifier behavior — low research leverage for high design cost.

### 3.8 Conventions (fixed across all outputs)

- `x_grid`, `y_grid`: **1-indexed** on disk, [1..48] and [1..90].
- `time_bucket`: **1-indexed** [1..288] (5-minute resolution).
- `hour` (active_taxis only): **0-indexed** [0..23].
- `day_index`: **1-indexed** [1..5], Mon=1 through Fri=5.
- Grid size: 0.01° (~1.1 km; unchanged from legacy).
- GPS bounds: computed globally from all 3 raw files combined.
- active_taxis neighborhood: 5×5 (k=2).

---

## 4. Architecture

**Principle:** one pass over raw GPS → an enriched event-stream DataFrame (single source of truth) → N deterministic *views*, each producing one output file.

```
raw_data/taxi_record_{07,08,09}_50drivers.pkl
                  │
                  ▼
       ┌────────────────────────────┐
       │  Ingestion + Enrichment    │   (single pass)
       ├────────────────────────────┤
       │  1. Load + concat          │
       │  2. Global GPS bounds      │
       │  3. Quantize:              │
       │     - gps → (x, y) 1-idx   │
       │     - seconds → time_bucket│
       │     - timestamp → day_idx  │
       │     - drop weekends        │
       │  4. Sort (plate_id, ts)    │
       │  5. Detect transitions     │
       │     (groupby.diff)         │
       │  6. Enrich: trajectory_id, │
       │     is_pickup, is_dropoff, │
       │     is_empty, segment_type │
       └──────────────┬─────────────┘
                      │
                      ▼
           ENRICHED EVENT STREAM
            (single pandas DataFrame)
                      │
     ┌──────┬─────────┼────────┬────────┬──────────┐
     ▼      ▼         ▼        ▼        ▼          ▼
 [pu_do] [trajs] [active_  [profile][mapping][metadata]
  view    view    taxis]    view    view      view
    │      │        │         │       │          │
    ▼      ▼        ▼         ▼       ▼          ▼
 pickup_ pst +    active_   ms_      mapping.  metadata.
 dropoff ms_*+    taxis_    profile. pkl       json
 counts  cal_*    5x5_h     pkl
```

The enriched event stream is the sole load-bearing representation; everything downstream is a pure function of it plus config.

---

## 5. Module structure

```
famail_temporal/data/source_generation/
├── __init__.py
├── config.py              # constants, deterministic seeds, GPS/grid/time params
├── raw_loader.py          # load & concat the 3 taxi_record_*.pkl files
├── quantization.py        # ONE definition each of:
│                          #   - gps_to_grid(lat, lon, bounds) → (x, y) 1-indexed
│                          #   - seconds_to_time_bucket(s)     → tb in [1, 288]
│                          #   - timestamp_to_day(ts)          → day in {1..5} or None
├── transitions.py         # per-driver transition detection:
│                          #   df.groupby('plate_id').passenger_indicator.diff()
├── event_stream.py        # orchestrate ingest + quantize + transition → DataFrame
├── views/
│   ├── __init__.py
│   ├── pickup_dropoff.py  # → pickup_dropoff_counts.pkl (dense dict)
│   ├── active_taxis.py    # → active_taxis_5x5_hourly.pkl (bundle w/ 'data' key)
│   ├── trajectories.py    # → passenger_seeking_trajs.pkl,
│   │                      #   ms_seeking_trajs.pkl,
│   │                      #   ms_driving_trajs.pkl,
│   │                      #   driver_index_mapping.pkl
│   ├── profile.py         # → ms_profile_features.pkl (with fallback diagnostics)
│   └── calendars.py       # → ms_seeking_calendar_days.pkl,
│                          #   ms_driving_calendar_days.pkl
├── invariants.py          # cross-view assertion suite (§6)
├── writer.py              # serialize output, bundle metadata, JSON sidecar
├── cli.py                 # `python -m famail_temporal.data.source_generation`
└── tests/
    ├── test_quantization.py
    ├── test_transitions.py
    ├── test_event_stream.py
    ├── test_views.py
    ├── test_invariants.py
    ├── test_profile_fallbacks.py
    └── test_golden.py     # synthetic ~50-record dataset with hand-computed outputs
```

**Why a sub-package under `famail_temporal/data/` instead of flat files:** producer-side and consumer-side serve different roles. `data/loader.py`, `data/aggregation.py`, `data/active_mask.py` are *consumer-side* — they shape raw-data files into the cache tensors the algorithm uses. `data/source_generation/` is the *producer side*. Keeping them separated avoids accidental coupling and keeps the consumer-side code easier to reason about.

Each view file is small (50–150 LOC) and can be understood and tested in isolation.

---

## 6. Invariants

### Enforced by construction
- **Single quantization function** for `lat/lon → (x, y)`. All views call the same function. No cell-coordinate drift.
- **Single time quantization** — `seconds_to_time_bucket` produces 1-indexed `[1, 288]` everywhere; `hour` extraction is 0-indexed `[0, 23]` only where named `hour`.
- **Single weekday filter** applied to the shared event stream. Saturday cannot leak into any view.
- **Single transition-detection pass.** Pickups, dropoffs, seeking segments, driving segments all flow from one `groupby.diff()` call.
- **`n_days`** is a scalar computed once from the shared event stream, passed to every writer.

### Asserted before writing (defense in depth)

Invariants are split into two classes based on what kind of failure they can indicate:

**Per-trajectory invariants** (a failure indicates one abnormal trajectory → **drop the trajectory, record the removal in `processing_metadata.json`, continue processing**):

1. Every trajectory's `state[-1]` at `(x, y, tb, day)` has the corresponding pickup (seeking) or dropoff (driving) count at that key `≥ 1`.
2. Every state in a trajectory has valid quantized coordinates: `x ∈ [1..48]`, `y ∈ [1..90]`, `time_bucket ∈ [1..288]`, `day ∈ {1..5}`.
3. Every trajectory has `≥ 2` states after quantization and filtering.
4. Every trajectory's states are chronologically non-decreasing in timestamp.

**Systemic invariants** (a failure indicates a pipeline bug that cannot be attributed to one trajectory → **abort with concrete diagnostic**):

5. After per-trajectory removals: `sum(pickup_counts) == len(seeking_trajectories)` AND `sum(dropoff_counts) == len(driving_trajectories)`. (Pickup/dropoff counts must be recomputed from the surviving trajectories, so this invariant holds by construction unless there's an extraction bug.)
6. Exactly 50 unique drivers present across all files. Driver-index mapping is bijective.
7. Profile features: exactly `50 × 11`, no NaN, normalized mean ≈ 0 and std ≈ 1 per feature.
8. The multiset of seeking-trajectory `state[-1]` cells equals the pickup-count distribution after any per-trajectory removals.
9. Active-taxis sanity: for every cell-hour-day with a pickup, at least one driver is counted as active (since that driver was by definition empty just before the pickup).
10. All records surviving the weekday filter have `day ∈ {1..5}` and `hour ∈ [0..23]`.

### Removal reporting

For every trajectory dropped under per-trajectory invariants, `processing_metadata.json` records:

- `driver_id` (plate_id) and `driver_idx` (integer index, if already assigned)
- `trajectory_index_within_driver` (ordinal among that driver's extracted trajectories)
- `which_invariant` (the numbered rule that failed)
- `failing_values` (e.g., the out-of-bounds `x_grid`, the `state[-1]` cell that lacked a matching pickup count, the length that fell below 2)
- `n_states_before_removal`
- `removal_reason_category` (one of: `"out_of_bounds"`, `"degenerate_length"`, `"no_matching_count"`, `"temporal_order"`)

A summary block also aggregates total counts per removal category, so the researcher can see at a glance whether removals are a handful of edge cases or a systemic signal about upstream data quality.

If the number of per-trajectory removals exceeds a configurable threshold (default: **5%** of total extracted trajectories), the tool emits a **loud warning** but does not abort — real-world GPS data routinely has some noise; this is a signal for the researcher to inspect the metadata, not an automatic failure.

On any systemic invariant failure (#5-#10), the tool aborts with a concrete diagnostic (which rows / cells / drivers violated), not just "invariant X failed".

---

## 7. Research-goal alignment

| Settled decision | Research effect |
|---|---|
| `state[-1]` = pickup-transition cell | Modifier operates on the actual pickup location. Paper claim "we relocate pickup X" is literally true. |
| Counts derived from same transitions as trajectories | User-stated invariant `pickup_3d[state[-1]] >= 1` true by construction. No more mass-balance bugs. |
| Active-taxis = available-only (empty at some point) | F_spatial DSR denominator is service-capacity, not traffic. Stronger fairness claim. |
| Weekdays-only filter applied globally | No weekend/commute regime confound in any metric or discriminator score. |
| Multi-stream regenerated consistently | After v3 retraining, F_fidelity calibration reflects the same data universe as F_spatial + F_causal. |
| Same 11 profile features with refined definitions | Clean before/after comparison for discriminator retraining; ablation is trivial. |
| `home_x/y` from `time_bucket == 1` mode | Measures physical home location, not a session-artifact of where last dropoff happened. |
| 5th/95th percentile shift_start/end | Robust to outlier pings (legacy min/max produced `shift_start=1, shift_end=287` for driver 0). |
| Deterministic driver indexing via sorted plate_ids | Reproducibility: identical raw GPS → identical outputs. Reviewer can verify. |
| Processing metadata sidecar | Every output is self-describing; convention provenance is auditable. |

---

## 8. Testing strategy

- **Unit tests** per module (quantization, transitions, each view).
- **Golden test**: a hand-built synthetic GPS stream (~50 records across 2 fake drivers) with hand-computed expected values for every output file. Runs on every CI invocation.
- **Property tests** (hypothesis): generate random valid GPS streams, assert all §6 invariants hold on the output.
- **Profile fallback tests**: drivers with no `time_bucket == 1` records must still produce a valid `home_x/y` via the cascade, with the fallback trigger recorded in metadata.
- **End-to-end smoke test**: run on a subset of real data (e.g., 1 driver × 1 week), confirm all 8 outputs produced, all invariants pass, no warnings except expected ones.
- **Consistency regression test**: once initial full-run output is committed as a reference, future changes assert SHA256-level equality of outputs (or a documented diff). Detects silent behavior drift.

---

## 9. Post-implementation tasks (knock-on effects)

Recorded here in the design spec as a single source of truth so nothing gets forgotten. These MUST be scheduled to reach a working end-to-end pipeline after the tool is built.

### Required — block end-to-end runs
- [ ] Run the new tool on the full 3-month raw GPS dataset; produce all 8 output files under `famail_temporal/raw_data/`.
- [ ] Update `famail_temporal/data/loader.py:95` — change filename from `passenger_seeking_trajs_45-800.pkl` to `passenger_seeking_trajs.pkl`.
- [ ] Regenerate preprocess cache: `python -m famail_temporal.preprocess --force`.
- [ ] Verify the end-to-end smoke test runs without error:
      `python -m famail_temporal.evaluation.runner --name post-regen-smoke --max-trajectories 200 -k 5 --override MAX_ITERATIONS=5`.

### Scheduled-next — research quality
- [ ] Retrain the v3 discriminator on the new `ms_seeking_trajs.pkl`, `ms_driving_trajs.pkl`, and `ms_profile_features.pkl`.
- [ ] Place the retrained checkpoint at `famail_temporal/discriminator_checkpoints/default/best.pt` so the runner uses a real (not `nn.Identity`) discriminator.
- [ ] Validation study comparing v3-on-new-data vs v3-on-legacy-data on held-out trajectories.

### Cleanup — follow-up PR, non-blocking
- [ ] Remove the `time_bucket=0` tolerance workaround in `famail_temporal/data/aggregation.py::time_bucket_to_hour` — new trajectories are 1-indexed by construction, so the workaround is obsolete. Replace with a strict check that fails loudly on any `time_bucket == 0`.
- [ ] Archive (or delete) the legacy tools: `new_all_trajs/`, `pickup_dropoff_counts/`, `active_taxis/`. Retain only as git history.
- [ ] Delete repo-root diagnostic scripts `diag_negative_cells.py` and `diag_trajectory_vs_pickup_3d.py`.

### Documentation & comparability
- [ ] CHANGELOG entry noting that experimental results from before regeneration are NOT directly comparable to post-regeneration results. Specifically flag: (a) the active_taxis definition changed (available-only vs any-presence), (b) the trajectory endpoint changed (pickup-transition vs last-seeking), (c) the `n_days` normalization may change.
- [ ] Update `famail_temporal/data/README.md` (if present) to reference `source_generation/` as the canonical raw-data producer.

---

## 10. Open items (all resolved)

| Item | Resolution |
|---|---|
| Memory strategy for the enriched event stream | No concern; use pandas with everything in RAM |
| DataFrame library | pandas |
| `active_taxis_5x5_hourly.pkl` bundle format | Yes: `{'data': ..., 'stats': ..., 'config': ..., 'version': '1.0.0'}` matching the data-dictionary |
| `active_taxis` daily and all-period variants | Not produced; outside `famail_temporal.preprocess` consumption |
| Where to document decisions + knock-on effects | This spec (§§3, 7, 9) |

---

End of spec.
