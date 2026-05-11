# Fairness-Attribution Export Tool — Design Notes

**Purpose.** Design reference for a forthcoming tool that exports per-
spatial-temporal-cell fairness attributions for downstream consumers —
primarily Manuel's GAN/GAIL agent (separate project) and a parallel
baseline-GAN the current project will train for evaluating the
trajectory-modification framework.

**Status.** **Unblocked 2026-04-24.** `config.T` transitioned from 4 to 24
on 2026-04-24 — TIME_BLOCKS now covers 24 hourly blocks (`hour_00` through
`hour_23`). Implementation of the export tool is ready to proceed; design
below is the agreed spec.

---

## What the consumers need

**Manuel (collaborator):** A per-(cell, time, day) dataset of fairness
attributions, so a GAN/GAIL agent operating in the 48 × 90 × T × 5
spatial-temporal environment can be taught the fairness concepts
defined by this project through reward shaping or imitation targets.

**This project's own baseline GAN:** Same dataset, same format. Used as
a baseline in the evaluation framework for trajectory modification.
(Having a single canonical export for both removes drift risk.)

---

## Decisions (frozen 2026-04-24)

### 1. Time granularity — broadcast `time_block` to `time_bucket`

Fairness is computed at `(x, y, time_block)` granularity; the export
broadcasts each time_block's attribution value across all 5-minute
time_buckets that fall within that block. A (cell, block)'s attribution
appears identically on every time_bucket in that block.

**Rationale.** Consumers reason in 5-min state resolution because the
RL simulator steps at that cadence. Broadcasting is semantically honest
(the value literally is constant across buckets within a block) and
removes a mapping step on the consumer side.

**Note.** `config.T = 24` as of 2026-04-24. Each time_block is one hour
and covers 12 time_buckets. Broadcast semantics remain identical and
T-agnostic by design.

### 2. Day aggregation — pooled across days, broadcast to all day indices

Fairness is computed pooled across `n_days`; the export broadcasts the
same (cell, block) attribution to all 5 `day_index` values. A row for
`(x, y, tb, day=Monday)` has identical attribution values to
`(x, y, tb, day=Wednesday)`.

**Rationale.** The underlying fairness computation is a single pooled
quantity, not a per-day quantity. Broadcasting exposes the `day`
column for format consistency without inventing per-day values that
the pipeline doesn't compute.

**Future research direction (carried to methodology notes §9).**
Per-day fairness analysis is a plausible extension — each day gets its
own attribution set, capturing weekday-to-weekday variation. See
[`F_CAUSAL_METHODOLOGY_NOTES.md`](F_CAUSAL_METHODOLOGY_NOTES.md) §9
for the note on this direction.

### 3. Active vs inactive cells — dense grid with NaN + `is_active` flag

Every (x, y, time_bucket, day) combination appears in the export.
Inactive cells (no supply, out of the Shenzhen boundary, or NaN
demographics) have `spatial_fairness_attribution = NaN`,
`causal_fairness_attribution = NaN`, and `is_active = False`.

**Rationale.** A GAN navigating the environment will attempt inactive
states. Dense + NaN lets the consumer decide whether to mask them out,
assign a "cannot-be-here" penalty, or interpolate — without forcing a
policy decision at export time. Sparse would require the consumer to
reconstruct inactive-state handling. Dense costs ~a few hundred
thousand additional rows of NaN; negligible for file size.

### 4. Sign convention — positive = contributes to fairness

The emitted `spatial_fairness_attribution` and
`causal_fairness_attribution` are **signed decomposition contributions**:
the sum across all active cells equals the overall `F_spatial` /
`F_causal` scalar. A positive value means "this cell's behavior
contributes to raising the overall fairness metric" (good for fairness);
a negative value means "this cell's behavior lowers overall fairness"
(harmful for fairness).

**Crucial documentation point for the README.** The OVERALL
metrics `F_spatial ∈ [0, 1]` and `F_causal ∈ [0, 1]` are in the unit
interval. PER-CELL attributions are NOT bounded in [0, 1] — they are
signed real numbers whose sum equals the overall metric. Consumers
needing a bounded per-cell scalar must normalize themselves (no
universal choice fits every use case).

### 5. Context columns — include D, S, Y alongside attributions

Include `demand_D`, `supply_S`, and `service_rate_Y` per row so the
consumer can sanity-check attributions, condition a model on them, or
compute derived quantities without a re-derivation pipeline.

### 6. Output formats — three views of the same data

Emit all three:

| File | Schema | Best for |
|---|---|---|
| `fairness_attribution_tuples.pkl` | Metadata preamble + list of `(x, y, tb, day, spatial_attr, causal_attr, is_active, D, S, Y)` tuples | Matches original verbal request; iterator-friendly; smallest no-dependency consumer |
| `fairness_attribution_long.pkl` | DataFrame with same columns | Pandas-friendly filtering and analysis |
| `fairness_attribution_dense.pkl` | `{'spatial': ndarray(48, 90, T), 'causal': ndarray(48, 90, T), 'active_mask': ndarray(48, 90, T) bool, 'D': ndarray(48, 90, T), 'S': ndarray(48, 90, T), 'Y': ndarray(48, 90, T), 'metadata': {...}}` | Fast tensor lookup for GAN training loop |

The three are algebraically equivalent; consumers pick by convenience.

### 7. Output location

`famail_temporal/exports/<timestamp>/` — follows the existing `results/`
pattern for timestamped artifact directories. Each export run gets a
fresh subdirectory (no in-place overwrites) so consumers have a stable
reference point.

### 8. Per-driver breakdown — NOT INCLUDED

The emitted attributions are spatial-temporal-only. Fairness, as defined
in this project, is a spatial-temporal property independent of driver
identity; per-driver attributions would be a different derivation
entirely and are out of scope.

---

## Row-level schema (for all three formats)

| Column | Type | Range | Description |
|---|---|---|---|
| `x_grid` | int | [1, 48] | Grid-cell x coordinate, 1-indexed |
| `y_grid` | int | [1, 90] | Grid-cell y coordinate, 1-indexed |
| `time_bucket` | int | [1, 288] | 5-minute time bucket, 1-indexed |
| `day` | int | [1, 5] | Weekday index (Mon=1 … Fri=5) |
| `spatial_fairness_attribution` | float (may be NaN) | signed | Per-cell decomposition contribution to `F_spatial`; positive = more fair |
| `causal_fairness_attribution` | float (may be NaN) | signed | Per-cell decomposition contribution to `F_causal`; positive = more fair |
| `is_active` | bool | — | Whether this cell is in the fairness audit |
| `demand_D` | float (NaN if inactive) | ≥ 0 | Mean hourly pickups in (cell, time_block) |
| `supply_S` | float (NaN if inactive) | ≥ 0 | Mean hourly active taxis in (cell, time_block) |
| `service_rate_Y` | float (NaN if inactive) | > 0 | `S / max(D, DEMAND_FLOOR)` |

The metadata preamble / sidecar carries:

- `schema_version`: "1.0.0"
- `generated_at`: ISO 8601 timestamp
- `source_data_git_sha`: git SHA of source_generation commit
- `config_snapshot`: T, TIME_BLOCKS, GRID_DIMS, DEMAND_FLOOR, SUPPLY_FLOOR, ACTIVE_SUPPLY_THRESHOLD, DEMOGRAPHIC_FEATURES
- `overall_F_spatial`: scalar float, the global F_spatial on this dataset
- `overall_F_causal`: scalar float, the global F_causal on this dataset
- `signal_regime_r2`: scalar float, diagnostic from F_CAUSAL_METHODOLOGY_NOTES.md §3
- `sign_convention`: "positive_is_fair"
- `n_active_cells_per_block`: list[int], the count per time_block

---

## README content (to accompany the export)

The export directory includes a README that covers:

1. **TL;DR.** "Per-cell fairness attributions for the FAMAIL
   spatial-temporal grid. Use `positive value = more fair` as reward
   signal. Attribution VALUES are signed — clamp to [0, 1] only if
   needed for your loss function."
2. **Sign convention** — verbatim from §4 above.
3. **Overall vs per-cell scale** — the `[0, 1]` clarification is prominent
   and explicit.
4. **Granularity notes** — time_block broadcasting (§1), day
   broadcasting (§2), active/inactive handling (§3).
5. **Field reference** — the row-level schema table.
6. **Example lookup code** for each of the three formats.
7. **Reproducibility** — which config values and source-data SHA
   produced the file.
8. **Contact / version info.**

---

## Implementation plan (for when we unblock)

**Module:** `famail_temporal/evaluation/export_fairness_attributions.py`.

**CLI:** `python -m famail_temporal.evaluation.export_fairness_attributions`.

**Steps the tool will take:**

1. Load a `DataBundle` via `DataBundle.load()`.
2. Compute `F_spatial` and `F_causal` on the cached active units using
   the existing fairness modules.
3. Extract per-cell attributions via the canonical functions
   `per_cell_fairness_attribution_spatial` and
   `per_cell_fairness_attribution_causal` — the same single canonical
   decompositions that the trajectory-modification algorithm uses.
   Both sum to their respective F-metric (1/N-shifted decomposition; see
   [`FAIRNESS_DECOMPOSITION_FORMULATION.md`](FAIRNESS_DECOMPOSITION_FORMULATION.md)).
4. Use the `UnitIndexMap` to map active-unit indices back to
   (x, y, time_block) coordinates.
5. Broadcast attributions along the time_bucket axis (all 5-min
   buckets within a time_block receive the same value).
6. Broadcast attributions along the day axis (all days receive the
   same value).
7. Build three output artifacts: tuples, DataFrame, dense tensors.
8. Emit the README with sign convention, granularity, and field reference.
9. Write a metadata JSON sidecar.

**Tests to add:**

- `test_export_tuples_format_and_sign_convention`: assert emitted tuples
  have correct schema, signed attributions, NaN for inactive cells.
- `test_export_sum_of_attributions_equals_overall_metric`: critical
  consistency invariant — `sum(spatial_attr over active) ≈ F_spatial`
  and similarly for causal.
- `test_export_broadcast_consistency`: tuples at
  `(x, y, tb1, day1)` and `(x, y, tb2, day2)` within the same time_block
  carry identical attributions.
- `test_export_three_formats_are_consistent`: same data in tuples,
  DataFrame, and dense tensors.

---

## Known constraints and future considerations

### T=24 transition complete (2026-04-24)

`config.T` moved from 4 to 24 on 2026-04-24. Export tool can now be
implemented; broadcasting semantics are T-agnostic by design.

### Sign-convention consistency with upstream

The project's `F_causal = 1 − r²_{demo}` formulation already has the
"higher = fairer" orientation at the scalar level. The per-cell
attribution emitted by this export must carry the same direction.
Implementation MUST verify this by checking sign orientation against
the overall metric at tool-run-time (if `F_causal > 0.5` and the sum
of attributions matches, sign is correct; the test
`test_export_sum_of_attributions_equals_overall_metric` covers this).

### Per-day attribution is out of scope (for now)

Decision 2 broadcasts pooled attributions to all days. Per-day
attributions would require re-running the fairness computation
separately on each day's subset of the data. Out of scope for the
current export; noted as a future research direction in
[`F_CAUSAL_METHODOLOGY_NOTES.md`](F_CAUSAL_METHODOLOGY_NOTES.md) §9.
If that research direction is pursued, the export tool can be
extended with an optional per-day output axis.

### Source-data provenance

Every export must embed the source-data `git_sha` and the
`processing_metadata.json` from the source_generation run that
produced the underlying `source_data/`. This is critical for
reproducibility and for Manuel to know which dataset his GAN was
trained against.

---

## Change log

- **2026-04-24 (initial)** — Initial design. Blocked on `T=4` → `T=24`
  configuration transition before implementation.
- **2026-04-24 (unblocked)** — `config.T` moved from 4 to 24 same day.
  Design approved; implementation can proceed.
