# Design: `action_space_violation` Per-Trajectory Invariant

**Date:** 2026-04-21
**Status:** Approved for implementation
**Scope:** Add one new per-trajectory invariant to the unified source-data generation tool.

## Motivation

The original expert-driver dataset (`all_trajs.pkl`, element 125) attaches an
**action code** 0–9 to every state, where codes 0–7 are the 8 compass directions
(Δ ∈ {-1, 0, +1} per axis with `max(|dx|, |dy|) = 1`), code 8 is "stay"
(dx = dy = 0), and code 9 is "stop" (terminal, used for the legacy (0, 0)
sentinel which does not apply to our 1-indexed grid). The label function
`judge_action(x, y, nx, ny)` in [`new_all_trajs/step2_processor.py:76`](../../../new_all_trajs/step2_processor.py#L76)
only reads the *sign* of the axis deltas, so it silently labels a multi-cell
jump like (5, 10) → (7, 10) as "east" — the label does not guarantee physical
consistency with the 9-action agent semantics.

Trajectories in the current `source_data/` that contain at least one transition
with `max(|dx|, |dy|) > 1` cannot be rollouts of a 9-action agent. Concretely:

| Stream | Trajectories with ≥ 1 non-adjacent transition |
|---|---|
| Seeking (213,477 total) | 107,989 (50.59 %) |
| Driving (179,120 total) | 86,688 (48.40 %) |

These transitions come from GPS dropouts, tunnel segments, and high-speed
movement between ~15–30 s GPS samples on a ~1 km grid. Training an imitation
agent on these trajectories teaches it that multi-cell jumps are a legal move,
which is false in the downstream RL simulator.

## Filter definition

For every trajectory (seeking *and* driving), for every consecutive state pair
`(traj[i], traj[i+1])`:

```python
max_axis_delta = max(abs(traj[i+1][0] - traj[i][0]),
                     abs(traj[i+1][1] - traj[i][1]))
if max_axis_delta > 1:
    # Whole trajectory is rejected with category "action_space_violation".
```

The first violating transition is recorded on the `RemovalRecord`. If every
consecutive pair has `max_axis_delta ≤ 1`, the trajectory passes.

**Strictness:** all-or-nothing. A single non-adjacent transition drops the
entire trajectory. The alternative — segment-splitting at the violation —
was considered and rejected because it would produce sub-trajectories that
no longer end at a pickup/dropoff transition record, breaking the mass-balance
invariant (`pickup_3d[endpoint] ≥ 1 for every seeking trajectory`) that the
famail_temporal pipeline depends on.

**Scope:** spatial delta only. Time-bucket delta is unconstrained — a "stay"
action that persists for multiple time buckets is still a valid stay action.

## Placement in the invariant cascade

Insert the new check between `temporal_order` and `implausibly_long`. Both
`temporal_order` and `action_space_violation` are per-consecutive-pair validity
checks; putting them adjacent groups the two transition-shape checks together:

| # | Category | What it checks |
|---|---|---|
| 1 | `degenerate_length` | `len(traj) < 2` |
| 2 | `temporal_order` | Same-day backward `time_bucket` |
| **3** | **`action_space_violation`** | **`max(|dx|, |dy|) > 1` on any pair (NEW)** |
| 4 | `implausibly_long` | Duration > `MAX_TRAJECTORY_DURATION_BUCKETS` or day wrapped backward |
| 5 | `out_of_bounds` | State outside grid / time-bucket / weekday range |
| 6 | `no_matching_count` | Endpoint missing from pickup/dropoff counts |

Short-circuit semantics match the existing checks: the first failure wins.

## Data contract

**`RemovalCategory` literal** (in [`removal.py`](../../../famail_temporal/data/source_generation/removal.py)):
add `"action_space_violation"` to the `Literal[...]`.

**`RemovalRecord.failing_values`** dict shape for this category:
```python
{
    "from": (x, y, tb, day),           # violating transition start state
    "to": (nx, ny, ntb, nday),         # violating transition end state
    "max_axis_delta": max_axis_delta,  # the magnitude (always > 1 when this fires)
    "transition_index": i,             # 0-based index of the first bad pair
}
```

**`which_invariant`:** `6` (next available after the existing 1–5).

**Naming:** `max_axis_delta` is used in code, tests, and `processing_metadata.json`
consistently. Not `chebyshev_distance` — the term is correct mathematically but
jargon-heavy, and this codebase prioritizes name-level legibility.

## Files touched

| File | Change |
|---|---|
| `famail_temporal/data/source_generation/removal.py` | Add `"action_space_violation"` to `RemovalCategory` |
| `famail_temporal/data/source_generation/invariants.py` | Insert new check block in `_validate_single_trajectory`, between temporal_order and implausibly_long |
| `famail_temporal/data/source_generation/tests/test_invariants.py` | Three new tests; one fixture tweak (see below) |
| `famail_temporal/source_data/` | Regenerate after code changes are merged |

**Not touched:** [`writer.py`](../../../famail_temporal/data/source_generation/writer.py)
(new categories are picked up automatically via `counts_by_category`),
[`views/`](../../../famail_temporal/data/source_generation/views),
[`cli.py`](../../../famail_temporal/data/source_generation/cli.py) — the filter
runs upstream of all views, and the existing mass-balance rebuild in
[`cli.py:94-105`](../../../famail_temporal/data/source_generation/cli.py#L94-L105)
already reconstructs `pickup_dropoff_counts` from surviving-trajectory endpoints
after per-trajectory removals.

## Tests (TDD order)

1. **`test_per_trajectory_drops_multi_cell_jump_as_action_space_violation`** —
   construct a trajectory with a (5, 10) → (7, 10) transition (max_axis_delta = 2);
   assert it's dropped with category `"action_space_violation"` and
   `which_invariant == 6`.

2. **`test_per_trajectory_accepts_all_nine_actions`** — for each
   `(dx, dy) ∈ {-1, 0, 1}²`, build a minimal trajectory that uses it and
   assert all nine are kept. Nails down stay, cardinal directions, and
   diagonals explicitly.

3. **`test_action_space_failing_values_records_first_violation`** — trajectory
   with two non-adjacent transitions (e.g., at index 0 and index 2); assert
   `failing_values` captures the one at index 0 (matches `temporal_order`'s
   first-violation-wins behavior) and includes the `max_axis_delta` key.

**Fixture tweak for existing test:** `test_per_trajectory_drops_out_of_bounds`
currently uses `[[5, 10, 1, 1], [999, 999, 1, 1], [6, 11, 2, 1]]`. With the
new check at step 3 (before out_of_bounds at step 5), the `(5, 10) →
(999, 999)` transition fires `action_space_violation` first. The fixture
must change so that every consecutive transition has `max_axis_delta ≤ 1`
*and* at least one state is out of bounds. Concretely, use
`[[0, 10, 1, 1], [1, 10, 1, 1], [1, 11, 2, 1]]`:

- Transition (0,10) → (1,10): `max_axis_delta = 1` — passes step 3.
- Transition (1,10) → (1,11): `max_axis_delta = 1` — passes step 3.
- Same day, forward time, `duration = 1` bucket — passes step 4.
- State 0: `x = 0`, which violates `1 ≤ x ≤ X_GRID_MAX` — fires step 5.

Each test still exercises exactly one invariant. The `pickup_counts` fixture
gains `(1, 11, 2, 1): (1, 0)` so the trajectory endpoint would otherwise
reach step 6 cleanly.

## Config

No new config constants. The 9-action semantics is defined by
`max(|dx|, |dy|) ≤ 1` exactly — there is nothing to parameterize.

**`REMOVAL_RATE_WARN_THRESHOLD`** ([`config.py:42`](../../../famail_temporal/data/source_generation/config.py#L42))
stays at `0.05`. Post-filter removal rate will be ~49 %, which fires the
warning. The warning is calibrated for *"something's unexpectedly wrong"* —
a ~50 % removal rate *is* deliberate here, well-documented in
`processing_metadata.json`, and a one-line WARN in the generation log
is a useful signpost rather than a bug.

## Expected impact

After regenerating `source_data/`:

| Stream | Current count | Expected after filter | Approximate loss |
|---|---|---|---|
| Seeking | 213,477 | ~105,500 | ~50.6 % |
| Driving | 179,120 | ~92,400 | ~48.4 % |

The resulting dataset is still larger than the legacy
`discriminator/multi_stream/extracted_data/` output (~75k trajectories).
Downstream compatibility:

- **`famail_temporal` tests** — unchanged; fewer trajectories flow through
  but the schema is identical.
- **`discriminator/multi_stream/dataset_generation`** — need to verify
  post-regeneration that each driver still has ≥ 2 days with ≥ 5 trajs per
  stream (required for Ren positive-pair sampling). This is a check on the
  regenerated data, not a design risk — the filter doesn't change
  dataset_generation's logic.
- **Mass balance** — preserved. `cli.py:94-105` rebuilds
  `pickup_dropoff_counts` from surviving-trajectory endpoints; the systemic
  invariant #5 (`sum(pickup_counts) == n_seeking`) continues to hold.

## Risks accepted

- GPS-dropout trajectories are not distinguishable from high-speed movement
  trajectories in this data. Both produce `max_axis_delta > 1` transitions
  and both are rejected. This is the cost of aligning with the 9-action
  agent semantics; the alternative (interpolation or segment-splitting)
  would break mass balance and is not worth the complexity.
- The filter is permanent in the source-data pipeline. If a future consumer
  wants the pre-filter trajectories, they regenerate without this check
  (not by post-processing the surviving data).

## Out of scope

- **Consecutive-state deduplication.** The legacy extractor deduplicated
  consecutive identical `(x, y, time_bucket)` states. The unified source
  generation tool does not, and this filter does not change that. Dedup
  would only reshape 80 % of transitions that are already valid (Δ = 0
  stays) and would not affect the Δ > 1 transitions this filter targets.
- **Temporal-delta constraints.** The filter only constrains spatial delta.
  A trajectory with consecutive states separated by many time buckets at the
  same `(x, y)` is a perfectly valid repeated-stay action sequence.
- **Action-code labels on output.** This filter validates that transitions
  are consistent with the 9 actions; it does not emit action codes on
  `passenger_seeking_trajs.pkl`. The source-data output schema stays at
  the 4-element state `[x, y, time_bucket, day]`. If action-code labels are
  ever wanted, they should be computed by the consumer (e.g., the
  discriminator's dataset_generation) from the existing state sequence.
