# Action-Space Violation Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new per-trajectory invariant to the unified source-data generation tool that rejects trajectories containing at least one consecutive-state transition with `max(|dx|, |dy|) > 1`, enforcing physical consistency with the 9 possible actions of the original `all_trajs.pkl` dataset's state vector (8 compass moves + stay).

**Architecture:** One new check block inside `_validate_single_trajectory` in [`famail_temporal/data/source_generation/invariants.py`](../../../famail_temporal/data/source_generation/invariants.py), placed between the `temporal_order` and `implausibly_long` checks. All-or-nothing strictness: one bad transition drops the whole trajectory. The filter lives upstream of all views; the existing `cli.py` mass-balance rebuild already handles surviving-endpoint count reconstruction, so no downstream consumer changes are needed.

**Tech Stack:** Python, pytest, pandas (indirect). No new dependencies.

**Reference spec:** [`docs/superpowers/specs/2026-04-21-action-space-violation-filter-design.md`](../specs/2026-04-21-action-space-violation-filter-design.md)

---

## File Structure

**Modified:**
- `famail_temporal/data/source_generation/removal.py` — add `"action_space_violation"` to `RemovalCategory` Literal
- `famail_temporal/data/source_generation/invariants.py` — insert new check in `_validate_single_trajectory`
- `famail_temporal/data/source_generation/tests/test_invariants.py` — three new tests, one fixture tweak

**Not modified (intentionally):**
- `writer.py`, `cli.py`, `views/` — the filter runs upstream of all views, and `counts_by_category` in the removal summary picks up new categories automatically.
- `discriminator/multi_stream/dataset_generation/` — consumes kept trajectories; transparent to the new filter.

**Regenerated (gitignored — nothing to commit):**
- `famail_temporal/source_data/` — via `python -m famail_temporal.data.source_generation` after code changes land.

---

## Task 1: Add `action_space_violation` to `RemovalCategory`

**Rationale:** This constant is referenced by the new tests in Task 3–5 and by the production code in Task 3. Land it first so tests can assert against a valid Literal value.

**Files:**
- Modify: `famail_temporal/data/source_generation/removal.py:7-13`

- [ ] **Step 1: Add the new literal value**

Edit the `RemovalCategory` Literal to include the new category:

```python
RemovalCategory = Literal[
    "out_of_bounds",
    "degenerate_length",
    "no_matching_count",
    "temporal_order",
    "implausibly_long",
    "action_space_violation",
]
```

- [ ] **Step 2: Verify the import works**

Run:
```bash
python -c "from famail_temporal.data.source_generation.removal import RemovalCategory; print(RemovalCategory.__args__)"
```

Expected output includes `'action_space_violation'`:
```
('out_of_bounds', 'degenerate_length', 'no_matching_count', 'temporal_order', 'implausibly_long', 'action_space_violation')
```

- [ ] **Step 3: Commit**

```bash
git add famail_temporal/data/source_generation/removal.py
git commit -m "feat(source_generation): add action_space_violation to RemovalCategory"
```

---

## Task 2: Pre-tweak the `out_of_bounds` test fixture

**Rationale:** The existing `test_per_trajectory_drops_out_of_bounds` fixture `[[5, 10, 1, 1], [999, 999, 1, 1], [6, 11, 2, 1]]` will fail the new action-space check *first* once Task 3 lands (the transition `(5,10) → (999,999)` has `max_axis_delta = 989`). We fix the fixture now, before implementing the check, so that Task 3 only adds behavior without breaking an existing test. Pre-tweaking is safe because the new fixture also passes under the current implementation.

**Files:**
- Modify: `famail_temporal/data/source_generation/tests/test_invariants.py`

- [ ] **Step 1: Replace the fixture**

Find the `test_per_trajectory_drops_out_of_bounds` function and replace its body:

```python
def test_per_trajectory_drops_out_of_bounds():
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [
            _valid_seeking_traj(),
            # x=0 is out of bounds (1 <= x <= X_GRID_MAX); every consecutive
            # transition has max_axis_delta <= 1 so out_of_bounds fires
            # rather than action_space_violation.
            [[0, 10, 1, 1], [1, 10, 1, 1], [1, 11, 2, 1]],
        ],
    })
    pickup_counts = {(6, 11, 2, 1): (1, 0), (1, 11, 2, 1): (1, 0)}
    dropoff_counts: dict = {}
    kept, removals = apply_per_trajectory_invariants(
        trajs, pickup_counts, dropoff_counts,
    )
    assert len(kept.seeking_by_plate["A"]) == 1
    assert len(removals) == 1
    assert removals[0].removal_reason_category == "out_of_bounds"
```

- [ ] **Step 2: Verify the existing test still passes**

Run:
```bash
python -m pytest famail_temporal/data/source_generation/tests/test_invariants.py::test_per_trajectory_drops_out_of_bounds -v
```

Expected: `1 passed`.

- [ ] **Step 3: Commit**

```bash
git add famail_temporal/data/source_generation/tests/test_invariants.py
git commit -m "test(source_generation): rework out_of_bounds fixture to avoid future conflict with action-space check"
```

---

## Task 3: Add the multi-cell-jump test + implement the check (TDD RED → GREEN)

**Rationale:** This is the core behavior. TDD discipline: write the test first, watch it fail because the check doesn't exist yet, then implement the minimal check that makes it pass.

**Files:**
- Modify: `famail_temporal/data/source_generation/tests/test_invariants.py`
- Modify: `famail_temporal/data/source_generation/invariants.py`

- [ ] **Step 1: Write the failing test**

Add this test to `famail_temporal/data/source_generation/tests/test_invariants.py`, right after `test_per_trajectory_drops_friday_to_monday_as_implausibly_long` (grouping transition-shape checks together):

```python
def test_per_trajectory_drops_multi_cell_jump_as_action_space_violation():
    """A trajectory containing a consecutive-state transition with
    max(|dx|, |dy|) > 1 cannot be a rollout of a 9-action agent and must
    be rejected. The first violating transition is recorded on the
    RemovalRecord's failing_values dict."""
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [[
            [5, 10, 1, 1],
            [7, 10, 2, 1],  # max_axis_delta = 2: non-adjacent, rejected
            [6, 11, 3, 1],
        ]],
    })
    pickup_counts = {(6, 11, 3, 1): (1, 0)}
    kept, removals = apply_per_trajectory_invariants(trajs, pickup_counts, {})
    assert kept.seeking_by_plate.get("A", []) == []
    assert len(removals) == 1
    assert removals[0].removal_reason_category == "action_space_violation"
    assert removals[0].which_invariant == 6
    assert removals[0].failing_values["max_axis_delta"] == 2
    assert removals[0].failing_values["transition_index"] == 0
```

- [ ] **Step 2: Run the test and verify it fails**

Run:
```bash
python -m pytest famail_temporal/data/source_generation/tests/test_invariants.py::test_per_trajectory_drops_multi_cell_jump_as_action_space_violation -v
```

Expected: `FAILED`, with an assertion error on one of the `assert` lines — most likely `assert kept.seeking_by_plate.get("A", []) == []` failing because the trajectory is currently kept (no action-space check exists yet).

- [ ] **Step 3: Implement the check in `_validate_single_trajectory`**

Open `famail_temporal/data/source_generation/invariants.py`. Find the block after the `temporal_order` for-loop:

```python
    for a, b in zip(traj, traj[1:]):
        ta, da = a[2], a[3]
        tb_, db = b[2], b[3]
        if da == db and tb_ < ta:
            return False, 4, "temporal_order", {
                "day_time_buckets": [(s[3], s[2]) for s in traj],
            }
    # Plausibility-of-duration check (design spec §6, invariant #5, research-
```

Insert the new check **between** the temporal_order for-loop and the "Plausibility-of-duration check" comment. The inserted block:

```python
    # Action-space-violation check (design spec §6, invariant #6). Enforces
    # physical consistency with the 9 possible actions of the original
    # all_trajs.pkl state vector: each consecutive-state transition must
    # satisfy max(|dx|, |dy|) <= 1 (8 compass moves + stay). Trajectories
    # with GPS-dropout or high-speed-movement jumps cannot be rollouts of
    # a 9-action agent and are rejected whole. First violation wins.
    for i, (a, b) in enumerate(zip(traj, traj[1:])):
        max_axis_delta = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        if max_axis_delta > 1:
            return False, 6, "action_space_violation", {
                "from": tuple(a),
                "to": tuple(b),
                "max_axis_delta": max_axis_delta,
                "transition_index": i,
            }
```

The resulting structure around the insertion point should read:
```python
    # Temporal-order check ...
    for a, b in zip(traj, traj[1:]):
        ...
        if da == db and tb_ < ta:
            return False, 4, "temporal_order", {...}
    # Action-space-violation check (design spec §6, invariant #6).
    for i, (a, b) in enumerate(zip(traj, traj[1:])):
        max_axis_delta = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        if max_axis_delta > 1:
            return False, 6, "action_space_violation", {...}
    # Plausibility-of-duration check ...
```

- [ ] **Step 4: Run the new test and verify it passes**

Run:
```bash
python -m pytest famail_temporal/data/source_generation/tests/test_invariants.py::test_per_trajectory_drops_multi_cell_jump_as_action_space_violation -v
```

Expected: `1 passed`.

- [ ] **Step 5: Run the full `test_invariants.py` to catch regressions**

Run:
```bash
python -m pytest famail_temporal/data/source_generation/tests/test_invariants.py -v
```

Expected: all tests pass (including `test_per_trajectory_drops_out_of_bounds` with its new fixture from Task 2). If any pre-existing test fails, the likely cause is a fixture that incidentally contains a non-adjacent transition — inspect the fixture and adjust so that each test exercises exactly one invariant.

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/data/source_generation/tests/test_invariants.py famail_temporal/data/source_generation/invariants.py
git commit -m "feat(source_generation): add action_space_violation per-trajectory invariant"
```

---

## Task 4: Add a test that covers all 9 agent actions

**Rationale:** Pin the exact action-space semantics — all eight compass moves and the stay action must be kept. If a future refactor accidentally changes the comparison (e.g., `>` to `>=`, which would reject "stay"), this test fires.

**Files:**
- Modify: `famail_temporal/data/source_generation/tests/test_invariants.py`

- [ ] **Step 1: Add the test**

Append after the test from Task 3:

```python
def test_per_trajectory_accepts_all_nine_actions():
    """All nine agent actions (8 compass moves + stay) must produce
    max_axis_delta <= 1 and be kept. This pins the action-space boundary
    against accidental off-by-one in the comparison (>= vs >)."""
    all_nine_deltas = [
        (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
    ]
    assert len(all_nine_deltas) == 9

    trajs = TrajectoriesResult(seeking_by_plate={}, driving_by_plate={})
    pickup_counts: dict = {}
    for i, (dx, dy) in enumerate(all_nine_deltas):
        plate = f"A{i}"
        start = (10, 20, 1, 1)
        # When dx=dy=0 (stay), bump time_bucket so the transition is a
        # same-cell state change at a later time; temporal_order still
        # passes because time_bucket increases.
        end = (10 + dx, 20 + dy, 2, 1)
        trajs.seeking_by_plate[plate] = [[list(start), list(end)]]
        pickup_counts[end] = (1, 0)

    kept, removals = apply_per_trajectory_invariants(
        trajs, pickup_counts, {},
    )
    assert len(removals) == 0, (
        f"Expected 0 removals, got {len(removals)}: "
        f"{[r.removal_reason_category for r in removals]}"
    )
    assert len(kept.seeking_by_plate) == 9
```

- [ ] **Step 2: Run the test and verify it passes**

Run:
```bash
python -m pytest famail_temporal/data/source_generation/tests/test_invariants.py::test_per_trajectory_accepts_all_nine_actions -v
```

Expected: `1 passed` (the implementation from Task 3 correctly handles all nine deltas because `max(|dx|, |dy|) ≤ 1` for every `(dx, dy) ∈ {-1, 0, 1}²`).

- [ ] **Step 3: Commit**

```bash
git add famail_temporal/data/source_generation/tests/test_invariants.py
git commit -m "test(source_generation): pin action-space boundary with all-9-actions coverage test"
```

---

## Task 5: Add a test for the first-violation-wins failing_values contract

**Rationale:** The `failing_values` dict surfaces in `processing_metadata.json` and is useful for debugging which trajectories got filtered. Pin that the dict contains the expected keys and records the *first* violation (matches the short-circuit behavior of the existing `temporal_order` check).

**Files:**
- Modify: `famail_temporal/data/source_generation/tests/test_invariants.py`

- [ ] **Step 1: Add the test**

Append after the test from Task 4:

```python
def test_action_space_failing_values_records_first_violation():
    """When a trajectory has multiple non-adjacent transitions, the
    RemovalRecord records the FIRST one (short-circuit behavior matching
    temporal_order). The failing_values dict carries from/to states,
    max_axis_delta, and transition_index."""
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [[
            [5, 10, 1, 1],
            [5, 10, 2, 1],
            [8, 10, 3, 1],   # transition 1: max_axis_delta = 3
            [8, 10, 4, 1],
            [20, 10, 5, 1],  # transition 3: max_axis_delta = 12
            [21, 11, 6, 1],
        ]],
    })
    pickup_counts = {(21, 11, 6, 1): (1, 0)}
    kept, removals = apply_per_trajectory_invariants(trajs, pickup_counts, {})
    assert len(removals) == 1
    r = removals[0]
    assert r.removal_reason_category == "action_space_violation"
    assert r.failing_values["transition_index"] == 1
    assert r.failing_values["max_axis_delta"] == 3
    assert r.failing_values["from"] == (5, 10, 2, 1)
    assert r.failing_values["to"] == (8, 10, 3, 1)
```

- [ ] **Step 2: Run the test and verify it passes**

Run:
```bash
python -m pytest famail_temporal/data/source_generation/tests/test_invariants.py::test_action_space_failing_values_records_first_violation -v
```

Expected: `1 passed`.

- [ ] **Step 3: Run the full `test_invariants.py` once more**

Run:
```bash
python -m pytest famail_temporal/data/source_generation/tests/test_invariants.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add famail_temporal/data/source_generation/tests/test_invariants.py
git commit -m "test(source_generation): pin first-violation-wins contract for action_space_violation"
```

---

## Task 6: Regenerate `source_data/` on real GPS data

**Rationale:** The code is now correct according to unit tests. Run the tool on real 3-month Shenzhen GPS data to (a) verify end-to-end that the new filter integrates cleanly with the rest of the pipeline, and (b) produce the dataset that downstream consumers will use.

**Files:**
- Regenerate: `famail_temporal/source_data/` (gitignored except README.md and .gitkeep)

- [ ] **Step 1: Back up the current `processing_metadata.json`**

For before/after comparison of removal counts:
```bash
cp famail_temporal/source_data/processing_metadata.json /tmp/pre_action_space_filter_metadata.json
```

- [ ] **Step 2: Regenerate `source_data/`**

Run the unified tool (takes ~10 minutes on real data):
```bash
python -m famail_temporal.data.source_generation --input-dir raw_data --output-dir famail_temporal/source_data
```

Expected terminal output includes:
```
INFO Extracted 214286 seeking + 179384 driving trajectories
INFO Applying per-trajectory invariants…
...
WARNING Per-trajectory removal rate XX.XX% exceeds threshold 5.00%
INFO Done: ~105,000 seeking + ~92,000 driving kept; ~195,000 removals; outputs at famail_temporal/source_data
```

The WARNING is expected (the removal rate is ~50% after this filter, well above the 5% warn threshold; see spec §Config).

- [ ] **Step 3: Verify post-filter counts match spec expectations**

Run this diagnostic:
```bash
python << 'EOF'
import json
with open("famail_temporal/source_data/processing_metadata.json") as f:
    meta = json.load(f)
rs = meta["removal_summary"]
print(f"Total extracted:  {rs['total_extracted']:,}")
print(f"Total removed:    {rs['n_removed']:,}")
print(f"Removal rate:     {100*rs['removal_rate']:.2f}%")
print("Counts by category:")
for cat, n in sorted(rs["counts_by_category"].items(), key=lambda kv: -kv[1]):
    print(f"  {cat:30s} {n:>10,}")
EOF
```

Expected:
- `removal_rate` is in the 0.45–0.55 range (spec predicts ~49%).
- `counts_by_category` contains `action_space_violation` with ~195,000 entries (the dominant new category).
- All previously-existing categories still present with their prior counts (approximately; minor rebalancing is normal because action_space_violation short-circuits before other checks for some trajectories).

- [ ] **Step 4: Verify the schema-contract invariants still hold on the regenerated data**

Run:
```bash
python -m pytest famail_temporal/data/source_generation/tests/test_schema_contract.py -v
```

Expected: 3 passed. This confirms that parallel-list lengths still match, calendar_day_map still resolves every index, and the profile `features` vs `features_normalized` split is preserved.

- [ ] **Step 5: No commit needed**

`famail_temporal/source_data/*.pkl` and `processing_metadata.json` are gitignored. The regeneration is a deployment action, not a code change. Confirm with:
```bash
git status --short
```

Expected: clean working tree (no uncommitted changes).

---

## Task 7: Full regression suite on the regenerated data

**Rationale:** Catch any downstream surprise from the ~50% data reduction — e.g., a test that expects a specific minimum-trajectory-count assertion.

**Files:** None modified.

- [ ] **Step 1: Run the full `famail_temporal` test suite**

Run:
```bash
python -m pytest famail_temporal/tests/ --run-slow --tb=short -q
```

Expected: all tests pass (the prior run was 258 passed, 1 skipped; this should match within ±2 on the skipped count depending on discriminator checkpoint presence).

**If `test_databundle_load_real_data` fails** because `bundle.unit_map.n_units < config.MIN_TOTAL_ACTIVE_UNITS`: the fewer surviving trajectories caused active-unit preprocessing to produce fewer active cells. Investigate by checking `config.MIN_TOTAL_ACTIVE_UNITS` against the new `unit_map.n_units` and report to the user before making any changes — this would be a data-sufficiency issue, not a code bug.

- [ ] **Step 2: Run the full `source_generation` test suite (unit tests only, fast)**

Run:
```bash
python -m pytest famail_temporal/data/source_generation/tests/ --tb=short -q \
    --ignore=famail_temporal/data/source_generation/tests/test_golden.py \
    --ignore=famail_temporal/data/source_generation/tests/test_cli.py \
    --ignore=famail_temporal/data/source_generation/tests/test_schema_contract.py
```

Expected: all unit tests pass (72 passed from the last verification, plus the 3 new tests from Tasks 3–5 = 75 passed).

- [ ] **Step 3: Run the integration tests (slow)**

Run:
```bash
python -m pytest famail_temporal/data/source_generation/tests/test_cli.py famail_temporal/data/source_generation/tests/test_golden.py famail_temporal/data/source_generation/tests/test_schema_contract.py --tb=short -v
```

Expected: 6 passed (the real-data smoke test takes ~10 minutes; other tests are fast).

- [ ] **Step 4: Summarize to the user**

Report to the user:
1. Total trajectory counts before/after the new filter (from Task 6 Step 3).
2. `counts_by_category` showing `action_space_violation` as the dominant removal.
3. Test results: unit + integration + famail_temporal all green.
4. Open question to the user: should they re-run the discriminator retraining now (`dataset_generation` + `train.py`) on the newly filtered data? This is their call — the dataset they generated earlier is still consistent but now includes multi-cell-jump trajectories we've decided are not 9-action-compatible.

---

## Self-Review Notes

**Spec coverage:**
- §Filter definition → Task 3 (implementation + test)
- §Placement in the invariant cascade → Task 3 Step 3 (insert between temporal_order and implausibly_long)
- §RemovalRecord schema → Task 1 (Literal), Task 3 (failing_values dict), Task 5 (test pins the contract)
- §Files touched → Tasks 1, 2, 3, 4, 5
- §Tests → Tasks 3, 4, 5
- §Fixture tweak → Task 2
- §Config (no new constants; warn threshold stays) → Task 6 Step 2 (warning is expected, not a bug)
- §Expected impact → Task 6 Step 3 (verification against ~50% prediction)
- §Risks accepted → acknowledged in spec; nothing new to plan

**Placeholder scan:** no TBD/TODO; every code step shows actual code; every shell step shows the expected output.

**Type / name consistency:** `max_axis_delta` used in code, tests, and the plan narrative; `"action_space_violation"` quoted consistently; `which_invariant=6` consistent.
