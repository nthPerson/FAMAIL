# `source_generation/tests/` — Unit + integration tests for the unified GPS producer

## Purpose

TDD-style tests for every module in `source_generation/`. Each production module
in Phase 1-4 of the implementation plan has its own test file; Phase 5 adds the
end-to-end golden test and the slow real-data smoke test. A full run is **81 tests,
2 real-data-gated skips**, typically under 2 seconds.

The test suite is where cross-file contracts get locked in: the golden test exercises
the whole pipeline end-to-end, including the CLI orchestrator's count-rebuild step
and the systemic-invariant check on a 2-driver synthetic fixture with relaxed
`expect_n_drivers=2`.

---

## Files

### Per-module unit tests (one per production module)

| Test file | Module under test | Coverage |
|---|---|---|
| `test_raw_loader.py` | `raw_loader.py` | Flat + nested day-list shapes; missing file; bad top-level type; multi-file concat |
| `test_quantization.py` | `quantization.py` | `GlobalBounds`, `gps_to_grid` (scalar + vectorized + edge clamp), `seconds_to_time_bucket` (midnight, hour boundary, last second, vectorized), `seconds_to_hour`, `timestamp_to_day` (weekdays + weekends + bad format) |
| `test_transitions.py` | `transitions.py` | Transition detection on a 9-state driver; per-driver isolation (diff doesn't leak across plates); `assign_segment_ids` places transition rows as LAST of their segment; per-driver segment_id resets |
| `test_event_stream.py` | `event_stream.py` | Output DataFrame has all required columns; weekend-day rows dropped; per-driver sort order; transitions land at correct row indices; `n_days` and `GlobalBounds` computed |
| `test_view_pickup_dropoff.py` | `views/pickup_dropoff.py` | Empty input; single pickup; multi-event aggregation; distinct cells |
| `test_view_active_taxis.py` | `views/active_taxis.py` | Empty; single empty ping + 5×5 neighborhood fan-out; occupied-only driver filtered out; deduplication across repeated pings; multi-driver addition; hour independence; grid-edge clamping |
| `test_view_trajectories.py` | `views/trajectories.py` | Lex-ordered driver mapping; seeking + driving extraction from a 9-row fixture; `state[-1]` = pickup-cell for seeking and dropoff-cell for driving; min-length-2 filter; incomplete trailing segment drop |
| `test_view_profile.py` | `views/profile.py` | Home from `time_bucket == 1` mode; 5th/95th percentile shift bounds; `zscore_normalize` shape/mean/std for a 50×11 matrix; `num_trips_per_day` = pickups / distinct calendar dates |
| `test_profile_fallbacks.py` | `views/profile.py::compute_home_xy_with_fallback` | Primary (`tb==1` mode); fallback 1 (first-hour mode); fallback 2 (all-records mode); empty driver raises |
| `test_view_calendars.py` | `views/calendars.py` | Sorted unique day extraction with dedup across multiple trajectories; missing driver produces empty list |
| `test_invariants.py` | `invariants.py` | Per-trajectory: drops out-of-bounds, degenerate length, no-matching-count, temporal-order (within-day backward), implausibly-long (Fri→Mon, Fri→mid-week, >120-bucket same-week), action-space-violation (multi-cell jump + first-violation short-circuit), plus accept-cases for short midnight crossing, long overnight shift within threshold, and all 9 agent actions. Systemic: count mismatch raises, wrong driver count raises. |
| `test_writer.py` | `writer.py` | All 10 files produced by `write_all_outputs`; active_taxis bundle has `{data, stats, config, version}`; metadata JSON records removals with category counts |
| `test_cli.py` | `cli.py` | End-to-end `run_generation` with 50 synthetic drivers produces all 10 expected output files; `n_seeking_kept` / `n_driving_kept` populated |

### End-to-end golden + smoke tests

| Test file | Purpose |
|---|---|
| `golden_fixtures.py` | Hand-built 2-driver synthetic raw-GPS fixture (AAA with 9 states crossing a full dropoff→seeking→pickup→driving→dropoff cycle; BBB with 3 states). Plus hand-computed expected trajectories and pickup-count endpoint. **Not a test file itself — a fixture module.** |
| `test_golden.py::test_golden_end_to_end` | Full pipeline on the golden fixture (with `expect_n_drivers=2`). Verifies: seeking trajectories match the hand-computed expected; AAA's pickup cell has `pickup_counts >= 1`; every seeking trajectory's `state[-1]` has a matching pickup count (invariant #1); `sum(pickup_counts) == n_seeking_trajectories` (systemic #5). |
| `test_golden.py::test_smoke_on_real_raw_if_present` | Same pipeline on the real 3-month Shenzhen raw GPS, if present under `raw_data/`. Skips via in-body `pytest.skip()` when the 3 required files aren't present. Marked `@pytest.mark.slow`. |

---

## How to run

```bash
# Full suite (81 tests, 2 real-data-gated skips, < 2 seconds)
.venv/bin/pytest famail_temporal/data/source_generation/tests/ -v

# Single module
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_view_trajectories.py -v

# Include the real-data smoke test (runs ~1-3 minutes if raw_data/ has the 3 taxi_record files).
# The `--run-slow` flag is registered in famail_temporal/tests/conftest.py (not here);
# the smoke test also has an in-body pytest.skip() guard so running this package's tests
# alone will correctly skip when raw data is absent.
.venv/bin/pytest famail_temporal/data/source_generation/tests/test_golden.py -v
```

---

## TDD conventions

Every module was written test-first. The pattern repeated across Phase 1-4:

1. Write the test file. It will fail at import time (`ModuleNotFoundError` on the production module).
2. Verify RED — the failure is "module missing," not "typo in test."
3. Write the production module.
4. Re-run — verify GREEN.
5. Commit (test + impl together).

This discipline is documented per task in [`docs/superpowers/plans/2026-04-20-unified-source-data-generation.md`](../../../../docs/superpowers/plans/2026-04-20-unified-source-data-generation.md).

---

## Known test-coverage gaps (non-blocking)

The whole-diff code review at the end of the implementation flagged these as follow-up work:
- No test for a `driving` kind `no_matching_count` drop (only `seeking` kind is exercised).
- No test for `zscore_normalize` on a mixed constant/varying column matrix (currently only all-varying and all-constant cases are covered transitively).

These are test-hardening opportunities; every branch is already exercised indirectly by the golden end-to-end test.

---

## Dependencies

- `pytest` (test runner)
- `pandas`, `numpy` (fixture construction, assertion helpers)
- Standard library `pickle` (writing and reading fixture files under `tmp_path`)
- `famail_temporal.data.source_generation.*` (modules under test)
