# FAMAIL Temporal — Source-Dataset Generation Quickstart

A researcher's guide to running the unified raw-GPS → source-dataset generation tool at
`famail_temporal/data/source_generation/` and interpreting its outputs.

This document answers three core operational questions:

1. **How do I regenerate the 8 source datasets from raw taxi GPS?**
2. **What did the tool actually produce, and is it trustworthy?** (Auditing via `processing_metadata.json`.)
3. **What do I need to do downstream after regeneration?** (Preprocess cache + v3 discriminator retraining.)

For the architectural deep-dive, see [`README.md`](README.md) in this directory.
For the full design rationale (why the tool exists, what each decision protects against),
see [`docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md`](../../../docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md).

> **Serialization note.** The tool reads and writes Python `.pkl` files because the downstream
> consumers (`famail_temporal/preprocess.py`, `famail_temporal/data/loader.py`, and the v3
> discriminator context builder) expect that format. All pickle I/O is against project-internal
> paths only — never against files from external or untrusted sources.

---

## When should I run this tool?

**Run it when** one of:

- Raw GPS data changed (new month, different driver set, upstream fix).
- A convention decision changed (e.g., the day filter or the active-taxi definition).
- A bug was discovered in the producer and fixed.
- You're setting up a fresh checkout and need to regenerate source data from scratch.

**Do NOT run it** before every experiment. This is a one-shot tool whose outputs land in
`famail_temporal/source_data/` and stay there. The evaluation runner does not invoke it.

---

## Prerequisites

Before your first regeneration run, confirm:

| Requirement | How to check |
|---|---|
| Raw GPS files present | `ls raw_data/taxi_record_{07,08,09}_50drivers.pkl` — all 3 exist. These are not committed to the repo (binary, large); obtain from the project's source. |
| `famail_temporal/source_data/` exists (output directory) | `ls famail_temporal/source_data/` — directory is present (may be empty except for `.gitkeep`). |
| `.venv` active with pandas + numpy + pytest | `.venv/bin/python -c "import pandas, numpy"` — both import cleanly. |
| Tests pass | `.venv/bin/pytest famail_temporal/data/source_generation/tests/ -q` — expect 81 passed, 1-2 real-data-gated skips. |

If any of the 3 raw GPS files is missing, the tool will raise `FileNotFoundError` immediately.
If all 3 are present but empty (e.g., a placeholder), the run fails at `compute_global_bounds`
with a clear error.

---

## Your first run

```bash
python -m famail_temporal.data.source_generation \
    --input-dir raw_data/ \
    --output-dir famail_temporal/source_data/ \
    --verbose
```

What to expect (approximate timings on a modern workstation):

```
2026-04-20 18:32:14,021 INFO Building event stream from raw_data
2026-04-20 18:32:47,118 INFO Building views...
2026-04-20 18:33:12,550 INFO Extracted 38914 seeking + 38864 driving trajectories
2026-04-20 18:33:12,610 INFO Applying per-trajectory invariants...
2026-04-20 18:33:21,883 INFO Computing profile features...
2026-04-20 18:33:23,904 INFO Checking systemic invariants...
2026-04-20 18:33:23,907 INFO Writing outputs to famail_temporal/source_data
2026-04-20 18:33:25,412 INFO Done: 38914 seeking + 38864 driving kept; 0 removals; outputs at famail_temporal/source_data
```

Total runtime: typically 30–90 seconds on the full 3-month dataset.

If you see the line `Per-trajectory removal rate X.XX% exceeds threshold 5.00%`, stop and read
`processing_metadata.json` before trusting the output — see "Understanding removals" below.

---

## What gets produced

Every run writes 10 files under `--output-dir`:

### The 8 source datasets (consumed by `famail_temporal.preprocess` + `famail_temporal.data.loader`)

| # | File | Schema |
|---|---|---|
| 1 | `pickup_dropoff_counts.pkl` | `dict[(x, y, tb, day)] -> (pickup, dropoff)` — sparse, Python ints |
| 2 | `active_taxis_5x5_hourly.pkl` | `{data: dict[(x, y, hour, day)] -> int, stats, config, version}` bundle |
| 3 | `passenger_seeking_trajs.pkl` | `dict[plate_id str] -> list[trajectory]` where each trajectory is `list[[x, y, tb, day]]` |
| 4 | `ms_seeking_trajs.pkl` | `dict[int driver_idx] -> list[trajectory]` |
| 5 | `ms_driving_trajs.pkl` | `dict[int driver_idx] -> list[trajectory]` |
| 6 | `ms_profile_features.pkl` | `{features, features_normalized, feature_names, normalization, n_features}` bundle |
| 7 | `ms_seeking_calendar_days.pkl` | `dict[int driver_idx] -> sorted list[int day_idx]` |
| 8 | `ms_driving_calendar_days.pkl` | same shape |

### 2 sidecars (for audit and reproducibility — not consumed by the algorithm)

| File | Purpose |
|---|---|
| `driver_index_mapping.pkl` | `{plate_to_idx: dict[str, int], idx_to_plate: dict[int, str]}` — the bijective lexicographic mapping used for the `ms_*` files' int keys. |
| `processing_metadata.json` | Run-level audit record: config snapshot, GPS bounds, `n_days`, git SHA, and the full removal summary with per-driver diagnostic records. |

### Conventions reminder

- `x` ∈ [1, 48], `y` ∈ [1, 90]: **1-indexed** grid cells (loader.py subtracts 1 when it sees them).
- `tb` ∈ [1, 288]: **1-indexed** 5-minute time buckets.
- `hour` ∈ [0, 23]: **0-indexed** (only the `active_taxis` file uses hour).
- `day` ∈ {1, 2, 3, 4, 5}: **Mon=1 .. Fri=5**. Saturday and Sunday never appear.

---

## Understanding removals — auditing via `processing_metadata.json`

Real-world GPS data has noise. The tool's per-trajectory invariants drop offending trajectories
and record each removal in `processing_metadata.json` so you can audit what was excluded and why.

Open the file and look at the `removal_summary` block:

```json
{
  "n_days": 65,
  "bounds": {"lat_min": 22.4425, "lat_max": 22.87, "lon_min": 113.7501, "lon_max": 114.5582},
  "git_sha": "c7e50bb",
  "config_snapshot": {
    "GRID_SIZE_DEG": 0.01,
    "NEIGHBORHOOD_SIZE": 5,
    "TIME_INTERVAL_MIN": 5,
    "WEEKDAY_DAYS": [1, 2, 3, 4, 5]
  },
  "removal_summary": {
    "total_seeking_extracted": 38914,
    "total_driving_extracted": 38864,
    "total_extracted": 77778,
    "n_removed": 0,
    "removal_rate": 0.0,
    "counts_by_category": {},
    "removals": []
  }
}
```

`n_removed = 0` is the happy path — every extracted trajectory survived all per-trajectory
invariants. If you see non-zero removals:

| `removal_reason_category` | What it means | What to do |
|---|---|---|
| `out_of_bounds` | A state in the trajectory had `x`, `y`, `tb`, or `day` outside its valid range. | Almost always indicates upstream data corruption — coordinates that escaped `gps_to_grid`'s clamp or a bogus timestamp. Inspect the per-record `failing_values.state` + `axis`. |
| `degenerate_length` | The surviving trajectory has <2 states after filtering. | Usually a very short seeking/driving segment at a day boundary. Safe to ignore unless the count is large. |
| `no_matching_count` | The trajectory's `state[-1]` endpoint had no matching pickup (seeking) or dropoff (driving) count. | **Should never happen under the new pipeline** — pickup/dropoff counts are re-derived from surviving trajectory endpoints, so this invariant now holds by construction. If you see it, something subtle is wrong upstream. |
| `temporal_order` | A trajectory had a non-monotonic `time_bucket` sequence (went backward). | Typically an artifact of mixing GPS pings from two trips that got assigned the same segment_id. Rare — check `failing_values.time_buckets`. |
| `implausibly_long` | The trajectory's elapsed duration exceeded the `MAX_TRAJECTORY_DURATION_BUCKETS` threshold (default: 120 buckets = 10 hours). A single seeking or driving episode shouldn't last this long — these are extraction artifacts where a segment was stitched across off-duty time (e.g., a Friday→Monday segment spanning the weekend because weekend records were filtered out). | A handful per 100K trajectories on clean data. Higher counts suggest upstream GPS data has long-gap segments that the extractor didn't split. Inspect `failing_values.duration_buckets` (the actual duration) and `failing_values.start` / `failing_values.end` (day/time_bucket endpoints). |
| `action_space_violation` | A consecutive-state transition exceeded `max(|dx|, |dy|) = 1`, i.e., the trajectory jumped more than one grid cell in a single 5-minute time bucket. The surviving trajectories are physically consistent with rollouts of a 9-action agent (8 compass moves + stay), which the downstream discriminator and RL models assume. | Expected at roughly 49% on raw Shenzhen GPS data — these are high-speed-movement segments and GPS-dropout artifacts. The surviving trajectories are the action-space-consistent subset. Inspect `failing_values.from`, `failing_values.to` (the two consecutive states), `failing_values.max_axis_delta` (the jump magnitude), `failing_values.transition_index` (the 0-indexed position of the first violating pair). |

The `removals` array contains one full `RemovalRecord` per dropped trajectory:

```json
{
  "driver_id": "粤SW794X",
  "driver_idx": 7,
  "trajectory_index_within_driver": 142,
  "kind": "seeking",
  "which_invariant": 2,
  "failing_values": {"state": [999, 50, 1, 3], "axis": "x"},
  "n_states_before_removal": 14,
  "removal_reason_category": "out_of_bounds"
}
```

The **5% removal-rate warning threshold** (`config.REMOVAL_RATE_WARN_THRESHOLD`) fires in the
CLI log but never aborts the run. If you see the warning, check `counts_by_category` to see
which category dominates before deciding whether to trust the outputs.

---

## Auditing the downstream pipeline invariant

After a run completes, you can verify (and should, if you changed anything in the producer) that
the load-bearing invariant — *every seeking trajectory's `state[-1]` has a corresponding pickup
count in `pickup_3d`* — holds in the actual output. This is a project-level invariant stated by
the research team and the entire reason this tool exists. A one-line check:

```python
import pickle

with open("famail_temporal/source_data/passenger_seeking_trajs.pkl", "rb") as f:
    trajs = pickle.load(f)
with open("famail_temporal/source_data/pickup_dropoff_counts.pkl", "rb") as f:
    pd_counts = pickle.load(f)

ghosts = [
    (plate, i, tuple(traj[-1]))
    for plate, tlist in trajs.items()
    for i, traj in enumerate(tlist)
    if pd_counts.get(tuple(traj[-1]), (0, 0))[0] < 1
]
print(f"{len(ghosts)} ghost trajectories (should be 0)")
```

Under the new pipeline, this **must** print `0`. If it doesn't, something regressed in
`cli.py::run_generation` (specifically the count-rebuild step that derives final pickup counts
from kept trajectory endpoints).

---

## Required downstream actions after regeneration

Regeneration is only step 1 of getting a working end-to-end system. Do these in order:

### 1. Regenerate the preprocess cache (required)

`famail_temporal.preprocess` consumes the new source files and produces the `(48, 90, T)`
tensors + active mask + hat matrices + g0 fit that the algorithm actually uses.

```bash
python -m famail_temporal.preprocess --force
```

The `--force` flag is mandatory: without it, `preprocess.py` reuses its cached artifacts even
if the raw files changed. After a successful preprocess, `ls famail_temporal/cache/` should show
the expected `*_T4_thr0.5.pkl` files with recent timestamps.

### 2. Verify end-to-end (fast smoke test)

```bash
python -m famail_temporal.evaluation.runner \
    --name post-regen-smoke \
    --max-trajectories 200 -k 5 \
    --override MAX_ITERATIONS=5
```

This runs in ~15 seconds and writes a `report.md` under `famail_temporal/results/<timestamp>_post-regen-smoke/`. Open `report.md` and confirm:

- Dataset section shows `n_trajectories = 200`, `n_drivers > 0`, `n_active_units > 1000`.
- Fairness section shows non-NaN values for `F_spatial` / `F_causal` before and after.
- No `effective_alphas` override banner (unless you know your `ALPHA_FIDELITY` intent).
- No exceptions in the run log.

If the runner errors out with `ValueError: pickup_N ... must not contain negative values`,
the regeneration didn't actually fix the pipeline — that was the original bug that motivated
this tool. Open an issue; something went wrong in `cli.py`'s count-rebuild.

### 3. Retrain the v3 discriminator (required for F_fidelity experiments)

The multi-stream files (`ms_seeking_trajs.pkl`, `ms_driving_trajs.pkl`, `ms_profile_features.pkl`)
regenerate with the new pipeline's conventions. The existing v3 discriminator checkpoint at
`famail_temporal/discriminator_checkpoints/default/best.pt` was trained on the legacy files
and its calibration will drift on the new data. Retrain before running any F_fidelity-enabled
experiments.

Until retraining is done, you can still run **fairness-only experiments** by setting
`ALPHA_FIDELITY = 0` (either in `config.py` or via `--override ALPHA_FIDELITY=0`). The runner
also automatically falls back to `nn.Identity` + `alpha_fidelity = 0` if the checkpoint isn't
found, so a missing checkpoint is not fatal — but don't silently report F_fidelity numbers
from an uncalibrated discriminator.

---

## Parameter sweeps and ablations

The tool has **no runtime config flags**. Every decision that affects the output
(neighborhood size, grid resolution, day filter, etc.) lives in
[`config.py`](config.py) as an uppercase constant.

If you want to experiment with a different convention:

1. Edit the constant in `config.py`.
2. Re-run the tool — it will regenerate all affected files with the new convention.
3. Note the change in `processing_metadata.json` (the `config_snapshot` block captures the current values, so a quick diff against a previous run's metadata tells you exactly what changed).
4. Re-run `python -m famail_temporal.preprocess --force`.

**Research-defensibility rationale for the no-flags design:** every config flag is a silent
failure mode. Two researchers running the tool with different flag combinations would produce
subtly different source files that downstream analysis cannot distinguish. Making convention
changes require a code edit + git commit means the decision is audit-visible in history, and
`config_snapshot` in the metadata makes it audit-visible in the output.

---

## Common pitfalls

### 1. Forgetting to regenerate the preprocess cache

Symptom: `famail_temporal.evaluation.runner` still produces the old numbers despite running the
source-generation tool.

Cause: `preprocess.py` finds its cache files and reuses them.

Fix: `python -m famail_temporal.preprocess --force`.

### 2. Running with only partial raw data

Symptom: `ValueError: compute_global_bounds: empty input` or garbage GPS bounds in the metadata.

Cause: One or more of the 3 `taxi_record_*.pkl` files is missing or was replaced with a 0-byte placeholder.

Fix: Verify all 3 files are present and non-empty. The tool requires all 3 months (07, 08, 09) to establish the global GPS bounding box correctly.

### 3. High removal rate

Symptom: `Per-trajectory removal rate 12.34% exceeds threshold 5.00%` in the CLI log.

Cause: Varies — see the `removal_summary.counts_by_category` in `processing_metadata.json`.

Fix: Inspect the `removals` array. If most removals are `out_of_bounds` with extreme `x`/`y` values, you likely have a raw-data issue (GPS coordinates far outside Shenzhen). If most are `degenerate_length`, the upstream transition detection may be fragmenting trips at day boundaries — rare but possible.

### 4. Systemic invariant raised

Symptom: `SystemicInvariantError: #5: sum(pickup_counts)=X != n_seeking=Y` or similar — the tool aborts before writing any output.

Cause: A bug in the producer pipeline (e.g., count rebuild step missed a trajectory).

Fix: This should never happen with the shipped tool. If it does, open an issue and capture
`stderr` — the exception message names the specific systemic invariant number from §6 of the
design spec.

### 5. Driver count ≠ 50

Symptom: `SystemicInvariantError: #6: got N unique drivers; expected 50`.

Cause: The raw GPS files don't contain exactly 50 unique drivers (either some drivers are missing or there are extra/duplicate drivers).

Fix: If intentional (e.g., you're testing on a subset), pass `expect_n_drivers=<N>` programmatically — the CLI doesn't expose this flag because it's intended as a testing convenience. If unintentional, audit the raw files.

---

## Where to look next

- **Architecture:** [`README.md`](README.md) — module layout, design choices, architecture diagram.
- **Design rationale:** [`docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md`](../../../docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md) — why each decision was made, with cross-references to the three root-cause bugs it addresses.
- **Implementation plan:** [`docs/superpowers/plans/2026-04-20-unified-source-data-generation.md`](../../../docs/superpowers/plans/2026-04-20-unified-source-data-generation.md) — per-task TDD breakdown, for anyone maintaining the tool.
- **CHANGELOG entry:** [`CHANGELOG.md`](../../../CHANGELOG.md) — `2026-04-20 — Unified source-data generation tool` section documents all semantic changes vs. the legacy tools.
- **Downstream:** [`famail_temporal/evaluation/EVALUATION_QUICKSTART.md`](../../evaluation/EVALUATION_QUICKSTART.md) — how to run experiments with the regenerated source data.

If something in the output looks wrong, `processing_metadata.json`'s `git_sha` + `config_snapshot` + `bounds` are almost always enough to reproduce the run from a clean checkout.
