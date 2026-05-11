# famail_temporal/source_data/

This directory holds the source datasets consumed by `famail_temporal` at load time:

- `python -m famail_temporal.preprocess` reads 4 of them to build the `(48, 90, T)` cache tensors.
- `famail_temporal.data.loader.DataBundle.load()` reads 6 more at load time to populate the trajectory list, multi-stream discriminator context, and profile features.

All `.pkl` files in this directory are gitignored (binary, large). The directory itself is tracked via `.gitkeep` and this README.

> **Naming note.** The repo-root `raw_data/` directory holds the *actually raw* taxi GPS pickle files (`taxi_record_*.pkl`), which are the **input** to `famail_temporal.data.source_generation`. This `famail_temporal/source_data/` directory holds the **output** of that tool — the source datasets consumed by the algorithm. "Raw" and "source" refer to two different stages of the pipeline.

---

## Files

Files fall into 3 groups depending on how they're provisioned:

### Group A — Produced by the unified source-generation tool (8 files)

These regenerate every time you run:

```bash
python -m famail_temporal.data.source_generation \
    --input-dir raw_data/ \
    --output-dir famail_temporal/source_data/
```

| Filename | Consumer | Purpose |
|---|---|---|
| `pickup_dropoff_counts.pkl` | `preprocess.py` | `pickup_3d`, `dropoff_3d` tensors |
| `active_taxis_5x5_hourly.pkl` | `preprocess.py` | `active_taxis_3d` tensor (bundle format with `data`/`stats`/`config`/`version`) |
| `passenger_seeking_trajs.pkl` | `loader.py::_load_trajectories` | `bundle.trajectories` |
| `ms_driving_trajs.pkl` | `loader.py::_load_multi_stream` | Discriminator driving stream |
| `ms_seeking_trajs.pkl` | `loader.py::_load_multi_stream` | Discriminator seeking stream |
| `ms_profile_features.pkl` | `loader.py::_load_multi_stream` | Discriminator per-driver profile (11 features, z-score normalized) |
| `ms_seeking_calendar_days.pkl` | `loader.py::_load_multi_stream` | Reserved (loaded but currently unconsumed) |
| `ms_driving_calendar_days.pkl` | `loader.py::_load_multi_stream` | Reserved (loaded but currently unconsumed) |

Each run also writes `driver_index_mapping.pkl` (sidecar for joining plate-keyed and int-keyed files) and `processing_metadata.json` (run-level audit record with config snapshot, GPS bounds, git SHA, and per-trajectory removal summary).

See [`../data/source_generation/SOURCE_DATASET_GENERATION_QUICKSTART.md`](../data/source_generation/SOURCE_DATASET_GENERATION_QUICKSTART.md) for operator instructions and [`../data/source_generation/README.md`](../data/source_generation/README.md) for architectural details.

### Group B — External inputs (2 files, manually provisioned)

These are NOT produced by the source-generation tool. They come from census / geographic data in the repo-root `source_data/` directory and must be copied in once per checkout:

| Filename | Source (relative to repo root) | Consumer |
|---|---|---|
| `cell_demographics.pkl` | `source_data/cell_demographics.pkl` | `preprocess.py`; `data/demographics.py` |
| `grid_to_district_mapping.pkl` | `source_data/grid_to_district_mapping.pkl` | `preprocess.py` (Shenzhen boundary mask) |

Copy with:

```bash
cp source_data/cell_demographics.pkl       famail_temporal/source_data/
cp source_data/grid_to_district_mapping.pkl famail_temporal/source_data/
```

---

## Provisioning a fresh checkout

1. Ensure the raw taxi GPS files are at repo-root `raw_data/taxi_record_{07,08,09}_50drivers.pkl`.
2. Copy the 2 external inputs from `source_data/` (see Group B above).
3. Run `python -m famail_temporal.data.source_generation` to produce the 8 Group-A files.
4. Run `python -m famail_temporal.preprocess --force` to build the cache tensors.
5. Verify with `pytest famail_temporal/tests/ -q` — all fast tests should pass, including `test_databundle_load_real_data`.

---

## Conventions (all producer outputs)

- `x`, `y` coordinates are **1-indexed** [1..48], [1..90] on disk; the loader subtracts 1 to produce 0-indexed grid coords.
- `time_bucket` is **1-indexed** [1..288] (5-minute resolution).
- `hour` (only in `active_taxis_5x5_hourly.pkl`) is **0-indexed** [0..23].
- `day_index` is **1-indexed** Mon=1 .. Fri=5 (Saturday and Sunday are excluded from all outputs).
