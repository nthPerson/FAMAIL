# famail_temporal/source_data/

This directory holds the source datasets consumed by `famail_temporal` at load time:

- `python -m famail_temporal.preprocess` reads 4 of them to build the `(48, 90, T)` cache tensors.
- `famail_temporal.data.loader.DataBundle.load()` reads 6 more at load time to populate the trajectory list, multi-stream discriminator context, and profile features.

All data files in this directory are gitignored. The directory itself is tracked via `.gitkeep` and this README.

## Provisioning a fresh checkout

**Recommended:** fetch from the project's public HuggingFace dataset:

```bash
python -m famail_temporal.fetch_data
python -m famail_temporal.preprocess
```

This downloads `source_data/`, `raw_data/`, and the discriminator checkpoint in one shot. No HF token required — the dataset is public. Add `--skip-raw` if you don't need the 418 MB raw GPS bundle. See [`../fetch_data.py`](../fetch_data.py) for flags.

Dataset: <https://huggingface.co/datasets/nthPerson/famail-temporal-data>

**Alternative — regenerate from raw GPS:**

1. Obtain the 3 raw files `raw_data/taxi_record_{07,08,09}_50drivers.pkl` from the HF dataset (`python -m famail_temporal.fetch_data --skip-raw=false` covers this).
2. Obtain the 2 external inputs `cell_demographics.pkl` and `grid_to_district_mapping.pkl` from the HF dataset and place them in this directory.
3. Run `python -m famail_temporal.data.source_generation --input-dir raw_data/ --output-dir famail_temporal/source_data/` to produce the 8 generated files.
4. Run `python -m famail_temporal.preprocess --force` to build the cache tensors.

> **Naming note.** "Raw" and "source" refer to two different stages of the pipeline. The `raw_data/` directory at the repository root holds the raw taxi GPS files (the *input* to `source_generation`). This `famail_temporal/source_data/` directory holds the *output* of that tool — the datasets the algorithm actually consumes.

---

## File inventory

**Produced by the source-generation tool (8 files + 2 sidecars):**

| Filename | Consumer | Purpose |
|---|---|---|
| `pickup_dropoff_counts.pkl` | `preprocess.py` | `pickup_3d`, `dropoff_3d` tensors |
| `active_taxis_5x5_hourly.pkl` | `preprocess.py` | `active_taxis_3d` (bundle: `data`/`stats`/`config`/`version`) |
| `passenger_seeking_trajs.pkl` | `loader.py` | `bundle.trajectories` |
| `ms_driving_trajs.pkl` | `loader.py` | Discriminator driving stream |
| `ms_seeking_trajs.pkl` | `loader.py` | Discriminator seeking stream |
| `ms_profile_features.pkl` | `loader.py` | Per-driver profile (11 features, z-score normalized) |
| `ms_seeking_calendar_days.pkl` | `loader.py` | Reserved (loaded but currently unconsumed) |
| `ms_driving_calendar_days.pkl` | `loader.py` | Reserved (loaded but currently unconsumed) |
| `driver_index_mapping.pkl` (sidecar) | — | Joins plate-keyed and int-keyed files |
| `processing_metadata.json` (sidecar) | — | Run-level audit (config snapshot, GPS bounds, git SHA, removals) |

See [`../data/source_generation/SOURCE_DATASET_GENERATION_QUICKSTART.md`](../data/source_generation/SOURCE_DATASET_GENERATION_QUICKSTART.md) for operator instructions and [`../data/source_generation/README.md`](../data/source_generation/README.md) for architectural details.

**External inputs (2 files):** Not produced by the source-generation tool — sourced from Shenzhen census and ArcGIS district data.

| Filename | Consumer |
|---|---|
| `cell_demographics.pkl` | `preprocess.py`; `data/demographics.py` |
| `grid_to_district_mapping.pkl` | `preprocess.py` (Shenzhen boundary mask) |

Both ship inside the HuggingFace dataset's `source_data/` directory and land here automatically via `fetch_data`.

---

## Conventions (all producer outputs)

- `x`, `y` coordinates are **1-indexed** [1..48], [1..90] on disk; the loader subtracts 1 to produce 0-indexed grid coords.
- `time_bucket` is **1-indexed** [1..288] (5-minute resolution).
- `hour` (only in `active_taxis_5x5_hourly.pkl`) is **0-indexed** [0..23].
- `day_index` is **1-indexed** Mon=1 .. Fri=5 (Saturday and Sunday are excluded from all outputs).
