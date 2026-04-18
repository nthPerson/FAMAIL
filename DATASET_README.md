# FAMAIL Dataset Archive

This archive contains the datasets used by the **FAMAIL** (Fairness-Aware Multi-Agent Imitation Learning) project — a trajectory-editing pipeline for improving spatial equity in Shenzhen taxi service.

**Study area**: Shenzhen, China. Grid: 48 (lat) × 90 (lon) cells (~1.1 km each). Time: 288 five-minute buckets/day. Drivers: 50 expert taxis across weekdays in July–September 2016.

---

## Coordinate & Indexing Conventions (read first)

- Grid origin `(0, 0)` is the **south-west** corner.
- `x_grid` ∈ [0, 47] = latitude (0=south, 47=north). `y_grid` ∈ [0, 89] = longitude (0=west, 89=east).
- `cell_id = x_grid * 90 + y_grid`.
- **Files under `source_data/` use 1-indexed coords** `[1..48] × [1..90]`. The loader (`trajectory_modification/data_loader.py`) subtracts 1 before use.
- **Trajectory state vectors are 0-indexed** — read directly as `(x_grid, y_grid)`.
- Time bucket: `time_bucket = hour * 12 + minute // 5`, range [0, 287].

---

## Directory Layout

```
source_data/                                  # Aggregated per-cell / per-time data + trajectories
discriminator/multi_stream/extracted_data/    # Pre-extracted streams for the discriminator
raw_data/                                     # Raw GPS records (source for extraction)
```

---

## 1. Core Pipeline Data (`source_data/`)

### Required for trajectory modification & objective function

| File | Size | Role | Structure |
|---|---|---|---|
| `passenger_seeking_trajs_45-800.pkl` | 11 MB | **Trajectories to edit.** Primary input to `TrajectoryModifier`. Consumed by the ST-iFGSM loop and the fidelity discriminator. | `Dict[driver_id] -> List[trajectory]`, where each trajectory is a list of `[x, y, time_bucket, day]` rows. ~50 drivers, 45–800 trajectories per driver. |
| `pickup_dropoff_counts.pkl` | 128 MB | **Demand signal.** Used by spatial fairness (Gini of DSR), causal fairness (for `g(D, x)` fitting), and as input to the modifier's gradient. | `Dict[(x, y, time_bucket, day)] -> (pickup_count, dropoff_count)`. 1-indexed coords. |
| `active_taxis_5x5_hourly.pkl` | 6.7 MB | **Supply signal.** Denominator of the Demand-Service Ratio; service measure `S` in causal fairness. | `Dict[(x, y, hour, day)] -> taxi_count`. 1-indexed; 24 hours x 5 days. |
| `cell_demographics.pkl` | 850 KB | **Confounders for causal fairness** (Option B/C). District-level demographic features joined to the 48x90 grid. | Dict with keys `demographics_grid` (ndarray shape `(48, 90, 13)`, float64) and `feature_names` (list of 13 strings). Schema in `cell_demographics.sample.json`. |
| `grid_to_district_mapping.pkl` | 80 KB | Maps grid cells -> Shenzhen districts; also provides the `valid_mask` of cells that have demographic coverage. | Dict with `grid_to_district` (`(48, 90)` int), `valid_mask` (`(48, 90)` bool), `district_names` (10 strings). Schema in `grid_to_district_mapping.sample.json`. |

### Optional (used by dashboards / alternate time resolutions)

| File | Size | Role | Structure |
|---|---|---|---|
| `all_trajs.pkl` | 389 MB | Full trajectory dataset (with rich state vectors) for visualization and attribution dashboards. Not required for core optimization. | `Dict[driver_id] -> List[List[state_vector]]`, each state vector is 126-dim. See **State Vector Schema** below. |
| `latest_traffic.pkl` | 748 MB | Traffic speed, volume, and wait times per cell/time. Used by the `new_all_trajs` trajectory-generation pipeline and visualization app. | Dict keyed by spatiotemporal tuple -> traffic feature vector. |
| `latest_volume_pickups.pkl` | 219 MB | Volume/pickup aggregations used by the `new_all_trajs` pipeline. | Spatiotemporal dict similar to traffic. |
| `active_taxis_5x5_time_bucket.pkl` | 78 MB | Fine-grained supply (288 buckets/day) as an alternative to hourly aggregation. | `Dict[(x, y, time_bucket, day)] -> taxi_count`. |
| `active_taxis_5x5_daily.pkl` | 217 KB | Daily-aggregated supply. Reference/exploratory. | `Dict[(x, y, day)] -> taxi_count`. |
| `active_taxis_5x5_all.pkl` | 44 KB | Single-period aggregation. Reference/exploratory. | `Dict[(x, y)] -> taxi_count`. |
| `all_demographics_by_district.csv` | 1.5 KB | District-level demographic source data. Used to regenerate `cell_demographics.pkl`. | 10 rows x 13 numeric columns (AreaKm2, PopDensity, HousingPrice, GDP, employment, etc.). |
| `demographics_by_district.csv` | 1.4 KB | Alternate/filtered demographics CSV. Reference only. | Same schema as above. |
| `train_airport.pkl` | 1.1 KB | Airport-region marker used by the `new_all_trajs` generation pipeline. | Small reference dict. |

---

## 2. Discriminator Multi-Stream Data (`discriminator/multi_stream/extracted_data/`)

Pre-extracted, ML-ready streams for the V3 multi-stream discriminator (driving/seeking/profile). Loaded by `MultiStreamDataLoader` in `trajectory_modification/data_loader.py`.

| File | Size | Role | Structure |
|---|---|---|---|
| `driving_trajs.pkl` | 11 MB | Empty-taxi "driving-mode" trajectories per driver. Discriminator input stream. | `Dict[driver_index: int] -> List[trajectory]`, 1-indexed coords. 50 drivers. |
| `seeking_trajs.pkl` | 13 MB | Empty-taxi "seeking-mode" trajectories per driver. Discriminator input stream. | `Dict[driver_index: int] -> List[trajectory]`, 1-indexed. 50 drivers. |
| `profile_features.pkl` | 13 KB | Per-driver 11-dim behavioral profile (z-score normalized). Discriminator profile stream. | `Dict['features_normalized'] -> Dict[driver_index: int] -> ndarray(11,) float32`. |
| `seeking_calendar_days.pkl` | 102 KB | Calendar-day index aligned with each seeking trajectory. Temporal encoding. | `Dict[driver_index: int] -> List[day_index]`. Indices match trajectory count. |
| `driving_calendar_days.pkl` | 129 KB | Same, for driving trajectories. | `Dict[driver_index: int] -> List[day_index]`. |
| `calendar_day_map.pkl` | 1 KB | Lookup mapping of dates -> calendar-day indices. | Dict / lookup table. |
| `extraction_metadata.json` | 5 KB | Provenance metadata for the extraction run (date, driver count, feature config, validation stats). | JSON. |
| `DATA_DICTIONARY.md` | — | Full field-level documentation of the extracted streams. | Markdown. |

---

## 3. Raw GPS Data (`raw_data/`)

Source records. Only needed if you intend to re-run the multi-stream extraction pipeline (`discriminator/multi_stream/extraction/`).

| File | Size | Role | Structure |
|---|---|---|---|
| `taxi_record_07_50drivers.pkl` | ~128 MB | Raw GPS for July 2016. | `Dict[plate_id] -> List[[lat, lon, timestamp, passenger_flag, ...]]` over ~50 drivers. |
| `taxi_record_08_50drivers.pkl` | ~128 MB | Raw GPS for August 2016. | Same schema. |
| `taxi_record_09_50drivers.pkl` | ~152 MB | Raw GPS for September 2016. | Same schema. |
| `demographics_by_district.csv` | 1.4 KB | Raw district demographics (source for `source_data/` CSVs). | CSV. |
| `grid_to_district_ArcGIS_table_raw.csv` | — | Raw ArcGIS grid-to-district mapping (source for `grid_to_district_mapping.pkl`). Note: the raw table uses a different (50x90) grid and may need re-alignment. | CSV. |

---

## State Vector Schema (for `all_trajs.pkl`)

Each timestep in a trajectory is a 126-element vector:

| Indices | Content |
|---|---|
| 0–1 | `(x_grid, y_grid)` — 0-indexed |
| 2 | `time_bucket` in [0, 287] |
| 3 | `day_index` |
| 4–24 | POI distances (21 features) |
| 25–49 | Pickup counts in 5x5 neighborhood (25 features) |
| 50–74 | Traffic volume in 5x5 neighborhood |
| 75–99 | Speed in 5x5 neighborhood |
| 100–124 | Wait times in 5x5 neighborhood |
| 125 | Action code |

---

## Minimal Working Set

If disk space is a concern, the **minimum files** needed to run trajectory modification with all three objective terms are:

```
source_data/passenger_seeking_trajs_45-800.pkl
source_data/pickup_dropoff_counts.pkl
source_data/active_taxis_5x5_hourly.pkl
source_data/cell_demographics.pkl
source_data/grid_to_district_mapping.pkl
discriminator/multi_stream/extracted_data/*.pkl
discriminator/multi_stream/extracted_data/extraction_metadata.json
```

Total: ~160 MB. Add `all_trajs.pkl` (~389 MB) for dashboards and attribution visualization.

---

## Loading the Data

All `.pkl` files are Python pickles (the project's canonical on-disk format). Load with `pickle.load(open(path, "rb"))`, or use the project's bundle loader:

```python
from trajectory_modification import DataBundle
bundle = DataBundle.load_default()   # loads the required files from source_data/
```

CSVs are plain UTF-8 and load with `pandas.read_csv`. Only open pickle files you trust — they can execute arbitrary code on load.

---

## Notes

- All `.pkl`, `.csv`, `.json`, `.npz`, `.pt`, `.pth` files are gitignored in the repo — this archive is the canonical transfer vehicle.
- Sample JSON files (`*.sample.json`) next to demographic pickles show a trimmed example of the full structure and are safe to open in any text editor.
- Coordinate convention is the single most common source of bugs. Re-read the top of this README before touching new loader code.
