# SF `source_data` Schema Contract (Task 3.0)

The SF build (`sf_build.py`) must emit these files into the SF source dir so that
`preprocess.py` → `cache/` → `DataBundle.load()` work **with no change to
`loader.py`, `preprocess.py`, `aggregation.py`, or `demographics.py`** — only a
city-switchable `config` (D1/D4). All `(x, y)` are **1-indexed**; `preprocess`/`loader`
subtract 1. Grid = 32×30 (D1), `T = 24` hourly, trajectory `time_bucket ∈ 1..288` (D4).

## Consumed by `preprocess.py` (`load_raw`)

| File | Format | Notes |
|---|---|---|
| `pickup_dropoff_counts.pkl` | `dict[(x, y, time_bucket, day)] -> (pickup:int, dropoff:int)` | 1-idx `x,y`; 1-idx `time_bucket` (1..288); `day` = calendar-day int. `dataset_n_days` counts distinct `key[3]`. `aggregate_pickup_dropoff` → `pickup_3d/dropoff_3d (32,30,24)` via `(tb-1)//12` → hour. |
| `active_taxis_5x5_hourly.pkl` | bundle `{"data": dict[(x, y, hour, day)] -> int, "stats":…, "config":…, "version":…}` | 1-idx `x,y`; **0-idx `hour` (0..23)**. `loader` unwraps `["data"]`. Value = active-taxi count (5×5 neighborhood supply). Floored to `SUPPLY_FLOOR`. |
| `cell_demographics.pkl` | `{"demographics_grid": (32,30,3) float, "feature_names": ["AvgHousingPricePerSqM","CompPerCapita","MigrantRatio"]}` | **NaN** for inactive/non-residential cells (the active-mask finite-demographics filter excludes them). Reusing the Shenzhen feature *names* (filled with ACS values) means `enrich_demographics` passes them through and `config.DEMOGRAPHIC_FEATURES` needs no change. |
| `grid_to_district_mapping.pkl` | `{"valid_mask": (32,30) bool}` | `True` = in-bounds/mappable cell. SF: `True` over land cells; water/out-of-extent → `False` (or rely on the demographics-NaN filter). |

`compute_active_mask(active_taxis_3d, valid_mask, demographics_selected)` →
`mask_3d`: a `(cell,t)` unit is active iff `active_taxis > ACTIVE_SUPPLY_THRESHOLD`
**and** `valid_mask[x,y]` **and** all selected demographics finite. Target `n_active ≈ 10–12k`.

## Consumed by `loader.py` (direct `source_data` reads)

| File | Format | Notes |
|---|---|---|
| `passenger_seeking_trajs.pkl` | `dict[plate_id:str] -> List[traj]`, `traj = List[[x, y, time_bucket, day]]` | 1-indexed; `loader._parse_trajectory` subtracts 1 from x,y; needs ≥2 states. The editor's trajectory corpus. |
| `driver_index_mapping.pkl` | `{"plate_to_idx": {plate:str -> int}, "idx_to_plate": {int -> plate}}` | int `driver_idx` in `[0, n_drivers)`. SF: n_drivers ≈ 536. |
| `ms_seeking_trajs.pkl` | `dict[driver_idx:int] -> List[traj]` | 1-indexed `[x,y,t,d]`. Discriminator seeking stream. |
| `ms_driving_trajs.pkl` | `dict[driver_idx:int] -> List[traj]` | 1-indexed. Discriminator driving stream. |
| `ms_profile_features.pkl` | bundle `{"features": {idx->raw 11d}, "features_normalized": {idx->z 11d}, "feature_names":[…11], "normalization":{"mean","std"}, "n_features":11}` | `loader` reads `features_normalized`; the retrain reads `features` + `normalization`. |
| `ms_seeking_calendar_days.pkl` | `dict[driver_idx:int] -> List[int]` | calendar-day ints per driver (seeking). |
| `ms_driving_calendar_days.pkl` | `dict[driver_idx:int] -> List[int]` | calendar-day ints per driver (driving). |

(`calendar_day_map.pkl` + `metadata.json` are written by the Shenzhen tool for provenance;
`preprocess` derives its own `metadata` cache artifact — `n_days` from the pickup_dropoff keys.)

## Build order (maps to Phase-3 tasks)

1. **3.1** raw loader → tidy `[driver_id, lat, lon, occupancy, time_utc]`.
2. **3.2** segment (occupancy + gap) → seeking/driving trajectories (gridded 1-idx `[x,y,tb,day]`) + pickup/dropoff events; per-driver calendar days.
3. **3.3** demographics → `(32,30,3)` grid named `{AvgHousingPricePerSqM, CompPerCapita, MigrantRatio}` (areal interp, D2).
4. **3.4** counts → `pickup_dropoff_counts.pkl`, `active_taxis_5x5_hourly.pkl`, `grid_to_district_mapping.pkl`.
5. **3.5** multi-stream → `ms_*` + `passenger_seeking_trajs.pkl` + `driver_index_mapping.pkl` + 11-dim profiles.
6. **3.6** city-switch `config`; run `preprocess`; assert `DataBundle.load()` + baseline `F_spatial/F_causal`.

## Profile features (11-dim, for 3.5 / retrain)

Per driver, from raw GPS: home cell (x,y), shift-start/-end percentiles, modal pickup cell (x,y),
avg seek distance, avg seek duration, avg drive distance, avg drive duration, trips/day. Stored raw
+ z-normalized with the `mean`/`std` used (the discriminator re-normalizes from raw).
