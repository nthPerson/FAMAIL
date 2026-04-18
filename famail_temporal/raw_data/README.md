# famail_temporal/raw_data/

This directory holds the 10 raw `.pkl` files required by `python -m famail_temporal.preprocess`. These files are gitignored (binary data) and must be copied manually from the main repository's data directories.

## Required Files

| Filename | Source (relative to repo root) |
|---|---|
| `pickup_dropoff_counts.pkl` | `source_data/pickup_dropoff_counts.pkl` |
| `active_taxis_5x5_hourly.pkl` | `source_data/active_taxis_5x5_hourly.pkl` |
| `cell_demographics.pkl` | `source_data/cell_demographics.pkl` |
| `grid_to_district_mapping.pkl` | `source_data/grid_to_district_mapping.pkl` |
| `passenger_seeking_trajs_45-800.pkl` | `source_data/passenger_seeking_trajs_45-800.pkl` |
| `ms_driving_trajs.pkl` | `discriminator/multi_stream/extracted_data/driving_trajs.pkl` |
| `ms_seeking_trajs.pkl` | `discriminator/multi_stream/extracted_data/seeking_trajs.pkl` |
| `ms_profile_features.pkl` | `discriminator/multi_stream/extracted_data/profile_features.pkl` |
| `ms_seeking_calendar_days.pkl` | `discriminator/multi_stream/extracted_data/seeking_calendar_days.pkl` |
| `ms_driving_calendar_days.pkl` | `discriminator/multi_stream/extracted_data/driving_calendar_days.pkl` |

## Copying (from repo root)

```bash
cp source_data/pickup_dropoff_counts.pkl       famail_temporal/raw_data/
cp source_data/active_taxis_5x5_hourly.pkl     famail_temporal/raw_data/
cp source_data/cell_demographics.pkl            famail_temporal/raw_data/
cp source_data/grid_to_district_mapping.pkl     famail_temporal/raw_data/
cp source_data/passenger_seeking_trajs_45-800.pkl famail_temporal/raw_data/
cp discriminator/multi_stream/extracted_data/driving_trajs.pkl          famail_temporal/raw_data/ms_driving_trajs.pkl
cp discriminator/multi_stream/extracted_data/seeking_trajs.pkl          famail_temporal/raw_data/ms_seeking_trajs.pkl
cp discriminator/multi_stream/extracted_data/profile_features.pkl       famail_temporal/raw_data/ms_profile_features.pkl
cp discriminator/multi_stream/extracted_data/seeking_calendar_days.pkl  famail_temporal/raw_data/ms_seeking_calendar_days.pkl
cp discriminator/multi_stream/extracted_data/driving_calendar_days.pkl  famail_temporal/raw_data/ms_driving_calendar_days.pkl
```

## Notes

- The first 4 files are used by `preprocess.py` to build the cached (48, 90, T) tensors, active mask, g0 fit, and hat matrices.
- `passenger_seeking_trajs_45-800.pkl` is loaded at `DataBundle.load()` time (not during preprocessing) to build trajectory objects.
- The 5 `ms_*` files are loaded at `DataBundle.load()` time to populate the `MultiStreamData` container for the fidelity discriminator.
- All coordinates in raw data files are **1-indexed** [1-48, 1-90]; the loader subtracts 1 to produce 0-indexed grid coordinates.
