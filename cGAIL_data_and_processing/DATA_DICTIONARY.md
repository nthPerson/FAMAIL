# cGAIL Trajectory Data Dictionary

This document describes the structure and provenance of the trajectory datasets used in the cGAIL experiments. It consolidates the exploratory notes from
[cGAIL_data_and_processing/cgail_trajs2_data_exploration.ipynb](cGAIL_data_and_processing/cgail_trajs2_data_exploration.ipynb),
[cGAIL_data_and_processing/all_trajs_data_exploration.ipynb](cGAIL_data_and_processing/all_trajs_data_exploration.ipynb), and the feature-building logic in
[cGAIL_data_and_processing/net-dis-cGAIL.ipynb](cGAIL_data_and_processing/net-dis-cGAIL.ipynb).

## Datasets at a Glance

- **all_trajs.pkl** (the feature-augmented trajectories sometimes referred to as `state_feature_trajs`)
  - Top-level: dictionary with the same 50 driver keys.
  - Value per driver: list of trajectories; each trajectory mirrors the sequencing of `cgail_trajs2.pkl` but with expanded per-step features and an action code.
  - Feature construction happens in `process_trajectories` and `processing_state_features` inside net-dis-cGAIL.

- **cgail_trajs2.pkl** (small sample of trajectory data; not to be used in FAMAIL analysis)
  - Top-level: dictionary with 50 driver keys.
  - Value per driver: list of trajectories.
  - Each trajectory: list of raw states.
  - Raw state layout: `[driver_id, x_grid, y_grid, time_bucket, day_index]`.

## State Vector Schema (all_trajs.pkl / state_feature_trajs)

Each `step_vector` has length 126. Index meanings (zero-based):

- **0**: `x_grid` — grid index for longitude.
- **1**: `y_grid` — grid index for latitude.
- **2**: `time_bucket` — discretized time-of-day slot.
- **3**: `day_index` — day indicator as stored in the trajectories.
- **4–24**: `poi_manhattan_distance[21]` — Manhattan distances to each train/airport point of interest loaded from `Data/features_condition/train_airport.pkl`.
- **25–49**: `pickup_count_norm[25]` — normalized pickup counts over a 5×5 window centered on `(x, y)` at `(t, day)` using `volume[(x,y,t,day)][0]` from `Data/features_condition/latest_volume_pickups.pkl` (defaults applied when missing).
- **50–74**: `traffic_volume_norm[25]` — normalized trip volumes over the same 5×5 window using `volume[(x,y,t,day)][1]` from `latest_volume_pickups.pkl` (defaults when missing).
- **75–99**: `traffic_speed_norm[25]` — normalized traffic speeds over the 5×5 window using `traffic[(x,y,t,day)][0]` from `Data/features_condition/latest_traffic.pkl` (defaults when missing).
- **100–124**: `traffic_wait_norm[25]` — normalized traffic waiting times over the 5×5 window using `traffic[(x,y,t,day)][1]` from `latest_traffic.pkl` (defaults when missing).
- **125**: `action_code` — movement label derived by `judging_action(current_xy, next_xy)`.

### Action Code Map

- `0`: move (x same, y increases)
- `1`: move (x increases, y increases)
- `2`: move (x increases, y same)
- `3`: move (x increases, y decreases)
- `4`: move (x same, y decreases)
- `5`: move (x decreases, y decreases)
- `6`: move (x decreases, y same)
- `7`: move (x decreases, y increases)
- `8`: stay (x unchanged, y unchanged)
- `9`: stop (triggered when current or next position is `(0, 0)`).

### Ordering Rationale

`processing_state_features` builds `[poi distances] + [25 pickup counts] + [25 trip volumes] + [25 speeds] + [25 waits]`; `process_trajectories` then prepends the base state `[x, y, t, day]` and appends the `action_code`. Because the downstream pipeline slices the base+feature portion as 125 elements before adding user metadata, the PoI list length resolves to 21 (`4 + 21 + 25*4 = 125`).

### Normalization Notes

- Pickup count baseline/scale: `-1.7411… / 8.6891…` when missing.
- Trip volume baseline/scale: `-241.1649… / 864.8101…` when missing.
- Traffic speed baseline/scale: `-0.009096… / 0.0077867…` when missing.
- Traffic wait baseline/scale: `-9.2149… / 20.8396…` when missing.
- These constants are embedded in `processing_state_features` and are assumed to be derived from dataset statistics prior to normalization.

## Raw State Schema (cgail_trajs2.pkl)

Each raw state in `cgail_trajs2.pkl` is a 5-tuple:

1. `driver_id` — identifier matching the top-level key.
2. `x_grid` — grid index for longitude.
3. `y_grid` — grid index for latitude.
4. `time_bucket` — discretized time-of-day slot.
5. `day_index` — day indicator.

These tuples are the inputs to feature construction; they become `[x, y, t, day]` after the driver_id is stripped during processing.

## Provenance of Feature Inputs

- Train/airport PoI coordinates: `Data/features_condition/train_airport.pkl` (used for Manhattan distances).
- Pickup/volume features: `Data/features_condition/latest_volume_pickups.pkl` via the `volume` dictionary.
- Traffic speed/wait features: `Data/features_condition/latest_traffic.pkl` via the `traffic` dictionary.
- Action labels: computed on-the-fly by comparing consecutive positions inside `process_trajectories` in `net-dis-cGAIL.ipynb`.

## Usage Pointers

- When sampling or inspecting `all_trajs.pkl`, indices 0–124 are deterministic features; index 125 is the supervised action label.
- To align a feature row back to its raw source, match `(x_grid, y_grid, time_bucket, day_index)` with the dictionaries above; the driver key provides the identity context.
- The ordering of the 5×5 blocks is row-major over `x_range = x-2…x+2`, `y_range = y-2…y+2`.
