#!/usr/bin/bash

# Copy raw data (see famail_temporal/raw_data/README.md)
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

# Copy discriminator checkpoint
cp discriminator/model/checkpoints/20260316_223817/best.pt famail_temporal/discriminator_checkpoints/default/best.pt
