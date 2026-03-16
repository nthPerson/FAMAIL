# Extracted Data Dictionary

Generated: 2026-03-16 | Source: `discriminator/multi_stream/extraction/`

## Overview

This directory contains dual-stream trajectory data and driver profile features extracted from raw Shenzhen taxi GPS records (July–September 2016, weekdays only). The data supports the multi-stream ST-SiameseNet discriminator (Ren et al., KDD 2020) with three input streams: seeking trajectories (LSTM_S), driving trajectories (LSTM_D), and profile features (FCN_P).

## Source Data

| Parameter | Value |
|-----------|-------|
| Raw input files | `taxi_record_07_50drivers.pkl`, `taxi_record_08_50drivers.pkl`, `taxi_record_09_50drivers.pkl` |
| Total raw records | 8,123,761 |
| Drivers | 50 |
| Study period | July–September 2016, weekdays (Mon–Fri) |
| GPS bounds | lat [22.4425, 22.8700], lon [113.7501, 114.5582] |

## Extraction Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `grid_size` | 0.01° | ~1.1 km cells |
| `time_interval` | 5 min | 288 buckets/day |
| `x_grid_offset` | +1 | 1-indexed x output (compatible with `data_loader.py` which subtracts 1) |
| `y_grid_offset` | +1 | 1-indexed y output |
| `min_segment_states` | 5 | Minimum states after deduplication |
| `min_segment_duration_sec` | 300 | 5-minute minimum segment duration |
| `max_segment_states` | 1000 | Truncation cap |
| `exclude_weekends` | true | Safety filter (raw data is weekday-only) |
| GPS bounding box | lat [22.44, 22.87], lon [113.75, 114.65] | Shenzhen metro area |

## Filtering Statistics

| Metric | Count |
|--------|-------|
| Records outside bounding box | 0 |
| Records on weekends | 0 |
| Records with invalid timestamps | 0 |
| States deduplicated (consecutive identical x,y,t) | 5,281,484 |
| Segments filtered (< 5 states after dedup) | 523,986 |
| Segments filtered (< 300s duration) | 15,629 |

---

## File: `seeking_trajs.pkl`

Passenger-seeking trajectories (passenger indicator = 0). These represent drivers cruising for passengers between dropoff and pickup events.

### Format

```python
{
    driver_index: [           # int, 0-indexed (0–49)
        [                     # trajectory 1
            [x, y, t, d],    # state: 4-element list of ints
            [x, y, t, d],
            ...
        ],
        [                     # trajectory 2
            ...
        ],
        ...
    ],
    ...
}
```

### State Vector

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| 0 | `x` | [1, 48] | Grid x-coordinate (latitude dimension, 1-indexed, south=1, north=48) |
| 1 | `y` | [1, 90] | Grid y-coordinate (longitude dimension, 1-indexed, west=1, east=90) |
| 2 | `t` | [0, 287] | Time bucket (5-minute intervals, 0 = midnight, 287 = 23:55) |
| 3 | `d` | [1, 5] | Day of week (1=Monday, 5=Friday) |

### Aggregate Statistics

| Metric | Value |
|--------|-------|
| Total trajectories | 50,604 |
| Total states | 1,057,073 |
| Avg trajectories/driver | 1,012 |

### Length Distribution

| Stat | Value |
|------|-------|
| Min | 5 |
| P25 | 7 |
| Median | 10 |
| Mean | 20.9 |
| P75 | 18 |
| Max | 1,000 |

### Segment Definition

A seeking trajectory begins when the passenger indicator transitions 1→0 (dropoff) and ends when it transitions 0→1 (pickup) or at a day boundary. Consecutive states that quantize to the same `(x, y, t)` are deduplicated (first occurrence kept). Segments with fewer than 5 deduplicated states or under 300 seconds raw duration are discarded.

---

## File: `driving_trajs.pkl`

Passenger-onboard (driving) trajectories (passenger indicator = 1). These represent drivers transporting passengers from pickup to dropoff.

### Format

Same structure as `seeking_trajs.pkl`.

### Aggregate Statistics

| Metric | Value |
|--------|-------|
| Total trajectories | 64,338 |
| Total states | 884,741 |
| Avg trajectories/driver | 1,287 |

### Length Distribution

| Stat | Value |
|------|-------|
| Min | 5 |
| P25 | 7 |
| Median | 10 |
| Mean | 13.8 |
| P75 | 17 |
| Max | 329 |

### Segment Definition

A driving trajectory begins when the passenger indicator transitions 0→1 (pickup) and ends when it transitions 1→0 (dropoff) or at a day boundary. Same deduplication and filtering as seeking trajectories.

---

## Per-Driver Trajectory Counts

| Idx | Plate | Seeking | Driving | Seek States | Drive States | Drive:Seek Ratio |
|-----|-------|---------|---------|-------------|--------------|------------------|
| 0 | 粤B010VY | 374 | 141 | 45,619 | 1,619 | 0.38 |
| 1 | 粤B0DR36 | 1,191 | 2,011 | 17,754 | 25,884 | 1.69 |
| 2 | 粤B0U3G7 | 645 | 1,018 | 16,475 | 14,734 | 1.58 |
| 3 | 粤B0U5G6 | 591 | 1,290 | 13,874 | 17,854 | 2.18 |
| 4 | 粤B0U8G2 | 760 | 1,189 | 15,858 | 21,116 | 1.56 |
| 5 | 粤B0U9G0 | 1,083 | 1,595 | 19,772 | 25,654 | 1.47 |
| 6 | 粤B0V3P7 | 649 | 1,146 | 15,744 | 14,056 | 1.77 |
| 7 | 粤B158ZD | 487 | 117 | 53,214 | 794 | 0.24 |
| 8 | 粤B1B243 | 1,008 | 1,687 | 17,433 | 23,445 | 1.67 |
| 9 | 粤B1BR46 | 701 | 1,356 | 14,552 | 19,113 | 1.93 |
| 10 | 粤B1U2G7 | 878 | 1,091 | 21,182 | 18,601 | 1.24 |
| 11 | 粤B1U6G1 | 1,231 | 1,890 | 19,508 | 25,732 | 1.54 |
| 12 | 粤B1U6G7 | 1,121 | 2,022 | 17,285 | 29,146 | 1.80 |
| 13 | 粤B1U9G0 | 762 | 958 | 19,857 | 13,775 | 1.26 |
| 14 | 粤B1VW71 | 1,039 | 1,659 | 15,276 | 16,046 | 1.60 |
| 15 | 粤B2ND30 | 994 | 798 | 17,841 | 7,337 | 0.80 |
| 16 | 粤B2U3Y7 | 979 | 1,533 | 22,080 | 22,649 | 1.57 |
| 17 | 粤B3387U | 1,100 | 1,913 | 16,875 | 25,760 | 1.74 |
| 18 | 粤B3U4Y6 | 1,131 | 1,771 | 19,060 | 29,229 | 1.57 |
| 19 | 粤B407ZE | 654 | 871 | 4,795 | 17,576 | 1.33 |
| 20 | 粤B413YZ | 1,418 | 584 | 28,863 | 5,138 | 0.41 |
| 21 | 粤B420ZE | 1,614 | 406 | 48,901 | 2,839 | 0.25 |
| 22 | 粤B443ZE | 1,561 | 629 | 29,794 | 6,033 | 0.40 |
| 23 | 粤B446ZB | 1,603 | 1,041 | 14,416 | 15,045 | 0.65 |
| 24 | 粤B476YZ | 1,705 | 732 | 14,905 | 9,656 | 0.43 |
| 25 | 粤B476ZE | 1,550 | 692 | 28,582 | 6,545 | 0.45 |
| 26 | 粤B4L6E1 | 1,337 | 2,478 | 16,599 | 27,827 | 1.85 |
| 27 | 粤B4NP17 | 473 | 707 | 10,669 | 6,206 | 1.49 |
| 28 | 粤B4U0Y3 | 1,271 | 1,972 | 19,488 | 26,627 | 1.55 |
| 29 | 粤B4U3Y9 | 1,069 | 1,686 | 20,333 | 27,276 | 1.58 |
| 30 | 粤B4U5Y9 | 1,221 | 1,635 | 21,012 | 24,873 | 1.34 |
| 31 | 粤B4U9Y1 | 1,273 | 1,938 | 17,265 | 29,618 | 1.52 |
| 32 | 粤B4V5X0 | 1,173 | 1,804 | 20,044 | 23,136 | 1.54 |
| 33 | 粤B4X2A9 | 291 | 1 | 24,318 | 5 | 0.00 |
| 34 | 粤B550ZB | 604 | 104 | 51,949 | 736 | 0.17 |
| 35 | 粤B553ZB | 991 | 297 | 57,862 | 2,054 | 0.30 |
| 36 | 粤B5AF47 | 1,089 | 1,839 | 17,111 | 25,379 | 1.69 |
| 37 | 粤B7V3P2 | 1,145 | 1,553 | 19,799 | 23,767 | 1.36 |
| 38 | 粤B7VU02 | 1,165 | 2,177 | 16,804 | 28,318 | 1.87 |
| 39 | 粤B957YZ | 934 | 1,230 | 6,634 | 21,792 | 1.32 |
| 40 | 粤B962ZD | 503 | 1,799 | 3,590 | 33,492 | 3.58 |
| 41 | 粤B9B35A | 1,158 | 1,295 | 25,073 | 16,894 | 1.12 |
| 42 | 粤B9VU26 | 1,113 | 1,869 | 18,314 | 26,888 | 1.68 |
| 43 | 粤B9W8F2 | 1,004 | 1,912 | 16,755 | 23,200 | 1.90 |
| 44 | 粤BP14B2 | 1,208 | 1,504 | 18,551 | 15,816 | 1.25 |
| 45 | 粤BS3Z81 | 1,114 | 1,720 | 16,266 | 22,915 | 1.54 |
| 46 | 粤BX2Y97 | 1,137 | 1,876 | 18,070 | 28,800 | 1.65 |
| 47 | 粤BX4Y80 | 998 | 1,723 | 18,058 | 25,559 | 1.73 |
| 48 | 粤SW794X | 715 | 464 | 16,538 | 3,503 | 0.65 |
| 49 | 粤SW948Y | 789 | 615 | 16,456 | 4,684 | 0.78 |

### Notable Outliers

- **Driver 33** (粤B4X2A9): Only 1 driving trajectory (5 states). This driver has 291 seeking trajectories with 24,318 states — suggesting extended cruising with almost no recorded pickups. May operate differently from typical taxis.
- **Driver 7** (粤B158ZD): 487 seeking / 117 driving, but seeking trajectories are very long (avg ~110 states). Low drive:seek ratio (0.24).
- **Driver 40** (粤B962ZD): Highest drive:seek ratio (3.58) — predominantly driving with passengers.

---

## File: `profile_features.pkl`

11-dimensional driver profile feature vector with z-score normalization, targeting the FCN_P stream of the multi-stream ST-SiameseNet (Ren et al., KDD 2020).

### Format

```python
{
    "features": {             # Raw feature values
        0: np.ndarray(11,),   # driver 0
        1: np.ndarray(11,),   # driver 1
        ...
    },
    "features_normalized": {  # Z-score normalized values
        0: np.ndarray(11,),
        ...
    },
    "feature_names": [        # 11 string names
        "home_x", "home_y", "shift_start", "shift_end",
        "freq_grid_x", "freq_grid_y",
        "avg_seeking_dist", "avg_seeking_time",
        "avg_driving_dist", "avg_driving_time",
        "num_trips_per_day"
    ],
    "normalization": {
        "mean": np.ndarray(11,),  # Population mean per feature
        "std": np.ndarray(11,)    # Population std per feature (clamped >= 1e-8)
    },
    "n_features": 11,
    "method": "z-score",
    "home_loc_coverage": {
        "from_pickle": 32,    # Drivers with home location from pickle file
        "from_fallback": 18,  # Drivers with home computed from seeking trajectories
        "total": 50
    }
}
```

### Feature Specification

| Idx | Name | Source | Raw Units | Description |
|-----|------|--------|-----------|-------------|
| 0 | `home_x` | Pickle (32) / fallback (18) | grid cell [1–48] | Home/rest location x-coordinate |
| 1 | `home_y` | Pickle (32) / fallback (18) | grid cell [1–90] | Home/rest location y-coordinate |
| 2 | `shift_start` | `start_finishing_time.pkl` | time bucket [0–287] | Typical shift start time |
| 3 | `shift_end` | `start_finishing_time.pkl` | time bucket [0–287] | Typical shift end time |
| 4 | `freq_grid_x` | Engineered | grid cell [1–48] | Most frequently visited x (all trajectories) |
| 5 | `freq_grid_y` | Engineered | grid cell [1–90] | Most frequently visited y (all trajectories) |
| 6 | `avg_seeking_dist` | Engineered | grid cells (Euclidean) | Mean total distance per seeking trajectory |
| 7 | `avg_seeking_time` | Engineered | time buckets | Mean time-bucket span per seeking trajectory |
| 8 | `avg_driving_dist` | Engineered | grid cells (Euclidean) | Mean total distance per driving trajectory |
| 9 | `avg_driving_time` | Engineered | time buckets | Mean time-bucket span per driving trajectory |
| 10 | `num_trips_per_day` | Engineered | count | Driving trajectories / unique calendar days |

### Feature Statistics (Raw, Pre-Normalization)

| Feature | Mean | Std | Min | Median | Max |
|---------|------|-----|-----|--------|-----|
| `home_x` | 14.66 | 7.35 | 6.00 | 13.50 | 42.00 |
| `home_y` | 27.14 | 10.90 | 9.00 | 29.00 | 54.00 |
| `shift_start` | 1.64 | 0.63 | 1.00 | 1.71 | 2.95 |
| `shift_end` | 282.27 | 9.75 | 236.68 | 284.55 | 287.89 |
| `freq_grid_x` | 14.26 | 7.33 | 6.00 | 11.50 | 40.00 |
| `freq_grid_y` | 26.76 | 10.84 | 6.00 | 29.00 | 53.00 |
| `avg_seeking_dist` | 81.69 | 262.01 | 5.32 | 14.33 | 1,722.60 |
| `avg_seeking_time` | 11.28 | 10.07 | 1.99 | 7.52 | 59.34 |
| `avg_driving_dist` | 20.15 | 17.62 | 4.53 | 16.35 | 90.39 |
| `avg_driving_time` | 3.32 | 0.77 | 1.79 | 3.42 | 4.73 |
| `num_trips_per_day` | 19.74 | 9.46 | 0.02 | 21.89 | 37.55 |

### Z-Score Normalization Parameters

| Feature | Mean (μ) | Std (σ) |
|---------|----------|---------|
| `home_x` | 14.6600 | 7.3474 |
| `home_y` | 27.1400 | 10.8959 |
| `shift_start` | 1.6417 | 0.6345 |
| `shift_end` | 282.2656 | 9.7487 |
| `freq_grid_x` | 14.2600 | 7.3343 |
| `freq_grid_y` | 26.7600 | 10.8417 |
| `avg_seeking_dist` | 81.6878 | 262.0119 |
| `avg_seeking_time` | 11.2839 | 10.0631 |
| `avg_driving_dist` | 20.1502 | 17.6210 |
| `avg_driving_time` | 3.3242 | 0.7734 |
| `num_trips_per_day` | 19.7399 | 9.4638 |

Normalized value: `z = (x - μ) / σ`

### Feature Source Details

**Features 0–1 (home location):**
- 32 drivers sourced from `home_loc_plates_dict_all.pkl`: raw GPS coordinates converted via `gps_to_grid()` + offset
- 18 drivers computed as fallback: most frequently visited grid cell across all seeking trajectory states (approximates Ren's "longest-staying grid cell")

**Features 2–3 (shift timing):**
- All 50 drivers sourced from `start_finishing_time.pkl`
- Values are time-bucket floats (0–287 range)
- shift_start mean ≈ 1.6 buckets ≈ 00:08 (early morning start), shift_end mean ≈ 282.3 buckets ≈ 23:31

**Features 4–5 (most visited grid):**
- Computed from all trajectories (seeking + driving combined)
- Uses `Counter.most_common(1)` over all `(x, y)` state pairs

**Features 6–7 (avg seeking metrics):**
- Distance: sum of `sqrt((x₂-x₁)² + (y₂-y₁)²)` between consecutive states, averaged over all seeking trajectories
- Time: `|last_time_bucket - first_time_bucket|` per trajectory, averaged
- High variance in distance (std=262 vs mean=82) due to outlier drivers with very long cruising patterns

**Features 8–9 (avg driving metrics):**
- Same computation as seeking, applied to driving trajectories
- Driving distances are shorter and less variable than seeking (mean=20 vs 82)

**Feature 10 (trips per day):**
- Count of driving trajectories / count of unique calendar dates in raw data
- Range: 0.02 (driver 33, near-zero driving) to 37.5 trips/day

### Cross-Validation

Profile features were cross-validated against `trip_info_dict_789.pkl`:

| Comparison | Pearson r | p-value |
|------------|-----------|---------|
| `avg_driving_dist` vs trip_info v1 (distance) | 0.9420 | 2.09e-24 |
| `avg_driving_time` vs trip_info v2 (time) | 0.4923 | 2.82e-04 |

The strong distance correlation (r=0.94) confirms correct driving segment identification. The moderate time correlation (r=0.49) is expected due to different measurement units (5-minute time buckets vs raw seconds) and segment filtering removing micro-trips.

---

## File: `extraction_metadata.json`

JSON file containing the full extraction configuration, GPS bounds, driver index mapping, and processing statistics.

### Key Fields

| Field | Description |
|-------|-------------|
| `timestamp` | Extraction date/time |
| `config` | Full `ExtractionConfig` parameters |
| `bounds` | `{lat_min, lat_max, lon_min, lon_max}` used for grid quantization |
| `driver_mapping` | `{"0": "粤B010VY", "1": "粤B0DR36", ...}` — index to plate ID |
| `stats` | Full `ExtractionStats` including all filtering counts |

---

## Coordinate System Reference

- **Origin**: (x=1, y=1) = south-west corner of the study area
- **x-axis**: latitude dimension (1=south, 48=north)
- **y-axis**: longitude dimension (1=west, 90=east)
- **Grid cells are 1-indexed** in the extracted data — downstream consumers (e.g., `data_loader.py`) subtract 1 to produce 0-indexed coordinates
- Geographic extent: lat [22.44, 22.87], lon [113.75, 114.56]

## Downstream Usage

### Discriminator Model (Phase 3+)

The extracted 4-element state vectors `[x, y, t, d]` are compatible with the existing `FeatureNormalizer` in `discriminator/model/model.py`, which expects input shape `[batch, seq_len, 4]` and normalizes to `[batch, seq_len, 6]` via:
- x/49, y/89 → spatial [0, 1]
- sin/cos(2π·t/288), sin/cos(2π·(d-1)/5) → cyclical temporal features

### FCN_P Architecture Target

The 11-feature profile vector feeds into an FCN with architecture `[11] → [64] → [32] → [8]` (ReLU activation), following Ren et al. Appendix A.3. The 8-dimensional output is the profile embedding used in the Siamese dissimilarity computation.

## Extraction Tool

To regenerate this data:

```bash
# Default configuration
python -m discriminator.multi_stream.extraction.run

# Custom filtering
python -m discriminator.multi_stream.extraction.run --min-states 10 --min-duration 600

# Extraction only (skip profile features)
python -m discriminator.multi_stream.extraction.run --skip-profiles

# Validate extracted data
python -m discriminator.multi_stream.extraction.validate
```
