# Level-1 Data-Quality Table v2 (driver-conditioned)

Edit source: `famail_temporal/results/2026-07-11T11-31-55_supply_lift_a10_sf12_filtered`

Eval drivers: 12

Validation gate (real-anchored): **PASSED** (matched real-d/real-d 0.958 vs mismatched real-d/real-d' 0.034, margin 0.20)

| Source | F_causal | F_spatial | Fidelity-A (identity, higher=better) | A separation (matched-mismatched) | Fidelity-B (divergence, lower=better) |
|---|---:|---:|---:|---:|---:|
| raw | 0.8752 | 0.1846 | 0.958 | +0.923 | 0.0000 |
| edited | 0.9067 | 0.1985 | 0.958 | +0.924 | 0.0978 |
| bc | 0.8789 | 0.1894 | 0.958 | +0.923 | 0.0100 |
| gan | 0.8794 | 0.1856 | 0.958 | +0.923 | 0.0269 |
