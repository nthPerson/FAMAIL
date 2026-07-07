# Level-1 Data-Quality Table v2 (driver-conditioned)

Edit source: `famail_temporal/results/2026-07-01T09-59-11_sf12-dual`

Eval drivers: 12

Validation gate (real-anchored): **PASSED** (matched real-d/real-d 0.958 vs mismatched real-d/real-d' 0.034, margin 0.20)

| Source | F_causal | F_spatial | Fidelity-A (identity, higher=better) | A separation (matched-mismatched) | Fidelity-B (divergence, lower=better) |
|---|---:|---:|---:|---:|---:|
| raw | 0.8752 | 0.1846 | 0.958 | +0.923 | 0.0000 |
| edited | 0.8891 | 0.1817 | 0.958 | +0.924 | 0.1058 |
| bc | 0.8789 | 0.1894 | 0.958 | +0.923 | 0.0100 |
| gan | 0.8794 | 0.1856 | 0.958 | +0.923 | 0.0269 |
