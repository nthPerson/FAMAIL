# Level-1 Data-Quality Table v2 (driver-conditioned)

Edit source: `famail_temporal/results/2026-07-13T17-04-22_supply_lift_v1_shz_4feat_filtered`

Eval drivers: 50

Validation gate (real-anchored): **PASSED** (matched real-d/real-d 0.848 vs mismatched real-d/real-d' 0.192, margin 0.20)

| Source | F_causal | F_spatial | Fidelity-A (identity, higher=better) | A separation (matched-mismatched) | Fidelity-B (divergence, lower=better) |
|---|---:|---:|---:|---:|---:|
| raw | 0.7253 | 0.1034 | 0.848 | +0.656 | 0.0000 |
| edited | 0.7473 | 0.1120 | 0.844 | +0.650 | 0.1794 |
| bc | 0.7223 | 0.1048 | 0.848 | +0.655 | 0.0108 |
| gan | 0.7385 | 0.1041 | 0.849 | +0.653 | 0.2913 |
