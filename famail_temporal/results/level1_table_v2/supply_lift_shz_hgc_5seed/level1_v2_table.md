# Level-1 Data-Quality Table v2 (driver-conditioned)

Edit source: `famail_temporal/results/2026-07-13T04-41-12_supply_lift_v1_shz_hgc_filtered`

Eval drivers: 50

Validation gate (real-anchored): **PASSED** (matched real-d/real-d 0.848 vs mismatched real-d/real-d' 0.193, margin 0.20)

| Source | F_causal | F_spatial | Fidelity-A (identity, higher=better) | A separation (matched-mismatched) | Fidelity-B (divergence, lower=better) |
|---|---:|---:|---:|---:|---:|
| raw | 0.8069 | 0.1034 | 0.848 | +0.655 | 0.0000 |
| edited | 0.8275 | 0.1091 | 0.845 | +0.646 | 0.1937 |
| bc | 0.8045 | 0.1048 | 0.848 | +0.656 | 0.0108 |
| gan | 0.8152 | 0.1041 | 0.849 | +0.655 | 0.2911 |
