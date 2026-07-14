# Level-1 Data-Quality Table v2 (driver-conditioned)

Edit source: `famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered`

Eval drivers: 50

Validation gate (real-anchored): **PASSED** (matched real-d/real-d 0.848 vs mismatched real-d/real-d' 0.193, margin 0.20)

| Source | F_causal | F_spatial | Fidelity-A (identity, higher=better) | A separation (matched-mismatched) | Fidelity-B (divergence, lower=better) |
|---|---:|---:|---:|---:|---:|
| raw | 0.7988 | 0.1034 | 0.848 | +0.655 | 0.0000 |
| edited | 0.8214 | 0.1095 | 0.844 | +0.650 | 0.1871 |
| bc | 0.7955 | 0.1048 | 0.849 | +0.656 | 0.0108 |
| gan | 0.8089 | 0.1041 | 0.849 | +0.655 | 0.2911 |
