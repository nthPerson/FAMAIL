# Level-1 Data-Quality Table v2 (driver-conditioned)

Edit source: `famail_temporal/results/2026-06-26T12-32-59_k-10000_causal_emphasis_no-dedup_cleaned`

Eval drivers: 50

Validation gate (real-anchored): **PASSED** (matched real-d/real-d 0.847 vs mismatched real-d/real-d' 0.193, margin 0.20)

| Source | F_causal | F_spatial | Fidelity-A (identity, higher=better) | A separation (matched-mismatched) | Fidelity-B (divergence, lower=better) |
|---|---:|---:|---:|---:|---:|
| raw | 0.8069 | 0.1034 | 0.847 | +0.654 | 0.0000 |
| edited | 0.8193 | 0.1025 | 0.842 | +0.673 | 0.1513 |
| bc | 0.8045 | 0.1048 | 0.848 | +0.655 | 0.0108 |
| gan | 0.8152 | 0.1041 | 0.849 | +0.654 | 0.2922 |
