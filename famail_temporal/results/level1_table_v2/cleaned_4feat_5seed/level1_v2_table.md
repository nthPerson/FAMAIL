# Level-1 Data-Quality Table v2 (driver-conditioned)

Edit source: `famail_temporal/results/2026-06-28T11-46-12_k-10000_causal_emphasis_no-dedup_cleaned_4feat`

Eval drivers: 50

Validation gate (real-anchored): **PASSED** (matched real-d/real-d 0.848 vs mismatched real-d/real-d' 0.193, margin 0.20)

| Source | F_causal | F_spatial | Fidelity-A (identity, higher=better) | A separation (matched-mismatched) | Fidelity-B (divergence, lower=better) |
|---|---:|---:|---:|---:|---:|
| raw | 0.7253 | 0.1034 | 0.848 | +0.655 | 0.0000 |
| edited | 0.7409 | 0.1024 | 0.842 | +0.673 | 0.1537 |
| bc | 0.7223 | 0.1048 | 0.847 | +0.655 | 0.0108 |
| gan | 0.7385 | 0.1041 | 0.849 | +0.654 | 0.2922 |
