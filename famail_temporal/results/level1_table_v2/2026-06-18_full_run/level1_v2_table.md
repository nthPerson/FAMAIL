# Level-1 Data-Quality Table v2 (driver-conditioned)

Edit source: `famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`

Eval drivers: 50

Validation gate (real-anchored): **PASSED** (matched real-d/real-d 0.840 vs mismatched real-d/real-d' 0.174, margin 0.20)

| Source | F_causal | F_spatial | Fidelity-A (identity, higher=better) | A separation (matched-mismatched) | Fidelity-B (divergence, lower=better) |
|---|---:|---:|---:|---:|---:|
| raw | 0.8052 | 0.0822 | 0.840 | +0.666 | 0.0000 |
| edited | 0.8180 | 0.0824 | 0.838 | +0.725 | 0.1689 |
| bc | 0.8070 | 0.0833 | 0.841 | +0.667 | 0.0103 |
| gan | 0.8146 | 0.0837 | 0.842 | +0.668 | 0.3228 |
