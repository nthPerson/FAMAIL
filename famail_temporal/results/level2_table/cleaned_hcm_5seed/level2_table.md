# Level-2 Usability Table (fairness transfer)

Edit source: `famail_temporal/results/2026-06-29T12-06-55_k-10000_causal_emphasis_no-dedup_cleaned_hcm`

Seeds: [0, 1, 2, 3, 4] | Eval drivers: 50

Validation gate (real-anchored): **PASSED** (matched 0.847 vs mismatched 0.193, margin 0.20)

Each cell is mean ± std across seeds (driver-conditioned BC trained on that source).

| Source (training data) | F_causal | F_spatial | Fidelity-A ↑ | Fidelity-B ↓ |
|---|---:|---:|---:|---:|
| raw | 0.8002 ± 0.0027 | 0.1053 ± 0.0009 | 0.8473 ± 0.0002 | 0.0128 ± 0.0008 |
| edited | 0.7990 ± 0.0018 | 0.1052 ± 0.0006 | 0.8480 ± 0.0005 | 0.0132 ± 0.0009 |
| bcgen | 0.7984 ± 0.0009 | 0.1048 ± 0.0005 | 0.8476 ± 0.0004 | 0.0170 ± 0.0007 |
| gangen | 0.8101 ± 0.0025 | 0.1043 ± 0.0010 | 0.8487 ± 0.0004 | 0.3264 ± 0.0164 |

## Paired fairness transfer (F_causal, by seed)

| Comparison | mean Δ ± std | n seeds | Wilcoxon p |
|---|---:|---:|---:|
| edited − raw | -0.0012 ± 0.0030 | 5 | 0.438 |
| edited − bcgen | +0.0005 ± 0.0021 | 5 | 0.812 |
| edited − gangen | -0.0112 ± 0.0023 | 5 | 0.062 |
