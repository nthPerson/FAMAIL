# Level-2 Usability Table (fairness transfer)

Edit source: `famail_temporal/results/2026-06-28T11-46-12_k-10000_causal_emphasis_no-dedup_cleaned_4feat`

Seeds: [0, 1, 2, 3, 4] | Eval drivers: 50

Validation gate (real-anchored): **PASSED** (matched 0.847 vs mismatched 0.193, margin 0.20)

Each cell is mean ± std across seeds (driver-conditioned BC trained on that source).

| Source (training data) | F_causal | F_spatial | Fidelity-A ↑ | Fidelity-B ↓ |
|---|---:|---:|---:|---:|
| raw | 0.7274 ± 0.0023 | 0.1053 ± 0.0009 | 0.8473 ± 0.0002 | 0.0128 ± 0.0008 |
| edited | 0.7264 ± 0.0016 | 0.1053 ± 0.0006 | 0.8479 ± 0.0004 | 0.0135 ± 0.0012 |
| bcgen | 0.7256 ± 0.0010 | 0.1048 ± 0.0005 | 0.8476 ± 0.0004 | 0.0170 ± 0.0007 |
| gangen | 0.7403 ± 0.0019 | 0.1043 ± 0.0010 | 0.8487 ± 0.0004 | 0.3264 ± 0.0164 |

## Paired fairness transfer (F_causal, by seed)

| Comparison | mean Δ ± std | n seeds | Wilcoxon p |
|---|---:|---:|---:|
| edited − raw | -0.0010 ± 0.0023 | 5 | 0.312 |
| edited − bcgen | +0.0007 ± 0.0009 | 5 | 0.188 |
| edited − gangen | -0.0139 ± 0.0015 | 5 | 0.062 |
