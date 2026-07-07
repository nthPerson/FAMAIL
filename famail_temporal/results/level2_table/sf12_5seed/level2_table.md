# Level-2 Usability Table (fairness transfer)

Edit source: `famail_temporal/results/2026-07-01T09-59-11_sf12-dual`

Seeds: [0, 1, 2, 3, 4] | Eval drivers: 12

Validation gate (real-anchored): **PASSED** (matched 0.958 vs mismatched 0.034, margin 0.20)

Each cell is mean ± std across seeds (driver-conditioned BC trained on that source).

| Source (training data) | F_causal | F_spatial | Fidelity-A ↑ | Fidelity-B ↓ |
|---|---:|---:|---:|---:|
| raw | 0.8742 ± 0.0053 | 0.1892 ± 0.0013 | 0.9575 ± 0.0001 | 0.0109 ± 0.0010 |
| edited | 0.8745 ± 0.0043 | 0.1923 ± 0.0024 | 0.9575 ± 0.0001 | 0.0130 ± 0.0008 |
| bcgen | 0.8801 ± 0.0028 | 0.1896 ± 0.0018 | 0.9577 ± 0.0001 | 0.0164 ± 0.0002 |
| gangen | 0.8779 ± 0.0015 | 0.1863 ± 0.0013 | 0.9576 ± 0.0001 | 0.0341 ± 0.0016 |

## Paired fairness transfer (F_causal, by seed)

| Comparison | mean Δ ± std | n seeds | Wilcoxon p |
|---|---:|---:|---:|
| edited − raw | +0.0004 ± 0.0033 | 5 | 0.812 |
| edited − bcgen | -0.0056 ± 0.0054 | 5 | 0.125 |
| edited − gangen | -0.0033 ± 0.0052 | 5 | 0.312 |
