# Level-2 Usability Table (fairness transfer)

Edit source: `famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`

Seeds: [0, 1, 2, 3, 4] | Eval drivers: 50

Validation gate (real-anchored): **PASSED** (matched 0.841 vs mismatched 0.174, margin 0.20)

Each cell is mean ± std across seeds (driver-conditioned BC trained on that source).

| Source (training data) | F_causal | F_spatial | Fidelity-A ↑ | Fidelity-B ↓ |
|---|---:|---:|---:|---:|
| raw | 0.8083 ± 0.0027 | 0.0831 ± 0.0002 | 0.8406 ± 0.0002 | 0.0121 ± 0.0014 |
| edited | 0.8061 ± 0.0025 | 0.0841 ± 0.0003 | 0.8408 ± 0.0007 | 0.0120 ± 0.0010 |
| bcgen | 0.8099 ± 0.0018 | 0.0833 ± 0.0002 | 0.8408 ± 0.0005 | 0.0163 ± 0.0007 |
| gangen | 0.8143 ± 0.0037 | 0.0839 ± 0.0008 | 0.8418 ± 0.0003 | 0.3507 ± 0.0106 |

## Paired fairness transfer (F_causal, by seed)

| Comparison | mean Δ ± std | n seeds | Wilcoxon p |
|---|---:|---:|---:|
| edited − raw | -0.0022 ± 0.0016 | 5 | 0.062 |
| edited − bcgen | -0.0038 ± 0.0034 | 5 | 0.125 |
| edited − gangen | -0.0081 ± 0.0047 | 5 | 0.062 |
