# Level-1 Data-Quality Table

Edit source: `famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`

Validation gate: **FAILED** (real-real 0.668 vs collapsed 0.660 / shuffled 0.668, margin 0.20)

_Fairness columns are single-seed (this table's internal coherence); the authoritative multi-seed fairness figures are the variance-suite 5-seed mean ± std._

| Source | F_causal (single-seed) | F_spatial (single-seed) | Fidelity-A (HuMID, higher=better) | Fidelity-B (divergence, lower=better) |
|---|---:|---:|---:|---:|
| raw | 0.8052 | 0.0822 | 0.668 (untrusted) | 0.0000 |
| edited | 0.8180 | 0.0824 | 0.665 (untrusted) | 0.0480 |
| bc | 0.8064 | 0.0827 | 0.664 (untrusted) | 0.0106 |
| gan | 0.8212 | 0.0847 | 0.664 (untrusted) | 0.3189 |
