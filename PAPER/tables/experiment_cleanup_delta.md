# E22: Experiment-level dirty-vs-clean robustness

Data cleanup = stuck-GPS sink filter (6 drivers, cell (28,52) removed).
All four stages of the argument are compared below.

## Stage L1-v2 — per-source F_causal (edited should stay fairest faithful)

| source | dirty F_causal | clean F_causal | Δ (clean−dirty) |
|--------|---------------|---------------|-----------------|
| raw | 0.8052 | 0.8069 | 0.0017 |
| edited | 0.8180 | 0.8193 | 0.0014 |
| bc | 0.8070 | 0.8045 | -0.0025 |
| gan | 0.8146 | 0.8152 | 0.0006 |

**Conclusion preserved?** PRESERVED — edited stays fairest faithful

## Stage L2 — vanilla-BC transfer: edited−raw paired Δ F_causal (should stay n.s.)

| | Δ F_causal (mean) | wilcoxon p |
|--|-------------------|------------|
| dirty | -0.0022 | 0.0625 |
| clean | -0.0009 | 0.4375 |

**Conclusion preserved?** PRESERVED — both dirty & clean n.s. (p≥0.05)

## Stage weighted-BC — paired Δ F_causal vs raw (edited_wN should stay significant + dose-responsive)

| arm | dirty Δ (p) | clean Δ (p) | status |
|-----|------------|------------|--------|
| edited | -0.0019 (p=0.03125) | -0.0008 (p=0.4375) | compared |
| edited_w10 | 0.0186 (p=0.03125) | 0.0175 (p=0.03125) | compared |
| edited_w20 | 0.0242 (p=0.03125) | 0.0222 (p=0.03125) | compared |
| edited_w30 | 0.0274 (p=0.03125) | 0.0260 (p=0.03125) | compared |
| most_fair_w10 | — (absent) | -0.0001 (p=1.0) | clean_only |
| most_fair_w20 | — (absent) | 0.0010 (p=0.4375) | clean_only |
| most_fair_w30 | — (absent) | 0.0012 (p=0.5625) | clean_only |
| random_w10 | — (absent) | -0.0007 (p=0.3125) | clean_only |
| random_w30 | — (absent) | -0.0004 (p=0.6875) | clean_only |

**Conclusion preserved?** PRESERVED — weighted arms stay significant in both dirty & clean; new clean-only arms: most_fair_w10, most_fair_w20, most_fair_w30, random_w10, random_w30

## Stage variance — b0 vs FAMAIL paired Δ F_causal (should stay ≈null)

| | dirty paired Δ F_causal | clean paired Δ F_causal | shift |
|--|------------------------|------------------------|-------|
| f_causal | -0.0011 | -0.0004 | 0.0008 |
| f_spatial | 0.0009 | -0.0002 | -0.0011 |

**Conclusion preserved?** PRESERVED — paired Δ F_causal remains near-zero in both

