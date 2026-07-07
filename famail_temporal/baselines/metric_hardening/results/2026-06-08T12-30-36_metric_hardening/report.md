# Metric hardening run report

**Command**: `python -m famail_temporal.baselines.run_metric_hardening`
**Edit source**: `famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`

## Transmission (does the data-level signal survive the LSTM?)

| Quantity | Value |
|---|---|
| JS(p_raw, p_edited) - *target* shift | **0.00753** bits |
| JS(p_gen_B0, p_gen_FAMAIL) - *transmitted* shift | **0.01259** bits |
| **Transmission ratio** (transmitted / target) | **1.672** |
| JS(p_gen_B0, p_raw) - B0 fidelity to raw target | 0.01108 |
| JS(p_gen_FAMAIL, p_edited) - FAMAIL fidelity to edited target | 0.01313 |

Reading: transmission_ratio ~ 1.0 means the generator faithfully transmitted
the edit; << 1 means MLE smoothing + multinomial sampling washed it out.

## Disparate impact (DI) - both Y conventions

|       | Y = supply/demand (primary; F_causal-aligned) | Y = demand/supply (supplementary) |
|---|---:|---:|
| B0     | 0.2637 | 0.1307 |
| FAMAIL | 0.2630 | 0.1384 |
| Delta DI | **-0.0008** | **+0.0077** |

Top-3 hukou districts: [5, 8, 3]; bottom-3: [2, 7, 6].
Both DIs should move in the *same* direction under FAMAIL editing (robustness).

## Localized F_causal (restricted to 1186 edited active units)

|       | F_causal_global | F_causal_localized |
|---|---:|---:|
| B0     | 0.8079 | 0.2724 |
| FAMAIL | 0.8107 | 0.2636 |
| Delta  | +0.0028 | **-0.0088** |

Note: f_causal_global here uses M=I (uniform weighting), the same formula as
f_causal_localized at different N. This is NOT the production F_causal in
b0_fairness/famail_fairness (which uses M=center). See MODEL_LEVEL_METRICS.md.

Reading: localized Delta should be substantially larger than global Delta because
the edit's effect concentrates in the touched units. If localized Delta is also
small, the headline is fragile and the data-level Pareto is the more honest framing.
