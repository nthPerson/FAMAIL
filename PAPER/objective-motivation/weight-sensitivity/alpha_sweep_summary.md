# α-sweep — empirical (ΔF_spatial, ΔF_causal) frontier (SZ primary, supply-lift editor, k=10000, +infeasible-trim filter)

| α (spatial, causal, fidelity) | ΔF_causal | ΔF_spatial | Pareto | source |
|---|---:|---:|:---:|---|
| (0, 0.9, 0.1) | +0.0221 | +0.0057 | — | `2026-07-09T17-11-50_alpha_sweep_s00_c90_f10_filtered` |
| (0.1, 0.8, 0.1) | +0.0226 | +0.0061 | — | `2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered` |
| (0.2, 0.7, 0.1) ★ shipped | +0.0222 | +0.0064 | — | `2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered` |
| (0.35, 0.55, 0.1) | +0.0217 | +0.0076 | — | `2026-07-10T10-32-00_alpha_sweep_s35_c55_f10_filtered` |
| (0.55, 0.35, 0.1) | +0.0227 | +0.0094 | ✓ | `2026-07-10T17-45-40_alpha_sweep_s55_c35_f10_filtered` |
| (0.8, 0.1, 0.1) | +0.0185 | +0.0117 | ✓ | `2026-07-10T23-30-57_alpha_sweep_s80_c10_f10_filtered` |

**Weight-selection criterion** (max ΔF_causal s.t. ΔF_spatial ≥ 0) selects **(0.55, 0.35, 0.1)**.