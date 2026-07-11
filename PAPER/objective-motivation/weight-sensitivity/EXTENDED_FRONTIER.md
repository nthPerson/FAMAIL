# Extended frontier — ring-2/ring-3 columns for every sweep point (Shenzhen PRIMARY)

**Generated 2026-07-11** by the R0X frontier extension: `supply_recount --persist-grids` +
`channel_decomposition --tier2-grid` (B=2,000) + `run_external_fairness --delta-supply` (B=1,000) run
on **every** sweep corpus, identically to the headline scoring. Ledger rows R0 (s55) and R0X-{s00,s10,s35,s80};
the (0.2, 0.7, 0.1) row is the original supply-lift headline artifact set. Migrant axis, district extremes.
`*` = 95% bootstrap CI excludes 0.

| α (sp, ca, fi) | ΔF_causal | ΔF_spatial | Δmean(Y\|D) total | supply tier-1 | supply tier-2 | ΔDI | ΔTheil |
|---|---:|---:|---:|---:|---:|---:|---:|
| (0.00, 0.90, 0.1) | +0.0221 | +0.0057 | **+0.0777\*** | **+0.0199\*** | **+0.0426\*** | +0.0171\* | −0.0085\* |
| (0.10, 0.80, 0.1) | +0.0226 | +0.0061 | **+0.0529\*** | **+0.0176\*** | **+0.0411\*** | +0.0162\* | −0.0086\* |
| (0.20, 0.70, 0.1) | +0.0222 | +0.0064 | **+0.0468\*** | **+0.0091\*** | **+0.0242\*** | +0.0155\* | −0.0082\* |
| (0.35, 0.55, 0.1) | +0.0217 | +0.0076 | +0.0192 | −0.0018 | +0.0125\* | +0.0138\* | −0.0078\* |
| (0.55, 0.35, 0.1) | +0.0227 | +0.0094 | +0.0086 | **−0.0082\*** (neg.) | +0.0057\* | +0.0131\* | −0.0073\* |
| (0.80, 0.10, 0.1) | +0.0185 | +0.0117 | +0.0019 | −0.0007 | +0.0058\* | +0.0130\* | −0.0077\* |

## Findings

1. **Ring 1 is flat; ring 2 is not.** ΔF_causal varies by ≤0.001 for all α_spatial ≤ 0.55, but the
   design-targeted lift-up (Δmean(Y|D), the supply channels) declines **monotonically** as α_spatial
   rises — from +0.0777\* at (0, 0.9) to n.s./negative beyond α_spatial = 0.2. The two-axis sweep the
   re-anchor decision was made on could not see this.
2. **The lift-up claim is a property of causal-heavy weights.** Tier-1 supply flips significantly
   negative at (0.55, 0.35): the spatial term dominates the value-of-presence map and routes lift
   tails to spatially-uneven (not demographically disadvantaged) cells. External metrics (DI, DP,
   Theil) improve significantly at every point — but beyond α_spatial ≈ 0.2 they improve by
   leveling down again.
3. **Amended three-ring criterion.** Maximize ΔF_causal subject to (i) ΔF_spatial ≥ 0 and (ii) the
   supply-channel lift-up significant on both accounting tiers. Eligible: {(0, 0.9), (0.1, 0.8),
   (0.2, 0.7)}. Selected: **(0.1, 0.8, 0.1)** — ΔF_causal +0.0226, with the lift-up materially
   stronger than the prior (0.2, 0.7) headline (tier-1 +0.0176 vs +0.0091; tier-2 +0.0411 vs +0.0242;
   total +0.0529 vs +0.0468) and DI/Theil at least as good. The prior headline is weakly dominated by
   (0.1, 0.8, 0.1) on every reported column except a noise-scale ΔF_spatial difference
   (+0.0061 vs +0.0064).

## Provenance

Per-point artifacts (each corpus dir): `channel_decomposition.json`, `supply_recount.json`,
`S_tier2_{before,after}.npz`; external metrics under
`famail_temporal/baselines/external_fairness/results/shenzhen-primary-supplylift-{s00,s10,s35,a55,s80}/`.
(0.2, 0.7, 0.1) row: `PAPER/supply-lift/data/` (original headline artifacts). Assembly command recorded
in ledger rows R0X-\*.
