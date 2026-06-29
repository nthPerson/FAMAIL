# F_causal feature-set sensitivity analysis (cleaned data)

Recomputes the **before-edit** F_causal and its per-cell attribution for alternative demographic feature subsets, reusing the production compute path (`precompute_hat_matrices` → `compute_fcausal_compact` / `per_cell_fairness_attribution_causal`). The Stage-1 residual R, the centering M, and the active set are held **fixed**; only the demographic projection H_demo changes.

## Sanity gate

- expected (editor before-edit): **0.806928**
- recomputed (cached hat matrices): **0.806928** (|Δ| = 0.00000006)
- recomputed (from-scratch demographics): **0.806928** (|Δ| = 0.00000006)
- tolerance: 0.001000
- **GATE PASSED**

## F_causal per subset

Top-K overlap uses K = 2293 (edited-N); N_units = 34524. Jaccard/Spearman are vs the baseline subset.

| subset | features | F_causal | top-K Jaccard | Spearman α | max VIF | status |
|---|---|---|---|---|---|---|
| baseline | AvgHousingPricePerSqM, GDPperCapita, CompPerCapita | 0.8069 | 1.0000 | 1.0000 | 2.54 | ok |
| +popdensity | AvgHousingPricePerSqM, GDPperCapita, CompPerCapita, LogPopDensity | 0.7253 | 0.9229 | 0.8353 | 2.87 | ok |
| +migrant | AvgHousingPricePerSqM, GDPperCapita, CompPerCapita, MigrantRatio | 0.7723 | 0.9776 | 0.9110 | 15.15 | ok |
| drop_gdp | AvgHousingPricePerSqM, CompPerCapita | 0.8071 | 1.0000 | 0.9997 | 1.22 | ok |
| drop_comp | AvgHousingPricePerSqM, GDPperCapita | 0.8155 | 0.9632 | 0.9694 | 1.49 | ok |
| drop_housing | GDPperCapita, CompPerCapita | 0.9030 | 0.5636 | 0.9013 | 2.08 | ok |
| logs | LogHousingPrice, LogGDP, LogCompensation | 0.7439 | 0.9582 | 0.7987 | 7.00 | ok |
| broad5 | AvgHousingPricePerSqM, GDPperCapita, CompPerCapita, LogPopDensity, MigrantRatio | 0.7251 | 0.9237 | 0.8506 | 24.25 | ok |

### Per-feature VIFs

- **baseline**: AvgHousingPricePerSqM=1.49, GDPperCapita=2.54, CompPerCapita=2.08
- **+popdensity**: AvgHousingPricePerSqM=2.87, GDPperCapita=2.71, CompPerCapita=2.09, LogPopDensity=1.95
- **+migrant**: AvgHousingPricePerSqM=4.11, GDPperCapita=8.66, CompPerCapita=2.10, MigrantRatio=15.15
- **drop_gdp**: AvgHousingPricePerSqM=1.22, CompPerCapita=1.22
- **drop_comp**: AvgHousingPricePerSqM=1.49, GDPperCapita=1.49
- **drop_housing**: GDPperCapita=2.08, CompPerCapita=2.08
- **logs**: LogHousingPrice=2.14, LogGDP=5.25, LogCompensation=7.00
- **broad5**: AvgHousingPricePerSqM=4.16, GDPperCapita=14.60, CompPerCapita=2.18, LogPopDensity=3.13, MigrantRatio=24.25

## Verdict

**FRAGILE**

- F_causal spread across recomputed subsets: 0.1779 (robust threshold ≤ 0.05)
- min top-K Jaccard vs baseline: 0.5636 (robust threshold ≥ 0.6) [worst: drop_housing = 0.5636]
- min Spearman α vs baseline: 0.7987 (robust threshold ≥ 0.9)
