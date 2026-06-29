# F_causal demographic feature-SELECTION analysis (cleaned data)

Base-3 {housing, gdp, comp} F_causal = **0.8069** (before-edit). Lower F_causal = more demographic-driven unfairness captured. Policy: VIF < 10, ≤ 5 features (10-district DOF). Jaccard/Spearman are top-2293 most-unfair cells vs the current base-3 (does the editor target the same cells?).

## 1. Marginal contribution (each feature added to base-3)

Sorted by ΔF_causal (most negative = adds most independent unfairness signal).

| + feature | axis | F_causal | ΔF_causal | added-feat VIF | set max VIF |
|---|---|---|---|---|---|
| EmployeeCompensation100MYuan | income | 0.7167 | -0.0902 | 2.68 | 2.90 |
| AvgEmployedPersons | scale | 0.7223 | -0.0846 | 1.92 | 2.59 |
| LogPopDensity | density | 0.7253 | -0.0816 | 1.95 | 2.87 |
| PopDensityPerKm2 | density | 0.7363 | -0.0706 | 2.60 | 3.68 |
| GDPin10000Yuan | income | 0.7386 | -0.0683 | 1.20 | 2.54 |
| LogGDP | income | 0.7439 | -0.0630 | 1.10 | 2.58 |
| LogCompensation | income | 0.7458 | -0.0611 | 1.95 | 2.76 |
| YearEndPermanentPop10k | scale | 0.7472 | -0.0597 | 1.81 | 4.17 |
| NonRegisteredPermanentPop10k | pop_structure | 0.7564 | -0.0505 | 2.98 | 5.84 |
| MigrantRatio | pop_structure | 0.7723 | -0.0347 | 15.15 | 15.15 |
| LogHousingPrice | housing | 0.7812 | -0.0257 | 14007.06 | 14084.28 |
| SexRatio100 | pop_structure | 0.7913 | -0.0156 | 2.16 | 2.84 |

## 2. Candidate-pool VIF + pairwise correlation

Whole-pool VIFs are **inf by design** — the pool carries alternative encodings of the same axis (housing≈loghousing, gdp≈loggdp, comp≈logcomp≈employed), so the pooled design is singular. **Judge collinearity per candidate SET (§3 max VIF), not over the raw pool.** The near-perfectly-collinear pairs that cause this:

| feature A | feature B | r |
|---|---|---|
| AvgHousingPricePerSqM | LogHousingPrice | +1.000 |
| EmployeeCompensation100MYuan | AvgEmployedPersons | +0.961 |
| GDPin10000Yuan | LogGDP | +0.970 |
| NonRegisteredPermanentPop10k | YearEndPermanentPop10k | +0.959 |

### Pairwise |Pearson r| (upper triangle; flagged if |r| ≥ 0.8)

| feature | AvgHousing | LogHousing | GDPperCapi | CompPerCap | EmployeeCo | GDPin10000 | LogGDP | LogCompens | MigrantRat | NonRegiste | SexRatio10 | PopDensity | LogPopDens | YearEndPer | AvgEmploye |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AvgHousingPricePerSqM | — | +1.00 | -0.58 | -0.42 | -0.63 | -0.14 | -0.15 | -0.52 | +0.80 | +0.47 | +0.14 | -0.74 | -0.63 | +0.30 | -0.62 |
| LogHousingPrice | +1.00 | — | -0.57 | -0.42 | -0.62 | -0.14 | -0.15 | -0.52 | +0.80 | +0.47 | +0.14 | -0.74 | -0.63 | +0.31 | -0.62 |
| GDPperCapita | -0.58 | -0.57 | — | +0.72 | +0.73 | +0.29 | +0.14 | +0.48 | -0.90 | -0.74 | +0.40 | +0.22 | +0.13 | -0.55 | +0.57 |
| CompPerCapita | -0.42 | -0.42 | +0.72 | — | +0.65 | +0.41 | +0.28 | +0.63 | -0.68 | -0.29 | +0.57 | +0.14 | +0.06 | -0.13 | +0.52 |
| EmployeeCompensation100MYuan | -0.63 | -0.62 | +0.73 | +0.65 | — | +0.69 | +0.58 | +0.82 | -0.84 | -0.25 | +0.37 | +0.60 | +0.57 | +0.01 | +0.96 |
| GDPin10000Yuan | -0.14 | -0.14 | +0.29 | +0.41 | +0.69 | — | +0.97 | +0.87 | -0.45 | +0.35 | +0.37 | +0.39 | +0.57 | +0.59 | +0.71 |
| LogGDP | -0.15 | -0.15 | +0.14 | +0.28 | +0.58 | +0.97 | — | +0.84 | -0.36 | +0.46 | +0.28 | +0.46 | +0.68 | +0.68 | +0.62 |
| LogCompensation | -0.52 | -0.52 | +0.48 | +0.63 | +0.82 | +0.87 | +0.84 | — | -0.67 | +0.18 | +0.25 | +0.59 | +0.68 | +0.43 | +0.84 |
| MigrantRatio | +0.80 | +0.80 | -0.90 | -0.68 | -0.84 | -0.45 | -0.36 | -0.67 | — | +0.60 | -0.23 | -0.55 | -0.46 | +0.36 | -0.75 |
| NonRegisteredPermanentPop10k | +0.47 | +0.47 | -0.74 | -0.29 | -0.25 | +0.35 | +0.46 | +0.18 | +0.60 | — | -0.10 | -0.05 | +0.18 | +0.96 | -0.10 |
| SexRatio100 | +0.14 | +0.14 | +0.40 | +0.57 | +0.37 | +0.37 | +0.28 | +0.25 | -0.23 | -0.10 | — | -0.28 | -0.14 | -0.02 | +0.16 |
| PopDensityPerKm2 | -0.74 | -0.74 | +0.22 | +0.14 | +0.60 | +0.39 | +0.46 | +0.59 | -0.55 | -0.05 | -0.28 | — | +0.91 | +0.13 | +0.74 |
| LogPopDensity | -0.63 | -0.63 | +0.13 | +0.06 | +0.57 | +0.57 | +0.68 | +0.68 | -0.46 | +0.18 | -0.14 | +0.91 | — | +0.36 | +0.68 |
| YearEndPermanentPop10k | +0.30 | +0.31 | -0.55 | -0.13 | +0.01 | +0.59 | +0.68 | +0.43 | +0.36 | +0.96 | -0.02 | +0.13 | +0.36 | — | +0.16 |
| AvgEmployedPersons | -0.62 | -0.62 | +0.57 | +0.52 | +0.96 | +0.71 | +0.62 | +0.84 | -0.75 | -0.10 | +0.16 | +0.74 | +0.68 | +0.16 | — |

## 3. Curated feature-SET search (sizes 3–5, distinct axes)

| set | features | axes | n | F_causal | max VIF | max \|r\| | Jaccard | Spearman α | verdict |
|---|---|---|---|---|---|---|---|---|---|
| baseline_h-g-c | AvgHousingPricePerSqM, GDPperCapita, CompPerCapita | housing,income | 3 | 0.8069 | 2.54 | 0.72 | 1.0000 | 1.0000 | ROBUST-EQUIVALENT |
| h-g-migrant | AvgHousingPricePerSqM, GDPperCapita, MigrantRatio | housing,income,pop_structure | 3 | 0.7774 | 14.96 | 0.90 | 0.9674 | 0.8639 | HIGH-VIF/UNSTABLE |
| h-c-migrant | AvgHousingPricePerSqM, CompPerCapita, MigrantRatio | housing,income,pop_structure | 3 | 0.7988 | 4.45 | 0.80 | 0.9607 | 0.9851 | ROBUST-EQUIVALENT |
| h-g-c-migrant | AvgHousingPricePerSqM, GDPperCapita, CompPerCapita, MigrantRatio | housing,income,pop_structure | 4 | 0.7723 | 15.15 | 0.90 | 0.9776 | 0.9110 | HIGH-VIF/UNSTABLE |
| h-g-c-logpop | AvgHousingPricePerSqM, GDPperCapita, CompPerCapita, LogPopDensity | density,housing,income | 4 | 0.7253 | 2.87 | 0.72 | 0.9229 | 0.8353 | ROBUST-AND-BETTER |
| h-c-migrant-logpop | AvgHousingPricePerSqM, CompPerCapita, MigrantRatio, LogPopDensity | density,housing,income,pop_structure | 4 | 0.7253 | 4.51 | 0.80 | 0.9318 | 0.8742 | ROBUST-AND-BETTER |
| h-g-migrant-logpop | AvgHousingPricePerSqM, GDPperCapita, MigrantRatio, LogPopDensity | density,housing,income,pop_structure | 4 | 0.7372 | 23.30 | 0.90 | 0.9457 | 0.8934 | HIGH-VIF/UNSTABLE |
| h-g-migrant-sexratio | AvgHousingPricePerSqM, GDPperCapita, MigrantRatio, SexRatio100 | housing,income,pop_structure | 4 | 0.7587 | 15.10 | 0.90 | 0.9269 | 0.9030 | HIGH-VIF/UNSTABLE |
| h-g-migrant-logpop-sexratio | AvgHousingPricePerSqM, GDPperCapita, MigrantRatio, LogPopDensity, SexRatio100 | density,housing,income,pop_structure | 5 | 0.7273 | 23.34 | 0.90 | 0.9180 | 0.8368 | HIGH-VIF/UNSTABLE |
| h-g-c-migrant-logpop | AvgHousingPricePerSqM, GDPperCapita, CompPerCapita, MigrantRatio, LogPopDensity | density,housing,income,pop_structure | 5 | 0.7251 | 24.25 | 0.90 | 0.9237 | 0.8506 | HIGH-VIF/UNSTABLE |
| h-g-c-nonreg | AvgHousingPricePerSqM, GDPperCapita, CompPerCapita, NonRegisteredPermanentPop10k | housing,income,pop_structure | 4 | 0.7564 | 5.84 | 0.74 | 0.9515 | 0.8379 | ROBUST-AND-BETTER |

## 4. Pareto view (lower F_causal × lower VIF; VIF<10, ≤5 feats)

- Pareto-frontier sets: ['baseline_h-g-c', 'h-g-c-logpop', 'h-c-migrant-logpop']
- Sets that DOMINATE base-3 (more unfairness, VIF<10, Jaccard≥0.6): ['h-g-c-logpop', 'h-c-migrant-logpop', 'h-g-c-nonreg']
- Best low-VIF set with a population/migrant axis that beats base-3: **h-c-migrant-logpop**

## 5. Per-set verdicts

- **baseline_h-g-c** → ROBUST-EQUIVALENT  (F_causal 0.8069, ΔF 0.0000, maxVIF 2.54, Jaccard 1.000, Spearman 1.000)
- **h-g-migrant** → HIGH-VIF/UNSTABLE  (F_causal 0.7774, ΔF -0.0296, maxVIF 14.96, Jaccard 0.967, Spearman 0.864)
- **h-c-migrant** → ROBUST-EQUIVALENT  (F_causal 0.7988, ΔF -0.0081, maxVIF 4.45, Jaccard 0.961, Spearman 0.985)
- **h-g-c-migrant** → HIGH-VIF/UNSTABLE  (F_causal 0.7723, ΔF -0.0347, maxVIF 15.15, Jaccard 0.978, Spearman 0.911)
- **h-g-c-logpop** → ROBUST-AND-BETTER  (F_causal 0.7253, ΔF -0.0816, maxVIF 2.87, Jaccard 0.923, Spearman 0.835)
- **h-c-migrant-logpop** → ROBUST-AND-BETTER  (F_causal 0.7253, ΔF -0.0816, maxVIF 4.51, Jaccard 0.932, Spearman 0.874)
- **h-g-migrant-logpop** → HIGH-VIF/UNSTABLE  (F_causal 0.7372, ΔF -0.0698, maxVIF 23.30, Jaccard 0.946, Spearman 0.893)
- **h-g-migrant-sexratio** → HIGH-VIF/UNSTABLE  (F_causal 0.7587, ΔF -0.0483, maxVIF 15.10, Jaccard 0.927, Spearman 0.903)
- **h-g-migrant-logpop-sexratio** → HIGH-VIF/UNSTABLE  (F_causal 0.7273, ΔF -0.0796, maxVIF 23.34, Jaccard 0.918, Spearman 0.837)
- **h-g-c-migrant-logpop** → HIGH-VIF/UNSTABLE  (F_causal 0.7251, ΔF -0.0818, maxVIF 24.25, Jaccard 0.924, Spearman 0.851)
- **h-g-c-nonreg** → ROBUST-AND-BETTER  (F_causal 0.7564, ΔF -0.0505, maxVIF 5.84, Jaccard 0.951, Spearman 0.838)
