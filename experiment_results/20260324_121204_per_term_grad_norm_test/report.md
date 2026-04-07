# Experiment Report: per_term_grad_norm_test

**Timestamp**: 2026-03-24T12:12:04
**Duration**: 116.5s

## Configuration

| Parameter | Value |
|-----------|-------|
| Trajectories to modify | 10 |
| ST-iFGSM step size | 0.1 |
| Max perturbation | 2.0 |
| Max iterations | 50 |
| Weight: spatial | 0.33 |
| Weight: causal | 0.33 |
| Weight: fidelity | 0.34 |
| Causal formulation | option_b |
| Discriminator | checkpoints/20260316_223817/best.pt |
| Gradient decomposition | True |
| Gradient normalization | True |

## Global Fairness Metrics

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| gini | 0.9031 | 0.9030 | -0.0001 |
| f_spatial | 0.0969 | 0.0970 | +0.0001 |
| f_causal | 0.9787 | 0.9788 | +0.0001 |
| f_fidelity | 0.5000 | 0.8941 | +0.3941 |
| combined | 0.5249 | 0.6590 | +0.1341 |

## Modification Summary

- **Trajectories modified**: 10
- **Converged**: 0
- **Mean iterations**: 50.0
- **Mean perturbation**: 2.26 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 119 | 49 | (41, 41) | (40, 43) | 2.24 | 0.6687 |
| 344 | 22 | (35, 15) | (35, 17) | 2.00 | 0.6507 |
| 15 | 19 | (29, 54) | (31, 55) | 2.24 | 0.6467 |
| 57 | 19 | (29, 54) | (31, 56) | 2.83 | 0.6450 |
| 94 | 22 | (28, 16) | (28, 16) | 0.00 | 0.6439 |
| 390 | 27 | (17, 38) | (17, 40) | 2.00 | 0.6265 |
| 111 | 9 | (17, 38) | (19, 40) | 2.83 | 0.6188 |
| 208 | 39 | (17, 38) | (19, 40) | 2.83 | 0.6177 |
| 150 | 2 | (17, 38) | (19, 40) | 2.83 | 0.6147 |
| 177 | 36 | (17, 38) | (19, 40) | 2.83 | 0.6115 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial fraction | 0.213 | 0.318 |
| Causal fraction | 0.107 | 0.167 |
| Fidelity fraction | 0.680 | 0.331 |
| Spatial-causal alignment | -0.211 | 0.601 |

**Interpretation**: Spatial and causal terms have **mixed** alignment — sometimes cooperating, sometimes conflicting.

## Convergence Statistics

- **Min iterations**: 50
- **Max iterations**: 50
- **Median iterations**: 50
- **Convergence rate**: 0%
