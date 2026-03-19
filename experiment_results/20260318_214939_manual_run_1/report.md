# Experiment Report: default

**Timestamp**: 2026-03-18T21:49:38
**Duration**: 534.9s

## Configuration

| Parameter | Value |
|-----------|-------|
| Trajectories to modify | 10 |
| ST-iFGSM step size | 0.1 |
| Max perturbation | 2.0 |
| Max iterations | 500 |
| Weight: spatial | 0.33 |
| Weight: causal | 0.33 |
| Weight: fidelity | 0.34 |
| Causal formulation | option_b |
| Discriminator | checkpoints/20260316_223817/best.pt |
| Gradient decomposition | True |

## Global Fairness Metrics

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| gini | 0.9031 | 0.9031 | -0.0000 |
| f_spatial | 0.0969 | 0.0969 | +0.0000 |
| f_causal | 0.9855 | 0.9855 | -0.0000 |
| f_fidelity | 0.5000 | 0.8723 | +0.3723 |
| combined | 0.5272 | 0.6538 | +0.1266 |

## Modification Summary

- **Trajectories modified**: 10
- **Converged**: 1
- **Mean iterations**: 490.4
- **Mean perturbation**: 2.59 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 505 | 35 | (20, 28) | (18, 29) | 2.24 | 0.6196 |
| 451 | 34 | (20, 28) | (18, 29) | 2.24 | 0.6179 |
| 295 | 35 | (20, 28) | (18, 29) | 2.24 | 0.6179 |
| 208 | 39 | (17, 38) | (19, 40) | 2.83 | 0.5925 |
| 579 | 40 | (17, 38) | (19, 40) | 2.83 | 0.5918 |
| 390 | 27 | (17, 38) | (19, 39) | 2.24 | 0.5858 |
| 111 | 9 | (17, 38) | (19, 40) | 2.83 | 0.5854 |
| 150 | 2 | (17, 38) | (19, 40) | 2.83 | 0.5802 |
| 177 | 36 | (17, 38) | (19, 40) | 2.83 | 0.5786 |
| 565 | 31 | (14, 31) | (16, 33) | 2.83 | 0.5720 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial fraction | 0.721 | 0.423 |
| Causal fraction | 0.000 | 0.000 |
| Fidelity fraction | 0.000 | 0.000 |
| Spatial-causal alignment | 0.000 | 0.000 |

**Interpretation**: Spatial and causal terms have **mixed** alignment — sometimes cooperating, sometimes conflicting.

## Convergence Statistics

- **Min iterations**: 404
- **Max iterations**: 500
- **Median iterations**: 500
- **Convergence rate**: 10%
