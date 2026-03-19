# Experiment Report: default

**Timestamp**: 2026-03-18T21:27:06
**Duration**: 121.3s

## Configuration

| Parameter | Value |
|-----------|-------|
| Trajectories to modify | 10 |
| ST-iFGSM step size | 0.1 |
| Max perturbation | 2.0 |
| Max iterations | 100 |
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
| f_fidelity | 0.5000 | 0.8707 | +0.3707 |
| combined | 0.5272 | 0.6532 | +0.1260 |

## Modification Summary

- **Trajectories modified**: 10
- **Converged**: 0
- **Mean iterations**: 100.0
- **Mean perturbation**: 2.59 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 451 | 34 | (20, 28) | (18, 29) | 2.24 | 0.6194 |
| 295 | 35 | (20, 28) | (18, 29) | 2.24 | 0.6193 |
| 505 | 35 | (20, 28) | (18, 29) | 2.24 | 0.6162 |
| 208 | 39 | (17, 38) | (19, 40) | 2.83 | 0.5926 |
| 111 | 9 | (17, 38) | (19, 40) | 2.83 | 0.5904 |
| 390 | 27 | (17, 38) | (19, 39) | 2.24 | 0.5899 |
| 579 | 40 | (17, 38) | (19, 40) | 2.83 | 0.5823 |
| 150 | 2 | (17, 38) | (19, 40) | 2.83 | 0.5812 |
| 177 | 36 | (17, 38) | (19, 40) | 2.83 | 0.5745 |
| 565 | 31 | (14, 31) | (16, 33) | 2.83 | 0.5702 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial fraction | 0.706 | 0.431 |
| Causal fraction | 0.000 | 0.000 |
| Fidelity fraction | 0.000 | 0.000 |
| Spatial-causal alignment | 0.000 | 0.000 |

**Interpretation**: Spatial and causal terms have **mixed** alignment — sometimes cooperating, sometimes conflicting.

## Convergence Statistics

- **Min iterations**: 100
- **Max iterations**: 100
- **Median iterations**: 100
- **Convergence rate**: 0%
