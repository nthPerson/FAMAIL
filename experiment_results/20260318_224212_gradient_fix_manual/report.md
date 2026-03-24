# Experiment Report: default

**Timestamp**: 2026-03-18T22:42:12
**Duration**: 163.7s

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

## Global Fairness Metrics

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| gini | 0.9031 | 0.9031 | -0.0000 |
| f_spatial | 0.0969 | 0.0969 | +0.0000 |
| f_causal | 0.9855 | 0.9855 | -0.0000 |
| f_fidelity | 0.5000 | 0.8708 | +0.3708 |
| combined | 0.5272 | 0.6533 | +0.1261 |

## Modification Summary

- **Trajectories modified**: 10
- **Converged**: 0
- **Mean iterations**: 50.0
- **Mean perturbation**: 1.43 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 505 | 35 | (20, 28) | (22, 27) | 2.24 | 0.6198 |
| 451 | 34 | (20, 28) | (19, 28) | 1.00 | 0.6186 |
| 295 | 35 | (20, 28) | (22, 27) | 2.24 | 0.6179 |
| 208 | 39 | (17, 38) | (17, 39) | 1.00 | 0.5934 |
| 111 | 9 | (17, 38) | (17, 39) | 1.00 | 0.5900 |
| 390 | 27 | (17, 38) | (17, 39) | 1.00 | 0.5825 |
| 150 | 2 | (17, 38) | (17, 39) | 1.00 | 0.5817 |
| 565 | 31 | (14, 31) | (16, 33) | 2.83 | 0.5790 |
| 579 | 40 | (17, 38) | (17, 39) | 1.00 | 0.5782 |
| 177 | 36 | (17, 38) | (17, 39) | 1.00 | 0.5757 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial fraction | 0.005 | 0.007 |
| Causal fraction | 0.444 | 0.328 |
| Fidelity fraction | 0.551 | 0.329 |
| Spatial-causal alignment | -0.494 | 0.609 |

**Interpretation**: Spatial and causal terms have **mixed** alignment — sometimes cooperating, sometimes conflicting.

## Convergence Statistics

- **Min iterations**: 50
- **Max iterations**: 50
- **Median iterations**: 50
- **Convergence rate**: 0%
