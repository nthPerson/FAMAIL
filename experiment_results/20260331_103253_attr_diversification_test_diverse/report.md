# Experiment Report: attr_diversification_test_diverse

**Timestamp**: 2026-03-31T10:32:53
**Duration**: 137.2s

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
| f_causal | 0.9787 | 0.9787 | +0.0000 |
| f_fidelity | 0.5000 | 0.8985 | +0.3985 |
| combined | 0.5249 | 0.6605 | +0.1355 |

## Modification Summary

- **Trajectories modified**: 10
- **Converged**: 0
- **Mean iterations**: 50.0
- **Mean perturbation**: 2.29 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 9 | 49 | (40, 42) | (42, 44) | 2.83 | 0.6672 |
| 84 | 22 | (24, 9) | (26, 11) | 2.83 | 0.6509 |
| 54 | 19 | (28, 54) | (30, 55) | 2.24 | 0.6493 |
| 15 | 19 | (29, 54) | (31, 55) | 2.24 | 0.6478 |
| 94 | 22 | (28, 16) | (28, 16) | 0.00 | 0.6470 |
| 80 | 19 | (22, 46) | (24, 48) | 2.83 | 0.6470 |
| 3 | 20 | (26, 8) | (26, 10) | 2.00 | 0.6325 |
| 32 | 18 | (13, 29) | (14, 27) | 2.24 | 0.6089 |
| 98 | 45 | (18, 29) | (16, 31) | 2.83 | 0.6053 |
| 35 | 8 | (10, 37) | (8, 39) | 2.83 | 0.6033 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial effective fraction | 0.300 | 0.161 |
| Causal effective fraction | 0.215 | 0.167 |
| Fidelity effective fraction | 0.485 | 0.232 |
| Spatial-causal alignment | 0.213 | 0.490 |

*Gradient normalization active — fractions reflect effective contribution to step direction (alpha weights renormalized over terms with nonzero gradient signal).*

**Interpretation**: Spatial and causal terms have **mixed** alignment — sometimes cooperating, sometimes conflicting.

## Convergence Statistics

- **Min iterations**: 50
- **Max iterations**: 50
- **Median iterations**: 50
- **Convergence rate**: 0%
