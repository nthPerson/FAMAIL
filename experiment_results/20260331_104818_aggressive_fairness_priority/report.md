# Experiment Report: aggressive_fairness_priority

**Timestamp**: 2026-03-31T10:48:18
**Duration**: 135.6s

## Configuration

| Parameter | Value |
|-----------|-------|
| Trajectories to modify | 10 |
| ST-iFGSM step size | 0.1 |
| Max perturbation | 2.0 |
| Max iterations | 50 |
| Weight: spatial | 0.45 |
| Weight: causal | 0.45 |
| Weight: fidelity | 0.1 |
| Causal formulation | option_b |
| Discriminator | checkpoints/20260316_223817/best.pt |
| Gradient decomposition | True |
| Gradient normalization | True |

## Global Fairness Metrics

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| gini | 0.9031 | 0.9030 | -0.0002 |
| f_spatial | 0.0969 | 0.0970 | +0.0002 |
| f_causal | 0.9787 | 0.9787 | +0.0000 |
| f_fidelity | 0.5000 | 0.9276 | +0.4276 |
| combined | 0.5340 | 0.5769 | +0.0428 |

## Modification Summary

- **Trajectories modified**: 10
- **Converged**: 0
- **Mean iterations**: 50.0
- **Mean perturbation**: 2.26 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 24 | 49 | (40, 42) | (42, 44) | 2.83 | 0.5498 |
| 9 | 49 | (40, 42) | (42, 42) | 2.00 | 0.5496 |
| 54 | 19 | (28, 54) | (30, 55) | 2.24 | 0.5451 |
| 84 | 22 | (24, 9) | (26, 11) | 2.83 | 0.5440 |
| 80 | 19 | (22, 46) | (24, 48) | 2.83 | 0.5440 |
| 15 | 19 | (29, 54) | (31, 55) | 2.24 | 0.5439 |
| 57 | 19 | (29, 54) | (31, 56) | 2.83 | 0.5438 |
| 94 | 22 | (28, 16) | (28, 16) | 0.00 | 0.5429 |
| 3 | 20 | (26, 8) | (26, 10) | 2.00 | 0.5394 |
| 98 | 45 | (18, 29) | (16, 31) | 2.83 | 0.5309 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial effective fraction | 0.476 | 0.277 |
| Causal effective fraction | 0.257 | 0.241 |
| Fidelity effective fraction | 0.267 | 0.317 |
| Spatial-causal alignment | 0.152 | 0.482 |

*Gradient normalization active — fractions reflect effective contribution to step direction (alpha weights renormalized over terms with nonzero gradient signal).*

**Interpretation**: Spatial and causal terms have **mixed** alignment — sometimes cooperating, sometimes conflicting.

## Convergence Statistics

- **Min iterations**: 50
- **Max iterations**: 50
- **Median iterations**: 50
- **Convergence rate**: 0%
