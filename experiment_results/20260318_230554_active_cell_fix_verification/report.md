# Experiment Report: active_cell_fix_verification

**Timestamp**: 2026-03-18T23:05:54
**Duration**: 43.7s

## Configuration

| Parameter | Value |
|-----------|-------|
| Trajectories to modify | 5 |
| ST-iFGSM step size | 0.1 |
| Max perturbation | 2.0 |
| Max iterations | 10 |
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
| f_causal | 0.9787 | 0.9787 | +0.0000 |
| f_fidelity | 0.5000 | 0.8392 | +0.3392 |
| combined | 0.5249 | 0.6403 | +0.1153 |

## Modification Summary

- **Trajectories modified**: 5
- **Converged**: 0
- **Mean iterations**: 10.0
- **Mean perturbation**: 0.20 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 15 | 19 | (29, 54) | (29, 55) | 1.00 | 0.6513 |
| 38 | 29 | (8, 16) | (8, 16) | 0.00 | 0.6147 |
| 12 | 2 | (11, 37) | (11, 37) | 0.00 | 0.6114 |
| 42 | 38 | (10, 37) | (10, 37) | 0.00 | 0.6041 |
| 35 | 8 | (10, 37) | (10, 37) | 0.00 | 0.5972 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial fraction | 0.102 | 0.180 |
| Causal fraction | 0.034 | 0.084 |
| Fidelity fraction | 0.864 | 0.230 |
| Spatial-causal alignment | 0.101 | 0.682 |

**Interpretation**: Spatial and causal terms have **mixed** alignment — sometimes cooperating, sometimes conflicting.

## Convergence Statistics

- **Min iterations**: 10
- **Max iterations**: 10
- **Median iterations**: 10
- **Convergence rate**: 0%
