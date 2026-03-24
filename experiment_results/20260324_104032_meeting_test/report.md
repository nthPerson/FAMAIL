# Experiment Report: meeting_test

**Timestamp**: 2026-03-24T10:40:32
**Duration**: 167.9s

## Configuration

| Parameter | Value |
|-----------|-------|
| Trajectories to modify | 15 |
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
| gini | 0.9031 | 0.9029 | -0.0002 |
| f_spatial | 0.0969 | 0.0971 | +0.0002 |
| f_causal | 0.9787 | 0.9788 | +0.0001 |
| f_fidelity | 0.5000 | 0.9026 | +0.4026 |
| combined | 0.5249 | 0.6619 | +0.1370 |

## Modification Summary

- **Trajectories modified**: 15
- **Converged**: 0
- **Mean iterations**: 50.0
- **Mean perturbation**: 2.16 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 119 | 49 | (41, 41) | (42, 42) | 1.41 | 0.6687 |
| 451 | 34 | (20, 28) | (18, 30) | 2.83 | 0.6526 |
| 384 | 19 | (29, 53) | (31, 54) | 2.24 | 0.6504 |
| 295 | 35 | (20, 28) | (22, 30) | 2.83 | 0.6504 |
| 15 | 19 | (29, 54) | (31, 56) | 2.83 | 0.6467 |
| 344 | 22 | (35, 15) | (35, 17) | 2.00 | 0.6459 |
| 57 | 19 | (29, 54) | (31, 55) | 2.24 | 0.6450 |
| 94 | 22 | (28, 16) | (28, 16) | 0.00 | 0.6439 |
| 200 | 19 | (29, 54) | (31, 55) | 2.24 | 0.6435 |
| 137 | 15 | (14, 15) | (16, 17) | 2.83 | 0.6328 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial fraction | 0.225 | 0.303 |
| Causal fraction | 0.196 | 0.259 |
| Fidelity fraction | 0.579 | 0.337 |
| Spatial-causal alignment | -0.224 | 0.716 |

**Interpretation**: Spatial and causal terms have **mixed** alignment — sometimes cooperating, sometimes conflicting.

## Convergence Statistics

- **Min iterations**: 50
- **Max iterations**: 50
- **Median iterations**: 50
- **Convergence rate**: 0%
