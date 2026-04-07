# Experiment Report: causal_dropout_compensation

**Timestamp**: 2026-03-31T10:50:24
**Duration**: 146.5s

## Configuration

| Parameter | Value |
|-----------|-------|
| Trajectories to modify | 10 |
| ST-iFGSM step size | 0.1 |
| Max perturbation | 2.0 |
| Max iterations | 50 |
| Weight: spatial | 0.35 |
| Weight: causal | 0.5 |
| Weight: fidelity | 0.15 |
| Causal formulation | option_b |
| Discriminator | checkpoints/20260316_223817/best.pt |
| Gradient decomposition | True |
| Gradient normalization | True |

## Global Fairness Metrics

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| gini | 0.9031 | 0.9030 | -0.0001 |
| f_spatial | 0.0969 | 0.0970 | +0.0001 |
| f_causal | 0.9787 | 0.9793 | +0.0006 |
| f_fidelity | 0.5000 | 0.9276 | +0.4276 |
| combined | 0.5983 | 0.6628 | +0.0645 |

## Modification Summary

- **Trajectories modified**: 10
- **Converged**: 0
- **Mean iterations**: 50.0
- **Mean perturbation**: 1.74 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 24 | 49 | (40, 42) | (42, 44) | 2.83 | 0.6290 |
| 9 | 49 | (40, 42) | (42, 42) | 2.00 | 0.6287 |
| 54 | 19 | (28, 54) | (30, 55) | 2.24 | 0.6219 |
| 84 | 22 | (24, 9) | (26, 10) | 2.24 | 0.6203 |
| 80 | 19 | (22, 46) | (23, 46) | 1.00 | 0.6202 |
| 15 | 19 | (29, 54) | (31, 55) | 2.24 | 0.6202 |
| 57 | 19 | (29, 54) | (31, 54) | 2.00 | 0.6199 |
| 94 | 22 | (28, 16) | (28, 16) | 0.00 | 0.6186 |
| 3 | 20 | (26, 8) | (25, 9) | 1.41 | 0.6134 |
| 98 | 45 | (18, 29) | (17, 28) | 1.41 | 0.6007 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial effective fraction | 0.423 | 0.209 |
| Causal effective fraction | 0.311 | 0.249 |
| Fidelity effective fraction | 0.265 | 0.227 |
| Spatial-causal alignment | -0.096 | 0.590 |

*Gradient normalization active — fractions reflect effective contribution to step direction (alpha weights renormalized over terms with nonzero gradient signal).*

**Interpretation**: Spatial and causal terms have **mixed** alignment — sometimes cooperating, sometimes conflicting.

## Convergence Statistics

- **Min iterations**: 50
- **Max iterations**: 50
- **Median iterations**: 50
- **Convergence rate**: 0%
