# Experiment Report: verification_run

> End-to-end verification of experiment framework

**Timestamp**: 2026-03-18T20:57:42
**Duration**: 37.5s

## Configuration

| Parameter | Value |
|-----------|-------|
| Trajectories to modify | 5 |
| ST-iFGSM step size | 0.1 |
| Max perturbation | 2.0 |
| Max iterations | 20 |
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
| f_fidelity | 0.5000 | 0.8558 | +0.3558 |
| combined | 0.5272 | 0.6482 | +0.1210 |

## Modification Summary

- **Trajectories modified**: 5
- **Converged**: 0
- **Mean iterations**: 20.0
- **Mean perturbation**: 1.62 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 15 | 19 | (29, 54) | (30, 55) | 1.41 | 0.6159 |
| 38 | 29 | (8, 16) | (9, 14) | 2.24 | 0.5876 |
| 35 | 8 | (10, 37) | (8, 38) | 2.24 | 0.5817 |
| 12 | 2 | (11, 37) | (11, 37) | 0.00 | 0.5796 |
| 42 | 38 | (10, 37) | (8, 38) | 2.24 | 0.5779 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial fraction | 0.758 | 0.406 |
| Causal fraction | 0.000 | 0.000 |
| Fidelity fraction | 0.000 | 0.000 |
| Spatial-causal alignment | 0.000 | 0.000 |

**Interpretation**: Spatial and causal terms have **mixed** alignment — sometimes cooperating, sometimes conflicting.

## Convergence Statistics

- **Min iterations**: 20
- **Max iterations**: 20
- **Median iterations**: 20
- **Convergence rate**: 0%
