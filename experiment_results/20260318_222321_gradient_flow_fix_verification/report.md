# Experiment Report: gradient_flow_fix_verification

> Verify all three objective terms now have non-zero gradients

**Timestamp**: 2026-03-18T22:23:20
**Duration**: 31.2s

## Configuration

| Parameter | Value |
|-----------|-------|
| Trajectories to modify | 3 |
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
| f_causal | 0.9855 | 0.9855 | -0.0000 |
| f_fidelity | 0.5000 | 0.8572 | +0.3572 |
| combined | 0.5272 | 0.6486 | +0.1214 |

## Modification Summary

- **Trajectories modified**: 3
- **Converged**: 0
- **Mean iterations**: 10.0
- **Mean perturbation**: 0.80 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 15 | 19 | (29, 54) | (30, 55) | 1.41 | 0.6870 |
| 35 | 8 | (10, 37) | (9, 37) | 1.00 | 0.6568 |
| 42 | 38 | (10, 37) | (10, 37) | 0.00 | 0.6501 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial fraction | 0.111 | 0.184 |
| Causal fraction | 0.000 | 0.000 |
| Fidelity fraction | 0.889 | 0.184 |
| Spatial-causal alignment | -0.128 | 0.250 |

**Interpretation**: Spatial and causal terms have **mixed** alignment — sometimes cooperating, sometimes conflicting.

## Convergence Statistics

- **Min iterations**: 10
- **Max iterations**: 10
- **Median iterations**: 10
- **Convergence rate**: 0%
