# Experiment Report: test_run_mf-pc

**Timestamp**: 2026-04-07T10:14:23
**Duration**: 55.3s

## Reproduce

```bash
python -m experiment_framework run --top-k 10 --name test_run_mf-pc --gradient-decomposition
```

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
| gini | 0.9031 | 0.9031 | -0.0000 |
| f_spatial | 0.0969 | 0.0969 | +0.0000 |
| f_causal | 0.9787 | 0.9784 | -0.0002 |
| f_fidelity | 0.5000 | 0.4914 | -0.0086 |
| combined | 0.5249 | 0.5220 | -0.0030 |

## Modification Summary

- **Trajectories modified**: 10
- **Converged**: 5
- **Mean iterations**: 33.2
- **Mean perturbation**: 1.73 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 98 | 45 | (18, 29) | (16, 31) | 2.83 | 0.5058 |
| 84 | 22 | (24, 9) | (22, 9) | 2.00 | 0.5018 |
| 80 | 19 | (22, 46) | (21, 44) | 2.24 | 0.4994 |
| 94 | 22 | (28, 16) | (27, 15) | 1.41 | 0.4980 |
| 15 | 19 | (29, 54) | (28, 55) | 1.41 | 0.4979 |
| 57 | 19 | (29, 54) | (28, 55) | 1.41 | 0.4974 |
| 54 | 19 | (28, 54) | (28, 55) | 1.00 | 0.4967 |
| 3 | 20 | (26, 8) | (25, 8) | 1.00 | 0.4960 |
| 9 | 49 | (40, 42) | (42, 42) | 2.00 | 0.4911 |
| 24 | 49 | (40, 42) | (42, 42) | 2.00 | 0.4907 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial effective fraction | 0.350 | 0.093 |
| Causal effective fraction | 0.264 | 0.138 |
| Fidelity effective fraction | 0.386 | 0.102 |
| Spatial-causal alignment | 0.226 | 0.637 |

*Gradient normalization active — fractions reflect effective contribution to step direction (alpha weights renormalized over terms with nonzero gradient signal).*

**Interpretation**: Spatial and causal terms have **mixed** alignment — sometimes cooperating, sometimes conflicting.

## Convergence Statistics

- **Min iterations**: 2
- **Max iterations**: 50
- **Median iterations**: 36
- **Convergence rate**: 50%
