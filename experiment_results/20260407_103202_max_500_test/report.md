# Experiment Report: max_500_test

**Timestamp**: 2026-04-07T10:32:02
**Duration**: 54.0s

## Reproduce

```bash
python -m experiment_framework run --top-k 10 --gradient-decomposition --max-trajectories 500 --name max_500_test
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
| f_fidelity | 0.5000 | 0.5001 | +0.0001 |
| combined | 0.5249 | 0.5249 | -0.0000 |

## Modification Summary

- **Trajectories modified**: 10
- **Converged**: 6
- **Mean iterations**: 33.2
- **Mean perturbation**: 1.97 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 111 | 9 | (17, 38) | (15, 39) | 2.24 | 0.5054 |
| 177 | 36 | (17, 38) | (15, 39) | 2.24 | 0.5046 |
| 150 | 2 | (17, 38) | (15, 39) | 2.24 | 0.5042 |
| 344 | 22 | (35, 15) | (34, 16) | 1.41 | 0.5025 |
| 208 | 39 | (17, 38) | (15, 39) | 2.24 | 0.5009 |
| 390 | 27 | (17, 38) | (19, 40) | 2.83 | 0.5004 |
| 94 | 22 | (28, 16) | (27, 15) | 1.41 | 0.4980 |
| 15 | 19 | (29, 54) | (28, 55) | 1.41 | 0.4979 |
| 57 | 19 | (29, 54) | (28, 55) | 1.41 | 0.4974 |
| 119 | 49 | (41, 41) | (43, 42) | 2.24 | 0.4929 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial effective fraction | 0.358 | 0.073 |
| Causal effective fraction | 0.263 | 0.132 |
| Fidelity effective fraction | 0.378 | 0.089 |
| Spatial-causal alignment | 0.317 | 0.676 |

*Gradient normalization active — fractions reflect effective contribution to step direction (alpha weights renormalized over terms with nonzero gradient signal).*

**Interpretation**: Spatial and causal terms have **mixed** alignment — sometimes cooperating, sometimes conflicting.

## Convergence Statistics

- **Min iterations**: 22
- **Max iterations**: 50
- **Median iterations**: 22
- **Convergence rate**: 60%
