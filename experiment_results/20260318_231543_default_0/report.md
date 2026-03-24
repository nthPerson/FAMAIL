# Experiment Report: default

**Timestamp**: 2026-03-18T23:15:42
**Duration**: 162.9s

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
| f_causal | 0.9787 | 0.9788 | +0.0001 |
| f_fidelity | 0.5000 | 0.8708 | +0.3708 |
| combined | 0.5249 | 0.6510 | +0.1261 |

## Modification Summary

- **Trajectories modified**: 10
- **Converged**: 0
- **Mean iterations**: 50.0
- **Mean perturbation**: 2.47 grid cells

## Top 10 Most Impactful Trajectories

| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |
|-------|--------|-----------|----------|-------------|-----------|
| 505 | 35 | (20, 28) | (22, 30) | 2.83 | 0.6526 |
| 451 | 34 | (20, 28) | (18, 30) | 2.83 | 0.6514 |
| 295 | 35 | (20, 28) | (22, 30) | 2.83 | 0.6507 |
| 208 | 39 | (17, 38) | (19, 37) | 2.24 | 0.6262 |
| 111 | 9 | (17, 38) | (18, 40) | 2.24 | 0.6228 |
| 390 | 27 | (17, 38) | (18, 40) | 2.24 | 0.6153 |
| 150 | 2 | (17, 38) | (19, 37) | 2.24 | 0.6145 |
| 565 | 31 | (14, 31) | (16, 32) | 2.24 | 0.6118 |
| 579 | 40 | (17, 38) | (18, 40) | 2.24 | 0.6110 |
| 177 | 36 | (17, 38) | (19, 40) | 2.83 | 0.6085 |

## Gradient Decomposition Summary

| Metric | Mean | Std |
|--------|------|-----|
| Spatial fraction | 0.026 | 0.035 |
| Causal fraction | 0.196 | 0.245 |
| Fidelity fraction | 0.777 | 0.261 |
| Spatial-causal alignment | -0.701 | 0.521 |

**Interpretation**: Spatial and causal terms are largely **conflicting** — they pull modifications in opposing directions.

## Convergence Statistics

- **Min iterations**: 50
- **Max iterations**: 50
- **Median iterations**: 50
- **Convergence rate**: 0%
