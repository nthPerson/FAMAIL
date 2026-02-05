# FAMAIL Project Documentation

**Fairness-Aware Multi-Agent Imitation Learning**  
San Diego State University

---

## Overview

This directory contains the definitive technical documentation for the FAMAIL project. The documents describe the trajectory modification algorithm, its mathematical foundations, the data infrastructure, and the design decisions that shaped the system.

FAMAIL addresses fairness in taxi service distribution by editing expert driver trajectories to reduce spatial and causal inequality while preserving behavioral fidelity. The system operates on a discretized spatial grid covering Shenzhen, China (48 × 90 cells), using historical GPS data from 50 expert taxi drivers.

## Documents

| Document | Description |
|----------|-------------|
| [Algorithm Overview](algorithm_overview.md) | High-level description of the FAMAIL project, its goals, and the trajectory modification approach. Covers the two-phase pipeline (attribution → modification), the six-step iterative algorithm, and the role of each component. |
| [Mathematical Foundations](mathematical_foundations.md) | Complete mathematical specification of the objective function, fairness terms, soft cell assignment, attribution methods, and gradient-based optimization. Provides all formulas needed for replication and validation. |
| [Data Reference](data_reference.md) | Description of the datasets used by the trajectory modification framework, their structure, coordinate systems, and aggregation strategies. |

## Quick Reference

The FAMAIL trajectory modification framework optimizes:

$$\mathcal{L} = \alpha_1 \cdot F_{\text{spatial}} + \alpha_2 \cdot F_{\text{causal}} + \alpha_3 \cdot F_{\text{fidelity}}$$

where:
- $F_{\text{spatial}} = 1 - \frac{1}{2}(G_{\text{DSR}} + G_{\text{ASR}})$ measures equity of service distribution
- $F_{\text{causal}} = \max(0, R^2)$ measures the causal relationship between demand and supply
- $F_{\text{fidelity}} = f(\tau, \tau')$ measures behavioral similarity to the original trajectory

The algorithm proceeds in two phases:
1. **Attribution**: Rank trajectories by fairness impact using LIS (spatial) and DCD (causal) scores, then select the top-$k$ least fair trajectories.
2. **Modification**: Apply gradient-based perturbations (modified ST-iFGSM) to the selected trajectories to maximize $\mathcal{L}$.

## Relationship to Codebase

| Document Section | Primary Code Reference |
|-----------------|----------------------|
| Objective Function | `trajectory_modification/objective.py` |
| Trajectory Modifier (ST-iFGSM) | `trajectory_modification/modifier.py` |
| Attribution (LIS / DCD) | `trajectory_modification/dashboard.py`, `objective_function/dashboards/components/attribution_integration.py` |
| Soft Cell Assignment | `objective_function/soft_cell_assignment/module.py` |
| Data Loading | `trajectory_modification/data_loader.py` |
| Discriminator (Fidelity) | `trajectory_modification/discriminator_adapter.py` |
| Trajectory Representation | `trajectory_modification/trajectory.py` |
| Global Metrics Tracking | `trajectory_modification/metrics.py` |
