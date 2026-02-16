# ST-iFGSM: Gradient-Based Trajectory Editing

FAMAIL adapts the **Spatio-Temporal iterative Fast Gradient Sign Method (ST-iFGSM)** — originally developed for adversarial attacks on trajectory classifiers — as a tool for fairness-aware trajectory editing. Instead of fooling a classifier, the algorithm moves pickup locations in the direction that most improves the combined fairness objective.

---

## Algorithm

<div class="algorithm-box">

**ALGORITHM:** FAMAIL Trajectory Modification

**INPUT:** $T$ trajectories, objective function $\mathcal{L}$, step size $\alpha$, perturbation bound $\epsilon$, max iterations $T_{\max}$, temperature schedule $\{\tau_t\}$

**OUTPUT:** Modified trajectories with improved fairness

---

**Phase 1: Attribution**

1. **FOR** each trajectory $\tau$:
    - Compute $\text{LIS}(\tau)$ — spatial inequality contribution
    - Compute $\text{DCD}(\tau)$ — demand-conditional deviation
    - $\text{Score}(\tau) = w_{\text{LIS}} \cdot \text{LIS}(\tau) + w_{\text{DCD}} \cdot \text{DCD}(\tau)$

2. **SELECT** top-$k$ trajectories by Score (highest impact on unfairness)

---

**Phase 2: Modification**

3. **FOR** each selected trajectory $\tau$:

    - Initialize: $\mathbf{p} \leftarrow$ original pickup location, $\delta_{\text{total}} \leftarrow 0$

4. &emsp;**FOR** $t = 1, 2, \ldots, T_{\max}$:

    - **(a)** Compute soft cell probabilities $\sigma_c(\mathbf{p};\, \tau_t)$
    - **(b)** Evaluate combined objective: $\mathcal{L} = \alpha_1 F_{\text{spatial}} + \alpha_2 F_{\text{causal}} + \alpha_3 F_{\text{fidelity}}$
    - **(c)** Compute gradient: $\nabla_\mathbf{p} \mathcal{L}$ via backpropagation
    - **(d)** Compute perturbation: $\delta = \text{clip}(\alpha \cdot \text{sign}(\nabla_\mathbf{p} \mathcal{L}),\, -\epsilon,\, \epsilon)$
    - **(e)** Update cumulative perturbation: $\delta_{\text{total}} = \text{clip}(\delta_{\text{total}} + \delta,\, -\epsilon,\, \epsilon)$
    - **(f)** Update pickup location: $\mathbf{p} \leftarrow \text{clip}(\mathbf{p}_{\text{original}} + \delta_{\text{total}},\, \text{grid\_bounds})$
    - **(g)** **IF** $|\mathcal{L}_t - \mathcal{L}_{t-1}| < \text{threshold}$: **BREAK** (converged)

5. &emsp;Update global pickup counts: decrement original cell, increment new cell

6. **EVALUATE** updated global fairness metrics

</div>

---

## Key Properties

**Sign gradient.** The $\text{sign}(\cdot)$ function normalizes the gradient to unit magnitude in each dimension, making the step size independent of gradient scale. This is the hallmark of FGSM-type methods.

**Cumulative clipping.** The total perturbation is always bounded by $[-\epsilon, \epsilon]$ per dimension, regardless of how many iterations run. A pickup can move at most $\epsilon$ cells from its original location.

**Sequential processing.** Trajectories are modified one at a time, with global counts updated after each edit. This ensures each modification accounts for previous changes to the service distribution.

**Temperature annealing.** The [soft cell assignment](soft-cell-assignment.md) temperature decreases over iterations, transitioning from smooth exploration (broad gradients) to precise assignment (accurate cell counts).

---

## Adaptation from Adversarial ML

| Concept | Original ST-iFGSM (Adversarial) | FAMAIL Adaptation |
|---------|-------------------------------|-------------------|
| **Objective** | Adversarial loss (fool classifier) | Fairness objective $\mathcal{L}$ |
| **Direction** | Maximize loss (attack) | Maximize fairness (improve) |
| **Perturbation space** | Continuous feature space | Continuous grid coordinates |
| **Constraint** | $L_\infty$ norm bound | $\epsilon$ grid cell bound |
| **Discretization** | N/A | [Soft cell assignment](soft-cell-assignment.md) bridges continuous → discrete |

!!! note "Reframing adversarial methods for social good"
    The core insight of FAMAIL is that adversarial perturbation techniques — designed to attack models — can be repurposed as optimization tools for improving fairness. The mathematical machinery is the same; only the objective changes.

---
