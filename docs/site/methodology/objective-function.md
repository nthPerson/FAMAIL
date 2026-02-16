# The Multi-Objective Function

At the heart of FAMAIL is a multi-objective function that simultaneously optimizes three goals. The objective is expressed as a weighted sum:

<div class="equation-box">

$$\mathcal{L} = \alpha_1 \cdot F_{\text{spatial}} + \alpha_2 \cdot F_{\text{causal}} + \alpha_3 \cdot F_{\text{fidelity}}$$

</div>

where $\sum_{i} \alpha_i = 1$ and each term lies in $[0, 1]$. The objective is **maximized** — higher values indicate fairer, more realistic outcomes.

---

## The Three Terms

| Term | Name | What It Measures | Goal |
|------|------|-----------------|------|
| $F_{\text{spatial}}$ | **Spatial Fairness** | Equality of taxi service distribution across all grid cells | Equalize service so no area is systematically over- or under-served |
| $F_{\text{causal}}$ | **Causal Fairness** | Whether service allocation aligns with demand rather than demographic factors | Ensure service is driven by genuine demand, not neighborhood wealth |
| $F_{\text{fidelity}}$ | **Trajectory Fidelity** | Whether modified trajectories remain behaviorally realistic | Preserve the authenticity of expert driving patterns |

---

## Spatial Fairness - $F_{\text{spatial}}$

Spatial fairness uses the **Gini coefficient** — a widely-used inequality measure from economics — applied to taxi service rates across the city grid. Service rates are computed by normalizing raw pickup and dropoff counts by the number of active taxis in each cell, isolating true service inequality from differences in taxi presence.

<div class="equation-box">

$$F_{\text{spatial}} = 1 - \frac{1}{2}\left(G_{\text{pickup}} + G_{\text{dropoff}}\right)$$

</div>

where:

- $G = 0$: Perfect equality (identical service rates everywhere)
- $G = 1$: Maximum inequality (all service concentrated in one cell)
- $F_{\text{spatial}} = 1$: Perfectly fair; $F_{\text{spatial}} = 0$: Maximally unfair

For more context on the Gini coefficient and its interpretation, see the [Fairness Definitions](../fairness.md) page.

---

## Causal Fairness - $F_{\text{causal}}$

Causal fairness asks a deeper question than spatial fairness: not just "is service equal?" but "is service driven by the right factors?" Specifically, it measures whether the variance in service ratios across the city can be explained by demand patterns rather than by sensitive demographic attributes like neighborhood income.

The term uses the **coefficient of determination** ($R^2$) to quantify how well a demand-based model predicts the observed service distribution. Higher $R^2$ means more of the service variation is explained by demand — and less by potentially unfair demographic factors.

---

## Trajectory Fidelity - $F_{\text{fidelity}}$

Fidelity ensures that trajectory edits remain realistic. A neural network [discriminator](discriminator.md) evaluates each modified trajectory against the original, producing a similarity score:

- $F_{\text{fidelity}} \approx 1$: The modified trajectory is indistinguishable from the original (good).
- $F_{\text{fidelity}} \approx 0$: The modification is so large that the trajectory no longer resembles the driver's behavior (bad).

Fidelity acts as a **regularizer**, preventing the optimizer from making arbitrarily large changes just to improve fairness.

---

## Differentiability

All three terms — and the entire optimization pipeline — are fully differentiable, enabling gradient-based optimization. This is made possible by a key technical innovation: [soft cell assignment](soft-cell-assignment.md), which bridges the gap between continuous pickup locations and discrete grid cell counts.

!!! info "Design Principle"
    By formulating all objectives as differentiable functions in $[0, 1]$ (higher is better), the terms can be freely combined with different weight configurations to explore the trade-off space between fairness and fidelity.

---
