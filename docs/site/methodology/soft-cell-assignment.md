# Soft Cell Assignment

One of the central technical challenges in FAMAIL is that the objective function depends on **discrete grid cell counts** (e.g., "how many pickups are in cell $(i, j)$?"), but gradient-based optimization requires **continuous, differentiable** functions. Assigning a pickup to a single grid cell is a step function — the gradient is zero almost everywhere.

---

## The Problem

When a pickup location sits at continuous coordinates $(3.2, 7.8)$, the hard assignment says it belongs to cell $(3, 7)$. But if we shift the location slightly to $(3.6, 7.8)$, it still maps to cell $(3, 7)$ — the gradient provides no signal about how to move toward a different cell. Only at the exact boundary does the assignment change, and this produces an undefined (discontinuous) gradient.

---

## The Solution

**Soft cell assignment** replaces the hard (one-cell) assignment with a probability distribution over nearby cells. Instead of saying "this pickup is in cell $(3, 7)$," we say "this pickup has probability 0.82 of being in cell $(3, 7)$, probability 0.08 in cell $(3, 8)$, probability 0.05 in cell $(4, 7)$," and so on.

This is implemented as a **Gaussian softmax** over a local neighborhood:

<div class="equation-box">

$$\sigma_c(\mathbf{p};\, \tau) = \frac{\exp\!\left(-\|\mathbf{p} - \mathbf{c}\|^2 \;/\; 2\tau^2\right)}{\displaystyle\sum_{c' \in \mathcal{N}} \exp\!\left(-\|\mathbf{p} - \mathbf{c}'\|^2 \;/\; 2\tau^2\right)}$$

</div>

where:

- $\mathbf{p}$ is the continuous pickup location
- $\mathbf{c}$ is the center of a grid cell
- $\mathcal{N}$ is the 5 × 5 neighborhood around the original cell
- $\tau$ is the temperature parameter

---

## Temperature Annealing

The temperature $\tau$ controls how "soft" or "hard" the assignment is:

| Temperature | Behavior | When Used |
|-------------|----------|-----------|
| High ($\tau \to \infty$) | Probability spread uniformly across neighborhood | *Not used (theoretical limit)* |
| Moderate ($\tau = 1.0$) | Smooth distribution; broad gradients | Early iterations (exploration) |
| Low ($\tau = 0.1$) | Concentrated on nearest cell; nearly hard assignment | Late iterations (precision) |

During optimization, the temperature follows an **exponential decay schedule**:

<div class="equation-box">

$$\tau_t = \tau_{\max} \cdot \left(\frac{\tau_{\min}}{\tau_{\max}}\right)^{t/T}$$

</div>

from $\tau_{\max} = 1.0$ to $\tau_{\min} = 0.1$. This creates a natural curriculum:

1. **Early iterations:** Soft assignments produce non-zero gradients for all nearby cells, allowing the optimizer to explore broadly and "see" many candidate locations.
2. **Late iterations:** Assignments sharpen to approximate the discrete reality, ensuring the final modified location corresponds to a definite grid cell.

---

## The Full Gradient Path

Soft cell assignment creates a differentiable chain from pickup location to objective value:

```
Pickup location (continuous, differentiable)
        ↓
Soft cell probabilities (Gaussian softmax)
        ↓
Differentiable pickup counts per cell
        ↓
Spatial fairness (Gini)  +  Causal fairness (R²)  +  Fidelity (discriminator)
        ↓
Combined objective L
        ↓
Gradient ∇_p L (via backpropagation)
        ↓
Perturbation δ = clip(α · sign(∇_p L), -ε, ε)
```

!!! tip "Why this matters"
    Without soft cell assignment, FAMAIL could not use gradient-based optimization at all. The entire pipeline — from continuous pickup locations through discrete cell counts to fairness metrics — would be non-differentiable. This module is the bridge that makes end-to-end learning possible.

---

<div style="display: flex; justify-content: space-between; margin-top: 2rem;">
<a href="../algorithm/" class="md-button">← The Algorithm</a>
<a href="../discriminator/" class="md-button md-button--primary">Next: ST-SiameseNet →</a>
</div>
