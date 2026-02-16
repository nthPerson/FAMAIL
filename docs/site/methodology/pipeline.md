# Two-Phase Trajectory Modification

The FAMAIL trajectory modification algorithm operates in two phases, applied iteratively.

## Phase 1: Attribution

**"Which trajectories should we modify?"**

Not all trajectories contribute equally to unfairness. The attribution phase ranks every trajectory by its impact on global inequality, using two complementary scoring methods:

### Local Inequality Score (LIS)

How much does this trajectory's pickup or dropoff cell deviate from the citywide average service rate? Trajectories in cells far above or below the mean receive high scores.

$$\text{LIS}_c = \frac{|c_{\text{count}} - \mu|}{\mu}$$

### Demand-Conditional Deviation (DCD)

How much does the actual service in this trajectory's pickup cell deviate from what we would expect given the level of demand? Trajectories in cells that are over- or under-served relative to demand receive high scores.

$$\text{DCD}_c = |Y_c - g(D_c)|$$

### Combined Ranking

These scores are normalized to $[0, 1]$ and combined into a single ranking:

$$\text{Score}_\tau = w_{\text{LIS}} \cdot \widetilde{\text{LIS}}_\tau + w_{\text{DCD}} \cdot \widetilde{\text{DCD}}_\tau$$

The top-$k$ highest-impact trajectories are selected for editing.

---

## Phase 2: Modification

**"How should we change them?"**

Selected trajectories are individually modified using a gradient-based optimization algorithm ([ST-iFGSM](algorithm.md)) that iteratively adjusts pickup locations to improve the combined fairness objective:

1. **Compute the gradient** of the objective function with respect to the trajectory's pickup location.
2. **Apply a small, bounded perturbation** in the gradient direction.
3. **Iterate** until the objective converges or a maximum number of iterations is reached.
4. **Update global service counts** to reflect the change before moving to the next trajectory.

---

## Key Constraints

The modification algorithm enforces several constraints to ensure realistic edits:

| Constraint | Description |
|------------|-------------|
| **Spatial bound** ($\epsilon$) | No pickup location can move more than $\epsilon$ grid cells (~3.3 km) from its original position in any direction |
| **Grid boundary** | Modified locations are projected back within the valid study area |
| **Fidelity validation** | A [discriminator network](discriminator.md) evaluates whether the edited trajectory remains behaviorally consistent with the original driver |
| **Sequential processing** | Trajectories are modified one at a time, with global counts updated after each edit |

---

<div style="display: flex; justify-content: space-between; margin-top: 2rem;">
<a href="../../approach/goals/" class="md-button">← Research Goals</a>
<a href="../objective-function/" class="md-button md-button--primary">Next: Objective Function →</a>
</div>
