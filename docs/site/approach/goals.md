# Research Goals

FAMAIL pursues four interrelated research objectives:

## 1. Multi-Objective Function

**Develop a multi-objective function** that balances fairness with trajectory fidelity.

The objective function must simultaneously capture spatial equality, demand alignment, and behavioral realism — and it must be fully differentiable to support gradient-based optimization.

$$\mathcal{L} = \alpha_1 \cdot F_{\text{spatial}} + \alpha_2 \cdot F_{\text{causal}} + \alpha_3 \cdot F_{\text{fidelity}}$$

[Learn more about the objective function →](../methodology/objective-function.md)

## 2. Trajectory Modification Algorithm

**Create a trajectory modification algorithm** that iteratively improves fairness.

The algorithm must efficiently identify which trajectories to modify, determine how to modify them, and do so within realistic spatial constraints. FAMAIL adapts the ST-iFGSM algorithm — originally designed for adversarial attacks on trajectory classifiers — as a tool for fairness-aware trajectory editing.

[Learn more about the algorithm →](../methodology/algorithm.md)

## 3. Discriminator Model

**Build a discriminator model** to validate trajectory authenticity.

Edited trajectories must remain indistinguishable from genuine expert trajectories. A Siamese LSTM neural network (modified ST-SiameseNet) serves as a learned "realism check," ensuring that modifications preserve the behavioral signature of the original driver.

[Learn more about the discriminator →](../methodology/discriminator.md)

## 4. Imitation Learning Agents

**Train imitation learning agents** on fairness-edited trajectories.

The ultimate downstream application: agents that learn equitable service patterns from edited expert demonstrations. By embedding fairness into the training data itself, the resulting policies should naturally produce more equitable service distributions.

!!! info "Current Status"

    Goals 1–3 are the focus of ongoing development. Goal 4 represents the downstream application that will follow once the trajectory editing framework is validated.

---
