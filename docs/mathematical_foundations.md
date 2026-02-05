# FAMAIL: Mathematical Foundations

**Complete mathematical specification for the FAMAIL trajectory modification framework.**

---

## Table of Contents

1. [Notation](#1-notation)
2. [Combined Objective Function](#2-combined-objective-function)
3. [Spatial Fairness Term](#3-spatial-fairness-term)
4. [Causal Fairness Term](#4-causal-fairness-term)
5. [Fidelity Term](#5-fidelity-term)
6. [Soft Cell Assignment](#6-soft-cell-assignment)
7. [Temperature Annealing](#7-temperature-annealing)
8. [Attribution Methods](#8-attribution-methods)
9. [ST-iFGSM Modification Algorithm](#9-st-ifgsm-modification-algorithm)
10. [Gradient Derivations](#10-gradient-derivations)
11. [G-Function Estimation](#11-g-function-estimation)

---

## 1. Notation

| Symbol | Definition |
|--------|-----------|
| $\mathcal{T}$ | Set of all trajectories |
| $\tau$ | A single trajectory (sequence of states) |
| $\tau'$ | Modified trajectory |
| $\mathbf{p} = (p_x, p_y)$ | Pickup location in continuous grid coordinates |
| $c = (i, j)$ | Grid cell index |
| $\mathcal{G}$ | The spatial grid, $\mathcal{G} \subset \mathbb{Z}^2$, with $|\mathcal{G}| = 48 \times 90 = 4320$ cells |
| $N_c^{\text{pickup}}$ | Total pickup count in cell $c$ |
| $N_c^{\text{dropoff}}$ | Total dropoff count in cell $c$ |
| $A_c$ | Active taxi count in cell $c$ (mean across time periods) |
| $D_c$ | Demand in cell $c$ (mean pickup count across time periods) |
| $S_c$ | Supply in cell $c$ (mean active taxis across time periods) |
| $Y_c$ | Service ratio: $Y_c = S_c / D_c$ |
| $g(d)$ | Expected service ratio given demand $d$, estimated from historical data |
| $\tau$ (temperature) | Soft cell assignment temperature parameter (context distinguishes from trajectory $\tau$) |
| $\alpha$ | Step size for ST-iFGSM perturbation |
| $\epsilon$ | Maximum perturbation bound per dimension |
| $\alpha_1, \alpha_2, \alpha_3$ | Objective function term weights |
| $k$ | Half-width of neighborhood window: neighborhood is $(2k+1) \times (2k+1)$ |
| $n$ | Number of grid cells (in Gini computation) or number of data points (in variance computation) |
| $G$ | Gini coefficient |
| $R^2$ | Coefficient of determination |

---

## 2. Combined Objective Function

The FAMAIL objective function combines three terms into a scalar value to be maximized:

$$\mathcal{L}(\tau') = \alpha_1 \cdot F_{\text{spatial}} + \alpha_2 \cdot F_{\text{causal}} + \alpha_3 \cdot F_{\text{fidelity}}$$

subject to:

$$\alpha_1 + \alpha_2 + \alpha_3 = 1, \quad \alpha_i \geq 0$$

**Default weights**: $\alpha_1 = 0.33$, $\alpha_2 = 0.33$, $\alpha_3 = 0.34$.

Each term is defined to lie in $[0, 1]$, so $\mathcal{L} \in [0, 1]$. Higher values indicate better outcomes (more fair, more faithful).

### 2.1 Optimization Direction

The objective is maximized: trajectory modifications should increase $\mathcal{L}$. The gradient $\nabla_\mathbf{p} \mathcal{L}$ points in the direction of steepest increase, and the ST-iFGSM algorithm follows this direction.

---

## 3. Spatial Fairness Term

### 3.1 Definition

Spatial fairness measures the equity of taxi service distribution using the Gini coefficient of service rates:

$$F_{\text{spatial}} = 1 - \frac{1}{2}\big(G_{\text{DSR}} + G_{\text{ASR}}\big)$$

where $G_{\text{DSR}}$ and $G_{\text{ASR}}$ are the Gini coefficients of the Departure Service Rate and Arrival Service Rate, respectively.

### 3.2 Service Rates

Service rates normalize raw pickup/dropoff counts by taxi availability, isolating true service inequality from differences in taxi presence:

$$\text{DSR}_c = \frac{N_c^{\text{pickup}}}{A_c}, \qquad \text{ASR}_c = \frac{N_c^{\text{dropoff}}}{A_c}$$

where $A_c > 0$ is the number of active taxis in cell $c$. Only cells with $A_c > \varepsilon$ (a small positive constant, default $\varepsilon = 10^{-8}$) are included. Cells with no active taxis are excluded from the computation to avoid undefined ratios.

### 3.3 Pairwise Gini Coefficient

The Gini coefficient is computed using the **pairwise difference formula**, which is fully differentiable:

$$G(\mathbf{x}) = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |x_i - x_j|}{2 n^2 \bar{x}}$$

where:
- $\mathbf{x} = (x_1, x_2, \ldots, x_n)$ is the vector of service rates across all $n$ cells
- $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i + \varepsilon$ is the mean (with numerical stability constant)

**Properties**:
- $G \in [0, 1]$: $G = 0$ means perfect equality; $G = 1$ means maximum inequality.
- The formula uses only element-wise operations (subtraction, absolute value) and reductions (sum, mean), making it fully compatible with automatic differentiation.
- The result is clamped to $[0, 1]$.

### 3.4 Differentiability of $F_{\text{spatial}}$

$F_{\text{spatial}}$ is differentiable with respect to pickup counts $N_c^{\text{pickup}}$ because:

1. $\text{DSR}_c = N_c^{\text{pickup}} / A_c$ is differentiable (division by a constant $A_c$).
2. The pairwise Gini formula uses only $|\cdot|$ (differentiable almost everywhere, with subgradient at 0), sums, and means.
3. The composition $1 - \frac{1}{2}(G_{\text{DSR}} + G_{\text{ASR}})$ preserves differentiability.

When using soft cell assignment, the pickup counts $N_c^{\text{pickup}}$ are themselves differentiable functions of the pickup location $\mathbf{p}$, completing the gradient chain.

---

## 4. Causal Fairness Term

### 4.1 Definition

Causal fairness measures whether the variance in service ratios across cells can be explained by demand, using the coefficient of determination:

$$F_{\text{causal}} = \max\!\big(0,\; R^2\big)$$

where:

$$R^2 = 1 - \frac{\text{Var}(R)}{\text{Var}(Y)}$$

### 4.2 Components

**Service ratio** (outcome variable):

$$Y_c = \frac{S_c}{D_c + \varepsilon}$$

where $S_c$ is the supply (active taxis) and $D_c$ is the demand (pickup count), both aggregated as **means** across time periods. The $\varepsilon$ term prevents division by zero.

**Expected service ratio** (from the $g(d)$ function):

$$\hat{Y}_c = g(D_c)$$

where $g: \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$ is a monotone non-increasing function estimated from historical data (see Section 11). The function $g$ is **frozen** during optimization — it receives no gradients and does not update.

**Residual**:

$$R_c = Y_c - g(D_c)$$

The residual captures the component of the service ratio that is *not* explained by demand. In a perfectly causally fair system, all variance would be explained by $g(d)$ and the residuals would be zero.

### 4.3 Variance Computation

The variances are computed over all active cells (those with $D_c \geq D_{\min}$, where $D_{\min} = 1.0$ by default):

$$\text{Var}(Y) = \frac{1}{|\mathcal{A}|} \sum_{c \in \mathcal{A}} (Y_c - \bar{Y})^2 + \varepsilon$$

$$\text{Var}(R) = \frac{1}{|\mathcal{A}|} \sum_{c \in \mathcal{A}} (R_c - \bar{R})^2$$

where $\mathcal{A} = \{c \in \mathcal{G} : D_c \geq D_{\min}\}$ is the set of active cells, and $\bar{Y}$, $\bar{R}$ are the respective means.

**Critical requirement**: The demand threshold $D_{\min}$ used during optimization **must match** the threshold used when fitting $g(d)$. A mismatch would include cells with artificially extreme $Y$ values (from tiny denominators), inflating $\text{Var}(Y)$ and producing artificially low $R^2$.

### 4.4 Differentiability of $F_{\text{causal}}$

$F_{\text{causal}}$ is differentiable with respect to pickup counts because:

1. $Y_c = S_c / (D_c + \varepsilon)$ is differentiable with respect to $D_c$ (which comes from pickup counts via soft cell assignment).
2. $g(D_c)$ is frozen (treated as a constant with respect to the optimization variable), so $R_c = Y_c - g(D_c)$ is differentiable through $Y_c$ only.
3. Variance and the $1 - \text{ratio}$ operations are standard differentiable operations.
4. The $\max(0, \cdot)$ clamp has a subgradient at 0.

### 4.5 Scale Matching Requirement

For $F_{\text{causal}}$ to be meaningful, the demand and supply tensors used during optimization must be at the **same scale** as those used to fit $g(d)$. Since $g(d)$ is fitted on hourly-aggregated data (with **mean** aggregation across time periods), the causal demand and supply grids must also use mean aggregation:

$$D_c = \frac{1}{|T_c|} \sum_{t \in T_c} D_{c,t}, \qquad S_c = \frac{1}{|T_c|} \sum_{t \in T_c} S_{c,t}$$

where $T_c$ is the set of time periods observed for cell $c$.

This is distinct from spatial fairness, which uses **sum** aggregation to capture total service volume.

---

## 5. Fidelity Term

### 5.1 Definition

$$F_{\text{fidelity}} = f(\tau, \tau')$$

where $f: \mathcal{T} \times \mathcal{T} \to [0, 1]$ is a pre-trained Siamese LSTM discriminator (ST-SiameseNet).

### 5.2 Discriminator Architecture

The discriminator processes two trajectories through shared-weight LSTM encoders and compares their representations:

$$\mathbf{h}_\tau = \text{LSTM}_\theta(\tau), \quad \mathbf{h}_{\tau'} = \text{LSTM}_\theta(\tau')$$

$$f(\tau, \tau') = \sigma\!\left(\text{MLP}\!\left(\mathbf{h}_\tau \ominus \mathbf{h}_{\tau'}\right)\right)$$

where:
- $\text{LSTM}_\theta$ has hidden dimensions $(200, 100)$ with bidirectional processing
- $\ominus$ denotes the combination operation (element-wise difference)
- $\sigma$ is the sigmoid function
- The model is frozen (parameters fixed, but gradients flow through to input features)

### 5.3 Input Format

Each trajectory is represented as a tensor of shape $[\text{seq\_len}, 4]$ where each row is $[x_{\text{grid}}, y_{\text{grid}}, \text{time\_bucket}, \text{day\_index}]$. A batch dimension is prepended for the discriminator: $[1, \text{seq\_len}, 4]$.

### 5.4 Differentiability

The discriminator uses standard PyTorch operations (`nn.LSTM`, `nn.Linear`, `nn.Sigmoid`, `nn.Dropout`), all of which are differentiable. The discriminator's parameters are frozen (`requires_grad=False`), but gradients flow through the computation graph from $F_{\text{fidelity}}$ back to the modified trajectory features $\tau'$.

### 5.5 Role in Optimization

$F_{\text{fidelity}}$ acts as a regularizer. Without it, the optimizer could move the pickup location to any cell that improves fairness, regardless of whether the resulting trajectory makes sense for the driver. The fidelity term penalizes modifications that make the trajectory too different from the original.

If $\alpha_3 = 0$, the fidelity term is skipped entirely and the discriminator is not required.

---

## 6. Soft Cell Assignment

### 6.1 Motivation

In a discrete grid, a pickup at location $\mathbf{p}$ is assigned to cell $c^* = \text{floor}(\mathbf{p})$. This hard assignment is a step function — the gradient is zero almost everywhere and undefined at cell boundaries. Soft cell assignment replaces this with a continuous relaxation.

### 6.2 Gaussian Softmax Formulation

Given a continuous location $\mathbf{p} = (p_x, p_y)$ and an original cell $c_0 = (i_0, j_0)$, the soft assignment probability for cell $c = (i, j)$ in the neighborhood $\mathcal{N}(c_0)$ is:

$$\sigma_c(\mathbf{p};\, \tau) = \frac{\exp\!\left(-\frac{\|\mathbf{p} - \mathbf{c}\|^2}{2\tau^2}\right)}{\sum_{c' \in \mathcal{N}(c_0)} \exp\!\left(-\frac{\|\mathbf{p} - \mathbf{c}'\|^2}{2\tau^2}\right)}$$

where:
- $\mathbf{c} = (i, j)$ is the center of cell $c$ (cells are unit-spaced, so the center of cell $(i, j)$ is at coordinates $(i, j)$)
- $\|\mathbf{p} - \mathbf{c}\|^2 = (p_x - i)^2 + (p_y - j)^2$ is the squared Euclidean distance
- $\mathcal{N}(c_0) = \{c' : \|c' - c_0\|_\infty \leq k\}$ is the $(2k+1) \times (2k+1)$ neighborhood (default $k = 2$, giving a $5 \times 5$ window)
- $\tau > 0$ is the temperature parameter

### 6.3 Properties

1. **Normalization**: $\sum_{c \in \mathcal{N}} \sigma_c = 1$ (by construction via softmax).
2. **Differentiability**: $\sigma_c$ is infinitely differentiable with respect to $\mathbf{p}$ and $\tau$.
3. **Temperature limit**: As $\tau \to 0^+$, $\sigma_c \to \mathbb{1}[c = c^*]$ where $c^* = \arg\min_c \|\mathbf{p} - \mathbf{c}\|^2$ (approaches hard assignment).
4. **Uniform limit**: As $\tau \to \infty$, $\sigma_c \to 1/|\mathcal{N}|$ (uniform distribution over neighborhood).

### 6.4 Differentiable Soft Counts

Using soft assignment, the differentiable pickup count for cell $c$ is:

$$\tilde{N}_c^{\text{pickup}} = N_c^{\text{base}} + \sigma_c(\mathbf{p};\, \tau)$$

where $N_c^{\text{base}}$ is the base count (total pickups minus the current trajectory's original contribution to the grid). This formulation:

- Removes the current trajectory's hard assignment from the base counts.
- Adds back the trajectory's contribution via soft assignment.
- Ensures the gradient $\partial \tilde{N}_c^{\text{pickup}} / \partial \mathbf{p} = \partial \sigma_c / \partial \mathbf{p} \neq 0$ for cells in the neighborhood.

### 6.5 Boundary Handling

When the neighborhood extends beyond the grid boundary, cells outside the valid range ($0 \leq i < 48$, $0 \leq j < 90$) are excluded from the count update (but remain in the softmax denominator for proper normalization). A validity mask identifies which neighborhood cells fall within the grid.

---

## 7. Temperature Annealing

### 7.1 Annealing Schedule

The temperature follows an exponential decay over the course of optimization:

$$\tau_t = \tau_{\max} \cdot \left(\frac{\tau_{\min}}{\tau_{\max}}\right)^{t/T}$$

where:
- $t$ is the current iteration ($0$-indexed)
- $T$ is the total number of iterations (specifically, $T - 1$ for the exponent base)
- $\tau_{\max} = 1.0$ is the initial temperature (soft, broad gradients)
- $\tau_{\min} = 0.1$ is the final temperature (hard, sharp assignment)

### 7.2 Rationale

- **Early iterations** ($\tau$ large): The soft assignment distributes probability across many cells, producing non-zero gradients everywhere in the neighborhood. This enables the optimizer to "see" many candidate cells and move the pickup toward the most beneficial direction.
- **Late iterations** ($\tau$ small): The assignment sharpens to nearly one-hot, giving accurate cell counts that match the discrete reality. This ensures the final modified pickup location corresponds to a definite grid cell.

### 7.3 Discrete Schedule Values

| Iteration $t$ (of $T = 50$) | $\tau_t$ | Behavior |
|-----|------|------|
| 0   | 1.000 | Broad, smooth gradients |
| 10  | 0.631 | Moderate sharpness |
| 25  | 0.316 | Increasingly focused |
| 40  | 0.158 | Nearly hard assignment |
| 49  | 0.100 | Hard assignment |

---

## 8. Attribution Methods

### 8.1 Overview

Attribution determines which trajectories to prioritize for modification. The FAMAIL framework uses two complementary attribution scores — one for each fairness dimension — combined into a single ranking score.

### 8.2 Local Inequality Score (LIS)

**Purpose**: Measure each trajectory's contribution to spatial inequality.

**Cell-level LIS**:

$$\text{LIS}_c = \frac{|N_c - \mu_N|}{\mu_N}$$

where $N_c$ is the count (pickup or dropoff) in cell $c$ and $\mu_N = \frac{1}{|\mathcal{G}|}\sum_{c' \in \mathcal{G}} N_{c'}$ is the global mean count.

LIS quantifies how far a cell is from the average. Cells with counts far above or below the mean have high LIS — both oversupply and undersupply contribute to inequality.

**Trajectory-level LIS**:

For trajectory $\tau$ with pickup cell $p$ and dropoff cell $d$:

$$\text{LIS}_\tau = \max\!\big(\text{LIS}_p^{\text{pickup}},\; \text{LIS}_d^{\text{dropoff}}\big)$$

The $\max$ aggregation ensures a trajectory is flagged if either endpoint is in a highly unequal cell. The pickup LIS uses the pickup count grid and the dropoff LIS uses the dropoff count grid.

### 8.3 Demand-Conditional Deviation (DCD)

**Purpose**: Measure each trajectory's contribution to causal inequality.

**Cell-level DCD**:

$$\text{DCD}_c = |Y_c - g(D_c)|$$

where:
- $Y_c = S_c / (D_c + \varepsilon)$ is the actual service ratio at cell $c$
- $g(D_c)$ is the expected service ratio for demand level $D_c$

DCD captures the residual — the unexplained deviation from what the demand-supply relationship predicts. High DCD means the cell receives either more or less service than expected given its demand.

**Trajectory-level DCD**:

For trajectory $\tau$ with pickup cell $p$:

$$\text{DCD}_\tau = \text{DCD}_p$$

DCD is evaluated at the pickup cell because the pickup location is the modifiable parameter.

### 8.4 Combined Attribution Score

**Normalization**: Both LIS and DCD scores are normalized to $[0, 1]$ by dividing by the maximum value across all trajectories:

$$\widetilde{\text{LIS}}_\tau = \frac{\text{LIS}_\tau}{\max_{\tau' \in \mathcal{T}} \text{LIS}_{\tau'}}, \qquad \widetilde{\text{DCD}}_\tau = \frac{\text{DCD}_\tau}{\max_{\tau' \in \mathcal{T}} \text{DCD}_{\tau'}}$$

**Weighted combination**:

$$\text{Score}_\tau = w_{\text{LIS}} \cdot \widetilde{\text{LIS}}_\tau + w_{\text{DCD}} \cdot \widetilde{\text{DCD}}_\tau$$

where the default weights are $w_{\text{LIS}} = w_{\text{DCD}} = 0.5$.

### 8.5 Selection Strategies

Given the ranked list of trajectories, $k$ are selected for modification:

**Top-$k$**: Select the $k$ trajectories with the highest $\text{Score}_\tau$.

**Diverse selection**: A greedy algorithm that penalizes spatial clustering. After selecting a trajectory from cell $c$, future candidates from cell $c$ have their effective score reduced:

$$\text{Score}_\tau^{\text{eff}} = \text{Score}_\tau \cdot (1 - \beta \cdot n_c)$$

where $n_c$ is the number of previously selected trajectories from cell $c$ and $\beta = 0.5$ is the penalty factor. This spreads modifications across the city.

---

## 9. ST-iFGSM Modification Algorithm

### 9.1 Algorithm Statement

**Input**: Trajectory $\tau$, objective function $\mathcal{L}$, step size $\alpha$, bound $\epsilon$, max iterations $T$, temperature schedule $\{\tau_t\}_{t=0}^{T-1}$.

**Output**: Modified trajectory $\tau'$.

$$
\begin{aligned}
&\textbf{Initialize:} \quad \mathbf{p}^{(0)} = \mathbf{p}_{\text{original}}, \quad \boldsymbol{\delta}^{(0)} = \mathbf{0} \\
&\textbf{For } t = 0, 1, \ldots, T-1: \\
&\quad 1.\; \text{Set temperature } \tau_t \text{ (if annealing enabled)} \\
&\quad 2.\; \text{Create differentiable tensor } \mathbf{p}^{(t)} \text{ with } \texttt{requires\_grad=True} \\
&\quad 3.\; \tilde{N}^{\text{pickup}} = \text{SoftCounts}\!\left(\mathbf{p}^{(t)},\, \mathbf{p}_{\text{original}},\, N^{\text{base}},\, \tau_t\right) \\
&\quad 4.\; \mathcal{L}^{(t)} = \alpha_1 F_{\text{spatial}}\!\left(\tilde{N}^{\text{pickup}}, N^{\text{dropoff}}, A\right) + \alpha_2 F_{\text{causal}}\!\left(D, S\right) + \alpha_3 F_{\text{fidelity}}\!\left(\tau, \tau'^{(t)}\right) \\
&\quad 5.\; \mathbf{g}^{(t)} = \nabla_{\mathbf{p}} \mathcal{L}^{(t)} \quad \text{(via backpropagation)} \\
&\quad 6.\; \boldsymbol{\delta}^{(t+1)} = \text{clip}\!\left(\boldsymbol{\delta}^{(t)} + \alpha \cdot \text{sign}\!\left(\mathbf{g}^{(t)}\right),\; -\epsilon,\; \epsilon\right) \\
&\quad 7.\; \mathbf{p}^{(t+1)} = \text{clip}\!\left(\mathbf{p}_{\text{original}} + \boldsymbol{\delta}^{(t+1)},\; \mathbf{0},\; \mathbf{p}_{\max}\right) \\
&\quad 8.\; \textbf{If } |\mathcal{L}^{(t)} - \mathcal{L}^{(t-1)}| < \theta: \quad \textbf{break} \\
&\textbf{Return } \tau' \text{ with pickup at } \mathbf{p}^{(T)}
\end{aligned}
$$

### 9.2 Key Details

**Cumulative perturbation**: The perturbation $\boldsymbol{\delta}$ accumulates across iterations but is always clipped to the $[-\epsilon, \epsilon]$ box. The new pickup location is always computed relative to the *original* position: $\mathbf{p}^{(t+1)} = \mathbf{p}_{\text{original}} + \boldsymbol{\delta}^{(t+1)}$.

**Sign gradient**: The $\text{sign}(\cdot)$ function normalizes the gradient to unit magnitude in each dimension, making the step size independent of gradient magnitude. This is the hallmark of FGSM-type methods.

**Grid projection**: The $\text{clip}(\cdot,\, \mathbf{0},\, \mathbf{p}_{\max})$ operation ensures $0 \leq p_x \leq 47$ and $0 \leq p_y \leq 89$.

**Convergence**: The algorithm terminates early if the objective change between iterations falls below $\theta = 10^{-4}$.

### 9.3 Relationship to Original ST-iFGSM

| Concept | Original ST-iFGSM (Adversarial) | FAMAIL Adaptation |
|---------|-------------------------------|-------------------|
| **Objective** | Adversarial loss (fool classifier) | Fairness objective $\mathcal{L}$ |
| **Input representation** | Trajectory embedding | Grid cell coordinates |
| **Perturbation space** | Continuous feature space | Continuous grid coordinates |
| **Constraint** | $L_\infty$ norm bound | $\epsilon$ grid cell bound |
| **Direction** | Maximize loss (attack) | Maximize fairness (improve) |
| **Discretization** | N/A | Soft cell assignment bridges continuous → discrete |

---

## 10. Gradient Derivations

### 10.1 Gradient of $\mathcal{L}$ with Respect to Pickup Location

By the chain rule:

$$\nabla_\mathbf{p} \mathcal{L} = \alpha_1 \nabla_\mathbf{p} F_{\text{spatial}} + \alpha_2 \nabla_\mathbf{p} F_{\text{causal}} + \alpha_3 \nabla_\mathbf{p} F_{\text{fidelity}}$$

### 10.2 Gradient Through Soft Cell Assignment

For any function $h(\tilde{N})$ that depends on the soft pickup counts:

$$\frac{\partial h}{\partial p_x} = \sum_{c \in \mathcal{N}} \frac{\partial h}{\partial \tilde{N}_c} \cdot \frac{\partial \sigma_c}{\partial p_x}$$

The gradient of the soft assignment with respect to location can be derived from the softmax:

$$\frac{\partial \sigma_c}{\partial p_x} = \sigma_c \left[\frac{-(p_x - c_x)}{\tau^2} - \sum_{c'} \sigma_{c'} \cdot \frac{-(p_x - c'_x)}{\tau^2}\right]$$

$$= \frac{\sigma_c}{\tau^2} \left[\sum_{c'} \sigma_{c'} (p_x - c'_x) - (p_x - c_x)\right]$$

This gradient is non-zero for all cells in the neighborhood (as long as $\sigma_c > 0$), enabling the optimizer to evaluate the effect of moving the pickup in any direction.

### 10.3 Gradient Through Spatial Fairness

Let $\text{DSR}_c = \tilde{N}_c^{\text{pickup}} / A_c$. The Gini coefficient gradient with respect to a specific count value $\tilde{N}_c$ involves the partial derivative of the pairwise sum of absolute differences. Since absolute value $|a - b|$ has gradient $\text{sign}(a - b)$ almost everywhere:

$$\frac{\partial G}{\partial \text{DSR}_c} = \frac{1}{2n^2 \bar{x}} \sum_{j \neq c} \text{sign}(\text{DSR}_c - \text{DSR}_j) \cdot 2 - \frac{G}{n \bar{x}}$$

In practice, PyTorch's autograd computes this exactly through the pairwise difference matrix.

### 10.4 Gradient Through Causal Fairness

Since $g(D_c)$ is frozen, the gradient flows only through $Y_c$:

$$\frac{\partial F_{\text{causal}}}{\partial Y_c} = \frac{\partial}{\partial Y_c}\left(1 - \frac{\text{Var}(R)}{\text{Var}(Y)}\right)$$

This involves derivatives of variance with respect to individual elements, which PyTorch autograd handles through the standard variance computation.

### 10.5 Gradient Through Fidelity

The gradient $\nabla_{\tau'} F_{\text{fidelity}}$ flows through the discriminator's LSTM and MLP layers. Because the modification only changes the pickup state (last state in the sequence), only the gradients corresponding to that state position in the LSTM sequence are non-trivial.

---

## 11. G-Function Estimation

### 11.1 Purpose

The function $g: \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$ models the expected relationship between demand and the service ratio:

$$g(d) = \mathbb{E}[Y \mid D = d]$$

It captures the empirical fact that areas with higher demand tend to have lower service ratios (more competition among passengers per available taxi).

### 11.2 Estimation Method: Isotonic Regression

The recommended estimation method is **isotonic regression** with a non-increasing constraint:

$$\hat{g} = \arg\min_{g \in \mathcal{M}^-} \sum_{c \in \mathcal{A}} (Y_c - g(D_c))^2$$

where $\mathcal{M}^-$ is the set of monotone non-increasing functions and $\mathcal{A}$ is the set of active cells (those with $D_c \geq D_{\min}$).

**Why isotonic regression**:
- Non-parametric: Does not impose a specific functional form (e.g., log-linear, power law).
- Monotone decreasing: Captures the economic intuition that higher demand areas have lower per-unit service ratios due to supply limitations.
- `out_of_bounds='clip'`: For demand values outside the training range, the prediction is clipped to the nearest boundary value.

### 11.3 Training Data

$g(d)$ is fitted on hourly-aggregated, per-cell data:

1. **Demand** ($D_c$): Mean pickup count per cell, aggregated hourly.
2. **Service ratio** ($Y_c$): Mean supply-to-demand ratio $S_c / D_c$ per cell, where supply is the active taxi count.
3. **Filtering**: Only cells with $D_c \geq D_{\min} = 1$ are included to avoid extreme ratios from near-zero demand.

The data pipeline:
```
pickup_dropoff_counts.pkl → extract demand → aggregate to hourly → match with active_taxis
→ compute Y = S/D → aggregate per-cell means → fit isotonic regression
```

### 11.4 Frozen During Optimization

$g(d)$ is computed once before the modification loop begins and remains fixed throughout. This prevents circular optimization: if $g(d)$ were updated as trajectories change, the "expected" relationship would shift to match the modifications, undermining the causal fairness criterion.

### 11.5 Diagnostics

The $g(d)$ fitting process reports:
- **$R^2$** of the fit (how well isotonic regression captures the demand-ratio relationship)
- **Number of data points** used for fitting
- **Demand and ratio ranges** in the training data
- **Aggregation method** (mean, sum, or max)

---

## Appendix A: Implementation Constants

| Constant | Value | Usage |
|----------|-------|-------|
| Grid dimensions | $48 \times 90$ | Spatial grid over Shenzhen |
| $\varepsilon$ (numerical) | $10^{-8}$ | Division safety in DSR, ASR, variance |
| $D_{\min}$ (demand threshold) | $1.0$ | Minimum demand for causal fairness cells |
| Min cells for variance | $2$ | Minimum active cells for meaningful $R^2$ |
| Step size $\alpha$ | $0.1$ | ST-iFGSM perturbation step size |
| Perturbation bound $\epsilon$ | $3.0$ | Maximum grid cells of perturbation per dimension |
| Max iterations $T$ | $50$ | Maximum ST-iFGSM iterations per trajectory |
| Convergence threshold $\theta$ | $10^{-4}$ | Early stopping criterion |
| Neighborhood size | $5 \times 5$ ($k = 2$) | Soft cell assignment window |
| $\tau_{\max}$ | $1.0$ | Initial temperature (soft) |
| $\tau_{\min}$ | $0.1$ | Final temperature (hard) |
| LIS weight $w_{\text{LIS}}$ | $0.5$ | Default attribution weight for spatial fairness |
| DCD weight $w_{\text{DCD}}$ | $0.5$ | Default attribution weight for causal fairness |
| Diverse selection penalty $\beta$ | $0.5$ | Spatial clustering penalty factor |

## Appendix B: Summary of Differentiability

| Component | Differentiable? | Mechanism |
|-----------|----------------|-----------|
| Pickup location $\mathbf{p}$ | Yes | Input tensor with `requires_grad=True` |
| Soft cell probabilities $\sigma_c$ | Yes | Gaussian softmax (smooth, continuous) |
| Soft pickup counts $\tilde{N}_c$ | Yes | Sum of base counts + soft probabilities |
| DSR, ASR | Yes | Division by constant $A_c$ |
| Pairwise Gini coefficient $G$ | Yes | Abs differences and mean (continuous ops) |
| $F_{\text{spatial}}$ | Yes | $1 - 0.5(G_{\text{DSR}} + G_{\text{ASR}})$ |
| Service ratio $Y_c$ | Yes | $S_c / (D_c + \varepsilon)$ |
| $g(D_c)$ | **Frozen** | Pre-computed, no gradient flow |
| Residual $R_c$ | Yes (through $Y_c$) | $Y_c - g(D_c)$ |
| $R^2$ | Yes | Standard variance operations |
| $F_{\text{causal}}$ | Yes | $\max(0, R^2)$ |
| Discriminator $f(\tau, \tau')$ | Yes (through $\tau'$) | LSTM + MLP, frozen parameters |
| $F_{\text{fidelity}}$ | Yes | Sigmoid output of discriminator |
| $\mathcal{L}$ | Yes | Weighted sum of terms |
