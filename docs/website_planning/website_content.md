# FAMAIL Website — Content Document

**Purpose:** This document serves as the informational basis for the public-facing FAMAIL project website. It contains all content that should appear on the site, organized by section. The website tells the story of the FAMAIL project — its motivation, methodology, data, and technical approach — without presenting any concrete experimental results (which remain preliminary).

**Color Palette (consistent with existing presentations):**
| Token | Hex | Usage |
|-------|-----|-------|
| Primary | `#A6192E` | Headings, accents, borders |
| Secondary | `#CDCDC8` | Backgrounds, dividers |
| Tertiary / Teal | `#008080` | Subheadings, highlights, info boxes |
| Charcoal | `#2D2828` | Body text |
| White | `#FFFFFF` | Card / section backgrounds |

**Affiliation:** San Diego State University · Department of Computer Science

---

## Table of Contents

1. [Hero Section](#1-hero-section)
2. [The Problem](#2-the-problem)
3. [Project Overview](#3-project-overview)
4. [Research Goals](#4-research-goals)
5. [Study Area & Data](#5-study-area--data)
6. [Methodology Overview](#6-methodology-overview)
7. [The Objective Function](#7-the-objective-function)
8. [The Trajectory Modification Algorithm](#8-the-trajectory-modification-algorithm)
9. [Soft Cell Assignment — Bridging Discrete and Continuous](#9-soft-cell-assignment--bridging-discrete-and-continuous)
10. [Discriminator: Trajectory Authenticity](#10-discriminator-trajectory-authenticity)
11. [Fairness Definitions](#11-fairness-definitions)
12. [Related Work](#12-related-work)
13. [Team](#13-team)
14. [Footer](#14-footer)

---

## 1. Hero Section

### Title
**FAMAIL: Fairness-Aware Multi-Agent Imitation Learning**

### Tagline
*Improving Spatial-Temporal Fairness in Taxi Services Through Targeted Trajectory Modification*

### One-Paragraph Summary
FAMAIL is a research project at San Diego State University that addresses spatial inequality in urban taxi services. Using GPS trajectory data from Shenzhen, China, we develop trajectory editing techniques that modify expert driver trajectories to improve fairness metrics — ensuring that all areas of a city receive equitable taxi service — while maintaining the behavioral realism of the original trajectories.

---

## 2. The Problem

### Section Title
**The Urban Taxi Fairness Problem**

### Content

Taxi services in cities like Shenzhen exhibit significant spatial inequality. Certain areas — often wealthier, centrally-located neighborhoods — receive disproportionately more service relative to demand, while other areas — frequently lower-income or peripheral neighborhoods — are systematically underserved.

This inequality arises naturally from expert driver behavior: drivers learn to optimize for personal revenue, gravitating toward high-demand, high-tip areas and avoiding less profitable zones. Over time, this creates a self-reinforcing cycle:

- **High-service areas** attract more taxis → shorter wait times → more riders → more revenue
- **Low-service areas** have fewer taxis → longer wait times → fewer riders → less incentive for drivers

When this spatial inequality correlates with socioeconomic factors — such as neighborhood income levels — it raises a systemic fairness issue in urban mobility.

### Key Question (callout box)

> *Can we modify how taxi drivers serve a city to make the distribution of service more equitable — without compromising the realism of driver behavior?*

---

## 3. Project Overview

### Section Title
**What is FAMAIL?**

### Content

FAMAIL (Fairness-Aware Multi-Agent Imitation Learning) takes a novel approach to the urban taxi fairness problem: rather than generating entirely new synthetic trajectories, we **edit existing expert driver trajectories** to improve fairness.

This editing approach is central to the project's philosophy:

| Approach | Description | FAMAIL? |
|----------|-------------|---------|
| **Trajectory Generation** | Create entirely new synthetic trajectories from scratch | ✗ |
| **Trajectory Editing** | Apply small, bounded adjustments to real expert trajectories | ✓ |

Editing is preferred for several reasons:

1. **Preserves expert knowledge:** Driving patterns, route choices, and temporal behaviors are retained from the original human drivers.
2. **Maximizes efficiency:** Not all trajectories contribute equally to global unfairness, so only the most impactful trajectories are modified.
3. **Bounded modifications:** Spatial constraints limit how far pickup locations can move, ensuring modifications remain realistic.
4. **Enables fidelity validation:** A neural network discriminator can verify that a modified trajectory still "looks like" it came from the same driver.

### Pipeline (visual diagram description)

The overall FAMAIL pipeline:

```
Expert GPS Trajectories (Shenzhen, 50 drivers)
        │
        ▼
┌────────────────────────┐
│  1. Identify Unfair     │   ← Which trajectories contribute
│     Trajectories        │      most to spatial inequality?
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  2. Edit Pickup         │   ← Move pickup locations toward
│     Locations           │      underserved areas
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  3. Validate Fidelity   │   ← Does the edited trajectory
│     (Discriminator)     │      still look realistic?
└────────┬───────────────┘
         │
         ▼
  Fairness-Improved Trajectories
        │
        ▼
  Train Imitation Learning Agents
  on Fairer Expert Demonstrations
```

### Long-Term Vision

The edited trajectories are ultimately intended to serve as training data for imitation learning agents. By training agents on fairness-improved demonstrations, we aim to produce autonomous taxi policies that naturally distribute service more equitably — embedding fairness directly into learned driving behavior.

---

## 4. Research Goals

### Section Title
**Research Goals**

### Content

FAMAIL pursues four interrelated research objectives:

1. **Develop a multi-objective function** that balances fairness with trajectory fidelity.

   The objective function must simultaneously capture spatial equality, demand alignment, and behavioral realism — and it must be fully differentiable to support gradient-based optimization.

2. **Create a trajectory modification algorithm** that iteratively improves fairness.

   The algorithm must efficiently identify which trajectories to modify, determine how to modify them, and do so within realistic spatial constraints.

3. **Build a discriminator model** to validate trajectory authenticity.

   Edited trajectories must remain indistinguishable from genuine expert trajectories. A neural network discriminator serves as a learned "realism check."

4. **Train imitation learning agents** on fairness-edited trajectories.

   The ultimate downstream application: agents that learn equitable service patterns from the edited demonstrations.

---

## 5. Study Area & Data

### Section Title
**Study Area and Data Sources**

### Study Area

The FAMAIL framework operates on real-world GPS taxi trajectory data from **Shenzhen, China** — one of the largest and most economically dynamic cities in southern China. Shenzhen's taxi fleet is one of the densest in the world, and the city's rapid economic growth has produced significant variation in neighborhood wealth and service access.

### Spatial and Temporal Resolution

| Parameter | Value |
|-----------|-------|
| **Geographic Area** | Shenzhen, China |
| **Spatial Grid** | 48 × 90 cells (~1.1 km per cell) |
| **Total Grid Cells** | 4,320 |
| **Temporal Resolution** | 288 time buckets per day (5-minute intervals) |
| **Analysis Period** | July–September 2016 (weekdays only) |
| **Fleet** | 50 expert taxi drivers (subset of 17,877 total) |

### Data Sources

The project uses three primary data sources, all derived from GPS traces of the 50 expert drivers:

**1. Taxi Trajectories**

GPS-derived trajectory sequences recording each driver's path from the start of passenger-seeking behavior to the pickup event. Each trajectory is a sequence of spatiotemporal states — grid cell coordinates and time-of-day — capturing the driver's movement through the city.

**2. Pickup and Dropoff Counts**

Aggregated event data recording how many passenger pickups and dropoffs occurred at each grid cell, in each 5-minute time window, on each day. This serves as the empirical measure of taxi service distribution across the city.

**3. Active Taxi Counts**

For each grid cell and time period, how many unique taxis were present in the surrounding neighborhood. This measures taxi supply — how many drivers were available to serve each area — and is essential for computing service rates that account for taxi availability.

### Demographic Data

District-level demographic data for Shenzhen's 10 administrative districts provides the basis for measuring whether service inequality correlates with socioeconomic factors. Key indicators include:

- GDP per capita
- Average housing price
- Employee compensation per capita
- Population density

Each grid cell is mapped to its majority-overlap Shenzhen district via ArcGIS spatial analysis, allowing demographic attributes to be associated with service patterns at the grid level.

### Study Area Map (description for visual)

> **Visual:** A map of Shenzhen divided into its 10 administrative districts, overlaid with the 48 × 90 grid. Districts are color-coded by a demographic indicator (e.g., GDP per capita). The grid cells are faintly outlined to show the spatial resolution of the analysis. A legend identifies each district.

---

## 6. Methodology Overview

### Section Title
**Methodology: Two-Phase Trajectory Modification**

### Content

The FAMAIL trajectory modification algorithm operates in two phases, applied iteratively:

### Phase 1: Attribution — "Which trajectories should we modify?"

Not all trajectories contribute equally to unfairness. The attribution phase ranks every trajectory by its impact on global inequality, using two complementary scoring methods:

- **Local Inequality Score (LIS):** How much does this trajectory's pickup or dropoff cell deviate from the citywide average service rate? Trajectories in cells far above or below the mean receive high scores.

- **Demand-Conditional Deviation (DCD):** How much does the actual service in this trajectory's pickup cell deviate from what we would expect given the level of demand? Trajectories in cells that are over- or under-served relative to demand receive high scores.

These scores are combined into a single ranking, and the top-$k$ highest-impact trajectories are selected for editing.

### Phase 2: Modification — "How should we change them?"

Selected trajectories are individually modified using a gradient-based optimization algorithm (ST-iFGSM) that iteratively adjusts pickup locations to improve the combined fairness objective. The algorithm:

1. Computes the gradient of the objective function with respect to the trajectory's pickup location.
2. Applies a small, bounded perturbation in the gradient direction.
3. Iterates until the objective converges or a maximum number of iterations is reached.
4. Updates global service counts to reflect the change before moving to the next trajectory.

### Key Constraints

The modification algorithm enforces several constraints to ensure realistic edits:

- **Spatial bound ($\epsilon$):** No pickup location can move more than $\epsilon$ grid cells (default: 3 cells, ~3.3 km) from its original position in any direction.
- **Grid boundary enforcement:** Modified locations are projected back within the valid study area.
- **Fidelity validation:** A discriminator network evaluates whether the edited trajectory remains behaviorally consistent with the original driver.

---

## 7. The Objective Function

### Section Title
**The Multi-Objective Function**

### Content

At the heart of FAMAIL is a multi-objective function that simultaneously optimizes three goals. The objective is expressed as a weighted sum:

$$\mathcal{L} = \alpha_1 \cdot F_{\text{spatial}} + \alpha_2 \cdot F_{\text{causal}} + \alpha_3 \cdot F_{\text{fidelity}}$$

where $\alpha_1 + \alpha_2 + \alpha_3 = 1$ and each term lies in $[0, 1]$. The objective is **maximized** — higher values indicate fairer, more realistic outcomes.

### The Three Terms

| Term | Name | What It Measures | Goal |
|------|------|-----------------|------|
| $F_{\text{spatial}}$ | **Spatial Fairness** | Equality of taxi service distribution across all grid cells | Equalize service so no area is systematically over- or under-served |
| $F_{\text{causal}}$ | **Causal Fairness** | Whether service allocation aligns with demand rather than demographic factors | Ensure service is driven by genuine demand, not neighborhood wealth |
| $F_{\text{fidelity}}$ | **Trajectory Fidelity** | Whether modified trajectories remain behaviorally realistic | Preserve the authenticity of expert driving patterns |

### Spatial Fairness ($F_{\text{spatial}}$)

Spatial fairness uses the **Gini coefficient** — a widely-used inequality measure from economics — applied to taxi service rates across the city grid. Service rates are computed by normalizing raw pickup and dropoff counts by the number of active taxis in each cell, isolating true service inequality from differences in taxi presence.

$$F_{\text{spatial}} = 1 - \frac{1}{2}\left(G_{\text{pickup}} + G_{\text{dropoff}}\right)$$

- $G = 0$: Perfect equality (identical service rates everywhere)
- $G = 1$: Maximum inequality (all service concentrated in one cell)
- $F_{\text{spatial}} = 1$: Perfectly fair; $F_{\text{spatial}} = 0$: Maximally unfair

### Causal Fairness ($F_{\text{causal}}$)

Causal fairness asks a deeper question than spatial fairness: not just "is service equal?" but "is service driven by the right factors?" Specifically, it measures whether the variance in service ratios across the city can be explained by demand patterns rather than by sensitive demographic attributes like neighborhood income.

The term uses the coefficient of determination ($R^2$) to quantify how well a demand-based model predicts the observed service distribution. Higher $R^2$ means more of the service variation is explained by demand — and less by potentially unfair demographic factors.

### Trajectory Fidelity ($F_{\text{fidelity}}$)

Fidelity ensures that trajectory edits remain realistic. A neural network discriminator evaluates each modified trajectory against the original, producing a similarity score:

- $F_{\text{fidelity}} \approx 1$: The modified trajectory is indistinguishable from the original (good).
- $F_{\text{fidelity}} \approx 0$: The modification is so large that the trajectory no longer resembles the driver's behavior (bad).

Fidelity acts as a regularizer, preventing the optimizer from making arbitrarily large changes just to improve fairness.

### Differentiability

All three terms — and the entire optimization pipeline — are fully differentiable, enabling gradient-based optimization. This is made possible by a key technical innovation: **soft cell assignment** (described below), which bridges the gap between continuous pickup locations and discrete grid cell counts.

---

## 8. The Trajectory Modification Algorithm

### Section Title
**ST-iFGSM: Gradient-Based Trajectory Editing**

### Content

FAMAIL adapts the **Spatio-Temporal iterative Fast Gradient Sign Method (ST-iFGSM)** — originally developed for adversarial attacks on trajectory classifiers — as a tool for fairness-aware trajectory editing. Instead of fooling a classifier, the algorithm moves pickup locations in the direction that most improves the combined fairness objective.

### Algorithm Pseudocode

```
ALGORITHM: FAMAIL Trajectory Modification
════════════════════════════════════════════════════════════════

INPUT:  T trajectories, objective function L, step size α,
        perturbation bound ε, max iterations T_max,
        temperature schedule {τ_t}

OUTPUT: Modified trajectories with improved fairness

─── Phase 1: Attribution ───────────────────────────────────────

1.  FOR each trajectory τ:
      Compute LIS(τ)  ← spatial inequality contribution
      Compute DCD(τ)  ← demand-conditional deviation
      Score(τ) = w_LIS · LIS(τ) + w_DCD · DCD(τ)

2.  SELECT top-k trajectories by Score (highest impact on unfairness)

─── Phase 2: Modification ──────────────────────────────────────

3.  FOR each selected trajectory τ:

      Initialize: p ← original pickup location
                  δ_total ← 0

4.    FOR t = 1, 2, ..., T_max:

        a.  Compute soft cell probabilities σ_c(p; τ_t)
            (distribute pickup probability across nearby cells)

        b.  Evaluate combined objective:
            L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity

        c.  Compute gradient: ∇_p L via backpropagation

        d.  Compute perturbation:
            δ = clip(α · sign(∇_p L), -ε, ε)

        e.  Update cumulative perturbation:
            δ_total = clip(δ_total + δ, -ε, ε)

        f.  Update pickup location:
            p ← clip(p_original + δ_total, grid_bounds)

        g.  IF |L_t - L_{t-1}| < convergence_threshold:
              BREAK (converged)

5.    Update global pickup counts:
        decrement original cell, increment new cell

6.  EVALUATE updated global fairness metrics
```

### Key Properties

- **Sign gradient:** The $\text{sign}(\cdot)$ function normalizes the gradient to unit magnitude in each dimension, making the step size independent of gradient scale. This is the hallmark of FGSM-type methods.

- **Cumulative clipping:** The total perturbation is always bounded by $[-\epsilon, \epsilon]$ per dimension, regardless of how many iterations run. A pickup can move at most $\epsilon$ cells from its original location.

- **Sequential processing:** Trajectories are modified one at a time, with global counts updated after each edit. This ensures each modification accounts for previous changes to the service distribution.

- **Temperature annealing:** The soft cell assignment temperature decreases over iterations, transitioning from smooth exploration (broad gradients) to precise assignment (accurate cell counts).

### Adaptation from Adversarial ML

| Concept | Original ST-iFGSM (Adversarial) | FAMAIL Adaptation |
|---------|-------------------------------|-------------------|
| **Objective** | Adversarial loss (fool classifier) | Fairness objective $\mathcal{L}$ |
| **Direction** | Maximize loss (attack) | Maximize fairness (improve) |
| **Perturbation space** | Continuous feature space | Continuous grid coordinates |
| **Constraint** | $L_\infty$ norm bound | $\epsilon$ grid cell bound |
| **Discretization** | N/A | Soft cell assignment bridges continuous → discrete |

---

## 9. Soft Cell Assignment — Bridging Discrete and Continuous

### Section Title
**Soft Cell Assignment: A Key Technical Innovation**

### Content

One of the central technical challenges in FAMAIL is that the objective function depends on **discrete grid cell counts** (e.g., "how many pickups are in cell $(i, j)$?"), but gradient-based optimization requires **continuous, differentiable** functions. Assigning a pickup to a single grid cell is a step function — the gradient is zero almost everywhere.

### The Solution

**Soft cell assignment** replaces the hard (one-cell) assignment with a probability distribution over nearby cells. Instead of saying "this pickup is in cell $(3, 7)$," we say "this pickup has probability 0.82 of being in cell $(3, 7)$, probability 0.08 in cell $(3, 8)$, probability 0.05 in cell $(4, 7)$," and so on.

This is implemented as a **Gaussian softmax** over a local neighborhood:

$$\sigma_c(\mathbf{p};\, \tau) = \frac{\exp\!\left(-\|\mathbf{p} - \mathbf{c}\|^2 \;/\; 2\tau^2\right)}{\displaystyle\sum_{c' \in \mathcal{N}} \exp\!\left(-\|\mathbf{p} - \mathbf{c}'\|^2 \;/\; 2\tau^2\right)}$$

where:
- $\mathbf{p}$ is the continuous pickup location
- $\mathbf{c}$ is the center of a grid cell
- $\mathcal{N}$ is the 5 × 5 neighborhood around the original cell
- $\tau$ is the temperature parameter

### Temperature Annealing

The temperature $\tau$ controls how "soft" or "hard" the assignment is:

| Temperature | Behavior | When Used |
|-------------|----------|-----------|
| High ($\tau \to \infty$) | Probability spread uniformly across neighborhood | *Not used (theoretical limit)* |
| Moderate ($\tau = 1.0$) | Smooth distribution; broad gradients | Early iterations (exploration) |
| Low ($\tau = 0.1$) | Concentrated on nearest cell; nearly hard assignment | Late iterations (precision) |

During optimization, the temperature follows an **exponential decay schedule** from $\tau_{\max} = 1.0$ to $\tau_{\min} = 0.1$. This creates a natural curriculum:

1. **Early iterations:** Soft assignments produce non-zero gradients for all nearby cells, allowing the optimizer to explore broadly and "see" many candidate locations.
2. **Late iterations:** Assignments sharpen to approximate the discrete reality, ensuring the final modified location corresponds to a definite grid cell.

### The Full Gradient Path

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

---

## 10. Discriminator: Trajectory Authenticity

### Section Title
**ST-SiameseNet: Validating Trajectory Authenticity**

### Content

When we modify a trajectory, how do we know the result still looks like a real taxi trip? FAMAIL uses a **Siamese LSTM neural network** — called ST-SiameseNet — that has been trained to determine whether two trajectory sequences were generated by the same driver.

### Architecture Overview

The discriminator takes two trajectories as input and outputs a similarity score between 0 and 1:

```
Original Trajectory τ ──→ ┌──────────┐
                          │  Shared   │
                          │  LSTM     │──→ Compare ──→ Similarity
                          │  Encoder  │       ↑         Score
Modified Trajectory τ′ ──→└──────────┘       │        ∈ [0, 1]
                          (shared weights)    │
                                         MLP + Sigmoid
```

- **Input:** Each trajectory is a sequence of spatiotemporal states: grid coordinates, time of day, and day of week.
- **Encoding:** A bidirectional LSTM with shared weights processes both trajectories into fixed-length representations.
- **Comparison:** The representations are compared via an MLP that outputs $P(\text{same driver})$.
- **Output:** A score near 1.0 means the trajectories appear to come from the same driver; near 0.0 means they look like different drivers.

### Role in the Optimization

During trajectory modification, the discriminator is **frozen** — its parameters are not updated. However, gradients flow through the discriminator back to the modified trajectory, allowing the optimizer to learn which modifications preserve driver-consistent behavior. The fidelity score acts as a regularization term, preventing the optimizer from making unrealistically large location shifts just to improve fairness.

---

## 11. Fairness Definitions

### Section Title
**What Does "Fairness" Mean in FAMAIL?**

### Content

FAMAIL operationalizes fairness through two complementary lenses:

### Spatial Fairness: Equal Access to Service

Spatial fairness asks: **Is taxi service distributed equally across the city?**

We measure this using the **Gini coefficient** — the same measure economists use to quantify income inequality — applied to taxi service rates. For each grid cell, we compute a service rate by normalizing the number of pickups (or dropoffs) by the number of active taxis available in that area. This controls for the fact that some areas simply have more taxis present.

A Gini coefficient of 0 represents perfect equality (every cell has the same service rate), while a coefficient of 1 represents maximum inequality (all service is concentrated in a single cell).

*Intuition:* If two neighborhoods have the same number of available taxis, they should have roughly similar pickup rates. Large deviations indicate that some areas are systematically favored over others.

### Causal Fairness: Service Driven by Demand, Not Demographics

Causal fairness asks a deeper question: **Is the service distribution explained by legitimate factors (demand) rather than sensitive factors (demographics)?**

It is acceptable — even expected — that areas with higher passenger demand receive more service. What is *not* acceptable is when two areas with the same demand receive different levels of service because one is wealthier than the other.

FAMAIL measures causal fairness by:

1. **Modeling the demand-service relationship:** What service level should each area receive, given its level of demand?
2. **Computing residuals:** How much does the actual service deviate from the demand-predicted level?
3. **Checking for demographic bias:** Do those residuals correlate with neighborhood demographics (income, housing prices, etc.)?

If service deviations are independent of demographics, the system achieves causal fairness — service allocation is driven by demand alone, not by where wealthy or poor residents live.

### Why Both?

These two fairness measures are complementary:

- **Spatial fairness** can be high even if the allocation is driven by demographics, as long as the total distribution happens to be uniform.
- **Causal fairness** can be high even if the distribution is unequal, as long as the inequality is fully explained by demand.

True fairness requires both: equitable distribution (spatial) that is driven by legitimate factors (causal).

---

## 12. Related Work

### Section Title
**Foundations and Related Work**

### Content

FAMAIL builds on research from several fields:

### Imitation Learning for Urban Mobility

The cGAIL (Conditional Generative Adversarial Imitation Learning) framework provides the foundation for modeling expert taxi driver behavior. cGAIL trains agents to replicate human driving patterns using GPS trajectory data, producing the expert trajectories that FAMAIL subsequently edits for fairness.

> Zhang, X., et al. "Conditional Generative Adversarial Imitation Learning for Taxi Driver Trajectories"

### Adversarial Trajectory Perturbation

The ST-iFGSM (Spatio-Temporal iterative Fast Gradient Sign Method) algorithm was originally developed for adversarial robustness testing of human mobility identification models. FAMAIL repurposes this technique: instead of attacking a model, we use gradient-based perturbation to *improve* a fairness objective.

> Zhang, X., et al. "ST-iFGSM: Enhancing Robustness of Human Mobility Signature Identification Model via Spatial-Temporal Iterative FGSM"

### Measuring Spatial Inequality in Transportation

The application of Gini coefficients to measure spatial inequality in taxi services — distinguishing between supply-side and demand-side inequality — provides the mathematical foundation for our spatial fairness term.

> Su, L., Yan, Z., & Cao, J. (2018). "Uncovering Spatial Inequality in Taxi Services in the Context of a Subsidy War among E-Hailing Apps"

### Causal Approaches to Fairness

The causal reasoning framework for algorithmic fairness — in particular, the idea that fairness requires ruling out the influence of sensitive attributes on outcomes — informs our causal fairness formulation.

> Kilbertus, N., et al. "Avoiding Discrimination through Causal Reasoning"

---

## 13. Team

### Section Title
**Team**

### Members

| Role | Name | Affiliation |
|------|------|-------------|
| **Researcher** | Robert Ashe | San Diego State University |
| **Advisor** | Dr. Xin Zhang | San Diego State University |

### Affiliation

San Diego State University · Department of Computer Science

### Project Timeline

2025–2026 (ongoing)

---

## 14. Footer

**FAMAIL** | Fairness-Aware Multi-Agent Imitation Learning
San Diego State University · Computer Science
© 2025–2026

---

## Appendix: Website Implementation Notes

### Recommended Page Structure

The content above can be organized as either a single-page scrolling site or a multi-page site. For a single-page site, each numbered section becomes a full-width panel. For a multi-page site, suggested groupings:

| Page | Sections |
|------|----------|
| **Home** | 1 (Hero), 2 (Problem), 3 (Overview) |
| **Methodology** | 6 (Two-Phase Pipeline), 7 (Objective Function), 8 (Algorithm), 9 (Soft Cell Assignment) |
| **Data** | 5 (Study Area & Data) |
| **Fairness** | 11 (Fairness Definitions), 4 (Research Goals) |
| **Technical** | 10 (Discriminator), 9 (Soft Cell Assignment — deeper version) |
| **About** | 12 (Related Work), 13 (Team) |

### Visual Assets Needed

| Asset | Description | Status |
|-------|-------------|--------|
| Shenzhen grid map | 48×90 grid over Shenzhen with district boundaries | Exists in GIS notebooks |
| Pipeline diagram | Three-step visual (Identify → Edit → Validate) | To create |
| Objective function diagram | Three-term visual with weights | To create |
| Soft cell assignment animation | Temperature annealing from soft → hard | To create |
| ST-SiameseNet architecture diagram | Siamese LSTM schematic | To create |
| Gini coefficient illustration | Visual explanation of the inequality measure | To create |

### Content Exclusions (Do Not Include on Website)

The following are explicitly excluded from the public website:

- **Numerical results** (metric values, convergence statistics, iteration counts)
- **Before/after modification comparisons** with specific numbers
- **Heatmaps showing actual service distribution** from the data
- **Discriminator accuracy or performance metrics**
- **Specific parameter tuning results** (which $\alpha$ values work best, etc.)
- **Per-trajectory modification details**
- **g(D) function fitting diagnostics or R² values**

These will be included in a future "Results" section once the research is further along.

### Math Rendering

The website should use **KaTeX** for math rendering (consistent with existing presentations). Key equations to render:

1. The combined objective: $\mathcal{L} = \alpha_1 \cdot F_{\text{spatial}} + \alpha_2 \cdot F_{\text{causal}} + \alpha_3 \cdot F_{\text{fidelity}}$
2. The Gini coefficient: $G(\mathbf{x}) = \frac{\sum_{i}\sum_{j} |x_i - x_j|}{2n^2 \bar{x}}$
3. Soft cell assignment: $\sigma_c(\mathbf{p};\, \tau) = \frac{\exp(-\|\mathbf{p}-\mathbf{c}\|^2 / 2\tau^2)}{\sum_{c'}\exp(-\|\mathbf{p}-\mathbf{c}'\|^2 / 2\tau^2)}$
4. ST-iFGSM perturbation: $\delta = \text{clip}(\alpha \cdot \text{sign}(\nabla_\mathbf{p} \mathcal{L}),\, -\epsilon,\, \epsilon)$
5. Temperature annealing: $\tau_t = \tau_{\max} \cdot (\tau_{\min} / \tau_{\max})^{t/T}$
