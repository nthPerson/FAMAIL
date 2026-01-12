# Trajectory Fidelity Term ($F_{\text{fidelity}}$) Development Plan

## Document Metadata

| Property | Value |
|----------|-------|
| **Term Name** | Trajectory Fidelity |
| **Symbol** | $F_{\text{fidelity}}$ |
| **Version** | 1.1.0 |
| **Last Updated** | 2026-01-12 |
| **Status** | Development Planning |
| **Author** | FAMAIL Research Team |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [Literature and References](#3-literature-and-references)
4. [Data Requirements](#4-data-requirements)
5. [Trajectory Modification Algorithm](#5-trajectory-modification-algorithm)
6. [Implementation Plan](#6-implementation-plan)
7. [Configuration Parameters](#7-configuration-parameters)
8. [Testing Strategy](#8-testing-strategy)
9. [Expected Challenges](#9-expected-challenges)
10. [Development Milestones](#10-development-milestones)
11. [Appendix](#11-appendix)

---

## 1. Overview

### 1.1 Purpose and Definition

The **Trajectory Fidelity Term** ($F_{\text{fidelity}}$) measures how authentic or realistic the edited trajectories remain compared to genuine expert driver trajectories. This term ensures that while improving fairness, the modified trajectories still resemble real taxi driver behavior.

**Core Principle**: Edited trajectories should be indistinguishable from real expert trajectories. High fidelity means that a classifier (discriminator) cannot easily distinguish edited trajectories from original ones.

**Key Role**: The fidelity term prevents the optimization from producing unrealistic trajectories that might achieve high fairness scores but would be impractical or impossible to execute in the real world.

### 1.2 Role in Objective Function

The fidelity term is a critical constraint in the FAMAIL objective function:

$$
\mathcal{L} = \alpha_1 F_{\text{causal}} + \alpha_2 F_{\text{spatial}} + \alpha_3 F_{\text{fidelity}}
$$

- **Weight**: $\alpha_3$ (typically 0.2-0.4 of total weight)
- **Optimization Direction**: **Maximize** (higher values = more authentic)
- **Value Range**: [0, 1]
  - $F_{\text{fidelity}} = 1$: Perfectly authentic (indistinguishable from real)
  - $F_{\text{fidelity}} = 0$: Completely artificial (easily detected as fake)

### 1.3 Relationship to Other Terms

| Related Term | Relationship |
|--------------|-------------|
| $F_{\text{spatial}}$, $F_{\text{causal}}$ | **Trade-off**: Improving fairness may require edits that reduce fidelity |
| Discriminator | **Implementation**: Fidelity is measured using the ST-SiameseNet discriminator |

### 1.4 Key Insights

**Why Fidelity Matters**:

1. **Realism**: Imitation learning requires realistic demonstrations
2. **Feasibility**: Trajectories must be physically possible
3. **Generalization**: Unrealistic trajectories may not generalize to real deployment
4. **Balance**: Prevents fairness optimization from "cheating" with impossible trajectories

**Example**:
- Original trajectory: Driver takes highway to downtown
- High-fidelity edit: Driver takes slightly different route via parallel street
- Low-fidelity edit: Driver teleports to downtown (unrealistic)

---

## 2. Mathematical Formulation

### 2.1 Core Formula

The fidelity term uses the **discriminator confidence score** as a measure of authenticity:

$$
F_{\text{fidelity}} = \frac{1}{|\mathcal{T}'|} \sum_{\tau' \in \mathcal{T}'} \text{Discriminator}(\tau')
$$

Where:
- $\mathcal{T}'$ = set of edited trajectories
- $\text{Discriminator}(\tau')$ = probability that $\tau'$ is classified as "real" (not edited)

### 2.2 Component Definitions

#### 2.2.1 Discriminator Model

The FAMAIL discriminator is based on **ST-SiameseNet** (Spatio-Temporal Siamese Network):

$$
\text{Discriminator}(\tau') = \sigma(f_{\theta}(\tau'))
$$

Where:
- $f_{\theta}$ = learned embedding/scoring function
- $\sigma$ = sigmoid activation (outputs probability in [0, 1])
- $\theta$ = model parameters (trained separately)

**Discriminator Architecture** (from FAMAIL discriminator module):
- Input: Trajectory sequence of state vectors
- Encoding: LSTM or Transformer-based temporal encoding
- Output: Single probability score ∈ [0, 1]

#### 2.2.2 Trajectory Representation

Each trajectory is represented as a sequence of states:

$$
\tau = [s_1, s_2, \ldots, s_T]
$$

Where each state $s_t$ is a 126-element vector (per `all_trajs.pkl` format):
- Spatial location: `x_grid`, `y_grid`
- Temporal context: `time_bucket`, `day_index`
- Environmental features: POI distances, pickup counts, traffic metrics
- Action: movement code

#### 2.2.3 Per-Trajectory Fidelity

For individual trajectory assessment:

$$
F_{\text{fidelity}}(\tau') = \text{Discriminator}(\tau')
$$

**Interpretation**:
- $F_{\text{fidelity}}(\tau') \approx 1$: Trajectory appears authentic
- $F_{\text{fidelity}}(\tau') \approx 0.5$: Discriminator uncertain
- $F_{\text{fidelity}}(\tau') \approx 0$: Trajectory appears artificial

### 2.3 Alternative Formulations

#### 2.3.1 Edit Distance-Based Fidelity

If no discriminator is available, use edit distance from original:

$$
F_{\text{fidelity}}^{\text{edit}} = 1 - \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \frac{\text{EditDist}(\tau, \tau')}{|\tau|}
$$

Where $\text{EditDist}(\tau, \tau')$ is the number of modified states.

#### 2.3.2 Constraint-Based Fidelity

Binary fidelity based on constraint satisfaction:

$$
F_{\text{fidelity}}^{\text{constraint}} = \mathbb{1}[\text{Discriminator}(\tau') \geq \theta]
$$

Where $\theta$ is a threshold (e.g., 0.5).

### 2.4 Derivation and Justification

**Why Discriminator-Based?**

1. **Learned representation**: Captures complex patterns in real trajectories
2. **End-to-end**: No need to hand-craft authenticity features
3. **Adversarial framework**: Natural fit with trajectory editing optimization
4. **Proven approach**: Used successfully in GANs and adversarial learning

**Connection to ST-iFGSM**:

The ST-iFGSM paper uses a similar discriminator to maintain trajectory authenticity during adversarial perturbation. FAMAIL adapts this approach for fairness optimization.

---

## 3. Literature and References

### 3.1 Primary Sources

#### 3.1.1 ST-iFGSM Paper

**"ST-iFGSM: Enhancing Robustness of Human Mobility Signature Identification Model via Spatial-Temporal Iterative FGSM"**

**Relevance**:
- Introduces discriminator for trajectory authenticity
- Provides framework for iterative trajectory editing
- Balances perturbation objectives with authenticity

**Location**: `FAMAIL/objective_function/docs/23-KDD-ST-iFGSM-Mingzhi-PDFA.pdf`

#### 3.1.2 FAMAIL Discriminator Module

**Location**: `FAMAIL/discriminator/model/`

Contains:
- ST-SiameseNet architecture
- Training scripts
- Pre-trained checkpoints

### 3.2 Theoretical Foundation

**Adversarial Learning**:
- Goodfellow et al. (2014): Generative Adversarial Networks
- Discriminator as authenticity measure

**Trajectory Similarity**:
- Siamese networks for sequence comparison
- Contrastive learning for trajectory embeddings

### 3.3 Related Work

| Reference | Relevance |
|-----------|-----------|
| Ho & Ermon (2016) | GAIL - Generative Adversarial Imitation Learning |
| Rao et al. (2020) | Trajectory similarity learning |
| Chen et al. (2021) | Mobility pattern recognition |

---

## 4. Data Requirements

### 4.1 Required Datasets

#### 4.1.1 Trajectory Data: `all_trajs.pkl`

**Location**: `FAMAIL/source_data/all_trajs.pkl`

**Structure**:
```python
{
    driver_key: [
        trajectory_0: [state_0, state_1, ...],  # List of 126-element states
        trajectory_1: [...],
        ...
    ],
    ...  # 50 drivers
}
```

**Usage**: Both original and edited trajectories in this format.

#### 4.1.2 Pre-Trained Discriminator

**Location**: `FAMAIL/discriminator/model/checkpoints/`

**Requirements**:
- Model weights (`.pt` or `.pth` file)
- Model configuration
- Normalization parameters

### 4.2 Data Preprocessing

#### 4.2.1 Trajectory Normalization

The discriminator expects normalized trajectories:

```python
def normalize_trajectory(
    trajectory: List[List[float]],
    normalization_params: Dict[str, Tuple[float, float]]
) -> np.ndarray:
    """
    Normalize trajectory features for discriminator input.
    
    Args:
        trajectory: List of 126-element state vectors
        normalization_params: Dict of (mean, std) for each feature
    
    Returns:
        Normalized trajectory array
    """
    traj_array = np.array(trajectory)
    normalized = np.zeros_like(traj_array)
    
    for idx, (mean, std) in normalization_params.items():
        if std > 0:
            normalized[:, idx] = (traj_array[:, idx] - mean) / std
        else:
            normalized[:, idx] = traj_array[:, idx] - mean
    
    return normalized
```

#### 4.2.2 Trajectory Padding/Truncation

For batch processing, trajectories may need uniform length:

```python
def prepare_trajectory_batch(
    trajectories: List[List[List[float]]],
    max_length: int = 500,
    pad_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare batch of trajectories with padding.
    
    Returns:
        (padded_trajectories, lengths)
    """
    batch = []
    lengths = []
    
    for traj in trajectories:
        traj_array = np.array(traj)
        
        if len(traj_array) > max_length:
            # Truncate
            traj_array = traj_array[:max_length]
        elif len(traj_array) < max_length:
            # Pad
            padding = np.full((max_length - len(traj_array), 126), pad_value)
            traj_array = np.vstack([traj_array, padding])
        
        batch.append(traj_array)
        lengths.append(min(len(traj), max_length))
    
    return np.array(batch), np.array(lengths)
```

### 4.3 Data Validation

```python
def validate_trajectory_for_discriminator(
    trajectory: List[List[float]],
    min_length: int = 2
) -> List[str]:
    """
    Validate trajectory format for discriminator.
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check minimum length
    if len(trajectory) < min_length:
        errors.append(f"Trajectory too short: {len(trajectory)} < {min_length}")
    
    # Check state vector dimensions
    for i, state in enumerate(trajectory):
        if len(state) != 126:
            errors.append(f"State {i} has wrong dimension: {len(state)} != 126")
    
    # Check for NaN/Inf values
    traj_array = np.array(trajectory)
    if np.any(np.isnan(traj_array)):
        errors.append("Trajectory contains NaN values")
    if np.any(np.isinf(traj_array)):
        errors.append("Trajectory contains Inf values")
    
    return errors
```

---

## 5. Trajectory Modification Algorithm

This section describes the trajectory modification algorithm that is central to the FAMAIL framework. The fidelity term plays a critical validation role in this algorithm.

### 5.1 Core Concept

The FAMAIL trajectory modification approach treats each taxi trajectory as a data sample contributing to overall fairness. The core insight is that:

1. **Each trajectory contributes to global fairness**: Some trajectories disproportionately contribute to unfair service distribution, analogous to how certain training examples contribute high loss in machine learning.
2. **Trajectories are interdependent**: Changing one route affects the fairness impact of other routes because it alters the active taxi count distribution ($N^p$) and service rates.
3. **Order matters**: The sequence in which trajectories are modified can influence outcomes (e.g., editing trajectory 1 then 2 vs. 2 then 1 may yield different results).

### 5.2 Editing vs. Generation

The algorithm focuses on **trajectory editing** (small adjustments to existing routes) rather than generating new trajectories from scratch:

- **Edits are locally bounded**: Changes keep the route within the original vicinity (within a defined $n \times n$ grid neighborhood)
- **Threshold-based scope**: A threshold determines how much a trajectory can be altered; issues beyond that threshold are noted as requiring new trajectory generation (future work)
- **Examples**:
  - ✅ **Valid edit**: Shifting a trip's start/end location to a neighboring underserved area (within ±n cells)
  - ❌ **Out of scope**: Teleporting to a distant underserved region (requires new trajectory generation)

### 5.3 Algorithm Steps

```
ALGORITHM: FAMAIL Fair Trajectory Modification
═══════════════════════════════════════════════════════════════════════

INPUT:
  - 𝒯: Original expert trajectory set
  - fairness_metrics: {spatial_fairness, causal_fairness}
  - discriminator: Trained ST-SiameseNet model
  - config:
    - neighborhood_size: n (for n×n grid neighborhood constraint)
    - max_iterations: int
    - convergence_threshold: float

OUTPUT:
  - 𝒯': Modified trajectory set with improved fairness

STEPS:

1. RANK TRAJECTORIES BY FAIRNESS IMPACT
   ─────────────────────────────────────────────────
   1.1 Calculate each trajectory's influence on global fairness
       using the spatial fairness metric (and causal fairness)
   1.2 Identify routes that most significantly reduce fairness
   
2. IDENTIFY WORST-OFFENDING TRAJECTORIES
   ─────────────────────────────────────────────────
   2.1 Apply "demand hierarchy" filter:
       - Consider areas with high ride demand first
   2.2 Within high-demand areas, select lowest-fairness trajectories
   2.3 These become initial targets for adjustment

3. MODIFY SELECTED TRAJECTORIES
   ─────────────────────────────────────────────────
   3.1 For each selected trajectory τ:
       3.1.1 Identify well-served (most fair) regions nearby
       3.1.2 Identify underserved (unfair) regions nearby
       3.1.3 Reallocate pickups/drop-offs from well-served
             to underserved regions within n×n neighborhood
       3.1.4 Keep overall route structure similar

4. VALIDATE TRAJECTORY FIDELITY
   ─────────────────────────────────────────────────
   4.1 For each edited trajectory τ':
       4.1.1 Run discriminator: score = Discriminator(τ')
       4.1.2 If score < threshold:
             - Reject edit OR reduce edit magnitude
       4.1.3 Ensure trajectory cannot be easily distinguished
             from real trajectories

5. RECOMPUTE FAIRNESS METRICS
   ─────────────────────────────────────────────────
   5.1 Update active taxi count distribution (N^p)
   5.2 Recompute spatial fairness: F_spatial(𝒯')
   5.3 Recompute causal fairness: F_causal(𝒯')
   5.4 Record improvement or regression

6. ITERATE IF NECESSARY
   ─────────────────────────────────────────────────
   6.1 Re-rank trajectories with updated metrics
   6.2 Make further adjustments as needed
   6.3 Repeat until:
       - Fairness metrics reach acceptable level, OR
       - No further significant gains can be made, OR
       - max_iterations reached

RETURN 𝒯'
```

### 5.4 Constraints and Efficiency Considerations

#### 5.4.1 Local Boundedness Constraint

All trajectory edits must stay within the original vicinity:

$$
\forall s' \in \tau', \exists s \in \tau : \|s' - s\|_\infty \leq n
$$

Where $n$ is the neighborhood size in grid cells. This ensures:
- **Physical feasibility**: Routes remain realistic
- **Minimal disruption**: Changes are subtle
- **Driver plausibility**: Edits represent natural route variations

#### 5.4.2 Computational Efficiency

To reduce computational load, the algorithm employs:

1. **Hold $N^p$ constant within iterations**: Assume active taxi counts don't change immediately during individual trajectory modifications
2. **Batch recalculation**: Only update $N^p$ after modifying a batch of worst-offending trajectories
3. **This saves time** while still maintaining accuracy between major iterations

### 5.5 Causal Fairness Integration

The algorithm incorporates causal fairness alongside spatial fairness:

| Fairness Type | Measures | Role in Algorithm |
|---------------|----------|-------------------|
| **Spatial Fairness** | **What** the service distribution is | Identifies geographic inequality |
| **Causal Fairness** | **Why** disparities exist (demand context) | Guides intelligent trajectory selection |

By combining demand estimates with fairness metrics, the algorithm can:
- Shift service from high-demand/well-served areas to high-demand/poorly-served areas
- Address root causes of unfairness, not just symptoms
- Prioritize modifications that have the greatest impact on demand-supply matching

### 5.6 Fidelity as Validation Gate

The fidelity term serves as a **validation gate** in the modification algorithm:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Trajectory Modification Loop                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Original τ ──► Propose Edit ──► τ' (candidate)                │
│                        │                                        │
│                        ▼                                        │
│              ┌─────────────────────┐                            │
│              │  Fidelity Check     │                            │
│              │  Discriminator(τ')  │                            │
│              └──────────┬──────────┘                            │
│                         │                                        │
│           ┌─────────────┴─────────────┐                         │
│           │                           │                         │
│           ▼                           ▼                         │
│     score ≥ θ                   score < θ                       │
│     ──────────                  ───────────                      │
│     Accept τ'                   Reject/Reduce                    │
│     Update 𝒯'                   Try smaller edit                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

This ensures that fairness improvements do not come at the cost of trajectory realism.

---

## 6. Implementation Plan

### 6.1 Algorithm Steps

```
ALGORITHM: Compute Trajectory Fidelity Term
═══════════════════════════════════════════════════════════════════════

INPUT:
  - edited_trajectories: Dict[driver_id → List[trajectory]]
  - discriminator: Trained ST-SiameseNet model
  - config: FidelityConfig
    - batch_size: 32
    - use_gpu: True
    - aggregation: "mean"

OUTPUT:
  - F_fidelity: float ∈ [0, 1] (higher = more authentic)

STEPS:

1. LOAD DISCRIMINATOR
   ─────────────────────────────────
   1.1 Load model weights from checkpoint
   1.2 Set model to evaluation mode
   1.3 Move to GPU if available

2. PREPARE TRAJECTORIES
   ─────────────────────────────────
   2.1 Flatten all trajectories into single list
   2.2 Normalize features using stored parameters
   2.3 Create batches of trajectories

3. COMPUTE DISCRIMINATOR SCORES
   ─────────────────────────────────
   3.1 For each batch:
       3.1.1 Prepare padded batch tensor
       3.1.2 Forward pass through discriminator
       3.1.3 Apply sigmoid to get probabilities
       3.1.4 Store scores

4. AGGREGATE
   ─────────────────────────────────
   4.1 If aggregation == "mean":
       F_fidelity = mean(all_scores)
   4.2 If aggregation == "min":
       F_fidelity = min(all_scores)
   4.3 If aggregation == "threshold":
       F_fidelity = proportion(scores > threshold)

5. RETURN F_fidelity
```

### 6.2 Pseudocode

```
function compute_fidelity(edited_trajectories, discriminator, config):
    # Step 1: Prepare data
    all_trajectories = flatten(edited_trajectories)
    normalized_trajs = normalize_all(all_trajectories, config.norm_params)
    
    # Step 2: Compute scores
    scores = []
    
    for batch in create_batches(normalized_trajs, config.batch_size):
        batch_tensor = prepare_batch_tensor(batch)
        
        with no_grad():
            logits = discriminator(batch_tensor)
            probs = sigmoid(logits)
        
        scores.extend(probs.tolist())
    
    # Step 3: Aggregate
    if config.aggregation == "mean":
        F_fidelity = mean(scores)
    elif config.aggregation == "min":
        F_fidelity = min(scores)
    elif config.aggregation == "geometric_mean":
        F_fidelity = geometric_mean(scores)
    
    return F_fidelity
```

### 6.3 Python Implementation Outline

```python
# File: objective_function/fidelity/term.py

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional, Literal
import numpy as np
import torch
import torch.nn as nn

from objective_function.base import ObjectiveFunctionTerm, TermMetadata, TermConfig


@dataclass
class FidelityConfig(TermConfig):
    """Configuration for fidelity term."""
    discriminator_checkpoint: str = "discriminator/model/checkpoints/best_model.pt"
    batch_size: int = 32
    max_trajectory_length: int = 500
    use_gpu: bool = True
    aggregation: str = "mean"  # "mean", "min", "geometric_mean", "threshold"
    threshold: float = 0.5  # For threshold aggregation


class FidelityTerm(ObjectiveFunctionTerm):
    """
    Trajectory Fidelity term based on discriminator confidence.
    
    Measures how authentic edited trajectories appear to a trained
    discriminator. Higher values indicate more realistic trajectories.
    """
    
    def __init__(self, config: FidelityConfig):
        super().__init__(config)
        self.discriminator = None
        self.device = None
        self.normalization_params = None
        self._load_discriminator()
    
    def _build_metadata(self) -> TermMetadata:
        return TermMetadata(
            name="fidelity",
            display_name="Trajectory Fidelity",
            version="1.0.0",
            description="Discriminator-based measure of trajectory authenticity",
            value_range=(0.0, 1.0),
            higher_is_better=True,
            is_differentiable=True,  # Through discriminator
            required_data=["discriminator_checkpoint"],
            optional_data=["normalization_params"],
            author="FAMAIL Team",
            last_updated="2026-01-09"
        )
    
    def _validate_config(self) -> None:
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.config.aggregation not in ["mean", "min", "geometric_mean", "threshold"]:
            raise ValueError(f"Invalid aggregation: {self.config.aggregation}")
    
    def _load_discriminator(self) -> None:
        """Load pre-trained discriminator model."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "discriminator/model"))
        
        from model import STSiameseNet  # Import from discriminator module
        
        # Determine device
        self.device = torch.device("cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu")
        
        # Load model
        checkpoint = torch.load(self.config.discriminator_checkpoint, map_location=self.device)
        
        self.discriminator = STSiameseNet(**checkpoint.get('model_config', {}))
        self.discriminator.load_state_dict(checkpoint['model_state_dict'])
        self.discriminator.to(self.device)
        self.discriminator.eval()
        
        # Load normalization parameters
        self.normalization_params = checkpoint.get('normalization_params', {})
    
    def compute(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> float:
        """Compute fidelity value for edited trajectories."""
        # Flatten trajectories
        all_trajs = []
        for driver_trajs in trajectories.values():
            all_trajs.extend(driver_trajs)
        
        if len(all_trajs) == 0:
            return 0.0
        
        # Compute scores
        scores = self._compute_discriminator_scores(all_trajs)
        
        # Aggregate
        return self._aggregate_scores(scores)
    
    def _compute_discriminator_scores(
        self,
        trajectories: List[List[List[float]]]
    ) -> List[float]:
        """Compute discriminator scores for all trajectories."""
        scores = []
        
        # Process in batches
        for i in range(0, len(trajectories), self.config.batch_size):
            batch = trajectories[i:i + self.config.batch_size]
            
            # Prepare batch
            batch_tensor, lengths = self._prepare_batch(batch)
            batch_tensor = batch_tensor.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                logits = self.discriminator(batch_tensor, lengths)
                probs = torch.sigmoid(logits)
            
            scores.extend(probs.cpu().numpy().tolist())
        
        return scores
    
    def _prepare_batch(
        self,
        trajectories: List[List[List[float]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch tensor with normalization and padding."""
        # Normalize and pad
        padded, lengths = prepare_trajectory_batch(
            trajectories,
            max_length=self.config.max_trajectory_length
        )
        
        # Apply normalization
        if self.normalization_params:
            padded = self._normalize(padded)
        
        return torch.FloatTensor(padded), torch.LongTensor(lengths)
    
    def _aggregate_scores(self, scores: List[float]) -> float:
        """Aggregate individual trajectory scores."""
        if len(scores) == 0:
            return 0.0
        
        scores_arr = np.array(scores)
        
        if self.config.aggregation == "mean":
            return float(np.mean(scores_arr))
        elif self.config.aggregation == "min":
            return float(np.min(scores_arr))
        elif self.config.aggregation == "geometric_mean":
            return float(np.exp(np.mean(np.log(scores_arr + 1e-10))))
        elif self.config.aggregation == "threshold":
            return float(np.mean(scores_arr >= self.config.threshold))
        else:
            return float(np.mean(scores_arr))
    
    def compute_gradient(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute gradient of fidelity with respect to trajectory features.
        
        Used for ST-iFGSM optimization.
        """
        # Enable gradients for this computation
        all_trajs = []
        for driver_trajs in trajectories.values():
            all_trajs.extend(driver_trajs)
        
        gradients = []
        
        for traj in all_trajs:
            traj_tensor = torch.FloatTensor(traj).unsqueeze(0).to(self.device)
            traj_tensor.requires_grad = True
            
            logit = self.discriminator(traj_tensor, torch.tensor([len(traj)]))
            prob = torch.sigmoid(logit)
            
            prob.backward()
            
            gradients.append(traj_tensor.grad.cpu().numpy())
        
        return np.array(gradients)
```

### 6.4 Computational Considerations

#### 5.4.1 Time Complexity

| Operation | Complexity |
|-----------|-----------|
| Trajectory normalization | $O(N \cdot T \cdot 126)$ |
| Batch preparation | $O(N \cdot T)$ |
| Discriminator forward pass | $O(N/B \cdot C_D)$ where $C_D$ = discriminator complexity |
| Aggregation | $O(N)$ |

Where $N$ = number of trajectories, $T$ = max trajectory length, $B$ = batch size.

#### 5.4.2 GPU Acceleration

- Discriminator forward passes are highly parallelizable
- Batch processing maximizes GPU utilization
- Typical speedup: 10-50x vs CPU

#### 5.4.3 Memory Considerations

- Batch size limits memory usage
- For 50 drivers × ~100 trajectories = ~5000 trajectories
- With batch_size=32 and max_length=500: ~10MB per batch

---

## 7. Configuration Parameters

### 7.1 Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `discriminator_checkpoint` | str | Path to trained discriminator weights |

### 7.2 Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 32 | Trajectories per batch |
| `max_trajectory_length` | int | 500 | Maximum states per trajectory |
| `use_gpu` | bool | True | Use GPU if available |
| `aggregation` | str | "mean" | Score aggregation method |
| `threshold` | float | 0.5 | Threshold for binary aggregation |
| `weight` | float | 1.0 | Weight in objective function ($\alpha_3$) |

### 7.3 Default Values and Rationale

```python
DEFAULT_CONFIG = FidelityConfig(
    # Model path
    discriminator_checkpoint="discriminator/model/checkpoints/best_model.pt",
    
    # Batch processing
    batch_size=32,               # Balance memory and speed
    max_trajectory_length=500,   # Cover most trajectories
    use_gpu=True,               # Much faster
    
    # Aggregation
    aggregation="mean",          # Smooth, differentiable
    threshold=0.5,              # Natural decision boundary
    weight=1.0,
)
```

**Rationale**:

- `aggregation="mean"`: Provides gradient signal from all trajectories
- `batch_size=32`: Fits comfortably in GPU memory
- `max_trajectory_length=500`: Covers 95%+ of trajectories without excessive padding

---

## 8. Testing Strategy

### 8.1 Unit Tests

#### 7.1.1 Discriminator Loading Tests

```python
class TestDiscriminatorLoading:
    """Test discriminator model loading."""
    
    def test_load_checkpoint(self):
        """Test loading from checkpoint file."""
        config = FidelityConfig(
            discriminator_checkpoint="path/to/test_checkpoint.pt"
        )
        term = FidelityTerm(config)
        
        assert term.discriminator is not None
        assert term.device is not None
    
    def test_missing_checkpoint(self):
        """Test error handling for missing checkpoint."""
        config = FidelityConfig(
            discriminator_checkpoint="nonexistent.pt"
        )
        
        with pytest.raises(FileNotFoundError):
            FidelityTerm(config)
```

#### 7.1.2 Score Computation Tests

```python
class TestFidelityComputation:
    """Test fidelity score computation."""
    
    @pytest.fixture
    def mock_discriminator(self):
        """Create mock discriminator for testing."""
        class MockDiscriminator(nn.Module):
            def forward(self, x, lengths):
                # Return high score for all
                return torch.ones(x.shape[0])
        
        return MockDiscriminator()
    
    def test_output_range(self, mock_discriminator):
        """Output always in [0, 1]."""
        # Create test trajectories
        trajectories = {
            "driver_0": [
                [[0.0] * 126 for _ in range(10)]
                for _ in range(5)
            ]
        }
        
        config = FidelityConfig()
        term = FidelityTerm(config)
        term.discriminator = mock_discriminator
        
        result = term.compute(trajectories, {})
        assert 0.0 <= result <= 1.0
    
    def test_aggregation_mean(self):
        """Test mean aggregation."""
        scores = [0.2, 0.4, 0.6, 0.8]
        
        config = FidelityConfig(aggregation="mean")
        term = FidelityTerm.__new__(FidelityTerm)
        term.config = config
        
        result = term._aggregate_scores(scores)
        assert abs(result - 0.5) < 0.01
    
    def test_aggregation_min(self):
        """Test min aggregation."""
        scores = [0.2, 0.4, 0.6, 0.8]
        
        config = FidelityConfig(aggregation="min")
        term = FidelityTerm.__new__(FidelityTerm)
        term.config = config
        
        result = term._aggregate_scores(scores)
        assert result == 0.2
```

### 8.2 Integration Tests

```python
class TestFidelityIntegration:
    """Integration tests with real discriminator."""
    
    def test_with_real_trajectories(self):
        """Test with actual trajectory data."""
        import pickle
        
        with open('source_data/all_trajs.pkl', 'rb') as f:
            all_trajs = pickle.load(f)
        
        # Select subset for testing
        test_trajs = {
            k: v[:5] for k, v in list(all_trajs.items())[:3]
        }
        
        config = FidelityConfig()
        term = FidelityTerm(config)
        
        result = term.compute(test_trajs, {})
        
        # Real trajectories should score high (if discriminator trained well)
        assert result > 0.5
    
    def test_original_vs_perturbed(self):
        """Original should score higher than perturbed."""
        import pickle
        
        with open('source_data/all_trajs.pkl', 'rb') as f:
            all_trajs = pickle.load(f)
        
        # Get sample trajectory
        driver = list(all_trajs.keys())[0]
        original = {"test": [all_trajs[driver][0]]}
        
        # Create perturbed version
        perturbed_traj = [
            [s + np.random.randn(126) * 0.5 for s in all_trajs[driver][0]]
        ]
        perturbed = {"test": perturbed_traj}
        
        config = FidelityConfig()
        term = FidelityTerm(config)
        
        original_score = term.compute(original, {})
        perturbed_score = term.compute(perturbed, {})
        
        # Original should score higher
        assert original_score > perturbed_score
```

### 8.3 Validation with Real Data

**Expected Behavior**:

1. **Original trajectories**: Score > 0.8 (highly authentic)
2. **Random trajectories**: Score < 0.3 (clearly fake)
3. **Subtle edits**: Score 0.5-0.8 (depends on edit magnitude)

---

## 9. Expected Challenges

### 9.1 Known Difficulties

#### 9.1.1 Discriminator Quality

**Challenge**: Fidelity is only as good as the discriminator.

**Impact**: Poor discriminator → misleading fidelity scores.

**Mitigation**:
- Validate discriminator performance separately
- Monitor discriminator accuracy during development
- Retrain if needed

#### 9.1.2 Gradient Stability

**Challenge**: Discriminator gradients may be unstable or vanishing.

**Impact**: Optimization may not improve fidelity effectively.

**Mitigation**:
- Use gradient clipping
- Consider spectral normalization
- Monitor gradient magnitudes

#### 9.1.3 Adversarial Overfitting

**Challenge**: Optimization may find adversarial examples that fool discriminator but are unrealistic.

**Impact**: High fidelity score but actually low-quality trajectories.

**Mitigation**:
- Multiple discriminators (ensemble)
- Regularization in objective
- Periodic discriminator retraining

### 9.2 Mitigation Strategies

| Challenge | Strategy | Implementation |
|-----------|----------|----------------|
| Discriminator quality | Validation metrics | Track accuracy, AUC separately |
| Gradient stability | Clipping | `torch.nn.utils.clip_grad_norm_` |
| Adversarial overfitting | Ensemble | Multiple discriminator models |
| Memory limits | Batching | Configurable batch size |

---

## 10. Development Milestones

### 10.1 Phase 1: Core Implementation (Week 1-2)

- [ ] **M1.1**: Set up directory structure
- [ ] **M1.2**: Implement `FidelityConfig` dataclass
- [ ] **M1.3**: Implement discriminator loading
- [ ] **M1.4**: Implement batch preparation utilities
- [ ] **M1.5**: Implement `compute()` method
- [ ] **M1.6**: Implement aggregation methods

**Deliverables**:
- Working `FidelityTerm` class
- Integration with existing discriminator

### 10.2 Phase 2: Testing and Validation (Week 2-3)

- [ ] **M2.1**: Unit tests for all components
- [ ] **M2.2**: Integration tests with real data
- [ ] **M2.3**: Validate discriminator-based fidelity
- [ ] **M2.4**: Implement `compute_gradient()` method
- [ ] **M2.5**: Performance benchmarking

**Deliverables**:
- Comprehensive test suite
- Gradient computation validated

### 10.3 Phase 3: Integration (Week 3-4)

- [ ] **M3.1**: Integrate with objective function
- [ ] **M3.2**: Test gradient flow in optimization
- [ ] **M3.3**: Document API and usage
- [ ] **M3.4**: Code review

**Deliverables**:
- Integration-ready module
- Complete documentation

---

## 11. Appendix

### 11.1 Code Snippets

#### 10.1.1 Fallback Edit-Distance Fidelity

```python
def compute_edit_distance_fidelity(
    original_trajectories: Dict[str, List[List[List[float]]]],
    edited_trajectories: Dict[str, List[List[List[float]]]]
) -> float:
    """
    Compute fidelity based on edit distance from original.
    
    Use when discriminator is unavailable.
    
    Returns:
        Fidelity score in [0, 1] (higher = closer to original)
    """
    total_edits = 0
    total_states = 0
    
    for driver_id in original_trajectories:
        orig_trajs = original_trajectories[driver_id]
        edit_trajs = edited_trajectories.get(driver_id, [])
        
        for orig, edit in zip(orig_trajs, edit_trajs):
            # Count changed states
            for orig_state, edit_state in zip(orig, edit):
                if not np.allclose(orig_state, edit_state, rtol=1e-5):
                    total_edits += 1
                total_states += 1
    
    if total_states == 0:
        return 1.0
    
    edit_ratio = total_edits / total_states
    return 1.0 - edit_ratio
```

### 11.2 Discriminator Interface

```python
# Expected discriminator interface
class DiscriminatorInterface(Protocol):
    """Protocol for discriminator models."""
    
    def forward(
        self,
        trajectories: torch.Tensor,  # [batch, seq_len, 126]
        lengths: torch.Tensor         # [batch]
    ) -> torch.Tensor:                # [batch] logits
        """Compute logits for trajectory authenticity."""
        ...
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load model weights."""
        ...
    
    def eval(self) -> None:
        """Set to evaluation mode."""
        ...
```

### 11.3 Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-09 | Initial development plan |
| 1.1.0 | 2026-01-12 | Added Trajectory Modification Algorithm section; Removed Quality Term references |

---

*This document serves as the comprehensive development guide for the Trajectory Fidelity term. All implementation should follow the specifications outlined here.*
