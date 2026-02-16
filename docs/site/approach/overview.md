# What is FAMAIL?

FAMAIL (Fairness-Aware Multi-Agent Imitation Learning) takes a novel approach to the urban taxi fairness problem: rather than generating entirely new synthetic trajectories, we **edit existing expert driver trajectories** to improve fairness.

## Trajectory Editing vs. Generation

This editing approach is central to the project's philosophy:

| Approach | Description | FAMAIL? |
|----------|-------------|:-------:|
| **Trajectory Generation** | Create entirely new synthetic trajectories from scratch | ✗ |
| **Trajectory Editing** | Apply small, bounded adjustments to real expert trajectories | ✓ |

## Why Editing?

Editing is preferred for several reasons:

1. **Preserves expert knowledge.** Driving patterns, route choices, and temporal behaviors are retained from the original human drivers.

2. **Maximizes efficiency.** Not all trajectories contribute equally to global unfairness, so only the most impactful trajectories are modified.

3. **Bounded modifications.** Spatial constraints limit how far pickup locations can move, ensuring modifications remain realistic.

4. **Enables fidelity validation.** A neural network discriminator can verify that a modified trajectory still "looks like" it came from the same driver.

## The FAMAIL Pipeline

<div class="pipeline">
  <div class="pipeline-step">
    <div class="step-number">1</div>
    <h4>Identify Unfair Trajectories</h4>
    <p>Which trajectories contribute most to spatial inequality?</p>
  </div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step">
    <div class="step-number">2</div>
    <h4>Edit Pickup Locations</h4>
    <p>Move pickup locations toward underserved areas</p>
  </div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step">
    <div class="step-number">3</div>
    <h4>Validate Fidelity</h4>
    <p>Does the edited trajectory still look realistic?</p>
  </div>
</div>

## Long-Term Vision

The edited trajectories are ultimately intended to serve as training data for **imitation learning agents**. By training agents on fairness-improved demonstrations, we aim to produce autonomous taxi policies that naturally distribute service more equitably — embedding fairness directly into learned driving behavior.

```
Expert GPS Trajectories
        │
        ▼
  FAMAIL Trajectory Editing
  (Identify → Edit → Validate)
        │
        ▼
  Fairness-Improved Trajectories
        │
        ▼
  Train Imitation Learning Agents
  on Fairer Expert Demonstrations
```

---

<div style="display: flex; justify-content: space-between; margin-top: 2rem;">
<a href="../../problem/" class="md-button">← The Problem</a>
<a href="../goals/" class="md-button md-button--primary">Next: Research Goals →</a>
</div>
