---
hide:
  - toc
---

<div class="hero" markdown>

# Welcome to the Fairness-Aware Multi Agent Imitation Learning Project

<p class="tagline">FAMAIL</p>

<p class="summary">
FAMAIL is a research project at San Diego State University that addresses <strong>spatial inequality in urban taxi services</strong>. Using GPS trajectory data from Shenzhen, China, we develop trajectory editing techniques that modify expert driver trajectories to improve fairness metrics — ensuring that all areas of a city receive equitable taxi service — while maintaining the behavioral realism of the original trajectories.
</p>

<p class="affiliation">Dr. Xin Zhang (Advisor)<br>Robert Ashe (Researcher)<br>San Diego State University · Department of Computer Science</p>

</div>

## The Challenge

Taxi services in cities like Shenzhen exhibit significant spatial inequality. Certain areas — often wealthier, centrally-located neighborhoods — receive disproportionately more service relative to demand, while other areas — frequently lower-income or peripheral neighborhoods — are systematically underserved. When this inequality correlates with socioeconomic factors, it raises a systemic fairness concern in urban mobility.

[Learn more about the problem →](problem.md){ .md-button }

## Our Approach

Rather than generating entirely new synthetic trajectories, FAMAIL **edits existing expert driver trajectories** — applying small, bounded adjustments to real-world GPS data to improve the equity of taxi service distribution. This preserves the expert knowledge embedded in real driver behavior while steering the overall service pattern toward fairness.

<div class="pipeline">
  <div class="pipeline-step">
    <div class="step-number">1</div>
    <h4>Identify</h4>
    <p>Rank trajectories by their contribution to spatial inequality</p>
  </div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step">
    <div class="step-number">2</div>
    <h4>Edit</h4>
    <p>Modify pickup locations using gradient-based optimization</p>
  </div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step">
    <div class="step-number">3</div>
    <h4>Validate</h4>
    <p>Verify edited trajectories remain behaviorally realistic</p>
  </div>
</div>

## Explore the Project

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
<!-- <div class="card-icon">🎯</div> -->

### Research Goals

Four interrelated objectives — from multi-objective optimization to imitation learning.

[Research Goals →](approach/goals.md)
</div>

<div class="feature-card" markdown>
<!-- <div class="card-icon">⚙️</div> -->

### Methodology

A two-phase pipeline combining attribution-based trajectory selection with gradient-based editing.

[Two-Phase Pipeline →](methodology/pipeline.md)
</div>

<div class="feature-card" markdown>
<!-- <div class="card-icon">📐</div> -->

### Objective Function

Three terms — spatial fairness, causal fairness, and trajectory fidelity — in a single differentiable objective.

[Objective Function →](methodology/objective-function.md)
</div>

<div class="feature-card" markdown>
<!-- <div class="card-icon">🗺️</div> -->

### Study Area & Data

Real-world GPS taxi data from Shenzhen, China — 50 expert drivers, 4,320 grid cells, 288 daily time buckets.

[Study Area →](data/study-area.md)
</div>

<div class="feature-card" markdown>
<!-- <div class="card-icon">⚖️</div> -->

### Fairness

Two complementary lenses: spatial equality of service and causal independence from demographics.

[Fairness Definitions →](fairness.md)
</div>

<div class="feature-card" markdown>
<!-- <div class="card-icon">🧠</div> -->

### Discriminator

A Siamese LSTM network that validates whether edited trajectories remain authentic.

[ST-SiameseNet →](methodology/discriminator.md)
</div>

</div>

---

<p style="text-align: center; color: #8a8585; font-size: 0.88rem; margin-top: 2rem;">
<em>The FAMAIL Project — 2025–2026</em>
</p>
