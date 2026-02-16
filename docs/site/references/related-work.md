# Foundations and Related Work

FAMAIL builds on research from several fields. The following works provide the theoretical and methodological foundations for the project.

---

## Imitation Learning for Urban Mobility

The **cGAIL** (Conditional Generative Adversarial Imitation Learning) framework provides the foundation for modeling expert taxi driver behavior. cGAIL trains agents to replicate human driving patterns using GPS trajectory data, producing the expert trajectories that FAMAIL subsequently edits for fairness.

> Zhang, X., et al. "Conditional Generative Adversarial Imitation Learning for Taxi Driver Trajectories"

---

## Adversarial Trajectory Perturbation

The **ST-iFGSM** (Spatio-Temporal iterative Fast Gradient Sign Method) algorithm was originally developed for adversarial robustness testing of human mobility identification models. FAMAIL repurposes this technique: instead of attacking a model, we use gradient-based perturbation to *improve* a fairness objective.

> Zhang, X., et al. "ST-iFGSM: Enhancing Robustness of Human Mobility Signature Identification Model via Spatial-Temporal Iterative FGSM"

---

## Measuring Spatial Inequality in Transportation

The application of **Gini coefficients** to measure spatial inequality in taxi services — distinguishing between supply-side and demand-side inequality — provides the mathematical foundation for our spatial fairness term.

> Su, L., Yan, Z., & Cao, J. (2018). "Uncovering Spatial Inequality in Taxi Services in the Context of a Subsidy War among E-Hailing Apps"

---

## Causal Approaches to Fairness

The **causal reasoning framework** for algorithmic fairness — in particular, the idea that fairness requires ruling out the influence of sensitive attributes on outcomes — informs our causal fairness formulation.

> Kilbertus, N., et al. "Avoiding Discrimination through Causal Reasoning"

---
