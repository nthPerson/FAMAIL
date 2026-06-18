# The Two-Level Argument (FAMAIL paper framing)

This document is the brief top-level outline of how the FAMAIL paper argues its case. It frames two companion claims — **data quality** (Level 1) and **usability** (Level 2) — and points to the detailed results/methodology docs for each.

> Established at Meeting 38 (2026-06-11) with Dr. Zhang; supersedes the earlier "lead with the data-level Pareto" plan. The Pareto figure is now Level-1 supporting evidence; the model-level intervention argument is set aside.

---

## Umbrella claim

> By **editing the unfair portion of real data**, FAMAIL produces data that is *more fair* (and closer to human behavior) than what generative approaches (behavior cloning, GAN) produce — **and** that fairness is *useful*: a model trained on the edited data inherits the advantage.

The argument deliberately separates **what the data is** (Level 1) from **what the data is good for** (Level 2).

---

## Level 1 — Data quality (DONE)

**Question:** Is the edited data itself higher quality than data produced by generative baselines?

**Design:** one comparable table over **four data sources** — `raw` (real Shenzhen taxi trajectories), `edited` (FAM-AIL fairness-edited), `bc` (behavior-cloning / MLE-generated), `gan` (adversarially fine-tuned) — scored on three axes:

- **Causal fairness** `F_causal` (1 = fairest)
- **Spatial fairness** `F_spatial` (1 = fairest)
- **Fidelity** (realism vs raw), measured on **two complementary axes**:
  - **Fidelity-A (identity):** the frozen HuMID Siamese discriminator's same-driver probability — *does a trajectory still read as its driver?*
  - **Fidelity-B (distributional):** discriminator-free Jensen-Shannon divergence of trajectory statistics vs raw — *do the trajectory distributions match?*

At Level 1, BC and GAN are **data generators** whose emitted data is compared — not models trained on edited data (that is Level 2).

**Result (v2, 2026-06-18):** the identity-fidelity gate **passes** (driver-conditioned generation + an in-distribution HuMID construction fixed v1's failed gate), so both fidelity axes are trustworthy. **Edited data is the fairest source (F_causal 0.818, highest) while remaining identity-faithful (Fidelity-A 0.838 ≈ raw 0.840).** Its only distributional cost is the terminal-cell/pickup distribution — exactly the relocation editing performs by design — with trajectory shape preserved. The GAN reads as same-driver yet collapses distributionally; reporting *both* fidelity axes is what exposes that. Edited data is the only source strong on fairness **and** both fidelity axes.

**Details:** [`famail_temporal/baselines/LEVEL1_V2_RESULTS.md`](../famail_temporal/baselines/LEVEL1_V2_RESULTS.md) (numbers), [`LEVEL1_V2_METHODOLOGY.md`](../famail_temporal/baselines/LEVEL1_V2_METHODOLOGY.md) (architectures + construction). v1: [`LEVEL1_RESULTS.md`](../famail_temporal/baselines/LEVEL1_RESULTS.md).

---

## Level 2 — Usability (IN DESIGN)

**Question:** Does the edited data's quality advantage **survive downstream model training** — i.e., does fairness *transfer* from data into a model trained on it?

**Design (planned):** train a downstream behavior-cloning policy on each data source (the core contrast being **raw vs edited**, with **BC-generated and GAN-generated training data** as the comparison baselines that make the claim viable for the paper), then evaluate each *trained policy's* generated demand on the same Level-1 axes (`F_causal`, `F_spatial`, Fidelity-A/B). If the policy trained on edited data produces fairer demand than the policy trained on raw (or on generated) data while staying faithful, fairness has propagated from dataset to model — the usability payoff.

**Key statistical caveat (carried into the spec):** the data-fairness gap to propagate is ~0.013 in `F_causal`, near the seed-noise floor (~0.012 bits) measured in the GAN-baseline work. So Level-2 must be **multi-seed and paired**, reported as mean ± std with a paired test — powered to distinguish "BC preserves the gap" from "BC washes it out."

**Status:** brainstorming the spec (scope: raw, edited, BC-gen, GAN-gen as training sources). Spec/plan will live under `docs/superpowers/`.

---

## How the levels connect

Level 1 establishes that the edited *dataset* is fairer and faithful. Level 2 asks whether that property is *inherited* by a model trained on it. Together they support the umbrella claim: editing real data is a better route to fair, usable behavioral data than generating it. The previously-explored **model-level** intervention (editing *inside* the imitation-learning loop) is set aside — its effect was null at n=5 — and is reframed as the Level-2 usability question (train *on* fair data rather than edit *during* training).

---

## Pointers

| Artifact | Location |
|---|---|
| Level-1 v2 results | `famail_temporal/baselines/LEVEL1_V2_RESULTS.md` |
| Level-1 v2 methodology / architectures | `famail_temporal/baselines/LEVEL1_V2_METHODOLOGY.md` |
| Level-1 v1 results + training curves | `famail_temporal/baselines/LEVEL1_RESULTS.md`, `TRAINING_CURVES.md` |
| Level-1 orchestrators | `famail_temporal/baselines/run_level1_table.py`, `run_level1_table_v2.py` |
| Specs / plans | `docs/superpowers/specs/`, `docs/superpowers/plans/` |
