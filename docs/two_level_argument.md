# The Two-Level Argument (FAMAIL paper framing)

This document is the brief top-level outline of how the FAMAIL paper argues its case. It frames two companion claims — **data quality** (Level 1) and **usability** (Level 2) — and points to the detailed results/methodology docs for each.

> Established at Meeting 38 (2026-06-11) with Dr. Zhang; supersedes the earlier "lead with the data-level Pareto" plan. The Pareto figure is now Level-1 supporting evidence; the model-level intervention argument is set aside.

---

## Umbrella claim

> By **editing the unfair portion of real data**, FAMAIL produces data that is *more fair* (and closer to human behavior) than what generative approaches (behavior cloning, GAN) produce — **and** that fairness is *useful*: a model trained on the edited data inherits the advantage.

The argument deliberately separates **what the data is** (Level 1) from **what the data is good for** (Level 2).

> **Empirical status (2026-06-18):** Level 1 holds — the edited *data* is fairer and faithful. Level 2's transfer clause (the second half of the umbrella claim) is **not supported**: a vanilla driver-conditioned BC model trained on the edited data does **not** inherit the fairness advantage (paired `edited − raw` F_causal −0.0022 ± 0.0016, n=5). The data-quality claim stands; the model-inheritance claim does not, for this downstream model. See Level 2 below.

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

## Level 2 — Usability (DONE — negative result)

**Question:** Does the edited data's quality advantage **survive downstream model training** — i.e., does fairness *transfer* from data into a model trained on it?

**Design:** train a driver-conditioned behavior-cloning (BC) policy on each of four matched, full-corpus data sources — **raw**, **edited**, **BC-generated**, **GAN-generated** — across 5 paired seeds, then evaluate each *trained policy's* generated demand on the same Level-1 axes (`F_causal`, `F_spatial`, Fidelity-A/B) and report paired per-seed differences. Pairing (`set_all_seeds(s)` before each arm) removes shared seed noise, essential because the data-fairness gap (~0.013 `F_causal`) sits near the seed-noise floor (~0.012 bits).

**Result (2026-06-18):** the gate passes (Fidelity-A trusted), but **fairness does not transfer.** The +0.0128 data-level F_causal advantage of edited over raw is absent in the trained policies: every policy — whatever it trained on — lands near the *raw-data* fairness level (~0.806–0.814), nowhere near edited's data-level 0.818. The headline paired `edited − raw` difference is **−0.0022 ± 0.0016** (edited marginally *lower*, negative in all 5 seeds; Wilcoxon p=0.0625; 95% CI [−0.0042, −0.0003]). It is not a fidelity trade-off (edited-trained Fidelity-A 0.8408 ≈ raw; Fidelity-B 0.0120 ≈ raw). GAN-generated again posts the highest F_causal (0.8143) purely via distributional collapse (Fidelity-B 0.3507) — the Level-1 artifact propagating into the trained policy. **Vanilla behavior cloning imitates the aggregate demand distribution and averages away the targeted edits (3,773 / 105,401 trajectories); data-level fairness is not inherited by a BC model trained on the data.**

**Details:** [`famail_temporal/baselines/LEVEL2_RESULTS.md`](../famail_temporal/baselines/LEVEL2_RESULTS.md). Spec/plan under `docs/superpowers/`.

---

## How the levels connect

Level 1 establishes that the edited *dataset* is fairer and faithful. Level 2 asked whether that property is *inherited* by a model trained on it — and the answer, for a vanilla driver-conditioned BC policy, is **no**: the data-level fairness advantage does not transfer (paired `edited − raw` F_causal −0.0022 ± 0.0016, n=5). So the umbrella claim holds at the **data** level (L1: editing yields fairer, faithful data than generating it) but **not** at the model level under behavior cloning (L2): a BC model trained on the edited data does not inherit the fairness advantage. The previously-explored **model-level** intervention (editing *inside* the imitation-learning loop) remains set aside — its effect was null at n=5. The Level-2 negative is reported as-is; what training procedure *would* realize the data-level fairness in a model is left open.

---

## Pointers

| Artifact | Location |
|---|---|
| Level-2 results (fairness transfer) | `famail_temporal/baselines/LEVEL2_RESULTS.md` |
| Level-1 v2 results | `famail_temporal/baselines/LEVEL1_V2_RESULTS.md` |
| Level-1 v2 methodology / architectures | `famail_temporal/baselines/LEVEL1_V2_METHODOLOGY.md` |
| Level-1 v1 results + training curves | `famail_temporal/baselines/LEVEL1_RESULTS.md`, `TRAINING_CURVES.md` |
| Level-1 orchestrators | `famail_temporal/baselines/run_level1_table.py`, `run_level1_table_v2.py` |
| Specs / plans | `docs/superpowers/specs/`, `docs/superpowers/plans/` |
