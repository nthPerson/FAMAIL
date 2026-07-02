# Evaluation procedures & statistical conventions

This doc describes *how* every claim is tested and the statistical rules that govern how the numbers
are read. Per-city result values are **not** here — they live in
[`05_results_shenzhen.md`](05_results_shenzhen.md) and [`06_results_sf.md`](06_results_sf.md); the same
protocol was run identically on both cities.

---

## 1. The two-pillar experimental design

The argument is decomposed into what the data *is* (Pillar 1) and what the data is *good for* (Pillar 2),
plus a model-level control. All model-based arms use **paired seeds** (`set_all_seeds(s)` before each
arm) so shared seed noise cancels in the per-seed differences.

### Pillar 1 — L1 data quality (four sources, one table)

Four candidate data sources are scored on the four metrics:

| source | what it is |
|---|---|
| **raw** | the real trajectories |
| **edited** | the FAMAIL fairness-edited trajectories |
| **bc-generated** | trajectories emitted by a driver-conditioned behavior-cloning (MLE) model |
| **gan-generated** | trajectories emitted by a (WGAN-GP) adversarial model |

Each source is scored on **F_causal, F_spatial, Fidelity-A, Fidelity-B**. At L1, BC and GAN are
**data generators** whose *emitted data* is compared — not models trained on edited data (that is L2).
The claim: edited is the fairest source *while remaining faithful on both fidelity axes*; a generator
can post a high fairness number only by collapsing distributionally (caught by Fidelity-B).

### L2 — vanilla transfer

A driver-conditioned BC policy is trained on each source and its generated demand is re-scored on the
same metrics, reporting the **paired edited − raw ΔF_causal** across seeds. This tests whether the
data-level fairness *transfers* into a model trained on it under vanilla training. (Result: null — the
edit is a small slice the MLE loss averages over. This is the null Pillar 2 must overcome.)

### Pillar 2 — weighted-BC recovery (with two controls)

The edited demonstrations are **upweighted** in the BC loss, and the paired ΔF_causal vs raw is
measured across a **dose-response** of weights (w10 / w20 / w30). Two control arms isolate the
mechanism:

- **random placebo** — upweight a size-matched *random* non-edited subset. Tests whether the gain is
  mere oversampling.
- **most-fair select** — upweight the *already-fairest existing* trajectories. Tests whether the gain
  is reproducible by *selecting* fair data rather than *editing*.

The claim survives only if the **edited** arm recovers (monotone, all-seed-consistent) while **both
controls do not** — i.e. the gain is edit-specific.

### Model-level variance

A paired comparison of a baseline policy (b0, raw-corpus BC) against the FAMAIL policy (edited-corpus
BC), MLE-only, reporting ΔF_causal. It is the model-level companion to the L2 null.

---

## 2. The real-anchored Fidelity-A validation gate

Fidelity-A is only meaningful if the identity discriminator actually separates same-driver from
different-driver *in our input construction*. A **real-anchored gate** tests exactly that, independent
of any generator's quality:

- `high_matched` = mean same-driver probability for **real-d vs real-d** pairs (same real driver).
- `low_mismatched` = mean for **real-d vs real-d′** pairs (different real drivers).
- **Gate passes iff `high_matched − low_mismatched ≥ 0.2` and `high_matched > low_mismatched`.**

When the gate passes, Fidelity-A is reported as **trusted** for all four sources. The gate is anchored
on **real** data (not generated data) deliberately: it measures whether the *metric* is well-posed in
our regime, and does not conflate metric validity with generator quality (a collapsed GAN must not be
able to invalidate the metric for raw and edited). This real-anchored construction is what fixed an
earlier version whose out-of-distribution single-trajectory input gave no separation (~0.67 ≈ 0.67,
"untrusted"). Both cities pass the gate; the numbers are in 05/06.

---

## 3. Statistical conventions (load-bearing)

The evidence is small-n and paired, so the reporting rules matter as much as the point estimates:

- **Paired seeds.** Every model-based Δ is a per-seed paired difference (edited − raw, or arm − raw),
  which removes shared seed noise — essential because the data-level fairness gap sits near the
  cross-seed noise band.
- **The n = 6 Wilcoxon floor.** With 6 paired seeds, the two-sided signed-rank test's *smallest
  possible* p-value is **`p = 0.03125`**, reached exactly when all 6 seeds share a sign. It is a
  **sign-unanimity certificate, not an effect size**, and no test survives multiple-comparison
  correction at this n. The weighted-BC recovery is therefore read from the **mean Δ + t-CIs +
  monotone dose-response + the null/negative control arms**, not from an uncorrected p.
- **The n = 5 case.** L2 and the variance suite use 5 seeds, where a two-sided Wilcoxon **cannot** reach
  p < 0.05 (floor 0.0625). Those nulls are reported as **effect-vs-noise** — the mean Δ against the
  cross-seed band — not as significant negatives.
- **The deterministic L1 gap.** Raw and edited F_causal are static **data-level rescores** with
  **std = 0** across seeds, so the edited − raw L1 gap is a **point comparison with no sampling CI**.
  It is also the editor's own optimization target, so it is expected by construction; its value is
  that it holds while Fidelity-A is unchanged.

Together these mean the argument does not rest on any single p-value: it rests on **effect direction +
magnitude + CI separation + dose-response + the behavior of the control arms**, replicated across two
cities.

---

## Sources / provenance

- L1 four-source design + the real-anchored gate + Fidelity-B components:
  `famail_temporal/baselines/LEVEL1_V2_METHODOLOGY.md`.
- Two-pillar arm structure + weighted-BC controls + variance suite:
  `PAPER/by_feature_set/housing-comp-migrant/README.md`, `PAPER/second-dataset/FINDINGS.md` §6.
- Statistical conventions: `PAPER/by_feature_set/housing-comp-migrant/README.md` (conventions block),
  `PAPER/feature_selection/tables/comparison_across_sets.md`. Per-city numbers:
  [`05_results_shenzhen.md`](05_results_shenzhen.md), [`06_results_sf.md`](06_results_sf.md).
