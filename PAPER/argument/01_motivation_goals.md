# Motivation & goals

## Why mobility-service inequity matters

Taxi and ride-hail service is a public good with private allocation: who gets picked up, how quickly,
and where they are taken shapes access to jobs, healthcare, and the rest of the city. In real fleets
that allocation is not demographically neutral — pickup and drop-off intensity correlates with
neighborhood characteristics such as housing price, income, and migrant/hukou population structure.
When a neighborhood's residents systematically receive less service than their demand warrants, the
inequity is baked into the operational data the industry runs on.

## How imitation-learned demand models inherit the bias

Modern demand and dispatch models are frequently learned by **imitation** from historical trajectory
data — they are trained to reproduce where drivers actually went. A model that faithfully imitates
biased data reproduces the bias, and a model deployed to guide dispatch can *amplify* it: it sends
supply where supply already went, reinforcing the under-service of the neighborhoods that were
under-served to begin with. Any intervention that only touches the model, and leaves the data
untouched, is fighting the training signal.

## Why edit real data instead of generating synthetic data

The natural alternative — *generate* fairer synthetic trajectories with a learned generator (behavior
cloning, GANs) — has two failure modes that motivate a different approach:

- **Distributional collapse / loss of realism.** Adversarial and free-running generators can drift
  away from real human behavior (degenerate lengths, collapsed coverage), producing data that scores
  well on a fairness number precisely because it no longer looks like real trajectories. A fairness
  gain bought with unrealistic data is not usable.
- **Untargeted change.** Generation rewrites the whole distribution, including the vast majority of
  behavior that was already fair, making it hard to attribute any change to the fairness objective.

**Editing** avoids both. It starts from real trajectories (so human fidelity is preserved by
construction), and it changes only a **small, attribution-targeted slice** — the pickups where
demographic unfairness concentrates — leaving the rest of the real data intact. The edit is a tiny,
bounded relocation of pickup cells, small enough that a driver-identity discriminator still recognizes
the trajectory as the same driver.

## Positioning — a fairness-oriented data-augmentation method

FAMAIL is framed as a **data-augmentation** method, in two moves:

1. **Edit** a small unfair slice of the real demonstrations to make the data fairer along a
   demographic axis, while keeping it realistic.
2. **Upweight** the edited demonstrations during downstream policy training, so the fairness
   propagates into the trained model instead of being averaged away by the unedited majority.

The scientific content is the pairing: editing alone yields fairer, faithful *data*, but a vanilla
model trained on it does not inherit the fairness (a genuine null) — the upweighting step is what
carries the data-level fairness into the model, and it does so *edit-specifically* (random
oversampling and selecting already-fair trajectories do not reproduce it). See
[`00_overview.md`](00_overview.md) for the argument in one page and
[`04_evaluation.md`](04_evaluation.md) for how each claim is tested.

## Contributions

- **A demand-adjusted demographic-fairness metric with per-cell attribution.** `F_causal` measures
  how much of the demand-adjusted service residual is explained by demographics (1 = fairest), and
  its per-(cell, time) attribution localizes *where* the unfairness sits. (Formulas and caveats:
  [`03_fairness_theory.md`](03_fairness_theory.md).)
- **An attribution-guided trajectory editor.** A signed-gradient step relocates the highest-attribution
  pickup cells within a small ε-ball, improving the fairness objective while a frozen identity
  discriminator keeps the edit realistic.
- **A two-pillar training recipe.** Pillar 1: edited data is the fairest faithful source (beats
  generating). Pillar 2: upweighting the edited demonstrations in behavior cloning recovers the
  fairness that vanilla training averages away — verified edit-specific against random and
  select controls.
- **Two-city validation.** The primary results on Shenzhen reproduce, with no algorithm change, on an
  independent city (San Francisco), establishing external validity.

## Sources / provenance

Framing draws on the project's umbrella-claim outline (`docs/two_level_argument.md`, historical) and
the current PI-approved two-pillar / data-augmentation positioning (`PAPER/second-dataset/FINDINGS.md`
§8; `famail_temporal/baselines/MEETING_41_PREP.md`). All experimental claims referenced here are
quantified and sourced in [`05_results_shenzhen.md`](05_results_shenzhen.md) and
[`06_results_sf.md`](06_results_sf.md).
