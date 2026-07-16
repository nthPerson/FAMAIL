# Fairness-intervention baseline — design spec (2026-07-16)

**Goal.** Answer reviewer-sim objection #2 ("no fairness-method baseline") with a real
fairness intervention compared against FAMAIL **at the model level** — rollout external
metrics on Shenzhen — under the paper's existing conventions. Approved by Robert
2026-07-16 (brainstorm in-session; comparison axis pre-decided: model-level/rollout is
the only viable option, since reweighing/in-processing produce models, not edited data).

**Deadline frame.** Paper Jul 26; new results must land ~Jul 22–24. Reweighing arm is
the banker (near-zero engineering); penalty arm follows (~1 day engineering). GPU is
occupied by the C1 dose-extension until ~Jul 16 late afternoon; these suites queue after.

## 1. Comparison structure (decided)

- **Baseline arms train on the RAW corpus** (`famail_temporal` Shenzhen PRIMARY, cleaned,
  same corpus the raw-BC arm uses). FAMAIL's arm is the committed **edited + upweighted
  (w30)** result. Raw-BC (uniform weights) anchors the bottom.
- **Claim shape:** intervention *placement* — "moving the fairness intervention into the
  demonstrations (edit + upweight) vs applying it at training time (reweigh / penalty)."
- **Pre-committed honesty rule:** every axis is reported as measured, whichever arm wins.
  The paper's claim is the placement + fairness-vs-distortion trade-off, NOT dominance on
  every metric. (House norm: surface, never smooth.)

## 2. Arms

### 2a. `reweigh` — Kamiran–Calders-style instance reweighing [kamirancalders2012]
- Per-trajectory weight = inverse of the origin group's supply-demand service ratio
  (trajectories originating in under-served-group cells get upweighted), computed once
  from the raw corpus's before-edit external-metrics grouping (migrant axis, district
  extremes — the SAME grouping every external table uses).
- Weights normalized to mean 1 (effective dataset size matched to the uniform arm).
- Weight construction is a pure function of (corpus, grouping) — deterministic, no seed.

### 2b. `penalty` — fairness-penalty BC (in-processing, à la [zheng2023])
- BC loss + λ · **differentiable DP-gap analog**: the absolute gap between
  group-conditional predicted service mass under the policy's next-step distributions
  (advantaged vs disadvantaged cell groups, same migrant/district-extremes grouping).
- **Deliberately NOT F_causal** — the baseline must optimize an external-family quantity;
  penalizing our own optimization label would be circular against the metric firewall
  (§4.1) and reviewers would call it.
- λ ∈ {λ_lo, λ_mid, λ_hi} — three values bracketing "penalty visible but training stable,"
  calibrated by a short pilot at seed 0 before the full suite; mirrors the w10/w20/w30
  dose convention. Pilot criterion: λ_hi = largest λ where held-out next-step accuracy
  degrades < 20% relative; λ_lo ≈ λ_hi/10; λ_mid geometric mean.

## 3. Scoring (identical pipeline to existing arms)

- **n = 6 paired seeds** (0–5), same seeds as the WBC suites; paired against raw-BC.
- **Fairness axes (rollout/model level):** DP gap, DI, Theil, mean(Y|disadv.) computed on
  each arm's rolled-out allocation via the existing rollout + external-fairness tooling
  (the same path that produced the §4.4 allocation-boundary numbers).
- **Quality axes:** Fidelity-A of generated/rolled-out trajectories under the frozen
  identity discriminator (as in L1v2) + distributional JS vs raw (as in the variance
  suite). No new metric machinery.
- Statistics: n=6 conventions as everywhere (Wilcoxon floor p=.03125 = sign-unanimity
  certificate; t-intervals reported).

## 4. Implementation shape (decided: Option 1 + guardrail)

- **Extend `famail_temporal/baselines/run_weighted_bc_smoke.py`** (which already owns
  corpus loading, per-trajectory weights, 6-seed paired training, rollout, JS plumbing):
  - new weight mode `--fairness-reweigh` (arm 2a);
  - new loss flag `--fairness-penalty <lambda>` (arm 2b) in its BC training loop.
- **Regression gate (Mission-3 lesson, REQUIRED before any new result is trusted):**
  after the code changes, re-run the UNMODIFIED edited w30 arm (1 seed suffices if
  deterministic; else all 6) and diff against the committed result — byte/float-identical
  or the change is rejected. The frozen-editor gate is untouched (this work never touches
  the editor).
- New arms write to `famail_temporal/results/weighted_bc_sweep/fairness_baseline_*`;
  ledger rows `FB-REWEIGH`, `FB-PENALTY-PILOT`, `FB-PENALTY`; per-landing sequence
  applies (curate → inventory → gates → commit).
- Tests: TDD for the weight-rule function (group weights sum/normalization, degenerate
  groups) and for the penalty term (gradient flows; zero when groups equal; λ=0 recovers
  vanilla loss) BEFORE wiring into the training loop.

## 5. Paper placement

- §4.5 gains a short "fairness-intervention baselines" paragraph + rows (or a compact
  sub-table) — final placement decided with the 8-page cut plan (appendix relocation is
  available; the first 8 pages keep at least the headline comparison sentence).
- Citations introduced/leaned on: kamirancalders2012, zheng2023 — both re-verified
  against the ACM DL/authoritative source during implementation (Dr. Kash directive,
  Meeting 43), verification recorded in the citation audit trail.

## 6. Out of scope

- SF replication of these arms (Shenzhen-only for the deadline; SF is a stated candidate
  for camera-ready).
- Any editor/algorithm change; any change to existing arms' semantics.
- Training-side allocation constraints (the §4.4 boundary's real fix — future work).

## 7. Risks

- **Penalty arm instability** at high λ → the pilot bounds λ; if training collapses, the
  paper reports reweigh + the pilot's finding (an honest "in-processing is brittle here"
  note) rather than shipping garbage.
- **A baseline wins an axis** → reported as measured; the trade-off table (fairness axes
  vs Fid-A/JS) is the story, per §1's pre-commitment.
- **Schedule** — reweigh suite ~10h GPU, penalty pilot ~1h + suite ~10h; if the penalty
  suite would land past Jul 23, ship reweigh alone (decided fallback).
