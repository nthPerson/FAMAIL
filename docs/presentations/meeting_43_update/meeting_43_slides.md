# Meeting 43 — Slide Plan (paper overview + road to KDD)

> **This doc is the slide spine** (hand-off to PPTX generation). 11 slides: *what happened this
> week (2) → the paper's argument, results-first, for critique (8, incl. Figure 1 and the
> baseline definitions) → the prioritized path to submission (1).* Each slide gives a **title**, the **on-slide content**
> (terse bullets + the one load-bearing table), and a **"the point"** speaker note (not on the
> slide). Numbers are α\*-era, committed; deep provenance lives in
> [`MEETING_43_PREP.md`](../../../famail_temporal/baselines/meeting_prep/MEETING_43_PREP.md) §5
> and `PAPER/supply-lift/data/a10/`.
>
> **Assets for the deck builder:**
> - **Figure 1 (draft):** TikZ source `paper/figures/figure-1/figure-1.tex` (+ street-map PNG in
>   the same dir); it renders as Fig. 1 in the compiled `paper/main.pdf` — screenshot/crop that
>   page for slide 4.
> - **Weight-sensitivity frontier:** `paper/figures/extended_frontier.pdf` (two panels, print-safe).
> - All tables below are ready to paste as markdown.
> - House rules: grayscale-safe, one accent color, no tool/product names on slides.

---

## Slide 1 — FAMAIL: paper status one week out

**On the slide**
- Fairness-Aware Mobility-data Augmentation via Imitation Learning — KDD '27 submission.
- **Status: manuscript is prose-complete AND experiment-complete.** Every §4 result cell is
  filled from a fresh, ledger-verified run at the adopted configuration; **zero open run
  markers**; one open decision (SF framing — slide 10, needs this meeting).
- Deadlines: abstract **Jul 19** (draft to Dr. Zhang by ~Jul 17), full paper **Jul 26**.

**The point:** the campaign finished — today is about pressure-testing the argument, not
reporting gaps.

---

## Slide 2 — Since Meeting 42 (1/2): the weights are now a measured decision

**On the slide**
- Completed the α-sweep: 6 full trim+lift editing runs across the weight simplex, each scored
  on **three rings** of metrics (optimized / design-targeted / external).
- ΔF_causal is **flat** across the frontier — but the **lift-up declines monotonically with
  α_spatial** and is significant on both supply tiers only for α_sp ≤ 0.2. A two-axis view
  would have picked a config whose lift-up is *dead*; our own checks halted that promotion.
- **Adopted α\* = (0.1, 0.8, 0.1)** by a pre-stated three-ring criterion (max ΔF_causal s.t.
  ΔF_spatial ≥ 0 and lift-up significant on both tiers). Nearly doubles the lift-up vs the old
  config (tier-2 +0.0411 vs +0.0242).
- *(Figure: `extended_frontier.pdf` — panel B shows the monotone decline.)*

**The point:** the weight choice is now a criterion-driven selection a reviewer can check — and
the sensitivity analysis surfaced a methodological finding (optimized metrics alone can hide the
property that matters).

---

## Slide 3 — Since Meeting 42 (2/2): the full re-run bill, paid

**On the slide**
- **Every reported number now comes from a fresh run at α\*** — ~30 ledger-wrapped runs in 4
  days: data-level + externals + channels (both cities), ablations, **both** rollouts, four-source
  tables ×3 feature sets, weighted-BC sweeps ×4, variance ×4, baseline arms, per-set externals.
- Reproducibility discipline held throughout: nothing ran without a ledger row (command, commit,
  frozen-editor gate, env capture, checksums); an **era audit** caught two stale numbers in §4 and
  the lint now guards old-era values mechanically.
- Three surprises, all disclosed in the paper rather than smoothed (details on slides 8–9):
  the δ=0 "no-op" claim retired; random jitter raises F_causal by breaking trajectories; the
  select-the-fairest control is significant on the alternate feature sets.
- Writing: intro, related work, methodology, experiments, conclusion, abstract — all drafted,
  compiled, convention-linted, twice-audited.

**The point:** camera-ready hygiene is done *now*, not deferred — reviewers get one era, full
provenance, and our anomalies stated in our own words.

---

## Slide 4 — Figure 1 (draft) — design critique wanted

**On the slide**
- *(The current Figure 1 render, full-bleed: three panels on a real Shenzhen street-map
  background — the service gap → trim relocates over-served pickups (under-served side visibly
  untouched) → lift reroutes final seeking minutes into the supply-gradient field.)*
- Caption it carries in the paper: two bounded editing modes under one differentiable objective;
  trim moves demand, lift moves supply *with* the driver; edits confined to a two-cell ball.

**The point (ask the room):** does the three-panel argument read at a glance? Is the street-map
background helping or noise? Is the lift detour legible in grayscale? This is a draft — critique
freely; the spec has three alternative designs if this direction is wrong.

---

## Slide 5 — The argument in one slide

**On the slide**
1. **Demonstrations encode service inequity; imitation learns it.** (Motivation, §1)
2. **Edit, don't generate:** attribution finds where unfairness lives; ≤k real trajectories are
   perturbed within a two-cell ball under a frozen identity discriminator. (§3)
3. **Demand-only editing levels down — structurally.** Diagnosis dictated the second mode:
   **lift** makes the supply consequence of a reroute differentiable and endogenous. (§3.4–3.5)
4. **It works on metrics we never optimized**, in two cities, and the gain **survives training**
   via upweighted imitation — with controls pinning the recovery to the edit itself. (§4)
5. Contributions: the trim+lift editor · the leveling-down diagnosis · the transfer recipe with
   edit-specificity controls · two-city validation on never-optimized measures.

**The point:** one sentence per layer; every layer has a table on the next four slides. Ask: is
any link in this chain under-defended?

---

## Slide 6 — Results I: data-level fairness (the headline)

**On the slide**
- Editing improves the optimized metric (ΔF_causal **+0.0226** SZ, **+0.0316** SF) — but the
  claim rides the **external** instruments (never in the objective), migrant axis shown:

| Metric (SZ, before → after) | Δ [95% CI] |
|---|---|
| Disparate impact ↑ | **+0.0162** [+0.0136, +0.0189] |
| DP gap ↓ | **−0.890** [−0.992, −0.785] |
| Theil ↓ | **−0.0087** [−0.0097, −0.0076] |
| mean(Y \| disadvantaged) ↑ | **+0.0529** [+0.0086, +0.0989] |

- The gap closes **from both ends**: over-service falls (21.27→20.44) *and* the under-served
  level **rises** — decomposed, the rise rides the **supply channel**, significant under both
  accounting conventions (tier-1 +0.0176\*, distinct-taxi tier-2 recount **+0.0411\***).

**The point:** first defensible "lifting-up" result — added taxi presence in under-served areas,
significant under the honest (distinct-taxi) accounting, not just the optimizer's convention.

---

## Slide 7 — Results II: why lift exists — the ablation the PI asked for

**On the slide**
- Same weights, same budget, trim-only vs trim+lift (SZ):

| | trim-only | trim+lift |
|---|---|---|
| ΔF_causal | +0.0146 | **+0.0226** |
| ΔF_spatial | −0.0011 | **+0.0061** |
| Δ mean(Y \| disadv.) | **7.0734 → 7.0734 (flat to 4 decimals)** | **+0.053** (CI excl. 0) |

- Trim-only is *pure leveling-down* — measured, and shown structural in §3.4 (selection,
  leverage, frozen supply). Lift is the non-perverse remedy, not an incremental tweak.
- **Honest boundary, now like-for-like at α\*:** rolled-out policies still tilt pickup share
  away from disadvantaged areas — trim+lift **−0.0033** vs trim-only **−0.0049** at w30:
  **~33% attenuated, not reversed.** Disclosed; motivates training-side constraints as future work.

**The point:** the ablation Zhang called "really necessary" is complete and textbook — and the
boundary disclosure is now era-clean after the audit (both rollouts re-run at α\*).

---

## Slide 8 — Results III: the gain survives training — with controls

**On the slide**
- Vanilla behavior cloning averages the edit away (SZ: +0.0022, n.s.) — a null we verify.
- **Upweighting the edited slice recovers it, dose-monotone and sign-unanimous:**

| w | SZ | SF | HGC | 4FEAT |
|---|---|---|---|---|
| 10 | +0.0217 | +0.0242 | +0.0173 | +0.0180 |
| 30 | **+0.0302** | **+0.0332** | **+0.0248** | **+0.0256** |

  (every cell 6/6 seeds; F_spatial *also* propagates on SZ — sig at every weight — not on SF:
  a city contrast we state plainly)
- **Controls:** random-upweighting is null-to-negative *everywhere*; select-the-fairest is null
  on the primary set and **significantly positive on both alternate sets (+0.0054 / +0.0072 —
  a fifth to a quarter of the gain; the edited arm is ≥3× larger at every dose)**. We report the
  strongest control we found — edit-specificity under its hardest test.

**The point:** four independent 6-seed sweeps, one shape. The new control rows (added this week)
turn a potential reviewer attack into evidence.

---

## Slide 9 — The four baselines, defined (before their results)

**On the slide**
- All four arms: **matched budget, the same 9,882 trajectories the headline edit selected**,
  none optimizes fairness, all scored on FAMAIL's own rails. Question they answer: *objective,
  or perturbation/resampling per se?*

| baseline | one-line definition | what it tests |
|---|---|---|
| iFGSM (rand. restart) | iterative signed-gradient attack on the frozen identity discriminator, ε=2 | gradient-guided bounded perturbation *without* our objective |
| FGSM (rand. restart) | single-step variant | does iteration matter? |
| random jitter | seeded uniform noise in the same ε-ball | does *any* bounded perturbation help? |
| demog. oversampling (+ placebo) | duplicate disadvantaged-origin trajectories (phantom IDs, ±1-cell jitter), demand *and* supply rebuilt, dose-matched | can **fabrication** substitute for **redistribution**? |

- Lineage note: iFGSM/FGSM repurpose the ST-iFGSM attack (the KDD template paper) as an
  **editing-quality** baseline — per Meeting 41, a fidelity comparison, not a fairness competitor.
- "Random restart" is honest naming: the textbook δ=0 init was pre-registered as a no-op
  ablation, measurement showed the deployed concatenation head is *not* stationary there — the
  paper reports what actually ran.

**The point:** be ready for "why these baselines?" — each arm removes one ingredient of FAMAIL
(the objective, the iteration, the gradient, the realism constraint) so the comparison isolates
what the method actually contributes.

---

## Slide 10 — Results IV: baselines + editing quality + what we disclose

**On the slide**
- Cross-arm comparison at matched budgets (SZ; fairness ↑ / inflation):

| arm | ΔF_causal | note |
|---|---|---|
| **FAMAIL trim+lift** | **+0.0226** | 0% inflation, king-compliant by construction |
| demog. oversampling (targeted) | +0.0153 | fabricates **10.5%** of the corpus |
| \: placebo (untargeted) | −0.0172 | fabrication without targeting *degrades* fairness |
| random jitter | +0.0135 | **98.8% adjacency violations**, divergence 0.447 vs 0.187 |
| iFGSM / FGSM (rand. restart) | −0.0057 / +0.0017 | fidelity baselines, as framed |

- Four-source data quality (both cities): the edited corpus is the **fairest source that stays
  faithful** (SZ 0.8214, SF 0.9067; identity fidelity ≈ raw).
- Disclosed anomalies: the SZ GAN's distributional score is **seed-bimodal** (3/5 seeds
  collapse — pattern seed-identical across all three feature sets); random jitter's gain comes
  from breaking trajectories; the δ=0 "provable no-op" claim was retired after measurement
  (concatenation head ≠ difference head) and §4.5 tells the measured story.

**The point:** nothing beats the edit without paying in realism or fabrication — and our
surprises are in the paper before a reviewer can find them.

---

## Slide 11 — Road to KDD: prioritized, dated, and one decision needed *today*

**On the slide**
- **⚠️ DECISION NEEDED THIS MEETING — SF fairness framing** (the paper's only open marker):
  supply channel positive-significant (+0.0209\*) but total mean(Y|D) net-negative (−0.0324\*)
  because lift also routes *pickups* into under-served cells. Both readings are drafted
  side-by-side; **Dr. Zhang picks which leads** (ratio reading vs external-metrics reading —
  emphasis, not omission).
- **P0 — submission-critical:**
  1. Abstract to Dr. Zhang (**by Jul 17**) → KDD abstract **Jul 19**.
  2. **Reduce to 8 pages** main content (current build ~10 pp in review mode) — biggest writing task.
  3. Final adversarial audit (number/convention + fidelity reviewers) *after* the length pass.
- **P1 — quality + pledge:**
  4. `REPRODUCIBILITY.md` capstone: every claim → curated artifact → ledger row → exact command
     (inputs complete: 38-file curated bundle + run ledger + data inventory).
  5. **Artifact pledge readiness:** repo presentable + anonymized for double-blind; goal is to
     pledge artifacts-on-publication in the submission.
  6. Figure 1 final (today's feedback); fold in the s10 replication (running; report both if it
     differs).
- **P2:** citation human-pass (Robert), read-aloud pass, Overleaf port, Dr. Cash acknowledgment
  (camera-ready), retire the ungrounded "54%" (already lint-banned).

**The point:** the experimental risk is retired; what remains is length, polish, and packaging —
and the one framing call only the PI can make.
