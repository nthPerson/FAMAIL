# Meeting 42 Prep — external fairness metrics + the leveling-down mechanism

**Date:** prep written 2026-07-07 · **Covers:** everything since Meeting 41 (2026-07-02).
**Purpose:** brief Dr. Zhang on the completed Meeting-41 P0 (**external fairness metrics, before→after
edit**) and the **definitive leveling-down mechanism finding**, and propose **reframing FAMAIL as an
over-service-reduction method** with a supply-side **uplift mechanism as Future Work**.

**Bottom line.** The editor improves fairness on **established metrics that are NOT in its objective**
(demographic parity, disparate impact, supply/demand ratio, Theil) — **unanimously and robustly on
Shenzhen**, weakly on SF. **But the gain is *leveling-down*, and we now know precisely why: it is
structural, not an optimizer quirk.** A demand-only editor over a frozen supply landscape can *only*
reduce over-service; there is **no non-perverse move that lifts the under-served group** (proven by a
flow analysis, a leverage argument, and an oracle bound; the downstream policy doesn't lift up either).
**Proposed framing:** present FAMAIL as a principled **over-service-reduction ("slack-trimming") fairness
editor** — turning the reviewer's leveling-down objection into a *demonstrated property of the problem* —
and add a **Future Work** section for the **supply-side uplift lever**. Full detail:
`PAPER/external-metrics/` (`FINDINGS.md`, `LEVELING_DOWN_MECHANISM.md`).

---

## 1. Results TL;DR (all curated in `PAPER/external-metrics/`)

**External metrics, before→after edit** (`FINDINGS.md`; metrics NOT in the objective):
- **Shenzhen: unanimous + significant + feature-set-robust.** Every axis × both groupings × Theil moves
  toward fairness, all Δ 95% CIs exclude 0. Headline (migrant, district-extremes, PRIMARY): **DI
  0.3325→0.3422, DP gap 14.20→13.60, Theil 0.155→0.149.** Across all 3 feature sets migrant DI Δ =
  +0.0097 / +0.0092 / +0.0086 (tight).
- **SF sf12: same direction, weaker** — compensation + Theil significant; **migrant NOT significant.**
- Two method notes: **housing-axis disparity direction is city-dependent** (Shenzhen low-housing is
  *over*-served, DI>1; SF *under*-served); **DP ≡ the supply/demand gap** by construction, so the honest
  distinct set is {DI, DP/gap, Theil} + the descriptive group levels.

**The leveling-down mechanism** (`LEVELING_DOWN_MECHANISM.md`): on the Shenzhen headline cell, the gap
closes **entirely by reducing the over-served (advantaged) group; the under-served group's level is
unchanged** (disadvantaged mean Y = 7.0734 before *and* after).

## 2. Why it levels down — structural (three verified causes + an oracle bound)

1. **The selection never sees the poor group.** All **2,455/2,455** edited pickups originated *and*
   landed in advantaged (low-migrant) cells — **zero** edits touched a disadvantaged cell. The
   α-attribution is residual-*variance*-based, and only over-served cells carry big residuals.
2. **The demand lever is ~inert on the poor side.** Adding demand to rich cells is **~32×** more
   Y-effective than removing it from poor cells, and **93%** of poor units sit at/below `DEMAND_FLOOR`
   (removal changes nothing).
3. **The real inequity is supply-side, and supply is frozen.** Median taxi presence: poor **1.8** vs rich
   **17.6** (~10×). The editor's only mutable quantity is the pickup location (demand); it has **no
   supply channel.**

**Oracle bound:** even a *perfect* demand-only editor could raise the poor group only by **deleting ~3k
of its recorded pickups** — perverse (it teaches downstream policies to serve poor areas *less*). So
leveling-down is the **constrained optimum**, not a failure of optimization.

**Downstream check — Option A rollout eval** (24 policies = raw/edited/w10/w30 × 6 seeds; §6 of the
mechanism doc): the trained policy **does not lift up either** — seeking-supply allocation to poor areas
is flat, and the **upweighted policies serve poor areas ~7–10% *less*** (0/6 seeds, p=.031) while rich
areas rise. **The published Pillar-2 rollout F_causal gain (+0.021/+0.031) is system-level over-service
trimming, not increased service to the under-served** — a load-bearing caveat to state honestly.

## 3. Reframed paper argument (concise)

**FAMAIL = a fair-data-augmentation method that reduces over-servedness.**
- **Pillar 1 — the edit.** Attribution-guided ST-iFGSM trims **over-served idle-slack** demand → the
  edited dataset is fairer on **established metrics it did not optimize** (unanimous + robust on
  Shenzhen) while staying realistic. On Shenzhen **no group's absolute recorded service falls** (pure
  slack-trimming).
- **Pillar 2 — downstream.** Upweighted BC propagates the fairness signal (rollout ΔF_causal
  +0.021/+0.031), described honestly as **system-level over-service reduction** (Option A).
- **Framing move (turn the objection into a contribution).** Engage the leveling-down / Parfit objection
  head-on, then answer with the **constrained-optimality result**: frozen supply + conserved demand +
  demand floor ⇒ **no non-perverse uplift direction exists for *any* demand-only editor** (quantified by
  the oracle bound). A reviewer attack becomes a proven property of the problem.
- **Future Work — the uplift / supply-side lever** (the door this opens):
  - **(B) Supply-aware editing / "seeking-tail rerouting"** — extend the editor to move the last few
    *seeking* states with the pickup via a differentiable ΔS channel; routes taxi presence into
    under-served cells → genuine lifting-up, combinable with the current trimming. Bonus: makes the
    fidelity discriminator load-bearing and cleanly differentiates FAMAIL from ST-iFGSM.
  - **(C) Supply augmentation** — add fidelity-screened synthetic seeking trajectories into under-served
    areas (cheaper; augmentation-native). Slogan: *"edit to trim over-service; reroute/augment to lift
    under-service."*

## 4. Asks / decisions for Dr. Zhang

1. **Approve the reframe** — present the current result as principled **over-service reduction** now, with
   the **supply-side uplift lever as Future Work**?
2. **Pillar-2 presentation** — state the rollout F_causal gain as **over-service trimming** (not uplift),
   per Option A?
3. **Where does the leveling-down / constrained-optimality result go** — main body as a **contribution**
   (problem property + motivation for the supply-side roadmap), or limitations?
4. **Scope for KDD** — keep B/C as Future Work, or attempt **(C) supply augmentation** before submission
   as a first uplift demonstration?

---

**Provenance:** results + numbers = `PAPER/external-metrics/FINDINGS.md`; mechanism + Option A =
`PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md` (scripts in `PAPER/external-metrics/scripts/`); tool +
method = `famail_temporal/baselines/EXTERNAL_FAIRNESS_RESULTS.md`, spec/plan under `docs/superpowers/`.
