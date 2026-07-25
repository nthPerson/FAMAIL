# Cut-Candidate Inventory — FAMAIL KDD manuscript (2026-07-18)

**Inventory only. No cuts applied, nothing committed, no source file modified.** This SUPERSEDES
`2026-07-15-cut-recon.md` in coverage (the paper has since grown: new teaser figure in §1, §4.5
fairness-method-baseline paragraphs, §4.7 two-tier/D1 SF material, n=12 flagship upgrades in both cities,
dose-saturation prose). Recon candidates that survive are carried forward and tagged `[recon Cx]`.

## Ground rules used

- **Target.** Paper renders ~12 pages incl. ~1 pg refs. Venue = **8 self-contained content pages + refs +
  UNLIMITED appendix reviewers need not read**. ~3 content pages (~330 single-column lines) must come out.
  Because the appendix is confirmed allowed, **RELOCATE is first-class** here (the recon had to hedge on it).
- **Savings unit.** Single-column rendered lines; two-column acmart ≈ **55 lines/column, ~110 lines/page**.
  - *Prose candidates:* I counted **source content lines** (excluding every `%` comment line — those cost
    zero page space per the constraints), then applied a **×1.2 wrap factor** (source is hard-wrapped ~72
    chars; an acmart 9.5pt column holds ~53) and **rounded down**. Deliberately conservative.
  - *Float candidates (figures/tables):* fractional-page footprints from the recon's measured build ×110,
    rounded down. The recon flags the local build over-counts vs Overleaf by ~0.5–0.75 pg; treat every
    number as ±0.5 pg build-uncertain and confirm on Overleaf before executing.
- Page map (build): §1 pp.1–2 (incl. `fig:teaser`), §2 pp.2–3, §3 pp.3–6, §4 pp.6–11, §5 p.11, refs pp.11–12.
- Floats present: 3 figures (`fig:teaser` §1; `fig:overview`/Fig-1 §3, full-width `figure*`;
  `fig:alpha-pareto`/Fig-2 §4.6) + 6 tables (`tab:external-sz`, `tab:channels`, `tab:ablation`, `tab:l1`,
  `tab:baselines`, `tab:featsets`).

---

## Candidates — §1 Introduction (`sections/01_introduction.tex`; has uncommitted author edits)

### `intro-tier-breakdown` `[recon C4]`
- **Location:** 01_introduction.tex 84–92 (the "+0.0176 (tier-1) … +0.0411 (tier-2)" clause).
- **Type:** SHORTEN — keep "raise F_causal by +0.0226 and … add statistically robust taxi presence to the
  under-served group"; drop the two tier numbers (both restated verbatim in §4.2 / `tab:channels`). Surviving:
  ~2 sentences.
- **Savings:** ~6 lines (counted ~4 source lines removed ×1.2; recon's 0.15 pg was optimistic).
- **What is lost:** intro no longer previews the tier-1/tier-2 supply split; +0.0226 headline stays.
- **Risk:** LOW (pure duplication of §4.2).
- **Dependencies:** none.

### `intro-contributions-compress`
- **Location:** 01_introduction.tex 116–136 (four-item `itemize`).
- **Type:** SHORTEN — tighten each bullet's second clause by ~1 line; keep all four contribution headers.
  Surviving: 4 bullets, ~1 line shorter each.
- **Savings:** ~6 lines.
- **What is lost:** some elaboration on each contribution (all restated in §3/§4).
- **Risk:** MEDIUM (the itemize is a scannable asset reviewers use to locate claims).
- **Dependencies:** none.

### `teaser-resize` / `teaser-remove`
- **Location:** 01_introduction.tex 19–48 (`fig:teaser`, single-column `figure[t]`, TikZ, 3 panels; **freshly
  author-added, uncommitted**).
- **Type:** FIGURE-OP (resize, MEDIUM) **or** REMOVE (HIGH).
- **Savings:** resize ~11 lines (trim panel height ~0.1 pg); full remove ~33 lines (~0.30 pg).
- **What is lost:** resize — legibility of the 3.0× service-ratio visual; remove — the paper's only "WHY"
  motivating figure and its 3.0× headline number (number also lives in §4/`tab:external-sz` context).
- **Risk:** resize MEDIUM; **remove HIGH** (author just added it as the motivating visual; SHORTEN variant =
  the resize, so the HIGH entry is admissible). Recommend resize over remove.
- **Dependencies:** caption cross-refs `fig:overview`; unaffected by resize.

---

## Candidates — §2 Related Work (`sections/02_related_work.tex`)

### `rw-contrast-tighten` `[recon C7]`
- **Location:** the closing FAMAIL-contrast sentence of each of the 5 themes (12–21, 27–35, 47–51, 62–67, 79–84).
- **Type:** SHORTEN — tighten each contrast sentence; **keep every `\cite`**. Surviving: 5 themed paragraphs.
- **Savings:** ~6 lines.
- **What is lost:** rhetorical polish on positioning; no citation dropped.
- **Risk:** MEDIUM (reviewer-sim obj. 10 already calls §2 thin — tighten prose only, never cut cites).
- **Dependencies:** none.

### `rw-recourse-compress`
- **Location:** 02_related_work.tex 53–67 ("Adversarial perturbation and recourse").
- **Type:** SHORTEN — the FGSM→ST-iFGSM→Gumbel/STE→recourse chain is re-explained in §3.5 "Shared machinery"
  (350–356); compress the mechanism detail here to a compact cite-carrying clause, keep the positioning.
  Surviving: ~3 sentences.
- **Savings:** ~4 lines.
- **What is lost:** nothing conceptual (mechanism detail survives in §3.5).
- **Risk:** LOW (duplication with §3.5).
- **Dependencies:** relies on §3.5 remaining intact for the mechanism; note if `meth-editor-impl-relocate` also fires.

### `rw-leveling-compress`
- **Location:** 02_related_work.tex 79–84 ("Both critiques are load-bearing…" preview sentence).
- **Type:** SHORTEN — this previews §3.3/§3.4 and §1; compress to one clause. Surviving: 1 sentence.
- **Savings:** ~3 lines.
- **What is lost:** a forward-pointer that §3.4 delivers in full.
- **Risk:** MEDIUM (sets up the structural-diagnosis contribution; keep the leveling-down + feedback cites).
- **Dependencies:** none.

---

## Candidates — §3 Methodology (`sections/03_methodology.tex`) — the appendix vein

### `meth-fcausal-derivation-relocate` `[absorbs recon C3 N×N identity]`
- **Location:** 03_methodology.tex 69–95 (the "Despite its appearance…" closed-form explanation, the constant-H /
  implicit-refit prose, the N×N normal-equations identity 80–84, and the Frisch–Waugh–Lovell paragraph 87–95).
- **Type:** RELOCATE to appendix. Keep in body: Eq.(1), a 2-sentence gloss ("Eq. 1 is the stage-two regression
  in closed form — numerator RSS, denominator TSS; because demographics are fixed, H is constant, so the
  measure re-fits the regression exactly at every gradient step, App. X"), and the boundary-case sentence
  (96–98, orients the 0–1 scale, cheap). Leave a one-line stub pointing to the appendix for FWL + the compact
  evaluation identity.
- **Savings:** ~24 lines (counted ~21 source content lines relocated ×1.2, rounded down).
- **What is lost from the 8 pages:** the FWL-exactness justification and the O(N) evaluation identity — a
  technical-soundness reviewer must open the appendix to see them.
- **Risk:** LOW (pure derivation, appendix-natural; the *interpretation* — partial R² = demand-adjusted
  demographic dependence — stays in the gloss).
- **Dependencies:** §3.3 attribution (79–80, 199–201) leans on "idempotence of its projections" — keep that
  half-sentence in the body gloss or the attribution exactness reads as unsupported.

### `meth-fspatial-gini-relocate`
- **Location:** 03_methodology.tex 125–136 (Eq.(2) Gini + F_spatial; the "never materialized … O(N log N)
  prefix-sum" remark 133–136).
- **Type:** two variants:
  - **(a) SHORTEN `[recon C3 Gini half]`** — drop only the O(N log N) / no-N²-materialization remark (133–136).
    Surviving: Eq.(2) + the "spatial smoothness regularizer" sentence. Savings ~3 lines. Risk **LOW**.
  - **(b) RELOCATE** — move all of Eq.(2) to appendix, keep a one-line "F_spatial = 1 − mean Gini of DSR/ASR
    (App. X)". Savings ~8 lines. Risk **MEDIUM** (F_spatial is a scored/optimized metric; self-containment
    prefers its definition stay).
- **What is lost:** (a) an efficiency aside; (b) the closed definition of a scored metric.
- **Dependencies:** none for (a).

### `meth-attribution-eq-relocate`
- **Location:** 03_methodology.tex 199–238 (Eq.(3) unit-attribution 202–208; Eq.(4) supply-gradient closed form
  231–237 + "against which the automatic gradient is verified").
- **Type:** RELOCATE closed forms to appendix:
  - **Eq.(4) + verification** (231–237): appendix-natural derivation. Savings ~8 lines. Risk **LOW**.
  - **Eq.(3)** (202–208): additional ~7 lines, but this is the "exact per-unit decomposition" that backs the
    *"two exact attribution mechanisms"* contribution; relocating it dims that claim's visibility. Risk **MEDIUM**;
    keep a one-line "an exact per-unit partition (App. X)" stub if relocated.
- **Savings:** ~8 (Eq.4 only) / ~15 (both).
- **What is lost:** the verifiable closed forms; the conceptual value-of-presence prose (211–230) stays.
- **Risk:** LOW (Eq.4) / MEDIUM (Eq.3).
- **Dependencies:** §3.4 (285) cites Eq.(unit-attr) by ref — cross-ref must retarget to the appendix.

### `meth-screen-detail-shorten`
- **Location:** 03_methodology.tex 240–260 (lift screen: per-offset tail translation, linearized-gain scoring,
  80k/95k, "nominates only").
- **Type:** SHORTEN mechanics (242–253) to ~2 sentences; keep the "screen nominates, editor derives the move"
  point and the 80k/95k eligibility.
- **Savings:** ~6 lines.
- **What is lost:** the exact scoring recipe of the supply screen (part of contribution 1's machinery).
- **Risk:** MEDIUM (a reviewer probing "how are candidates chosen?" would want it — but it is appendix-natural).
- **Dependencies:** none.

### `meth-editor-impl-relocate`
- **Location:** 03_methodology.tex 350–413, three sub-blocks: Eq.(5) step rule + Gumbel/STE bridge (359–372);
  lift taper constants w_j=0.25/0.5/0.75/1.0, 1/12 presence mass, s0=0.1 floor (384–392); king-move
  backward-reachability repair mechanics (399–410).
- **Type:** RELOCATE/SHORTEN protocol detail to appendix. **Keep in body:** the ε=2 identity-budget
  reinterpretation, "supply is endogenous" sentence, and the ~5% (118/2,455) infeasible-revert **disclosure**
  (that is load-bearing honesty, not plumbing). Move the constants and the repair algorithm.
- **Savings:** ~17 lines (bundle; counted ~14 source lines ×1.2, rounded down).
- **What is lost:** exact editor constants and the repair procedure — reproducibility detail, appendix-natural.
- **Risk:** LOW (implementation detail). **Do not** relocate the "Budget and phase order" two-phase-as-control
  paragraph (415–428) — it justifies the ablation and is DO-NOT-CUT.
- **Dependencies:** the ~5% revert number is also referenced in §4.2 provenance; keep both consistent.

### `meth-weight-dup-compress`
- **Location:** 03_methodology.tex 166–186 ("Scalarization" weight-selection prose) vs §4.6 462–471
  (weight-sensitivity results). The three-class criterion, "ΔF_causal flat within 0.001", and "lift-up declines
  monotonically" are stated in **both** places.
- **Type:** SHORTEN §3.2 to a forward-pointer ("α=(0.1,0.8,0.1) selected empirically by a three-class criterion;
  the sweep and criterion are reported in §4.6/Fig. 2"), letting §4.6 carry the detail. Surviving: ~2 sentences.
- **Savings:** ~8 lines.
- **What is lost:** the eager justification at definition-time; the full account survives in §4.6.
- **Risk:** LOW (duplication).
- **Dependencies:** **couples with `exp-figure2-relocate`** — if Fig. 2 relocates to the appendix, this
  forward-pointer must point to the appendix, and the two together must still leave the criterion in the body.

### `meth-figure1-resize` `[recon C8]`
- **Location:** 03_methodology.tex 317–343 (`fig:overview`/Figure 1, full-width `figure*`, TikZ).
- **Type:** FIGURE-OP — `figure*`→single-column, or width 0.85.
- **Savings:** ~15 lines (0.12–0.20 pg).
- **What is lost:** the method explainer renders smaller / narrower.
- **Risk:** MEDIUM (the overview figure carries the trim-vs-lift intuition reviewers value).
- **Dependencies:** none.

**DO-NOT-CUT in §3:** the leveling-down diagnosis §3.4 including its numbers (2,455/2,455 flow; 32× leverage;
93% at demand floor; median presence 1.8 vs 17.6; oracle perversity) — that is the *structural-diagnosis
contribution*, not protocol; the demand-endogeneity caveat (101–111); the two-phase-as-control paragraph (415–428).

---

## Candidates — §4 Experiments (`sections/04_experiments.tex`)

### `exp-setup-stats-shorten`
- **Location:** 04_experiments.tex 36–49 (protocol/statistics: Wilcoxon-floor arithmetic).
- **Type:** SHORTEN — keep "paired seeds; p-floor read as a sign-unanimity certificate; n=12 flagship survives
  correction; bootstrap intervals first-order"; relocate the exact floor values (0.03125/0.0625/.00049 derivation)
  to a footnote or appendix. Surviving: ~3 sentences.
- **Savings:** ~4 lines.
- **What is lost:** the explicit floor arithmetic (appendix-natural).
- **Risk:** LOW.
- **Dependencies:** the ".00049 survives Holm" claim is echoed in §5 bounds — keep both.

### `exp-setup-instruments-shorten`
- **Location:** 04_experiments.tex 63–73 ("External fairness instruments" — defines DP/DI/Theil/levels/groupings).
- **Type:** SHORTEN definitions to a compact clause; **keep the DP≡gap disclosure** (avoids double-counting —
  load-bearing). Surviving: ~2 sentences + the DP≡gap note.
- **Savings:** ~4 lines.
- **What is lost:** textbook definitions of standard instruments (appendix-natural).
- **Risk:** LOW.
- **Dependencies:** DP≡gap note also governs `tab:featsets`/`tab:external-sz` row choices — keep.

### `exp-tables-merge-A` `[recon C1]`
- **Location:** `tab:ablation` (178–195) + `tab:baselines` (425–443).
- **Type:** FIGURE-OP (merge) — both are arm × {ΔF_causal, ΔF_spatial} at matched budget; fold FAMAIL trim-only
  (ablation) as a row into the cross-arm table, keep SF ablation rows as a 2-row sub-panel or inline. One caption saved.
- **Savings:** ~18 lines (~0.16 pg).
- **What is lost:** nothing — same axes, one fewer caption; arguably *strengthens* (FAMAIL + ablation + all
  non-fairness arms in one glance).
- **Risk:** LOW.
- **Dependencies:** SF ablation rows (`tab:ablation` 190–192) don't share the inflation column — sub-panel them.

### `exp-tables-merge-B` `[recon C2]`
- **Location:** `tab:external-sz` (91–108) + `tab:channels` (128–144).
- **Type:** FIGURE-OP (merge) — shared `mean(Y|disadv)` +0.0529 row (it is the "Total" of the channel table and
  a row of the external table). Stack DP/DI/Theil/levels above the channel decomposition under one caption.
- **Savings:** ~13 lines (~0.12 pg).
- **What is lost:** nothing — one fewer caption; slightly denser.
- **Risk:** LOW.
- **Dependencies:** none.

### `exp-table6-relocate` `[recon C9; now full relocate]`
- **Location:** `tab:featsets` (512–536, the **largest table**, ~0.28 pg) + the alternate-set prose (494–510).
- **Type:** RELOCATE table + supporting prose to appendix. **Keep in body** (2–3 sentences): "the argument
  reproduces directionally on both alternate feature sets; the supply channel is tier-2-significant on all three
  (+0.0411/+0.0211/+0.0771); the most-fair-select control **leaks +0.0054/+0.0072** under the alternate sets —
  the edited arm still ≥3× larger." That leak sentence is a **disclosure and stays in the body.**
- **Savings:** ~35 lines (table ~31 + net prose trim after keeping the disclosure).
- **What is lost from the 8 pages:** the full 3-way robustness grid (per-metric deltas across feature sets).
- **Risk:** MEDIUM — reviewer-sim obj. 7 & 9 rest on the multi-set reproduction and the most-fair-select leak;
  safe **only because** the grid lands in the appendix and the leak + tier-2 sig numbers stay in body.
- **Dependencies:** body prose currently says "see Tab.~\ref{tab:featsets}"/"see §4.x" cross-refs inside the
  table — retarget to the appendix; keep the `most-fair-select` disclosure inline.

### `exp-figure2-relocate` `[recon C10; now full relocate]` — **largest single candidate**
- **Location:** `fig:alpha-pareto`/Figure 2 (474–492, ~0.40 pg) + §4.6 weight-sensitivity prose (462–471).
- **Type:** RELOCATE figure to appendix. **Keep in body** (2 sentences): "ΔF_causal is flat (within 0.001)
  across α_sp∈[0,0.55]; the supply-channel lift-up declines monotonically and is tier-1-significant only for
  α_sp≤0.2; the adopted (0.1,0.8,0.1) is the three-class criterion's best frontier point (Fig. App-X)."
- **Savings:** ~44 lines (figure float; +~6 more if §4.6 prose is trimmed to the 2 kept sentences).
- **What is lost from the 8 pages:** the visual weight-choice frontier — the α re-anchor defense's picture.
- **Risk:** MEDIUM — this figure *is* the weight-choice defense; safe only with the appendix holding it and the
  criterion staying in body.
- **Dependencies:** **couples with `meth-weight-dup-compress`** (§3.2 forward-pointer must aim at the appendix);
  §3.2 line 170 cross-refs the metric classes, unaffected.

### `exp-fourseource-gan-shorten`
- **Location:** 04_experiments.tex 246–253 (GAN Fidelity-B seed-bimodality digression in §4.3).
- **Type:** SHORTEN to 1 sentence ("the Shenzhen GAN's Fidelity-B is seed-bimodal; its worst seeds fail
  broadband — the known collapse mode — while the SF GAN stays healthy, §4.7"); relocate the per-seed spread.
- **Savings:** ~5 lines.
- **What is lost:** the per-seed 0.197–0.295 / 0.03–0.04 detail (secondary robustness).
- **Risk:** LOW–MEDIUM (a nice honesty beat; the pillar does not lean on it).
- **Dependencies:** none.

### `exp-dose-saturation-shorten`
- **Location:** 04_experiments.tex 271–281 (§4.4 w40/w50 extension + per-step increment sequence).
- **Type:** SHORTEN — keep "extending the dose shows saturation, not unbounded growth; w30 sits at the knee";
  relocate the exact w40/w50 values and the +0.0050/+0.0035/+0.0021/+0.0016 increment list.
- **Savings:** ~5 lines.
- **What is lost:** the numeric saturation curve (the "not a tuned endpoint" defense keeps its sentence).
- **Risk:** LOW–MEDIUM.
- **Dependencies:** the SF twin `exp-sf-downstream-shorten` should get the same treatment for parity.

### `exp-variance-shorten`
- **Location:** 04_experiments.tex 293–301 (§4.4 "Model-level variance", n=10 suite).
- **Type:** SHORTEN to ~2 sentences (keep "+0.0030±0.0022, n=10, p=.0039, an order of magnitude below the
  upweighted gains, so weighting carries the transfer"); relocate the n=5-vs-n=10 comparison aside.
- **Savings:** ~4 lines.
- **What is lost:** the "effect unchanged from the five-seed suite" reassurance (secondary).
- **Risk:** LOW–MEDIUM.
- **Dependencies:** SF variance twin (624–629) parallels this — treat together.

### `exp-provenance-shorten`
- **Location:** 04_experiments.tex 157–165 (§4.2 "Provenance disclosures").
- **Type:** SHORTEN — **keep the skip-on-infeasible ~5% disclosure**; relocate the oracle-ceiling arithmetic
  (+0.786, 2.6×, +0.882, realized +0.053) to the appendix, leaving "the realized lift sits far below the
  realism-free oracle ceiling (App. X)."
- **Savings:** ~4 lines.
- **What is lost:** the oracle headroom numbers (secondary; the disclosure survives).
- **Risk:** LOW–MEDIUM.
- **Dependencies:** oracle gate is also alluded to in §3.4 (commented-out) — no live cross-ref.

### `exp-baselines-perturbation-note-shorten` `[recon C5]`
- **Location:** 04_experiments.tex 335–344 (the δ=0 / concatenation-head naming note in §4.5).
- **Type:** SHORTEN paragraph → 2 sentences ("the gradient arms are iFGSM/FGSM with random restart; a
  pre-registered δ=0 no-op ablation did not stall because the deployed head compares by concatenation, so the
  random-start arms are the reported ones").
- **Savings:** ~8 lines.
- **What is lost:** the full stationarity-assumption honesty beat (baseline hygiene; not load-bearing).
- **Risk:** MEDIUM (a genuine honesty beat, but nothing in a pillar or headline needs it).
- **Dependencies:** none.

### `exp-fairness-penalty-shorten`
- **Location:** 04_experiments.tex 388–419 (§4.5 "Fairness-method baselines" — Kamiran–Calders reweigh +
  in-processing penalty λ-sweep).
- **Type:** SHORTEN — **keep** the two headline conclusions with numbers ("Kamiran–Calders reweigh moves
  fairness the *wrong* way, −0.0227, 6/6; the in-processing DP penalty is inert at every trainable dose and
  destructive only where it dominates; neither reproduces the editing recovery"). Relocate the full λ-grid
  (λ∈{1,3.16,10,100,1000}, signed vs absolute, the −0.2053/−0.1293 collapse detail, the 10⁻⁵-of-loss derivation)
  to the appendix. Surviving: ~4 sentences.
- **Savings:** ~9 lines.
- **What is lost:** the exhaustive λ-sweep evidence (the *conclusion* + reweigh number stay in body).
- **Risk:** MEDIUM — this is the reviewer-requested fairness-method comparison (reviewer-sim obj. 2); the
  "wrong-way / inert" verdict and its numbers must stay in the 8 pages; only the grid relocates.
- **Dependencies:** none.

### `exp-filtering-shorten`
- **Location:** 04_experiments.tex 539–546 (§4.6 "Filtering is not a substitute").
- **Type:** SHORTEN to 1 sentence ("removing the K least-fair trajectories *inverts* the gain — F_causal falls
  to 0.7935 at K=2,455 — whereas editing the same slice raises it to 0.8214"); or RELOCATE wholesale to appendix.
- **Savings:** ~5 lines.
- **What is lost:** the "why not just filter?" rebuttal detail (keep the one-sentence answer in body).
- **Risk:** MEDIUM (a reviewer defense; one surviving sentence mitigates).
- **Dependencies:** none.

### `exp-sf-downstream-shorten`
- **Location:** 04_experiments.tex 599–642 (§4.7 downstream reproduction: extended-dose saturation 612–618,
  variance 624–629, four-source reproduction 636–642).
- **Type:** SHORTEN — the SF downstream **restates the Shenzhen structure**; keep the n=12 flagship (+0.0333,
  12/12, p=.00049) and the "F_spatial does not propagate on SF" city-difference; compress the SF saturation,
  variance, and four-source paragraphs to one sentence each (they mirror §4.3/§4.4 and can point back).
- **Savings:** ~8 lines.
- **What is lost:** SF-specific saturation/variance/four-source detail (the reproduction claim + n=12 stay).
- **Risk:** LOW (reproduction detail; the SF conclusions and the D1 tier-2 material are untouched).
- **Dependencies:** must NOT touch the tier-2/Reading-B block (567–597) — that is fresh and DO-NOT-CUT.

**DO-NOT-CUT in §4:** the SF tier-1/tier-2 + D1 "Reading B" resolution (567–597, headline disclosure);
the allocation-boundary paragraphs both cities (303–319, 630–635, disclosure); the most-fair-select
sig-positive leak (449–460, disclosure); the SF caveats block (644–654, disclosure, may shorten only);
the n=12 flagship numbers both cities; the oversampling fabrication/placebo-degrades result (368–386);
`tab:external-sz` (Pillar 1 numbers) and `tab:l1` (four-source claim) content even if merged.

---

## Candidates — §5 Conclusion (`sections/05_conclusion.tex`)

### `concl-restatement-shorten` `[recon C6]`
- **Location:** 05_conclusion.tex 5–18 (the §1-restatement paragraph).
- **Type:** SHORTEN the restatement to ~4 sentences; **keep the bounds paragraph (20–37) intact.**
- **Savings:** ~6 lines.
- **What is lost:** restatement verbosity only.
- **Risk:** LOW.
- **Dependencies:** none.

### `concl-future-shorten`
- **Location:** 05_conclusion.tex 39–47 ("Two directions follow…").
- **Type:** SHORTEN to ~3 sentences.
- **Savings:** ~3 lines.
- **What is lost:** elaboration on the single-pass and transfer future directions.
- **Risk:** LOW.
- **Dependencies:** none.

**DO-NOT-CUT in §5:** the bounds paragraph (20–37) — associational caveat, demand-endogeneity, the
multiple-comparison honesty and the n=12 p=.00049 exception, city-specificity, identity-not-shape fidelity.
May shorten wording but not relocate; it is the paper's honesty ledger.

---

## Candidates — Abstract (`main.tex` 42–69)

### `abstract-tighten`
- **Location:** main.tex 42–69 (~215 words).
- **Type:** SHORTEN ~30 words.
- **Savings:** ~4 lines.
- **What is lost:** minor phrasing; every headline claim must survive.
- **Risk:** MEDIUM (shop window; low yield for the risk — deprioritize).
- **Dependencies:** none.

---

## Totals

**28 candidates** (some with sub-variants): RELOCATE 5 · SHORTEN 17 · REMOVE 1 (teaser, HIGH) · FIGURE-OP 5
(teaser-resize, Figure-1-resize, merge-A, merge-B, plus the two relocate-figures count as RELOCATE above).
Type tally by primary action: **RELOCATE 5, SHORTEN 17, FIGURE-OP 4 (2 merges + 2 resizes), REMOVE 1.**

| Risk class | Candidates | Summed savings (single-col lines) | ≈ pages |
|---|---|---:|---:|
| **LOW only** | fcausal-derivation-relocate, fspatial-gini(a), attribution-eq4, editor-impl, setup-stats, setup-instruments, merge-A, merge-B, fourseource-gan, dose-saturation, variance, provenance, sf-downstream, weight-dup, rw-recourse, intro-tier, concl-restatement, concl-future | **~144** | ~1.3 |
| **LOW + MEDIUM** | + table6-relocate, figure2-relocate, perturbation-note, fairness-penalty, filtering, screen-detail, attribution-eq3, fspatial-eq2(b), rw-contrast, rw-leveling, intro-contributions, teaser-resize, figure1-resize, abstract | **~308** | ~2.8 |
| add HIGH (teaser-remove, instead of resize) | +22 net | **~330** | ~3.0 |

**Five largest candidates:** `exp-figure2-relocate` (~44) · `exp-table6-relocate` (~35) · `teaser-remove`
(~33, HIGH) · `meth-fcausal-derivation-relocate` (~24) · `exp-tables-merge-A` (~18). *(Excluding the HIGH
teaser-remove, #5 is `meth-editor-impl-relocate` ~17.)*

---

## DO-NOT-CUT register (negative space the controller should see)

- **Pillar 1 (editing makes data fairer + faithful):** `tab:external-sz` content (DP/DI/Theil + levels, +0.0226
  F_causal, +0.0529 mean(Y|disadv)); `tab:l1` four-source "edited is fairest faithful source"; Fidelity-A≈raw.
- **Pillar 2 (upweighted BC carries fairness, edit-specifically):** the n=12 flagship recovery both cities
  (+0.0297/+0.0333 w30, 12/12, p=.00049); vanilla-BC null; both edit-specificity controls (random + most-fair).
- **Disclosures (main-text, load-bearing — shorten-only, never relocate):** allocation-boundary drain both
  cities; most-fair-select sig-positive leak; leveling-down-analogy scoping (§1 + §3.4); SF caveats block;
  §5 bounds paragraph; DP≡gap; tier-1-vs-tier-2 accounting + the D1/Reading-B SF resolution; skip-on-infeasible
  ~5%; oversampling fabrication + placebo-degrades.
- **The structural-diagnosis contribution:** §3.4 numbers (2,455/2,455; 32×; 93% at floor; presence 1.8 vs 17.6;
  oracle perversity) — this is a *result*, not protocol.
- **Two-phase-as-scientific-control paragraph** (§3.5 415–428) — it justifies the ablation's cleanness.

---

## Constraint feasibility note

No single required constraint is *impossible* — but **the 3-page target is NOT reachable within LOW-risk cuts
alone** (~1.3 pg). Reaching ~3 pages forces the two MEDIUM RELOCATEs (`exp-figure2-relocate`,
`exp-table6-relocate`) plus the §3 derivation relocations and table merges — i.e. **the 3-page cut is only
feasible *because the appendix is allowed*.** If the venue forbade the appendix, ~3 content pages could not be
recovered without shortening a disclosure or dropping a robustness defense below reviewer-probe safety; in that
counterfactual the honest ceiling of loss-tolerant cuts is ~1.3–1.5 pages (LOW) and the target would be
infeasible without a HIGH-risk move. All figures are ±0.5 pg build-uncertain (local over-count) — **confirm on
an Overleaf compile before executing**, since the font correction alone may absorb ~0.5–0.75 pg of the deficit.
