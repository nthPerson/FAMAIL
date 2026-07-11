# Meeting 43 prep — progress since Meeting 42

**Prepared:** 2026-07-10. **Baseline for "progress since":** Meeting 42, held 2026-07-09 (Robert + Dr.
Zhang). Grounding: the Meeting-42 record was extracted from Notion — both the AI summary
([`MEETING_42_SUMMARY_EXTRACT.md`](MEETING_42_SUMMARY_EXTRACT.md), unverified) and the **full transcript**
([`MEETING_42_TRANSCRIPT_EXTRACT.md`](MEETING_42_TRANSCRIPT_EXTRACT.md), ground truth, read in full) —
because Notion summaries have previously fabricated/omitted content. All progress claims below cite
committed artifacts.

---

## 1. Meeting-42 recap (what was actually agreed)

**Action items voiced (T1–T6, transcript §1):** (T1) finish the GPU BC-propagation eval; (T2) implement
the other data-augmentation baselines; (T3) human-review all AI-assisted citations before use; (T4)
motivate the new attribution variant alongside the other objective terms; (T5) **start writing — the
methodology section first** (PI directive); (T6) **draft abstract to Dr. Zhang by next week**.

**PI decisions:** start paper-writing now (methodology first); **the trim-vs-trim+lift ablation is
necessary** ("this kind of design is really necessary"); abstract-as-placeholder is acceptable; general
endorsement of the trim+lift direction pending the BC/GAIL propagation result; timeline confidence.

**Corrections to the Notion record (transcript §6 — worth stating at Meeting 43):** the summary marked
T1 and T2 as done `[x]` when both were open at meeting time; it erased **Dr. Cash's credit** for the
"lower half of trajectories" insight that motivated the lift phase (provenance that should survive into
acknowledgments); "~80 model combinations" was actually spoken as "60 or 80"; the king-moves rule's
source paper (the cGAIL/"Seagale" preprocessing convention) was dropped. No fabrications this time.

---

## 2. Progress since Meeting 42, by action item

### T1 — BC-propagation eval: ✅ LANDED (same day, post-meeting)
The GPU run completed and the supply-lift workstream merged to `main` (2026-07-09): weighted-BC
propagation under trim+lift reaches **+0.0310 @ w30, 6/6 seeds** (Shenzhen), and **F_spatial now
propagates** too. Disclosure carried with it: the rollout-allocation drain is **attenuated ~40%, not
reversed**. Curated in `PAPER/supply-lift/`.

### T2 — data-augmentation baselines: ✅ BUILT (4 arms); results in hand for 1 of 4
- **3-arm perturbation suite (iFGSM / FGSM / random-jitter):** built + reviewed + merged 2026-07-09
  (same day as the meeting, hours after "still on my list"). GPU runs **pending** behind the α-sweep.
  Paper-facing caveats locked: the gradient arms are **"iFGSM/FGSM with random restart" (PGD-style), not
  vanilla ST-iFGSM** (the identity head is stationary at δ=0 — vanilla is a provable no-op, kept as an
  ablation row); FGSM numbers must come from the corrected engine (`6da3d27`+).
- **4th arm — Demographic Oversampling:** brainstormed → spec'd → planned → built → **run (9 CPU arms)**
  → adversarially reviewed → merged 2026-07-10. **Headline (matched budget k/dose = 10,000):** targeted
  mean ΔF_causal **+0.0153** (dose-monotone) vs. random-oversampling placebo **−0.0172** vs. FAMAIL
  trim+lift **+0.0222** — targeting is necessary (placebo *degrades* fairness) AND insufficient (below
  FAMAIL while fabricating **10.5%** of the corpus; FAMAIL: zero inflation). The placebo's DP gap
  explodes (+2.8) via fabricated supply landing in advantaged cells — the demand-endogeneity probe
  working as designed. Full record: `PAPER/baselines/demographic-oversampling/FINDINGS.md`.
- **Paper organization:** new **`PAPER/baselines/`** bundle (2026-07-10) — one subdirectory per baseline
  approach + `comparison/` reserved for the 6-row cross-arm table (lands with the GPU runs). Nothing
  else in `PAPER/` was baseline work that is cleanly separable (scope note in `PAPER/baselines/README.md`).

### T3 — human review of AI-assisted citations: ◐ PARTIALLY ADDRESSED (Robert to confirm)
The Mission-2 literature pass was already **citation-verified against primary sources** (2 fabrications
caught and removed — Zheng "67%"/"2.3%" and a Corbett-Davies misquote; audit trail in
`mission_2_citation_audit.md`), and the Mission-3 lit-scan (`DATA_AUG_BASELINE_CANDIDATES.md`) recorded a
per-entry verification URL. **Robert's own final human pass over the references remains his call** — T3
should stay open until he declares it done.

### T4 — motivate the new attribution variant: ◐ IN PROGRESS
The objective-function motivation bundle (`PAPER/objective-motivation/`) is merged. The **α-Pareto weight
sweep is running** to upgrade "Why these weights" from a documented criterion to an **empirical
(ΔF_spatial, ΔF_causal) frontier** — point 1 of 5 done as of 2026-07-10, ETA ~2026-07-11 morning; folding
into `MOTIVATION.md` follows (trim+lift is now canonical for all reporting, so the frontier is presented
as the shipped editor's α-frontier). Whether the *new attribution variant specifically* is motivated to
the standard Robert wants — his call to confirm against `PAPER/objective-motivation/`.

### T5 / T6 — methodology section + draft abstract: Robert's writing track
No repo-side artifacts yet; the running argument doc-set (`PAPER/argument/`, 8 docs) is positioned as the
assembly source. Status to report at Meeting 43 is Robert's.

---

## 3. New decision since Meeting 42 (Robert, 2026-07-09/10)

**Trim+lift results center ALL PAPER reporting; trim-only appears only in the trim-vs-trim+lift ablation**
(plus rare fringe mentions). This operationalizes Dr. Zhang's "the ablation is necessary" decision: the
ablation is *the* sanctioned home for trim-only numbers.

---

## 4. Suggested discussion items for Meeting 43

1. **Demographic-oversampling result as the naive-lifting-up contrast** — the targeted-vs-placebo-vs-FAMAIL
   triple directly services the leveling-down discussion from Meeting 42 §3a (the very caveat that
   motivated trim+lift). Proposed framing: ratio metrics can be gamed by fabrication; FAMAIL's gains come
   from redistributing observed behavior.
2. **Dataset fact for PI awareness:** the MigrantRatio and CompPerCapita disadvantaged-origin pools are
   the **same 4,907 trajectories** (their bottom-third district sets coincide), and the distinct
   disadvantaged pool (8,241) cannot supply a 10,000-duplicate budget without re-duplication — relevant
   to how independent the equity axes really are, and a disclosed limitation of naive oversampling.
3. **The "54%" figure needs grounding before it is ever written down.** The only quantified headline
   spoken at Meeting 42 was F_causal improving "a little bit over 54%"; the committed records use
   absolute deltas (SZ **+0.0222**, SF **+0.0328**). Reconcile what the 54% refers to (or retire it) so
   the paper never carries an unsourced relative number.
4. **α-Pareto frontier** (expected complete by Meeting 43): the empirical "why these weights" table +
   scatter — the design-necessity companion to the trim-vs-lift ablation.
5. **GPU queue confirmation:** α-sweep (running) → 3 perturbation arms (minutes each) → 6-row comparison
   table → `PAPER/baselines/comparison/`.
6. **Record hygiene:** the two Notion checkboxes wrongly marked done, and Dr. Cash's credit (transcript
   §6) — decide where his contribution gets acknowledged.

---

## 5. Numbers cheat-sheet (all committed, provenance in the linked bundles)

| Result | Value | Where |
|---|---|---|
| FAMAIL trim+lift headline (SZ / SF) | ΔF_causal **+0.0222** / **+0.0328** | `PAPER/supply-lift/` |
| Weighted-BC propagation under trim+lift | **+0.0310** @ w30, 6/6 seeds | `PAPER/supply-lift/` |
| Oversampling targeted d10k (mean, 3 seeds) | ΔF_causal **+0.0153**, dose-monotone | `PAPER/baselines/demographic-oversampling/` |
| Oversampling placebo d10k (mean, 3 seeds) | ΔF_causal **−0.0172**; ΔDP **+2.8** | same |
| Corpus inflation at d10k | **10.5%** (n_corpus 95,297) | same |
| Distinct disadvantaged-origin pool | **8,241** (< 10,000); migrant ≡ comp pool (4,907) | same, FINDINGS §4 |
| Attribution coverage (trim+lift) | ~2,400 → 7,500+ trajectories (~10% cap) | Meeting-42 transcript §3c |
| BC eval scale | "60 or 80" model combos, 2 datasets | Meeting-42 transcript §3d |
