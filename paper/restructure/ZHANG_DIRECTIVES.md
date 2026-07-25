# ZHANG DIRECTIVES — request ledger (email + revised draft + meeting)

Every request from Dr. Zhang's restructuring email, tracked with a disposition. The
2026-07-24 meeting supersedes the email where they conflict (meeting evidence:
`meeting/analysis_A_decisions.md`, `meeting/analysis_B_actions.md`,
`meeting/analysis_C_claims.md`). Robert's binding rule: **never misrepresent the
algorithm to satisfy a directive** — she has said she is fine with unfulfilled
requests in that case.

Legend: ✅ FULFILL · 🟡 FULFILL-WITH-CAREFUL-WORDING · 🔵 NEEDS-DECISION (Robert) ·
🔴 CANNOT-AS-STATED (would misrepresent; alternative proposed) · ⚪ SUPERSEDED-BY-MEETING

| # | Request (email) | Disposition | Notes |
|---|---|---|---|
| E1 | Overall order: 1 Intro · 2 Overview · 3 Methodology · 4 Experiments · 5 Related Work · 6 Conclusion | ✅ | Related Work moves after Experiments. |
| E2 | §2 Overview = current §3.1 "directly"; only data, basic quantities, problem; NO fairness objective or editing detail; boldface defined terms per cGAIL | ✅ | Strip/retarget the two forward refs in current §3.1 (to §3.2 and §3.4). Meeting adds: itemized challenges likely live HERE (see A-report; confirm). |
| E3 | §3 opens with a FATE-overview paragraph: two stages (budgeted, constrained editing → edit-aware weighting), components mapped to the Introduction's challenges | ✅ | Two-stage framing is accurate. Challenge mapping must point wherever the challenge list ends up (intro vs overview). |
| E4 | §3.1 Collective Fairness Surrogate: design-requirements list (corpus-level, demand-aware, differentiable, attributable); why raw parity is wrong; define residual then F_demo; explain r² reading; why-useful list; then the COMPLETE editing objective (demo + spatial regularization + fidelity guardrail); derivations stay in appendix | ✅ | All four design requirements and all four why-useful properties are true of F_demo. Keep the associational caveat (protected register) — she did not mention it but it must survive. Content ≈ current §3.2 reorganized behind a requirements-first lead. |
| E5 | §3.2.1 Attribution under budget: surrogate global vs edits local; outcome-side attribution (per-unit deficit decomposition → trajectories ending in high-deficit units); resource-side attribution v_i = ∂L/∂S_i; ranking/selection under budget K; attribution is the allocation mechanism, not post-hoc explanation | 🟡 | All accurate. Careful item: K is not allocated by one ranked list — trim takes its deficit-attribution selection, lift fills the remainder with positive-score nominees (SZ: 2,455+7,545). Describe the split faithfully. |
| E6 | §3.2.2 Outcome-side vs resource-aware editing: definitions, trim mechanics, the limitation paragraph; lift with 7 named elements (value-of-resource map, candidate screening, bounded reachable region, prefix preservation, final-tail-only, fixed anchor + tapered displacement, moved states update supply map); closing key distinction | ✅ | Every element exists in the real algorithm. 🔵 sub-decision: how much of current §3.3 (empirical 2,455-pickup fact, structural reasons i–iii, demand endogeneity) stays in main text vs moves to appendix — protected-register items must survive somewhere. |
| E7 | §3.2.3 Budgeted validity- and fidelity-constrained editing: K vs ε; constraint list; identity model described as identity-level behavioral-fidelity guardrail, NOT full realism; 6-step edit pipeline | 🟡 | Constraint list and K-vs-ε are accurate; the guardrail framing is already our claim discipline. **Careful item R1**: her pipeline step "evaluate fidelity → accept, skip, or revert" reads as a fidelity threshold gate. In reality accept/skip/revert is driven by king-move VALIDITY; fidelity is a weighted objective term each iteration + an evaluation-time gate (Fidelity-A). Use the corrected 6-step pipeline in ALGORITHM_FACTS.md §Validity. |
| E8 | §3.3 Edit-aware weighting (her outline typo says 3.4): dilution problem, then the upweighting strategy | ✅ | Current §3.4 "Downstream recipe" + §4.3 evidence. Number it 3.3. |
| E9 | Experiments answer Q1–Q5 (fairness under limited budget; resource presence; fairness/budget/validity/fidelity trade-off; downstream preservation; robustness) | 🟡 | Q1, Q2, Q4, Q5 map directly onto existing subsections. Q3's "edit budget" axis has NO k-sweep in the results (see E10) — frame the trade-off with what exists: ε-bound + validity stats + Fidelity-B dose behavior + the α sweep; present k as the configured budget. |
| E10 | "add or emphasize an analysis across different edit budgets if possible" | ⚪ | DROPPED by the meeting (verified by exhaustive absence — "budget" spoken once, as positioning; Zhang spent the back half REMOVING scope: "submit whatever we have"). Disposition: no k-sweep, no text implying one (C D16); emphasize existing budget-adjacent evidence (k as configured budget ≈ a tenth of the corpus; trim-vs-trim+lift composition; upweighting dose-response; oracle ceiling) and add a k-sweep future-work sentence (T7). |
| E11 | Draft a solution-framework figure (cGAIL Fig. 2 / ST-iFGSM Fig. 3 style) | ✅ | Meeting refined: abstract SEQUENCE diagram, inputs → stages → outputs, style per `meeting/fig2_style_screenshot.png` (= ST-iFGSM Fig. 3). Replaces current 3-panel Figure 2. Spec: `figures/FIG2_FRAMEWORK_SPEC.md`. |
| E12 | Figure 1 = her disparity-aggregation teaser (the PNG) | ✅ | Meeting: COMMITTED unmodified; use the PNG directly for submission (\includegraphics); TikZ remake = least priority, only after ALL content is final ([56:27]). Spec: `figures/FIG1_TEASER_SPEC.md`. |
| E13 | Follow the logic of her revised abstract + introduction | 🟡 | Adopt substance near-verbatim; repair broken cites ([35?] and the anonymous-link [?]); enforce house style (era numbers, claim discipline). See ZHANG_DRAFT_DELTA.md for the exact list of stale-vs-authoritative pieces. |
| E14 | Her draft's intro claim: "We made our code and dataset available … via an anonymous link" | 🔵 | Anon repo + §1 URL swap is already on Robert's checklist; but "and dataset" must match what the repo will actually hold by Sunday (raw data only if 100% anonymous — M43/M44 decision). Either scope the sentence to code or confirm data inclusion. |
| E15 | "You can boldface each term you defined following the format in the cgail paper" | ✅ | Apply in §2 Overview. |
| E16 | Template completeness (from meeting): ACM Reference Format + permissions block visible; resolve Overleaf-vs-local differences; finalize compilation | ✅ | Our main.tex currently SUPPRESSES these (`\settopmatter{printacmref=false}` + `\setcopyright{none}`). Flip to template-default so the blocks render as in her PDF; keep `anonymous,review`. Verify locally + on Overleaf. |
| E17 | "I would like you to decide the final wording and presentation" | ✅ | Explicit latitude — meeting reinforced: section names from the email are not binding. |

## Standing constraints that no directive overrides
1. Era discipline: α* = (0.1, 0.8, 0.1); SZ 2,455 selected → 2,337 net trim + 7,545
   lift; SF 1,330 + 629; headline Δs +0.0226 SZ / +0.0316 SF. lint.sh guards these.
2. Protected register: associational caveat, demand-endogeneity bound, leveling-down
   analogy (analogy ONLY), accounting-tier labels, Fidelity-B disclosure, p=0.031
   discipline — each survives somewhere (main text or appendix).
3. Citations: any \cite/refs.bib change updates CITATION_PRIORITY_CHECKLIST.md the same
   session; nobody ticks Robert's boxes but Robert.
4. No product/tool names; no "54%" figure; SF "reproduces", never "beats".

## Open items for Robert (synced to MEETING_DIGEST §4 / TASK_BOARD Lane 0)
- Q1 workspace (archive dir / branch / both).
- Q2 challenge set (five 1:1 per ADJ-3 vs three-merged).
- Q3 Fig-2 layout (Option A hybrid vs Option B three-band).
- Q4 anonymous-link sentence scope (code-only vs code+data) — was E14.
- A0 drop source assets into `zhang/`; A1 clarify who clicks submit.
- (E10 budget sweep: RESOLVED — dropped by meeting. E6 §3.3 depth: resolved
  operationally by T4's design — compressed in-block, protected register intact,
  overflow to appendix; Robert adjusts in review.)
