# FINAL WHOLE-BRANCH REVIEW — findings (2026-07-24 late, base 39dd01a → ea6dd25 + T10 in flight)

Verdict: **FIX-FIRST** — blockers #1–#4, plus #5 (T10 diff needs its own claim-safety
read after commit). #6–#14 are minors. Everything else verified clean: every numeral
era-checked, all 17 E-items + 10 digest decisions implemented, D15/D16 fully
discharged, protected register intact, no hunk-boundary regressions, zero ??/dangling
refs, lint silent.

Status legend: ⚖ = needs Robert's ruling · 🔧 = mechanical fix · ✅ = applied.
APPLIED at c921518 (2026-07-24 late): #2 #3 #4 #6 #7 #9 #14 ✅ (incl. framework.tex
op-box + regenerated fig2-for-zhang.png/preview). #5: T10 committed at d874d8d;
reviewer spot-checked its in-flight diff CLEAN; one focused claim-read of the
committed diff remains recommended. STILL OPEN: #1 ⚖ (Fig-1 PNG ruling +
\Description repair after it), #8 ⚖ (her ¶3 'global' wording — PI text, Robert's
call), #10 #11 #13 ⚖ optional, #12 note-for-PI.

## Blockers

1. **[Critical] ⚖ ROBERT — Figure 1 PNG (Zhang's own figure) depicts the perverse
   demand move.** Pixel-verified: right panel's orange passenger sits at
   panel-relative x=659 vs boundary x=696 — i.e. ~37px on the ADVANTAGED side —
   while the disadvantaged half drops from 4 passengers to 3. Read as an exchange,
   the figure shows a pickup relocated OUT of the disadvantaged district, which
   §3.3 explicitly rules out ("not one landed in a disadvantaged cell"; the
   "perverse" deletion option). The taxi half is accurate (lift adds presence).
   Likely sloppy icon placement in her Keynote, not intent (her label says
   "Disadvantaged: Service: Increased"). OPTIONS: (a) nudge the orange passenger
   ~80–100px right so it sits clearly in the disadvantaged half beside the orange
   taxi (one-icon PIL edit; reads as one clean lift edit); (b) leave as-is (it is
   the PI's figure; ambiguity is 2.7% of panel width); (c) raise with Dr. Zhang in
   the hand-off email. E12 said "unmodified", but the prime directive authorizes
   raising it. REGARDLESS of a/b/c: 🔧 the `\Description` in 01_introduction.tex is
   factually wrong (dashed line runs boundary→black passenger ~130px above the
   orange pair; no passenger crossing is described) — repair it to describe what
   the PNG actually shows once (a)/(b) is decided.
2. **[Critical] 🔧 Fig-2 caption (03_methodology.tex:46) + op-box
   (framework.tex:152) assert a per-trajectory fairness-contribution score**
   (do-not-claim item 3 / D3 — the exact artifact D3 said to fix in the
   replacement). Fix both to C-6's frame: "Attribution scores how much a bounded,
   admissible edit to each trajectory would improve the corpus's collective
   fairness, and selects the k ≪ |T| best until the budget is spent." (op-box
   short form: "score the fairness improvement a bounded edit to each trajectory
   would buy").
3. **[Important] 🔧 C2's forward pointer (02_overview.tex:103) goes to
   §3.1 (sec:objective) but C2's answering block is §3.4 (sec:phys-validity)** —
   breaks the one binding invariant at the navigation layer. Fix:
   `(\S\ref{sec:phys-validity})` or the two-ref form.
4. **[Important] 🔧 Ring-(iii) qualifier missing from the headline fairness claim
   in abstract + §1** ("improves multiple fairness measures" / "consistently
   reduces demographic service disparity" read as the optimized metric). Fix
   (5 words): abstract → "improves multiple established fairness measures the
   objective never optimizes"; §1 results sentence + contribution 3 → add "on
   measures the objective never optimizes" (once each).
5. **[Important] T10's diff postdates the review snapshot** — reviewer spot-checked
   the in-flight working tree and found it CLEAN (cites travelled, madry2018pgd
   survives, oversampling disclosures merged not duplicated, lint-allow intact),
   but the committed T10 diff still needs one focused claim-safety read before the
   PI hand-off.

## Minors (#6–#14)

6. 🔧 03_methodology.tex:19–21 — §3 opener logic inverted: the blocks follow the
   steps, not the reverse. Fix: "…whose three coarse steps (attribute, edit,
   upweight) the five blocks below follow."
7. 🔧 04_experiments.tex:21 — "five questions, each answered by one subsection
   below" is false (RQ1+RQ2 → §4.2; RQ5 spans §4.5–4.6). Fix: "…answered across
   the subsections below."
8. 🔧 01_introduction.tex:121–125 — undefined synonym "global" (×2) for the defined
   term. Fix: "collective fairness is a corpus-level property, but edits are
   local" / "trace this corpus-level disparity".
9. 🔧 CITATION_PRIORITY_CHECKLIST.md — standing rule not discharged: add one dated
   Coverage line (§1 cite groups rebuilt 9→5, key set unchanged, 02_overview
   carries no cites, related work renders as §5). Tick nothing.
10. ⚖ optional — 01_introduction.tex:167 contribution 1 "connects the edit budget
    … to its corpus-level fairness impact" primes a budget-curve expectation
    (D16-adjacent). Optional soften: "connects a bounded edit budget on local
    demonstrations to corpus-level fairness."
11. ⚖ optional — edit-specificity controls no longer promised in abstract/§1
    (survive in §3.5/§4.3/§6). Optional: append "with random-subset and most-fair
    controls isolating the edits" to contribution 3.
12. (note for PI) — decision #8's "more visually striking" SZ↔SF contrast was
    discharged in prose only; no table/figure emphasis. Defensible under the
    lowered bar.
13. ⚖ optional — Fig-2 caption could pair the vocabularies once: "trim
    (outcome-side)" / "lift (resource-aware)".
14. 🔧 03_methodology.tex:273–275 — T4 compression dropped "for $Y$" from the
    leverage sentence. Restore: "more consequential for $Y$ than removing it from
    starved ones."

## Verified clean (reviewer's explicit list)
All added-line numerals vs ALGORITHM_FACTS; the new SZ-demand-channel comparative
(+0.0352 CI straddles 0 → "positive and not significant") checks; D15 fully repaired
in all three prior locations; D16 discharged; protected register intact; all E-items
and digest decisions land incl. 1:1 challenge↔block mapping, no-itemize rules, ACM
blocks, both figure rulings; no src/lint-allow losses; Fig-2 inside geometry gates.
