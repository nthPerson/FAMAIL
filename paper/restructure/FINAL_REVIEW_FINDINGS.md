# FINAL WHOLE-BRANCH REVIEW — findings (2026-07-24 late, base 39dd01a → ea6dd25 + T10 in flight)

> **→ PART 3 at the bottom of this file is the current state (2026-07-25).** Every
> finding below is dispositioned there, the 8-page limit is now met, and Part 3 carries
> the two claim defects that finding #5 turned up plus the one item still needing
> Robert's ruling. The statuses in this header describe 07-24 and are superseded.

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

---

# PART 2 — systematic prose re-read (2026-07-25, Robert's directive; base 6edbf0c)

Scope: every line of prose in the abstract and §§1–6, read in order, by the
orchestrator (not delegated). Criteria: sentence flow, fidelity to the argument,
specificity (explicit over implicit), repetition, wordiness. Smaller repairs were
applied; anything needing Robert's sign-off is #15–#21 below. Gates green at every
step: `latexmk` clean, `lint.sh` silent, 0 undefined refs/citations, no Overfull > 5pt.

**Page effect.** 14pp → 13pp, and the §6 spill onto p9 went from **29 rendered lines
to 20** (p9 now carries lines 929–948: the bounds-¶ tail plus "Three directions").
Nine lines recovered without touching the protected bounds paragraph, without an
appendix relocation, and without cutting a claim, a number, or a disclosure.
**Still 20 lines over the hard 8-page limit** — see #16 for the menu.

## Applied (no sign-off needed, but three changed meaning — marked ⚑)

**Abstract** — "applies bounded local edits WITH the greatest impact" → "TO THOSE
with" (selection ranks trajectories, not edits; matches §1). "service-provision" →
"resource-provision process" (the term §1 and contribution 2 use). ⚑ The dilution
pair was de-hedged: "their fairness signal can be diluted" → "training that weights
every demonstration equally averages their fairness signal away" — the mechanism we
actually measure (the §4.3 vanilla null), and it drops a third statement of the
budget. Closing sentence "further preserves" → "carries these gains into", matching
§6's verb.

**§1** — the prior-work paragraph's closing sentence was the third consecutive
statement of "prior work cannot attribute disparity to trajectories"; it now states
the intervention instead. Challenge 1 lost "As the left panel of Figure 1 shows,
disparity emerges from the aggregation of many trajectories, not from any single
demonstration" — ¶1 says it and the Figure-1 caption says it almost verbatim
("emerges from the aggregation of local trajectories"). Challenge 3's first two
sentences joined (a restatement of the budget requirement dropped).

**§2** — "The imitation we demonstrate on this corpus is supervised" → "The imitation
setting is supervised". "(N ≈ 34,500 on the primary city)" → "on Shenzhen, the
primary city" (first use of the term). Problem-definition (ii) reworded so the
discriminator is the agent of the sentence.

**§3** — ⚑ **accuracy repair, the substantive one:** §3.1 said "Both stages are
evaluated in closed form through the regression's projection (hat) matrix … every
edit step re-fits the regression exactly". That is wrong for stage one and vague
about stage two. Only the stage-two demographic regression re-fits implicitly via
the hat matrix (Appendix A already states it correctly); g_0 is fitted **once**
during preprocessing and then frozen, and it is evaluated under `no_grad` — which is
why the gradient clause correctly says "through Y". The body now says so, and
"a flexible curve g_0(D)" became "a four-term power-basis curve" (the estimator it
actually is: `g0_power_basis.py`, basis [1, 1/(D+1), 1/√(D+1), √(D+1)]).
⚑ zietlow2022's claim was narrowed from "data augmentation is **the one**
intervention observed to help the disadvantaged group" (reads as a claim about the
literature) to "in a study of fair image classifiers, data augmentation was the one
strategy that helped …", matching the verified refs.bib note — **flagged for your
citation pass** (see the checklist's new coverage note).
Also: C3 opener "falls on the objective" → "is a requirement on the objective" and
its duplicate business-district example dropped (kept in §2, where the reader meets
C3 first); "spatial smoothness regularizer" → "regularizes toward spatial equality"
(Gini measures equality, not smoothness); "Trim retains its selection" → "Trim
selects"; one double-colon sentence split (the non-perverse-lever sentence); §3.4's
"(about 5%) … before computing fairness metric effects" now points to §4.2, which
states the fraction and the timing precisely (the disclosure is unchanged, stated
once); C5's opener "is why editing alone is necessary but not sufficient" → "is why
editing alone is not sufficient" (nothing there argues necessity); "an artifact of
oversampling" → "of upweighting any subset", which is what the random-subset control
tests (RQ3's own wording); two related-work asides removed from §3.5 (feldman2015
taxonomy, the zheng2023 model-level contrast — both keys still cited elsewhere,
coverage note appended to the checklist).

**§4** — "The lift-up claim rides the supply channel" → "rests on"; "rather than
celebrate it" → "rather than take it at face value"; "The strict count of distinct
external instruments is therefore {DI, DP/gap, Theil}" → "The distinct external
instruments are therefore DI, DP/gap, and Theil" (a count is not a set); "the
paper's configured value in each city" → "fixed per city"; the four-source paragraph
stated its claim twice ("the claim under test is that X. The claim holds: X") and
now states it once; "Extending the dose saturates" → "The gain saturates as the dose
rises" (the dose does not saturate); "Fairness methods proper, applied where they
usually live at training time" → "Two established fairness methods, applied at
training time … where they usually live"; §4.5's opener no longer leans on feature
sets "introduced below" in the passive voice; the weight-sensitivity paragraph's
second sentence stopped restating the first; one more double colon split in §4.6,
and "observed in the wild" → "now observed in data".

**§5** — the Zheng sentence split in two (the "which is the gap FATE closes" tail was
tangled); "recovers driver policies faithfully … optimizes how faithfully" echo
fixed; "surveys these five lines" → "these lines" (the paragraph does not present
five countable lines).

**§6** — ⚑ "FATE makes fairness **a property of** the demonstrations … the fairness it
repairs is collective, **a property of** the whole corpus's service allocation" used
one word for two levels in one sentence, the exact ambiguity the D3 guard exists for.
It now reads "places the fairness intervention in the demonstrations", with the
collective definition intact. "confirm the edit itself is what carries them" →
"is responsible" (carries/carries); "Three directions follow naturally" → "follow".
**The bounds paragraph was not touched** (byte-identical protected register).

## New findings for Robert

15. **[Important] ⚖ District-vs-tract granularity: §2 states a Shenzhen fact as a
    global one.** §2 says "These covariates resolve at district granularity, which
    bounds the resolution of any demographic analysis built on them", but §4.1 says
    San Francisco's features are "filled from ACS tract values", and
    `PAPER/argument/02_datasets.md` confirms the asymmetry (SZ = 10 district
    profiles; SF = ACS 2006–2010 tracts → cells). Knock-on: §3.1's caveat "With
    about ten district profiles the association is ecological" and the **protected**
    §6 bounds sentence "a partial R² over roughly ten district-level demographic
    profiles" are both stated globally. For SF the caveat is conservative in the safe
    direction (it claims a coarser resolution than SF has), so this is a precision
    problem, not an overclaim — but a reviewer reading §2 then §4.1 sees the tension.
    Proposed §2 wording (yours to approve, since it touches the caveat register that
    the protected ¶ shares): "These covariates resolve at administrative granularity
    (ten districts on Shenzhen; census tracts mapped to cells on San Francisco,
    \S\ref{sec:exp-setup}), which bounds the resolution …". I did not change §3.1 or
    §6.
16. **[Page budget] 20 rendered lines still over.** Menu, cheapest-first, with
    estimates. Nothing here is done; all of it is your call.
    - **Figure 2 height** (currently ~290pt ≈ 25 text lines): trimming it to ~250pt
      costs no prose at all. **~3–4 lines.** Cheapest real lever in the paper.
    - **§6 "Three directions" ¶** (11 lines): each direction can be one clause
      instead of two. **~4–5 lines**, no claim lost, bounds ¶ untouched.
    - **§1 ¶4 vs contribution 2** state the outcome-side/resource-aware distinction
      near-verbatim (see #18). **~3–4 lines.**
    - **Table 2 (cross-arm baselines) → appendix**, leaving RQ4 to prose. **~10
      lines**, but it is the table that answers "is it the objective or just
      perturbation?" — I would not do this before the other four.
    - §3.4 ↔ §5 both spell out the recourse framing (#20): **~2 lines.**
    - Fidelity-B 0.187 appears in §4.2, §4.3, §4.4; §4.3's is droppable: **~1 line.**
    - §4.1's 106,677-pickup cleanup detail → appendix: **~1–2 lines.**
17. **⚖ §1 ¶3 (her prose): difficulties 1 and 3 both cover the budget.** Difficulty 1
    ends "trace this global disparity to influential trajectories and identify the
    local edits with the largest corpus-level effect"; difficulty 3 opens "fairness
    intervention must be effective under a limited edit budget" and closes "achieve
    substantial collective fairness improvement by modifying only a small subset".
    The reader meets the same challenge twice, and the closing five-label list
    (C1…C5) runs in a different order than the three prose difficulties. Defensible
    as-is (the closing sentence says the five labels make the difficulties
    *precise*, not that they map 1:1) — but if you want ¶3 restructured to three
    difficulties that map cleanly onto C1/C4 + the C2/C3/C5 trio, that is a rewrite
    of her paragraph and I did not attempt it.
18. **⚖ optional — §1 ¶4 and contribution 2 duplicate the trim/lift distinction.**
    ¶4: "distinguishes outcome-side edits, which change the measured allocation
    statistic, from resource-aware edits, which alter the underlying
    resource-provision process. The former may reduce disparity without benefiting
    disadvantaged groups …". Contribution 2 says the same in different words. Some
    repetition between narrative and contributions is conventional; this is close to
    verbatim. ~3–4 lines if you want ¶4's version shortened (keep contribution 2 —
    it carries the C4 insight).
19. **⚖ Abstract carries no number.** Zhang's text states no quantitative result. The
    headline (+0.0226 F_demo, or the 3.0× disparity, or "12/12 seeds, p = .00049")
    would give a reviewer something to anchor on, at a cost of half a line. Your
    call — it is her abstract, and she may have dropped numbers deliberately.
20. **(note) Duplications I deliberately left.** §2's C4 and §3.3's opener state the
    same finding — by design (decision #6: each block names its challenge in its
    opening sentence). §3.4 and §5 both spell out the constructive-recourse framing
    with the same five citations — the §5 version belongs in related work and the
    §3.4 version is the methodological justification. "Small fraction of the corpus"
    twice in the abstract reads as a deliberate bookend for the budget claim.
21. **(scope) Appendix prose was not re-read line by line** in this pass — main
    content only, since the 8-page pressure and your close read are both there. I did
    read Appendix A's derivations (that is where the g_0 repair was verified) and the
    SF results block. Say the word and I will do A–E next.

---

# PART 3 — findings applied (2026-07-25, Robert approved all but #1; base 44763ea)

**Headline: the paper now fits KDD's hard 8-page limit.** `pdftotext -f 9 -l 9`
opens with REFERENCES; main content ends on page 8. Total 13pp. Gates green
throughout: `latexmk` clean, `lint.sh` silent, 0 undefined refs, 0 `??` in the PDF,
no Overfull > 5pt.

Trajectory of the §6 spill onto p9: **29 lines (07-24) → 20 (Part 2) → 0 (now).**

## ⚠ Zero margin — read this before the close read

Page 8 holds **116 of 116 available lines in both columns**, the same count as every
float-free page (verified against p5 and p7). There is no slack. Any net addition
during the close read pushes content back onto p9, and the Overleaf port is a real
risk here: a different TeX Live version can change hyphenation and reflow a line.

Reserve levers, cheapest first, none of them yet used:
- **Figure 2's height** (~290pt). I did NOT touch it — the limit was met without it,
  and it is the other session's style reference. Realistic yield on inspection is
  ~0.5 line, not the 3–4 I estimated in #16: the vertical rhythm is set by text
  blocks and by the stage-3 title band, which the left op-box cannot pass. Lower
  yield than it looks.
- The abstract's second sentence and §1 ¶1–¶2 (PI prose, so far untouched): ~2 lines.
- §4.6's San Francisco caveats: ~1 line of wordiness.
- Table 1 → appendix (~7 lines). Last resort; it backs the headline RQ1 result.
- If the other session's TikZ Figure 1 comes in under 135pt, that is free margin.

## Disposition of every open finding

| # | Status |
|---|---|
| 1 | Handed to the parallel Figure-1 TikZ session (not touched here) |
| 5 | ✅ **discharged, and it found two real defects — see below** |
| 8 | ✅ applied: "collective fairness is **global**" ×2 → corpus-level property / "the disparity" |
| 10 | ✅ applied: contribution 1 → "connects **a bounded** edit budget on local demonstrations to corpus-level fairness" (drops the budget-curve priming, D16-adjacent) |
| 11 | ✅ applied: contribution 3 gains "with random-subset and most-fair controls isolating the edit" |
| 12 | No action (note for the PI; defensible under the meeting's lowered bar) |
| 13 | ✅ applied: Fig-2 caption now pairs the vocabularies once — "trim (outcome-side) and lift (resource-aware)" |
| 15 | ✅ applied in §2 + §3.1; **one residual needs your ruling** — see below |
| 16 | ✅ all five cheap levers applied **plus** Table 2 → appendix; Figure 2 untouched |
| 17 | ✅ applied: ¶3 restructured so difficulties 1 and 3 no longer both argue the budget |
| 18 | ✅ applied: ¶4's restatement of the trim/lift distinction dropped; contribution 2 keeps the C4 insight |
| 19 | ✅ applied — but **not** with the number I first proposed; see below |
| 20 | Unchanged by design |
| 21 | Still open (appendix not re-read line by line; #5 covered the T10-relocated blocks) |

## #5 found two claim defects in T10's relocated appendix block

Both in Appendix B, "Why demand-only editing has no lever in under-served units",
the block T10 moved out of §3.3. Repaired, with the reasoning in a comment beside each:

1. **"so under-served cells are never edit candidates" — dropped.** The claim asserts
   a structural guarantee the selection rule does not provide. Trim ranks by *signed*
   per-unit attribution (`attribution.py::rank_trajectories`: ascending, "most-negative
   α_i first"), and that score is built from squared residual projections — so a
   strongly *under*-served unit whose residual is demographically predictable can also
   score negative and be nominated. Under-served cells going unselected is an
   **empirical** fact about this corpus (the 2,455/2,455 finding, already in §3.3), not
   a property of the rule. The surviving clause — candidates *concentrate* in
   over-served, high-residual cells — is what the mechanism supports, and Leverage plus
   Supply-side inequity carry the rest of the argument unchanged.
2. **"an upper bound on any editor" → "on any demand-only editor".** Read literally,
   the original said no editor can help the under-served — the exact opposite of the
   paper's central result, since the lift channel does precisely that. Compression
   dropped the scope; §3.3 frames the same bound correctly as "the constrained optimum
   of the demand-only problem".

Also verified for #5: every numeric token T10 removed from §2/§3/§4 still exists in the
paper (21 tokens checked mechanically; the one apparent miss, 98.8%, was reworded, not
lost, and survives in §4.4). #14's "for $Y$" is present at appendix.tex:147 — that fix
did land, despite the T10 diff appearing to drop it.

## #19: the number I planned would have been false

I proposed the 3.0× disparity or "about a tenth of the corpus". **Both fail as two-city
claims** and I checked before writing either: disparate impact before editing is 0.3325
on Shenzhen (3.01×) but 0.7076 on San Francisco (1.41×), and the budget share is
10.5% on Shenzhen (k=10,000/95,297) versus 18.4% on San Francisco (k=2,000/10,887).
The abstract now carries the flagship instead, which *is* two-city-safe: **"on 12 of 12
paired seeds in both cities ($p = .00049$)"**, with its `% src` pointer and a comment
recording why the other two candidates were rejected.

## ⚖ The one residual of #15 (protected register — your call)

§2 now says the covariates "resolve at coarse administrative units (districts or census
tracts)", true of both cities, and §3.1's ecological caveat is scoped ("about ten
district profiles **on Shenzhen**"). The §6 bounds paragraph still states it globally:
"a partial $R^2$ over roughly ten district-level demographic profiles". That sentence is
**byte-identical protected register** from M44, so I did not touch it. Two words
("on Shenzhen") would make it consistent with §3.1. For San Francisco the claim errs
conservative — it asserts coarser resolution than the tract data has — so this is
precision, not overclaim.

## Also applied while working (each a genuine repair, none requested)

- Two Overfull boxes my §1 rewrites created (5.01pt and 5.56pt) were closed by
  tightening rather than by reverting: "is an *associational* quantity" → "is
  *associational*"; "the same assumption turns out to bound" → "bounds"; "In the
  taxi-mobility instantiation studied in this paper" → "In our taxi-mobility
  instantiation"; "FATE further applies" → "FATE applies".
- §4.2's ablation gloss ("trim only redistributes existing service; lift adds taxi
  presence where it was missing") dropped as the **second** statement of it inside the
  same subsection — the ablation's own opening sentence already says it.
- §4.3's one-sentence "Model-level variance" paragraph folded into the end of the
  upweighting paragraph: same claim, same appendix pointer, one fewer paragraph break.
- §4.1's stuck-GPS cleanup sentence (106,677 pickups) moved to Appendix B's grid
  block with its `% src`; §4.1 now points there.

---

# PART 4 — Figure 1 swapped to TikZ (2026-07-25)

`01_introduction.tex` now does `\input{figures/figure-1/figure-1-teaser}` instead of
`\includegraphics{...teaser.png}`, and the `\Description` was replaced (the old one
described the PNG, and described it wrongly). **Finding #1 is closed by construction:**
the depiction defect is gone rather than nudged.

Verified independently of the other session's report, from the coordinate lists in the
figure's config block (`\tzBx` = 0.517 is the boundary fraction, shared by both panels):

- Advantaged taxis `0.086, 0.228, 0.442, 0.086, 0.360` → 5, all left of the boundary;
  right panel drops `0.442/0.287` → 4. Disadvantaged taxis `0.886, 0.934` → 2 in both
  panels, plus the amber taxi at 0.630 → 3. **One leaves, one arrives; 7 both panels.**
- Advantaged passengers `0.119, 0.047, 0.098` → 3, drawn in both panels.
- Disadvantaged passengers `0.655, 0.920, 0.861` plus the served one at 0.740 → **4 in
  both panels, same coordinates**, the served one recolored rather than moved. This is
  the fix: demand does not move, which is what her own "Demand: Similar" label claims.
- Edit path runs from the vacated taxi position (0.442, 0.287) down and east to 0.560,
  crossing 0.517, ending with the only arrowhead in the figure at the arriving taxi.
- Label strings are her wording verbatim; no era numbers; no forbidden vocabulary.
- Box re-measured from the harness myself: **239.50 × 116.83pt** (cap 241.15 × 135.0).
- Grayscale render inspected: counts, boundary dash, edit dash and arrowhead all survive.

**Page budget: no net gain, contrary to what the 18.2pt looked like it would buy.** The
figure is 18.2pt shorter than the PNG, but LaTeX absorbed that into float separation on
page 1 rather than converting it into a text line: p1 still holds 116 lines, §6 still
ends at line 927, p8 is still full to the same baseline as p7. 13pp, p9 = REFERENCES,
Fig-1 still on p1. The one real gain is elasticity — page 1 now carries ~18pt of
compressible glue that can absorb a line if the close read adds one.

**Still awaiting Robert (from the other session's report, unchanged by the swap):**
Conflict A (area tints: amber/cobalt default vs her green/pink, `\tzTintScheme`),
Conflict B (taxi color: neutral default vs her blue, `\tzTaxiMode`), and D11 (the trim
overlay exists, verified correct, and is OFF). Each flips with one macro.

---

# PART 5 — page-1 front matter fixed; the zero-margin warning is retired (2026-07-25)

Robert spotted the ACM Reference Format block breaking across the column boundary on
page 1. Diagnosis: that block needs 4 lines in column 1, but only 2 fit before acmart's
permission block claims the bottom, so its last 2 lines ("Knowledge Discovery and Data
Mining (KDD '27). ACM, New York, NY, USA, / 13 pages.") landed in column 2 between the
Figure-1 caption and the §1 heading. The fix had to be worth **2** lines, not 1: freeing
one would have pulled back only the first and orphaned "13 pages." in column 2.

Applied (Robert's choice of four options, the hybrid): the lowest-significance CCS
concept "Applied computing~Transportation" (100) removed, taking the CCS block from 3
rendered lines to 2, plus two abstract trims worth ~10 words, both of which removed
repetition rather than content — "training that weights every demonstration equally"
→ "training under uniform weights" (the paper's own term in §4.3, §6 and Figure 2), and
"editing a small fraction of the corpus improves" → "these edits improve" (the phrase
"small fraction of the corpus" appeared twice in four sentences).

**Result, and it is larger than the fix itself: the spill is gone, and main content now
ends at line 921 of the 928 available on page 8 — roughly 7 rendered lines of headroom,
where Part 3 recorded zero.** REFERENCES now begins on page 8. The close read has room
to breathe, and the Overleaf-reflow risk Part 3 flagged is much reduced.

Also this round: contribution 1's "while only a small subset of individual trajectories
**can be modified**" was factually wrong (nothing prevents editing any trajectory; the
editor applies to whatever attribution selects). It now reads "while only a small subset
**is edited by design**" — Robert's pick from four alternatives, and 2 words shorter.

⚠ Open, deliberately not touched: the abstract now reads "…gig-worker traces, **encode**
not only human decision-making strategies" but two sentences later "When such data
**is** used". Both treatments of *data* are defensible; the abstract should pick one.
§1 uses the singular ("HSTD **is** not a neutral record").

---

# PART 6 — three swallowed-sentence defects I introduced, found and fixed (2026-07-25)

Robert caught §3.1 rendering as "…with nothing trained inside the editing loop. **form,**
its exact per-unit attribution…". Cause was mine and it was systematic: when I inserted
multi-line `%` provenance comments immediately above existing prose, the last comment
line absorbed the opening words of the sentence that followed. Three instances, all from
this session's edits:

| Where | Swallowed | Rendered as |
|---|---|---|
| §3.1, the g_0 accuracy repair | "The closed" | "…editing loop. form, its exact…" |
| §3.3, the zietlow2022 citation repair | "The analogy is inexact in one" | "…pulling others down [45]. instructive way: trim…" |
| §6, the D3 property/intervention repair | "It has two" | "…service allocation. stages: a budgeted…" |

Only the first was visible to a reader as obvious nonsense; the other two produced
sentences that still parsed and would likely have survived a read-aloud.

**Reusable detector** (this is what found the other two — worth running before any
hand-off, since it also catches dropped words generally):

```
pdftotext main.pdf - | tr '\n' ' ' \
  | grep -oE "\. [a-z][a-z]+[^.]{0,55}" \
  | grep -vE "\. (vs|cf|e\.g|i\.e|et al|pp|https|arXiv)"
```

It flags every sentence that begins with a lowercase word. Remaining hits after the fix
are all genuine abbreviations ("n.s.\ at w20", "iFGSM (rand.\ restart)") — a candidate
for a lint.sh rule with an abbreviation allowlist, which I did not add this close to the
deadline. §3.1's pointer was restored in active voice ("Appendix A gives the closed form,
the exact per-unit attribution, and the $O(N)$ evaluation identity"), which fixes the
sentence, removes a dangling "its", and occupies the space the broken text did. The other
two were restored verbatim; §3.3's is protected register.
