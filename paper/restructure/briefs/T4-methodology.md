# TASK BRIEF T4 — reorganize §3 Methodology: leading paragraph + five challenge-mapped blocks

Reorganize `paper/sections/03_methodology.tex` from its current three subsections
(objective / why-demand-only-fails / the editor) into the restructure's shape: a
leading paragraph, then five blocks, each opening by naming the challenge it answers
(C1–C5, defined in §2 Overview). This is a RE-GROUPING of existing, battle-tested
prose plus a small amount of new connective text — not a rewrite. Every load-bearing
sentence, number, disclosure, `% src:` and `% lint-allow:` comment survives, relocated.

## Read first
1. `paper/restructure/MEETING_DIGEST.md` — decisions #6 (leading ¶ + invariant), #9
   (appendix policy), ADJ-2 note "prose never merges the phases", ADJ-3 (the mapping).
2. `paper/sections/03_methodology.tex` — current state (post-T3: problem formulation
   already moved out; a TODO comment marks where your leading ¶ goes).
3. `paper/sections/02_overview.tex` — the landed C1–C5 wordings your block openers
   must echo (echo the IDEA in a clause, don't re-quote).
4. `paper/restructure/ALGORITHM_FACTS.md` — the whole file; §Validity is binding for
   block 4 and §Editor for blocks 2–3.
5. `paper/restructure/meeting/analysis_C_claims.md` — §2 C-2 (two-stage overview
   wording, adopt its proposal), C-6 (attribution wording — its "Proposed accurate
   wording" paragraph is pre-approved raw material), C-7 (why prose never describes
   one-selection-feeding-both-editors), §4.2 (the metric-description content order),
   §5 do-not-claim list.
6. `paper/restructure/ZHANG_DRAFT_DELTA.md` §Terminology (outcome-side ↔ trim,
   resource-aware ↔ lift pairing on first use).

## Target structure

**Leading paragraph** (replaces the TODO comment; no subsection heading): names FATE;
states the two stages as one method (adopt analysis_C C-2's proposed wording:
budgeted, constrained edit + edit-aware weighting, "neither stage is sufficient
alone"); references `Figure~\ref{fig:overview}`; then ONE compact mapping sentence
tying components to challenges, e.g. "The objective answers C3, attribution under the
budget answers C1, the two editing channels answer C4, the edit constraints answer
C2, and edit-aware weighting answers C5." (your drafting; keep it one or two
sentences, plain).

**Block 1 — The Collective Fairness Objective (answers C3).** Content = current
§"The Fairness Objective" reorganized per the email's shape: open with the
design-requirements sentence (the objective must be corpus-level, demand-aware,
differentiable, and attributable to local units — all four are true; one sentence,
not a list); then the existing prose in its current order (wrong-target opener →
two-stage residual construction → Eq. (1) → smoothness/attribution properties →
caveats ¶ → F_spatial ¶ → fidelity guardrail ¶ → scalarization Eq. (2)). Add one
"why this objective serves the editor" sentence covering the four properties
(collective over the full corpus; differentiable w.r.t. the allocation; exact
per-unit decomposition; supports attribution from global disparity to local
trajectories) if it is not already implied where the attribution pointer sits.
KEEP: the associational/ecological caveats paragraph verbatim; the
`\label{sec:objective}`; "collective" may join the subsection title (heading wording
is yours). Title suggestion: "The Collective Fairness Objective".

**Block 2 — Attribution under an Edit Budget (answers C1).** Content = the
attribution material currently at the top of §"The Editor" plus the budget/phase
material: the objective is optimized one trajectory at a time; TWO mechanisms, one
per phase — demand deficit attribution (exact per-unit partition of r²_demo; signed
variant; trajectories whose pickups land in highest-deficit units) and
supply-gradient attribution (v_i = ∂L/∂S_i at ΔS=0, value-of-presence map,
linearized-offset screen that only NOMINATES); how k is allocated (trim takes its
deficit-attribution selection, lift fills the remaining budget with positive-score
nominees; SZ split 2,455 selected/7,545, supply gradient computed on the POST-TRIM
state); close with the email's emphasis sentence: attribution is not a post-hoc
explanation — it is the mechanism that decides which trajectories receive the
budget. Use analysis_C C-6's proposed paragraph as raw material. NEVER: z-scoring
language, a single ranked list feeding both phases, per-trajectory fairness scores.
The `\label{sec:attribution}` anchors here.

**Block 3 — Outcome-Side and Resource-Aware Editing (answers C4).** Three beats:
(a) *Outcome-side editing (the trim operation)* — define per the email (changes the
measured allocation statistic without changing the provision process); trim
mechanics from the current trim paragraph (pickup relocates within the ε-ball,
padding recorded demand into over-served supply-rich cells; supply frozen).
(b) *The limitation* — current §"Why Demand-Only Editing Cannot Help the
Under-Served" COMPRESSED to roughly half its length, keeping VERBATIM-OR-NEAR: the
2,455-pickup empirical fact (every trim pickup originated in advantaged cells, none
landed disadvantaged); the leveling-down ANALOGY-ONLY framing with its
conservation caveat; the three structural reasons tightened to one clause each
(selection concentrates in over-served cells; ∂Y/∂D leverage + 93% of disadvantaged
units at the demand floor; median presence 1.8 vs 17.6 untouchable by demand
edits); the demand-endogeneity paragraph (bounds both metric and editor); the
"one non-perverse lever" close (∂Y_i/∂S_i > 0 everywhere → adding presence).
`\label{sec:leveling}` anchors ON THIS MATERIAL (many referrers — see label rules).
(c) *Resource-aware editing (the lift operation)* — the current lift paragraphs
(value-of-presence question, screen nominates, tail = pickup + up to 4 states,
taper + fixed anchor, moved states carry supply differentiably, supply endogenous,
shared running state, fidelity scores the actual rerouted tail every iteration);
close with the email's key distinction sentence (outcome-side changes the measured
outcome; resource-aware changes the provision process and can directly benefit
under-served areas). The two-phase scientific-control statement (lift runs after
trim and never alters trim's edits; demand-only results carry over; the ablation
isolates lift; single-pass editor = future work) lands at the end of this block —
its current plain-language wording is Robert-approved from 272bb47; move it intact.

**Block 4 — Validity- and Fidelity-Constrained Editing (answers C2).** Content =
the bounded-perturbation machinery + physical validity, currently split between the
editor intro and the "Physical validity" paragraph: the two intervention limits
(corpus-level budget k vs per-trajectory bound ε — one sentence each, K-vs-ε per the
email); the per-iteration signed-gradient update Eq. (3) with the cumulative clip
and the ε-as-identity-budget reinterpretation; soft cell assignment pointer (one
clause, detail stays App B); king-move rule + exact backward-reachability repair;
infeasible edits NOT applied (lift skips in-loop; ~5% of trim edits reverted post
hoc BEFORE metrics); then the SIX-STEP pipeline summary from ALGORITHM_FACTS
§Validity VERBATIM in structure: propose gradient edit → clip to ε → discretize +
repair continuity (skip/revert if infeasible) → objective (fairness + fidelity
terms) evaluated each iteration, best iterate kept → accepted edit updates the
shared corpus state → repeat until the budget is spent. The identity model is
described as an identity-level behavioral-fidelity guardrail, NOT a per-edit
accept/reject gate and NOT a realism guarantee (Fidelity-B measured at evaluation,
§4 pointer). `\label{sec:phys-validity}` anchors here (referenced from §4 setup).
Also keep the constructive-reading sentences (adversarial-perturbation lineage read
constructively, recourse spirit) — currently in the editor intro; they can open
this block or stay with Eq. (3).

**Block 5 — Edit-Aware Weighting (answers C5).** Content = the current "Downstream
recipe: upweighted imitation" paragraph, retitled (her email's name for the stage);
dilution problem first (edited slice ≈ a tenth; uniform weighting averages it away
and relearns the bias; the null verified in §4), then the upweighting recipe
(instance reweighing transplanted), then the two edit-specificity controls
forward-ref. `\label{sec:downstream}` anchors here. Do NOT claim downstream gains
from editing alone (D2).

## Label rules (do this FIRST, mechanically)
`grep -rn "sec:objective\|sec:leveling\|sec:editor\|sec:attribution\|sec:downstream\|sec:phys-validity" paper/sections/ paper/main.tex`
— every label with referrers MUST anchor on the content that matches what referrers
mean by it. Expected placements: sec:objective → block 1; sec:attribution → block 2;
sec:leveling → block 3(b); sec:editor → block 2 or the leading ¶'s vicinity (check
each referrer's sentence: most mean "the editor/attribution machinery" — pick the
block that reads correctly for ALL of them and say which you chose); sec:downstream
→ block 5; sec:phys-validity → block 4. Zero undefined refs after (baseline build
first, compare).

## Register + policy
- Meeting: methodology register is OURS ("whatever makes sense for us") — keep the
  existing technical prose; your new text matches its voice. Explicit over clever;
  em-dashes only to save space; no coinages.
- Compression target: the section should come out NO LONGER than it went in
  (ideally ~10–15 lines shorter via the §3.3 compression). If something must
  overflow, the ONLY sanctioned overflow is appendix B (editor detail) with a
  one-clause pointer left behind — say what you moved in your report.
- Protected register (relocated, never deleted): associational caveats ¶;
  demand-endogeneity; leveling-down analogy-only + conservation; 2,455 disclosure;
  5% revert disclosure; ε reinterpretation; taper/anchor facts; vanilla-null
  prediction; the scientific-control property.
- Era numbers unchanged: 2,455 selected → (2,337 net appears in §4, not here);
  7,545 lift; k=10,000; ε=2; α=(0.1,0.8,0.1); N≈34,500.
- New vocabulary: pair on first use — "outcome-side editing (the \emph{trim}
  operation)" / "resource-aware editing (the \emph{lift} operation)"; after that,
  trim/lift as now. "value-of-presence map" stays (our term); you may gloss it once
  as "a value-of-resource map over the city" if it helps her vocabulary land —
  optional.

## Gates + checks
- Baseline build first (undefined-ref count), then after your edit:
  `cd /home/robert/FAMAIL/paper && latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`
- Zero undefined refs; no duplicate labels; `% lint-allow: ablation` lines that
  moved still suppress correctly (lint exit 0 proves it).
- `pdftotext` read of §3 start-to-end: verify the five block headings render in
  order and each opener names its challenge label.
- Report REFERENCES start page + §6-tail spill (telemetry only).

## Rules
- Touch ONLY `paper/sections/03_methodology.tex`. 04_experiments.tex's uncommitted
  edit stays untouched. No git commands.
- If Write/Edit is blocked: full file contents as text, status BLOCKED(write-denied).

## Final reply (machine-read, ≤20 lines)
Status; the five block titles you chose + which label anchors where; the leading ¶
verbatim; what (if anything) moved to the appendix; net line delta of the section;
gate results; undefined-ref count before/after; anything from the do-not-claim list
you had to steer around.
