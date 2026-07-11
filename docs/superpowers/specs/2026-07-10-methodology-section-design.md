# Design: KDD paper scaffold + methodology section (T5) + draft abstract (T6)

**Date:** 2026-07-10 · **Status:** approved (Robert, 2026-07-10 night)
**Task provenance:** Meeting-42 PI directives — T5 "start writing, methodology first" (Dr. Zhang) and
T6 "draft abstract to Dr. Zhang by next week." Deadlines: abstract to Zhang ~Jul 17, KDD abstract
deadline ≈ Jul 19, paper Jul 26. KDD template = the ST-iFGSM paper (ACM `acmart` format).

## Goal

Create the paper manuscript scaffold (`paper/`, LaTeX, compilable end-to-end from day one), draft the
full methodology section (`sections/03_methodology.tex`), and write a real ~200-word draft abstract in
`main.tex`. Everything else in the paper is stubs with pointer comments, to be filled by later tasks.

## Decisions locked during brainstorm (Robert, 2026-07-10)

1. **Deliverable = LaTeX scaffold + section** (not markdown-first, not hybrid).
2. **Scope = method-only.** The evaluation protocol (external metrics, two-pillar arms, baselines,
   statistical conventions) is deferred to a later Experiments task.
3. **Narrative = mechanism subsection.** The method is presented final-form, with the leveling-down
   mechanism given its own short subsection that analytically motivates the supply channel.
4. **Abstract = real draft now**, marked draft-pending-final-results (Zhang accepted placeholder-level,
   so a true draft over-delivers safely).
5. **Workflow:** the repo is the writing source of truth; Robert ports completed sections to the
   existing Overleaf for Dr. Zhang's review. The old Overleaf content is not touched by this work.
   Each section file must therefore stay self-contained enough to paste (shared custom macros live in
   one documented preamble block in `main.tex`).
6. **Toolchain:** TeX Live installed locally (verified: `pdflatex`, `latexmk`, `acmart.cls`) →
   `latexmk -pdf` is the compile gate.

## 1. The scaffold — new top-level `paper/` directory

`paper/` (lowercase) is the **manuscript**; the existing `PAPER/` remains the results/provenance
bundle. Layout:

```
paper/
  main.tex              acmart (sigconf), anonymous draft toggle (KDD is double-blind);
                        title, the REAL draft abstract, \input's the section files
  sections/
    01_introduction.tex   stub — skeleton + % pointer comments → PAPER/argument/01
    02_related_work.tex   stub — % pointers → PAPER/objective-motivation/MOTIVATION.md contrast
                          paragraphs + REFERENCES.md
    03_methodology.tex    ★ THE deliverable of this task (outline in §2)
    04_experiments.tex    stub — % pointers → PAPER/supply-lift, PAPER/external-metrics,
                          PAPER/baselines, PAPER/by_feature_set, PAPER/second-dataset
    05_conclusion.tex     stub
  refs.bib              seeded from PAPER/objective-motivation/REFERENCES.md (metadata verified
                        2026-07-08; header comment notes Robert's T3 human pass is still pending)
  README.md             build instructions (latexmk) + the writing conventions below, so every
                        future writing session inherits the same rules
```

Stubs are near-empty on purpose: the paper compiles end-to-end from day one, and each later writing
task fills exactly one file.

### Writing conventions (pinned in `paper/README.md`)

All previously locked decisions, enforced in one place:

- **Trim+lift centers ALL reporting**; trim-only numbers appear **only** in the trim-vs-trim+lift
  ablation (decision 2026-07-09/10; operationalizes Zhang's "the ablation is necessary").
- **`F_causal` keeps its label + associational caveat**; no causality *argument*; no `F_demo` rename
  (pending PI decision — do not pre-empt).
- **The spoken "54%" figure is banned** until grounded — absolute deltas only
  (+0.0222 SZ / +0.0328 SF).
- **`p = 0.031` never appears without** mean Δ + t-CI + monotone dose-response (it is the n=6
  Wilcoxon sign-unanimity floor, not an effect size).
- **SF *reproduces* Shenzhen, never "beats" it** (F_causal is city-specific and associational).
- **Every number carries a `%` provenance comment** naming its `PAPER/` source file.
- **Any single supply number states its accounting tier** — tier-1 (fractional presence, the
  optimizer's convention) vs tier-2 (distinct-taxi recount from raw GPS)
  (`PAPER/supply-lift/LIFT_ALGORITHM_REFERENCE.md` §10).
- **Three-ring metric firewall** (`LIFT_ALGORITHM_REFERENCE.md` §13): (i) optimized
  (F_spatial/F_causal/F_fidelity), (ii) design-targeted-not-optimized (mean(Y|D)/SDR family),
  (iii) genuinely external (DP, DI, Theil, per-group levels, tier-2 recount, channel decomposition).
  "Improves metrics we never optimized" claims ride ring (iii) only.

## 2. `03_methodology.tex` — outline and source map

Method-only scope. Target ~2.5–3 two-column pages. Nearly number-free — numbers live in Experiments —
except the mechanism subsection's structural facts and the α weights.

| § | subsection | content | assembles from |
|---|---|---|---|
| 3.1 | Problem formulation | grid/time units; demand D, supply S (5×5 presence), service ratio Y = S/max(D, DEMAND_FLOOR); district-level demographics; the edit-then-upweight data-augmentation goal | `PAPER/argument/02`, `03`; conventions table `LIFT_ALGORITHM_REFERENCE.md` §3 |
| 3.2 | Fairness objective | **F_causal** (FWL double regression, conditional-statistical-parity framing, associational caveat), **F_spatial** (differentiable pairwise Gini), **F_fidelity** (frozen ST-SiameseNet, honest guardrail framing); linear scalarization + "why these weights" citing the empirical α-Pareto frontier (final numbers slot in when the sweep lands ~2026-07-11 AM) | `PAPER/objective-motivation/MOTIVATION.md` (near camera-ready), `argument/03` |
| 3.3 | **Attribution — two mechanisms, one per edit mode** | **(a) Deficit attribution (drives trim):** the exact per-unit decomposition of r²_demo — *where existing unfairness concentrates* → which pickups to relocate. **(b) Supply-gradient attribution (drives lift):** one backward pass yields ∂L/∂S at every active unit (the value-of-presence map; the F_causal component has a closed form, autograd-verified — **included as a display equation**, see judgment call 1); a summed-area-table 5×5 box sum + linearized best-δ screen ranks all ~95k trajectories by predicted gain of rigidly translating their seeking tail; the screen *nominates*, the per-edit optimizer decides. Parallel framing: (a) attributes the *deficit*, (b) attributes the *remedy* | `argument/03`; `LIFT_ALGORITHM_REFERENCE.md` §4.1–4.3 |
| 3.4 | Why demand-only editing levels down | ∂Y/∂D leverage asymmetry (~32×), 93% of disadvantaged units at DEMAND_FLOOR, frozen supply ⇒ leveling-down is the *constrained optimum*, not an optimizer quirk; demand-endogeneity tie-in (Ensign et al. 2018; Lum & Isaac 2016); ⇒ the one non-perverse lever is the numerator: ∂Y/∂S = 1/max(D, 0.5) > 0, ΔY = 2·ΔS at the floor. One sentence on the Stage-0 oracle as a design gate (see judgment call 2) | `PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md`, `PAPER/objective-motivation/LEVELING_DOWN.md`, `LIFT_ALGORITHM_REFERENCE.md` §1–2 |
| 3.5 | The trim+lift editor | shared ST-iFGSM machinery (ε-L∞ bound, soft-cell 5×5 Gaussian smoothing, τ annealing, best-iterate); **trim** = pickup relocation (published mechanism, optimization byte-identical); **lift** = tapered whole-tail translation (taper 0.25/0.5/0.75/1.0, anchor fixed) with **endogenous differentiable ΔS** (presence mass moves with the tail; objective sees clamp(S+ΔS, SUPPLY_FLOOR)) and live fidelity scoring of the rerouted tail; king-move constraint (source-preprocessing provenance) + provably-exact backward-reachability repair + skip-on-infeasible rule; trim-precedence budget fill (k = 10,000 → 2,455 trim + 7,545 lift). Closes with the *two-phase-as-scientific-control* rationale: trim frozen ⇒ published results reproduce bit-for-bit inside the combined run ⇒ trim-only vs trim+lift is a clean ablation; unified one-pass editing named as future work | `LIFT_ALGORITHM_REFERENCE.md` §3, §5–6; `docs/presentations/meeting_42_update/trim_plus_lift_explainer.md` §1–3 (narrative register); `PAPER/supply-lift/FINDINGS.md` §2 |
| 3.6 | Downstream training recipe | upweighting edited demonstrations = instance reweighing (Kamiran & Calders 2012) transplanted to imitation learning; one paragraph on why vanilla BC averages the edit away (the null itself is an Experiments result) | `MOTIVATION.md` downstream § |

**Method-overview figure:** `\begin{figure}` TODO placeholder, with the explainer's three-panel
concept (gap / trim / lift — `trim_plus_lift_explainer.md` §4) noted in a comment as the design
candidate. Figure production is its own later task.

**Judgment calls (approved as-is; flip on request):**
1. The ∂F_causal/∂S closed form **goes in the paper** (§3.3b, one display equation) — autograd-verified,
   shows the supply gradient has interpretable structure, strengthens attribution-as-contribution.
2. The Stage-0 oracle gets **one sentence in §3.4**, not a subsection — it is a design-validation
   device; its numbers belong with Experiments/appendix.

## 3. The draft abstract (in `main.tex`)

~200 words, real content, marked `% DRAFT pending final results`. Arc: imitation-learned demand models
inherit service inequity → FAMAIL edits an attribution-targeted slice of real trajectories (trim+lift)
under a frozen-discriminator realism bound, then upweights the edited demonstrations → fairness
improves on established metrics **not** in the objective (DP/DI/Theil, before→after, two cities);
first statistically robust supply-channel lift-up of the under-served group's service; the recovery
through weighted BC is edit-specific (random/select controls null); naive demographic oversampling
fails despite 10.5% corpus inflation. Absolute deltas only; no "54%".

## 4. Verification

1. **Compile gate:** `latexmk -pdf` builds clean after every task.
2. **Number + convention audit:** a fresh-agent pass tracing every number in `03_methodology.tex` +
   the abstract to its `PAPER/` source file, plus a grep-lint for banned patterns ("54%",
   causal-effect phrasing, trim-only numbers outside ablation context, supply numbers missing a tier
   label, product names) — the verification pattern that took `PAPER/argument/` to 0 errors.

## Out of scope (this task)

- Content for introduction / related work / experiments / conclusion (stubs + pointers only).
- Figure production (placeholder only).
- The α-sweep fold-in itself (separate task; §3.2's "why these weights" carries the partial-frontier
  finding with a TODO marker until the sweep lands, then the fold-in task finalizes it).
- Overleaf porting (Robert's workflow).
- Any Experiments-section numbers or tables.

## Dependencies & risks

- **α-sweep** (5/5 ETA ~2026-07-11 AM): §3.2's weight-justification sentence initially cites the
  4-point partial finding (flat frontier) with a `% TODO(alpha-sweep)` marker; the fold-in task
  replaces it. Not blocking.
- **`F_demo` rename** is a pending PI decision — the section writes `F_causal` + caveat and must not
  pre-empt the rename.
- **KDD anonymity:** draft compiles with the `anonymous` toggle available; author block behind a flag.
- The Meeting-42 transcript's "54%" figure must not enter any file (convention above).

## Sources read for this design (grounding)

`PAPER/supply-lift/LIFT_ALGORITHM_REFERENCE.md` (authoritative mechanism record, 2026-07-10),
`PAPER/supply-lift/FINDINGS.md`, `PAPER/objective-motivation/{MOTIVATION,LEVELING_DOWN}.md`,
`PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md`, `PAPER/argument/` (00–04, 07),
`PAPER/baselines/README.md`, `docs/presentations/meeting_42_update/{supply_lift_briefing,
trim_plus_lift_explainer,SUPPLY_LIFT_UPDATE}.md` (pre-run drafts, marked as such),
`famail_temporal/baselines/meeting_prep/MEETING_43_PREP.md` (numbers cheat-sheet).
