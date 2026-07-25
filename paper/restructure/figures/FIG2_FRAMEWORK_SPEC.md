# FIGURE 2 SPEC — FATE framework/sequence diagram (replaces the 3-panel city figure)

Sources: meeting [12:55–27:49, 34:45, 52:07–52:14, 56:27]; style target
`../meeting/fig2_style_screenshot.png` (= ST-iFGSM Fig. 3, drawn in Keynote; the green
handwriting on it is Robert's live annotation "S1 → S2 → T1" = the topology to copy);
claim constraints from `../meeting/analysis_C_claims.md` (C-6, C-7, D3–D9) as
adjudicated in `../MEETING_DIGEST.md` ADJ-2. Requirements source of record:
`../meeting/analysis_B_actions.md` §3.

**What Zhang requires (COMMITTED):**
- Input → stages → output, nothing more. Abstract, NOT algorithm-faithful; mechanism
  detail goes to the appendix.
- The attribution beat must show PROCEDURE, not result: original trajectories in →
  scoring against the fairness deficit → the budgeted few selected (k ≪ N; her words:
  "100 out of 1000").
- Advantaged/disadvantaged contrast appears WITHOUT a literal city grid (districts
  "not necessarily" kept; abstract color contrast is enough).
- Stylized trajectories (polyline glyphs), not cars/stick-figures, as the data motif.
- ST-iFGSM Fig-3 style: outer black-bordered canvas; flat color band per stage; bold
  in-band stage titles ("Stage 1: …"); gray/white data boxes with τ₁ τ₂ … τ_k glyphs;
  arrows carrying small boxed operation labels; terminal CONTRAST boxes (their red
  "Wrong Label!" vs green "Correct Label!").
- The methodology's leading paragraph will reference this figure; its stage order
  should match the methodology block order (Robert, [35:42], unopposed).

**Accuracy constraints (binding; ADJ-2 as amended by Robert's 07-25 ruling):**
- Robert APPROVED the abstracted single-attribute-then-edit flow for the FIGURE
  (C D9 permits visual merging): phase 1 selects k ≪ N, phase 2 edits them with
  trim+lift. The caption/labels stay silent on mechanism count or defer to §3
  ("attribution mechanisms detailed in §3.2") — PROSE never merges the phases and
  never describes one ranked list feeding both editors.
- Trajectory-group labels avoid per-trajectory-fairness verdicts (C D6): score by
  "contribution to global fairness" / "fairness impact of an edit", not "this
  trajectory is unfair". (Robert's group names are flexible by his own note.)
- No z-score/normalization vocabulary on scores.
- No "attack"/"perturbation" vocabulary anywhere.
- Trim edits act in over-served areas; ONLY lift touches the under-served side (by
  adding presence). The output panel must not show demand moved INTO the
  disadvantaged region.
- Caption calls the figure a schematic/framework overview; never "case study".
- Fidelity appears as a constraint annotation ("bounded ε; frozen identity
  discriminator in the objective"), never as an accept/reject gate glyph.

## ADOPTED LAYOUT — Robert's three-phase design (⚖ RULED 2026-07-25; supersedes the
earlier Option A/B drafts, which are preserved in git history at bf64eee)

Three logical phases, ST-iFGSM-style bands, colors consistent ACROSS all phases;
simplicity mandate: "abstract the framework into logical steps… not perfectly
represent the exact algorithm" (Robert). Stage order matches the methodology blocks.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ [INPUT strip] Raw taxi trajectory corpus  τ₁ τ₂ … τ_N                   │
│  (stylized gray polylines; optional advantaged/disadvantaged tint pair) │
├─────────────────────────────────────────────────────────────────────────┤
│ [PHASE 1 band] Attribute — select the trajectories that matter          │
│   trajectories drawn in THREE GROUPS, colored by attribution outcome:   │
│     ● selected, high fairness impact (k ≪ N)      [color A, e.g. orange]│
│     ● scored, low impact — left unchanged          [color B, e.g. blue] │
│     ● not selected — left unchanged                [gray]               │
│   arrow label in: "score each trajectory's contribution to the          │
│   corpus's collective fairness"; annotation: "edit budget k ≪ N"        │
├─────────────────────────────────────────────────────────────────────────┤
│ [PHASE 2 band] Trim + Lift — bounded gradient edits                     │
│   selected (color-A) trajectories pass through "gradient ascent on the  │
│   fairness objective L"; two labeled edit glyphs: trim (pickup          │
│   relocated, ≤ ε) and lift (final seeking tail rerouted, ≤ ε);          │
│   constraint strip: "every edit ≤ ε cells · continuity repaired ·       │
│   frozen driver-identity discriminator in the objective"                │
│   output: edited corpus (k edited in color A-dashed, N−k untouched)     │
├─────────────────────────────────────────────────────────────────────────┤
│ [PHASE 3 band] Upweight — edit-aware weighted training                  │
│   edited corpus → imitation training, arrow label "upweight the k       │
│   edited demonstrations" → trained policy                               │
│   terminal contrast pair: ✗ uniform weights: bias reproduced            │
│                           ✓ upweighted: fairer service allocation       │
└─────────────────────────────────────────────────────────────────────────┘
```

Design notes:
- The three trajectory groups are THE phase-1 content (Robert's explicit spec);
  group names flexible but avoid per-trajectory-fairness verdicts (see accuracy
  constraints above) — "high impact / low impact / not selected" reads well.
- Color continuity: color A marks the selected-and-edited trajectories in ALL three
  bands (solid when selected, dashed after editing — consistent with Fig-1's
  dashed-orange edited-trajectory vocabulary). Gray = untouched everywhere.
- Phase 2 shows trim and lift as two glyphs INSIDE one band (approved abstraction);
  no separate selection arrows into each.
- Terminal contrast pair mirrors the exemplar's Wrong/Correct Label boxes and encodes
  the vanilla-null vs upweighted result honestly.
- Grayscale-safe: dashing + shape + labels carry meaning, not color alone.

## Implementation plan

1. Build as standalone TikZ at `paper/figures/figure-2/framework.tex` with test
   harness `framework-test.tex` (pattern: figures/figure-1/ has the same). TikZ is OUR
   fast path (compiles inside the gates; no Keynote available); Zhang ranked tool
   freedom explicitly — content first, polish later.
2. Palette: match the exemplar's flat bands (light blue / peach / light green at low
   saturation), grayscale-safe (band labels + glyph shapes carry meaning, not color
   alone). Trajectories: dark-gray polylines; edited ones dashed orange (consistent
   with Fig-1's legend vocabulary).
3. Column width, target height ≤ 0.5\textheight. \scriptsize minimum text.
4. Caption (draft matching the ADOPTED three-phase layout; adjust in review): "FATE
   framework. (1) Attribution scores every trajectory's contribution to the corpus's
   collective fairness and selects the k ≪ N whose bounded edits would improve it
   most (the attribution mechanisms are detailed in \S\ref{sec:editor}). (2) The
   selected trajectories receive bounded gradient edits under the fairness objective:
   trim relocates recorded pickups within over-served areas, and lift reroutes final
   seeking states into under-served cells; every edit stays within \(\varepsilon\)
   cells, is repaired for continuity, and is scored by a frozen driver-identity
   discriminator inside the objective. (3) The k edited demonstrations are upweighted
   during imitation so the corpus-level fairness gain survives training; at uniform
   weights it is averaged away." (Numbers stay out; k, N, ε appear symbolically. The
   §-defer clause is what keeps the merged phase-2 band honest.)
5. Gate: renders under latexmk + lint; then send the standalone PDF to Zhang EARLY
   (meeting A5: cheap veto while revert is possible).
6. `\label{fig:overview}` is kept (current §3.4 opener and Fig-1 caption reference it).
7. The retired 3-panel figure's source stays in `figures/figure-2/` (git history +
   possible case-study salvage per ADJ-1) but leaves main.tex.
