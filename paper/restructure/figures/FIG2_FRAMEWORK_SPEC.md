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

**Accuracy constraints (binding; from ADJ-2):**
- NEVER draw one shared selection feeding both editors: trim's selection (demand
  deficit attribution) and lift's (supply-gradient attribution on the POST-TRIM state)
  are separate. Two score-map glyphs, sequential order.
- No z-score/normalization vocabulary on scores; labels say "share of the fairness
  deficit" (trim side) and "value of added presence" (lift side).
- No "attack"/"perturbation" vocabulary anywhere.
- Trim edits act in over-served areas; ONLY lift touches the under-served side (by
  adding presence). The output panel must not show demand moved INTO the
  disadvantaged region.
- Caption calls the figure a schematic/framework overview; never "case study".
- Fidelity appears as a constraint annotation ("bounded ε; frozen identity
  discriminator in the objective"), never as an accept/reject gate glyph.

## Recommended layout — Option A "hybrid two-stage" (matches her exemplar + email's
two-stage FATE; adjudicated ADJ-2; ⚖ pending Robert)

```
┌───────────────────────────────────────────────────────────────────────────┐
│ [INPUT band — light blue]                                                 │
│  ┌─────────────────────┐        Raw taxi trajectory corpus                │
│  │ Raw HSTD corpus     │        τ₁ τ₂ … τ_N   (stylized polylines,        │
│  │ τ₁ τ₂ … τ_N         │        N ≈ 95k; advantaged/disadvantaged         │
│  └─────────┬───────────┘        context strip: two tinted blocks)         │
├────────────┼──────────────────────────────────────────────────────────────┤
│ [STAGE 1 band — peach]  Stage 1: Budgeted Fairness-Aware Editing          │
│            │                                                              │
│   ┌────────▼─────────┐   ┌──────────────────┐   ┌─────────────────────┐   │
│   │ deficit map      │──▶│ trim: relocate   │──▶│ presence-value map  │─┐ │
│   │ (share of the    │   │ pickups in over- │   │ on the EDITED corpus│ │ │
│   │ fairness deficit │   │ served areas     │   │ (value of added     │ │ │
│   │ per unit)        │   │ [k_trim selected]│   │ presence per unit)  │ │ │
│   └──────────────────┘   └──────────────────┘   └─────────────────────┘ │ │
│      arrow label:            arrow label:            arrow label:       │ │
│      "attribute"             "bounded edits ≤ ε"     "re-score"         │ │
│   ┌──────────────────────────────────────────────────────────────────┐  │ │
│   │ lift: reroute final seeking states into under-served cells       │◀─┘ │
│   │ [k_lift selected; budget k = k_trim + k_lift ≪ N]                │    │
│   └────────────────────────────┬─────────────────────────────────────┘    │
│   constraint strip (small): "every edit ≤ ε cells · continuity repaired · │
│   frozen driver-identity discriminator in the objective"                  │
├────────────────────────────────┼───────────────────────────────────────────┤
│ [STAGE 2 band — green]  Stage 2: Edit-Aware Weighted Training              │
│   ┌──────────────────┐         ▼                ┌───────────────────────┐  │
│   │ Edited corpus    │──────▶ imitation ──────▶ │ trained policy        │  │
│   │ τ'…(k edited,    │        training          └───────────┬───────────┘  │
│   │ N−k untouched)   │        arrow label:                  │              │
│   └──────────────────┘        "upweight the k edits"        │              │
├───────────────────────────────────────────────────────────────────────────┤
│ TERMINAL CONTRAST:  ┌─────────────────────────┐  ┌───────────────────────┐ │
│  (uniform weights)  │ ✗ bias reproduced       │  │ ✓ fairer service      │ │
│                     │ (edits averaged away)   │  │ allocation            │ │
│                     └─────────────────────────┘  └───────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────┘
```

Reading: the k≪N budget beat lives in the bracketed counts + one bold annotation
("budget k ≪ N: only the trajectories that matter change"). The two score-map glyphs
make per-phase attribution visible without prose. The terminal contrast pair encodes
the vanilla-null vs upweighted result (mirrors her exemplar's Wrong/Correct Label pair)
— that is the honest version of "the payoff lands downstream."

## Option B "three bands" (analysis_C C-7's shape; simpler, taller)

Band 1: Attribute & Trim (deficit map → bounded pickup relocations).
Band 2: Re-score & Lift (presence-value map on edited corpus → tail reroutes).
Band 3: Weight (upweighted imitation → fairer policy) + terminal contrast.
Same glyph and label rules. Choose if Option A renders too wide/busy at \columnwidth.

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
4. Caption (draft, adjust to final layout): "FATE framework. Stage 1 spends a fixed
   edit budget k ≪ N: demand deficit attribution selects trajectories whose pickups
   over-serve advantaged areas and trim relocates those pickups under a bounded offset;
   supply-gradient attribution, computed on the edited corpus, then selects
   trajectories whose final seeking states lift reroutes into under-served cells. All
   edits satisfy spatial bounds, continuity repair, and a frozen driver-identity
   guardrail. Stage 2 upweights the k edited demonstrations during imitation so the
   corpus-level fairness gain survives training; at uniform weights it averages away."
   (Numbers stay out of the caption; k and N appear symbolically.)
5. Gate: renders under latexmk + lint; then send the standalone PDF to Zhang EARLY
   (meeting A5: cheap veto while revert is possible).
6. `\label{fig:overview}` is kept (current §3.4 opener and Fig-1 caption reference it).
7. The retired 3-panel figure's source stays in `figures/figure-2/` (git history +
   possible case-study salvage per ADJ-1) but leaves main.tex.
