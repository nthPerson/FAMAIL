# FIG-1 TikZ RE-IMPLEMENTATION — report (2026-07-25)

Deliverable: `figure-1-teaser.tex` (body) + `figure-1-teaser-test.tex` (harness),
a drop-in TikZ alternative to Dr. Zhang's `teaser.png` (untouched, still the
shipping fallback; `01_introduction.tex` still points at the PNG and builds
unchanged — full gates re-verified below). Nothing in the manuscript was edited.

## 1. Measured box and gates

From the harness (`\newsavebox` measurement, printed by the compiler):

- **Width 239.50285 pt** vs `\columnwidth` = 241.14749 pt (1.64 pt to spare;
  Figure 2 is 239.80 pt, so the two figures are visually the same width).
- **Height 115.54004 pt** vs the 135.0 pt hard cap → **19.46 pt of headroom**
  (the PNG renders at 135.0 pt, so the TikZ version is ~1.7 text lines
  SHORTER — a small page-budget gain if swapped in).
- Harness log: 0 errors, 0 Overfull.

Full manuscript gates (PNG still in place), all green:

```
Pages:           13
overfull>5pt: none
undefined count: 0
fig1-on-p1: 1        (pdftotext p1 finds "Collective service disparity")
```

## 2. Font sizes and legibility call

| Element | Macro | Size |
|---|---|---|
| Panel titles | `\tzTitleFont` | `\scriptsize` (7 pt) |
| Advantaged / Disadvantaged | `\tzSideFont` | `\fontsize{6}{7}` bold (6 pt) |
| Service/Demand lines | `\tzInfoFont` | `\tiny` (5 pt) |
| Legend entries | `\tzLegendFont` | `\tiny` (5 pt) |
| FATE center label | `\tzFateFont` | `\scriptsize` bold (7 pt) |

Legibility at 100 % zoom of the compiled PDF: titles and side labels are
comfortably legible; the 5 pt Service/Demand and legend lines are legible but
small — **the same effective size the PNG itself renders at** (its in-panel
text is ~5.5 pt at column width), so the conversion does not lose ground.
The spec's "labels ≥ `\scriptsize`" acceptance check is met for titles and
side names but **not** for the Service/Demand and legend lines; meeting it
there would require either a taller figure (against the 135 pt cap) or
shorter label text (the PI's wording, not to be improved). Stated trade-off,
not silently shrunk: bump `\tzInfoFont`/`\tzLegendFont` to `\scriptsize` and
the figure still fits the width, but the label band and legend grow ~6 pt
combined — still under 135 pt if Robert prefers that trade.

## 3. §5 invariant checklist (all checked)

Verified two ways: by construction (the coordinate lists in the config block
drive every icon) and by count on the 300 dpi render.

1. ✅ Disadvantaged passengers identical in both panels: `\tzDisPicks` (3) +
   the served passenger (1) are drawn in BOTH panels → 4/4. (The PNG had
   4→3; this is the semantic fix.)
2. ✅ Advantaged passengers identical: `\tzAdvPicks` (3) drawn in both → 3/3.
3. ✅ Conservation: advantaged taxis 5→4 (`\tzAdvTaxisR` = `\tzAdvTaxis`
   minus the vacated entry), disadvantaged 2→3 (+1 amber taxi). One leaves,
   one arrives; corpus total 7 in both panels.
4. ✅ Every amber element on the disadvantaged side is added presence: the
   amber taxi (with a small "+" mark, Figure-2's added-presence glyph), the
   dashed rerouted trajectory, and the amber passenger — which is one of the
   four disadvantaged passengers RECOLORED in place (same coordinates in
   both panels), i.e. "the passenger who now gets served", not moved demand.
5. ✅ Trim overlay (OFF by default): both endpoints at x-fractions
   0.200/0.310, boundary at 0.517 — both inside the advantaged tint.
   Verified in the toggle smoke test render.
6. ✅ Grayscale: `figure-1-teaser-preview-gray.png`. The 5-vs-2 / 4-vs-3
   taxi asymmetry, bold side labels, boundary dash, edit dash + arrowhead
   and "+" mark all survive with the tints gone.
7. ✅ The edited trajectory is the right panel's most salient element: the
   only dashed amber stroke (0.9 pt vs 0.5 pt gray solids), with the only
   arrowhead in either panel, ending at the only amber taxi.

## 4. The semantic fix (FINAL_REVIEW_FINDINGS #1)

Pixel-verified defect in the PNG: orange passenger at panel-fraction 0.479
vs boundary 0.517 (advantaged side), disadvantaged dark passengers 4→3 —
read together, a pickup relocated OUT of the disadvantaged district, the
exact move §3.3 rules out. Fix implemented (per the brief's preferred
option): demand is untouched (4/4 passengers, matching her own "Demand:
Similar" label), the orange passenger is a recolored in-place disadvantaged
passenger, and the edit is drawn as what lift actually does — a rerouted
trajectory from the vacated advantaged taxi position (faint open circle,
Figure-2's vacated-original mark) across the boundary into the arriving
amber taxi (arrowhead). The taxi half of her figure, which was correct, is
preserved exactly.

## 5. Visual deviations from the PNG (each reversible)

Decisions **awaiting Robert** are marked ⚖.

| # | Deviation | Rationale | Reverse with |
|---|---|---|---|
| D1 | Semantic fix above (passenger counts, orange pair meaning, explicit rerouted polyline + ghost) | §3.3 / ALGORITHM_FACTS: only lift touches the under-served side, by adding presence; prime directive beats PNG fidelity | Not reversible by macro by design — reverting reinstates the perverse edit. `\showVacatedGhostfalse` hides the ghost; `\showAddedMarkfalse` hides the "+" |
| D2 ⚖ | **Conflict A — area tints**: amber/cobalt (`tzsel!14`/`tzlow!12`, Figure-2's area grammar) instead of her green/pink | Figure consistency; deut/prot CVD safety; green=advantaged / red=disadvantaged reads as a value judgment about neighborhoods | `\def\tzTintScheme{greenpink}` (uses her measured hexes #F0F8F1/#FBEFEF) |
| D3 ⚖ | **Conflict B — taxi color**: neutral charcoal `tzink` instead of her blue | In Figure 2, cobalt means "scored, low fairness impact"; gray=untouched, amber=edited is the shared grammar | `\def\tzTaxiMode{blue}` (her measured #639CE2) |
| D4 | Outer card tints dropped (left card pink / right card green in the PNG) | Duplicate color coding; the titles already name the worlds; frees contrast for the tints that carry meaning | `\showCardTintstrue` (her measured #FEF5F2/#F7F9F6) |
| D5 | Side-label colors follow the tint scheme (dark amber / dark cobalt) instead of her dark green / dark red | Labels reinforce the area coding instead of introducing a second good/bad axis; her colors return with `greenpink` | automatic with `\tzTintScheme`; or override `tzAdvLabel`/`tzDisLabel` colorlets |
| D6 | Edit color is Figure-2 amber #D97706, not her orange #F7BB07 | Same object, same color across adjacent figures | `\colorlet{tzEdit}{tzZedit}` (one line, noted in the file) |
| D7 | Streets drawn (faint lattice + 14 gray polylines), not her rasterized map | Spec forbids reproducing the real map; raster prints muddy at this size | `\useRasterBackgroundtrue` (swaps in `corpus_background_raw.png`) |
| D8 | Orange pair moved from ~(0.61,0.39) to (0.63,0.55)/(0.74,0.59) | The PNG position collides with a dark passenger once the pair is enlarged for print; open lower-middle keeps the focal edit clean; served passenger sits at the SAME spot in both panels | `\tzOrangeTaxiX/Y`, `\tzServedX/Y` |
| D9 | "+" added-presence mark beside the amber taxi (not in the PNG) | Figure-2's grayscale carrier for "added presence"; kills any residual "moved demand" reading | `\showAddedMarkfalse` |
| D10 | FATE label bold (PNG regular) | Salience of the one named actor in the figure | `\tzFateFont` |
| D11 | Trim-channel overlay exists but is OFF | One clean channel beats two crowded ones; overlay verified correct (both endpoints advantaged) | `\showTrimEdittrue` |

All toggle paths compile-tested (Zhang look = D2+D3+D4 flipped together;
trim overlay; raster background): 0 errors each, renders inspected.

## 6. Integration snippet (for the OTHER session to apply — I did not touch
`sections/01_introduction.tex`)

In `paper/sections/01_introduction.tex`, replace

```latex
  \includegraphics[width=\columnwidth]{figures/figure-1/teaser.png}
```

with

```latex
  \input{figures/figure-1/figure-1-teaser}
```

Nothing else is required: `main.tex` already loads `tikz` + `arrows.meta`
(the only libraries used), the body ends in `%`, and `\figonedir` is
`\providecommand`-defaulted for the manuscript path. The comment block above
the `\includegraphics` (raster adoption, style-mismatch note) should be
updated or dropped by whoever applies the swap. Note the figure is ~19.5 pt
shorter than the PNG, so page breaks can shift: re-check `Pages: 13` and
Fig-1-on-p1 after the swap. The `\Description` MUST be replaced at the same
time (the current one is wrong for the PNG and wrong for this figure):

```latex
  \Description{Two stylized city maps side by side, each split by a vertical
  dashed district boundary into an advantaged half and a disadvantaged half,
  drawn over a faint street grid with dark gray polylines standing in for
  existing trajectories. In the left map, titled Biased Service in HSTD, the
  advantaged half holds five taxis and three passengers and is labeled
  Service: High, Demand: Low, while the disadvantaged half holds two taxis
  and four passengers and is labeled Service: Low, Demand: High. A thick
  arrow labeled FATE leads to the right map, titled FATE for Fairer Service.
  There the advantaged half holds four taxis, and a dashed orange line
  starts at the vacated position of the missing taxi, crosses the district
  boundary, and ends with an arrowhead at an orange taxi on the
  disadvantaged side, beside a small plus sign marking added service
  presence. The passenger nearest the arriving taxi is drawn in orange to
  mark the passenger who now gets served; passenger counts are unchanged on
  both sides. The halves are labeled Service: Moderate, Demand: Similar and
  Service: Increased, Demand: Similar. A legend identifies the taxi and
  passenger icons, the solid gray existing trajectories, and the dashed
  orange edited trajectory.}
```

(If Robert instead ships the PNG, the `\Description` repair decided under
finding #1 still applies and this draft does NOT fit the PNG — the PNG shows
a different, defective scene.)

## 7. Files

- `figure-1-teaser.tex` — figure body (config block drives everything)
- `figure-1-teaser-test.tex` — harness with compiler-checked size gates
- `figure-1-teaser-preview.png` / `figure-1-teaser-preview-gray.png` —
  150 dpi renders (color + grayscale)
- `fig1-for-zhang.png` — 300 dpi figure+caption crop, email-ready (mirrors
  `figure-2/fig2-for-zhang.png`)

## 8. Not done / notes

- The trim overlay defaults OFF (D11); nobody has ruled on showing two
  channels in the teaser.
- Zero use of new packages or TikZ libraries; nothing added to `main.tex`.
- `teaser.png` is byte-identical (conversion reference + fallback + the
  still-live one-icon PIL nudge plan B for finding #1).
- The spec's `0.45\textheight` height allowance is stale; the harness
  encodes the 135.0 pt hard cap instead (documented there).
