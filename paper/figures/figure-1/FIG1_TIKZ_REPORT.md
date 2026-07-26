# FIG-1 TikZ RE-IMPLEMENTATION — report (2026-07-25, v6)

**v6 (Robert's trajectory-correlation pass, 2026-07-25):**
1. **Road-true trajectories.** All 13 trajectories re-traced so every
   waypoint sits on a road of the underlying SZ map (left horizontal road,
   top-left/mid-left blocks, SW diagonal, S-curve artery, right vertical
   artery, NE diagonal, bottom-right blocks, center artery + 4 background
   lines).
2. **Icons anchored to trajectories.** Taxis sit ON intermediate waypoints
   (horizontal segments preferred, so cars read as driving the road);
   passengers stand at trajectory ENDS (every trajectory ends in a
   pickup). The icon lists in the config block literally repeat waypoints
   from the trajectory paths, with comments naming each host (A1, A2, A4,
   A5 = advantaged; D1, D2, D3 = disadvantaged; B* = iconless texture).
3. **Trim redrawn**: the pickup at trajectory A1's end — dark in the left
   panel — is vacated (FAINT passenger, `icon-passenger-faint.png`) in the
   right panel, with the amber dashed arrow from that position to the
   amber passenger at the new location. Both endpoints inside the
   advantaged tint.
4. **Lift redrawn as the fig-2-detail composition** ("more realistic
   trajectory edit"): host trajectory T_L runs down the S-curve artery
   with its taxi ON the route and its recorded pickup at the end, just
   inside the under-served side. In the right panel the tail branches at
   the ANCHOR (dark node dot — the state that never moves): the original
   tail grays out (faint line + faint state dots + faint pickup), and the
   amber dashed reroute (with amber state dots) carries the pickup AND
   the taxi deeper into the under-served side — amber taxi ON the dashed
   line past the boundary, "+" mark, arrowhead into the amber passenger.
   "The real Lift edit moves the pickup, too" (Robert; ALGORITHM_FACTS:
   pickup moves by the full offset δ).
   **Design decision flagged:** the recorded pickup was placed just
   INSIDE the disadvantaged half so the moved pickup travels WITHIN the
   under-served district (deeper, toward the high-value interior). This
   keeps the per-panel demand counts identical (4/4) and keeps any
   demand-out-of-district reading impossible; presence still visibly
   flows advantaged→disadvantaged via the vacated-taxi ghost and the
   amber taxi. Placing the recorded pickup on the ADVANTAGED side
   instead would show demand crossing into the district (also a true
   lift outcome) at the cost of the count invariant — two-macro change
   (`\tzLiftPickOrig*`) if preferred.
5. New knobs: `\showTailNodes` (state dots), `\tzTailDotR`, the lift
   path/anchor/taxi/pickup macros (`\tzLift*`), `icon-passenger-faint.png`
   (PIL alpha ×0.32). Box unchanged: 239.50 × 126.51 pt.

**v5 (Robert's legend + font pass, 2026-07-25):**
1. **Legend evenly spaced, by computation.** The four icon+label group
   widths are measured at compile time (`\settowidth` in the actual
   legend font, in the config block — measuring inside the tikzpicture
   silently returns 0) and the leftover band width is split into 5 EQUAL
   gaps (both end margins + three between-group gaps, 0.486 cm each at
   current settings). Editing a label string, the legend font, an icon
   size, `\tzLegSample` or `\tzLegPad` re-spaces the band automatically.
   Render-verified: inter-group ink gaps 59/56/55 px at 300 dpi (±2 px
   is antialiasing + the icons' transparent padding).
2. **Fonts up one more level**: panel titles `\small` (9 pt), side names
   `\footnotesize` bold (8 pt). Bands grown to fit: `\tzTitleH` 0.40,
   `\tzLabelH` 0.94, info-line offset −0.37. Info/legend stay 7 pt, FATE
   8 pt bold (Robert's hand-tuned v4 overlay values preserved: backing
   0.78 × 0.33, arrow 0.70).
3. Measured box **239.50 × 126.51 pt** (width unchanged; height +2.8 pt,
   still 8.5 pt under the 135 pt cap and ~0.7 line shorter than the PNG).

**v4 (Robert's overlay-legibility pass, 2026-07-25):**
1. **Icons cleared from the overlay zone.** The middle-right disadvantaged
   passenger moved from the PNG's (0.920,0.409) to (0.870,0.455) in BOTH
   panels — he stood with his feet under the arrow shaft. (The v3 taxi
   nudge to x=0.145 plus the smaller v4 arrow clears the right panel's
   side; no other icon or trajectory touched the zone.)
2. **Semi-transparent backing behind FATE**: a white rounded rectangle at
   `\tzFateBgOpacity` (default 0.72) drawn under the label, so panel
   content shows through washed out. Fully tinker-friendly:
   `\tzFateBgW`/`\tzFateBgH` (0.94 × 0.40 cm) and `\tzFateBgDy` (center
   height above the arrow axis). TikZ cannot blur, so semi-transparency
   is the whole effect; opacity 1 gives a solid plate.
3. **Arrow shrunk to the backing's length**: total length `\tzArrLen` =
   0.94 cm = `\tzFateBgW` (was 1.10), shaft/head half-heights
   `\tzArrShaftH` 0.075 / `\tzArrHeadH` 0.14 (was 0.11/0.20), head length
   `\tzArrHeadL` 0.26 — every dimension a macro. Box unchanged:
   239.50 × 123.66 pt.

**v3 (Robert's three post-integration changes, 2026-07-25):**
1. **Trim channel shown** (`\showTrimEdittrue`, now the default). Drawn so
   the invariants survive: trim takes the RIGHT MEMBER of the advantaged
   passenger pair (`\tzTrimFrom` = the PNG's (0.098,0.615) passenger, dark
   and unmoved in the left panel) and relocates it to `\tzTrimTo`
   (0.245,0.775) — ghost circle at the recorded position, dashed amber
   arrow, amber passenger at the new position, BOTH endpoints inside the
   advantaged tint. Advantaged passenger count stays 3 in both panels
   (moved, not added). Confusion judgment: low — the two channels use the
   same ghost→dashed→amber vocabulary and each stays in its lane (trim
   entirely inside the advantaged half, lift crossing into the
   disadvantaged half); the lift path remains the dominant element.
   `\showTrimEditfalse` returns to the lift-only teaser.
2. **Zhang's overlay-arrow composition**: the panels now sit `\tzGap` =
   0.24 cm apart (≈1.2 grid cells) and the FATE label + block arrow are
   drawn AFTER the panels, overlaying their inner edges as in her PNG.
   Panels grew 3.60 → 4.08 cm wide (+13%); all icon positions are
   panel-fractions, so the layout rescaled automatically. The one manual
   consequence: the mid-left advantaged taxi moved from the PNG's
   x=0.086 to x=0.145 in BOTH panels so the arrowhead lands clear of it.
3. **Fonts one level up**: titles `\footnotesize` (8 pt), side names
   `\scriptsize` bold (7 pt), Service/Demand + legend `\scriptsize`
   (7 pt), FATE `\footnotesize` bold (8 pt). The spec's ≥`\scriptsize`
   acceptance check is now met by EVERY text element (v1/v2's 5-6 pt
   shortfall is gone). Icons bumped to `\tzTaxiW`=0.50 / `\tzPickH`=0.33
   to hold Zhang's icon:panel ratio. Bands grew to fit (title 0.36,
   label 0.88, legend 0.44 cm): measured box now
   **239.50 × 123.66 pt** — still 11.3 pt under the 135 pt cap.

Deliverable: `figure-1-teaser.tex` (body) + `figure-1-teaser-test.tex` (harness),
a drop-in TikZ alternative to Dr. Zhang's `teaser.png` (untouched, still the
shipping fallback; `01_introduction.tex` still points at the PNG and builds
unchanged — full gates re-verified below). Nothing in the manuscript was edited.

**v2 (Robert's four refinements, 2026-07-25):**
1. **Zhang's own icons adopted.** Her taxi and passenger glyphs were extracted
   from the teaser.png legend band (clean background there) with PIL
   flood-fill alpha, and recolored variants generated by hue-remapping the
   blue paint: `icon-taxi-{neutral,blue,amber}.png`,
   `icon-passenger{,-amber}.png` (blue + dark = her originals untouched).
   Default taxi is the neutral recolor (Conflict B still stands; the `blue`
   mode now uses her literal icon). `\useZhangIconsfalse` restores the drawn
   fallback glyphs. No web sourcing was needed.
2. **Grid ~2.4× finer** (19×13 lattice vs the old 8×6), drawn at reduced
   weight/opacity so it reads as cell structure without competing with the map.
3. **SZ street map** (`figures/figure-2/SZ_street_background_5x4_rotated.png`,
   byte-identical to the `paper-pre-rewrite-2026-07-23` copy Robert named)
   now sits under each panel at `\tzMapOpacity` (0.55), included at panel
   width and clipped (equivalent to Robert's cut-in-half framing: the left/
   right halves of the map land in the advantaged/disadvantaged panes).
   The 16 existing-trajectory polylines were re-traced to follow the map's
   visible roads (S-curve artery, right vertical artery, the y≈0.42
   horizontal, NE + SW diagonals, block grids) with rounded corners and
   mixed lengths: short 3–4 segments, medium 5–6, long 8–10, segment length
   ≈ one grid cell. Toggle: `\useMapBackgroundfalse` = tints+grid only.
4. **Panel cards ON by default** (this supersedes v1's D4-off default):
   cautionary pale pink behind the biased panel, calm pale green behind the
   FATE panel. Her outer-card hexes #FEF5F2/#F7F9F6 were too faint to read
   as a signal, so the cards use her stronger in-panel tint hexes
   #FBEFEF/#F0F8F1 (still colors measured from her PNG).
   `\showCardTintsfalse` removes them.

## 1. Measured box and gates

From the harness (`\newsavebox` measurement, printed by the compiler):

- **Width 239.50285 pt** vs `\columnwidth` = 241.14749 pt (1.64 pt to spare;
  Figure 2 is 239.80 pt, so the two figures are visually the same width).
- **Height 126.50525 pt** (v5) vs the 135.0 pt hard cap → **8.49 pt of
  headroom** (the PNG renders at 135.0 pt, so the TikZ version is still
  ~0.7 text line SHORTER than the PNG despite two rounds of font bumps).
- Harness log: 0 errors, 0 Overfull.

Full manuscript gates (PNG still in place), all green:

```
Pages:           13
overfull>5pt: none
undefined count: 0
fig1-on-p1: 1        (pdftotext p1 finds "Collective service disparity")
```

## 2. Font sizes and legibility call

v5 sizes (titles + side names one further level up per Robert):

| Element | Macro | Size |
|---|---|---|
| Panel titles | `\tzTitleFont` | `\small` (9 pt) |
| Advantaged / Disadvantaged | `\tzSideFont` | `\footnotesize` bold (8 pt) |
| Service/Demand lines | `\tzInfoFont` | `\scriptsize` (7 pt) |
| Legend entries | `\tzLegendFont` | `\scriptsize` (7 pt) |
| FATE center label | `\tzFateFont` | `\footnotesize` bold (8 pt) |

Legibility at 100 % zoom: everything is comfortably legible; every text
element now meets the spec's ≥`\scriptsize` acceptance check (the v1/v2
5–6 pt shortfall is resolved). "Service: Increased", the longest info
line, still sits fully inside its half-panel column at 7 pt.

## 3. §5 invariant checklist (all checked)

Verified two ways: by construction (the coordinate lists in the config block
drive every icon) and by count on the 300 dpi render.

(v6 wording — the lift now shows the moved pickup, per Robert)

1. ✅ Disadvantaged passengers identical in both panels: left = D1/D2/D3
   ends (3 dark) + the lift's recorded pickup (dark, just inside the
   district) = 4; right = the same 3 dark + the moved pickup in amber
   (the faint vacated marker does not count) = 4. Demand never leaves the
   district — the lift pickup moves DEEPER into it.
2. ✅ Advantaged passengers identical: left = A4/A5 ends (2) + the trim
   subject dark at A1's end = 3; right = 2 kept + the SAME passenger
   relocated in amber = 3. Moved, not added.
3. ✅ Conservation: advantaged taxis 5→4 (`\tzAdvTaxisR` = `\tzAdvTaxis`
   minus the lift driver), disadvantaged 2→3 (+1 amber taxi ON the
   reroute). One leaves, one arrives; corpus total 7 in both panels.
4. ✅ Amber on the disadvantaged side = the lift edit only: added presence
   (amber taxi + "+" mark) and the lift-moved pickup, which RELOCATES
   WITHIN the under-served district (v6 revision of the old "never
   relocated demand" wording — the real lift moves the pickup by the full
   offset δ; the forbidden direction, demand moved OUT of the district,
   remains impossible to read because the vacated marker sits inside the
   district too). The trim channel's amber passenger sits on the
   ADVANTAGED side.
5. ✅ Trim edit: recorded position x=0.270 (A1's end), relocated position
   x=0.135, arrowhead x=0.158 — all left of the 0.517 boundary, inside
   the advantaged tint. Verified in source and render.
6. ✅ Grayscale: `figure-1-teaser-preview-gray.png`. The 5-vs-2 / 4-vs-3
   taxi asymmetry, bold side labels, boundary dash, both edit dashes +
   arrowheads, the faint-vs-dark vacated marks and the "+" all survive
   with the tints gone.
7. ✅ The edit vocabulary is the right panel's most salient element class;
   within it the lift dominates (branch node, grayed tail, longest dashed
   stroke crossing the boundary, amber taxi + "+", arrowhead into the
   amber passenger). The trim arrow is a shorter parallel statement fully
   inside the advantaged half.

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
| D4 | ~~Outer card tints dropped~~ **superseded in v2 (Robert 2026-07-25): cards ON by default**, using her stronger in-panel hexes #FBEFEF/#F0F8F1 because her card hexes are near-invisible | cautionary-left / calm-right signal Robert asked for | `\showCardTintsfalse` removes; hexes in the config block |
| D5 | Side-label colors follow the tint scheme (dark amber / dark cobalt) instead of her dark green / dark red | Labels reinforce the area coding instead of introducing a second good/bad axis; her colors return with `greenpink` | automatic with `\tzTintScheme`; or override `tzAdvLabel`/`tzDisLabel` colorlets |
| D6 | Edit color is Figure-2 amber #D97706, not her orange #F7BB07 | Same object, same color across adjacent figures | `\colorlet{tzEdit}{tzZedit}` (one line, noted in the file) |
| D7 | v2 (Robert 2026-07-25): SZ street-map render under drawn street-following trajectories + a 19×13 grid | trajectories read as real driving paths; map at 0.55 opacity keeps tints/icons in front | `\useMapBackgroundfalse` = tints+grid+trajectories only; `\tzMapFile`/`\tzMapOpacity` |
| D8 | Orange pair moved from ~(0.61,0.39) to (0.63,0.55)/(0.74,0.59) | The PNG position collides with a dark passenger once the pair is enlarged for print; open lower-middle keeps the focal edit clean; served passenger sits at the SAME spot in both panels | `\tzOrangeTaxiX/Y`, `\tzServedX/Y` |
| D9 | "+" added-presence mark beside the amber taxi (not in the PNG) | Figure-2's grayscale carrier for "added presence"; kills any residual "moved demand" reading | `\showAddedMarkfalse` |
| D10 | FATE label bold (PNG regular) | Salience of the one named actor in the figure | `\tzFateFont` |
| D11 | v3 (Robert 2026-07-25): trim-channel overlay ON — both edit channels shown, method depicted in full | drawn as a relocation of an existing advantaged passenger (counts conserved); low confusion risk (see v3 note 1) | `\showTrimEditfalse` returns to lift-only |
| D12 | v3: mid-left advantaged taxi at x=0.145 (PNG: 0.086), both panels | keeps the overlaid FATE arrowhead from landing on it | `\tzAdvTaxis`/`\tzAdvTaxisR` lists |

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
  dashed district boundary into an advantaged half and a disadvantaged
  half, drawn over a faint street map. Solid gray polylines trace existing
  taxi trajectories along the streets; taxi icons sit on the routes and a
  passenger icon stands at the end of each route, marking its pickup. In
  the left map, titled Biased Service in HSTD, the advantaged half holds
  five taxis and three passengers and is labeled Service: High, Demand:
  Low, while the disadvantaged half holds two taxis and four passengers
  and is labeled Service: Low, Demand: High. A thick arrow labeled FATE
  leads to the right map, titled FATE for Fairer Service, where two edits
  are drawn in orange. In the advantaged half, a dashed orange arrow leads
  from a faded passenger at the end of one route to an orange passenger at
  a new location, showing a recorded pickup relocated within the
  advantaged district. On a long route that ends just inside the
  disadvantaged district, the final segments fade out and a dashed orange
  line branches from a marked point on the route, crosses the boundary
  carrying an orange taxi and a small plus sign, and ends with an
  arrowhead at an orange passenger deeper in the district, showing the
  rerouted trajectory that moves taxi presence and its pickup further into
  the under-served district. Passenger counts are unchanged in both
  halves, which are labeled Service: Moderate, Demand: Similar and
  Service: Increased, Demand: Similar. A legend identifies the taxi and
  passenger icons, the solid gray existing trajectories, and the dashed
  orange edited trajectory.}
```

(v6: the manuscript currently carries the v1-era `\Description`; it no
longer matches the figure and NEEDS this replacement.)

(If Robert instead ships the PNG, the `\Description` repair decided under
finding #1 still applies and this draft does NOT fit the PNG — the PNG shows
a different, defective scene.)

## 7. Files

- `figure-1-teaser.tex` — figure body (config block drives everything)
- `icon-taxi-{neutral,blue,amber}.png`, `icon-passenger{,-amber}.png` —
  Zhang's icons (extracted from her teaser.png legend band) + PIL-recolored
  variants; `blue` taxi and dark passenger are her originals untouched
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
