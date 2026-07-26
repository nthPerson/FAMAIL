# FIGURE 1 SPEC — teaser (Dr. Zhang's PNG; optional TikZ remake)

> **v6 (Robert, 2026-07-25, trajectory correlation):** all trajectories
> re-traced onto the map's roads; taxis ON routes, passengers at route
> ENDS; trim = amber arrow from the faded vacated pickup at trajectory
> A1's end to the relocated amber passenger (both advantaged); lift =
> the fig-2-detail composition (anchor node, grayed original tail +
> faded recorded pickup just inside the under-served side, amber dashed
> reroute w/ state dots carrying the pickup AND an amber taxi deeper
> into the district, "+" mark, arrowhead into the amber passenger).
> Demand counts stay 4/4 and 3/3 per panel (the moved lift pickup stays
> WITHIN the under-served district; the adv-side origin alternative is a
> two-macro change, see report v6 note 4). New: \showTailNodes,
> \tzLift* macros, icon-passenger-faint.png. ⚠ the manuscript's
> \Description is now stale — replacement drafted in report §6.
>
> **v5 (Robert, 2026-07-25, legend + fonts):** legend groups now spaced by
> compile-time measurement (`\settowidth` per label, leftover split into
> 5 equal gaps — self-re-spacing on any wording/font/icon change; the
> measuring block must stay OUTSIDE the tikzpicture, where it returns 0);
> titles `\small` 9 pt, side names `\footnotesize` bold 8 pt; bands grown
> (`\tzTitleH` 0.40, `\tzLabelH` 0.94). Measured 239.50 × 126.51 pt.
>
> **v4 (Robert, 2026-07-25, overlay legibility):** middle-right passenger
> (0.920,0.409)→(0.870,0.455) both panels, clear of the arrow shaft;
> semi-transparent white backing behind FATE (`\tzFateBgW/H/Dy/Opacity`,
> 0.94×0.40 cm @ 0.72); arrow shrunk to the backing's length
> (`\tzArrLen` 0.94, heights 0.075/0.14, all macros). Box unchanged.
>
> **v3 (Robert, 2026-07-25, post-integration):** trim channel ON (drawn as
> a relocation of one existing advantaged passenger, counts conserved,
> both endpoints inside the advantaged tint; `\showTrimEditfalse` = back
> to lift-only); Zhang's overlay-arrow composition (panels 0.24 cm apart
> ≈ 1.2 grid cells, FATE label+arrow drawn over the panels' inner edges,
> panels 3.60→4.08 cm; mid-left taxi nudged x 0.086→0.145 clear of the
> arrowhead); ALL fonts one level up (titles/FATE 8 pt, everything else
> 7 pt — the ≥\scriptsize acceptance check now passes everywhere); icons
> 0.50/0.33 cm keep her icon:panel ratio. Measured 239.50 × 123.66 pt.
>
> **v2 REFINEMENTS (Robert, 2026-07-25, same day):** Zhang's own taxi/
> passenger icons adopted (extracted from teaser.png's legend + PIL
> recolor variants, `icon-*.png`); grid ~2.4x finer (19x13); SZ street
> map (`figures/figure-2/SZ_street_background_5x4_rotated.png`) under the
> panels at 0.55 opacity with the existing-trajectory polylines re-traced
> along its roads (mixed 3-10 segment lengths); panel cards ON by default
> (cautionary pink left / calm green right, her in-panel hexes). Measured
> box now 239.50 x 116.83 pt. Each refinement macro-reversible
> (`\useZhangIcons`, `\useMapBackground`, `\showCardTints`, `\tzGridN*`).
>
> **PHASE 2 EXECUTED (2026-07-25).** The TikZ remake is BUILT at
> `paper/figures/figure-1/figure-1-teaser.tex` (harness:
> `figure-1-teaser-test.tex`; previews + email-ready `fig1-for-zhang.png`
> beside it). Measured 239.50 × 115.54 pt — under `\columnwidth` and under
> the 135.0 pt hard cap that SUPERSEDES this spec's stale
> `0.45\textheight` allowance (the PNG renders at 241.15 × 135.0 pt and
> the paper is over the page budget, so taller is not allowed). It is a
> drop-in: `01_introduction.tex` still ships the PNG until Robert rules.
> Full build/deviation/decision record: `figure-1/FIG1_TIKZ_REPORT.md`.
> Spec deviations, macro-reversible (report §5 has the full table):
> - SEMANTIC FIX for FINAL_REVIEW_FINDINGS #1: disadvantaged passengers
>   4/4 in both panels (the PNG drops one), the orange passenger is a
>   recolored in-place disadvantaged passenger (per this spec's own
>   placement, which the PNG deviated from), and the edit is an explicit
>   rerouted polyline (vacated advantaged position → boundary → arriving
>   amber taxi, arrowhead) in fig-2's edit vocabulary.
> - Tints default to fig-2's amber/cobalt (⚖ Conflict A, Robert to rule;
>   `\tzTintScheme{greenpink}` restores Zhang's pair), taxis default
>   neutral charcoal (⚖ Conflict B; `\tzTaxiMode{blue}` restores), outer
>   card tints dropped (`\showCardTintstrue` restores), trim overlay
>   built but OFF (`\showTrimEdittrue`).
> - In-panel label sizes: titles/side names 7/6 pt, Service-Demand and
>   legend 5 pt — the last two are below this spec's ≥`\scriptsize`
>   check, matching the PNG's own effective ~5.5 pt (trade-off stated in
>   the report §2, not silently accepted).

MEETING UPDATE (analysis_B §2, [10:42–11:21], [56:27]): Zhang's version is COMMITTED
with **no modifications requested**, and she ranked TikZ re-implementation "least
priority — after all content is final". So the plan is two-phase:

- **Phase 1 (submission path): use her PNG directly.** Robert drops the file at
  `paper/restructure/zhang/teasing.png`; copy to `paper/figures/figure-1/teaser.png`;
  `\includegraphics[width=\columnwidth]{figures/figure-1/teaser.png}` replaces the
  current TikZ `\input` in 01_introduction.tex. Keep `\label{fig:teaser}`. New caption =
  hers (below) + a one-clause `\Description` rewrite. A Fig-1(raster)/Fig-2(TikZ) style
  mismatch is tolerated per the meeting.
- **Phase 2 (OPTIONAL, only after all content is final): TikZ remake** per the visual
  spec below, for typographic consistency. Do not start this before every Lane-1 task
  and Fig-2 are done.

Caption (hers, adopt): "Collective service disparity emerges from the aggregation of
local trajectories (left). FATE edits a small set of influential trajectories to
improve corpus-level fairness."
Note: the current intro's ¶1 3.0× sentence and the old caption's 20%/11% chip metrics
are teaser-specific prose — the intro rewrite decides what survives around the new
figure (the 3.0× sentence has its own `% src:` and remains truthful; the 20%→11% pair
belonged to the RETIRED TikZ teaser and does not carry into her caption).

## Visual spec (for the OPTIONAL Phase-2 TikZ remake only)

## Layout (single column width, ~2:1.12 aspect)

Two side-by-side map panels + center transition + bottom legend strip.

1. **Left panel — "Biased Service in HSTD"** (title above panel, black text).
   - Background: a stylized street map split by a vertical dashed district boundary.
     Left half tinted pale green (advantaged), right half pale red/pink
     (disadvantaged). Street network: thin gray polylines (existing trajectories double
     as streets — abstract, hand-drawn feel; a TikZ approximation with ~12-18 random
     orthogonal-ish polylines is fine; do NOT try to reproduce the real map).
   - Advantaged (left) half: 6 taxi icons, 3 passenger icons — service-rich, demand-low.
   - Disadvantaged (right) half: 1-2 taxi icons near the top edge, 5 passenger icons —
     service-poor, demand-high.
   - Below panel, two column labels: "**Advantaged**" (dark green bold) with
     "Service: High / Demand: Low"; "**Disadvantaged**" (dark red bold) with
     "Service: Low / Demand: High".
2. **Center transition**: bold "FATE" label above a thick right-pointing gray gradient
   arrow.
3. **Right panel — "FATE for Fairer Service"** (title above panel).
   - Same base map + boundary + tints.
   - Advantaged half: 4 taxi icons (reduced), passengers unchanged.
   - Disadvantaged half: 2-3 taxi icons, one **orange/amber taxi + orange passenger**
     linked by an **orange dashed edited-trajectory polyline** crossing the boundary —
     THE focal element (an edited trajectory now serving the disadvantaged side).
   - Below panel: "**Advantaged**" — "Service: Moderate / Demand: Similar";
     "**Disadvantaged**" — "Service: Increased / Demand: Similar".
4. **Legend strip** (bottom, full width, light gray band): taxi icon "Taxi";
   person icon "Passenger"; solid dark-gray line "Existing trajectory"; orange dashed
   line "Edited trajectory".

## TikZ implementation notes
- Icons: simple geometric taxi (rounded rectangle + cab light) and passenger
  (circle head + body path) as reusable `\pic`s — no external images, no emoji fonts.
  Blue-gray taxis, dark-gray passengers; orange (#E8A33D-ish) for the edited pair.
- Meeting nuance (Zhang): icons may be replaced with stylized trajectories if clearer —
  panels must still read as "many taxis+pickups on one side, few on the other". Keep
  icon count exactly asymmetric; the count asymmetry IS the message.
- Grayscale-safe: the green/red tints must survive grayscale printing via the text
  labels (Advantaged/Disadvantaged) and icon counts, not color alone; the edited
  trajectory is dashed (pattern carries it in grayscale).
- Column width target: \columnwidth, ≤ 0.45\textheight tall including labels.
- File: `paper/figures/figure-1/figure-1-teaser.tex` (new; keep the old figure-1.tex
  in place until the swap commit so history is clean).
- Label stays `fig:teaser`; §1 references it.

## Acceptance checks
- Renders with the paper gates (latexmk + lint) at column width without Overfull.
- Legible at 100% zoom in the compiled PDF (labels ≥ \scriptsize).
- The four Service/Demand label pairs match the source PNG wording exactly.
- Left/right panel taxi-count asymmetry obvious at a glance; edited trajectory is the
  single most salient element of the right panel.
