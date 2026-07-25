# FIGURE 1 SPEC — teaser (Dr. Zhang's PNG; optional TikZ remake)

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
