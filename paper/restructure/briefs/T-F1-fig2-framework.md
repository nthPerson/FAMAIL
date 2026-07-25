# TASK BRIEF T-F1 — build the Figure-2 framework diagram (TikZ, standalone first)

You are building the single biggest deliverable of the restructure sprint: the new
FATE framework figure. It replaces the current 3-panel stylized-city Figure 2. It will
be emailed to the PI EARLY for a cheap veto, so the standalone render is the product
of this task — integration into the manuscript happens in a later task, not here.

## Read first, in this order
1. `paper/restructure/figures/FIG2_FRAMEWORK_SPEC.md` — your requirements. The
   "ADOPTED LAYOUT — Robert's three-phase design" section IS the layout (ruled
   2026-07-25). The "Accuracy constraints" section is BINDING — violating any bullet
   is a task failure even if the figure is beautiful.
2. `paper/restructure/meeting/fig2_style_screenshot.png` — the style target
   (ST-iFGSM Fig. 3): flat color bands per stage, bold in-band stage titles, gray
   data boxes with τ glyphs, arrows carrying small boxed operation labels, terminal
   contrast boxes. Look at it before writing any TikZ. Ignore the green handwriting
   (live annotation, not content).
3. `paper/figures/figure-2/figure-2.tex` — the OLD figure you are replacing: use it
   only for repo TikZ idiom (how figures here handle sizing, fonts, color
   definitions). Do not reuse its city-map content.
4. `paper/figures/figure-1/figure-1.tex` + `figure-1-test.tex` — the repo's
   standalone-harness pattern and its color vocabulary (edited trajectories are
   dashed orange — your "color A" must match that reading).
5. `paper/restructure/CONTEXT.md` §"Repo ground rules" — era numbers etc. (your
   figure uses only symbols k, N, ε — no numerals).

## Deliverables (all inside `paper/figures/figure-2/` — touch NO other directory)
- `framework.tex` — the figure body: a single `tikzpicture` sized for
  `\columnwidth`, ready to be `\input` inside a `figure` environment later. No
  `\begin{figure}`, no caption (the caption lives in the manuscript; a draft caption
  is in the spec's Implementation plan §4 for your reference of what the labels must
  support).
- `framework-test.tex` — standalone harness (mirror the figure-1-test.tex pattern:
  minimal preamble, article or standalone class, loads tikz + arrows.meta, sets
  \columnwidth-like width ≈ 3.33in, inputs framework.tex).
- `framework-test.pdf` — compiled proof.
- `framework-preview.png` — `pdftoppm -png -r 150 framework-test.pdf framework-preview`
  (single page → framework-preview-1.png is fine; name it clearly).

## Build loop
Compile ONLY the harness, inside `paper/figures/figure-2/`:
`cd /home/robert/FAMAIL/paper/figures/figure-2 && pdflatex -interaction=nonstopmode framework-test.tex`
NEVER compile `paper/main.tex` — another task owns the root build right now, and the
aux files would collide. Iterate: compile → render preview → LOOK at the preview with
the Read tool → fix. Do not declare done without having visually inspected the final
preview yourself (text overlaps, arrow collisions, band overflow are all on you).

## Hard requirements recap (spec is authoritative; these are the ones people miss)
- Three bands top-to-bottom after an input strip: (1) Attribute — trajectories in
  THREE groups colored by attribution outcome (color A = selected/high impact,
  color B = scored/low impact, gray = not selected), with the "edit budget k ≪ N"
  annotation; (2) Trim + Lift — selected trajectories through "gradient ascent on the
  fairness objective", trim glyph (pickup dot relocated ≤ ε) + lift glyph (final tail
  rerouted ≤ ε), constraint strip, output = edited corpus with color-A-DASHED edited
  trajectories; (3) Upweight — edited corpus → imitation training (arrow label
  "upweight the k edited demonstrations") → policy, then the terminal contrast pair
  (✗ uniform weights: bias reproduced / ✓ upweighted: fairer service allocation).
- Color continuity across ALL bands: color A (orange family, matching Fig-1's edited-
  trajectory orange) follows the selected trajectories through the whole figure —
  solid when selected, dashed once edited. Gray = untouched everywhere.
- Band fills: low-saturation light blue (input), peach (band 1... use your judgment:
  the exemplar's band colors are input=light blue, stage bands peach/green — assign
  so adjacent bands differ and color A stays legible on every band).
- Trajectory glyphs: stylized 3-5-segment polylines. NO cars, NO stick figures, NO
  city grid or street map. A small advantaged/disadvantaged tint pair in the input
  strip is optional — include only if it doesn't crowd.
- Group labels avoid per-trajectory-fairness verdicts: "high fairness impact
  (selected, k ≪ N)" / "low impact" / "not selected" — NOT "unfair trajectories".
- No "attack"/"perturbation"/"z-score" vocabulary anywhere. Fidelity appears only in
  the constraint strip as "frozen driver-identity discriminator in the objective".
- Trim acts in over-served areas; only lift touches under-served (by adding
  presence). If your band-2 glyphs gesture at areas at all, respect that.
- Grayscale-safe: dashing, shapes, and text labels carry every distinction; color is
  reinforcement only. All text ≥ \scriptsize. Total height ≤ 0.5\textheight at
  column width. ASCII-safe TikZ source (no unicode in the .tex; τ is `$\tau$`).
- Keep the source READABLE: named colors defined once (`\definecolor`), coordinates
  via named nodes/positioning, comments marking each band. Someone edits this at
  1 a.m. tomorrow.

## Rules
- Do NOT run `git commit` or `git add`. Leave your files in the working tree.
- Do NOT edit any file outside `paper/figures/figure-2/`.
- If the harness blocks your Write/Edit calls, return each file's COMPLETE contents
  in your final reply under clear `=== FILE: path ===` headers with status
  BLOCKED(write-denied); otherwise write the files and keep the reply short.

## Final reply (machine-read)
Status (DONE / DONE_WITH_CONCERNS / BLOCKED + reason); files written; compile result;
preview path; any deviation from the spec and why (one line each); open questions.
Max 20 lines plus any BLOCKED file dumps.
