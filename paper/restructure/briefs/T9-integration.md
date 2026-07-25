# TASK BRIEF T9 — integration pass: appendix check, two-city add, page budget, render QA

The final implementation task before the whole-branch review. Everything upstream
(T1–T7) has landed; your job is coherence and geometry across the whole PDF, plus
two rulings Robert made during the sprint. (The former T8 appendix check is folded in
here as item 1.)

## Read first
1. `paper/restructure/TASK_BOARD.md` — every ⚠ HANDOFF note and ⚖ ruling (T-M1's
   Fig-1 float note is resolved by T2; the T9 rulings below are binding).
2. `paper/restructure/MEETING_DIGEST.md` decision #10 (bar + space levers) and §3.
3. `paper/restructure/ALGORITHM_FACTS.md` (for any wording you touch).

## Work items

1. **Appendix coherence pass** (`paper/sections/appendix.tex`, read start to end):
   every `\S\ref{...}` and `\ref{sec:...}` now resolves against the NEW numbering
   (§2 Overview; §3.1–3.5 blocks; §5 Related Work; §6 Conclusion) — check each
   referrer's SENTENCE still reads correctly, not just that the ref resolves
   (e.g. "deferred from §3.2" style phrasings whose content moved blocks). Fix
   prose-fit only; no content changes. Same one-pass check for
   `02_related_work.tex` (renders §5) if T6 left anything stale — read it once.
2. **Two-city add (⚖ Robert, 07-24)**: in `00_abstract.tex`, the evaluation
   sentence names both cities — smallest change, e.g. "We instantiate FATE for taxi
   mobility and evaluate it on real-world HSTD from Shenzhen and San Francisco."
   Rebuild `kdd27-abstract-only.tex` too (it shares the file; a stale-aux latexmk
   artifact may need its aux state deleted — see T1's note on the board).
3. **Cross-ref + citation sweep**: zero undefined references/citations; zero
   multiply-defined labels; every `\cite` key has a CITATION_PRIORITY_CHECKLIST.md
   row (lint enforces); no `??` anywhere in the rendered PDF (`pdftotext main.pdf -
   | grep -n "??"`).
4. **Page budget**: target is strict 8 pages of content — `pdftotext -f 9 -l 9
   main.pdf -` should open with REFERENCES. Measure the current spill (§6 tail
   lines on p9). If spilled, apply ONLY the meeting-sanctioned levers, in this
   order, re-measuring after each: (a) tighten YOUR OWN generation-era prose first
   (never Robert-approved sentences); (b) cross-city dedup residue in §4.6 that T5
   left; (c) figure/caption geometry (Fig-2's caption can lose its parenthetical
   clause if needed); (d) move the App-E-overflow candidates NONE — appendix content
   stays. If after (a)–(c) the spill remains, STOP and report the residual with a
   lever menu — Robert decides; do not cut protected-register or PI-approved text.
5. **Render QA** (the swallowed-sentence class): read the FULL rendered PDF via
   pdftotext page by page (pp. 1–9 minimum) hunting: sentences that end mid-thought,
   duplicated fragments, stray "%" absorption artifacts, encoding garbage, figure
   captions colliding with body text. Also visually check pp. 1–3 renders
   (`pdftoppm -png -r 110 -f 1 -l 3 main.pdf …` then Read the PNGs): Figure 1 (PNG
   teaser) placement + legibility, Figure 2 (framework) placement + legibility,
   the ACM blocks on p1.
6. **Consistency micro-sweep** (grep-driven, fix only clear inconsistencies):
   "budget-aware" vs "budget aware" (pick hyphenated); "edit-aware weighting"
   spelled consistently; C1–C5 labels referenced consistently (plain text, no
   \ref); k vs K for the budget (paper uses lowercase k — her email's K does NOT
   override); |T| vs N roles (N = active units only); "departure-service ratio"
   (never "demand-service" in the paper); no "attack"/"perturbation" applied to
   FATE's own edits (baseline contexts exempt); no em-dashes in prose added by this
   sprint (pre-existing sanctioned ones stay).

## Gates (final)
`cd /home/robert/FAMAIL/paper && latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`
— exit 0 both; then the abstract-only build; then the page-9 check.

## Rules
- You may touch: `appendix.tex`, `00_abstract.tex`, `02_related_work.tex` (prose-fit
  only), and — for lever (a)–(c) fixes — the specific sentences you name in your
  report. You may NOT touch: numbers, tables, protected-register disclosures,
  Robert's Editor-configuration paragraph edit in 04 (it is committed by now; treat
  as his), the C1–C5 wordings, PI-adopted abstract/intro sentences beyond the
  two-city add.
- No git commands.
- If Write/Edit is blocked: exact edits as text, BLOCKED(write-denied).

## Final reply (machine-read, ≤20 lines)
Status; appendix fixes made (one line each); the two-city sentence as landed; page
budget verdict (REFERENCES page + spill lines before/after; levers applied with
line savings); render-QA findings (or "clean"); consistency fixes; gate results.
