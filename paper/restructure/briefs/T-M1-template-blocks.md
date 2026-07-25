# TASK BRIEF T-M1 — make the ACM Reference Format + permissions blocks render

Context: the PI requires the compiled PDF to show the ACM Reference Format block and
the permissions/copyright block (currently suppressed), per the 2026-07-24 meeting
(action A25 in `paper/restructure/meeting/analysis_B_actions.md` §4 — read that
section first). Her instruction: make the blocks APPEAR; do not rewrite boilerplate.
Her own template PDF shows: the "Permission to make digital or hard copies…" block,
"© … Copyright held by the owner/author(s). Publication rights licensed to ACM.",
an ISBN line, and an "ACM Reference Format:" block.

## The edit (in `/home/robert/FAMAIL/paper/main.tex` ONLY)
1. Remove the line `\settopmatter{printacmref=false}`.
2. Replace `\setcopyright{none}` with `\setcopyright{acmlicensed}` (this is what
   renders "Publication rights licensed to ACM", matching her template).
3. KEEP the real venue metadata block (`\acmConference[KDD '27]…`, `\acmYear`,
   `\copyrightyear`) exactly as is, and KEEP `\acmISBN{}`/`\acmDOI{}` empty — the
   deliberate choice recorded in the comment there (no fake identifiers) stands; the
   blocks will render with the real venue line and simply omit ISBN/DOI.
4. Update the comments around those lines so the file records: suppression retired
   2026-07-25 per the PI's template-completeness instruction (Meeting 07-24); the
   `printacmref=false`/`none` pair may return at camera-ready if the venue's final
   instructions differ.

## Gates + verification (run all)
- `cd /home/robert/FAMAIL/paper && latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`
- `pdftotext -f 1 -l 1 main.pdf - | grep -c "ACM Reference Format"` → must be ≥ 1.
- `pdftotext -f 1 -l 1 main.pdf - | grep -c "Permission to make digital"` → ≥ 1.
- `pdftotext -f 1 -l 1 main.pdf - | grep -c "Anonymous Author"` → ≥ 1 (anonymity intact).
- Report where REFERENCES starts now: `for p in 8 9 10; do echo "p$p:"; pdftotext -f $p -l $p main.pdf - | head -3; done`
  (the block costs ~0.2 column on page 1; report the spill, do NOT try to fix it —
  task T9 owns the page budget).

## Rules
- Touch ONLY `paper/main.tex`. The working tree contains an uncommitted edit to
  `paper/sections/04_experiments.tex` that belongs to Robert — do not touch, stage,
  or revert it. Do NOT run `git commit` or `git add`.
- If Write/Edit is blocked by the harness, return the exact replacement lines (old →
  new, with 2 lines of surrounding context each) as text with status
  BLOCKED(write-denied).

## Final reply (machine-read, ≤15 lines)
Status; the diff you made (compact); gate results (latexmk exit, lint output tail);
the three grep counts; the REFERENCES page report; any surprises.
