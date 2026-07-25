# paper/restructure/ — coordination hub for the 07-24 Zhang restructure

Read order for any agent joining this effort:

1. `CONTEXT.md` — mission, sources of direction (email < her draft < 07-24 meeting),
   who's who, deadline (draft to Zhang **Saturday night 07-25**), prime directive
   (never misrepresent the algorithm), gates.
2. `ALGORITHM_FACTS.md` — ground truth about FATE. Paper text must not contradict it.
3. `ZHANG_DIRECTIVES.md` — request ledger: every email directive with a disposition
   (fulfill / careful wording / needs-decision / cannot-as-stated).
4. `ZHANG_DRAFT_DELTA.md` — what in her revision PDF is authoritative (abstract, intro,
   Fig-1 concept) vs stale (ALL body sections; ours supersede). Terminology map.
5. `meeting/` — the 07-24 meeting record: `transcript_readable.txt` (authoritative),
   `plaud_summary.md`, `highlights_raw.html`, `fig2_style_screenshot.png` (Figure-2
   style target), and `analysis_A_decisions.md` / `analysis_B_actions.md` /
   `analysis_C_claims.md` / `analysis_D_verification.md` (extraction reports).
6. `MEETING_DIGEST.md` — verified synthesis of the meeting (decisions + action items)
   after the analyst/verifier pass. Written by the orchestrator; trust this over the
   raw analyst reports on conflicts.
7. `TASK_BOARD.md` — the work breakdown, statuses, and dependency lanes. Update your
   task's checkbox + status line when you finish something.
8. `figures/FIG1_TEASER_SPEC.md`, `figures/FIG2_FRAMEWORK_SPEC.md` — figure specs.
9. `zhang/` — source PDFs/PNG from Dr. Zhang (Robert drops these in; may be empty
   early on. The digests in 3–4 are faithful transcriptions in the meantime).

Ground rules for every implementer:
- Gates before any commit: `cd /home/robert/FAMAIL/paper && latexmk -pdf -g
  -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`.
- Moving a text block moves its `% src:` provenance and `% lint-allow:` comments with it.
- Any `\cite`/`refs.bib` change updates `paper/CITATION_PRIORITY_CHECKLIST.md` in the
  same commit (add rows UNTICKED — only Robert ticks).
- Era numbers and protected-register disclosures per `CONTEXT.md`.
- Explicit staging (`git add <files>`, never `-A`); Robert pushes; commit messages
  follow the existing `paper(...)` style.
