# TASK BRIEF T6+T7 — Related Work position re-check + Conclusion spine update

Two small, related tasks in one dispatch: §5 Related Work now renders AFTER the
experiments (the file `02_related_work.tex` kept its name — do not rename), and §6
Conclusion needs the new spine's vocabulary. Both files are short and already
well-edited; make the minimum changes that fit them to the restructure.

## Read first
1. `paper/restructure/MEETING_DIGEST.md` decisions #1–#3 (spine, demotion of the
   fidelity trade-off, the C-1 wording rule) + ADJ-3.
2. `paper/sections/02_related_work.tex` and `paper/sections/05_conclusion.tex`.
3. `paper/sections/01_introduction.tex` + `02_overview.tex` + `03_methodology.tex`
   (post-T2/T3/T4) — the vocabulary you must be consistent with (budget-aware
   editing, collective fairness, outcome-side/resource-aware pairing, edit-aware
   weighting, C1–C5 labels).
4. `paper/restructure/ALGORITHM_FACTS.md` §Things-FATE-is-NOT;
   `analysis_C_claims.md` §5 items 1, 2, 8, 11 and D16.

## T6 — `02_related_work.tex` (renders as §5)
1. Position check: the section now FOLLOWS the method and results. Fix any prose
   that assumed it preceded them (e.g., forward-pointing phrasings like "the setting
   this paper requires" are fine; anything saying "below" or promising an upcoming
   definition is not — hunt and adjust tense/direction). Read it start to finish
   with fresh eyes as if you were a reviewer arriving from §4.
2. Vocabulary: one or two touches maximum — e.g., where the FATE-contrast sentences
   say what FATE does, prefer the restructure's terms ("edits a bounded, budgeted
   slice", "collective fairness of the corpus's service allocation") IF a sentence
   is already being touched; do not rewrite sentences solely to inject vocabulary.
3. The leveling-down closer and the (\S\ref{sec:leveling}) pointer stay; verify the
   ref resolves to the post-T4 location and the sentence still reads correctly
   given where that content now lives.
4. Appendix pointer (Appendix~\ref{app:related}) stays.

## T7 — `05_conclusion.tex` (renders as §6)
1. First paragraph: update to the restructure's one-method-two-stages framing —
   FATE = budgeted, attribution-directed editing + edit-aware weighting; collective
   fairness of the service allocation; the C-1 causal chain (the edited slice is
   small, uniform training averages it away, the weighting stage is why the gains
   arrive). Keep its existing factual clauses exactly as they are (never-optimized
   measures improve; the supply channel adds statistically significant taxi
   presence; the control arms confirm the edit itself carries the gains) — reword
   the framing around them only.
2. The bounds paragraph ("The claims come with stated bounds…") survives VERBATIM —
   protected register; do not touch a word of it, including its % src comment.
3. Future-work paragraph: keep the two existing directions; ADD the budget-sweep
   sentence (D16 discharge), e.g.: "Third is the budget itself: k is configured,
   not swept, in this study; characterizing fairness as a function of the edit
   budget is a natural next experiment." (your wording; must plainly say a sweep was
   NOT run here). Also verify \S\ref{sec:editor} and Appendix~\ref{app:results}
   pointers resolve post-T4.
4. Register: conclusion mirrors §1 qualitatively, no numbers except the ones already
   there (p=.00049 line stays with its src comment).

## Gates + checks
- `cd /home/robert/FAMAIL/paper && latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`
- Zero undefined refs; report REFERENCES start page + §6 tail lines.

## Rules
- Touch ONLY `02_related_work.tex` and `05_conclusion.tex`. 04_experiments.tex's
  uncommitted edit stays untouched. No git commands.
- If Write/Edit is blocked: full file contents as text, BLOCKED(write-denied).

## Final reply (machine-read, ≤14 lines)
Status; every sentence you changed in T6 (old → new, compact); the conclusion's new
first-paragraph framing sentences verbatim; the future-work budget sentence
verbatim; gate results; REFERENCES page + §6 tail lines.
