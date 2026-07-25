# TASK BRIEF T3 — create §2 Overview (problem formulation moves here + the challenge list)

You are creating the paper's new §2 "Overview" per the PI's restructure: the current
§3.1 Problem Formulation moves here largely as-is, followed by the five labeled
challenges. This is the section every later section leans on, so labels and challenge
wording matter.

## Read first
1. `paper/restructure/MEETING_DIGEST.md` — decisions #4, #5, #6 and ADJ-3 (the
   five-challenge mapping, RULED).
2. `paper/sections/03_methodology.tex` — the current §3.1 block (from
   `\subsection{Problem Formulation}` through the end of the Task paragraph, before
   `\subsection{The Fairness Objective}`): this text MOVES.
3. `paper/sections/01_introduction.tex` lines 110–132 — the current C1–C5 itemize:
   your challenge-content source material (it gets REMOVED from the intro later, in
   T2 — not by you).
4. `paper/restructure/ALGORITHM_FACTS.md` §Setting + §Things-FATE-is-NOT.
5. `paper/restructure/meeting/analysis_C_claims.md` §5 do-not-claim list.

## Deliverables

### A. New file `paper/sections/02_overview.tex`
Structure (per meeting [31:30]: definitions → problem definition → challenges):

1. `\section{Overview}\label{sec:overview}`
2. **Definitions** — the moved §3.1 content, reorganized under the email's three
   beats but WITHOUT subsection headings (bold lead-ins or plain paragraphs, your
   judgment): (a) the data and its representation — corpus T of real per-driver
   trajectories, seeking states ending in a pickup; (b) service allocation the data
   induces — active units, demand D, supply S, service ratio Y with the demand
   floor, demographics x_i at district granularity + its resolution caveat; (c) keep
   every formula, number, and `% src:` comment from the moved text intact.
   **Boldface each term at its definition** (cGAIL style): \textbf{active unit},
   \textbf{demand}, \textbf{supply}, \textbf{service ratio}, the corpus/trajectory
   terms — bold ONCE, at definition, sparingly elsewhere. The existing `\emph{}` on
   defined terms upgrades to `\textbf{}`.
   Adopt the abstract's HSTD framing in ONE opening clause (the corpus is
   human-generated spatial-temporal data) — do not force the acronym into every
   paragraph.
3. **Problem definition** — the moved "Task." paragraph, kept intact including the
   data-augmentation positioning sentence, the k/ε budget sentence, and the
   downstream pointer. Bold lead-in \textbf{Problem definition.} (replacing
   \textbf{Task.}) is fine.
4. **Challenges** — five, per ADJ-3 (RULED). Format: ONE paragraph block, NO itemize
   environment (meeting [32:12]), each challenge a bold inline label + 1–2 sentences,
   stacked: `\textbf{C1 (budget).} … \textbf{C2 (fidelity).} …` etc. Draft wording
   from these skeletons (source material = the old intro items; tighten freely):
   - **C1 (budget)** [REWRITTEN from old "scarcity"]: real demonstrations are
     irreplaceable, so only a small share of the corpus may change; the difficulty is
     spending a budget of k ≪ N edits where they move COLLECTIVE fairness most.
     (This is the meeting's "unique challenge" [06:02] — how can a small portion of
     edited trajectories move the fairness of the whole corpus.)
   - **C2 (fidelity)**: old C2 content — edits must remain real driver behavior;
     wording must NOT say the discriminator "gates" each edit (C D15): bounded edits
     + a frozen driver-identity guardrail.
   - **C3 (the wrong target)**: old C3 — equal service is the wrong target; the
     objective must score only the service variation demand does not explain.
   - **C4 (level up, not down)**: old C4 — measured fairness can improve while the
     under-served gain nothing; the editor needs a channel that adds real presence.
   - **C5 (surviving training)**: old C5 — sparse edits are averaged away by uniform
     training; their influence must be preserved.
   One forward-pointer clause per challenge at most (e.g. "(\S\ref{sec:objective})").
   Keep the old intro items' section refs where they still make sense.
   Order C1→C5 as above (budget first — it is the spine).
5. Challenges are referenced from later sections as plain text ("C1", "C4") — no
   \label machinery needed for them.

### B. Edit `paper/sections/03_methodology.tex`
Remove the moved §3.1 block (subsection heading through the Task paragraph). Leave
`\section{Methodology}\label{sec:method}` in place with a one-line TODO comment
`% T4 rewrites this opening: leading paragraph (FATE, two stages, Fig 2 ref, challenge map)`
so the next task knows. Do NOT renumber or rewrite anything else in 03 — the
remaining subsections keep their labels; LaTeX renumbers automatically.
⚠ If the moved block's labels (e.g. the subsection's `\label{...}` — check what it
is) are referenced elsewhere (`grep -rn "ref{<label>" paper/sections paper/main.tex`),
the label MOVES with the content into 02_overview.tex so refs keep resolving.

### C. Edit `paper/main.tex` (two lines)
- Insert `\input{sections/02_overview}` after the 01_introduction input.
- MOVE the `\input{sections/02_related_work}` line to after 04_experiments (Related
  Work becomes §5; the FILE KEEPS ITS NAME — renames are deliberately out of scope).
- Result order: 01_introduction · 02_overview · 03_methodology · 04_experiments ·
  02_related_work · 05_conclusion.

## Known-acceptable transient state
Until T2 rewrites the intro, the old C1–C5 itemize still sits in §1 while your new
challenge block sits in §2 — duplicated content, compiles fine, expected. Also §2
Related Work's text says things like "(\S\ref{sec:leveling})" — those refs must still
RESOLVE (check the log for undefined refs) but their prose fit is T6's job.

## Gates + checks
- `cd /home/robert/FAMAIL/paper && latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`
- `grep -c "LaTeX Warning: Reference" main.log` → report count (must not INCREASE vs
  a pre-edit baseline build you run first; zero undefined refs after).
- `pdftotext main.pdf - | grep -n "Overview"` — §2 exists where expected.
- Report where REFERENCES starts (p9 expected; report only).

## Rules
- Files you may touch: `02_overview.tex` (create), `03_methodology.tex` (the removal
  only), `main.tex` (the two input lines only). NOTHING else. Robert's uncommitted
  edit in `04_experiments.tex` stays untouched. No git commands.
- Moved text keeps its `% src:` comments attached to the sentences they source.
- If Write/Edit is blocked, return complete file contents / exact edits as text,
  status BLOCKED(write-denied).

## Final reply (machine-read, ≤18 lines)
Status; the five challenge sentences you drafted (verbatim — Robert reviews these);
what label the moved subsection carried and where its refs point now; gate results;
undefined-ref count before/after; REFERENCES page.
