# RESTRUCTURE CONTEXT — read this first

**Mission.** Dr. Xin Zhang (PI, she/her) has directed a restructuring of the FATE KDD 2027
manuscript (repo: `paper/`, sections in `paper/sections/*.tex`). The new spine is
**budget-aware trajectory editing for collective fairness** — replacing the current
fairness-vs-fidelity framing. Sources of direction, in order of recency (later wins on
conflict):

1. Zhang's restructuring email (digest below; PDF: `zhang/Zhang_restructuring_email.pdf`
   if Robert has dropped it in — otherwise the digest in `ZHANG_DIRECTIVES.md` is the
   reference).
2. Zhang's revised draft PDF (`zhang/Zhang_paper_revision.pdf` when available): her new
   title/abstract/introduction/Figure-1 grafted onto an OLDER snapshot of our draft.
   See `ZHANG_DRAFT_DELTA.md` for what is authoritative vs stale in it.
3. **The 2026-07-24 meeting (Robert ↔ Zhang, 62 min) — most recent and most binding.**
   Transcript: `meeting/transcript_readable.txt` (`[mm:ss] Speaker:` format).
   Plaud auto-summary: `meeting/plaud_summary.md`. Highlights JSON (5 marked moments):
   `meeting/highlights_raw.html`. Figure-2 style target screenshot (ST-iFGSM Fig. 3):
   `meeting/fig2_style_screenshot.png`.

**Who's who in the transcript.** `Speaker 3` = Robert Ashe (RA, SDSU; the paper's
day-to-day author). `Dr. Xin Zhang` = PI. `Speaker 1` = stray mislabeled fragments
(156 chars total; ignore). Plaud speaker labels are auto-assigned; judge by content.
The auto-summary sometimes writes "FADE" — transcription artifact for **FATE**.

**Deadline (updated in the meeting).** Robert delivers the restructured draft + figures
to Zhang by **Saturday night 2026-07-25**; Zhang does the final pass and submits
**Sunday night 2026-07-26 (AoE)**. Effective working window: ~24 hours.

**Prime directive from Robert (binding on every agent).** Dr. Zhang does not have a
perfect understanding of the FATE algorithm. **Never misrepresent the actual approach to
satisfy one of her requests.** She is explicitly OK with requests going unfulfilled when
fulfilling them would misrepresent the work. Ground truth for what FATE actually does:
`ALGORITHM_FACTS.md` (verified against `paper/sections/03_methodology.tex` and
`paper/sections/appendix.tex`). Track every such conflict in
`ZHANG_DIRECTIVES.md` → request ledger.

**Paper state (commit `272bb47`, tree clean).**
- Title is ALREADY Zhang's: "Mitigating Demonstration Bias via Fairness-Aware
  Trajectory Editing" (adopted 07-18).
- Current structure: 1 Intro (challenges C1–C5) · 2 Related Work (compact, 38 lines) ·
  3 Methodology (3.1 problem formulation, 3.2 fairness objective, 3.3 why demand-only
  editing cannot help the under-served, 3.4 editor: attribution/trim/lift + downstream
  upweighting recipe) · 4 Experiments (setup / data-level SZ / downstream SZ /
  baselines / robustness / SF external validity) · 5 Conclusion · Appendices A–E.
- Figures: Fig 1 teaser (TikZ, `figures/figure-1/`), Fig 2 three-panel stylized city
  (TikZ, `figures/figure-2/`), Fig 3 weight sweep (matplotlib, appendix).
- Strict 8-page content limit at submission (references may spill to p9+). Current
  draft has a ~7-line spill of §5 onto p9 — restructure will re-shuffle lengths anyway.
- Gates before ANY commit in `paper/`:
  `cd /home/robert/FAMAIL/paper && latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`

## Zhang email digest (target structure)

```
1 Introduction
2 Overview                (current §3.1 goes here largely as-is)
    HSTD and Trajectory Representation
    Service Allocation Induced by HSTD
    Problem Definition
3 Methodology
    FATE overview paragraph (two stages: budgeted+constrained editing; edit-aware
      weighting), connecting components to the Introduction's challenges
    3.1 Collective Fairness Surrogate  (design requirements; why raw parity is wrong;
        residual; F_demo; why-useful list; full editing objective incl. spatial
        regularization + fidelity guardrail; derivations stay in appendix)
    3.2 FATE Editing
        3.2.1 Attribution-Guided Editing under a Limited Budget
              (outcome-side attribution; resource-side attribution v_i = ∂L/∂S_i;
               ranking/selection under budget K; attribution is the budget-allocation
               mechanism, not post-hoc explanation)
        3.2.2 Outcome-Side and Resource-Aware Trajectory Editing
              (outcome-side = trim, its limitation paragraph;
               resource-aware = lift with 7 named elements; closing key distinction)
        3.2.3 Budgeted Validity- and Fidelity-Constrained Editing
              (K vs ε; constraint list; identity model = identity-level
               behavioral-fidelity guardrail, NOT full realism; 6-step edit pipeline)
    3.3 Preserving the Influence of Sparse Edits via Edit-Aware Weighting
        (numbered 3.4 in her outline — typo; her prose says 3.3)
4 Experiments  (5 research questions; budget analysis "if possible"; framework figure)
5 Related Work   (MOVED after experiments)
6 Conclusion
```

Boldface defined terms in §2 following the cGAIL format. Section names from her email
are NOT binding per the meeting ("do not strictly mirror section names").

## Repo ground rules that survive the restructure

- Era numbers: α* = (0.1, 0.8, 0.1); SZ k=10,000 (2,455 trim selected → 2,337 net +
  7,545 lift); SF k=2,000 (1,330 + 629). Headline Δs: +0.0226 SZ / +0.0316 SF.
- Any `\cite`/`refs.bib` change updates `paper/CITATION_PRIORITY_CHECKLIST.md` in the
  same session; NEVER tick Robert's checkboxes.
- Writing conventions: `paper/README.md` (locked decisions list). Explicit-over-clever
  prose; em-dashes only to save space; no coinages; every load-bearing number carries a
  `% src:` provenance comment.
- Protected register: headline numbers and disclosures must survive somewhere
  (main text or appendix) — do not silently delete them.
