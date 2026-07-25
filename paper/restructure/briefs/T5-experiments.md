# TASK BRIEF T5 — experiments: leading paragraph, RQ framing, transferability reframe, dedup

Bring §4 in line with the restructure: a leading paragraph stating aims and
organization as research questions, the San Francisco subsection reframed as the
transferability block, duplicated cross-city prose collapsed, and budget-aware
framing added WITHOUT implying a budget sweep. §4's content and numbers are
battle-tested — this is framing surgery, not a rewrite.

## Read first
1. `paper/restructure/MEETING_DIGEST.md` decision #8 + §3 corrections (the [41:10]/
   [42:40] nuance: her durable instruction is the PREFERENCE at [44:52], not a claim
   about the current draft).
2. `paper/sections/04_experiments.tex` — current state. ⚠ The working tree carries
   Robert's OWN uncommitted edit near the top ("Editor configuration" paragraph: new
   wording active, old wording commented out). PRESERVE IT EXACTLY, including the
   commented block — it ships in your commit as his change.
3. `paper/sections/03_methodology.tex` — post-T4 state: verify every §3 label you
   reference (sec:objective, sec:leveling, sec:editor, sec:phys-validity,
   sec:downstream) still resolves and points where §4's prose assumes.
4. `paper/restructure/ALGORITHM_FACTS.md` §Headline results + §Things-FATE-is-NOT
   (the no-k-sweep line is binding: D16).
5. `paper/restructure/meeting/analysis_C_claims.md` C-9/C-10 (transferability and
   parity wording constraints) + §5 items 8, 11, 12.

## Deliverables (all inside `04_experiments.tex`)

1. **Leading paragraph** right after `\section{Experiments}` (before the Setup
   subsection): states what the experiments must show and how the section is
   organized, as research questions mapped to subsections. Draft ~5 questions that
   are honest to the existing content, in this spirit (your wording):
   - RQ1: does spending the edit budget where attribution directs it improve the
     corpus's collective fairness, on measures the objective never optimizes?
     (\S data-level)
   - RQ2: does resource-aware editing add real taxi presence to under-served areas,
     rather than only improving measured ratios? (\S data-level supply channel +
     the trim-only ablation)
   - RQ3: do the gains survive downstream training, and are they specific to the
     edits rather than to reweighting or selection? (\S downstream)
   - RQ4: does the improvement come from the fairness objective, or would bounded
     perturbation or resampling per se produce it? (\S baselines)
   - RQ5: are the conclusions robust to the demographic feature set and objective
     weights, and do they reproduce on a second city? (\S robustness + \S SF)
   Close the paragraph with one sentence: Shenzhen answers every question; San
   Francisco then tests whether the conclusions reproduce (the meeting's structure).
   The trade-off among fairness, budget, validity, and fidelity is REPORTED where it
   arises (fidelity/validity disclosures in data-level; the α sweep in robustness) —
   if you mention it in the lead, phrase it as "reported throughout", never as a
   budget analysis. NO sentence may imply k was varied (D16).
2. **Transferability reframe**: retitle the SF subsection toward the meeting's frame
   (suggestion: "Transferability: San Francisco" — keep `\label{sec:exp-sf}`);
   its opening sentence states the claim as reproduction on a second city with
   different geography, sampling, and fleet size — NEVER "generalizes to other
   cities" (D13). The existing caveats block stays verbatim.
3. **Cross-city dedup + contrast** (meeting: identical observations stated once;
   genuine differences kept and made "more visually striking" [46:10]): hunt
   sentences in the SF subsection that restate a Shenzhen observation verbatim-in-
   substance and collapse them to "as on Shenzhen (§ref), X reproduces: <numbers>";
   where SZ↔SF genuinely DIFFER (the tier-1 net-negative total read as demand
   endogeneity; F_spatial not propagating on SF; DI's grouping dependence; the GAN
   bimodality being city-specific), sharpen the juxtaposition — inline SZ-vs-SF
   number pairs in one sentence beat two separated paragraphs. No new tables. Every
   supply number keeps its accounting-tier label. Expected net: a modest line
   REDUCTION; report the delta.
4. **Budget framing, honestly**: where k appears (setup ¶), one clause may frame it
   as the paper's configured edit budget (≈ a tenth of the SZ corpus). If you add a
   forward-looking sentence anywhere, a k-sweep is FUTURE WORK (that sentence
   belongs to the conclusion task, not here — avoid duplicating it).
5. Do not touch: the three-ring metrics-and-claim-discipline block; Table 1/2
   contents; the protocol/statistics block; any number; any `% src:` or
   `% lint-allow:` comment (they move only if their sentence moves).

## Gates + checks
- Baseline build first (undefined-ref count), then:
  `cd /home/robert/FAMAIL/paper && latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`
- Zero undefined refs; lint exit 0 (the ablation lint-allow lines must still
  suppress).
- `pdftotext` read of §4's first page + the SF subsection; report REFERENCES start
  page + §6 tail lines (telemetry).

## Rules
- Touch ONLY `04_experiments.tex`. No git commands (your commit is made by the
  orchestrator; Robert's pending edit ships with it, unchanged).
- If Write/Edit is blocked: full file contents as text, BLOCKED(write-denied).

## Final reply (machine-read, ≤18 lines)
Status; the leading paragraph verbatim; the SF subsection's new title + opening
sentence; what you deduped (one line per collapse) and the net line delta; any
SZ↔SF contrast you sharpened; gate results; undefined-ref count; REFERENCES page.
