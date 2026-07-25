# TASK BRIEF T2 — rebuild the Introduction on the PI's revised intro (+ Fig-1 PNG swap)

Replace the current §1 with the PI's revised introduction, repaired where wording
would misstate the method, wired to the new §2 Overview, and with her teaser PNG
installed as Figure 1. Her intro's LOGIC and paragraph order are authoritative; exact
sentences may be smoothed for grammar and house style (her email: "I would like you
to decide the final wording and presentation").

## Read first
1. `paper/restructure/MEETING_DIGEST.md` — decisions #1–#5 (spine, challenge
   placement) and §2 ADJ-3.
2. `paper/sections/02_overview.tex` — LANDED by task T3: your intro's brief challenge
   mention must be consistent with its five challenges (C1 budget, C2 fidelity, C3
   target, C4 level-up, C5 training). Read the final wordings there.
3. `paper/sections/01_introduction.tex` — current file (being replaced): note its
   header comments, the 3.0× hook sentence + `% src:` at lines ~23–29, and the
   anonymous-repo footnote at lines ~150–155 (both survive, see below).
4. `paper/restructure/ALGORITHM_FACTS.md` + `paper/restructure/meeting/analysis_C_claims.md`
   §5 (do-not-claim list) + C-1/C-3 proposed wordings.
5. `paper/sections/00_abstract.tex` — T1's landed abstract: the intro must not
   contradict its wording (it won't if you follow this brief).
6. `paper/restructure/figures/FIG1_TEASER_SPEC.md` — Phase 1 (the PNG swap).

## Her introduction, verbatim (ASR-free source: her revision PDF). Bracketed notes
are MY instructions, not her text.

¶1:
> Human generated spatial-temporal data (HSTD), such as taxi trajectories, passenger
> trip records, and gig-worker traces, is increasingly used to learn human
> decision-making strategies for mobility modeling, simulation, and downstream
> decision applications [35?]. However, HSTD is not a neutral record of human
> behavior. It captures not only how human agents make sequential decisions, but also
> how these decisions collectively shape where services are repeatedly provided and
> where they remain scarce. The resulting fairness is therefore a collective property
> of the HSTD corpus: it emerges from how many local decisions jointly allocate
> services across locations and demographic groups. In urban mobility, for example,
> taxi trajectories encode both drivers' passenger-seeking strategies and the
> resulting spatial distribution of service across neighborhoods. Models trained on
> such data may reproduce not only useful human expertise but also demographic
> disparities embedded in the demonstrated service allocation. Once deployed, these
> learned decisions can reinforce historical allocation patterns, propagating bias
> from HSTD to downstream applications [6, 21].

[Repairs ¶1: "Human generated" → "Human-generated" (matches abstract). Broken cite
[35?] → `\cite{zhang2019cgail,zhang2022cgail}`; if refs.bib has a Feng-2020
learning-to-simulate entry, add it for "simulation" — check with grep; do NOT create
new bib entries. [6,21] → `\cite{ensign2018,lumisaac2016}`. Optional but recommended:
after the "In urban mobility, for example…" sentence, graft the current intro's
concrete hook with its src comment: "On the Shenzhen corpus we study, the advantaged
district receives $3.0\times$ the taxi service per unit of demand that the
disadvantaged district receives (Figure~\ref{fig:teaser})." — it is accurate, sourced,
and gives her generic paragraph a number. Flag in your report if you judge it breaks
her flow and omit.]

¶2:
> Existing methods address bias either by modifying data or by constraining a model,
> but they do not explicitly bridge the local-to-collective gap in HSTD: how
> individual trajectories jointly produce corpus-level service disparity. Data-level
> trajectory perturbation and fairness-aware generation alter individual samples or
> the overall distribution [15, 31, 34], but are not designed to attribute collective
> service disparity back to specific trajectories and their local decisions.
> Algorithm-level methods, including training-time reweighting and fairness-aware
> objectives [17, 37], mitigate bias for a particular model without identifying which
> demonstrations generate the observed collective disparity or repairing their
> contribution to it. This motivates a data-centric intervention that attributes
> collective service disparity to individual demonstrations and modifies their local
> decisions according to their aggregate effects.

[Cites: [15,31,34] → `\cite{hu2023stifgsm,vanbreugel2021decaf,xu2018fairgan}`;
[17,37] → `\cite{kamirancalders2012,zheng2023}`. All exist in refs.bib. No other
changes needed — this paragraph is the C-3 differentiation, accurately stated.]

¶3 (the brief challenge mention — keep as PROSE, no itemize; this is the meeting's
"briefly tell what are the challenges" beat):
> Realizing such an intervention is challenging: First, collective fairness is
> global, but edits are local. As shown in the left panel of Fig. 1, disparity
> emerges from the aggregation of many trajectories, not from any single
> demonstration. An editor must therefore trace this global disparity to influential
> trajectories and identify the local edits with the largest corpus-level effect.
> Second, reducing disparity does not necessarily improve disadvantaged-group
> welfare. An intervention may narrow a group gap by decreasing the measured outcome
> of the advantaged group rather than increasing the resources received by the
> disadvantaged group. In taxi mobility, for example, reducing the contribution of
> trips from heavily served areas can narrow the measured service gap without
> directing additional taxi presence to under-served neighborhoods. Third, fairness
> intervention must be effective under a limited edit budget. Modifying a large
> portion of HSTD can be costly and may distort the human behaviors that make the
> data valuable. The editor must therefore achieve substantial collective fairness
> improvement by modifying only a small subset of trajectories, while preserving the
> validity and downstream influence of these sparse edits.

[Repairs ¶3: keep her First/Second/Third prose — her closing clause ("while
preserving the validity and downstream influence of these sparse edits") already
gestures at C2 and C5, so all five §2 challenges are covered without a count claim.
"Fig. 1" → `Figure~\ref{fig:teaser}` (house style; "left panel" matches the new PNG).
END the paragraph with one forward sentence of your drafting, e.g.: "Section~\ref{sec:overview}
states these challenges precisely (C1–C5), and each component of our method answers
one of them." Do not italicize the three lead-ins beyond her \emph pattern if you
keep it; either style is fine.]

¶4 + contributions:
> To address these challenges, we introduce FATE, a Fairness-Aware Trajectory Editing
> framework for budgeted fairness intervention in HSTD. Given a limited edit budget,
> FATE attributes corpus-level disparity to individual trajectories and applies
> bounded modifications to those with the greatest aggregate fairness impact, as
> illustrated in Fig. 1. FATE distinguishes outcome-side edits, which change the
> measured allocation statistic, from resource-aware edits, which alter the
> underlying resource-provision process. The former may reduce disparity without
> benefiting disadvantaged groups, whereas the latter directly changes where
> resources are supplied. Because the edited demonstrations constitute only a small
> fraction of the corpus, FATE further applies edit-aware weighting to preserve their
> influence during downstream training. In the taxi-mobility instantiation studied in
> this paper, outcome-side and resource-aware edits correspond to modifying pickup
> endpoints and rerouting passenger-seeking trajectories, respectively. Experiments
> on real-world HSTD show that FATE consistently reduces demographic service
> disparity, increases resource presence in under-served areas, and preserves
> metric-level fairness improvements in learned policies. Our contributions are
> summarized as follows:
>
> • We formulate fairness mitigation in HSTD as a budgeted local-to-collective
> problem: disparity emerges from the aggregate effects of many sequential decisions,
> while only a small subset of individual trajectories can be modified. This
> formulation connects the edit budget on local demonstrations to its corpus-level
> fairness impact.
> • We develop FATE, which attributes collective disparity to individual trajectories
> and distinguishes edits that change a measured allocation statistic from those that
> alter the underlying resource-provision process. We further characterize why
> outcome-side editing can reduce disparity without improving disadvantaged-group
> resources and introduce a differentiable resource-aware intervention to address
> this limitation.
> • We instantiate FATE for taxi mobility and evaluate it on real-world HSTD from
> Shenzhen and San Francisco. The results show consistent reductions in demographic
> service disparity, increased resource presence in under-served areas, and transfer
> of metric-level fairness improvements to downstream imitation-learned policies. We
> made our code and dataset available to contribute to the research community via an
> anonymous link [?]

[Repairs ¶4/contributions:
- "greatest aggregate fairness impact" → "greatest estimated aggregate fairness
  impact" (matches T1's abstract).
- "rerouting passenger-seeking trajectories" → "rerouting final passenger-seeking
  segments" (the anchor/prefix is preserved; whole-trajectory rerouting is not what
  lift does).
- "increases resource presence in under-served areas" stays; if the sentence can
  cheaply carry it, prefer "adds taxi presence, counted as distinct vehicles, to
  under-served areas" ONCE (either here or leave for §4 — your call, flag it).
- "preserves metric-level fairness improvements in learned policies" is fine (the
  weighting sentence precedes it, so no D2 violation).
- Contributions render as a LaTeX itemize (the meeting's no-itemize rule was for the
  CHALLENGES in §2, not the contributions).
- Contribution 3: "We made our code and dataset available" → "We make our code and
  dataset available to the research community via an anonymous
  link.\footnote{\url{https://anonymous.4open.science/r/FATE-review}. % TODO(Robert,
  before submission): create the anonymous repository and replace this placeholder
  URL; an empty repository at submission is acceptable per the PI (Meeting 44).}"
  — carry the existing footnote + its TODO comment forward (Q4 ruling: code+data
  wording KEPT; the repo must hold both by Sunday — Robert's checklist).
- Do NOT reintroduce: "the wrong target" phrasing, C1–C5 itemize, or the old
  contributions list. The old intro is fully replaced; its header comment block is
  replaced by a new one recording this rewrite (date 2026-07-24 late, restructure T2,
  source = PI revision draft; pre-rewrite in git history + paper-pre-restructure-2026-07-25/).]

## Figure 1 swap (in the same file)
- `cp /home/robert/FAMAIL/paper/restructure/zhang/teasing.png /home/robert/FAMAIL/paper/figures/figure-1/teaser.png`
- Replace the current figure environment's `\input{figures/figure-1/figure-1}` with
  `\includegraphics[width=\columnwidth]{figures/figure-1/teaser.png}` (add graphicx?
  acmart loads graphics already — verify by compiling).
- Caption (hers): "Collective service disparity emerges from the aggregation of
  local trajectories (left). FATE edits a small set of influential trajectories to
  improve corpus-level fairness."
- New `\Description{...}`: 2–3 sentences describing the two-panel map (biased service
  left: many taxis/low demand advantaged side, few taxis/high demand disadvantaged
  side; after FATE right: moderated advantaged service, increased disadvantaged
  service, one edited trajectory shown dashed). Keep `\label{fig:teaser}` and the
  `[t]` placement; position the figure env after ¶1 (page-1 target; T-M1's board
  note explains the ACM-block displacement — if it still lands on p2, report, don't
  fight it).
- The retired TikZ teaser files stay on disk untouched.

## Gates + checks
- `cd /home/robert/FAMAIL/paper && latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`
- No undefined references (grep main.log); no duplicate-label warnings.
- `pdftotext -f 1 -l 2 main.pdf -` — eyeball ¶ flow and where Figure 1 landed; report.
- Report REFERENCES start page + §5-tail line count (page budget telemetry for T9).

## Rules
- Files you may touch: `01_introduction.tex`, the `figures/figure-1/teaser.png` copy.
  NOTHING else; 04_experiments.tex's uncommitted edit stays untouched. No git
  commands. New cites must already exist in refs.bib (you are not adding entries; if
  a needed key is missing, flag it instead).
- If Write/Edit is blocked: full file contents as text, status BLOCKED(write-denied).

## Final reply (machine-read, ≤18 lines)
Status; whether the 3.0× hook was kept and where; the exact forward-pointer sentence
you drafted for ¶3; contribution-3 final wording; where Figure 1 landed (page/col);
gate results; undefined-ref count; REFERENCES page + §5 tail lines; anything flagged.
