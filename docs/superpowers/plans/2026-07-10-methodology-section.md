# KDD Paper Scaffold + Methodology Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the compilable KDD manuscript scaffold (`paper/`), draft the full methodology section (`sections/03_methodology.tex`, 6 subsections), seed `refs.bib` from the verified reference list, and write a real draft abstract.

**Architecture:** One `acmart` (sigconf, anonymous) LaTeX project at repo root `paper/`; five `\input` section files (four stubs + the methodology deliverable); a convention lint script that runs with the compile gate after every task. Prose is *assembled* from the committed `PAPER/` bundles named per subsection — the sources are authoritative; the tasks are writing + condensation, not research.

**Tech Stack:** TeX Live (verified installed: `pdflatex`, `latexmk`, `acmart.cls`), BibTeX (`ACM-Reference-Format`), bash lint script.

**Spec:** `docs/superpowers/specs/2026-07-10-methodology-section-design.md` (approved 2026-07-10).

## Global Constraints

Every task's requirements implicitly include all of these:

- **Compile gate after every task:** `cd /home/robert/FAMAIL/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` → exit 0, `main.pdf` produced. LaTeX *warnings* are tolerated during drafting; *errors* are not.
- **Lint gate after every task:** `bash /home/robert/FAMAIL/paper/lint.sh` → exit 0.
- **Trim+lift centers ALL reporting.** Trim-only numbers (+0.0144 / +0.0128 / +0.0139) may appear ONLY in explicit ablation context (append `% lint-allow: ablation` to the line).
- **`F_causal` keeps its label + associational caveat.** No causality-claim language ("causal effect", "causally drives", …). No `F_demo` rename (pending PI decision).
- **The "54%" figure is banned.** Absolute deltas only.
- **`p = 0.031` never appears without** mean Δ + t-CI + monotone dose-response context (n=6 sign-unanimity floor). (Methodology should not need p-values at all.)
- **SF *reproduces* Shenzhen, never "beats" it.**
- **Every load-bearing number gets a provenance comment** on its line or the line above: `% src: PAPER/<path>` (or `LIFT_ALGORITHM_REFERENCE.md §n`).
- **Any single supply number states its tier** (tier-1 fractional-presence vs tier-2 distinct-taxi).
- **No custom LaTeX macros** — plain LaTeX everywhere (`F_{\mathrm{causal}}`, not `\Fc`) so sections paste cleanly into the shared Overleaf.
- **No product/tool names** (Claude, Anthropic, Cowork, Copilot, ChatGPT) anywhere.
- **Methodology is nearly number-free** — results numbers live in Experiments. Allowed in §3: design constants (ε=2, k=10,000, α=(0.2,0.7,0.1), floors 0.5/0.1, tail length 4, taper 0.25/0.5/0.75/1.0, 5×5, budget split 2,455/7,545, ~95k corpus, ~34,500 active units, 10 district profiles) and §3.4's structural facts (~32×, 93%, 2,455/2,455, 1.8 vs 17.6).
- **Commit after every task** with a `paper:`-prefixed conventional message.

## Notation (single source of truth for Tasks 3–9 — use these symbols verbatim)

| Symbol | Meaning |
|---|---|
| `g`, `t`, `i` | grid cell; hour block; active unit `i = (g,t)`, `i = 1..N` (N ≈ 34,500 on Shenzhen) |
| `D_i`, `S_i` | per-unit demand (recorded pickups); supply (active-taxi presence over the 5×5 neighborhood) |
| `d_0 = 0.5`, `s_0 = 0.1` | demand floor; supply floor |
| `Y_i = S_i / \max(D_i, d_0)` | service ratio |
| `x_i`, `X` | district-level demographic covariates (z-scored); their design matrix |
| `g_0(D)` | power-basis demand map (Stage-1 OLS) |
| `R_i = Y_i - g_0(D_i)` | demand-adjusted residual |
| `H`, `M` | projection onto `[1, X]`; centering matrix |
| `F_{\mathrm{causal}} = R^\top(I-H)R / R^\top M R = 1 - r^2_{\mathrm{demo}}` | primary fairness term |
| `F_{\mathrm{spatial}} = 1 - \tfrac{1}{2}(\mathrm{Gini}(\mathrm{DSR}) + \mathrm{Gini}(\mathrm{ASR}))` | spatial term |
| `F_{\mathrm{fidelity}}` | frozen driver-identity discriminator similarity |
| `\mathcal{L} = \alpha_{\mathrm{sp}} F_{\mathrm{spatial}} + \alpha_{\mathrm{ca}} F_{\mathrm{causal}} + \alpha_{\mathrm{fi}} F_{\mathrm{fidelity}}` | objective, `\alpha = (0.2, 0.7, 0.1)` |
| `\tau`, `\mathcal{T}` | a trajectory; the corpus (≈95k trajectories) |
| `k = 10{,}000`, `\varepsilon = 2` | edit budget; L∞ (Chebyshev) edit-ball radius in cells |
| `\delta \in [-\varepsilon,\varepsilon]^2` | integer edit offset |
| `\ell = 4` | tail length (last `\ell` seeking states + pickup move) |
| `w_j = j/\ell` | linear taper weights (0.25, 0.5, 0.75, 1.0) |
| `\Delta S` | differentiable endogenous supply delta |

## File Structure

```
paper/
  main.tex                 acmart sigconf+anonymous; title/authors (hidden); abstract; \input's; bib hookup
  sections/01_introduction.tex   stub (pointer comments)
  sections/02_related_work.tex   stub (pointer comments)
  sections/03_methodology.tex    the deliverable — subsections land in Tasks 3–8
  sections/04_experiments.tex    stub (pointer comments)
  sections/05_conclusion.tex     stub (pointer comments)
  refs.bib                 all entries from PAPER/objective-motivation/REFERENCES.md
  README.md                build instructions + writing conventions
  lint.sh                  convention lint (banned patterns)
  .gitignore               LaTeX build artifacts
```

---

### Task 1: Compilable scaffold + conventions + lint

**Files:**
- Create: `paper/main.tex`, `paper/sections/01_introduction.tex`, `paper/sections/02_related_work.tex`, `paper/sections/03_methodology.tex`, `paper/sections/04_experiments.tex`, `paper/sections/05_conclusion.tex`, `paper/refs.bib` (header-only), `paper/README.md`, `paper/lint.sh`, `paper/.gitignore`

**Interfaces:**
- Produces: the compile gate + lint gate commands every later task runs; the file layout every later task edits; `\section{Methodology}` numbered 3 (Tasks 3–8 write `\subsection`s inside `03_methodology.tex`).

- [ ] **Step 1: Create the directory and `.gitignore`**

`paper/.gitignore`:
```gitignore
*.aux
*.bbl
*.blg
*.log
*.out
*.fls
*.fdb_latexmk
*.synctex.gz
*.pdf
```

- [ ] **Step 2: Write `paper/main.tex`**

```latex
% FAMAIL — KDD manuscript.
% Source of truth for writing is THIS repo; Robert ports completed sections to the
% shared Overleaf for Dr. Zhang's review. Build: latexmk -pdf main.tex
% Conventions: see README.md in this directory (trim+lift canonical, F_causal label
% + associational caveat, provenance comments on every load-bearing number, etc.).
\documentclass[sigconf,review,anonymous]{acmart}
% KDD is double-blind: `anonymous` hides the author block; `review` adds line numbers.
% Remove `review` for the PI-facing PDF if line numbers distract.
\settopmatter{printacmref=false}
\setcopyright{none}

\begin{document}

\title{FAMAIL: Fairness-Aware Trajectory Editing as Data Augmentation for
Imitation-Learned Mobility Models}
% Working title — revisit with PI before submission.

% Author list/order provisional — PI decision; hidden by the `anonymous` option.
\author{Robert Ashe}
\affiliation{%
  \institution{San Diego State University}
  \city{San Diego}\state{CA}\country{USA}}
\email{robertashe@sdsu.edu} % placeholder address — confirm before de-anonymizing
\author{Xin Zhang}
\affiliation{%
  \institution{San Diego State University}
  \city{San Diego}\state{CA}\country{USA}}
\email{xzhang@sdsu.edu} % placeholder address — confirm before de-anonymizing

\begin{abstract}
Draft abstract lands in a later task (plan Task 9).
\end{abstract}

\keywords{fairness, imitation learning, data augmentation, trajectory editing,
urban mobility}

\maketitle

\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_methodology}
\input{sections/04_experiments}
\input{sections/05_conclusion}

\bibliographystyle{ACM-Reference-Format}
\bibliography{refs}

\end{document}
```

- [ ] **Step 3: Write the five section files**

`paper/sections/01_introduction.tex`:
```latex
\section{Introduction}\label{sec:intro}
% STUB — filled by a later task.
% Assemble from: PAPER/argument/01_motivation_goals.md (why inequity matters; why edit
% not generate; contributions) + PAPER/argument/00_overview.md (elevator argument).
% Narrative spine: external-metrics leveling-down caveat MOTIVATED trim+lift
% (Meeting-42 §3a→3b); lead with the two-pillar thesis.
```

`paper/sections/02_related_work.tex`:
```latex
\section{Related Work}\label{sec:related}
% STUB — filled by a later task.
% Assemble from: PAPER/objective-motivation/MOTIVATION.md ("Contrast / novelty" blocks
% per term) + REFERENCES.md groupings (fairness metrics; transportation equity;
% imitation/TUL/fidelity; adversarial perturbation & recourse; leveling-down ethics;
% feedback loops). Zheng et al. 2023 = closest applied neighbor (in-processing;
% FAMAIL is pre-processing/data-side).
```

`paper/sections/03_methodology.tex`:
```latex
\section{Methodology}\label{sec:method}
% Subsections land in plan Tasks 3-8:
% 3.1 problem formulation · 3.2 objective · 3.3 attribution (two mechanisms) ·
% 3.4 leveling-down mechanism · 3.5 trim+lift editor · 3.6 downstream recipe.
```

`paper/sections/04_experiments.tex`:
```latex
\section{Experiments}\label{sec:experiments}
% STUB — filled by a later task.
% Assemble from: PAPER/supply-lift/FINDINGS.md (+ tables/) = headline & channel decomposition;
% PAPER/external-metrics/FINDINGS.md = external-metrics protocol & results;
% PAPER/baselines/ = 6-row comparison (pending GPU runs) + demographic-oversampling FINDINGS;
% PAPER/by_feature_set/ = 3-feature-set robustness; PAPER/second-dataset/ = SF external validity;
% PAPER/argument/04_evaluation.md = two-pillar design + statistical conventions.
% Disclosure ledger to carry: rollout-allocation drain attenuated ~40% NOT reversed;
% leveling-down caveat; skip-on-infeasible provenance; tier-1 vs tier-2 supply accounting;
% SF mean(Y|D) tension (open PI framing).
```

`paper/sections/05_conclusion.tex`:
```latex
\section{Conclusion}\label{sec:conclusion}
% STUB — filled by a later task.
% Assemble from: PAPER/argument/07_limitations.md (candid limitations) + supply-lift
% FINDINGS §10 (deferrals) + future work: unified one-pass trim+lift editing;
% training-side allocation constraints.
```

- [ ] **Step 4: Write `paper/refs.bib` header (entries land in Task 2)**

```bibtex
% FAMAIL manuscript bibliography.
% Source of truth: PAPER/objective-motivation/REFERENCES.md (metadata verified
% 2026-07-08 against arXiv/DOI/ACM DL/IEEE Xplore/DBLP/Crossref).
% NOTE: Robert's own human citation pass (Meeting-42 T3) is still pending — these
% entries are machine-verified only.
```

- [ ] **Step 5: Write `paper/lint.sh`**

```bash
#!/usr/bin/env bash
# Convention lint for the FAMAIL manuscript (see README.md). Exit 1 on any hit.
# A line may carry `% lint-allow: <reason>` to suppress a hit (e.g. ablation context).
set -u
cd "$(dirname "$0")"
fail=0
check() { # $1 = grep -E pattern, $2 = description
  local hits
  hits=$(grep -RInE "$1" --include='*.tex' . | grep -v 'lint-allow' || true)
  if [ -n "$hits" ]; then
    echo "LINT FAIL — $2:"
    echo "$hits"
    fail=1
  fi
}
check '54 ?(\\%|%|percent)'                        'ungrounded "54%" figure (banned until grounded)'
check 'causal (effect|impact|estimate)|causally'   'causality-claim language (F_causal is associational)'
check '0\.0144|0\.0128|0\.0139'                    'trim-only headline number outside ablation context'
check '[Cc]laude|[Aa]nthropic|[Cc]owork|[Cc]opilot|ChatGPT' 'tool/product name'
check '(beats|outperforms) (Shenzhen|the first city)' 'SF must reproduce, not beat'
exit $fail
```

- [ ] **Step 6: Write `paper/README.md`**

```markdown
# FAMAIL — KDD manuscript

**This repo is the writing source of truth.** Robert ports completed sections to the
shared Overleaf for Dr. Zhang's review (old Overleaf content is out of scope). Each
`sections/*.tex` file is self-contained plain LaTeX (no custom macros) so it pastes
cleanly.

## Build

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex   # compile gate
    bash lint.sh                                                    # convention lint

Both must pass before every commit. Warnings are tolerated during drafting; errors are not.

## Writing conventions (locked decisions — do not relitigate in prose)

1. **Trim+lift centers ALL reporting.** Trim-only numbers appear ONLY in the
   trim-vs-trim+lift ablation (mark the line `% lint-allow: ablation`).
2. **F_causal keeps its label + associational caveat.** No causality-claim language;
   no F_demo rename (pending PI decision).
3. **The spoken "54%" figure is banned** until grounded. Absolute deltas only
   (+0.0222 SZ / +0.0328 SF).
4. **p = 0.031 never appears without** mean Δ + t-CI + monotone dose-response — it is
   the n=6 Wilcoxon sign-unanimity floor, not an effect size.
5. **SF *reproduces* Shenzhen, never "beats" it** (F_causal is city-specific and
   associational; absolute baselines are not cross-city comparable).
6. **Every load-bearing number carries a provenance comment**: `% src: PAPER/<path>`.
7. **Any single supply number states its accounting tier** — tier-1 (fractional
   presence, optimizer convention) vs tier-2 (distinct-taxi recount from raw GPS).
   See PAPER/supply-lift/LIFT_ALGORITHM_REFERENCE.md §10.
8. **Three-ring metric firewall** (LIFT_ALGORITHM_REFERENCE.md §13):
   (i) optimized: F_spatial/F_causal/F_fidelity; (ii) design-targeted, not optimized:
   mean(Y|D)/SDR family; (iii) genuinely external: DP, DI, Theil, per-group levels,
   tier-2 recount, channel decomposition. "Improves metrics we never optimized"
   claims ride ring (iii) only.
9. **No product/tool names** anywhere.

## Layout

`main.tex` (acmart sigconf, anonymous+review) → `sections/01..05` → `refs.bib`
(seeded from PAPER/objective-motivation/REFERENCES.md; T3 human pass pending).
```

- [ ] **Step 7: Run the compile gate (this is the failing-test moment — run it BEFORE the files are complete only if you want to see the failure; the required check is that it passes now)**

Run: `cd /home/robert/FAMAIL/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
Expected: exit 0; `main.pdf` exists (title page + empty numbered sections 1–5). An "Empty bibliography" BibTeX warning is fine.

- [ ] **Step 8: Run the lint gate**

Run: `bash /home/robert/FAMAIL/paper/lint.sh`
Expected: exit 0, no output.

- [ ] **Step 9: Verify git sees all files (repo has aggressive global ignores)**

Run: `cd /home/robert/FAMAIL && git status --short paper/`
Expected: all 10 files listed as untracked (`main.pdf` and build artifacts NOT listed). If any `.tex`/`.bib`/`.md`/`.sh` file is missing, fix `.gitignore` interactions before committing.

- [ ] **Step 10: Commit**

```bash
cd /home/robert/FAMAIL
git add paper/
git commit -m "paper: compilable acmart scaffold + writing conventions + lint gate"
```

---

### Task 2: Seed `refs.bib` from the verified reference list

**Files:**
- Modify: `paper/refs.bib`

**Interfaces:**
- Consumes: `PAPER/objective-motivation/REFERENCES.md` (the single citation source of truth, metadata verified 2026-07-08).
- Produces: the exact citation keys Tasks 4–8 use in `\cite{}` (list below — these keys are load-bearing; do not improvise different ones).

- [ ] **Step 1: Transcribe every entry of `PAPER/objective-motivation/REFERENCES.md` into BibTeX**

One `@inproceedings`/`@article`/`@book`/`@misc` entry per REFERENCES.md bullet, fields copied from the verified metadata (title, authors, venue, year, pages, DOI where given). Include the entry-specific caution notes as trailing `%` comments (e.g. Zheng: cite the absolute MPE-gap 0.361→0.084, not percentage headlines; Parfit: title must match the cited edition; Mittelstadt: year matches version).

**The exact key set** (Tasks 4–8 cite these; full inventory for later sections too):

```
corbettdavies2017  feldman2015     kamirancalders2012  vermarubin2018   barocas2023
frischwaugh1933    lovell1963      horchergraham2021   karner2024       atkinson1970
theil1967          demaio2007      zheng2023           hoermon2016      zhang2019cgail
zhang2022cgail     pan2020xgail    ren2020stsiamese    gao2017tuler     zhou2018tulvae
miao2020deeptul    feng2020simulate goodfellow2015fgsm kurakin2017ifgsm hu2023stifgsm
ustun2019recourse  wachter2018counterfactual  karimi2020recourse  karimi2021recourse
jang2017gumbel     maddison2017concrete  bengio2013ste  parfit1997      temkin1993
temkin2000         mittelstadt2024 zietlow2022         pinzon2022      ensign2018
lumisaac2016
```

Three worked examples of the required format (repeat the pattern for all):

```bibtex
@inproceedings{corbettdavies2017,
  author    = {Corbett-Davies, Sam and Pierson, Emma and Feller, Avi and Goel, Sharad and Huq, Aziz},
  title     = {Algorithmic Decision Making and the Cost of Fairness},
  booktitle = {Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year      = {2017},
  pages     = {797--806}
} % conditional statistical parity FORMALIZED here (not originated) — building on Kamiran 2013 / Dwork 2012

@article{zhang2022cgail,
  author  = {Zhang, Xin and Li, Yanhua and Zhou, Xun and Luo, Jun},
  title   = {{cGAIL}: Conditional Generative Adversarial Imitation Learning --- An Application in Taxi Drivers' Strategy Learning},
  journal = {IEEE Transactions on Big Data},
  volume  = {8},
  number  = {5},
  pages   = {1288--1300},
  year    = {2022},
  doi     = {10.1109/TBDATA.2020.3039810}
} % venue is IEEE Trans. Big Data, NOT TKDE (citation-audit correction 2026-07-08)

@article{lumisaac2016,
  author  = {Lum, Kristian and Isaac, William},
  title   = {To Predict and Serve?},
  journal = {Significance},
  volume  = {13},
  number  = {5},
  pages   = {14--19},
  year    = {2016},
  doi     = {10.1111/j.1740-9713.2016.00960.x}
}
```

- [ ] **Step 2: Verify the bibliography parses and renders (temporary `\nocite{*}`)**

Add `\nocite{*}` on the line before `\bibliographystyle` in `main.tex`, then:

Run: `cd /home/robert/FAMAIL/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex && grep -c '\\bibitem' main.bbl`
Expected: exit 0; `\bibitem` count equals the number of entries (≈40); no "Warning--" lines from BibTeX in `main.blg` other than capitalization/style notes: check with `grep -i 'warning' main.blg || true` and resolve anything about missing fields.

- [ ] **Step 3: Remove `\nocite{*}`, recompile**

Run: `cd /home/robert/FAMAIL/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`
Expected: both exit 0 (bibliography is empty again — that is correct until sections cite).

- [ ] **Step 4: Commit**

```bash
cd /home/robert/FAMAIL
git add paper/refs.bib paper/main.tex
git commit -m "paper: seed refs.bib from verified REFERENCES.md (T3 human pass pending)"
```

---

### Task 3: §3.1 Problem formulation

**Files:**
- Modify: `paper/sections/03_methodology.tex`

**Interfaces:**
- Consumes: the plan-level Notation table (above) — verbatim symbols.
- Produces: `\subsection{Problem Formulation}\label{sec:problem}`; the notation every later subsection uses; labels `sec:problem`.

- [ ] **Step 1: Read the sources**

Read: `PAPER/argument/02_datasets.md` (grid/units, district demographics), `PAPER/argument/03_fairness_theory.md` (metric setup), `PAPER/supply-lift/LIFT_ALGORITHM_REFERENCE.md` §3 (exact conventions table: presence mass, floors, tail, taper, edit ball, king-move rule).

- [ ] **Step 2: Write `\subsection{Problem Formulation}` (~0.4 page)**

Required content (all of it; nothing else):
1. Setting: a city discretized into 0.01° grid cells `g` × hourly blocks `t`; **active units** `i=(g,t)`; per-unit demand `D_i` (recorded pickups) and supply `S_i` (active-taxi presence aggregated over the 5×5 neighborhood, matching how the supply grid is built from GPS).
2. Service ratio `Y_i = S_i/\max(D_i, d_0)` with `d_0 = 0.5` (floor rationale: near-empty cells otherwise explode the ratio). `% src: LIFT_ALGORITHM_REFERENCE.md §3`
3. Demographics: district-level covariates `x_i` (housing price, per-capita compensation, migrant share) — resolve to ~10 district profiles on the primary city (sets up the ecological caveat cited in §3.2). `% src: PAPER/argument/02_datasets.md`
4. The corpus `\mathcal{T}` of real per-driver trajectories (seeking states + pickup); trajectories, not grids, are the editable objects.
5. **The task statement** (this is the paragraph the whole paper hangs on): given `\mathcal{T}` encoding demographic service inequity, produce an edited corpus `\mathcal{T}'` that (a) is measurably fairer, (b) remains realistic at the individual-trajectory level (a frozen driver-identity discriminator must still recognize each edited trajectory as its driver), and (c) transfers its fairness into a policy trained on it — via editing a small attribution-targeted slice (budget `k`, per-edit L∞ bound `\varepsilon`) and upweighting that slice downstream. Explicitly: FAMAIL is **data augmentation** (edits real trajectories; generates nothing).

- [ ] **Step 3: Compile + lint gates**

Run: `cd /home/robert/FAMAIL/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh`
Expected: both exit 0.

- [ ] **Step 4: Self-check against the required-content list (5 items present, no results numbers, provenance comments on constants)**

- [ ] **Step 5: Commit**

```bash
cd /home/robert/FAMAIL
git add paper/sections/03_methodology.tex
git commit -m "paper: methodology 3.1 problem formulation"
```

---

### Task 4: §3.2 Fairness objective

**Files:**
- Modify: `paper/sections/03_methodology.tex`

**Interfaces:**
- Consumes: notation from Task 3; cite keys from Task 2 (`corbettdavies2017, frischwaugh1933, lovell1963, feldman2015, horchergraham2021, karner2024, ren2020stsiamese, gao2017tuler, zhou2018tulvae, miao2020deeptul, hoermon2016, feng2020simulate`).
- Produces: `\subsection{The Fairness Objective}\label{sec:objective}`; equation labels `eq:fcausal`, `eq:objective` (Task 5 and 7 reference them).

- [ ] **Step 1: Read the sources**

Read: `PAPER/objective-motivation/MOTIVATION.md` (the why/how paragraphs — written to be near camera-ready; adapt, don't re-research) and `PAPER/argument/03_fairness_theory.md` (formulas).

- [ ] **Step 2: Write `\subsection{The Fairness Objective}` (~0.8 page)**

Required content:
1. **F_causal** (primary): why raw parity is wrong (service legitimately tracks demand) → adjust for demand first = conditional statistical parity `\cite{corbettdavies2017}`. Two-stage construction: Stage 1 power-basis `g_0(D)`, residual `R`; Stage 2 partial R² of demographics on `R` via FWL `\cite{frischwaugh1933,lovell1963}`; fairness-as-predictability lineage `\cite{feldman2015}`. Display equation (label `eq:fcausal`):
   ```latex
   F_{\mathrm{causal}} \;=\; \frac{R^\top (I-H)\, R}{R^\top M\, R} \;=\; 1 - r^2_{\mathrm{demo}},
   ```
   with one sentence on boundary cases (R ∈ span(X) ⇒ 0; R ⊥ X ⇒ 1). **The associational caveat, verbatim in spirit:** F_causal is an associational partial R² on observational district-level demographics (no identification, no counterfactual; the name is historical); ~10 district profiles ⇒ ecological-fallacy exposure. **Demand-endogeneity forward-pointer**: conditioning on recorded demand assumes it exogenous; §3.4 (`\ref{sec:leveling}`) shows the same assumption bounds the editor. `% src: PAPER/objective-motivation/MOTIVATION.md`
2. **F_spatial** (secondary): differentiable pairwise Gini over supply-normalized rates (DSR/ASR); Gini = transportation-equity standard `\cite{horchergraham2021,karner2024}`; demographic-independent smoothness term.
3. **F_fidelity** (guardrail): frozen ST-SiameseNet driver-identity discriminator `\cite{ren2020stsiamese}` (TUL premise: mobility signatures identify `\cite{gao2017tuler,zhou2018tulvae,miao2020deeptul}`); frozen to avoid a live adversarial game `\cite{hoermon2016}`; **honest framing**: at ε=2 its gradient w.r.t. the edit is ≈0 — it is a realism *guardrail*, not an edit driver; distributional collapse is guarded separately by a JS-divergence check `\cite{feng2020simulate}` (evaluation-side).
4. **Scalarization** (label `eq:objective`):
   ```latex
   \mathcal{L} \;=\; \alpha_{\mathrm{sp}} F_{\mathrm{spatial}} + \alpha_{\mathrm{ca}} F_{\mathrm{causal}} + \alpha_{\mathrm{fi}} F_{\mathrm{fidelity}},
   \qquad \alpha = (0.2,\, 0.7,\, 0.1).
   ```
   Why these weights: the selection criterion (maximize ΔF_causal subject to ΔF_spatial ≥ 0) + the gradient geometry is causal-dominated (weights reflect, not force, where editable signal lies) + one sentence citing the empirical α-Pareto frontier confirming the adopted point sits on it, marked:
   ```latex
   % TODO(alpha-sweep): finalize frontier sentence when 5/5 lands (~2026-07-11 AM);
   % partial 4-point finding: frontier flat (ΔF_causal +0.0217..+0.0226), shipped point on it.
   % src: famail_temporal/results/alpha_sweep/summary/ (pending)
   ```
   Do NOT cite the trim-only +0.0128 here (superseded framing; MOTIVATION.md's fold-in is a separate task).

- [ ] **Step 3: Compile + lint gates** (same commands; both exit 0)

- [ ] **Step 4: Self-check** — 4 items present; associational caveat present; no causality language (lint enforces); the only numbers are α weights + district count; every `\cite` key exists in refs.bib (grep each against refs.bib).

- [ ] **Step 5: Commit**

```bash
cd /home/robert/FAMAIL
git add paper/sections/03_methodology.tex
git commit -m "paper: methodology 3.2 fairness objective"
```

---

### Task 5: §3.3 Attribution — two mechanisms

**Files:**
- Modify: `paper/sections/03_methodology.tex`

**Interfaces:**
- Consumes: notation (Task 3); `eq:fcausal` (Task 4).
- Produces: `\subsection{Attribution: Localizing the Deficit and the Remedy}\label{sec:attribution}`; equation labels `eq:unit-attr`, `eq:supply-grad` (Task 7 references `eq:supply-grad`).

- [ ] **Step 1: Read the sources**

Read: `PAPER/argument/03_fairness_theory.md` (per-cell attribution), `PAPER/supply-lift/LIFT_ALGORITHM_REFERENCE.md` §4.1–4.3 (supply gradient, analytic anchor, screen, plan assembly).

- [ ] **Step 2: Write the subsection (~0.6 page), parallel two-mechanism structure**

Required content:
1. Framing sentence: the objective is optimized by editing individual trajectories, so FAMAIL needs to know *which* trajectories to edit; it uses two attribution mechanisms — one locating **where existing unfairness concentrates** (drives trim), one locating **where added supply would most improve fairness** (drives lift). One attributes the *deficit*, the other the *remedy*.
2. **(a) Deficit attribution.** Because `M` and `(I−H)` are idempotent, `r²_demo` admits an exact per-unit decomposition (label `eq:unit-attr`):
   ```latex
   r^2_{\mathrm{demo}} \;=\; \sum_i \frac{(MR)_i^2 - \bigl((I-H)R\bigr)_i^2}{R^\top M R},
   ```
   an exact partition of the fairness deficit (not a heuristic saliency); a signed variant separates over- from under-served units; the top-attribution units select the pickups trim relocates. `% src: PAPER/argument/03_fairness_theory.md`
3. **(b) Supply-gradient attribution.** Ask, at every active unit, "how much fairer would service be with marginally more taxi presence here?" — one backward pass through `\mathcal{L}` w.r.t. a zero-initialized `\Delta S` answers all ~34,500 units at once (a value-of-presence map). The F_causal component has a closed form (label `eq:supply-grad`), against which the autograd gradient is verified:
   ```latex
   \frac{\partial F_{\mathrm{causal}}}{\partial S_i}
   \;=\; \frac{2}{R^\top M R}\,
   \frac{\bigl((I-H)R\bigr)_i - F_{\mathrm{causal}}\,(MR)_i}{\max(D_i,\, d_0)} .
   ```
   `% src: LIFT_ALGORITHM_REFERENCE.md §4.1 (autograd-verified closed form)`
4. **The screen**: each trajectory's tail (last `\ell` seeking states + pickup) is rigidly translated by each integer `\delta \in [-\varepsilon,\varepsilon]^2\setminus\{0\}`; the linearized gain sums presence-mass × the (5×5 box-summed) gradient change over tail states; a trajectory's score is its best δ; all ≈95k trajectories are ranked. **The screen only nominates** — the per-edit optimizer (§`\ref{sec:editor}`) re-derives each actual move under the full objective. Budget assembly: trim's selection takes precedence; lift fills the remaining budget with positive-score nominees. `% src: LIFT_ALGORITHM_REFERENCE.md §4.2–4.3`

- [ ] **Step 3: Compile + lint gates** (both exit 0)

- [ ] **Step 4: Self-check** — parallel structure present; both equations labeled; "nominates, optimizer decides" present; no results numbers beyond design constants.

- [ ] **Step 5: Commit**

```bash
cd /home/robert/FAMAIL
git add paper/sections/03_methodology.tex
git commit -m "paper: methodology 3.3 two-mechanism attribution"
```

---

### Task 6: §3.4 Why demand-only editing levels down

**Files:**
- Modify: `paper/sections/03_methodology.tex`

**Interfaces:**
- Consumes: notation (Task 3); cite keys `parfit1997, mittelstadt2024, zietlow2022, ensign2018, lumisaac2016`.
- Produces: `\subsection{Why Demand-Only Editing Levels Down}\label{sec:leveling}` (referenced from §3.2's forward-pointer and §3.5).

- [ ] **Step 1: Read the sources**

Read: `PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md` (structural analysis + numbers), `PAPER/objective-motivation/LEVELING_DOWN.md` (ethical/fair-ML framing), `LIFT_ALGORITHM_REFERENCE.md` §1–2 (condensed facts + oracle).

- [ ] **Step 2: Write the subsection (~0.5 page)**

Required content:
1. The observed property: a demand-only editor improves the fairness metrics **only by leveling down** — on the primary city, all 2,455/2,455 edited pickups originated *and* landed in advantaged cells; the under-served group's service was untouched. Name the leveling-down objection `\cite{parfit1997}` and the fair-ML formalization (leveling-up prescription) `\cite{mittelstadt2024}`; augmentation as the one strategy that helped the disadvantaged group `\cite{zietlow2022}`. `% src: PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md`
2. Why it is the **constrained optimum**, not an optimizer quirk — three structural causes: (i) residual-variance attribution only selects over-served cells; (ii) with supply frozen, `\partial Y/\partial D = -S/D^2` gives ~32× more leverage to padding demand into high-supply (advantaged) cells, and **93% of disadvantaged units sit at the demand floor** where demand edits are inert; (iii) the actual inequity is supply-side (median presence 1.8 vs 17.6 taxis) and supply is frozen. A greedy oracle confirms the only demand-side path that raises the under-served ratio is deleting their recorded pickups — perverse. `% src: LEVELING_DOWN_MECHANISM.md; LIFT_ALGORITHM_REFERENCE.md §1`
3. **Demand endogeneity** as the unifying cause: recorded demand is suppressed by historical under-supply (feedback-loop pathology `\cite{ensign2018,lumisaac2016}`); the metric's blind spot and the editor's leveling-down are the same phenomenon.
4. The consequence that motivates §3.5: the one non-perverse lever is the numerator — `\partial Y/\partial S = 1/\max(D_i, d_0) > 0` everywhere (at the floor, `\Delta Y = 2\,\Delta S`) — so the editor needs a **supply channel**: rerouting real seeking behavior into under-served cells. One sentence noting a pre-build oracle bounded the achievable supply-channel headroom well above a pre-registered go threshold (numbers in Experiments). `% src: LIFT_ALGORITHM_REFERENCE.md §2`

- [ ] **Step 3: Compile + lint gates** (both exit 0)

- [ ] **Step 4: Self-check** — 4 items present; the structural numbers carry `% src:` comments; oracle = one sentence, no oracle numbers; no "constraint-forced" over-claim (Pinzón analogy stays out of methodology — it belongs in related work if anywhere).

- [ ] **Step 5: Commit**

```bash
cd /home/robert/FAMAIL
git add paper/sections/03_methodology.tex
git commit -m "paper: methodology 3.4 leveling-down mechanism"
```

---

### Task 7: §3.5 The trim+lift editor + method-overview figure placeholder

**Files:**
- Modify: `paper/sections/03_methodology.tex`

**Interfaces:**
- Consumes: notation (Task 3); `eq:objective` (Task 4); `eq:supply-grad` + screen description (Task 5); `sec:leveling` (Task 6); cite keys `goodfellow2015fgsm, kurakin2017ifgsm, hu2023stifgsm, jang2017gumbel, maddison2017concrete, bengio2013ste, ustun2019recourse, wachter2018counterfactual, zhang2022cgail`.
- Produces: `\subsection{The Trim+Lift Editor}\label{sec:editor}`; figure placeholder `fig:overview`.

- [ ] **Step 1: Read the sources**

Read: `LIFT_ALGORITHM_REFERENCE.md` §3, §5–6 (conventions, pipeline order, per-edit optimizer), `PAPER/supply-lift/FINDINGS.md` §2 (method summary), `docs/presentations/meeting_42_update/trim_plus_lift_explainer.md` §1–3 (the narrative register to aim for — but that doc predates the first trim+lift run; numbers come only from FINDINGS/REFERENCE).

- [ ] **Step 2: Write the subsection (~0.9 page)**

Required content:
1. **Shared machinery** (one paragraph): both edit modes repurpose bounded adversarial perturbation `\cite{goodfellow2015fgsm,kurakin2017ifgsm}` — the group's spatio-temporal instantiation `\cite{hu2023stifgsm}` — *constructively*, in the spirit of algorithmic recourse `\cite{ustun2019recourse,wachter2018counterfactual}`: per iteration a signed-gradient step on the edit offset, cumulative L∞ clip at `\varepsilon = 2` cells (the identity-preservation budget); discrete grid bridged by temperature-annealed soft cell assignment over the 5×5 supply-matched window `\cite{jang2017gumbel,maddison2017concrete,bengio2013ste}`; best-iterate selection. Include the two-line step display:
   ```latex
   \delta \leftarrow \mathrm{clip}\bigl(\eta \cdot \mathrm{sign}(\nabla_{\delta}\,\mathcal{L}),\, -\varepsilon,\, \varepsilon\bigr),
   \qquad
   \delta_{\mathrm{total}} \leftarrow \mathrm{clip}(\delta_{\mathrm{total}} + \delta,\, -\varepsilon,\, \varepsilon).
   ```
2. **Trim mode** (short — it is the published mechanism): deficit-attribution-selected pickups relocate within the ε-ball, pulling demand out of over-served hotspots; its optimization path is unchanged from the demand-only editor.
3. **Lift mode** (the new mechanism, most of the page): supply-gradient-nominated trajectories have their **whole seeking tail** translated — pickup moves the full offset, earlier tail states move by linear taper `w_j = j/\ell` (0.25/0.5/0.75/1.0), the anchor state never moves — so the trajectory *physically cruises through* the under-served area before pickup. The moved states carry their presence mass **differentiably**: each state contributes `1/12` hourly presence over its 5×5 neighborhood to `\Delta S`, and the objective evaluates supply as `\mathrm{clamp}(S + \Delta S,\, s_0)` — **supply is endogenous**, so the optimizer is rewarded for *providing* service, not only redistributing demand. The fidelity term scores the actual rerouted tail each iteration. Every edit updates a shared running state (later edits see earlier edits' effects). `% src: LIFT_ALGORITHM_REFERENCE.md §5–6`
4. **Physical validity**: source preprocessing (cGAIL lineage `\cite{zhang2022cgail}`) enforces the king-move rule (`\max(|dx|,|dy|)\le 1` per transition); discretized edits pass through an exact backward-reachability repair that returns a compliant assignment nearest the tapered targets or reports infeasibility; **infeasible ⇒ the edit is skipped** (applied uniformly to both modes). This closes a latent inconsistency: bounded pickup-only moves of ≥2 cells otherwise violate the rule that the source data itself is filtered on.
5. **Budget + phase order**: trim keeps its published selection; lift fills the remaining budget (k = 10,000 → 2,455 trim + 7,545 lift on the primary city), with the supply gradient computed on the post-trim state. Close with the **two-phase-as-scientific-control** rationale: trim's optimization is frozen byte-identical to the published editor, so (a) published results reproduce inside the combined run and (b) trim-only vs trim+lift is a clean ablation attributing every delta to the new mechanism; unified one-pass editing (gradient chooses each edit's character) is future work. `% src: LIFT_ALGORITHM_REFERENCE.md §5; trim_plus_lift_explainer.md §3`

- [ ] **Step 3: Add the method-overview figure placeholder**

```latex
\begin{figure*}[t]
  \centering
  % TODO(figure): three-panel overview — (1) the service gap (over-served cluster vs
  % under-served area); (2) trim: pickups nudged out of hotspots (levels down);
  % (3) lift: value-of-presence heat map + a tail bending into the glow (lifts up).
  % Design candidate: docs/presentations/meeting_42_update/trim_plus_lift_explainer.md §4.
  \fbox{\parbox{0.9\textwidth}{\centering\vspace{2em}Method overview figure — placeholder.\vspace{2em}}}
  \caption{FAMAIL trim+lift editing overview (placeholder).}
  \label{fig:overview}
\end{figure*}
```

- [ ] **Step 4: Compile + lint gates** (both exit 0)

- [ ] **Step 5: Self-check** — 5 content items + figure present; taper/floors/budget constants carry `% src:`; the trim-only vs trim+lift *ablation* mention is design rationale with no trim-only numbers (no lint-allow needed); `\ref{sec:leveling}`/`\ref{eq:supply-grad}` resolve (check `main.log` for "undefined references": `grep -i 'undefined' paper/main.log` → only the intentionally-empty stubs' absence of refs, i.e. no hits).

- [ ] **Step 6: Commit**

```bash
cd /home/robert/FAMAIL
git add paper/sections/03_methodology.tex
git commit -m "paper: methodology 3.5 trim+lift editor + overview figure placeholder"
```

---

### Task 8: §3.6 Downstream training recipe

**Files:**
- Modify: `paper/sections/03_methodology.tex`

**Interfaces:**
- Consumes: notation (Task 3); cite keys `kamirancalders2012, feldman2015, zheng2023`.
- Produces: `\subsection{Downstream Recipe: Upweighted Imitation}\label{sec:downstream}` — completes the section.

- [ ] **Step 1: Read the source**

Read: `PAPER/objective-motivation/MOTIVATION.md` ("Downstream pairing" §) and `PAPER/argument/04_evaluation.md` §1 (L2/Pillar-2 arm structure — for accurate one-sentence characterization only; the results stay in Experiments).

- [ ] **Step 2: Write the subsection (~0.3 page)**

Required content:
1. Editing is necessary but not sufficient: the edited slice is small (≈10% of the corpus), and a vanilla behavior-cloning objective averages it away — the old bias is relearned (verified as a genuine null; results in Experiments, `\ref{sec:experiments}`).
2. FAMAIL therefore **upweights** the edited demonstrations in the imitation loss — instance reweighing for fairness `\cite{kamirancalders2012}` (pre-processing family `\cite{feldman2015}`) transplanted to imitation learning.
3. Edit-specificity is validated against two controls (upweight a random subset; upweight the already-fairest trajectories) — design named here, outcomes in Experiments.
4. Positioning sentence: this targets, on the training data, the same demand-model bias addressed downstream by in-processing regularization `\cite{zheng2023}`; FAMAIL intervenes on the demonstrations instead.

- [ ] **Step 3: Compile + lint gates** (both exit 0)

- [ ] **Step 4: Self-check** — 4 items; no p-values, no Δ values; §3 now reads 3.1→3.6 with no gaps.

- [ ] **Step 5: Commit**

```bash
cd /home/robert/FAMAIL
git add paper/sections/03_methodology.tex
git commit -m "paper: methodology 3.6 downstream upweighting recipe"
```

---

### Task 9: Draft abstract

**Files:**
- Modify: `paper/main.tex` (replace the placeholder abstract)

**Interfaces:**
- Consumes: the completed §3 (Tasks 3–8) for terminology consistency; headline numbers from `famail_temporal/baselines/meeting_prep/MEETING_43_PREP.md` §5.
- Produces: the real draft abstract (T6 deliverable — Robert sends to Dr. Zhang).

- [ ] **Step 1: Replace the abstract body with the draft below, then edit for flow (~200 words). Content is binding; wording may improve.**

```latex
\begin{abstract}
% DRAFT (2026-07-10) — pending final results; numbers below are committed headline
% values. % src: PAPER/supply-lift/FINDINGS.md; PAPER/external-metrics/FINDINGS.md;
% PAPER/baselines/demographic-oversampling/FINDINGS.md
Demand models learned by imitation from urban mobility data inherit the demographic
service inequity encoded in their demonstrations. We present FAMAIL, a fairness-oriented
data-augmentation method that \emph{edits}, rather than regenerates, a small
attribution-targeted slice of real taxi trajectories, then upweights the edited
demonstrations so the fairness survives training. The editor combines two bounded,
gradient-guided mechanisms under one differentiable objective: \emph{trim} relocates
pickups out of over-served hotspots, and \emph{lift} reroutes drivers' final seeking
minutes into under-served areas, with the supply consequences of each move made
differentiable and endogenous to the objective. A frozen driver-identity discriminator
bounds every edit to preserve realism. On two cities (Shenzhen; San Francisco), editing
improves established fairness measures that the objective never optimizes ---
demographic parity, disparate impact, and the Theil index --- and, unlike demand-only
editing, raises the under-served group's service ratio through a statistically robust
supply channel rather than by leveling down. Upweighted behavior cloning propagates the
data-level gains into trained policies edit-specifically: random-upweighting and
select-the-fairest controls are null, and a naive demographic-oversampling baseline
degrades fairness despite fabricating 10.5\% of the corpus.
\end{abstract}
```

- [ ] **Step 2: Compile + lint gates** (both exit 0)

- [ ] **Step 3: Word-count check**

Run: `cd /home/robert/FAMAIL/paper && awk '/\\begin\{abstract\}/,/\\end\{abstract\}/' main.tex | grep -v '^%' | wc -w`
Expected: ≈180–230.

- [ ] **Step 4: Commit**

```bash
cd /home/robert/FAMAIL
git add paper/main.tex
git commit -m "paper: draft abstract (T6, pending final results)"
```

---

### Task 10: Coherence pass + fresh-agent number/convention audit + fix wave

**Files:**
- Modify: `paper/sections/03_methodology.tex`, `paper/main.tex` (fixes only)

**Interfaces:**
- Consumes: everything above.
- Produces: the verified deliverable.

- [ ] **Step 1: Whole-section coherence read** — read `03_methodology.tex` top-to-bottom once; fix register shifts, duplicate definitions, dangling `\ref`s, notation drift against the plan's Notation table.

- [ ] **Step 2: Dispatch a fresh read-only audit agent** with exactly this brief:

> Read `paper/sections/03_methodology.tex` and the abstract in `paper/main.tex`. For EVERY number (including design constants) verify it against the named `% src:` file (read that file); flag any number without a `% src:` comment, any mismatch, and any violation of the conventions in `paper/README.md` (trim-only numbers outside ablation context; causality-claim language; "54%"; SF "beats"; supply numbers without a tier label; p-values without required context; product names). Also verify every `\cite{key}` exists in `paper/refs.bib` and every `\ref`/`\label` pair resolves. Report findings as a list (file, line, claim, verdict, evidence). Do NOT edit any file.

- [ ] **Step 3: Apply the fix wave** — fix every confirmed finding; for disputed findings, record disposition in the commit message rather than silently dropping.

- [ ] **Step 4: Final gates**

Run: `cd /home/robert/FAMAIL/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex && bash lint.sh && grep -ci 'undefined' main.log`
Expected: compile + lint exit 0; the final `grep -ci` prints `0` (no undefined references/citations).

- [ ] **Step 5: Commit**

```bash
cd /home/robert/FAMAIL
git add paper/
git commit -m "paper: methodology audit fix wave (fresh-agent number/convention verification)"
```

---

## Execution notes

- **Order:** Tasks are strictly sequential (each consumes the previous task's labels/keys/notation).
- **The α-sweep dependency is non-blocking:** Task 4 ships with the `% TODO(alpha-sweep)` marker; the fold-in is a separate task outside this plan.
- **No GPU, no data access needed** — this is a writing plan; every source is a committed markdown/JSON artifact.
- **Reviewer guidance (for subagent-driven execution):** review = check the task's required-content list item-by-item against the written LaTeX + run both gates + verify provenance comments; prose taste fixes are fair game, factual drift is a rejection.
