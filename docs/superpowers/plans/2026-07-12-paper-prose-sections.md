# Paper Prose Sections (Intro / Related Work / Conclusion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draft `paper/sections/01_introduction.tex`, `02_related_work.tex`, and
`05_conclusion.tex` per the approved spec, then verify every citation-claim pair with Opus
subagents, so the manuscript's prose is complete except for run-gated numbers.

**Architecture:** Three drafting tasks (intro → related work → conclusion, in that order so
the intro sets the register), followed by a citation-verification task (Opus subagents via
the Agent tool, one batch per section) and a whole-paper consistency gate. Prose "tests" =
the build gate (`latexmk` + `lint.sh`) plus targeted grep sweeps defined per task.

**Tech Stack:** LaTeX (acmart sigconf), `latexmk`, `bash lint.sh`, BibTeX (`refs.bib`, 41
verified entries), Agent tool with `model: "opus"` for citation verification.

## Global Constraints

Copied from `docs/superpowers/specs/2026-07-12-paper-prose-sections-design.md` — every task
inherits all of these:

- **Source of truth = drafted abstract (`paper/main.tex:38-65`) + §3 + §4.** `PAPER/argument/`
  docs are pre-supply-lift skeleton ONLY; never copy a number or mechanism description from
  them (their +0.0144/+0.0139 headlines are lint-banned).
- **Committed Shenzhen numbers (α\* = (0.1, 0.8, 0.1), promoted s10 corpus), safe to hard-code:**
  editor ΔF_causal **+0.0226**; DI Δ **+0.0162**; lift-up tier-1 **+0.0176** / tier-2 **+0.0411**.
  Any OTHER number needs a `% src:` trace or a run-wired `TODO`.
- **Every SF value is a slot:** `X.XXXX % TODO(run:<stage> -> <artifact path>)` using the same
  paths §4's markers use.
- **Numbered contributions; the word "pillar" never appears in the manuscript.**
- **All 10 `paper/README.md` conventions + `bash lint.sh` must pass** (trim+lift centers
  reporting; no causality-claim language; SF *reproduces*, never "beats"; three-ring firewall
  vocabulary — "improves metrics we never optimized" claims ride ring (iii) only; every
  load-bearing number carries `% src:`; no product names).
- **Terminology locks:** *trim*, *lift*, active unit, demand deficit attribution,
  supply-gradient attribution, value-of-presence map, service ratio $Y$, leveling down /
  lift-up.
- **Voice:** match the abstract and §3. Per `feedback_paper_prose_style.md` memory: explicit
  referents (no dangling "throughout/this/baseline"), no AI-sounding flourishes, no coinages,
  words over notation that skims like a typo.
- **New `refs.bib` entries:** allowed only when no existing entry serves; tag the entry
  `% verify-pending` on creation; Task 4 must clear every tag. F_demo rename and "54%" figure
  never appear.
- **Length targets:** §1 ≈ 1.2 column, §2 ≈ 0.7 column, §5 ≈ 0.35 column; whole paper ≤ 9 pp
  (current build: 7 pp with stubs; `review` option adds line numbers, so page count is
  indicative, not a gate).

---

### Task 1: Draft §1 Introduction

**Files:**
- Modify: `paper/sections/01_introduction.tex` (replace the 6-line stub entirely; keep
  `\section{Introduction}\label{sec:intro}`)

**Interfaces:**
- Consumes: abstract (`paper/main.tex:38-65`); §3 terminology (`sections/03_methodology.tex`);
  bib keys listed in Step 2.
- Produces: the contribution list C1–C4 (Task 3's conclusion restatement mirrors it verbatim
  in structure); the intro's SF slot-marker strings (Task 5 sweeps them).

- [ ] **Step 1: Read the current-era sources (skeleton last)**

Read, in order: `paper/main.tex:38-65` (abstract — the voice anchor);
`PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md` (the 2,455/2,455 flow finding);
`PAPER/supply-lift/FINDINGS.md` §1-2 (lift-up headline + tier accounting);
`sections/03_methodology.tex:181-260` (attribution + leveling-down vocabulary);
`PAPER/argument/01_motivation_goals.md` (¶1 skeleton ONLY — no numbers from it).

- [ ] **Step 2: Locate the SF slot paths and the +0.0226 provenance comment**

Run: `grep -n "0.0226" paper/sections/04_experiments.tex` → copy that line's `% src:` comment
verbatim for the intro's use of +0.0226.
Run: `grep -n "r1\b" famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md` → note r1's output
directory; the intro's SF editor-delta slot marker is
`X.XXXX % TODO(run:r1 -> <that output dir>/metrics.json)` unless
`grep -n "San Francisco" -B2 -A4 paper/sections/04_experiments.tex` reveals §4 already has an
SF editor-delta marker — if it does, copy §4's marker string exactly.

- [ ] **Step 3: Write the section**

Five paragraphs + contribution list. Must-hit beats per paragraph (write full prose, not
these bullets):

1. *Problem.* Service allocation in real fleets correlates with neighborhood demographics;
   demand models learned by imitation \cite{zhang2022cgail} reproduce it; deployed models
   can amplify it via the allocation feedback loop \cite{ensign2018,lumisaac2016}. No
   numbers.
2. *Why the data side; why edit.* Model-side mitigation fights the training signal
   \cite{zheng2023}; generation rewrites the whole distribution and risks losing realism;
   FAMAIL edits ≤ k attribution-targeted real trajectories inside an ε = 2 ball, generates
   nothing, and upweights the edited slice downstream. This paragraph carries the paper's
   identity claim (title language: "trajectory editing as data augmentation").
3. *The turn.* Demand-only editing improves fairness measures by leveling down — on Shenzhen
   every one of the 2,455 demand-only edits originated in advantaged cells
   (\cite{parfit1997,mittelstadt2024}; `% src: PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md`);
   this finding motivated lift: differentiable, endogenous supply. Numbers here: SZ
   ΔF_causal +0.0226 (src from Step 2); lift-up +0.0176 (tier-1) / +0.0411 (tier-2) with
   `% src: PAPER/supply-lift/FINDINGS.md` and tier vocabulary per README convention #7; SF
   ΔF_causal slot from Step 2.
4. *Transfer + external validation.* Vanilla BC null; upweighting recovers edit-specifically
   (random + select-fairest controls null) \cite{kamirancalders2012}; external measures never
   optimized improve — demographic parity, disparate impact (SZ DI Δ +0.0162,
   `% src: PAPER/external-metrics/FINDINGS.md`), Theil \cite{theil1967}; oversampling baseline
   captures only part of the gain at 10.5\% fabrication
   (`% src: PAPER/baselines/demographic-oversampling/FINDINGS.md`).
5. *Contributions* (`\begin{itemize}` or enumerated, C1–C4 as in the spec):
   C1 trim+lift editor (bounded, gradient-guided, supply endogenous; demand deficit
   attribution + supply-gradient attribution); C2 leveling-down diagnosis + supply-channel
   remedy; C3 upweighted-imitation recipe + edit-specificity controls; C4 two-city validation
   on never-optimized measures + oversampling-baseline comparison.

- [ ] **Step 4: Build and lint**

Run: `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex 2>&1 | tail -3 && bash lint.sh && echo LINT PASS`
Expected: `Output written on main.pdf`, `LINT PASS`, exit 0.

- [ ] **Step 5: Section sweeps**

Run: `grep -in "pillar" paper/sections/01_introduction.tex` → expect empty.
Run: `grep -c "TODO(run:" paper/sections/01_introduction.tex` → expect ≥ 1 (the SF slot(s)).
Run: `grep -in "0\.0144\|0\.0139\|54" paper/sections/01_introduction.tex` → expect empty.
Run: `grep -i "undefined" paper/main.log` → expect empty (all cite keys resolve).

- [ ] **Step 6: Commit**

```bash
git add paper/sections/01_introduction.tex
git commit -m "paper: draft introduction (framework lede, leveling-down turn, C1-C4)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Draft §2 Related Work

**Files:**
- Modify: `paper/sections/02_related_work.tex` (replace the 7-line stub; keep
  `\section{Related Work}\label{sec:related}`)
- Modify (only if a new entry is unavoidable): `paper/refs.bib` (tag `% verify-pending`)

**Interfaces:**
- Consumes: `PAPER/objective-motivation/MOTIVATION.md` (per-term "Contrast / novelty"
  blocks); `PAPER/objective-motivation/REFERENCES.md` (groupings); the 41 existing bib keys.
- Produces: the final citation set Task 4 verifies; any `% verify-pending` tags.

- [ ] **Step 1: Read sources**

Read `PAPER/objective-motivation/MOTIVATION.md` (contrast blocks) and
`PAPER/objective-motivation/REFERENCES.md` in full; skim `sections/03_methodology.tex` cite
sites (`grep -n "cite{" paper/sections/03_methodology.tex`) to avoid repeating its sentences.

- [ ] **Step 2: Write the section**

Five paragraphs, each ending with a one-sentence FAMAIL contrast. Paragraph → key map
(cite keys are existing `refs.bib` entries):

1. *Fairness interventions in ML* — kamirancalders2012, feldman2015, corbettdavies2017,
   vermarubin2018, barocas2023. Contrast: pre-processing transplanted to imitation learning;
   the intervention edits demonstrations, not features or labels.
2. *Fairness in urban mobility / transportation equity* — zheng2023 (closest applied
   neighbor: in-processing regularization for ride-hailing demand prediction),
   horchergraham2021, karner2024, theil1967, atkinson1970, demaio2007. Contrast: FAMAIL moves
   the intervention from the model to the demonstrations and evaluates on external measures
   the objective never optimizes.
3. *Imitation learning for mobility & trajectory identity* — zhang2019cgail, zhang2022cgail,
   pan2020xgail, feng2020simulate; gao2017tuler, zhou2018tulvae, miao2020deeptul,
   ren2020stsiamese; hoermon2016. Contrast: FAMAIL proposes no new generator — it edits the
   demonstrations such models train on, and repurposes the identity literature as a realism
   guardrail.
4. *Adversarial perturbation & recourse* — goodfellow2015fgsm, kurakin2017ifgsm,
   hu2023stifgsm, ustun2019recourse, wachter2018counterfactual, jang2017gumbel,
   maddison2017concrete, bengio2013ste. Contrast: the same bounded-perturbation machinery
   used constructively, with ε reinterpreted as an identity-preservation budget.
5. *Leveling-down ethics & feedback loops* — parfit1997, mittelstadt2024, zietlow2022,
   ensign2018, lumisaac2016. Contrast: FAMAIL operationalizes leveling up via the supply
   channel, and treats demand endogeneity as a stated bound on what any demand-adjusted
   metric can see.

If any paragraph genuinely needs an entry not in `refs.bib`, add it with complete fields and
a trailing `% verify-pending` comment on the `@` line — Task 4 clears it or the citation goes.

- [ ] **Step 3: Build and lint**

Run: `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex 2>&1 | tail -3 && bash lint.sh && echo LINT PASS`
Expected: `Output written on main.pdf`, `LINT PASS`.

- [ ] **Step 4: Section sweeps**

Run: `grep -i "undefined" paper/main.log` → expect empty.
Run: `grep -c "cite{" paper/sections/02_related_work.tex` → expect ~20–30 (breadth check).
Run: `grep -in "pillar\|beats" paper/sections/02_related_work.tex` → expect empty.

- [ ] **Step 5: Commit**

```bash
git add paper/sections/02_related_work.tex paper/refs.bib
git commit -m "paper: draft related work (5 themes, per-paragraph FAMAIL contrasts)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Draft §5 Conclusion

**Files:**
- Modify: `paper/sections/05_conclusion.tex` (replace the 5-line stub; keep
  `\section{Conclusion}\label{sec:conclusion}`)

**Interfaces:**
- Consumes: Task 1's C1–C4 list (restatement mirrors its structure); limitation items below.

- [ ] **Step 1: Write the section**

Three paragraphs:

1. *Restatement.* What FAMAIL is (edit + upweight, trim + lift) and what was shown, mirroring
   C1–C4 order; committed numbers only, or fully qualitative — no new numbers, no SF values.
2. *Limitations* (one compact paragraph, every clause already load-bearing elsewhere):
   $F_{\mathrm{causal}}$ is associational/ecological on ~10 district profiles; adjusting for
   recorded demand inherits demand endogeneity (suppressed demand where service was thin);
   small-$n$ significance floors mean the evidence rests on direction + magnitude + t-CIs +
   dose-response + controls rather than uncorrected p-values; SF *reproduces* the Shenzhen
   conclusions — city-specific baselines are not comparable; the fidelity guardrail certifies
   driver-identity preservation, not trajectory-shape realism.
3. *Future work.* (a) The unified one-pass editor — supply endogenous everywhere, the
   gradient choosing each edit's character — echoing §3.5's own closing text (do not repeat
   its sentence verbatim; `grep -n "unified single-pass" paper/sections/03_methodology.tex`
   to see the wording being echoed); (b) broader transfer: other imitation objectives,
   whether a GAN/WGAN trained on the edited corpus inherits the fairness, additional cities,
   allocation constraints on the training side.

- [ ] **Step 2: Build and lint**

Run: `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex 2>&1 | tail -3 && bash lint.sh && echo LINT PASS`
Expected: `Output written on main.pdf`, `LINT PASS`.

- [ ] **Step 3: Section sweeps**

Run: `grep -in "pillar\|F_demo\|f_demo" paper/sections/05_conclusion.tex` → expect empty.
Run: `grep -cn "X\.XXXX" paper/sections/05_conclusion.tex` → expect 0 (no slots in §5).

- [ ] **Step 4: Commit**

```bash
git add paper/sections/05_conclusion.tex
git commit -m "paper: draft conclusion (restatement, limitations recap, future work)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Citation verification (Opus subagents)

**Files:**
- Create: `paper/reviews/2026-07-12-prose-citation-audit.md`
- Modify (fixes): `paper/sections/01_introduction.tex`, `02_related_work.tex`,
  `05_conclusion.tex`, `paper/refs.bib`

**Interfaces:**
- Consumes: every `\cite{...}` instance in the three new sections + its full surrounding
  sentence; the matching `refs.bib` entries.
- Produces: the audit file; zero `% verify-pending` tags remaining.

- [ ] **Step 1: Extract citation-claim pairs**

For each of the three sections, list every `\cite` with its full sentence:
`grep -n "cite{" paper/sections/01_introduction.tex paper/sections/02_related_work.tex paper/sections/05_conclusion.tex`
Then, for each hit, copy the complete sentence (not the line — the sentence) and the cite
key(s) into a working list in the scratchpad, one block per section. Pull each cited key's
full BibTeX entry from `refs.bib` into the same block.

- [ ] **Step 2: Dispatch one Opus verification subagent per section (3 agents, parallel)**

Use the Agent tool, `subagent_type: "general-purpose"`, `model: "opus"`, one call per
section, all three in one message so they run concurrently. Prompt template (fill the
placeholders with the section's blocks from Step 1):

```
You are verifying citations for a KDD paper section. For EACH citation-claim pair below,
using WebSearch/WebFetch against publisher pages, DBLP, arXiv, or Semantic Scholar:

1. EXISTENCE: confirm the work exists and the BibTeX fields (authors, title, venue, year,
   pages) are correct. Report any field mismatch.
2. CLAIM SUPPORT: the sentence below cites this work for a specific claim. Confirm the work
   actually contains/supports that claim. Quote the supporting passage (abstract or body)
   and give its URL. If you cannot find affirmative evidence, say so plainly — do NOT
   extrapolate from the title.

Default skeptical: if uncertain after searching, verdict is UNSUPPORTED. This project
previously caught fabricated citations; your job is to refute, not to confirm.

Return a markdown table: | key | existence (OK/FIX:field/NOT FOUND) | claim support
(SUPPORTED/PARTIAL/UNSUPPORTED) | evidence quote + URL | notes |

Pairs to verify:
[PASTE the section's citation-claim blocks: sentence, cite key(s), BibTeX entry]
```

- [ ] **Step 3: Consolidate the three tables into the audit file**

Write `paper/reviews/2026-07-12-prose-citation-audit.md`: header (date, scope = §1/§2/§5,
verifier = Opus subagents, protocol = existence + claim-support), the three tables, and a
disposition column you fill in Step 4. Model on `mission_2_citation_audit.md` (repo root).

- [ ] **Step 4: Fix every non-SUPPORTED verdict**

For each `FIX:field` → correct `refs.bib`. For each `PARTIAL` → weaken or reword the claim
sentence until the evidence covers it. For each `UNSUPPORTED`/`NOT FOUND` → swap in a
verified citation or delete the citation and rewrite the sentence to stand uncited. Record
each disposition in the audit file. Remove every `% verify-pending` tag whose entry passed.

- [ ] **Step 5: Re-verify changed pairs**

If any sentence was reworded or any citation swapped in Step 4, dispatch one follow-up Opus
agent with just the changed pairs (same template). Repeat until every pair is SUPPORTED or
the citation is gone.

- [ ] **Step 6: Gate and commit**

Run: `grep -rn "verify-pending" paper/refs.bib` → expect empty.
Run: `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex 2>&1 | tail -3 && bash lint.sh && echo LINT PASS` → expect pass.

```bash
git add paper/reviews/2026-07-12-prose-citation-audit.md paper/sections/ paper/refs.bib
git commit -m "paper: citation audit for prose sections (Opus-verified, all pairs supported)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Whole-paper consistency gate

**Files:**
- Modify (fixes only): `paper/sections/*.tex`

**Interfaces:**
- Consumes: all prior tasks' output.
- Produces: the final pre-handoff build.

- [ ] **Step 1: Terminology and convention sweeps (whole manuscript)**

Run each; every one must return empty (or only `% lint-allow` lines):
`grep -in "pillar" paper/sections/*.tex`
`grep -in "what-if\|setminus\|heuristic saliency" paper/sections/*.tex`
`grep -inE "F_demo|f_demo" paper/sections/*.tex`
`grep -inE "deficit attribution" paper/sections/*.tex | grep -iv "demand deficit"` (line-wrap
continuations of "demand ⏎ deficit attribution" are the one acceptable hit — check context)
`bash paper/lint.sh` (from `paper/`: `bash lint.sh`)

- [ ] **Step 2: Number-provenance sweep**

Every numeral in §1/§2/§5 either matches a Global-Constraints committed value with a `% src:`
comment, or is inside a `TODO(run:...)` slot:
`grep -nE "[0-9]+\.[0-9]+" paper/sections/01_introduction.tex paper/sections/02_related_work.tex paper/sections/05_conclusion.tex`
Inspect each hit against the rule; fix stragglers.

- [ ] **Step 3: Voice pass**

Reread the three sections once, aloud-style, against the four checks in
`feedback_paper_prose_style.md`: dangling referents; AI flourishes; coinages; notation that
skims as a typo. Fix inline.

- [ ] **Step 4: Final build + page count**

Run: `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex 2>&1 | grep "Output written"`
Expected: `Output written on main.pdf (N pages, ...)` with N ≤ 10 in review mode (line
numbers inflate; flag N > 10 to Robert rather than cutting content unilaterally).

- [ ] **Step 5: Commit**

```bash
git add paper/
git commit -m "paper: consistency gate over intro/related-work/conclusion

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
