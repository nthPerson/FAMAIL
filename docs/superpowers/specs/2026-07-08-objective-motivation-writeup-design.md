# Objective-Function Motivation Write-up — Design Spec

**Date:** 2026-07-08 · **Branch:** `objective-motivation-writeup` (worktree off `main` `1e51471`)
**Status:** approved (brainstorming) → ready for writing-plans
**Mission:** Meeting-41 P0 #2 — the literature-grounded "why + how" motivation for the FAMAIL objective function.

---

## 1. Context & motivation

FAMAIL's objective (`L = α_spatial·F_spatial + α_causal·F_causal + α_fidelity·F_fidelity`, maximized,
causal-emphasis `α=(0.2,0.7,0.1)`) and its ST-iFGSM editor are described *operationally* in
`PAPER/argument/03_fairness_theory.md` (formulas, intuition, caveats) — but that doc's "External lineage"
section explicitly says **"exact bibliographic references to be finalized by the authors"** and lists
cGAIL / HuMID / FGSM / FWL as bare names. This mission produces that missing *why* — a paper-ready,
literature-grounded motivation for each objective component and the editor, with a **verified** reference
apparatus and reviewer rebuttals.

The literature was gathered by an external deep-research session and then **citation-verified** (2026-07-08,
5-agent read-only web pass). The verification is authoritative: it caught **fabricated content** (a Zheng
"67%/2.3%" quote confabulation; a Corbett-Davies misquote) and several metadata errors, and confirmed the
rest. All citation facts in this build come from that audit, **not** from the raw research report's
unverified strings.

There is **no LaTeX manuscript yet** — the "paper" is the `PAPER/argument/` doc-set + results bundles — so
the deliverable is a durable *paper-source* bundle, not a `.tex` insert.

## 2. Goals / non-goals

**Goals**
- A self-contained `PAPER/objective-motivation/` bundle: paper-ready per-component "why + how" prose,
  reviewer defense, the leveling-down + demand-endogeneity framing, and a verified reference list.
- Light back-fill of the argument set: fill `03`'s "references to be finalized" placeholder; add a
  demand-endogeneity limitation note to `07`.
- Every citation audit-correct; every empirical claim traceable to an existing authoritative doc.

**Non-goals**
- **No new numbers or experiments.** The α-Pareto grid-search sweep is a *later, separate* run (GPU busy);
  this write-up justifies the weights from existing records and flags the sweep as reported/planned
  sensitivity.
- No rewrite of the curated argument narrative beyond the two targeted back-fills.
- Not committing the raw external research report into `PAPER/` (it stays an untracked working input).
- No merge to `main` — that is a separate step the user approves later.

## 3. Deliverable structure

```
PAPER/objective-motivation/
  README.md              role, relation to argument/03, provenance (citations verified 2026-07-08)
  MOTIVATION.md          paper-ready why+how per component  (the text that becomes paper prose)
  REVIEWER_DEFENSE.md    anticipated objection → rebuttal, per component
  LEVELING_DOWN.md       ethics + fair-ML framing + demand-endogeneity legitimacy (xref, no dup)
  REFERENCES.md          consolidated verified/corrected reference list (single citation source-of-truth)
```
plus back-fill: `PAPER/argument/03_fairness_theory.md`, `PAPER/argument/07_limitations.md`.

## 4. Per-file specification

### 4.1 README.md
What the bundle is; its relation to `argument/03` (this = *why*; 03 = *what/formula*); reading order;
a provenance note stating citations were verified 2026-07-08 against arXiv/DOI/ACM/IEEE/DBLP/Crossref and
that `REFERENCES.md` is the single citation source-of-truth. Cross-links to `argument/03`, `argument/04`–`05`
(Pillar 2), `external-metrics/LEVELING_DOWN_MECHANISM.md`.

### 4.2 MOTIVATION.md
Paper-ready prose. Structure — executive thesis, then one subsection per component. Depth ∝ novelty:
- **Executive thesis** — the one-paragraph "why + how" for the whole objective.
- **F_causal** (DEEP) — demand-adjusted demographic fairness. Why demand-adjustment before attributing
  disparity (conditional statistical parity, Corbett-Davies 2017, building on Kamiran 2013/Dwork 2012);
  how = residualize-then-project = **FWL** (Frisch-Waugh 1933; Lovell 1963); fairness-as-predictability
  (Feldman 2015); exact per-unit attribution as the edit selector. **Include a short honest paragraph on
  demand being *legitimate but endogenous*** (pointer to §4.4 / LEVELING_DOWN.md). Associational-not-causal
  + ecological caveats stated once.
- **F_spatial** (moderate) — Gini over supply-normalized service; transportation-equity practice
  (Hörcher & Graham 2021; Karner et al. 2024); Theil/Atkinson as alternatives; differentiable pairwise form.
- **F_fidelity** (moderate) — frozen driver-identity discriminator as realism regularizer; ST-SiameseNet/
  HuMID (Ren et al. 2020) + TUL lineage; frozen-vs-adversarial rationale; ≈0-gradient honesty + the JS
  distributional backstop.
- **ST-iFGSM editor + soft discretization** (DEEP) — iFGSM (Goodfellow 2015; Kurakin 2017) repurposed
  constructively via algorithmic recourse (Ustun 2019; Wachter 2018); ε-L∞ as identity-preservation budget;
  Gumbel-softmax/straight-through (Jang 2017; Maddison 2017; Bengio 2013) + 5×5 supply-matched smoothing + τ.
  Cite the group's own **ST-iFGSM (Hu et al., KDD 2023)**.
- **Downstream upweighting** (BRIEF, per user) — edit-then-upweight as pre-processing/data-augmentation
  fairness (Kamiran & Calders 2012 reweighing; Feldman 2015 disparate-impact removal); one-paragraph
  motivation, then defer Pillar-2 *results* to `argument/04`–`05`.
- **"Why these weights" — the scalarization** — linear scalarization of competing fairness/realism
  objectives; justify `(0.2,0.7,0.1)` from the recorded selection (see §5.2): the criterion, the
  before/after numbers, and the gradient-dominance facts. Present the α-Pareto sweep as **reported/planned
  sensitivity** — write the paragraph so it is complete and honest *whether or not* the sweep later runs
  (no dangling placeholder, no "TODO").

Each component subsection = *supporting literature (verified cites) → drafted why+how paragraph →
contrast/novelty ("to our knowledge")*. Draft source = the research report's per-component paragraphs
(§7 below), corrected per the audit.

### 4.3 REVIEWER_DEFENSE.md
Objection → literature-grounded rebuttal, per component. Cover at least: "F_causal isn't causal"
(concede: associational; rename to F_demo pending); ecological fallacy (10 district DOF); "why Gini not
Theil/Atkinson" (report Theil/Atkinson robustness); "frozen discriminator can be gamed" (ε-bound is the
real limiter); "iFGSM is just an attack" (recourse legitimizes constructive reuse); "ε=2 arbitrary" (tie
to driver signature + supply window; ε-sensitivity); "upweighting is ad hoc" (reweighing + random/
select-already-fair controls); **leveling-down** (→ LEVELING_DOWN.md); **demand-endogeneity** (§5.3);
**fixed weights** (§5.2). A fairness reviewer *will* raise leveling-down and demand-endogeneity — these
get the fullest rebuttals.

### 4.4 LEVELING_DOWN.md
The ethics + fair-ML *framing* (NOT the empirical proof — that lives in
`external-metrics/LEVELING_DOWN_MECHANISM.md`; cross-reference, do not duplicate numbers). Content:
Parfit (leveling-down objection) + Temkin; Mittelstadt et al. (leveling-up / minimum-rate constraints);
Zietlow et al. (augmentation was the one strategy that helped the disadvantaged) → ties to FAMAIL's
augmentation stance; Pinzón et al. (leveling-down can be constraint-forced) as an *analogy* (not a direct
theorem about FAMAIL — FAMAIL's own oracle/structural bound is load-bearing). **Then the demand-endogeneity
legitimacy argument (§5.3)** as the unifying thread: the metric's blind spot and the editor's leveling-down
are the same phenomenon; supply-side lever is the future direction. Frame over-service reduction under a
frozen-supply constraint as a principled constrained optimum.

### 4.5 REFERENCES.md
Consolidated verified reference list, grouped by theme (fairness metrics; transportation equity; imitation
learning / TUL / fidelity; adversarial / recourse / discretization; egalitarian ethics + leveling-down),
each flagged foundational vs recent. **Every entry uses the audit-corrected metadata (§5.1).** This is the
single source-of-truth other files cite against.

### 4.6 argument/03_fairness_theory.md (back-fill — light)
Replace the "External lineage … references to be finalized by the authors" bullet block with the verified
citations (cGAIL = Zhang et al. ICDM'19 / *IEEE Trans. Big Data* 2022; ST-SiameseNet/HuMID = Ren et al.
KDD'20; FGSM/iFGSM = Goodfellow'15 / Kurakin'17; **ST-iFGSM = Hu et al. KDD'23**; FWL = Frisch-Waugh'33 /
Lovell'63) + a one-line cross-link to `../objective-motivation/`. **Do not otherwise edit 03** (it is
tight, number-authoritative, and adversarially reviewed).

### 4.7 argument/07_limitations.md (back-fill — light)
Add a short **demand-endogeneity** limitation note (recorded demand is suppressed by historical
under-supply, so demand-adjustment can under-detect latent inequity; ties to leveling-down) + a pointer to
`../objective-motivation/LEVELING_DOWN.md`.

## 5. Ground-truth ledgers (embedded so the build is self-contained)

### 5.1 Verified / corrected citations (from `mission_2_citation_audit.md`)
**Must-fix content:**
- **Zheng et al. 2023** — "Fairness-Enhancing Deep Learning for Ride-Hailing Demand Prediction," *IEEE Open
  J. Intell. Transp. Syst.* 4:551–569, DOI 10.1109/OJITS.2023.3297517. Model SA-Net, Chicago TNC. The
  **"67%" and "2.3%" figures are FABRICATED — do NOT use.** Cite the paper's actual absolute result: MPE
  gap black vs non-black **drops 0.361 → 0.084**. Any percentage is *our* derivation, labeled as such, never
  quoted.
- **Corbett-Davies et al. 2017** — KDD 2017, pp. 797–806. Quote-b ("Conditional statistical parity requires
  that one define the 'legitimate' factors ℓ(X)…") is **verbatim, keep**. Quote-a is a **misquote** — the
  paper says "prior convictions," not "previous arrests." Use the verbatim text; say conditional statistical
  parity is **formalized** there (building on Kamiran 2013 / Dwork 2012), not "defined."

**Metadata corrections:**
- **cGAIL** = *IEEE Trans. Big Data* 8(5):1288–1300, **2022** (NOT TKDE) + conference version IEEE **ICDM
  2019** pp. 1480–1485; authors Zhang, Li, Zhou, Luo.
- **ST-iFGSM** = Hu, Zhang, Li, Zhou, Luo, **KDD 2023**, pp. 764–774, DOI 10.1145/3580305.3599513.
- **Zietlow et al. 2022** = CVPR 2022, pp. **10400–10411** (not 10410–10421), arXiv:2203.04913.
- **Mittelstadt, Wachter & Russell** = *Michigan Tech. Law Rev.* 30(1), **2024** (arXiv:2302.02404 = 2023).
- **Wachter, Mittelstadt & Russell** = *Harvard J. Law & Tech.* 31(2):841–887, **2018**.
- **Karner et al.** = *Transportation* 52:1399–1427, online 2024 / print 2025, DOI 10.1007/s11116-023-10460-7.
- **DROP "Wilms & Heitz (FAccT 2026)"** — real paper but contains no leveling-down content; not an anchor.

**Confirmed as cited** (safe): FWL (Frisch-Waugh 1933 *Econometrica* 1(4):387–401; Lovell 1963 *JASA*
58(304):993–1010), Feldman 2015 (KDD, BER↔disparate-impact), Kamiran & Calders 2012 (*KAIS* 33(1)),
Verma & Rubin 2018, Barocas-Hardt-Narayanan (2019 web / 2023 MIT Press), Pinzón et al. 2022 (AAAI
36(7):7993–8000), Parfit ("Equality and Priority," *Ratio* 10(3):202–221, 1997; "Equality or Priority?"
1991 Lindley Lecture), Temkin (*Inequality*, OUP 1993; chapter pp. 126–161, 2000), Hörcher & Graham 2021,
Atkinson 1970, Theil 1967, De Maio 2007, Goodfellow 2015 (arXiv:1412.6572), Kurakin 2017 (arXiv:1607.02533),
Jang 2017 / Maddison 2017 / Bengio 2013, Ustun 2019, Karimi 2020/2021, ST-SiameseNet (Ren et al. KDD 2020),
TULER (Gao IJCAI'17) / TULVAE (Zhou IJCAI'18) / DeepTUL (Miao AAMAS'20), GAIL (Ho & Ermon NeurIPS'16),
xGAIL (Pan KDD'20), Feng et al. 2020 (KDD).

### 5.2 Weight (α) justification — recorded facts
- **Selection criterion (matches the search intent):** `α=(0.2,0.7,0.1)` is "a balanced multi-objective
  (spatial + fidelity terms active) that **matches pure-causal gain without gaming a single metric**,
  F_spatial flat" (`baselines/STATUS.md:41`); it **superseded the pure-causal `(0,1,0)` headline** (commit
  `325b531`, 2026-05-27, "adopt validated causal-emphasis edit config"). = *maximize ΔF_causal s.t.
  ΔF_spatial ≥ 0*.
- **Numbers:** shipped config ΔF_causal **+0.0128**, ΔF_spatial **+0.0003**
  (`results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`); earlier +0.0087/+0.0093 with F_spatial
  flat at k=1k/10k.
- **Mechanism rationale (why causal-heavy is principled):** F_causal drives ~**97.5%** of gradient-sign
  decisions; F_spatial's gradient is ~**20× smaller** (never wins); F_fidelity is **dormant** at ε=2.
- **Not yet run:** a full multi-combination α-Pareto table (methodology doc **R3** lists it as open;
  supply-lift spec notes a "grid-search protocol exists" = capability, not a committed table). → present as
  reported/planned sensitivity.

### 5.3 Demand-endogeneity argument (the write-up's key value-add)
`F_causal` conditions out demand, treating it as a *legitimate* factor. But recorded demand (pickups) in
under-served areas is itself **suppressed by historical under-supply** (latent/censored demand; feedback
loop), so conditioning on demand can **launder away real inequity**. Smoking gun: **93% of poor-area units
sit at/below `DEMAND_FLOOR`** (`LEVELING_DOWN_MECHANISM.md`) → F_causal sees ≈no residual there → never
selects them. The metric's blind spot and the editor's leveling-down are the **same phenomenon**. Cite the
feedback-loop literature: **Ensign et al. 2018** (Runaway Feedback Loops); **Lum & Isaac 2016** (To Predict
and Serve?). *(These two are additions beyond the research report — verify them in the same audited manner
before finalizing REFERENCES.md.)* Addressing this head-on unifies metric + leveling-down + supply-side
future work and pre-empts a predictable fairness-reviewer objection.

## 6. Content rules & conventions
- Citations **only** from §5.1 (audit-corrected). No unverified strings from the raw research report.
- Ensign 2018 / Lum & Isaac 2016 (§5.3) are new → verify before inclusion in REFERENCES.md.
- Novelty framed "to our knowledge"; novelty #1 (FWL-partial-R²) softened (adjacent precedent exists).
- **No new experimental numbers**; all empirical claims point to `05`/`06`, `external-metrics/`,
  `second-dataset/`, or `baselines/STATUS.md`/methodology doc for the weight facts.
- Keep the associational-`F_causal` caveat wherever the metric is characterized.
- Match the argument set's tone: focused docs, cross-links, provenance; no authoring-tool names.

## 7. Source materials (untracked, in the MAIN working tree — read by absolute path)
- `/home/robert/FAMAIL/mission_2_context.md` — the brief (objective defs + design decisions).
- `/home/robert/FAMAIL/supporting_literature_and_why+how_FAMAIL_objective_function.md` — the research
  report; **draft source** for the MOTIVATION.md paragraphs (adapt + correct per audit).
- `/home/robert/FAMAIL/mission_2_citation_audit.md` — the citation corrections (§5.1 is its digest).
- In-repo: `PAPER/argument/03`,`04`,`05`,`07`; `PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md`;
  `famail_temporal/baselines/STATUS.md`; `famail_temporal/docs/TRAJECTORY_EDITING_METHODOLOGY.md`.

## 8. Environment / process
- Worktree `.claude/worktrees/objective-motivation-writeup`, branch `objective-motivation-writeup` off
  `main` `1e51471`. Keeps everything off `supply-lift-editing`; needs **no GPU**.
- Implement via subagent-driven development after writing-plans. Commit on the branch; **no merge to main**
  without user approval.

## 9. Risks & self-review notes
- **Citation drift** — mitigated: §5.1 is the locked ledger; anything new (Ensign/Lum & Isaac) is verified
  before use.
- **Duplicating the leveling-down proof** — avoided by design: LEVELING_DOWN.md is *framing only* and
  cross-references the mechanism doc.
- **Weight subsection reading as incomplete** — write it standalone from §5.2; the α-sweep is "reported
  sensitivity," not a placeholder.
- **Bloating argument/03** — back-fill is limited to replacing the placeholder block + one cross-link.

## 10. Success criteria
- The 5-file bundle exists, internally consistent, every citation audit-correct, no fabricated quote/number.
- `argument/03` placeholder replaced with verified cites + cross-link; `07` has the demand-endogeneity note.
- MOTIVATION.md paragraphs are drop-in-with-light-editing paper prose for the objective-motivation section.
- Demand-endogeneity is developed (F_causal + LEVELING_DOWN + 07) as the unifying limitation/defense.
- No new experiments; the α-sweep is cleanly flagged as planned sensitivity.
