# FAMAIL Objective-Function Motivation Write-up — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the `PAPER/objective-motivation/` bundle (paper-ready "why + how" motivation for the FAMAIL objective, verified citations, reviewer defense, leveling-down + demand-endogeneity framing) and lightly back-fill `argument/03` and `argument/07`.

**Architecture:** A documentation bundle — no code. Each task writes one Markdown deliverable (or one back-fill) and ends with a **runnable consistency check** (grep gates + citation-resolution) in place of unit tests, then a commit. `REFERENCES.md` is built first as the single citation source-of-truth; all other docs cite against it. Everything happens in the worktree, on branch `objective-motivation-writeup`; nothing merges to `main`.

**Tech Stack:** Markdown; `grep`/`bash` consistency checks; `git`. No build, no runtime, no GPU.

**Spec:** `docs/superpowers/specs/2026-07-08-objective-motivation-writeup-design.md` (read it — §5.1/§5.2/§5.3 are the ground-truth ledgers this plan draws on).

## Global Constraints

*(Every task implicitly includes these.)*

- **Worktree/branch:** work only in `/home/robert/FAMAIL/.claude/worktrees/objective-motivation-writeup`, branch `objective-motivation-writeup`. Run git as `git -C <worktree> …` or from the worktree root. **Never** touch `main` or `supply-lift-editing`; **no merge to main**.
- **Citations:** use **only** the audit-corrected metadata in spec §5.1. Two new cites — **Ensign et al. 2018** and **Lum & Isaac 2016** — must be web-verified (Task 1) before use. No unverified strings from the raw research report.
- **No new experimental numbers.** The α-Pareto sweep is *planned/reported sensitivity*, not run. All empirical claims trace to `PAPER/argument/05`–`06`, `PAPER/external-metrics/`, `PAPER/second-dataset/`, or `famail_temporal/baselines/STATUS.md` + methodology doc (weight facts, spec §5.2).
- **Forbidden strings** (must NOT appear anywhere in the bundle): `67%`, `2.3%` (the fabricated Zheng figures), `Wilms`/`Heitz`, `TKDE` (cGAIL is *IEEE Trans. Big Data*), `previous arrests` (the Corbett-Davies misquote → use "prior convictions"), and `TODO`/`TBD`/`FIXME`.
- **Novelty** claims phrased "to our knowledge"; novelty #1 (FWL-partial-R²) softened (adjacent precedent exists).
- **Associational-`F_causal` caveat** appears wherever the metric is characterized (it is a partial R² on ~10 district profiles; no identification/counterfactual; rename to `F_demo` pending).
- **No authoring-tool names** anywhere in `PAPER/` content.
- **Commit per task** on the branch. Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Source docs** (untracked at repo root when this plan was written; tracked 2026-07-14 under `PAPER/objective-motivation/sources/`):
  - `PAPER/objective-motivation/sources/mission_2_context.md` — the brief.
  - `PAPER/objective-motivation/sources/supporting_literature_and_why+how_FAMAIL_objective_function.md` — the research report; **draft source** for the why+how paragraphs (adapt + correct).
  - `PAPER/objective-motivation/sources/mission_2_citation_audit.md` — the citation corrections.

---

### Task 1: `REFERENCES.md` — the verified citation source-of-truth

**Files:**
- Create: `PAPER/objective-motivation/REFERENCES.md`

**Interfaces:**
- Consumes: spec §5.1 (audit-corrected metadata); `PAPER/objective-motivation/sources/mission_2_citation_audit.md`.
- Produces: the canonical reference list. Every downstream doc cites by **surname + year** and those must resolve here. Grouped headings other docs can point to.

- [ ] **Step 1: Web-verify the two new feedback-loop citations**

These are additions beyond the audited set (spec §5.3) and must be confirmed before inclusion. Use WebSearch/WebFetch (arXiv/ACM/DOI). Expected canonical metadata to confirm:
- **Ensign, Friedler, Neville, Scheidegger, Venkatasubramanian (2018)** — "Runaway Feedback Loops in Predictive Policing," *Conference on Fairness, Accountability and Transparency (FAT\*)* 2018, PMLR 81:160–171.
- **Lum & Isaac (2016)** — "To predict and serve?," *Significance* 13(5):14–19, DOI 10.1111/j.1740-9713.2016.00960.x.

Record the confirmed metadata; if either differs, use what the evidence shows.

- [ ] **Step 2: Write `REFERENCES.md`**

Consolidated list grouped by theme, each entry flagged **[foundational]** or **[recent]**. Use the exact audit-corrected metadata. Required groups and the non-negotiable corrected entries:

- **Fairness metrics & definitions:** Corbett-Davies et al. 2017 (KDD, pp. 797–806); Feldman et al. 2015 (KDD, pp. 259–268); Kamiran & Calders 2012 (*Knowl. Inf. Syst.* 33(1):1–33); Verma & Rubin 2018 (FairWare@ICSE); Barocas, Hardt & Narayanan (2019 web / 2023 MIT Press); Frisch & Waugh 1933 (*Econometrica* 1(4):387–401); Lovell 1963 (*JASA* 58(304):993–1010).
- **Transportation / spatial equity:** Hörcher & Graham 2021 (*Transportation* 48:2521–2544); Karner, Pereira & Farber 2024 (*Transportation* 52:1399–1427, DOI 10.1007/s11116-023-10460-7); Atkinson 1970 (*J. Econ. Theory* 2(3):244–263); Theil 1967 (*Economics and Information Theory*); De Maio 2007 (*JECH* 61(10):849–852); **Zheng et al. 2023** ("Fairness-Enhancing Deep Learning for Ride-Hailing Demand Prediction," *IEEE Open J. Intell. Transp. Syst.* 4:551–569, DOI 10.1109/OJITS.2023.3297517) — annotate: *report the absolute MPE-gap result 0.361→0.084; the "67%"/"2.3%" figures are not in the paper.*
- **Imitation learning / TUL / fidelity:** Ho & Ermon 2016 (NeurIPS, pp. 4565–4573, arXiv:1606.03476); **cGAIL** — Zhang, Li, Zhou, Luo, conf. IEEE **ICDM 2019** pp. 1480–1485 + journal *IEEE Trans. Big Data* **8(5):1288–1300, 2022** (DOI 10.1109/TBDATA.2020.3039810); xGAIL — Pan et al. KDD 2020 (pp. 1334–1343); ST-SiameseNet — Ren, Pan, Li, Zhou, Luo, KDD 2020 (pp. 1306–1315); TULER — Gao et al. IJCAI 2017; TULVAE — Zhou et al. IJCAI 2018; DeepTUL — Miao et al. AAMAS 2020; Feng et al. 2020 (KDD, "Learning to Simulate Human Mobility").
- **Adversarial / recourse / discretization:** Goodfellow, Shlens & Szegedy 2015 (ICLR, arXiv:1412.6572); Kurakin, Goodfellow & Bengio 2017 (ICLR Workshop, arXiv:1607.02533); **ST-iFGSM — Hu, Zhang, Li, Zhou, Luo, KDD 2023, pp. 764–774** (DOI 10.1145/3580305.3599513); Ustun, Spangher & Liu 2019 (FAT\*); Wachter, Mittelstadt & Russell **2018** (*Harvard J. Law & Tech.* 31(2):841–887); Karimi et al. 2020/2021; Jang, Gu & Poole 2017 (ICLR); Maddison, Mnih & Teh 2017 (ICLR); Bengio, Léonard & Courville 2013 (arXiv:1308.3432).
- **Egalitarian ethics & leveling-down:** Parfit 1997 ("Equality and Priority," *Ratio* 10(3):202–221) + Parfit 1991 Lindley Lecture ("Equality or Priority?"); Temkin 1993 (*Inequality*, OUP) + Temkin 2000 (chapter, pp. 126–161); Mittelstadt, Wachter & Russell **2024** (*Michigan Tech. Law Rev.* 30(1); arXiv:2302.02404); Zietlow et al. 2022 (CVPR, pp. **10400–10411**; arXiv:2203.04913); Pinzón et al. 2022 (AAAI 36(7):7993–8000).
- **Feedback loops / demand endogeneity:** Ensign et al. 2018 + Lum & Isaac 2016 (from Step 1).

Add a one-line header note: "Metadata verified 2026-07-08 (arXiv/DOI/ACM/IEEE/DBLP/Crossref); see the citation audit."

- [ ] **Step 3: Verify — forbidden strings absent, corrected entries present**

Run from the worktree root:
```bash
cd PAPER/objective-motivation
echo "forbidden (expect 0 each):"; grep -ciE 'Wilms|Heitz|TKDE|previous arrests|67%|2\.3%|TODO|TBD' REFERENCES.md
echo "required (expect >=1 each):"; \
  grep -c 'Big Data' REFERENCES.md; \
  grep -cE 'ST-iFGSM|Hu' REFERENCES.md; \
  grep -ciE 'Ensign' REFERENCES.md; \
  grep -ciE 'Lum' REFERENCES.md; \
  grep -cE '10400' REFERENCES.md
```
Expected: first count `0`; each required count `≥1`.

- [ ] **Step 4: Commit**

```bash
git add PAPER/objective-motivation/REFERENCES.md
git commit -m "docs(objective-motivation): verified reference list (citation source-of-truth)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `MOTIVATION.md` — paper-ready why+how per component

**Files:**
- Create: `PAPER/objective-motivation/MOTIVATION.md`

**Interfaces:**
- Consumes: `REFERENCES.md` (cite by surname+year); research report (draft paragraphs, absolute path); spec §5.1 (corrections), §5.2 (weight facts).
- Produces: the drop-in-with-light-editing paper prose for the objective-motivation section.

- [ ] **Step 1: Draft `MOTIVATION.md`**

Read the research report (`PAPER/objective-motivation/sources/supporting_literature_and_why+how_FAMAIL_objective_function.md`) for the drafted why+how paragraphs and adapt them, applying every spec-§5.1 correction. Structure (depth ∝ novelty):

1. **Executive thesis** — one paragraph: edit-don't-generate; demand-adjusted demographic fairness (FWL + conditional statistical parity); Gini spatial term; frozen identity discriminator; iFGSM-as-recourse editor; upweighting so fairness survives cloning.
2. **F_causal (DEEP)** — *supporting lit* → *why+how paragraph* → *contrast/novelty ("to our knowledge")*. Cite: Corbett-Davies et al. 2017 (conditional statistical parity — **formalized there, building on Kamiran 2013 / Dwork 2012**; use the verbatim quote-b, never "previous arrests"); Frisch & Waugh 1933 / Lovell 1963 (FWL = residualize-then-project); Feldman et al. 2015 (fairness-as-predictability); note exact per-unit attribution drives edit selection. Include the associational + ecological caveats **once**. Add a short honest paragraph: **demand is a *legitimate but endogenous* control** (pointer to `LEVELING_DOWN.md`).
3. **F_spatial (moderate)** — Gini over supply-normalized service; Hörcher & Graham 2021; Karner et al. 2024; Theil/Atkinson as reported alternatives; differentiable pairwise form; demographic-independent.
4. **F_fidelity (moderate)** — frozen driver-identity discriminator as realism regularizer; Ren et al. 2020 (ST-SiameseNet/HuMID) + TUL lineage (Gao'17/Zhou'18/Miao'20); frozen-vs-adversarial (GAN instability); honest ≈0-gradient at ε=2 + the JS distributional backstop; Feng et al. 2020 for JS-over-mobility-stats practice.
5. **ST-iFGSM editor + soft discretization (DEEP)** — Goodfellow 2015 / Kurakin 2017; constructive reuse via recourse (Ustun 2019; Wachter 2018); ε-L∞ reinterpreted as identity-preservation budget; Gumbel-softmax/straight-through (Jang 2017; Maddison 2017; Bengio 2013) + 5×5 supply-matched smoothing + τ; **cite ST-iFGSM = Hu et al. KDD 2023**.
6. **Downstream upweighting (BRIEF)** — edit-then-upweight as pre-processing/data-augmentation fairness (Kamiran & Calders 2012 reweighing; Feldman 2015 disparate-impact removal); one paragraph; defer Pillar-2 *results* to `argument/04`–`05`; may cite Zheng et al. 2023 as the applied bias this targets upstream (absolute-number framing only).
7. **Why these weights — the scalarization** — linear scalarization of competing objectives; justify `α=(0.2,0.7,0.1)` from spec §5.2: the selection criterion (*max ΔF_causal s.t. ΔF_spatial ≥ 0*, superseding pure-causal `(0,1,0)`, commit `325b531`), the numbers (ΔF_causal +0.0128, ΔF_spatial +0.0003), and the gradient-dominance facts (F_causal ~97.5% of sign decisions; F_spatial ~20× smaller; F_fidelity dormant at ε=2). Present the α-Pareto sweep as **reported/planned sensitivity** — the paragraph must read as complete whether or not the sweep runs (no placeholder).

- [ ] **Step 2: Verify — forbidden strings + caveat present + weight numbers exact**

```bash
cd PAPER/objective-motivation
echo "forbidden (expect 0):"; grep -ciE 'Wilms|Heitz|TKDE|previous arrests|67%|2\.3%|TODO|TBD' MOTIVATION.md
echo "associational caveat (expect >=1):"; grep -ciE 'associational' MOTIVATION.md
echo "weight facts (expect >=1 each):"; grep -cE '0\.0128' MOTIVATION.md; grep -cE '97\.5' MOTIVATION.md
```
Expected: forbidden `0`; caveat `≥1`; weight facts `≥1`.

- [ ] **Step 3: Commit**

```bash
git add PAPER/objective-motivation/MOTIVATION.md
git commit -m "docs(objective-motivation): per-component why+how motivation (paper-ready)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `REVIEWER_DEFENSE.md` — objection → rebuttal

**Files:**
- Create: `PAPER/objective-motivation/REVIEWER_DEFENSE.md`

**Interfaces:**
- Consumes: `REFERENCES.md`; research report ("Reviewer risks + rebuttals" blocks); spec §5.3.
- Produces: the rebuttal-prep companion (not paper prose).

- [ ] **Step 1: Draft `REVIEWER_DEFENSE.md`**

One "Objection → Rebuttal" entry per item, each rebuttal grounded in a `REFERENCES.md` cite. Required objections:
- "F_causal isn't causal" → concede associational; rename to `F_demo` pending; FWL is an algebraic identity about conditional association.
- "Ecological fallacy" → 10 district DOF; report as limitation; no individual-level claims.
- "Why Gini, not Theil/Atkinson?" → Gini standard/interpretable/parameter-free; report Theil/Atkinson robustness (Karner et al. 2024).
- "A frozen discriminator can be gamed" → the bounded ε-L∞ edit is the real limiter; JS-divergence is the collapse guard.
- "iFGSM is just an attack / a gimmick" → bounded signed-gradient ascent is method-agnostic; recourse legitimizes constructive reuse (Ustun 2019).
- "Is ε=2 arbitrary?" → tie to the driver signature + the 5×5 supply window; report fidelity/JS-vs-ε.
- "Upweighting is ad hoc" → reweighing (Kamiran & Calders 2012) + random / select-already-fair controls.
- "Leveling-down" → summarize; defer to `LEVELING_DOWN.md`.
- "Demand endogeneity" → the fullest rebuttal (spec §5.3); cite Ensign et al. 2018, Lum & Isaac 2016; tie to the 93%-at-`DEMAND_FLOOR` finding and the supply-side future lever.
- "Fixed weights are unjustified" → the recorded selection procedure (spec §5.2); α-sweep as planned sensitivity.

- [ ] **Step 2: Verify**

```bash
cd PAPER/objective-motivation
echo "forbidden (expect 0):"; grep -ciE 'Wilms|Heitz|TKDE|previous arrests|67%|2\.3%|TODO|TBD' REVIEWER_DEFENSE.md
echo "coverage (expect >=1 each):"; grep -ciE 'endogen' REVIEWER_DEFENSE.md; grep -ciE 'leveling|levelling' REVIEWER_DEFENSE.md
```
Expected: forbidden `0`; coverage `≥1`.

- [ ] **Step 3: Commit**

```bash
git add PAPER/objective-motivation/REVIEWER_DEFENSE.md
git commit -m "docs(objective-motivation): reviewer objections + rebuttals

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `LEVELING_DOWN.md` — ethics framing + demand-endogeneity

**Files:**
- Create: `PAPER/objective-motivation/LEVELING_DOWN.md`

**Interfaces:**
- Consumes: `REFERENCES.md`; spec §5.3; `PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md` (cross-ref target — **do not duplicate its numbers/proof**).
- Produces: the leveling-down/demand-endogeneity narrative; the pointer target for `argument/07`.

- [ ] **Step 1: Draft `LEVELING_DOWN.md`**

Sections:
1. **The objection (ethics).** Parfit (leveling-down objection; note the 1997 *Ratio* "Equality and Priority" vs. 1991 Lindley "Equality or Priority?" title split) + Temkin (non-instrumental egalitarianism).
2. **Algorithmic-fairness grounding.** Mittelstadt et al. 2024 (leveling-up / minimum-rate constraints); Zietlow et al. 2022 (augmentation was the one strategy that helped the disadvantaged → ties to FAMAIL's augmentation stance); Pinzón et al. 2022 as an **analogy** (leveling-down can be constraint-forced) — state that FAMAIL's own oracle/structural bound is the load-bearing formal claim, not Pinzón.
3. **FAMAIL's position.** Over-service reduction under a **frozen-supply** constraint is the constrained optimum, not an optimizer bug — cross-reference `../external-metrics/LEVELING_DOWN_MECHANISM.md` for the structural proof (cite it; don't restate the numbers). Supply-side lever = future direction.
4. **Demand endogeneity (the unifying thread).** Recorded demand is suppressed by historical under-supply → conditioning on it can launder inequity; the metric's blind spot and the editor's leveling-down are the same phenomenon; the 93%-at-`DEMAND_FLOOR` finding is the evidence. Cite Ensign et al. 2018, Lum & Isaac 2016.

- [ ] **Step 2: Verify — framing only (no duplicated proof), cross-ref present**

```bash
cd PAPER/objective-motivation
echo "forbidden (expect 0):"; grep -ciE 'Wilms|Heitz|TKDE|67%|2\.3%|TODO|TBD' LEVELING_DOWN.md
echo "cross-ref to mechanism doc (expect >=1):"; grep -c 'LEVELING_DOWN_MECHANISM' LEVELING_DOWN.md
echo "endogeneity developed (expect >=1):"; grep -ciE 'endogen' LEVELING_DOWN.md
```
Expected: forbidden `0`; cross-ref `≥1`; endogeneity `≥1`.

- [ ] **Step 3: Commit**

```bash
git add PAPER/objective-motivation/LEVELING_DOWN.md
git commit -m "docs(objective-motivation): leveling-down ethics framing + demand-endogeneity

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `README.md` — bundle guide

**Files:**
- Create: `PAPER/objective-motivation/README.md`

**Interfaces:**
- Consumes: the four docs from Tasks 1–4 (describes them accurately).
- Produces: entry point + provenance.

- [ ] **Step 1: Draft `README.md`**

Content: what the bundle is (the literature-grounded *why* behind the objective); its relation to `argument/03` (this = *why*; 03 = *what/formula*); reading order table (MOTIVATION → REVIEWER_DEFENSE → LEVELING_DOWN → REFERENCES); a provenance note (citations verified 2026-07-08; `REFERENCES.md` is the single citation source-of-truth; no new experimental numbers — the α-sweep is planned sensitivity). Cross-links to `../argument/03_fairness_theory.md`, `../argument/04_evaluation.md`, `../argument/05_results_shenzhen.md`, `../external-metrics/LEVELING_DOWN_MECHANISM.md`. No authoring-tool names.

- [ ] **Step 2: Verify — links resolve, forbidden absent**

```bash
cd PAPER/objective-motivation
echo "forbidden (expect 0):"; grep -ciE 'Wilms|Heitz|TKDE|67%|2\.3%|TODO|TBD' README.md
echo "referenced siblings exist:"; for f in MOTIVATION.md REVIEWER_DEFENSE.md LEVELING_DOWN.md REFERENCES.md; do test -f "$f" && echo "ok $f" || echo "MISSING $f"; done
echo "argument/03 link target exists:"; test -f ../argument/03_fairness_theory.md && echo ok || echo MISSING
```
Expected: forbidden `0`; all `ok`.

- [ ] **Step 3: Commit**

```bash
git add PAPER/objective-motivation/README.md
git commit -m "docs(objective-motivation): bundle README + provenance

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Back-fill `argument/03_fairness_theory.md`

**Files:**
- Modify: `PAPER/argument/03_fairness_theory.md` (the "External lineage" bullet block, ~lines 145–151)

**Interfaces:**
- Consumes: `REFERENCES.md` (for exact citation strings).
- Produces: verified lineage in the canonical argument doc + a cross-link to the bundle.

- [ ] **Step 1: Replace the placeholder block**

Locate the block that currently reads *"External lineage (as grounded in the repo's methodology docs; exact bibliographic references to be finalized by the authors):"* with its cGAIL / HuMID / FGSM / FWL bullets. Replace **only that block** with the verified citations:
- cGAIL — Zhang, Li, Zhou & Luo (IEEE ICDM 2019; journal *IEEE Trans. Big Data* 8(5):1288–1300, 2022).
- ST-SiameseNet / HuMID — Ren, Pan, Li, Zhou & Luo (KDD 2020).
- FGSM / iFGSM — Goodfellow et al. (ICLR 2015); Kurakin et al. (ICLR 2017 Workshop).
- ST-iFGSM — Hu, Zhang, Li, Zhou & Luo (KDD 2023).
- FWL — Frisch & Waugh (1933); Lovell (1963).

Add one line: *"Full literature-grounded motivation + reviewer defense: [`../objective-motivation/`](../objective-motivation/README.md)."* Do **not** edit anything else in `03`.

- [ ] **Step 2: Verify — placeholder gone, cites in, only intended change**

```bash
cd PAPER/argument
echo "placeholder removed (expect 0):"; grep -c 'to be finalized by the authors' 03_fairness_theory.md
echo "verified cites in (expect >=1 each):"; grep -cE 'Big Data|ICDM' 03_fairness_theory.md; grep -cE 'KDD 2023|Hu' 03_fairness_theory.md
echo "cross-link (expect >=1):"; grep -c 'objective-motivation' 03_fairness_theory.md
git -C ../.. diff --stat -- PAPER/argument/03_fairness_theory.md
```
Expected: placeholder `0`; cites `≥1`; cross-link `≥1`; diff touches only `03_fairness_theory.md`.

- [ ] **Step 3: Commit**

```bash
git add PAPER/argument/03_fairness_theory.md
git commit -m "docs(argument): fill objective lineage with verified citations + xref motivation bundle

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Back-fill `argument/07_limitations.md`

**Files:**
- Modify: `PAPER/argument/07_limitations.md`

**Interfaces:**
- Consumes: `LEVELING_DOWN.md` (pointer target); spec §5.3.
- Produces: the demand-endogeneity limitation in the canonical limitations doc.

- [ ] **Step 1: Add the demand-endogeneity limitation note**

Read `07_limitations.md` first to match its format (heading/bullet style). Add a short entry: recorded demand (pickups) in under-served areas is itself suppressed by historical under-supply, so `F_causal`'s demand-adjustment can under-detect latent inequity (feedback loop; Ensign et al. 2018; Lum & Isaac 2016); this is the same phenomenon as the editor's leveling-down (the 93%-at-`DEMAND_FLOOR` finding). Pointer: *"see [`../objective-motivation/LEVELING_DOWN.md`](../objective-motivation/LEVELING_DOWN.md)."* Change nothing else.

- [ ] **Step 2: Verify**

```bash
cd PAPER/argument
echo "endogeneity note (expect >=1):"; grep -ciE 'endogen|suppressed by' 07_limitations.md
echo "pointer (expect >=1):"; grep -c 'objective-motivation/LEVELING_DOWN' 07_limitations.md
git -C ../.. diff --stat -- PAPER/argument/07_limitations.md
```
Expected: note `≥1`; pointer `≥1`; diff touches only `07_limitations.md`.

- [ ] **Step 3: Commit**

```bash
git add PAPER/argument/07_limitations.md
git commit -m "docs(argument): note demand-endogeneity limitation + xref leveling-down

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Bundle-wide consistency gate

**Files:**
- (No new file — verification + a final README provenance touch-up only if a check fails.)

**Interfaces:**
- Consumes: the whole bundle + the two back-fills.
- Produces: a green consistency gate (the "integration test").

- [ ] **Step 1: Forbidden-string sweep (whole bundle + back-fills)**

```bash
cd /home/robert/FAMAIL/.claude/worktrees/objective-motivation-writeup
grep -rniE 'Wilms|Heitz|TKDE|previous arrests|\b67%|\b2\.3%|TODO|TBD|FIXME' \
  PAPER/objective-motivation/ PAPER/argument/03_fairness_theory.md PAPER/argument/07_limitations.md
```
Expected: **no output** (exit 1 from grep). Any hit is a failure — fix the offending file, re-commit, re-run.

- [ ] **Step 2: Citation-resolution check**

Every in-text `(Surname … YEAR)` must resolve in `REFERENCES.md`. Extract distinct surname+year pairs from the prose docs and confirm each appears in `REFERENCES.md`:
```bash
cd PAPER/objective-motivation
grep -rhoE '[A-Z][a-zA-Z-]+ (et al\.? )?[0-9]{4}' MOTIVATION.md REVIEWER_DEFENSE.md LEVELING_DOWN.md \
  | sed -E 's/ et al\.?//' | sort -u \
  | while read name year; do grep -qE "$name.*$year|$year" REFERENCES.md || echo "UNRESOLVED: $name $year"; done
```
Expected: no `UNRESOLVED` lines. (Manually sanity-check any that look like false positives, e.g. a year used as data.)

- [ ] **Step 3: Spec success-criteria check**

Confirm by inspection against spec §10:
- 5 bundle files exist; `argument/03` placeholder gone; `argument/07` has the note.
- MOTIVATION.md reads as drop-in paper prose; demand-endogeneity developed across F_causal + LEVELING_DOWN + 07.
- No new experimental numbers; α-sweep flagged as planned sensitivity.
```bash
cd /home/robert/FAMAIL/.claude/worktrees/objective-motivation-writeup
ls PAPER/objective-motivation/ ; echo "---" ; git -C . log --oneline main..objective-motivation-writeup
```
Expected: five `.md` files listed; the commit log shows Tasks 1–7 (+ the spec/plan commits).

- [ ] **Step 4: Commit (only if Step 1–3 produced fixes)**

```bash
git add -A PAPER/
git commit -m "docs(objective-motivation): consistency-gate fixes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
If nothing needed fixing, skip the commit and record "gate green, no changes."

---

## Self-Review (author check against the spec)

**Spec coverage:** README (Task 5 ↔ spec §4.1); MOTIVATION incl. augmentation + weights (Task 2 ↔ §4.2/§5.2); REVIEWER_DEFENSE (Task 3 ↔ §4.3); LEVELING_DOWN + demand-endogeneity (Task 4 ↔ §4.4/§5.3); REFERENCES + Ensign/Lum verification (Task 1 ↔ §4.5/§5.1/§6); argument/03 (Task 6 ↔ §4.6); argument/07 (Task 7 ↔ §4.7); no-new-numbers + forbidden-strings + citation-resolution (Global Constraints + Task 8 ↔ §6/§9/§10). No spec section is unmapped.

**Placeholder scan:** the plan itself contains no TBD/TODO; every task has concrete content specs, exact citation metadata, runnable checks, and full commit commands.

**Type/name consistency:** file paths and the `PAPER/objective-motivation/{README,MOTIVATION,REVIEWER_DEFENSE,LEVELING_DOWN,REFERENCES}.md` names are identical across tasks; downstream tasks cite against `REFERENCES.md` (built in Task 1); pointer targets (`LEVELING_DOWN.md`, `LEVELING_DOWN_MECHANISM.md`) match their producers.
