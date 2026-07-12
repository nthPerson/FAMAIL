# Paper Prose Sections (Intro / Related Work / Conclusion) — Design

**Date:** 2026-07-12
**Status:** Approved (brainstorm session with Robert, 2026-07-11→12)
**Goal:** Draft the three remaining prose sections of `paper/` — `01_introduction.tex`,
`02_related_work.tex`, `05_conclusion.tex` — while Fable access lasts, so voice and level of
detail stay consistent with the already-drafted abstract, §3 (methodology), and §4
(experiments). The experiments re-run campaign (r-chain + q1–q8) is mid-flight; nothing in
these sections may hard-code a number the campaign could change.

## Context & source-of-truth ruling

- §3 (415 lines) and §4 (444 lines, 29 run-gated slot-markers) are drafted; abstract drafted
  (committed headline values only). §1/§2/§5 are stubs.
- **The `PAPER/argument/` docs are pre-supply-lift era** (trim-only editor, banned +0.0144/+0.0139
  headlines, no lift / external metrics / leveling-down). They are *skeleton only*.
  **Source of truth for the current narrative = drafted abstract + §3 + §4**, plus:
  `PAPER/objective-motivation/MOTIVATION.md` (contrast blocks), `REFERENCES.md` (groupings),
  `PAPER/external-metrics/FINDINGS.md` + `LEVELING_DOWN_MECHANISM.md`,
  `PAPER/supply-lift/FINDINGS.md` + `LIFT_ALGORITHM_REFERENCE.md`,
  `PAPER/argument/{00,01,07}` (bones only, re-derive every claim against current-era docs).

## Locked framing decisions (Robert, AskUserQuestion session)

1. **Headline identity = the framework**: edit-don't-generate + upweight recipe (matches title
   and abstract). Trim+lift and external metrics are supporting evidence, not the lede.
2. **Numbers: both cities, SF slot-marked.** Shenzhen committed values hard-coded with `% src`
   comments (ΔF_causal +0.0226, DI +0.0162, lift-up tier-1 +0.0176 / tier-2 +0.0411 — promoted
   s10 corpus). Every SF value is an `X.XXXX` slot with `% TODO(run:r1 -> <output path>)`
   wired to the same artifact paths §4's markers use.
3. **Contributions: numbered list, no "pillar" coinage.** The data-level/training-level
   structure is described in prose; the internal "two pillars" vocabulary stays internal.
4. **Conclusion scope:** contribution restatement + compact limitations recap + future work
   limited to (a) unified one-pass editor and (b) broader transfer. Seeking-sensitive
   discriminator excluded. F_demo rename excluded (pending PI decision — never in the paper).

## §1 Introduction (~1.2 columns + contribution list)

Five-paragraph arc:

1. **Problem.** Taxi service allocation encodes demographic inequity; demand models learned by
   imitation inherit it; deployed models can amplify it (feedback loops: ensign2018,
   lumisaac2016). Re-derived from `PAPER/argument/01_motivation_goals.md`, current register.
2. **Why the data side; why edit, not generate.** Model-side fixes fight the training signal
   (zheng2023 as the model-level counterpart); generation risks distributional collapse and
   untargeted change; FAMAIL edits a small, attribution-targeted, bounded slice of real
   trajectories and generates nothing.
3. **The turn (Meeting-42 spine).** Demand-only editing improves fairness metrics by leveling
   down (parfit1997, mittelstadt2024); that finding *motivated* trim+lift: the supply channel
   is differentiable and endogenous, so gains are no longer pure leveling-down. SZ numbers
   hard-coded here; SF slot-marked.
4. **Transfer.** Vanilla BC averages the edit away (null); upweighting recovers it
   edit-specifically (random + select-fairest controls); external measures the objective never
   optimizes improve (ring-iii vocabulary only, per README convention #8).
5. **Contributions (numbered):**
   - C1 — the trim+lift editor: bounded, gradient-guided, supply endogenous; two exact
     attribution mechanisms (demand deficit attribution; supply-gradient attribution).
   - C2 — the leveling-down diagnosis of demand-only editing and the supply-channel remedy.
   - C3 — the upweighted-imitation recipe with edit-specificity controls.
   - C4 — two-city validation on established fairness measures never optimized, plus the
     demographic-oversampling baseline comparison.

No roadmap paragraph (space; section titles carry it).

## §2 Related Work (~0.7 column)

Five thematic paragraphs, **each ending with a one-sentence contrast positioning FAMAIL**
(MOTIVATION.md contrast blocks are pre-built for this):

1. Fairness interventions in ML — pre-/in-/post-processing; kamirancalders2012, feldman2015,
   corbettdavies2017, barocas2023. *Contrast: pre-processing transplanted to imitation
   learning; edits demonstrations, not features/labels.*
2. Fairness in urban mobility / transportation equity — zheng2023 (closest applied neighbor,
   in-processing), horchergraham2021, karner2024, theil1967/atkinson1970/demaio2007 as metric
   lineage. *Contrast: intervention moved from model to demonstrations; evaluated on external
   measures never optimized.*
3. Imitation learning for mobility + trajectory identity — zhang2019cgail, zhang2022cgail,
   pan2020xgail, feng2020simulate; TUL: gao2017tuler, zhou2018tulvae, miao2020deeptul;
   ren2020stsiamese. *Contrast: no new generator; edits the demonstrations those models train
   on; identity literature repurposed as realism guardrail.*
4. Adversarial perturbation & recourse — goodfellow2015fgsm, kurakin2017ifgsm, hu2023stifgsm,
   ustun2019recourse, wachter2018counterfactual; discreteness machinery (jang2017gumbel,
   maddison2017concrete, bengio2013ste). *Contrast: bounded-perturbation machinery used
   constructively; ε reinterpreted as identity-preservation budget.*
5. Leveling-down ethics + feedback loops — parfit1997, mittelstadt2024, zietlow2022,
   ensign2018, lumisaac2016. *Contrast: FAMAIL operationalizes level-up via the supply
   channel; demand endogeneity acknowledged as bounding what any demand-adjusted metric sees.*

**Citation discipline:** draw from the 41 verified `refs.bib` entries. Any *new* citation must
be verified against the publisher record before entering `refs.bib` (Mission-2 audit rule;
cf. hoaglinwelsch1978 precedent). §2 may re-cite works §3 cites inline — repeat citations,
never sentences.

## §5 Conclusion (~0.35 column)

Three paragraphs:

1. **Restatement** mirroring the contribution list; committed numbers only (or qualitative).
2. **Limitations recap** (compact): associational/ecological metric on ~10 district profiles;
   demand endogeneity (tied to leveling-down); small-n significance floors with the
   direction + magnitude + t-CI + dose-response + controls defense; SF reproduces, never
   beats; one clause on fidelity certifying identity, not shape.
3. **Future work:** unified one-pass editor (pre-motivated by §3.5's own text); broader
   transfer — other imitation objectives, GAN/WGAN on the edited corpus, additional cities,
   training-side allocation constraints.

## Cross-cutting guardrails

- **Voice:** match abstract + §3 register. Per Robert's prose-style feedback (memory
  `feedback_paper_prose_style.md`): explicit referents (no dangling "throughout/baseline"),
  no AI-sounding flourishes, no coinages where a plain term exists, prefer words over
  notation that skims like a typo.
- **All README conventions + `lint.sh` apply** — notably: trim+lift centers all reporting; no
  trim-only numbers outside the ablation; no causality-claim language; SF *reproduces*;
  three-ring firewall vocabulary; no product names; every load-bearing number carries `% src`.
- **Terminology locks:** *trim*, *lift*, active unit, demand deficit attribution,
  supply-gradient attribution, value-of-presence map, service ratio Y, leveling down /
  lift-up.
- **Length budget:** compiled paper currently 7 pp with stubs; KDD limit 9 pp + refs. Targets:
  §1 ≈ 1.2 col, §2 ≈ 0.7 col, §5 ≈ 0.35 col.
- **Verification gate per section:** `latexmk` clean + `bash lint.sh` pass + citation keys all
  resolve in `main.bbl` + no un-slotted SF numbers (grep for digits near "SF"/"San Francisco").

## Out of scope

§4's 29 run-gated slots (fill-in pass when runs land); the two placeholder figures; the
PI-framing decision on which SF fairness story leads (line ~422 of §4); any abstract rewrite;
F_demo rename.
