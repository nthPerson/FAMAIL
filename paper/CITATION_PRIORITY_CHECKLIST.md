# Citation Priority Checklist — human verification pass (Meeting-42 T3 / Meeting-43 mandate)

**Purpose:** Robert's manual, final verification of EVERY citation in the FAMAIL manuscript.
Per Dr. Kash (2026-07-16): *"don't just take what Google Scholar gives you. Seek the
authoritative source, be it ACM."* Machine cross-checks are done (see Evidence base below);
this list is ranked so the highest-risk items get human eyes first.

**How to use:** work top to bottom; tick `[x]` when you have personally confirmed the entry
against the linked authoritative source (and, where flagged, the claim in the paper text).
The "check" line tells you the *specific* thing machine verification could not settle.

**Verification standard:** ACM Digital Library for ACM venues → publisher page
(IEEE/Springer/Wiley/PMLR/IJCAI/CVF/AAAI/NeurIPS/T&F/JSTOR) → arXiv abs page only for
arXiv-only works. Google Scholar / DBLP / Crossref are corroboration, never terminal.

**Evidence base (machine passes):**
- `reviews/2026-07-12-prose-citation-audit.md` — §1/§2 claim+existence audit (29 instances, 0 fabrications).
- `reviews/2026-07-16-citation-audit-s3-s5.md` — §3–§5 audit + all-41-entry metadata/full-name
  re-verification (0 fabrications; findings F1–F5; ready-to-apply refs.bib diffs live there).
- `PAPER/objective-motivation/sources/mission_2_citation_audit.md` — the 2 fabrications caught
  earlier (why this checklist exists).

**Coverage (verified 2026-07-16):** the paper's citations live entirely in
`sections/01_introduction.tex`, `02_related_work.tex`, `03_methodology.tex` — §4, §5, the
abstract, and `main.tex` carry zero `\cite`. 36 distinct keys cited (P0–P3 below) + 5
bib-only entries (P4) = all 41 `refs.bib` entries covered. Self-check after any edit:

```bash
cd paper && diff <(grep -rho 'cite{[^}]*}' --include='*.tex' . | sed 's/cite{//;s/}//' | tr ',' '\n' | sort -u) \
              <(grep -o '^- \[.\] \*\*[a-z0-9]*\*\*' CITATION_PRIORITY_CHECKLIST.md | grep -o '[a-z0-9]*\*\*$' | sed 's/\*//g' | sort -u)
# lines starting "<" = cited in the paper but MISSING from this checklist (P4 entries appear as ">")
```

**Maintenance rule:** whenever a `\cite` is added/removed in `sections/*.tex` or `refs.bib`
changes, THIS FILE must be updated in the same session — add/remove the entry's block,
re-rank it, and reset its checkbox if metadata changed. (A memory entry enforces this for
Claude sessions; see `memory/project_citation_checklist.md`.)

Status legend for what machines already did:
**PAGE** = live publisher page fetched & matched · **DEPOSIT** = publisher-deposited
Crossref/DOI metadata matched (front-end bot-blocked) · **NONE** = no publisher record exists.

**Applied-changes log (controller sessions; checkboxes above/below remain Robert's alone):**
- 2026-07-24 (late; Robert's F_spatial-lineage correction): ONE NEW entry `su2018taxigini`
  (Robert supplied the source PDF — the paper the ASR/DSR service-rate formulation was
  drawn from during objective design). §3.2 Spatial equity rewritten to name the lineage
  explicitly; key also added to the App A (eq:fspatial context) and App E
  (transportation-equity) Gini cite lists. Terminology corrected paper-wide with it:
  DSR expansion "demand-service ratio" → "departure-service ratio" (their DSR =
  departure service rate; formula D_i/S_i UNCHANGED; code keys + RESEARCHER_HANDOFF.md
  keep the legacy expansion — mapping recorded in paper/README.md conventions).
  Count now 51 entries.
- 2026-07-24 (Meeting-44 rewrite batch 2 — intro citation classes): FIVE NEW entries + P0
  rows (top of P0), filling Dr. Zhang's ≥2-per-class + 2025-26 recency mandate. Hunter
  subagent verified against primary sources, then the controller INDEPENDENTLY re-verified
  all five (DBLP JSON / frontiersin.org / aclanthology.org / proceedings.mlr.press fetched
  live this session); full evidence trail:
  `docs/presentations/meeting_44_update/debrief/analysis/CITATION_CANDIDATES.md`. §1 slots:
  in-processing sentence now `zheng2023,lamalfa2026fairppo,zhao2025fairdrlst`; generation
  sentence extended with a diffusion clause (`hastingsblow2025diffaug`); new ¶1 sentence
  cites `zhao2017menshopping,wang2021directional` as SUPERVISED bias-amplification evidence
  (deliberate honest scope: no IL-specific bias-inheritance paper exists); reweighing
  sentence now `kamirancalders2012,feldman2015` (existing key re-used, no new entry).
  Class (e) (fairness-motivated trajectory EDITING) deliberately left empty: the hunter
  found no such work, which supports FATE's novelty claim. Count now 50 entries.
- 2026-07-22 (Robert's §4.4 citation-gap pass): perturbation arms now cite their methods —
  `kurakin2017ifgsm` + `goodfellow2015fgsm` on the iFGSM/FGSM label, `hu2023stifgsm` on the
  identity-discriminator attack clause (the ST instantiation), and ONE NEW ENTRY
  `madry2018pgd` (DBLP-verified; row above) on "PGD-style random initialization". The
  demographic-oversampling arm now cites its class: `kamirancalders2012,zietlow2022`
  attached to "duplicate real trajectories originating in demographically disadvantaged
  regions" (sampling-for-fairness + augmentation-helps-the-worst-off). All reused keys already have rows; the
  fairness-method paragraph needed no change (kamirancalders2012 + zheng2023 already
  cited). Count now 45 entries.
- 2026-07-21 (later, P0 verification): `xu2018fairgan` + `vanbreugel2021decaf` machine-verified
  against primary sources (proceedings.neurips.cc fetched live for DECAF; DBLP for both;
  FairGAN DOI resolves to IEEE Xplore). refs.bib updated: FairGAN + pages 570--575 +
  DOI 10.1109/BigData.2018.8622525; DECAF + pages 22221--22233. Claim-fit of the intro's
  "synthesizing fair data" sentence confirmed for both. P0 rows' machine-status lines
  updated accordingly; checkboxes untouched (Robert's manual pass still required — FairGAN's
  terminal source is IEEE Xplore, bot-blocked, needs human eyes).
- 2026-07-21: cite MOVES during the cut campaign + Robert's read-aloud pass (no new refs, no
  removals from refs.bib): Robert's guardrail cuts removed `hoermon2016` and
  `feng2020simulate` from §3 (both remain cited in §2 — adversarial-perturbation and
  world-model themes); `feng2020simulate` re-added at §4.2's Fidelity-B definition (the
  Jensen--Shannon gloss moved there when §3's monitoring sentence was cut). NOTE: §4 now
  carries \cite (feng2020simulate, plus kamirancalders2012/zheng2023 from W-earlier) — the
  "coverage" note above (¶ 'citations live entirely in 01/02/03') is stale on that point;
  the self-check command remains the authority. W2/W3/W7 also relocated cite-bearing
  derivation/implementation text into sections/appendix.tex (FWL, Gumbel/STE, cGAIL cites
  now also appear there); all keys unchanged.
- 2026-07-18: TWO NEW refs.bib entries + P0 rows per Dr. Zhang's abstract feedback (intro
  data-generation category cites): `xu2018fairgan`, `vanbreugel2021decaf` — both UNVERIFIED
  (controller-knowledge metadata, no pages/DOI on purpose); MUST clear Robert's manual pass
  before submission. Also NEW intro usage of existing `kamirancalders2012` (rebalancing
  category) and reworded in-processing sentence retains `zheng2023`. Count now 43 entries.
- 2026-07-18 (later): NEW intro usage of existing `ren2020stsiamese` (Robert) — cites
  ST-SiameseNet at its first named mention (frozen driver-identity discriminator). Claim-check
  for the manual pass: architecture-name attribution only, no result claimed. Intro also now
  carries tab:external-sz's DI/DP-gap/Theil deltas verbatim (verified against the table +
  external_fairness.json this session).
- 2026-07-17: NEW §4.5 usages of existing keys `kamirancalders2012` (reweighing baseline arm) and
  `zheng2023` ("in the spirit of" — the in-processing penalty arm). Claim-check note for the manual
  pass: both are METHOD-LINEAGE attributions for our baseline implementations, not claims about the
  cited papers' results; verify the lineage reads fairly (our penalty is a DP-gap analog, not
  Zheng's exact regularizer).
- 2026-07-16: the audit's §3 metadata diff + §4 full-name normalization APPLIED to `refs.bib`
  (34 entries; fixes: zietlow2022 pages 10410–10421; mittelstadt2024 title "Leveling" + DOI;
  karimi2020recourse title/pages; karimi2021recourse pages/booktitle; temkin2000 publisher +
  editor; 4 verified DOIs added). The four "cosmetic, decide-don't-default" items were NOT
  applied — they remain your calls (feldman2015 "21th" [sic], vermarubin booktitle form,
  zhang2022cgail em-dash, kurakin/goodfellow first-name consistency). F1 prose reworded in
  §3.2 (feldman2015 direction acknowledged). `lint.sh` now fails if a cited key is missing
  from this checklist. If you verify an entry against the publisher page, tick it — the
  metadata you're checking is the post-fix state.

---

## P0 — Decisions & unresolved items (do these first; each needs human judgment)

**(2026-07-24 late — Robert-supplied source; the fastest human tick on the list)**

- [ ] **su2018taxigini** — Su, Fang, Xu & Huang, *Uncovering Spatial Inequality in Taxi
  Services in the Context of a Subsidy War among E-Hailing Apps*, ISPRS Int. J. Geo-Inf.
  7(6):230, 2018, DOI 10.3390/ijgi7060230 — NEW 2026-07-24: cited at §3.2 Spatial equity
  (rewritten to name it as the source of the ASR/DSR pair + Gini-per-rate measurement)
  and in the App A / App E transportation-Gini cite lists. **Machine status: PAGE-grade —
  metadata read directly from the author PDF Robert supplied (title page + DOI footer);
  this is the paper the formulation was drawn from during objective design (his
  account).** Check (~2 min): the MDPI landing page via https://doi.org/10.3390/ijgi7060230,
  and that §3.2's one-sentence description of their method reads fairly against their §4
  (per-TAZ ASR/DSR; Gini per rate). Note their DSR = **departure** service rate — the
  reason our expansion was corrected from "demand-service ratio".

**(2026-07-24 batch — Meeting-44 intro citation classes; machine-verified twice, human pass pending)**

- [ ] **lamalfa2026fairppo** — La Malfa, Zhang, Luck & Black, *Fairness Aware Reinforcement
  Learning via Proximal Policy Optimization*, AAAI 2026, pp. 22725--22733,
  DOI 10.1609/aaai.v40i27.39434 — NEW 2026-07-24: cited at §1's in-processing sentence.
  **Machine status: PAGE (2026-07-24) — DBLP JSON (conf/aaai/MalfaZLB26: authors/venue/
  pages/DOI) + arXiv:2502.03953 abstract; claim-fit: adds a fairness penalty (demographic
  parity / conditional statistical parity variants) to PPO's objective.** Check: AAAI OJS
  via https://doi.org/10.1609/aaai.v40i27.39434 (publisher page not bot-fetched; needs
  human eyes).
- [ ] **zhao2025fairdrlst** — Zhao, Shao, Chan, Xu & Salim, *FairDRL-ST: Disentangled
  Representation Learning for Fair Spatio-Temporal Mobility Prediction*, ACM SIGSPATIAL
  2025, pp. 103--106, DOI 10.1145/3748636.3762713 — NEW 2026-07-24: cited at §1's
  in-processing sentence; the tightest zheng2023 companion (fair mobility-demand
  prediction). **Machine status: PAGE (2026-07-24) — DBLP JSON (SIGSPATIAL/GIS 2025) +
  arXiv:2508.07518 ("Accepted as a Research Paper (short) at ACM SIGSPATIAL 2025").**
  Check: ACM DL via https://doi.org/10.1145/3748636.3762713 (ACM DL bot-blocked; needs
  human eyes). Note it is a SHORT paper.
- [ ] **hastingsblow2025diffaug** — Hastings Blow, Qian, Gibson, Obiomon & Dong, *Data
  Augmentation via Diffusion Model to Enhance AI Fairness*, Frontiers in Artificial
  Intelligence 8:1530397, 2025, DOI 10.3389/frai.2025.1530397 — NEW 2026-07-24: cited at
  §1's generation sentence (diffusion clause). **Machine status: PAGE (2026-07-24) —
  frontiersin.org article page fetched LIVE (title/authors/volume/DOI + Tab-DDPM
  fairness-improvement claim).** Check: quick glance at the Frontiers page; lowest risk
  of the five.
- [ ] **zhao2017menshopping** — Zhao, Wang, Yatskar, Ordonez & Chang, *Men Also Like
  Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints*, EMNLP
  2017, pp. 2979--2989, DOI 10.18653/v1/D17-1323 — NEW 2026-07-24: cited at §1 ¶1's
  bias-amplification sentence, which is deliberately worded for SUPERVISED scope ("models
  trained on human-sourced data"), NOT imitation-specific (no IL-specific work exists;
  see CITATION_CANDIDATES.md class c). **Machine status: PAGE (2026-07-24) —
  aclanthology.org/D17-1323 fetched LIVE.** Check: quick glance + confirm the ¶1 wording
  keeps the supervised scope after any edits.
- [ ] **wang2021directional** — Wang & Russakovsky, *Directional Bias Amplification*,
  ICML 2021, PMLR 139, pp. 10882--10893 — NEW 2026-07-24: cited alongside
  zhao2017menshopping, same supervised-scope framing. **Machine status: PAGE
  (2026-07-24) — proceedings.mlr.press/v139/wang21t.html fetched LIVE.** Check: quick
  glance.

- [ ] **xu2018fairgan** — Xu, Yuan, Zhang & Wu, *FairGAN: Fairness-aware Generative
  Adversarial Networks*, IEEE Big Data 2018 — NEW 2026-07-18 (Zhang-feedback citation for the
  intro's data-generation category, `01:~58`). **Machine status: PAGE (2026-07-21) — DBLP
  record matched all fields; pages 570--575 + DOI 10.1109/BigData.2018.8622525 now in
  refs.bib; claim-fit vs abstract confirmed (fair synthetic data generation via GAN matches
  the intro's "synthesizing fair data").** Check: confirm on IEEE Xplore itself
  (https://doi.org/10.1109/BigData.2018.8622525) — DBLP is corroboration, not terminal
  under the verification standard above.
- [ ] **vanbreugel2021decaf** — van Breugel, Kyono, Berrevoets & van der Schaar, *DECAF:
  Generating Fair Synthetic Data Using Causally-Aware Generative Networks*, NeurIPS 2021 —
  NEW 2026-07-18 (same intro category, `01:~58`). **Machine status: PAGE (2026-07-21) —
  proceedings.neurips.cc abstract page fetched live and matched title/authors/NeurIPS 34;
  pages 22221--22233 (DBLP) now in refs.bib; claim-fit confirmed (fair synthetic tabular
  data via causal-DAG GAN, inference-time debiasing).** Check: eyeball
  https://proceedings.neurips.cc/paper/2021/hash/ba9fab001f67381e56e410575874d967-Abstract.html
  — the publisher page is the terminal source and was already fetched; this one only needs
  your eyes.

- [ ] **goodfellow2015fgsm** — Goodfellow, Shlens & Szegedy, *Explaining and Harnessing
  Adversarial Examples* — used at `02:55`, `03:334` (FGSM origin / bounded adversarial perturbation).
  Machine status: arXiv PAGE-verified (https://arxiv.org/abs/1412.6572), but **the "ICLR 2015"
  venue has NO publisher record anywhere** — ICLR 2015 had no formal proceedings and the arXiv
  page has no Comments field.
  **Check/decide:** keep community-standard "ICLR 2015" booktitle, or recast as `@misc` arXiv
  citation. Look at how the ST-iFGSM paper (the KDD template) cites it and match that.
  Query: `google: "Explaining and Harnessing Adversarial Examples" ICLR 2015 site:iclr.cc`

- [ ] **madry2018pgd** — Madry, Makelov, Schmidt, Tsipras & Vladu, *Towards Deep Learning
  Models Resistant to Adversarial Attacks*, ICLR 2018 — NEW 2026-07-22 (Robert's §4.4
  citation-gap pass): cited at the perturbation arms' "PGD-style random initialization
  within the $\varepsilon$ ball" (`04:~262`). **Machine status: PAGE (2026-07-22) — DBLP
  matched authors/venue/year (ICLR 2018 poster, arXiv:1706.06083); claim-fit: random start
  within the $\varepsilon$ ball is this paper's contribution.** Check: eyeball
  https://openreview.net/forum?id=rJzIBfZAb (ICLR 2018 has OpenReview as its record; no
  formal page numbers).
  Query: `google: "Towards Deep Learning Models Resistant to Adversarial Attacks" ICLR 2018`

- [ ] **feldman2015** — Feldman et al., *Certifying and Removing Disparate Impact*, KDD 2015,
  pp 259–268 — used at `02:13` (DI remover), `03:92` (**fairness-as-predictability — F1
  finding**), `03:425` (DI repair, pre-processing family).
  Machine status: DEPOSIT (ACM DL Cloudflare-blocked); claim at 03:92 read in full text —
  **prediction direction is inverted** vs their Theorem 4.1 (they predict the protected
  attribute FROM the data; F_causal runs demographics → outcome residual).
  **Check:** (1) confirm on the rendered DL page https://dl.acm.org/doi/10.1145/2783258.2783311;
  (2) decide the 03:92 rewording (proposal in the 2026-07-16 report §1-F1); (3) style call:
  ACM's official proceedings title is "…**21th** ACM SIGKDD…" [sic] — bib has "21st".
  Query: `site:dl.acm.org "Certifying and Removing Disparate Impact"`

- [ ] **mittelstadt2024** — Mittelstadt, Wachter & Russell, Michigan Technology Law Review
  30(1), 2024 — used at `01:43`, `02:74`, `03:270` (leveling-down formalized for ML; prescribe
  level-up). ⚑ prose being rewritten to analogy-only (Meeting-43).
  Machine status: PAGE (https://repository.law.umich.edu/mtlr/vol30/iss1/3/).
  **Check:** apply/confirm the title fix — the journal edition spells "**Leveling** Down"
  (bib has preprint's "Levelling"); do NOT add page numbers (publisher's citation lists none;
  bepress "firstpage=3" is an article slot); optional DOI 10.36645/mtlr.30.1.unfairness.

- [ ] **zietlow2022** — Zietlow et al., *Leveling Down in Computer Vision…*, CVPR 2022 —
  used at `02:76`, `03:272` (data aug uniquely helps the disadvantaged group). ⚑ leveling-down prose.
  Machine status: PAGE (CVF Open Access).
  **Check:** apply/confirm the page fix — **10410–10421**, not 10400–10411 (CVF meta tags +
  CVF BibTeX both; the 07-12 audit missed it by trusting arXiv).
  Link: https://openaccess.thecvf.com/content/CVPR2022/html/Zietlow_Leveling_Down_in_Computer_Vision_Pareto_Inefficiencies_in_Fair_Deep_CVPR_2022_paper.html

- [ ] **karner2024** — Karner, Pereira & Farber, *Advances and pitfalls in measuring
  transportation equity*, Transportation 52(4):1399–1427, **2025** — used at `02:26`, `03:119`
  (Gini widely used in transport equity).
  Machine status: PAGE (Springer, direct); year-2025 issue record confirmed correct.
  **Check:** a published **Correction** exists (2024-03-06, DOI 10.1007/s11116-024-10474-9) —
  open it and confirm it doesn't touch the Gini/Lorenz content we cite.
  Link: https://link.springer.com/article/10.1007/s11116-023-10460-7

- [ ] **kurakin2017ifgsm** — Kurakin, Goodfellow & Bengio, *Adversarial Examples in the
  Physical World* — used at `02:55`, `03:334` (iterative FGSM).
  Machine status: arXiv PAGE (https://arxiv.org/abs/1607.02533); **ICLR 2017 Workshop Track
  status rests on a search snippet** (OpenReview bot-blocked).
  **Check:** one click — https://openreview.net/forum?id=HJGU3Rodl confirms Workshop Track.
  Name note: arXiv renders "Ian Goodfellow" (no "J.") here — inconsistent with
  goodfellow2015fgsm's "Ian J."; records differ, your call on harmonizing.

- [ ] **jang2017gumbel** — Jang, Gu & Poole, *Categorical Reparameterization with
  Gumbel-Softmax* — used at `02:58` (continuous relaxation), `03:353` (temperature-annealed
  soft cell assignment — the discrete-grid bridge in the editor).
  Machine status: arXiv PAGE (https://arxiv.org/abs/1611.01144); ICLR 2017 via arXiv+DBLP only.
  **Check:** one click — https://openreview.net/forum?id=rkE3y85ee (ICLR 2017 poster).

- [ ] **maddison2017concrete** — Maddison, Mnih & Teh, *The Concrete Distribution…* — used at
  `02:58`, `03:353` (same claim pair as jang2017gumbel).
  Machine status: arXiv PAGE (https://arxiv.org/abs/1611.00712); ICLR 2017 via arXiv+DBLP only.
  **Check:** one click — https://openreview.net/forum?id=S1jE5L5gl.

---

## P1 — Load-bearing attributions verified only via publisher DEPOSIT (spot-check the rendered page)

All fields matched ACM/IEEE-deposited metadata; the residual risk is page-level rendering only.
Open each DL/Xplore page in a browser (they block our fetchers, not humans) and eyeball
authors/title/pages.

- [ ] **hu2023stifgsm** — *ST-iFGSM…*, KDD 2023, pp 764–774 — `02:56`, `03:335` (the editor's
  direct lineage; the KDD template paper). DEPOSIT.
  https://dl.acm.org/doi/10.1145/3580305.3599513
- [ ] **zhang2022cgail** — *cGAIL…*, IEEE Trans. Big Data 8(5):1288–1300, 2022 — `01:9`,
  `02:39`, `03:383` (framework + preprocessing pipeline; king-move claim body-verified via
  NSF PAR manuscript). DEPOSIT.
  https://ieeexplore.ieee.org/document/9266753 · cosmetic: IEEE renders title em-dash unspaced.
- [ ] **zhang2019cgail** — cGAIL conference version, ICDM 2019, pp 1480–1485 — `02:39`. DEPOSIT.
  https://ieeexplore.ieee.org/document/8970802 · full name "Xin Zhang" confirmed (resolves the
  [32]/[33] X. Zhang / Xin Zhang collision — same person).
- [ ] **zheng2023** — *Fairness-Enhancing Deep Learning for Ride-Hailing Demand Prediction*,
  IEEE OJ-ITS 4:551–569 — `01:21`, `02:30`, `03:432` (in-processing regularizer; the
  0.361→0.084 MPE-gap numbers verified exact on 07-12). DEPOSIT.
  https://doi.org/10.1109/OJITS.2023.3297517
- [ ] **corbettdavies2017** — *Algorithmic Decision Making and the Cost of Fairness*, KDD 2017,
  pp 797–806 — `02:16`, `03:54` (conditional statistical parity — anchors F_causal's logic;
  ⚑ near the F_causal-rename caveat). DEPOSIT.
  https://dl.acm.org/doi/10.1145/3097983.3098095
- [ ] **ustun2019recourse** — *Actionable Recourse in Linear Classification*, FAT* 2019,
  pp 10–19 — `02:62`, `03:337` (claim: algorithmic recourse — the same counterfactual-
  perturbation tooling used *constructively*; the framing FAMAIL borrows for its editor). DEPOSIT.
  https://dl.acm.org/doi/10.1145/3287560.3287566
- [ ] **ren2020stsiamese** — *ST-SiameseNet*, KDD 2020, pp 1306–1315 — `02:47`, `03:144`
  (the fidelity discriminator's family). DEPOSIT.
  https://dl.acm.org/doi/10.1145/3394486.3403183
- [ ] **pan2020xgail** — *xGAIL*, KDD 2020, pp 1334–1343 — `02:40` (claim: explainable-GAIL
  extension exposing the recovered policies, in §2's imitation-learning lineage). DEPOSIT.
  https://dl.acm.org/doi/10.1145/3394486.3403186 · "W. Huang" = Weixiao Huang.
- [ ] **feng2020simulate** — *Learning to Simulate Human Mobility*, KDD 2020, pp 3426–3433 —
  `02:41`, `03:153` (JSD-over-mobility-statistics precedent — claim body-verified from the
  authors' camera-ready). DEPOSIT for metadata.
  https://dl.acm.org/doi/10.1145/3394486.3412862
- [ ] **vermarubin2018** — *Fairness Definitions Explained*, FairWare 2018, pp 1–7 — `02:14`
  (claim: taxonomy of formal fairness *definitions* — NOT the pre/in/post-processing
  intervention taxonomy; §2 was reworded to this exact claim on 07-12, disposition #3).
  DEPOSIT. https://dl.acm.org/doi/10.1145/3194770.3194776
  **Check:** claim wording stays as reworded; venue-string style — ACM renders "Proceedings
  of the International Workshop on Software Fairness" (no "@ICSE" suffix; that's DBLP style).
- [ ] **parfit1997** — *Equality and Priority*, Ratio 10(3):202–221 — `01:43`, `02:72`,
  `03:268` (the leveling-down objection; ⚑ analogy-only rewrite). DEPOSIT (Wiley blocked).
  https://doi.org/10.1111/1467-9329.00041 · NB: cites the 1997 Ratio article, NOT the 1991
  Lindley Lecture "Equality or Priority?" — title must stay as-is.

---

## P2 — DEPOSIT-verified, routine (older stats/econ journals; low risk, quick confirms)

- [ ] **frischwaugh1933** — Frisch & Waugh, Econometrica 1(4):387–401 — `03:87` (FWL theorem).
  Claim verified via Basu's YFWL history. Econometric Society page fetched live; JSTOR blocked.
  https://www.jstor.org/stable/1907330 · trap to ignore: EconSoc's website lists Waugh first —
  CMS artifact; Frisch-first is correct (their own DOI deposit + citation tradition).
- [ ] **lovell1963** — JASA 58(304):993–1010 — `03:87` (claim: the FWL generalization —
  arbitrary regressor partitions; claim verified via Basu's YFWL history, same as frischwaugh1933).
  https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10480682
- [ ] **hoaglinwelsch1978** — *The Hat Matrix in Regression and ANOVA*, Amer. Statistician
  32(1):17–22 — `03:63` (claim: canonical hat-matrix reference — H as the projection onto the
  design matrix's column space, the closed-form machinery behind Eq. F_causal; their abstract
  says "a projection matrix known as the hat matrix"). DOI verified resolving.
  https://www.tandfonline.com/doi/abs/10.1080/00031305.1978.10479237
- [ ] **atkinson1970** — *On the Measurement of Inequality*, JET 2(3):244–263 — `02:27`
  (claim: seminal source of the welfare-based Atkinson index, in §2's inequality-measures list).
  https://doi.org/10.1016/0022-0531(70)90039-6 (ScienceDirect blocked our fetchers).
- [ ] **demaio2007** — *Income inequality measures*, JECH 61(10):849–852 — `02:27`
  (claim: survey of inequality measures — when Gini vs Theil vs Atkinson is preferred).
  https://doi.org/10.1136/jech.2006.052969 · known artifact: BMJ's Crossref deposit shows
  "…: Figure 1" appended to the title — the true title has no suffix (Europe PMC confirms).
- [ ] **hoermon2016** — Ho & Ermon, GAIL, NIPS 2016 — `02:51` (IL-as-adversarial-game analogy),
  `03:147` (the "live adversarial game" whose instability the frozen discriminator avoids —
  note the cite anchors the *game*; "instability" is the field's characterization, judged
  defensible on 07-16). NeurIPS page PAGE-verified; **pages 4565–4573 are printed-proceedings
  convention** (the web page shows no pagination) — standard, keep.
  https://proceedings.neurips.cc/paper/2016/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html

---

## P3 — Fully PAGE-verified live (lowest risk; tick after a quick glance)

- [ ] **kamirancalders2012** — Reweighing, KAIS 33(1):1–33 — `01:33` (pre-processing lineage),
  `02:12` (pre-processing family), `03:424` (claim: canonical source of instance Reweighing,
  which FAMAIL transplants to the imitation loss — the downstream recipe's anchor cite).
  https://link.springer.com/article/10.1007/s10115-011-0463-8 (open access)
- [ ] **horchergraham2021** — Gini for transport demand imbalance, Transportation 48:2521–2544 —
  `02:26`, `03:119` (claim: proposes/applies the Gini index to public-transport demand
  imbalances — half of the "widely used in transportation equity" pair justifying F_spatial).
  https://link.springer.com/article/10.1007/s11116-020-10138-4
- [ ] **theil1967** — *Economics and Information Theory*, North-Holland — `02:27`
  (claim: origin of the Theil index — entropy-based, between/within-group decomposable).
  https://openlibrary.org/works/OL1393206W (book; optional additions: co-publisher Rand
  McNally, series "Studies in Mathematical and Managerial Economics v.7")
- [ ] **gao2017tuler** — TULER, IJCAI 2017, pp 1689–1695 — `02:45` (TUL method), `03:142`
  (shared claim with the next two: mobility signatures are highly identifying — the TUL
  literature's premise, which motivates the driver-identity fidelity discriminator).
  https://www.ijcai.org/proceedings/2017/234
- [ ] **zhou2018tulvae** — TULVAE, IJCAI 2018, pp 3212–3218 — `02:45`, `03:142` (same claim
  pair as gao2017tuler). https://www.ijcai.org/proceedings/2018/446
- [ ] **miao2020deeptul** — DeepTUL, AAMAS 2020, pp 878–886 — `02:45`, `03:142` (same claim
  pair). https://www.ifaamas.org/Proceedings/aamas2020/ (camera-ready p878.pdf read; ACM DL
  hosts it under 10.5555 id only — IFAAMAS is publisher of record)
- [ ] **ensign2018** — *Runaway Feedback Loops in Predictive Policing*, PMLR 81:160–171 —
  `01:17`, `02:78`, `03:294` (claim: feedback-loop pathology / censored demand signal).
  ⚠ known valence gotcha (07-12 disposition #5): the papers' stated harm is predictive-policing
  *over*-allocation; the under-service reading is OUR domain mapping and §2 marks it as such —
  when reviewing, check that framing survives any prose edits.
  https://proceedings.mlr.press/v81/ensign18a.html
- [ ] **lumisaac2016** — *To Predict and Serve?*, Significance 13(5):14–19 — `01:17`, `02:78`,
  `03:294` (same claim + same disposition-#5 gotcha as ensign2018).
  https://academic.oup.com/jrssig/article/13/5/14/7029190 (journal moved Wiley→OUP;
  legacy DOI 10.1111/j.1740-9713.2016.00960.x still resolves — fine as cited)
- [ ] **wachter2018counterfactual** — Harvard JOLT 31(2):841–887 — `02:62`, `03:337` (claim:
  counterfactual explanations = smallest input change to an alternative decision; paired with
  ustun2019recourse for the "constructive perturbation" framing of the editor). Subtitle
  "…: Automated Decisions and the GDPR" was added to the bib on 07-12 (disposition #2).
  https://jolt.law.harvard.edu/assets/articlePDFs/v31/Counterfactual-Explanations-without-Opening-the-Black-Box-Sandra-Wachter-et-al.pdf
  (pages confirmed on the journal's own PDF: starts 841, 47 pp)
- [ ] **bengio2013ste** — STE, arXiv:1308.3432 (arXiv-only; abs page IS the authority) —
  `02:59` (straight-through gradient estimation — deliberately split from the "relaxation"
  phrase on 07-12, disposition #4), `03:353` (inside the "temperature-annealed soft cell
  assignment" umbrella — pre-cleared as defensible). https://arxiv.org/abs/1308.3432
- [ ] **barocas2023** — *Fairness and Machine Learning*, MIT Press 2023 — `02:10` (claim:
  carrier of the pre-/in-/post-processing intervention taxonomy — moved onto this cite alone
  on 07-12, disposition #3). https://fairmlbook.org/ (canonical BibTeX there matches; entry
  pins the 2023 MIT Press edition — the free web edition is 2019)

---

## P4 — In refs.bib but currently UNCITED (verify only if they enter the paper)

Metadata upgrades for these are already in the 2026-07-16 report diff, ready if needed.

- [ ] **pinzon2022** — AAAI 2022, 36:7993–8000 — likeliest to return (flagged "analogy only"
  for the leveling-down rewrite). PAGE-verified: https://ojs.aaai.org/index.php/AAAI/article/view/20770
  (issue 7 + DOI 10.1609/aaai.v36i7.20770 available).
- [ ] **karimi2020recourse** — NeurIPS 2020. PAGE-verified; **bib is missing the published
  subtitle ": A Probabilistic Approach" and pages 265–277**.
  https://proceedings.neurips.cc/paper/2020/hash/02a3c7fb3f489288ae6942498498db20-Abstract.html
- [ ] **karimi2021recourse** — FAccT 2021. DEPOSIT; add pages 353–362.
  https://doi.org/10.1145/3442188.3445899
- [ ] **temkin1993** — *Inequality*, OUP 1993 (hardcover; paperback is 1996). Verified on a
  Wayback capture of the OUP product page. https://global.oup.com/academic/product/inequality-9780195078602
- [ ] **temkin2000** — chapter in *The Ideal of Equality* — **bib missing publisher (Palgrave
  Macmillan)**; year 2000 = original Macmillan hardcover (Palgrave softcover record says ©2002);
  chapter title correctly spells "**Levelling**" (British — unlike mittelstadt2024).
  https://link.springer.com/book/9780333971192 · pages 126–161 rest on chapter scans/PhilPapers.

---

## Cross-cutting items (once, not per-entry)

- [ ] Apply the **R21 full-name normalization** (complete author-field list in
  `reviews/2026-07-16-citation-audit-s3-s5.md` §4) and confirm the rendered bibliography no
  longer mixes "X. Zhang"/"Xin Zhang".
- [ ] After any refs.bib edit: rebuild and **proofread the rendered reference list aloud**
  (your standard practice) — BibTeX + ACM-Reference-Format can re-abbreviate or drop fields.
- [ ] Double-blind sweep already clean (no "our prior work" phrasing anywhere); re-check if
  §3.5 prose around `zhang2022cgail` gets rewritten.
- [ ] The two in-flight terminology decisions (leveling-down → analogy-only; F_causal rename)
  will rewrite prose around ⚑-marked entries — re-read those claim attachments after the rewrite.

<!-- Coverage note (2026-07-24 late, restructure review fix #9): the restructure
     rebuilt §1's cite groups (9 -> 5 groups; five keys re-homed into the PI's new
     intro paragraphs: zhao2017menshopping, wang2021directional,
     hastingsblow2025diffaug, lamalfa2026fairppo, zhao2025fairdrlst) and moved
     related work to render as §5 (file name unchanged). KEY SET UNCHANGED — no
     row added or removed; 02_overview.tex carries no cites; lint coverage green
     at d874d8d. The pre-restructure Coverage paragraph above describes the old
     file layout. -->

<!-- Coverage note (2026-07-25, Robert's systematic re-read / prose pass): two body
     citations were DROPPED from §3.5 (Edit-Aware Weighting) as part of removing two
     related-work asides from the method section. Neither key is orphaned and no row
     changes: feldman2015 remains cited in Appendix A (Derivations) and Appendix E
     (full related-work survey); zheng2023 remains cited in §1, §5 (Related Work),
     §4.4 (Fairness-method baselines) and Appendix E. KEY SET UNCHANGED.
     One claim-attachment item DOES need a human look during the P0 pass:
     zietlow2022 in §3.3 previously read "data augmentation is THE ONE intervention
     observed to help the disadvantaged group" (a claim about the literature); it now
     reads "in a study of fair image classifiers, data augmentation was the one
     strategy that helped ...", matching this file's own verified refs.bib note
     ("adaptive augmentation was the one strategy that helped the disadvantaged
     group", verified 2026-07-16). Verify the domain qualifier is how you want it. -->

<!-- Coverage note (2026-07-25, findings round: #5/#8/#10/#11/#13/#15/#16/#17/#18/#19):
     NO key-set change and no \cite added or removed. The only cite-adjacent edit was in
     §3.4, where the recourse gloss was dropped for the page budget; the citation group
     \cite{ustun2019recourse,wachter2018counterfactual} is unchanged and §5 still states
     the same idea with the same five perturbation/recourse keys. Table 2 moved to
     Appendix C and carries no citations. Lint coverage green. Nothing ticked. -->
