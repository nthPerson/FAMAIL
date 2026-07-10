# Data-augmentation baseline candidates: verified literature scan

**Task:** Mission-3 Task 6 (non-blocking, gated). **Date:** 2026-07-09.

**Purpose:** FAMAIL already has three built baseline arms — PGD-style ST-iFGSM, plain FGSM, and
random jitter — all bounded ε=2-grid-cell whole-trajectory perturbations, scored on fidelity
(frozen Siamese driver-identity discriminator + distributional JS) and fairness (expected NOT to
improve, since none of them optimize a fairness objective; see
`famail_temporal/baselines/stifgsm_baseline.py`). This memo surveys the literature for **3-5
additional candidate baselines** a KDD reviewer might expect to see, so the PI/user can decide
whether any becomes a 4th arm. **No code was written for this task.**

Search themes (per the Task-6 brief / Meeting-41 plan): trajectory augmentation for deep mobility
models; perturbation-based augmentation for GPS/spatio-temporal data; fairness-aware data
augmentation for mobility; counterfactual trajectory generation.

**Citation-verification standard applied:** every citation below was checked against its
arXiv abstract page, an ACM DOI resolved through the Semantic Scholar Graph API (ACM Digital
Library itself returned HTTP 403 to direct fetches — see rejected/blocked list), or a Dagstuhl
LIPIcs DOI landing page. No metadata was taken from memory or from AI-search summaries; only
primary bibliographic pages. Verification URL is recorded per entry. All quoted phrases are
verbatim from the fetched page.

---

## Candidate 1 — Geographic point-perturbation trajectory augmentation

**Citation:** Yaksh J. Haranwala, Gabriel Spadon, Chiara Renso, Amílcar Soares. "A Data
Augmentation Algorithm for Trajectory Data." *1st ACM SIGSPATIAL International Workshop on
Methods for Enriched Mobility Data (EMODE@SIGSPATIAL)*, 2023. DOI: 10.1145/3615885.3628008.
Companion open-source framework: Yaksh J. Haranwala, "AugmentTRAJ: A framework for point-based
trajectory data augmentation," arXiv:2311.15097 (submitted 25 Nov 2023).

**Verification:**
- Workshop paper — verified via Semantic Scholar Graph API keyed on the DOI (ACM DL page itself
  403'd on direct fetch): `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3615885.3628008?fields=title,authors,venue,year,abstract,externalIds`
  — title, all 4 authors, venue "EMODE@SIGSPATIAL", year 2023 confirmed; DBLP key
  `conf/emode/HaranwalaSRS23` cross-confirms indexing.
- Framework paper — verified directly: `https://arxiv.org/abs/2311.15097`.

**Method summary:** The workshop paper proposes "a novel strategy for augmenting trajectory data
that applies a geographical perturbation on trajectory points along a trajectory," producing
"controlled changes in the raw trajectory and, consequently, changes in the trajectory feature
space." Tested on two trajectory datasets, it reports "a performance improvement of approximately
20% when contrasted with the baseline" on downstream (non-fairness) tasks. AugmentTRAJ is the
authors' accompanying open-source Python framework generalizing this into a reusable point-wise
augmentation toolkit for mobility datasets.

**Applicability to FAMAIL:** This is the closest literature analogue to FAMAIL's own random-jitter
arm, but the perturbation is described as geographically informed/point-local rather than an
independent-per-point draw inside a fixed L∞ ball — i.e. a different noise *shape*, not a
different noise *budget*. Adapting it would mean swapping the noise-generation function in the
existing FGSM/jitter harness (continuous perturbation → snap back onto the 48×90 grid, respect the
existing ε=2-cell budget) while reusing the rest of the scoring pipeline unchanged.
**Build cost: S** (harness, discriminator, and fidelity/fairness scorers all already exist; only
the noise generator changes).

**Recommendation: DEFER.** Conceptually adjacent to the already-built random-jitter arm (same
perturb-then-snap-to-grid pattern); the marginal reviewer-facing distinctiveness of adding a
second, geometry-aware jitter variant is low relative to the other candidates below unless a
quick smoke test shows the correlated-noise structure changes fidelity results materially.

---

## Candidate 2 — Geo-indistinguishability (formal DP-style location perturbation)

**Citation:** Miguel E. Andrés, Nicolás E. Bordenabe, Konstantinos Chatzikokolakis, Catuscia
Palamidessi. "Geo-Indistinguishability: Differential Privacy for Location-Based Systems."
*Proceedings of the 2013 ACM SIGSAC Conference on Computer and Communications Security (CCS'13)*,
pp. 901-914. arXiv:1212.1984 (submitted 10 Dec 2012, last revised 20 Feb 2014).

**Verification:** `https://arxiv.org/abs/1212.1984` — title, all 4 authors, and CCS'13
venue/page numbers confirmed directly on the arXiv abstract page.

**Method summary:** Introduces geo-indistinguishability, "a formal notion of privacy for
location-based systems that protects the user's exact location, while allowing approximate
information [to be released]." The mechanism draws controlled random noise from a 2D Laplace
distribution in polar coordinates around the true point, calibrated to a privacy radius/budget,
and the paper shows this "offers the best privacy guarantees, for the same utility, among all
those which do not depend on the prior." It predates deep-learning-era augmentation work but is
the foundational reference the location-privacy/perturbation literature still cites for
*principled, radially-symmetric* location noise (as opposed to an ad hoc bounded perturbation).

**Applicability to FAMAIL:** This is a genuinely different noise family from what's already
built — radially-symmetric calibrated Laplace noise around each point vs. whole-trajectory
L∞-bounded PGD/jitter — so it would be a distinguishable 4th arm rather than a variant of an
existing one. Adaptation needed: sample Laplace-in-polar-coordinates noise per seeking-state,
snap the perturbed continuous point back onto the 48×90 grid cell, and choose an ε that maps
sensibly onto the existing 2-cell perturbation budget for comparability. **Build cost: S/M**
(new noise sampler + grid-snap step; everything downstream — discriminator scoring, fidelity JS,
fairness metrics — is reused as-is).

**Recommendation: worth discussing (lean ADOPT-CANDIDATE).** This is plausibly the specific
comparison a reviewer familiar with the location-privacy literature would ask for ("why not
compare against calibrated/principled noise rather than only ad hoc bounded perturbation"), and
it is cheap to build by reusing the existing harness. Still gated on user go-ahead per the
decision gate below.

---

## Candidate 3 — LSTM-TrajGAN (GAN-based full-trajectory counterfactual generation)

**Citation:** Jinmeng Rao, Song Gao, Yuhao Kang, Qunying Huang. "LSTM-TrajGAN: A Deep Learning
Approach to Trajectory Privacy Protection." *11th International Conference on Geographic
Information Science (GIScience 2021), Part I*. Leibniz International Proceedings in Informatics
(LIPIcs), Volume 177, Article 12. DOI: 10.4230/LIPIcs.GIScience.2021.I.12.

**Verification:** `https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GIScience.2021.I.12`
— title, all 4 authors, LIPIcs volume/venue, and DOI confirmed directly on the Dagstuhl landing
page. (Note: the landing page's own metadata lists publication year 2020 for the GIScience 2021
proceedings volume — reported here as it appears on the source, not reconciled further.)

**Method summary:** An end-to-end seq2seq LSTM generator/discriminator GAN that outputs a full
synthetic replacement trajectory per real trajectory (rather than perturbing the original
in-place), trained with a composite "TrajLoss" spatial+temporal+categorical similarity metric so
the synthetic trace defeats a trajectory-user-linking (re-identification) classifier while
preserving aggregate spatial, temporal, and thematic properties of the original.

**Applicability to FAMAIL:** Conceptually the "counterfactual trajectory generation" arm the
search themes call for, and its discriminator-defeating objective is philosophically close to
what FAMAIL's own frozen HuMID discriminator scores. But it requires training a full generator
network from scratch on FAMAIL's discretized 48×90×24h seeking corpus, re-deriving TrajLoss for
grid coordinates/time-buckets, and conditioning the generator per-driver so its outputs are
scorable by HuMID at all. **Build cost: L.**

**Recommendation: DEFER.** High build cost, and the project's own prior GAN-baseline work
(memory: model-level F_causal NULL at n=5, edit signal below the seed-noise floor; adversarial
fine-tune reproducibly collapses — see `famail_temporal/baselines/gan/` and
`B0_DECISION_BRIEF.md`) already demonstrates the same training-instability failure mode this
would likely hit again. Unlikely to justify the cost for an arm that is expected, like the other
three, to show no fairness improvement.

---

## Candidate 4 — Fairness-aware data augmentation via demographic subgroup oversampling

> **✅ SELECTED 2026-07-09 (user decision) — human name: "Demographic Oversampling".** To be implemented on a
> NEW branch off `main` via brainstorm→spec→plan. Locked-in-principle design: a *resampling* baseline that
> duplicates (with light jitter) real seeking trajectories originating in under-served demographic cells;
> rebuild the fairness grid **additively with BOTH demand (pickups) AND supply (seeking presence, via the
> tier-2 supply recount)** — the load-bearing decision; demand-only is perverse. Frame as the naive baseline
> for the **supply-lift** editor (lifting-up) and a direct probe of the demand-endogeneity / leveling-down
> limitation; report a small oversampling dose-response + a random-oversampling placebo, on the same fidelity
> + external-metrics harness. Disclose: duplicates trivially pass fidelity (they are real), so the cost is
> corpus inflation / fabricated (unobserved) demand, not realism. Cost S–M (the memo's "S" is optimistic once
> the additive grid builder + supply recount + dose levels are included).

**Citation:** Ioannis Pastaltzidis, N. Dimitriou, K. Quezada-Tavárez, S. Aidinlis, Thomas
Marquenie, Agata Gurzawska, D. Tzovaras. "Data augmentation for fairness-aware machine learning:
Preventing algorithmic bias in law enforcement systems." *2022 ACM Conference on Fairness,
Accountability, and Transparency (FAccT '22)*. DOI: 10.1145/3531146.3534644.

**Verification:** Semantic Scholar Graph API keyed on the DOI (ACM DL page 403'd on direct
fetch): `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3531146.3534644?fields=title,authors,venue,year,abstract,externalIds`
— title, all 7 authors, venue "Conference on Fairness, Accountability and Transparency", year
2022 confirmed; DBLP key `conf/fat/PastaltzidisDQA22` cross-confirms indexing.

**Method summary:** Not trajectory data — the paper works on RWF-2000, a video violent-activity
dataset — but it is a direct hit on the "fairness-aware data augmentation" search theme and is
exactly the paradigm a KDD reviewer would expect cited. It identifies "issues of
overrepresentation of minority subjects in violence situations that limit the external validity
of the dataset" and "propose[s] data augmentation techniques to rebalance the dataset," showing
synthetically generated samples can create "more balanced datasets," mitigating subgroup bias
without touching the learning algorithm.

**Applicability to FAMAIL:** The transferable idea is demographic-targeted oversampling rather
than perturbation: identify under-served demographic grid-cells (by the same district-level
demographic features FAMAIL's own fairness metrics use) and duplicate/lightly-jitter real
seeking trajectories originating there to shift the corpus's supply-demand balance, instead of
editing existing trajectories toward a fairness objective. This is a resampling baseline, not a
perturbation baseline — orthogonal to the other three built arms, and it directly probes a
different question than the already-run random-subset placebo (memory: [[placebo-pickup]]):
here the oversampling target is demographically *informed*, not random. **Build cost: S**
(mostly bookkeeping — select and duplicate/jitter trajectories weighted by demographic-cell
under-service; reuses the existing fairness and fidelity scorers unchanged).

**Recommendation: ADOPT-CANDIDATE.** Cheapest of the five, most directly answers the
"fairness-aware augmentation for mobility" theme a reviewer will look for, and is a genuinely
different mechanism (resampling vs. perturbation) from what's already built — the clearest case
for a 4th arm if the user wants one.

---

## Candidate 5 — Selection-strategy-driven trajectory augmentation

**Citation:** Adam Nordling. "A Systematic Approach for Selecting Trajectories for Data
Augmentation." arXiv:2606.10938 (submitted 9 June 2026).

**Verification:** `https://arxiv.org/abs/2606.10938` — title, author, and submission date
confirmed directly on the arXiv abstract page. **No peer-reviewed venue could be confirmed** —
this is currently an arXiv-only preprint; flagged accordingly below.

**Method summary:** Evaluates five systematic strategies (Outlierness, Diversity,
Representativeness, Uncertainty, Random) for choosing *which* trajectories in a corpus to
augment, across four datasets (animal behavior, maritime traffic, urban traffic) and several ML
models. Finds "systematic strategies, particularly Outlierness and Uncertainty, demonstrated
superior stability compared to random sampling," but that "augmentation benefits are
conditional — while it successfully repairs sparse datasets, it can introduce noise in
high-quality datasets," and identifies "physical limitations in high-velocity domains where
standard perturbation techniques diverge in feature space."

**Applicability to FAMAIL:** This is a meta-level *targeting* question (which trajectories to
touch) rather than a new perturbation mechanism (how to touch them) — orthogonal to all four
other candidates and to FAMAIL's existing arms. It could in principle be layered onto FAMAIL's
own editor as an alternative to its current trajectory-selection heuristic (e.g., prioritize
outlier or high-uncertainty seeking trajectories rather than the current selection rule), testing
whether smarter targeting changes fairness/fidelity outcomes independent of the edit itself.
**Build cost: S** (a selection heuristic layered on the existing pipeline) but it answers a
different research question than "baseline perturbation methods," so it doesn't slot cleanly
into the same 4-arm comparison table.

**Recommendation: DEFER.** Very recent (weeks old at review time), single-author, arXiv-only
preprint with no confirmed peer review — thin evidentiary basis to lean on for a KDD submission.
Also answers "which trajectories to select" rather than "how to perturb them," which is a
different axis than this memo was scoped to compare. Worth revisiting only if reviewers
specifically probe augmentation-target selection.

---

## Sources considered and not carried forward (topical fit, not failed verification)

These surfaced during the search but were set aside for scope/topic reasons, not because
citation verification failed — noted for completeness since the brief asks what was rejected:

- **ST-TrajGAN, TCAC-GAN, DP-TrajGAN** (ScienceDirect/IEEE Xplore GAN-family trajectory
  generators) — same family as Candidate 3 (LSTM-TrajGAN); not pursued further once Candidate 3's
  build-cost/prior-failure-mode analysis made the whole GAN-generation family a DEFER.
- **Ioannis Pastaltzidis et al.'s FAccT paper aside**, no other "fairness-aware mobility
  augmentation"-specific (i.e., trajectory-native) paper surfaced; the mobility-fairness hits
  found (e.g., FairST demand prediction, AAAI; ride-hailing rebalancing work) are fairness-aware
  *prediction/allocation* models, not *data augmentation* methods, so off-theme for this memo.
- **"Comprehensive Bias Mitigation and Evaluation Framework"** (IJCA, a non-indexed venue) — not
  pursued given inability to confirm indexing/peer-review status in the time available.
- **CHOP** (arXiv:2603.02004, visuomotor navigation obstacle avoidance) and **Counterfactual
  Fairness Filter for Fair-Delay Multi-Robot Navigation** (arXiv:2305.11465) — real papers, but
  about robot-navigation fairness/counterfactuals, not GPS taxi trajectory augmentation; too far
  off-topic to adapt within scope.

**Verification-access note:** the ACM Digital Library returned HTTP 403 to direct WebFetch for
two DOIs (Candidates 1 and 4); both were independently confirmed via the Semantic Scholar Graph
API keyed on the same DOI, cross-checked against DBLP conference keys. No citation in this memo
was accepted on the strength of an AI-generated search summary alone — every entry above has a
primary-source verification URL.

---

**Decision gate: none of these are built without an explicit user go-ahead.**
