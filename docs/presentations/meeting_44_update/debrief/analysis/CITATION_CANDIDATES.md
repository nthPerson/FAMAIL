# Citation Candidates — FATE (KDD 2027), Related-Work Reinforcement

**Task:** Find VERIFIED citation candidates so every class of related approach named in the
introduction has ≥ 2 citations, with recent (2025–2026) works where they genuinely fit.
**Prepared:** 2026-07-23. **Rule enforced:** every proposed work below was verified during
this task by loading a primary source (venue page, DBLP record, and/or arXiv/PMLR/ACL abstract
page); metadata is copied verbatim; each carries a verbatim abstract quote. No memory-only entries.

**Verification-access notes.**
- **OpenReview** is behind a browser bot-check and could not be loaded; where a work surfaced only
  there I re-verified via DBLP + arXiv/PMLR/ACL abstract pages instead.
- **ACM DL / AAAI OJS** publisher pages were not fetched directly; venue/pages/DOI for those were
  taken from the **DBLP** publication records (JSON API) and the abstract from the arXiv mirror →
  graded `VERIFIED-DBLP+ABSTRACT` (not `VERIFIED-VENUE`).
- DBLP JSON API (`https://dblp.org/search/publ/api?q=...&format=json`) was fetchable throughout.

**Confidence legend:** `VERIFIED-VENUE` = publisher/proceedings page loaded · `VERIFIED-DBLP+ABSTRACT`
= DBLP record + abstract page loaded · `ARXIV-ONLY` = arXiv abstract loaded, no peer-reviewed venue
(permitted only for 2025–2026 recency slots).

---

## Class (a) — HIGH — Model-level / in-processing fairness for IL / RL / mobility policy learning
Existing intro cite: `zheng2023` (fairness-regularized ride-hailing demand prediction; the only cite for this class).
**Result: 2 strong recommended companions (both peer-reviewed, both recent), + 2 verified arXiv extras.**

### ★ Candidate a1 — `lamalfa2026fairppo`  · VERIFIED-DBLP+ABSTRACT
- **Title (verbatim):** Fairness Aware Reinforcement Learning via Proximal Policy Optimization
- **Authors:** Gabriele La Malfa, Jie M. Zhang, Michael Luck, Elizabeth Black
- **Venue:** Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2026, pp. 22725–22733
- **DOI:** 10.1609/aaai.v40i27.39434 · **arXiv mirror:** 2502.03953
- **URLs loaded:** `https://arxiv.org/abs/2502.03953` (abstract) · `https://dblp.org/search/publ/api?q=Fairness+Aware+Reinforcement+Learning+Proximal+Policy+Optimization&format=json` (AAAI 2026 record: pages + DOI) · `https://dblp.org/search?q=Gabriele+La+Malfa` (author page confirming AAAI 2026 acceptance)
- **Abstract quote:** "This paper introduces fairness in Proximal Policy Optimization (PPO) with a penalty term derived from a fairness definition such as demographic parity, counterfactual fairness, or conditional statistical parity. The proposed method, which we call Fair-PPO, balances reward maximisation with fairness by integrating two penalty components…"
- **Intro claim it supports:** "In-processing fairness for sequential decision-making augments the policy-optimization objective with a fairness penalty while leaving the demonstration data unchanged [zheng2023, lamalfa2026fairppo]."
- **Note:** Cleanest RL-side companion; adds a fairness term to the training objective (the exact mechanism our class names). Peer-reviewed top venue, 2026 → fills the "recent" slot.

### ★ Candidate a2 — `zhao2025fairdrlst`  · VERIFIED-DBLP+ABSTRACT
- **Title (verbatim):** FairDRL-ST: Disentangled Representation Learning for Fair Spatio-Temporal Mobility Prediction
- **Authors:** Sichen Zhao, Wei Shao, Jeffrey Chan, Ziqi Xu, Flora Salim
- **Venue:** ACM SIGSPATIAL 2025 (Advances in Geographic Information Systems), pp. 103–106 (short/research paper)
- **DOI:** 10.1145/3748636.3762713 · **arXiv mirror:** 2508.07518
- **URLs loaded:** `https://arxiv.org/abs/2508.07518` (abstract) · `https://dblp.org/search/publ/api?q=FairDRL-ST+disentangled+fair+spatio-temporal+mobility&format=json` (SIGSPATIAL 2025 record: pages + DOI)
- **Abstract quote:** "we propose a novel framework, FairDRL-ST, based on disentangled representation learning, to address fairness concerns in spatio-temporal prediction, with a particular focus on mobility demand forecasting. By leveraging adversarial learning and disentangled representation learning, our framework learns to separate attributes that contain sensitive information."
- **Intro claim it supports:** "Model-level fairness for mobility learning regularizes the training objective (e.g., adversarial/disentangled representations) to equalize outcomes across demographic groups [zheng2023, zhao2025fairdrlst]."
- **Note:** Same subclass as the existing `zheng2023` (in-processing fairness for a mobility-prediction model) — the tightest domain match. Peer-reviewed, 2025.

### Extra a3 (optional survey cite) — `reuel2024fairrlsurvey`  · ARXIV-ONLY
- **Title (verbatim):** Fairness in Reinforcement Learning: A Survey · **Authors:** Anka Reuel, Devin Ma · **arXiv:** 2405.06909, 2024
- **URL loaded:** `https://arxiv.org/abs/2405.06909`
- **Abstract quote:** "We start by reviewing where fairness considerations can arise in RL, then discuss the various definitions of fairness in RL … [and] highlight the methodologies researchers used to implement fairness in single- and multi-agent RL systems."
- **Use only if** a survey-style umbrella cite is wanted; arXiv-only, so subordinate to a1/a2.

### Extra a4 (optional recent framework) — `cimpean2025farel`  · ARXIV-ONLY
- **Title (verbatim):** Fairness-Aware Reinforcement Learning (FAReL): A Framework for Transparent and Balanced Sequential Decision-Making
- **Authors:** Alexandra Cimpean, Nicole Orzan, Catholijn Jonker, Pieter Libin, Ann Nowé · **arXiv:** 2509.22232, 2025
- **URL loaded:** `https://arxiv.org/abs/2509.22232`
- **Abstract quote:** "To capture fairness, we propose an extended Markov decision process, $f$MDP, that explicitly encodes individuals and groups … [and] formulate a fairness framework that computes fairness measures over time."
- **Use only if** a second RL example is wanted alongside a1; arXiv-only.

---

## Class (b) — HIGH — Fair data GENERATION (synthesizing fair data)
Existing intro cites: `xu2018fairgan` (FairGAN, tabular), `vanbreugel2021decaf` (DECAF, causal GAN, tabular).
**Result: 2 recent companions (1 peer-reviewed diffusion, 1 arXiv). No trajectory-specific *fair* generator exists — see honesty note.**

### ★ Candidate b1 — `hastingsblow2025diffaug`  · VERIFIED-VENUE
- **Title (verbatim):** Data augmentation via diffusion model to enhance AI fairness
- **Authors:** Christina Hastings Blow, Lijun Qian, Camille Gibson, Pamela Obiomon, Xishuang Dong
- **Venue:** Frontiers in Artificial Intelligence, vol. 8, article 1530397, 2025 · **DOI:** 10.3389/frai.2025.1530397
- **URL loaded:** `https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1530397/full`
- **Abstract quote:** "Tabular Denoising Diffusion Probabilistic Model (Tab-DDPM) … was utilized with different amounts of generated data for data augmentation. … Experimental results demonstrate that the synthetic data generated by Tab-DDPM improves fairness in binary classification."
- **Intro claim it supports:** "A complementary line synthesizes fair training data — e.g., diffusion-based generation of samples that make downstream classifiers fairer [xu2018fairgan, vanbreugel2021decaf, hastingsblow2025diffaug]."
- **Note:** Hits the "diffusion-based fair synthesis" target squarely and is peer-reviewed. **Tabular** (like the two existing cites), not trajectory. Frontiers is a peer-reviewed journal but lower-prestige than the paper's other venues — fine for a related-work cite.

### Candidate b2 — `sikder2024fair4free`  · ARXIV-ONLY
- **Title (verbatim):** Fair4Free: Generating High-fidelity Fair Synthetic Samples using Data Free Distillation
- **Authors:** Md Fahim Sikder, Daniel de Leng, Fredrik Heintz · **arXiv:** 2410.01423, 2024 (DBLP lists **CoRR only** — no peer-reviewed venue)
- **URLs loaded:** `https://arxiv.org/abs/2410.01423` (abstract) · `https://dblp.org/search/publ/api?q=Fair4Free+data+free+distillation+synthetic&format=json` (confirms CoRR-only)
- **Abstract quote:** "This work presents Fair4Free, a novel generative model to generate synthetic fair data using data-free distillation in the latent space. … our synthetic samples outperform state-of-the-art models in all three criteria (fairness, utility and synthetic quality) … for both tabular and image datasets."
- **Intro claim it supports:** "Recent fair generative models produce synthetic samples optimized jointly for fairness, utility, and fidelity [sikder2024fair4free]."
- **Note:** Recent, explicitly a *fair* generator (tabular + image). arXiv/CoRR-only → use as the recency companion to the peer-reviewed b1, not as sole support.

**Honesty note for (b):** I found **no** genuinely *fair* trajectory/mobility/spatiotemporal **generator**. The closest,
ATLAS (Li, Hong, Shirakawa, Chang, "Learning Demographic-Conditioned Mobility Trajectories with Aggregate Supervision,"
arXiv 2603.03275, 2026; abstract loaded), conditions trajectory generation on demographics but targets demographic
**realism/heterogeneity**, *not* fairness/debiasing — quoting it as "fair generation" would misrepresent it, so it is
**not** proposed. Consistent with the existing tabular cites, the honest recent additions for this class are tabular/diffusion (b1, b2).

---

## Class (c) — HIGH — Evidence that IL / LfD inherits or AMPLIFIES bias in the data
Existing cites: `zhang2022cgail` (cGAIL capability cite), `ensign2018` + `lumisaac2016` (feedback loops, not IL-specific).
**Result: no IL-specific empirical demographic-bias-inheritance paper exists. Two peer-reviewed *supervised* bias-amplification neighbors proposed, labeled honestly.**

### ★ Candidate c1 — `zhao2017menshopping`  · VERIFIED-VENUE  *(supervised neighbor, NOT IL)*
- **Title (verbatim):** Men Also Like Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints
- **Authors:** Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, Kai-Wei Chang
- **Venue:** Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2979–2989 (Best Long Paper)
- **URL loaded:** `https://aclanthology.org/D17-1323/`
- **Abstract quote:** "models trained on these datasets further amplify existing bias. For example, the activity cooking is over 33% more likely to involve females than males in a training set, and a trained model further amplifies the disparity to 68% at test time."
- **Intro claim it supports:** "Models trained to reproduce patterns in human-sourced data do not merely inherit but can *amplify* the demographic biases in that data [zhao2017menshopping]."
- **HONEST LABEL:** Supervised **structured prediction** (vision + NLP), *not* imitation learning. Cite as the canonical empirical demonstration of the *phenomenon* (bias amplification from human-sourced training data), explicitly framed as a neighbor to our IL setting.

### ★ Candidate c2 — `wang2021directional`  · VERIFIED-VENUE  *(supervised neighbor, NOT IL)*
- **Title (verbatim):** Directional Bias Amplification
- **Authors:** Angelina Wang, Olga Russakovsky
- **Venue:** Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139, 2021, pp. 10882–10893
- **URL loaded:** `https://proceedings.mlr.press/v139/wang21t.html`
- **Abstract quote:** "we focus on one aspect of the problem, namely bias amplification: the tendency of models to amplify the biases present in the data they are trained on. … We introduce and analyze a new, decoupled metric for measuring bias amplification, $BiasAmp_{\rightarrow}$ (Directional Bias Amplification)."
- **Intro claim it supports:** "Bias amplification — models emitting biased predictions at a higher rate than the training data warrants — is a measurable, general phenomenon in learning from human-labeled data [zhao2017menshopping, wang2021directional]."
- **HONEST LABEL:** Supervised vision; a metric/analysis refinement of c1. Within the 2020–2026 window.

**Honesty note for (c):** Despite targeted searches (behavioral cloning / GAIL / policy learning + demographic disparity /
service inequity across driving, dispatch, healthcare), **no** paper empirically shows IL/LfD *inheriting or amplifying
demographic bias* from demonstrations. "Data Quality in Imitation Learning" (Belkhale, Cui, Sadigh, arXiv 2306.02437;
abstract loaded) concerns distribution-shift/data-quality, *not* demographic bias — **excluded**. The genuine evidence is
the supervised bias-amplification line above (c1, c2), which the task explicitly permits as a labeled neighbor; it pairs
naturally with the already-cited feedback-loop works (`ensign2018`, `lumisaac2016`). A possible IL-adjacent *recency*
option — RLHF/reward-model group-bias work (e.g., "Benchmarking Group Fairness in Reward Models," arXiv 2503.07806, not
loaded/verified) — is LLM-alignment-specific and far from the mobility setting; I do **not** recommend forcing it.

---

## Class (d) — MEDIUM — Preprocessing by reweighing/resampling for fairness
Existing cites: `kamirancalders2012` (seminal), `zietlow2022`. Task ceiling: **at most 1** recent companion, only if genuinely fitting.
**Result: 1 fitting recent companion (arXiv-only).**

### Candidate d1 — `yang2025iffair`  · ARXIV-ONLY
- **Title (verbatim):** IFFair: Influence Function-driven Sample Reweighting for Fair Classification
- **Authors:** Jingran Yang, Min Zhang, Lingfeng Zhang, Zhaohui Wang, Yonggang Zhang · **arXiv:** 2512.07249, 2025
- **URL loaded:** `https://arxiv.org/abs/2512.07249`
- **Abstract quote:** "we propose a pre-processing method IFFair based on the influence function. … IFFair only uses the influence disparity of training samples on different groups as a guidance to dynamically adjust the sample weights during training without modifying the network structure, data features and decision boundaries."
- **Intro claim it supports:** "Preprocessing approaches reweight training samples to mitigate bias before learning [kamirancalders2012, yang2025iffair]."
- **Note:** Directly a *reweighting-for-fairness* method (the exact subclass), and recent — a clean fit for the single allowed companion. arXiv-only (Dec 2025); include only if a recent reweighting citation is desired, otherwise seminal-only is defensible for a MEDIUM class.

---

## Class (e) — LOW — Trajectory editing/perturbation as fairness data augmentation
Existing cites: `kurakin2017ifgsm`, `goodfellow2015fgsm`, `madry2018pgd`, `hu2023stifgsm`. Highest bar (close-competitor territory).
**Result: NO FIT — recommend adding nothing. Absence is a useful finding (supports novelty). See below.**

---

## CLASSES WITH NO GOOD RECENT FIT

- **Class (e) — fairness-motivated trajectory/spatiotemporal *editing* as data augmentation: KEEP AS-IS (no addition).**
  Targeted searches surfaced trajectory *editing* only for realism/environmental-constraint satisfaction (not fairness),
  and fairness *counterfactual augmentation* only for **graph/GNN** representations — nearest neighbor Fair-ICD
  ("Improving Fairness in Graph Neural Networks via Counterfactual Debiasing," arXiv 2508.14683; surfaced in search, **not**
  loaded/verified, **not** proposed) operates on graphs, not trajectories. No work *edits* a real spatiotemporal/trajectory
  corpus for a *fairness* objective. This gap is exactly the space FATE occupies — the honest recommendation is to claim the
  absence rather than cite a non-fit.

- **Class (b) — fair *trajectory* generation specifically: none exists (tabular/diffusion added instead).**
  See the (b) honesty note: the only demographic-conditioned trajectory generator found (ATLAS) targets realism, not fairness.

- **Class (c) — IL-specific empirical bias-inheritance: none exists (supervised neighbors added instead).**
  See the (c) honesty note: proposed neighbors are supervised bias-amplification, labeled as such.

---

## Summary table

| Class | Key | Venue / Year | Confidence | Recommend |
|---|---|---|---|---|
| a | `lamalfa2026fairppo` | AAAI 2026 | VERIFIED-DBLP+ABSTRACT | ★ primary |
| a | `zhao2025fairdrlst` | ACM SIGSPATIAL 2025 | VERIFIED-DBLP+ABSTRACT | ★ primary |
| a | `reuel2024fairrlsurvey` | arXiv 2024 | ARXIV-ONLY | optional |
| a | `cimpean2025farel` | arXiv 2025 | ARXIV-ONLY | optional |
| b | `hastingsblow2025diffaug` | Frontiers in AI 2025 | VERIFIED-VENUE | ★ primary |
| b | `sikder2024fair4free` | arXiv/CoRR 2024 | ARXIV-ONLY | companion |
| c | `zhao2017menshopping` | EMNLP 2017 | VERIFIED-VENUE | ★ (supervised neighbor) |
| c | `wang2021directional` | ICML 2021 | VERIFIED-VENUE | ★ (supervised neighbor) |
| d | `yang2025iffair` | arXiv 2025 | ARXIV-ONLY | optional (≤1) |
| e | — | — | — | none (novelty) |

**Do NOT re-propose (already in refs.bib):** zheng2023, xu2018fairgan, vanbreugel2021decaf, zhang2022cgail,
ensign2018, lumisaac2016, kamirancalders2012, zietlow2022, kurakin2017ifgsm, goodfellow2015fgsm, madry2018pgd, hu2023stifgsm.
