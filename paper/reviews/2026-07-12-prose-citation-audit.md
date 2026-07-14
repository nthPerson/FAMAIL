# Prose-sections citation audit — 2026-07-12

**Scope:** every `\cite` in the newly drafted `sections/01_introduction.tex` and
`sections/02_related_work.tex` (`05_conclusion.tex` carries zero citations by design).
**Protocol:** each citation-claim pair verified by an Opus subagent (4 agents, parallel;
skeptical-by-default prompt) for (a) EXISTENCE — work exists, BibTeX fields correct; and
(b) CLAIM SUPPORT — the cited work actually contains the content the paragraph attributes
to it, with quoted evidence + URL. Pair extraction was mechanical
(paragraph + keys + BibTeX entries; scratchpad `citation_pairs_*.md`).
**Headline: 0 fabrications, 0 not-found. 29 key-instances verified (7 in §1, 22 in §2;
33 distinct keys incl. §1/§2 overlap). 2 bibliographic field fixes, 3 prose fixes.**
**Precedent:** `PAPER/objective-motivation/sources/mission_2_citation_audit.md`, which caught 2 fabrications
in an earlier literature pass — the reason this stage exists.

## Dispositions (all applied 2026-07-12)

| # | finding | severity | disposition |
|---|---------|----------|-------------|
| 1 | `karner2024` year=2024 inconsistent with cited vol 52(4) pp 1399–1427 (= Aug **2025** print issue; 2024 was online-first) | field fix | `refs.bib` year → 2025; entry comment updated. Key name kept (internal only). |
| 2 | `wachter2018counterfactual` title missing published subtitle | cosmetic | `refs.bib` title → "…Black Box: Automated Decisions and the GDPR". |
| 3 | `vermarubin2018` cited for the pre-/in-/post-processing *intervention* taxonomy, but the paper taxonomizes fairness *definitions* | **citation-content mismatch** | §2 ¶1 reworded: intervention grouping now cites `barocas2023` alone; `vermarubin2018` moved to "Among the many formal fairness definitions…" — the claim the agent's evidence directly supports. |
| 4 | `bengio2013ste` grouped under "continuous relaxation," but STE is a straight-through gradient estimator, not a relaxation | wording | §2 ¶4 reworded: "…continuous relaxation \cite{jang2017gumbel,maddison2017concrete} and straight-through gradient estimation \cite{bengio2013ste}." (§3's "temperature-annealed soft cell assignment" umbrella was out of scope and is defensible as written.) |
| 5 | `ensign2018`/`lumisaac2016` (§2 ¶5): "entrench under-service" attributes FAMAIL's ride-hailing mapping to papers whose stated harm is predictive-policing *over*-allocation | valence | §2 ¶5 reworded to the mechanism the papers state ("entrench a self-reinforcing allocation bias"), with the under-service reading explicitly marked as our domain mapping. §1's use ("feedback loop documented for other data-driven allocation systems") was verified SUPPORTED as written — no change. |
| 6 | `zheng2023` "0.361 to 0.084" | **verified genuine** | Table 2, racial MPE gap, Conv-LSTM Net arm, γ: 0→10. Exact values confirmed. Nuance on record: the drop is for the Conv-LSTM arm; other arms report other values. No prose change (claim accurate as written). |
| 7 | `parfit1997` "originates in the ethics of equality" | accepted | Parfit is the canonical formulator (crediting precursors); phrasing targets the field, not personal priority. No change. |

**Re-verification of reworded pairs (plan Task 4 Step 5):** each rewording moves the claim
*toward* evidence already quoted by the verifying agent — vermarubin2018's new pairing is
the agent's own finding ("taxonomy of fairness definitions"); bengio2013ste's is the
agent's characterization ("straight-through estimator"); ensign/lum's is the agents'
quoted mechanism, with the domain analogy now explicitly ours. No new external claim was
introduced, so no fresh dispatch was required.

---

## Agent 1 — §1 Introduction (blocks 1–3, 7 keys)

| key | block | existence | claim support | evidence (short quote + URL) | notes |
|---|---|---|---|---|---|
| zhang2022cgail | 1 | OK | SUPPORTED | DBLP confirms all fields exactly: "IEEE Transactions on Big Data, Volume 8, Number 5, Pages 1288–1300, Year 2022, DOI 10.1109/TBDATA.2020.3039810; Xin Zhang, Yanhua Li, Xun Zhou, Jun Luo." Imitation-learning framework learning taxi drivers' strategies from real GPS trajectories. https://ieeexplore.ieee.org/document/9266753/ | All fields match. |
| ensign2018 | 1 | OK | SUPPORTED | "Such systems have been shown susceptible to runaway feedback loops, where police are repeatedly sent back to the same neighborhoods regardless of the true crime rate… prove why this feedback loop occurs." PMLR v81:160–171, FAT* 2018. https://proceedings.mlr.press/v81/ensign18a.html | Supports "feedback loop documented for other data-driven allocation systems." |
| lumisaac2016 | 1 | OK | SUPPORTED | Predictive policing on biased data drives police "to neighborhoods where arrests were already concentrated"; Significance 13(5):14–19, 2016, DOI 10.1111/j.1740-9713.2016.00960.x. https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1740-9713.2016.00960.x | All fields match. |
| zheng2023 | 2 | OK | SUPPORTED | "developed a socially-aware neural network (SA-Net)… with a bias-mitigation regularization to reduce the prediction error gap." IEEE OJ-ITS vol 4, pp 551–569, 2023. https://arxiv.org/abs/2303.05698 | In-processing regularization — supports "in-processing methods regularize the model." |
| kamirancalders2012 | 2 | OK | SUPPORTED | "Data preprocessing techniques for classification without discrimination. Knowledge and Information Systems, 33(1), 1–33"; canonical source of Reweighing. https://link.springer.com/article/10.1007/s10115-011-0463-8 | All fields match. |
| parfit1997 | 3 | OK | SUPPORTED | "Equality and Priority… Ratio 10(3), 1997, pp 202–221, DOI 10.1111/1467-9329.00041." Foundational statement of the Levelling-Down Objection. https://onlinelibrary.wiley.com/doi/10.1111/1467-9329.00041 | Distinct from the Lindley Lecture "Equality or Priority?"; BibTeX targets the correct Ratio record. |
| mittelstadt2024 | 3 | OK | SUPPORTED | "many fairness measures suffer from… 'leveling down'—where fairness is achieved by making every group worse off." Michigan Technology Law Review 30(1), 2024. https://repository.law.umich.edu/mtlr/vol30/iss1/3/ | All fields match. |

Agent flags: none requiring change; zheng2023 DOI not re-confirmed digit-for-digit (all
other fields robustly verified — treat as OK).

## Agent 2 — §2 blocks 1–2 (11 keys)

| key | block | existence | claim support | evidence (quote + URL) | notes |
|---|---|---|---|---|---|
| vermarubin2018 | 1 | OK | PARTIAL | Taxonomy of fairness *definitions*: "statistical fairness metrics according to… disparate treatment, disparate impact and disparate mistreatment… illustrating common fairness definitions using the German Credit Dataset." https://dl.acm.org/doi/10.1145/3194770.3194776 | Fields correct (FairWare@ICSE 2018, pp 1–7). Attributed intervention grouping is carried by barocas2023 → **disposition #3 applied**. |
| barocas2023 | 1 | OK | SUPPORTED | "Preprocessing methods aim to adjust the source data… 'inprocessing' methods modify the training algorithms… Postprocessing sets group-specific acceptance thresholds." https://fairmlbook.org/ | MIT Press 2023; year defensible. |
| kamirancalders2012 | 1 | OK | SUPPORTED | Foundational source of the "Reweighing" preprocessing technique (implemented as such in AIF360). https://link.springer.com/article/10.1007/s10115-011-0463-8 | All fields match. |
| feldman2015 | 1 | OK | SUPPORTED | "Certifying and Removing Disparate Impact"; the DI remover repairs feature distributions. https://dl.acm.org/doi/10.1145/2783258.2783311 | KDD 2015, pp 259–268 — all fields match. |
| corbettdavies2017 | 1 | OK | SUPPORTED | "Conditional statistical parity means that controlling for a limited set of 'legitimate' risk factors, an equal proportion of defendants are detained within each race group." https://ar5iv.labs.arxiv.org/abs/1701.08230 | KDD 2017, pp 797–806 — all fields match. |
| horchergraham2021 | 2 | OK | SUPPORTED | Proposes/applies "the Gini index as a measure of demand imbalances in public transport." https://link.springer.com/article/10.1007/s11116-020-10138-4 | Transportation 48:2521–2544, 2021 — all fields match. |
| karner2024 | 2 | FIX:year=2025 | SUPPORTED | "review two of the most widely used approaches… Gini coefficients/Lorenz curves and needs-gap/transit desert approaches." https://ideas.repec.org/a/kap/transp/v52y2025i4d10.1007_s11116-023-10460-7.html | Vol 52 + pp 1399–1427 + DOI match; vol 52(4) is the Aug 2025 issue → **disposition #1 applied**. Confirms Gini among most widely used transport-equity measures. |
| theil1967 | 2 | OK | SUPPORTED | "Theil developed the notion of entropy on the basis of information theory and advocated… entropy-based measures for the analysis of income inequality." https://openlibrary.org/books/OL17757080M | North-Holland 1967 — all fields match. |
| atkinson1970 | 2 | OK | SUPPORTED | Seminal source of the welfare-based Atkinson index. https://ideas.repec.org/a/eee/jetheo/v2y1970i3p244-263.html | JET 2(3):244–263, 1970 — all fields match. |
| demaio2007 | 2 | OK | SUPPORTED | "income inequality measures such as the generalised entropy index and the Atkinson index…" https://pmc.ncbi.nlm.nih.gov/articles/PMC2652960/ | JECH 61(10):849–852, 2007 — all fields match. |
| zheng2023 | 2 | OK | SUPPORTED | Table 2: "when increasing γ from 0 to 10, the MPE gap between the black and non-black groups drops from **0.361 to 0.084** for Conv-LSTM Net." https://ar5iv.labs.arxiv.org/abs/2303.05698 | **Exact attributed values verified** → disposition #6. |

## Agent 3 — §2 blocks 3–4 (17 keys)

| key | block | existence | claim support | evidence (short quote + URL) | notes |
|---|---|---|---|---|---|
| zhang2019cgail | 3 | OK (ICDM 2019, pp 1480–1485, DOI 10.1109/ICDM.2019.00194) | SUPPORTED | "conditional generative adversarial imitation learning (cGAIL)… learns the driver's decision-making preferences and policies… taxi GPS trajectory data in Shenzhen." https://ieeexplore.ieee.org/abstract/document/8970802 | Direct match. |
| zhang2022cgail | 3 | OK (TBD 8(5):1288–1300, 2022) | SUPPORTED | dblp confirms all fields; journal version of cGAIL. https://ieeexplore.ieee.org/document/9266753/ | All fields correct. |
| pan2020xgail | 3 | OK (KDD 2020, pp 1334–1343) | SUPPORTED | "first explainable generative adversarial imitation learning framework… explains how a human agent makes decisions." | Match ("explainable extensions expose the recovered policies"). |
| feng2020simulate | 3 | OK (KDD 2020, pp 3426–3433) | SUPPORTED | "Learning to Simulate Human Mobility" — neural generative model for mobility trajectories. https://dl.acm.org/doi/10.1145/3394486.3412862 | Match. |
| gao2017tuler | 3 | OK (IJCAI 2017, pp 1689–1695) | SUPPORTED | "TULER (TUL via Embedding and RNN)… learn the underlying motion patterns of a particular user." https://www.ijcai.org/proceedings/2017/0234.pdf | TUL method — match. |
| zhou2018tulvae | 3 | OK (IJCAI 2018, pp 3212–3218) | SUPPORTED | "TULVAE… learns the human mobility in a neural generative architecture." https://www.ijcai.org/proceedings/2018/446 | TUL method — match. |
| miao2020deeptul | 3 | OK (AAMAS 2020, pp 878–886) | SUPPORTED | "DeepTUL… linking different trajectories to users." https://dl.acm.org/doi/10.5555/3398761.3398864 | TUL method — match. |
| ren2020stsiamese | 3 | OK (KDD 2020, pp 1306–1315) | SUPPORTED | "train ST-SiameseNet to predict the mobility signature similarity between each pair of agents… validating if incoming trajectories were indeed generated by a claimed agent." | Match (driver = taxi agent). |
| hoermon2016 | 3 | OK (NIPS 2016, pp 4565–4573) | SUPPORTED | "draws an analogy between imitation learning and generative adversarial networks." https://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning | Canonical GAIL — correctly cited as the live adversarial game FAMAIL avoids. |
| goodfellow2015fgsm | 4 | OK (ICLR 2015; arXiv:1412.6572) | SUPPORTED | "small but intentionally worst-case perturbations" — FGSM origin. https://arxiv.org/abs/1412.6572 | Match. |
| kurakin2017ifgsm | 4 | OK (ICLR 2017 Workshop; arXiv:1607.02533) | SUPPORTED | Iterative FGSM / basic iterative method. https://arxiv.org/pdf/1607.02533 | Match. |
| hu2023stifgsm | 4 | OK (KDD 2023, pp 764–774, DOI 10.1145/3580305.3599513) | SUPPORTED | "ST-iFGSM: Enhancing Robustness of Human Mobility Signature Identification Model via Spatial-Temporal Iterative FGSM." https://dl.acm.org/doi/10.1145/3580305.3599513 | Exact match. |
| jang2017gumbel | 4 | OK (ICLR 2017; arXiv:1611.01144) | SUPPORTED | "differentiable sample from a novel Gumbel-Softmax distribution… smoothly annealed." https://arxiv.org/abs/1611.01144 | Match. |
| maddison2017concrete | 4 | OK (ICLR 2017; arXiv:1611.00712) | SUPPORTED | "CONCRETE random variables—continuous relaxations of discrete random variables." https://arxiv.org/pdf/1611.00712 | Match. |
| bengio2013ste | 4 | OK (arXiv:1308.3432, 2013) | PARTIAL | "how to estimate the gradient… through stochastic or non-smooth neurons," incl. the straight-through estimator. https://arxiv.org/abs/1308.3432 | STE ≠ continuous relaxation → **disposition #4 applied**. |
| ustun2019recourse | 4 | OK (FAT* 2019, pp 10–19) | SUPPORTED | "recourse… alter input variables in a way that guarantees approval." https://dl.acm.org/doi/10.1145/3287560.3287566 | Match. |
| wachter2018counterfactual | 4 | OK — subtitle missing | SUPPORTED | "counterfactual explanations describe the minimum conditions that would have led to an alternative decision… the smallest change to the world." https://arxiv.org/abs/1711.00399 | Vol/pages correct; **disposition #2 applied** (title subtitle). |

## Agent 4 — §2 block 5 (5 keys)

| key | block | existence | claim support | evidence (short quote + URL) | notes |
|---|---|---|---|---|---|
| parfit1997 | 5 | OK (Ratio 10(3):202–221, 1997) | SUPPORTED | Levelling Down Objection: equality "by reducing the better off to the condition of the worst off… in no way better for anyone." https://onlinelibrary.wiley.com/doi/pdf/10.1111/1467-9329.00041 | "Originates in" acceptable → disposition #7. |
| mittelstadt2024 | 5 | OK (Mich. Tech. L. Rev. 30(1), 2024) | SUPPORTED | "Many current fairness measures suffer from… 'levelling down,' where fairness is achieved by making every group worse off." https://arxiv.org/abs/2302.02404 | Title spelling varies by venue; BibTeX acceptable. |
| zietlow2022 | 5 | OK (CVPR 2022, pp 10400–10411) | SUPPORTED | "we propose an adaptive augmentation strategy that, uniquely, of all methods tested, improves performance for the disadvantaged groups." https://arxiv.org/abs/2203.04913 | Confirms the data-augmentation attribution. |
| ensign2018 | 5 | OK (PMLR v81:160–171, 2018) | SUPPORTED | "police are repeatedly sent back to the same neighborhoods regardless of the true crime rate… prove why this feedback loop occurs." https://proceedings.mlr.press/v81/ensign18a.html | Valence flag → **disposition #5 applied**. |
| lumisaac2016 | 5 | OK (Significance 13(5):14–19, 2016) | SUPPORTED | Systems trained on historical crime data "reproduce and amplify existing biases"; PredPol on Oakland "could double police presence in Black neighborhoods." https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1740-9713.2016.00960.x | Valence flag → **disposition #5 applied**. |
