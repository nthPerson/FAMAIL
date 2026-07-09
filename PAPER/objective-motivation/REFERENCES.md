# References — FAMAIL objective-function motivation

**Metadata verified 2026-07-08** (arXiv / DOI / ACM DL / IEEE Xplore / DBLP / Crossref); corrections tracked
in the project's citation audit. **This is the single citation source-of-truth for the objective-motivation
bundle** — the other docs cite by *surname + year* against these entries. Entries are flagged
**[foundational]** or **[recent]**.

---

## Fairness metrics & definitions

- **Corbett-Davies, S., Pierson, E., Feller, A., Goel, S., & Huq, A. (2017).** "Algorithmic Decision Making
  and the Cost of Fairness." *Proc. KDD 2017*, pp. 797–806. **[foundational]** — *conditional statistical
  parity* is formalized here (building on Kamiran et al. 2013 and Dwork et al. 2012), not originated.
- **Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015).**
  "Certifying and Removing Disparate Impact." *Proc. KDD 2015*, pp. 259–268. **[foundational]** — links
  disparate impact to the balanced-error-rate *predictability* of the protected attribute.
- **Kamiran, F., & Calders, T. (2012).** "Data Preprocessing Techniques for Classification without
  Discrimination." *Knowledge and Information Systems* 33(1):1–33. **[foundational]** — suppression,
  massaging, and reweighing.
- **Verma, S., & Rubin, J. (2018).** "Fairness Definitions Explained." *FairWare@ICSE 2018*, pp. 1–7.
  **[recent]**
- **Barocas, S., Hardt, M., & Narayanan, A. (2019/2023).** *Fairness and Machine Learning: Limitations and
  Opportunities.* Free web edition (fairmlbook.org) and MIT Press (2023). **[foundational]** — cite the
  edition used.
- **Frisch, R., & Waugh, F. V. (1933).** "Partial Time Regressions as Compared with Individual Trends."
  *Econometrica* 1(4):387–401. **[foundational]** — the FWL partial-regression identity (with Lovell 1963).
- **Lovell, M. C. (1963).** "Seasonal Adjustment of Economic Time Series and Multiple Regression Analysis."
  *Journal of the American Statistical Association* 58(304):993–1010. **[foundational]**

## Transportation & spatial equity

- **Hörcher, D., & Graham, D. J. (2021).** "The Gini index of demand imbalances in public transport."
  *Transportation* 48:2521–2544. **[recent]**
- **Karner, A., Pereira, R. H. M., & Farber, S. (2024).** "Advances and pitfalls in measuring transportation
  equity." *Transportation* 52:1399–1427. DOI 10.1007/s11116-023-10460-7 (online 2024; print vol. 52, 2025).
  **[recent]** — Gini/Theil/Atkinson side by side; the dispersion-not-level caveat.
- **Atkinson, A. B. (1970).** "On the Measurement of Inequality." *Journal of Economic Theory* 2(3):244–263.
  **[foundational]** — the Atkinson index (tunable inequality aversion).
- **Theil, H. (1967).** *Economics and Information Theory.* North-Holland, Amsterdam. **[foundational]** — the
  Theil index (between/within-group decomposable).
- **De Maio, F. G. (2007).** "Income inequality measures." *Journal of Epidemiology & Community Health*
  61(10):849–852. **[foundational]** — when Gini vs. Theil vs. Atkinson is preferred.
- **Zheng, Y., Wang, Q., Zhuang, D., Wang, S., & Zhao, J. (2023).** "Fairness-Enhancing Deep Learning for
  Ride-Hailing Demand Prediction." *IEEE Open Journal of Intelligent Transportation Systems* 4:551–569.
  DOI 10.1109/OJITS.2023.3297517. **[recent]** — SA-Net on Chicago TNC data; the closest applied neighbor.
  *Cite the paper's absolute MPE-gap reduction (0.361 → 0.084); avoid the percentage headline figures that
  circulate in secondary summaries — they do not appear in the source.*

## Imitation learning, trajectory-user linking & fidelity

- **Ho, J., & Ermon, S. (2016).** "Generative Adversarial Imitation Learning." *NeurIPS 2016* (Advances in
  NIPS 29), pp. 4565–4573. arXiv:1606.03476. **[foundational]**
- **Zhang, X., Li, Y., Zhou, X., & Luo, J. (2019).** "Unveiling Taxi Drivers' Strategies via cGAIL:
  Conditional Generative Adversarial Imitation Learning." *IEEE ICDM 2019*, pp. 1480–1485. **[recent]** —
  cGAIL, conference version.
- **Zhang, X., Li, Y., Zhou, X., & Luo, J. (2022).** "cGAIL: Conditional Generative Adversarial Imitation
  Learning — An Application in Taxi Drivers' Strategy Learning." *IEEE Transactions on Big Data*
  8(5):1288–1300. DOI 10.1109/TBDATA.2020.3039810 (early access 2020). **[recent]** — cGAIL, journal
  version; the venue is *IEEE Transactions on Big Data*. The taxi-imitation base FAMAIL descends from.
- **Pan, M., Huang, W., Li, Y., Zhou, X., & Luo, J. (2020).** "xGAIL: Explainable Generative Adversarial
  Imitation Learning for Explainable Human Decision Analysis." *Proc. KDD 2020*, pp. 1334–1343. **[recent]**
- **Ren, H., Pan, M., Li, Y., Zhou, X., & Luo, J. (2020).** "ST-SiameseNet: Spatio-Temporal Siamese Networks
  for Human Mobility Signature Identification." *Proc. KDD 2020*, pp. 1306–1315. **[recent]** — the HuMID
  driver-identity discriminator the fidelity model follows.
- **Gao, Q., Zhou, F., Zhang, K., Trajcevski, G., Luo, X., & Zhang, F. (2017).** "Identifying Human Mobility
  via Trajectory Embeddings" (TULER). *IJCAI 2017*, pp. 1689–1695. **[recent]**
- **Zhou, F., Gao, Q., Trajcevski, G., Zhang, K., Zhong, T., & Zhang, F. (2018).** "Trajectory-User Linking
  via Variational AutoEncoder" (TULVAE). *IJCAI 2018*, pp. 3212–3218. **[recent]**
- **Miao, C., Wang, J., Yu, H., Zhang, W., & Qi, Y. (2020).** "Trajectory-User Linking with Attentive
  Recurrent Network" (DeepTUL). *AAMAS 2020*, pp. 878–886. **[recent]**
- **Feng, J., Yang, Z., Xu, F., Yu, H., Wang, M., & Li, Y. (2020).** "Learning to Simulate Human Mobility."
  *Proc. KDD 2020*, pp. 3426–3433. **[recent]** — JS-divergence over mobility statistics as the realism check.

## Adversarial perturbation, algorithmic recourse & discretization

- **Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015).** "Explaining and Harnessing Adversarial Examples"
  (FGSM). *ICLR 2015*. arXiv:1412.6572. **[foundational]**
- **Kurakin, A., Goodfellow, I., & Bengio, S. (2017).** "Adversarial Examples in the Physical World"
  (iterative / basic-iterative FGSM). *ICLR 2017 Workshop*. arXiv:1607.02533. **[foundational]**
- **Hu, M., Zhang, X., Li, Y., Zhou, X., & Luo, J. (2023).** "ST-iFGSM: Enhancing Robustness of Human Mobility
  Signature Identification Model via Spatial-Temporal Iterative FGSM." *Proc. KDD 2023*, pp. 764–774.
  DOI 10.1145/3580305.3599513. **[recent]** — the spatio-temporal iterative-FGSM the editor adapts.
- **Ustun, B., Spangher, A., & Liu, Y. (2019).** "Actionable Recourse in Linear Classification." *Proc. FAT\*
  2019*, pp. 10–19. **[recent]** — constructive reuse of counterfactual-perturbation tooling.
- **Wachter, S., Mittelstadt, B., & Russell, C. (2018).** "Counterfactual Explanations without Opening the
  Black Box." *Harvard Journal of Law & Technology* 31(2):841–887. **[foundational]** — (SSRN preprint 2017).
- **Karimi, A.-H., von Kügelgen, J., Schölkopf, B., & Valera, I. (2020).** "Algorithmic Recourse under
  Imperfect Causal Knowledge." *NeurIPS 2020*; and **Karimi, A.-H., Schölkopf, B., & Valera, I. (2021).**
  "Algorithmic Recourse: from Counterfactual Explanations to Interventions." *Proc. FAccT 2021*. **[recent]**
- **Jang, E., Gu, S., & Poole, B. (2017).** "Categorical Reparameterization with Gumbel-Softmax." *ICLR 2017*.
  arXiv:1611.01144. **[foundational]**
- **Maddison, C. J., Mnih, A., & Teh, Y. W. (2017).** "The Concrete Distribution: A Continuous Relaxation of
  Discrete Random Variables." *ICLR 2017*. arXiv:1611.00712. **[foundational]**
- **Bengio, Y., Léonard, N., & Courville, A. (2013).** "Estimating or Propagating Gradients Through Stochastic
  Neurons for Conditional Computation" (straight-through estimator). arXiv:1308.3432. **[foundational]**

## Egalitarian ethics & leveling-down

- **Parfit, D. (1997).** "Equality and Priority." *Ratio* 10(3):202–221. **[foundational]** — the leveling-down
  objection. (The 1991 Lindley Lecture version is titled "Equality or Priority?"; cite the exact title used.)
- **Temkin, L. S. (1993).** *Inequality.* Oxford University Press. **[foundational]**
- **Temkin, L. S. (2000).** "Equality, Priority, and the Levelling Down Objection." In *The Ideal of Equality*
  (eds. M. Clayton & A. Williams), pp. 126–161. **[foundational]**
- **Mittelstadt, B., Wachter, S., & Russell, C. (2024).** "The Unfairness of Fair Machine Learning: Levelling
  Down and Strict Egalitarianism by Default." *Michigan Technology Law Review* 30(1). arXiv:2302.02404 (2023).
  **[recent]** — "levelling up" via minimum-rate constraints. (Cite year to match the version: journal 2024 /
  preprint 2023.)
- **Zietlow, D., Lohaus, M., Balakrishnan, G., Kleindessner, M., Locatello, F., Schölkopf, B., & Russell, C.
  (2022).** "Leveling Down in Computer Vision: Pareto Inefficiencies in Fair Deep Classifiers." *CVPR 2022*,
  pp. 10400–10411. arXiv:2203.04913. **[recent]** — adaptive augmentation was the one strategy that helped
  the disadvantaged group.
- **Pinzón, C., Palamidessi, C., Piantanida, P., & Valencia, F. (2022).** "On the Impossibility of Non-trivial
  Accuracy in Presence of Fairness Constraints." *AAAI 2022*, 36(7):7993–8000. **[recent]** — leveling-down
  can be constraint-forced (use as analogy; FAMAIL's own oracle bound is the load-bearing formal claim).

## Feedback loops & demand endogeneity

- **Ensign, D., Friedler, S. A., Neville, S., Scheidegger, C., & Venkatasubramanian, S. (2018).** "Runaway
  Feedback Loops in Predictive Policing." *Proc. FAT\* 2018* (PMLR 81), pp. 160–171. **[recent]** —
  *verified 2026-07-08.* Recorded discovered-incident data understates true rates where enforcement was
  historically concentrated → self-reinforcing loop.
- **Lum, K., & Isaac, W. (2016).** "To Predict and Serve?" *Significance* 13(5):14–19.
  DOI 10.1111/j.1740-9713.2016.00960.x. **[foundational]** — *verified 2026-07-08.* Biased historical records
  as a suppressed/censored signal of true demand.
