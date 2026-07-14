# Mission 2 — Citation Verification Audit

**Companion to** `supporting_literature_and_why+how_FAMAIL_objective_function.md` (the deep-research report)
and `mission_2_context.md` (the brief). Produced 2026-07-08 by a 5-agent read-only web-verification pass
(arXiv / publisher DOI pages / ACM DL / IEEE Xplore / DBLP / Crossref — evidence-backed, not from model
memory).

> The header previously carried a "do NOT commit onto the `supply-lift-editing` branch" hold. That branch
> merged and was deleted on 2026-07-09; the hold is retired and these sources were tracked here on
> 2026-07-14.

## Verdict at a glance
~30 citations checked. **The report is highly reliable — every paper it cites is real.** The only failures are
the classic LLM-research failure mode: **confabulated *precise* content** (a fabricated quantitative quote, one
misquote) plus routine metadata drift. **Both items the report itself flagged "verify manually" are now
resolved.** Nothing here undermines the substantive framings we decided to take (conditional statistical parity +
FWL; recourse; Zietlow/Mittelstadt leveling-down) — those are all confirmed.

| Bucket | Count | Items |
|---|---|---|
| 🔴 Fabricated content — MUST fix before use | 2 | Zheng "67%"/"2.3%"; Corbett-Davies quote (a) |
| 🟡 Metadata correction | 5 | cGAIL venue; Zietlow pages; Mittelstadt year; Wachter year; Wilms&Heitz framing |
| 🟢 Self-flagged item now RESOLVED | 2 | ST-iFGSM (author list recovered); cGAIL (venue corrected) |
| ✅ Confirmed as-is | ~21 | see list below |

---

## 🔴 MUST FIX — fabricated / misquoted content

### 1. Zheng et al. 2023 — the "67%" and "2.3%" figures are FABRICATED
- **Bibliography is fully correct:** "Fairness-Enhancing Deep Learning for Ride-Hailing Demand Prediction,"
  Yunhan Zheng, Qingyi Wang, Dingyi Zhuang, Shenhao Wang, Jinhua Zhao. *IEEE Open Journal of Intelligent
  Transportation Systems*, vol. 4, pp. 551–569, 2023. DOI 10.1109/OJITS.2023.3297517. Model "SA-Net"
  (socially aware neural network) ✓, Chicago TNC data ✓. (arXiv preprint: 2303.05698.)
- **The two quantitative claims do NOT appear in the paper.** No "67%" and no "2.3%" anywhere in abstract or
  full text (explicit substring search of the arXiv full text). The paper reports **absolute** results:
  - MPE gap black vs non-black **drops from 0.361 to 0.084** (de-biasing SA-Net).
  - MAE reductions of **0.12 for black communities and 0.05 for non-black communities**.
- ⚠️ The verifier found that **web search AI-overviews now assert "67%"/"2.3%" as if quoting the paper** — a
  hallucination that has leaked into the search layer. Do not trust a casual re-check; the primary text is
  authoritative.
- **Action:** DELETE both figures from Key-Finding #7 and the §Downstream drafted paragraph. If a quantitative
  hook is wanted, cite the paper's actual numbers (gap 0.361→0.084). A percent-reduction (≈77% by
  (0.361−0.084)/0.361) is *our* derivation, not the paper's claim — label it as such, never in quotation marks.

### 2. Corbett-Davies et al. 2017 — quote (a) is a MISQUOTE; quote (b) is verbatim ✓
- **Metadata correct:** "Algorithmic Decision Making and the Cost of Fairness," KDD 2017, pp. 797–806.
- **Quote (b) — VERBATIM ✓:** "Conditional statistical parity requires that one define the 'legitimate'
  factors ℓ(X), and this choice significantly impacts results." Keep as-is.
- **Quote (a) — NOT verbatim.** The report's string ("...conditional on certain permitted characteristics
  (e.g. previous arrests)") does not appear, and the example is wrong. The paper's **actual** text:
  > "Conditional statistical parity means that controlling for a limited set of 'legitimate' risk factors, an
  > equal proportion of defendants are detained within each race group (Kamiran et al., 2013; Dwork et al.,
  > 2012)."
  with the example "among defendants who have the same number of **prior convictions**, black and white
  defendants are detained at equal rates." (Formal: 𝔼[d(X)|ℓ(X),g(X)] = 𝔼[d(X)|ℓ(X)].)
- **Provenance nuance:** Corbett-Davies *formalize/name* conditional statistical parity but attribute the notion
  to **Kamiran et al. 2013** and **Dwork et al. 2012**. The report's "defined there" slightly overstates
  originality — cite as "formalized by Corbett-Davies et al. 2017 (building on Kamiran et al. 2013; Dwork et al.
  2012)." (This does **not** weaken our use of it; it strengthens the citation chain.)
- **Action:** Replace quote (a) with the verbatim text above (and "prior convictions", not "previous arrests");
  soften "defined there" → "formalized there."

---

## 🟡 METADATA CORRECTIONS (real papers, fix the details)

### 3. cGAIL — WRONG JOURNAL (was flagged "verify manually" — now resolved)
Two real, distinct papers, both by **Xin Zhang, Yanhua Li, Xun Zhou, Jun Luo** ("Zhang, Li et al." ✓):
- **Conference:** "Unveiling Taxi Drivers' Strategies via cGAIL: Conditional Generative Adversarial Imitation
  Learning." **IEEE ICDM 2019**, pp. 1480–1485.
- **Journal:** "cGAIL: Conditional Generative Adversarial Imitation Learning—An Application in Taxi Drivers'
  Strategy Learning." **IEEE Transactions on Big Data**, **8(5):1288–1300, 2022** (DOI 10.1109/TBDATA.2020.3039810;
  early-access 2020). **NOT** IEEE TKDE — the report's "IEEE TKDE (≈2020–2021)" is wrong on both venue and the
  citable year. Note: Xin Zhang here = the project PI.

### 4. Zietlow et al. 2022 — page range
- Correct canonical range is **pp. 10400–10411** (Crossref DOI 10.1109/CVPR52688.2022.01016 + DBLP), not
  10410–10421. (The wrong range does circulate — CVF's own generated BibTeX has it — but the publisher record
  disagrees.) Title / authors / CVPR 2022 / arXiv:2203.04913 all correct. Quote **verbatim ✓**.

### 5. Mittelstadt, Wachter & Russell — year (arXiv 2023 vs journal 2024)
- The *Michigan Technology Law Review* 30(1) version of record is **2024**; 2023 is only the arXiv preprint
  (2302.02404). Pick one convention and be consistent (journal → 2024; preprint → 2023). Quote **verbatim ✓**;
  "minimum rate constraints" / "minimum acceptable harm thresholds" **verbatim ✓**. (Journal-of-record spells
  it American "Leveling Down"; preprint uses British "Levelling Down" — match your cited version.)

### 6. Wachter, Mittelstadt & Russell — year
- The Harvard JOLT version is **31(2):841–887, 2018**. The report's "2017" is the SSRN/preprint date and is
  internally inconsistent with citing "JOLT 31(2)" (a 2018 issue). Use 2018 if citing the journal.

### 7. Wilms & Heitz (FAccT 2026) — real paper, but does NOT support the leveling-down framing
- The paper exists: "Fairness vs Performance: Characterizing the Pareto Frontier of Algorithmic Decision
  Systems," Wilms & Heitz, FAccT 2026 (DOI 10.1145/3805689.3812302; arXiv:2605.10604). **But** a full-text
  search found **zero** occurrences of "leveling down"/"levelling down" — it's about the fairness–performance
  Pareto frontier, adjacent but not leveling-down terminology.
- The report already treated it as a non-anchor "forward-looking pointer." **Recommendation: drop it** (2026,
  outside the peer-reviewed window, doesn't say what it'd be cited for). ⚠️ It's also the citation most worth a
  30-second human eyeball, being a 2026 item verified by an automated pass.

---

## 🟢 RESOLVED — the report's own "verify before submission" items

### 8. ST-iFGSM (KDD 2023) — CONFIRMED + author list recovered
- "ST-iFGSM: Enhancing Robustness of Human Mobility Signature Identification Model via Spatial-Temporal
  Iterative FGSM." Authors **Mingzhi Hu, Xin Zhang, Yanhua Li, Xun Zhou, Jun Luo**. **KDD 2023 main research
  track**, pp. 764–774. DOI 10.1145/3580305.3599513. (Xin Zhang = project PI ✓ "the group's own paper.")
- (cGAIL, the other self-flagged item, corrected in #3 above.)

---

## ✅ CONFIRMED AS-IS (spot-checked metadata; safe to cite)

| Citation | Confirmed detail |
|---|---|
| Frisch & Waugh 1933 | *Econometrica* 1(4):387–401 |
| Lovell 1963 | *JASA* 58(304):993–1010 |
| Yule 1907 | Proc. Royal Soc. A — "anticipated FWL" framing reasonable |
| Corbett-Davies 2017 (metadata + quote b) | KDD 2017, pp. 797–806 |
| Feldman et al. 2015 | KDD 2015, pp. 259–268; BER/predictability↔disparate-impact is genuinely in the paper |
| Verma & Rubin 2018 | "Fairness Definitions Explained," FairWare@ICSE 2018, pp. 1–7 |
| Barocas–Hardt–Narayanan | free web ed. (fairmlbook.org, draft stamped 2018) + MIT Press 2023 — both real |
| Kamiran & Calders 2012 | *Knowl. Inf. Syst.* 33(1):1–33; reweighing/massaging/suppression confirmed |
| Pinzón et al. 2022 | AAAI 36(7):7993–8000 |
| Parfit — "Equality and Priority" | *Ratio* 10(3):202–221, 1997 |
| Parfit — "Equality or Priority?" | 1991 Lindley Lecture (title split is correct) |
| Temkin — *Inequality* | OUP 1993 |
| Temkin — chapter | "Equality, Priority, and the Levelling Down Objection," *The Ideal of Equality* 2000, pp. 126–161 |
| Karner, Pereira & Farber | *Transportation* 52:1399–1427; online 2024, print 2025; DOI 10.1007/s11116-023-10460-7 ✓ |
| Hörcher & Graham 2021 | *Transportation* 48:2521–2544 |
| Atkinson 1970 | *J. Econ. Theory* 2(3):244–263 |
| Theil 1967 | *Economics and Information Theory*, North-Holland |
| De Maio 2007 | *J. Epidemiol. Community Health* 61(10):849–852; Gini/Theil/Atkinson comparison confirmed |
| Goodfellow, Shlens & Szegedy 2015 | ICLR 2015, arXiv:1412.6572 |
| Kurakin, Goodfellow & Bengio 2017 | ICLR 2017 Workshop, arXiv:1607.02533 |
| Jang, Gu & Poole 2017 | ICLR 2017, arXiv:1611.01144 |
| Maddison, Mnih & Teh 2017 | ICLR 2017, arXiv:1611.00712 |
| Bengio, Léonard & Courville 2013 | arXiv:1308.3432 (straight-through) |
| Ustun, Spangher & Liu 2019 | "Actionable Recourse in Linear Classification," FAT* 2019 |
| Karimi et al. 2020/2021 | causal-recourse (NeurIPS 2020) + "from counterfactuals to interventions" (FAccT 2021) |
| ST-SiameseNet (Ren et al. 2020) | KDD 2020, pp. 1306–1315; HuMID term confirmed |
| TULER (Gao et al. 2017) | IJCAI 2017, pp. 1689–1695 |
| TULVAE (Zhou et al. 2018) | IJCAI 2018, pp. 3212–3218 |
| DeepTUL (Miao et al. 2020) | AAMAS 2020, pp. 878–886 |
| Ho & Ermon 2016 (GAIL) | NeurIPS 2016, pp. 4565–4573, arXiv:1606.03476 |
| xGAIL (Pan et al. 2020) | KDD 2020, pp. 1334–1343 |
| Feng et al. 2020 | "Learning to Simulate Human Mobility," KDD 2020, pp. 3426–3433; JSD-over-mobility-stats eval ✓ |

---

## Corrected reference snippets (paste-ready)
- **cGAIL (conf):** X. Zhang, Y. Li, X. Zhou, J. Luo. "Unveiling Taxi Drivers' Strategies via cGAIL: Conditional
  Generative Adversarial Imitation Learning." *IEEE ICDM*, 2019, pp. 1480–1485.
- **cGAIL (journal):** X. Zhang, Y. Li, X. Zhou, J. Luo. "cGAIL: Conditional Generative Adversarial Imitation
  Learning—An Application in Taxi Drivers' Strategy Learning." *IEEE Transactions on Big Data*, 8(5):1288–1300,
  2022.
- **ST-iFGSM:** M. Hu, X. Zhang, Y. Li, X. Zhou, J. Luo. "ST-iFGSM: Enhancing Robustness of Human Mobility
  Signature Identification Model via Spatial-Temporal Iterative FGSM." *KDD*, 2023, pp. 764–774.
- **Zheng et al. 2023:** Y. Zheng, Q. Wang, D. Zhuang, S. Wang, J. Zhao. "Fairness-Enhancing Deep Learning for
  Ride-Hailing Demand Prediction." *IEEE Open J. Intell. Transp. Syst.*, 4:551–569, 2023. — *cite the absolute
  gap 0.361→0.084; do NOT quote "67%"/"2.3%."*
- **Zietlow et al. 2022:** *CVPR*, 2022, pp. 10400–10411. arXiv:2203.04913.
- **Mittelstadt, Wachter & Russell:** *Michigan Technology Law Review*, 30(1), 2024 (arXiv:2302.02404, 2023).
- **Wachter, Mittelstadt & Russell:** *Harvard J. Law & Technology*, 31(2):841–887, 2018.
