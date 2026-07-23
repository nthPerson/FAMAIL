# §3–§5 citation audit — 2026-07-16

**Scope:** every `\cite` in `sections/03_methodology.tex`, `04_experiments.tex`,
`05_conclusion.tex`, plus the corresponding `refs.bib` entries and the render-QA R21
bib-wide hygiene items (author-name style normalization; the "X. Zhang"/"Xin Zhang"
unification). **§4 and §5 carry zero citations** (grep-confirmed for `\cite|\citep|\citet`;
consistent with the 2026-07-12 audit's "05 carries zero citations by design"), so every
row below is in `03_methodology.tex`. The §1/§2 claims audited on 2026-07-12
(`reviews/2026-07-12-prose-citation-audit.md`) were NOT redone; §3 usages of those keys
were checked for *claim drift* against the audited claim, and re-verified from the source
only where the §3 claim differs.

**Protocol:** 29 key-instances in §3 (26 distinct keys). Three keys never audited
(`frischwaugh1933`, `lovell1963`, `hoaglinwelsch1978`) got full existence + metadata +
claim verification. Three audited keys carry a *different* claim in §3 (`feldman2015`
"fairness-as-predictability", `feng2020simulate` JSD check, `zhang2022cgail` preprocessing
pipeline) and got fresh body-level claim verification. The other 17 reuse the audited
claim (textual comparison, recorded per-row). Separately, ALL 41 `refs.bib` entries were
re-verified for author full names + metadata against publisher records (10 parallel web
agents; ACM DL first for ACM venues, then IEEE/Springer/PMLR/Wiley/IJCAI/IFAAMAS/CVF/
AAAI/NeurIPS/MTLR/OUP; arXiv only for arXiv-only works; Google Scholar and aggregators
never treated as terminal — where a publisher front-end bot-blocked fetches, the
*publisher-deposited* Crossref/DOI record was used and is flagged below).

**Headline: 0 fabrications, 0 not-found, 0 outright claim mismatches.
1 claim-precision finding (feldman2015 direction inversion), 2 genuine metadata errors
(`zietlow2022` page range; `mittelstadt2024` title spelling vs the cited edition), a
ready-to-apply R21 full-name normalization for all 41 entries, and an explicit
could-not-reach-publisher-page list for Robert's manual pass.**

This is a machine cross-check feeding Robert's manual verification (Meeting-43, Dr. Kash),
not a substitute for it. READ-ONLY session: no repo file was edited; all fixes below are
*proposals* for the controller session.

---

## 1. Ranked findings

### F1 — feldman2015 (§3.2, `03_methodology.tex:92`): prediction direction inverted — MEDIUM-HIGH (claim precision)

The sentence: *"measuring fairness by how well a protected attribute predicts an outcome
follows the fairness-as-predictability logic of \cite{feldman2015}."*

Feldman et al.'s formal result (Theorem 4.1, read in full text, arXiv:1412.3756 =
KDD 2015) runs the **other way**: a dataset admits disparate impact iff the **protected
attribute is predictable from the remaining attributes** (balanced error rate of
predicting X from Y) — *"If Bob cannot predict X given the other attributes of D, then A
is fair with respect to Bob on D."* FAMAIL's F_causal uses the protected attribute as
the *predictor* of a service-residual *outcome* (partial R²). Statistical dependence is
symmetric, so the citation is defensible in spirit — but a reviewer who knows the paper
will notice the inversion, and this cite sits in the paper's core metric derivation.

**Proposed prose fix (controller's call on wording):** soften the attribution to the
shared logic rather than the specific direction, e.g.
*"…follows the predictability-based logic of \cite{feldman2015}, which certifies
disparate impact through the statistical dependence between protected attributes and the
data (there, the protected attribute predicted from the data; here, the demographics
predicting the residual)."* — or any one-clause equivalent acknowledging the direction.
The second feldman2015 usage (§3.6, line 425, "disparate-impact repair") is exact:
their rank-preserving repair to a median distribution is canonical pre-processing.
CONFIRMED, no change.

### F2 — zietlow2022: page range wrong in refs.bib — METADATA-FIX

CVF Open Access (publisher of record; fetched directly, both `citation_*` meta tags and
CVF's own BibTeX block) gives **pp. 10410–10421**; `refs.bib` has 10400–10411. The
2026-07-12 audit verified this entry against arXiv and missed it — exactly Dr. Kash's
point about aggregator-vs-publisher. Fix in diff block §3 below.

### F3 — mittelstadt2024: title spelling doesn't match the cited edition — METADATA-FIX

The entry cites the journal version (Michigan Technology Law Review 30(1), 2024) and its
own comment says "title must match the cited edition" logic elsewhere in the file — but
the title uses the *preprint's* British spelling "Levelling Down". The published MTLR
version (repository.law.umich.edu/mtlr/vol30/iss1/3/, fetched directly) renders
**"The Unfairness of Fair Machine Learning: Leveling Down and Strict Egalitarianism by
Default"** (US spelling), DOI 10.36645/mtlr.30.1.unfairness, online 2024-11-12. Also: the
publisher's recommended citation lists **no page numbers** (the bepress "firstpage=3" is
an article-sequence slot, not law-review pagination — do not add it). The 2026-07-12
audit noted "title spelling varies by venue; BibTeX acceptable" — under the stricter
match-the-cited-edition standard, it should be "Leveling". Fix in diff block §3.
(Prose caution: §3.4 and §2 spell the *concept* "leveling down" already, so the fix
*removes* an inconsistency. Note the concept prose itself is being rewritten per
Meeting-43 — see Context flags.)

### F4 — venues with NO publisher record — for Robert's manual pass

`goodfellow2015fgsm`: the arXiv abs page (designated authority for this entry) has **no
Comments field and never states ICLR 2015**. ICLR 2015 had no formal proceedings; the
venue attribution is community-standard (and universal in the literature) but there is
literally no publisher record that says it. Options: keep as-is (standard practice), or
cite as arXiv:1412.6572. Same class of issue, milder: `kurakin2017ifgsm` ICLR-2017-
Workshop status rests on the OpenReview forum (bot-blocked; confirmed via snippet only),
and `jang2017gumbel`/`maddison2017concrete` ICLR 2017 status rests on arXiv + DBLP
corroboration because OpenReview is Cloudflare-blocked from this environment. All four
are correct by community convention; none has a fetched publisher page asserting the venue.

### F5 — hygiene: 5 uncited refs.bib entries

`karimi2020recourse`, `karimi2021recourse`, `temkin1993`, `temkin2000`, `pinzon2022` are
cited nowhere in the paper. Harmless (BibTeX emits cited-only; no `\nocite`), but two of
them carry metadata gaps fixed in diff §3 in case they get cited (pinzon2022 is flagged
"analogy only" for the leveling-down rewrite and may well come back). Keep-or-cut is the
controller's call.

---

## 2. Per-citation table (§3, in order of first appearance)

Verdicts: **OK** = existence + metadata + claim all check. "Audit §1/§2" in the
existence column = verified 2026-07-12 (URL in that report); only *new* verification
URLs are repeated here. "⚑LD" = attached prose will change (leveling-down rewrite,
Meeting-43); verify-normally-but-expect-rewording.

| key | where used | claim attached | existence source | metadata | claim | proposed fix |
|---|---|---|---|---|---|---|
| corbettdavies2017 | 03:54 | conditional statistical parity restricts disparities to those not explained by permitted factors | audit §1/§2; ACM deposit 10.1145/3097983.3098095 re-confirmed | OK | SAME as audited — OK | none |
| hoaglinwelsch1978 | 03:63 | hat matrix = projection onto column space of design matrix | NEW: tandfonline.com/doi/abs/10.1080/00031305.1978.10479237 (publisher-deposited record; DOI verified resolving) | OK (exact) | SUPPORTED — abstract: "A projection matrix known as the hat matrix" | optional: add doi field |
| frischwaugh1933 | 03:87 | FWL theorem: residualize-then-project = full-regression coefficients | NEW: econometricsociety.org article page (direct HTTP 200) + jstor.org/stable/1907330 | OK (exact) | SUPPORTED — FW 1933 pp. 394–396 prove b=b′ for the trend case; joint cite with Lovell is exactly right (Basu, *Analytical History of the YFWL Theorem*, read in full) | optional: add doi 10.2307/1907330 |
| lovell1963 | 03:87 | FWL theorem (general case) | NEW: tandfonline.com/doi/abs/10.1080/01621459.1963.10480682 + jstor.org/stable/2283327 | OK (exact) | SUPPORTED — Lovell partitions regressors "without any restrictions", the generalization the prose needs | optional: add doi |
| feldman2015 (1st) | 03:92 | "fairness-as-predictability logic" | audit §1/§2; full text re-read at ar5iv 1412.3756 | OK | **PARTIAL — direction inverted (finding F1)** | one-clause prose rewording (F1) |
| horchergraham2021 | 03:119 | Gini widely used in transportation equity | audit §1/§2; Springer page re-fetched directly | OK (issue 5 exists; omitted by file convention) | SAME — OK | none |
| karner2024 | 03:119 | same | audit §1/§2; Springer page re-fetched directly | OK (2025/52(4)/1399–1427 re-confirmed correct) | SAME — OK | none; FYI a published Correction exists (2024-03-06, DOI 10.1007/s11116-024-10474-9) — Robert may want to check it doesn't touch the cited content |
| gao2017tuler | 03:142 | mobility signatures highly identifying — premise of TUL literature | audit §1/§2; ijcai.org/proceedings/2017/234 re-fetched directly | OK | SAME (mild extension: "highly identifying" = the literature's premise; defensible) | none |
| zhou2018tulvae | 03:142 | same | audit §1/§2; ijcai.org/proceedings/2018/446 direct | OK | SAME — OK | none |
| miao2020deeptul | 03:142 | same | audit §1/§2; ifaamas.org camera-ready PDF p878 (publisher of record) | OK (pp 878–886 confirmed on the PDF itself) | SAME — OK | none |
| ren2020stsiamese | 03:144 | frozen driver-identity discriminator, ST-SiameseNet family | audit §1/§2; ACM deposit 10.1145/3394486.3403183 | OK | SAME — OK | none |
| hoermon2016 | 03:147 | "instability of a live adversarial game" (GAIL) | audit §1/§2; proceedings.neurips.cc direct | OK (pages 4565–4573 are printed-proceedings convention; not on the web page) | SAME construct; LOW nuance — the cite correctly anchors the adversarial game; "instability" is the field's characterization, not a claim Ho & Ermon make. Defensible; no change proposed | none |
| feng2020simulate | 03:153 | JSD over aggregate mobility statistics as the collapse/realism check | camera-ready PDF read (vonfeng.github.io, KDD 2020); ACM deposit 10.1145/3394486.3412862 | OK | **SUPPORTED (body-level)** — §4.2: six mobility-pattern metrics (Distance, Radius/r_g, Duration, DailyLoc, G-rank, I-rank), all scored by JSD; Table 1 headed "Metrics(JSD)" | none |
| parfit1997 ⚑LD | 03:268 | classic leveling-down objection to equalization | audit §1/§2; Wiley deposit 10.1111/1467-9329.00041 re-confirmed | OK | SAME — OK | none (prose will be rewritten as analogy-only) |
| mittelstadt2024 ⚑LD | 03:270 | formalized leveling-down for ML fairness + prescribe level-up | audit §1/§2; repository.law.umich.edu/mtlr/vol30/iss1/3/ direct | **METADATA-FIX (F3: title "Leveling", not "Levelling")** | SAME — OK | diff §3 |
| zietlow2022 ⚑LD | 03:272 | data augmentation the one intervention helping the disadvantaged group | audit §1/§2; openaccess.thecvf.com direct (curl) | **METADATA-FIX (F2: pages 10410–10421)** | SAME — OK ("the one intervention" tracks the paper's "uniquely, of all methods tested") | diff §3 |
| ensign2018 | 03:294 | feedback-loop pathology; suppressed/censored demand in under-served areas | audit §1/§2 (disposition #5); PMLR page re-fetched directly | OK (venue string matches PMLR exactly) | SAME as the reworded mechanism; the censored-signal reading matches the 2026-07-08-verified bib comments | none |
| lumisaac2016 | 03:294 | same | audit §1/§2; OUP page (Significance moved Wiley→OUP; legacy DOI still resolves) | OK | SAME — OK | none |
| goodfellow2015fgsm | 03:334 | bounded adversarial perturbation (FGSM origin) | audit §1/§2; arXiv abs direct | OK, but see **F4** (ICLR 2015 has no publisher record) | SAME — OK | manual-pass decision (F4) |
| kurakin2017ifgsm | 03:334 | iterative FGSM | audit §1/§2; arXiv abs direct | OK (workshop status via OpenReview snippet — F4) | SAME — OK | none |
| hu2023stifgsm | 03:335 | spatio-temporal instantiation (ST-iFGSM) | audit §1/§2; ACM deposit 10.1145/3580305.3599513 verbatim | OK | SAME — OK | none |
| ustun2019recourse | 03:337 | algorithmic recourse: minimal beneficial change | audit §1/§2; ACM deposit 10.1145/3287560.3287566 | OK | SAME — OK | none |
| wachter2018counterfactual | 03:337 | counterfactual explanations as minimal input change | audit §1/§2; journal's own PDF read page-by-page (vol 31(2), starts p. 841, 47 pp → 841–887 confirmed) | OK | SAME — OK | none |
| jang2017gumbel | 03:353 | temperature-annealed soft cell assignment (umbrella) | audit §1/§2; arXiv direct | OK (F4 note) | SAME — pre-cleared by audit disposition #4 ("§3 umbrella defensible as written") | none |
| maddison2017concrete | 03:353 | same | audit §1/§2; arXiv direct | OK | SAME — OK | none |
| bengio2013ste | 03:353 | same (STE inside the umbrella) | audit §1/§2; arXiv abs direct (arXiv-only: authoritative) | OK ("Léonard" diacritic confirmed) | SAME — pre-cleared (disposition #4) | none |
| zhang2022cgail | 03:383 | source data preprocessed following cGAIL's pipeline → king-move rule | NSF PAR accepted manuscript read (par.nsf.gov/servlets/purl/10225184) + IEEE deposit | OK (cosmetic: IEEE renders title em-dash unspaced) | **SUPPORTED (body-level)** — cGAIL §4.2: action set = 8 neighboring cells + stay, deterministic transitions ⇒ max(\|dx\|,\|dy\|)≤1 by construction; 0.01° grid matches. Nuance: cGAIL states this as the MDP action-space *definition*, not an explicit filtering rule — FAMAIL's filter is its own implementation detail, consistent with the cited pipeline. No text change needed | optional cosmetic (diff §3) |
| kamirancalders2012 | 03:424 | instance reweighing, pre-processing family | audit §1/§2; Springer page re-fetched directly | OK | SAME — OK | none |
| feldman2015 (2nd) | 03:425 | disparate-impact repair | (as above) | OK | **CONFIRMED** (rank-preserving median-distribution repair) | none |
| zheng2023 | 03:432 | in-processing regularization at the model level | audit §1/§2; IEEE deposit 10.1109/OJITS.2023.3297517 | OK | SAME — OK | none |

---

## 3. Ready-to-apply refs.bib metadata diff (controller applies)

```
--- refs.bib (metadata fixes)

@ zietlow2022:                                            [F2 — real error]
-  pages     = {10400--10411}
+  pages     = {10410--10421}
   (CVF Open Access meta tags + CVF's own BibTeX agree; also update the trailing
    entry comment if it repeats the old range)

@ mittelstadt2024:                                        [F3 — match cited edition]
-  title   = {The Unfairness of Fair Machine Learning: Levelling Down and Strict Egalitarianism by Default}
+  title   = {The Unfairness of Fair Machine Learning: Leveling Down and Strict Egalitarianism by Default}
   optionally add:
+  doi     = {10.36645/mtlr.30.1.unfairness}
   (do NOT add pages — the publisher's recommended citation lists none; the bepress
    "firstpage=3" is an article slot, not pagination)

@ karimi2020recourse (currently uncited):                 [publisher title + pages]
-  title     = {Algorithmic Recourse under Imperfect Causal Knowledge}
+  title     = {Algorithmic Recourse under Imperfect Causal Knowledge: A Probabilistic Approach}
+  pages     = {265--277}
   (proceedings.neurips.cc official BibTeX; publisher renders sentence-case —
    title-case above matches the file's house style)

@ karimi2021recourse (currently uncited):                 [pages + venue]
+  pages     = {353--362}
-  booktitle = {Proceedings of the ACM Conference on Fairness, Accountability, and Transparency (FAccT)}
+  booktitle = {Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (FAccT)}

@ temkin2000 (currently uncited):                         [missing publisher]
+  publisher = {Palgrave Macmillan}
   (year note: the Palgrave softcover record is copyright 2002 / published 26 May 2000;
    the original 2000 hardcover was Macmillan Press — year 2000 is defensible as-is.
    Chapter spelling confirmed "Levelling" — British, correct for THIS chapter title.)

Optional DOI additions (all verified resolving): frischwaugh1933 {10.2307/1907330},
lovell1963 {10.1080/01621459.1963.10480682}, hoaglinwelsch1978
{10.1080/00031305.1978.10479237}, pinzon2022 {10.1609/aaai.v36i7.20770}.

Cosmetic, decide-don't-default (flag for Robert):
- feldman2015 booktitle: ACM's OFFICIAL rendering is "…21th ACM SIGKDD…" [sic — ACM's
  own typo, kept in their record]. File has grammatical "21st". Either is defensible;
  matching the publisher exactly means adopting their typo.
- vermarubin2018 booktitle: ACM renders "Proceedings of the International Workshop on
  Software Fairness" (acronym FairWare '18); the file's "(FairWare@ICSE)" suffix is
  DBLP-style, not ACM's.
- zhang2022cgail title: IEEE renders the em-dash unspaced ("Learning—An Application");
  file has spaced " --- ". LaTeX: "Learning---An Application".
- hoermon2016: NIPS pages 4565–4573 come from the printed proceedings (the web page has
  no pagination) — standard practice, keep.
```

## 4. R21 author-name normalization diff (full first names, publisher-verified)

Every author string below was verified against the publisher record (or
publisher-deposited DOI metadata — see §5). Entries already in full form
(`corbettdavies2017`, `zhang2022cgail`, `lumisaac2016`) are unchanged. This unifies the
R21 "X. Zhang" [32] / "Xin Zhang" [33] collision: **both are Xin Zhang** (IEEE-deposited
records for ICDM 2019 and TBD 2022).

```
@ feldman2015:        author = {Feldman, Michael and Friedler, Sorelle A. and Moeller, John and Scheidegger, Carlos and Venkatasubramanian, Suresh}
@ kamirancalders2012: author = {Kamiran, Faisal and Calders, Toon}
@ vermarubin2018:     author = {Verma, Sahil and Rubin, Julia}
@ barocas2023:        author = {Barocas, Solon and Hardt, Moritz and Narayanan, Arvind}
@ frischwaugh1933:    author = {Frisch, Ragnar and Waugh, Frederick V.}
@ lovell1963:         author = {Lovell, Michael C.}
@ hoaglinwelsch1978:  author = {Hoaglin, David C. and Welsch, Roy E.}
@ horchergraham2021:  author = {H{\"o}rcher, Daniel and Graham, Daniel J.}
@ karner2024:         author = {Karner, Alex and Pereira, Rafael H. M. and Farber, Steven}
@ atkinson1970:       author = {Atkinson, Anthony B.}
@ theil1967:          author = {Theil, Henri}
@ demaio2007:         author = {De Maio, Fernando G.}
@ zheng2023:          author = {Zheng, Yunhan and Wang, Qingyi and Zhuang, Dingyi and Wang, Shenhao and Zhao, Jinhua}
@ hoermon2016:        author = {Ho, Jonathan and Ermon, Stefano}
@ zhang2019cgail:     author = {Zhang, Xin and Li, Yanhua and Zhou, Xun and Luo, Jun}
@ pan2020xgail:       author = {Pan, Menghai and Huang, Weixiao and Li, Yanhua and Zhou, Xun and Luo, Jun}
@ ren2020stsiamese:   author = {Ren, Huimin and Pan, Menghai and Li, Yanhua and Zhou, Xun and Luo, Jun}
@ gao2017tuler:       author = {Gao, Qiang and Zhou, Fan and Zhang, Kunpeng and Trajcevski, Goce and Luo, Xucheng and Zhang, Fengli}
@ zhou2018tulvae:     author = {Zhou, Fan and Gao, Qiang and Trajcevski, Goce and Zhang, Kunpeng and Zhong, Ting and Zhang, Fengli}
@ miao2020deeptul:    author = {Miao, Congcong and Wang, Jilong and Yu, Heng and Zhang, Weichen and Qi, Yinyao}
@ feng2020simulate:   author = {Feng, Jie and Yang, Zeyu and Xu, Fengli and Yu, Haisu and Wang, Mudan and Li, Yong}
@ goodfellow2015fgsm: author = {Goodfellow, Ian J. and Shlens, Jonathon and Szegedy, Christian}
@ kurakin2017ifgsm:   author = {Kurakin, Alexey and Goodfellow, Ian and Bengio, Samy}
@ hu2023stifgsm:      author = {Hu, Mingzhi and Zhang, Xin and Li, Yanhua and Zhou, Xun and Luo, Jun}
@ ustun2019recourse:  author = {Ustun, Berk and Spangher, Alexander and Liu, Yang}
@ wachter2018counterfactual: author = {Wachter, Sandra and Mittelstadt, Brent and Russell, Chris}
@ karimi2020recourse: author = {Karimi, Amir-Hossein and von K{\"u}gelgen, Julius and Sch{\"o}lkopf, Bernhard and Valera, Isabel}
@ karimi2021recourse: author = {Karimi, Amir-Hossein and Sch{\"o}lkopf, Bernhard and Valera, Isabel}
@ jang2017gumbel:     author = {Jang, Eric and Gu, Shixiang and Poole, Ben}
@ maddison2017concrete: author = {Maddison, Chris J. and Mnih, Andriy and Teh, Yee Whye}
@ bengio2013ste:      author = {Bengio, Yoshua and L{\'e}onard, Nicholas and Courville, Aaron}
@ parfit1997:         author = {Parfit, Derek}
@ temkin1993:         author = {Temkin, Larry S.}
@ temkin2000:         author = {Temkin, Larry S.}
                      editor = {Clayton, Matthew and Williams, Andrew}
                      (chapter byline is "Larry Temkin"; "Larry S. Temkin" matches his book
                       records — either defensible, S. recommended for consistency w/ temkin1993)
@ mittelstadt2024:    author = {Mittelstadt, Brent and Wachter, Sandra and Russell, Chris}
@ zietlow2022:        author = {Zietlow, Dominik and Lohaus, Michael and Balakrishnan, Guha and Kleindessner, Matth{\"a}us and Locatello, Francesco and Sch{\"o}lkopf, Bernhard and Russell, Chris}
@ pinzon2022:         author = {Pinz{\'o}n, Carlos and Palamidessi, Catuscia and Piantanida, Pablo and Valencia, Frank}
@ ensign2018:         author = {Ensign, Danielle and Friedler, Sorelle A. and Neville, Scott and Scheidegger, Carlos and Venkatasubramanian, Suresh}
```

Name-notes: `kurakin2017ifgsm` — arXiv (the designated authority) renders the middle
author plain "Ian Goodfellow", no "J."; keeping it distinct from `goodfellow2015fgsm`'s
"Ian J. Goodfellow" matches the respective records exactly (a human might prefer
consistency — Robert's call). Diacritics verified: Hörcher, Léonard, Kügelgen, Schölkopf,
Matthäus, Pinzón.

URL/DOI line-break cosmetics (R21 third item): not addressed here — low priority per the
render-QA note; purely a BibTeX-style/`\usepackage{url}` matter for the controller.

---

## 5. Entries NOT verified to the authoritative-source standard — Robert's manual-pass priority

Several publisher front-ends bot-block this environment (Cloudflare 403 / challenge
pages: dl.acm.org, ieeexplore.ieee.org, onlinelibrary.wiley.com, tandfonline.com,
jstor.org, sciencedirect.com, jech.bmj.com, openreview.net, mitpress.mit.edu). For those,
verification used the **publisher's own Crossref/DOI metadata deposit** (publisher-authored,
one step removed from the rendered page) plus corroborating snippets of the publisher
page. Fields matched in every case. In priority order for the manual pass:

1. **`goodfellow2015fgsm` — venue attribution has NO publisher record anywhere** (F4).
   Decide: keep community-standard "ICLR 2015" or cite as arXiv.
2. **OpenReview-blocked ICLR entries** — `jang2017gumbel`, `maddison2017concrete`
   (venue via arXiv+DBLP), `kurakin2017ifgsm` (workshop status via snippet). One click
   each on OpenReview settles them.
3. **ACM-DL-blocked** (verified via ACM-deposited DOI metadata): `corbettdavies2017`,
   `feldman2015`, `vermarubin2018`, `pan2020xgail`, `ren2020stsiamese`,
   `feng2020simulate`, `hu2023stifgsm`, `ustun2019recourse`, `karimi2021recourse`.
   Robert should spot-check at least the §3 load-bearing ones (`feldman2015`,
   `hu2023stifgsm`) on the rendered DL pages.
4. **IEEE-blocked** (IEEE-deposited metadata): `zhang2019cgail`, `zhang2022cgail`,
   `zheng2023`.
5. **Other blocked front-ends** (publisher-deposited metadata, all fields consistent):
   `frischwaugh1933` (JSTOR; but the Econometric Society's own page WAS fetched directly),
   `lovell1963`, `hoaglinwelsch1978` (T&F), `atkinson1970` (Elsevier), `demaio2007`
   (BMJ; note their Crossref deposit carries a title artifact "…: Figure 1" — true title
   confirmed via Europe PMC), `parfit1997` (Wiley).
6. **Edition subtleties**: `temkin1993` (verified on a Wayback capture of the OUP product
   page — hardcover 1993 correct); `temkin2000` (chapter pages 126–161 rest on chapter
   PDF scans + PhilPapers; Springer hosts no chapter record); `hoermon2016` (pages are
   printed-proceedings convention).

**Fully verified on the live publisher page, no caveat:** `kamirancalders2012`,
`horchergraham2021`, `karner2024` (Springer), `gao2017tuler`, `zhou2018tulvae` (IJCAI),
`miao2020deeptul` (IFAAMAS camera-ready PDF), `ensign2018` (PMLR), `lumisaac2016` (OUP),
`wachter2018counterfactual` (journal's own PDF), `karimi2020recourse` (NeurIPS
proceedings + official BibTeX), `zietlow2022` (CVF), `pinzon2022` (AAAI OJS),
`mittelstadt2024` (MTLR repository), `bengio2013ste` (arXiv = authoritative for
arXiv-only), `barocas2023` (fairmlbook.org canonical BibTeX), `feng2020simulate`'s
*content* (author camera-ready PDF read for the claim check).

---

## 6. Context flags (annotate-only, per session brief)

- **Leveling-down rewrite in flight (Meeting-43, Dr. Kash):** the `parfit1997` /
  `mittelstadt2024` / `zietlow2022` cluster (03:268–272) is verified as written but its
  attached prose will be recast as analogy-only. The F3 title fix ("Leveling") is
  orthogonal and should be applied regardless. `pinzon2022` (uncited, "analogy only" per
  its own comment) may re-enter during that rewrite — its metadata is now fully verified
  (AAAI OJS direct) with issue 7 + DOI available.
- **F_causal rename (F_demo floated):** the citations nearest the naming caveat
  (03:98–107) are `corbettdavies2017` (03:54) and `feldman2015` (03:92). Neither claim
  depends on the term's name; the F1 rewording, if adopted, should be drafted
  rename-agnostically.
- **Double-blind:** no self-revealing phrasing found (grep for "our prior/previous/
  earlier work", "we previously" — zero hits). "following the pipeline of
  \cite{zhang2022cgail}" (03:383) is third-person and fine. Five of 26 cited keys are
  the same research-group lineage (cGAIL ×2, xGAIL, ST-SiameseNet, ST-iFGSM) — inherent
  to the domain, neutrally phrased throughout; no action.
- **`karner2024` Correction** (Transportation, 2024-03-06, DOI 10.1007/s11116-024-10474-9):
  existence noted; whether it touches the Gini-usage claim was not checked — one-click
  item for the manual pass.

---

*Verification agents: 10 parallel web agents (4 dispatched + coordinator-spawned
sub-batches), 2026-07-16. Working notes: session scratchpad (`s3_usage_map.md`,
`names_results.md`). No repo files modified other than creating this report.*
