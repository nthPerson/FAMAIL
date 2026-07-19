# Draft reply to Dr. Zhang's abstract feedback (Robert to edit/send)

Subject: Re: Abstract feedback — all changes applied; two questions

Hi Dr. Zhang,

Thank you for the detailed feedback — every item is now applied to the draft. A summary of
what changed, point by point:

1. **Title.** Adopted as you wrote it: "Mitigating Demonstration Bias via Fairness-Aware
   Trajectory Editing." I will initiate the submission with this title.

2. **Abstract.** Restructured to the sequence you described — problem, then motivation and
   gap, then the proposed solution, then experimental results — and it now closes with a
   summary statement of effectiveness. The term "demand model" has been removed throughout
   the paper (the opening now speaks of imitation-learned mobility policies).

3. **Introduction structure.** Reordered to your six-step sequence; the visualization now
   sits at step 4, illustrating the research gap after the categories of existing approaches
   are presented.

4. **Citations for the intervention categories.** The paragraph you flagged now cites every
   category: in-processing (Zheng et al. 2023), fairness-aware synthetic data generation
   (FairGAN, Xu et al. 2018; DECAF, van Breugel et al. 2021), and reweighing/resampling
   (Kamiran & Calders 2012). The two generation citations are newly added and I am
   completing authoritative-source verification of their metadata before submission.

5. **Terminology.** "Generative repair" and similar nonstandard phrasings are gone; the
   categories now use the terms you suggested (synthetic data generation, data rebalancing,
   bias mitigation through data modification), and the discussion leans on the literature's
   own categorization.

6. **Related work.** Each group of methods now closes with an explicit statement of its
   limitation in this paper's context, followed by the distinction from our approach.

7. **Figure symbols.** Both figures now use a small taxi/car glyph for taxi presence and a
   passenger stick figure for service pickups, with legends updated accordingly — the
   resource-redistribution story reads much more directly, thank you for the suggestion.

8. **Method name.** With the new title, the old FAMAIL acronym (Fairness-Aware Multi-Agent
   Imitation Learning) no longer described the method, so the paper now calls it **FATE
   (Fairness-Aware Trajectory Editing)**, matching the title exactly. Please flag if you'd
   prefer a different name.

Two questions before I finalize:

**(a) Which figure did you mean by "the main visualization"?** Your note suggested reducing
it to about half-page width and retaining only Part (c). Figure 1 (the motivating figure) is
the only one with (a)/(b)/(c) parts, but it is already half-page (single-column) width;
Figure 2 (the method overview) is the full-width figure, but it has no lettered parts. If
you meant Figure 1: should I cut parts (a) and (b) and keep only the two-futures panel? If
Figure 2: I can reduce it to single-column width. (Brighter, more engaging colors are coming
to both either way.)

**(b) The F_causal rename is still open.** From Meeting 41: the optimized fairness term is
associational, not causal, and we agreed the causality *claim* should be dropped — the paper
already frames it that way — but the symbol is still $F_{\mathrm{causal}}$. Before the
final pass I'd like to settle the rename (F_demo was the earlier candidate, matching its
demographic-dependence meaning). Do you have a preference?

One confirmation: I've recorded the author list as You (Robert), Manuel, Charles, Dr. Xin
Wang, Dr. Yanhua Li (WPI), Dr. Kash, and you.

Many thanks,
Robert

---
*Controller notes (not part of the email): FATE name collision worth Robert's awareness
before sending — WeBank's "FATE" federated-learning framework and the FAccT/"FATE"
(Fairness, Accountability, Transparency, Ethics) community label both exist; method-name
collisions are common and rarely reviewer-relevant, and the FAccT resonance arguably fits a
fairness paper, but flag-don't-surprise. All changes committed through `d5c5d67`; the two
new refs are P0-flagged UNVERIFIED in CITATION_PRIORITY_CHECKLIST.md.*
