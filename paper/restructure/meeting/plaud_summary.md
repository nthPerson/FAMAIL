# Plaud auto-summary — 07-24 Meeting: Paper Positioning and Core Contribution (Budget-Aware Trajectory Editing)

> Machine-generated summary (verbatim from Plaud, light re-formatting). 62-minute
> meeting, 2026-07-24 ~16:46 PT. Participants: Robert ("Speaker 3"), Dr. Xin Zhang.
> "FADE" below is a transcription artifact for FATE. Verify anything load-bearing
> against `transcript_readable.txt`.

## Meeting notes

**Paper positioning and core contribution (budget-aware trajectory editing)**
- The team will reframe the paper around "budget-aware trajectory editing": modify only
  a portion of trajectories to influence downstream model fairness.
- Distinction from fair GAN and model-side fairness: focus on data-level
  collective/global fairness via selective edits.
- [HIGHLIGHTED] The prior novelty on the editing–fidelity trade-off is retained, but
  emphasis shifts to differentiating from prior approaches and highlighting collective
  fairness impact.
- Clear claim: small, targeted corrections may not greatly change raw data fairness but
  can significantly affect downstream models.
- Conclusion: Adopt the "budget-aware trajectory editing" framing and foreground
  collective fairness and differentiation from model-side methods.

**Algorithm overview and visual communication (Figure 1 and Figure 2 redesign)**
- Figure 1: Use the clearer "collective service disparity emerges from aggregation"
  version.
- Figure 2: Prefer a framework/sequence diagram over algorithm-faithful visuals; depict
  inputs, stages, and outputs simply.
- Three-step method: attribution (select trajectories), trim, lift; keep abstract for
  clarity, with detailed implementation in the appendix.
- [HIGHLIGHTED] The "Locate" step must clearly contrast advantaged vs disadvantaged
  districts and show procedure (before/after attribution, inputs, outputs).
- Consider replacing icons (cars/people) with stylized trajectories; districts remain
  conceptually important but can be abstracted.
- Tools: Keynote is acceptable; TikZ is preferred for consistency if time permits;
  prioritize content clarity over style.
- Conclusion: Rework figures into a simple framework sequence diagram and a clearer
  disparity illustration; ensure "Locate" conveys process and contrast.

**Writing structure and methodology flow**
- Introduction: Use accessible language (KDD audience) to avoid losing readers early.
- Overview section: Provide definitions and the problem statement, then challenges and
  solutions; mention challenges briefly in the intro and itemize them in the overview.
- Methodology: Start with a leading paragraph on the FATE approach; reference the
  framework figure; then detail parts (e.g., fairness objective/surrogate) in a logical
  order, each addressing a specific challenge.
- Do not strictly mirror section names from examples; align content to challenges and
  logical blocks.

**[HIGHLIGHTED] Experiments organization (Shenzhen and San Francisco)**
- Start with a leading paragraph on experiment aims and organization.
- Two strategies considered: side-by-side per question, or Shenzhen-focused with San
  Francisco as transferability evidence.
- Baselines: more complete for Shenzhen due to compute; fairness and propagation show
  parity between cities.
- Avoid redundancy: if observations are the same across cities, state them once;
  separately highlight meaningful differences for defensibility.
- Conclusion: Prefer a Shenzhen-focused main narrative with San Francisco as
  transferability, minimizing duplication and explicitly noting differences.

**Overleaf/KDD formatting and compilation details**
- Ensure KDD/ACM template completeness (ACM Reference Format and permissions sections
  present).
- Resolve Overleaf vs local environment differences; finalize compilation before the
  deadline.
- Dr. Zhang will review the final version and perform a last pass before submission.

**Timeline, workload, and future meetings**
- This is the third revision; significant time pressure with a Sunday night deadline.
- Robert will deliver an updated draft (target: by Saturday night) and figures; will
  use AI assistance for visuals.
- Dr. Zhang will do a final pass and submit by Sunday night.
- This Thursday's biweekly meeting is canceled; future meetings will be rescheduled.
- Post-submission: prepare for the KDD rebuttal; Robert intends to support the rebuttal.

## Next arrangements (Plaud checklist)
- [ ] Emphasize "budget-aware trajectory editing" and collective fairness in the
      paper's positioning
- [ ] Replace Figure 1 with the clearer disparity-aggregation version provided by
      Dr. Zhang
- [ ] Redesign Figure 2 as a framework/sequence diagram (inputs → stages → outputs)
- [ ] Clarify the "Locate" attribution step with before/after and trajectory scoring
      visuals
- [ ] Reorganize methodology: lead with the FATE overview and the framework figure,
      then sequential parts
- [ ] Draft experiments with a Shenzhen focus and San Francisco for transferability;
      avoid redundant text
- [ ] Ensure KDD/ACM template completeness (ACM Reference Format, permissions) and
      finalize PDF compilation
- [ ] Send updated draft and figures to Dr. Zhang for a final pass before Sunday night
- [ ] Prepare the appendix with sufficient implementation details
- [ ] Define and standardize the approach name and finalize fairness metrics and
      normalization/scoring in the main text

## Plaud "AI suggestions" (unresolved items flagged by the tool)
1. Define and consistently use the term/name for the approach (FATE) across the paper.
2. Specify exact figure numbering and placements (which is Figure 1 vs Figure 2).
3. Finalize the fairness metric(s) used for attribution and clearly describe
   normalization/scoring in the main text.
4. Decide the minimum set of baselines required in San Francisco to support
   transferability claims without overextending compute.
5. Confirm tooling for figure creation (Keynote vs TikZ) and ensure style consistency
   if mixed tools are used.

## Highlights (the 5 marked moments, from highlights_raw.html)
1. [04:40] Budget-aware trajectory editing — original trajectory-editing-alone story
   hard to defend → reframe as budgeted/budget-aware editing; central claim: modifying
   a small, impactful portion of data can effectively influence downstream model
   fairness; collective fairness across the dataset, not fairer models per instance.
   ACTION: Robert updates intro + overall structure accordingly, consistent with
   implementation and results.
2. [05:44] Zhang: distinguishing from FairGAN-style work matters MORE than the
   editing-vs-fidelity trade-off; differentiator = collective-fairness perspective
   (small edits → global fairness of whole dataset). Robert agreed.
3. [16:17] Current "locate"/attribute panel fails to convey procedure (what is
   attributed and how). Show input (original trajectories) → attribution process
   (scoring trajectories for fairness) → output, not just the end state.
4. [24:39] DECISION: replace the complex figure with a simpler abstract
   framework/sequence diagram in the style of the attached screenshot
   (`fig2_style_screenshot.png` = ST-iFGSM Fig. 3: colored stage bands, input → stages
   → output, labeled arrows). Clarity over mechanical detail.
5. [39:57] Experiments: leading paragraph stating purpose + organization; use Shenzhen
   to answer ALL experimental questions, then show transferability on San Francisco
   (saves space). Reproducibility lives in the appendix for this version.
