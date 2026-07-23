# Plaud auto-summary — Meeting 44 (2026-07-23, "KDD Paper Revisions and Submission Strategy")

> ⚠️ **POST-DEBRIEF CORRECTIONS (Robert, 2026-07-23) — parts of this document are superseded.**
> 1. Dr. Zhang WAS reviewing the CURRENT paper content (Robert transferred it to Overleaf pre-meeting; she views and will edit there). Retract every "stale copy / older version / discount-as-render-artifact" inference in this file — ALL her feedback binds against the current text.
> 2. Template: per Robert, conform to her direction to NOT use the `\keywords{...}` block (the "corrected template adds keywords" reading is superseded); verify against KDD template standards.
> 3. Raw data: releasable IF 100% anonymous (not a flat no); in-paper data references must not leak identifying information.
> 4. Hat-matrix citation: stays in the main body (derivation content still moves to the appendix).
> Authoritative record: MEETING_44_DEBRIEF.md (§2, §3, §6, §7).

> Source: Plaud recording `2db74f59261d6bdc55ed4d8bb5db3a55`, duration ~2h00m.
> Speaker mapping: "Speaker 3" = Robert; "Dr. Xin Zhang" merges Plaud Speakers 1/2/4.
> The referenced marked photo is saved alongside as `plaud_marked_photo.png` (if download succeeded).

> Date: 2026-07-23 10:53:00
> Location: [Insert Location]
> Participants: [Dr. Xin Zhang] [Speaker 3]

## Meeting Notes

### Paper Template and Formatting
- The current KDD template used initially was incorrect (included CCS concepts instead of keywords). Dr. Xin Zhang provided a corrected template with a keywords section; the team will proceed with this version.
- Formatting issues include package conflicts affecting figure insertion and excessive blank space. Speaker 3 (using TikZ) will adjust spacing and increase text size in figures to be slightly smaller than the main text for readability.
- Figure captions are overly detailed; captions should highlight only essential information due to the eight-page limit. The main text must clearly reference and explain figures; bolding references has some formatting challenges.

### Introduction Clarity and Positioning
- Dr. Xin Zhang found the introduction confusing, particularly around "in-processing methods" and "objective and training signal conflict," and felt the reasons existing approaches fail were not clearly explained.
- Speaker 3 stated the introduction was rewritten to explain the problem's "why" and the proposed solution within space constraints, using compact, semantically dense language supported by citations.
- Both agreed the core problem is the fairness versus realism (or fidelity) trade-off: existing approaches may improve fairness but often fail to retain realism; the paper claims to achieve both.
- Plan to reorganize the introduction to emphasize:
  - Clear motivation centered on the fairness–fidelity trade-off.
  - Limitations of existing approaches (with at least two citations per category and more recent works from 2025–2026).
  - A concise one-paragraph summary of the proposed approach.
  - A motivating example (e.g., GAIL inheriting/amplifying unfairness) and, if possible, early experimental evidence in Figure 1 showing existing approaches cannot solve fairness.
- Add a "Challenges" section to articulate why the problem is difficult and why prior methods are insufficient. Consider cutting or relocating content to make space for clearer explanations, while avoiding loss of critical support for later arguments.

### Terminology and Scope
- Debate on terminology: prefer "fidelity" over "realism" for clarity, while distinguishing general domain usage from paper-specific definitions (e.g., fidelity via ST-SiameseNet discriminator and distributional similarity). Finalize consistent terminology in the introduction.
- Clarify "in-processing methods" versus "regularization approaches" as model-level interventions that keep data biased; ensure citations substantiate claims.
- Position the method as targeted data augmentation for behavior cloning (imitation learning) with minimal trajectory editing (~10%) plus upweighting of edited samples to ensure fairness persists through training; note future extensions beyond behavior cloning as follow-up work.

[PLAUD NOTE photo was attached here in the original — see plaud_marked_photo.png]

### Figures: Clarity, Metrics, and Consistency
- Figure 1 is not self-explanatory; reviewers struggled with understanding grid cells and color coding. **[Plaud highlight]** Adopt Figure 2's style: include labels for "Advantaged district" and "Disadvantaged district," duplicate legends, and background map for clarity.
- Revise Figure 1 to be self-contained:
  - Show explicit fairness metrics (e.g., service ratio values) directly within the figure; the "three times service gap" must be visually evident without relying on the caption.
  - Keep totals of passengers and vehicles consistent across panels to reflect relocation rather than removal, or provide clear justification if they differ.
  - Ensure color coding is consistent across figures and that Figure 1 (motivation) and Figure 2 (solution) jointly convey the narrative.
- Define and standardize fairness metrics (e.g., "taxi service per unit demand," disparate impact) across text and figures; avoid ambiguous icons or assumptions. Clarify the exact ratio definition (supply/demand or demand/supply) and display it consistently.

### Methodology and Derivations
- The derivation of the service ratio formula (F_demo) is complex. Keep the core equation (e.g., 1 − R^2_demo), its meaning, and importance in the main text, with detailed derivations moved to the appendix. Retain reference to the seminal "hat matrix" work (ref [13]) for reviewer context.
- Rename "Task" to "Problem Formulation." Emphasize challenges-first framing aligned with KDD style. Streamline structure, reduce duplication between problem formulation and experimental setup (e.g., avoid repeating grid cell resolution), and move implementation-specific details (e.g., 0.01-degree grid cells) to the appendix to stress generalizability.

### Writing Process and AI Use
- Speaker 3's AI-assisted workflow produces a first draft that is then humanized; the appendix is the least humanized. The paper should be more problem-driven, explaining why choices were made rather than documenting steps.
- Guidance: write first, then use GPT for refinement; avoid GPT-first drafts with overly general language. Replace vague phrases (e.g., "rebalancing models," "data generation shifts the distribution") with precise, domain-specific terms, specifying the level of intervention (data, model, hyperparameters) and the exact distribution(s) referenced.

### Paper Length and Submission Logistics
- The manuscript initially exceeds the eight-page limit (~11 pages). Temporarily going over is acceptable, but the final must be ≤8 pages. Prioritize clarity and ease of reading, especially in the introduction, and tighten formatting (bulleted lists, remove extra spaces).
- Submission targeted by Sunday to avoid deadline ambiguity; verify authoritative cutoff time and timezone on the official site versus OpenReview.

### Data Availability and Artifacts
- Source data from Dr. Yan Hua Li; do not release raw data. Share anonymized data and code via an anonymous GitHub link included in the submission. An empty repository is acceptable initially; artifact pledge can proceed with code availability. Document licensing constraints to ensure compliance.

## Next Arrangements
- [ ] Follow the corrected KDD template for all formatting, including captions.
- [ ] Speaker 3 to reduce spacing and enlarge text within figures; update Figure 1 with legends, labels, background map, and visible fairness metrics; maintain consistent totals or justify differences.
- [ ] Ensure captions are concise and the main text provides clear, explicit figure references.
- [ ] Define and standardize fairness metrics and the "three times" service ratio; align captions and visuals.
- [ ] Reorganize the introduction around the fairness–fidelity trade-off: motivation, limitations with sufficient and current citations, concise approach summary, motivating example, and potential early evidence in Figure 1; add a "Challenges" section.
- [ ] Clarify terminology ("in-processing" vs. "regularization," "fidelity" vs. "realism") and finalize consistent usage; distinguish paper-specific definitions from general usage.
- [ ] Position the method as data augmentation for behavior cloning with trajectory editing and upweighting; specify edited trajectory percentage and the upweighting scheme in methodology; note future extensions.
- [ ] Move detailed derivations (F_demo) and implementation specifics (e.g., grid cell size) to the appendix; keep core equations and meanings in the main body; rename "Task" to "Problem Formulation."
- [ ] Streamline structure, remove duplication between sections, and reduce overall manuscript to ≤8 pages while maintaining readability and completeness.
- [ ] Tighten formatting and remove extra spaces; avoid redundant restatements.
- [ ] Add an anonymized GitHub link for code (and placeholder for anonymized data); confirm artifact pledge requirements.
- [ ] Verify the exact submission deadline and timezone; submit by Sunday.

## AI Suggestions
- Finalize a clear strategy and ownership for rewriting the paper to be problem-driven; assign a lead for revising the introduction and captions.
- Resolve whether to expand the introduction by cutting later supporting points; document the trade-off decision.
- Decide and document consistent terminology choices ("in-processing" vs. "regularization," "fidelity" vs. "realism") and distinguish general versus paper-specific meanings.
- Select and standardize fairness metrics across text and figures (e.g., disparate impact vs. alternatives); define the service ratio precisely and display values.
- Determine whether passenger/vehicle totals must remain constant across Figure 1 panels or if actual measured changes will be shown; justify and document.
- Specify the percentage of edited trajectories and the upweighting scheme; include citations and methodology details.
- Confirm inclusion of early experimental results in Figure 1 to demonstrate existing approaches' limitations; choose which results to show.
- Reconcile submission deadline discrepancies (official website vs. OpenReview); document the authoritative cutoff and timezone.
- Document data licensing constraints from Dr. Yan Hua Li to ensure compliance with the artifact pledge and any future sharing.
