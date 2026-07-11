# FAMAIL Meeting 42 — Notion AI Summary Extraction

> **SOURCE:** Notion page "FAMAIL Meeting 42"
> **URL:** https://app.notion.com/p/398eb30651108020a026c9c50dee1e86
> **Page meeting date (Scheduled Time & Date):** 2026-07-09
> **Page created:** 2026-07-09T17:51:04.299Z
> **Extraction date:** 2026-07-10
>
> **CAVEAT — READ FIRST:** This file transcribes the **unverified Notion AI meeting SUMMARY** only.
> It is NOT the raw transcript and NOT ground truth. This project has previously found that Notion
> AI meeting summaries both **fabricate** and **omit** content (e.g., the Meeting-41 summary invented
> a "3-tier metric framework" that was never said). A separate transcript agent establishes ground
> truth. Everything below reflects what the SUMMARY claims, transcribed faithfully without editorializing.
> The transcript itself was deliberately NOT fetched here (a sibling agent handles it).

---

## Metadata

- **Meeting title:** FAMAIL Meeting 42
- **Group/Attendees (as recorded in page properties):** Dr. Xin Zhang
  - (Robert is the presenter/research assistant throughout the summary but is not enumerated in the Group/Attendees property.)
- **Scheduled date:** 2026-07-09 (date only, no time component)
- **Created Date & Time:** 2026-07-09 17:51:04 UTC
- **Page URL:** https://app.notion.com/p/398eb30651108020a026c9c50dee1e86
- **Ancestor path:** Research Dashboard → Meetings → Meetings Database
- **Related Projects:** https://app.notion.com/p/259eb30651108071a41bc632dcd49167
- **Summary internal heading:** "Research Progress Update: Fairness Algorithm & Paper Development" (carries an inline mention-date of `2025-10-12` — see Suspicious/Ambiguous section)

---

## TODO & Action Items (verbatim from the summary's "Action Items" list)

Checkbox state is preserved as the summary recorded it ([x] = checked/done, [ ] = unchecked/open).

- [x] Wait for GPU results on fairness propagation through trained behavioral cloning models (expected same afternoon)
- [x] Robert to implement data augmentation baselines
- [ ] Robert to human-review all AI-assisted literature references before including them in the paper
- [ ] Begin assembling the methodology section of the paper
- [ ] Robert to prepare a draft abstract for advisor review by next week

(5 action items total: 2 marked done, 3 open.)

---

## Decisions / Guidance attributed to the PI (Dr. Xin Zhang) — "Advisor" in the summary

The summary attributes the following to the advisor (Dr. Zhang):

- **Advisor suggested starting on the methodology section** of the paper.
- Advisor guidance to **submit a draft abstract ahead of the abstract deadline** so the advisor has time to review it (paired with the open action item "prepare a draft abstract for advisor review by next week").
- **Advisor noted the abstract can serve as a placeholder and be refined later.**
- **Next weekly meeting scheduled** to review further progress.

(Note: the summary does not attribute any explicit accept/reject decision on the Trim & Lift algorithm or the metric results to the PI — those are presented as Robert's reported progress, not as PI rulings.)

---

## Topic-by-Topic Summary Content (condensed but faithful; all specifics preserved)

### External Fairness Metrics
- Robert implemented **three external fairness metrics not optimized for**: **disparate impact**, **demographic parity gap**, and the **Theil index**.
- **Fairness improved across all three metrics**, demonstrating that improvements are not simply a byproduct of optimizing self-defined metrics.
- **Caveat (recorded in the summary):** the improvement was achieved by **removing service from over-served (advantaged) areas** (higher income, higher property values, lower migrant ratio), which **conflicts with the paper's goal of maintaining service while improving fairness**.

### New Algorithm: Trim & Lift
- **Root cause identified:** in the previous algorithm, **only demand was differentiable** (Y = supply / demand), so the algorithm could only manipulate pickup locations, resulting in reductions to advantaged-area service.
- **Solution:** made **supply also differentiable** using **Gaussian softmax smoothing** over the **5×5 grid cell counts of active taxis** — the same technique previously used for demand.
- The algorithm now has **two phases**:
  - **Trim:** manipulates demand by perturbing the **final pickup state** (previous approach).
  - **Lift:** manipulates supply by modifying the **last ~4 trajectory states prior to the pickup**, pushing taxis toward underserved areas.
- **King moves constraint enforced:** trajectories are restricted to **one-cell movements in any direction (including diagonals)**, consistent with pre-processing rules; the **previous algorithm violated this by allowing two-cell jumps**.
- This makes all edits more **realistic** and closes a potential reviewer criticism about simply reducing advantaged service.

### Results
- **F-causal metric improved by >54%** — described as a major gain.
- Improvements also **propagate to all external fairness metrics**.
- Results **consistent across both the Shenzhen and San Francisco datasets**.
- **Attribution coverage grew from ~2,400 trajectories to 7,500+ (~10% of the dataset)**, enabling a stronger argumentative arc: significant fairness gains with only a small fraction of data modified.
- A clear **ablation study between trim-only and trim+lift** has been recorded.

### Pending Evaluation
- GPU is currently running rollouts for **~80 (estimated) model combinations** (across two datasets) to evaluate whether fairness improvements **propagate through trained behavioral cloning models**.
- Results **expected later that afternoon**; this wraps the previous evaluation stage.

### Literature Review & Paper Argument
- Robert conducted an **AI-assisted literature review** to motivate the objective function and fairness terms.
- **Supporting references found** for each fairness term, attribution, and related design choices.
- **AI-generated references will be manually reviewed before use** to ensure accuracy.
- A **preliminary argument has been drafted for each component of the objective function**.

### Paper Writing
- Robert has maintained a **running argument document for ~1.5 months**; each algorithm change has been reflected in the document, making paper writing **primarily an assembly task**.
- Advisor suggested starting on the **methodology section**.
- Robert aims to submit a **draft abstract ahead of the abstract deadline** so the advisor has time to review it.
- Advisor noted the **abstract can serve as a placeholder and be refined later**.
- **Next weekly meeting scheduled** to review further progress.

---

## Cross-reference to requested topics of interest

- **External fairness metrics:** Present — disparate impact, demographic parity gap, Theil index; all three improved; leveling-down caveat recorded (over-served/advantaged areas reduced). See "External Fairness Metrics" above.
- **Supply-lift / trim+lift editor:** Present and central — the "Trim & Lift" algorithm; supply made differentiable via Gaussian softmax smoothing over 5×5 active-taxi counts; two phases (Trim=demand/final pickup state, Lift=supply/last ~4 states); King-moves one-cell constraint; F-causal >54% gain; trim-only vs trim+lift ablation recorded.
- **Data-augmentation baselines (ST-iFGSM / FGSM / random / oversampling):** Only a generic mention — action item "Robert to implement data augmentation baselines" (marked done [x]). The summary does NOT name any specific baseline (no ST-iFGSM, FGSM, random, or oversampling terminology appears).
- **α-weight sensitivity / Pareto:** NOT mentioned anywhere in the summary. (Only a passing reference to "current fairness metric weights" is absent here; no α-sweep or Pareto content.)
- **SF second dataset:** Present — results stated "consistent across both the Shenzhen and San Francisco datasets"; ~80 model combinations run "across two datasets."
- **KDD timeline:** No explicit "KDD" mention and no explicit calendar dates in the summary. Timeline content is limited to: draft abstract "by next week" / "ahead of the abstract deadline," abstract as a refinable placeholder, and a next weekly meeting.
- **Writing / drafting assignments:** Present — assemble methodology section; prepare draft abstract for advisor review; human-review AI-assisted literature references; maintain the running argument document.

---

## Anything Ambiguous or Suspicious (possible summary artifacts — flagged, not corrected)

1. **Internal heading date mismatch.** The summary's top heading "Research Progress Update: Fairness Algorithm & Paper Development" carries an inline mention-date of **2025-10-12**, which is inconsistent with the page's meeting date of 2026-07-09. Likely a Notion AI artifact; do not treat 2025-10-12 as the meeting date.

2. **">54%" F-causal improvement is a large, un-baselined figure.** The summary states "F-causal metric improved by >54%" with no absolute before/after values or denominator. This project's tracked headline supply-lift gains are reported in ΔF_causal absolute terms (e.g., Shenzhen +0.0222, SF +0.0328). A ">54%" relative framing is not something the internal records use and should be verified against the transcript and result artifacts before quoting — it may be a summary re-expression (relative-to-baseline) or an artifact.

3. **"~80 (estimated) model combinations."** The count is explicitly hedged as "estimated" in the summary; treat as approximate.

4. **Attendees under-listed.** The Group/Attendees property lists only "Dr. Xin Zhang," yet Robert is the presenter throughout. This is a properties-field limitation, not necessarily who attended.

5. **Data-augmentation baselines named only generically.** Given the project has specific baselines (ST-iFGSM/FGSM/random/oversampling), the summary's bare "implement data augmentation baselines" is an omission of specifics rather than a contradiction — flag for the transcript agent to fill in.

6. **No α-sweep / Pareto and no explicit KDD dates.** Their absence is notable given they are active workstreams; likely a summary omission rather than evidence they weren't discussed. Defer to the transcript.

7. **Metric naming.** The summary writes "F-causal" (hyphenated) and "demographic parity gap" / "Theil index." Internal docs use "F_causal" and note DP ≡ gap. Naming is consistent in substance; noted for exactness.
