# Meeting 43 — Notion AI Auto-Generated Summary (raw, verbatim)

**Source:** Notion page "FAMAIL Group Bi-Weekly Summer 4" (Meetings Database), `<summary>` block.
Page: https://app.notion.com/p/39feb306511080019bcecab390801cc2
Fetched: 2026-07-16T18:17Z. Meeting date: 2026-07-16.

WARNING — **UNTRUSTED SOURCE.** This is the Notion auto-summary, which has fabricated or omitted decisions in Meetings 40, 41, and 42. The transcript is ground truth. Inline footnote links (`[^https://...]`) pointing into the transcript page (39feb306511080e7a3b1ffc019c9189b) have been stripped for readability; every bullet below carried one or more such anchors in the original.

---

### Action Items
- [ ] Robert Ashe to send a polished abstract to Dr. Zhang by tomorrow for review
- [ ] Robert Ashe to set up the paper on Overleaf for collaborative editing
- [ ] Robert Ashe to send Figure 1 (and abstract) to advisors via email or Slack for feedback
- [ ] Robert Ashe to add a "teasing figure" to the introduction (refer to ST-FGSM KDD paper for inspiration)
- [ ] Remove all personally identifiable information (e.g., passwords, names) from code before submitting anonymous GitHub repo link
- [ ] Manually verify all citations from authoritative sources (ACM, etc.) — do not rely on AI-generated or Google Scholar citations
- [ ] Revise "leveling down" framing in paper and Figure 1 — use it only as an analogy, not as the primary description of the effect
- [ ] Reduce paper length from ~9 pages to the KDD maximum of 8 main content pages
- [ ] Dr. Zhang to begin reviewing the paper next week

---

### Meeting Overview
- Robert Ashe presented a detailed progress update on a paper about fairness-aware trajectory modification for taxi GPS data, targeting a **KDD submission**
- Two upcoming deadlines: **abstract due this Sunday**, **full paper due the following Sunday**

---

### Research Progress
- The core method — the **FAMAIL Trim+Lift algorithm** — modifies taxi trajectories to improve global fairness: "trim" reduces over-service to advantaged areas; "lift" reroutes taxi supply to underserved regions
- External fairness metrics (**disparate impact**, **demographic parity**, **TEAL index**) confirm improvement on metrics not directly optimized for
- A **grid search** over objective function weights found more optimal parameters than prior trial-and-error; this triggered a full rerun of the experimental stack (~1 week of GPU time)
- Results are shown on two datasets: **Shenzhen** (primary) and **San Francisco**, demonstrating generalizability
- Ablation baselines include: random jitter, iterative FGSM, and **demographic oversampling** (a competitive but weaker baseline)
- Fairness-aware training (up-weighting edited trajectories) shows fairness propagates through trained models in a **monotonically increasing** fashion
- All experimental runs are logged down to the exact command for full **reproducibility**

---

### Paper Status
- Prose for the full paper is written; now needs trimming, citation checks, and final validation
- Submission will be **anonymous** — author information must be commented out
- An anonymous GitHub repository link will be provided for code; the repo doesn't need to be fully polished before submission
- Data/results organization and public availability decisions can be finalized **after** the submission deadline
- AI adversarial review has been used throughout to find argument gaps and verify citations

---

### Dr. Kash's Comments
- **Citations:** Citations are receiving significantly more scrutiny at venues like KDD; papers have been desk-rejected over citation issues. All citations must be pulled manually from authoritative sources (e.g., ACM digital library); AI-generated citations and Google Scholar output should not be trusted. Using AI tools (e.g., GPT-based detectors) to flag hallucinated citations was suggested as an additional check
- **Figure 1 — "Leveling Down" Framing:** The classic "leveling down" framing does not accurately describe the algorithm's effect — in this context, pickups are **relocated**, not eliminated, so there is a conservation of service rather than true destruction of value. Recommended approach: **eliminate "leveling down" as a primary description**; it may be kept as an **analogy** when explaining the earlier, trim-only behavior, but should not label the overall effect. The **"trim and lift" terminology** should be kept as the paper's own bespoke framing

---

### Dr. Zhang's Comments
- Suggested adding a **"teasing figure"** in the introduction to highlight the problem being solved and the overall claim; referenced the ST-FGSM KDD paper as a stylistic example
- Will begin reviewing the paper next week and start the OpenReview submission process
- Abstract and title can be refined up until the paper submission deadline; current versions are acceptable as placeholders
