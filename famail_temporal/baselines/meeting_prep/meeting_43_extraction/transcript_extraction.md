# Meeting 43 — Structured Extraction (from full transcript)

- **Meeting:** FAMAIL Group Bi-Weekly Summer 4 = **Meeting 43**, held **2026-07-16**
- **Source of truth:** `transcript_raw.md` in this directory (fetched from Notion page `39feb306-5110-8001-9bce-cab390801cc2`). The Notion auto-summary was checked against the transcript this time and found substantially faithful (see Surprises for the exceptions).
- **Attendees:** Robert Ashe (presented), Dr. Xin Zhang (PI), Dr. Kash. Manu present or expected but "We [don't] have much time for Manu to update." No speaker labels in transcript; attributions below are inferred from content and are high-confidence.
- **Transcription garbles:** "Dr. Chong" = Dr. Zhang; "Dr. Cash" = Dr. Kash; "TEAL"/"TEO index" = **Theil index**; "STFJSM" = ST-iFGSM; "word leaf" = Overleaf; "GPT-0" = GPTZero.

---

## 1. DECISIONS

### D-1. "Leveling down" framing ELIMINATED as the primary description (Figure 1 + paper) — Dr. Kash, accepted by Robert
The single biggest content decision of the meeting.
- Kash's argument: the algorithm **relocates** pickups, it does not destroy them — so it is not leveling down in the classic (loans-classifier) sense: *"you're not actually eliminating pickups in this picture right you're just kind of relocating those pickups so like there's a conservation of pickups going on here ... so it's not like leveling down in that classic sense"* (Dr. Kash).
- What trim-only DOES share with leveling down, and what to emphasize instead: it moves pickups *"from like the most advantaged place to like a ... moderately less advantaged place ... Which ... shares the important feature with ... leveling down that it's not helping the truly disadvantaged at all. Right, and I think that's the point to be emphasizing there. But it's not ... truly destructive in that way."* (Dr. Kash)
- Ruling: *"I would eliminate leveling down as the description of what's going on."* (Dr. Kash)
- Robert asked: remove entirely, or qualify as "pseudo leveling down"? Kash: *"I might keep leveling down as sort of an analogy when explaining what's going on ... I wouldn't even label the effect as quasi-"* — so: **analogy only (when explaining the earlier trim-only behavior); no "quasi/pseudo leveling down" label; never the primary description.**
- Terminology kept: *"I think you can keep the trim and lift terms, right, which are kind of your own bespoke thing."* (Dr. Kash)
- Robert accepted explicitly ("I hear you." / "now I feel pretty justified") — this decision is FINAL, and it obligates edits to Figure 1 and to every paper/doc passage that frames the trim effect as "leveling down" (notably the external-metrics leveling-down caveat and `PAPER/objective-motivation/LEVELING_DOWN.md`-derived prose).

### D-2. Submission logistics — Dr. Zhang
- **Anonymous submission**: *"for the submission, it's gonna be anonymous. So no auto[r] information in the paper"* (Zhang). Robert: *"Got the anonymous thing. It's a little one line comment out."*
- Zhang will initiate the **OpenReview** submission and add the author list; Robert sends him the abstract: *"you can just send me ... the abstract, initiated submission ... open review and [I'll add us] to the author list"* (Zhang).
- **Zhang begins reviewing the paper next week**: *"I'll be start reviewing your paper sometime next week"* — starting with *"the paper in terms of its organization and everything."*
- **Abstract/title are placeholders until the paper deadline**: *"for the abstract, we can always modify before the paper submission deadline. And title too ... as long as it's not far off. ... currently we just want to put a meaningful abstract, a meaningful title as a placeholder"* (Zhang).

### D-3. Code/data release scope — Dr. Zhang
- Code goes out as an **anonymous GitHub repository link**; it *"doesn't need to be perfect itself"*; **scrub PII first**: *"Make sure that your code doesn't have any kind of personal information like a password that has your name on it"* (Zhang).
- **Priority order decided**: *"the priority would be having all the paper contents ready, having all these experiments ready, and the data set as well as the code can be cleaned up after the submission"* (Zhang). Results-directory organization and what to make public: *"those can be finalized after the submission deadline"* (Zhang).

### D-4. Teasing figure for the introduction — Dr. Zhang (new request, accepted)
*"maybe like one kind of a teasing figure, [in] the introduction to try to highlight what's the problem we are trying to solve or what the overall claim looks like. You can refer to the other KDD paper ... ST-FGSM for kind of some idea"* (Zhang). Robert: "Love it."

### D-5. Page budget
KDD max is **8 main-content pages**; paper is at ~9: *"KDD gives us a maximum length of eight main content and right now I'm sitting at about nine pages"* (Robert). Shrinking is now a task (Overleaf's smaller fonts may absorb some of it).

### NOT decided / NOT discussed (asked-about items that did not occur)
- **SF framing (Reading A "ratio" vs Reading B "external-metrics/demand-endogeneity") and the D1 SF tier-2 recount engineering: NEVER RAISED.** No approval, no rejection — the open PI decision from MEETING_43_PREP item 2 remains open. (Robert asked Dr. Kash to stay for a private chat after the meeting — *"Dr. Cash, can I talk to you for a couple of minutes?"* — which is NOT in the transcript; if the SF framing was discussed, it happened there, unrecorded.)
- **F_causal → F_demo rename: NEVER RAISED.** Robert did not re-float it; no naming decision exists from this meeting.
- **Runs menu (A1/B1/B2/C1) / "fairness-baseline" comparison: not discussed by name.** No run was approved or rejected; the only run-related remark was Robert's *"there's a couple other experimental results that I'm popping out. Things that basically just round out tables."* No new experimental runs were requested by anyone.

---

## 2. ADVICE, PER PERSON

### Dr. Kash
1. **Citations (his headline warning):** *"it can't be emphasized enough that citations are getting so much more scrutiny than they used to. I have seen multiple papers just get essentially desk rejected because of citation issues."* Concrete guidance: *"we cannot let any AI citations get through ... don't accept any citation that AI gave you. Make sure that you've pulled all of the citations that are in the paper manually and ideally from a reputable source. So don't just take what Google Scholar gives you. I've actually seen papers get rejected because of essentially including garbage that Google Scholar returns. ... Seek the authoritative source for the citation, be it ACM, whatever ... a proper digital repository will give you a proper clean citation."*
2. **Figure 1 / leveling-down framing** — see Decision D-1 (eliminate as description, keep as analogy at most, keep trim+lift).

### Dr. Zhang (PI)
1. Teasing figure in the introduction, modeled on the ST-iFGSM KDD paper (D-4).
2. Anonymity mechanics + he drives OpenReview and the author list (D-2).
3. Anti-hallucinated-citation tactic (adding to Kash's point): *"put it into GPT-0 [GPTZero] or use the AI tool to help you to track hallucinated [citations]"*.
4. Triage: paper content and experiments first; repo/dataset cleanup and disclosure decisions after the deadline (D-3).
5. Abstract/title perfectionism deferred — meaningful placeholders now, refine until the paper deadline (D-2).

### Robert (self-set, for the record)
- Wants the group's "very human review" of the full paper; will send **Figure 1 + abstract** by email/Slack for cold-read scrutiny (deliberately not explaining Figure 1 first, to avoid biasing readers); wants a couple of days for his own first pass before opening Overleaf.
- Will pledge the **dataset + reproduction document** with the submission (*"I've heard that that's a major bonus for KDD reviewers"*); every run tracked *"down to the exact command"* so reviewers *"could even just hit go on a single script and reproduce all of our results."*

---

## 3. ACTION ITEMS

| # | Item | Owner | Due |
|---|------|-------|-----|
| 1 | Send polished abstract to Dr. Zhang for review/final say | Robert | **Jul 17** ("by tomorrow. It might happen today") |
| 2 | Set up paper on Overleaf for collaborative editing (after his own first pass, "a couple days") | Robert | ~Jul 18-19 |
| 3 | Send Figure 1 + abstract to advisors via email/Slack for cold-read feedback | Robert | before abstract deadline |
| 4 | Revise leveling-down framing per D-1 (Figure 1 + all paper prose; analogy-only, no quasi- label; keep trim+lift) | Robert | before paper deadline |
| 5 | Add "teasing figure" to the introduction (model: ST-iFGSM KDD paper) | Robert | before paper deadline |
| 6 | Manually verify ALL citations against authoritative sources (ACM etc.); no AI- or Google-Scholar-sourced citations; optionally GPTZero-style hallucination check; human eyes final | Robert | before paper deadline |
| 7 | Cut paper from ~9 pages to KDD's 8 main-content pages | Robert | before paper deadline |
| 8 | Scrub PII (passwords, names) from code; prepare anonymous GitHub repo link | Robert | before paper deadline (repo polish can lag) |
| 9 | Comment out author info for anonymous submission | Robert | at submission |
| 10 | Initiate OpenReview submission, add author list; begin reviewing the paper (organization first) | Dr. Zhang | next week (w/o Jul 20) |
| 11 | Dataset pledge + reproduction document for reviewers | Robert | with submission |
| 12 | (Deferred by decision) Results-dir organization, public-disclosure choices, repo cleanup | Robert | after Jul 26 |

**Deadlines confirmed in-meeting:** abstract **Sunday Jul 19**; paper **Sunday Jul 26**; abstract to Zhang **Jul 17**.

---

## 4. SURPRISES / CORRECTIONS vs standing assumptions

1. **Leveling-down framing reversal (D-1).** The project's standing docs (`PAPER/objective-motivation/LEVELING_DOWN.md`, the external-metrics leveling-down caveat, Figure 1) use "leveling down" as a load-bearing frame. Dr. Kash killed it as a *description*: relocation/conservation of pickups ≠ classic leveling down. Keep only as analogy for the earlier trim-only behavior; no "quasi/pseudo" label; the point to emphasize about trim-only is that it *"is not helping the truly disadvantaged at all"* — which is exactly what motivates lift. This requires a paper-wide + Figure-1 edit pass.
2. **The SF framing decision did NOT get made** despite the brief being ready (MEETING_43_PREP item 2). It was never raised in the recorded meeting; possibly covered in the unrecorded private Robert–Kash chat afterwards. Until Robert says otherwise, it is STILL OPEN.
3. **F_causal → F_demo rename was not re-floated.** No decision; standing state (optimization label kept, rename undecided) unchanged.
4. **No runs-menu adjudication.** Nobody approved/rejected A1/B1/B2/C1 or the fairness-baseline comparison; no new runs requested. The reproducibility/single-script packaging emphasis was implicitly endorsed by Zhang's comments.
5. **Page budget is 8, not ~10.** Paper had been reviewed in "10pp mode"; KDD main content max is 8 and the paper sits at ~9 — a real cut is required (Overleaf fonts may help).
6. **New anonymity/PII workstream** (author info comment-out, repo PII scrub, anonymous GitHub link, Zhang-driven OpenReview) — none of this was on the campaign map before.
7. **Robert publicly credited Dr. Kash for the lift idea** in front of the group ("very key realization from Dr. Cash there") — repairing the Meeting-42 summary's erasure of that credit.
8. **Notion-record artifacts to distrust:** the meeting-notes block carries a `mention-date` of **2025-10-12** (wrong; the meeting was 2026-07-16); the summary and transcript say "TEAL/TEO index" for **Theil index**; "Dr. Chong" = Dr. Zhang. Unlike Meetings 40-42, the auto-summary this time contained no detected fabrications against the transcript — but the transcript remains ground truth.
9. **Zhang's review starts next week** — i.e., between the abstract (Jul 19) and paper (Jul 26) deadlines, focused first on organization. Robert bought "a couple days" for his own emphasis/minimize pass before Overleaf goes live.
10. Robert flagged, vaguely but on the record, that *"there are results that aren't exactly what we expect"* and he intends to present them as a *"realism improvement for the paper"* (e.g., demographic oversampling being "surprisingly good" but weaker than trajectory editing) — consistent with the surface-don't-hide doctrine.
