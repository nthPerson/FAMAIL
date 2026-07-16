# Meeting 43 (2026-07-16) — Notes + Auto-Summary Extraction

**Meeting:** Notion "FAMAIL Group Bi-Weekly Summer 4" = project Meeting 43. Held 2026-07-16. PI Dr. Xin Zhang; Dr. Kash present; Robert Ashe presenting.
**Sources:** `notes_raw.md` (Robert's manual notes, sparse) + `summary_raw.md` (Notion auto-summary).
**Page:** https://app.notion.com/p/39feb306511080019bcecab390801cc2 (transcript subpage: 39feb306511080e7a3b1ffc019c9189b — fetched by a parallel agent).

---

## ⚠️ CAVEAT — READ FIRST

This project has repeatedly caught the Notion auto-summary **fabricating or omitting decisions** (Meetings 40, 41, 42 all had summary errors: fabricated frameworks, understated decisions, erased attributions). **The auto-summary is UNTRUSTED; the transcript is the ground truth.** Everything below tagged [SUMMARY] must be cross-checked against the transcript extraction before it drives any paper edit or run decision. Items tagged [NOTES] are corroborated by Robert's own manual notes (still terse — the transcript remains authoritative).

**Load-bearing items that MUST be transcript-verified:**
1. "Eliminate leveling down as primary description, keep only as analogy for trim-only behavior" — this would force edits to Figure 1, `PAPER/objective-motivation/LEVELING_DOWN.md`-derived prose, and §4 external-metrics caveat text. Corroborated by [NOTES], but the exact scope (Figure 1 vs. all prose) needs the transcript.
2. "Keep trim+lift as bespoke terminology" — [SUMMARY]+[NOTES], but notes end with a truncated "Don't use" bullet whose object is missing. What is it we should NOT use? Transcript needed.
3. Deadline readings: "abstract due this Sunday" (= Jul 19, matches KDD abstract deadline) and "full paper due the following Sunday" (= Jul 26). Consistent with known deadlines, but confirm nothing moved.
4. "TEAL index" is almost certainly a mis-transcription of **Theil index** (the external metric we actually compute). Do not propagate "TEAL" anywhere.
5. "ST-FGSM KDD paper" is presumably the **ST-iFGSM** paper (our KDD template). Same mis-transcription risk.
6. The summary describes the grid search/rerun and demographic-oversampling baseline as if presented facts — fine — but check the transcript for whether the PI **approved or requested any additional runs** (see "Not found" below).
7. Attribution risk: Meeting 42's summary erased Dr. Kash's credit for the lift idea. Verify the Kash/Zhang attributions below against the transcript.
8. Notion artifact: the meeting-notes block carries a stray date mention "2025-10-12"; page properties say 2026-07-16 (correct).

---

## Decisions

- **[SUMMARY]+[NOTES] "Leveling down" framing revision (Dr. Kash):** classic leveling-down does not describe the algorithm — pickups are *relocated*, not eliminated (conservation of service, not destruction of value; "not leveling down in the classic sense"). Eliminate it as the primary description in the paper and Figure 1; permissible only as an analogy for the earlier *trim-only* behavior.
- **[SUMMARY]+[NOTES] Keep "trim and lift"** as the paper's own bespoke terminology (Dr. Kash endorsed).
- **[SUMMARY] Anonymous submission:** author info commented out; anonymous GitHub repo link for code; repo need not be fully polished pre-submission.
- **[SUMMARY] Data/results public-availability decisions can wait until after the submission deadline.**
- **[SUMMARY] Abstract/title are placeholders** — refinable until the paper deadline; current versions acceptable (Dr. Zhang).

## Action Items (owner in bold)

- **Robert**: send polished abstract to Dr. Zhang **by tomorrow (Jul 17)** for review. [SUMMARY]
- **Robert**: set up the paper on **Overleaf** for collaborative editing. [SUMMARY]
- **Robert**: send **Figure 1 + abstract** to advisors (email or Slack) for feedback. [SUMMARY]
- **Robert**: add a **"teasing figure"** to the introduction (style ref: ST-[i]FGSM KDD paper). [SUMMARY]+[NOTES] (Dr. Zhang's suggestion)
- **Robert**: remove all PII (passwords, names) from code before the anonymous GitHub link. [SUMMARY]+[NOTES]
- **Robert**: manually verify ALL citations against authoritative sources (ACM DL etc.); do not trust AI-generated or Google Scholar citations; optionally use AI tools as an extra hallucination check. [SUMMARY]+[NOTES] (Dr. Kash)
- **Robert**: revise "leveling down" framing in paper + Figure 1 (analogy only). [SUMMARY]+[NOTES]
- **Robert**: cut the paper from ~9 pages to the **KDD 8-page main-content max**. [SUMMARY]
- **Dr. Zhang**: begin reviewing the paper next week; start the **OpenReview** submission process. [SUMMARY]

## Advice (attributed)

- **Dr. Kash — citations/desk-rejection:** citation quality is under significantly more scrutiny at venues like KDD; papers have been **desk-rejected over citation issues**. Pull every citation manually from authoritative sources. [SUMMARY], corroborated [NOTES] "Citations! Getting more scrutiny".
- **Dr. Kash — Figure 1 / leveling-down:** as under Decisions.
- **Dr. Zhang — teasing figure** in the intro highlighting problem + overall claim (ST-FGSM KDD paper as stylistic example). [SUMMARY]+[NOTES]
- **Dr. Zhang — process:** will review next week; abstract/title refinable to the deadline. [SUMMARY]

## Targeted topics — findings

- **(a) SF framing decision (Reading A "ratio" vs Reading B "external-metrics"):** **NOT MENTIONED** in either notes or summary. If it was discussed, the summary omitted it (a known failure mode) — check the transcript. The summary only says SF results "demonstrate generalizability".
- **(b) F_causal rename / F_demo re-float:** **NOT MENTIONED** in notes or summary. Transcript check required.
- **(c) Additional experimental runs:** no new run approvals/requests appear. Summary reports the α grid search + full ~1-week GPU rerun and reproducibility logging as *presented status*, and calls demographic oversampling "competitive but weaker". No PI directive for further experiments recorded — verify in transcript.
- **(d) Citation quality / desk-rejection:** YES — Dr. Kash, as above (the meeting's strongest process warning).
- **(e) Deadlines / 8 pages / Overleaf / abstract:** abstract due "this Sunday" (Jul 19); full paper "the following Sunday" (Jul 26); reduce ~9 → 8 pages (KDD max); Overleaf setup action item; polished abstract to Zhang by Jul 17; Zhang review + OpenReview next week. [SUMMARY]

## Suspicious / ambiguous

- "TEAL index" (→ Theil), "ST-FGSM" (→ ST-iFGSM): transcription slips; do not copy into any document.
- Notes bullet "Don't use ___" is truncated — object unknown.
- Summary claims external metrics "confirm improvement on metrics not directly optimized for" with no city/metric caveats — our own findings show SF is weaker (migrant n.s.) and leveling-down caveats; the summary may be over-smoothing what was actually said.
- No mention of the SF framing decision or F_demo rename anywhere — either not discussed or omitted by the summary.
