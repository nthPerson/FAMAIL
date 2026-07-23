# Meeting 44 — Decisions, Actions & Submission Logistics

**Meeting:** 44 · 2026-07-23 (Thu) · ~2h00m · Robert (RA, he/him) + Dr. Xin Zhang (PI, she/her)
**Purpose:** Full-draft review of the FATE KDD 2027 paper. **Global verdict: paper too complicated → problem-driven simplification rewrite before submission.**
**Submission target discussed:** Sunday **2026-07-26** (ahead of the Monday **2026-07-27** deadline).

**Lens of this record:** what was decided, who owns what, submission mechanics. Extracted from the transcript as the primary source; Plaud auto-summaries used only as cross-checks and corrected where they diverge.

**Reading notes / speaker & name cautions (verified against transcript):**
- "Dr. Xin Zhang" merges Plaud voice-IDs 1/2/4; "Robert" = Speaker 3. Judged by content throughout.
- **"Laura" = Robert.** Diarization artifact; only two people attended. Confirmed at [1:39:40] where Dr. Zhang addresses Robert's work as "Laura is… your work. You have full control." Every Plaud "Laura" owner is corrected to Robert.
- **"Katie" ([0:19:16] "the templates that Katie gave us")** is not an attendee and is almost certainly a mis-transcription of "KDD" (the template source). Not used as an owner anywhere. (Flagged, not load-bearing.)
- "Tim" [0:09:33] and "Manuel" [1:58:38] are mentioned in passing (an email; a person Dr. Zhang left to meet). Neither owns any action here.
- Timeline anchor: 07-23 Thu (meeting), 07-24 Fri, 07-25 Sat, **07-26 Sun (submission target)**, **07-27 Mon (deadline)**.

---

## A. DECISION TABLE

| ID | Decision | Status | Decider | Source [h:mm:ss] + quote |
|----|----------|--------|---------|--------------------------|
| **D1** | Switch to the **corrected KDD template**: remove the **CCS-concepts** block, replace with a **keywords** block. Dr. Zhang already made the edit in her copy. | **DECIDED** | Dr. Zhang | [0:11:22] "the template is wrong but it's okay, I have… modified it to the correct version"; [0:11:47] "this is a correct version, so we can just use this one" |
| **D2** | **Global rewrite before submission**: reorganize the paper to be **problem-driven** ("why we did it", not "what we did"). Rest of the days pre-deadline = rewriting on top of Robert's draft. | **DECIDED** | Dr. Zhang (PI) | [0:50:46] "We'll be doing the rest of the days before… the deadline… rewriting the paper based on what you have already done" |
| **D3** | **Reorganize the introduction**: crisp motivation + existing-works limitations; **shrink the FATE-approach description to one paragraph** (details live in methodology). | **DECIDED** | Dr. Zhang | [0:50:46] "reorganize it so it's having a very clear motivation and existing works limitation… shrink the FATE's proposed approach part into one paragraph" |
| **D4** | **Add a "Challenges" subsection** to the introduction: what makes the problem hard and why prior methods fail. | **DECIDED** | Dr. Zhang | [1:25:46]/[1:26:22] "what we forgot to mention… in the introduction… is the challenges… what makes this problem challenging" |
| **D5** | **Add a motivating example** to the intro (existing approaches e.g. GAIL inherit/amplify unfairness). Figure 1 does part of the work. | **DECIDED** — *mechanism OPEN* | Dr. Zhang | [0:51:43] "a motivating example… existing approaches like GAIL… amplifying or inheriting the unfairness"; **OPEN sub-choice:** "One approach is… using some news… Another approach… experiment results… in the first figure" |
| **D6** | **Rename section "Task" → "Problem Formulation"** (define trajectories, reward, states/actions). | **DECIDED** | Dr. Zhang | [1:25:21] "There's not a problem formulation. It's just a task. You can just give it… problem… formulation" (also Plaud highlight #2) |
| **D7** | **Shrink Related Work to ~half a column** in the main body; move the **full version to the appendix**. | **DECIDED** | Dr. Zhang | [1:20:02] "we can shrink it into minimum of a half column… leave a complete version in the appendix" |
| **D8** | **Move the F_demo derivation to the appendix**; keep only the **core equation `1 − R²_demo` + its meaning** in the main text, with a pointer to the appendix. (Hat-matrix ref [13] travels with the derivation — see Discrepancies.) | **DECIDED** | Dr. Zhang | [1:23:32] "point readers towards appendix in terms of how it's derived"; [1:24:14] R: "move all of this to the appendix?… keep the one minus R square demo?" Z: "Yes, exactly" |
| **D9** | **Move implementation details (0.01° grid cells) to the appendix**; keep a generalizable state-space framing in methodology to stress the approach transfers (transit logs, logistics data). | **DECIDED** | Dr. Zhang | [1:29:27] "we don't need to… write in the detail about… 0.01 degree grid cells. Those are… implementation details that could be pushed into the appendix" |
| **D10** | **Remove cross-section duplication** — esp. problem-formulation details repeated in experimental setup. | **DECIDED** | Dr. Zhang | [1:37:28] "make sure those information doesn't appear in your experimental setup" |
| **D11** | **Redesign Figure 1** in Figure 2's style: **map background, legends, explicit "advantaged/disadvantaged" district labels**, larger text, tighter spacing; make it **self-contained** (Fig 1 and Fig 2 must each stand alone — they sit far apart). | **DECIDED** | Joint (Robert accepts) | [1:18:40] R: "I will make sure that the legends and everything are duplicated on both"; [1:17:25] Z: "the style of Figure two would work better to replace these two smaller figures in Figure one" |
| **D12** | Figure 1 must keep **total taxis and total passengers constant** across before/after (relocation, not removal). | **DECIDED** | Dr. Zhang | [1:18:40] "we are not removing taxis or removing passengers… we are relocating them… the number of taxis stays the same" |
| **D13** | **Show the fairness metric visually inside Figure 1** (pre-editing fairness score / the "3× service gap"), not only in the caption. | **DECIDED** | Dr. Zhang | [0:53:37]→[0:53:52] "we'd better show that information in the figure"; [1:04:57] "it's not clear why… the advantage district receives three times the taxi service" |
| **D14** | **Figure text slightly smaller than caption/body text; captions concise** (only the most important info, because space is 8 pages). *(An early "don't modify captions, leave as is" remark at [0:19:16] was superseded by this.)* | **DECIDED** | Dr. Zhang | [0:15:10] "text in the figure… slightly smaller than this caption size"; [0:17:24] "we just explain the most important information in the caption" |
| **D15** | **Citations**: any *grouped class* of related work needs **≥2 citations** (not one); prefer **more recent works (2025–2026)**. | **DECIDED** | Dr. Zhang | [0:39:12] "at least two"; [0:32:33] "more current… like 2025, 2026" |
| **D16** | **Replace vague AI phrasing with precise domain terms** — specify the *level* of intervention (data / model / hyperparameters) and *which* distribution; avoid coinages like bare "in-processing", "rebalancing", "shifts the distribution". | **DECIDED** | Dr. Zhang | [1:46:49] "GPT is usually gonna use very general language… not easily understandable by domain experts"; [0:39:32] "in processing is… too broad" |
| **D17** | **Terminology: "realism" → "fidelity"** (Dr. Zhang recommends against "realism"). | **OPEN** | leaning Dr. Zhang; Robert flags conflict | [1:41:47] "I wouldn't recommend using realism because realism usually means other things instead of fidelity." **Competing reading (Robert, unresolved):** paper already defines **Fidelity-A/B** specifically, so using "fidelity" generically in the intro overclaims that cited works meet *our* definition — [1:49:56]/[1:52:14]. No clean resolution reached. |
| **D18** | **Writing workflow: draft manually first, use GPT only to refine** — not GPT-first drafts. | **DECIDED** (guidance) | Dr. Zhang | [1:45:36] "you write it first, and you ask GPT to refine it… the problem with asking GPT to write it first… will be very subtle" |
| **D19** | **Data availability: do NOT release raw data** (source: Dr. Yanhua Li). Release only **anonymized data + code** via an anonymous GitHub link. | **DECIDED** | Dr. Zhang | [1:57:06] "we don't release the raw data, we only release some… anonymous data… We just give people… GitHub link" |
| **D20** | **Anonymous GitHub link placed in the introduction; an empty repo at submission is acceptable.** | **DECIDED** | Dr. Zhang | [0:51:43] "an anonymous link telling people… where… the code will be released"; [1:59:06] "It's okay… GitHub is empty for now" |
| **D21** | **Data-availability statement** will say *code is shared via a GitHub link*; say little/nothing about the (unreleased) data. | **DECIDED** | Dr. Zhang | [1:59:17] "We can just say that we share our code via a GitHub link… We don't need to say too much about the data" |
| **D22** | **Proceed with the artifact pledge on the basis of code availability** (Robert noted the pledge implies data+code available; Dr. Zhang accepted proceeding with code only). | **DECIDED** (soft) | Dr. Zhang | [1:59:29] R: "artifact pledge… implies… data and code are available." [1:59:48] Z: "Yeah, it's okay." |
| **D23** | **Detailed reproducibility documentation deferred until AFTER submission.** | **DEFERRED-POST-SUBMISSION** | Dr. Zhang | [1:58:38] "Can we… leave this after the submission deadline?"; [1:58:53] "as long as we… leave a link to one anonymous GitHub" |
| **D24** | **Submit on Sunday 2026-07-26**, ahead of the Monday deadline, to avoid last-minute failure. | **DECIDED** | Robert commits; Dr. Zhang concurs | [1:55:52] "Sunday's what I've committed to in my mind"; [1:53:51] Z: "the deadline is this Sunday, right? So hopefully we can finish… before that" |
| **D25** | **Verify the authoritative deadline + timezone** (OpenReview *local time* vs official site; AoE = "Hawaiian time"). The exact cutoff was not conclusively pinned in-meeting. | **OPEN** | — (verification pending) | [1:54:22] "anywhere on earth means… Hawaiian time"; [1:55:27] "Our open review is showing local time"; [1:55:52] "Monday… July twenty seventh at eleven fifty nine… is what the open review says" |
| **D26** | **Page budget: final ≤ 8 pages main content.** Going over **temporarily is fine**; length is "the least important thing" and always compressible; appendix absorbs moved content. | **DECIDED** | Dr. Zhang | [0:47:35] "totally okay to go over length at the current stage as long as we cut it in eight pages before the deadline"; [1:28:07] "length is the least important thing… we can always shrink it within eight pages" |
| **D27** | **Dr. Zhang will (time permitting) draft her own version of the intro/abstract in a new file**; Robert may adopt it or not. (No overwrite risk — she uses a new file.) | **DECIDED** (optional deliverable) | Dr. Zhang | [0:52:42] "If I have time, I'm gonna rewrite a version, and it's up to you whether you want to use the written version or not"; [2:00:03] "I'm gonna create a new text file" |
| **D28** | **Scope/positioning: frame the contribution as targeted data augmentation for imitation learning (behavior cloning)** — fairness survives to the learned policy via **upweighting** edited samples; extension beyond BC is future work. | **DECIDED** (converged after debate) | Joint | [1:03:10] Z: "we are designing an approach specifically for imitation learning so that the learned policy… is able to provide more fair services. That is a claim"; [1:03:53] R reframes as a "dual finding", not watering down |

**Decision counts:** DECIDED **25** · OPEN **2** (D17 terminology, D25 deadline) · DEFERRED-POST-SUBMISSION **1** (D23) · REJECTED **0**. *(D5 is DECIDED with an OPEN sub-choice on the motivating-example mechanism.)*

---

## B. ACTION TABLE

All owners are real attendees (Robert / Dr. Zhang / joint). Due dates are shown **only where the transcript states one**; otherwise "unstated (pre-submission)" — meaning it must land before the Sunday target but no explicit per-item date was given.

| ID | Action | Owner | Due (if stated) | Source | Deps / order |
|----|--------|-------|-----------------|--------|--------------|
| **A1** | Adopt Dr. Zhang's corrected template (CCS-concepts block → keywords block) into the working draft. | Robert (adopt); Dr. Zhang authored the fix | unstated (pre-submission) | [0:11:56]/[0:12:08] | **Blocker:** handoff path for the corrected file is unspecified (see Logistics §1) |
| **A2** | Enlarge figure text (slightly smaller than caption); reduce figure spacing/gaps to compact figures. | Robert | unstated | [0:15:44]/[0:13:31] | — |
| **A3** | Redesign Figure 1 (Fig-2 style: map background, legends, advantaged/disadvantaged labels, self-contained; colors consistent with Fig 2). | Robert | unstated | [1:18:40]/[1:17:25] | Enables A5 |
| **A4** | Ensure Figure 1 keeps total taxis + total passengers constant across panels (relocation). | Robert | unstated | [1:18:40] | Part of A3 |
| **A5** | Show pre-editing fairness score / visualize the "3× service gap" inside Figure 1. | Robert | unstated | [1:04:57]/[0:53:37] | After A3 |
| **A6** | Shorten figure captions to essentials. | Robert | unstated | [0:16:47] | — |
| **A7** | Reorganize the introduction (motivation + existing-works limitations; FATE approach → 1 paragraph). | Robert (lead); Dr. Zhang optional draft (A22) | unstated (pre-submission) | [0:50:46] | Coordinate w/ A22/A25 |
| **A8** | Add a "Challenges" subsection to the introduction. | Robert | unstated (pre-submission) | [1:26:22] | Part of A7 |
| **A9** | Add a motivating example to the intro (GAIL/existing approaches amplify unfairness); decide news vs early experimental result in Fig 1. | Robert | unstated | [0:51:43] | Mechanism OPEN (D5) |
| **A10** | Rename "Task" → "Problem Formulation"; define trajectories, reward, states/actions. | Robert | unstated (pre-submission) | [1:25:21]/[1:32:12] | — |
| **A11** | Shrink Related Work to ~half column; move full version to appendix. | Robert | unstated | [1:20:02] | Frees main-body space |
| **A12** | Move F_demo derivation (incl. hat-matrix material) to appendix; keep `1 − R²_demo` + meaning + appendix pointer in main. | Robert | unstated | [1:24:14] | — |
| **A13** | Move 0.01° grid-cell implementation detail to appendix; keep generalizable framing in methodology. | Robert | unstated | [1:29:27] | — |
| **A14** | Remove cross-section duplication (problem-formulation vs experimental setup). | Robert | unstated | [1:37:28] | — |
| **A15** | Strengthen citations: ≥2 per grouped class; add recent (2025–2026) works. | Robert | unstated | [0:39:12]/[0:32:33] | — |
| **A16** | Replace vague terms (in-processing, rebalancing, "shifts the distribution") with precise domain language; resolve realism/fidelity (D17). | Robert | unstated | [1:49:18]/[1:49:56] | D17 OPEN |
| **A17** | Tighten formatting — bulleted lists, remove blank spaces — to reclaim space. | Robert | unstated | [1:40:30] | — |
| **A18** | Create the anonymous GitHub repo and place the anonymous link in the introduction (empty repo acceptable at submission). | Robert (implied; not explicitly assigned) | unstated (pre-submission) | [1:59:06]/[0:51:43] | — |
| **A19** | Write the data-availability/licensing statement (code shared via GitHub; raw data not released). | Robert | unstated (pre-submission) | [1:57:06]/[1:59:17] | — |
| **A20** | Verify the authoritative deadline + timezone (OpenReview local time vs official site; AoE/Hawaiian). | Joint (unassigned lead) | before Sunday | [1:54:22]/[1:55:27] | Gates A21 |
| **A21** | Submit the paper. | Robert (lead); Dr. Zhang support | **Sunday 2026-07-26** | [1:55:52] | After A20; all above |
| **A22** | Dr. Zhang to draft a rewritten intro/abstract in a new file (time permitting) and share it with Robert. | Dr. Zhang | unstated (conditional "if I have time") | [0:52:42]/[2:00:03] | Feeds A7 |
| **A23** | Dr. Zhang to email further suggestions; Robert to incorporate them. | Joint | unstated | [1:39:19]/[1:53:51] | — |
| **A24** | Compress final manuscript to ≤8 pages main content before the deadline (losslessly where possible). | Robert | before deadline | [0:47:35]/[1:28:07] | Final step w/ A21 |
| **A25** | Reproducibility documentation (claim→artifact map, re-run recipes). | Robert | **DEFERRED — after submission** | [1:58:38] | Post-07-27 |
| **A26** | Coordinate so Robert does not overwrite Dr. Zhang's intro work (she uses a new file). | Joint | unstated | [1:55:55]/[2:00:03] | — |

**Action counts:** total **26** · owner Robert **~18** · Dr. Zhang **1** (A22; + co-author of A1) · joint **4** (A20, A23, A26; A21 lead-Robert) · with an explicit due date **2** (A21 Sunday; A25 deferred post-submission) · all others "unstated (pre-submission)".

---

## C. LOGISTICS DOSSIER

### C.1 — TEMPLATE

**What was wrong:** the KDD template Robert started from **included a CCS-concepts block**; the group's prior version **replaced that with a keywords block**. Robert notes the document-class line is standard acmart sigconf anonymous review (his line 47) and that CCS-vs-keywords is "just a one-line insert into the text file."
- [0:10:09] Z: "you are removing the CCS concepts part and replace it by keywords."
- [0:11:04] Z: "they are also including this CCS concept part. But in our previous version… this part is replaced by keywords."
- [0:10:41] R: "documents class SigConf anonymous review ACM Art… line forty seven."

**What Dr. Zhang did:** she edited a copy herself to the corrected (keywords) version and confirmed she did it directly.
- [0:11:22] "the template is wrong but it's okay, I have… modified it to the correct version."
- [0:11:56]→[0:12:08] R: "you made the change directly in—" Z: "Yes."
- She also tried adding the TikZ figures into her copy but they wouldn't render — **package conflict**, which she attributed to her own unfamiliarity with LaTeX-source figures (she prefers screenshots): [0:12:15] "By using the package that you use, but it doesn't seem to be showing correctly… a package conflict issue."

**WHERE the corrected template lives — UNSTATED (finding).** No platform is named (not Overleaf, email, or repo). She modified her own working copy; the **handoff path to Robert is unspecified.** She separately offers to email suggestions [1:39:19] and, for the intro, says she'll "create a new text file" [2:00:03]. **Robert must confirm how he receives the corrected template** (this is the practical blocker behind A1).

**What Robert must do to adopt it:** swap the CCS-concepts block for the keywords block (a small header-metadata edit). 

**Does it change layout/page count?** No statement that it does. It is a header-block swap within the same acmart **sigconf, anonymous, review** class — not a column/format change — so no page-budget impact is indicated.

### C.2 — DEADLINE / SUBMISSION

**Sunday target:** both agreed to aim for **Sunday 2026-07-26**, ahead of the deadline, because last-minute submission is risky (Robert cites how long the *abstract* submission unexpectedly took).
- [1:53:51] Z: "the deadline is this Sunday, right? So hopefully we can finish… before that."
- [1:55:52] R: "deadlines freak me out… Sunday's what I've committed to."

**Deadline ambiguity (unresolved in-meeting → D25/A20):**
- Dr. Zhang initially thought **Sunday July 26** [1:55:15] and had checked the website [1:54:31].
- Robert read the **OpenReview page** as **Monday July 27, "11:59"** — he says **"eleven fifty nine a.m."** and "Monday morning" [1:55:52]; but he recalls the *abstract* deadline as **"11:59 p.m."** [transcript around 1:54:31/0:? — the AM/PM is internally inconsistent].
- Timezone: **official = "anywhere on earth" = Hawaiian time (AoE)** [1:54:22]; **OpenReview shows local time** [1:55:27], which for their timezone pushes the wall-clock to Monday [1:55:32].
- Dr. Zhang conceded she "did misread" [1:55:48]; net reconciled reading = **Monday 07-27 on OpenReview local time**, but the **exact hour (11:59 AM vs PM) was not verified.**

**Bottom line:** target **Sunday 07-26**; deadline **Monday 07-27** (AoE officially / local time on OpenReview); **exact cutoff hour still to be verified** (A20). The AM/PM discrepancy is a genuine finding — do not treat "23:59" as confirmed.

**Updating the already-submitted abstract:** **NOT RAISED.** The abstract is mentioned only as a deadline reference point ([1:54:10] Robert recalls it "was extended a day to Monday"). **No decision or action to update/revise the submitted abstract.**

### C.3 — DATA / CODE AVAILABILITY

- **Raw data NOT released.** Source is **Dr. Yanhua Li**; Robert asks whether her permission is needed. [1:57:06] Z: "we don't release the raw data, we only release some… anonymous data… we don't need to release the data for now."
- **Anonymized data + code** shared via an **anonymous GitHub link placed in the introduction** [0:51:43].
- **Empty repo at submission is acceptable.** [1:59:06] "It's okay… GitHub is empty for now."
- **Who creates the anonymous repo:** not explicitly assigned; **Robert by implication** (he owns repo/reproducibility work). Placement of the link may land via Dr. Zhang's intro draft.
- **Data-availability statement wording:** state that code is shared via the GitHub link; do not say much about the data. [1:59:17] "We can just say that we share our code via a GitHub link… We don't need to say too much about the data or anything."
- **Artifact pledge / badging:** Robert flags that the pledge "implies that the data and code are available"; Dr. Zhang accepts proceeding on **code availability** — [1:59:48] "Yeah, it's okay." (No badging tier discussed; soft acceptance.)
- **Reproducibility documentation deferred until after submission.** Robert has a full claim→artifact reproducibility doc ready to show [1:57:24]/[1:58:10]; Dr. Zhang, out of time (leaving to meet Manuel), asks to defer it: [1:58:38] "Can we… leave this after the submission deadline?"

### C.4 — PAGE BUDGET

- **Final constraint: ≤ 8 pages main content.** Going over **temporarily is explicitly fine.** [0:47:35] "totally okay to go over length at the current stage as long as we cut it in eight pages before the deadline… all the important information are complete… instead of… throwing away some of the details."
- **Length is de-prioritized:** [1:27:17] "I don't think length will be a problem. For desk rejection?"; [1:28:07] "The length is the least important thing. We can always… shrink it within… eight pages." Dr. Zhang attributes prior length pressure to an over-long related-work section [1:27:22].
- **Appendix:** used as **overflow** — full related work [1:20:02], F_demo derivation [1:24:14], grid-cell implementation detail [1:29:27] all move there. **No page cap on the appendix was discussed** (treated as unlimited overflow). **Caveat that governs what may move:** reviewers are **not required to read the appendix**, so the main body must remain self-sufficient — [1:22:57] R: "The reviewers aren't required to read the appendix, so I want to provide them in the body with as much as they need."
- **Current draft state (per Robert):** main content is **just under 8 pages** ([0:17:35] "Right now we're under eight pages as far as main content"), with **~3–4 lines of slack** ([0:47:24]). The **"~11 pages"** figure is **Robert's pre-compression draft** ([0:48:44] "Landed at about eleven pages, so then… compacting all of that argument into eight pages") — **not the current version** (see Discrepancies).

---

## D. PRIOR-AGENDA RECONCILIATION

Robert's prep agenda vs what actually happened:

| # | Prep item | Verdict | Detail |
|---|-----------|---------|--------|
| (i) | Ratify the recent **"argument triage"** (allocation-boundary / SF-detail / recount-mechanics → appendix) | **NOT RAISED by name — but the general strategy was endorsed and extended** | No item was reviewed by name. The *principle* of appendix-residence was affirmed and Dr. Zhang directed **more** to move there (related work [1:20:02], F_demo derivation [1:24:14], grid-cell detail [1:29:27]). Robert referenced appendix-migrated content generally at [0:20:37]/[0:31:27]. The specific prior triage decisions were never itemized or explicitly ratified. |
| (ii) | Approve the **new figure color scheme** | **DISCUSSED — NOT approved; critiqued** | The yellow/blue coloring was reviewed; Dr. Zhang found it **unclear** ([1:09:04] "it looks… like this yellow area is where people park their cars") and required explicit "advantaged/disadvantaged" labels + consistent colors across figures [1:05:24]/[1:18:40]. Outcome = rework, not approval. |
| (iii) | Walk through the **D1 / "Reading B" / distinct-taxi supply-science results** | **NOT RAISED as a results walk-through** | The supply-science numbers surfaced **only** as a figure-clarity issue (the "3× service per unit demand" ratio, [1:04:57]–[1:13:39]), and Dr. Zhang cut the numeric drill-down short: [1:09:37] "if we're… talking about all the details in such a way, it's gonna take us like forever." No walk-through of supply_tier2 / total_tier2 / distinct-taxi findings. |
| (iv) | The **teaser-figure caption** | **DISCUSSED** | Figure 1 is the teaser. Decisions: keep captions **concise** [0:16:47], move key info (fairness score / 3× gap) **into the figure** rather than the caption [0:53:37]/[1:04:57]. Teaser "doesn't have to be rigorous" [1:13:42]. |
| (v) | **Citation-verification status (FairGAN / DECAF)** | **NOT RAISED by name** | Neither FairGAN nor DECAF was mentioned. Only **general** citation guidance was given: ≥2 per grouped class [0:39:12] and recency (2025–2026) [0:32:33]. |
| (vi) | **Anonymity / PII checks** | **NOT RAISED as a PII sweep** | Anonymity came up only as (a) the **anonymous submission format** (documentclass anonymous/review) and (b) the **anonymous GitHub link** [0:51:43]. No manuscript PII/de-anonymization check was discussed. |

**Summary — NOT raised:** (i) argument-triage ratification *(by name; general strategy endorsed)*, (iii) D1/Reading-B/distinct-taxi results walk-through, (v) FairGAN/DECAF citation verification, (vi) PII/anonymity check. **Raised:** (ii) color scheme *(critiqued, not approved)*, (iv) teaser caption.

---

## E. CORRECTED PLAUD CHECKLIST

Reproduces the discussion-summary "Plan" rows with **corrected owner / date / status** and what the transcript actually supports. **Every "Laura" → Robert.** Dates of **2026-07-30** fall *after* the 07-27 deadline and are wrong for any pre-submission task (corrected to "pre-submission"); the one exception is the deferred reproducibility doc, where a post-deadline date is consistent.

**Topic 1 — Narrative / Structure / Positioning**

| Plaud row (owner / date) | Corrected owner / date / status | Transcript support |
|---|---|---|
| Rewrite abstract + intro, problem-driven — *Dr. Zhang, Laura* (07-26) | **Robert (lead) + Dr. Zhang (optional draft); pre-submission; DECIDED** | D2/D3/D27. Dr. Zhang's rewrite is **conditional** ("if I have time") and optional to adopt [0:52:42]. |
| Rename "task" → "Problem Formulation" — *Speaker 3* (07-30) | **Robert; pre-submission (07-30 is post-deadline, wrong); DECIDED** | D6 [1:25:21]. |
| Add "Challenges" subsection — *Speaker 3* (07-30) | **Robert; pre-submission (07-30 wrong); DECIDED** | D4 [1:26:22]. |
| Shrink Related Work to half-column, full → appendix — *[unassigned]* | **Robert; pre-submission; DECIDED** | D7 [1:20:02]. |
| Move derivations (R²_demo) + grid-cell detail → appendix — *Speaker 3* (07-30) | **Robert; pre-submission (07-30 wrong); DECIDED** | D8/D9 [1:24:14]/[1:29:27]. |
| Remove duplicated statements — *Laura* (07-26) | **Robert; pre-submission; DECIDED** | D10 [1:37:28]. |
| Adjust formatting (bulleted lists) — *Laura* (07-26) | **Robert; pre-submission; DECIDED** | A17 [1:40:30] (Robert volunteers). |
| Complete rewrite + finalize for submission by **Sunday 2026-07-26** — *Laura, Dr. Zhang* | **Robert (lead) + Dr. Zhang (support); Sunday 2026-07-26; DECIDED** | D24 [1:55:52]/[1:53:51]. **Date correct** (07-26 = Sunday). |

**Topic 2 — Figure Readability**

| Plaud row | Corrected | Support |
|---|---|---|
| Revise Fig 1 self-explanatory (map bg, legends, icons) — *Speaker 3* | **Robert; unstated; DECIDED** | D11 [1:18:40]. |
| Fig 1 constant taxis/passengers — *Speaker 3* | **Robert; unstated; DECIDED** | D12 [1:18:40]. |
| Add pre-editing fairness score + "3× gap" in Fig 1 — *[unassigned]* | **Robert; unstated; DECIDED** | D13 [1:04:57]. |
| Enlarge figure text — *Speaker 3* | **Robert; unstated; DECIDED** | D14 [0:15:44]. |
| Reduce spacing/gaps — *Speaker 3* | **Robert; unstated; DECIDED** | A2 [0:13:31]. |
| Shorten captions — *[unassigned]* | **Robert; unstated; DECIDED** | D14 [0:16:47]. |

**Topic 3 — Terminology / Language**

| Plaud row (owner / date) | Corrected | Support |
|---|---|---|
| Replace "realism" → "fidelity" — *Laura* (07-26) | **Robert; unstated; OPEN** (not clean-DECIDED) | D17 — Robert flags Fidelity-A/B conflict [1:49:56]/[1:52:14]; Dr. Zhang recommends against "realism" [1:41:47] but tension unresolved. |
| Replace vague "rebalancing" w/ precise terms; name the related work — *Laura* (07-26) | **Robert; unstated; DECIDED** | D16 [1:46:49]/[1:49:18]. |
| Manual rewrite-first workflow, AI for refinement — *Laura* (07-26) | **Robert; ongoing practice (not a dated deliverable); DECIDED** | D18 [1:45:36]. |

**Topic 4 — Technical / Formatting**

| Plaud row (owner / date) | Corrected | Support |
|---|---|---|
| Add anonymous GitHub link in intro — *Dr. Zhang* | **Robert (creates repo/link; link may land via Dr. Zhang's intro draft); pre-submission; DECIDED** | D20 [0:51:43]/[1:59:06]. Owner-as-Dr.-Zhang is only partly right — repo creation is Robert's. |
| Prepare data-availability statement (code available, raw data not) — *Laura* (07-26) | **Robert; pre-submission; DECIDED** | D19/D21 [1:57:06]/[1:59:17]. |
| Defer detailed reproducibility documentation until after submission — *Laura* (07-30) | **Robert; after 07-27 (07-30 is consistent here); DEFERRED-POST-SUBMISSION** | D23 [1:58:38]. **The one row where a post-deadline date is correct.** |

*(Plaud "AI Suggestions" rows are advisory only and are folded into the OPEN items D5, D17, D25 above; they are not attributed actions.)*

---

## FINDINGS & DISCREPANCIES vs the Plaud summaries

1. **"~11 pages" is not the current draft.** Plaud auto-summary (line 53) says "the manuscript initially exceeds the eight-page limit (~11 pages)." Transcript: the 11-page figure is **Robert's pre-compression draft** ([0:48:44]); the **current** draft is **just under 8 pages** main content with ~3–4 lines of slack ([0:17:35]/[0:47:24]). This resolves the prompt's flagged "~11 pages" concern — it is a drafting-history artifact, **not** a version/template discrepancy in the live document.
2. **Hat-matrix ref [13].** Plaud auto-summary (line 45) says "retain reference to the seminal hat-matrix work (ref [13]) **for reviewer context**" — implying it stays in the main body. Transcript: Dr. Zhang directs the **entire F_demo derivation to the appendix** [1:24:14]; her "It's okay" at [1:22:40] about losing ref [13] from the body is **ambiguous** and, in context, the ref **travels to the appendix with the derivation**. Treat "keep [13] in main text" as **unsupported** — main body keeps only `1 − R²_demo` + meaning + an appendix pointer.
3. **Owner artifacts corrected.** Plaud discussion-summary lists **"Laura"** as an owner on 7 rows and attributes 4 rows to dates of **2026-07-30 (after the deadline)**. Laura = Robert; the 07-30 dates are wrong for all pre-submission tasks (corrected to pre-submission). Only the **deferred reproducibility doc** legitimately falls after the deadline.
4. **Deadline hour unverified.** Plaud says "submit by Sunday; verify cutoff." Transcript shows an unresolved **AM/PM conflict** (Robert reads OpenReview as Monday "11:59 **a.m.**" but recalls the abstract as "11:59 **p.m.**"). **Do not record 23:59 as confirmed** — A20 must verify.
5. **"Add anonymous GitHub link — Dr. Zhang" is only half right.** The link *placement* may ride along with Dr. Zhang's optional intro draft, but **repo/link creation is Robert's** (unassigned explicitly; implied).
6. **Caption guidance reversed within the meeting.** An early remark to "not modify the caption… leave it as is" [0:19:16] is **superseded** by the later, firmer decision to shorten captions and push key info into the figure [0:16:47]/[0:53:37]. Only the later decision (D14/D13) stands.
7. **"Katie" / "Manuel" / "Tim"** are non-attendees in the transcript: "Katie" ≈ mis-transcribed "KDD" (template source); "Manuel" = a person Dr. Zhang left to meet; "Tim" = an email reference. None own actions.
8. **Positioning shift is real (D28).** The paper's scope is explicitly narrowed to **data augmentation for imitation learning / behavior cloning** (fairness surviving via upweighting; extension = future work). Robert frames it as a "dual finding" rather than a reduced claim, and Dr. Zhang agrees the imitation-learning-specific framing is the claim.

## OPEN QUESTIONS (carry forward)

- **D17** — realism vs fidelity in the intro: unresolved (Fidelity-A/B specificity vs general usage).
- **D25/A20** — authoritative deadline hour + timezone (AM/PM; AoE vs local).
- **D5/A9** — motivating-example mechanism: recent *news* vs *early experimental result* placed in Figure 1.
- **A1 handoff** — where the corrected template lives and how Robert obtains it (platform never named).
- **A18 ownership** — who creates the anonymous repo (implied Robert; never explicitly assigned).
