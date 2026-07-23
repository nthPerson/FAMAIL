# MEETING 44 DEBRIEF — the simplification mandate

**Meeting:** 2026-07-23, ~2h00m, Robert + Dr. Xin Zhang (full-draft review, KDD 2027 Cycle 1).
**Produced:** 2026-07-23 by 4 parallel Opus transcript analysts + orchestrator synthesis and
web verification. This file is the synthesis; the four detailed reports are in `analysis/`
and the primary sources (transcript, Plaud summaries, her pen-markup photo) sit beside it.

| Source | File |
|---|---|
| Cleaned transcript (341 utterances, timestamped) | `meeting_44_transcript.txt` (raw JSON: `meeting_44_transcript_raw.txt`) |
| Plaud auto-summary / discussion summary / highlights | `plaud_summary.md`, `plaud_discussion_summary_raw.txt`, `plaud_highlights_raw.txt` |
| Dr. Zhang's pen markup on Figure 1 | `plaud_marked_photo.png` |
| Itemized feedback ledger (54 items F1–F54) | `analysis/FEEDBACK_LEDGER.md` |
| Decisions (28) + actions (26) + logistics | `analysis/DECISIONS_AND_ACTIONS.md` |
| Narrative mandate + gap analysis + target outline | `analysis/NARRATIVE_STRATEGY.md` |
| Element-level figure/formatting spec | `analysis/FIGURE_REVISION_SPEC.md` |

---

## 1. Bottom line

Dr. Zhang's verdict is structural, not cosmetic — and explicitly not about length or the science:

> [0:21:58] "the current version is a kind of ... good documentation of what has been done
> instead of ... a good way of telling the story that what the problem we want to solve and
> what strategies we have proposed to solve this problem."
> [0:39:52] "when I'm reading this paper, it's really taking me a lot of time in terms of understanding."
> [1:36:29] "there is a difference about a tech report versus a paper ... when it's a paper, it has to be problem driven."
> [1:43:36] "I know how the KDD community usually read papers, and I think it's going to be
> difficult to work if we are submitting in the current way."

Scale: rewrite the **introduction** (and abstract) around a problem-driven story, restructure
the **related work** to half a column, reshape the **methodology front matter** (Problem
Formulation + derivation policy), rebuild **Figure 1**, and run a **terminology/precision
sweep**. She is emphatic the results are sound and that **"length is the least important
thing"** [1:28:07] — the ≤8pp submission limit is the only hard constraint; being over
temporarily during the rewrite is fine.

---

## 2. Hard logistics (safety-critical items first)

### 2.1 Deadline — CORRECTED, verified against the official CFP (post-meeting web check)
- Official [KDD 2027 Research Track CFP](https://kdd2027.kdd.org/research-track-call-for-papers/):
  Cycle 1 **paper deadline July 26, 2026, AoE** — "strict and no extensions, regardless of
  circumstances."
- **July 26 AoE ends Monday 2026-07-27 ~04:59 AM Pacific.** The project's long-standing
  "Mon 07-27 23:59" belief was wrong by ~19 hours. In the meeting Robert read OpenReview as
  "11:59 **AM** Monday" (local rendering of the same AoE instant); the exact hour was left
  unverified in-meeting — this check closes that action (A20).
- Agreed plan: **submit Sunday 2026-07-26** (D24) — treat Sunday evening as the real deadline.

### 2.2 Template
- The compile Dr. Zhang reviewed showed a **CCS-concepts block; the corrected template
  replaces it with keywords** (header-metadata swap, same acmart sigconf/review/anonymous
  class — no layout/page-count change indicated). She made the edit in **her own copy**, but
  **where that copy lives was never named** (not said to be Overleaf/email/repo). → Robert
  must ask her where it is, or reconcile directly. Note: local `main.tex` (identical to
  origin's) already has `\keywords{...}` and **no** `CCSXML`/`\ccsdesc` block — reconcile
  before assuming any local change is needed.

### 2.3 Data & code availability (D19)
- **Raw data is NOT released** (source: Dr. Yanhua Li) — and she said not to dwell on it in
  the paper. The Plaud action item "document licensing constraints" **inverts her actual
  instruction** — disregard it.
- Ship **anonymized data + code via an anonymous GitHub link in the introduction**;
  **empty repo at submission is acceptable**; artifact pledge proceeds on code availability.
  Repo/link creation implicitly Robert's.
- Data-availability statement: minimal — code shared via GitHub. Detailed reproducibility
  documentation **deferred until after submission** (PAPER/REPRODUCIBILITY.md remains our
  internal seed).

---

## 3. What she actually reviewed (read before acting on any figure/layout complaint)

- Her PDF was a **broken and stale render**: figures failed with a package conflict, she
  worked partly from **screenshots**, and she **could not locate Figure 2**. Whitespace and
  figure-placement complaints should be re-checked against a clean compile of the current
  draft before spending effort.
- Her copy predates the last ~23 local commits (origin sits at `cdf3e4d`, cut-wave W8).
  Already done locally, which her copy lacked:

| Her ask | Local status |
|---|---|
| "Task" → "Problem Formulation" [1:25:21] | Heading already renamed (`03_methodology.tex:3`); **residual `\textbf{Task.}` run-in at l.29 must go**; rigorous state/action/reward definitions only partly present |
| ≤8 pages | Content already under 8pp (References ~95% down p8) — Robert told her so live [0:17:35]; "~11 pages" was Robert describing his **pre-compression** draft [0:48:44], not her page count |
| Shorter teaser caption | One trim already landed 07-22; her bar ("essential info only" [0:17:24]) is stricter — trim further |
| Move detail to appendix | Argument-triage moves (allocation boundary, SF detail, recount mechanics) already landed 07-22; her policy extends this (F_demo derivation, grid-cell config) |

- Misc transcription artifacts: "Laura" (even at [1:39:40]) = Robert; "Katie" = "KDD"; only
  two attendees.

---

## 4. The mandate — seven planks

1. **Fairness-vs-fidelity trade-off is the spine.** Existing methods buy fairness at the
   cost of fidelity; FATE achieves both. "the fairness versus realism tradeoff is a major
   thing we want to solve. So that is kind of core of this problem" [0:38:14]. Difficulty is
   anchored to model-level regularization (`zheng2023`).
2. **Problem-driven story, reviewer-legible from §1 alone** [0:49:07]. KDD reviewers skim:
   problem + why prior work fails must land in the introduction. Kill the "third position"
   framing in favor of "here is why a new approach is necessary."
3. **A Challenges beat in §1** [1:25:48]: what makes the problem hard = the same reasons
   existing works fail; the methodology should then be organized to answer each challenge
   ("optimize your methodology against your challenges" [1:39:40]).
4. **Motivating example + early evidence in Figure 1**: an imitation learner (GAIL)
   inheriting/amplifying unfairness [0:51:43]; show the **pre-editing fairness score** in
   Figure 1 and, if feasible, early experimental evidence that existing approaches can't fix
   it (news story = fallback) [0:53:04].
5. **Precise, literature-named language; ≥2 citations per approach class, with recent
   (2025–26) works** [0:32:37]; split the data side into **trajectory-editing vs
   trajectory-generation** [0:33:57]. Grep-confirmed: in-processing and rebalancing classes
   currently ride on one citation each. Her specific confusions: "in-processing methods,"
   "objective and training signal conflict," unexplained failure reasons; always name which
   distribution shifts and at what level (data / model / hyperparameter) the intervention acts.
6. **Ruthless main-text/appendix split**: §2 related work → **half a column**, full version
   to appendix [1:20:02]; §1 approach summary → **one paragraph** + anonymous code link
   [0:50:46]; **F_demo derivation (including the hat-matrix reference) → appendix**, main
   text keeps `1−R²_demo` + meaning + pointer (adjudicated below, §7); grid-cell/implementation
   config → appendix; remove §3/§4 duplication (0.01° grid stated in both).
7. **Terminology + de-AI-ification**: "realism" → **"fidelity"** in technical contexts
   [1:41:47] while keeping paper-specific Fidelity-A/B distinct [1:51:04]; replace vague
   GPT-flavored phrases ("rebalancing models," "data generation shifts the distribution");
   workflow reversed — **human-write-first, AI refine after** [0:19:38ff].

Full quote-anchored detail + a paragraph-level target outline for §1/§2:
`analysis/NARRATIVE_STRATEGY.md` Part 3.

---

## 5. Figures (element-level spec in `analysis/FIGURE_REVISION_SPEC.md`)

Top-line: **Figure 2 is the approved reference standard** [1:17:25]; Figure 1 must be rebuilt
in its style and stand alone (Robert's "read them in combination" was overruled).

1. Map background + explicit **"Advantaged district" / "Disadvantaged district" labels** on
   the rollout panels (her pen markup literally writes "A."/"D." on them) + duplicated legend.
2. **Conservation rule — currently VIOLATED**: left panel has 7 taxis/5 pickups, right has
   6/4. Totals must match across before/after so the story is *relocation, not removal*
   [1:05:32]; recast the "+ added presence" motif as *moved*.
3. **Fairness numbers inside the figure** (pre-/post-edit service value; the 3.0× gap must be
   visible from the glyph distribution — her worked example: 6 taxi/2 pickup advantaged vs
   1 taxi/1 pickup disadvantaged [1:12:04]) — stylized, "teaser need not be rigorous."
4. **In-figure text enlarged** to slightly-smaller-than-caption (kill ~5pt `\tiny`), glyphs
   larger, internal gaps tighter — both figures.
5. **One ratio direction, one metric name** across text + figures (§3.2 currently defines
   both supply/demand and demand/supply — a real text inconsistency, not just figure polish).
6. Captions: essential info only; every figure **and table** `\ref`'d in text (tables were
   Robert's own flagged risk [0:19:02]).
7. Palette: keep the existing CVD/grayscale-verified blue-amber system (`figink/figaccent/
   figtrim/figtint*`) — the colors were **critiqued-not-approved** in the sense that clarity
   complaints stand, but no directive to change colors was given; don't discard the verified
   palette while rebuilding.
8. Conflicts to resolve deliberately (C1–C9 in the spec): conservation vs visible-3×;
   in-figure numbers vs compactness; map background vs the logged map-free design decision;
   her broken render vs real whitespace issues; imitation-model layer stays (her removal
   suggestion was withdrawn — core to the transfer claim).

---

## 6. Open decisions Robert must make (genuine disagreements or unassigned choices)

1. **Who writes the new §1/abstract**: she said she may draft her own rewrite in a new .tex
   file [1:52:42]/[2:00:03]. Coordinate before either side writes — merge risk (and the
   editor-conflict recipe applies if both touch the same files).
2. **Top-line scope claim**: her repeated narrowing to "data augmentation for imitation
   learning / behavior cloning" (beyond-BC = future work, D28) vs Robert's "dual finding"
   framing. Abstract/intro currently claim the general version.
3. **"Fidelity" vs "realism"**: her new rule (fidelity) collides with her own earlier
   literature-backed "realism" rule; Robert was mid-defense when the topic closed. OPEN
   (F44) — decide one, sweep consistently, keep Fidelity-A/B distinct.
4. **Robert's standing defenses** (PUSHED-BACK-STANDS or OPEN in the ledger): "in-processing"
   as a term, the "distribution shift" claim, intentional §3/§4 duplication. Decide which to
   concede in the rewrite vs keep with better wording.
5. **Which early evidence goes into Figure 1** (GAIL number? news fallback?) — and whether it
   fits without wrecking the teaser's simplicity.
6. **Hat-matrix bare cite**: mandate sends the derivation + ref to the appendix; keeping a
   bare `\cite{hoaglinwelsch1978}` in body prose costs ~0 lines if Robert wants reviewer
   cover — style call.
7. **Challenges content**: the C1–C4 candidate set in NARRATIVE_STRATEGY.md Part 3 is
   inference — Robert picks/words the actual challenges.
8. **Caption interpretation** (low risk): shorten caption *content* to essentials [0:17:24]
   while following template caption *formatting* [0:19:16] — confirm reading with her only if
   convenient.

---

## 7. Corrections log (what the record now says vs earlier beliefs/sources)

- **Deadline**: SUN 2026-07-26 AoE (official, strict) — not Mon 07-27 23:59. §2.1.
- **Hat-matrix ref**: transcript [1:23:32]–[1:24:14] sends the whole derivation including
  the seminal ref to the appendix; Plaud's "retain ref [13] in main" and one analyst's
  same reading are unsupported ("It's okay" at [1:22:40] = it's okay to lose it from body).
- **"~11 pages"**: Robert's description of his pre-compression draft, not her page count;
  no 11-page compile was reviewed.
- **Plaud hallucinations**: "document licensing constraints" (inverts her instruction);
  owner "Laura" + all per-row dates in the discussion summary (incl. impossible post-deadline
  2026-07-30 dates) are machine-generated; "background map" idea originated with Robert
  (she endorsed Figure-2 style).
- **Colors**: critiqued for clarity, not rejected; no directive to change the palette.
- **Prior-agenda items NOT raised** in the meeting: argument-triage ratification by name,
  D1/Reading-B science walk-through (she cut the numbers drill-down short), FairGAN/DECAF
  citation verification, PII/anonymity checks. These remain on our internal checklist.

---

## 8. Proposed schedule (deadline-driven; today = Thu 07-23)

| When | What |
|---|---|
| Thu eve | Robert reads this debrief + answers §6 decisions; asks Dr. Zhang where the corrected template lives; watch for her §1 .tex |
| Fri | §1/abstract rewrite (problem-driven, Challenges, one-¶ approach, anon link) + §2 half-column/appendix split; citation additions (≥2/class, 2025–26) |
| Sat | Figure 1 rebuild (conservation, labels, in-figure numbers) + methodology reshape (derivation → appendix, dedup, Task run-in) + terminology sweep (fidelity, precision fixes) |
| Sun | Full gates, page check, PII/anonymity re-check, template reconciliation, anonymous repo link live, **submit** (well before AoE midnight) |

Working rules unchanged: protected register (headline numbers + disclosures survive somewhere),
era discipline, citation-checklist same-session rule, explicit staging, surface-never-smooth.
