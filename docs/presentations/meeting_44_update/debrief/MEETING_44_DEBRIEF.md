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

### 2.2 Template (corrected per Robert, 2026-07-23 post-debrief)
- The template she reviewed surfaced a **CCS-concepts vs keywords** problem, and she has
  already fixed the template in her copy on **Overleaf** (Robert's clarification: she views
  and will edit there — the earlier "location never named" blocker is dissolved).
- **Robert's directive (authoritative):** in the upcoming edit, **conform to Dr. Zhang's
  direction to NOT use the `\keywords{...}` block**, and **double-check compliance with the
  KDD template standards**. Note: the Plaud summary claims the corrected template *adds* a
  keywords section — Robert's account (he attended) supersedes the machine summary;
  reconcile against her actual Overleaf template when adopting.
- Local `main.tex` currently has `\keywords{...}` and no `CCSXML`/`\ccsdesc` block → the
  concrete local action is to remove/adjust the keywords block to match her template.

### 2.3 Data & code availability (D19 — refined by Robert post-debrief)
- **Raw data: releasable ONLY IF 100% anonymous** (Robert's reading of the discussion; this
  refines the flat "not released" in the meeting notes). Default posture remains no-release;
  any release path requires complete anonymization. Source: Dr. Yanhua Li.
- **In-paper caution (new):** how the paper references the data must not leak identifying
  information (collection/provider specifics, uncleaned coordinates, anything traceable) —
  fold this into the PII pass.
- Ship **anonymized data + code via an anonymous GitHub link in the introduction**;
  **empty repo at submission is acceptable**; artifact pledge proceeds on code availability.
  Repo/link creation implicitly Robert's.
- Data-availability statement: minimal — code shared via GitHub. Detailed reproducibility
  documentation **deferred until after submission** (PAPER/REPRODUCIBILITY.md remains our
  internal seed). The Plaud action item "document licensing constraints" **inverts her
  actual instruction** (don't dwell on it in-paper) — disregard it.

---

## 3. What she reviewed — CORRECTED (Robert, 2026-07-23 post-debrief; supersedes the first reading)

**Dr. Zhang WAS working with the current paper content.** Robert transferred the full
updated draft to Overleaf before the meeting; Overleaf is where she views it and where she
plans to make her own edits. The debrief's initial "broken, stale compile" reading is
**RETRACTED**, and every inference built on it is void: **no complaint may be discounted as
a version artifact — all feedback binds against the current text.** The in-meeting
rendering trouble on her side (figures failing with a package conflict [0:12:15], working
from screenshots, not finding Figure 2 on screen) was an environment/navigation hiccup
during the call, not evidence of stale content.

Reinterpretations that follow:

- **"It's just a task" [1:25:21] targets the CURRENT text.** The §3 heading already reads
  "Problem Formulation" (`03_methodology.tex:3`), so her critique lands on the
  `\textbf{Task.}` paragraph run-in (l.29) and on the formulation's lack of rigor. The
  substantive ask is fully open: rename the run-in and add explicit
  trajectory/state/action/reward definitions [1:32:12].
- **The teaser caption she called overly detailed is the already-trimmed 07-22 version** —
  her bar ("essential info only" [0:17:24]) requires cutting further.
- **Whitespace, in-figure text size, spacing, and figure-clarity complaints are about the
  real, current figures.** All stand as actionable now; nothing is deferred to a re-check.
- **Her appendix policy extends (not duplicates) the 07-22 argument-triage moves**: she saw
  the draft with those moves already made and still wants more moved (F_demo derivation,
  grid-cell config, full related work).
- Unchanged attribution facts (independent of what she viewed): "~11 pages" was Robert
  describing his own **pre-compression** draft [0:48:44], not her page count — content is
  under 8pp and Robert told her so live [0:17:35]; "Laura" (even at [1:39:40]) = Robert;
  "Katie" = "KDD"; only two attendees.

**Overleaf is the coordination surface**: her corrected template and her possible new
§1/abstract .tex will appear there; local git remains our working source of truth, with a
merge/reconcile event planned ~1–2 days out (§6.1, §8).

---

## 4. The mandate — seven planks

**Governing style rule (Robert, 2026-07-23, binding for the whole rewrite):** every
statement self-explanatory; every section self-contained (assume the reviewer scans in
mid-paper, has not read earlier sections, and will not read the cited works); prioritize
domain-specific language and avoid ALL unnecessary generalities, cleverness, and patterns
that force the reader to self-interpret; the story carries the minimum detail necessary to
introduce and validate the method, structured as Dr. Zhang instructed.

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
   [0:50:46]; **F_demo derivation → appendix**, main text keeps `1−R²_demo` + meaning +
   pointer **+ the bare hat-matrix citation** (`hoaglinwelsch1978` stays in body per
   Robert's §6.6 call; the derivation content moves per [1:23:32]–[1:24:14]);
   grid-cell/implementation config → appendix; remove §3/§4 duplication (0.01° grid
   stated in both).
7. **Terminology + de-AI-ification**: her ask was "realism" → **"fidelity"** in technical
   contexts [1:41:47] while keeping paper-specific Fidelity-A/B distinct [1:51:04]
   (execution refined by Robert's §6.3 answer: neither word is forced — the concept must
   be self-explanatory wherever introduced, and "realism" is optional); replace vague
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
   imitation-model layer stays (her removal suggestion was withdrawn — core to the transfer
   claim). (The spec's "her render vs real issues" caveat is void per §3: all whitespace/
   clarity complaints are about the current figures and stand.)

---

## 6. Open decisions — ANSWERED (Robert, 2026-07-23 post-debrief)

1. **§1/abstract ownership → parallel tracks.** Dr. Zhang makes her own edits in a separate
   environment (Overleaf); we also write a new §1/abstract ourselves; the two versions get
   merged/reconciled before the deadline (Robert: "probably in a day or two"). Treat the
   reconcile as a real scheduled event (3-way against the pre-rewrite base; editor-conflict
   recipe applies).
2. **Top-line scope → dual finding retained, scoped per claim (working position).** Robert
   leans dual-finding and asked for a defensibility check; Fable's recommendation (recorded
   as the working position, confirm during the rewrite): keep the dual finding — it IS the
   argument (the vanilla-BC null shows data-level gains alone don't survive training; the
   upweighting recovery completes it) — but scope each claim honestly: data-level claims
   are learner-agnostic (the corpus is measured before any training), while the transfer
   claim is stated as "demonstrated with behavior cloning" wherever made, with one
   future-work clause beyond BC (which also satisfies her D28). Rationale: an unscoped
   IL-wide transfer claim is attackable, especially with GAIL as the motivating example and
   no GAIL training run in the paper.
3. **Fidelity vs realism → neither word is forced.** Wherever the concept appears, the
   sentence must be self-explanatory so the reader needs no prior knowledge of the term;
   "realism" may be used or dropped ("it's not a requirement to use it"). Her Overleaf
   edits may still impose "fidelity" — reconcile at merge. Fidelity-A/B stay distinct as
   the paper-specific instruments.
4. **Standing defenses → balanced concession.** Keep terms that are genuinely the most
   reasonable expression ("in-processing" probably stays) but give every such term enough
   in-place context to be understood cold — Dr. Zhang's confusion came from missing
   context, not the word itself. Binding assumption: the reader has NOT read the paper
   start-to-finish and will NOT read the cited works; every section/paragraph must be
   self-contained for a scanning reader.
5. **Fig-1 early evidence → deferred.** Re-pose the question concretely when the Figure-1
   edit starts (with FIGURE_REVISION_SPEC on the table).
6. **Hat-matrix citation → STAYS in the main body** (bare cite next to `1−R²_demo`); the
   derivation content still moves to the appendix.
7. **Challenges → C1–C4 approved, C5 added.** C1 fidelity-under-editing, C2 demand-adjusted
   fairness target, C3 level-up-not-down, C4 survival-through-training (NARRATIVE_STRATEGY
   §3.1), plus **C5: human-derived demonstration data is scarce, so filtering out the
   unfair data is not an option** (scarcity forces editing over filtering/regeneration).
   Additionally: sweep the current §1/§2 for further implicit challenges worth promoting,
   and keep the challenge set consistent everywhere it appears (intro beat, methodology
   mapping, contributions).
8. **Captions → essentials-only, confirmed.** No Zhang check needed; both agree. Template
   caption *formatting* still follows her corrected template.

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
- **Post-debrief corrections from Robert (2026-07-23)** — supersede the corresponding
  first-pass readings above and in `analysis/`:
  1. **Version**: the "broken, stale compile" reading is RETRACTED — she reviewed the
     CURRENT content on Overleaf (Robert transferred it pre-meeting); all feedback binds
     against the current text (§3).
  2. **Template**: do NOT use the `\keywords{...}` block (Robert's account supersedes the
     Plaud summary's "corrected template adds keywords"); verify against KDD template
     standards (§2.2).
  3. **Raw data**: anonymity-conditional release (OK if 100% anonymous), not a flat no;
     plus a new in-paper data-reference PII caution (§2.3).
  4. **Hat-matrix**: the bare citation stays in the main body (Robert's call); only the
     derivation content moves (§4 plank 6, §6.6).

---

## 8. Schedule (deadline-driven; today = Thu 07-23)

| When | What |
|---|---|
| Thu eve ✅ | Debrief delivered; Robert answered §6; corrections folded into this doc |
| Fri | Our §1/abstract rewrite (problem-driven, C1–C5, one-¶ approach, anon code link) + §2 half-column/appendix split + citation additions (≥2/class, 2025–26; checklist rows same session). Dr. Zhang works in parallel on Overleaf |
| Sat | Figure 1 rebuild (conservation, district labels, in-figure numbers) + §3 reshape (derivation → appendix keeping the bare hat-matrix cite, grid-cell dedup, `\textbf{Task.}` run-in fix, rigorous state/action/reward definitions) + self-containment/terminology sweep |
| Sat/Sun | **Merge/reconcile with Dr. Zhang's parallel edits** (her §1/abstract .tex + her template; 3-way vs pre-rewrite base) |
| Sun | Full gates, page check, PII pass **including the data-reference leak check**, template compliance (no `\keywords{...}`), anonymous repo link live, **submit well before AoE midnight** |

Working rules unchanged: protected register (headline numbers + disclosures survive somewhere),
era discipline, citation-checklist same-session rule, explicit staging, surface-never-smooth.
