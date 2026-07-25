# TASK BOARD — restructure sprint (draft to Zhang: Saturday night 2026-07-25)

Update your task's checkbox and the Status column when you finish something. Statuses:
`todo · in-progress · blocked(<on what>) · review · DONE`. Every prose task obeys:
gates before commit; moved blocks carry their `% src:`/`% lint-allow:` comments; cite
changes update CITATION_PRIORITY_CHECKLIST.md (rows UNTICKED); era numbers + protected
register per `CONTEXT.md`; wording constraints per `meeting/analysis_C_claims.md` §5
(the do-not-claim list) as adjudicated in `MEETING_DIGEST.md`.

⚠ Working-tree note: `paper/sections/04_experiments.tex` carries Robert's uncommitted
wording tweak ("All edits are made using…", old line commented). PRESERVE it; fold it
into T5's first commit.

## Lane 0 — gate decisions (Robert; everything else keys off these)

- [x] Q1 workspace — RULED 07-25: archive dir + main. Archive created:
      `paper-pre-restructure-2026-07-25/` (+ README_ARCHIVE.md) — STATUS: DONE
- [x] Q2 challenge set — RULED 07-25: FIVE, 1:1 per ADJ-3 mapping — STATUS: DONE
- [x] Q3 Fig-2 layout — RULED 07-25: Robert's three-phase design (Attribute w/ three
      colored trajectory groups / Trim+Lift / Upweight); spec updated — STATUS: DONE
- [x] Q4 anon-link scope — RULED 07-25: CODE + DATA wording (her full claim); the
      Sunday anonymity pass is now load-bearing; dataset (or pointer) must be in the
      anon repo by submission — STATUS: DONE
- [x] A0 source assets: `teasing.png` DELIVERED 07-25 (2750×1540 RGBA — T-F2
      unblocked). The two PDFs remain nice-to-have (digests in this hub cover them).
      — STATUS: DONE (PNG)
- [ ] A1 Robert↔Zhang: confirm who clicks submit Sunday ([57:24] ambiguity) — STATUS: open

## Lane 1 — prose restructure (sequential; one implementer at a time; each task = gates-green commit)

- [x] T1 `00_abstract.tex` — DONE 07-24 (f7b34df): her abstract near-verbatim; repair
      1 = "All edits satisfy spatial and continuity constraints, and a frozen
      driver-identity discriminator scores behavioral fidelity in the editing
      objective"; repair 2 = "greatest estimated aggregate fairness impact"; all
      claims verified; both builds green (standalone build had a pre-existing
      clock-skew latexmk artifact — delete its aux state if it recurs).
      ⚠ WATCH ITEMS for downstream tasks: (a) "attributes corpus-level disparity to
      influential trajectories" is abstract-level compression — §2/§3 keep the exact
      partition at ACTIVE-UNIT level, never a per-trajectory fairness score (D3);
      (b) "direct additional resources toward under-served areas" holds only as
      redirection under conservation — body keeps that explicit (T4); (c) her
      abstract says "real-world HSTD", dropping the explicit two-city naming —
      ⚖ RULED 07-24: Robert wants both cities named; T9 folds in the add (see T9).
      — STATUS: DONE
- [ ] T2 `01_introduction.tex` — rebuild on her intro's logic (accessible register,
      [36:50]): hook ¶s; brief PROSE mention of challenges (no itemize; forward-ref
      §2); spine wording per C-1 proposal (NOT her literal spoken claim); FairGAN/
      model-side differentiation per C-3; contributions list rebuilt (collective-
      fairness contribution leads, C-refs point to §2 labels, C D15 "enforced by" →
      constraint wording); repair her broken cites ([35?] → zhang2019cgail+zhang2022cgail
      or +feng2020, implementer judges context; [?] anon link → existing footnote,
      CODE + DATA wording per Q4 ruling); decide fate of the 3.0× hook sentence (keep if it fits her flow —
      it is accurate and sourced); Fig-1 swap per T-F2. New cites → checklist rows.
      — STATUS: todo (after T3 lands labels)
- [ ] T3 NEW `02_overview.tex` — move current §3.1 content here (label `sec:overview`;
      moved labels travel with content): definitions with cGAIL-style **boldfaced
      terms** → problem definition → five-challenge block (stacked bold lead-ins, NO
      itemize env; C1 budget / C2 fidelity / C3 target / C4 level-up / C5 training).
      main.tex: insert 02_overview input; move 02_related_work input after 04.
      SCOPE CHANGE 07-24: NO file renames this sprint (02_related_work.tex and
      05_conclusion.tex keep their names; filenames lag section order — post-deadline
      cleanup item). Brief: `briefs/T3-overview.md`. — STATUS: brief ready
- [ ] T4 `03_methodology.tex` — leading ¶ (names FATE, two stages, refs
      Figure~\ref{fig:overview}, maps components → challenge labels); then five blocks
      per ADJ-3 mapping, each opening with its challenge: (1) collective fairness
      objective [current §3.2 reorganized: design requirements → why raw parity is
      wrong → residual → F_demo → why-useful → scalarization; caveats ¶ SURVIVES],
      (2) attribution under a budget [both mechanisms; k split described faithfully,
      C-6 wording; "not post-hoc" emphasis], (3) outcome-side vs resource-aware
      editing [trim mechanics; §3.3's limitation argument COMPRESSED here with the
      2,455-pickup fact + leveling-down analogy + endogeneity (protected register;
      overflow detail → appendix); lift with the 7 named elements; closing key
      distinction], (4) validity- and fidelity-constrained editing [K vs ε; constraint
      list; corrected 6-step pipeline per ALGORITHM_FACTS §Validity — NO fidelity
      gate], (5) edit-aware weighting [dilution problem; upweighting; controls
      forward-ref]. Keep labels sec:objective/sec:leveling/sec:editor/sec:downstream
      alive (aliases fine) — §4, §5, appendix reference them. — STATUS: todo
- [ ] T5 `04_experiments.tex` — leading ¶ (aims + research questions mapped to
      subsections; count Robert-approved in review); fold Robert's pending wording
      tweak; SF subsection re-framed as consolidated transferability block
      ("reproduces on a second city", never "generalizes"); trim duplicated cross-city
      prose (differences stay, made "more visually striking" [46:10] — e.g., the
      SZ-vs-SF contrast sentences juxtaposed); budget framing sentences (k as
      CONFIGURED budget; no k-sweep implication, C D16). — STATUS: todo
- [ ] T6 `02_related_work.tex` (file keeps its name; now inputted as §5) — position
      handled by T3's main.tex reorder; this task = content re-check: adjust any
      pointers whose prose assumed Related Work preceded the method; opening line
      tense/position touch; content otherwise stands (already M44-compressed).
      — STATUS: todo
- [ ] T7 `05_conclusion.tex` (file keeps its name; renders as §6) — spine vocabulary
      update (budgeted intervention, collective fairness); future-work gains the
      k-sweep sentence (D16); bounds ¶ SURVIVES verbatim. — STATUS: todo
- [ ] T8 `appendix.tex` — absorb any detail displaced from T4 (the "tricks" →
      appendix rule, C-12: correctness conditions keep one main-text clause each);
      verify all \ref targets after renumbering; App E related-work survey pointer
      still correct from new §5. — STATUS: todo
- [ ] T9 Integration pass — full gates; strict page check (`pdftotext -f 9 -l 9` —
      p9 should open with REFERENCES; if not, apply the meeting's space levers:
      stacked challenges, SZ/SF dedup, appendix overflow); render QA read of every
      changed page (the swallowed-sentence class); cross-ref sweep (no ?? in log);
      CITATION_PRIORITY_CHECKLIST coverage green.
      ⚖ RULED 07-24 (Robert): fold in the abstract's two-city add — name Shenzhen
      and San Francisco in the evaluation sentence of 00_abstract.tex (e.g.
      "…evaluate it on real-world HSTD from Shenzhen and San Francisco."); smallest
      change that names both cities; re-check the standalone abstract build too.
      Also verify Fig-1 page placement (T-M1 note) and Fig-1/Fig-2 render QA.
      — STATUS: todo

## Lane 2 — figures (parallel with Lane 1 from the start)

- [ ] T-F1 Fig-2 framework diagram per `figures/FIG2_FRAMEWORK_SPEC.md` ADOPTED
      LAYOUT (Robert's three-phase design, ruled 07-25): standalone TikZ + test
      harness in `paper/figures/figure-2/`; integrate as `fig:overview`; retired
      3-panel source stays in the dir, leaves main.tex. DELIVERABLE ALSO: standalone
      PDF for Robert to email Zhang EARLY (meeting A5 cheap-veto). — STATUS: ready
- [~] T-F2 Fig-1 swap — FOLDED INTO T2 (one writer per file: the intro rewrite owns
      the figure environment). Spec reference stands: `figures/FIG1_TEASER_SPEC.md`
      Phase 1. — STATUS: merged into T2
- [ ] T-F3 STRETCH (default SKIP per ADJ-1): real-data case-study figure for §4 from
      existing artifacts; only if T1–T9 + T-F1/F2 are done with time left; never
      relabel the schematic. — STATUS: parked

## Lane 3 — mechanics (independent; can run anytime)

- [x] T-M1 `main.tex` template blocks — DONE 07-24 (wave 1): `acmlicensed` renders
      both blocks with real KDD '27 venue line, no fake ISBN/DOI; gates green; 13pp
      unchanged; anonymity intact. ⚠ HANDOFF NOTE for T2/T9: the ACM block took
      Fig-1's page-1 col-2-top slot — the teaser now floats to p2. T2's intro
      rewrite + PNG swap re-decides page-1 layout; T9 verifies the teaser is back on
      p1 (or Robert accepts p2). Pre-existing acmart warning "ACM keywords are
      mandatory" (keywords retired for CCS 07-24) — benign under review class,
      re-check at camera-ready. — STATUS: DONE
- [ ] T-M2 Robert-owned, post-restructure: port to Overleaf main.tex + final-compile
      check there (different engine possible, [50:33]); email/ping Zhang on
      review-ready (A3). — STATUS: waits on T9

## Robert's standing submission-hygiene list (no Zhang mandate in the meeting; still his checklist)

- [ ] 9-row P0 citation pass in CITATION_PRIORITY_CHECKLIST.md (+ any rows added by
      T1/T2) — human ticks only
- [ ] Anonymous repo live + §1 footnote URL swap — Q4 ruling: repo must hold CODE and
      DATASET (or a pointer) by Sunday; the PII/anonymity pass gates the data half
- [ ] CCS concept_ids regenerated at dl.acm.org/ccs (main.tex TODO comment)
- [ ] Sunday PII/data-leak pass before submission

## Execution notes for the orchestrator

- Subagent-driven: fresh Opus implementer per task (explicit model param), task review
  after each, final whole-branch review before the Saturday hand-off. Lane-1 tasks are
  SEQUENTIAL (interlocking labels/refs); T-F1 and T-M1 may run parallel to Lane 1
  because file ownership is disjoint (figures/figure-2/*, main.tex header block) — but
  never two writers on one file; main.tex edits serialize (T3's \input reorder vs
  T-M1's header: T-M1 first, it is 2 lines).
- Task briefs point implementers at: CONTEXT.md, ALGORITHM_FACTS.md, MEETING_DIGEST.md,
  the relevant analysis file section, and the specific current .tex files. Not the
  whole hub.
- Commit style: `paper(restructure T<n>): <what>` — one gates-green commit per task;
  explicit staging; Robert pushes; Robert must Revert File in his editor after each
  commit he has the file open for (stale-buffer clobber happened twice on 07-24).
- Page budget: no number was spoken in the meeting, but the repo's strict-8pp target
  stands until Robert says otherwise; T9 owns the check.
