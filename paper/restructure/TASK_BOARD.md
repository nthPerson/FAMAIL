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

- [ ] Q1 workspace mechanics (archive dir / branch / both) — STATUS: asked 07-25
- [ ] Q2 challenge set (five 1:1 vs three-merged; ADJ-3 mapping) — STATUS: asked 07-25
- [ ] Q3 Fig-2 layout (Option A hybrid two-stage vs Option B three-band) — STATUS: asked 07-25
- [ ] Q4 anonymous-link sentence scope (code-only vs code+data) — STATUS: asked 07-25
- [ ] A0 Robert drops source assets into `paper/restructure/zhang/`:
      `Zhang_restructuring_email.pdf`, `Zhang_paper_revision.pdf`, `teasing.png`
      (the teaser PNG is REQUIRED for T-F2; the PDFs are nice-to-have since digests
      exist) — STATUS: requested
- [ ] A1 Robert↔Zhang: confirm who clicks submit Sunday ([57:24] ambiguity) — STATUS: open

## Lane 1 — prose restructure (sequential; one implementer at a time; each task = gates-green commit)

- [ ] T1 `00_abstract.tex` — adopt Zhang's abstract (ZHANG_DRAFT_DELTA §Authoritative)
      with repairs: fix the fidelity-gate wording ("verifies" → guardrail phrasing,
      C D15); verify every claim against ALGORITHM_FACTS; keep "distinct-taxi
      presence" tier language; no em-dash violations. Also check
      `kdd27-abstract-only.tex` still compiles (shares this file). — STATUS: todo
- [ ] T2 `01_introduction.tex` — rebuild on her intro's logic (accessible register,
      [36:50]): hook ¶s; brief PROSE mention of challenges (no itemize; forward-ref
      §2); spine wording per C-1 proposal (NOT her literal spoken claim); FairGAN/
      model-side differentiation per C-3; contributions list rebuilt (collective-
      fairness contribution leads, C-refs point to §2 labels, C D15 "enforced by" →
      constraint wording); repair her broken cites ([35?] → zhang2019cgail+zhang2022cgail
      or +feng2020, implementer judges context; [?] anon link → existing footnote,
      scope per Q4); decide fate of the 3.0× hook sentence (keep if it fits her flow —
      it is accurate and sourced); Fig-1 swap per T-F2. New cites → checklist rows.
      — STATUS: todo (after T3 lands labels)
- [ ] T3 NEW `02_overview.tex` — move current §3.1 content here (label `sec:overview`;
      keep sub-labels working): definitions with cGAIL-style **boldfaced terms** →
      problem definition (Task ¶) → challenge list per Q2 (stacked bold lead-ins in
      running text, NO itemize env, labels C1… referenced from §3 blocks). Strip/
      retarget the old forward refs (the §3.2/§3.4 pointers). Renumber section files:
      02_related_work.tex → 05_related_work.tex; new file order in main.tex
      (01, 02_overview, 03, 04, 05_related_work, 06_conclusion). — STATUS: todo
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
- [ ] T6 `05_related_work.tex` (renamed from 02) — position after Experiments; adjust
      any "below/§3.3" style pointers; opening line may need a tense/position touch;
      content otherwise stands (already M44-compressed). — STATUS: todo (mostly done
      inside T3's rename; this task is the content re-check)
- [ ] T7 `06_conclusion.tex` (renamed from 05_conclusion) — spine vocabulary update
      (budgeted intervention, collective fairness); future-work gains the k-sweep
      sentence (D16); bounds ¶ SURVIVES verbatim. — STATUS: todo
- [ ] T8 `appendix.tex` — absorb any detail displaced from T4 (the "tricks" →
      appendix rule, C-12: correctness conditions keep one main-text clause each);
      verify all \ref targets after renumbering; App E related-work survey pointer
      still correct from new §5. — STATUS: todo
- [ ] T9 Integration pass — full gates; strict page check (`pdftotext -f 9 -l 9` —
      p9 should open with REFERENCES; if not, apply the meeting's space levers:
      stacked challenges, SZ/SF dedup, appendix overflow); render QA read of every
      changed page (the swallowed-sentence class); cross-ref sweep (no ?? in log);
      CITATION_PRIORITY_CHECKLIST coverage green. — STATUS: todo

## Lane 2 — figures (parallel with Lane 1 from the start)

- [ ] T-F1 Fig-2 framework diagram per `figures/FIG2_FRAMEWORK_SPEC.md` (layout per
      Q3): standalone TikZ + test harness in `paper/figures/figure-2/`; integrate as
      `fig:overview`; retired 3-panel source stays in the dir, leaves main.tex.
      DELIVERABLE ALSO: standalone PDF for Robert to email Zhang EARLY (meeting A5
      cheap-veto). — STATUS: todo (Q3 gates layout; drafting can start on Option A)
- [ ] T-F2 Fig-1 swap per `figures/FIG1_TEASER_SPEC.md` Phase 1 (needs A0 PNG):
      \includegraphics + her caption + new \Description; old TikZ teaser retired in
      place. — STATUS: blocked(A0)
- [ ] T-F3 STRETCH (default SKIP per ADJ-1): real-data case-study figure for §4 from
      existing artifacts; only if T1–T9 + T-F1/F2 are done with time left; never
      relabel the schematic. — STATUS: parked

## Lane 3 — mechanics (independent; can run anytime)

- [ ] T-M1 `main.tex` template blocks: remove `\settopmatter{printacmref=false}` +
      `\setcopyright{none}` so the ACM Reference Format + permission blocks render
      (meeting A25–A26: make them appear, do not edit boilerplate); verify the
      anonymous+review options still hide authors; check rendered p1 footer.
      KEEP our real venue metadata lines (more correct than template placeholders).
      — STATUS: todo
- [ ] T-M2 Robert-owned, post-restructure: port to Overleaf main.tex + final-compile
      check there (different engine possible, [50:33]); email/ping Zhang on
      review-ready (A3). — STATUS: waits on T9

## Robert's standing submission-hygiene list (no Zhang mandate in the meeting; still his checklist)

- [ ] 9-row P0 citation pass in CITATION_PRIORITY_CHECKLIST.md (+ any rows added by
      T1/T2) — human ticks only
- [ ] Anonymous repo live + §1 footnote URL swap (scope per Q4)
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
