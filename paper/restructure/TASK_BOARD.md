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
- [x] T2 `01_introduction.tex` — DONE 07-24 (44e4956): PI's intro logic near-verbatim
      (HSTD ¶1 + her cite map repaired incl. feng2020simulate; 3-challenge prose ¶ +
      C1–C5 forward pointer; contributions rebuilt, code+data anon-link footnote per
      Q4); 3.0× hook kept with fig-pointer moved off the number; PI teaser PNG in as
      Fig 1 (p1 top-right); 5 orphaned bib keys rescued into her cite groups; two
      Overfull-driven rewords flagged for Robert's read-aloud ("We address these
      challenges with FATE"; "the resources the disadvantaged group receives");
      §2 HSTD parenthetical deduped at commit. — STATUS: DONE
- [x] T3 NEW `02_overview.tex` — DONE 07-24 (c699369): definitions boldfaced,
      problem definition, five-challenge stacked block (C1 budget / C2 fidelity /
      C3 target / C4 level-up / C5 training; C1 uses k ≪ |T|, notation catch);
      sec:problem label moved with content (3 appendix referrers verified);
      main.tex order 1–6; §3.1 excised from 03. Orchestrator polish: C2 clause
      restructured; Meeting-44→07-24-meeting comment misnomer fixed.
      ⚠ HANDOFF: §2 opens by defining HSTD — T2's intro ¶1 also defines it (hers,
      authoritative); orchestrator dedups §2's parenthetical AT T2 COMMIT TIME.
      NO file renames this sprint (02_related_work.tex renders as §5). — STATUS: DONE
- [x] T4 `03_methodology.tex` — DONE 07-24 (d60491b): leading ¶ + five blocks (3.1
      objective C3 / 3.2 attribution+budget C1 [sec:editor+sec:attribution] / 3.3
      outcome-vs-resource C4 [sec:leveling on the limitation beat] / 3.4
      validity+fidelity C2 [sec:phys-validity] / 3.5 weighting C5 [sec:downstream]);
      framework figure integrated as Fig 2 p3 (schematic caption, k ≪ |T|); §3.3(b)
      compressed ~1/3 with full protected register (orchestrator render-verified:
      2,455 ×3, analogy, endogeneity ×3, 93%, 1.8/17.6, control sentence). Gates
      green; 0 undefined refs. COST: §3 ≈ +14 rendered lines; §6 heading moved to
      p9 (T9 page-budget input). ⚠ HANDOFF to T6/T7: conclusion's "single-pass
      editor \S\ref{sec:editor} anticipates" now points one block early — fix the
      ref/wording to match where the future-work sentence lives (end of 3.3).
      — STATUS: DONE
- [x] T5 `04_experiments.tex` — DONE 07-24 (585de04, ships Robert's editor-config
      tweak): RQ1–RQ5 leading ¶; SF → "Transferability: San Francisco"
      (reproduces + magnitude caveat); dedup −35 words with sharpened SZ/SF inline
      contrasts; budget stated as configured (no sweep implied); §4 refs repaired
      post-T4 (sec:editor→sec:method for "the trim+lift editor of §3"); frozen
      blocks byte-identical; 40/40 src comments intact. — STATUS: DONE
- [x] T6+T7 — DONE 07-24 (945b9e7): §5 position-checked (1 ref fix, 2 vocab
      touches, fidelity gate→signal per D15); §6 reframed on the two-stage
      collective spine with the C-1 chain sentence; bounds ¶ byte-identical;
      single-pass pointer fixed to §3.3; "Three directions" adds the k-sweep
      future-work sentence (plainly unmeasured here). — STATUS: DONE
- [~] T8 — FOLDED INTO T9 (T4 displaced nothing new into the appendix; what remains
      is a coherence/ref check, which is T9 item 1). — STATUS: merged into T9
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

- [x] T-F1 Fig-2 framework diagram — DONE 07-24 (09f1fed): three-phase per Robert's
      design; 239.8×300.7pt (inside both gates); grayscale-verified; weighting fork
      encodes the vanilla-null honestly; |T| notation (N stays the active-unit
      count). EMAIL ASSETS for the Zhang early-veto (meeting A5):
      `figures/figure-2/fig2-for-zhang.png` (300dpi crop w/ draft caption) or
      `framework-test.pdf`. INTEGRATION (in T4): wrap `framework.tex` in the figure
      env as `fig:overview`, caption per FIG2_FRAMEWORK_SPEC Implementation §4 with
      k ≪ |T| notation; retire the 3-panel `figure-2.tex` from 03. — STATUS: DONE
      (integration pending in T4)
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
