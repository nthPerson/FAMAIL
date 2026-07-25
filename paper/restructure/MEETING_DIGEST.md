# MEETING DIGEST — adjudicated synthesis of the 2026-07-24 meeting

Written by the orchestrator after the A/B/C extraction reports and the D verification
pass. **On any conflict, trust THIS file**, then C (claims) for wording constraints,
then A/B for detail, then the transcript itself. Timestamp corrections from D are
applied here. Items marked ⚖ PENDING await Robert's ruling — do not treat them as
decided until this file replaces PENDING with his answer.

## 1. The ten load-bearing decisions

1. **Spine = budget-aware trajectory editing for collective fairness.** The old
   fairness-vs-fidelity trade-off is DEMOTED to "a good point" / a constraint
   (retained in methodology + disclosures, out of the novelty sentence). The
   differentiator vs FairGAN/model-side methods: FATE measures and repairs fairness at
   the level of the WHOLE corpus's service allocation, by editing a small portion of
   trajectories. [03:38–07:01]
2. **Why the reframe:** FATE is editing + upweighting; a story sold as "editing alone"
   cannot account for the second stage. Edit-aware weighting is co-equal, not an
   afterthought — and NOT appendix-able. [03:38, 04:12]
3. **⚠ The spine must be worded OUR way, not her literal sentence.** Her spoken claim
   ("small portion won't much affect raw-data fairness but will affect downstream
   models") is wrong in both halves: data-level fairness moves reliably (that is our
   headline evidence), and downstream movement requires the weighting (vanilla BC is
   null). Use the C-1 proposed wording (analysis_C §2): small budget → measurable
   corpus-level gain on never-optimized measures → weighting carries it into training,
   where uniform weighting averages it away. Her written abstract in the revision PDF
   is already accurate — adopt THAT register, never the spoken one.
4. **Structure:** 1 Intro · 2 Overview · 3 Methodology · 4 Experiments · 5 Related
   Work · 6 Conclusion. Current §3.1 moves to the Overview. Email section names are
   NOT binding (said 3×). [29:26, 34:16–34:45, 39:14]
5. **Challenges:** brief prose mention in the Intro (no itemization); defined and
   labeled (C1, C2, …) in the Overview AFTER definitions + problem definition; NO
   LaTeX itemize environment — stacked bold lead-ins in running text. Count was NOT
   fixed in the meeting ("C one, C two, C three" was illustrative). [30:46–32:41]
6. **The binding structural invariant: each challenge ↔ exactly one methodology
   block, in logical order**; §3 opens with a leading paragraph naming FATE, giving
   the two-stage overview, and pointing to the framework Figure 2. [30:07, 33:20,
   34:45]
7. **Figures:** Fig 1 = Zhang's disparity-aggregation teaser, unmodified. Fig 2 =
   abstract framework/sequence diagram in the ST-iFGSM Fig-3 style (colored stage
   bands, boxed arrow labels, terminal contrast), REPLACING the 3-panel city figure;
   it must show attribution as a procedure (input → scoring → selected k≪N), not just
   a result. Tool free choice; content first; TikZ polish = least priority. AI
   drafting explicitly endorsed. [10:42–11:21, 12:55–27:49, 52:07–52:14, 56:27]
8. **Experiments:** leading paragraph (aims + organization); research-question
   framing (count not fixed); Shenzhen answers every question, San Francisco appears
   as a consolidated transferability block; identical cross-city observations stated
   once; genuine SZ↔SF differences still surfaced (Robert's defensibility carve-out,
   unopposed) and made "more visually striking" [46:10]. Baselines stay SZ-only —
   accepted. NO new experiments; the email's "budget analysis if possible" was never
   raised and is DROPPED (no k-sweep exists; future-work sentence instead). [39:14,
   41:10–48:31]
9. **Appendix:** absorbs displaced mechanism detail; must hold "enough detail about
   this implementation"; reproducibility is NOT a deliverable this round. [19:28,
   20:24, 40:30]
10. **Timeline + bar:** Robert delivers restructured draft + figures together,
    target Saturday night 07-25, and pings Zhang the moment it is review-ready
    (she starts on notification); Zhang does the final pass; submission Sunday night
    07-26. Bar explicitly lowered: "we can just submit whatever we have"; burnout
    caution issued. Send the framework figure EARLY for a cheap veto. [37:09–38:19,
    51:03–51:46, 57:01–57:24, 53:29]

## 2. Adjudications of the D-report conflicts

**ADJ-1 · Case study reuse of the retired 3-panel figure → C wins, with A/B's path
conditionally open.** A schematic is never labeled a case study (C-8/D12). The case
study is OPTIONAL ("no time is totally okay"). If built, it must plot a REAL edit from
existing artifacts (a real lift reroute: before/after tail geometry, value-of-presence
context, resulting ΔS), and only then may it inherit the retired figure's visual
language. Default plan: SKIP; revisit only if everything else lands early.

**ADJ-2 · Fig-2 stage grouping → hybrid, encoded in figures/FIG2_FRAMEWORK_SPEC.md.**
Zhang's skeleton (one selection of k, then trim+lift as one editing stage) is a
WOULD-MISREPRESENT flow if drawn literally (C D7): trim and lift have separate
selections, and lift's scores are computed on the post-trim state. Her deeper asks are
(a) input → stages → output, (b) the k≪N budgeted-selection beat made visible,
(c) two stages like her exemplar. All three are satisfiable accurately: Stage 1 =
budgeted editing band showing the two passes IN SEQUENCE (deficit map → trim edits,
then presence-value map on the edited corpus → lift reroutes) under one budget
annotation; Stage 2 = edit-aware weighting band; terminal contrast boxes. Visual
merging of trim+lift into one BAND is fine (C D9); a single shared selection arrow is
not. Prose never merges the phases.

**ADJ-3 · Challenge count → recommend FIVE, mapped 1:1 to five methodology blocks**
(B's "three" was a misreading per D). The mapping that satisfies invariant #6 with the
email's own block set:
  - C-budget (rewritten from old C1 scarcity): only a small share of the corpus may
    change, and it must be the share that matters → block: attribution under a budget.
  - C-target (old C3): equal service is the wrong target → block: the collective
    fairness objective.
  - C-levelup (old C4): measured fairness can improve while the under-served gain
    nothing → block: outcome-side vs resource-aware editing (trim's limitation → lift).
  - C-fidelity (old C2): edits must remain real driver behavior → block: validity- and
    fidelity-constrained editing (where the demoted trade-off now lives).
  - C-training (old C5): sparse edits must survive training → block: edit-aware
    weighting.
  ⚖ PENDING — Robert decides the final count/wording.
  Note: her revision PDF's intro has three italicized challenges — the INTRO PROSE can
  gesture at them without count commitment; the labeled list lives in the Overview.

**ADJ-4 · Zhang's markup on the discarded snapshot → A wins.** The 12-hour-old draft
Robert emailed her that morning is DITCHED per [10:10], unopposed. Nothing to
re-apply. Her durable inputs are the restructuring email + the revision PDF's
abstract/intro/Fig-1 + this meeting.

## 3. Corrections to carry (from D)

- "use whatever language that makes sense" = [36:40] (not [36:45]).
- "I have the context of the project" = [07:01] (not [07:23]).
- Three of B's "→ Zhang: yes" confirmations are diarizer run-ons inside Robert's own
  turns ([52:14], [34:16], [49:33]) — treat those points as Robert-stated and
  uncontradicted, not Zhang-confirmed. The substantive decisions stand on other turns
  ([52:07], [34:45]).
- [48:31] is a topic close, not an endorsement; the SF-differences carve-out stands as
  "stated and unopposed".
- The email remains the binding baseline; the meeting is additive emphasis ([08:47]) —
  the ZHANG_DIRECTIVES ledger dispositions stand except where this digest notes an
  override (challenges location, budget-analysis DROPPED, section names non-binding,
  reproducibility deferred).
- Current §4 already has SF material both inline and in a final SF subsection
  ([42:40]: the fully separate treatment was already dissolved once); the target state
  is: minimal inline SF mentions (differences only) + one consolidated transferability
  block.

## 4. Open items that survived to the Robert-question round

1. Workspace mechanics (archive dir vs branch) — his standing pattern is an archive
   copy. ⚖ PENDING.
2. Challenge count/wording (ADJ-3). ⚖ PENDING.
3. Fig-2 layout choice (spec offers the hybrid two-stage and a three-band variant).
   ⚖ PENDING.
4. Anonymous-link sentence scope (code-only vs code+data). ⚖ PENDING.
5. Who physically clicks submit ([57:24] ambiguity) — Robert clarifies with Zhang
   directly; plan assumes Zhang submits per [38:19].
6. Source assets into `zhang/` (2 PDFs + teaser PNG) — requested from Robert.
