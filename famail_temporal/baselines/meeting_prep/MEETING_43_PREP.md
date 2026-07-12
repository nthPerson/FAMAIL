# Meeting 43 prep — progress since Meeting 42

**Prepared:** 2026-07-10 · **Updated:** 2026-07-11 (post weight-re-anchor; re-run campaign in flight).
**Baseline for "progress since":** Meeting 42, held 2026-07-09 (Robert + Dr. Zhang). Grounding: the
Meeting-42 record was extracted from Notion — both the AI summary
([`MEETING_42_SUMMARY_EXTRACT.md`](MEETING_42_SUMMARY_EXTRACT.md), unverified) and the **full transcript**
([`MEETING_42_TRANSCRIPT_EXTRACT.md`](MEETING_42_TRANSCRIPT_EXTRACT.md), ground truth, read in full) —
because Notion summaries have previously fabricated/omitted content. All progress claims below cite
committed artifacts.

> **⚠️ Headline-number change since the first draft of this prep:** the objective weights were
> **re-anchored to α\* = (0.1, 0.8, 0.1)** on 2026-07-11 (see §3). Numbers quoted at Meeting 42 from the
> (0.2, 0.7, 0.1) era (SZ +0.0222 / SF +0.0328 / WBC +0.0310) are **superseded**; the current committed
> headlines are in §5. Do not mix eras when speaking to the PI.

---

## 1. Meeting-42 recap (what was actually agreed)

**Action items voiced (T1–T6, transcript §1):** (T1) finish the GPU BC-propagation eval; (T2) implement
the other data-augmentation baselines; (T3) human-review all AI-assisted citations before use; (T4)
motivate the new attribution variant alongside the other objective terms; (T5) **start writing — the
methodology section first** (PI directive); (T6) **draft abstract to Dr. Zhang by next week**.

**PI decisions:** start paper-writing now (methodology first); **the trim-vs-trim+lift ablation is
necessary** ("this kind of design is really necessary"); abstract-as-placeholder is acceptable; general
endorsement of the trim+lift direction pending the BC/GAIL propagation result; timeline confidence.

**Corrections to the Notion record (transcript §6 — worth stating at Meeting 43):** the summary marked
T1 and T2 as done `[x]` when both were open at meeting time; it erased **Dr. Cash's credit** for the
"lower half of trajectories" insight that motivated the lift phase (provenance that should survive into
acknowledgments); "~80 model combinations" was actually spoken as "60 or 80"; the king-moves rule's
source paper (the cGAIL/"Seagale" preprocessing convention) was dropped. No fabrications this time.

---

## 2. Progress since Meeting 42, by action item

### T1 — BC-propagation eval: ✅ LANDED, now RE-RUNNING at the adopted weights
The supply-lift eval landed 2026-07-09 (weighted-BC recovery +0.0310 @ w30, 6/6 seeds; F_spatial
propagated; rollout drain attenuated ~40%, not reversed — all on the (0.2,0.7,0.1) corpus, now the
prior era). Under the re-anchor (§3) the **entire downstream suite re-runs on the α\* corpora**; the
SZ weighted-BC sweep is running overnight 2026-07-11 (campaign stage r4), rollout re-run queued behind
it. The *qualitative* claims (recovery, F_spatial propagation, drain-attenuated disclosure) are the
expectations these re-runs test; the paper carries only α\*-era numbers.

### T2 — data-augmentation baselines: ✅ BUILT (4 arms); oversampling DONE; perturbation arms RUNNING
- **Demographic Oversampling (done, committed):** targeted mean ΔF_causal **+0.0153** (dose-monotone) vs
  placebo **−0.0172** vs FAMAIL **+0.0226** at zero inflation (comparator updated to the α\* headline) —
  targeting is necessary AND insufficient; the placebo's DP gap explodes (+2.8) via fabricated supply
  landing in advantaged cells. Full record: `PAPER/baselines/demographic-oversampling/FINDINGS.md`.
- **3-arm perturbation suite (iFGSM / FGSM / random-jitter):** running NOW on CPU (2026-07-11, campaign
  stage q1 under the memory-guarded companion launcher) against the α\* headline corpus; the 6-row
  cross-arm table lands in `PAPER/baselines/comparison/` when the arms finish. Paper-facing caveats
  locked: arms are **"iFGSM/FGSM with random restart", not vanilla ST-iFGSM** (vanilla at δ=0 is a
  provable no-op, kept as an ablation row); FGSM numbers from the corrected engine (`6da3d27`+).

### T3 — human review of AI-assisted citations: ◐ OPEN (Robert's pass)
Machine verification is done (2 fabrications caught and removed; audit in `mission_2_citation_audit.md`;
`paper/refs.bib` header flags the pending human pass). **Robert's own final pass remains open.**

### T4 — motivate the attribution variant / objective: ✅ DONE — and it produced a finding
The α-Pareto sweep completed (5 points + anchor, all full k=10,000 trim+lift runs) and the
"Why these weights" story is now **fully empirical and stronger than planned** — see §3 for the
re-anchor narrative. `PAPER/objective-motivation/MOTIVATION.md` ("Why these weights") and methodology
§3.2 are folded in at the adopted configuration; the canonical decision record is
`PAPER/objective-motivation/weight-sensitivity/{DECISION.md, EXTENDED_FRONTIER.md}`.

### T5 — methodology section: ✅ DRAFTED, COMPILED, TWICE-AUDITED
The manuscript scaffold exists (`paper/`, ACM `acmart`, compile + convention-lint gates): methodology
§3.1–3.6 complete (problem formulation; objective; **two-mechanism attribution** incl. the
autograd-verified ∂F_causal/∂S closed form; the leveling-down mechanism subsection; the trim+lift
editor; the upweighted-imitation recipe). Verified by two independent read-only agents
(number/convention auditor + FAMAIL-fidelity reviewer); their fix wave is applied. **The Experiments
section (5.1–5.7) is also drafted** — setup, data-level fairness, the ablation (complete, see §5),
downstream propagation, baselines, robustness/sensitivity, SF external validity — with pending cells
wired to campaign stages that slot in as runs land. Workflow: repo = writing source of truth; Robert
ports finished sections to the shared Overleaf.

### T6 — draft abstract: ✅ DRAFTED (in `paper/main.tex`, marked draft)
~215 words, absolute deltas only, audited against the honesty boundaries (the first draft's two
overclaims were caught by the fidelity audit and fixed **before** any PI hand-off). Ready to port to
Overleaf once Robert is happy with it; numeric claims re-verify automatically as the α\* campaign
lands its remaining suites.

---

## 3. New decisions since Meeting 42 (chronological — brief the PI on all three)

1. **Trim+lift centers ALL PAPER reporting; trim-only appears only in the ablation** (Robert,
   2026-07-09/10). Operationalizes Zhang's "the ablation is necessary."
2. **Full re-run bill + maximal reproducibility discipline** (Robert, 2026-07-11): every reported
   number comes from re-runs on the adopted corpora; nothing runs without a row in
   `famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md` (command, commit SHA, frozen-editor gate,
   environment capture, SHA-256 checksums, per-dir PROVENANCE); a `PAPER/REPRODUCIBILITY.md`
   claims→artifacts→ledger→command capstone is planned. Framing for reviewers: full transparency.
3. **⭐ WEIGHTS RE-ANCHORED to α\* = (0.1, 0.8, 0.1)** (Robert, 2026-07-11 — the headline item for
   Meeting 43). The story matters as much as the outcome:
   - The completed sweep showed a **flat ΔF_causal frontier** with (0.55, 0.35, 0.1) weakly dominating
     the shipped (0.2, 0.7, 0.1) on the two optimized axes. Robert chose to re-anchor there
     (camera-readiness: never ship a dominated configuration).
   - **The promotion was HALTED by our own checks:** the (0.55,…) corpus's channel decomposition showed
     the tier-1 supply channel **significantly negative** — at spatial-heavy weights, lift routes tails
     toward spatial evenness instead of disadvantaged cells, and the lift-up dies. The two-axis frontier
     was an **incomplete decision basis**.
   - The frontier was extended with **ring-2/ring-3 columns for every sweep point**
     (`EXTENDED_FRONTIER.md`): the lift-up **declines monotonically with α_spatial** and is significant
     on both supply tiers only for α_sp ≤ 0.2, while ΔF_causal stays flat and external metrics improve
     everywhere. An **amended three-ring criterion** (max ΔF_causal s.t. ΔF_spatial ≥ 0 AND lift-up
     significant on both tiers) selects **(0.1, 0.8, 0.1)** — which weakly dominates the old headline on
     every reported column and nearly **doubles the lift-up** (tier-1 +0.0176 vs +0.0091; tier-2 +0.0411
     vs +0.0242).
   - Proposed PI framing: this is a *strength* — the weight choice is now a measured, criterion-driven
     selection with a table a reviewer can check, and the sensitivity analysis itself surfaces a
     methodological point (optimizing-ring metrics alone can hide the property that matters).
   - Hedge: an s10 replication run is scheduled before camera-ready (tie-nondeterministic edit
     ordering), with pre-commitment to report both runs.

---

## 4. Suggested discussion items for Meeting 43

1. **The weight re-anchor + extended frontier** (§3.3) — walk Zhang through the two-decision story and
   the three-ring table; confirm he's comfortable with (0.1, 0.8, 0.1) and with presenting the
   incomplete-basis finding openly in the sensitivity subsection.
2. **⚠️ DECISION REQUIRED — the SF fairness framing (PI call, blocking a `% TODO(PI-framing)` marker in
   `paper/sections/04_experiments.tex`, §"External Validity: San Francisco").** At α\* the SF tension
   reproduces exactly: the supply channel is positive-significant (**+0.0209**, CI [+0.0122, +0.0300])
   but lift also routes *pickups* into under-served cells, so recorded demand there rises and the total
   mean(Y|D) is net-negative (**−0.0324**, CI excludes 0; demand channel −0.0533\*). The draft presents
   BOTH readings side by side and Zhang picks which leads:
   - **Ratio reading:** the SF lifting-up claim is withheld (the S/D ratio for the under-served group falls).
   - **External-metrics reading:** more rides *served* in under-served areas is a demand-side parity gain
     — DP, DI, and Theil all improve, and the falling ratio is precisely the demand-endogeneity mechanism
     the paper's §3.4 predicts (recorded demand was suppressed; serving it raises the denominator).
   Supporting context that strengthens the external-metrics reading since Meeting 42: SF's migrant DP is
   now significant under **both** grouping conventions at α\* (extremes −0.0729\*, median-split −0.0370\*);
   the SF weighted-BC recovery reproduces (+0.0332 @ w30, 6/6, controls fail); and the SF four-source
   table reproduces with fresh generators (edited fairest 0.9067, identity-faithful). Whichever reading
   leads, the other stays disclosed — the decision is about *emphasis*, not omission.
3. **Ablation is COMPLETE and textbook** (his "really necessary" item): at identical weights/budget,
   trim-only = +0.0146 F_causal, *negative* F_spatial, disadvantaged level flat to 4 decimals
   (7.0734 → 7.0734) — pure leveling-down; trim+lift = +0.0226, +0.0061, +0.053 lift-up (CI excl. 0).
   On SF both editors move DP significantly at α\* — the contrast there is magnitude (~40% larger) +
   the supply channel, stated as measured (the old "immovable migrant axis" phrasing was retired).
4. **Demographic-oversampling result as the naive-lifting-up contrast** — targeting necessary
   (placebo degrades), insufficient (below FAMAIL at 10.5% fabrication vs zero). Pool-exhaustion
   disclosure: migrant ≡ comp origin pools (4,907); distinct pool 8,241 < 10,000.
5. **The "54%" figure still needs grounding or retiring** — committed records use absolute deltas
   (now +0.0226 SZ / +0.0316 SF); nothing in the repo supports a "54%" relative claim.
6. **Writing status + Overleaf hand-off plan** — methodology + abstract + experiments drafted and
   audited; which sections Robert ports first; abstract-to-Zhang timing vs the Jul 19 deadline.
7. **Record hygiene:** the two wrongly-checked Notion boxes; **Dr. Cash's acknowledgment** placement.
8. **Campaign/GPU queue status:** overnight = SZ weighted-BC (r4); then rollout re-run, SF weighted-BC,
   L1v2 four-source tables (fresh generators), variance suites, alternate-feature-set matrix (q6–q8),
   s10 replication. Concurrency: one memory-guarded CPU companion at a time (measured footprints;
   host-RAM guard at 10 GB kill floor).

---

## 5. Numbers cheat-sheet (α\*-era, committed; provenance in the linked bundles)

| Result | Value | Where |
|---|---|---|
| **FAMAIL trim+lift headline (SZ / SF)** at α\*=(0.1,0.8,0.1) | ΔF_causal **+0.0226** / **+0.0316**; ΔF_spatial **+0.0061** / **+0.0139** | `PAPER/supply-lift/data/a10/` |
| SZ lift-up (design-targeted, ring 2) | Δmean(Y\|D) **+0.0529** [+0.0086, +0.0989]; supply tier-1 **+0.0176\*** / tier-2 **+0.0411\*** | same + `weight-sensitivity/EXTENDED_FRONTIER.md` |
| SZ external metrics (ring 3) | DI **+0.0162\***; DP **−0.890\***; Theil **−0.0087\*** | same |
| Ablation (SZ): trim-only vs trim+lift | +0.0146 / −0.0011 / **flat 7.0734** vs +0.0226 / +0.0061 / **+0.053\*** | `paper/` §5.3 + `data/a10/` |
| Ablation (SF) | +0.0144, DP −0.052\* vs +0.0316, DP −0.073\* (both sig; contrast = magnitude + supply channel) | same |
| SF channel (the tension, PI flag) | supply **+0.0209\***, demand −0.0533\*, total **−0.0324\*** | `data/a10/sf12_a10_channel_decomposition.json` |
| Extended frontier finding | lift-up monotone-declines with α_sp; sig both tiers only α_sp ≤ 0.2; ΔF_causal flat | `weight-sensitivity/EXTENDED_FRONTIER.md` |
| Oversampling targeted / placebo d10k | **+0.0153** dose-monotone / **−0.0172**, ΔDP +2.8; inflation 10.5% | `PAPER/baselines/demographic-oversampling/` |
| SUPERSEDED (0.2,0.7,0.1)-era numbers | +0.0222 SZ / +0.0328 SF / WBC +0.0310 — do NOT quote as current | prior-era `PAPER/supply-lift/` root |
| PENDING re-runs | SZ WBC (overnight), rollout, SF WBC, L1v2 ×2, variance ×2, per-set matrix, 6-row baseline table | `EXPERIMENTS_RUN_LEDGER.md` + `driver.sh --status` |
| Attribution coverage (trim+lift) | ~2,400 → ~9,900 edited trajectories (2,337+7,545 post-filter, ~10% of corpus) | s10 corpus `metrics.json` |
