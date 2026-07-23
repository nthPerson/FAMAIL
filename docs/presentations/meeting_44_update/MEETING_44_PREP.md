# Meeting 44 prep — status update + the Figure-1 decision

*Prepared 2026-07-20 for Robert's meeting with Dr. Zhang (week of 2026-07-20);
updated 2026-07-22 with the submission-week execution record and the new
reproducibility document (§3); updated 2026-07-23: **the length problem is
solved** — content is under the strict 8-page limit after the argument-triage
campaign (agenda item 5 is now a ratification, not a decision). New location
note: meeting prep now lives in `docs/presentations/<meeting>/` (earlier
meetings used `famail_temporal/baselines/meeting_prep/`).*

**Deadlines:** abstract **submitted 2026-07-19** (KDD 2027 Research Track
Cycle 1, OpenReview; "we can always modify the submission"). Full paper due
**Monday 2026-07-27 23:59**.

---

## 1. What has happened since Meeting 43 (2026-07-16)

**Submission + naming**
- **Abstract re-written and submitted.** Rebuilt to Dr. Zhang's four-beat
  structure, then polished by Robert; single-sourced in
  `paper/sections/00_abstract.tex` (shared by the manuscript and the
  standalone `kdd27-abstract-only.tex`, so the two can never drift).
- **Title:** "Mitigating Demonstration Bias via Fairness-Aware Trajectory
  Editing" (per Dr. Zhang's suggestion to lead with the problem).
- **Method named FATE** (Fairness-Aware Trajectory Editing; replaces the
  FAMAIL working name). **F_causal renamed F_demo** per Dr. Zhang's approval
  (2026-07-20): all prose, equations, tables, and the regenerated frontier
  figure; the associational caveat stays. Code and artifact keys keep
  `f_causal` (mapping recorded in `paper/README.md`).

**Writing**
- **Introduction re-written** to Dr. Zhang's six-beat outline: gap →
  why-existing-methods-fall-short (new intervention-categories paragraph) →
  FATE's position → mechanism summary → results with intervals (DI +0.0162,
  DP gap −0.890 (14.199 → 13.309), Theil −0.0087) → contributions.
  ST-SiameseNet is now cited at first mention.
- **Introduction reference check:** referenced resources verified to exist.
  The two citations added for the categories paragraph (FairGAN, DECAF)
  were machine-verified against primary sources on 2026-07-21 (pages/DOI
  added, claim-fit confirmed); Robert's human pass per Dr. Kash's mandate
  is the remaining step — see agenda item 2.
- **Related work:** each of the four themes now closes with the concrete
  limitation FATE addresses (Dr. Zhang's "state the contrast" note).

**Figures**
- **Figure 2 (method overview):** glyph vocabulary per Dr. Zhang — passenger
  stick figure = service pickup, car glyph = taxi presence — plus legend
  updates.
- **Figure 1 redesigned** — see §2, the decision item for this meeting.
- **Color refresh applied to both figures** (the "more engaging colors"
  promised in Robert's email): a two-hue system — muted cobalt = added/
  edited (the FATE intervention), muted amber = excess/trimmed, charcoal
  neutrals, and pale amber/blue *regional* tints marking over-/under-served
  areas in both figures' maps. Figure 1's corpus boxes now show real
  GPS-trace renderings (the edited one with its blue slice and "+").
  Verified grayscale-safe and CVD-safe (blue–amber is the colorblind-safe
  axis; simulated deuteranopia/protanopia separation is large; shapes,
  dashes, and "+" marks still carry all semantics without color).

**Experiments closed since Meeting 43** (all landed in §4; no runs pending)
- **SF two-tier supply recount (D1):** counting taxis as *distinct vehicles*
  from raw GPS, the lift-up supply channel is +0.1027 (CI-significant) and
  the tier-2 total is +0.0493 (significantly positive) — the earlier tier-1
  net-negative was a fractional-presence accounting artifact. §4.7 now makes
  the two-tier statement. *Walk-through owed to Dr. Zhang — agenda item.*
- **Flagship n=12 in both cities:** the w30 recovery replicated on twelve
  paired seeds per city (p = .00049, 12/12 positive both cities; SZ
  +0.0297 ± 0.0029, SF +0.0333 ± 0.0050).
- **SF n=12 controls:** random-slice upweighting degrades fairness;
  most-fair-slice selection is positive but ~6× smaller than the edited
  slice — the effect is edit-specific.
- **Penalty-formulation probe:** the fairness-penalty baseline's failure is
  formulation-independent (absolute-value variant tracks the signed one);
  §4.5 records it.

**Manuscript logistics — LENGTH SOLVED (2026-07-23)**
- Main content cut **10.6 → under 8.0 pages** (references now begin ~95%
  down page 8): eight measured waves + §3/§4 restructure (07-21), then the
  **argument-triage campaign** (07-22/23) executed with Robert's per-item
  tier approvals. KDD rules: 8 content pages at submission; 9 content + 12
  total only on acceptance.
- Argument triage, what moved to the appendix (all restorable at
  camera-ready, every number preserved with origin notes): the rollout
  **allocation-boundary** disclosure (both cities; §5's future-work clause
  anchors it), the **SF downstream detail** (n=12 control spreads, extended
  doses, grouping-convention CIs), the **distinct-taxi recount mechanics**,
  and several depth items (δ=0 ablation aside, penalty-formulation detail,
  saturation/oversampling compression). Kept on Robert's call: the
  leveling-down "analogy is inexact" refinement, the random-jitter surprise
  paragraph, the intervention-categories paragraph, all five §2 theme
  closers in Dr. Zhang's contrast cadence.
- **Zero protected content lost**: all headline numbers, disclosures, and
  caveats survive in main text or appendix; a ~55-number audit of §4 found 0
  mismatches. Lint gate at 5pt. ⚠️ The margin is knife-edge: almost any
  main-text addition pushes references back to page 9.
- Robert's read-aloud pass now covers **§1 through §4 complete**; next:
  polish §5 and check the appendix. The pass also drove a reader-clarity
  sweep: "tier-1/tier-2" renamed **fractional-presence / distinct-taxi**
  accounting; Fidelity-A/B now defined at their §3.2 source + a two-axes
  appendix gloss; the k/n seed-count notation defined in §4.1; "oracle"
  and other flashy register retired; em-dashes near-eliminated.
- Citations: FairGAN + DECAF machine-verified (pages/DOI added); a §4.4
  citation-gap pass added method cites (iFGSM/FGSM/PGD/oversampling class)
  incl. one new DBLP-verified entry (Madry et al., PGD). Human pass: three
  refs remain (see agenda item 2).
- A PII/anonymity scan of the built PDF is clean (anonymous author block,
  no metadata leaks, self-citations in compliant third person).
- Anonymous sigconf build with real venue metadata (KDD '27, San Jose).

---

## 2. Figure 1: options, tradeoffs, sizes (the decision for this meeting)

Dr. Zhang's request ("only showing c would be sufficient") is **already
executed** — the live manuscript figure is the (c)-only build. The question
for the meeting is whether to keep it or adopt the (b)+(c) variant.

| Option | Content | Size (figure+caption) | Saved vs 3-strip |
|---|---|---|---|
| A — 3-strip (archived) | problem → asset → stakes | 15.4 cm | — |
| B — (b)+(c) counter-proposal | asset → stakes | 12.0 cm | ~9 lines |
| C — (c)-only (**live**) | stakes only | 11.4 cm | ~11 lines |

**Tradeoff in one sentence: B costs only ~0.6 cm ≈ 1–2 text lines more than
C, and it is the only variant in which the paper's defining beat — the data
is valuable human expertise, so FATE *edits* rather than regenerates — has a
visual home.** Both B and C get nearly all their savings from dropping the
city panel (a); C then has to re-introduce labeled corpus chips as arrow
sources, which claws back most of panel (b)'s removal. The page budget does
not hinge on the difference: the cut plan reaches 8.0 pages via appendix
relocations and keeps ≥2 lines of slack for this swap.

**RESOLVED (Robert, 2026-07-21): C stays.** With the 8-page limit binding
hard after the cut campaign, Robert has settled on the live (c)-only figure
("it does the job, and the simplicity is a strength"); the (b)+(c)
counter-proposal is retired from the meeting agenda. The B variant and the
comparison material remain in the repo as design history.

Refinements applied to the live figure since the redesign (also on B where
applicable): an explicit **FATE provenance arrow** from the raw-corpus chip
to the edited-corpus chip (the edited corpus visibly *comes from* the raw
corpus), and a **boxed, centered legend** with a third entry glossing the
accent "+" marks ("changed by the edit").

**Meeting materials** (in `paper/figures/figure-1/design-archive/`):
`comparison-2026-07-20.png` (all three variants side by side, measured sizes
in the headers) · `preview-3strip-final.png` (the archived original) ·
`zhang-meeting-talking-points.md` (the full argument + anticipated
objection).

---

## 3. NEW: the reproducibility record — `PAPER/REPRODUCIBILITY.md` (2026-07-21)

*The T17 capstone from the Meeting-40 task list, landed this week. One
document that makes every number in the paper independently re-derivable.*

**What it is.** The map from **every headline paper claim** to (a) its
curated, git-tracked artifact, (b) the raw results directory that produced
it, (c) the run-ledger row that launched it, and (d) the exact command and
environment record. 39 claim rows covering the Shenzhen editor, downstream
suites, baselines, feature-set robustness, and the San Francisco
replication.

**Why it exists (four purposes):**
1. **Reviewer and PI trust.** "Where does this number come from?" is now a
   one-lookup question for any value in §4 — claim → artifact → command,
   with per-artifact SHA-256 checksums and environment fingerprints
   (Python/Torch/CUDA/GPU + pip-freeze hash).
2. **Era discipline, made mechanical.** The project's biggest recurring
   hazard has been stale-era numbers. The document gives the verification
   rule: never trust a directory name or prose — read the artifact's own
   `config_snapshot` (α\* = 0.1/0.8/0.1, TAIL_LEN = 4) and its edit-count
   fingerprint (Shenzhen 2,337 + 7,545; San Francisco 1,330 + 629). It also
   flags the one trap a re-runner would hit: the committed config's ALPHA
   defaults are *not* α\*; the weights are applied per run via overrides.
3. **The name translations, recorded once.** Repo/code "famail" = paper
   "FATE"; artifact key `f_causal` = paper symbol F_demo (code keys
   unchanged); the 2026-05-14 sign-convention erratum; "3feat" = HGC;
   "4feat" = PRIMARY + logpopdensity. This is the mapping-of-record the
   paper README points to.
4. **Seed for the anonymized artifact repository** (the Meeting-43
   anonymity workstream): the document is PII-free by construction — no
   names, emails, or institutions — so it can be copied into the anonymous
   repo as its reproducibility README.

**The evidence behind it (not just claims):**
- A 2026-07-15 read-only audit certified the whole input chain: every
  checked §4 number matches its artifact JSON to full precision; all 38
  curated artifact twins are byte-identical to their raw sources; **zero
  correctness-critical discrepancies**.
- **The headline corpus re-derives exactly**: an end-to-end replication
  under clean `main` (S10-REPLICATION) reproduced every metric and count of
  the promoted Shenzhen corpus, including ΔF_demo +0.022561.
- Re-run recipes per experiment class (editor, external fairness, channel
  decomposition, weighted-BC, fidelity, variance, baselines), commands
  verbatim from the run ledger, GPU/CPU noted.

**Honest gaps it records** (disclosed, not hidden): the oversampling
baseline arms predate environment fingerprinting; some older `PAPER/` prose
docs still state superseded-era numbers (the document says which, and to
trust `config_snapshot` instead); and no data-availability/licensing
statement exists yet for the datasets — a decision needed for the anonymous
artifact repo (agenda item 3).

**Rider closed this week:** the one reviewer-facing statistics gap the
audit flagged — the feature-set robustness table's lift-up cells carried no
significance statement — is closed: the channel decompositions for both
alternate feature sets exist (bootstrap B = 2,000), all intervals exclude
zero, and the table now marks significance directly (HGC total +0.0594
[+0.0181, +0.1013]; 4FEAT +0.1461 [+0.0900, +0.2039]).

---

## 4. Other agenda items

1. **Reading-B / D1 acknowledgment** — the SF two-tier framing in §4.7 has
   not yet been walked through with Dr. Zhang (decided and executed
   post-Meeting-43); this meeting is the slot.
2. **Citation verification status** — FairGAN, DECAF, and the new
   Madry-et-al. (PGD) entry are all **machine-verified against primary
   sources** (NeurIPS proceedings page fetched live; DBLP + IEEE DOI;
   pages/DOI added to refs.bib; claim-fit confirmed). What remains is
   Robert's human pass per Dr. Kash's Meeting-43 mandate — DECAF and PGD
   need a proceedings/OpenReview eyeball; FairGAN needs the IEEE Xplore
   page (bot-blocked, genuinely needs human eyes).
3. **Anonymity / artifact repo** — the manuscript-side scan is **clean**
   (anonymous author block, no PDF metadata leaks, third-person
   self-citations). Remaining for the anonymous artifact repo: scrub of
   repo docs/comments, and a **data-availability/licensing statement**,
   which exists nowhere yet — needs a decision with Dr. Zhang
   (`REPRODUCIBILITY.md` is ready to seed the repo, see §3).
4. **Colors** — the refreshed palette is already applied to both figures
   (and to both Figure-1 variants, so the A/B/C comparison also previews
   it); confirm Dr. Zhang is happy with the direction.
5. **RESOLVED (2026-07-23): the length decision was made and executed** as
   the argument-triage campaign (Robert's per-item approvals; see the
   manuscript-logistics block above for exactly what moved and what was
   kept). Content is now under the strict 8-page limit. The ask for Dr.
   Zhang shrinks to **ratifying the triage**: the moved items are in the
   appendix with origin notes and return at camera-ready (9 content + 12
   total on acceptance).
