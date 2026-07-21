# §3 + §4 restructuring proposal (2026-07-21, after cut waves 1–6)

State: content ≈ 8.86 pp at `6e15a5d`; target 8.0 (strict, submission); appendix A–D
≈ 2.4 pp. This proposal is the "streamlined packaging" pass Robert requested: same
evidence, same claims, fewer words and fewer parallel tellings. Nothing here executes
without approval. Estimated landing if ALL of it fires: **≈ 8.1–8.2 pp**, with the
final ~0.1–0.2 closed by the two deferred cosmetic levers (§F below) and the polish
pass. Everything relocated is restorable at camera-ready (9 pp content allowed there).

## A. The invariant — what a reviewer must still get (unchanged)

1. The objective: Eq. (1), what each term means, the associational + demand-endogeneity
   caveats, the scalarization and its empirical selection.
2. The two named attribution mechanisms and their exactness (a contribution bullet).
3. The leveling diagnosis: trim-only helps no one under-served; structural, not
   incidental; demand endogeneity as root cause; hence lift.
4. Editor mechanics at concept level: bounded ε, tapered tail reroute, endogenous
   supply, king-rule validity, budget/phase order as a scientific control.
5. Downstream recipe + the two controls' design and n=12 outcomes.
6. Pillar 1 (data-level, with the external-instruments table + channel decomposition),
   pillar 2 (downstream recovery), the ablation certifying (3), the baselines with the
   fabrication disclosure, weight/feature-set robustness, the SF replication with the
   two-tier reading, and every register disclosure. All stay in the 8-page body.

## B. §3 restructure (~20 lines)

**B1. Merge §3.3 (Attribution) + §3.5 (Editor) into one phase-organized subsection,
"The Editor: Attribution, Trim, and Lift" (~12 lines).** Today each mechanism is told
twice — §3.3 says what *selects* (deficit attribution → trim; supply gradient →
lift), §3.5 re-introduces the same phases to say what *edits*. The merged section
walks each phase once, selection→edit: shared machinery (Eq. 3, ε-budget); **Trim:**
deficit attribution (exact partition, App. A) → in-ball relocation → §3.4 pointer;
**Lift:** supply-gradient attribution (ΔS pathway, said once — today the ΔS mechanism
is introduced in §3.3 and again in §3.5's lift paragraph) → screen (nominates only) →
tapered tail reroute with endogenous supply; then validity, budget/phase-order,
downstream run-ins as now. Both mechanism names stay prominent (bold run-ins), so the
contribution bullet's referent is intact. §3.4 sits between the objective and the
merged editor section, exactly where its diagnosis motivates lift.
Savings: duplicated phase framing, the double ΔS exposition, one subsection heading,
two transitions.

**B2. §3.4 as a compact enumerated argument (~8 lines).** The three structural
reasons become an `enumerate` with the numbers inline (selection: attribution never
nominates under-served cells; leverage: the 32×/93%-at-floor asymmetry; supply-side
inequity: 1.8 vs 17.6 median presence — a demand-only editor cannot touch supply),
followed by the oracle sentence, the demand-endogeneity cause, and the §leveling
analogy passage EXACTLY as it stands (protected). The essay flow becomes a list a
reviewer can audit at a glance; every number and the analogy scoping survive verbatim.

## C. §4 restructure (~35 lines nominal)

**C1. §4.1: merge "Metric classes" + "External fairness instruments" into one
"Metrics and claim discipline" block (~5).** The class-(iii) enumeration currently
appears in both. One block: three classes, then the instrument list + DP≡gap
disclosure + strict-count sentence, one pointer to App. D.

**C2. §4.2: let Table 1 carry the numbers (~4).** The channel paragraph re-narrates
what the table's bottom panel shows; keep the decomposition logic sentence, the
both-tiers significance claim, and the MAE-0.0 validity sentence; drop re-stated
magnitudes (they sit 3 cm away in the table).

**C3. Four-source comparison (Table 2) → Appendix C, claim stays in prose (~18).**
Body keeps three sentences: the claim under test; "the edited corpus is the fairest
source and identity-faithful (gate passed, matched 0.848 vs mismatched 0.193), with
Fidelity-B 0.187 = the disclosed cost of the bounded edits"; and the GAN-bimodality
honesty sentence with its SF contrast. The table + per-source walk-through join the
per-seed spread already in App. C. This is the largest single move; it is a secondary
result (neither pillar rides it) and returns at camera-ready.

**C4. §4.3 downstream: dose-detail hygiene (~4).** The fidelity-dose sentence
(0.020→0.027) and the F_spatial per-dose values join their siblings in App. C; body
keeps both verdicts ("Fidelity-A unchanged at every weight; Fidelity-B rises
dose-monotonically, disclosed", "F_spatial propagates, both controls degrade it").

**C5. §4.4 baselines prose tightening (~6).** The random-jitter story keeps its
surprise + broken-trajectories explanation but loses the secondary "why identity
fidelity cannot rank editing quality" elaboration (one clause survives); the
oversampling paragraph keeps headline contrast + fabrication/placebo disclosures
verbatim and sends the re-duplication arithmetic (8,241/1,759) to App. C.

## D. Auxiliary passes (~10 lines)

- §2 third pass at Robert's "length is secondary" reading (~5): harder compression of
  theme middles; cites and closes invariant (count-checked again).
- Connective-tissue sweep across §4 subsection openers (~3).
- §4.6/SF: none — the section is protected-dense and already pointer-compressed.
- Fig-2 caption: already tightened in W4; no further.

## E. Line budget and honest landing estimate

| Block | Nominal | Realized (×0.75) |
|---|---:|---:|
| B1 editor merge | 12 | 9 |
| B2 §3.4 enumerate | 8 | 6 |
| C1–C2 setup/data-level | 9 | 7 |
| C3 Table-2 relocation | 18 | 14–16 |
| C4–C5 downstream/baselines | 10 | 7 |
| D auxiliary | 10 | 7 |
| **Total** | **67** | **≈ 50–52 (~0.45 pp)** |

Landing: **8.86 − ~0.45 ≈ 8.4; float re-packing at these boundaries has historically
added ~0.1 → ≈ 8.3.**

## F. Closing the last ~0.3 (needs Robert's word, all cosmetic-or-deferred)

1. `\footnotesize` on the two merged tables (deferred earlier): ~5 lines.
2. Channel bottom-panel of Table 1 → two prose lines (numbers verbatim): ~6 lines.
3. Fig-2 TikZ vertical-padding trim (figure surgery, measured): ~3 lines.
4. Ablation bottom-panel of Table 3 → prose (numbers verbatim): ~6 lines.
5. Residual: the T18-style polish audit always recovers a handful.
Together ≈ 20 nominal ≈ 15 realized ≈ 0.13 pp → **≈ 8.15–8.2**; the remaining sliver
(~0.15) is where the (b)+(c) slack decision and the final widow/orphan polish live —
tight, but inside reach for the first time.

## G. Execution shape (on approval)

One restructure wave (Opus implementer, same verification discipline: gates, protected
greps, render-QA, byte-checks on protected blocks) for B+C+D; then measure; then the F
items Robert approves as a final micro-wave; then lint 8pt→5pt + the full §4 audit +
Robert's read-aloud pass of the restructured sections (B1/B2 change HIS argumentative
flow — his proofread is the real gate).
