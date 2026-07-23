# KDD Reviewer Simulation — FAMAIL manuscript (2026-07-15)

**Read against:** `paper/` sections 01–05, rendered PDF (11 pp, review-mode local build, α\*=(0.1,0.8,0.1) era).
**Format assumed:** KDD research track, double-blind; template/precedent = ST-iFGSM (Hu 2023).
**Purpose:** feed the 2026-07-16 PI meeting. Report-only; no repo edits.
**Method:** three passes — R1 fairness-in-ML, R2 mobility/IL empiricist, R3 skeptical senior ("reasons to reject").

A parallel session is assembling a *runs menu*. Objections that **existing results already answer** are marked
🟢 REBUTTAL-ONLY; objections needing **new prose** are 🟡 PROSE; objections a **new run** would materially
strengthen are 🔴 RUN and stated precisely enough to map onto that menu.

---

## 1. Scorecard (1–5; 3 = borderline). Consensus of the three passes.

| Dimension | Score | One-paragraph justification |
|---|---|---|
| **Novelty** | **3** | The genuinely new pieces are the *supply-gradient attribution* (Eq. 5, autograd-verified closed form), the *endogenous supply channel* that makes a reroute's presence differentiable inside the objective, and the *structural leveling-down diagnosis* (§3.3) that motivates it. That is a real contribution. But the surface reads as a **combination of two off-the-shelf tools** — bounded signed-gradient perturbation (ST-iFGSM lineage) + instance reweighing (Kamiran–Calders 2012) — applied to trajectory data. R3 will anchor "incremental" here and R1 will agree the *metric* (partial-R²) is standard; only R2 sees the supply channel as clearly novel. Splits the panel — hence borderline. |
| **Technical quality** | **4** | Strong. Closed-form F_causal via the hat matrix with a materialization-free evaluation identity; FWL-theorem justification for the partial-R² reading; exact per-unit attribution from projection idempotence; O(N log N) Gini; a closed-form ∂F_causal/∂S checked against autograd. The metric firewall (three rings, claims labelled by ring) is above the bar for the venue. Deductions only for the fidelity term being *inert* (near-zero gradient — see obj. 9) and effect sizes being small. |
| **Experimental rigor** | **3** | Real strengths: paired-seed design, a clean frozen-trim ablation, dose–response monotonicity, and **two edit-specificity controls** (random-subset, most-fair-select) that most fairness papers omit. Real weaknesses that cap it at borderline: n=5/6 seeds with a Wilcoxon floor of 0.031 and no multiplicity correction across dozens of reported tests; **no fairness-method baseline**; tiny datasets (50-driver SZ, 12-driver SF); and the external-validity city fails the headline lift-up claim under its natural metric (obj. 3). |
| **Clarity** | **4** | Genuinely well-written and, unusually, *honest* — disclosures are foregrounded, not buried. Costs: very dense (§3 is notation-heavy), and the flagship metric is named `F_causal` while the text spends a paragraph explaining it is **not** causal — an avoidable friction (obj. 8). |
| **Reproducibility** | **3** | Methodology is detailed enough to reimplement (hyperparameters, floors, budgets, closed forms all present). But: the primary corpus is a proprietary Shenzhen cGAIL sample; there is **no reproducibility/code-availability statement in the manuscript**; and n=5/6 makes exact-number reproduction seed-fragile (the paper even schedules an s10 replication as a hedge). Cabspotting (SF) is public, which helps. |

**Likely disposition:** borderline (weak-accept ↔ weak-reject), decided by whether reviewers accept the
"data-augmentation, editing-quality-baselines-by-design" framing or read it as scope-narrowing to dodge a
fairness comparator. The writing quality and honesty pull toward accept; novelty framing and the SF tension
pull toward reject.

---

## 2. Top objections, ranked by rejection risk

### 1. 🔴🟡 "This is just data augmentation / an engineering combination of known tools" (R3, novelty)
- **Attacks:** the framing throughout — §1 "third position," Contributions list, §3.4 (perturbation = ST-iFGSM
  repurposed), §3.6 (reweighing transplanted). Compounded by **dataset scale**: a 50-driver Shenzhen subsample
  and a 12-driver SF subsample (§4.1).
- **Rebuttal (existing):** the supply-gradient attribution + endogenous supply channel is not in ST-iFGSM or
  in reweighing; the leveling-down *diagnosis* (§3.3) is a standalone conceptual result; the density-matching
  rationale (full 536-taxi fleet saturates the supply measure) is a defensible reason the samples are small,
  and cGAIL precedent uses comparable driver counts.
- **Fix:** 🟡 sharpen the novelty paragraph in §1 so the *supply-endogenous editing* and the *structural
  diagnosis* lead, and the ST-iFGSM/reweighing lineage reads as "repurposed machinery," not "the method."
  Move the "combination of known tools" honesty into related work, not the contributions framing.
  🔴 *if the menu can afford it:* a single larger-fleet run (or a supply-measure rescaling that admits more
  drivers without saturation) would blunt the scale objection more than any prose. Flag as high-value/hard.

### 2. 🔴 No fairness-method baseline — only editing-quality + oversampling arms (R2/R3, rigor)
- **Attacks:** §4.5 and the contributions claim. All five baselines (iFGSM, FGSM, random jitter, targeted
  oversampling, placebo) are explicitly *non-fairness* baselines. There is **no comparison to an actual
  fairness intervention** — not the in-processing regularizer the paper itself cites as its closest neighbor
  (Zheng 2023, §2), not a reweighing-only baseline, not generative repair.
- **Rebuttal (existing):** the arms are "editing-quality baselines by design" (Meeting-41 framing) — a fairness
  competitor would optimize the very objective FAMAIL firewalls, so head-to-head on F_causal is circular. The
  oversampling arm *is* the naive fairness-by-fabrication comparator and FAMAIL beats it (+0.0226 vs +0.0153 at
  zero vs 10.5% inflation).
- **Why it still bites:** "circular on F_causal" does not excuse the absence of a comparator **on the external
  ring** (DP/DI/Theil), where FAMAIL's whole pitch lives. A reviewer wants to see FAMAIL vs an in-processing
  fairness-regularized BC scored on the *same external metrics before→after*.
- **Fix:** 🔴 **RUN** — add one fairness-intervention baseline (in-processing reweighing or a fairness-penalty
  BC à la Zheng 2023) evaluated on the external ring (DP/DI/Theil) on Shenzhen, matched budget. This is the
  single most rebuttal-proof addition available and maps cleanly onto the runs menu.

### 3. 🔴🟡 San Francisco fails the lift-up claim under its natural metric; rescued by switching metrics (R2/R3, external validity)
- **Attacks:** §4.7 "The supply channel replicates; the ratio does not." The external-validity city's total
  mean(Y|disadv) moves **net-negative (−0.0324, CI excludes 0)**; the demand channel is significantly negative
  (−0.0533). The paper keeps the headline alive only via the "external-metrics reading."
- **Rebuttal (existing):** the *supply channel itself* is positive-significant in SF (+0.0209, CI [+0.0122,
  +0.0300]) — the mechanism replicates; DP/DI/Theil all improve; the falling ratio is exactly the demand
  endogeneity §3.3 predicts (serving suppressed demand raises the denominator).
- **Why it still bites:** to a skeptic this is **post-hoc metric selection** — the pre-registered ring-(ii)
  quantity goes the wrong way and the paper leans on ring-(iii) to save it. This is also the **open PI framing
  decision** (`TODO(PI-framing)`, §4.7). Whichever reading Zhang picks, a reviewer sees the other.
- **Fix:** 🔴 **RUN** — plumb the **tier-2 distinct-taxi supply recount for the SF pipeline** (§4.7 states it is
  "not plumbed"). Shenzhen's most convincing lift-up evidence is tier-2 (+0.0411); giving SF the same
  distinct-taxi recount would let SF stand on supply-side evidence *independent of the contaminated ratio*,
  converting the tension from a weakness into a replicated mechanism. Highest-leverage SF run on the menu.
  🟡 regardless of the run, state the two readings as a **finding about demand endogeneity**, not as an
  unresolved fork the reader must referee.

### 4. 🟡 Downstream allocation boundary — trained policies still under-serve disadvantaged areas (R3, practical value)
- **Attacks:** §4.4 "The honest boundary." Rolled-out policies allocate pickups **away** from disadvantaged
  areas (−0.0033 at w30), only ~33% attenuated vs demand-only, **not reversed**; seeking-state shares n.s.
- **Rebuttal (existing):** the paper's *claims* are data-level lift-up + metric propagation, both of which
  hold; the allocation boundary is disclosed, not hidden, and motivates training-side constraints as future
  work. Honesty here is a credibility asset.
- **Why it still bites:** the method's ultimate purpose is fairer *deployed allocation*, and the deployed
  policy still tilts against the disadvantaged group. A reviewer can frame the whole contribution as "improves
  audit metrics but not the behavior the audit is a proxy for."
- **Fix:** 🟡 pre-empt in §1/§5 by scoping the claim explicitly to *demonstration-level and metric-level*
  fairness and naming training-side allocation constraints as the acknowledged open problem — so the boundary
  reads as delimited scope, not a buried failure. (A training-side-constraint run would answer it but is a
  whole second method — out of scope for this cycle; do **not** promise it as a result.)

### 5. 🟡 Leveling-down asymmetry — the "lifts up" gain is ~16× smaller than the level-down (R1, fairness)
- **Attacks:** §1 and §4.2. On the headline axis the advantaged mean falls **−0.837** while the disadvantaged
  mean rises **+0.0529** — i.e. ~94% of the gap closure is still *from the top*. The supply-channel portion
  that is the actual "lift-up" is +0.0176 (tier-1). A leveling-down-focused reviewer will say the paper trades
  one clean leveling-down editor for one that is *mostly* still leveling down.
- **Rebuttal (existing):** the paper's claim is deliberately narrow — the **first statistically robust
  *positive* movement** of the under-served group's absolute service, where demand-only gives exactly zero
  (7.0734 → 7.0734). "Some genuine lift-up" is a categorically different result from "none," even if small,
  and every external instrument improves.
- **Fix:** 🟡 report the up:down ratio honestly in-text and frame the contribution as *"breaks the
  zero-lift-up barrier,"* not *"lifts up."* Do not let §1 imply the gap closes primarily from the bottom. The
  extended frontier shows lift-up cannot be increased without losing F_causal significance, so this is
  intrinsic — own it rather than let a reviewer surface it.

### 6. 🔴 Statistical thinness — n=5/6, p floored at 0.031, no multiplicity correction (R2, rigor)
- **Attacks:** §4.1 protocol; every downstream/variance claim. n=6 Wilcoxon floors at 0.03125 (only sign
  unanimity), n=5 cannot reach p<0.05; the conclusion concedes "no single test survives multiple-comparison
  correction."
- **Rebuttal (existing):** the paper does not rest on any single p — it uses dose–response monotonicity, CIs,
  and null controls, and treats p=0.031 only as a sign-unanimity certificate. That is the right way to read a
  seed-floored design and is stated plainly.
- **Why it still bites:** a rigor-first reviewer still wants ≥1 headline effect that clears p<0.05 *with*
  correction. It's cheap insurance.
- **Fix:** 🔴 **RUN** — raise the headline downstream WBC suite (SZ, edited arm, w30) to **n≥10 (target n=20)**
  paired seeds so the Wilcoxon floor drops below 0.05 and at least the flagship recovery survives Holm/BH
  correction. Low-risk, directly answers the objection, maps to the menu.

### 7. 🟡 The external metrics are not independent corroboration (R1, metric validity)
- **Attacks:** §4.1/§4.2 "every established instrument improves." DP (=the signed gap of the two group means),
  DI (=their ratio), and the two group *levels* are all **algebraic functions of the same two numbers**
  (mean(Y|adv), mean(Y|disadv)). Only Theil is structurally distinct (and it is between-region). So "three
  external instruments agree" is largely **one movement viewed three ways**.
- **Rebuttal (existing):** the paper already discloses DP≡gap and reports it once; Theil is genuinely
  independent (entropy, grouping-free) and moves the same way; the levels are reported separately so the reader
  sees the raw movement. The instruments are *standard*, which is the point — improvement on measures the
  objective never optimized.
- **Fix:** 🟡 add one sentence conceding DI/DP algebraic dependence and lean the "never-optimized" weight on
  **Theil + the per-group levels + the tier-2 distinct-taxi recount** (the structurally independent evidence),
  rather than on a count of instruments. Cheap, and it disarms the sharpest R1 line.

### 8. 🟡 `F_causal` names an associational quantity "causal" (R1, clarity/validity)
- **Attacks:** §3.2. A partial-R² of a cross-sectional regression over ~10 district-level (ecological)
  demographic profiles, named `F_causal`, with a paragraph explaining the name is "historical."
- **Rebuttal (existing):** fully disclosed; it is used only as an *optimization label*; no individual-level or
  counterfactual claim is made anywhere.
- **Fix:** 🟡 seriously consider renaming to **`F_demo`** (already floated internally) or `F_assoc`. The honesty
  is currently spent *defending a misleading name*; a rename spends nothing and removes an easy reviewer jab
  and a genuine misreading risk. If the PI wants continuity with prior notation, keep `F_causal` but add a
  one-line footnote at first use rather than a full caveat paragraph.

### 9. 🟡 The three-term objective is effectively one term; fidelity is decorative; a control leaks (R3, rigor)
- **Attacks:** §3.2 (α=(0.1,0.8,0.1), fidelity gradient "near zero"), §4.6 (most-fair-select control is
  **significantly positive on 2 of 3 feature sets**, +0.0054/+0.0072 at w30).
- **Rebuttal (existing):** fidelity is honestly framed as a *guardrail* (constrains via the ε-ball/king-move
  repair + gate, doesn't steer); F_spatial is a secondary regularizer; and the edited arm still exceeds the
  leaking control by ≥3× at every dose, which is the edit-specificity claim under its hardest test — reported,
  not averaged away.
- **Fix:** 🟡 in §3.2, state up front that under bounded edits the objective is *F_causal-dominant by design*
  and F_fidelity/F_spatial are constraint/regularizer roles — so a reviewer doesn't "discover" it. For the
  control leak, keep the honest ≥3× framing but make sure §1 doesn't claim the controls are *null* (they are
  null on the primary set, positive-but-dominated elsewhere).

### 10. 🟡 Reproducibility statement + a related-work gap (R3)
- **Attacks:** no code/data-availability statement in the manuscript; the operational **fair ride-hailing
  dispatch / online fair matching** literature is absent from §2 (which covers fairness-ML, transport-equity
  Gini, IL-for-mobility, adversarial/recourse, leveling-down — but not fair *assignment/dispatch*).
- **Rebuttal (existing):** methodology detail is high; Cabspotting is public; the ledger discipline exists in
  the repo even if not in the paper.
- **Fix:** 🟡 add a short reproducibility statement (anonymized code, public SF data, ledger-backed provenance)
  and 1–2 sentences + citations positioning FAMAIL against fair-dispatch/matching work (data-side editing
  vs. runtime assignment). Both are low-cost and remove free reject-reasons.

---

## 3. "What would make this an accept" — per reviewer

**R1 (fairness-in-ML).** Rename `F_causal` (obj. 8); concede DI/DP algebraic dependence and re-anchor the
never-optimized claim on Theil + levels + the tier-2 recount (obj. 7); and report the leveling-down
up:down ratio honestly while framing the contribution as *breaking the zero-lift-up barrier* (obj. 5). With
those three prose moves the fairness story becomes airtight and honest, and R1 flips to accept — the
leveling-down diagnosis is exactly the kind of result this reviewer rewards.

**R2 (mobility/IL empiricist).** Two runs decide it: (a) one **fairness-intervention baseline on the external
ring** (obj. 2), and (b) **SF tier-2 distinct-taxi recount** so the second city stands on replicated supply
evidence rather than a rescued metric (obj. 3). Bumping the headline downstream suite to n≥10 (obj. 6) is the
cheap third. With a fairness comparator and a clean SF replication, R2 sees a rigorous, well-controlled
empirical paper and accepts.

**R3 (skeptical senior).** Needs the novelty reframed so the *supply-endogenous editing + structural
leveling-down diagnosis* lead and the ST-iFGSM/reweighing lineage reads as repurposed machinery (obj. 1);
the allocation boundary scoped as delimited claim rather than buried failure (obj. 4); and the reproducibility
statement + fair-dispatch related work added (obj. 10). R3 will never love the dataset scale, but with the
contribution correctly sized and the fairness baseline from R2's list present, the "just data augmentation"
thesis loses its footing and R3 moves from reject to borderline-accept.

---

## Cross-reference for the cut recon (do NOT cut these)
Objections above depend on: §3.3 leveling-down diagnosis (obj 1,5); §4.2 channel decomposition / Table 2
(obj 5,7); §4.4 allocation-boundary paragraph (obj 4); §4.6 control-leak disclosure + Table 6 (obj 9); §4.7
SF section (obj 3). The cut recon treats each as **load-bearing** — compress prose at most, never delete.
