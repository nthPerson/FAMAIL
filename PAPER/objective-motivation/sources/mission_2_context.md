# FAMAIL — objective-function motivation: deep-research context brief

> **You are a deep-research agent.** Your job: **find and synthesize supporting academic literature for
> the design of the FAMAIL objective function, and draft the "why + how" narrative** that motivates each
> component for a KDD methods paper. This brief is **self-contained** — you have no access to the FAMAIL
> codebase, so everything you need is below. Produce a **cited markdown report** in the format specified
> in §7. Cite only **real, verifiable** works (foundational + recent, ~2015–2025); do not fabricate
> references. Flag gaps where no strong prior work exists — a gap is a finding, not a failure.

---

## 1. What FAMAIL is (orientation)

**FAMAIL** (Fairness-Aware Multi-agent Imitation Learning) is a method to make **urban taxi-trajectory
data fairer** along a demographic axis **while keeping it realistic**, so that demand/dispatch models
learned from it by imitation do not inherit and amplify demographic service inequity.

**The problem.** Taxi/ride-hail service is a public good with private allocation. In real fleets, where
taxis pick up and drop off correlates with neighborhood characteristics (housing price, income,
migrant/hukou share). Demand models are commonly learned by **imitation** from historical trajectories;
a model that faithfully imitates biased data reproduces the bias, and when deployed to guide dispatch
can **amplify** it (sending supply where it already went). An intervention that only touches the model,
leaving the data untouched, fights the training signal.

**The approach — edit, don't generate.** Rather than generating synthetic "fair" trajectories (which
risks distributional collapse / loss of human fidelity, and rewrites the already-fair majority), FAMAIL
**edits a small, attribution-targeted slice** of the *real* trajectories: it identifies the pickups
where demographic unfairness concentrates and **relocates those pickup cells within a tiny bounded
radius**, leaving the rest of the real data intact. It is positioned as a **fairness-oriented
data-augmentation / pre-processing** method, paired with **upweighting** the edited demonstrations in
downstream training so the fairness propagates instead of being averaged away.

**The data (for grounding — you do not need to reproduce it).** Primary dataset: a 50-driver sample of
Shenzhen taxi GPS trajectories (seeking + driving segments), discretized to a **48×90 spatial grid**
(0.01° cells) × **24 hourly** time buckets. The unit of analysis is an **active `(cell, time)` unit**
(N ≈ 34,500). Each unit has: **pickups** (demand proxy), **dropoffs**, **active-taxi count** (supply
proxy), and **neighborhood demographics** that resolve to **10 district-level profiles**. The primary
demographic feature set is **{housing price, per-capita compensation, migrant/hukou population share}**.
An independent second city (San Francisco Cabspotting, ACS demographics) is used for external validity
with no algorithm change.

---

## 2. The object of study — the FAMAIL objective function

The editor maximizes a weighted sum of three terms, each in `[0, 1]`, weights summing to 1:

```
L = α_spatial · F_spatial  +  α_causal · F_causal  +  α_fidelity · F_fidelity
```

Headline runs use **causal-emphasis weights α = (α_spatial, α_causal, α_fidelity) = (0.2, 0.7, 0.1)** —
the demographic-fairness term is the objective; spatial and fidelity are minor/regularizing terms.
**Higher = fairer/more realistic for all three** (fairness convention: 1 = fairest). Below, each term,
its exact formulation, and the design intent you must find literature to support.

### 2.1 `F_causal` — demand-adjusted demographic fairness (the primary term)

**Question it operationalizes:** *does neighborhood demographic composition explain the service a
neighborhood gets, beyond what its demand already explains?*

A **two-stage double regression** over the `N` active `(cell, time)` units:

- **Stage 1 — control for demand.** Fit a power-basis map `g₀(D)` from per-unit demand `D` to the
  supply-to-demand ratio `Y = S / max(D, DEMAND_FLOOR)`, take the residual `R = Y − g₀(D)`. Power basis
  `g₀(D) = β₀ + β₁/(D+1) + β₂/√(D+1) + β₃√(D+1)` (linear-in-parameters, OLS-fit).
- **Stage 2 — regress demographics on the residual.** With `X̃` the z-scored demographics,
  `H` the projection (hat matrix) onto `[intercept, demographics]`, and `M = I − 11'/N` the centering
  matrix:

  ```
  F_causal = R'(I − H)R / R'MR  =  1 − r²_demo
  ```

  `r²_demo` is the share of the demand-adjusted service residual explained by demographics. If
  demographics explain a lot, service is systematically predicted by neighborhood composition → unfair
  → `F_causal` low. Boundaries: `R ∈ span(X) ⇒ F_causal = 0` (fully unfair); `R ⊥ X ⇒ F_causal = 1`
  (fully fair).
- **Per-cell attribution** (drives the edit): because `M` and `(I−H)` are idempotent, `r²_demo` admits
  an **exact per-unit decomposition** `r²_demo = Σ_i [(MR)_i² − ((I−H)R)_i²] / R'MR`. Each term is unit
  `i`'s contribution to the demographic-explained variance — this localizes *where* the unfairness sits
  and selects which pickups to move.

**Load-bearing caveats (must be handled honestly in the narrative):**
- **Associational, not causal.** It is the partial R² of a cross-sectional OLS on observational
  demographics — no identification, no counterfactual. The "causal" in the name is historical; a rename
  to **`F_demo`** is under consideration. (Research need: how have others named/positioned
  regression-residual demographic-disparity measures without overclaiming causality?)
- **Ecological / few-DOF.** Demographics resolve to ~10 district profiles → few degrees of freedom and
  an **ecological-fallacy** exposure (district-level association ≠ individual-level effect).

**Design choices in `F_causal` needing literature:** (a) *demand-adjustment before* measuring demographic
disparity (vs. raw demographic parity / disparate impact); (b) the **residualize-then-project /
partialling-out** construction (this is the **Frisch–Waugh–Lovell theorem**); (c) using **supply/demand
ratio** as the service/outcome variable; (d) the R²-style "proportion of variance explained by protected
attributes" as a fairness quantity; (e) the exact additive per-unit attribution as an
explainability/localization device.

### 2.2 `F_spatial` — spatial service equity (secondary term)

A **Gini-based** measure of how evenly service is distributed across active units (1 = fairest):

```
F_spatial = 1 − 0.5 · ( Gini(DSR) + Gini(ASR) )
```

where `DSR = pickups / active_taxis` and `ASR = dropoffs / active_taxis` per unit. A **differentiable
pairwise Gini** is used so it can enter the gradient objective. `F_spatial` is
**demographic-independent** (pure spatial distribution).

**Design choices needing literature:** (a) the **Gini coefficient as a service/accessibility-equity
metric** in transportation; (b) supply-normalized service rates (per-taxi demand/supply) as the
equity target; (c) alternatives and their trade-offs — **Theil index, Atkinson index, coefficient of
variation, Palma ratio** — and when each is preferred (e.g. Theil's between/within-group
decomposability); (d) differentiable/relaxed inequality measures for use in gradient-based objectives.

### 2.3 `F_fidelity` — realism regularizer

Realism that the edit must preserve, measured by a **frozen, driver-conditioned, 3-stream Siamese
discriminator** (ST-SiameseNet / HuMID-style): a **seeking-trajectory BiLSTM** + a **driving-trajectory
LSTM** + an **11-dim driver profile**, shared weights, embedded and compared by an MLP that outputs
`P(same driver | trajectory₁, trajectory₂)`. Trained once and **frozen** during editing — it supplies
gradient signal but is never updated. Inside the objective, `F_fidelity` is the discriminator similarity
score acting as a **regularizer** discouraging large edits. (Empirically its gradient w.r.t. the edited
pickup is ≈0 because edits are tiny; it is retained as a realism backstop, and a separate
distributional check — Jensen–Shannon divergence of trajectory-statistic distributions — guards against
generator-style collapse.)

**Design choices needing literature:** (a) **trajectory-user linking / human-mobility identification**
as a fidelity/realism criterion (the HuMID / ST-SiameseNet line, and the broader TUL literature); (b)
using a **frozen (non-adversarial) discriminator as a realism regularizer** vs. a live GAN adversary
(stability, mode-collapse avoidance); (c) Siamese / contrastive metric learning for trajectory identity;
(d) distributional realism metrics for synthetic/edited mobility data (JS/KL divergence of mobility
statistics — radius of gyration, trip length, coverage).

### 2.4 The editor — attribution-guided ST-iFGSM with soft (differentiable) discretization

FAMAIL adapts the **Spatio-Temporal iterative Fast Gradient Sign Method (ST-iFGSM)** — originally an
**adversarial-attack** technique — as a **fairness-editing** tool. The per-`(cell,time)` attribution
selects the highest-deficit pickup cells; the algorithm moves those pickups in the direction that most
improves `L`.

**Signed-gradient step (per iteration), ε-bounded in L∞:**
```
δ       = clip( step · sign(∇_p L), −ε, ε )     # sign-gradient: scale-independent
δ_total = clip( δ_total + δ, −ε, ε )            # cumulative L∞ bound
```
Headline runs use **ε = 2 grid cells** — a pickup moves at most 2 cells from its origin, no matter how
many iterations run. This bounded, targeted edit is what keeps the trajectory inside the driver's
identity signature.

**Soft cell counts / differentiable discretization.** Trajectories live on a **discrete** grid, but
gradient ascent needs a **differentiable** objective. FAMAIL makes cell counts differentiable via
**Gaussian smoothing over a 5×5 neighborhood**, controlled by a **temperature `τ`** (how aggressive the
smoothing is). The 5×5 window is chosen because it matches how the active-taxi supply is aggregated and
retains a usable gradient signal; smoothing over the whole 48×90 grid collapses the signal to a uniform
"mush." **`τ` is itself claimed as a contribution** — a controllable knob for the continuous↔discrete
adaptation. (A temperature-annealed **soft cell assignment** bridges the continuous perturbation back to
a discrete cell: broad early exploration → precise late assignment.)

**Design choices needing literature:** (a) **repurposing adversarial-example methods (FGSM/iFGSM) for a
constructive / beneficial goal** — fairness editing, data augmentation, counterfactual generation —
rather than attack; (b) **signed-gradient, L∞-ε-bounded perturbations** as a *minimal, bounded* edit
principle (imperceptibility ↔ identity-preservation analogy); (c) **temperature-annealed / Gumbel-softmax
/ straight-through** relaxations for differentiating through discrete structure; (d) **Gaussian /
kernel smoothing of spatial histograms** to obtain gradients over grid data; (e) **soft-to-hard
annealing** schedules.

### 2.5 The downstream pairing (context for the augmentation framing)

FAMAIL is a **data-augmentation** method: (1) edit the unfair slice; (2) **upweight** the edited
demonstrations during downstream **behavior-cloning** so fairness propagates (vanilla training averages
it away — a genuine null — while *edit-specific* upweighting recovers it, verified against random and
"select-already-fair" controls).

**Design choices needing literature:** (a) **pre-processing fairness interventions** (editing/massaging
the training data) vs. in-processing / post-processing; (b) **instance reweighing / importance weighting
for fairness** (e.g. reweighing to break the label–protected-attribute association); (c) fairness in
**imitation learning / behavior cloning**; (d) **data augmentation for fairness** generally.

---

## 3. The design decisions to motivate (the crux of "why")

Turn each of these into a literature-backed justification. For each: what established idea supports it,
who did something analogous, and how FAMAIL's choice is standard, novel, or a defensible departure.

1. **Editing real data beats generating synthetic data** for fairness under a realism constraint.
2. **Demand-adjusting service before measuring demographic disparity** (so you don't penalize a
   low-demand poor area, or credit a high-demand one) — vs. naive demographic parity / disparate impact.
3. **Residualize-then-project (FWL / partial regression) as a fairness metric**, yielding an
   R²-style "variance explained by protected attributes."
4. **Gini over supply-normalized service rates** as spatial equity; and the Theil/Atkinson alternatives.
5. **A frozen identity discriminator as a realism regularizer** (trajectory-user-linking as fidelity).
6. **Adversarial-attack machinery (iFGSM) repurposed constructively** for bounded, targeted fair edits.
7. **Bounded (ε-L∞) minimal edits** as the mechanism that preserves human/driver fidelity.
8. **Temperature-controlled soft discretization** to make a discrete grid objective differentiable.
9. **A weighted linear scalarization of competing fairness/realism objectives** (multi-objective
   trade-off; the causal↔spatial terms partially conflict) — and how to justify fixed weights.
10. **Upweighting edited demonstrations** so data-level fairness survives into an imitation-learned model.

---

## 4. Known lineage / anchor citations (find the exact refs + situate FAMAIL)

FAMAIL already builds on these; find the authoritative citations, confirm what each does, and identify
where FAMAIL diverges or contributes beyond them. Then expand outward to adjacent/most-recent work.

- **cGAIL** — conditional generative-adversarial imitation learning for taxi driver trajectories (the
  imitation-learning base FAMAIL's realism model and framing descend from). *Find the exact paper(s).*
- **HuMID / Ren et al.** — the driver-identity ("human mobility identification") discriminator the
  fidelity model follows; and **ST-SiameseNet** for taxi-driver identification. *Find exact refs.*
- **FGSM / iFGSM** — Goodfellow et al. (2015), *Explaining and Harnessing Adversarial Examples*;
  Kurakin et al. (2017), *Adversarial Examples in the Physical World* (iterative FGSM). The
  "ST-iFGSM" the editor adapts. *Confirm and find any spatio-temporal-iFGSM source.*
- **Frisch–Waugh–Lovell theorem** — the partial-regression identity underpinning `F_causal`'s
  residualize-then-project construction.
- **Fairness metrics** — the taxonomy the external-metric evaluation used (Verma & Rubin 2018,
  *Fairness Definitions Explained*; Barocas, Hardt & Narayanan, *Fairness and Machine Learning*) — for
  positioning `F_causal`/`F_spatial` against demographic parity, disparate impact, statistical parity.

---

## 5. Research questions (the core ask — prioritized)

**A. Metric design (highest priority — this is the objective's heart).**
- How is **demand/exposure adjustment** handled in fairness and in transportation-equity metrics? Find
  work that residualizes out a legitimate driver (demand) before attributing disparity to a protected
  attribute. Is FAMAIL's demand-then-demographics double regression standard, novel, or reinventing a
  known device (e.g. "conditional demographic parity", "fairness given legitimate factors")?
- Precedents for **"proportion of outcome variance explained by protected attributes" as a fairness
  measure** (R²/partial-R² style; FWL/partialling-out; also "balance"/independence tests).
- **Spatial / accessibility equity metrics** in transportation: Gini vs Theil vs Atkinson vs Palma;
  supply-demand ratio equity; when each is used; decomposability arguments for Theil.
- How do others build **differentiable relaxations of inequality metrics** (Gini) for optimization?

**B. The editing mechanism.**
- Prior art on **repurposing adversarial perturbations for constructive ends** (fair representation
  editing, counterfactual/recourse generation, data augmentation, "adversarial for good").
- **Minimal / bounded (L∞) perturbation** as a design principle transferred from imperceptibility
  (attacks) to fidelity-preservation (edits).
- **Differentiating through discrete spatial structure**: Gumbel-softmax (Jang et al.; Maddison et al.),
  straight-through estimators (Bengio et al.), temperature annealing, kernel-smoothed histograms.

**C. Realism / fidelity.**
- **Trajectory-user linking (TUL)** and human-mobility identification as a **realism/fidelity criterion**
  for synthetic or edited mobility data; Siamese/contrastive trajectory identity models.
- **Frozen vs. adversarial discriminators** as regularizers (training stability, collapse avoidance).

**D. Method framing / positioning.**
- **Pre-processing fairness** (data editing / massaging / reweighing: Kamiran & Calders; disparate-impact
  remover: Feldman et al.) vs in-/post-processing — where FAMAIL sits.
- **Fairness in imitation learning / behavior cloning**, and **importance weighting for fairness**.
- **Fair urban mobility / ride-hailing equity** applications (closest applied neighbors).

**E. The leveling-down angle (recent finding — §6).**
- The **leveling-down objection** in egalitarian ethics (Parfit) and its treatment in algorithmic
  fairness (fairness achieved by *harming the advantaged* rather than *helping the disadvantaged*).
- Literature distinguishing **"leveling down" vs "leveling up"** interventions, and any formal results on
  when fairness objectives are satisfiable only by reduction.

---

## 6. Recent internal finding to weave into the "why" (leveling-down)

An empirical analysis of the current editor found — and *proved structurally* — that it improves fairness
**only by reducing over-service to advantaged areas, never by lifting service to under-served areas**
("leveling-down"). Root causes: (i) the attribution selects only over-served cells (residual-variance
based); (ii) the demand lever is near-inert on the under-served side (a demand floor + weak gradient);
(iii) **the true inequity is on the supply side, and supply is frozen** — the editor can only move
pickups (demand). An oracle bound shows any *demand-only* editor could raise the under-served group only
by *deleting its recorded pickups* — perverse. So leveling-down is the **constrained optimum** of the
demand-only/frozen-supply problem, not an optimizer failure; a supply-side lever is the future direction.

**Why this matters for Mission 2:** the objective-motivation story should (a) motivate the demand-side
design *and* honestly frame its constrained-optimality, and (b) pre-empt the reviewer's leveling-down
objection with the egalitarian-ethics + algorithmic-fairness literature (question E). Find literature
that lets us present "over-service reduction under a frozen-supply constraint" as a principled,
defensible contribution with a clear supply-side future direction.

---

## 7. Deliverable — what to return

A **markdown report** structured as:

1. **Executive summary** — the 5–8 strongest literature anchors and the one-paragraph "why + how" thesis
   for the FAMAIL objective.
2. **Per-component sections** — one each for `F_causal`, `F_spatial`, `F_fidelity`, the ST-iFGSM editor
   (incl. soft discretization), and the augmentation/upweighting framing. In each:
   - **Supporting literature** — the established methods/ideas that justify or parallel the choice, with
     **full inline citations**.
   - **How it motivates FAMAIL** — a drafted **"why + how" paragraph** usable (with light editing) in the
     paper's objective-motivation section: *why* this component exists and *how* the design follows from
     prior work.
   - **Contrast / novelty** — what FAMAIL does differently and why that is defensible.
   - **Reviewer risks + rebuttals** — the objections a KDD reviewer would raise (e.g. "F_causal isn't
     causal," "why Gini not Theil," "iFGSM is just an attack," "leveling-down"), and the
     literature-grounded answers.
3. **The leveling-down framing** — a short section marshaling the egalitarian-ethics + fairness
   literature to position over-service-reduction as principled, with a supply-side future direction.
4. **Consolidated reference list** — every cited work, full bibliographic detail, grouped by theme,
   flagged foundational vs recent. Note any **gaps** where the design lacks clear precedent (candidate
   novelty claims).

**Framing constraints:** target venue is **KDD** (applied data-mining/ML methods). Keep `F_causal`
described as **associational** (do not assert causal identification). Prefer foundational works +
2018–2025 developments. Every citation must be real and verifiable; where you are unsure, say so.

---

*This brief is a hand-off artifact for an external research session; it is not committed to the FAMAIL
repository. Return your markdown report and it will be folded into the paper's objective-motivation
(Mission 2) work.*
