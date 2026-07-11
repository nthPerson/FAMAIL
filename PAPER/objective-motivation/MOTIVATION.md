# Motivating the FAMAIL objective — a "why + how" for each term

This is the literature-grounded motivation for the FAMAIL editing objective
`L = α_spatial·F_spatial + α_causal·F_causal + α_fidelity·F_fidelity` (maximized; each term in `[0,1]`;
causal-emphasis weights `α = (0.2, 0.7, 0.1)`) and the ST-iFGSM editor that optimizes it. The paragraphs
below are written to be usable, with light editing, in the paper's objective-motivation section. Formulas and
intuition live in [`../argument/03_fairness_theory.md`](../argument/03_fairness_theory.md); this doc supplies
the *why*. All citations resolve in [`REFERENCES.md`](REFERENCES.md) (metadata verified 2026-07-08).

---

## Executive thesis

Demand/dispatch models learned by imitation from historical taxi trajectories inherit — and, once deployed
to guide dispatch, can amplify — the demographic service inequity encoded in the demonstrations they clone.
FAMAIL intervenes at the *data* level rather than the model level, but under a hard realism constraint that
purely generative approaches violate: synthesizing "fair" trajectories risks distributional collapse and
rewrites the already-fair majority. FAMAIL therefore *edits* a small, attribution-targeted slice of the real
trajectories. Its objective decomposes into a primary demographic-fairness term operationalized as a
partial-R² via the Frisch–Waugh–Lovell theorem after adjusting for legitimate demand — the transportation
analogue of conditional statistical parity (Corbett-Davies et al. 2017) — a secondary Gini-based spatial-equity
term drawn from transportation-equity practice, and a frozen trajectory-identity discriminator
(ST-SiameseNet lineage; Ren et al. 2020) that regularizes edits toward realism. The edit repurposes bounded
ε-L∞ adversarial-perturbation machinery (Goodfellow et al. 2015; Kurakin et al. 2017) *constructively*, made
differentiable over a discrete grid by temperature-annealed soft assignment (Jang et al. 2017). Downstream,
edited demonstrations are upweighted so the fairness survives behavior cloning instead of being averaged
away — instance reweighing (Kamiran & Calders 2012) transplanted to imitation learning.

---

## F_causal — demand-adjusted demographic fairness (primary term)

**Why.** Raw demographic parity would flag any neighborhood whose service differs from the mean as unfair —
but taxi service legitimately tracks demand: a business district *should* see more pickups than a sleepy
residential block. FAMAIL therefore measures fairness only in the component of service that demand does *not*
explain. Adjusting for a *legitimate* factor before attributing residual disparity to a protected attribute
is exactly **conditional statistical parity**, which Corbett-Davies et al. (2017) formalize (building on
Kamiran et al. 2013 and Dwork et al. 2012): restrict disparities to those not explained by permitted factors.
The same source cautions that the choice of "legitimate" factors significantly affects the result — a caution
we take up directly in the demand-endogeneity discussion below.

**How.** We first fit a flexible power-basis map from per-unit demand to the supply-to-demand ratio and take
the residual; we then ask how much of that residual neighborhood demographics explain. This
residualize-then-project construction is the **Frisch–Waugh–Lovell theorem** (Frisch & Waugh 1933;
Lovell 1963): the Stage-2 coefficients on demographics after partialling out demand equal their coefficients
in the full multiple regression, so `F_causal = 1 − r²_demo` is an interpretable "proportion of demand-adjusted
service inequity attributable to demographics." Because the centering and hat matrices are idempotent,
`r²_demo` admits an *exact* additive per-unit decomposition — localizing *where* the unfairness concentrates
and selecting which pickups to move. Using how well a protected attribute is predicted from the rest of the
data as the fairness quantity follows the fairness-as-predictability logic of Feldman et al. (2015).

**Caveat — associational, not causal.** `F_causal` is an **associational** quantity: the partial R² of a
cross-sectional OLS on observational demographics, with no identification and no counterfactual. The "causal"
in the name is historical; a rename to `F_demo` is under consideration. Because demographics resolve to ~10
district-level profiles, the association is ecological and few-DOF — we report district-level associations and
avoid individual-level claims (ecological-fallacy exposure).

**Demand is legitimate — but endogenous.** Quotienting out demand assumes demand is an exogenous, legitimate
control. Recorded demand (pickups), however, is itself suppressed by historical under-supply in under-served
areas: latent demand is censored by the very inequity we aim to measure. Conditioning on such demand can
*under-detect* real inequity — the metric's central limitation, and the same phenomenon as the editor's
leveling-down behavior. We treat this head-on in
[`LEVELING_DOWN.md`](LEVELING_DOWN.md) (feedback-loop grounding: Ensign et al. 2018; Lum & Isaac 2016).

**Contrast / novelty.** Conditional statistical parity is normally a binary-classifier constraint on predicted
labels; FAMAIL transplants it to a continuous, spatial supply/demand-ratio outcome and turns the residual
partial-R² into a differentiable *objective* with exact per-unit attribution used for editing. To our
knowledge no prior work uses an FWL-partial-R² of demographics on a demand-residualized service ratio as an
editable fairness objective in transportation; residualization-for-fairness has adjacent precedent, so the
novel contribution is the specific combination, not the primitive.

---

## F_spatial — spatial service equity (secondary term)

**Why & how.** Beyond demographic fairness, we want service to be spatially even relative to the taxi supply
actually present. We include a Gini-based term over supply-normalized service rates — pickups per active taxi
and dropoffs per active taxi — where a lower Gini means service is spread more evenly across active cells.
The Gini coefficient is the most widely used and interpretable inequality index in transportation-equity
research (Hörcher & Graham 2021; Karner et al. 2024), and a differentiable pairwise formulation lets it enter
the gradient objective directly. Because `F_spatial` is demographic-independent, it acts as a general spatial
smoothness regularizer complementing the demographic `F_causal` term.

**Contrast / novelty.** The contribution is not the metric but its *differentiable pairwise* form embedded in
a gradient-based editor. The choice of Gini over Theil or Atkinson is defensible but not dominant — Gini is
parameter-free and interpretable, while Theil is group-decomposable and Atkinson encodes an inequality-aversion
parameter (Karner et al. 2024); since `F_causal` already isolates the demographic axis, Theil's decomposability
advantage is partly redundant here. We report Theil/Atkinson as robustness alternatives.

---

## F_fidelity — realism regularizer

**Why & how.** An edit that improves fairness but produces trajectories no real driver would generate is
useless for training a realistic demand model. We regularize edits with a **frozen** driver-identity
discriminator from the ST-SiameseNet / HuMID family (Ren et al. 2020): because individual mobility signatures
are highly identifying — the premise of the entire trajectory-user-linking literature, from TULER (Gao et al.
2017) through TULVAE (Zhou et al. 2018) and DeepTUL (Miao et al. 2020) — a trajectory that still "reads as" the
same driver after editing has preserved human fidelity. We freeze the discriminator (trained once, never
updated during editing) to avoid the instability and mode collapse of a live adversarial game; Ho & Ermon
(2016) note the GAN analogy in imitation learning that this design deliberately sidesteps. A separate
Jensen–Shannon-divergence check on aggregate mobility statistics guards against generator-style distributional
collapse, following standard mobility-generation evaluation practice (Feng et al. 2020).

**Contrast / novelty.** ST-SiameseNet was built for driver verification; FAMAIL repurposes a frozen instance
as a fairness-editing realism regularizer. Because edits are tiny and ε-bounded, its gradient with respect to
the edited pickup is ≈0 — so it is retained honestly as a distributional *guardrail* rather than an active
driver of the edit, with the JS-divergence check doing the operative collapse-guarding.

---

## The editor — attribution-guided ST-iFGSM with soft, differentiable discretization

**Why & how.** To edit pickups we must move probability mass on a discrete grid in the direction that most
improves the objective, under a hard "stay realistic" budget. Adversarial-example methods solve almost exactly
this problem in reverse: FGSM and iterative FGSM (Goodfellow et al. 2015; Kurakin et al. 2017) compute the
minimal signed-gradient perturbation that changes a model's output, bounded in L∞ so the change is
imperceptible. FAMAIL inverts the *intent* — the same ε-L∞-bounded signed-gradient step now moves a pickup
toward *greater* fairness, capped at ε = 2 grid cells so a pickup can never drift outside the driver's identity
signature no matter how many iterations run. This constructive reuse mirrors **algorithmic recourse** (Ustun
et al. 2019; Wachter et al. 2018), where the same counterfactual-perturbation tooling prescribes minimal,
bounded, beneficial changes; reinterpreting the ε-bound as an *identity-preservation* budget rather than an
attack-stealth budget is a clean conceptual transfer. Because the grid is discrete but gradient ascent needs
differentiability, we make cell counts differentiable via Gaussian smoothing over a 5×5 neighborhood (matching
how active-taxi supply is aggregated), with a temperature `τ` that anneals from broad early exploration to
precise late assignment — the soft-to-hard schedule of Gumbel-softmax and straight-through relaxations (Jang
et al. 2017; Maddison et al. 2017; Bengio et al. 2013). The spatio-temporal instantiation is the group's own
ST-iFGSM (Hu et al. 2023).

**Contrast / novelty.** Two defensible, "to our knowledge" novelties: (1) using an adversarial-*attack* method
as a fairness *data-editing* operator on spatio-temporal mobility (recourse edits an individual feature vector
to flip a decision; FAMAIL edits real trajectory pickups to reduce population-level demographic inequity); and
(2) temperature-annealed soft cell assignment with a 5×5 supply-matched smoothing window as an explicit,
tunable continuous↔discrete bridge.

---

## Downstream pairing — fair data augmentation with upweighting (brief)

Editing the unfair slice is necessary but not sufficient: retraining by vanilla behavior cloning on the edited
dataset averages the small fair slice away and relearns the old bias — a genuine null we verify empirically. We
therefore *upweight* the edited demonstrations during downstream training so the intervention propagates into
the learned policy. This is **instance reweighing for fairness** (Kamiran & Calders 2012) — part of the
pre-processing family alongside disparate-impact removal (Feldman et al. 2015) — transplanted to imitation
learning, and validated against random-reweighting and select-already-fair controls to show the recovery is
*edit-specific*. It targets, upstream on the demonstrations, the same demand-model bias documented downstream
by Zheng et al. (2023), whose socially-aware model reduces the black vs. non-black mean-percentage-error gap
from 0.361 to 0.084 via an in-processing regularizer; FAMAIL intervenes on the training data instead. The
full Pillar-2 experimental results live in [`../argument/04_evaluation.md`](../argument/04_evaluation.md) and
[`../argument/05_results_shenzhen.md`](../argument/05_results_shenzhen.md).

---

## Why these weights — the empirically selected scalarization

FAMAIL combines three competing objectives by **linear scalarization** into a single differentiable `L`.
The adopted weights **`α = (0.1, 0.8, 0.1)`** are selected *empirically*, by sweeping the weight simplex
(fidelity fixed at 0.1) with full trim+lift editing runs and scoring every point on **all three metric
rings** — the optimized metrics, the design-targeted supply/demand family, and the external instruments
(`PAPER/objective-motivation/weight-sensitivity/EXTENDED_FRONTIER.md`; decision record `DECISION.md`,
2026-07-11). The sweep exposes an asymmetry the optimized metrics alone would hide: `ΔF_causal` is **flat**
(within 0.001) across `α_spatial ∈ [0, 0.55]`, while the **supply-channel lift-up of the under-served group
declines monotonically** with `α_spatial`, losing significance beyond `α_spatial = 0.2` — at spatial-heavy
weights the value-of-presence map is dominated by evenness rather than the demographic residual, and the
editor reverts to leveling down. The selection criterion is therefore three-part: *maximize `ΔF_causal`
subject to (i) `ΔF_spatial ≥ 0` and (ii) the supply-channel lift-up remaining significant under both
accounting tiers.* `(0.1, 0.8, 0.1)` is the frontier's best point under this criterion
(`ΔF_causal = +0.0226`, `ΔF_spatial = +0.0061`, lift-up tier-1 `+0.0176` / tier-2 `+0.0411`, both
CI-significant). Two properties make the choice robust rather than delicate: the primary gain is
insensitive to the weights over a wide range, and `F_fidelity` is dormant at ε = 2 (the bounded edit keeps
every move inside the training distribution) — the discriminator constrains, but does not steer, the
optimization. A causal-heavy weighting thus reflects, rather than forces, where the editable *lifting-up*
signal lies; the full three-ring sweep is reported as the sensitivity analysis.
