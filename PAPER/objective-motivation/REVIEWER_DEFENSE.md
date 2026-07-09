# Reviewer objections & rebuttals — the FAMAIL objective

Anticipated KDD-reviewer objections to the objective and editor, with literature-grounded rebuttals. This is
rebuttal-prep, not paper prose; citations resolve in [`REFERENCES.md`](REFERENCES.md). The two objections a
fairness reviewer is most likely to press — **leveling-down** and **demand endogeneity** — get the fullest
answers and are developed in [`LEVELING_DOWN.md`](LEVELING_DOWN.md).

---

### Objection: "The `F_causal` term is called *causal* but it isn't causal."
**Rebuttal.** Conceded explicitly. `F_causal` is an **associational** quantity — the partial R² of a
cross-sectional OLS on observational demographics — with no identification and no counterfactual. Frisch–Waugh–
Lovell (Frisch & Waugh 1933; Lovell 1963) is an algebraic identity about *conditional association*, not
causation. We state this once, up front, and a rename to `F_demo` is under consideration. The name is
historical; the metric is unchanged.

### Objection: "District-level demographics invite the ecological fallacy."
**Rebuttal.** Agreed as a scope limit. Demographics resolve to ~10 district-level profiles, so associations are
ecological and few-DOF. We report district-level associations only and make no individual-level claims; this is
stated as a limitation, not hidden.

### Objection: "Why Gini for `F_spatial`, rather than Theil or Atkinson?"
**Rebuttal.** Gini is the standard, interpretable, parameter-free inequality index in transportation-equity
research (Hörcher & Graham 2021; Karner et al. 2024). Theil is group-decomposable and Atkinson encodes an
inequality-aversion parameter, but since `F_causal` already isolates the demographic axis, Theil's
between/within decomposability advantage is partly redundant here. We keep Gini and report Theil/Atkinson as
robustness alternatives (Karner et al. 2024).

### Objection: "A frozen discriminator can be gamed by the editor."
**Rebuttal.** True in principle, but the operative realism constraint is not the discriminator — it is the
bounded ε-L∞ edit (ε = 2 cells), which limits exploitation far more than a live adversary could, plus the
discriminator-free Jensen–Shannon-divergence check that guards against distributional collapse. The frozen
discriminator is retained as a guardrail (its gradient w.r.t. the edited pickup is ≈0), not as the primary
defense; freezing avoids the instability/mode-collapse of a live adversarial game (the GAN analogy noted by
Ho & Ermon 2016).

### Objection: "iFGSM is an *attack* method — using it here is a gimmick."
**Rebuttal.** The mathematics — bounded signed-gradient ascent on a differentiable objective — is
method-agnostic as to intent. Repurposing counterfactual-perturbation tooling *constructively* is already
legitimized by algorithmic recourse (Ustun et al. 2019; Wachter et al. 2018), which uses the same minimal,
bounded perturbations to prescribe beneficial changes. Reinterpreting the ε-bound as an identity-preservation
budget rather than an attack-stealth budget is a clean conceptual transfer, not a gimmick.

### Objection: "Is ε = 2 arbitrary?"
**Rebuttal.** No — ε is tied to the empirical spatial scale of the driver signature and to the 5×5 supply
aggregation window (a pickup must stay inside the driver's identity signature). We report editor sensitivity to
ε (and to τ and the smoothing-window size) via fidelity / JS-divergence curves; empirically the editing ceiling
is intrinsic to the ~1–3% editable slice, not to the ε choice.

### Objection: "Upweighting the edited demonstrations is ad hoc."
**Rebuttal.** It is grounded in **instance reweighing for fairness** (Kamiran & Calders 2012) — a standard
pre-processing lever — transplanted to imitation learning, and it is validated against two controls: random
reweighting and select-already-fair reweighting. Only *edit-specific* upweighting recovers the fairness, which
rules out "oversampling per se" as the explanation.

### Objection: "Fairness improves only by leveling down — reducing the advantaged, not lifting the disadvantaged."
**Rebuttal.** We show this and argue it is the **constrained optimum** of a demand-only, frozen-supply problem,
not an optimizer bug: attribution selects over-served cells, the demand lever is near-inert on the under-served
side, and the true inequity is supply-side while supply is frozen. Normatively, over-service reduction has
defensible egalitarian standing (Parfit 1997; Temkin 1993), and the fair-ML literature treats leveling-down as
a design property, not an inevitability (Mittelstadt et al. 2024), sometimes constraint-forced (Pinzón et al.
2022). Following the leveling-up prescription and Zietlow et al.'s (2022) augmentation finding, the supply-side
lever is our stated future direction. Full framing: [`LEVELING_DOWN.md`](LEVELING_DOWN.md).

### Objection: "Adjusting for demand launders away real inequity (demand endogeneity)."
**Rebuttal.** This is the sharpest objection and we address it directly rather than defensively. Recorded demand
is **endogenous**: pickups in under-served areas are suppressed by historical under-supply, so demand is a
censored signal of true need (the feedback-loop phenomenon of Ensign et al. 2018 and Lum & Isaac 2016).
Conditioning on such demand can under-detect inequity — and the evidence is internal: ~93% of poor-area units
sit at or below the demand floor, where the editor sees ≈no residual and never selects them. We therefore frame
demand-adjustment's endogeneity as the metric's central limitation, note that the metric's blind spot and the
editor's leveling-down are the *same* phenomenon, and position a supply-side lever as the principled remedy
([`LEVELING_DOWN.md`](LEVELING_DOWN.md)).

### Objection: "The fixed weights (0.2, 0.7, 0.1) are unjustified."
**Rebuttal.** The weights follow an explicit selection criterion — maximize `ΔF_causal` subject to
`ΔF_spatial ≥ 0` — and were adopted after matching the pure-causal `(0, 1, 0)` gain without gaming a single
metric (`ΔF_causal = +0.0128`, `ΔF_spatial = +0.0003`). The weighting reflects the objective's gradient
geometry: `F_causal` drives ~97.5% of gradient-sign decisions, `F_spatial`'s gradient is ~20× smaller, and
`F_fidelity` is dormant at ε = 2. A `(ΔF_spatial, ΔF_causal)` Pareto sweep over the weight simplex is reported
as sensitivity to confirm the frontier around the adopted point.
