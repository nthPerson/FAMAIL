# Fairness metrics & the editor — theory and resources

This doc gives the **core formulas + intuition** for the two fairness metrics, the two fidelity
metrics, and the editing algorithm. Full derivations live in the methodology docs linked in
**Resources**; this is the reference a reviewer or slide-builder needs to state the claims correctly.

---

## F_causal — demand-adjusted demographic fairness (primary metric)

**Question it answers:** *do demographics explain the service a neighborhood gets, beyond what its
demand already explains?*

It is a two-stage (double-regression) construction over `N` active `(cell, time)` units:

- **Stage 1 — control for demand.** Fit a power-basis map `g₀(D)` from per-unit demand `D` to the
  supply-to-demand ratio `Y = S / max(D, DEMAND_FLOOR)`, and take the residual `R = Y − g₀(D)`. The
  power basis is `g₀(D) = β₀ + β₁/(D+1) + β₂/√(D+1) + β₃√(D+1)` (linear-in-parameters, OLS-fit).
- **Stage 2 — regress demographics on the residual.** With `X̃` the z-scored demographics,
  `H = [1, X̃](·)⁺` the projection onto [intercept, demographics], and `M = I − 11'/N` the centering
  matrix,

  ```
  F_causal = R'(I − H)R / R'MR = 1 − r²_demo
  ```

**Intuition (1 = fairest):** `r²_demo` is the share of the demand-adjusted service residual that
demographics explain. If demographics explain a lot (`r²_demo` high), service is systematically
predicted by neighborhood composition → **unfair**; `F_causal = 1 − r²_demo` is then low. Boundary
cases: `R ∈ span(X) ⇒ F_causal = 0` (fully unfair), `R ⊥ X ⇒ F_causal = 1` (fully fair). The
complement orients "maximize F_causal" = "maximize fairness," matching the editor's objective.

**Per-cell attribution (drives the edit).** Because `M` and `(I − H)` are idempotent, `r²_demo`
admits an *exact* per-unit decomposition:

```
r²_demo = Σ_i [ (MR)_i² − ((I−H)R)_i² ] / R'MR
```

Each term is unit `i`'s contribution to the demographic-explained variance — a mathematically exact
partition of the fairness deficit, not a heuristic weight. A signed variant multiplies by
`sign((HR)_i)` to separate over- from under-served units. This attribution is what tells the editor
*which* pickup cells to move.

**Caveats (load-bearing — keep on any F_causal slide):**
- **Associational, not causal.** It is an associational quantity — the partial R² of a
  cross-sectional OLS on observational demographics — with no identification and no counterfactual.
  The "causal" in the name is historical.
- **10 district-level DOF (Shenzhen).** Demographics resolve to 10 district profiles, so the
  regression has few degrees of freedom and an **ecological-fallacy** exposure (district-level
  association ≠ individual-level effect).
- **Naming.** A rename `F_causal → F_demo` (to drop the causal connotation) is a **pending PI
  decision**; the metric is unchanged, so this doc keeps `F_causal` + the caveat.

---

## F_spatial — spatial service equity (secondary metric)

A Gini-based measure of how evenly service is distributed across active units (1 = fairest):

```
F_spatial = 1 − 0.5 · ( Gini(DSR) + Gini(ASR) )
```

where `DSR = pickups / active_taxis` and `ASR = dropoffs / active_taxis` per unit. A differentiable
pairwise Gini is used so it can enter the gradient objective. F_spatial is **demographic-independent**
(it uses spatial attribution, grid channel 0), which is why it is reported as a secondary metric and
why the data cleanup (a spatial filter) is analyzed against it.

---

## Fidelity — is an edited trajectory still realistic?

Realism is measured on **two complementary axes**, because a source can look realistic on one and
collapse on the other:

- **Fidelity-A (identity).** A **frozen, driver-conditioned 3-stream Siamese discriminator**
  (ST-SiameseNet, HuMID-style): a seeking-trajectory BiLSTM + a driving-trajectory LSTM + an 11-dim
  driver profile, with shared weights, embedded and compared by an MLP that outputs
  `P(same driver | trajectory₁, trajectory₂)`. It is trained once and **frozen** during editing —
  it supplies gradient signal but is never updated. Fidelity-A asks *does the edited trajectory still
  read as its driver?* **Profile-dominance property:** because the editing use case shares the same
  driver's profile across both branches, the discriminator can achieve its score largely from the
  profile stream; empirically its gradient w.r.t. the edited pickup cell is ~0 (Shenzhen 4.7e-6, SF
  2.6e-11). So Fidelity-A certifies **identity preservation**, not fine-grained trajectory-shape
  realism — a property of the whole mechanism on both cities (see [`07_limitations.md`](07_limitations.md)).
- **Fidelity-B (distributional).** A **discriminator-free** Jensen–Shannon divergence between the
  edited and raw distributions of trajectory statistics (length, displacement, coverage, radius of
  gyration, net displacement, terminal cell). It asks *do the trajectory distributions still match
  real data?* — the axis that exposes a generator that has collapsed (0 = identical to raw; larger =
  more divergent).

Inside the editor, the objective's fidelity term (`F_fidelity`) is the discriminator similarity
score, acting as a **regularizer** that discourages arbitrarily large edits. In the four-source
data-quality evaluation, Fidelity-A and Fidelity-B are the two realism metrics scored per source.

---

## The editor — attribution-guided ST-iFGSM

FAMAIL adapts the **Spatio-Temporal iterative Fast Gradient Sign Method (ST-iFGSM)** — originally an
adversarial-attack technique — as a fairness-editing tool. The per-(cell, time) attribution above
selects the highest-deficit pickup cells; the algorithm then moves those pickups in the direction that
most improves the combined objective.

**Objective (maximized; each term in [0, 1], weights sum to 1):**

```
L = α_spatial · F_spatial + α_causal · F_causal + α_fidelity · F_fidelity
```

The headline runs use **causal-emphasis weights α = (0.2, 0.7, 0.1)** — the causal axis is the
objective, spatial and fidelity are minor terms.

**The signed-gradient step (per iteration):**

```
δ = clip( step · sign(∇_p L), −ε, ε )      # sign-gradient, scale-independent
δ_total = clip( δ_total + δ, −ε, ε )        # cumulative L∞ bound
```

The `sign(·)` makes the step independent of gradient magnitude; the cumulative clip enforces an
**ε-grid-cell L∞ bound** — a pickup can move **at most ε cells** from its original location, no matter
how many iterations run. The headline runs use **ε = 2**. A temperature-annealed soft cell assignment
bridges the continuous perturbation to discrete grid cells (broad early exploration → precise late
assignment). Because the edit is small and bounded, it stays inside the driver's identity signature —
which is why F_fidelity barely moves under it.

---

## Resources

**Internal methodology (full derivations & implementation):**
- `famail_temporal/docs/F_CAUSAL_METHODOLOGY_NOTES.md` — the F_causal formulation, power basis,
  DEMAND_FLOOR rationale, diagnostics.
- `famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md` — the fairness decomposition /
  per-unit attribution.
- `famail_temporal/fairness/README.md` — the pooled-metric module (spatial Gini, causal Option B,
  attribution sum-property).
- `docs/mathematical_foundations.md` — consolidated mathematical foundations.
- `docs/site/methodology/{objective-function, discriminator, algorithm, soft-cell-assignment}.md` —
  the objective, the ST-SiameseNet discriminator, ST-iFGSM, and the soft-assignment discretization.
- `famail_temporal/docs/TRAJECTORY_EDITING_METHODOLOGY.md` — the editing methodology end-to-end.

**External lineage** (bibliographic references verified 2026-07-08; full literature-grounded motivation +
reviewer defense: [`../objective-motivation/`](../objective-motivation/README.md)):
- **cGAIL** — the conditional generative-adversarial imitation-learning base for taxi trajectories
  (Zhang, Li, Zhou & Luo; *IEEE ICDM* 2019; journal *IEEE Trans. Big Data* 8(5):1288–1300, 2022).
- **ST-SiameseNet / HuMID** — the driver-identity discriminator this fidelity model follows
  (Ren, Pan, Li, Zhou & Luo; *KDD* 2020).
- **FGSM / iFGSM** — the signed-gradient editing step (Goodfellow, Shlens & Szegedy, *ICLR* 2015;
  Kurakin, Goodfellow & Bengio, *ICLR Workshop* 2017); the spatio-temporal instantiation is **ST-iFGSM**
  (Hu, Zhang, Li, Zhou & Luo; *KDD* 2023).
- **Frisch–Waugh–Lovell theorem** — the partial-regression identity underpinning F_causal's
  residualize-then-project construction (Frisch & Waugh, *Econometrica* 1933; Lovell, *JASA* 1963).

---

## Sources / provenance

Formulas and properties are quoted from `famail_temporal/docs/F_CAUSAL_METHODOLOGY_NOTES.md`,
`famail_temporal/fairness/README.md`, and `docs/site/methodology/{objective-function, discriminator,
algorithm}.md`. The profile-dominance / fidelity-gradient property is documented in
`PAPER/second-dataset/FINDINGS.md` §5.4 and `PAPER/second-dataset/tables/fidelity_sensitivity.csv`.
No new numbers are introduced here; experimental values live in
[`05_results_shenzhen.md`](05_results_shenzhen.md) and [`06_results_sf.md`](06_results_sf.md).
