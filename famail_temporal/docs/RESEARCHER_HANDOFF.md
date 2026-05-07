# FAMAIL Temporal — Researcher Handoff: Trajectory-Modification Algorithm and Fairness Formulations

**Document date:** 2026-05-07

This document is intended as a sanity-check enabler for collaborating researchers with no prior `famail_temporal/` context.

**Status:** Written against config.T = 24, source-data git SHA a532ead.

---

## TL;DR

FAMAIL Temporal optimizes taxi-service equity across a city's active `(cell, time-block)` units using a three-term objective `L = α_s · F_spatial + α_c · F_causal + α_f · F_fidelity`, where `F_spatial` measures Gini-based service exposure equity, `F_causal` measures how much demographics — rather than demand — explain the service rate `Y`, and `F_fidelity` constrains modified trajectories to remain realistic. The trajectory-modification algorithm identifies pickup locations whose per-cell fairness attribution falls below the uniform baseline and reroutes them via a Spatial-Temporal iterative Fast Gradient Sign Method (ST-iFGSM) loop that maximizes `L` over a soft-cell assignment. Per-driver fairness, supply-side modification, and real-time deployment are out of scope.

---

## §1. Project context

FAMAIL Temporal asks whether a city's taxi-service supply is fair across both space and time, and whether trajectories can be algorithmically rerouted to make it fairer.

The research question is: given drivers' historical GPS traces, does service exposure differ systematically across urban areas in ways that correlate with neighborhood demographics rather than demand, and can individual pickup locations be perturbed to reduce that disparity without producing unrealistic trajectories?

The project's contribution is two-part. First, it delivers a fairness audit at `(cell, time-block)` granularity — distinguishing, for example, whether a residential district is underserved in the morning peak versus the evening peak, rather than collapsing service inequality into a single spatial scalar. Second, it provides a trajectory-modification algorithm that identifies which pickups contribute most to unfairness and reroutes them via an iterative gradient-based perturbation loop.

`famail_temporal/` is a ground-up rewrite of prior FAMAIL iterations that operated on 2D spatial grids with four coarse time blocks; explicit hourly granularity at T = 24 is the headline methodological change. The dataset covers 50 drivers in Shenzhen across three months, weekdays only.

Three topics are explicitly out of scope for this project:

- **Per-driver fairness.** The audit and rerouting operate on aggregate `(cell, time-block)` units, not on individual driver allocation.
- **Supply-side modification.** The algorithm shifts pickup locations within existing trajectories; it does not add drivers, reassign shifts, or alter fleet size.
- **Real-time deployment.** All processing is offline and batch; no latency or streaming constraints are assumed.

Three load-bearing claims that the rest of the document defends: (1) a supply-based active-unit mask — not a demand-based one — correctly identifies which `(cell, time-block)` units participate in fairness evaluation; (2) demographics, not demand, are the right causal variable to partial out when measuring service inequity; and (3) a Siamese discriminator pre-trained on real driver traces is a sufficient proxy for trajectory realism.

For the architectural quickstart and the four invariants every module must respect, see `../README.md`.

The remainder of the document defines what 'fair' means in this project (§3–§7) and how trajectories are rerouted (§8).

---

## §2. Dataset and active-unit construction

Fairness is measured over a discrete set of active spatial-temporal units; this section defines that set.

The dataset covers 50 Shenzhen taxi drivers across three calendar months (July–September), weekdays only. The spatial grid is 48 × 90 cells; T = 24 hourly time blocks span a full operating day. Three primary tensors are constructed from the raw GPS records (see [../data/README.md](../data/README.md)):

- `pickup_3d` — mean hourly pickups per `(cell, block)`.
- `dropoff_3d` — mean hourly dropoffs per `(cell, block)`.
- `active_taxis_3d` — mean hourly active taxis per `(cell, block)`.

All three share a unified aggregation rule: sum 5-minute GPS buckets to hourly within each time block, average across hours in the block, then average across qualifying weekdays. The result is a mean-hourly rate for each `(cell, block)`. For the source datasets that feed aggregation, see [../source_data/README.md](../source_data/README.md).

**Active-unit filter.** A `(cell, t)` unit participates in the fairness audit only when all three conditions hold: (1) `active_taxis_3d[c, t] > ACTIVE_SUPPLY_THRESHOLD` (0.5 mean taxis per hour), (2) cell `c` is inside the Shenzhen administrative boundary per `grid_to_district_mapping.pkl`, and (3) every selected demographic feature for cell `c` is finite. The conjunction ensures audit units are geographically valid, operationally reachable, and covariate-complete. The current dataset yields N = 5,834 active units.

**Why supply, not demand, defines the mask.** The filter uses taxi supply, not observed passenger demand, as the reachability criterion. Observed demand is endogenous to historical service patterns: a residential cell chronically under-served may show near-zero pickups not because demand is absent, but because residents gave up on taxi service and found alternatives. A demand-based threshold would conflate "no service territory" with "unfair service territory" — and would specifically excise the cells most relevant to the fairness question. Supply measures whether taxis physically traverse the cell, determined by road networks and geography rather than by service allocation history; `ACTIVE_SUPPLY_THRESHOLD = 0.5` admits cells where taxis *can* serve regardless of whether they *do*. Full rationale is in [F_CAUSAL_METHODOLOGY_NOTES.md](F_CAUSAL_METHODOLOGY_NOTES.md) §5.

**Canonical active-unit ordering.** The active set is enumerated once at preprocess time: cells in row-major order (x = 0..47, y = 0..89), and within each cell the T = 24 blocks in ascending order (0..23). This ordering is serialized in `cache/unit_index_map_*.pkl` and asserted at every load boundary. Every `(N,)` array in the system — pickup counts, supply ratios, fairness attributions, hat-matrix rows — shares this ordering; a length mismatch raises an assertion error before it can propagate silently downstream.

**Demographics.** Each active cell carries three z-scored district-level features: housing price per square metre (`AvgHousingPricePerSqM`), GDP per capita (`GDPperCapita`), and compensation per employed person (`CompPerCapita`). These are standardized via `StandardScaler` before entering the hat-matrix projection in `F_causal`. NaN in any feature disqualifies the cell under condition (3); demographic completeness is part of the active-unit definition, not a post-hoc filter.

Four load-bearing claims a reviewer could contest: (a) supply-based masking cleanly separates "unreachable" from "unfairly served" — the endogeneity argument is the defense; (b) mean-hourly aggregation is scale-consistent — Gini is scale-invariant and `g_0(D)` is re-fit at the same scale; (c) the three-condition conjunction is necessary — dropping any one admits boundary-invalid, taxi-inaccessible, or covariate-incomplete units; (d) three demographic features suffice for the causal audit — parsimonious by design and consistent with prior FAMAIL iterations, but an empirical choice open to extension.

All N-vectors and (48, 90, T) tensors in the rest of the document share the active-unit ordering established here.

---

## §3. The objective at a glance

Three terms compose the optimization objective: two fairness metrics and one realism check.

`L` is the top-level scalar that the ST-iFGSM loop maximizes. It is a weighted sum of three component scalars, each in [0, 1] where higher is better:

```
L = α_s · F_spatial + α_c · F_causal + α_f · F_fidelity
```

Default weights from `../config.py`: α_s ≈ 0.33, α_c ≈ 0.33, α_f ≈ 0.34 (sum ≈ 1). No renormalization is applied inside the objective.

**F_spatial** is `1 − ½(Gini(DSR) + Gini(ASR))`, the average of two pooled Gini coefficients over all N active units — `DSR` is the demand-service ratio and `ASR` is the arrival-service ratio. Perfect equality across both yields F_spatial = 1, full concentration yields 0 (see §4).

**F_causal** is a double-regression metric: `1 − r²_demo`, where `r²_demo` is the R² from a demographic projection on the residual `R = Y − g_0(D)` left over after a first-stage power-basis fit of service rate on demand. Zero demographic explanatory power yields F_causal = 1 (see §5).

**F_fidelity** is the Multi-Stream Siamese discriminator score for a modified trajectory against the real-trace distribution — scores near 1 are realistic, near 0 are implausible (see §6).

**Clean ablation.** Setting `ALPHA_FIDELITY = 0` in `../config.py` removes F_fidelity from L entirely; no GPU memory is consumed by the discriminator and no checkpoint is required.

Three load-bearing claims a reviewer could push back on: (1) equal weights for F_spatial and F_causal assert that Gini-based exposure equity and demographic causal fairness are commensurable — a normative choice with no empirical ground truth; (2) all three terms are monotone in the same direction, which assumes the discriminator score and both fairness scalars improve jointly under the same perturbations — a coupling assumption that warrants empirical validation; (3) three terms are sufficient to characterize fairness for this dataset — this is a design boundary, not a completeness theorem.

The objective is implemented in `../algorithm/README.md` (`FAMAILObjective.forward()`); all weights are in `../config.py`.

§4–§6 give each term in full; §7 decomposes the two fairness terms per cell; §8 puts everything inside the trajectory-modification loop.

---

## §4. F_spatial — pooled Gini fairness

F_spatial is a Gini-based measure of equity in service exposure across active units.

**Ratio definitions.** Each active `(cell, t)` unit `u` carries two service-rate scalars derived from its mean-hourly counts and its supply `S_u`:

- `DSR_u = pickup_u / S_u` — demand-service ratio: how many pickups occur per mean active taxi.
- `ASR_u = dropoff_u / S_u` — arrival-service ratio: how many dropoffs occur per mean active taxi.

Both ratios use `S_u` as the denominator rather than raw pickup counts, so units where taxis are present but not picking up passengers register as low-ratio rather than absent.

**F_spatial formula.**

```
F_spatial = 1 − ½(Gini(DSR) + Gini(ASR))
```

`Gini(x)` is applied to the full N-vector of values across all active units simultaneously. The scalar result is in [0, 1].

**Pairwise Gini formula.**

```
G(x) = Σ_i Σ_j |x_i − x_j| / (2 N² mean(x))
```

This form is differentiable with respect to `x` everywhere except at measure-zero ties — gradient flow through `F_spatial` during ST-iFGSM is well-defined almost surely.

**Sign convention.** `F_spatial = 1` indicates perfect equality: every active unit receives the same service ratio for both DSR and ASR. `F_spatial = 0` indicates maximum concentration: one unit absorbs all service mass and every other unit receives none.

**Design choices.**

1. **Pooled, not block-averaged.** The Gini is computed once over all N active units rather than computed per time-block and then averaged across blocks. Time-blocks with more active units carry proportionally more weight, which reflects their larger contribution to total service exposure across the operating day.

2. **DSR + ASR equal weighting.** Pickups and dropoffs are treated as dual signals of service: pickups capture where taxis initiate service and dropoffs capture where riders arrive. Weighting either alone biases the metric toward the origin view (pickup-only) or the destination view (dropoff-only); equal weighting treats service as a round-trip phenomenon.

**Implementation pointer.** `pairwise_gini()` and `compute_fspatial()` are in `../fairness/spatial.py`; see `../fairness/README.md` for the full API. The module receives only the N-vectors; it has no knowledge of the `(48, 90, T)` grid geometry.

**Load-bearing claims.** Three claims that a reviewer could push back on: (1) supply `S_u` is the right denominator — using raw counts rather than a rate would make high-supply cells structurally advantaged, but the choice that `S_u` correctly normalizes for taxi availability rather than demand is an assumption about what "fair exposure" means; (2) the measure-zero differentiability guarantee holds in practice during optimization — empirically, ties are rare across N = 5,834 units, but a gradient blackout at a tie is not theoretically impossible; (3) equal weighting of DSR and ASR is normatively neutral — it encodes the judgment that origin equity and destination equity matter equally, which may not hold in all city contexts.

F_spatial enters the objective in §3 as the first term and is decomposed per cell in §7.

---

## §5. F_causal — demographic-projection R²

F_causal asks whether demographics — not demand — explain the service rate, via a double regression.

**Stage 1: power-basis fit g_0(D).** The service rate for each active unit is `Y_u = S_u / max(D_u, DEMAND_FLOOR)`. A four-term power basis is fitted via OLS across all N active units:

```
g_0(D) = β₀ + β₁/(D+1) + β₂/√(D+1) + β₃√(D+1)
```

This produces a baseline prediction of service rate from demand alone. The `(D+1)` offsets prevent singularity at `D = 0`, independently of the `DEMAND_FLOOR` clamp that stabilizes `Y`. The resulting coefficient vector `[β₀, β₁, β₂, β₃]` is fixed after preprocessing; it is not re-estimated during the ST-iFGSM loop.

**Stage 2: demographic projection on residuals.** The residual `R_u = Y_u − g_0(D_u)` strips demand-explained variation; `R` (length N) carries only the service-rate component that demand cannot account for. The demographic hat matrix is formed from z-scored features with a prepended intercept column, `X̃` (N × p+1):

```
H_demo = X̃(X̃'X̃)⁻¹X̃'
```

`H_demo` projects any N-vector onto the column space of the z-scored demographics plus intercept. Three district-level features enter `X̃` — housing price per square metre, GDP per capita, and compensation per employed person — each standardized via `StandardScaler` across the active set.

**Final form and sign convention.** Let `M = I − 11'/N` be the centering matrix. The causal fairness scalar is:

```
F_causal = R'(I − H_demo)R / R'MR = 1 − r²_demo
```

where `r²_demo = R'H_demo R / R'MR` is introduced here as the demographic-explained variance fraction of the demand-adjusted residual. The sign convention is deliberate: `r²_demo` high means demographics explain a large share of the residual — the service rate still tracks neighborhood wealth after demand is removed — which is the unfair outcome. `F_causal` high means demographics explain little residual variance, i.e., the service distribution is not systematically aligned with socioeconomic composition. Boundary cases: `R ∈ span(X̃)` gives `F_causal = 0` (fully unfair); `R ⊥ X̃` gives `F_causal = 1` (fully fair).

**Design choices.**

1. **Power basis for g_0.** The form `[1, 1/(D+1), 1/√(D+1), √(D+1)]` is linear in parameters, so OLS produces a closed-form solution and the hat-matrix algebra for F_causal remains exact. The four terms together capture hyperbolic saturation at low demand (dominant `1/(D+1)` and `1/√(D+1)` behavior where `D ≈ 0`) plus a sub-linear growth term (`√(D+1)`) at high demand — a shape confirmed by the Pearson correlation `log(D)·log(Y) = −0.89` on signal-regime cells.

2. **DEMAND_FLOOR = 0.5 as a clamp, not a filter.** Cells with `D_raw < 0.5` retain their identity in the active set; only their demand value is replaced by 0.5 inside `Y = S/D` — filtering would remove them, rendering unfairness in underserved regions invisible. The value 0.5 is chosen for residual-scale balance: at the prior value of 0.01, clamped cells produced `Y` up to 2,947 (two to three orders of magnitude above signal-regime scale), causing `R'MR` to be dominated by floor-regime variance; at 0.5 the clamped-cell `Y` max is 63.5, placing both regimes on comparable scale for the pooled regression.

3. **Two-R² diagnostic.** The all-cells fit and the signal-regime fit (cells with `D_u ≥ 0.5`) are both reported, separating model-class adequacy — does the power basis fit the demand-service law where demand is identifiable? — from audit-set composition, which drives the all-cells R² down because ~85% of active cells have near-zero demand. The all-cells coefficients define `g_0` downstream; the signal-regime R² is a diagnostic only.

4. **g_0 evaluated under `torch.no_grad()` in the modifier loop.** During ST-iFGSM, `g_0(D)` is detached from the autograd graph before `F_causal` is evaluated, so the modifier improves `F_causal` only by moving service relative to the fixed baseline rather than by adjusting the baseline itself. Without this boundary, perturbing pickups to change `D` would simultaneously shift baseline and residual and double-count the demand effect.

**Pointer-outs.** Full methodology rationale — including the two-R² diagnostic, DEMAND_FLOOR sensitivity table, and paper-ready text — is in `F_CAUSAL_METHODOLOGY_NOTES.md` (sibling in this directory). The power-basis fitting routine is in `../fairness/g0_power_basis.py`; hat-matrix precomputation (H_demo and M) is in `../fairness/hat_matrices.py`.

**Load-bearing claims.** Six claims a reviewer could contest: (1) demand is the correct first-stage control — the Frisch-Waugh-Lovell theorem grounds this; partial regression on `R` isolates the demographic effect after demand is partialled out; (2) the power basis suits the demand-service relationship — signal-regime R² of 0.69 and log-log Pearson correlation −0.89 are the empirical anchors; (3) `DEMAND_FLOOR = 0.5` produces a well-scaled residual — the sensitivity table in `F_CAUSAL_METHODOLOGY_NOTES.md` §6 shows values ≤ 0.1 fail the scale-balance criterion; (4) three demographic features span socioeconomic variation adequately — parsimonious by design, not a completeness claim; (5) all-cells OLS coefficients are the correct source for `g_0` — signal-regime coefficients would misspecify the baseline for floor-regime cells, breaking per-cell attribution consistency in §7; (6) `r²_demo` supports a causal interpretation — this rests on demand being exogenous to cell demographics, plausible under the supply-based mask (§2) but not tested instrumentally.

F_causal enters the objective in §3 as the second term and is decomposed per cell in §7. DEMAND_FLOOR's empirical justification is reprised in §9 as a sensitivity-study opportunity.

---

## §6. F_fidelity — discriminator-based realism

F_fidelity is a similarity score from a pre-trained discriminator that constrains modified trajectories to remain realistic.

**Discriminator.** The realism check is carried out by the Multi-Stream Siamese discriminator, ported from the parent codebase as an opaque inference-only module in `famail_temporal/fidelity/`. Four classes are ported: `FeatureNormalizer`, `SiameseLSTMEncoder`, `ProfileEncoder`, and `MultiStreamSiameseDiscriminator`. No training code or deprecated architectures are included.

**Inputs and output.** Each call takes two trajectories — an anchor from the real-trace distribution and the modified trajectory — each rendered as a multi-stream context: driving stream, seeking stream, and profile features. The discriminator returns a similarity score in [0, 1]; F_fidelity = 1 means the modified trajectory is indistinguishable from an authentic expert trace.

**Design choices.**

1. **Opaque inference-only port.** The parent codebase contains 1,297 lines across eight classes including training loops and five deprecated architectures; only the four inference classes are ported. `load_discriminator()` asserts the presence of an `architecture_config` key and raises specifically on partial loads — silent mismatches are not tolerated.

2. **ALPHA_FIDELITY = 0 as a clean ablation.** Setting `config.ALPHA_FIDELITY` to zero causes `FAMAILObjective.forward()` to skip the discriminator call entirely; no GPU memory is consumed and no checkpoint is required, making zero-weight runs a true ablation rather than a downweighted one.

The discriminator's LSTM requires `torch.backends.cudnn.flags(enabled=False)` during the forward pass because the cuDNN RNN kernel does not support backward passes in eval mode; without this flag, `loss.backward()` after a discriminator call raises a `RuntimeError`.

**Pointer-outs.** API surface, ported-class inventory, and multi-stream context decisions: `../fidelity/README.md`. Checkpoint provenance and architecture config format: `../discriminator_checkpoints/README.md`.

**Load-bearing claims.** Three claims a reviewer could contest: (1) a discriminator pre-trained on Shenzhen traces is a sufficient realism proxy — it is not tested on out-of-distribution perturbations; (2) collapsing the multi-stream context to a single scalar is adequate for gradient guidance — spatial, temporal, and profile signals are pooled without per-stream interpretability; (3) eval-mode behavior with dropout disabled is equivalent to the training-time forward for gradient signal — a behavioral assumption about the V3 checkpoint, not a theorem.

F_fidelity enters the objective in §3 as the third term. Unlike the fairness terms, it is not decomposed per cell — it is a per-trajectory check, not a per-unit audit.

---

## §7. Per-cell fairness attribution

Both fairness metrics admit a per-cell decomposition that sums to F itself, signed so that positive = fair.

**The decomposition problem.** F_spatial and F_causal are each a scalar in [0, 1] ("higher = fairer"). A per-cell audit requires distributing that scalar across all N active units as a signed N-vector α with Σ_i α_i = F — not Σ_i α_i = 1 − F. A decomposition summing to the complement forces consumers to track an implicit sign flip against the published metric. The natural per-cell terms for both metrics land on the unfairness side — Gini is a sum of non-negative pairwise terms, and r²_demo decomposes as a squared-residual difference, both summing to 1 − F — so the decomposition re-anchors each term against a uniform baseline 1/N.

**1/N-shifted decomposition.**

```
α_i = (1/N) − unfairness_contrib_i
Σ_i α_i = F
```

The unfairness contribution is metric-specific.

**For F_spatial:**

```
unfairness_contrib_i = ½(gini_dsr_i + gini_asr_i)

where  gini_i(x) = Σ_j |x_i − x_j| / (2 N² mean(x))
```

Each gini_i(x) is the per-unit contribution to the pooled Gini coefficient; summing over i recovers Gini(x). Equal weighting of DSR and ASR mirrors the F_spatial definition in §4.

**For F_causal:**

```
unfairness_contrib_i = ((MR)_i² − ((I − H_demo)R)_i²) / R'MR
```

where M = I − 11'/N is the centering matrix, H_demo is the demographic hat matrix, and R = Y − g_0(D) is the demand-adjusted residual. Each term is the per-cell difference between the centered squared residual and the post-demographic-fit squared residual, normalized by total centered variance. Summing over i gives r²_demo = 1 − F_causal.

**Sign-convention table.**

| α_i band | Cell semantics | Priority |
|---|---|---|
| α_i > 1/N | Above-baseline fair; cell contributes more than its uniform share to F | Low — not a target |
| α_i ≈ 1/N | Neutral; cell carries its uniform share | Monitor only |
| 0 < α_i < 1/N | Mildly underperforming the baseline; positive but sub-uniform contribution | Low–medium |
| α_i < 0 | Drags fairness below baseline; cell's unfairness contribution exceeds 1/N | Highest — primary modification target |

**Justification for the uniform 1/N baseline.** The uniform baseline is the minimum-assumption prior: no auxiliary signal — demand, supply, or demographics — enters α_i beyond the metric's own unfairness term; any deviation from 1/N is attributable entirely to unfairness_contrib_i. Perfect-fair limit: Gini = 0 or r²_demo = 0 gives every α_i = 1/N and Σ α_i = 1 = F. Perfect-unfair limit: Gini = 1 or r²_demo = 1 drives outlier α_i toward −1 and Σ α_i = 0 = F.

**Load-bearing claims.** Four claims a reviewer could push back on: (1) Σ α_i = F follows algebraically from the 1/N shift — verify that the pairwise Gini per-cell form and the squared-residual causal form each sum to their 1 − F complement before accepting the equality; (2) the uniform prior 1/N carries no weighting by cell area, demand, or supply — contestable on the grounds that higher-demand cells deserve a larger baseline share; (3) the per-cell causal term is signed, not non-negative — cells where demographic regression worsens local fit yield negative unfairness_contrib_i and α_i > 1/N even when F_causal < 1, which is correct behavior; (4) α_i is recomputed at each gradient step of ST-iFGSM because pickup perturbations change R, so gradient flow through R is live throughout the loop.

Full formulation — including worked examples, the decision audit trail, and the relationship to the prior (1 − F) decompositions — is in `FAIRNESS_DECOMPOSITION_FORMULATION.md` (sibling in this directory).

Per-cell α_i drives trajectory selection in §8 (cells with α_i < 0 are highest-priority modification targets) and is the primary export downstream tooling consumes.
