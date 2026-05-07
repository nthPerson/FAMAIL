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
