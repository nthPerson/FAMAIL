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
