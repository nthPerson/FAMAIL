# FAMAIL Temporal — Researcher Handoff: Trajectory-Modification Algorithm and Fairness Formulations

**Document date:** 2026-05-07

This document is intended as a sanity-check enabler for collaborating researchers who have lab context but no prior `famail_temporal/` context.

**Status:** Written against config.T = 24, source-data git SHA a532ead.

---

## TL;DR

FAMAIL Temporal optimizes taxi-service equity across a city's active `(cell, time-block)` units using a three-term objective `L = α_s · F_spatial + α_c · F_causal + α_f · F_fidelity`, where `F_spatial` measures Gini-based service exposure equity, `F_causal` measures how much demographics — rather than demand — explain the service rate `Y`, and `F_fidelity` constrains modified trajectories to remain realistic. The trajectory-modification algorithm identifies pickup locations whose per-cell attribution `α_i` falls below the uniform baseline `1/N` and reroutes them via a Spatial-Temporal iterative Fast Gradient Sign Method (ST-iFGSM) loop that maximizes `L` over a soft-cell assignment. Per-driver fairness, supply-side modification, and real-time deployment are out of scope.

---
