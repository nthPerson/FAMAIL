# Supply-lift editing — design

**Date:** 2026-07-08
**Status:** Approved design (brainstorm 2026-07-08) → ready for implementation plan
**Workstream name:** **supply-lift editing** (mechanism term: *seeking-tail rerouting*). Branch: `supply-lift-editing`.
**Protocol note:** this is a trajectory-editing algorithm change; the design was explicitly approved by
Robert in the 2026-07-08 brainstorm (satisfies the algorithm-change protocol). Trim-side numerics are
preserved byte-identical (§5, §7 gate G3).

---

## 1. Motivation

The leveling-down mechanism analysis (`PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md`) proved the
current editor **cannot lift up** the under-served group: it is demand-only, and under
`Y = supply/demand` with frozen supply, every fairness-improving demand move either pads over-served
rich cells (leveling-down — what it does) or deletes recorded service from poor cells (perverse — what
it must never do). The Option-A rollout evaluation confirmed the consequence propagates through
training: weighted-BC policies *reduce* poor-area pickup share dose-dependently (0/6 seeds, p=.031).

The one non-perverse lever is **supply**: taxi seeking presence. `∂Y/∂S = 1/max(D, 0.5) > 0`
everywhere, and for the 93% of poor-group units at the demand floor, `ΔY = 2·ΔS`. In the data, supply
*is* the seeking trajectory — so rerouting seeking tails into under-served cells raises their service
ratio while **adding** (not deleting) recorded service behavior for downstream learning.

**Bonus repair:** the current editor violates the data's own action space — it moves only
`states[-1]` by up to ε=2 cells (`utils/trajectory.py:76-87`) with **no adjacency check on editor
output** (the king-move filter `max(|dx|,|dy|) ≤ 1` runs only in source-data generation,
`data/source_generation/invariants.py:35-47`). Supply-lift's tapered tail translation makes every
edited trajectory fully action-space compliant — more physically valid than today's output.

## 2. Goals / non-goals

**Goals**
- Add a **lift** edit mode: reroute selected trajectories' seeking tails (last `TAIL_LEN` states +
  pickup) toward under-served cells, with supply endogenous to the objective via a differentiable ΔS
  channel.
- Keep **trim** (the existing pickup/demand edits) byte-identical; trim+lift run together in one
  editing pass. The published trim-only results become the ablation row.
- New headline for the KDD paper: trim+lift edits on all 4 datasets → external metrics **without the
  leveling-down caveat** + `Δmean(Y|D) > 0` (the lifting-up test, currently 0.000) + weighted-BC
  downstream re-run + Option-A allocation shares re-run (poor-area shares should rise).
- Full king-move adjacency compliance of edited trajectories.

**Non-goals (v1)**
- Detour *insertion* (adding states): changes trajectory-length distributions → Fidelity-B risk;
  deferred to v2.
- Objective asymmetry (weighting under-served residuals more): the supply gradient already points at
  under-served residuals; YAGNI for v1.
- Temporal edits: spatial-only, time buckets untouched (current convention).
- Multi-loop / re-attribution: single-pass, matching the locked decision.
- Editing the driving stream or profiles: seeking tails only.

## 3. Edit mechanism — tapered tail translation

- Edit unit: the last **`TAIL_LEN = 4`** seeking states + the pickup (5 states total; configurable).
- One continuous δ ∈ ℝ² per trajectory (unchanged parameterization, ε-ball `EPSILON_BALL = 2.0`
  unchanged), applied with fixed taper weights **`TAIL_TAPER = (0.25, 0.5, 0.75, 1.0)`** from the
  oldest tail state toward the pickup (pickup gets full δ). The state before the tail
  (`states[-TAIL_LEN-2]`) is the anchor and never moves.
- Each offset position runs through the existing `SoftCellAssignment` (5×5 Gaussian softmax,
  τ-annealed) — its `forward` is already batch-capable; the injection helper gains a batched variant.
- **Discretization:** round per-state offsets, then a greedy adjacency repair (from the anchor toward
  the pickup) enforces `max(|dx|,|dy|) ≤ 1` on every consecutive transition. With |δ|∞ ≤ 2 and
  TAIL_LEN ≥ 2 a compliant rounding always exists. Trajectories edited in trim mode also get the
  taper (their δ came from the demand gradient; the tail follows), so **all** edited output becomes
  adjacency-compliant.
- Time buckets and `day_index` of every state are preserved.

## 4. The ΔS supply channel

`active_taxis` ground truth: **distinct taxis** per (5×5 neighborhood, hour), from **all** GPS streams
(seeking + driving) of the 50 drivers, block-aggregated as `sum(hourly counts)/(n_hours·n_days)`,
floored at `SUPPLY_FLOOR = 0.1` (`data/aggregation.py:120-158`; `active_taxis/generation.py:70-146`).

- **Soft convention (optimization):** a 5-min seeking state = **1/12 of one hourly taxi-presence**.
  Its contribution to `ΔS` = soft-assignment probability map, **box-blurred with the 5×5 presence
  kernel** (same size as the active-taxis neighborhood; implemented as conv2d with a ones kernel),
  × `1/(12 · n_hours_per_block[t] · n_days)`, **added** at the state's new position and
  **subtracted** at its original position, in the state's own time block.
  `S′_N = clamp(S_base_N + ΔS_N, min=SUPPLY_FLOOR)`; `S′_N` replaces the frozen buffer at exactly two
  call sites: `objective.py:126` (F_spatial's DSR/ASR denominators) and `objective.py:139`
  (F_causal's `supply_N`).
- **Hard tier 1 (every evaluation):** identical arithmetic at the discrete post-rounding positions
  (no softmax smear). The **soft-vs-hard gap** is an explicit gate (G2) — this failure mode killed
  multi-loop re-attribution in June.
- **Hard tier 2 (gold, headline runs):** exact **distinct-taxi recount** from raw GPS with the edited
  seeking states swapped in (Shenzhen: `raw_data/taxi_record_07_50drivers.pkl`; SF: Cabspotting
  pings). If tier 2 disagrees materially with tier 1, **the paper reports tier-2 numbers**.
  Known bias to quantify: the presence-fraction relaxation over-counts removals/additions when the
  driver has other pings in the same neighborhood-hour.

## 5. Lift attribution & selection

- New closed-form **per-unit supply gradient** `g_i = ∂L/∂S_i` (same residual algebra as the demand
  attribution — its mirror; F_causal part analytic from `R`, `H`, `M`; F_spatial part from the Gini
  decomposition). Positive `g_i` = adding supply at unit *i* raises the objective (under-served units).
- **Lift candidates:** trajectories whose tail states lie within ε of cells with high positive
  supply-gradient mass; score = achievable `Σ g_i · ΔS_i` for the best in-ball target. Ranked
  descending; top-`LIFT_BUDGET` selected.
- **Budget:** total k = 10,000 unchanged. Trim keeps its strictly-negative-α selection
  (byte-identical, ~2,455 on Shenzhen PRIMARY); lift fills from the remaining budget
  (`LIFT_BUDGET = k − n_trim` by default; configurable). A trajectory selected by both is edited once,
  in trim mode (trim precedence keeps the ablation clean).
- Single-pass; no re-attribution between edits.

## 6. Optimizer & fidelity integration

- `algorithm/supply.py` (new): ΔS math — presence-mass constants, box-blur kernel, batched soft
  injection, hard tier-1 materialization, the supply-gradient attribution.
- `algorithm/modifier.py`: tail-aware perturbation (taper application, per-state soft-assign calls,
  ΔS accumulation, adjacency-repairing discretization); persists tail-modified trajectories in
  `ModificationHistory` (downstream `build_edited_corpus` then works unchanged).
- `algorithm/objective.py`: `forward` accepts an optional differentiable `delta_supply_N`; when
  absent, behavior is byte-identical to today (gate G3).
- **Fidelity goes live on the tail:** the L tail rows of the cached discriminator tensors
  (`tau_prime_features`, `ms_kwargs['x2']` slot 0) are spliced per-iteration — today only the pickup
  row is (`modifier.py:439-450`). The seeking-BiLSTM finally *sees* the edit; F_fidelity becomes
  load-bearing instead of gradient-dead (paper point).
- Config additions: `TAIL_LEN`, `TAIL_TAPER`, `LIFT_BUDGET`.

## 7. Gates

- **G0 — Stage-0 supply oracle (BEFORE any build):** greedy oracle on real corpus geometry under
  exact distinct-count semantics → achievable `Δmean(Y|D)` ceiling from rerouting eligible tails.
  **Gate: ceiling ≥ ~+0.3** (meaningful vs trim's −0.60 gap-close and the demand oracle's +1.54);
  below → stop, take fallback rung 1.
- **G1 — baseline preservation:** with lift disabled, the full pipeline reproduces the published
  trim-only metrics exactly.
- **G2 — soft-vs-hard:** optimization-time soft ΔS gains survive hard tier-1 re-materialization
  (report the gap); headline numbers survive tier-2 distinct-count recomputation.
- **G3 — trim invariance:** trim-selected trajectories' pickup cells identical with the new code
  (taper changes their tails, not their pickups; demand grid unchanged).
- **G4 — adjacency:** 100% of edited trajectories king-move compliant (vs. violated today).
- **G5 — fidelity:** Fidelity-A ≈ raw; Fidelity-B stable (translation preserves length).
- **G6 — the lifting-up test:** `Δmean(Y|D) > 0` with bootstrap CI excluding 0 on Shenzhen PRIMARY;
  external metrics improve; F_causal not regressed.

## 8. Evaluation plan (headline)

1. Trim+lift edit runs on all 4 datasets (Shenzhen PRIMARY first; then gdp-comp, logpop, SF sf12 —
   SF hard-tier-2 needs its Cabspotting pings; budget a day for that path).
2. External fairness metrics via the existing harness with a **supply override** (small change:
   `service_ratio_Y(pickup_3d, bundle, supply_3d=None)`; group levels expose lift vs trim).
3. Weighted-BC sweeps on the trim+lift corpus (machinery unchanged — histories carry the modified
   trajectories; ~10h GPU per city-sweep).
4. Option-A allocation-share evaluation re-run on the new policies (poor-area pickup + seeking-state
   shares should now **rise**; kills the perverse-drain caveat).
5. Curation into `PAPER/` mirroring the external-metrics bundle; the published trim-only rows carried
   as the ablation.

## 9. Schedule (indicative — expected to shift; fallback ladder is the binding part)

All-in target (user choice): build lands ~Jul 14; edit runs + external metrics ~Jul 16; BC sweeps +
Option-A + curation by ~Jul 19 (abstract); tables/figures + buffer to Jul 26 (paper).

**Fallback ladder (pre-agreed):**
1. G0 fails → paper gets limitations + the oracle ceiling number (quantified future work).
2. Build slips past ~Jul 16 → BC re-run drops out; downstream cites trim-only results, labeled.
3. Metrics weak/late → additive-subsection shape (mechanism finding → pilot); existing headline
   results stand unchanged.

## 10. Risks

- **Soft-vs-hard ΔS gap** (the June failure mode) — mitigated by G2's two tiers + Stage-0 running on
  *hard* semantics from the start.
- **Distinct-count dampening:** removals at origin may not reduce hard counts (driver's other pings) —
  quantified by tier 2; note the *asymmetry favors lift* (additions in poor cells, where the 50-driver
  presence is sparse, are likely real new presence).
- **Fidelity pressure:** tail edits are visible to the discriminator; if F_fidelity resists lifts,
  α-weights may need retuning (grid-search protocol exists) — surfaced, not silently tuned.
- **SF tier-2 pipeline** is new plumbing (Cabspotting ping recount).
- **Schedule** — governed by the fallback ladder, not by optimism.

## 11. Provenance / inputs

- Mechanism analysis + Option-A results: `PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md` (+
  `scripts/leveling_analysis.py`, `scripts/option_a_rollout_eval.py`) — committed to `main` in
  `1e51471` (Meeting-42 prep) as the motivation record.
- Verified code facts cited throughout from: `utils/trajectory.py`, `algorithm/{modifier,objective,
  soft_cell_assignment,attribution}.py`, `fairness/{causal,spatial}.py`, `data/aggregation.py`,
  `active_taxis/generation.py`, `fidelity/context.py`, `data/source_generation/invariants.py`.
