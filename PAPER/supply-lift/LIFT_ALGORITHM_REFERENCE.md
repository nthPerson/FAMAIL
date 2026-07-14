# Supply-Lift & Supply-Gradient Attribution — Foundational Reference

**What this is.** The exhaustive reference for the *lift* side of the FAMAIL trajectory editor and the
supply-gradient attribution that drives it: motivation, methodology, mathematics, implementation (with
verified code anchors), validation history, results, and the invariants/gotchas any future human, agent,
or paper-writer needs. [`FINDINGS.md`](FINDINGS.md) is the curated *results* narrative; this document is
the *mechanism* record. Where the two overlap, FINDINGS.md numbers are canonical (each traceable via
[`data_provenance.md`](data_provenance.md)).

**Provenance of this document.** Written 2026-07-10 against `main` after the supply-lift merge
(branch `supply-lift-editing`, built 2026-07-07 → 07-09, fast-forward merged 2026-07-09). All file:line
anchors below were verified against the merged tree at the time of writing; commit SHAs are the original
branch commits (preserved by the fast-forward merge).

---

## 1. Motivation: why the editor needed a supply lever

Full analysis: [`PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md`](../external-metrics/LEVELING_DOWN_MECHANISM.md).
The hard facts:

1. **The published mechanism ("trim") levels down.** External fairness metrics improve after trim
   editing, but by reducing the over-served group's service while the under-served group's absolute
   service stays *flat*. On the published Shenzhen PRIMARY edit, **all 2,455/2,455 edited pickups
   originated and landed in advantaged (low-migrant) cells** — zero edits touched a disadvantaged cell.
2. **This is structural, not an optimizer quirk.** With service ratio `Y = S / max(D, DEMAND_FLOOR)` and
   supply `S` frozen, the editor's only lever is demand `D`. The demand-side leverage `∂Y/∂D = −S/D²`
   is ~32× larger in high-supply (advantaged) cells; **93% of disadvantaged units sit at the demand
   floor** (D ≤ 0.5) where demand removal changes nothing; median taxi presence is 1.8 (disadvantaged)
   vs 17.6 (advantaged). A demand-only editor can only raise the poor group's ratio by *deleting
   recorded service from poor areas* — a greedy oracle proved that ceiling exists (+1.54) but is
   perverse. Leveling-down is the demand-only editor's constrained optimum.
3. **The training-side alternative failed.** BC policies trained on trim-edited data allocate *fewer*
   pickups to poor areas, dose-dependently (share 0.0500 → 0.0452 at w30, 0/6 seeds, Wilcoxon p = .031)
   — the "Option A" rollout evaluation, negative
   ([`data/rollout_trimonly_prior_summary.json`](data/rollout_trimonly_prior_summary.json)).
4. **Therefore: the numerator.** `∂Y/∂S = 1/max(D, 0.5) > 0` everywhere, and at the demand floor
   `ΔY = 2·ΔS`. Adding taxi *presence* (supply) to under-served cells is the one non-perverse lever
   that raises their service ratio. Supply in this project is *seeking-taxi presence* derived from GPS —
   so the editor must reroute *seeking behavior*.

## 2. The go/no-go gate (G0): Stage-0 oracle

Before any build, a greedy oracle bounded the achievable effect (Task 1, commit `0fac8f7`; tool:
`famail_temporal/analysis/supply_lift_oracle.py`; artifact [`data/oracle.json`](data/oracle.json)).

- **Setup:** for every trajectory whose tail (last `TAIL_LEN+1` states) passes within Chebyshev-2 of a
  disadvantaged-group cell (migrant axis, district extremes), evaluate all 24 integer translations
  δ ∈ [−2,2]²\{0} of the tail; score the *net* effect on `mean(Y|D)` (supply mass added at new 5×5
  boxes, removed at old ones, pickup demand mass relocated); apply greedily best-first against running
  grids, re-scoring at pop, up to budget k = 10,000.
- **Results (Shenzhen PRIMARY, baseline mean(Y|D) = 7.0734, N_D = 6,950):** ceiling **+0.882**
  (fraction semantics) / **+0.827** (distinct-seeking approximation); 5,538 candidates, 2,571 applied;
  runtime ~16 s. Gate threshold: **≥ +0.3**.
- **The decomposition that made the gate honest.** Inspection of the top edits showed the nominal
  ceiling partly rode a demand-floor artifact (moving 0.2 pickup mass out of near-empty disadvantaged
  cells inflates Y; e.g. a cell with D = 0.8, S = 29.8 jumps Y 37→50). A decomposition re-run split the
  +0.882 into **supply +0.376 / demand +0.506**, and a supply-only greedy (demand pinned) achieved
  **+0.786 — 2.6× the threshold on the honest channel alone** (5,528 of 5,538 candidates applied; the
  edit budget never binds — candidate supply does). The gate passed on the supply-only number.
- **Caveats recorded at the time:** the oracle is an upper bound unconstrained by realism/fidelity/the
  optimizer; its "distinct-seeking" semantics is an approximation (exact recount = tier-2, §10).

## 3. Conventions (exact values; the vocabulary of everything below)

| Concept | Definition | Where |
|---|---|---|
| Service ratio | `Y = S / max(D, DEMAND_FLOOR)` per active (cell, hour-block) unit | `DEMAND_FLOOR = 0.5`, `famail_temporal/config.py:56` |
| Supply floor | endogenous supply clamped `S' = clamp(S_base + ΔS, min=SUPPLY_FLOOR)` | `SUPPLY_FLOOR = 0.1`, `config.py:57` |
| Presence mass | one 5-min seeking state = **1/12 hourly presence**; per-state mass `= 1/(12 · n_hours_per_block[t] · n_days)` | `state_presence_mass`, `famail_temporal/algorithm/supply.py:26` |
| Presence spread | each state's mass covers its **clipped 5×5 neighborhood** (replicating ones-kernel, *not* mass-conserving — matches how `active_taxis_3d` was built) | `PRESENCE_KERNEL_SIZE = 5`, `supply.py:23` |
| Pickup demand mass | `1/(n_hours_per_block[t] · n_days)` at its **single** cell (12× a state's presence mass; a count, not a presence fraction) | `baselines/datasets.py` (`pickup_mass`) |
| Tail | the **last** `L_eff = min(TAIL_LEN, len−2)` seeking states **+ the pickup** (`states[-(L_eff+1):]`; `states[-1]` *is* the pickup). The first states and the "anchor" (state before the tail) never move. Trajectories with `len < 3` are never tail-edited | `config.TAIL_LEN = 4`, `config.py:85` |
| Taper | pickup moves the full offset δ; tail state j gets `round(taper_j · δ)` with linear weights `j/L_eff` (= `TAIL_TAPER = (0.25, 0.5, 0.75, 1.0)` at L_eff = 4) | `taper_weights`, `famail_temporal/utils/trajectory.py:11`; `config.py:86` |
| Edit ball | δ ∈ [−ε, ε]² integer, ε = `EPSILON_BALL = 2.0` (Chebyshev), δ = (0,0) excluded from screens | `config.py:83` |
| King-move rule | every consecutive transition satisfies `max(|dx|,|dy|) ≤ 1`; source preprocessing enforces it, the legacy editor violated it on ≥2-cell moves (latent double standard, repaired by the taper machinery) | `apply_tail_perturbation`, `trajectory.py:173` |
| Lift budget | `LIFT_BUDGET = None` → lift fills `k_total − n_trim` slots | `config.py:87` |
| Edits are spatial-only | `time_bucket` and `day_index` never change | enforced throughout |

## 4. Supply-gradient attribution (the mathematics)

### 4.1 The gradient: ∂L/∂S at baseline

`supply_gradient_N(bundle, objective)` (`supply.py:177`): construct a zero tensor `δS ∈ ℝᴺ` (one entry
per active unit, `N` = 34,524 on Shenzhen) with `requires_grad=True`; call
`objective.forward(pickup_3d, delta_supply_N=δS)`; one backward; return `δS.grad` as a numpy `(N,)`.
The objective applies `active_taxis_N = clamp(active_taxis_N + delta_supply_N, min=SUPPLY_FLOOR)`
(`famail_temporal/algorithm/objective.py:125-126`) feeding **both** `compute_fspatial` and
`compute_fcausal_from_compact` — so the gradient reflects the full weighted objective
(α = (0.2, 0.7, 0.1) in production).

**Analytic anchor (tested, not just asserted).** For the F_causal term with residual `R = Y − g0(D)`,
demographic hat matrix `H = X(XᵀX)⁻¹Xᵀ`, and centering `M`:

```
∂F_causal/∂S_i = (2 / RᵀMR) · [ ((I − H) R)_i  −  F · (M R)_i ] / max(D_i, 0.5)
```

The autograd gradient is asserted equal to this closed form (rtol 1e-3) on all clamp-inactive units in
`famail_temporal/tests/test_supply.py::test_supply_gradient_matches_analytic_fcausal` — the mathematical
anchor test of the feature (verbatim from the plan, passed unmodified).

**Known property:** at units where `S_base ≤ SUPPLY_FLOOR` the clamp subgradient can zero the gradient.
Accepted by design — those units can only gain from added supply.

### 4.2 The screen: linearized best-δ ranking

`lift_candidates(bundle, grad_N, tail_len=config.TAIL_LEN, epsilon=config.EPSILON_BALL)`
(`supply.py:289`): embed `grad_N` into the (gx, gy, T) grid, precompute a 5×5 **box-summed gradient
grid** via a summed-area table (`_box_sum_grid`, `supply.py:204` — verified exact against brute force
on 200 random grids during review), so a state's whole-neighborhood gradient value is one lookup. Then
per trajectory (len ≥ 3): for each of the 24 integer δ, the linearized gain is
`Σ_tail-states mass · (G_box[new position] − G_box[old position])` (positions clipped to grid); the
trajectory's score is its best δ. Returns `(trajectory_idx, score)` sorted descending. δ = (0,0) is
excluded (documented); scores ≤ 0 are dropped downstream.

This is a *nomination* device only — first-order, frozen at the post-trim state (see §5 ordering). The
per-edit optimizer (§6) re-derives the actual move.

### 4.3 Plan assembly: trim precedence + budget fill

`assemble_edit_plan(trim_indices, lift_scored, k_total, lift_budget=None)` (`supply.py:236`): trim
entries first, in their given order; then lift entries from the descending-sorted screen, skipping any
index already claimed by trim and any score ≤ 0, stopping after
`lift_budget if lift_budget is not None else k_total − len(trim_indices)` entries. Production Shenzhen:
k = 10,000 → 2,455 trim + 7,545 lift. SF: k = 2,000 → 1,371 trim + 629 lift.

## 5. Pipeline integration (runner)

`famail_temporal/evaluation/runner.py`, `run_experiment`:

- **Ordering.** The trim phase runs first, unchanged, via `editing_loop.run_editing_rounds`
  (`run_editing_rounds` is byte-untouched by the supply-lift work). Only **after** trim completes does
  the lift-selection block run (`lift_enabled = config.TAIL_LEN > 0 and config.LIFT_BUDGET != 0`,
  `runner.py:378`; selection at `runner.py:477-495`), so the supply gradient is computed on the
  **post-trim** state. The lift edits then run as a second pass over the assembled plan, calling
  `modify_single(traj, mode="lift", ...)` on trajectories provably disjoint from the trim set.
- **Legacy short-circuit (G1).** With `TAIL_LEN = 0` or `LIFT_BUDGET = 0`, the entire block is skipped —
  `supply_gradient_N`/`lift_candidates` are never called and every code path is bit-for-bit the
  pre-supply-lift pipeline (proven by an exact-equality end-to-end test,
  `tests/test_runner.py::test_legacy_mode_end_to_end_byte_identical`).
- **After-metrics.** `metrics_after` is computed with endogenous supply:
  `np.clip(active_taxis_3d + delta_supply_3d, SUPPLY_FLOOR, None)` through the standard
  `build_fairness_grid` (all four fairness channels see one consistent supply denominator).
- **Persistence.** `delta_supply_3d.npz` (key `delta_supply_3d`, float64) + `metrics.json` gains
  `n_trim`, `n_lift`, `n_taper_infeasible_trim`, `n_taper_infeasible_lift`,
  `supply_totals {added, removed}` (`evaluation/persistence.py`). Note: `added ≈ removed` is *expected*
  for interior edits (unclipped 5×5 boxes conserve mass); asymmetry arises only from grid-edge clipping.
- **External-metrics override.** `service_ratio_Y(pickup_3d, bundle, supply_3d=None)` in
  `baselines/external_fairness_io.py` + `--delta-supply <npz>` on `baselines/run_external_fairness.py`
  apply `S'` to the AFTER side only (commit `fc903fe`; default paths byte-identical).

## 6. The per-edit optimizer in lift mode (`modify_single`, `algorithm/modifier.py:392`)

Both modes share the loop (ε-ball, temperature-annealed soft cell assignment, best-iterate selection,
`total.backward(retain_graph=True)`); `mode="trim"` is the default kwarg and its optimization-path
argument lists are unchanged (G1). Lift specifics, per trajectory:

- **Constants block** (`modifier.py:486-547`): local demand clone (sanitized `clamp(min=0)` — incident
  #1, §9), tail cell/time/mass vectors, taper weights, and `removal_const` — the **once-computed,
  constant** hard ΔS removal (sign −1) at the tail's *original* positions (`modifier.py:543-547`).
- **Per iteration:** soft tail positions `pos_j = orig_j + taper_j · δ_tensor` (one `(2,)` differentiable
  leaf carries the whole edit); one **batched** soft-assign over all M = L_eff+1 moving rows;
  `soft_delta_supply(probs_new, ..., signs=+1)` (`supply.py:42`: embed each row's 5×5 probs at its
  clipped window → `F.conv2d` with a ones 5×5 kernel → × mass × sign → accumulate into the row's
  hour-block slice) plus `removal_const` gives the trajectory's live ΔS;
  `delta_supply_N = (self._delta_supply_3d + traj_soft_ds)[mask_3d]` enters the objective
  (`modifier.py:650-657`). The pickup's soft demand injection is exactly the trim/legacy mechanism.
- **Live fidelity tail.** The fidelity features splice **all M moving rows** per iteration (trim splices
  only the pickup row) — `modifier.py:667-694` — so the discriminator scores the actual rerouted tail
  during optimization.
- **Discretization.** `Trajectory.apply_tail_perturbation(δ, TAIL_LEN, grid_dims)`
  (`utils/trajectory.py:173`): per-axis rounded pickup offset; tapered integer targets; **backward
  reachability-interval repair** that returns a king-compliant assignment closest to the targets, or
  `None` exactly when none exists even after deepening the tail to the whole trajectory (review
  brute-forced 3,000 random cases: zero false-None, zero false-some). Lift on `None`: **skip the edit
  entirely** (demand restored, accumulator untouched, `n_taper_infeasible_lift` incremented).
- **Persistence per edit:** demand moved as in legacy; the *final discrete* move's hard tier-1 ΔS
  (`hard_delta_supply`, `supply.py:126`) is accumulated into the shared `self._delta_supply_3d`
  (init `modifier.py:184`; exported float64 via `current_delta_supply_3d()`, `modifier.py:202`), so
  every subsequent edit optimizes against the running supply state.

**Trim's relationship to ΔS (evaluation honesty).** In taper mode trim tails also move (at
discretization only, via `_discretize_trim`, `modifier.py:244` — which first computes the **exact
legacy deployed cell** by the verbatim legacy arithmetic, then hands the repair an integer offset, so
trim pickups/demand grid reproduce legacy exactly = G3). Their hard ΔS enters the accumulator for
*evaluation*, but **never enters their own optimization** (trim's objective calls pass nothing for
`delta_supply_N`). Trim repair-infeasible → legacy pickup-only fallback, counted
(`n_taper_infeasible_trim`, `modifier.py:286`) — 115/2,455 on Shenzhen, 47/1,371 on SF; under the
**skip-on-infeasible rule** (user decision 2026-07-08, rule adopted *before* its metric effects were
computed) those fallbacks were post-filtered to their originals in the headline "filtered" datasets
(FINDINGS §8; the re-derivation is replay-exact and locked as a regression test).

## 7. Implementation map (verified anchors + original commits)

| Component | Location | Commit(s) |
|---|---|---|
| ΔS math (soft diff. + hard tier-1) | `algorithm/supply.py:23-175` | `67fd50a` + fix `ca593bc` |
| Objective `delta_supply_N` (None path byte-identical) | `algorithm/objective.py:93-126` | `eaa4151` |
| Supply gradient + SAT + screen | `supply.py:177-334` | `1941b83` |
| Tapered tail perturbation + adjacency repair | `utils/trajectory.py:11, 173+` | `4bd70cc` |
| Edit-plan assembly | `supply.py:236` | `d25a985` |
| Modifier lift mode + `_discretize_trim` | `algorithm/modifier.py:184-211, 244+, 392+` | `92383c4` + G3 fix `e18fe5e` |
| Runner wiring + persistence + G1 gate | `evaluation/runner.py:378-495+`, `evaluation/persistence.py` | `2005e29` |
| Tier-2 distinct-count recount | `analysis/supply_recount.py` | `c0536b4` |
| External-metrics supply override | `baselines/external_fairness_io.py`, `run_external_fairness.py` | `fc903fe` |
| Stage-0 oracle | `analysis/supply_lift_oracle.py` | `0fac8f7` |
| Incident fixes (float32 family, §9) | modifier / runner | `c660140`, `abfef82`+`1bd7f29`, `0011cd4` |
| Discriminator constant-encoding cache | `fidelity/model.py` (`cache_constant_streams`), `modifier.py:570-572` | `85c6dbc` |
| Tests | `tests/test_supply.py`, `test_tail_perturbation.py`, `test_modifier.py`, `test_runner.py`, `test_fidelity_cache.py`, extensions to `test_objective.py`, pins in `test_editing_loop.py` | per-task |

Design & plan (methodology of record): `docs/superpowers/specs/2026-07-08-supply-lift-editing-design.md`,
`docs/superpowers/plans/2026-07-08-supply-lift-editing.md`. Execution ledger (every incident, review,
and number's origin — **gitignored scratch, machine-local**): `.superpowers/sdd/progress.md`.

## 8. Gates: what was proven, with final numbers

| Gate | Claim | Outcome |
|---|---|---|
| **G0** | Oracle ceiling ≥ +0.3 | **+0.786 supply-only** (2.6×); full +0.882 (§2) |
| **G1** | `TAIL_LEN=0`/`LIFT_BUDGET=0` ⇒ bit-for-bit the published pipeline | Exact-equality end-to-end test green; lift selection provably never invoked on legacy path |
| **G2** | Editor's ΔS convention vs honest recount | Recount reproduces the production supply grid **exactly** (MAE 0.0, corr 1.0, 34,524 cells); **100%** of 9,885 edit histories matched to raw pings; tier-1 vs tier-2 quantified (§10) |
| **G3** | Trim pickups + demand grid identical to legacy in combined runs | Proven at production scale: combined run reproduced the published 2,455 trim edits and F_causal 0.7988→0.8132 (+0.0144) *before* lift began |
| **G4** | King-move compliance | Lift: **0 violations** (skip-on-infeasible by construction). Trim: 115/2,455 legacy fallbacks (pre-filter 98.85% compliant) → **100% absolute** after the skip-on-infeasible filter (SZ); SF **edit-relative 100%**, absolute 87.40% vs raw baseline 84.95% (14.9% of raw SF trajectories pre-violate — source-data property) |
| **G5** | Fidelity stable | Fidelity-A stable both cities (SZ edited −0.0033 vs published trim-only −0.002; lift-mode −0.0031 ≤ trim-mode −0.0059). Fidelity-B: lift-mode ~0.265 vs trim ~0.16 — by-design distributional cost of tail relocation, disclosed (FINDINGS §6) |
| **G6** | Δmean(Y|D) > 0 with CI; F_causal not regressed; external metrics | SZ: F_causal **+0.0222**; mean(Y|D) **+0.0468** CI [+0.0022, +0.0932]; **supply channel +0.0091 CI [+0.0054, +0.0130] significant** (tier-2 +0.0242, also significant); demand channel n.s. SF: supply channel replicates (+0.0195 sig.) but total mean(Y|D) −0.0330 (demand-endogeneity tension, FINDINGS §5.2 — open PI framing) |

Two hard human checkpoints gated execution: after G0 (2026-07-07) and after the full gate package
(2026-07-08/09), both user-approved.

## 9. Production incidents — the float32-residual family (and their fixes)

The editor's per-edit demand accounting (`-= mass` / `+= mass` on a shared float32 grid, a *legacy*
mechanism) leaves ~1e-9-scale residues on fully-drained cells. Three distinct faces surfaced during the
k = 10,000 validation runs; all were reproduced, root-caused, minimally fixed *without touching the
legacy bit-reproduction path*, and regression-tested:

1. **Crash at first lift edit** (validation attempt 1, after a perfect trim phase): a drained cell's
   residual rounds to −1.86e-9 (proof: 67 aggregated pickups − 67 masses in float32) and
   `compute_fspatial`'s strict non-negativity check raises. Fix `c660140`: one non-autograd
   `torch.clamp(min=0)` on **lift's local demand clone** (unreachable from trim/legacy); regression test
   plants the exact production residual through the real objective.
2. **Silent 12-hour stall on one trim edit** (attempt 2; diagnosed live via `py-spy --locals`: identical
   trajectory and frozen iteration counter across 64 s; a single backward ≥64 s vs ~30 ms normal).
   Attributed to denormal-float poisoning of CPU FP (subnormals are 10–100× slower). Notable: edit
   *order* is tie-nondeterministic across runs (exact ties exist in edit scores), so identical configs
   can meet different float32 states. Hardening `abfef82`+`1bd7f29`: `torch.set_flush_denormal(True)`
   guarded on `TAIL_LEN > 0` (all taper-mode runs; `TAIL_LEN=0` legacy keeps the historical FP
   environment for bit-reproduction — documented process-global side effect at the call site,
   `runner.py:387-407`), plus per-100-edit progress/timing logs and flushed logging.
3. **Crash in final after-metrics** (the 300+150 GPU smoke): the *final* pickup grid carries the same
   negative residuals; legacy passes `current_pickup_3d()` raw and survives on luck (trim-only rarely
   fully drains cells — empirically 28/200 drained cells round negative; lift's clustered relocation
   breaks the luck). Fix `0011cd4`: `np.clip(pickup_after, 0, None)` at the single fetch point,
   `TAIL_LEN > 0` guard; the reviewer independently reproduced the pre-fix crash through the real
   `run_experiment` path.

**Standing invariants from this family** (binding on future work):
- **Trim edits must never run *after* lift edits within a run** unless the trim read path gains the same
  sanitization (current plan order — all trims first — is load-bearing).
- **Downstream consumers of the mutated grids must treat |v| < 1e-6 as zero** (the drift predates the
  branch; published results were computed on top of it).
- **FTZ is process-global**: a taper-mode `run_experiment` leaves flush-denormal on for any later call in
  the same process (runner CLI is one-shot per process; library callers beware).

## 10. Two-tier supply accounting (tier-1 vs tier-2)

- **Tier-1 (the optimizer's convention):** fractional presence mass (1 state = 1/12 hour, 5×5
  replication). This is what `delta_supply_3d.npz` stores and what the objective optimizes.
- **Tier-2 (the honest semantics):** `analysis/supply_recount.py` re-derives supply from **raw GPS** as
  *distinct taxis per 5×5-neighborhood-hour* with edited tails substituted for their original pings,
  using the *production* counting/aggregation functions (reproduction of the untouched grid is exact:
  MAE 0.0). A driver already present in a neighborhood-hour adds no distinct presence — at smoke scale
  ~95% of tier-1-touched cells netted zero, and tier-2 |ΔS| ≈ 0.25–0.30× tier-1.
- **The production surprise (why both tiers must always be cited):** on the metric that matters, tier-2
  *exceeded* tier-1 — Shenzhen supply-channel Δmean(Y|D) **+0.0242 (tier-2) vs +0.0091 (tier-1)**, both
  significant. Smaller total mass, but the *distinct* supply that does land concentrates where it counts.
  Cite tier explicitly whenever quoting a single supply number.
- **SF gap:** the tier-2 recount is **not yet plumbed for SF** (Cabspotting ping pipeline differs; the
  flag exists but exits with deferred status). SF's supply number (+0.0195) is tier-1 only — a lower
  bound of the honest count on Shenzhen's evidence.

## 11. Results (summary — FINDINGS.md is canonical)

Headline (Shenzhen PRIMARY filtered): F_causal 0.7988 → **0.8210 (+0.0222**, vs +0.0144 trim-only);
mean(Y|D) for the migrant disadvantaged group **rises for the first time** (+0.0468, CI excl. 0), with
the supply channel significant under both accounting tiers; Theil −0.0082 (CI-solid); DI +0.0155. SF:
every external metric improves (incl. the previously-immovable migrant axis) and the supply channel
replicates (+0.0195, sig.), but total mean(Y|D) is net-negative (−0.0330) because lift also routes
*demand* into under-served cells — the demand-endogeneity tension (FINDINGS §5.2, open PI framing
decision). Downstream: weighted-BC propagates F_causal (+0.0310 @ w30, 6/6) and — new vs trim-only —
F_spatial (+0.0057 @ w30, 6/6); rollout allocation drain is attenuated ~40% but **not reversed**
(−0.0029 vs −0.0048 @ w30) — claim data + metric levels, disclose the allocation boundary.

Run logistics for reproduction: Shenzhen validation run = 7h54m on an RTX 3070 (k = 10,000, cached
stack); results dirs (gitignored, with their own `PROVENANCE.md`):
`famail_temporal/results/2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered/` and
`...T22-43-06_supply_lift_v1_sf12_filtered/`.

## 12. Performance notes

- **The fidelity discriminator's LSTM dominates the editor** (~70–75% of every optimization iteration,
  both modes). Lift is *not* intrinsically slower than trim per iteration — lift candidates are ~2×
  longer trajectories (mean ~50 states vs ~22 corpus-wide), a **length confound**; any lift-vs-trim
  timing comparison must control for `n_states`.
- **Constant-encoding cache** (`85c6dbc`): `cache_constant_streams()` context manager memoizes the
  iteration-invariant stream encodings (only the live tail stream `trip_s2` is recomputed);
  **bitwise-verified on CUDA** (objective values, every term, gradients — cached vs uncached, both
  modes, multiple lengths), active on all paths including legacy. Production effect: lift edits ~6.2 s →
  ~2.5 s; the full k = 10,000 run 15.5 h (projected uncached) → 7h54m measured.
- **Not shipped (candidate, needs sign-off):** batching x2's constant slots would cut most remaining
  encodes but measured **non-bitwise** on CUDA (~7e-9 GEMM drift) — rejected to preserve bit-exactness;
  revisit only with an explicit numerics decision.
- Ops: long runs need `PYTHONUNBUFFERED=1` (4 KB block-buffering made a 14 h run blind); healthy GPU
  signature on the RTX 3070 eGPU is P2/P3 at ~20–40% util (the loop is Python/launch-bound, not
  GPU-bound); a silent stall now alarms within 30 min via the per-100-edit progress lines.

## 13. Methodological positioning (metric firewall)

Three rings, drawn honestly (the paper should state this):
1. **Optimized:** F_spatial, F_causal, F_fidelity — what the editor's gradient sees (supply now inside).
2. **Design-targeted, not optimized:** mean(Y|D)/SDR-family — the supply mechanism was *aimed* at these
   (G0/G6 gate criteria); improvement is confirmatory, not surprising, and must be labeled as targeted.
3. **Genuinely external:** DP gap, disparate impact, Theil, per-group service levels, the tier-2
   distinct-taxi recount, and the channel decomposition. "Improves metrics we never optimized" claims
   ride on this ring only.

## 14. Known limitations & open items

From FINDINGS §10 plus implementation-level items not listed there:
- SF tier-2 recount plumbing deferred; SF supply number is tier-1 only (§10).
- SF total mean(Y|D) net-negative — PI framing decision pending (FINDINGS §5.2).
- Filtered "survivors" were not re-optimized after the skip-on-infeasible reverts (approved
  negligible-coupling trade-off; grids exact for "surviving edits applied to base").
- Two alternate Shenzhen feature-set runs deferred (robustness parity with the trim-only sweep).
- Rollout allocation still net-negative (attenuated ~40%); seeking-state shares unmoved — motivates
  training-side allocation constraints as future work.
- **Unified one-pass trim+lift is the natural v2** (run everything supply-endogenous; let the gradient
  set each edit's character). Deliberately not done this cycle: freezing trim is what yields bit-level
  reproduction of published results inside the combined run and a clean trim-only vs trim+lift ablation.
  Unification requires a common selection currency (trim's attribution vs lift's linearized gain) and
  periodic gradient recomputation if interleaved.
- Minor code-quality items from reviews (docstring nits, defensive asserts, test-device pinning for the
  bitwise cache test, a bounded assert before the lift-path clamp) are recorded in the execution ledger
  (`.superpowers/sdd/progress.md`, "minors-for-final-triage" entries).

## 15. Document index (everything that bears on lift)

| Document | Role |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | Canonical curated results narrative |
| [`data_provenance.md`](data_provenance.md) | Every load-bearing number → artifact path + commit |
| [`data/`](data/), [`tables/`](tables/), [`figures/`](figures/) | Committed artifacts (JSON, report tables, forest plots) |
| `../external-metrics/LEVELING_DOWN_MECHANISM.md` | Motivation deep-dive (mechanism analysis, oracle bounds, rollout eval §6.4) |
| `../external-metrics/FINDINGS.md` | The trim-only external-metrics result this responds to |
| `docs/superpowers/specs/2026-07-08-supply-lift-editing-design.md` | Approved design (gates G0–G6 defined here) |
| `docs/superpowers/plans/2026-07-08-supply-lift-editing.md` | Implementation plan of record (11 tasks, TDD code) |
| `docs/presentations/meeting_42_update/SUPPLY_LIFT_UPDATE.md` | Presentation summary (numbers verified) |
| `docs/presentations/meeting_42_update/supply_lift_briefing.md`, `trim_plus_lift_explainer.md` | Slide-source explainers (uncommitted drafts) |
| `.superpowers/sdd/progress.md` + task/perf reports | Execution ledger (gitignored, machine-local) |
| `../objective-motivation/` | Objective-function motivation incl. the demand-endogeneity argument invoked by FINDINGS §5.2 |
