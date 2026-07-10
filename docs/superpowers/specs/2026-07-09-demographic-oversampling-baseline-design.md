# Demographic Oversampling Baseline (Mission-3 4th arm) — Design Spec

**Date:** 2026-07-09 · **Branch:** `worktree-demographic-oversampling` (worktree off `main` `238acec`)
**Status:** approved (brainstorming) → ready for writing-plans
**Mission:** Meeting-41 P0 #3, 4th arm — the naive *resampling* baseline for the supply-lift editor and a
direct empirical probe of the demand-endogeneity / leveling-down limitation. Selected from the lit-scan
(`famail_temporal/baselines/DATA_AUG_BASELINE_CANDIDATES.md`, Candidate 4, Pastaltzidis et al. FAccT'22).

---

## 1. Context & purpose

The three built Mission-3 arms (iFGSM / FGSM / random) are *perturbation* baselines. This arm is the
*resampling* counterpart: duplicate real seeking trajectories originating in demographically disadvantaged
regions to shift the corpus's service balance — the naive cousin of the **supply-lift (trim+lift) editor**,
which is canonical for all PAPER reporting (user decision, 2026-07-09).

The claim the arm supports (pre-registered): *naive demographic oversampling can move the ratio-based
external fairness metrics, but only by fabricating unobserved supply and demand (phantom drivers, phantom
pickups) and inflating the corpus; FAMAIL's gains come from redistributing real observed behavior at zero
corpus inflation.* Mechanically we EXPECT the ratio metrics to improve (each duplicate adds supply along its
whole trail but demand at only one pickup, so Y = S/D rises in targeted regions) — the arm quantifies **how
much apparent fairness pure fabrication buys**, and the dose-response gives the fabrication-to-gain exchange
rate. It is also the direct probe of the demand-endogeneity critique (a duplicate's pickup is *unobserved*
demand) and the naive form of "lifting up" (raises the disadvantaged group without touching the advantaged —
the intended contrast with the leveling-down caveat in `PAPER/external-metrics/FINDINGS.md`).

Comparator: the SZ-primary filtered trim+lift headline
`famail_temporal/results/2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered/`
(`deltas.f_causal = +0.022218`, `deltas.f_spatial = +0.006357`), transcribed via the existing stub-file
convention, never recomputed.

## 2. Decisions locked during brainstorming

1. **Data-level v1** (user): inflate the corpus and rescore data-level fairness like the other 3 arms;
   training-level oversampling is already covered in spirit by weighted-BC + its placebo.
2. **Additive demand+supply grid rebuild** (user, pre-decided at kickoff): rebuild BOTH demand (pickups)
   and supply (tier-2 seeking presence) additively. Demand-only is perverse (adds demand to under-served
   cells → lowers their service ratio).
3. **Synthetic driver IDs** (user): each duplicate is a phantom driver with a fresh namespaced plate ID.
   Required by tier-2 distinct-count semantics — a duplicate under its source plate adds zero supply
   wherever the real driver was already present per (5×5 neighborhood, hour). Physical story: "an extra
   taxi ran the same seeking run." Disclosed as fabricated supply — which IS the naive-baseline story.
4. **Targeting = all three equity axes via the evaluation convention** (user): per axis in `EQUITY_AXES`
   (housing, comp, migrant), disadvantaged regions = `region_extremes(frac=1/3, DISADVANTAGED_HIGH[axis])`
   over the cell-level demographic values — identical to the definition the external-metrics reporting
   uses, so `n` regions is convention-anchored, not arbitrary. Eligibility: a seeking trajectory's
   **origin cell** (first seeking state) lies in the axis's disadvantaged regions. (Rejected: pickup-cell
   targeting — adds fabricated demand exactly where under-service is claimed, muddying the probe;
   single-axis targeting — user requires all three axes to shape the sample.)
5. **Per-axis quota allocation** (user): dose B split ≈ B/3 per axis (deterministic remainder), uniform
   without-replacement within each axis's pool, strata drawn in fixed axis order so overlap trajectories
   are not drawn twice. Pool-smaller-than-quota degrades to with-replacement + warning + diagnostic flag.
6. **Budget-parity headline dose = 10,000 duplicates** (user), mirroring FAMAIL's k=10000 edit budget;
   smaller doses {2,500, 5,000} form the dose-response curve.
7. **Rigid whole-trajectory shift** (approved): one offset per duplicate, uniform over the L∞ radius-1
   ball excluding (0,0), clipped at grid boundaries; time buckets and day index unchanged. Preserves
   internal adjacency exactly (`adjacency_violation_rate` trivially 0), guarantees non-identity.
   (Rejected: per-state independent jitter — manufactures adjacency violations; pure re-weighting — adds
   no tier-2 supply at all.)
8. **Random-oversampling placebo** (approved): identical machinery, sources drawn uniformly from the
   WHOLE seeking corpus. Separates demographic *targeting* from mere corpus *inflation* — the additive
   mirror of the weighted-BC random placebo.
9. **Approach A — standalone, self-scored module** (user): new module + runner; zero changes to the frozen
   editor, `evaluation/runner.py`, or the existing substitution-semantics CLIs (`run_external_fairness`,
   `supply_recount`). Additive semantics live in one new, independently testable place.
10. **Scope = Shenzhen only, v1** (approved): consistent with `supply_recount`'s SZ-only scoping; SF
    replication deferred.
11. **Fidelity not re-scored** (approved): duplicates are (near-)copies of real trajectories → Fidelity-A/B
    trivially perfect by construction; stated + disclosed per `baselines/STATUS.md`. Optional one-off CPU
    spot-check only if a reviewer-facing number is wanted.

## 3. Components

### 3.1 Engine — `famail_temporal/baselines/demographic_oversampling.py` (new)

Pure functions, no CLI I/O. Core API (names final):

```python
EPS_SHIFT = 1          # rigid-offset L-inf radius (cells)
REGION_FRAC = 1.0/3.0  # region_extremes convention, shared with evaluation

def disadvantaged_cell_masks(selected_grid) -> Dict[str, np.ndarray]
    # {axis: (GX, GY) bool} via region_extremes(frac=REGION_FRAC, DISADVANTAGED_HIGH[axis])
    # applied to the cell-level values from external_fairness_io._enriched_selected_grid().

def eligible_pools(trajectories, masks) -> Dict[str, np.ndarray]
    # {axis: indices of seeking trajectories whose ORIGIN cell is in the axis's mask}.

def sample_duplicates(pools, dose, seed, placebo=False) -> List[DuplicateSpec]
    # Per-axis quotas (deterministic remainder), sequential without-replacement with cross-stratum
    # dedupe; with-replacement fallback flagged. placebo=True ignores pools: uniform over the corpus.
    # DuplicateSpec records: source index, drawing stratum, eligible-axes set, offset, phantom ID.

def make_phantom(traj, spec) -> Trajectory
    # Deep copy; fresh namespaced phantom plate ID (cannot collide with real plates);
    # rigid (dx, dy) shift on every state, clipped to grid bounds; times/days unchanged.

def additive_demand(bundle, phantoms) -> np.ndarray
    # D' = bundle.pickup_3d(float64) + pickup_mass(bundle, t) at each phantom's pickup (cell, hour).
    # Existing per-event mass convention (datasets.pickup_mass), ADDED, never relocated/floored.

def additive_supply(bundle, phantoms) -> np.ndarray
    # S' = bundle.active_taxis_3d + Σ per-phantom presence grids. Per phantom: mark the clipped
    # 5×5 neighborhood of each state at its hour, OR over the phantom's states per (cell, hour)
    # — tier-2 distinct count for a driver that did not previously exist — then normalize
    # IDENTICALLY to the production active_taxis counter (mean-hourly over n_days; the exact
    # normalizer is pinned against data/source_generation/views/active_taxis.py in the plan and
    # enforced by the fixture + dose-0 tests). Phantom IDs are fresh → contributions are
    # independent and purely additive; no raw-GPS resegmentation is needed.
```

### 3.2 Runner — `famail_temporal/baselines/run_demographic_oversampling.py` (new)

CLI: `--dose N --seed S --variant targeted|placebo [--city shenzhen] --out-root ...`. Steps:

1. Load the cached SZ PRIMARY `DataBundle` (same bundle the trim+lift headline used).
2. Build pools → sample → materialize phantoms.
3. Build `(D', S')`; score:
   - **External metrics:** `Y_before = service_ratio_Y(bundle.pickup_3d, bundle)`,
     `Y_after = service_ratio_Y(D', bundle, S')`; then the same spec set as `run_external_fairness`
     (DP / DI / SDR gap+means / Theil; median_split + region_extremes per axis; `paired_bootstrap`
     CIs) computed via the **pure** functions in `external_fairness.py`, emitting the same JSON
     schema as the existing external-fairness reports.
   - **F_causal / F_spatial** under `(D', S')` via the substituted-grid evaluation seam
     `supply_recount` already validated (reuse/lift its call pattern; exact helper pinned in the plan).
4. Write the arm dir `results/<ts>_baseline_demo_oversample_{targeted|placebo}_d<dose>_s<seed>_shenzhen/`:
   - `duplicates.pkl` — phantoms + full provenance (DuplicateSpecs). **Deliberately NOT named
     `histories.pkl`**: the substitution-semantics tools (`run_external_fairness`, `supply_recount`)
     must fail loudly on this dir rather than silently mis-score an additive corpus.
   - `metrics.json` — `"arm"` (config + diagnostics), `"fairness"` (before/after/Δ F_causal,
     F_spatial), `"external_fairness"` (per-metric before/after/Δ + CIs), shaped so
     `assemble_baseline_table` can ingest the row (adapter verified/extended in the plan).

Diagnostics always recorded in `metrics.json["arm"]`: n_duplicates, per-axis draw counts, overlap counts,
with-replacement flag, origin- and pickup-region-escape fractions (post-shift), boundary-clip count,
corpus inflation fraction, seed, dose, variant.

### 3.3 Table integration

Headline row (targeted @ 10,000) added to the Mission-3 baseline comparison table beside raw / FAMAIL /
ifgsm / fgsm / random; dose curve + placebo to an appendix table and a small Δmetric-vs-dose figure
(targeted vs placebo lines). Fidelity cells for this arm read "≈ perfect by construction (real duplicates)"
— disclosed, not computed.

## 4. Experiment matrix (all CPU; no GPU contention with the α-sweep)

| Variant | Doses | Seeds | Purpose |
|---|---|---|---|
| targeted | 2,500 / 5,000 / 10,000 | 0 | dose-response curve |
| targeted | 10,000 | 0, 1, 2 | sampling-variance check at the headline dose |
| placebo | 5,000 / 10,000 | 0 (+ 1, 2 at 10,000) | inflation-only control at matched dose |

Nine arm dirs total. City: Shenzhen (v1). Targeted-minus-placebo at matched dose isolates the effect of
demographic targeting from corpus inflation.

## 5. Testing

Unit tests (`famail_temporal/baselines/tests/test_demographic_oversampling.py` +
`test_run_demographic_oversampling.py`), mirroring the other arms' suites:

- Region selection reproduces the `region_extremes(frac=1/3)` evaluation convention exactly.
- Quota allocation: deterministic remainder; cross-stratum dedupe (overlap trajectory drawn once);
  with-replacement fallback triggers + flags on a small synthetic pool.
- Rigid shift: offset ≠ (0,0); boundary clip; internal adjacency and time buckets preserved;
  same-seed determinism (identical corpus byte-for-byte).
- Phantom supply: two states in the same (5×5 neighborhood, hour) count once; two phantoms in the same
  cell-hour count twice; normalization matches the production `active_taxis` counter on a synthetic
  fixture.
- Additive demand: `sum(D' − D_base) = Σ pickup_mass(bundle, t_dup)` (mass conservation, no flooring).
- **Dose-0 identity:** at dose 0 the pipeline reproduces the unmodified baseline metrics exactly.
- Phantom-ID non-collision with real plates; provenance round-trips through `duplicates.pkl`.
- `metrics.json` ingestible by `assemble_baseline_table`.

## 6. Error handling

- Empty axis pool → hard error (misconfiguration; SZ PRIMARY should never produce one).
- Pool < quota → with-replacement + stderr warning + `with_replacement: true` diagnostic.
- Boundary clips, origin/pickup region-escape fractions: always reported, never silently constrained.
- NaN conventions follow the existing external-fairness code (excluded units labeled −1, NaN-safe means).

## 7. Ship gates

- Frozen-algorithm gate: `git diff main -- famail_temporal/algorithm/ famail_temporal/evaluation/runner.py`
  = 0 lines throughout the branch.
- Full suite green: `python -m pytest famail_temporal/ -q` (baseline at branch time: 849 passed, 8 skipped).
- Run-book + results appended to `famail_temporal/baselines/STATUS.md` Mission-3 section.

## 8. Deferred (explicitly out of scope for v1)

- SF replication (needs SF supply plumbing; same deferral as `supply_recount`).
- Training-level oversampling probe (Pillar-2 question; weighted-BC + placebo already cover the
  training-level story).
- Fidelity-A/B spot-check on duplicates (only if a reviewer-facing number is requested).
- Severity-weighted (service-deficit-proportional) allocation — a *less* naive variant; noted as future
  work, not built.
