# Straight-Through (Hard-Metric) Editing — Design

- **Date:** 2026-06-06
- **Branch:** `algorithm-improvements` (builds on the engine + gate from the multi-loop side project)
- **Status:** Design approved (conversational); gated change, directed by the user → spec → plan → implement.
- **Scope:** A single, focused change — make the trajectory editor optimize/select on the **realizable hard-grid** F-metrics via a straight-through estimator (STE) — plus **one** disambiguating experiment. This does **not** reopen the broader continuous/OT reformulation.

---

## 1. Motivation

§8.7 established that multi-loop re-attribution *degrades* F_causal and bigger ε does nothing, with a characterized root cause: the editor **optimizes and selects on the soft (differentiable) cell-assignment metric but deploys and measures on the hard (discrete) grid**. Two consequences:

- **Best-iterate prefers snap-fragile solutions.** A diffuse (high-temperature) soft distribution can score a better *soft* F_causal via fractional cell allocation; best-iterate keeps it; the hard snap then discards the fractional gain. Direct evidence: A3 round 2's 1,666 edits each improved the soft objective yet collectively dropped hard F_causal by 6.4e-4.
- **Re-attribution dithers.** Each round measures the hard grid, optimizes soft, snaps back — the disagreement at the fine scale turns iterative rounds net-negative.

**The fix is not to snap less often** (deferring the snap chases an unrealizable fractional upper bound — the integrality gap collapses at the final projection). The fix is to **measure the hard metric every iteration while keeping the soft gradient for search**: a straight-through estimator.

**Why it's worth one run — it disambiguates the +0.0128 ceiling:**
- If STE multi-loop **accumulates past +0.0128** → the ceiling was an optimization artifact (the soft-vs-hard gap); we've broken it.
- If it **plateaus at +0.0128** (and, with hard best-iterate, does *not* degrade) → the ceiling is **intrinsic** (the ~1–3% editable slice, each ~1 cell from fair; consistent with the ε=5 result); we've proven it isn't a solver failure.

Either outcome is a stronger result than the current §8.7 state.

## 2. The change

A new opt-in flag routes the per-iteration objective through a straight-through estimator in `modify_single` ([modifier.py](famail_temporal/algorithm/modifier.py)).

**Today** (per iteration): `soft_3d = inject_soft_counts_into_3d(base_3d, probs, …)` → `objective(soft_pickup_3d=soft_3d)`. The forward value is the *soft* (fractional) F-metric; best-iterate and the acceptance gate read those soft values.

**With STE** (`use_ste=True`): also build the **hard** grid — full pickup mass at the cell `int(current_pickup)`, which is *exactly* the cell the persist step writes (`new_cx = int(modified.pickup_state.x_grid)`). Note we snap by `int(current_pickup)`, **not** `argmax(probs)`: the soft assignment measures distance to cell *centers* (`int+0.5`) while the pickup sits at the integer corner, so `argmax` can tie-break to the wrong cell at integer coordinates (e.g. iter-0). Then stitch:

```python
soft_3d = inject_soft_counts_into_3d(base_3d, probs, (orig_cx, orig_cy),
                                     t_block, k=self.soft_assign.k, pickup_mass=pickup_mass)
if self.use_ste:
    k = self.soft_assign.k
    # Hard cell = int(current_pickup): the exact cell the persist step writes.
    snap_x, snap_y = int(current_pickup[0]), int(current_pickup[1])
    ox, oy = snap_x - orig_cx + k, snap_y - orig_cy + k       # index into (2k+1, 2k+1)
    hard_probs = torch.zeros_like(probs)
    if 0 <= ox < probs.shape[0] and 0 <= oy < probs.shape[1]:
        hard_probs[ox, oy] = 1.0
    hard_3d = inject_soft_counts_into_3d(base_3d, hard_probs, (orig_cx, orig_cy),
                                         t_block, k=k, pickup_mass=pickup_mass)
    objective_grid = hard_3d + (soft_3d - soft_3d.detach())   # forward = hard; grad via soft
else:
    objective_grid = soft_3d
total, terms = self.objective(soft_pickup_3d=objective_grid, …)
```

`hard_probs` (one-hot) carries no gradient, so `hard_3d` is constant; `soft_3d - soft_3d.detach()` is numerically zero but carries the soft gradient. Therefore:

- **Forward value** of `total`/`terms` = the **realizable hard** F-metrics (mass concentrated in one cell, exactly what gets persisted and what cGAIL consumes).
- **Backward** gradient flows through `soft_3d` → the continuous pickup coordinate (a biased-but-informative search direction; the true hard-metric gradient is zero a.e.).

## 3. What this automatically fixes (no other code changes)

Because the existing best-iterate tracking and the acceptance gate read `total`/`terms.f_causal`/`terms.f_spatial`, feeding them the hard-valued stitched grid makes **selection and gating hard-based for free**:

- **Best-iterate** now keeps the iterate with the best *realizable* objective → it can no longer prefer diffuse, snap-fragile solutions.
- **iter-0 baseline** (`cumulative_delta=0` → pickup at original → snapped to original cell) = the trajectory's true pre-edit hard state, so the gate compares hard-to-hard.
- **Multi-loop cannot degrade**: with hard best-iterate (and either gate), a round only persists edits that improve the realizable metric. Worst case it equals single-pass; it cannot dither negative.
- **No final-snap surprise**: every iterate is already evaluated on the hard grid the persist will write (the STE hard cell is `int(current_pickup)`, identical to the persist's `int(modified.pickup_state.x_grid)`), so optimized == measured == deployed.

## 4. CLI / config surface

| Knob | Default | Meaning |
|---|---|---|
| `--ste` (config `STE_ENABLED`) | `False` | Route the objective through the straight-through hard-metric estimator. Off = today's soft behavior, bit-identical. |

Constructed like the other modifier knobs: `TrajectoryModifier(..., use_ste=None→config.STE_ENABLED)`; threaded `runner → run_experiment → modifier`. No other flags change.

## 5. Backward compatibility

`--ste` defaults off → `objective_grid = soft_3d` → **bit-identical** to current behavior. All existing results, tests, and the §8.7 findings stand. Verified by the full suite (the new flag is purely additive).

## 6. Experiment (one run + comparisons that already exist)

**E2 — STE multi-loop.** Identical to A3 except `--ste`:
```
-k 10000 --max-rounds 20 --round-convergence-tol 1e-5 --round-patience 2 \
--epsilon-cap 2 --accept-rule objective --ste \
--override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.0
```
One run yields everything:
- **Round 1 = STE single-pass** → compare to the +0.0128 baseline and the soft α_fi=0 single-pass reference (A3 round 1 = +0.01271). Does selecting on the hard metric change the single pass?
- **Rounds 2+** → compare the round curve to A3 (soft multi-loop, which degraded: +0.01271 → +0.01213). Accumulate, plateau, or (it shouldn't) degrade?

Isolation is clean: **A3 vs E2 differ only in STE on/off.** α_fi=0 keeps it cheap (~15 min) and is a validated proxy (§8.7 finding 2).

## 7. Testing strategy (TDD)

- **STE forward equals the hard grid.** Synthetic: for a fixed pickup, `objective(stitched).detach()` equals `objective(hard_3d).detach()` (the soft term cancels in value).
- **Gradient still flows.** With `use_ste=True`, `pickup_tensor.grad` is non-None and finite after backward (the soft path carries it).
- **Default off is bit-identical.** `use_ste=False` reproduces current `modify_single` output on a synthetic trajectory (same modified cell, same history).
- **STE hard cell == persist cell** — for representative pickups (including integer coords like iter-0), the STE one-hot lands at `int(current_pickup)`, matching `int(modified.pickup_state.x_grid)`.
- **STE multi-loop does not degrade** (synthetic drag bundle): the hard round ΔF_causal is ≥ −tol each round (no net-negative rounds), in contrast to the soft path.
- Full suite green (the flag is additive; defaults preserve behavior).

## 8. Documentation

Methods doc **§8.8** — the STE result and its interpretation: whether the +0.0128 ceiling is an optimization artifact (STE accumulates) or intrinsic (STE plateaus, no degradation). Update `STATUS.md` and memory only if the shipped editing config changes (i.e., only if STE beats +0.0128).

## 9. Risks / honest caveats

- **STE gradients are biased** (soft gradient for a step-function objective). This is a known heuristic; near the hard optimum the true gradient is zero, so a biased direction is acceptable — and the hard best-iterate/gate prevent it from *committing* a bad move. Worst case: STE proposes moves the hard metric rejects → no-ops (graceful plateau).
- **The integrality gap is real but now respected.** STE optimizes the realizable hard metric directly; it does not claim the unrealizable soft upper bound. So a plateau is a genuine result, not a solver artifact.
- **Likely outcome (my prior):** given the ε=5 / intrinsic-ceiling evidence, a *plateau with a small lift* is more likely than a step-change — but the run is worth it precisely because it converts a hypothesis into a definitive, paper-ready statement.
- **Deadline:** one focused ~1-day change reusing the existing engine + gate + multi-loop; not a reopening of the continuous reformulation.

## 10. Success criteria

- `--ste` implemented, opt-in, default-off bit-identical, tested.
- E2 run completed; round curve recorded.
- A definitive §8.8 statement: the +0.0128 ceiling is **artifact** (STE > +0.0128 → new best config, update STATUS/memory) or **intrinsic** (STE ≈ +0.0128, non-degrading → ceiling confirmed fundamental).
- Algorithm-change protocol respected (gated change, user-directed); all on `algorithm-improvements`.
