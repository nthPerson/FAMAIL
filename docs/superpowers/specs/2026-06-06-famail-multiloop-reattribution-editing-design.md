# Multi-Loop Re-Attribution Editing + Non-Regression Acceptance — Design

- **Date:** 2026-06-06
- **Branch:** `algorithm-improvements` (branched from `implement-gan-baselines`)
- **Status:** Design approved (brainstorming); spec under review → writing-plans next.
- **Scope:** A time-boxed side trip to squeeze more ΔF_causal out of the
  trajectory-editing algorithm itself, before resuming the parked main thread
  (GAN baselines / model-level / Phase 4). Two algorithm changes are
  pre-authorized by the team (multi-loop re-attribution; a both-metrics
  acceptance rule). Everything else affecting *what the algorithm computes*
  remains gated per the algorithm-change protocol.

---

## 1. Motivation & questions

Today's editing pipeline is a **single pass**: attribute once → select top-k
negative-α trajectories → perturb each within ε=2 → done. The strongest result
to date is the reference baseline:

- Dir: `famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`
- Config: `α=(0.2, 0.7, 0.1)`, `k=10000`, no `--max-per-unit`, `MAX_ITERATIONS=50`.
- Result: **ΔF_causal = +0.0128**, ΔF_spatial = +0.0003, 3,773 modified, 3,765 converged.

Questions to answer:

1. **(Primary)** Does running multiple "re-attribute → edit all attributed"
   rounds beat the single pass on ΔF_causal?
2. What is the optimal number of rounds?
3. Is a convergence threshold more principled than a fixed round count?
4. (Survey, separate) Other low-hanging fruit that does **not** change the
   fairness-metric formulations.

Overarching goal (per user): **find the ceiling** — how much F_causal
improvement is achievable — prioritizing F_causal as the most direct proxy for
the human bias the edits aim to remove.

## 2. Background that constrains the design

- **ε=2 (`EPSILON_BALL`) within a single edit is inviolable** — it matches the
  cGAIL IL model's 5×5 training distribution. The *cumulative across-rounds*
  policy is the new degree of freedom (see §3.3).
- **§8.2 — fine-grained re-attribution was already null.** `--iterative-topk`
  (re-attribute after *every single* edit) was tested at k=20 and produced
  **bit-identical** results to single-pass batch: per-edit pickup mass
  (~0.04/block) is 0.1–1% of a destination cell's demand, too small to flip the
  αᵢ ranking between consecutive single edits. The doc flagged this might change
  "at larger k, longer iter budgets." → Batch granularity (re-attribute after a
  *whole batch* of ~3,773 edits) is the untested regime where re-attribution can
  have teeth. This is the core bet.
- **§8.3 — ε=2 is almost never binding.** At k=1000: mean movement 1.15 cells,
  median 1.00, **only 2.6%** of trajectories reach the ε-ball boundary.
  Implication: cumulative-ε relaxation buys movement mostly for (a) that ~2.6%
  and (b) trajectories whose *re-attributed* gradient points outward — modest a
  priori upside, but real cost (drift past ε=2 leaves the 5×5 IL window and may
  wake the currently-dormant F_fidelity term). Hence ε-cap is a *tunable
  ablation axis*, not a hardcoded choice.
- **F_spatial is nearly insensitive.** Its gradient is ~20× smaller than
  F_causal's; it "never wins" a gradient-sign decision; corpus ΔF_spatial ~1e-4
  vs ΔF_causal ~1e-2. This is why the acceptance rule is **non-regression**, not
  strict-improve-both (see §3.2).
- **F_fidelity is dormant at ε=2** (discriminator saturates), so `α_fi=0` is a
  faithful, ~13× cheaper proxy for bounded-ε runs (per-iter ~94 ms → ~7 ms).

## 3. Design

### 3.1 Unified re-attribution-loop engine (orthogonal knobs)

A single outer-loop engine wraps the existing per-trajectory modifier. Today's
single pass is the **R=1 special case**, so default behavior is unchanged.

Each **round**: re-attribute against the live grid
(`modifier.current_pickup_3d()` → `compute_per_unit_attribution`) → select the
eligible editable set → edit it via `modify_single` → evaluate the stop rule.

Four **orthogonal** knobs (your "two parallel algorithms" are points in this space):

- **Granularity B** — `batch` (default; edit *all* eligible negative-α this
  round, against the round-start attribution snapshot) or `--iterative-topk`
  (B=1; edit the single most-negative eligible trajectory, re-attribute every
  edit).
- **Outer loop** — `--max-rounds R` (default **1**) and/or
  `--round-convergence-tol τ_outer` with `--round-patience P_outer`.
- **Cumulative-ε cap C** — `--epsilon-cap` (default **2.0**; accepts `inf`).
- **Inner gate** — `--accept-rule {objective, non-regression}` (default
  **objective**, for backward-compat).

**Eligibility (per trajectory, per round):** `αᵢ < 0` **and** cumulative
displacement from the *true original* cell `< C`. For B=1, additionally
edit-count `<` the multi-edit cap (`--iterative-topk-max-edits`; `0` = unlimited).
Batch mode has no separate edit-count cap — the ε-cap `C` is the natural limiter
(a trajectory at cumulative `C` becomes ineligible). No `--max-per-unit` in the
headline runs (full-batch = "edit all attributed").

**Within-round vs between-round:** batch mode edits all eligible against a single
round-start attribution snapshot (within-round interference/pile-up is *expected*
and is exactly what the next round's re-attribution is meant to correct). B=1
re-attributes after every edit (no within-round pile-up).

### 3.2 Inner-loop non-regression acceptance gate

At iteration 0 a trajectory sits at its original pickup, so `F_causal₀`,
`F_spatial₀` (global, this-trajectory-included) are captured for free
(`modify_single` already computes the objective terms there). The gate changes
**which iterate is persisted as "best"** and what resets the patience counter:

- `objective` (default, current behavior): keep the iterate with the best
  weighted total objective `L` (`modifier.py:473`).
- `non-regression` (the team's rule, chosen reading): keep the best-`L` iterate
  **among those satisfying** `F_causal(it) ≥ F_causal₀ + τ` **and**
  `F_spatial(it) ≥ F_spatial₀ − τ`. If none qualify → no edit (delta=0; existing
  fallback). τ reuses `CONVERGENCE_TOL` (1e-6, above the float64 noise floor).

The gradient still optimizes the weighted objective as today; only acceptance
changes.

> **Team-reconciliation note (surface to PI/team):** this implements "improve
> both" as *improve-primary / non-regress-other*. §2 shows strict-improve-both
> would likely veto most F_causal-improving edits (F_spatial dips on ~half of
> them) and collapse the edit count. The strict variant is available as the
> `objective`-vs-`non-regression` ablation axis if the team wants the data.

### 3.3 Cumulative-ε cap from the true original

`modify_single` currently measures ε from `trajectory.states[-1]`
(`modifier.py:286-313`, `:426-437`), so feeding a modified trajectory back in
*naturally* yields per-round ε=2 with unbounded cumulative stacking. To support a
bounded cap the engine tracks each trajectory's **true original cell** across
rounds and constrains `‖pickup − original‖_∞ ≤ C`:

- `C = 2.0` (default): bounded to the 5×5 IL window; a re-edited trajectory only
  uses leftover headroom (~0.85 cells avg given mean movement 1.15). Most
  defensible.
- `C = inf`: your literal per-round-ε=2 proposal; unbounded drift. Used to probe
  the ceiling and to detect fidelity activation.

Implementation: the engine passes `original_cell` and `C` into `modify_single`;
the cumulative-delta projection (`modifier.py:426-437`) clips against the
true-original-anchored ball rather than the round-start cell.

### 3.4 Outer-loop stop rule

The loop halts at the **first** of:

1. `--max-rounds` reached — the hard ceiling, **always enforced** (also bounds
   the `C=inf` case if it chases a drifting gradient);
2. **convergence** — round-over-round global ΔF_causal `< τ_outer` for
   `P_outer` consecutive rounds (best-round tracked, mirroring the inner-loop
   best-iterate+patience design in §2.2 of the methods doc);
3. **pool exhaustion** — no eligible (negative-α, under-cap) trajectories remain
   (always on).

In convergence mode (`--round-convergence-tol` set), `--max-rounds` acts as the
safety ceiling — set it to e.g. 20. A warning fires if convergence is enabled
while `--max-rounds` is still at its default 1 (which would force a single pass).

Round-by-round global F_causal / F_spatial / n_edited are logged so the round
curve answers Q2 (optimal #rounds = where the curve flattens) and Q3
(convergence round vs fixed-R, post-hoc, no extra runs).

### 3.5 CLI / config surface

New CLI flags on `runner.py` (config defaults in `config.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--max-rounds INT` | `1` | Outer-loop hard ceiling, always enforced; doubles as the convergence-mode safety cap (`MAX_ROUNDS`). R=1 = current single pass. |
| `--round-convergence-tol FLOAT` | `None` (off) | Enable convergence stop; τ_outer on round ΔF_causal (`ROUND_CONVERGENCE_TOL`). |
| `--round-patience INT` | `2` | Outer-loop patience (`ROUND_PATIENCE`). |
| `--epsilon-cap FLOAT` | `EPSILON_BALL` (2.0) | Cumulative L∞ displacement cap from true original; accepts `inf` (`EPSILON_CAP`). |
| `--accept-rule {objective,non-regression}` | `objective` | Inner acceptance gate (`ACCEPT_RULE`). |
| `--iterative-topk-max-edits INT` | `1` | Multi-edit cap for B=1 (`--iterative-topk`); `0` = unlimited (`ITERATIVE_TOPK_MAX_EDITS`). |

Existing flags unchanged: `--iterative-topk` (B=1 preset), `-k`,
`--max-per-unit`, `--max-per-cell`, `--patience`, `--convergence-tol`,
`--device`, `--diagnostics`, `--override`. The "batch multi-loop" needs **no new
preset** — it is the default batch path with `--max-rounds > 1` and/or
`--round-convergence-tol` set.

### 3.6 Backward-compatibility & equivalence locks (tested)

- `--max-rounds 1` + batch + `objective` gate + `epsilon-cap 2` ⟹ **bit-identical**
  to today's batch-topk (`_modify_with_progress`).
- `--iterative-topk` + `--iterative-topk-max-edits 1` + `--max-rounds`≥pool ⟹
  **bit-identical** to today's `_iterative_topk_modify` (the §8.2 property: same
  IDs, order, destinations, mean_iters).

These locks make the unification a numerics-preserving refactor under the
algorithm-change protocol, not a silent behavior change. Any unavoidable
reordering must be documented and justified.

## 4. Scope

**Build:** the unified engine; the four orthogonal knobs; the non-regression
gate; cumulative-ε tracking from true original; outer-loop controls + round-curve
logging; the multi-edit `--iterative-topk` option (approved — and reused by the
A4 ablation); tests; docs.

**Defer / out of scope:** any change to attribution math, fairness-metric
formulations, soft-cell assignment, or the within-edit ε=2; B1
differentiable-fairness work; all GAN-baselines / Phase-4 work (parked,
untouched). The question-4 "low-hanging fruit" survey is delivered as a short
written list, not implemented here.

## 5. Experiment matrix

All runs: full-batch (edit all attributed), `k=10000`, no `--max-per-unit`,
`α_spatial=0.2 / α_causal=0.7`, convergence-stopped (`--round-convergence-tol`
set, `--round-patience 2`) unless noted. Reference R0 already exists. `α_fi` is
toggled (0 vs 0.1) with α_spatial/α_causal held fixed; since the ST-iFGSM step is
sign-based (`STEP_SIZE_ALPHA`-scaled), no renormalization is needed and the
`α_fi=0`-vs-`0.1` contrast cleanly isolates the fidelity term's influence.

| Run | B | α_fi | Gate | C | Purpose |
|---|---|---|---|---|---|
| **R0** *(exists)* | K | 0.1 | objective | 2 | single-pass reference (+0.0128) |
| R0′ | K | 0 | objective | 2 | single-pass α_fi=0 reference |
| **A1** | K | 0 | non-reg | 2 | bounded-ε result; round curve → Q2/Q3 |
| **A2** | K | 0 | non-reg | ∞ | unbounded ceiling = max reachable F_causal |
| **A3** | K | 0 | objective | 2 | gate ablation (vs A1) |
| **A4** | **1** | 0 | non-reg | 2 | **B=1 vs B=K equivalence ablation** (vs A1); cost-gated, see below |
| **H1** | K | 0.1 | non-reg | 2 | defensible in-distribution headline vs +0.0128 |
| **H2** | K | 0.1 | non-reg | ∞ | ceiling at α_fi=0.1; fidelity-activation check |

**Headline** = whichever of H1/H2 has the best F_causal, with the
in-distribution (C=2) vs out-of-distribution (C=∞) nuance documented explicitly.
Report *both* numbers.

**Comparisons:** Q1 = H1 vs R0 (and A1/A2 vs R0′); optimal-rounds/convergence
(Q2/Q3) = A1/A2 round curves; ε-stacking = A2 vs A1; acceptance gate = A3 vs A1;
α_fi / fidelity realism = H1 vs A1 and H2 vs A2; **B-granularity equivalence =
A4 vs A1** (does §8.2's bit-identity survive into the multi-loop regime, or does
granularity now matter?).

**A4 setup:** A4 uses unlimited multi-edit (`--iterative-topk-max-edits 0`) with
the ε-cap `C=2` as the limiter, matching A1's re-edit policy so the **only**
difference vs A1 is granularity `B`.

**A4 cost-gate:** B=1 re-attributes per edit (thousands of full-corpus
attributions at k=10000). The plan must **probe per-attribution wall-time first**
and, if a full k=10000 B=1 run is impractical, run A4 at the largest affordable k
(noted as a granularity check, not a ceiling run) and supplement with the §8.2
(k=20, bit-identical) citation.

A one-shot `α_fi=0`-vs-`0.1` sanity check at small k confirms the α_fi=0 proxy
before relying on the cheap runs.

## 6. Testing strategy (TDD)

- **Equivalence locks (§3.6):** batch R=1 == current batch-topk; iterative-topk
  max-edits=1 == current `_iterative_topk_modify` (§8.2 property).
- **Non-regression gate:** synthetic case where an iterate lifts F_causal but
  dips F_spatial → rejected under `non-regression`, kept under `objective`.
- **Cumulative-ε cap:** across ≥2 rounds, total displacement ≤ 2 for C=2; > 2
  permitted for C=∞; per-round within-edit move still ≤ ε=2.
- **Outer-loop stop conditions:** convergence-τ fires after P_outer flat rounds;
  max-rounds caps; pool-exhaustion terminates; MAX_ROUNDS safety holds.
- **Multi-edit:** a trajectory can be re-selected across rounds up to the cap;
  cap=1 forbids re-edit; cap=0 unlimited.

Use existing synthetic bundle helpers in `famail_temporal/tests/`.

## 7. MAX_ITERATIONS — measure, don't guess

The non-regression gate may delay per-trajectory convergence (fewer qualifying
improvements). Capture `mean_best_iter` and converged-fraction in A1/A3 and bump
`MAX_ITERATIONS` (currently 50) **only if** trajectories actually hit the cap.
Document the measurement either way.

## 8. Documentation & memory updates

- Methods doc **§8.7** — new calibration entry: multi-loop results, the gate
  effect, the ε-stacking effect, the A4 granularity finding, the round curve.
- `famail_temporal/baselines/STATUS.md` — record the shipped editing config (it
  feeds Phase 4's FAMAIL edit source).
- **ε-convention revision:** if multi-loop ships, the "ε=2 inviolable *across
  loops*" claim in the methods doc and auto-memory must be revised to "ε=2
  within-edit; cumulative cap C across rounds (default 2)." Flag exactly what
  changed.
- New best-config result committed to `results/` and reflected in §8.7. If we do
  **not** beat +0.0128, document the negative finding clearly.

## 9. Open risks / findings to surface (do not silently patch)

- If multi-loop makes fairness *worse* or fails to beat +0.0128 → surface and
  interpret; do not patch the algorithm to "fix" it.
- If non-regression collapses the edit count → report it; it's a real result.
- If C=∞ wakes F_fidelity / drifts far out of distribution → that's the realism
  finding; report displacement histograms.
- If A4 shows B=1 ≠ B=K → the §8.2 equivalence does not generalize; new finding.

## 10. Success criteria

- A clear empirical answer to Q1–Q3 with the round curve and the ablations above.
- A recommended stop rule (fixed-R or convergence-τ) with its evidence.
- The non-regression gate implemented, tested, benchmarked, documented.
- A new best-config result that beats +0.0128 **or** an honest documented
  trade-off / negative result.
- A short question-4 low-hanging-fruit list.
- All work on `algorithm-improvements`, algorithm-change protocol respected,
  equivalence locks green.
