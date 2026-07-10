# Mission 3 — Data-Augmentation Baselines (vanilla ST-iFGSM / FGSM / random) — Design Spec

**Date:** 2026-07-09 · **Branch:** `mission3-baselines` (worktree off `main` `0c5e652`)
**Status:** approved (brainstorming) → ready for writing-plans
**Mission:** Meeting-41 P0 #3 — baseline editors that contextualize FAMAIL's edit quality.

---

## 1. Context & purpose

Meeting 41 (PI decisions, canonical): **ST-iFGSM is a FIDELITY / editing-quality baseline, NOT a fairness
comparison** — it is not fairness-aware, so a fairness head-to-head is not apples-to-apples; we still report
its fairness numbers to show fairness does **not** improve. Implement **vanilla ST-iFGSM pre-discretization
on the continuous seeking trajectories** (the paper's native mode: whole-trajectory edits on continuous
data), then discretize and measure. A "ST-iFGSM-for-fairness" arm was **rejected as circular**. Also named:
**plain non-iterative FGSM**; "searching for others" left open.

The claim the baseline table supports: *at equal edit budget (same trajectories, same ε), a
non-fairness-aware editor of the same signed-gradient family neither improves fairness nor preserves
driver identity as well as FAMAIL* — i.e., FAMAIL's gains come from its objective, not from bounded
editing per se.

## 2. Decisions locked during brainstorming

1. **Edit set = FAMAIL's** (user-chosen): the baselines edit the exact trajectory set of the filtered
   supply-lift headline (9,885 = 2,340 trim + 7,545 lift), same ε, same count. Selection held constant so
   the table isolates edit *direction*.
2. **Arms (3) + lit-scan** (user-chosen): `ifgsm` (vanilla iterative attack), `fgsm` (single-step),
   `random` (direction placebo), plus a non-blocking literature scan for additional candidates (gated).
3. **Scope = Shenzhen, full metric panel** (user-chosen): Fidelity-A + Fidelity-B + F_causal/F_spatial +
   external metrics. SF replication deferred (cheap later re-run under `FAMAIL_CITY=sf12`).
4. **Whole-trajectory continuous edits** (PI, Meeting 41): perturb every seeking state's float coords, not
   just the pickup/tail.
5. **Standalone module** (approach A): zero changes to `modifier.py` / `objective.py` /
   `evaluation/runner.py` — the frozen algorithm is untouched (algorithm-change protocol; the α-sweep is
   running against that code concurrently).

## 3. Components

### 3.1 Attack engine — `famail_temporal/baselines/stifgsm_baseline.py` (new)

Pure engine, no I/O. Core API (names final):

```python
def attack_trajectories(
    trajectories,          # List[Trajectory] — ORIGINALS from the bundle (never mutated)
    disc,                  # frozen 3-stream discriminator (eval mode, params frozen)
    profiles,              # driver profile features (per trajectory's driver)
    mode,                  # "ifgsm" | "fgsm" | "random"
    epsilon=2.0,           # per-coordinate cumulative L-inf bound, grid units
    step=0.1,              # signed-gradient step (parity: config.STEP_SIZE_ALPHA)
    max_iterations=50,     # parity: config.MAX_ITERATIONS
    patience=10,           # parity: config.PATIENCE (best-attack-loss early stop)
    seed=0,                # random arm only
    device="auto",
    batch_size=256,
) -> AttackResult
```

- **Perturbed variables:** the `x_grid, y_grid` floats of **every seeking state**; `time_bucket` /
  `day_index` frozen. Originals are deep-copied; inputs never mutated.
- **Loss:** `p = disc(original_seq, perturbed_seq, profile)` — the same original-vs-edited same-driver
  pairing FAMAIL's `F_fidelity` uses. `ifgsm` descends `p` (the KDD'23 attack direction: make the
  discriminator say "different driver") via `delta = clip(delta - step * sign(grad), -eps, +eps)` per
  coordinate, keeping the best (lowest-`p`) iterate, patience-stopped.
- **`fgsm`:** identical loop, `max_iterations=1`, step = ε (one full-budget signed step).
- **`random`:** seeded `delta ~ Uniform{-1,+1}^(S,2) * eps` (full-budget random direction), no gradient,
  no discriminator calls during the "attack" (scored after).
- **Batched** over trajectories (padded/masked `(B, S, 4)`); there is no shared state between
  trajectories, unlike FAMAIL's demand-grid-coupled sequential editor. A `batch_size=1` path must produce
  identical results (test).
- `AttackResult`: per-trajectory perturbed float states + final `p`, iteration counts, and the applied δ.

### 3.2 Discretize + package — same module

- Round perturbed coords to grid ints; clamp to grid bounds (48×90).
- **No adjacency repair** — vanilla ST-iFGSM has none. Instead **report the king-move adjacency violation
  rate per arm** (consecutive-state `max(|dx|,|dy|) > 1`) as realism evidence, contrasted with FAMAIL's
  100% compliance (filtered headline).
- Emit the edited corpus in the standard histories/edit-dir format the measurement harnesses consume
  (same contract as the editor's results dirs: original + modified trajectories with IDs), one results
  dir per arm: `results/<ts>_baseline_<mode>_shz_primary/`.

### 3.3 Runner CLI — `famail_temporal/baselines/run_stifgsm_baseline.py` (new)

`python -m famail_temporal.baselines.run_stifgsm_baseline --edit-dir <filtered_headline_dir>
--mode {ifgsm,fgsm,random} [--epsilon 2.0 --seed 0 --device auto ...]`

1. Load bundle + frozen discriminator (existing `fidelity/checkpoint.py` path; `FAMAIL_CITY` selects city).
2. Read the **trajectory IDs** from the headline dir's `histories.pkl`; fetch the ORIGINAL trajectories
   from the bundle (baselines edit originals, not FAMAIL's edited versions).
3. Run the arm → discretize → write the arm's results dir + `metrics.json` (config snapshot, mode, ε,
   seed, iteration stats, adjacency-violation rate, F_causal/F_spatial before→after rescoring).
4. Print the one-line summary (house style).

### 3.4 Measurement chain (existing harnesses, no new metrics code)

Per arm, mirroring the headline's protocol exactly:
- **Fidelity-A:** `fidelity_eval.humid_identity_fidelity` on (original, edited) pairs + the
  matched/mismatched `identity_validation_gate`.
- **Fidelity-B:** the discriminator-free JS-divergence stats eval.
- **Fairness rescoring:** F_causal / F_spatial before→after (existing rescore path).
- **External metrics:** `run_external_fairness --edit-dir <arm_dir>` **with tier-2 supply recount**
  (`analysis/supply_recount.py`; the attack moves seeking states → supply moves; the headline's external
  metrics used recounted supply, so the baselines must too or the comparison is apples-to-oranges).
- Comparison rows in the assembled table: **raw**, **FAMAIL (filtered supply-lift headline)**, `ifgsm`,
  `fgsm`, `random`.

### 3.5 Lit-scan (parallel, non-blocking)

A short memo (`famail_temporal/baselines/DATA_AUG_BASELINE_CANDIDATES.md`): 3–5 candidate trajectory
data-augmentation baselines from the literature, each with citation (verified — same standard as the
Mission-2 audit), one-paragraph description, and an adopt/defer recommendation. **Gate: user decides**
whether any becomes a 4th arm; nothing is built from the scan without that decision.

## 4. Expected outcomes (hypotheses, stated for honesty)

- `ifgsm`: Fidelity-A collapses toward the mismatched band (it is the attack objective); fairness ≈ flat.
- `fgsm`: milder fidelity damage, fairness ≈ flat.
- `random`: fidelity mildly degraded, fairness ≈ flat (placebo: bounded perturbation alone does nothing).
- If any arm *improves* fairness materially, that is a FINDING to surface (per feedback protocol), not to
  frame away — it would weaken the "objective did it" claim and the PI must see it.

## 5. Tests (house style, `famail_temporal/baselines/tests/test_stifgsm_baseline.py`)

- ε-clip invariant: `max|perturbed − original| ≤ ε` per coordinate, all arms, incl. after many iterations.
- `fgsm` ≡ `ifgsm(max_iterations=1, step=ε)` bitwise.
- `random` determinism: same seed → identical output; different seed → different.
- Batched ≡ sequential (`batch_size=1`) equivalence on a small synthetic set.
- Originals never mutated (object identity + value checks).
- Discretization: all coords integer, in-grid; adjacency-violation counter correct on a crafted case.
- Time/day coords bitwise unchanged.
- CLI smoke on a tiny synthetic bundle (existing `_helpers.make_traj_at` fixtures).

## 6. Non-goals (v1)

- No SF run (deferred; the code must be city-agnostic via `FAMAIL_CITY`, but v1 measures Shenzhen only).
- No downstream weighted-BC training of baseline corpora (Meeting 41 framed these as data-level L1
  baselines; ~10h/arm GPU if the PI later asks).
- No changes to editor/algorithm code, no adjacency repair for baselines, no fairness-aware ST-iFGSM
  (rejected as circular).
- No PAPER/ curation in v1 (follows once numbers exist, mirroring the external-metrics bundle pattern).

## 7. Constraints & environment

- **GPU:** build + tests are GPU-free (CPU-capable tiny fixtures). The real runs (~minutes–1h) wait for
  the α-sweep to finish (driver: `famail_temporal/results/alpha_sweep/driver.sh`) or slot between points.
- **Worktree:** `.claude/worktrees/mission3-baselines`, branch `mission3-baselines` off `main` `0c5e652`.
  Commit per task; **no merge to main without user approval**.
- Seeds fixed (default 0); every results dir carries a config snapshot + provenance (house convention).
- Fairness convention: 1 = fairest (F_spatial/F_causal); Fidelity-A high = identity preserved.

## 8. Success criteria

- Three arm runs produce standard results dirs scoreable by ALL existing harnesses without modification.
- The assembled 5-row comparison table (raw / FAMAIL / ifgsm / fgsm / random) exists with the full panel:
  Fidelity-A (+gate), Fidelity-B, ΔF_causal, ΔF_spatial, external metrics, adjacency-violation rate.
- Tests pass; the frozen algorithm files are untouched (`git diff main -- famail_temporal/algorithm/
  famail_temporal/evaluation/runner.py` is empty).
- Lit-scan memo delivered with verified citations and a user-gated recommendation.
