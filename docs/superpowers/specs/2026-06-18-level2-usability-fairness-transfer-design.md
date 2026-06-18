# Level-2 Usability: Fairness Transfer — Design Spec

**Date:** 2026-06-18
**Status:** Draft for review
**Branch:** `level-2-usability`
**Framing:** [`docs/two_level_argument.md`](../../two_level_argument.md). Level 1 (data quality) is done ([`LEVEL1_V2_RESULTS.md`](../../../famail_temporal/baselines/LEVEL1_V2_RESULTS.md)); this is the Level-2 (usability) companion.

---

## 1. Goal

Test whether the edited data's **fairness advantage survives downstream behavior-cloning training** — i.e., does fairness *transfer* from a dataset into a model trained on it? Train a driver-conditioned BC policy on each of four data sources (raw, FAM-AIL edited, BC-generated, GAN-generated), then evaluate each *trained policy's* generated demand on the Level-1 axes. Framed as a **hypothesis test**: a null (no transfer) is a legitimate, publishable result, consistent with the model-level null finding.

**Success criteria:** (1) a four-source × five-seed **paired** table of {F_causal, F_spatial, Fidelity-A (identity, gated), Fidelity-B (enriched)} with mean ± std; (2) a paired-difference report — headline **edited − raw**, secondary **edited − {BC-gen, GAN-gen}** — with a paired significance test; (3) Fidelity-A/B reported as a **guardrail** (the edited-trained policy must not be less faithful than the raw-trained one); (4) an honest verdict on whether fairness transfers, either way.

---

## 2. Scope

### In scope
- Four **matched** downstream training datasets over the **full corpus** (raw, edited, BC-gen, GAN-gen), differing only in trajectory content.
- A driver-conditioned BC (MLE) downstream policy trained per source, per seed, with a **paired** seed design (same seed → same init + minibatch order across all four arms).
- Evaluation of each trained policy's generated demand on the Level-1 axes (F_causal, F_spatial, identity Fidelity-A with the real-anchored gate, enriched Fidelity-B), **reusing the Level-1 v2 scoring**.
- Paired multi-seed statistics (5 seeds, pre-registered scale-to-10 trigger), with a paired significance test.
- A Level-2 orchestrator + results doc.

### Out of scope
- **Non-BC downstream models.** The downstream policy is behavior cloning (MLE) only. GAN appears solely as a *data source*, never as a downstream model. No adversarial downstream training.
- **A separate downstream task** (e.g., demand prediction, driver identification). Evaluation is self-scoring on the Level-1 axes.
- **Multiple edit configurations.** A single committed edit run is used (the no-dedup causal-emphasis edit).
- **Changes to the editing algorithm, HuMID, the generators, or the Level-1 scoring functions** — all reused as-is. HuMID stays frozen/read-only (`torch.no_grad`, `train(False)`). No edits to `algorithm/`, `fairness/`, `fidelity/`.
- **GAN-architecture stabilization** — the GAN-gen arm intentionally uses the Level-1 (collapsed) GAN as a data source; that collapse is part of what the experiment measures.

---

## 3. Locked design decisions (user-confirmed 2026-06-18)

1. **Downstream policy = driver-conditioned BC** (the Level-1 v2 `TrajectoryLSTM` with `n_drivers`), trained by `train_mle` (MLE = behavior cloning). Driver-conditioning makes all Level-1 axes — including identity Fidelity-A — evaluable on the trained policy's output, so Level 2 mirrors Level 1 exactly.
2. **Matched-per-seed dataset construction.** All four training sets are index-aligned to the same real (driver, start-context) seeds, so they share size and driver/start-context distribution; only trajectory **content** varies.
3. **Full corpus.** Training and evaluation use the entire corpus (105,401 trajectories) — no `max_tokens` filtering (the Level-1 cap of 256 excluded 763 ≈ 0.72% outliers; L2 includes them). Memory for the ~0.7% very long trajectories is bounded by **token-budgeted batching** (§7).
4. **5 paired seeds (+ scale-to-10 trigger).** The same five seeds control weight-init and minibatch order across all four arms. Report mean ± std and a paired test on the per-seed differences. If the headline (edited − raw) paired CI crosses zero, scale to 10 seeds (pre-registered).
5. **Outcome = fairness transfer, fidelity as guardrail, framed as a hypothesis test.** Primary metric = F_causal of each trained policy's generated demand. Headline comparison = paired edited − raw; secondary = edited − {BC-gen, GAN-gen}. Fidelity-A/B are guardrails. A null is publishable.
6. **Real-anchored identity gate** (real-d vs real-d′) computed once — it is a property of HuMID + the input construction, independent of which policy produced a trajectory — and reused to flag Fidelity-A as trusted (carried over from Level-1 v2).

---

## 4. The four matched training datasets

For the full set of real seeds (every corpus trajectory `t`, carrying `t.driver_id` and start-context `trajectory_context(t)`), build four datasets, each index-aligned to the seeds so all share size N = 105,401 and the same driver/start-context distribution:

- **`D_raw`** = the real trajectories. Source: the edit run's `augmented_trajs_before.pkl` (identical to `bundle.trajectories`; using the edit dir's before/after pair guarantees raw and edited are aligned).
- **`D_edited`** = the edited corpus. Source: the edit run's `augmented_trajs_after.pkl` (the full corpus with the modified trajectories replaced — same basis as Level-1's `metrics_after`).
- **`D_bcgen`** = for each seed, **one** trajectory generated by the **Level-1 BC** generator conditioned on that seed's driver + start-context (`generate_trajectories(..., driver_idxs=[idx_d])`), labeled with that driver. Rollouts are capped at `MAX_GEN_LEN` (as in Level 1); an empty rollout falls back to a single-cell trajectory at the start cell so N is preserved.
- **`D_gangen`** = same as `D_bcgen`, from the **Level-1 GAN** generator.

The Level-1 BC and GAN generators are (re)trained driver-conditioned on the full raw corpus once (via the existing `_train_and_generate_cond` path) to produce `D_bcgen` / `D_gangen`. **Note (data distillation):** `D_bcgen` is data generated by a BC trained on raw, so a downstream BC trained on `D_bcgen` is a self-distillation baseline (expected ≈ raw if BC is faithful; large degradation reveals generation loss). `D_gangen` tests whether the GAN's distributional collapse propagates into a trained policy.

---

## 5. Downstream training (paired seeds)

For each source `X ∈ {raw, edited, bcgen, gangen}` and each seed `s ∈ {0,1,2,3,4}`:

1. `set_all_seeds(s)` — then construct a fresh `TrajectoryLSTM(n_drivers=len(driver_to_idx))` (same init across all four `X` for a given `s`) and run `train_mle(model, sequences_X, contexts_X, ..., driver_idxs=driver_idxs_X)` over the full corpus. Because the seed is fixed before both init and the `randperm` minibatch ordering, the four arms for seed `s` differ **only** in their training data — this is the pairing.
2. Cache the trained policy `π_{X,s}` (or evaluate immediately and discard, to bound memory).

Training is MLE (behavior cloning); no adversarial phase. The same `mle_epochs` as Level-1 (20) is the default.

---

## 6. Evaluation (reuses Level-1 v2 scoring)

For each trained policy `π_{X,s}`:

- **Fairness:** driver-conditioned `generate_pickups(π_{X,s}, contexts, driver_idxs=...)` over the full corpus → `pickups_to_pickup_3d` → `data_level_fairness(bundle, pickup_3d=grid)` → `F_causal`, `F_spatial`.
- **Fidelity-A (identity):** build matched pairs (real-d branch vs `π_{X,s}`-generated-for-d branch) exactly as in `run_level1_table_v2`, score `humid_identity_fidelity`. The **real-anchored gate** (real-d vs real-d′) is computed once per run and reused as the trusted flag.
- **Fidelity-B (enriched):** the five-key `distributional_fidelity` (length, mean-displacement, coverage, radius-of-gyration, net-displacement) on a shared grid vs raw + the terminal-cell-distribution JS, aggregated as in Level-1 v2.

Evaluation generation uses the same `MAX_GEN_LEN` cap as Level 1 for all arms (consistent across sources).

---

## 7. Statistics

- For each metric `m` and seed `s`, record `m(X, s)` for all four `X`. Compute per-seed **paired differences**: `Δ_raw(m,s) = m(edited,s) − m(raw,s)`, `Δ_bcgen(m,s) = m(edited,s) − m(bcgen,s)`, `Δ_gangen(m,s) = m(edited,s) − m(gangen,s)`.
- Report, per source, mean ± std across seeds for every metric.
- **Headline:** the paired distribution of `Δ_raw(F_causal, ·)` — mean, std, and a paired test (Wilcoxon signed-rank; paired-t as a secondary). State the per-seed differences explicitly (n = 5 is small).
- **Secondary:** `Δ_bcgen(F_causal, ·)`, `Δ_gangen(F_causal, ·)`.
- **Fidelity guardrail:** report Fidelity-A and Fidelity-B per arm; flag if the edited-trained policy is meaningfully **less** faithful than the raw-trained policy (identity-A lower, or distributional-B higher) — fairness transfer is only meaningful if fidelity is not sacrificed.
- **Scale-to-10 trigger (pre-registered):** if the headline `Δ_raw(F_causal, ·)` paired 95% CI crosses zero, rerun with 10 seeds before drawing a conclusion. Always disclose the seed count and that the gap (~0.013) sits near the seed-noise floor (~0.012 bits).

---

## 8. Data flow

```
edit dir ─► D_raw (augmented_trajs_before) , D_edited (augmented_trajs_after)        [full corpus, aligned]
raw corpus ─► train L1 BC + L1 GAN (driver-conditioned) ─► generate 1 traj / real seed ─► D_bcgen , D_gangen

for s in seeds (paired):
  for X in {raw, edited, bcgen, gangen}:
     set_all_seeds(s) ─► TrajectoryLSTM(n_drivers) ─► train_mle(D_X, driver_idxs_X)  ─► π_{X,s}
     π_{X,s} ─► generate_pickups (driver-cond, full corpus) ─► data_level_fairness ─► F_causal, F_spatial
     π_{X,s} ─► identity Fidelity-A (matched vs real-d; gate real-d vs real-d′ once) ─► A
     π_{X,s} ─► enriched Fidelity-B (5 stats + terminal-cell JS vs raw)             ─► B

paired differences (edited−raw headline; edited−{bcgen,gangen} secondary) ─► mean±std + Wilcoxon
assemble ─► level2_metrics.json + level2_table.md + LEVEL2_RESULTS.md (+ training curves)
```

---

## 9. Components (files)

- **Create:** `famail_temporal/baselines/run_level2_table.py` — the Level-2 orchestrator + CLI: builds the four matched full-corpus training sets, runs the paired-seed train/evaluate loop, computes paired statistics, persists results.
- **Create:** a matched-dataset builder (`_build_training_sets`) and a paired-statistics helper (`_paired_diff_stats`) — pure, unit-tested.
- **Create:** `famail_temporal/baselines/LEVEL2_RESULTS.md` (results) and update [`docs/two_level_argument.md`](../../two_level_argument.md) Level-2 status.
- **Reuse (read-only / unchanged):** `train_mle` + rollout (`driver_idxs`), `run_level1_table_v2` scoring helpers (identity Fidelity-A, gate, enriched Fidelity-B — import, do not duplicate), `drivers.py`, `data_level_fairness`, `_train_and_generate_cond` (to (re)build the L1 BC/GAN generators for the gen training sets), the edit dir's `augmented_trajs_before/after.pkl`, the variance-suite multi-seed pattern.
- **Token-budgeted batching:** add an optional `max_batch_tokens` path to `train_mle` (or a thin batching helper) so a batch's total padded tokens are bounded — long outliers form smaller batches. Backward compatible: default `None` = current fixed-`batch_size` behavior (Level-1 unaffected, regression-tested).

---

## 10. Error handling & edge cases

- **Memory (full corpus).** The ~0.7% of trajectories over 256 tokens (max 1654) are included; `max_batch_tokens` bounds peak logit memory `(B·L·VOCAB)`. The GPU smoke must confirm no OOM with the longest trajectories before the full run.
- **Empty generated rollout** when building `D_bcgen`/`D_gangen`: fall back to a single-cell trajectory at the start cell (preserves N and the driver label); count in `n_empty`.
- **Raw/edited alignment:** assert `len(augmented_trajs_before) == len(augmented_trajs_after)` and that driver_ids align index-wise; fail loud otherwise.
- **Paired integrity:** assert the same seed list is used for all four arms; the per-seed model init must be reconstructed after `set_all_seeds(s)` (not reused across arms).
- **Gate may fail:** non-fatal; Fidelity-A flagged untrusted (same discipline as Level 1). Expected to pass (real-anchored, validated in Level-1 v2).
- **Unknown driver / missing profile:** reuse Level-1 v2 handling (raise on unknown driver index at generation; zero profile fallback with a warning).

---

## 11. Testing

Unit (no GPU):
- matched-dataset builder: four sets equal length; index alignment to seeds; driver labels correct; empty-rollout fallback preserves N.
- paired-stats helper: per-seed differences, mean/std, Wilcoxon on a hand-built example; correct headline/secondary selection.
- token-budgeted batching: a batch with a long trajectory is split so total tokens ≤ budget; `max_batch_tokens=None` reproduces current batching exactly (regression).
- orchestrator pure helpers: result-dict/JSON round-trip; table render with the paired columns.

Real-data smoke (GPU, manual): 2 seeds, few MLE epochs, full-corpus training (confirm no OOM on the longest trajectories), populated paired table.

---

## 12. Open decisions — resolved / deferred

- Downstream policy = **driver-conditioned BC** (resolved, §3.1).
- Training-set construction = **matched per real seed, full corpus** (resolved, §3.2–3.3).
- Seeds = **5 paired + scale-to-10 trigger** (resolved, §3.4).
- Outcome = **fairness transfer, fidelity guardrail, hypothesis test** (resolved, §3.5).
- **Deferred:** non-BC downstream models; a separate downstream task; multiple edit configs; GAN stabilization.
- **Plan-time detail to pin:** the `max_batch_tokens` budget value (from the observed length distribution + a GPU memory check) and whether to cache all 20 policies or evaluate-and-discard per arm.
