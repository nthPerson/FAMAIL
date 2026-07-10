# FAMAIL Baselines — Status

Living status of the GAN-baseline work that motivates and evaluates FAMAIL trajectory editing.
Design spec: [`docs/superpowers/specs/2026-05-27-famail-gan-baselines-design.md`](../../docs/superpowers/specs/2026-05-27-famail-gan-baselines-design.md).

**Last updated:** 2026-07-10

---

## Argument (what the baselines prove)

| Baseline | Claim | Status |
|---|---|---|
| **B0** — generator on raw data | Bias propagates to the trained model | Implemented (Phase 2 MLE keystone + Phase 3 adversarial) |
| **B1** — + differentiable fairness loss | A model-level fairness fix is insufficient | Deferred (Phase 4) |
| **B2** — generate-then-filter | Filtering buys fairness only by discarding scarce data | Data-level done; model-level deferred (Phase 4) |
| **FAMAIL** — edit pickups (ε=2) | Editing wins fairness *and* retention | Editing validated; model-level deferred (Phase 4) |

Headline = **model-level** (edited-data model fairer than raw-data model); fallback = **data-level Pareto**.

---

## Phase 1 — data-level Pareto — DONE

Module `famail_temporal/baselines/` (16 tests passing: `python -m pytest famail_temporal/baselines/ -q`).

| File | What it does |
|---|---|
| `datasets.py` | `pickup_unit_of`, `pickup_mass`, `rank_unfair_trajectory_indices`, `build_filtered_pickup_3d` (filtered demand grid by subtracting each removed trajectory's pickup mass; modified cells floored at `DEMAND_FLOOR`) |
| `metrics.py` | `data_level_fairness(bundle, pickup_3d)` → `{f_spatial, f_causal, gini_dsr, gini_asr}` via the canonical `build_fairness_grid` + scalar reduction (1 = fairest) |
| `pareto.py` | `ParetoPoint`, `raw_point`, `filtered_points(k_levels)`, `edited_point`, `points_to_json` |
| `figure.py` | `plot_pareto(points, path, metric)` — retention × fairness scatter/curve |
| `run_data_pareto.py` | CLI: load corpus → raw + filtered sweep → optional edited point (via `run_experiment`) → JSON + PNG |

### Results (real corpus, `python -m famail_temporal.baselines.run_data_pareto`)
- **Reuse seam validated:** `raw.f_causal = 0.8052` — matches the known corpus baseline (0.805).
- **Filtering finding:** removing the top-K most-unfair trajectories does **NOT** raise data-level F_causal — it slightly *lowers* it (0.8052 → 0.8016 at 3,773 removed). Mechanistically, filtering perturbs a still-active unit's demand rather than removing it from the regression. → At the data level, **editing strictly dominates filtering** (better fairness AND full retention). The conventional filtering cost (less training data → weaker model) is a *model-level* effect, measured in Phase 2.
  - Decision (2026-05-28): report as-is; defer B2's cost to model-level. No filtering-criterion change (gated). See spec §8.

### Editing (FAMAIL) — validated strongest config
- Config: `--max-per-unit 1`, **α = (0.2, 0.7, 0.1)** (causal-emphasis) — a balanced multi-objective (spatial + fidelity terms active) that matches pure-causal gain without gaming a single metric. `run_data_pareto._run_edit` uses it.
- Run `2026-05-27T22-29-57_1000k_causal_emphasis_dedup` (`k=1000`): **ΔF_causal = +0.0087** (0.8052 → 0.8139); F_spatial flat (−0.0003); 999/1000 converged.
- Run `2026-05-28T00-22-24_10-000k_causal_emphasis_dedup` (`k=10000`): **ΔF_causal = +0.0093** (0.8052 → 0.8145); F_spatial flat (−0.0003); 1184/1186 converged.

**Unit-distinct editing budget (finding, 2026-05-28):** with `--max-per-unit 1`, selection **caps at 1,186 trajectories regardless of K** (k=10000 still selected only 1,186). Reason: 2,829 of 34,524 active units have negative (drag) causal attribution, but only 1,186 of them contain a pickup to relocate — editing moves *existing* pickups, so drag-units with no demand (active by supply only) are unreachable, and `max-per-unit 1` takes ≤1 trajectory per unit. So ~1.1% of the corpus is the natural editable slice; K stops binding above ~1,186. Pushing ΔF higher needs a different lever (relax `max-per-unit` → pile-up risk, or re-attribution rounds — **the latter was tested 2026-06-06 and refuted, see below**), not bigger K.

**Multi-loop re-attribution + non-regression gate — tested & REFUTED (2026-06-06).** A unified re-attribution engine (`algorithm/editing_loop.py`) plus a both-metrics (non-regression) acceptance gate were built (opt-in CLI: `--max-rounds`, `--round-convergence-tol`, `--epsilon-cap`, `--accept-rule`, `--iterative-topk-max-edits`; **defaults preserve the single pass exactly**) and run on the full corpus. **Multi-loop degrades F_causal, gate-independently** — round 1 (= the single pass) is best (+0.0127 objective gate / +0.0124 non-regression, both ≈ the +0.0128 baseline) and rounds 2+ are net-negative. Root cause: the **soft-relaxation-vs-discrete-grid gap** (edits accepted on the differentiable soft objective collectively degrade the hard pickup grid); the non-regression gate likewise protects *soft* F_spatial while *hard* F_spatial slips. **The shipped FAMAIL editing config is unchanged: single-pass, α=(0.2,0.7,0.1), no-dedup k=10000 → ΔF_causal=+0.0128 (`results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`), and remains the FAMAIL edit source for Phase 4.** Full write-up: methods doc §8.7–§8.8. Side note: A3 round 1 ≈ +0.0128 validated both the engine refactor (reproduces the single pass on real data) and the α_fidelity=0 speed proxy. **Follow-up (STE, 2026-06-06):** a straight-through estimator (`--ste`; forward=hard grid, grad=soft) was added to test whether the soft-vs-hard gap *bounds* ΔF_causal — it does NOT. STE fixes the multi-loop degradation (rounds go flat, not net-negative) but does **not** accumulate, so the ceiling is confirmed **intrinsic** (the ~1–3% editable slice + local gradient geometry); and STE single-pass even slightly *underperforms* soft (+0.01044 vs +0.01271 at α_fi=0) because the straight-through gradient is a worse search direction than the soft gradient. So **four** levers (multi-loop, non-reg gate, larger ε, STE) all fail to beat soft single-pass **+0.0128**, which stays shipped; STE is retained opt-in as a diagnostic only. See §8.8.

---

## Phase 2 — B0 generative baseline (MLE keystone) — DONE

Plan: [`docs/superpowers/plans/2026-05-27-famail-gan-baselines-phase2-b0-generative.md`](../../docs/superpowers/plans/2026-05-27-famail-gan-baselines-phase2-b0-generative.md).
Module `famail_temporal/baselines/gan/` (14 tests passing: `python -m pytest famail_temporal/baselines/gan/ -q`). Built via subagent + two-stage review (spec compliance PASS; code quality Approved-with-minors, the two Important items fixed).

| File | What it does |
|---|---|
| `config.py` | Cell vocabulary (`N_CELLS`=4320, BOS/EOS/PAD, `VOCAB_SIZE`), `N_TBLOCKS`, generator/training/generation hyperparameters |
| `sequences.py` | `flat_cell`/`unflat_cell`, `trajectory_to_tokens` (BOS…EOS), `trajectory_context` → (start cell, start t-block) |
| `generator.py` | `TrajectoryLSTM`: conditional LSTM LM; `forward` (teacher-forced) + `step` (O(L) carried-state decode) |
| `train_mle.py` | Next-token cross-entropy training (teacher forcing, PAD-ignored), padded batching |
| `rollout.py` | Autoregressive sampling → terminal-cell pickups → demand grid via `pickup_mass` (out-of-grid pickups skipped; no-op in production) |
| `b0.py` | `run_b0` orchestrator: train→generate→grid→fairness, returns `{generated, corpus, n_generated}` |
| `run_b0.py` | CLI → writes `results/b0/b0_fairness.json` |

**Design notes:** full-sequence generation (terminal cell = pickup); conditioning = (start cell, start t-block); pickup t-block inherits the conditioning block (Phase-2 simplification); one rollout per real context (corpus-matched). Adversarial training, the discriminator, and B1's fairness loss are **Phase 3**.

**Not yet run:** the real-data B0 smoke (`python -m famail_temporal.baselines.gan.run_b0`) — needs the cache + ideally GPU; expected `corpus.f_causal ≈ 0.805` with `generated.f_causal` near it (bias reproduced).

---

## Phase 3 — adversarial training subsystem + standard-adversarial B0 — DONE

Plan: [`docs/superpowers/plans/2026-05-28-famail-gan-baselines-phase3-adversarial.md`](../../docs/superpowers/plans/2026-05-28-famail-gan-baselines-phase3-adversarial.md).
Adds the adversarial stage Phase 2 deferred, completing the spec's standard-adversarial B0. Module `famail_temporal/baselines/gan/` now at 43 tests passing across the whole `baselines/` suite (`python -m pytest famail_temporal/baselines/ -q`). Built via subagent + two-stage review per task + a final holistic review.

| File | What it does |
|---|---|
| `generator.py` | + `step_embed` (decode from a precomputed input embedding); `step` delegates to it |
| `gumbel.py` | `gumbel_rollout` — differentiable straight-through Gumbel-softmax rollout (fixed `max_len`; feeds `y @ cell_embed.weight` back; records first EOS in `lengths`) |
| `critic.py` | `SequenceCritic` — real-vs-fake LSTM over the cell vocabulary; `forward_ids` (hard) + `forward_soft` (soft, differentiable via `soft_onehot @ embed.weight`) |
| `train_adversarial.py` | `adversarial_finetune` — non-saturating GAN loop (D-step on real vs detached fake, G-step on a re-rolled differentiable fake), annealed Gumbel temp, separate G/critic optimizers |
| `model_level.py` | `fit_and_evaluate` — MLE pretrain → adversarial fine-tune → generate → grid → fairness; returns `{generated, corpus, n_generated, mle_losses, adv_losses}` |
| `run_b0_adversarial.py` | CLI → writes `results/b0_adversarial/b0_adversarial_fairness.json` |
| `config.py` | + adversarial hyperparameters (`ADV_EPOCHS`, `ADV_LR_G/D`, `ADV_BATCH_SIZE`, `GUMBEL_TAU_START/END`, `D_HIDDEN_DIM`) |

**Design notes:** the critic is a fresh vocab-embedding LSTM (mirrors the Siamese *design*, per spec decision #8 — the trained Siamese net stays reserved for eval-time realism); critic is unconditioned; fixed-length rollout (EOS recorded in `lengths`, no early break) keeps a static differentiable batch; straight-through hard Gumbel. B1's differentiable fairness loss and the FAMAIL/B2 model-level dataset swaps are **Phase 4** (the B1 reuse seam — `FAMAILObjective` + a terminal-soft-pickup scatter — is documented in the Phase-3 plan).

**Not yet run:** the real-data adversarial-B0 smoke (`python -m famail_temporal.baselines.gan.run_b0_adversarial --mle-epochs 5 --adv-epochs 3 --device auto`) — needs the cache + GPU; expected `corpus.f_causal ≈ 0.805` with `generated.f_causal` near it. Watch the loss histories for D-collapse / amplification (a finding to record, not a bug to patch).

---

## Phase 4 — model-level (MLE-only B0/FAMAIL) — METRIC HARDENING DONE (2026-06-08)

Plan: [`docs/superpowers/plans/2026-06-06-metric-hardening.md`](../../docs/superpowers/plans/2026-06-06-metric-hardening.md).

Adds the model-level transmission + dynamic-range metrics (`baselines/transmission.py`,
`district_metrics.py`, `localized_metrics.py`, `run_metric_hardening.py`). Both
generators (B0 + FAMAIL) train MLE-only via `fit_and_evaluate(..., adv_epochs=0,
train_trajectories=...)`. The collapsing adversarial GAN remains an opt-in
"amplification" ablation per the `B0_DECISION_BRIEF.md` pivot.

### Results — first real-data run (2026-06-08, seed=0, RTX 3070, ~5 min)

| Metric | B0 | FAMAIL | Delta |
|---|---:|---:|---:|
| Transmission ratio (JS_generated / JS_target) | — | — | **1.672** |
| DI_primary (supply/demand, F_causal-aligned)   | 0.2637 | 0.2630 | -0.0008 |
| DI_supplementary (demand/supply)               | 0.1307 | 0.1384 | +0.0077 |
| F_causal_localized (M=I, n=1,186 edited units) | 0.2724 | 0.2636 | **-0.0088** |
| F_causal_global (M=I, n=34,524)                | 0.8079 | 0.8107 | +0.0028 |
| F_causal (production, M=center)                | 0.8080 | 0.8108 | +0.0028 |

Reading: **signal transmits (ratio = 1.67 >> the 0.3 fragility threshold) but model-level fairness translation is direction-inconsistent — localized goes the wrong way, global goes a tiny right way, DI is essentially flat**. The data-level Pareto (+0.0128 ΔF_causal, intrinsic ceiling per §8.7-§8.8) remains the safer headline. Full writeup:
[`baselines/metric_hardening/RESULTS.md`](metric_hardening/RESULTS.md);
methodology: [`docs/MODEL_LEVEL_METRICS.md`](../docs/MODEL_LEVEL_METRICS.md);
artifacts: `baselines/metric_hardening/results/2026-06-08T12-30-36_metric_hardening/`.

---

## Deferred (Phase 4+)

- **B1** differentiable fairness loss (`λ·(1−F_causal)` on Gumbel generations via `FAMAILObjective` + a differentiable terminal-soft-pickup grid).
- **B2 / FAMAIL** model-level dataset swaps (build edited/filtered *trajectory* datasets, then reuse `fit_and_evaluate`).
- Pure-GAN ablation (skip MLE pretrain); multi-seed paired training; eval-time Siamese realism critic + JS-divergence utility.
- Signal-maximization sweeps (large k, coordinate-descent rounds — gated); District disparate-impact ratio metric (Phase 1b; needs a confirmed per-district supply/demand definition — gated).

---

## How to run

```bash
# All baseline tests (Phase 1 + Phase 2 + Phase 3)
python -m pytest famail_temporal/baselines/ -q

# Data-level Pareto (no GAN); add --with-edit to also run the FAMAIL editing point
python -m famail_temporal.baselines.run_data_pareto --k-levels 100 500 1000 5000

# B0 generative baseline — MLE only (Phase 2; GPU recommended)
python -m famail_temporal.baselines.gan.run_b0 --epochs 5 --device auto

# B0 generative baseline — MLE + adversarial fine-tune (Phase 3; GPU recommended)
python -m famail_temporal.baselines.gan.run_b0_adversarial --mle-epochs 5 --adv-epochs 3 --device auto
```

PI-facing diagrams live in the research vault: `research-vault/FAMAIL/famail_temporal/diagrams/` (experimental design + fairness×retention Pareto).

---

## Mission 3 baselines (built, awaiting GPU)

Adds the data-augmentation baseline comparison spec'd in
[`docs/superpowers/specs/2026-07-09-mission3-data-aug-baselines-design.md`](../../docs/superpowers/specs/2026-07-09-mission3-data-aug-baselines-design.md) (2026-07-09): three
vanilla data-augmentation editors (**ST-iFGSM**, **FGSM**, **random**) that
attack the same trajectories the FAMAIL headline edited, packaged and rescored
the identical way, so the comparison table (Task 5) has a real
apples-to-apples row set alongside FAMAIL and raw. **ST-iFGSM is a FIDELITY
baseline, not a fairness one** (Meeting-41 framing) — it demonstrates a
plausible off-the-shelf adversarial-perturbation alternative, not a
competing fairness method.

Module `famail_temporal/baselines/{stifgsm_baseline,run_stifgsm_baseline,assemble_baseline_table}.py`
(+ 3 test files) built via Tasks 1-6; frozen-algorithm gate + this run-book =
Task 7. Not yet run against the real headline dir — needs the GPU, currently
held by the alpha-sweep (`famail_temporal/results/alpha_sweep/driver.sh --status`).

### Run-book (execute once the GPU is free)

```bash
# 1) Attack the same trajectories the headline edited, 3 arms (~minutes/arm on GPU).
H=famail_temporal/results/2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered
for MODE in ifgsm fgsm random; do
  python -m famail_temporal.baselines.run_stifgsm_baseline \
    --edit-dir "$H" --mode "$MODE" --seed 0 --device auto --score-fidelity
done
# Each invocation prints its packaged arm dir, e.g.
#   famail_temporal/results/<ts>_baseline_<mode>_shenzhen/
# with metrics.json["arm" | "fairness" | "fidelity"] populated
# (package_arm + _rescore + score_fidelity, run_stifgsm_baseline.py).

# 1b) Vanilla-no-op demonstration variant (textbook iFGSM/FGSM init, delta=0,
#     instead of the default PGD-style random start; --no-random-start is
#     ignored by mode=random so only run it for ifgsm/fgsm):
# for MODE in ifgsm fgsm; do
#   python -m famail_temporal.baselines.run_stifgsm_baseline \
#     --edit-dir "$H" --mode "$MODE" --seed 0 --device auto --score-fidelity \
#     --no-random-start
# done

# 2) Per arm dir: external fairness + tier-2 supply recount (existing CLIs,
#    no new code — reuse seam per the Task-7 brief).
for ARM in famail_temporal/results/*_baseline_{ifgsm,fgsm,random}_shenzhen; do
  python -m famail_temporal.baselines.run_external_fairness \
    --edit-dir "$ARM" --dataset "baseline-$(basename "$ARM")"
  python -m famail_temporal.analysis.supply_recount \
    --edit-dir "$ARM" --city shenzhen --persist-grids
done

# 3) Assemble the 5-row comparison table (raw, FAMAIL, ifgsm, fgsm, random).
#    --famail-json / --raw-json are small hand-authored stub files transcribing
#    the already-published headline numbers (never recomputed) in the schema
#    documented at the top of assemble_baseline_table.py.
python -m famail_temporal.baselines.assemble_baseline_table \
  --arm-dirs famail_temporal/results/*_baseline_ifgsm_shenzhen \
             famail_temporal/results/*_baseline_fgsm_shenzhen \
             famail_temporal/results/*_baseline_random_shenzhen \
  --famail-json famail_temporal/baselines/famail_headline_stub.json \
  --raw-json famail_temporal/baselines/raw_stub.json \
  --out famail_temporal/baselines/baseline_table
```

### Gate — verified 2026-07-09

`python -m pytest famail_temporal/ -q` → **849 passed, 8 skipped** (0 failed);
`git diff main -- famail_temporal/algorithm/ famail_temporal/evaluation/runner.py | wc -l` →
**0** (frozen-algorithm gate holds — Task 2 only *imports* `ModificationHistory`,
modifies nothing in the editing algorithm or the evaluation runner).

### Paper-facing notes (carry into the write-up)

- **Naming — these arms are "iFGSM / FGSM with random restart," NOT "vanilla ST-iFGSM."** The frozen
  driver-identity discriminator is a stationary point at an identical (original, original) pair — its
  |emb₁−emb₂| head has zero subgradient there — so a textbook vanilla iFGSM/FGSM starting at δ=0 cannot move
  and produces a *no-op* editor. The gradient arms therefore start from a PGD-style random point inside the
  ε-ball by necessity. `--no-random-start` is retained precisely to **demonstrate the vanilla no-op
  empirically** (a legitimate ablation row). The paper must label the arms accordingly.
- **Correctness catch (methodology rigor).** The final whole-branch review caught that the attack loop
  originally scored-then-stepped, so the single FGSM step was discarded and the arm returned its
  initialization; fixed (post-step scoring pass) + a dedicated gradient-path test. Any FGSM numbers must come
  from the corrected engine (commit `6da3d27`+).
- **Fidelity is trivially high for a resampling baseline.** For the Demographic-Oversampling arm (below,
  built + run), duplicated trajectories *are* real, so Fidelity-A ≈ perfect / Fidelity-B ≈ 0 by construction —
  the axis of interest there is fairness lift vs. corpus inflation / fabricated demand, not the discriminator.

### 4th arm — Demographic Oversampling (BUILT + RUN)

A **resampling** baseline (not perturbation): duplicate real seeking trajectories originating in
demographically disadvantaged regions (all three `EQUITY_AXES`, region-extremes convention) under fresh
phantom driver IDs, rebuild the demand + supply grids **additively on both channels** (demand-only is
perverse — it adds demand to already under-served cells and lowers their service ratio), and rescore the
identical way as the other three Mission-3 arms. The naive cousin of the **supply-lift (trim+lift)** editor
and a direct empirical probe of the demand-endogeneity / leveling-down limitation
(`PAPER/external-metrics/FINDINGS.md`) — a duplicate's pickup is *unobserved* demand, and the arm quantifies
how much apparent fairness pure fabrication buys, at what corpus-inflation cost. Selected from the lit-scan
(`DATA_AUG_BASELINE_CANDIDATES.md`, Candidate 4, Pastaltzidis et al. FAccT'22); scored alongside a
random-oversampling **placebo** (identical machinery, sources drawn uniformly over the whole corpus) that
isolates demographic *targeting* from mere corpus *inflation*.

Module `famail_temporal/baselines/{demographic_oversampling,run_demographic_oversampling}.py` (+2 test
files, 23 new tests) built via Tasks 1-6 on this branch, zero changes to the frozen editor or evaluation
runner. Design spec:
[`docs/superpowers/specs/2026-07-09-demographic-oversampling-baseline-design.md`](../../docs/superpowers/specs/2026-07-09-demographic-oversampling-baseline-design.md);
plan:
[`docs/superpowers/plans/2026-07-09-demographic-oversampling-baseline.md`](../../docs/superpowers/plans/2026-07-09-demographic-oversampling-baseline.md).

**Deferred:** the spec §3.3 comparison-table row (this arm placed beside raw/FAMAIL/ifgsm/fgsm/random via
`assemble_baseline_table`) is not yet assembled — ingestion of this arm's `metrics.json` schema is already
tested (`test_arm_metrics_ingest_into_baseline_table`), but the row is held until the three perturbation
arms' (ifgsm/fgsm/random) GPU runs complete, so the full 6-row table lands together with the rest of the
Mission-3 GPU run-book.

### Run-book (executed 2026-07-10, CPU only, Shenzhen v1)

```bash
# 1) Symlink the gitignored data into the worktree (see plan Task 6 Step 1), then smoke-test
#    with a dose-100 run before the real matrix.

# 2) Run the 9-arm matrix (sequential, CPU; minutes per arm).
PY=/home/robert/FAMAIL/.venv/bin/python
for spec in "targeted 2500 0" "targeted 5000 0" "targeted 10000 0" \
            "targeted 10000 1" "targeted 10000 2" \
            "placebo 5000 0" "placebo 10000 0" "placebo 10000 1" "placebo 10000 2"; do
  set -- $spec
  $PY -m famail_temporal.baselines.run_demographic_oversampling \
      --variant "$1" --dose "$2" --seed "$3" \
      2>&1 | tee -a famail_temporal/results/demo_oversample_runs.log
done

# 3) Assemble the summary (dose-response table + figure).
$PY -m famail_temporal.baselines.run_demographic_oversampling \
  --summarize famail_temporal/results/*_baseline_demo_oversample_*_shenzhen \
  --out famail_temporal/baselines/demographic_oversampling_results
```

### Results — 9-arm dose-response (2026-07-10)

`famail_temporal/baselines/demographic_oversampling_results/summary.md`:

| Arm | seed | inflation | ΔF_causal | ΔF_spatial | ΔDP (migrant/extremes) | ΔDI (migrant/extremes) | ΔTheil |
|---|---:|---:|---:|---:|---:|---:|---:|
| oversample-placebo-d5000 | 0 | 0.052 | -0.0099 | +0.0078 | +1.4904 | -0.0154 | +0.0071 |
| oversample-placebo-d10000 | 0 | 0.105 | -0.0179 | +0.0139 | +2.7730 | -0.0255 | +0.0118 |
| oversample-placebo-d10000 | 1 | 0.105 | -0.0168 | +0.0140 | +2.7733 | -0.0262 | +0.0110 |
| oversample-placebo-d10000 | 2 | 0.105 | -0.0169 | +0.0136 | +2.8115 | -0.0268 | +0.0125 |
| oversample-targeted-d2500 | 0 | 0.026 | +0.0059 | +0.0018 | -0.1759 | +0.0348 | +0.0016 |
| oversample-targeted-d5000 | 0 | 0.052 | +0.0097 | +0.0030 | -0.1737 | +0.0557 | +0.0035 |
| oversample-targeted-d10000 | 0 | 0.105 | +0.0175 | +0.0052 | +0.0601 | +0.0835 | +0.0087 |
| oversample-targeted-d10000 | 1 | 0.105 | +0.0141 | +0.0051 | +0.2662 | +0.0762 | +0.0083 |
| oversample-targeted-d10000 | 2 | 0.105 | +0.0144 | +0.0054 | +0.1930 | +0.0805 | +0.0086 |

**Headline (targeted vs. placebo, matched budget k/dose = 10,000):** targeted mean ΔF_causal =
**+0.0153** (seeds +0.0175 / +0.0141 / +0.0144), dose-monotone (+0.0059 @2,500 → +0.0097 @5,000 →
+0.0153 @10,000), vs. **placebo mean ΔF_causal = −0.0172** — fabrication *without* demographic
targeting **degrades** F_causal. Placebo ΔDP explodes (+1.49 @5,000, +2.77 to +2.81 @10,000): measured
directly from the d10,000 s0 arms' `external_fairness/external_fairness.json`
(`metrics.MigrantRatio.district_extremes.supply_demand_ratio`), uniform fabricated supply raises
`mean_advantaged` by **+3.22** (21.27 → 24.49) while `mean_disadvantaged`
rises only **+0.45** (7.07 → 7.52) — most of the placebo's fabricated supply lands in already-advantaged
cells (consistent with, but not solely explained by, `service_ratio_Y` dividing by `max(demand,
DEMAND_FLOOR)`: floored-demand cells amplify any added supply). Even **targeted** oversampling raises
`mean_advantaged` by **+3.15** (21.27 → 24.42) alongside `mean_disadvantaged`'s own +3.09 lift (7.07 →
10.16) — demographic targeting concentrates the *demand-side* draw but the additive *supply* trails still
leak into advantaged cells regardless of variant. Targeted ΔDI improves monotonically with dose (+0.035 →
+0.083); targeted ΔDP is mixed (−0.18
at low dose, +0.06 to +0.27 at d10,000 — DP is the scale-sensitive gap metric, consistent with the
DP≡gap caveat in `PAPER/external-metrics/FINDINGS.md`). ΔTheil is small and positive in every arm.

**FAMAIL comparator (side by side, not recomputed):** the trim+lift SZ headline
(`famail_temporal/results/2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered`) is **ΔF_causal =
+0.0222**. At the same k = 10,000 edit/duplicate budget, this naive baseline reaches **+0.0153 mean**
while fabricating **10.5% of the corpus**; FAMAIL redistributes real observed behavior at **zero**
corpus inflation.

### Disclosures (spec §2/§3.3, carried verbatim)

- **Phantom drivers and their pickups are fabricated, unobserved supply and demand** — each duplicate
  is a synthetic driver (fresh namespaced plate ID) added under the "an extra taxi ran the same
  seeking run" story; nothing about it was actually observed in the data.
- **Duplicated trajectories trivially pass fidelity checks by construction** — they are (near-)copies
  of real trajectories, so Fidelity-A/B are not meaningfully discriminative here. **Fidelity is NOT
  scored for this arm** (the axis of interest is fairness lift vs. corpus inflation / fabricated
  demand, not the discriminator).
- **Corpus inflation equals the dose**: `n_edited / n_corpus`, reported per arm, never hidden. At the
  real Shenzhen PRIMARY corpus (`n_corpus = 95,297` seeking trajectories), d10,000 = **10.5%**
  inflation of the whole corpus.
- **SUPPLY_FLOOR asymmetry**: `additive_supply` adds phantom presence on top of the already
  floor-clamped production `active_taxis_3d` grid (`config.SUPPLY_FLOOR = 0.1`, applied in
  `data/aggregation.py`'s `aggregate_active_taxis`), not on top of an unclamped true recount. In cells
  where the true recount sat below the floor, the additive S′ can therefore slightly exceed what a true
  recount-plus-phantoms would show — a conservative convention for the FAMAIL contrast (it can only
  flatter, never penalize, the naive baseline), and it applies identically to both the before and after
  sides so it cannot bias the reported delta.

**Additional diagnostics disclosed per arm (review-verified against `metrics.json`):**
- `origin_escape_frac` (fraction of shifted origins that leave the targeted disadvantaged region) =
  **0.177–0.189** across doses/seeds — higher than naively expected; a boundary-geometry property of
  the rigid radius-1 shift near region edges, consistent across all targeted arms (not a bug).
- `n_with_replacement` = **1,759** at d10,000 (~17.6% of the 10,000 draws), all of them in the
  **MigrantRatio** stratum (0 for AvgHousingPricePerSqM / CompPerCapita; measured from
  `duplicates.pkl` of the `..._targeted_d10000_s0_shenzhen` arm dir — `Counter(s.stratum for s in specs
  if s.with_replacement)` → `{"MigrantRatio": 1759}`) — the with-replacement fallback engages exactly as
  the spec's error-handling section requires (flagged, never silent). Root cause (verified via
  `eligible_pools`, real Shenzhen PRIMARY corpus): MigrantRatio's disadvantaged-origin pool and
  CompPerCapita's are the **same 4,907 trajectories** (`pools["MigrantRatio"] == pools["CompPerCapita"]`
  exactly, disjoint from Housing's 41,964-trajectory pool). Because `EQUITY_AXES` order draws
  CompPerCapita's 3,333-quota first from that shared pool, only 1,574 members remain unclaimed when
  MigrantRatio's turn comes (`quota 3,333 − 1,574 without-replacement = 1,759 with-replacement`,
  matching the measured count exactly) — the corpus cannot supply the budget-parity dose for this axis
  without re-duplication once a sibling axis has already drawn from the identical pool. Within
  MigrantRatio's 3,333 specs only **2,590 distinct source trajectories** appear (measured: `len({s.source_index
  for s in specs if s.stratum == "MigrantRatio"})`); across all 10,000 specs in the arm only **8,241**
  distinct source trajectories appear in total (`10,000 − 1,759`, exact — every with-replacement draw in
  this run happened to land on a trajectory some non-with-replacement draw had already claimed, since the
  shared 4,907-trajectory pool is fully exhausted between CompPerCapita's and MigrantRatio's own
  non-with-replacement allocations). A limitation of naive oversampling worth stating plainly: FAMAIL needs
  no re-duplication to reach its ΔF_causal gain at the same budget.
- `adjacency_violation_rate` = **0.0** in all 9 arms — the rigid whole-trajectory shift preserves
  internal adjacency exactly, as designed.

Dose-response figure:
[`demographic_oversampling_results/dose_response.png`](demographic_oversampling_results/dose_response.png)
(targeted vs. placebo ΔF_causal and ΔDP-migrant lines vs. dose).
