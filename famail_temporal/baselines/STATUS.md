# FAMAIL Baselines — Status

Living status of the GAN-baseline work that motivates and evaluates FAMAIL trajectory editing.
Design spec: [`docs/superpowers/specs/2026-05-27-famail-gan-baselines-design.md`](../../docs/superpowers/specs/2026-05-27-famail-gan-baselines-design.md).

**Last updated:** 2026-05-28

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
