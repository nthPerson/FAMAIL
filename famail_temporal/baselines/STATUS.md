# FAMAIL Baselines — Status

Living status of the GAN-baseline work that motivates and evaluates FAMAIL trajectory editing.
Design spec: [`docs/superpowers/specs/2026-05-27-famail-gan-baselines-design.md`](../../docs/superpowers/specs/2026-05-27-famail-gan-baselines-design.md).

**Last updated:** 2026-05-28

---

## Argument (what the baselines prove)

| Baseline | Claim | Status |
|---|---|---|
| **B0** — generator on raw data | Bias propagates to the trained model | Implemented (Phase 2, MLE keystone) |
| **B1** — + differentiable fairness loss | A model-level fairness fix is insufficient | Deferred (Phase 3) |
| **B2** — generate-then-filter | Filtering buys fairness only by discarding scarce data | Data-level done; model-level deferred |
| **FAMAIL** — edit pickups (ε=2) | Editing wins fairness *and* retention | Editing validated; model-level deferred |

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
- Run `2026-05-27T22-29-57_1000k_causal_emphasis_dedup`: `k=1000`, `--max-per-unit 1`, **α = (0.2, 0.7, 0.1)** (causal-emphasis).
- **ΔF_causal = +0.0087** (0.8052 → 0.8139); F_spatial flat (−0.0003); 999/1000 converged.
- Balanced multi-objective (spatial + fidelity terms active) that matches pure-causal gain without gaming a single metric → preferred FAMAIL config. `run_data_pareto._run_edit` uses it.

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

## Deferred (Phase 3+)

- Real-vs-fake discriminator (reuse Siamese encoder) + Gumbel-softmax adversarial fine-tune.
- B1 differentiable fairness loss; B2/FAMAIL model-level dataset swaps; pure-GAN ablation.
- Multi-seed paired training; signal-maximization sweeps (large k, coordinate-descent rounds — gated).
- District disparate-impact ratio metric (Phase 1b; needs a confirmed per-district supply/demand definition — gated).

---

## How to run

```bash
# All baseline tests (Phase 1 + Phase 2)
python -m pytest famail_temporal/baselines/ -q

# Data-level Pareto (no GAN); add --with-edit to also run the FAMAIL editing point
python -m famail_temporal.baselines.run_data_pareto --k-levels 100 500 1000 5000

# B0 generative baseline (trains the MLE generator; GPU recommended)
python -m famail_temporal.baselines.gan.run_b0 --epochs 5 --device auto
```

PI-facing diagrams live in the research vault: `research-vault/FAMAIL/famail_temporal/diagrams/` (experimental design + fairness×retention Pareto).
