# FAMAIL GAN Baselines Design Specification

**Date:** 2026-05-27
**Status:** Draft — awaiting user review
**Target directory:** `famail_temporal/baselines/` (new)
**Reuses (does not modify):** `famail_temporal/fairness/`, `famail_temporal/algorithm/`, the Siamese discriminator encoder in `discriminator/model/`
**Standing constraint:** the trajectory-editing algorithm and its intermediate calculations are frozen; any change to *what it computes* requires explicit user sign-off via AskUserQuestion before implementation (see §3.4).

---

## 1. Executive summary

This spec defines a suite of generative-model baselines that **motivate and evaluate** the FAMAIL trajectory-editing method. The central claim moves from *data-level* fairness (what `famail_temporal` measures today) to *model-level* fairness: **a trajectory GAN trained on FAMAIL-edited data produces rollouts that are measurably fairer than one trained on raw data — at equal data quantity — and fairer-per-retained-trajectory than filtering.**

All baselines share **one generator architecture trained on different dataset variants**; only the intervention changes. This single-architecture / vary-the-data design controls for model differences and is the strongest footing for an ablation argument.

### Scope at a glance

| Area | Decision |
|---|---|
| Primary claim | **Model-level**: edited-data generator → fairer rollouts than raw-data generator |
| Fallback claim | **Data-level Pareto**: editing dominates filtering on the dataset's fairness×retention frontier (no GAN needed) |
| Bonus claim | **Amplification**: GAN rollouts can be *less* fair than training data (mode collapse drops minority modes) |
| Representation | Grid-cell trajectory sequences on the 48×90×T grid (no 126-dim feature re-derivation) |
| Generator | Autoregressive **LSTM**, categorical next-cell head, **Gumbel-softmax** sampling (differentiable for B1) |
| Conditioning | **Light**: `(start cell, start time-block)`; **no driver-profile** |
| Discriminator (training) | Real-vs-fake sequence critic, reusing the Siamese LSTM **encoder** retasked to single-trajectory realism |
| Training paradigm | **Primary**: MLE pretrain → Gumbel-softmax adversarial fine-tune (standard SeqGAN recipe). **Ablation**: pure Gumbel-softmax GAN |
| Baselines | **B0** raw, **B1** raw + differentiable fairness loss, **B2** filtered@K, **FAMAIL** edited — all four in scope |
| Signal strategy | Maximize SNR within an **inviolable ε=2** cap (see §3) |
| Evaluation critic | Existing same-agent Siamese discriminator reused as an independent realism critic at eval time |
| Statistics | Paired multi-seed (raw vs edited share seed/init), mean±std + paired tests |

### Why these baselines exist

The argument chain (from the FAMAIL Evaluation & Argumentation Strategy Outline) has four links, each earned by one baseline:

| Argument link | Claim to establish | Baseline |
|---|---|---|
| Bias propagates to IL | A model trained on biased data reproduces/amplifies the bias | **B0** |
| "GANs have no mechanism" | A model-level fairness mechanism is insufficient/inferior | **B1** |
| Data scarcity / can't discard | Filtering buys fairness only by destroying irreplaceable data | **B2** |
| Solution: editing | Editing wins fairness *and* retention simultaneously | **FAMAIL** |

Each baseline is the rebuttal to a specific reviewer objection ("maybe the data is fine" / "just add a fairness loss" / "just discard the bad trajectories").

---

## 2. Design decisions (recorded during brainstorming)

| # | Decision | Rationale |
|---|---|---|
| 1 | Paper anchored on the **model-level** claim, with the data-level Pareto prepared as a fallback | Strongest contribution; the data Pareto is nearly free and de-risks a null model-level result |
| 2 | Generator operates on **grid-cell sequences**, not 126-dim states | Rollouts aggregate directly into `pickup_3d/dropoff_3d`; sidesteps the GPS-feature re-derivation that has blocked the end-to-end test |
| 3 | **All four** baselines (B0/B1/B2/FAMAIL) | B1 directly rebuts "just add a fairness loss"; full reviewer-proofing |
| 4 | Generator must support **differentiable discrete generation** (Gumbel-softmax) | Required by B1's fairness loss; also enables relaxed adversarial training |
| 5 | Training = **MLE pretrain → adversarial fine-tune** (primary); pure Gumbel-softmax GAN as ablation | Standard SeqGAN recipe; MLE base keeps the generator faithful so the small editing signal isn't buried under mode-collapse noise; ablation honors the "pure GAN" instinct and showcases amplification |
| 6 | **LSTM** backbone | Matches the existing Siamese LSTM, fits the 8 GB RTX 3070, fast across many variants×seeds |
| 7 | **Light conditioning** `(start cell, start time-block)`, no driver | Captures the comparable-grid and paired-comparison signal benefits at low cost; driver identity adds little to aggregate fairness |
| 8 | Discriminator reuses the Siamese **encoder** design retasked real-vs-fake | Reuses existing, tested feature-normalization + encoder code |
| 9 | Existing same-agent Siamese discriminator reused as an **eval-time realism critic** | Independent utility measure ("does a rollout look like real driver behavior?") |
| 10 | **ε=2 is inviolable**; signal maximized by other means (§3) | ε=2 matches the cGAIL IL training distribution (a hard constraint, not a tunable knob) |
| 11 | New code in `famail_temporal/baselines/`; editing algorithm untouched except gated §3 items | Keeps the frozen algorithm frozen; isolates new work |

---

## 3. Signal maximization (dedicated; ε=2 inviolable)

The headline rests on detecting a fairness difference that is **small at the data level** (validated editing moves F_causal ~+0.8%, r² 0.195→~0.186) after it passes through a lossy, stochastic GAN. The model-level signal can be written as:

```
signal  =  (data-level ΔF)  ×  (GAN transmission fidelity)  −  noise
```

All three terms are levers. The plan below maximizes SNR **without ever relaxing ε=2 from the original pickup cell**. Ranked by expected SNR gain per unit of effort:

### 3.1 Noise reduction (cheapest, largest wins)
- **Paired training.** The raw-data and edited-data generators share random seed, initialization, and hyperparameters, differing *only* in training data. Seed-induced variance then largely cancels in the difference; report paired statistics (paired t-test / Wilcoxon) across seeds.
- **Generation-level pairing.** Because the generator is conditioned on `(start cell, time-block)`, generate raw-model and edited-model rollouts from the **identical set of contexts**. The difference in their aggregated pickup grids is then pure learned-behavior difference, not start-distribution noise.
- **Large generation sample.** Estimate each model's fairness from a large rollout set (default: corpus-matched ~105k, optionally more) so the metric estimate has low variance. Sampling is cheap.

### 3.2 Numerator gains (bigger data-level ΔF, ε untouched)
- **Large `k` (editing count).** Edit more trajectories. `k` is a selection-count knob, not an algorithm change — no sign-off needed. Default sweep extends well past the validated `k=1000` (e.g., `k ∈ {1000, 5000, 10000, 25000}`), bounded by the S1 diversity constraint (`--max-per-unit 1`). *(Convention: lowercase `k` = editing selection count; uppercase `K` = filtering removal count in B2 — kept distinct throughout this spec.)*
- **Objective–metric alignment.** When headlining F_causal / disparate-impact, edit with `α=(0,1,0)` (pure F_causal), the validated strongest config.
- **Coordinate-descent re-attribution rounds** *(gated — see §3.4)*. Re-attribute and re-edit over `R` passes; each pickup remains capped at **ε=2 from its original cell** (cumulative cap, never stacked to 4). Re-attribution after each round recovers improvement that the greedy single pass leaves on the table. First investigate the existing `--iterative-topk` flag to see whether this is already implemented.

### 3.3 Metric lens (larger dynamic range, low risk)
- **District-level disparate-impact ratio** as a reporting headline: a *transform of existing quantities* (district supply/demand + hukou), not a new differentiable objective, so it carries none of the development risk of F_spatial/F_causal. Larger dynamic range than F_causal's r² → more detectable.
- **Localized reporting.** Report fairness within the edited districts/units alongside the global metric; the localized effect is larger than the washed-out global aggregate.
- **Optional hukou `NonRegisteredRatio` feature** in F_causal *(gated — see §3.4)*: the strongest single bias-relevant variable, currently unused.

### 3.4 Algorithm-change gates
Two items above change *what the editing algorithm computes* and therefore require explicit user sign-off (AskUserQuestion) **before implementation**, per the standing protocol:
1. **Coordinate-descent / multi-round editing** (if not already covered by `--iterative-topk`).
2. **Adding the hukou `NonRegisteredRatio` feature** to F_causal.

Everything else in §3 (large K, paired training, generation sample size, DI-ratio reporting, objective-metric alignment with existing `α`) is a knob or a reporting choice and needs no sign-off.

---

## 4. Architecture

### 4.1 Shared generative model

- **Trajectory representation.** A trajectory is a sequence of grid cells `(x, y)` with a time-block index over the 48×90×T grid (currently T=24 hourly). We model the passenger-seeking segments — the corpus FAMAIL already edits — whose terminal state is the fairness-relevant **pickup** cell.
- **Generator G.** Autoregressive LSTM (default 2 layers, 128 hidden — matching the discriminator). At each step it emits a categorical distribution over the next cell; sampling uses **Gumbel-softmax** with annealed temperature (default 1.0 → ~0.5), making generation differentiable end-to-end for B1.
- **Conditioning context.** `(start cell, start time-block)` embedded and supplied to G's initial state. No driver-profile.
- **Discriminator D (training).** Real-vs-fake single-trajectory critic reusing the Siamese LSTM **encoder + feature normalizer** design (`discriminator/model/model.py`), with a real/fake head replacing the same-agent head.
- **Eval-time realism critic.** The *existing trained* same-agent Siamese discriminator, used unmodified to score whether a rollout looks like coherent driver behavior (independent of D).

### 4.2 Training paradigm

**Primary (per data variant):**
1. **MLE pretrain** G by next-cell maximum likelihood on that dataset's trajectories.
2. **Adversarial fine-tune** G vs D with Gumbel-softmax relaxation.
3. Select the best checkpoint by a validation criterion (held-out realism + stability).

**Ablation:** identical nets, skip step 1 (pure adversarial from scratch). Secondary; used for robustness + the amplification sub-claim.

### 4.3 The four baselines

| | Training data | Objective | Expectation |
|---|---|---|---|
| **B0** | raw corpus | standard adversarial | rollout fairness ≈ data fairness (or worse via amplification) |
| **B1** | raw corpus | standard + `λ·(1−F)` differentiable fairness penalty on Gumbel generations (reuses `famail_temporal.fairness`); λ swept (e.g., {0.1, 1, 10}) | modest gain, fights realism + GAN instability → motivates data-level intervention |
| **B2** | filtered@`K` (remove top-`K` most-unfair trajectories by attribution) at several retention levels | standard adversarial | must drop a large fraction to match editing's fairness |
| **FAMAIL** | edited corpus (ε=2, signal-max config from §3) | standard adversarial | high fairness at 100% retention |

### 4.4 Rollout → grid → metrics

Sampled rollouts are aggregated into `pickup_3d/dropoff_3d` grids (48×90×T) using the existing aggregation path, then fed to the existing F-metric code. This is the seam that makes the model-level claim computable without GPS-feature reconstruction.

---

## 5. Evaluation protocol

- **Fairness metrics.** F_spatial, F_causal (canonical convention, **1 = fairest**) + district disparate-impact ratio, computed on rollout-aggregated grids.
- **Utility / realism.** (i) eval-time Siamese critic score; (ii) JS divergence vs held-out real data on pickup distribution, trip-length distribution, and OD-flow distribution; (iii) optional downstream next-cell prediction accuracy.
- **Data retention.** Fraction of corpus used in training (100% for B0/B1/FAMAIL; <100% for B2).
- **Headline figures.**
  1. Fairness × retention Pareto: B2 curve vs FAMAIL point vs B0 point.
  2. Fairness × utility scatter across all baselines.
  3. Rollout-fairness bar chart (B0/B1/B2/FAMAIL) with seed error bars.
  4. The one-liner: "to match editing's fairness by filtering, you must discard X% of irreplaceable data."
- **Statistics.** N seeds (default 5), paired across raw/edited; report mean±std and paired-test p-values.
- **Splits.** Hold out trajectories for realism/utility evaluation to avoid overfitting fairness on the training set.

### Success criteria
- **Primary (model-level):** edited-data generator's rollout fairness exceeds the raw-data generator's, statistically significant under the paired test on at least one fairness metric (F_causal or DI ratio), at matched utility.
- **Pareto:** the FAMAIL point dominates the B2 fairness-retention curve.
- **B1:** the fairness-loss GAN achieves less fairness gain per unit of utility loss than editing.
- **Fallback (if model-level is null):** the data-level Pareto still demonstrates editing > filtering, and the work is framed as a data-auditing+editing method.

---

## 6. Code organization

New package `famail_temporal/baselines/`:

| File | Responsibility |
|---|---|
| `config.py` | Hyperparameters, K levels, seed list, λ sweep, generation sample size |
| `data.py` | Dataset-variant builders: raw / filtered@K / edited; reuses `algorithm` attribution + editing |
| `generator.py` | Autoregressive LSTM generator + Gumbel-softmax sampling |
| `discriminator.py` | Real-vs-fake critic (reuses Siamese encoder design) |
| `train.py` | MLE pretrain + adversarial fine-tune loops; B1 fairness-loss hook; pure-GAN flag |
| `rollout.py` | Conditioned sampling → grid aggregation |
| `evaluate.py` | Fairness / utility / retention metrics + figures |
| `experiment.py` | Matrix runner (baselines × variants × seeds), paired protocol |
| `tests/` | Real tests: shape/contract, MLE overfit-a-batch, rollout→grid invariants, metric wiring |

**Reuses:** `famail_temporal.fairness` (F-metrics, differentiable for B1), `famail_temporal.algorithm` (attribution + editing for B2/FAMAIL variants), `discriminator/model` (encoder design + trained same-agent critic).

---

## 7. Build order (de-risked — bankable results first)

1. **Data-variant builders + data-level Pareto** (the fallback claim; no GAN). Fast, bankable.
2. **B0 end-to-end**: G + D + MLE pretrain + rollout→grid + fairness eval (the keystone infrastructure everything reuses).
3. **FAMAIL + B2** (reuse B0 infra; just swap the training dataset).
4. **B1** (differentiable fairness loss — most engineering).
5. **Pure-GAN ablation + signal-max iterations + multi-seed scale-up.**

---

## 8. Risks and open items

| Risk / item | Mitigation / note |
|---|---|
| Detectability of the small effect | The entire §3 plan; paired design + large samples are the cheapest big wins |
| GAN training instability | MLE pretrain + LSTM backbone; pure-GAN kept as a secondary ablation only |
| Algorithm-change gates | Coordinate-descent editing and hukou feature gated behind sign-off (§3.4) |
| Compute budget (RTX 3070, 8 GB) | One arch × ~6 dataset variants × N seeds × 2 stages — bound before scale-up; defaults N=5 seeds, editing-`k` levels=4, B2 retention levels=6 (§9) |
| Conditioning fidelity | Rollouts must aggregate to a corpus-comparable grid; validated by comparing B0 generations to the raw corpus distribution |
| `--iterative-topk` semantics | Investigate before building coordinate-descent editing (may already exist) |

---

## 9. Open parameters (concrete defaults; tunable)

| Parameter | Default |
|---|---|
| Seeds (paired) | 5 |
| B2 retention levels | remove {1, 2, 5, 10, 20, 40}% + one level matched to editing's fairness |
| Editing `k` sweep | {1000, 5000, 10000, 25000} with `--max-per-unit 1`, `α=(0,1,0)` |
| Coordinate-descent rounds R *(gated)* | 1 (single pass) until sign-off; then sweep {1, 2, 3} |
| Generation sample size | corpus-matched (~105k) rollouts per model |
| LSTM | 2 layers, 128 hidden |
| Gumbel temperature anneal | 1.0 → 0.5 |
| B1 fairness weight λ | sweep {0.1, 1, 10} |
