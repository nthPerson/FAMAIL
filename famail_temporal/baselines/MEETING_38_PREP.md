# Meeting 38 Prep — Overnight Results for Dr. Zhang's Meeting-37 Action Items

**Date:** 2026-06-11 (runs executed overnight 2026-06-10 → 06-11, RTX 3070, ~3.5 h GPU)
**Branch:** `variance-suite-wgan` (off `implement-gan-baselines`)
**Artifacts:** `famail_temporal/baselines/variance_suite/results/2026-06-11T00-04-19_seeds0-4/` (headline) and `famail_temporal/results/overnight_2026-06-10/` (ablations, WGAN, logs)

---

## TL;DR

1. **Your 5-model ask was decisive — it killed a fragile headline.** With converged (20-epoch) generators and 5 paired seeds, the model-level F_causal advantage disappears: paired ΔF_causal = **-0.0011 ± 0.0019**. The single-seed +0.0028 from 2026-06-08 was seed noise. The JS analysis explains why: the data-level edit signal (0.0075 bits) sits **below the seed-to-seed noise floor of BC training (0.0123 bits)** — two B0 re-trainings differ more than B0 differs from FAMAIL.
2. **The "more pretraining" hypothesis is refuted; WGAN partially works.** BCE adversarial training collapses identically at 10 and 20 pretraining epochs and with critic-slowing. WGAN-GP is the first config that changes the dynamics: it holds a near-faithful length distribution for a full epoch before drifting — which makes **WGAN + third-party-metric early stopping** (your two suggestions combined) the first viable adversarial recipe.
3. **Generator gradient direction is verified correct** (your diagnostic #1) — unit tests now assert the descent direction for both loss modes.

---

## Action-item scoreboard

| # | Meeting-37 ask | Status |
|---|---|---|
| 1 | 5 separate BC models, mean ± std of F_spatial/F_causal | **DONE** — paired design (same seed, only training data differs), n=5, results below |
| 2 | Quantify the noise floor via JS divergence | **DONE** — within-variant pairwise JS across seeds = the noise floor; see §2 |
| 3 | Try Wasserstein GAN | **DONE** — WGAN-GP, two schedules; collapse delayed, not prevented; see §4 |
| 4 | More generator pretraining epochs | **DONE & REFUTED** — 10 and 20 epochs collapse identically; see §3 |
| 5 | "B1" = generator trained on edited trajectories vs B0 | **DONE** — that is the FAMAIL column of the variance suite (n=5, error bars) |
| 6 | PR to main | Held per Robert's direction; branch ready when wanted |
| 7 | (Dr. Zhang) improve fairness metrics | — hers; the noise-floor numbers below give a concrete target |

All generators in every experiment were strengthened to 20 MLE pretraining epochs
(loss converges: ~1.95 → ~0.69; per-epoch curves persisted per seed in `seed_<k>.json`).

---

## 1. Variance suite — paired B0 vs FAMAIL, n=5 seeds, MLE-only

| Metric | B0 (mean ± std) | FAMAIL (mean ± std) | paired Δ (FAMAIL − B0) |
|---|---:|---:|---:|
| F_spatial | 0.0828 ± 0.0001 | 0.0837 ± 0.0004 | **+0.0009 ± 0.0005** |
| F_causal | 0.8062 ± 0.0028 | 0.8051 ± 0.0015 | **−0.0011 ± 0.0019** |
| DI_primary (supply/demand) | 0.2616 ± 0.0023 | 0.2612 ± 0.0015 | −0.0004 ± 0.0022 |
| DI_supplementary (demand/supply) | 0.1360 ± 0.0033 | 0.1362 ± 0.0053 | +0.0002 ± 0.0025 |
| F_causal localized (M=I, 1,186 edited units) | 0.2367 ± 0.0124 | 0.2197 ± 0.0069 | −0.0170 ± 0.0149 |

**Readings:**
- **F_causal: no model-level effect.** The paired delta is slightly negative and within one std of zero. The 2026-06-08 single-seed result (+0.0028) does not replicate — it was inside the ±0.0028 seed variance of B0 itself.
- **F_spatial: tiny but sign-consistent.** All 5 paired deltas are positive (min +0.0001, max +0.0013). Real but ~1% of the metric's level; worth mentioning, not worth leading with.
- **DI (both conventions): flat.** Consistent with the F_causal null.
- **Localized F_causal: directionally negative** (consistent with the 2026-06-08 single-seed −0.0088), high variance. The edited units do not become locally fairer in the trained model.

## 2. The JS noise floor — why the model-level signal cannot be seen (action item 2)

| Quantity | mean ± std | n |
|---|---:|---:|
| **Within-B0 pairwise JS (the seed noise floor)** | **0.01232 ± 0.00093** | 10 |
| Within-FAMAIL pairwise JS | 0.01251 ± 0.00071 | 10 |
| Cross-variant paired JS (B0_i vs FAMAIL_i) | 0.01154 ± 0.00106 | 5 |
| JS(p_raw, p_edited) — the data-level edit signal | **0.00753** | 1 |

**The one-sentence finding:** the generated terminal-cell distributions of B0 and FAMAIL differ *no more* than two independently-seeded B0 trainings differ from each other — the data-level edit (0.0075 bits) is below the training noise floor (0.0123 bits), so at this edit magnitude **no single-model model-level comparison can detect transmission**.

This also retro-explains the 2026-06-08 "transmission ratio = 1.67": any two retrained generators differ by ~1.6× the target JS purely from seed noise. The ratio measured noise, not transmission.

**Constructive implication for the paper:** the honest model-level claim is this noise-floor characterization itself. To make the gains "more obvious" (Meeting-37 goal), either the edit budget must grow (the data-level ceiling is +0.0128 at ε=2; ε is inviolable) or evaluation variance must shrink (e.g., averaging generations across many models — at n=25 models the floor would shrink ~√5× to ≈0.005, just below the signal; cheap to run if wanted).

## 3. Pretraining ablation — BCE adversarial (action item 4): refuted

| Run | g_loss (3 epochs) | d_loss | fake length (real = 18.2) | gen length |
|---|---|---|---|---|
| mle10 | 5.15 → 8.09 → 9.23 | 0.48 → 0.33 → 0.327 (floor) | 44.6 → 50.0 → 49.3 | 48.6 |
| mle20 | 4.99 → 7.88 → 8.94 | 0.50 → 0.33 → 0.327 (floor) | 46.3 → 53.4 → 53.0 | 52.6 |
| mle20 + d-update-every 2 | 3.45 → 6.62 → 7.61 | 0.67 → 0.34 → 0.332 | 41.7 → 53.7 → 54.1 | 54.2 |

Identical collapse signature in all three: generator loss explodes, critic loss pins at the
label-smoothing floor (0.327), fake lengths blow to ~3× real **within the first epoch**.
Doubling (and quadrupling) pretraining does not change the trajectory; critic-slowing only
softens the first epoch. This is strong evidence the failure is **structural** (the critic's
last-timestep readout gives it a trivially separable length feature + teacher-forced losses
don't constrain free-running generation), not a pretraining-strength problem.

## 4. WGAN-GP (action item 3): collapse delayed, not prevented — but a viable recipe emerges

| Run | g_loss | d_loss (Wasserstein + GP) | fake length | gen length | F_causal |
|---|---|---|---|---|---|
| n_critic=5 (standard) | 0.17 → 2.51 → 4.29 | −6.3 → −14.7 → −8.7 | **16.1** → 41.2 → 51.0 | 52.5 | 0.8198 |
| gen-heavy (d-every-2) | −1.88 → −1.39 → −0.39 | −2.0 → −10.1 → −16.4 | 23.2 → 41.1 → 54.9 | 57.6 | 0.8174 |

**Readings:**
- **Epoch 1 under standard WGAN-GP is nearly faithful** (fake length 16.1 vs real 18.2) — no BCE config ever achieved this. The 1-Lipschitz constraint genuinely tempers the critic.
- Drift resumes in epochs 2–3 (exposure bias is still unfixed), ending at the same ~50-length blowup.
- The critic-heavy schedule (the standard WGAN recipe) held epoch 1 *better* than the gen-heavy schedule — the gen-favoring instinct helps BCE but not WGAN.
- The F_causal values (0.8198 / 0.8174) are **above** the MLE band (0.8062 ± 0.0028) — this is the predicted "degraded generator scores higher on a terminal-cell-blind metric" artifact (Dr. Zhang's own "rubbish increases fairness" caveat from Meeting 37), not a real gain.
- **Proposed synthesis of her two suggestions:** WGAN-GP + early stopping on a third-party distributional metric (e.g., JS to real or length fidelity) would have stopped at epoch 1 with the first working adversarial generator. One epoch of WGAN fine-tune ≈ 10 min — cheap to validate if we want the "amplification" ablation to use a non-collapsed adversarial model.

## 5. Gradient-direction verification (her diagnostic #1)

Unit tests (`gan/tests/test_wgan.py`) now assert that gradient descent on the generator
loss pushes critic scores on fakes upward (toward "real") in **both** loss modes:
- BCE non-saturating: `g_loss = BCE(D(fake), 1)` → `d(loss)/d(score) = σ(score) − 1 < 0`. Correct.
- Wasserstein: `g_loss = −mean(D(fake))` → gradient −1/B < 0. Correct.

The loss formulation was never the problem.

## 6. What this means for the paper

1. **Lead with the data-level Pareto** (ΔF_causal = +0.0128 at the intrinsic ceiling, full data retention) — unchanged, now backed by a rigorous model-level noise-floor analysis instead of a fragile single-seed number.
2. **Report the model-level work as the noise-floor characterization** (§2): a quantified explanation of *why* single-model model-level evaluation cannot adjudicate a ~1%-of-corpus edit, with the n=25-model averaging path as future work.
3. **F_spatial sign-consistency** (5/5 positive) is an honest secondary observation; at +0.0009 it should be framed as suggestive only.
4. **The adversarial story is now complete and well-evidenced**: BCE collapses structurally (5 May runs + 3 overnight ablations), WGAN-GP delays but doesn't prevent (2 runs), and the WGAN+early-stop recipe is the constructive path if an adversarial ablation is wanted in the paper.

## 7. Suggested discussion points for the meeting

1. Does the noise-floor finding change which model-level experiments are worth running before the deadline? (n=25 averaging ≈ 4 h GPU vs dropping model-level claims entirely.)
2. Is the WGAN+early-stop validation (one ~15-min run) worth doing for the amplification ablation, or do we freeze the adversarial story as-is?
3. District-level discrimination pivot (her Meeting-37 framing direction) — DI infrastructure is built and reported here; what threshold of ΔDI would she consider "obvious"?
4. PR timing (action item 6).

## Reproduction

```bash
# Headline (variance suite):
python -m famail_temporal.baselines.run_variance_suite --seeds 0,1,2,3,4 --mle-epochs 20 --device auto
# Ablations / WGAN (see famail_temporal/results/overnight_2026-06-10/driver.sh + driver_wgan.sh
# for the exact five run_b0_adversarial invocations)
```

Engineering note: the first WGAN launch crashed on a cuDNN limitation (RNN double-backward
unsupported, needed by the gradient penalty); fixed in `7ffcd98` by disabling cuDNN for the
GP forward pass, verified on GPU, and re-run. Logs of the failed attempt retained in
`logs/wgan_*.log` history.
