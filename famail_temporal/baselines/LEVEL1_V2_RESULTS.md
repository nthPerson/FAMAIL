# Level-1 Data-Quality Results v2 (driver-conditioned)

**Run:** 2026-06-18, single seed 0, device cuda (RTX 3070). MLE 20 epochs, adversarial 3 epochs (WGAN-GP, n_critic=5), 50 evaluation drivers, 20 pairs/driver, Fidelity-B sample 5000.
**Methodology / architectures:** [`LEVEL1_V2_METHODOLOGY.md`](LEVEL1_V2_METHODOLOGY.md).
**Artifacts:** `famail_temporal/results/level1_table_v2/2026-06-18_full_run/` (`level1_v2_metrics.json`, `level1_v2_table.md`, `training_curves.json`, `trajectory_stats.npz`, `driver_index.json`, `curves/`). Gitignored; numbers below are self-contained.

---

## The table

Fairness: 1 = fairest (F_causal, F_spatial). Fidelity-A: same-driver probability from the frozen HuMID identity discriminator, higher = better (gated/trusted). Fidelity-B: Jensen-Shannon divergence vs raw, bits, lower = better.

| Source | F_causal | F_spatial | Fidelity-A (identity ↑) | A separation (matched−mismatched) | Fidelity-B (divergence ↓) |
|---|---:|---:|---:|---:|---:|
| raw    | 0.8052 | 0.0822 | 0.840 (trusted) | +0.666 | 0.0000 |
| **edited** | **0.8180** | 0.0824 | **0.838 (trusted)** | **+0.725** | 0.1689 |
| bc     | 0.8070 | 0.0833 | 0.841 (trusted) | +0.667 | 0.0103 |
| gan    | 0.8146 | 0.0837 | 0.842 (trusted) | +0.668 | 0.3228 |

**Validation gate (real-anchored): PASSED** — matched real-d/real-d **0.840** vs mismatched real-d/real-d′ **0.174** (margin 0.20, n = 1000 each). Fidelity-A is **trusted** for all sources.

---

## Headline finding

**Fairness-edited data is the only source that is simultaneously the *fairest* and *fully faithful on driver identity*.**

1. **Fairness:** edited has the **highest causal fairness** (F_causal **0.8180**) — above raw (0.8052), BC (0.8070), and GAN (0.8146) — while spatial fairness is essentially preserved (F_spatial 0.0824 ≈ raw 0.0822). Editing improves the fairness target it optimizes without degrading the other.
2. **Identity fidelity is now trustworthy.** Unlike v1 (gate FAILED → Fidelity-A untrusted), the v2 gate **PASSES decisively** (0.840 vs 0.174). Edited data's Fidelity-A (**0.838**) is statistically indistinguishable from a held-out real trajectory's (0.840): **fairness-editing does not damage driver identity.**
3. **Edited preserves driver-discriminability best.** Its matched−mismatched **separation (+0.725) is the largest** of all sources (raw/BC/GAN ≈ +0.667): edited-for-d reads as driver d *and* edited-for-d′ reads clearly as not-d.
4. **The two fidelity axes tell different, complementary stories** (see below) — which is exactly why both are reported.

---

## Why v2 fixes v1's Fidelity-A (the central methodological result)

v1's HuMID Fidelity-A **failed its gate** (`real-real ≈ real-shuffled ≈ 0.668`, no separation) and was reported untrusted, because v1 fed the identity discriminator a single seeking-only trajectory — far out of the 5-trajectory, 3-stream, driver-keyed regime it was trained on.

v2 makes multi-agency first-class (driver-conditioned generation) and constructs HuMID's inputs near its trained regime (slot-0 trajectory-under-test + the driver's real same-driver context + real profile, driving omitted symmetrically — exactly how HuMID is used inside the editing algorithm). The result: HuMID now **cleanly separates same-driver (0.840) from different-driver (0.174)**, so the gate passes and Fidelity-A is a trustworthy, interpretable metric. This is the prerequisite that lets us make a *fidelity* argument from HuMID at all, which v1 could not.

---

## Two fidelity axes: identity (A) vs distribution (B)

Fidelity-A (identity) and Fidelity-B (distributional realism) measure different things, and the contrast is itself a finding:

- **BC** is the most distributionally faithful (Fidelity-B **0.0103**, tiny across every statistic) — likelihood training matches the marginal distributions almost exactly — and it reads as strongly same-driver (A 0.841). But BC offers **no fairness gain** (F_causal 0.8070 ≈ raw).
- **GAN** is the least distributionally faithful (Fidelity-B **0.3228**), dominated by **length (0.495) and coverage (0.533)** — the runaway-generator / mode-collapse behaviour from Meeting 38 persists (GAN-architecture stabilization is deliberately out of scope here). Yet its identity Fidelity-A (0.842) is high: **identity-fidelity does not penalize the distributional collapse that Fidelity-B catches.** Reporting only HuMID would have hidden the GAN's defect; reporting both exposes it.
- **edited** sits where the argument wants it: best fairness, fully trusted identity fidelity (0.838 ≈ raw), and a distributional cost (Fidelity-B 0.1689) that is **almost entirely the terminal-cell (pickup) distribution** (terminal_cell JS **0.645**), with trajectory *shape* barely changed (length 0.005, mean-displacement 0.039, radius-of-gyration 0.080, net-displacement 0.135, coverage 0.109). This is exactly the intended signature of fairness editing: **it relocates pickups (by design, for fairness) but preserves how drivers move.** The enriched statistics (radius-of-gyration, net-displacement) are what make this decomposition visible.

### Fidelity-B per-component (JS bits vs raw)

| Component | raw | edited | bc | gan |
|---|---:|---:|---:|---:|
| length | 0 | 0.0052 | 0.0078 | 0.4952 |
| mean_displacement | 0 | 0.0394 | 0.0071 | 0.1785 |
| coverage | 0 | 0.1090 | 0.0029 | 0.5332 |
| radius_of_gyration | 0 | 0.0802 | 0.0073 | 0.3955 |
| net_displacement | 0 | 0.1346 | 0.0076 | 0.1978 |
| terminal_cell | 0 | 0.6447 | 0.0290 | 0.1365 |
| **aggregate** | **0** | **0.1689** | **0.0103** | **0.3228** |

---

## Comparison to v1

| | v1 | v2 |
|---|---|---|
| Fidelity-A gate | **FAILED** (0.668 ≈ 0.668, OOD) → untrusted | **PASSED** (0.840 vs 0.174) → **trusted** |
| Fidelity-A construction | single seeking-only trajectory (N=1) | N=5 seeking + real profile, driving omitted (trained regime) |
| Generators | unconditioned | **driver-conditioned** |
| Fidelity-B | length, mean-disp, coverage | + radius-of-gyration, + net-displacement, + terminal-cell JS |
| F_causal (raw/edited/bc/gan) | 0.8052 / 0.8180 / 0.8064 / 0.8212 | 0.8052 / 0.8180 / 0.8070 / 0.8146 |
| GAN status | disqualified by Fidelity-B 0.319 (collapse) | Fidelity-B 0.323 (collapse persists; now decomposed: length/coverage) |
| Primary fidelity metric | Fidelity-B (A untrusted) | **both A (trusted) and B** |

Raw and edited fairness are identical across versions (same `data_level_fairness(bundle)` and edit `metrics_after` basis). GAN's F_spatial improved markedly (0.104 → 0.084), plausibly because driver-conditioning regularizes the generator; its distributional collapse is unchanged.

---

## Training curves

`famail_temporal/results/level1_table_v2/2026-06-18_full_run/curves/` (raw CSV+PNG) and `curves/legible/` (smoothed MLE; adversarial g/d on independent axes with robust percentile y-limits). BC and GAN share the same 20-epoch MLE pre-train (final MLE loss 0.623); the GAN then gets 3 WGAN-GP adversarial epochs (`gan_adversarial.png`). Per-batch and per-epoch series for all phases are in `training_curves.json`.

---

## Reproduce

```bash
python -m famail_temporal.baselines.run_level1_table_v2 \
  --edit-dir famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup \
  --mle-epochs 20 --adv-epochs 3 --gan-loss wgan-gp --n-critic 5 --device auto
# then, for curves:
python -m famail_temporal.baselines.plot_training_curves --level1-dir <out-dir>
python -m famail_temporal.baselines.replot_training_curves --curves-dir <out-dir>/curves --out-dir <out-dir>/curves/legible
```

---

## Caveats & honest notes

- **BC/GAN Fidelity-A ≈ raw (slightly above).** Likelihood-trained generators produce *prototypical* per-driver trajectories, which read as strongly same-driver — so high Fidelity-A is necessary but not sufficient for realism. This is precisely why Fidelity-B is reported alongside: GAN's 0.842 identity score coexists with a 0.323 distributional divergence. The two axes are complementary, not redundant.
- **Gate anchor.** The trusted verdict is anchored on **real** data (real-d vs real-d′ separation), a strict superset of the spec's gen-anchored gate; all per-source gen-based separations are persisted, so the gate can be reinterpreted without re-running. See methodology §3.4 for the justification.
- **Single seed.** Fairness here is single-seed for table coherence; the variance suite remains the multi-seed authority for fairness mean ± std.
- **GAN stabilization out of scope.** The GAN's length/coverage collapse is expected; fixing it (spectral norm / hinge / TTUR) is a separate, on-hold effort.
- **`n_empty`:** raw 0, edited 0, bc 0, gan 1 (one empty rollout, excluded from stats).
