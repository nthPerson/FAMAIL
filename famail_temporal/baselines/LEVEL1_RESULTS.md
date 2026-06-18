# Level-1 Data-Quality Results

**Two-Level Argument · Level 1 (Data Quality).** Does FAM-AIL's *edited data* beat
raw / BC-generated / GAN-generated data on causal fairness, spatial fairness, and
fidelity (realism)? Spec: [`docs/superpowers/specs/2026-06-17-level1-data-quality-table-design.md`](../../docs/superpowers/specs/2026-06-17-level1-data-quality-table-design.md).

**Run:** `2026-06-17T21-43-07` · seed 0 · `--mle-epochs 20` (BC + GAN MLE), GAN
adversarial `--gan-loss wgan-gp --adv-epochs 3 --n-critic 5` · device CUDA.
Edit source: `famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`
(k_modified = 3,773 of 105,401 trajectories; α_causal 0.7 / α_spatial 0.2; ε = 2).

## The table

| Source | F_causal (↑) | F_spatial (↑) | Fidelity-A — HuMID (↑) | Fidelity-B — distributional JS (↓) |
|---|---:|---:|---:|---:|
| raw | 0.8052 | 0.0822 | 0.668 *(untrusted)* | 0.0000 |
| **FAM-AIL edited** | **0.8180** | 0.0824 | 0.665 *(untrusted)* | 0.0480 |
| BC-generated | 0.8064 | 0.0827 | 0.664 *(untrusted)* | 0.0106 |
| GAN-generated | 0.8212 | 0.0847 | 0.664 *(untrusted)* | 0.3189 |

Fairness columns are **single-seed** (this table's internal coherence). The
authoritative multi-seed fairness figures remain the variance-suite 5-seed
mean ± std. Raw/BC/GAN fairness are full-corpus `data_level_fairness`; the edited
row is the edit pipeline's own full-corpus `metrics_after` (same basis as raw —
the edit's `metrics_before.f_causal` = 0.8052 = the raw value here).

## Headline finding

**Among faithful data sources, FAM-AIL edited data has the highest causal
fairness.** Restricting to sources that are actually realistic (raw, edited,
BC — all Fidelity-B ≤ 0.048):

- Edited **0.8180** > BC 0.8064 (+0.0116) > raw 0.8052 (+0.0128 edit gain).
- Spatial fairness is essentially held (edited 0.0824 ≈ raw 0.0822): the edit
  targets causal fairness without degrading spatial fairness.

GAN-generated shows a *nominally* higher F_causal (0.8212), but it is
**disqualified by fidelity**: its trajectories collapse to the generation-length
cap (`gan_max_len = 64` vs real mean ~18), so its "fairness" is an artifact of
unrealistically long rollouts blanketing the grid, not genuine demand structure.
BC-generated is faithful but yields **no fairness gain** over raw. Only the
edited data improves fairness *while staying faithful* — that is the Level-1
data-quality claim.

## Fidelity — why Fidelity-B is the primary metric here

**The validation gate FAILED** (real-vs-real 0.668 vs real-vs-collapsed 0.660,
gap 0.008 ≪ 0.20 margin; real-vs-shuffled 0.668 — i.e. the discriminator scores
shuffled trajectories *identically* to real). This is the spec's **designed
fallback (§6)**, not a defect: the HuMID discriminator was trained on
multi-stream, driver-keyed data, and Level-1 feeds it single, reduced,
identity-less trajectories — out of distribution. It cannot rank real above
garbage in this regime, so **Fidelity-A (HuMID) is reported but flagged
untrusted**, and the discriminator-free **Fidelity-B is primary**.

Fidelity-B (Jensen-Shannon divergence of trajectory-statistic distributions vs
raw; lower = more faithful) separates the sources cleanly, and the per-statistic
breakdown shows exactly *why* the GAN fails:

| Source | length JS | displacement JS | coverage JS | aggregate |
|---|---:|---:|---:|---:|
| edited | 0.005 | 0.030 | 0.109 | 0.048 |
| BC | 0.022 | 0.006 | 0.004 | 0.011 |
| **GAN** | **0.493** | 0.017 | **0.448** | **0.319** |

The GAN's divergence is driven by **length (0.49)** and **coverage (0.45)** —
the collapse signature (over-long rollouts visiting far more cells than real
trips). Edited and BC are both close to raw (the edited residual is dominated by
a small coverage difference from scoring the 3,773-edited subset against the raw
sample; the edited trajectories are real data with only the pickup relocated).

## Training curves

Per-batch + per-epoch curves are captured for every trained model and exported
to CSV + PNG. See [`TRAINING_CURVES.md`](TRAINING_CURVES.md) for the catalog.
Highlights from this run:

- **BC / GAN MLE pretraining** converges cleanly: epoch loss 1.95 → 0.65 over 20
  epochs (65,400 per-batch points each). BC and GAN share the MLE phase (same
  seed); the GAN then diverges in the adversarial phase.
- **GAN adversarial (WGAN-GP)** is unstable: per-epoch critic loss swings
  −6.3 → −15.0 → +41.8 while generator loss rises 0.16 → 2.5 → 4.8 — the critic
  overpowering the generator, the mechanism behind the length collapse above.

## Reproduce

```bash
python -m famail_temporal.baselines.run_level1_table \
    --edit-dir famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup \
    --mle-epochs 20 --device auto --seed 0
# artifacts -> famail_temporal/results/level1_table/<timestamp>/
#   level1_metrics.json · level1_table.md · training_curves.json · trajectory_stats.npz
python -m famail_temporal.baselines.plot_training_curves \
    --level1-dir famail_temporal/results/level1_table/<timestamp>
```

## Caveats

- **Single-seed v1** for BC/GAN generation (fidelity is a coarse, gated +
  cross-checked realism check; multi-seed is a documented future upgrade).
- **Fidelity-A untrusted** this run (gate failed — OOD discriminator use). The
  data-quality conclusion rests on **F_causal + Fidelity-B**.
- The edited row's fairness is the edit's authoritative `metrics_after`
  (full-corpus); an earlier orchestrator bug computed it from only the 3,773
  modified pickups (a sparse, non-comparable grid → spurious 0.673) and was
  fixed (`fix(baselines): edited-source fairness from edit metrics_after`).

Related: [`MEETING_38_PREP.md`](MEETING_38_PREP.md) (Two-Level Argument framing),
the variance suite (authoritative multi-seed fairness), and the Pareto figure
(edited point at F_causal 0.818).
