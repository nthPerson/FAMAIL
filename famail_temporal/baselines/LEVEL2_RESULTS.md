# Level-2 Usability Results — Fairness Transfer

> **Headline (honest, plain): the edited data's Level-1 fairness advantage does NOT transfer through downstream behavior-cloning training.** A driver-conditioned BC policy trained on FAM-AIL-edited data is *not* more causally-fair than one trained on raw data — across 5 paired seeds the difference is **−0.0022 ± 0.0016** (edited slightly *lower*, unanimous in direction). The edit's quality is a property of the *dataset*; vanilla BC does not inherit it.

Run `2026-06-18T17-27-34` · seeds `[0,1,2,3,4]` · 50 eval drivers · 20 MLE epochs · `gen_max_tokens=256` · `max_batch_tokens=8192` · device cuda. Artifacts: `famail_temporal/results/level2_table/2026-06-18T17-27-34/` (`level2_metrics.json`, `level2_table.md`, `driver_index.json`; gitignored).

---

## The question

Level-1 established that the *edited dataset* is fairer than raw / BC-generated / GAN-generated data while staying faithful (edited F_causal 0.8180 vs raw 0.8052 — a **+0.0128** data-level gap). Level-2 asks: **does that advantage survive downstream model training** — i.e. is a model *trained on* the edited data fairer than one trained on raw data? We train a driver-conditioned behavior-cloning (BC) policy on each of four matched, full-corpus data sources, then score each *trained policy's* generated demand on the Level-1 axes.

## The table

Validation gate (real-anchored, real-d vs real-d′): **PASSED** — matched 0.8408 vs mismatched 0.1736 (n=1000 each, margin 0.20). This reproduces the Level-1 v2 gate (0.840 / 0.174) exactly, so **Fidelity-A is trusted** here as well.

Each cell is mean ± std across the 5 paired seeds (driver-conditioned BC trained on that source, then evaluated):

| Source (training data) | F_causal ↑ | F_spatial | Fidelity-A ↑ | Fidelity-B ↓ |
|---|---:|---:|---:|---:|
| raw | 0.8083 ± 0.0027 | 0.0831 ± 0.0002 | 0.8406 ± 0.0002 | 0.0121 ± 0.0014 |
| edited | 0.8061 ± 0.0025 | 0.0841 ± 0.0003 | 0.8408 ± 0.0007 | 0.0120 ± 0.0010 |
| bcgen | 0.8099 ± 0.0018 | 0.0833 ± 0.0002 | 0.8408 ± 0.0005 | 0.0163 ± 0.0007 |
| gangen | 0.8143 ± 0.0037 | 0.0839 ± 0.0008 | 0.8418 ± 0.0003 | **0.3507 ± 0.0106** |

**Reference — Level-1 *data*-level F_causal (the thing we asked to transfer):** raw 0.8052 · **edited 0.8180** · bc 0.8070 · gan 0.8146.

The contrast is the whole story: at the data level edited sits at **0.8180**; after BC training every policy — regardless of training source — sits near the **raw-data** level (~0.806–0.814). The edited data's 0.8180 is nowhere in the trained-policy column.

## Paired fairness transfer (F_causal, by seed)

Paired design: for each seed `s`, `set_all_seeds(s)` is called immediately before constructing and training each arm, so all four policies share weight-init and minibatch ordering and differ **only** in training data. This removes shared seed noise — essential because the data-level gap (~0.013) sits near the seed-noise floor (~0.012 bits, measured in the GAN-baseline work).

| Comparison | mean Δ ± std | per-seed diffs | n | Wilcoxon p | 95% CI (t) |
|---|---:|---|---:|---:|---:|
| edited − raw | **−0.0022 ± 0.0016** | −0.0013, −0.0003, −0.0032, −0.0043, −0.0020 | 5 | 0.0625 | [−0.0042, −0.0003] |
| edited − bcgen | −0.0038 ± 0.0034 | −0.0031, −0.0016, +0.0002, −0.0071, −0.0074 | 5 | 0.125 | — |
| edited − gangen | −0.0081 ± 0.0047 | −0.0053, −0.0092, −0.0045, −0.0160, −0.0057 | 5 | 0.0625 | — |

The headline `edited − raw` difference is **negative in all 5 seeds** (Wilcoxon p=0.0625 is the smallest value attainable at n=5, i.e. a perfectly consistent direction). The 95% CI excludes zero on the negative side, so the pre-registered scale-to-10 trigger ("fire if the CI crosses zero") did **not** fire; we conclude at n=5.

## What this says

1. **Fairness does not transfer through vanilla driver-conditioned BC.** The +0.0128 data-level F_causal advantage of edited over raw is absent in the trained policies. If anything the edited-trained policy is *marginally less* causally-fair than the raw-trained one (−0.0022, unanimous across seeds) — a small effect, but a real and consistent one, not a wash in the favorable direction.

2. **It is not a fidelity trade-off.** The edited-trained policy is as realistic as the raw-trained one on both axes — Fidelity-A 0.8408 ≈ raw 0.8406, Fidelity-B 0.0120 ≈ raw 0.0121. The policies are equally faithful; the fairness simply does not propagate.

3. **All real/clean sources land together near raw-data fairness.** raw 0.8083, edited 0.8061, bcgen 0.8099 all cluster ~0.806–0.810. BC imitates the *aggregate* conditional demand distribution, which is dominated by the unedited 96.4% of trajectories; the 3,773 / 105,401 relocated pickups are averaged away during imitation. **Data-level fairness ≠ model-level fairness under behavior cloning.**

4. **gangen's apparent F_causal "win" is the collapse artifact, again.** gangen has the highest trained-policy F_causal (0.8143) but a catastrophic Fidelity-B (0.3507) — the same length/coverage collapse seen in Level-1's GAN (0.323), now propagated into the policy trained on its data. Reporting *both* fidelity axes disqualifies it, exactly as at Level-1. The single-axis reading would have falsely crowned the collapsed source.

5. **bcgen (self-distillation) ≈ raw.** Training on BC-generated data yields a policy near the raw-trained one (0.8099) with clean fidelity (Fidelity-B 0.0163) — neither improving nor collapsing.

## Caveats & honest notes

- **n = 5 paired seeds.** The headline effect is tiny (≈0.2% of F_causal) and the Wilcoxon p (0.0625) is at the n=5 resolution floor; the claim we stand behind is the *qualitative* one (no positive transfer; a small, directionally-unanimous negative), not a precise effect size. The scale-to-10 trigger was not met (CI excludes zero), so we did not extend; 10 seeds would only sharpen the borderline p, not change the direction.
- **Proximity to the noise floor.** The data-level gap (~0.013) is barely above the ~0.012-bit seed-noise floor. The paired design is what gives the n=5 comparison its power; without pairing the effect would be invisible.
- **Scope = vanilla driver-conditioned BC (MLE), one downstream task.** This result speaks to behavior cloning of demand. It does not test other downstream models or training procedures; "does not transfer through *this* model" is the precise claim.
- **Generators are L1-consistent.** The BC/GAN generators that supply bcgen/gangen are pretrained on the 256-token-capped corpus (identical to Level-1 v2), not the full corpus — the ~0.7% long outliers (up to 1654 tokens) cannot be batched by the adversarial trainer, and generated trajectories are capped at 64 tokens regardless, so capping is lossless. The downstream policies still train on the full corpus (the generated sets are full-corpus-*sized*, one rollout per real seed across all 105,401, trained with the token budget). See `--gen-max-tokens` (commit `31f7b84`).
- **n_empty (empty rollouts replaced by `[BOS, start_cell, EOS]`):** raw `[0,0,0,1,0]`, edited `[0,0,0,0,0]`, bcgen `[2,0,1,2,2]`, gangen `[0,0,0,0,0]` — negligible.

## Methodology (for paper-writing)

**Design.** Four content-matched, full-corpus training sets, index-aligned to the same real seeds so only the *content* varies:
- **raw** — `bundle.trajectories` (105,401 real Shenzhen taxi trajectories).
- **edited** — the corpus with the 3,773 FAM-AIL-modified trajectories swapped in by `trajectory_id` (edit run `2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`).
- **bcgen** — one driver-conditioned rollout per real seed from a pure-MLE generator.
- **gangen** — one driver-conditioned rollout per real seed from a WGAN-GP adversarially-fine-tuned generator.

**Downstream policy.** A driver-conditioned `TrajectoryLSTM` (additive cell + time-block + driver embeddings), trained by MLE (`train_mle`, 20 epochs) with token-budgeted batching so the full corpus — including the long outliers — trains without OOM. No adversarial training downstream: the GAN is a *data source*, not a downstream model.

**Evaluation (reuses the Level-1 v2 scoring helpers verbatim).** Each trained policy's generated demand is scored on: causal/spatial fairness (`data_level_fairness` on the policy's generated pickup grid), identity **Fidelity-A** (frozen HuMID same-driver probability, two-pass matched/mismatched over 50 eval drivers, d vs d′), and enriched **Fidelity-B** (Jensen–Shannon divergence of trajectory statistics vs raw + terminal-cell JS). The real-anchored validation gate (real-d vs real-d′) is computed once and is policy-independent.

**Statistics.** Per-metric paired per-seed differences `edited − other` with a Wilcoxon signed-rank test (`_paired_diff_stats`), the headline being `edited − raw` F_causal.

**Architecture details** (generator, HuMID, the identity-branch construction, the two fidelity axes) are shared with Level-1 v2 — see [`LEVEL1_V2_METHODOLOGY.md`](LEVEL1_V2_METHODOLOGY.md). The Level-2 design and the four locked decisions are in `docs/superpowers/specs/2026-06-18-level2-usability-fairness-transfer-design.md`; the build in `docs/superpowers/plans/2026-06-18-level2-usability-fairness-transfer.md`.

## Reproduce

```bash
PYTHONPATH=$(pwd) python -m famail_temporal.baselines.run_level2_table \
  --seeds 0,1,2,3,4 --mle-epochs 20 --adv-epochs 3 --gan-loss wgan-gp \
  --n-critic 5 --device cuda
# ~4.5 h on an RTX 3070 (one-time generator pretraining ~1 h dominated by the
# WGAN-GP phase; ~34 min per seed for the 4-policy paired loop once warm).
```
