# Training Curves — Catalog

Every trained generator now records its full training loss curve at **per-batch**
(global-step) *and* per-epoch granularity. Curves export to CSV (raw data) + PNG
(plots) via `plot_training_curves.py`. PNG/CSV files live under the (gitignored)
run dirs; the one-line regenerate commands below recreate them from the persisted
JSON, so nothing is lost if a `curves/` dir is cleared.

## What models exist, and what curves they have

| Model | Training | Curve(s) | Source data |
|---|---|---|---|
| **B0** (variance suite, 5 seeds) | MLE only (`adv_epochs=0`) | MLE loss vs epoch (per-seed) | `variance_suite/.../seed_*.json` |
| **FAM-AIL** (variance suite, 5 seeds) | MLE only | MLE loss vs epoch (per-seed) | `variance_suite/.../seed_*.json` |
| **BC** (Level-1) | MLE only, 20 epochs | MLE loss vs batch + epoch | `level1_table/<ts>/training_curves.json` |
| **GAN** (Level-1) | MLE 20 ep + WGAN-GP 3 ep | MLE loss + adversarial g/d loss (vs batch + epoch) | `level1_table/<ts>/training_curves.json` |

The variance-suite B0/FAM-AIL models are **pure MLE** (`adv_epochs=0`) — they have
no adversarial/GAN curve by design. The only adversarial (GAN) curve in active use
is the **Level-1 GAN** baseline below. (The variance suite's existing seed files
predate per-batch capture, so its curves are per-epoch; a future re-run would add
per-batch — not done, as it only refines granularity.)

## Variance suite — B0 & FAM-AIL MLE curves (existing data)

```bash
python -m famail_temporal.baselines.plot_training_curves \
  --variance-dir famail_temporal/baselines/variance_suite/results/2026-06-11T00-04-19_seeds0-4
```
Output (`.../curves/`): `b0_mle.png`, `famail_mle.png` (each overlays all 5 seeds)
+ `b0_seed{0..4}_mle.csv`, `famail_seed{0..4}_mle.csv`.
**Reading:** clean MLE convergence, loss 1.95 → 0.65 over 20 epochs, tight across
seeds — the convergence evidence behind the 20-epoch pretraining choice.

## Level-1 — BC & GAN curves (run `2026-06-17T21-43-07`)

```bash
python -m famail_temporal.baselines.plot_training_curves \
  --level1-dir famail_temporal/results/level1_table/2026-06-17T21-43-07
```
Output (`.../curves/`):

- `bc_mle.png` — BC MLE, 65,400 per-batch points (+ 20-epoch). Loss 1.95 → 0.65.
- `gan_mle.png` — GAN's MLE pretrain (same shape as BC; shared MLE phase, seed 0).
- `gan_adversarial.png` — **the GAN training dynamics:** generator (g) and critic
  (d) loss. Per-batch g = 981 points (g-step every `n_critic`=5 batches), d = 4,905
  points (d-step every batch). **Reading:** the critic overpowers the generator —
  per-epoch d loss swings −6.3 → −15.0 → +41.8 while g loss rises 0.16 → 2.5 → 4.8.
  This instability is the mechanism behind the GAN's length collapse
  (`gan_max_len` 64 vs real ~18) that Fidelity-B flags (see `LEVEL1_RESULTS.md`).

## Data schema (for custom plots)

- **Level-1** `training_curves.json`: `{bc, gan}` → `mle_epoch_losses`,
  `mle_batch_losses`, and (GAN only) `adv: {g_epoch_losses, d_epoch_losses,
  g_batch_losses, d_batch_losses}` (BC `adv` is `null`).
- **Variance suite** `seed_<k>.json`: `b0`/`famail` → `mle_losses` (per-epoch),
  `mle_batch_losses` (if present), `adv_curve` (null for MLE-only runs).
- CSVs are two columns: `step,loss`.

Capture is **observability only** — training numerics are unchanged (the loss
values recorded are exactly those already computed each step).
