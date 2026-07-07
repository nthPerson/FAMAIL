# Variance suite report (seeds [0, 1, 2, 3, 4])

Paired B0 (raw corpus) vs FAMAIL (edited corpus), MLE-only, same seed within
each pair. Sample std (ddof=1), n=5. 20 MLE pretraining epochs (converged;
loss ~1.95 -> ~0.69), batch 32, max_tokens 256, ~104,638 training trajectories
and one generated rollout per training context.

## Definitions

### Models

- **B0** — the baseline generator: an LSTM over grid-cell token sequences,
  trained with teacher-forced MLE only (no adversarial stage) on the RAW,
  unedited trajectory corpus. Equivalent to behavioral cloning (Dr. Zhang's
  framing, Meeting 37).
- **FAMAIL** — the identical architecture, training recipe, hyperparameters,
  and seed, trained on the EDITED corpus: the persisted ST-iFGSM editing run
  (no-dedup, k=10000, causal-emphasis), in which 3,773 of 105,401 trajectories
  had their pickup relocated within the eps=2 cell budget
  (data-level Delta F_causal = +0.0128).
- **Paired (by seed)** — within a seed, B0 and FAMAIL share initialization and
  batch shuffling (same RNG seed); the ONLY difference inside a pair is the
  training corpus. Paired deltas (FAMAIL - B0) therefore cancel seed-level
  variance that would swamp small effects in unpaired comparisons.

### Fairness metrics (convention: higher = fairer, 1 = fairest)

- **f_spatial** — spatial fairness of the generated demand distribution:
  measures the geographic evenness of pickups across grid cells.
- **f_causal** — causal fairness: 1 minus the fraction of the supply/demand
  residual explained by district demographics. With Y = supply/demand and
  R = Y - g0(D) (g0 = frozen demand-only baseline),
  f_causal = R'(I - H_demo)R / R'MR. High = demographics explain little of
  the service-rate residual.
- **f_causal_localized** — the same orthogonality computation (with M = I,
  uniform weighting) restricted to the 1,186 active (cell, time-block) units
  the edit actually relocated pickups out of. Concentrates rather than
  dilutes the edit signal.
- **f_causal_global_mi** — the M = I variant computed on ALL ~34.5k active
  units: the apples-to-apples global comparator for f_causal_localized.
  Numerically ~equal to the production f_causal row here because the residual
  is approximately mean-zero (see docs/MODEL_LEVEL_METRICS.md section 3.3).

### Disparate impact (DI)

- **di_primary** — district-level disparate-impact ratio with outcome
  Y = supply/demand (clamped at DEMAND_FLOOR), aligned with f_causal's
  outcome variable: the mean district-level Y over the top-3 districts by
  hukou ratio, divided by the same mean over the bottom-3 districts.
  Hukou ratio = NonRegisteredPermanentPop / YearEndPermanentPop per district
  (from cell_demographics.pkl). 1.0 = parity between high-migrant and
  low-migrant districts; < 1 means high-migrant districts get less supply
  per unit demand.
- **di_supplementary** — the same two-level ratio under the flipped outcome
  Y = demand/supply (demand-pressure-per-cab lens); a robustness check on
  the choice of orientation.

### JS (Jensen-Shannon divergence)

A symmetric, bounded measure of the difference between two probability
distributions — here, between terminal-pickup-cell histograms over the 4,320
grid cells (one histogram per generator, built from ~104,638 rollouts).
Log base 2, so units are bits: 0 = identical distributions, 1 = disjoint.
Terminal cells are the comparison space because f_spatial/f_causal depend
ONLY on each rollout's terminal pickup cell — this distribution is the
channel any model-level fairness effect must pass through.

- **within-B0 / within-FAMAIL pairwise JS (the seed noise floor)** — JS
  between same-variant generators that differ only by seed (all C(5,2) = 10
  pairs): how different two re-trainings look from seed randomness alone.
- **cross-variant paired JS (the signal)** — JS(B0_seed_i, FAMAIL_seed_i),
  seed held fixed: how different the two variants look.
- **JS(p_raw, p_edited) (the data-level target)** — the shift the editing
  created in the corpus itself; the signal we would like the generators to
  transmit.
- **transmission ratio** — cross-variant JS / data-level target.

### Statistics

Mean +/- std uses the sample standard deviation (ddof=1) over n = 5 seeds
(10 pairs for within-variant JS). "Paired Delta" is the per-seed difference
FAMAIL - B0, aggregated the same way.

## Fairness metrics, mean +/- std

| Metric | B0 | FAMAIL | paired Delta (FAMAIL - B0) |
|---|---:|---:|---:|
| f_spatial | 0.0828 +/- 0.0001 | 0.0837 +/- 0.0004 | **+0.0009 +/- 0.0005** |
| f_causal | 0.8062 +/- 0.0028 | 0.8051 +/- 0.0015 | **-0.0011 +/- 0.0019** |
| di_primary | 0.2616 +/- 0.0023 | 0.2612 +/- 0.0015 | **-0.0004 +/- 0.0022** |
| di_supplementary | 0.1360 +/- 0.0033 | 0.1362 +/- 0.0053 | **+0.0002 +/- 0.0025** |
| f_causal_localized | 0.2367 +/- 0.0124 | 0.2197 +/- 0.0069 | **-0.0170 +/- 0.0149** |
| f_causal_global_mi | 0.8062 +/- 0.0027 | 0.8051 +/- 0.0015 | **-0.0011 +/- 0.0019** |

## JS noise floor vs transmitted signal (terminal-cell histograms, bits)

| Quantity | mean +/- std | n |
|---|---:|---:|
| within-B0 pairwise JS (seed noise floor) | 0.01232 +/- 0.00093 | 10 |
| within-FAMAIL pairwise JS | 0.01251 +/- 0.00071 | 10 |
| cross-variant paired JS (signal) | 0.01154 +/- 0.00106 | 5 |
| JS(p_raw, p_edited) (data-level target) | 0.00753 | 1 |
| transmission ratio (cross / target) | 1.532 +/- 0.141 | 5 |

Reading: the cross-variant JS is a real distributional signal only if it
clears the within-variant noise floor. If cross ~ within, the generated
distributions of B0 and FAMAIL differ no more than two B0 re-trainings do.
