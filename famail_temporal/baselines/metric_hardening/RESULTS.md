# Metric Hardening — First Real-Data Run

> Paper-ready summary of the model-level transmission + dynamic-range
> diagnostics for FAMAIL. Methodology lives in
> [`../../docs/MODEL_LEVEL_METRICS.md`](../../docs/MODEL_LEVEL_METRICS.md).
> Status is tracked in [`../STATUS.md`](../STATUS.md). Plan that drove
> the work: [`../../../docs/superpowers/plans/2026-06-06-metric-hardening.md`](../../../docs/superpowers/plans/2026-06-06-metric-hardening.md).

**Last updated:** 2026-06-08
**Run dir:** `results/2026-06-08T12-30-36_metric_hardening/` (single seed)

---

## TL;DR

| edit_dir | seed | transmission_ratio | Delta_DI_primary | Delta_DI_supplementary | Delta_F_causal_localized | Delta_F_causal_global (production) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup` | 0 | 1.672 | -0.0008 | +0.0077 | -0.0088 | +0.0028 | fragile — lead with data-level Pareto |

**Headline verdict.** The editing signal transmits through the LSTM
(transmission ratio = 1.67, well above the ~0.3 fragility threshold),
but the model-level fairness translation is **direction-inconsistent**:
production F_causal moves in the intended direction (+0.0028), DI's
primary lens is essentially flat (-0.0008), DI's supplementary lens
shows a tiny improvement (+0.0077), and the localized F_causal on the
1,186 touched units moves the WRONG way (-0.0088). With one seed and
deltas at the metric noise floor, this is not safe to lead with. The
data-level Pareto (+0.0128 ΔF_causal at the no-dedup k=10000 config,
intrinsic ceiling per §8.7-§8.8 of the editing methodology) remains
the honest headline; the model-level triple is a robustness /
limitations characterization.

---

## Headline numbers (single seed, RTX 3070, ~5 min wall-clock)

**Training context.** 5 MLE epochs each for B0 (full corpus) and FAMAIL
(edited corpus); n_generated = 104,638 rollouts each; seed = 0; both
runs MLE-only (the collapsing adversarial GAN is the opt-in
"amplification" ablation per `B0_DECISION_BRIEF.md`). MLE losses
descended cleanly (B0: 1.95 -> 0.78; FAMAIL: 1.96 -> 0.81), so neither
generator under-trained.

**Edit source.** `famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`
(shipped no-dedup k=10000 edit: data-level ΔF_causal = +0.0128,
ΔF_spatial = +0.0003). The edit touches `n_edited_active_units = 1,186`
deduplicated `(x, y, t_block)` units (matches the "~1,186 unit-distinct
editing budget" finding from the §8 calibration log).

### Transmission — does the edit survive the LSTM?

| Quantity | Value | Reading |
|---|---:|---|
| `js_target` = JS(p_raw, p_edited) | 0.00753 bits | The signal we want to transmit |
| `js_generated` = JS(p_gen_B0, p_gen_FAMAIL) | 0.01259 bits | The signal that transmitted |
| **`transmission_ratio`** = transmitted / target | **1.672** | Signal survives AND is amplified |
| `js_b0_vs_raw` | 0.01108 | B0 fidelity to raw target |
| `js_famail_vs_edited` | 0.01313 | FAMAIL fidelity to edited target |

### Disparate Impact — both Y conventions

|       | Y = supply/demand (primary; F_causal-aligned) | Y = demand/supply (supplementary) |
|---|---:|---:|
| B0     | 0.2637 | 0.1307 |
| FAMAIL | 0.2630 | 0.1384 |
| **Delta** | **-0.0008** | **+0.0077** |

Top-3 hukou districts: `[5, 8, 3]`; bottom-3: `[2, 7, 6]`.

### Localized F_causal (M = I, restricted to 1,186 edited units)

|       | F_causal_global (M=I, n=34,524) | F_causal_localized (M=I, n=1,186) |
|---|---:|---:|
| B0     | 0.8079 | 0.2724 |
| FAMAIL | 0.8107 | 0.2636 |
| **Delta** | +0.0028 | **-0.0088** |

These are the M = I (uniform weighting) form of F_causal — NOT the
production F_causal. The two fields are paired so the "did the local
signal beat the global dilution?" question is answerable. See
[`../../docs/MODEL_LEVEL_METRICS.md`](../../docs/MODEL_LEVEL_METRICS.md) §3.3.

### Production F_causal (M = I - 11'/N, the paper-headline number)

|       | F_spatial | F_causal (production) | gini_dsr | gini_asr |
|---|---:|---:|---:|---:|
| Corpus | 0.0822 | 0.8052 | 0.9384 | 0.8973 |
| B0     | 0.0837 | 0.8080 | 0.9353 | 0.8973 |
| FAMAIL | 0.0863 | 0.8108 | 0.9300 | 0.8973 |
| **B0 vs corpus** | +0.0015 | +0.0028 | -0.0031 | 0.0 |
| **FAMAIL vs corpus** | +0.0041 | +0.0056 | -0.0084 | 0.0 |
| **FAMAIL vs B0** | +0.0026 | +0.0028 | -0.0053 | 0.0 |

---

## Interpretation

### Transmission — signal survives, with anomalous amplification

`transmission_ratio = 1.67` puts us comfortably above the ~0.3 fragility
threshold, so MLE smoothing is NOT washing the edit out. In fact the
generator over-transmits: the JS between the two generators' rollouts
is 67% LARGER than the JS between the raw and edited corpora. Three
non-exclusive explanations:

1. **Independent training-seed drift.** B0 and FAMAIL were trained from
   the same seed but on different corpora, so their MLE optima drift in
   different directions in cell-distribution space. Some of
   `js_generated` reflects that drift rather than the edit itself.
2. **Long-tail compression.** The LSTM concentrates probability mass on
   high-frequency terminal cells; the marginal differences in those
   high-frequency bins between the two training corpora are amplified
   relative to the raw counts.
3. **Multinomial sampling noise** at the per-rollout level, while
   ergodic in expectation, can produce a >1 ratio on a single run.

For our purposes, the amplification is a finding to report — NOT a
bug, NOT a reason to discard the run — but it means `js_generated`
alone cannot be used as a faithful proxy for "the edit's signal at
the model level." We still have transmission; we just do not have a
faithful copy.

### Disparate Impact — signal at or below the metric's noise floor

DI's primary and supplementary lenses should move in the SAME direction
under a real fairness gain. They do not: primary is essentially flat
(-0.0008), supplementary shows a tiny improvement (+0.0077). The
absolute magnitudes are within the band we would expect from
single-seed sampling noise at a ~1% editing budget. This is the
textbook diagnostic for "the signal is at or below the metric's
resolution"; it neither confirms nor refutes the data-level claim,
but it cannot be used as positive evidence of a model-level gain.

### Localized F_causal — the strongest red flag

The localized M = I metric on the 1,186 touched units says **FAMAIL is
less fair than B0** at the local level: Delta = -0.0088, a regression
roughly an order of magnitude larger than the global Delta. The
paired global M = I metric says the opposite (+0.0028), and the
production F_causal also says the opposite (+0.0028 vs B0, +0.0056 vs
corpus). This is direction inconsistency: the same M = I metric points
in opposite directions over the touched subset vs the full corpus,
which is incompatible with the editing pipeline's intended mechanism
(the edit's effect should concentrate in the touched units).

The most likely explanations: (a) MLE smoothing absorbed the edit
inside those 1,186 units, and the residual noise dominates the local
direction; (b) the LSTM's terminal-cell pickup choices in those units
drifted in response to global training dynamics that happen to act
against the edit's direction at the local level; (c) pure single-seed
sampling noise on `N_local = 1,186`. We cannot rank these from one
run.

### Verdict

Lead with the **data-level Pareto** (+0.0128 ΔF_causal at the no-dedup
k=10000 config, ΔF_spatial = +0.0003 retention-preserving, intrinsic
ceiling validated in §8.7-§8.8 of the editing methodology). Report the
model-level transmission + DI + localized triple as a **robustness /
limitations characterization** — showing that the editing signal
transmits through the generator (and is even amplified) but does NOT
translate into a consistent model-level fairness improvement at this
signal magnitude. The +0.0028 production F_causal gain is real-sized,
but with one seed it is statistically indistinguishable from sampling
variance.

If the model-level claim ever needs to be defended, the right
follow-up is **multi-seed paired statistics** (>= 5 seeds per arm,
paired same-seed B0/FAMAIL training, paired Wilcoxon on the per-seed
Delta) plus either a larger edit budget (which §8 shows requires a
different lever than larger K — `max-per-unit` relaxation with
pile-up risk, or the soft-vs-hard gap closed in some other way) or
an editing pipeline that better targets the LSTM's gradient
sensitivity. The current data does not support a model-level headline.

---

## Reproduction

```bash
python -m famail_temporal.baselines.run_metric_hardening \
    --edit-dir famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup \
    --mle-epochs 5 --device auto --seed 0
```

Per-run artifacts (research artifact, NOT committed):

- `metrics.json` — full numerical block
- `terminal_cell_histograms.npz` — `p_raw`, `p_edited`, `p_gen_b0`, `p_gen_famail`
- `report.md` — auto-generated per-run human-readable report

Wall-clock ~5 minutes on an RTX 3070 (training dominates; metric
computation is sub-second).

See `famail_temporal/baselines/metric_hardening/results/<timestamp>_metric_hardening/`.
