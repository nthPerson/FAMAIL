# Model-Level Metric Hardening — Methodology

> Companion to [`TRAJECTORY_EDITING_METHODOLOGY.md`](TRAJECTORY_EDITING_METHODOLOGY.md).
> The trajectory editing pipeline operates at the *data level* (over the
> corpus pickup grid). To translate the data-level result into a paper
> headline, we train MLE-only B0 and FAMAIL generators (full corpus vs.
> edited corpus) and ask: does the editing signal survive the LSTM, and
> does the resulting model-level fairness number actually change in the
> intended direction? Three diagnostic metrics — one transmission check
> plus two dynamic-range metrics — make that question answerable. This
> doc fixes each metric's motivation, formula, reading rules, and
> reproduction details so they remain a single source of truth.

The metrics live in:
- [`baselines/transmission.py`](../baselines/transmission.py) — JS terminal-cell transmission
- [`baselines/district_metrics.py`](../baselines/district_metrics.py) — district disparate impact (DI)
- [`baselines/localized_metrics.py`](../baselines/localized_metrics.py) — localized F_causal
- [`baselines/run_metric_hardening.py`](../baselines/run_metric_hardening.py) — orchestrator CLI

First real-data run results are written up in
[`../baselines/metric_hardening/RESULTS.md`](../baselines/metric_hardening/RESULTS.md).

---

## 1. Transmission check — Jensen-Shannon over terminal cells

### 1.1 Motivation

F_causal and F_spatial only see each rollout's **terminal pickup cell**:
the rest of the trajectory is irrelevant to the metric. So the model-level
B0-vs-FAMAIL contrast reduces to a question about the marginal distribution
over `(x, y)` of the last token. A LSTM trained with MLE on a corpus whose
terminal-cell distribution shifted by ~1% (the editing budget) may or may
not faithfully reproduce that shift in its sampled rollouts: multinomial
sampling, EOS-truncation, and KL-style averaging can collectively smooth
the signal away.

Without an explicit transmission check, a flat model-level headline could
mean either (a) the editing signal washed out in MLE smoothing or (b) the
signal transmitted but the metric is insensitive to it. These have very
different remedies. The transmission check separates them up-front.

### 1.2 Formula

For each of the four corpora — `raw`, `edited`, `gen_B0`, `gen_FAMAIL` —
build a normalized histogram over flat cell ids `flat = x * GY + y`
(length `N_CELLS = 4320`), taking the **last state** of each trajectory
as the pickup. Out-of-grid pickups are dropped (a no-op in production;
guards the small synthetic bundle in tests).

Jensen-Shannon divergence (in bits, base-2 log) for two distributions
`p, q` over the same support:

```
m = 0.5 * (p + q)
JS(p, q) = 0.5 * KL_bits(p || m) + 0.5 * KL_bits(q || m)
KL_bits(a || b) = sum_{i: a_i > 0} a_i * (log2(a_i + eps) - log2(b_i + eps))
```

Properties: symmetric (`JS(p, q) = JS(q, p)`), bounded in `[0, 1]` bits
(0 iff `p == q`; 1 iff disjoint support). The `eps = 1e-12` clip avoids
`log(0)` on missing-support bins without distorting the value.

Reported quantities:

| Name | Definition | What it measures |
|---|---|---|
| `js_target` | JS(p_raw, p_edited) | The marginal shift the edit introduced — the signal we WANT to transmit |
| `js_generated` | JS(p_gen_B0, p_gen_FAMAIL) | The marginal shift between the two generators' rollouts — the signal that DID transmit |
| `transmission_ratio` | `js_generated / js_target` | ~1 = faithful; <<1 = washed out; >>1 = generator amplifies the signal |
| `js_b0_vs_raw` | JS(p_gen_B0, p_raw) | B0 fidelity to its own training corpus |
| `js_famail_vs_edited` | JS(p_gen_FAMAIL, p_edited) | FAMAIL fidelity to its own training corpus |

### 1.3 Reading rules

- **`transmission_ratio > ~0.3`** — signal survives. The model-level
  metric is at least *capable* of seeing the edit; if it still reads
  flat, the limitation is in the fairness metric's sensitivity, not in
  MLE smoothing.
- **`transmission_ratio < ~0.3`** — signal does not survive. The MLE
  generator washes out the editing budget; chasing a model-level
  headline at the current edit magnitude is futile. Lead with the
  data-level Pareto and report transmission as the failure mode.
- **`transmission_ratio >> 1`** — signal is amplified. Possible
  amplification mechanisms include correlated sampling drift between the
  two seeds, divergence in MLE optima between the two training runs, or
  the LSTM compressing the long tail of the terminal-cell distribution
  in a way that exaggerates the marginal difference. This is not a
  bug per se, but cuts against using `js_generated` directly as a proxy
  for the edit's strength: the model-level shift is no longer a faithful
  copy of the data-level shift, even if it points in the same direction.
- **`js_b0_vs_raw`** and **`js_famail_vs_edited`** are sanity checks: if
  either is much larger than `js_target`, the generator drifted further
  from its corpus than the edit did from the raw target, and the per-
  generator noise dominates the signal we wanted to read off.

### 1.4 Reproduction

```bash
python -m famail_temporal.baselines.run_metric_hardening \
    --edit-dir famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup \
    --mle-epochs 5 --device auto --seed 0
```

Histograms are persisted as `terminal_cell_histograms.npz` alongside
`metrics.json`. Edge-case notes:
- Generators with `max_len < typical-trajectory-length` will record EOS
  early; the recorded terminal cell is then the LSTM's chosen pickup at
  the truncation point. We keep `max_len = MAX_GEN_LEN` (config default)
  to match Phase-2 conventions.
- `js_target` can be zero if the edit didn't shift the marginal cell
  distribution at all (e.g., an edit that perfectly preserved every
  trajectory's terminal cell); the ratio is then `nan` and we
  effectively have no transmission signal to evaluate.

---

## 2. District-level Disparate Impact (DI) — both Y conventions

### 2.1 Motivation

The production F_causal is `1 - r^2_demo`, where `r^2_demo` is the
demographic variance share in `R = Y - g_0(D)` at the unit level. Two
known limitations of this scalar:

1. **Small dynamic range.** Around the corpus baseline (~0.805), a
   ~1% data-level edit produces a delta on the order of `10^-3`, which
   is comparable to the noise floor of a single training seed. Reading
   the editing signal off F_causal alone is statistically fragile.
2. **No demographic interpretability.** The scalar tells you whether
   demographics explain residual variance, but it does not tell you
   *which* demographic group fares better or worse. For the "fairer for
   hukou minorities" framing of the FAMAIL claim, we need a metric
   indexed by demographic group.

DI fills both gaps: it has a larger dynamic range (a ratio of group
means, not a 1 minus an `r^2`), and it explicitly groups districts by
the hukou (non-registered permanent population) ratio.

### 2.2 Formula

For each district `d`, restrict to its active units (`mask_3d=True`)
and define two per-unit ratios:

```
Y_primary(unit)       = supply_N / max(demand_N, DEMAND_FLOOR)
Y_supplementary(unit) = demand_N / max(supply_N, supply_floor)
```

Then average to a per-district scalar and the DI is the ratio of the
mean over the top-`n_top` hukou districts to the mean over the
bottom-`n_bottom` districts (both ranked by hukou-ratio, ascending):

```
Y_d_primary       = mean_{units in district d, active} Y_primary(unit)
Y_d_supplementary = mean_{units in district d, active} Y_supplementary(unit)

DI_primary       = mean_{d in top-n_top  hukou} Y_d_primary
                   / mean_{d in bottom-n_bottom hukou} Y_d_primary

DI_supplementary = mean_{d in top-n_top  hukou} Y_d_supplementary
                   / mean_{d in bottom-n_bottom hukou} Y_d_supplementary
```

Default `n_top = n_bottom = 3` (matches the spec's "top-3 vs bottom-3"
framing). Districts with zero active units are dropped before grouping.

Two-level averaging (within-district, then across-district) is
intentional: it normalizes for within-district size differences AND
between-district population differences, and treats each district as
one unit of analysis.

**Primary vs supplementary.** `Y_primary = supply / demand` is the same
ratio that F_causal regresses on (a district where supply chases demand
is "well-served"), so a positive `Delta DI_primary` aligns with a
positive `Delta F_causal`. `Y_supplementary = demand / supply` reads
the same disparity from the other direction (a district with high
demand pressure on supply is "underserved"). Both numbers should move
in the SAME direction under a real fairness improvement — they are
informative jointly as a robustness check.

### 2.3 Data-source provenance

The hukou ratio is computed as:

```
NonRegisteredRatio = NonRegisteredPermanentPop10k / YearEndPermanentPop10k
```

The raw counts live in **`cell_demographics.pkl`'s `district_demographics`
dict** (in `config.SOURCE_DATA_DIR`, i.e., `famail_temporal/source_data/`),
NOT in the project-root CSV `all_demographics_by_district.csv` (which
does not carry a literal `NonRegisteredRatio` column; it would require
the same ratio derivation off the same upstream counts).

The pkl is the canonical in-package convention used by `preprocess.py`
and the `MigrantRatio` derivation in
[`famail_temporal/data/demographics.py`](../data/demographics.py). The
two pkls — `grid_to_district_mapping.pkl` and `cell_demographics.pkl` —
must agree on the `district_to_id` mapping; the loader asserts this so
that downstream DI never indexes the wrong hukou row per district.

### 2.4 Reading rules

- **Both conventions should move the same way.** A real fairness
  improvement that registers in F_causal should produce sign-aligned
  raw deltas on BOTH conventions (the prompt's "same direction" rule:
  both `Delta DI_primary` and `Delta DI_supplementary` should be
  positive, or both negative, under a real edit-driven shift; the
  metric's sign is a research-time convention rather than a strict
  per-group fairness arithmetic). If the two deltas disagree on sign,
  or one is essentially flat while the other moves, the signal is at
  or below the metric's noise floor and DI is not a reliable
  discriminator at this edit magnitude.
- **DI on a hardened pickup grid is a level number, not a delta from
  parity.** `DI_primary > 1` means the top-hukou group has more
  supply-per-demand than the bottom-hukou group; `DI_primary < 1`
  means the reverse. The interesting paper-time quantity is
  `Delta DI = DI_FAMAIL - DI_B0`, not the level.
- **Magnitude.** With the current ~1% editing budget, even a "real"
  signal will move DI by `10^-3` to `10^-2`. Anything below that band
  is consistent with single-seed noise. The smoke run finds
  `Delta DI_primary = -0.0008`, `Delta DI_supplementary = +0.0077` — a
  signal at the noise floor in opposite directions, which is the
  textbook diagnostic for "below the metric's resolution."

### 2.5 Reproduction

DI is computed inside `run_metric_hardening`; there is no separate CLI.
Edge-case notes:
- Districts outside Shenzhen carry `district_id = -1` on the grid;
  these are never active in `mask_3d`. If any active unit ever lands
  on a `-1` cell, `n_active_per_district[d=-1]` would surface that;
  the current loader does not run that check explicitly but the
  index-by-district step would error if it tried to use `-1` as a
  valid row.
- `n_top + n_bottom` must not exceed the number of covered districts.
  `compute_di` raises if that fails.

---

## 3. Localized F_causal — restricted to edited units

### 3.1 Motivation

F_causal averages over all ~34,524 active units in the corpus, but the
editing pipeline only touches ~1,186 of them (the unit-distinct editing
budget surfaced in the §8 calibration log). The data-level
`Delta F_causal = +0.0128` is, in that sense, a heavily diluted version
of the change concentrated in the touched units. Restricting the metric
to those units gives the editing signal a much larger denominator share
and isolates the question "did the model translate the data-level edit
into a local model-level effect?" from the dilution.

It also gives a paired comparison: localized F_causal uses the same
`M = I` (uniform weighting) form of the metric on a subset of N; the
global F_causal at `M = I` over ALL N is its directly-comparable
counterpart. Both numbers appear in the per-run `metrics.json` so the
"local vs global dilution" question is answerable in one place.

### 3.2 Formula

Both fields share the residual definition (same as production
F_causal):

```
demand_N = pickup_3d[mask]            # over all active units
supply_N = active_taxis_3d[mask]
D_clamp  = max(demand_N, DEMAND_FLOOR)
Y        = supply_N / D_clamp
g0       = bundle.g0_func(D_clamp)    # frozen, fit once at preprocess time
R        = Y - g0
X_demo   = bundle.hat_matrices['X_demo']   # (N, p) standardized demographics
```

Both fields then use the M = I (uniform-weighting) form:

```
F_causal(M=I) = 1 - r^2_demo
              = R' (I - H_demo) R / (R' R)
              = 1 - R' H_demo R / (R' R)
```

where `H_demo = X_demo (X_demo' X_demo)^{-1} X_demo'`. The Frisch-Waugh-
Lovell (FWL) identity lets us avoid materializing the dense `N x N`
hat matrix at production `N ~ 34k`:

```
R' (I - H_demo) R = R' R - (X' R)' (X' X)^{-1} (X' R)
```

so the computation is `O(N * p)` in flops and memory — matching the
convention in [`fairness/hat_matrices.py`](../fairness/hat_matrices.py).

`f_causal_global` uses ALL active units. `f_causal_localized` restricts
`R` and `X_demo` to the rows whose `(x, y, t_block)` matches a unit in
`histories.pkl::original.states[-1]` (the ORIGINAL pickup unit of each
edited trajectory — that is the unit whose demand the edit moves mass
OUT of, so the change is concentrated there). The histories list is
deduplicated to unit-level (a single unit touched by N trajectories
counts once) and intersected with `mask_3d=True` (active units only),
yielding `n_edited_active_units` rows for the localized regression.

Degenerate cases:
- `R` zero-norm -> return 1.0 (no residual variance to explain).
- `X_demo` zero columns / rank 0 -> defensive 0.0 fallback (the formula
  limit at `H = 0` would be 1.0; the plan-spec fallback is intentionally
  conservative).

### 3.3 The M=I global vs production M=center: a note on numerical coincidence

`localized_f_causal` returns two F_causal fields built with M=I (uniform
weighting):

- `f_causal_localized` — restricted to the edited active units (~1k-4k).
- `f_causal_global` — over ALL active units (~34k).

The production F_causal in `data_level_fairness` uses `M = I − 11'/N`
(centering). On this data the two formulations agree to ~1e-4: the smoke
run shows `f_causal_global` = 0.8079 (M=I) and `b0_fairness.f_causal` =
0.8080 (M=center) for the same B0 generator. This is not a coincidence:
the residual `R = Y − g_0(D)` is approximately mean-zero by construction
(g_0 fits the demand-only marginal), so `R'R − (Σ R)²/N ≈ R'R` and the
two denominators essentially match. The pair appears in `metrics.json`
for completeness and to keep the localized regression formally clean —
NOT because the two formulations yield different numbers on this data.

The substantive comparisons of interest are between **generators** (B0
vs FAMAIL) at fixed M-convention, and between **scales** (full N vs
edited subset) at fixed M=I. Not between M-conventions.

### 3.4 Reading rules

- **`Delta_localized` should be substantially larger than the
  corresponding global `Delta`** under a healthy model-level
  translation: the edit's effect is concentrated in those 1,186 units,
  so removing the 33k diluting units should magnify whatever signal is
  there.
- **`Delta_localized` going the WRONG way** is the strongest red flag
  among the three metrics. It says the model translated the data-level
  edit into a local fairness *regression* on the very units the edit
  was supposed to help. Possible causes: (1) MLE smoothing absorbed
  the edit and the residual noise dominates the local direction, (2)
  the LSTM's terminal-cell pickup choices in those units drifted in
  response to a different signal that the edit interacted with, (3)
  pure single-seed sampling noise on a small `N_local`.
- **Both deltas small, both same direction** is the "real but weak
  signal" pattern; multi-seed paired tests would be the next step.

### 3.5 Reproduction

```bash
python -m famail_temporal.baselines.run_metric_hardening \
    --edit-dir famail_temporal/results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup \
    --mle-epochs 5 --device auto --seed 0
```

Edge-case notes:
- `histories.pkl` must exist in `--edit-dir`. The current Phase-3 edit
  source ships it; future edit dirs should preserve it.
- The deduplication treats `(x, y, t_block)` as the unit key — so two
  edited trajectories with the same original pickup unit count as one
  localized row. This matches the data-level F_causal's unit-of-analysis
  convention (one row per active unit, not one row per trajectory).
- If the edit dir's `histories.pkl::original` list is empty or all
  entries fall outside `mask_3d`, `n_edited_active_units = 0` and
  `f_causal_localized = nan`. The orchestrator does not error; the
  null is informative.

---

## 4. Joint reading — the three-metric robustness picture

A "real" model-level fairness gain shows all three signals consistently:

| Metric | Healthy signature |
|---|---|
| Transmission ratio | In `[~0.3, ~2]` — signal survives without anomalous amplification |
| DI primary + supplementary | Both move in the same direction, magnitude > noise floor |
| Localized F_causal Delta | Same sign as data-level Delta, magnitude > global Delta |

When any of these disagree, the model-level headline is fragile. The
correct paper response is to lead with the data-level Pareto (where
ground truth is observable and the intrinsic-ceiling argument has
already been validated in §8.7-§8.8 of the editing methodology) and
report the model-level triple as a robustness characterization, not a
headline.

---

## 5. See also

- [`TRAJECTORY_EDITING_METHODOLOGY.md`](TRAJECTORY_EDITING_METHODOLOGY.md) —
  data-level editing pipeline and the §8.7/§8.8 intrinsic-ceiling argument.
- [`../baselines/STATUS.md`](../baselines/STATUS.md) — phase status, Phase 4
  metric-hardening entry.
- [`../baselines/metric_hardening/RESULTS.md`](../baselines/metric_hardening/RESULTS.md) —
  first real-data run numbers and paper-ready interpretation.
- [`F_CAUSAL_METHODOLOGY_NOTES.md`](F_CAUSAL_METHODOLOGY_NOTES.md) —
  M = I vs M = I - 11'/N derivation.
