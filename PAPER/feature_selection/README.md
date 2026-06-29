# feature_selection — demographic-feature vetting & cross-set comparison

**Purpose.** The analysis that justifies the F_causal demographic feature choice, and the side-by-side that shows the
two-pillar argument is robust to it. F_causal is a **feature-set-specific** measure, so the choice of demographic axes
must be defended explicitly; this directory holds that defense.

## Contents

### `tables/`
| file | content | source |
|---|---|---|
| `fcausal_feature_selection.md` / `.json` (in `data/`) | per-feature marginal-contribution table, VIF / correlation matrix, set search, (F_causal, VIF) Pareto frontier | `analysis/fcausal_feature_sensitivity.py` |
| `fcausal_feature_sensitivity.md` / `.json` (in `data/`) | leave-one-out + broad-set sensitivity sweep (verdict, F_causal spread, min Jaccard/Spearman) | `analysis/fcausal_feature_sensitivity.py` |
| `comparison_3v4.md` | 3-feature ↔ 4-feature side-by-side (does the story hold; how much does the scale shift) | both sensitivity result sets |

### `figures/`
| file | shows | source |
|---|---|---|
| `fig_feature_robustness.png` | dumbbell of the four headline numbers under two feature sets; null rows marked `null (CI ∋ 0)` | both sensitivity result sets via `analysis/paper_figures.py` |

## How the PRIMARY set was chosen (and why it is defensible)

The PRIMARY metric uses **{housing, comp, migrant}** — three equity-salient axes: neighborhood wealth (housing
price), income (compensation), and **migrant/hukou population structure** (a real underserved-group axis in Shenzhen).
It is chosen for **construct validity as a *demographic* fairness lens**, on these grounds:

1. **Not an unfairness-maximizing lens.** Its before-edit F_causal is **0.799 — higher** than the density-augmented
   set's 0.725. So the choice does *not* cherry-pick the feature set that makes the baseline look most unfair; it
   leaves apparent unfairness on the table for construct validity. This directly defuses the "you chose the metric
   that makes your baseline worst" circularity concern.
2. **Well-conditioned.** Max VIF 4.45 (< the < 10 policy); targeting-stable vs the original {housing, GDP, comp}
   (top-cell Jaccard 0.96).
3. **Purely demographic.** It contains no demand-geography covariate (contrast the density set below).

## Honest caveats a reviewer will probe (and our answers)

- **The density-augmented set's F_causal drop is essentially all LogPopDensity, not population structure.** In
  {housing, comp, migrant, logpopdensity}, decomposing the before-edit drop from the base-3 metric (0.807 → 0.725,
  ΔF −0.082): **LogPopDensity supplies ~−0.073 (~90%)**; MigrantRatio's incremental contribution on top of the
  others is ~0. **LogPopDensity is a demand-density / geography variable, not a protected attribute.** We therefore
  report the density set as a **robustness check** ("does the story survive conditioning on demand density?"), *not*
  as the headline demographic lens — and we do **not** credit "population structure" for the scale change.
- **Co-dominance, not strict dominance.** At the density set's F_causal, an alternative {housing, GDP, comp,
  logpopdensity} **ties** (ΔF ≈ 3e-5) at **lower** max VIF (2.87 vs 4.51). The two sit on the same (F_causal, VIF)
  Pareto frontier — they are **co-dominant**, not one strictly dominating the other. We do not claim strict
  dominance; the density set was retained (in the sensitivity slot) for axis coverage, the alternative is noted.
- **Robustness is within the *housing-retaining* family.** The full feature-sensitivity sweep is formally
  **FRAGILE** (F_causal spread 0.178; min top-cell Jaccard 0.56) — driven entirely by *dropping housing*, which
  collapses editor targeting (drop_housing Jaccard 0.56, F_causal jumps to 0.90). **Housing is a load-bearing axis we
  retain by design.** All three reported sets retain housing; within that family targeting is stable (Jaccard ≥ 0.92,
  per-cell Spearman ≥ 0.84). Claims of robustness are scoped to this family, not to "any feature choice."
- **Ecological resolution = 10 districts.** The demographics resolve to only **10 distinct district-level profiles**
  broadcast onto cells, so VIF / correlation / F_causal projection have ~10 effective DOF, not the ~34.5k active
  units. Cell-level attribution via district covariates carries a standard **ecological-fallacy** caveat; the
  correlations behind feature drops (e.g. migrant×GDP r ≈ −0.90) are estimated from 10 points and are uncertain. The
  4-feature cap is itself DOF-driven.
- **Associational, not causal.** F_causal is the partial R² of a cross-sectional OLS of the demand-adjusted residual
  on observational district demographics — no identification. It measures demographic *predictability* of service,
  not a causal effect. *(Paper-facing rename of the construct is a pending PI decision.)*

## Cross-set comparison status

`comparison_3v4.md` documents the two **sensitivity** sets (3-feature ↔ density 4-feature) and shows every directional
conclusion reproduces, only the absolute scale shifting. A full **3-way** comparison that adds the PRIMARY
{housing, comp, migrant} set will be added here when the PRIMARY re-run completes (see top-level README status).
