# FAMAIL — paper results bundle

**Purpose.** A single, self-contained, auditable directory holding the **final** results, tables, and figures for
the FAMAIL paper, so nothing has to be reconstructed from the broader (git-ignored) `famail_temporal/results/`
tree. Every artifact below names the exact source file it was derived from (provenance). The numbers here are the
**cleaned-data** results under the **4-feature F_causal** formulation (the paper target); the prior 3-feature set is
preserved for comparison (see `famail_temporal/results/RESULTS_INDEX.md`).

**Study in one paragraph.** Raw Shenzhen taxi GPS data contained per-driver "stuck-GPS" pickup-sink artifacts; we
detect and filter them (data cleanup), then run the FAMAIL trajectory editor (attribution + ST-iFGSM) to make the
data fairer, and test whether the fairness propagates into trained behavior-cloning policies. F_causal (causal
fairness, 1 = fairest) uses four demographic axes {AvgHousingPricePerSqM, CompPerCapita, MigrantRatio,
LogPopDensity}, selected and defended by a feature-sensitivity/dominance analysis.

## Headline findings (all on cleaned data, 4-feature F_causal)
- **Data cleanup:** 106,677 phantom pickups removed (removal rate 49.7% → 38.95%); the headline stuck-GPS sink at
  grid (29,53) alone accounted for +0.0885 of the per-cell F_spatial recovery (net global F_spatial +0.0213).
- **Pillar 1 (L1 — data quality):** the edited dataset is the **fairest faithful** source (F_causal 0.741 vs raw
  0.725, bc 0.722; GAN-generated disqualified by distributional collapse); identity-faithful; validation gate passed.
- **L2 (vanilla transfer):** driver-conditioned BC trained on edited data does **not** transfer the fairness
  (edited−raw ΔF_causal −0.0010, n.s.) — it averages it away.
- **Pillar 2 (weighted-BC):** **upweighting** edited demos in BC **recovers** it — ΔF_causal +0.019/+0.026/+0.027
  at weights 10/20/30 (all Wilcoxon p = 0.031, dose-responsive).
- **Edit vs select vs random (PI-requested ablation):** **EDIT ≫ SELECT > RANDOM** — editing the unfair
  trajectories (+0.027) beats upweighting the already-fairest ones (+0.002, ~12× smaller) which beats the random
  placebo (null). Filtering out the unfair trajectories does **not** help either (Pareto). So FAMAIL's gain is
  specific to *editing*, not reproducible by *selecting* or *removing* data.
- **Robustness:** every conclusion above also holds under the prior 3-feature F_causal (only the absolute scale
  shifts) — the argument is robust to the demographic-feature choice (editor targeting Jaccard 0.93). See
  `tables/comparison_3v4.md`.

## Contents

### `figures/`
| file | shows | source |
|---|---|---|
| `fig_dose_response.png` | **headline:** edit vs most-fair vs random ΔF_causal by upweight dose, with CIs + p | `data/weighted_bc_4feat_sweep.json` (`paired_vs_raw`) |
| `fig_l1_data_quality.png` | L1 per-source F_causal/F_spatial (edited = fairest faithful), error bars | `data/L1v2_4feat_multiseed.json` |
| `fig_l2_negative_transfer.png` | L2 edited−raw transfer Δ (n.s.) vs the weighted-BC recovery | `data/L2_4feat_metrics.json`, `data/weighted_bc_4feat_sweep.json` |
| `fig_fidb_components.png` | Fidelity-B component breakdown (relocates pickups, preserves shape) | `data/L1v2_4feat_multiseed.json` (`fidelity_b_per_component`) |
| `fig_feature_robustness.png` | 3-feature vs 4-feature headline numbers (conclusions hold) | 3-feat + 4-feat result dirs (see `tables/comparison_3v4.md`) |
| `sink_spatial_attr_before_after.png` | E16: per-cell spatial αᵢ dirty vs cleaned; sinks circled | the two editor runs' `grid_before.pkl` |
| `pareto_causal_4feat.png` / `pareto_spatial_4feat.png` | edit vs raw vs filter Pareto (F_causal / F_spatial) | `tables/pareto_points_4feat.csv` |

### `tables/`
| file | content | source |
|---|---|---|
| `dataset_summary.md` | dirty-vs-clean cleanup stats (removal rate, phantom pickups, sink cells) | `source_data{,_dirty}/processing_metadata.json` |
| `cleanup_delta_editor.csv` | editor dirty-vs-clean F_spatial/F_causal delta (sink-removal effect) | the two editor `metrics.json` |
| `experiment_cleanup_delta.md` | dirty-vs-clean L1/L2/wbc/variance headline numbers (cleanup changed no conclusion) | the 3-feat experiment dirs vs the dirty baselines |
| `sink_f_spatial_decomposition.md` | per-sink share of the F_spatial recovery (headline sink (29,53) dominates) | the two editor runs' `grid_before.pkl` (channel 0) |
| `comparison_3v4.md` | **3-feature vs 4-feature F_causal robustness** (every conclusion holds) | both result sets |
| `fcausal_feature_selection.md` / `fcausal_feature_sensitivity.md` | the demographic feature-set sensitivity + selection analysis (marginal table, VIF, set search, Pareto) that justified the 4-feature choice | `famail_temporal/analysis/fcausal_feature_sensitivity.py` |
| `pareto_points_4feat.csv` | dual-metric Pareto points (raw / filter@K / edit) | `results/analysis/pareto_4feat/` |

### `data/` — the raw result JSONs the tables/figures are computed from (copied for self-containment)
| file | source result dir |
|---|---|
| `editor_4feat_metrics.json` | `results/2026-06-28T11-46-12_…_cleaned_4feat/metrics.json` |
| `L1v2_4feat_multiseed.json` | `results/level1_table_v2/cleaned_4feat_5seed/` |
| `weighted_bc_4feat_{sweep,paired_stats,dose_response}.json` | `results/weighted_bc_sweep/cleaned_4feat_6seed/` |
| `L2_4feat_metrics.json` | `results/level2_table/cleaned_4feat_5seed/` |
| `variance_4feat_aggregate.json` | `results/variance_suite/cleaned_4feat_5seed/` |
| `fcausal_feature_sensitivity.json` / `fcausal_feature_selection.json` | `results/analysis/fcausal_feature_sensitivity/` |
| `sink_f_spatial_decomposition.json`, `dataset_summary.json` | `results/analysis/{sink_decomposition,dataset_summary}/` |

## Provenance / reproducibility
- **Editor (4-feature) edit-dir:** `results/2026-06-28T11-46-12_k-10000_causal_emphasis_no-dedup_cleaned_4feat/`
  (causal-emphasis α = 0.2/0.7/0.1, k = 10000; before-edit F_causal 0.7253). It carries the per-run provenance
  bundle (`manifest.json` with git SHA, argv, env; `timings.jsonl`) and the enrichment artifacts
  (`attribution_distribution.npz`, `convergence_curve.npz`, enriched `trajectories.csv`).
- **F_causal feature set** committed at `7fb1fb2` (`famail_temporal/config.py`). Cache:
  `cache/hat_matrices_T24_thr0.5_feat-housing-comp-migrantratio-logpopdensity.pkl`.
- **Figures/tables** are regenerated by `famail_temporal/analysis/paper_figures.py` and the analysis modules under
  `famail_temporal/analysis/` (all committed). Each weighted-BC / L2 run dir also has its own `manifest.json`.
- **3-feature comparison set** is preserved and recomputable — see `famail_temporal/results/RESULTS_INDEX.md`.

_Note: this directory is committed to git as the curated paper deliverable; the underlying `results/` tree it
copies from is git-ignored (on-disk only)._
