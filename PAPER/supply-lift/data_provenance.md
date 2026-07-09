# Supply-lift bundle — data provenance

Every load-bearing number in [`FINDINGS.md`](FINDINGS.md) mapped to its source artifact and the commit that
produced it. Rule of this bundle: **numbers come from the artifacts, not from memory** — each was re-read from the
JSON/report copied into `data/`, `tables/`, `figures/` (or referenced by path for large gitignored files).

**Branch:** `supply-lift-editing` · **bundle curated at HEAD:** `44ac837` (initial `0ffe731`); updated
2026-07-09 with the landed rollout-allocation eval + SF supply-lift fidelity.
**Source-of-truth ledger:** `.superpowers/sdd/progress.md` (+ task reports in `.superpowers/sdd/`).

## Datasets (gitignored source dirs)
| short name | path | derived at |
|---|---|---|
| SZ filtered | `famail_temporal/results/2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered/` | git_sha `85c6dbc` |
| SZ source (unfiltered) | `famail_temporal/results/2026-07-08T14-03-03_supply_lift_v1_shz_primary/` | git_sha `85c6dbc` |
| SF filtered | `famail_temporal/results/2026-07-08T22-43-06_supply_lift_v1_sf12_filtered/` | git_sha `8605915` |
| oracle | `famail_temporal/analysis/supply_lift_oracle_out/oracle.json` | commit `0fac8f7` (Task 1) |
| BC sweep | `famail_temporal/results/weighted_bc_sweep/supply_lift_v1_shz_primary_filtered_6seed/` | manifest git_sha `e8f7d26` (dirty) |
| prior rollout (trim-only) | `famail_temporal/baselines/external_fairness/results/option_a_rollout/summary.json` | 2026-07-07 run |
| rollout (supply-lift) | `famail_temporal/baselines/external_fairness/results/option_a_rollout_supplylift/summary.json` | 2026-07-09 run |
| SF fidelity (supply-lift) | g5 scratchpad `g5_fidelity_{a,b}_result_sf12.json`; durable record = `.superpowers/sdd/g5-fidelity-report.md` "SF supply-lift fidelity" | 2026-07-09 |

Tooling commits: filter `2e8a83c`, recount `--persist-grids` `edfdb6e`, channel decomposition `8605915`,
decomposition unit tests `9d6e5cc`, SF replay-identification fix `e8f7d26`.

---

## §1 Motivation chain
| claim | value | source |
|---|---|---|
| Leveling-down mechanism (2455/2455 edits in advantaged; 32x leverage; 93% at floor; presence 1.8 vs 17.6) | — | `PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md` §0-§5 |
| Prior trim-only rollout: poor pickup share 0.0500 -> 0.0452 @ w30, 0/6, p=.031 | raw 0.0500+-0.0004; edited_w30 0.0452+-0.0011; mean delta -0.004808; n_pos 0; wilcoxon_p 0.03125 | `data/rollout_trimonly_prior_summary.json` (`edited_w30.MigrantRatio.pickups.share_D_delta`); `LEVELING_DOWN_MECHANISM.md` §6.4 |
| Oracle G0: baseline 7.0734; ceiling_fraction +0.882; ceiling_distinct +0.827; threshold +0.3 | 7.073448 / 0.882001 / 0.827378 / 0.3; N_D 6950; n_candidates 5538; n_applied 2571; runtime 16.0s | `data/oracle.json` |
| Supply-only greedy ceiling +0.786 | +0.786 (n_applied 5528) | `.superpowers/sdd/progress.md` Task-1 controller decomposition (scratchpad `oracle_decomp.py`, not committed) |

## §2 Method
| claim | source |
|---|---|
| TAIL_LEN=4; TAIL_TAPER [0.25,0.5,0.75,1.0]; 5x5 neighborhood; EPSILON_BALL 2.0; ACCEPT_RULE objective | `data/shz_primary_filtered_metrics.json` -> `config_snapshot` |
| Budget split SZ 2455->2340 trim / 7545 lift; SF 1371->1324 / 629 | `data/shz_primary_filtered_metrics.json`, `data/sf12_filtered_metrics.json` (`n_trim`,`n_lift`,`n_skipped_infeasible_trim`) |
| Provably-exact adjacency repair (3000-case brute force) | `.superpowers/sdd/progress.md` Task 5; spec/plan `docs/superpowers/{specs,plans}/2026-07-08-supply-lift-editing*` |
| G1 legacy byte-identical | `.superpowers/sdd/progress.md` Task 8 |

## §3 Shenzhen PRIMARY (filtered) headline
All from `data/shz_primary_filtered_metrics.json` unless noted.
| metric | before -> after (delta) | source |
|---|---|---|
| F_causal | 0.798795 -> 0.821013 (+0.022218) | metrics.json `metrics_before/after/deltas.f_causal` |
| F_spatial | 0.103427 -> 0.109785 (+0.006357) | metrics.json `.f_spatial` |
| gini_dsr | 0.898093 -> 0.885558 (-0.012535) | metrics.json `.gini_dsr` |
| Theil | 0.1550 -> 0.1468 (-0.0082, CI [-0.0092,-0.0072]) | `tables/shenzhen-primary-filtered.md`; `data/external_fairness_shz_primary_filtered.json` |
| mean(Y|D) migrant | 7.073448 -> 7.120293 (+0.046845, CI [+0.002215,+0.093177]) | `data/shz_primary_filtered_channel_decomposition.json` `levels` + `channels.total` |
| SDR gap migrant (adv-dis) | 14.1989 -> 13.3412 (-0.8576, CI [-0.9603,-0.7573]) | `tables/shenzhen-primary-filtered.md` (MigrantRatio district_extremes DP) |
| DI migrant | 0.3325 -> 0.3480 (+0.0155, CI [+0.0128,+0.0182]) | `tables/shenzhen-primary-filtered.md` |
| adv migrant level 21.2723 -> 20.4615 (-0.811) | | `tables/shenzhen-primary-filtered.md` |
| trim-only comparators (+0.0144 F_causal; migrant level flat 7.0734->7.0734; DI +0.0097) | | `PAPER/external-metrics/` FINDINGS §2, §4.1; `tables/shenzhen-primary.md` |

## §4 Channel decomposition (Shenzhen)
All from `data/shz_primary_filtered_channel_decomposition.json` (migrant, district-extremes, N_D=6950, B=2000, seed 0).
| channel | point (CI) sig | key |
|---|---|---|
| supply tier-1 | +0.009100 [+0.005353,+0.012955] SIG | `channels.supply` |
| supply tier-2 | +0.024208 [+0.020756,+0.027880] SIG | `channels.supply_tier2` |
| demand | +0.037745 [-0.006505,+0.083627] n.s. | `channels.demand` |
| total tier-1 | +0.046845 [+0.002215,+0.093177] SIG | `channels.total` |
| total tier-2 | +0.061953 [+0.016714,+0.107554] SIG | `channels.total_tier2` |
| supply_first / demand_second | +0.010492 SIG / +0.036354 n.s. | `channels.supply_first`, `.demand_second` |
| Tier-2: MAE 0.0, corr 1.0, 34524 cells; 9885/9885 histories matched, 0 unmatched; F_causal tier-2 0.815774 (+0.016979) | | `data/shz_primary_filtered_supply_recount.json` (`sanity_check_1...`, `substitution_stats`, `metrics`) |

## §5 SF (filtered)
| metric | before -> after (delta, CI) | source |
|---|---|---|
| F_causal | 0.875151 -> 0.907916 (+0.032765) | `data/sf12_filtered_metrics.json` `.deltas.f_causal` |
| F_spatial | 0.184629 -> 0.202652 (+0.018023) | `data/sf12_filtered_metrics.json` |
| gini_dsr | 0.826567 -> 0.789490 (-0.037076) | `data/sf12_filtered_metrics.json` |
| Theil | 0.2137 -> 0.2056 (-0.0081, CI [-0.0095,-0.0067]) | `tables/sf12-filtered.md`; `data/external_fairness_sf12_filtered.json` |
| migrant DP (district-extremes) | 2.1466 -> 2.0708 (-0.0758, CI [-0.1141,-0.0317]) | `tables/sf12-filtered.md` |
| migrant DI (district-extremes) | 0.7076 -> 0.7137 (+0.0061, CI [+0.0012,+0.0105]) | `tables/sf12-filtered.md` |
| migrant median_split DP -0.0341 [-0.0693,+0.0021] n.s. | | `tables/sf12-filtered.md` |
| supply channel | +0.019468 [+0.011148,+0.027857] SIG | `data/sf12_filtered_channel_decomposition.json` `channels.supply` |
| demand channel | -0.052484 [-0.077679,-0.030197] SIG (neg) | `channels.demand` |
| total | -0.033016 [-0.059895,-0.009052] SIG (neg) | `channels.total` |
| mean(Y|D) migrant before 5.194497 -> tier1 5.161481 | | `channel_decomposition.json` `levels` |
| trim-only SF +0.0139 comparator | | `PAPER/second-dataset/` FINDINGS §2 |
| raw-dir total -0.0363 (persists post-filter) | | `.superpowers/sdd/task-11e-sf-eval-report.md` §4 |

## §6 Fidelity
### §6.1 Shenzhen
| claim | value | source |
|---|---|---|
| Fidelity-A gate PASSED | matched 0.8489 / mismatched 0.1920 | `.superpowers/sdd/task-11a-report.md` (filtered) |
| Fidelity-A filtered edited-combined 0.8457 vs raw 0.8489 (-0.0033) | | `.superpowers/sdd/task-11a-report.md` |
| Fidelity-A mode split trim -0.0059 / lift -0.0031 | trim 0.8428, lift 0.8457, raw 0.8487 | `.superpowers/sdd/g5-fidelity-report.md` §1 (unfiltered) |
| Fidelity-B lift 0.2645 vs trim 0.1601 (~1.65x); trim-only ref 0.1689 | | `.superpowers/sdd/g5-fidelity-report.md` §2 |

### §6.2 SF (filtered corpus, sf_12 discriminator)
All from `.superpowers/sdd/g5-fidelity-report.md` "SF supply-lift fidelity" (scoring on
`..._supply_lift_v1_sf12_filtered`, batch 32; scripts env-parameterized `run_g5_fidelity_{a,b}.py`).
| claim | value | source / cross-check |
|---|---|---|
| Fidelity-A gate PASSED | matched 0.9578 / mismatched 0.0344 (n 240/240; separation 0.92) | g5 report; **cross-check to ~1e-7**: `PAPER/second-dataset/data/eval_l1v2_sf12_metrics.json` `gate` (0.9578103/0.0343547) — cite the **sources node** per house precedence, not the multiseed mean |
| Fidelity-A raw 0.9578 -> edited-combined 0.9581 (+0.0003) | == published trim-only raw->edited (+0.0003; sources.edited 0.9581168) | g5 report; `eval_l1v2_sf12_metrics.json` `sources.raw/edited.fidelity_a` |
| Fidelity-A trim-mode 0.9582 / lift-mode 0.9577 | lift n=236 pairs, 12 drivers | g5 report SF table |
| Fidelity-B trim 0.1087 ~= published trim-only 0.1058 | | g5 report; `eval_l1v2_sf12_metrics.json` `sources.edited.fidelity_b` 0.10578 |
| Fidelity-B lift 0.2649 ~= Shenzhen lift 0.2645 (city-independent cost) | combined 0.1145; all edited scored (1324+629, no sampling) | g5 report SF table + §2 (SZ) |
| Profile-dominance caveat maintained (SF A = weak instrument) | | g5 report SF section; `PAPER/second-dataset/` FINDINGS §5 |

## §7 Weighted-BC sweep (filtered PRIMARY corpus)
All paired diffs from `data/weighted_bc_paired_stats.json`; identity gate from `data/weighted_bc_manifest.json`.
| arm | F_causal mean (wilcoxon_p) | key |
|---|---|---|
| edited w1 | +0.002317 (0.15625) n.s. | `f_causal.edited` |
| w10 / w20 / w30 | +0.023163 / +0.028049 / +0.031014 (all 0.03125, 6/6) | `f_causal.edited_w10/20/30` |
| random w10 / w30 | -0.001088 (0.5625) / -0.002730 (0.09375) null | `f_causal.random_w10/w30` |
| most_fair w10/w20/w30 | +0.003425 / +0.001413 / +0.000664 (0.094/0.56/0.84) null | `f_causal.most_fair_w10/20/30` |
| F_spatial propagates w10/w20/w30 | +0.004172 / +0.004767 / +0.005677 (all 0.03125) | `f_spatial.edited_w10/20/30` |
| Fidelity-A edited ~+0.0001..+0.0006 (negligible) | | `fidelity_a.edited_w*` |
| Fidelity-B edited w30 +0.015666 (0.03125) | | `fidelity_b.edited_w30` |
| identity gate matched 0.8475 / mismatched 0.1931 | | `data/weighted_bc_manifest.json` `gate_matched/mismatched` |

## §8 Skip-on-infeasible
| claim | value | source |
|---|---|---|
| SZ 115 reverted, same 115 on re-derivation, G4 100% | 115; ids listed | `data/shz_primary_filtered_PROVENANCE.md` |
| SZ delta-S rebuild == persisted, max abs diff 0.0 | sum -2.74999961 both | `data/shz_primary_filtered_PROVENANCE.md`; `data/shz_primary_filtered_metrics.json` `provenance.delta_supply_reconstruction_equivalence` |
| SZ F_causal +0.0209 (unfiltered) -> +0.0222 (filtered) | 0.81972 vs 0.82101 | `.superpowers/sdd/task-11a-report.md` |
| SF 47 reverted; edit-relative 100%; absolute 87.40% (raw baseline 84.95%); 14.9% raw pre-violate | | `data/sf12_filtered_PROVENANCE.md` (`compliance`), `.superpowers/sdd/task-11e-sf-eval-report.md` §1 |
| SF F_causal +0.0223 -> +0.0328 | | `data/sf12_filtered_metrics.json`; `.superpowers/sdd/task-11a-report.md` |
| Rule adopted 2026-07-08 (precedes numbers) | user_decision_date 2026-07-08 | both `*_PROVENANCE.md` |

## §9 Rollout-allocation eval (landed 2026-07-09)
Supply-lift = `data/rollout_supplylift_summary.json`; prior trim-only = `data/rollout_trimonly_prior_summary.json`.
Key = `<arm>.MigrantRatio.pickups.share_D_delta` (allocation) / `...states.share_D_delta` (cruising).
| claim | value (supply-lift vs prior) | keys |
|---|---|---|
| w30 pickup drain attenuated ~40%, not reversed | -0.0028992 (0/6, p=.03125) vs -0.0048082 (0/6, p=.03125); ratio 0.603 | `edited_w30.MigrantRatio.pickups.share_D_delta` both files |
| w10 | -0.0022843 (0/6, .03125) vs -0.0032695 (0/6, .03125) | `edited_w10....pickups...` |
| w1 | -0.0008250 (0/6, .03125) vs +0.0003361 (3/6, .5625 n.s.) | `edited....pickups...` |
| seeking-STATE share n.s. all arms, both eras | supply-lift p = .3125/.21875/.84375 (w1/w10/w30); prior p = .84375/.09375/.5625 | `edited*.MigrantRatio.states.share_D_delta` both files |
| prior baseline framing 0.0500 -> 0.0452 @ w30 | | `LEVELING_DOWN_MECHANISM.md` §6.4 reference rows (see §1) |

## §10 Limitations
| claim | source |
|---|---|
| SF tier-2 recount deferred (tier2_grid null) | `data/sf12_filtered_channel_decomposition.json` `tier2_grid`; `.superpowers/sdd/task-11e-sf-eval-report.md` §5 |
| SF raw adjacency 14.9% / GPS gaps to 18.6 cells | `.superpowers/sdd/task-11e-sf-eval-report.md` §1 |
| 2 alternate SZ feature sets deferred | `.superpowers/sdd/progress.md` Task-10 checkpoint scope (e) |
| Rollout allocation attenuated, not reversed (honest boundary) | `data/rollout_supplylift_summary.json` (see §9) |
| SF Fidelity-A weak instrument (profile-dominated, 236 lift pairs) | `.superpowers/sdd/g5-fidelity-report.md` SF section (see §6.2) |
