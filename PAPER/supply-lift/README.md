# PAPER/supply-lift/ — supply-lift editing (the lifting-up mechanism)

Curated, committed, self-contained results bundle for the **supply-lift** workstream: extending the FAMAIL
trajectory editor with a **supply lever** (seeking-tail rerouting + endogenous delta-S) so it can raise the
**under-served** group's service ratio, rather than only reducing the over-served group's (the "leveling-down"
limitation documented in `PAPER/external-metrics/`). Headline datasets are the **filtered** Shenzhen PRIMARY and
SF sf12 supply-lift runs. Built + validated on branch `supply-lift-editing` (2026-07-07 -> 07-09).

## Read this first
- **[`FINDINGS.md`](FINDINGS.md)** — the full narrative: motivation chain (leveling-down -> negative rollout ->
  oracle gate), method, Shenzhen headline, the **channel decomposition** (the framing result), SF + the honest
  **demand-endogeneity tension**, fidelity, the weighted-BC sweep, the skip-on-infeasible disclosure, and limitations.
- **[`data_provenance.md`](data_provenance.md)** — every load-bearing number -> its source artifact path + commit.

## One-line result
Shenzhen PRIMARY (filtered): F_causal **+0.0222** (vs +0.0144 trim-only); the under-served migrant group's service
ratio **rises for the first time** (`mean(Y|D)` +0.047, CI excl. 0), with the **supply channel significant on both
cities** (SZ +0.009 tier-1 / +0.024 tier-2 distinct-taxi; SF +0.020). Central caveat: on **SF** the *external*
metrics all improve (incl. the migrant axis) but the **total** `mean(Y|D)` is net-negative — the demand-endogeneity
tension in `FINDINGS.md` §5.2, presented both ways. **Rollout-allocation eval is PENDING (§9 stub).**

## Contents
- `FINDINGS.md`, `README.md`, `data_provenance.md`.
- `data/` — committed copies of the small JSON/PROVENANCE artifacts (the gitignored run outputs are the source of
  truth; these are the durable record). Filtered-run `metrics.json` + `channel_decomposition.json` +
  `supply_recount.json` + `PROVENANCE.md` for both cities; external-fairness JSON for both; the weighted-BC sweep
  (`paired_stats` / `dose_response` / `manifest`); the Stage-0 `oracle.json`; the prior trim-only rollout
  `summary.json` (the negative baseline the pending eval must beat).
- `tables/` — the per-city external-fairness report tables (`shenzhen-primary-filtered.md`, `sf12-filtered.md`):
  supply/demand levels, DP, DI, Theil, all with paired-bootstrap 95% CIs.
- `figures/` — the per-city external-fairness forest plots (`*_delta.png`).
- **Large artifacts NOT copied** (referenced by path in `data_provenance.md`): `histories.pkl`, the augmented
  trajectory `.pkl`s, and all `.npz` grids (`delta_supply_3d.npz`, `S_tier2_*.npz`) live under the gitignored
  `famail_temporal/results/..._filtered/` dirs.

## Headline datasets (gitignored source dirs, with their own `PROVENANCE.md`)
- Shenzhen PRIMARY: `famail_temporal/results/2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered/`
- SF sf12: `famail_temporal/results/2026-07-08T22-43-06_supply_lift_v1_sf12_filtered/`

## Regenerate
See `FINDINGS.md` §11 (design spec + plan + execution ledger). Code lives on branch `supply-lift-editing`
(`famail_temporal/algorithm/supply.py`, `modifier.py`, `analysis/{supply_lift_oracle,filter_infeasible_trims,
channel_decomposition,supply_recount}.py`). External-fairness + weighted-BC harnesses are the published ones
pointed at the `_filtered` edit dirs.
