# PAPER/external-metrics/ — external fairness metrics (before → after edit)

Curated, committed results bundle for **Mission 1** of the Meeting-41 plan: demonstrating the trajectory
editor improves fairness on **established metrics NOT in its objective** (supply/demand ratio, demographic
parity, disparate impact, Theil index), computed **before-edit → after-edit** over Shenzhen (3 feature
sets) + SF sf12, with paired-bootstrap 95 % CIs.

## Read this first
- **[`FINDINGS.md`](FINDINGS.md)** — the full narrative: method summary, headline result, external
  validity, the **key findings for PI discussion** (leveling-down mechanism; city-dependent housing;
  SF migrant not significant; feature-set robustness; DP ≡ gap), limitations, and exact reproduce commands.

## Contents
- `tables/` — the per-dataset report tables (`shenzhen-primary.md`, `shenzhen-gdp-comp.md`,
  `shenzhen-logpop.md`, `sf12.md`), the cross-dataset `combined.md`, and the machine-readable
  `*.json` (before/after/Δ + CIs for every metric × axis × grouping). **Committed copies** of the
  gitignored run outputs — this is the durable record.
- `figures/` — the per-dataset forest plots (`*_delta.png`).

## One-line result
Shenzhen: **unanimous, significant, feature-set-robust** improvement. SF: same direction, weaker;
compensation + Theil significant, **migrant not significant**. Central caveat: the gains are
**leveling-down** (the over-served group is reduced; the under-served group is nearly untouched) — see
`FINDINGS.md` §4.1.

## Regenerate
See `FINDINGS.md` §6, or the code-side pointer `famail_temporal/baselines/EXTERNAL_FAIRNESS_RESULTS.md`.
Design/method in `docs/superpowers/specs/2026-07-02-external-fairness-metrics-design.md`; full recipe in
`docs/superpowers/plans/2026-07-02-external-fairness-metrics.md`.
