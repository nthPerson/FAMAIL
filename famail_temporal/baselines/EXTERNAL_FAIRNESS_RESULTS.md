# External fairness metrics — results & how-to (code-side pointer)

**The curated results + full findings live in [`PAPER/external-metrics/`](../../PAPER/external-metrics/)**
(`FINDINGS.md` is the narrative; `tables/` + `figures/` are the committed record). This file is the
code-side entry point.

## What this is
Established fairness metrics computed **before-edit → after-edit** over active `(cell,t)` units — to prove
the editor improves fairness on metrics **not in its objective** (Meeting 41 P0). Metrics:
**supply/demand ratio (group levels), demographic parity, disparate impact, Theil index**, on
`Y = supply/demand`; two groupings (district-extremes, median-split) × three equity axes
(housing, comp, migrant); paired unit-level bootstrap CIs (`B=1000`).

## Modules
- `external_fairness.py` — pure metrics / grouping / regions / `paired_bootstrap` (numpy in, scalar out).
- `external_fairness_io.py` — `service_ratio_Y`, `per_unit_demographics`, `build_edited_pickup_3d`.
- `run_external_fairness.py` — `assemble_results`, `write_json` / `render_markdown` / `write_figure`, CLI.
- Tests: `tests/test_external_fairness*.py` (24 tests).

## Run
```bash
python -m famail_temporal.baselines.run_external_fairness \
  --edit-dir <results_dir_with_histories.pkl> --dataset <label> --bootstrap 1000 --seed 0
# SF: prefix FAMAIL_CITY=sf12 (DataBundle.load() has no city arg; the env var selects the city).
# Combine: --combine <json1> <json2> ... --out-dir <dir>
```
Outputs → `famail_temporal/baselines/external_fairness/results/<label>/` (**gitignored** — curate into
`PAPER/external-metrics/` to keep). Exact edit-dir paths for all 4 datasets: `PAPER/external-metrics/FINDINGS.md` §6.

## Headline
Shenzhen: unanimous, significant, robust across all 3 feature sets. SF sf12: same direction, weaker
(compensation + Theil significant; **migrant not significant**). **Central caveat — leveling-down:** the
gap closes by reducing the over-served group, not raising the under-served (FINDINGS §4.1). Do **not**
ship the tables to the paper before reading FINDINGS §4 (leveling-down, city-dependent housing, DP ≡ gap).

Design: `docs/superpowers/specs/2026-07-02-external-fairness-metrics-design.md` ·
Recipe: `docs/superpowers/plans/2026-07-02-external-fairness-metrics.md`.
