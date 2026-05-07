# Using FAMAIL Fairness-Attribution Exports

Assumes you have read **`../docs/RESEARCHER_HANDOFF.md`**. If you have not, start there.

## TL;DR

This document tells you how to load a FAMAIL fairness-attribution export and feed its per-cell fairness signal into a GAIL, GAN, or generic offline-RL training loop. It covers loading patterns, the sign convention you must respect, the axis semantics that have a broadcast trap inside them, three self-contained training-method recipes, a numbered pitfalls catalogue, and a sanity-check checklist.

It does **not** re-derive the fairness math (that lives in [`../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`](../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md)), it does **not** prescribe a framework, and it does **not** opine on which metric to use, how to normalize, or how to weight your fairness term against your other losses. It shows you how to apply each option correctly; the choice is yours.

## What you have

Each export directory at `famail_temporal/exports/<timestamp>_<name>/` contains:

- **Three `.pkl` artifacts** — `fairness_attribution_dense.pkl` (block-level tensors for fast lookup), `fairness_attribution_long.pkl` (pandas DataFrame for filtering), `fairness_attribution_tuples.pkl` (dependency-free row iteration). Algebraically equivalent; pick by convenience.
- **`metadata.json`** — provenance sidecar (git SHA, source-data SHA, config snapshot, overall F values, active-cell counts per block).
- **`README.md`** — the auto-generated reference card for that specific export. Carries the export's actual F values and `n_days`. This how-to is the prescriptive companion to that reference card.
