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

---

## §1. Shared preamble

Read this once before any recipe. The recipes in §2 link back here rather than restating shared content.

### §1.1 Loading the export

Three artifacts; pick by convenience. They carry the same data.

```text
# Dense — fastest tensor lookup; best for a training loop
load fairness_attribution_dense.pkl as dense
spatial = dense["spatial"]            # (gx, gy, T) float, NaN on inactive
causal  = dense["causal"]             # (gx, gy, T) float, NaN on inactive
mask    = dense["active_mask"]        # (gx, gy, T) bool
metadata = dense["metadata"]

# Long — pandas DataFrame; best for filtering and analysis
load fairness_attribution_long.pkl as payload
df       = payload["dataframe"]
metadata = payload["metadata"]

# Tuples — list of row-tuples; best for dependency-free iteration
load fairness_attribution_tuples.pkl as payload
columns  = payload["columns"]
rows     = payload["rows"]
metadata = payload["metadata"]
```

Decision rule: **dense for tensor lookups inside a training loop, long for pandas filtering, tuples for dependency-free iteration**. The metadata sidecar (`metadata.json`) carries the same dict that is embedded inside each `.pkl`; either source is fine.
