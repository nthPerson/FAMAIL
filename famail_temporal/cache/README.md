# famail_temporal/cache/

Preprocessed artifacts produced by `python -m famail_temporal.preprocess`. All files are gitignored.

## Filename Scheme

```
{artifact}_T{T}_thr{threshold}[_feat-{feature_tokens}].pkl
```

- `T` — number of time blocks (e.g., `T4`)
- `thr` — active supply threshold (e.g., `thr0.5`)
- `feat-...` — only present for artifacts that depend on the demographic feature set (e.g., `feat-housing-gdp-comp`)

Example: `hat_matrices_T4_thr0.5_feat-housing-gdp-comp.pkl`

This encoding lets multiple configurations coexist in the same cache directory without invalidation.

## Artifact Types

| Artifact Name | Shape / Type | Description |
|---|---|---|
| `pickup_counts` | `(48, 90, T)` float32 | Mean hourly pickups per (cell, block) |
| `dropoff_counts` | `(48, 90, T)` float32 | Mean hourly dropoffs per (cell, block) |
| `active_taxis` | `(48, 90, T)` float32 | Mean hourly active taxis per (cell, block) |
| `active_mask` | `(48, 90, T)` bool | Active-unit mask (supply + valid + finite demographics) |
| `unit_index_map` | `UnitIndexMap` | Canonical ordering of active (cell, t) units |
| `g0_power_basis` | `G0Function` | Fitted g_0(D) power-basis function |
| `hat_matrices` | `dict` | `I_minus_H_demo`, `M`, scaler params, diagnostics (includes features suffix) |
| `metadata` | `dict` | `n_days`, config snapshot (`config_T`, `config_GRID_DIMS`, `config_ACTIVE_SUPPLY_THRESHOLD`, `config_DEMAND_FLOOR`, `config_DEMOGRAPHIC_FEATURES`) |

## Regenerating

```bash
python -m famail_temporal.preprocess          # skip existing artifacts
python -m famail_temporal.preprocess --force   # overwrite all
```

## Staleness warning

Running `python -m famail_temporal.preprocess` (without `--force`) skips
artifacts that already exist on disk. If you modify the raw data or change
`config.py` values that affect the same suffix (e.g., threshold, features),
use `--force` to regenerate all artifacts:

    python -m famail_temporal.preprocess --force

Without `--force`, you may get an inconsistent mix of old and new artifacts.
