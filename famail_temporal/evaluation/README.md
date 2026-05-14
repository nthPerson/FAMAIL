# famail_temporal.evaluation

End-to-end evaluation framework for the FAMAIL trajectory-modification pipeline.

## Quickstart

CLI:

    python -m famail_temporal.evaluation.runner --name demo

Programmatic:

    from famail_temporal.evaluation import run_experiment

    result = run_experiment(
        name="tighter-epsilon",
        config_overrides={"EPSILON_BALL": 1.5, "MAX_ITERATIONS": 20},
        k=100,
    )
    print(result.experiment_id, result.f_spatial_before, result.f_spatial_after)

## CLI flags

| Flag | Purpose |
|---|---|
| `--name <slug>` | Appended to the experiment ID for readability |
| `--max-trajectories N` | Limit the dataset (useful for quick iterations) |
| `--max-drivers N` | Limit the number of drivers loaded |
| `-k N` | Number of top-attribution trajectories to modify (default 100) |
| `--no-diagnostics` | Skip Tier A gradient decomposition and Tier C sensitivity grids |
| `--override KEY=VALUE` | Override any `famail_temporal.config` attribute. Repeat the flag. |

## What gets written

The authoritative artifact list and schemas live in the design spec in the parent monorepo
(available upon request); the summary below covers what each run writes under
`famail_temporal/results/{experiment_id}/`:

- `metrics.json` - config snapshot + provenance + before/after scalars
- `grid_before.pkl` / `grid_after.pkl` - (48, 90, T, 4) fairness grids
- `augmented_trajs_before.pkl[.gz]` / `augmented_trajs_after.pkl[.gz]` - full 8-element-state datasets
- `modified_trajectory_ids.json` - which trajectories were modified + cell moves
- `histories.pkl` - full per-iteration modification history
- `trajectories.csv` - one row per top-k modified trajectory
- `per_unit_attribution.csv` - one row per active unit
- `gradient_sensitivity_{before,after}.pkl` - (48, 90, T, 2) when diagnostics are on
- `report.md` - tables-only human-readable summary
