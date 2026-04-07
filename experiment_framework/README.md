# Experiment Framework

Run, record, and analyze FAMAIL trajectory modification experiments from the command line.

## Quick Start

```bash
# Run with defaults (top_k=10, equal weights)
python -m experiment_framework run

# Override parameters via CLI flags
python -m experiment_framework run --top-k 20 --alpha-spatial 0.5 --alpha-causal 0.3 --alpha-fidelity 0.2

# Run from a config file (with optional CLI overrides)
python -m experiment_framework run --config my_config.json --top-k 30

# Enable per-term gradient decomposition (3x slower, records which objective term drives each step)
python -m experiment_framework run --gradient-decomposition --top-k 10
```

## Commands

| Command | Purpose |
|---------|---------|
| `run` | Execute a single experiment |
| `sweep` | Run a Cartesian-product parameter sweep |
| `dashboard` | Launch the Streamlit analysis dashboard |
| `summarize` | Generate a CSV comparing multiple runs |

### Parameter Sweep

```bash
python -m experiment_framework sweep \
  --base-config base.json \
  --sweep '{"alpha_spatial": [0.2, 0.5, 0.8], "top_k": [10, 20]}'
```

This runs 6 experiments (3 x 2) and saves each to `experiment_results/`.

### Summarize

```bash
python -m experiment_framework summarize --results-dir experiment_results/ --output sweep_summary.csv
```

Produces a single CSV with one row per run, comparing key metrics across experiments.

### Dashboard

```bash
python -m experiment_framework dashboard --results-dir experiment_results/
```

Opens a Streamlit dashboard for interactive exploration of experiment results.

## Key CLI Flags (`run`)

| Flag | Default | Description |
|------|---------|-------------|
| `--top-k` | 10 | Number of trajectories to select and modify |
| `--alpha-spatial` | 0.33 | Weight for spatial fairness (Gini) term |
| `--alpha-causal` | 0.33 | Weight for causal fairness (R²) term |
| `--alpha-fidelity` | 0.34 | Weight for discriminator fidelity term |
| `--alpha` | 0.1 | ST-iFGSM step size |
| `--epsilon` | 2.0 | Max perturbation in grid cells |
| `--max-iterations` | 50 | Max ST-iFGSM iterations per trajectory |
| `--causal-formulation` | `option_b` | Causal term variant: `baseline`, `option_b`, `option_c` |
| `--gradient-decomposition` | off | Record per-term gradient vectors (3x backward cost) |
| `--no-discriminator` | off | Disable the fidelity discriminator |
| `--max-trajectories` | 100 | Max trajectories to load from source data |
| `--seed` | 42 | Random seed |
| `--config` | — | Load all settings from a JSON file |
| `--output-dir` | `experiment_results` | Where to save results |
| `-v` | off | Verbose (DEBUG-level) logging |

Weights should sum to ~1.0. The framework validates this on startup.

## Output Files

Each run creates a timestamped directory under `experiment_results/`:

```
experiment_results/20260324_121204_per_term_grad_norm_test/
├── config.json             # Full configuration used for this run
├── results.json            # Complete results (config + all data below in one file)
├── summary.json            # High-level before/after metrics and improvement deltas
├── report.md               # Auto-generated markdown report with tables and interpretation
├── trajectories.csv        # One row per modified trajectory (original/modified cell, convergence, scores)
├── iterations.csv          # One row per iteration per trajectory (objective terms, gradients, perturbation)
├── global_snapshots.csv    # Fairness metrics after each trajectory is modified (tracks cumulative progress)
└── attribution_scores.csv  # Phase 1 attribution scores (LIS + DCD) for all candidate trajectories
```

### What each file is for

- **config.json** — Reproduce the experiment. Pass it back with `--config` to re-run with identical settings.
- **results.json** — Single source of truth. Contains everything: metadata, config, summary, attribution scores, per-trajectory iteration histories, and global snapshots. Used by `ExperimentResult.load()` and the dashboard.
- **summary.json** — Quick check of before/after fairness metrics (Gini, F_spatial, F_causal, F_fidelity, combined) and improvement deltas without parsing the full results.
- **report.md** — Human-readable experiment report with configuration table, before/after metrics, top-10 most impactful trajectories, gradient decomposition summary (if enabled), and convergence statistics.
- **trajectories.csv** — One row per modified trajectory: original and modified pickup cells, convergence status, final objective term values, and perturbation magnitude. Good for quick spreadsheet analysis.
- **iterations.csv** — Detailed per-iteration trace for every trajectory: objective value, individual F_spatial/F_causal/F_fidelity scores, gradient norms, cumulative perturbation, and (if `--gradient-decomposition` is on) per-term gradient vectors with contribution fractions and spatial-causal alignment.
- **global_snapshots.csv** — Tracks how the global fairness metrics evolve as each trajectory is modified in sequence. Useful for plotting cumulative improvement curves.
- **attribution_scores.csv** — Phase 1 output: LIS (Local Inequality Score), DCD (Demand-Conditional Deviation), and combined attribution scores for all candidate trajectories. Shows which trajectories were ranked highest for modification.

## Programmatic API

```python
from experiment_framework import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(top_k=20, alpha_spatial=0.5, alpha_causal=0.3, alpha_fidelity=0.2)
runner = ExperimentRunner(config)
result = runner.run()
run_dir = result.save()  # returns path to the output directory
```

Load previous results:

```python
from experiment_framework import ExperimentResult

result = ExperimentResult.load("experiment_results/20260324_121204_per_term_grad_norm_test")
print(result.initial_snapshot)  # baseline metrics
print(result.final_snapshot)    # post-modification metrics
```

## Module Files

| File | Role |
|------|------|
| `cli.py` | CLI entry point (argparse, `run`/`sweep`/`dashboard`/`summarize` commands) |
| `experiment_config.py` | `ExperimentConfig` dataclass + `SweepConfig` for parameter sweeps |
| `experiment_runner.py` | Orchestrates the full pipeline: data load, attribution, modification, snapshots |
| `experiment_result.py` | Result dataclasses, JSON/CSV serialization, markdown report generation |
| `gradient_decomposition.py` | Per-term gradient decomposition during ST-iFGSM (optional, expensive) |
| `analysis_dashboard.py` | Streamlit dashboard for interactive result exploration |
