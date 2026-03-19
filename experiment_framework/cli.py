"""
Command-line interface for the FAMAIL Experiment & Analysis Framework.

Usage:
    python -m experiment_framework.cli run --top-k 20 --alpha-spatial 0.5
    python -m experiment_framework.cli run --config experiment_config.json
    python -m experiment_framework.cli sweep --base-config base.json --sweep '{"alpha_spatial": [0.2, 0.5, 0.8]}'
    python -m experiment_framework.cli dashboard --results-dir experiment_results/
    python -m experiment_framework.cli summarize --results-dir experiment_results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='FAMAIL Experiment & Analysis Framework',
        prog='experiment_framework',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # --- run ---
    run_parser = subparsers.add_parser('run', help='Run a single experiment')
    run_parser.add_argument('--config', type=str, help='Path to config JSON file')
    run_parser.add_argument('--name', type=str, help='Experiment name')
    run_parser.add_argument('--description', type=str, help='Experiment description')
    run_parser.add_argument('--top-k', type=int, help='Number of trajectories to modify')
    run_parser.add_argument('--max-trajectories', type=int, help='Max trajectories to load')
    run_parser.add_argument('--max-iterations', type=int, help='Max ST-iFGSM iterations')
    run_parser.add_argument('--alpha', type=float, help='ST-iFGSM step size')
    run_parser.add_argument('--epsilon', type=float, help='Max perturbation (grid cells)')
    run_parser.add_argument('--alpha-spatial', type=float, help='Spatial fairness weight')
    run_parser.add_argument('--alpha-causal', type=float, help='Causal fairness weight')
    run_parser.add_argument('--alpha-fidelity', type=float, help='Fidelity weight')
    run_parser.add_argument('--causal-formulation', type=str, choices=['baseline', 'option_b', 'option_c'])
    run_parser.add_argument('--checkpoint', type=str, help='Discriminator checkpoint path')
    run_parser.add_argument('--no-discriminator', action='store_true', help='Disable discriminator')
    run_parser.add_argument('--gradient-decomposition', action='store_true', help='Record per-term gradients')
    run_parser.add_argument('--seed', type=int, help='Random seed')
    run_parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'])
    run_parser.add_argument('--output-dir', type=str, help='Output directory')
    run_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    # --- sweep ---
    sweep_parser = subparsers.add_parser('sweep', help='Run a parameter sweep')
    sweep_parser.add_argument('--base-config', type=str, required=True, help='Base config JSON')
    sweep_parser.add_argument('--sweep', type=str, required=True, help='JSON dict of param -> values')
    sweep_parser.add_argument('--output-dir', type=str, default='experiment_results')
    sweep_parser.add_argument('--verbose', '-v', action='store_true')

    # --- dashboard ---
    dash_parser = subparsers.add_parser('dashboard', help='Launch analysis dashboard')
    dash_parser.add_argument('--results-dir', type=str, default='experiment_results')
    dash_parser.add_argument('--port', type=int, default=8501)

    # --- summarize ---
    sum_parser = subparsers.add_parser('summarize', help='Generate sweep summary CSV')
    sum_parser.add_argument('--results-dir', type=str, required=True)
    sum_parser.add_argument('--output', type=str, default='sweep_summary.csv')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'run':
        _cmd_run(args)
    elif args.command == 'sweep':
        _cmd_sweep(args)
    elif args.command == 'dashboard':
        _cmd_dashboard(args)
    elif args.command == 'summarize':
        _cmd_summarize(args)


def _cmd_run(args):
    """Execute a single experiment run."""
    from experiment_framework.experiment_config import ExperimentConfig
    from experiment_framework.experiment_runner import ExperimentRunner

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    # Load base config or defaults
    if args.config:
        config = ExperimentConfig.from_json(args.config)
    else:
        config = ExperimentConfig()

    # Apply CLI overrides
    overrides = {}
    if args.name:
        overrides['experiment_name'] = args.name
    if args.description:
        overrides['description'] = args.description
    if args.top_k is not None:
        overrides['top_k'] = args.top_k
    if args.max_trajectories is not None:
        overrides['max_trajectories'] = args.max_trajectories
    if args.max_iterations is not None:
        overrides['max_iterations'] = args.max_iterations
    if args.alpha is not None:
        overrides['alpha'] = args.alpha
    if args.epsilon is not None:
        overrides['epsilon'] = args.epsilon
    if args.alpha_spatial is not None:
        overrides['alpha_spatial'] = args.alpha_spatial
    if args.alpha_causal is not None:
        overrides['alpha_causal'] = args.alpha_causal
    if args.alpha_fidelity is not None:
        overrides['alpha_fidelity'] = args.alpha_fidelity
    if args.causal_formulation:
        overrides['causal_formulation'] = args.causal_formulation
    if args.checkpoint:
        overrides['discriminator_checkpoint'] = args.checkpoint
    if args.no_discriminator:
        overrides['use_discriminator'] = False
    if args.gradient_decomposition:
        overrides['record_gradient_decomposition'] = True
    if args.seed is not None:
        overrides['random_seed'] = args.seed
    if args.device:
        overrides['device'] = args.device
    if args.output_dir:
        overrides['output_dir'] = args.output_dir

    if overrides:
        config = config.with_overrides(**overrides)

    runner = ExperimentRunner(config)
    result = runner.run()
    run_dir = result.save()
    print(f"\nResults saved to: {run_dir}")


def _cmd_sweep(args):
    """Execute a parameter sweep."""
    from experiment_framework.experiment_config import ExperimentConfig, SweepConfig
    from experiment_framework.experiment_runner import ExperimentRunner

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    base = ExperimentConfig.from_json(args.base_config)
    base = base.with_overrides(output_dir=args.output_dir)
    sweep_params = json.loads(args.sweep)

    sweep = SweepConfig(base_config=base, sweep_params=sweep_params)
    configs = sweep.generate_configs()
    print(f"Sweep: {len(configs)} configurations")

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {cfg.experiment_name}")
        runner = ExperimentRunner(cfg)
        result = runner.run()
        run_dir = result.save()
        print(f"  -> {run_dir}")

    print(f"\nSweep complete. Results in: {args.output_dir}")


def _cmd_dashboard(args):
    """Launch the Streamlit analysis dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / 'analysis_dashboard.py'
    if not dashboard_path.exists():
        print(f"Dashboard not found at: {dashboard_path}")
        sys.exit(1)

    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        str(dashboard_path),
        '--', '--results-dir', args.results_dir,
    ]
    print(f"Launching dashboard on port {args.port}...")
    subprocess.run(cmd + ['--server.port', str(args.port)])


def _cmd_summarize(args):
    """Generate a summary CSV from multiple experiment runs."""
    import csv
    from experiment_framework.experiment_result import ExperimentResult

    results_dir = Path(args.results_dir)
    run_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / 'results.json').exists()
    ])

    if not run_dirs:
        print(f"No experiment results found in: {results_dir}")
        sys.exit(1)

    print(f"Found {len(run_dirs)} experiment runs")

    rows = []
    for rd in run_dirs:
        result = ExperimentResult.load(str(rd))
        summary = result._build_summary()
        cfg = result.config
        rows.append({
            'run_dir': rd.name,
            'experiment_name': cfg.get('experiment_name', ''),
            'top_k': cfg.get('top_k', ''),
            'alpha_spatial': cfg.get('alpha_spatial', ''),
            'alpha_causal': cfg.get('alpha_causal', ''),
            'alpha_fidelity': cfg.get('alpha_fidelity', ''),
            'causal_formulation': cfg.get('causal_formulation', ''),
            'initial_gini': summary['initial'].get('gini', ''),
            'final_gini': summary['final'].get('gini', ''),
            'delta_gini': summary['improvement'].get('delta_gini', ''),
            'initial_combined': summary['initial'].get('combined', ''),
            'final_combined': summary['final'].get('combined', ''),
            'delta_combined': summary['improvement'].get('delta_combined', ''),
            'n_converged': summary.get('n_converged', ''),
            'mean_iterations': summary.get('mean_iterations', ''),
            'mean_perturbation': summary.get('mean_perturbation_magnitude', ''),
            'duration_s': result.duration_seconds,
        })

    output = Path(args.output)
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary written to: {output}")


if __name__ == '__main__':
    main()
