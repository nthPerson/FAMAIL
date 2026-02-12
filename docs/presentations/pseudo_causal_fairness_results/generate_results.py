"""
Generate trajectory modification results for progress report.

This script runs the trajectory modification algorithm with the specified
configuration and saves all results to a JSON file for visualization.

Configuration:
- Convergence threshold: 1.0e-6
- Epsilon: 3.0
- Alpha: 0.10
- Max iterations: 50
- Objective weights: ~1/3 each
- Selection mode: "Top-k by Fairness Impact", k=10
- Discriminator: pass-seek_5000-20000_(84ident_72same_44diff)/best.pt
- Temperature annealing: enabled (1.0 → 0.1)
- Gradient mode: soft_cell
"""

import sys
from pathlib import Path
import json
import pickle
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from trajectory_modification import (
    DataBundle,
    TrajectoryModifier,
    FAMAILObjective,
    GlobalMetrics,
    DiscriminatorAdapter,
)


def compute_cell_lis_scores(cell_counts: np.ndarray) -> np.ndarray:
    """Compute Local Inequality Score (LIS) for all cells."""
    mean_val = cell_counts.mean()
    if mean_val < 1e-8:
        return np.zeros_like(cell_counts)
    lis = np.abs(cell_counts - mean_val) / mean_val
    return lis


def compute_cell_dcd_scores(
    demand_counts: np.ndarray,
    supply_counts: np.ndarray,
    g_function,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute Demand-Conditional Deviation (DCD) for all cells."""
    dcd = np.zeros_like(demand_counts, dtype=float)

    if g_function is None:
        # Fallback: use deviation from mean service ratio
        mask = demand_counts > eps
        if not mask.any():
            return dcd
        Y = np.zeros_like(demand_counts, dtype=float)
        Y[mask] = supply_counts[mask] / (demand_counts[mask] + eps)
        mean_Y = Y[mask].mean()
        dcd[mask] = np.abs(Y[mask] - mean_Y)
        return dcd

    # Only compute for cells with sufficient demand
    mask = demand_counts > eps

    if not mask.any():
        return dcd

    # Compute actual service ratio Y = S / D for active cells
    Y = np.zeros_like(demand_counts, dtype=float)
    Y[mask] = supply_counts[mask] / (demand_counts[mask] + eps)

    # Compute expected ratio g(D) for all cells
    D_flat = demand_counts.flatten()
    g_d_flat = g_function(D_flat)
    g_d = g_d_flat.reshape(demand_counts.shape)

    # DCD = |Y - g(D)| for active cells
    dcd[mask] = np.abs(Y[mask] - g_d[mask])

    return dcd


def compute_trajectory_attribution_scores(
    trajectories,
    pickup_counts: np.ndarray,
    dropoff_counts: np.ndarray,
    supply_counts: np.ndarray,
    g_function,
    lis_weight: float = 0.5,
    dcd_weight: float = 0.5,
):
    """Compute combined attribution scores for all trajectories."""
    # Compute cell-level LIS scores
    pickup_lis = compute_cell_lis_scores(pickup_counts)
    dropoff_lis = compute_cell_lis_scores(dropoff_counts)

    # Compute cell-level DCD scores
    dcd_scores = compute_cell_dcd_scores(pickup_counts, supply_counts, g_function)

    # Compute per-trajectory scores
    trajectory_scores = []

    for idx, traj in enumerate(trajectories):
        # Get pickup cell (last state) and dropoff cell (first state)
        pickup_state = traj.states[-1]
        dropoff_state = traj.states[0]

        pickup_cell = (int(pickup_state.x_grid), int(pickup_state.y_grid))
        dropoff_cell = (int(dropoff_state.x_grid), int(dropoff_state.y_grid))

        # Get LIS values for this trajectory's cells
        px, py = pickup_cell
        dx, dy = dropoff_cell

        lis_pickup = 0.0
        if 0 <= px < pickup_lis.shape[0] and 0 <= py < pickup_lis.shape[1]:
            lis_pickup = float(pickup_lis[px, py])

        lis_dropoff = 0.0
        if 0 <= dx < dropoff_lis.shape[0] and 0 <= dy < dropoff_lis.shape[1]:
            lis_dropoff = float(dropoff_lis[dx, dy])

        # Trajectory LIS = max of pickup and dropoff LIS
        traj_lis = max(lis_pickup, lis_dropoff)

        # Get DCD value at pickup cell
        traj_dcd = 0.0
        if 0 <= px < dcd_scores.shape[0] and 0 <= py < dcd_scores.shape[1]:
            traj_dcd = float(dcd_scores[px, py])

        trajectory_scores.append({
            'index': idx,
            'trajectory_id': getattr(traj, 'trajectory_id', idx),
            'driver_id': getattr(traj, 'driver_id', 'N/A'),
            'lis_score': traj_lis,
            'dcd_score': traj_dcd,
            'pickup_cell': pickup_cell,
            'dropoff_cell': dropoff_cell,
        })

    # Normalize scores
    lis_values = np.array([s['lis_score'] for s in trajectory_scores])
    dcd_values = np.array([s['dcd_score'] for s in trajectory_scores])

    lis_max = lis_values.max() if lis_values.max() > 0 else 1.0
    dcd_max = dcd_values.max() if dcd_values.max() > 0 else 1.0

    for s in trajectory_scores:
        s['lis_score_normalized'] = s['lis_score'] / lis_max
        s['dcd_score_normalized'] = s['dcd_score'] / dcd_max
        s['combined_score'] = (
            lis_weight * s['lis_score_normalized'] +
            dcd_weight * s['dcd_score_normalized']
        )

    return trajectory_scores


def select_top_k(trajectory_scores, k: int):
    """Select top-k trajectory indices by combined score."""
    sorted_scores = sorted(trajectory_scores, key=lambda x: x['combined_score'], reverse=True)
    return [s['index'] for s in sorted_scores[:k]]


def main():
    """Run trajectory modification and save results."""
    print("=" * 70)
    print("FAMAIL Trajectory Modification Results Generation")
    print("=" * 70)

    # Configuration
    config = {
        'convergence_threshold': 1.0e-6,
        'epsilon': 3.0,
        'alpha': 0.10,
        'max_iterations': 50,
        'alpha_spatial': 0.33,
        'alpha_causal': 0.33,
        'alpha_fidelity': 0.34,
        'k': 10,
        'discriminator_checkpoint': 'discriminator/model/checkpoints/pass-seek_5000-20000_(84ident_72same_44diff)/best.pt',
        'gradient_mode': 'soft_cell',
        'temperature_annealing': True,
        'tau_max': 1.0,
        'tau_min': 0.1,
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Load data
    print("\n" + "-" * 70)
    print("Loading data...")
    bundle = DataBundle.load_default(
        max_trajectories=100,
        estimate_g_from_data=True,
        aggregation='mean',
    )
    print(f"✓ Loaded {len(bundle.trajectories)} trajectories")
    print(f"✓ Pickup grid shape: {bundle.pickup_grid.shape}")
    print(f"✓ g(d) diagnostics: R² = {bundle.g_function_diagnostics.get('r_squared', 'N/A')}")

    # Compute attribution scores
    print("\n" + "-" * 70)
    print("Computing attribution scores...")
    trajectory_scores = compute_trajectory_attribution_scores(
        trajectories=bundle.trajectories,
        pickup_counts=bundle.pickup_grid,
        dropoff_counts=bundle.dropoff_grid,
        supply_counts=bundle.active_taxis_grid,
        g_function=bundle.g_function,
        lis_weight=0.5,
        dcd_weight=0.5,
    )

    # Select top-k
    selected_indices = select_top_k(trajectory_scores, config['k'])
    print(f"✓ Selected top-{config['k']} trajectories by attribution score")
    print(f"  Indices: {selected_indices}")

    # Initialize discriminator
    print("\n" + "-" * 70)
    print("Loading discriminator...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = PROJECT_ROOT / config['discriminator_checkpoint']

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Discriminator checkpoint not found: {checkpoint_path}")

    adapter = DiscriminatorAdapter(checkpoint_path=checkpoint_path, device=device)
    discriminator = adapter.model
    print(f"✓ Discriminator loaded on {device}")

    # Create objective function
    print("\n" + "-" * 70)
    print("Initializing objective function...")
    objective = FAMAILObjective(
        alpha_spatial=config['alpha_spatial'],
        alpha_causal=config['alpha_causal'],
        alpha_fidelity=config['alpha_fidelity'],
        g_function=bundle.g_function,
        discriminator=discriminator,
    )
    print("✓ Objective function initialized")

    # Create modifier
    print("\n" + "-" * 70)
    print("Initializing trajectory modifier...")
    modifier = TrajectoryModifier(
        objective_fn=objective,
        alpha=config['alpha'],
        epsilon=config['epsilon'],
        max_iterations=config['max_iterations'],
        convergence_threshold=config['convergence_threshold'],
        gradient_mode=config['gradient_mode'],
        temperature=config['tau_max'],
        temperature_annealing=config['temperature_annealing'],
        tau_max=config['tau_max'],
        tau_min=config['tau_min'],
    )
    print("✓ Modifier initialized")

    # Set global state
    modifier.set_global_state(
        pickup_counts=torch.tensor(bundle.pickup_grid, device=device, dtype=torch.float32),
        dropoff_counts=torch.tensor(bundle.dropoff_grid, device=device, dtype=torch.float32),
        active_taxis=torch.tensor(bundle.active_taxis_grid, device=device, dtype=torch.float32),
        causal_demand=torch.tensor(bundle.causal_demand_grid, device=device, dtype=torch.float32) if bundle.causal_demand_grid is not None else None,
        causal_supply=torch.tensor(bundle.causal_supply_grid, device=device, dtype=torch.float32) if bundle.causal_supply_grid is not None else None,
    )
    print("✓ Global state configured")

    # Initialize metrics
    metrics = GlobalMetrics(
        g_function=bundle.g_function,
        alpha_weights=(config['alpha_spatial'], config['alpha_causal'], config['alpha_fidelity'])
    )
    metrics.initialize_from_data(
        bundle.pickup_grid,
        bundle.dropoff_grid,
        bundle.active_taxis_grid,
    )

    initial_snapshot = metrics.compute_snapshot()
    print(f"\nInitial Metrics:")
    print(f"  Gini Coefficient: {initial_snapshot.gini_coefficient:.6f}")
    print(f"  F_spatial: {initial_snapshot.f_spatial:.6f}")
    print(f"  F_causal: {initial_snapshot.f_causal:.6f}")
    print(f"  Combined L: {initial_snapshot.combined_objective:.6f}")

    # Run modification
    print("\n" + "-" * 70)
    print(f"Running modification on {len(selected_indices)} trajectories...")

    histories = []
    for i, idx in enumerate(selected_indices):
        print(f"\n  [{i+1}/{len(selected_indices)}] Processing trajectory {idx}...")
        traj = bundle.trajectories[idx]

        try:
            history = modifier.modify_single(traj)
            histories.append(history)

            # Update metrics
            if history.iterations:
                orig_state = traj.states[-1]
                mod_state = history.modified.states[-1]
                metrics.update_pickup(
                    old_cell=(int(orig_state.x_grid), int(orig_state.y_grid)),
                    new_cell=(int(mod_state.x_grid), int(mod_state.y_grid)),
                    fidelity_score=history.iterations[-1].f_fidelity,
                )

                print(f"    ✓ Converged: {history.converged}, Iterations: {history.total_iterations}")
                print(f"    ✓ Final objective: {history.final_objective:.8f}")
                print(f"    ✓ Pickup moved: ({orig_state.x_grid:.2f}, {orig_state.y_grid:.2f}) → ({mod_state.x_grid:.2f}, {mod_state.y_grid:.2f})")
            else:
                print(f"    ✗ No iterations recorded")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    final_snapshot = metrics.compute_snapshot()
    print(f"\nFinal Metrics:")
    print(f"  Gini Coefficient: {final_snapshot.gini_coefficient:.6f} (Δ{final_snapshot.gini_coefficient - initial_snapshot.gini_coefficient:+.6f})")
    print(f"  F_spatial: {final_snapshot.f_spatial:.6f} (Δ{final_snapshot.f_spatial - initial_snapshot.f_spatial:+.6f})")
    print(f"  F_causal: {final_snapshot.f_causal:.6f} (Δ{final_snapshot.f_causal - initial_snapshot.f_causal:+.6f})")
    print(f"  Combined L: {final_snapshot.combined_objective:.6f} (Δ{final_snapshot.combined_objective - initial_snapshot.combined_objective:+.6f})")

    # Prepare results for JSON export
    print("\n" + "-" * 70)
    print("Preparing results for export...")

    results = {
        'config': config,
        'initial_metrics': {
            'gini_coefficient': float(initial_snapshot.gini_coefficient),
            'f_spatial': float(initial_snapshot.f_spatial),
            'f_causal': float(initial_snapshot.f_causal),
            'f_fidelity': float(initial_snapshot.mean_fidelity),
            'combined_objective': float(initial_snapshot.combined_objective),
        },
        'final_metrics': {
            'gini_coefficient': float(final_snapshot.gini_coefficient),
            'f_spatial': float(final_snapshot.f_spatial),
            'f_causal': float(final_snapshot.f_causal),
            'f_fidelity': float(final_snapshot.mean_fidelity),
            'combined_objective': float(final_snapshot.combined_objective),
        },
        'attribution_scores': [
            {
                'index': s['index'],
                'trajectory_id': int(s['trajectory_id']) if isinstance(s['trajectory_id'], (int, np.integer)) else str(s['trajectory_id']),
                'driver_id': int(s['driver_id']) if isinstance(s['driver_id'], (int, np.integer)) else str(s['driver_id']),
                'lis_score': float(s['lis_score']),
                'dcd_score': float(s['dcd_score']),
                'combined_score': float(s['combined_score']),
                'pickup_cell': [int(s['pickup_cell'][0]), int(s['pickup_cell'][1])],
                'dropoff_cell': [int(s['dropoff_cell'][0]), int(s['dropoff_cell'][1])],
            }
            for s in trajectory_scores if s['index'] in selected_indices
        ],
        'modifications': [],
    }

    # Add modification details
    for i, (idx, history) in enumerate(zip(selected_indices, histories)):
        traj = bundle.trajectories[idx]

        if not history.iterations:
            continue

        orig_pickup = history.original.states[-1]
        mod_pickup = history.modified.states[-1]

        mod_data = {
            'trajectory_index': int(idx),
            'trajectory_id': int(getattr(traj, 'trajectory_id', idx)),
            'driver_id': int(getattr(traj, 'driver_id', 0)),
            'original_pickup': {
                'x': float(orig_pickup.x_grid),
                'y': float(orig_pickup.y_grid),
            },
            'modified_pickup': {
                'x': float(mod_pickup.x_grid),
                'y': float(mod_pickup.y_grid),
            },
            'converged': bool(history.converged),
            'total_iterations': int(history.total_iterations),
            'final_objective': float(history.final_objective),
            'iterations': [],
        }

        # Add iteration details
        for iter_idx, result in enumerate(history.iterations):
            iter_data = {
                'iteration': iter_idx + 1,
                'objective_value': float(result.objective_value),
                'f_spatial': float(result.f_spatial),
                'f_causal': float(result.f_causal),
                'f_fidelity': float(result.f_fidelity),
                'gradient_norm': float(result.gradient_norm),
                'perturbation_x': float(result.perturbation[0]),
                'perturbation_y': float(result.perturbation[1]),
            }
            mod_data['iterations'].append(iter_data)

        results['modifications'].append(mod_data)

    # Save results
    output_path = Path(__file__).parent / 'results.json'
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("✓ Results generation complete!")
    print("=" * 70)
    print(f"\nNext step: Open index.html in a browser to view the report.")


if __name__ == '__main__':
    main()
