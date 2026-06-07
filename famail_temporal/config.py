"""
Configuration constants for famail_temporal.

Every reviewer-visible knob lives here. The cache/ filenames encode the values
of this config so multiple configurations can coexist without invalidation.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

# Paths
PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_DATA_DIR = PACKAGE_ROOT / "source_data"
CACHE_DIR = PACKAGE_ROOT / "cache"
DISCRIMINATOR_CHECKPOINT_DIR = PACKAGE_ROOT / "discriminator_checkpoints"
DISCRIMINATOR_CHECKPOINT_FILENAME = "default/best.pt"

# Grid geometry (fixed by the Shenzhen dataset)
GRID_DIMS: Tuple[int, int] = (48, 90)
N_TIME_BUCKETS: int = 288

# Time blocks — each hourly block spans (h, h+1). No wraparound needed at
# hourly resolution. Names are zero-padded (hour_00 .. hour_23) for stable
# lexicographic ordering. The prior 4-block configuration (morning_peak,
# midday, evening_peak, night) was retained during framework validation
# and superseded at T=24 on 2026-04-24.
TIME_BLOCKS: List[Tuple[str, int, int]] = [
    (f"hour_{h:02d}", h, h + 1) for h in range(24)
]
T: int = len(TIME_BLOCKS)

# Active-unit filter
ACTIVE_SUPPLY_THRESHOLD: float = 0.5
# DEMAND_FLOOR is a CLAMP, not an activity filter: cells with observed D <
# DEMAND_FLOOR have their D substituted with DEMAND_FLOOR before computing
# Y = S/D. Keeping them in the active set (rather than filtering them out)
# preserves the ability of F_causal to detect unfairness in reachable-but-
# low-demand areas. See docs/F_CAUSAL_METHODOLOGY_NOTES.md §4 for the
# 0.5-value rationale (residual-scale balance against signal-regime Y).
DEMAND_FLOOR: float = 0.5
SUPPLY_FLOOR: float = 0.1

# Demographics
DEMOGRAPHIC_FEATURES: List[str] = [
    "AvgHousingPricePerSqM",
    "GDPperCapita",
    "CompPerCapita",
]

# Objective weights
ALPHA_SPATIAL: float = 0.33
ALPHA_CAUSAL: float = 0.33
ALPHA_FIDELITY: float = 0.34

# ST-iFGSM
STEP_SIZE_ALPHA: float = 0.1
EPSILON_BALL: float = 2.0
MAX_ITERATIONS: int = 50
# Convergence: the optimizer runs to MAX_ITERATIONS by default. Inside the
# loop we track the best-seen objective and apply *patience-based* early
# stopping — terminate when no iter has improved the best objective by more
# than CONVERGENCE_TOL for PATIENCE consecutive iterations. This replaces the
# old "|ΔL| < tol on consecutive iters" criterion, which fired prematurely
# under ST-iFGSM's sign-only step rule (any near-stationary point looked
# converged after one step). CONVERGENCE_TOL now plays the role of "minimum
# improvement that counts" — set above the metric's numerical noise floor;
# F-metrics are computed in float64 internally so 1e-6 is well above noise.
# Set PATIENCE=None to disable early stopping and always run MAX_ITERATIONS.
CONVERGENCE_TOL: float = 1e-6
PATIENCE: int = 10

# Multi-loop re-attribution editing (algorithm-improvements side project,
# spec 2026-06-06). The defaults below reproduce the historical single-pass
# batch behavior exactly: MAX_ROUNDS=1 (one round), ACCEPT_RULE="objective"
# (weighted-objective best-iterate, unchanged), EPSILON_CAP=EPSILON_BALL
# (cumulative cap equals the per-edit ball ⇒ no extra clip for a single edit).
MAX_ROUNDS: int = 1
# Outer-loop convergence: stop when the best round F_causal has not improved by
# more than ROUND_CONVERGENCE_TOL for ROUND_PATIENCE consecutive rounds. None
# disables convergence (fixed MAX_ROUNDS). Set above the F-metric noise floor.
ROUND_CONVERGENCE_TOL: float | None = None
ROUND_PATIENCE: int = 2
# Cumulative L-inf displacement cap from each trajectory's TRUE original pickup
# cell, enforced across rounds. EPSILON_BALL (2.0) keeps edits in the cGAIL 5x5
# IL window; set to float('inf') for unbounded per-round-epsilon stacking.
EPSILON_CAP: float = EPSILON_BALL
# Inner-loop acceptance gate. "objective": keep the best weighted-objective
# iterate (historical). "non-regression": additionally require the persisted
# iterate to improve F_causal and not regress F_spatial vs the trajectory's
# iter-0 state.
ACCEPT_RULE: str = "objective"
# Max times the iterative (B=1) preset may edit the same trajectory across
# rounds. 1 = historical no-re-edit; 0 = unlimited (epsilon-cap is the limiter).
ITERATIVE_TOPK_MAX_EDITS: int = 1

# Straight-through (hard-metric) editing (spec 2026-06-06). When True, modify_single
# evaluates/selects/gates on the realizable HARD grid (forward = hard, gradient =
# soft) instead of the soft relaxation, closing the soft-vs-hard gap of §8.7.
# Default False = historical soft behavior (bit-identical).
STE_ENABLED: bool = False

# Gradient diagnostics
DIAGNOSTICS_ENABLED: bool = True

# Soft cell assignment
SOFT_NEIGHBORHOOD_SIZE: int = 5
TAU_MAX: float = 1.0
TAU_MIN: float = 0.1
ANNEAL_TEMPERATURE: bool = True

# Numerical stability
EPS: float = 1e-8
MIN_ACTIVE_UNITS_PER_BLOCK: int = 10
MIN_TOTAL_ACTIVE_UNITS: int = 100

# Reproducibility
DEFAULT_SEED: int = 42


def cache_suffix(include_features: bool = False) -> str:
    """Build the config-encoded filename suffix for cached artifacts."""
    base = f"T{T}_thr{ACTIVE_SUPPLY_THRESHOLD}"
    if include_features:
        tokens = []
        for f in DEMOGRAPHIC_FEATURES:
            token = f.lower().replace("percapita", "").replace("avg", "").replace("price", "")
            token = token.replace("persqm", "").strip("_")
            tokens.append(token)
        base += "_feat-" + "-".join(tokens)
    return base
