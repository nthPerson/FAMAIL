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

# Time blocks — end > 24 encodes wraparound
TIME_BLOCKS: List[Tuple[str, int, int]] = [
    ("morning_peak", 7, 10),
    ("midday",       10, 16),
    ("evening_peak", 16, 20),
    ("night",        20, 31),  # 20 to 07 next day
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
CONVERGENCE_TOL: float = 1e-6

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
